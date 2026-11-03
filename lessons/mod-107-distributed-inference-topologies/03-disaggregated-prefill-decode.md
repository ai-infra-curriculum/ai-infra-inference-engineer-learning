# Chapter 03 — Disaggregated Prefill-Decode Serving

> Prefill is compute-bound; decode is memory-bandwidth-bound. On one worker you solve the interference with chunked prefill (module 103). When one worker's mix is no longer tractable, you split them into two worker pools and ship the KV cache between them. DistServe, Splitwise, and NVIDIA Dynamo are the three deployment references.

## Motivation

Module 103 teaches you to tame the prefill-decode interference problem on a single worker: chunked prefill splits large prompts into chunks that interleave with decodes in the scheduler. For the many workloads where prompts are modest (a few thousand tokens), the batch size is reasonable, and the SLO is forgiving, chunked prefill is correct and sufficient.

Three regimes break this:

1. **Long context + high QPS.** 32k- or 128k-token prefills interleaved with decodes on the same worker stall the decode loop by the chunk size's prefill time. TTFT becomes acceptable for the prefill-request itself but TPOT becomes miserable for every other request on the worker.
2. **Different SLOs for the two phases.** A product that promises "TTFT ≤ 300 ms" but allows "TPOT ≤ 50 ms" has two different latency targets. On a flat worker, you cannot optimise prefill latency and decode throughput independently — the batching knob trades one for the other.
3. **Mismatched hardware for the two phases.** Prefill wants compute (lots of FLOPs); decode wants memory bandwidth (lots of HBM-bandwidth-per-token). A fleet with heterogeneous GPUs (H100 for compute, A10 for decode, say) cannot route efficiently if prefill and decode are mingled per worker.

The response: **disaggregate** — run two worker pools, one for prefill and one for decode, with a router that sends every request through prefill first, ships the resulting KV cache to a decode worker, and then streams tokens from the decode worker to the client. DistServe (OSDI 2024), Splitwise (ISCA 2024), and NVIDIA Dynamo (production reference implementation) are the three canonical references.

By the end of this chapter you can:

- State the architecture — prefill pool, decode pool, KV router, cross-node KV transfer — and draw it on a whiteboard.
- Decide when a workload has graduated from "chunked prefill is enough" to "needs disaggregation."
- Pick the TP / PP configuration for the two pools independently; they usually differ.
- Deploy a disaggregated topology on Dynamo or SGLang (vLLM also has disaggregated-serving support in later releases).
- Measure the goodput envelope (prefill QPS × decode QPS inside SLO) and reason about placement — how many prefill workers vs. how many decode workers.
- Recognise the three main failure modes (KV transfer bandwidth cliff, imbalanced pool sizing, router hotspotting) and the mitigations.

<!-- needs-research: disaggregated prefill-decode has rapidly evolved since late 2024. The specific flag spellings and architecture components (Dynamo's NIXL, SGLang's PD-disaggregation, vLLM's PDD) rotate across releases. Verify every example below against the pinned version. -->

## The architecture

A disaggregated fleet has three or four tiers in order of request flow:

```
Client
  │
  ▼
[ Front-end router / OpenAI-compatible API ]
  │
  ▼                                              (optional pre-router cache lookup)
[ Prefill pool: workers P0, P1, ..., Pn ]        ← chapter-01 topology; TP/PP/EP shape
  │
  │  (KV cache produced; shipped to the decode-side)
  ▼
[ KV router / KV transport layer ]
  │
  ▼
[ Decode pool: workers D0, D1, ..., Dm ]         ← chapter-01 topology; typically different shape
  │
  ▼
[ Streaming response back to the client ]
```

### The prefill pool

Each prefill worker runs the full model and executes the prefill phase: tokenise, compute embeddings, run all transformer layers over the entire prompt, and produce the KV cache. The output of a prefill is:

- The sequence's full KV cache (per layer, per head, per token).
- The last-token hidden state (used to produce the first output token).
- A sequence-level handle that identifies this KV cache for the decode pool.

Prefill workers are **compute-bound** at long context. Their batching discipline is: pack as many tokens as fit in the compute budget per iteration (chapter 103 chunked prefill applies within the pool), but no decodes.

### The decode pool

Each decode worker runs the same model, but only executes the decode phase: receive a KV cache, generate tokens one at a time, stream back to the router.

Decode workers are **memory-bandwidth-bound**: TPOT ≈ (model_weight_bytes + per-token-KV-read) / HBM-bandwidth. Their batching discipline is: continuous batching across many sequences, maximising HBM-bandwidth utilisation per decode step.

The pool-sizing problem: given a workload mix, how many prefill vs. decode workers? Depends on average prompt length vs. average generation length, QPS, and SLO.

### The KV router

The KV router is the critical new component. It:

- Receives every incoming request.
- Routes the request to a prefill worker (load-balanced, possibly with prefix-cache-aware routing — chapter 104 composes).
- Receives notification that the prefill is complete, with a handle to the KV cache on the prefill worker.
- Chooses a decode worker (load-balanced by TPOT headroom; sometimes also by locality to the prefill worker).
- Initiates the KV transfer: prefill worker streams the KV cache to the chosen decode worker.
- Routes the decode worker's streaming output back to the client.

Policy decisions the router makes:

- **Prefix hit.** If the prompt begins with a known prefix cached on some decode worker (or on a shared KV store — chapter 104 chapter 04), the router may skip prefill entirely and route directly to that decode worker. (This is the DistServe-with-prefix-caching composition.)
- **Decode-worker affinity.** Repeated requests from the same user or session may route to the same decode worker to amortise KV transfer.
- **Priority.** Gold-tier tenants get priority for decode slot admission; silver tenants queue.

### Cross-node KV transfer

The KV cache is the payload. For a 70B model with 80 layers, 8 KV heads (GQA), head_dim 128, 2048-token prompt, BF16:

```
kv_bytes = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_bytes
        = 2 × 80 × 2048 × 8 × 128 × 2 = 640 MiB per sequence
```

For a 128k-token prompt on the same model: ~40 GiB per sequence. Large. The transport library matters:

- **NIXL (NVIDIA Inference Xfer Library).** NVIDIA Dynamo's canonical KV transport; built on UCX; uses GPUDirect RDMA where available (GPU memory → NIC → NIC → GPU memory, bypassing CPU). Peak bandwidth close to the IB NIC's line rate (~400 Gbps per NIC on CX-7).
- **UCX / NCCL point-to-point.** Alternative transports; some engines route KV over NCCL send/recv on the training-style path.
- **TCP fallback.** Last resort. Observable as "KV transfer takes 500 ms for 640 MiB" — the fabric is not being used.

**GPUDirect RDMA requires:**

- IB or RoCE fabric.
- GPUDirect RDMA driver (`nvidia-peermem` module loaded).
- NIC-to-GPU PCIe affinity (same root complex / same NUMA domain, chapter 06).
- Firmware and OFED userspace that match the NIC and driver versions.

If any of these is missing, the KV transfer falls back to CPU-staged copy — 10-100× slower.

### KV transfer timing on the critical path

The KV transfer is on the critical path for TTFT: the first decode token can't start until the KV arrives on the decode worker. Its time budget is:

```
kv_transfer_time ≈ kv_bytes / effective_bandwidth
```

For 640 MiB over 400 Gbps GPUDirect RDMA: ~13 ms. For 40 GiB (128k context) same fabric: ~800 ms — this starts to be a significant chunk of TTFT.

Chapter-level mitigations:

- **Overlap with the first decode step.** Some engines overlap KV transfer with the preparation of the first decode step on the decode worker — mild but real win.
- **Streaming KV transfer.** Transfer KV layer-by-layer as prefill completes each layer, so the decode worker starts receiving before prefill finishes the last layer. More complex; some engines support it.
- **Shorter prompts.** The whole architecture favours moderate-prompt workloads. For 1M-token prompts, the KV transfer *is* the bottleneck; different approaches (streaming, chunked re-prefill on the decode side) are needed.

## When to graduate from chunked-prefill to disaggregation

A decision checklist. **Do NOT disaggregate** when:

- Workload is uniform (prompts and generations in a similar byte range).
- SLO allows the chunked-prefill interleaving on one worker.
- Fleet is small (< 8 workers) — the operational cost of a second pool exceeds the win.
- No IB / RoCE fabric. Disaggregation over TCP is a loss.

**Consider disaggregating** when:

- Long-context workloads dominate (32k+ prompts) and TPOT suffers on the mixed worker.
- Different SLOs for prefill and decode phases (e.g. "TTFT ≤ 300 ms, TPOT ≤ 50 ms" independently).
- Hardware heterogeneity: compute-dense GPUs available for prefill, memory-bandwidth-dense GPUs for decode.
- Workload has strong prefix-sharing (chapter 104) — prefix hits on the decode pool amortise prefill cost dramatically.
- Fleet is large enough (16+ workers) that pool specialisation pays.

**DistServe's goodput framing.** DistServe (Zhong et al., OSDI 2024) formalises the trade-off: *goodput* is the request rate at which both TTFT and TPOT are satisfied. On a flat fleet, the goodput envelope is capped by the worst of the two latencies on the shared worker. On a disaggregated fleet, each pool is optimised for its own SLO; the goodput envelope expands.

## Launching a disaggregated fleet

Below are the shapes; verify flag spellings against your engine's current release.

### NVIDIA Dynamo

Dynamo provides a reference implementation of disaggregated serving with NIXL transport, a KV router, and separate prefill / decode pool configurations. The config shape (simplified):

```yaml
# dynamo-config.yaml (illustrative — verify against the Dynamo docs)
model: meta-llama/Llama-3.1-70B-Instruct
prefill_pool:
  workers: 4
  tensor_parallel_size: 4
  pipeline_parallel_size: 1
  max_batched_tokens: 32768
  gpu_memory_utilization: 0.90
decode_pool:
  workers: 8
  tensor_parallel_size: 2
  pipeline_parallel_size: 1
  max_num_seqs: 256
  gpu_memory_utilization: 0.90
router:
  policy: round-robin                  # or prefix-aware, affinity
  kv_transport: nixl                   # the KV transfer library
  kv_cache_dtype: fp8                  # optional KV quantisation during transfer
```

<!-- needs-research: Dynamo's config shape and defaults rotate; verify against the current Dynamo documentation (https://docs.nvidia.com/dynamo/). The example above is illustrative. -->

### SGLang — PD disaggregation

SGLang has native PD-disaggregated serving (also called P-D split in docs). The launch command (illustrative):

```
# Prefill workers:
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device <ib_hca_name> \
    --port 30000

# Decode workers:
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 2 \
    --disaggregation-mode decode \
    --disaggregation-ib-device <ib_hca_name> \
    --port 30001

# Router (SGLang's `sglang_router`):
python -m sglang.router \
    --prefill-urls http://prefill-0:30000 http://prefill-1:30000 \
    --decode-urls http://decode-0:30001 http://decode-1:30001 \
    --policy prefill-decode-disaggregated
```

<!-- needs-research: SGLang's P-D disaggregation flag spellings and routing policies rotate; verify against the SGLang docs' "PD Disaggregation" section for your release. -->

### vLLM — PD disaggregation (later releases)

vLLM added PD-disaggregated serving in a later release with the `KVTransferConfig` abstraction. The flag shape (verify current):

```
# Prefill:
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' \
    --port 8100

# Decode:
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --kv-transfer-config '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
    --port 8200
```

<!-- needs-research: vLLM's PD-disaggregation configuration (`--kv-transfer-config`, connectors) is new and the API is likely to change. Verify against the pinned release's `distributed-serving` or `kv-transfer` docs. -->

## Pool sizing — a worked example

A fleet to serve a chat workload with:

- Average prompt: 2000 tokens.
- Average generation: 400 tokens.
- Target: 500 QPS, TTFT ≤ 500 ms, TPOT ≤ 50 ms.
- Base model: Llama-3.1-70B BF16.
- Hardware: H100 80 GB, NVLink intranode, 400 Gbps IB internode.

Steps:

1. **Prefill capacity per worker.** TP=4 Llama-70B. Prefill at 2000 tokens / request. Compute: ~2000 × 70e9 × 2 FLOPs / request = ~2.8e14 FLOPs. At H100's ~500 TFLOPs BF16 effective, that's ~0.6 s per request on one GPU, or ~0.15 s on TP=4 (near-linear scaling for prefill). With chunked-prefill batching, aggregated throughput ~15-25 QPS of prefill at target TTFT. Verify empirically.

2. **Decode capacity per worker.** TP=2 Llama-70B. Decode is HBM-bandwidth bound. Per-token read: ~70e9 × 2 bytes = 140 GB. H100 HBM3 ~3.35 TB/s per GPU, TP=2 doubles bandwidth; per-token time ~140 / 6.7 ≈ 21 ms. With continuous batching across many sequences, aggregated throughput improves (one HBM read serves many sequences' tokens for the same layer).

3. **Pool sizing.** 500 QPS × 400 generated tokens / sec per request ÷ (decode-worker aggregated TPS) = decode-worker count. For a 500-generated-tokens workload, each decode worker handling 50 concurrent sequences at ~20 tokens/s per sequence = ~1000 tokens/s. Need (500 × 400 / 1000) = 200 decode-worker-seconds per sec = many parallel sessions on fewer workers; approximate at ~8 decode workers.

4. **Prefill-pool sizing.** 500 QPS of prefill at ~20 QPS per prefill worker = ~25 prefill workers.

Those are napkin numbers. The real sizing is driven by measurement on the actual workload. Chapter-level rule: build the harness, measure, iterate.

## Failure modes

### Failure 1 — KV transfer bandwidth cliff

The disaggregated fleet is deployed; TTFT regressed instead of improved. Diagnosis: `NCCL_DEBUG=INFO` or the Dynamo logs show KV transfer taking 200 ms where it should take 20 ms. Root cause is almost always one of:

- GPUDirect RDMA not enabled (`nvidia-peermem` not loaded).
- NIC-to-GPU affinity wrong (chapter 06).
- IB fabric unhealthy on one port.
- `NCCL_IB_HCA` restricting to a slow HCA.

Fix: verify GPUDirect RDMA is live (`nvidia-smi` lists the NIC under GPUDirect devices on compatible setups); verify `ibv_devinfo` on each node; run `nccl-tests sendrecv_perf` between prefill and decode nodes to measure the baseline. Chapter 04 names expected bandwidth numbers.

### Failure 2 — pool imbalance

Prefill pool is at 20 % utilization; decode pool is at 95 % and queueing. Diagnosis: workload has longer-than-expected generations. Fix: resize pools (more decode workers, fewer prefill workers) — the autoscaling story is module 109, but the pool-level signal (queue depth, utilization per pool) belongs here.

### Failure 3 — router hotspotting

Round-robin router sends sequence patterns that cluster on one decode worker (e.g. a tenant's workload all hashes to decode-0 because the router keyed by tenant). One decode worker saturates; others idle. Fix: audit the router's load-balancing policy; add least-busy or least-concurrent routing; expose decode-worker utilization to the router.

### Failure 4 — prefix hit routed to the wrong pool

A request with a known prefix cached on decode-5 is routed to prefill-3 (fresh prefill); the prefix hit is wasted. Fix: router consults the prefix index before deciding which pool. DistServe-with-prefix-caching discusses the composition; verify your engine supports it or add a prefix-aware pre-router.

### Failure 5 — KV layout mismatch across engine versions

Prefill workers are on vLLM 0.6; decode workers on vLLM 0.7; the KV layout changed between versions. Transfers succeed byte-wise but the decode output is garbage. **Enforce engine-version parity** across the pools. Also enforce model + revision parity (chapter 02 fallback 2 scaled up).

### Failure 6 — abandoned sequences on the decode pool

Clients disconnect; the decode worker continues to generate until it hits the max tokens; HBM is wasted. Need a cancellation path from the router to the decode worker. Verify your engine honours cancellation.

### Failure 7 — long-context pool saturation

A 128k-context workload saturates decode-pool HBM at low batch size (KV cache dominates). The pool runs at 10 % compute utilization but 100 % HBM. Fix: KV quantization (chapter 104 KV FP8/INT8), tiered KV offload (chapter 104 LMCache / Mooncake), or re-sizing with more decode workers.

## Composing with other chapters

- **Chapter 01 (topology selection).** Each pool has its own topology. Prefill often picks higher TP (compute-dense); decode often picks lower TP but more replicas (memory-bandwidth-dense). The decision tree applies twice.
- **Chapter 02 (TP / PP mechanics).** Each pool is launched with the full TP / PP launch mechanics; the health checks of chapter 02 run on each pool independently.
- **Chapter 04 (collective cost).** The KV transfer is the dominant communication cost in disaggregated serving; chapter 04 names the fabric-level bandwidth and latency numbers you compare against.
- **Chapter 05 (MoE).** MoE disaggregation adds EP into each pool. The prefill pool's EP all-to-all + the decode pool's EP all-to-all are both on IB; the design envelope is tighter.
- **Chapter 06 (placement).** The NIC-to-GPU-to-NUMA affinity for GPUDirect RDMA is critical for KV transfer bandwidth. Chapter 06 covers the diagnostic.

## The boundary to the performance peer

This chapter does not teach:

- How NIXL's registration-and-descriptor API minimises descriptor-exchange overhead — library-internal craft.
- How to tune UCX's `UCX_NET_DEVICES` or NCCL's send/recv for a specific message-size curve — algorithm craft.
- How to write a custom KV-transport connector in the engine — kernel-and-driver craft; this is where vLLM's `KVTransferConnector` interface lives.
- How to overlap KV transfer with the decode worker's first-token compute at the kernel level — fusion craft.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) and sometimes at the data-centre networking team. The operator diagnoses "KV transfer slow" from the bandwidth and `nccl-tests` numbers; the kernel / driver / library fix is deferred.

## Summary

- Disaggregated prefill-decode runs two worker pools with a KV router between them. DistServe (OSDI 2024), Splitwise (ISCA 2024), and NVIDIA Dynamo are the three canonical references.
- The architecture: prefill pool → KV router → decode pool, with cross-node KV transfer over NIXL / UCX / NCCL on IB / RoCE.
- Each pool has its own TP / PP topology; prefill tends compute-dense, decode tends bandwidth-dense.
- GPUDirect RDMA is the fast path for KV transfer; its enablement checklist is a chapter-06 placement concern.
- The graduation signal: long context + mixed SLOs + ≥ 16-worker fleet. Below that, chunked prefill (module 103) is enough.
- Deploy on Dynamo, SGLang's PD mode, or vLLM's `--kv-transfer-config`; verify flag spellings against the current release.
- Common failures are KV-transport bandwidth cliff, pool imbalance, router hotspots, prefix-hit routed to the wrong pool, KV layout mismatch across versions, abandoned sequences, and long-context pool saturation.
- Kernel-level questions (NIXL internals, UCX / NCCL algorithm tuning, kernel fusion with KV transfer) route to the performance peer.
