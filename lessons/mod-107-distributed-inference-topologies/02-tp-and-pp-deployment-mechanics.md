# Chapter 02 — Tensor-Parallel and Pipeline-Parallel Deployment Mechanics

> Chapter 01 picked the topology. This chapter actually deploys it — the shard geometry under TP, the stage layout under PP, the launch flags, the NCCL requirements, and the health checks that confirm the engine is doing what you told it to.

## Motivation

Operators who first touch TP / PP serving almost always encounter the same two surprises:

1. **The engine quietly fell back to a different topology.** The launch flag said `tensor_parallel_size = 8` but one GPU in the group has a different driver, the NCCL communicator fails to build, and the engine starts in TP = 1 with one GPU — serving at 1/8 the expected throughput for days before someone notices.
2. **The topology ran, but the collectives did not go over NVLink.** `nvidia-smi topo -m` would have shown that two of the eight GPUs are on a different PCIe root complex; the NCCL path for those pairs is PCIe-via-host-memory, not NVLink; the per-layer AllReduce takes 20× longer than the specs promised.

Both are visible with the right health check at startup. Neither needs a kernel edit. This chapter writes the health checks and the launch procedure.

By the end of it you can:

- Draw the TP shard geometry for the attention and MLP projections — which dimension is sharded, which collective closes it, where the KV cache is sharded by head.
- Draw the PP stage boundary — which layers each stage holds, what the activation hand-off looks like, where the bubble comes from.
- Produce the launch command for vLLM / SGLang / TRT-LLM with the TP and PP flags and the NCCL environment variables for the fleet's interconnect.
- Verify the engine is actually sharding (not replicating), that NCCL built the right topology, and that `nvidia-smi topo -m` matches the sharding plan.
- Recognise the three "silent fallback" modes and the health checks that catch them.

<!-- needs-research: engine-specific launch-flag spellings change across releases (vLLM, SGLang, TensorRT-LLM). Verify the examples below against your pinned engine's `--help` output and release notes before committing to a config. -->

## TP shard geometry

A transformer layer has two large weight matrices per attention block (QKV projection + output projection) and two per MLP block (gate + up on the input side, down on the output side). TP shards each along one dimension; the shard choice is set by what collective closes the layer.

### Attention projection — column-sharded in, row-sharded out

The QKV projection is **column-parallel** (sharded along the output dimension). Each rank holds a slice of the heads:

```
Rank r holds Q_r, K_r, V_r corresponding to its assigned head slice.
Input X is replicated across ranks (no communication at input).
Each rank computes Q_r = X · W_q^r (and similarly K_r, V_r) — no communication.
```

The KV cache under TP is **sharded by head** — each rank stores KV for only its heads. This divides the KV footprint by TP degree. Chapter 104's KV byte-math composes directly.

The output projection `W_o` is **row-parallel** (sharded along the input dimension). Each rank computes a partial result; an **AllReduce** combines the partials into the final output. One AllReduce per attention block.

```
Rank r computes: AttnOut_r = Attn_r · W_o^r → partial result
AllReduce across ranks → AttnOut (full)
```

### MLP — column-parallel in, row-parallel out

The up-projection and gate-projection are **column-parallel**. Each rank computes `up_r = X · W_up^r` and `gate_r = X · W_gate^r` — no communication, each rank holds its slice of the hidden dim.

The activation (SiLU, GELU, or SwiGLU) is applied per rank — no communication.

The down-projection is **row-parallel**. Each rank computes a partial; an **AllReduce** closes the layer:

```
Rank r computes: hidden_r = activation(up_r) * gate_r (SwiGLU variant)
Rank r computes: MLP_out_r = hidden_r · W_down^r → partial
AllReduce across ranks → MLP_out (full)
```

Two AllReduces per transformer layer — one for the attention block, one for the MLP block.

### Sequence parallel — the AllGather + ReduceScatter variant

Vanilla TP has an AllReduce at the attention output and at the MLP output. Each AllReduce communicates `batch × seq × hidden × bytes` of data.

**Sequence parallel** (introduced in the Megatron-LM paper — see `resources.md`) splits the layer-norm, dropout, and residual tensors along the sequence dimension across the TP ranks. The result: AllReduce decomposes into AllGather + ReduceScatter, which have the same total bytes-communicated per rank as AllReduce but allow the layer-norm / residual to be done on only a slice of the sequence — saving activation memory.

Most inference engines expose this as a flag ("`--enable-sequence-parallel`" or equivalent in training; inference engines inherit the pattern). At inference time the activation-memory saving matters at long context; the collective cost is unchanged.

<!-- needs-research: sequence-parallel at inference is not uniformly exposed across inference engines. Verify your engine's docs; some apply it internally without a flag, some do not implement it. -->

### Which TP degrees work

TP degree must divide:

- The number of attention heads (so each rank holds a whole number of heads).
- The hidden dimension and intermediate dimension (so column-parallel and row-parallel splits are integer-sized).

Typical valid degrees: 1, 2, 4, 8 for Llama-architecture 7B / 8B / 13B / 70B. 405B's attention heads also admit TP=8 cleanly. Degree 16 is possible for models with large head counts; verify against the model's config.json (`num_attention_heads`, `num_key_value_heads` for GQA).

For GQA models, the KV-head count can be lower than the Q-head count. TP shards across KV heads first (so each rank holds a whole KV head); TP > num_kv_heads is not supported cleanly in standard engines.

## PP stage layout

Pipeline parallel cuts the transformer along layers. For an `L`-layer model and `K` pipeline stages:

- Stage 0 holds layers `[0, L/K)` plus the input embedding.
- Stage k holds layers `[k·L/K, (k+1)·L/K)` — a contiguous block.
- Stage K-1 holds the last block plus the final layer-norm and the LM head.

### Hand-off between stages

Between stages, the activation tensor crosses a node boundary (usually; PP is typically used across nodes). The hand-off is a **send/recv** pair:

- Stage `k` runs micro-batch `m` through its layers; produces activations `A_{k,m}`.
- Stage `k` sends `A_{k,m}` to stage `k+1`.
- Stage `k+1` receives, continues computation.

The payload per hand-off: `micro_batch_size × seq_len × hidden_dim × bytes`. For a 70B model with hidden_dim 8192, micro-batch 8, seq_len 2048 at BF16: `8 × 2048 × 8192 × 2 = 256 MiB`. This crosses IB / RoCE at a one-time cost per micro-batch per stage boundary — tens of microseconds to a millisecond.

### Micro-batching and the bubble

Inference PP typically uses a forward-only pipeline (training uses 1F1B or interleaved schedules). The simplest shape:

```
Request batch of B inputs.
Split into M micro-batches of size B/M.
At t=0: stage 0 runs micro-batch 0, stages 1..K-1 idle.
At t=1: stage 0 runs micro-batch 1, stage 1 runs micro-batch 0, stages 2..K-1 idle.
At t=K-1: stage 0 runs micro-batch K-1, ..., stage K-1 runs micro-batch 0.
At t=M+K-2: last micro-batch exits stage K-1.
```

**Bubble fraction** = (K - 1) / (M + K - 1). For K=2 and M=16, bubble is ~6 %; for K=4 and M=8, bubble is ~27 %. At K=8 and M=8, bubble is 47 % — the serve loop runs half-empty.

The implication for TTFT: the **first token** of the first request in a batch waits for the first micro-batch to traverse all K stages. This adds `(K - 1) × per_stage_time` to that request's TTFT. At K=2, it's one extra per-stage time — tolerable. At K=4, it's three extra — SLO-breaking for tight latency targets.

### 1F1B at inference

Some engines implement a 1F1B-like forward schedule that overlaps multiple micro-batches across stages, maximising steady-state throughput at the cost of complicated scheduling. Verify your engine's PP implementation documents the schedule it uses; the operator-visible knobs are typically `--pipeline-parallel-size` and (sometimes) `--micro-batch-size`.

<!-- needs-research: inference-PP implementations in vLLM, SGLang, and TRT-LLM each have different schedulers and different maturity levels. vLLM added PP support in a later release; verify the current state. SGLang's PP maturity varies by release. TRT-LLM has had PP since early releases. -->

## Launch mechanics

### vLLM — TP within one node

```
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

This spawns 4 worker processes (one per GPU) within one node, builds a NCCL communicator across them, and shards the model. Verify:

- Startup log shows `tensor_parallel_size = 4` and the per-worker footprint matches the shard byte-math (70B ÷ 4 shards ≈ 35 GiB per rank at BF16).
- `nvidia-smi` shows 4 GPUs busy, each with similar HBM usage.
- `vllm` metrics or the engine's status endpoint shows all 4 ranks healthy.

### vLLM — TP + PP across nodes

```
# Node 0 (head):
VLLM_HOST_IP=<node-0-ip> \
VLLM_PORT=<port> \
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --dtype bfloat16 \
    --max-model-len 8192

# Node 1 (worker — joins the Ray cluster):
ray start --address=<node-0-ip>:<ray-port>
```

Required: a Ray cluster (or `mp` with equivalent orchestration) across both nodes, a working IB / RoCE fabric for the inter-stage send/recv, and NCCL that knows about the IB fabric.

Verify:

- Startup log reports `pipeline_parallel_size = 2`, `tensor_parallel_size = 8`, total world size 16.
- Each node reports 8 workers (one per GPU) that joined the Ray cluster.
- A test request completes end-to-end and returns tokens.

<!-- needs-research: vLLM's exact PP+TP launch mechanics (Ray vs. mp, env-var spellings, port layout) rotate across releases. Verify against the current `vllm serve --help` and the `distributed-serving.md` doc for your pinned version. -->

### SGLang — TP

```
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 30000
```

Verify as above.

### TRT-LLM — TP + PP via build-time

TRT-LLM compiles a *fixed* TP + PP topology at build time:

```
# Build:
trtllm-build \
    --checkpoint_dir <converted-ckpt> \
    --output_dir <engine-dir> \
    --tp_size 4 \
    --pp_size 2 \
    --max_batch_size 32 \
    --max_input_len 4096

# Serve via Triton + tensorrtllm_backend; the engine .plan files
# are loaded by MPI launchers across TP*PP ranks.
```

The TP and PP sizes are baked into the engine; changing them means a rebuild. This is a cost compared to vLLM / SGLang (which allow runtime TP changes with a restart but no rebuild).

<!-- needs-research: TRT-LLM's build process has been refactored several times. The specific flag spellings and converter scripts rotate; verify against the TRT-LLM release notes and the LLM-API docs for your pinned version. -->

## NCCL environment and health checks

NCCL is the collective library every engine uses on NVIDIA hardware (AMD has RCCL as a drop-in). Its environment variables control topology discovery, algorithm selection, and interconnect binding. The operator-level knobs:

### `NCCL_DEBUG=INFO`

Prints what NCCL is doing at startup — critical for first-time deployments. Key lines to look for:

```
NCCL INFO Topology detected: ...
NCCL INFO Channel 00/00 : ...  → the ring/tree channels
NCCL INFO Launch mode Parallel
NCCL INFO Using network IB      (or Socket, if IB didn't bind)
```

If NCCL falls back to `Socket` on a fleet that was supposed to use IB, something is wrong — the IB driver isn't loaded, the OFED stack isn't configured, or `NCCL_IB_HCA` is restricted. Collective cost will be dominated by TCP overhead; chapter 04's cost model does not apply.

### `NCCL_IB_HCA`

Which IB HCAs to use. Default is "all." Can restrict to specific HCAs for NIC-to-GPU pairing (see chapter 06).

### `NCCL_SOCKET_IFNAME`

Which Ethernet interface to use for bootstrap and (if applicable) data transport. If multiple interfaces exist on a node, pick the right one; otherwise NCCL may use a management interface.

### `NCCL_P2P_LEVEL`

Which peer-to-peer path NCCL may take. Common settings:

- `NVL` — require NVLink for intra-node P2P. Fails if a pair of GPUs is not NVLink-connected.
- `PIX` — allow PCIe switch P2P (same PCIe switch).
- `PHB` — allow P2P across PCIe host bridges (slower, PCIe host memory copy).
- `SYS` — allow P2P anywhere, including via host memory.

Setting `NCCL_P2P_LEVEL=NVL` is a strong guard against the "collective falls back to PCIe" error — the communicator fails loudly at build time rather than quietly later.

### `NCCL_ALGO`, `NCCL_PROTO`

Algorithm (Ring / Tree / CollNet) and protocol (LL / LL128 / Simple). Operator rarely sets these; defaults are usually right. The performance peer tunes.

### Startup health checks

Three checks at every TP / PP deployment startup:

1. **`nvidia-smi topo -m`** on each node. Verify every GPU-pair the engine will shard across has `NV#` (NVLink) in the matrix, not `PIX` or `SYS`. If `SYS` appears between two GPUs that will be in the same TP group, that's a placement problem (chapter 06).
2. **`NCCL_DEBUG=INFO`** at engine startup. Capture the first ~500 lines of NCCL log and inspect for `Using network IB` (not `Socket`), channel count matches TP degree, and no `WARN` entries.
3. **`nccl-tests all_reduce_perf`** on the exact nodes and GPU ids the engine will use. For TP=8 on one node, run `all_reduce_perf -b 1M -e 1G -g 8`. Expect bus-bandwidth within 70-90 % of NVLink spec (chapter 04 names exact numbers). If measured bandwidth is 2-5× below spec, the NCCL path is wrong (PCIe fallback, bad channel count, driver mismatch) — fix before serving.

## Silent-fallback failure modes

### Fallback 1 — world size mismatches at launch

Launch flag `--tensor-parallel-size 8` but only 7 GPUs are visible (one failed health-check at the orchestrator). The engine may hang waiting for the 8th worker, or (worse) fall back to a smaller world size silently depending on the engine. **Mitigation:** pre-launch GPU health check (`nvidia-smi` returns 8 healthy GPUs); fail the launch if not.

### Fallback 2 — PCIe path instead of NVLink

The GPUs are "there," but two of them are on a different PCIe root complex (common on dual-socket servers with asymmetric GPU-to-CPU wiring). NCCL finds no NVLink between those GPUs and routes via PCIe host memory copy. Observable in `NCCL_DEBUG=INFO` as `via SYS` instead of `via NVL`. The AllReduce bus-bandwidth drops by an order of magnitude. **Mitigation:** `NCCL_P2P_LEVEL=NVL` fails fast; verify `nvidia-smi topo -m`; use NUMA-aware placement (chapter 06).

### Fallback 3 — IB not bound, falls back to TCP

The fleet has IB installed, but the OFED userspace isn't configured right on one node. NCCL's inter-node transport falls back to TCP. The per-collective latency rises by 10-100×. Observable in `NCCL_DEBUG=INFO` as `Using network Socket`. **Mitigation:** verify `ibv_devinfo` returns healthy HCAs on every node; verify `NCCL_SOCKET_IFNAME` doesn't accidentally direct NCCL away from IB; fail the launch if `NCCL_DEBUG=INFO` reports Socket.

### Fallback 4 — engine ran TP=1 instead of TP=N

A rare but observed failure: the engine's launcher spawned one worker (not N) because an env var was wrong. The one worker holds the full model (or OOMs). Startup log says `tensor_parallel_size = 1` even though the flag said 4. **Mitigation:** always read back the startup log's reported TP / PP values, not the flag — they can diverge.

### Fallback 5 — some ranks ran, others silently died

The engine spawned N workers; one died early (CUDA OOM, driver error). The remaining ranks hang in the collective because they can't reach the dead rank. Observable as all ranks "working" for 5 minutes (NCCL retry) then a cascade failure. **Mitigation:** inspect per-rank logs at startup, not just the head rank's; alert on any rank failing within the first minute.

## The composition with chapter 103 — continuous batching under TP

Chapter 103's continuous-batching scheduler composes directly with TP: the scheduler on rank 0 (typically the driver rank) decides the batch, broadcasts the per-iteration batch metadata to all ranks (small payload), and each rank runs its shard of the forward pass over the full batch. The chunked-prefill knobs (chapter 103 chapter 03) continue to work. The *collective overhead* adds to iteration time — this is the TP cost that chapter 04 quantifies.

Under PP, the scheduler also lives on rank 0; micro-batches flow through stages; iteration time per micro-batch is the per-stage time; aggregated iteration throughput is one micro-batch per stage per unit time once the pipeline is primed. The scheduler needs to produce enough micro-batches to keep the pipeline full — small-batch workloads under PP under-use the pipeline.

## The composition with chapter 104 — KV cache under TP

The KV cache is sharded by head along TP. For a model with `num_kv_heads` KV heads and TP degree `t`, each rank holds `num_kv_heads / t` KV heads for every request.

```
per_rank_kv_bytes = (num_layers × 2 × seq_len × num_kv_heads/t × head_dim × kv_dtype_bytes) per request
```

The chapter-104 PagedAttention block table is per-rank (each rank manages its slice). The engine assigns blocks on all ranks for the same request simultaneously; eviction is coordinated. The operator rarely tunes this directly, but the KV budget per rank is `(total KV budget) / TP`.

## Where the engine's distributed executor hooks in (very briefly)

For context on the engine side (deep internals belong to module 102):

- **vLLM.** `Ray`-based or `mp`-based (multi-processing) distributed executor. One worker per rank, process communication over NCCL. The scheduler lives on the driver worker; it broadcasts per-iteration metadata via Ray / mp primitives.
- **SGLang.** Native multi-process TP; one worker per GPU; NCCL for collectives. The scheduler is centralised.
- **TRT-LLM.** MPI-launched; one process per rank. The TRT engines on each rank are built with the TP / PP topology baked in.

For deeper internals, see module 102 chapters 01-03. This chapter stays at operator depth.

## The boundary to the performance peer

This chapter does not teach:

- How the TP-sharded Flash Attention kernel composes with the attention AllReduce (fused epilogue? separate kernel? CUDA-graph'd?) — kernel craft.
- How NCCL's ring algorithm maximises bus bandwidth or when the tree algorithm is better — algorithm craft.
- How to overlap the AllReduce with the next layer's GEMM in the forward pass — fusion craft.
- How the TP shard of SwiGLU / RMSNorm / rotary embedding is kernel-fused — kernel craft.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). The operator who runs the TP / PP topology is not expected to open these kernels. The operator *is* expected to verify the topology is doing what the flag said, that NCCL built on the right fabric, and that `nvidia-smi topo -m` matches the plan — chapter 06 formalises the diagnostic flowchart.

## Summary

- TP shards attention and MLP along the output / input dimensions respectively; two AllReduces close each transformer layer. The KV cache is sharded by head. TP degree must divide num_heads (and num_kv_heads for GQA).
- Sequence-parallel decomposes each AllReduce into AllGather + ReduceScatter at the layer-norm boundary, saving activation memory at long context; collective bytes/rank are unchanged.
- PP cuts layers into stages. The hand-off between stages is a send/recv per micro-batch; the bubble is `(K-1)/(M+K-1)`; TTFT absorbs `(K-1) × per_stage_time` at pipeline fill.
- Launch mechanics differ by engine: vLLM via Ray / mp runtime flags; SGLang native multi-process; TRT-LLM baked at build time.
- NCCL environment variables (`NCCL_DEBUG`, `NCCL_IB_HCA`, `NCCL_P2P_LEVEL`, `NCCL_SOCKET_IFNAME`) control topology discovery and binding; the operator sets them, the performance peer tunes the algorithms.
- Three startup health checks: `nvidia-smi topo -m`, `NCCL_DEBUG=INFO` log inspection, `nccl-tests` run on the actual fleet layout.
- Five silent-fallback modes to catch: world-size mismatch, PCIe fallback instead of NVLink, IB → Socket fallback, TP=1 instead of TP=N, and partial-rank failure.
- Kernel-level questions (algorithm, fusion, overlap, Flash-Attention-with-TP) route to the performance peer.
