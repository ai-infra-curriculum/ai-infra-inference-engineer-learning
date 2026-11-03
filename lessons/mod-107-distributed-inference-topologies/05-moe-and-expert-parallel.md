# Chapter 05 — MoE Serving: Expert-Parallel Topology and Routing Imbalance

> Dense models at this scale fit a straightforward playbook. MoE serving pulls two levers the dense playbook doesn't have — routing (which experts a token picks) and expert distribution (which rank hosts which experts). Both levers interact with the all-to-all collectives that dominate the critical path at multi-node EP. This chapter operates them without re-implementing the dispatch kernels.

## Motivation

A platform team receives a request to serve Mixtral 8×7B at 1000 QPS. The engine spins up with `--tensor-parallel-size 8` and no EP; HBM is tight but the model fits. Serving starts; throughput is a fraction of the expected peak; expert utilization monitor shows experts 3 and 5 running 80 % of the time and expert 7 running 4 % of the time. The team adds `--enable-expert-parallel` and `--ep-size 4`; throughput improves 2.3×. A month later a tenant's new workload triggers a token-routing shift; two experts become hot; throughput regresses; the team raises the capacity factor; some tokens are dropped at the capacity cap; quality on edge-case prompts degrades. The team moves to `balanced-routing` with a slightly larger capacity factor; the quality recovers and throughput stays. All of this is operator-depth work — none of it is a kernel edit.

MoE serving is not a dense model with a few extra flags. It is a parallelism and routing problem whose knobs sit next to each other, and whose default settings are rarely optimal for a production workload. By the end of this chapter you can:

- State the EP topology — experts partitioned across ranks, dispatch and combine all-to-alls per MoE layer, how it composes with TP for the attention and shared layers.
- Reason about token-routing imbalance (hot experts, cold experts, capacity drop) and name the mitigations.
- Pick the EP degree and the capacity factor for a given fleet shape.
- Deploy Mixtral, DeepSeek-V2 / V3, and Qwen-MoE on vLLM / SGLang / TensorRT-LLM at operator depth.
- Diagnose the three common MoE failure modes (expert hotspotting, all-to-all bandwidth cliff, capacity-factor drops) and the operator fixes.
- Cite the kernel boundary to the performance peer — the dispatch / combine kernels, the gating softmax, the expert-GEMM batching are kernel craft and route to that peer.

<!-- needs-research: MoE serving is one of the fastest-changing areas in production inference. DeepSeek V3's deployment patterns, SGLang's EP support, vLLM's expert-parallel implementation, and the DeepEP / Hopper-specific all-to-all kernels have all changed within the last year. Verify every specific example against pinned release notes. -->

## The MoE layer structure

In a transformer MoE, every MoE layer replaces the dense MLP with:

```
y = router(x) → top-k experts per token, with weights
y_i = sum over selected experts e of w_i,e * MLP_e(x_i)
```

Specifically:

- **Router / gating network.** A small linear + softmax that scores each token against each expert, selects the top-k experts (k is 1, 2, or more depending on the model), and produces mixing weights.
- **Experts.** `E` independent MLP blocks, each with its own `W_up`, `W_gate`, `W_down`. For Mixtral: `E=8`, `top-k=2`. For DeepSeek V3: `E=256` fine-grained routed experts + 1 shared expert, `top-k=8`. For Qwen-MoE variants: varies.
- **Shared experts.** Some architectures (DeepSeek V2 / V3, some Qwen variants) have one or more "shared experts" that process *every* token in addition to the routed experts. Shared experts are structurally like a dense MLP; EP typically does not split them (they're replicated across ranks or TP-sharded).
- **Combine step.** Mix the selected experts' outputs via the router weights.

The compute per token is `top-k × expert_MLP_cost`. For Mixtral top-k=2, each token consumes 2 experts' worth of compute per MoE layer. For DeepSeek V3 top-k=8 (plus the shared expert), 9 experts' worth. Compared to a dense MLP of the same hidden-size, this is a fraction (the dense MLP compute is roughly `E` experts' worth per layer, so top-k=2 of E=8 is 25 %; top-k=8 of E=256 is 3.1 %).

The compute savings come from routing. The infrastructure cost comes from the collective that implements routing — the all-to-all.

## The expert-parallel topology

Expert parallel partitions the `E` experts across `P` ranks. Each rank holds `E / P` experts (assuming even partitioning). For Mixtral 8-expert MoE with EP=4: 2 experts per rank. For DeepSeek V3 256-expert MoE with EP=32: 8 experts per rank.

Each MoE layer:

1. **Gate.** Every rank computes gating for every token in its local batch (gating weights are replicated).
2. **Dispatch all-to-all.** Each rank sends tokens to the ranks that host the experts those tokens picked. Rank `r` receives tokens that selected its local experts.
3. **Expert compute.** Each rank runs its local experts on the tokens it received.
4. **Combine all-to-all.** Each rank sends expert outputs back to the ranks where the tokens originated.
5. **Weighted sum.** Each origin rank combines the top-k outputs with the router weights.

The two all-to-alls per layer are the critical-path collectives.

### Composition with TP

The dense portions of the model (attention projections, embedding, LM head, shared experts) are TP-sharded within each EP rank's local group. The MoE layer's expert FFNs are *not* TP-sharded — each expert is replicated or sharded within its rank's local resources but the inter-rank partition is EP, not TP.

A plausible shape for DeepSeek V3 serving on 4 nodes × 8 H100:

```
World size = 32
TP = 8 (within each node)   — for attention + shared expert
EP = 32 (across all GPUs)   — for routed experts (one node of 8 ranks holds 64 experts; 4 nodes hold 256)
PP = 1
```

Vendor-recommended shapes may differ; verify in the deployment references.

<!-- needs-research: specific DeepSeek V3 serving topologies rotate; verify in the SGLang DeepSeek-V3 docs, vLLM DeepSeek guide, and the DeepSeek-V3 technical report. -->

### Composition with DP

For small MoE models at high concurrency (Mixtral at modest QPS), **DP replicate** is simpler than EP — each worker holds all 8 experts; no all-to-all. The decision is identical to the dense DP vs. TP decision (chapter 01): if the model fits comfortably on one worker, DP is simpler.

For large MoE models (DeepSeek V3), EP is non-optional because the model doesn't fit on one worker.

## The all-to-all cost — a chapter-04 recap applied

For a batch of `T` tokens routed to `E` experts with top-k=k, the dispatch all-to-all moves tokens' hidden states:

```
dispatch_bytes_per_rank = (T × k × hidden × bytes) / P  (balanced-routing case)
```

Across the EP group of size `P`, each rank sends to `(P-1)` others; the per-rank sent/received bytes:

```
per_rank_sent ≈ dispatch_bytes_per_rank × (P-1) / P
```

For DeepSeek V3 EP=32, hidden=7168 (DeepSeek V3 hidden dim; verify), T=1024 prefill tokens, k=8, BF16:

```
dispatch_bytes_per_rank ≈ (1024 × 8 × 7168 × 2) / 32 ≈ 3.6 MiB
per_rank_sent ≈ 3.6 × 31/32 ≈ 3.5 MiB
```

Over IB at 50 GB/s per rank, each dispatch all-to-all takes `~70 µs` + latency floor. Two per MoE layer × ~61 MoE layers in DeepSeek V3 = ~8.5 ms per iteration *just for dispatch/combine* at this batch size.

**Imbalanced routing** makes the per-rank traffic skew — some ranks send/receive more than others. The slowest rank sets the critical path (chapter 06 names this as the straggler pattern). The capacity factor is the knob that bounds the skew.

### The capacity factor

Not every expert receives exactly `T × k / E` tokens — the routing is data-dependent. If an expert gets more than its share, the kernel either:

- **Drops** the excess tokens at the capacity cap (quality cost).
- **Reroutes** them to other experts (compute cost).
- **Pads and permits** the overflow (memory cost).

The **capacity factor** (CF) multiplies the per-expert allocation: `capacity_per_expert = CF × T × k / E`. CF=1 is tight (any imbalance drops); CF=1.25 is a common conservative default; CF=1.5-2 absorbs more imbalance at the cost of memory.

At inference time, token drop is almost always worse than compute cost — a dropped token's expert slot is wasted and the model may produce incorrect outputs. **Pick a CF large enough that drops are rare** at your workload's imbalance profile. Measure (see "triage" below) before tuning.

## The three canonical MoE deployment references

### Mixtral 8×7B (Mistral AI, 2024)

- 32 layers; MoE in every layer; `E=8`, `top-k=2`.
- Total params ~47B (not 56B — shared projections aren't duplicated per expert).
- BF16: ~94 GiB — doesn't fit on one H100; fits on 2× H100 (TP=2) or 4× H100 (TP=4) comfortably.
- Serving in vLLM / SGLang: works out of the box with `--enable-expert-parallel` or equivalent; EP degree bounded above by `E=8`.
- Common shape: TP=4, EP=2 within one node; DP across replicas.
- Experts are big enough that per-expert utilization matters; a routing-skew workload (same user context pattern dominates) shows the imbalance clearly.

### DeepSeek-V2 (DeepSeek, 2024)

- Introduced MLA (Multi-head Latent Attention) — a KV-compression scheme that reduces KV footprint dramatically. KV ends up ~5-10× smaller than standard MHA. Chapter 104 composes.
- Fine-grained + shared experts (`E=160` routed + 2 shared in V2).
- ~236B total params; ~21B activated per token.
- Fits one node of 8× H100 at FP8 with TP + EP.
- `top-k=6`.

### DeepSeek-V3 (DeepSeek, 2024)

- Scaled V2: `E=256` + 1 shared, `top-k=8` with a sigmoid-gated router, auxiliary-loss-free load balancing.
- ~671B total; ~37B activated per token.
- Needs multi-node. Vendor-recommended serving uses TP + EP + occasionally PP.
- MLA kept from V2.
- EP all-to-all is the dominant communication cost at serving scale.

The DeepSeek-V3 technical report (arXiv:2412.19437) names specific serving configurations; the SGLang DeepSeek-V3 guide and the vLLM DeepSeek-V3 pages document the deployment flags. **Verify against those references, not the examples below.**

### Qwen-MoE (Alibaba)

- Qwen1.5-MoE-A2.7B: smaller MoE with 60-expert configuration.
- Qwen2 / Qwen2.5 MoE variants: varying expert counts.
- Deployment patterns similar to Mixtral for the smaller variants; similar to DeepSeek for the larger ones.
- Qwen's gating scheme has evolved across variants (verify in the model card).

<!-- needs-research: specific Qwen-MoE deployment numbers (expert count, top-k, serving topology) rotate across model releases. Verify against the Qwen model card for your pinned variant. -->

## Deployment examples

### vLLM — Mixtral

```
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --dtype bfloat16
```

vLLM handles Mixtral's MoE routing internally; EP is implicit (the expert ops are batched within each TP rank). Explicit `--enable-expert-parallel` is a newer feature; verify.

<!-- needs-research: vLLM's MoE handling has evolved. In older releases, Mixtral was served as a specialised dense-like path; in newer releases, explicit `--enable-expert-parallel` flags exist. Verify for your pinned version. -->

### SGLang — DeepSeek V3

```
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --ep-size 32 \
    --mem-fraction-static 0.9 \
    --disable-cuda-graph    # or enable if supported for MoE
```

SGLang's DeepSeek-V3 support includes the DeepEP integration — a specialised all-to-all kernel optimised for Hopper. Enabling it may require specific flags; verify against the SGLang DeepSeek-V3 serving guide.

<!-- needs-research: SGLang's EP flags (`--ep-size`, `--enable-ep-moe`, `--enable-deepep-moe`, etc.) have changed. Verify. -->

### TensorRT-LLM — Mixtral / DeepSeek

TRT-LLM has MoE support with specific plugin builds. Example (verify flag spellings):

```
trtllm-build \
    --checkpoint_dir <converted-ckpt> \
    --moe_plugin bfloat16 \
    --output_dir <engine-dir> \
    --tp_size 4 \
    --max_batch_size 16
```

Serving via Triton + `tensorrtllm_backend`. The MoE plugin configures expert distribution at build time; runtime EP changes require rebuild.

<!-- needs-research: TRT-LLM MoE plugin flags rotate; verify. -->

## Measuring per-expert utilization

The critical operator-level metric for MoE serving. Measurement approaches:

### Engine-exposed metrics

Most inference engines expose per-expert routing counts in metrics (name varies: `expert_hits`, `router_decisions`, `moe_tokens_per_expert`). Scrape these; produce a per-expert histogram.

### Expected shape

A healthy routing has moderate variance across experts — some are 1.5× others; none are 10× any other, and none are consistently idle. Over hundreds of thousands of tokens, routing should trend toward the auxiliary-loss-free balance (for DeepSeek-V3 style) or the auxiliary-loss-balanced distribution (for older MoEs).

### Unhealthy shapes

- **Hot expert.** One expert gets 3-5× its share. Dispatch all-to-all is bottlenecked at the rank hosting it.
- **Cold expert.** One expert gets < 20 % of its share. Capacity on that rank is wasted.
- **Workload-specific skew.** A tenant's workload consistently routes to the same 2-3 experts. This is the hardest case; chapter 06 (placement) composes.

### The capacity-factor dial

If measurement shows the imbalance is small and routing is close to uniform: capacity factor can be tight (1.1-1.2) with little drop.

If measurement shows a hot expert pattern: either raise capacity factor to absorb (memory cost), or rebalance by expert redistribution (see below), or accept the drops (quality cost).

## Mitigations for routing imbalance

### Mitigation 1 — raise capacity factor

Simplest. Trades memory and all-to-all bytes for drop-rate. Works for mild imbalance.

### Mitigation 2 — balanced routing variants

Some engines expose a "balanced-routing" mode that re-routes overflow tokens to the next-best expert. Mimics training-time load balancing at inference. Marginal quality cost; moderate throughput cost.

### Mitigation 3 — expert redistribution

Rearrange which experts live on which ranks so hot experts are not clustered on one rank. For DeepSeek-V3, the research literature on "expert placement" and "expert redistribution" discusses this; some engines expose it as a config knob.

<!-- needs-research: expert-redistribution support in vLLM / SGLang / TRT-LLM rotates. Verify for your pinned version. -->

### Mitigation 4 — speculative / shadow experts

Advanced: replicate the hottest experts across ranks so the all-to-all can land tokens on multiple candidate ranks. Research-grade; occasionally in production.

### Mitigation 5 — workload-aware routing hint

Per-tenant workload patterns can be analysed offline; if a tenant's workload routes heavily to experts A, B, C, the router can prefer to co-locate that tenant's requests on ranks that host A, B, C. Chapter 06 composes.

## Failure modes

### Failure 1 — EP all-to-all over slow fabric

Deployed EP=32 across 4 nodes; `nccl-tests alltoall_perf` shows per-rank bus-bw at 10 GB/s where it should be 40 GB/s. Dispatch and combine all-to-alls dominate iteration time; TPOT inflates. **Diagnosis:** chapter 04 flow. **Fix:** NIC-to-GPU affinity (chapter 06); verify GPUDirect RDMA is enabled; verify `NCCL_IB_HCA` is sane.

### Failure 2 — capacity drops under load

A tenant's workload shows degraded output quality. `moe_tokens_dropped` metric > 0. Capacity factor too tight for the imbalance. **Fix:** raise CF; if too much memory, mitigation 2 or 3.

### Failure 3 — serving on dense-model code path

The engine was built without EP support and treats Mixtral as 8× dense MLP concatenated — no routing. Throughput is 4× worse than expected. **Fix:** `--enable-expert-parallel` or equivalent; verify startup log reports MoE path active.

### Failure 4 — expert hot-spotting on one rank

One rank consistently has 2× the dispatch traffic. Diagnosis: per-expert routing histogram shows experts on that rank are hot. **Fix:** mitigation 3 (redistribution) or mitigation 4 (replication).

### Failure 5 — shared-expert bottleneck

DeepSeek V2/V3 has shared experts that process every token. If the shared expert is TP-sharded across the full EP group, its collective overhead can exceed the routed-expert collective. **Fix:** place shared experts on a smaller TP group; verify in the engine's config.

### Failure 6 — DeepEP / specialised kernel not loaded

The engine supports a Hopper-optimised all-to-all kernel (DeepEP for DeepSeek V3) but it isn't loaded (missing dependency, wrong GPU arch, flag not set). All-to-all falls back to a generic NCCL path. **Fix:** verify kernel-level feature flags; chapter 06 composes.

### Failure 7 — mixed EP and TP that violate math constraints

EP degree doesn't divide the expert count evenly; some ranks have more experts than others; one rank straggles. **Fix:** pick EP degree so `E mod EP == 0`.

### Failure 8 — gating in FP32 vs. BF16 mismatch

The router / gating network's dtype differs from the serving dtype; downcast errors in the router produce routing instability (sticky picking of certain experts). **Fix:** verify router dtype in the engine's config.

## Composition with other chapters

- **Chapter 01 (topology selection).** EP is a required axis for MoE; the decision tree branches on `is_moe`.
- **Chapter 02 (TP / PP mechanics).** EP composes with TP; the launch flags include both.
- **Chapter 03 (disaggregation).** Disaggregating MoE adds EP all-to-all into both pools; the design envelope is tighter.
- **Chapter 04 (collectives).** The all-to-all cost model drives EP's design envelope.
- **Chapter 06 (placement).** Expert redistribution and NIC-to-GPU affinity for all-to-all live there.

## The boundary to the performance peer

This chapter does not teach:

- How DeepEP / FastMoE / grouped-gemm kernels batch per-expert GEMMs into a single kernel launch — kernel craft.
- How the dispatch kernel avoids an explicit memcpy for the no-op case — kernel craft.
- How the all-to-all can be fused with the activation of the preceding layer — fusion craft.
- How the gating softmax is quantised without hurting routing stability — numerical craft.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). The operator diagnoses "per-expert utilization is skewed" from the metric; the kernel-level improvements to the dispatch kernel are deferred.

## Summary

- MoE adds routed experts to the transformer MLP; each MoE layer runs a dispatch all-to-all and a combine all-to-all; EP partitions the experts across ranks.
- EP degree bounded above by expert count and by per-rank capacity; composes with TP for the dense parts (attention, shared experts).
- Capacity factor bounds routing imbalance; too tight drops tokens (quality cost); too loose wastes memory.
- Three canonical references: Mixtral 8×7B (small, 8 experts), DeepSeek V2/V3 (fine-grained + shared, 160/256 experts), Qwen-MoE (varies).
- Measure per-expert utilization; mitigate imbalance by capacity-factor dial, balanced-routing, expert redistribution, or workload-aware routing.
- Common failures: all-to-all over slow fabric, capacity drops under load, non-EP code path for MoE, hot-spotted rank, shared-expert bottleneck, specialised kernel not loaded, mis-divisible EP degree, gating dtype mismatch.
- Kernel-level questions (grouped-gemm, dispatch-kernel fusion, DeepEP, gating-softmax quantization) route to the performance peer.
