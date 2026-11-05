# Chapter 01 — Topology Selection: TP, PP, EP, DP for a Given Model and Fleet Shape

> Before any launch flag, the first decision is where the model lives. Four axes, one decision tree, one byte-math pass per candidate. The wrong answer costs more than any quantization decision made at module 105.

## Motivation

A platform team receives three serving requests in one quarter:

1. **"Serve Llama-3.1-70B at 4000 QPS with sub-300 ms TTFT."** The checkpoint is 140 GiB in BF16. One H100 holds 80 GiB. The model does not fit on one GPU; it does fit on an 8× H100 node (640 GiB aggregate HBM). The question is not *whether* to shard — the question is *how*.
2. **"Serve DeepSeek V3 (671B total, 37B activated) at 2000 QPS."** The checkpoint is roughly 1.3 TiB in BF16 (dense) or ~440 GiB in BF8. One node holds 640 GiB in BF16 or 320 GiB in FP8. The experts are routed, so most of the parameters are dormant per token. A TP-only layout wastes expert capacity; a PP-only layout saturates one pipeline stage.
3. **"Serve Llama-3.2-3B at 50,000 QPS."** The checkpoint is 6 GiB. One A10 holds 24 GiB. The model fits on one GPU many times over. The question is not whether to shard but how many parallel workers to run and whether to shard attention across them (DP-attention) for burst-concurrency.

These are three different topologies: TP within a node (request 1), EP across many nodes with a residual TP + PP (request 2), DP replicate-and-balance (request 3). Picking the wrong one is a four-week fire — the fleet shows throughput that is 30-70 % of its hardware peak, TTFT is above SLO, and the symptom reads like a kernel problem even though it is a placement problem.

By the end of this chapter you can:

- Enumerate the four axes along which a model is partitioned (TP / PP / EP / DP) and state what each one costs.
- Walk a decision tree from `(model size, batch shape, HBM per GPU, GPUs per node, interconnect)` to a topology recommendation.
- Produce the byte-math behind the recommendation (weights + KV + activations + reserves must fit).
- Name the collectives each topology implies, which the later chapters cost out.
- Recognise the "small model, big batch" DP-attention regime, the "doesn't fit on one node" PP boundary, and the "MoE" EP boundary.

Specific thresholds (e.g. how far TP scales inside a node before degrading) depend on the GPU generation and the interconnect; the numbers quoted in this chapter are directional — verify against your engine's distributed-serving documentation and the hardware vendor's bandwidth specs for your target fleet.

## The four axes

A transformer forward pass has three large tensor-structures the engine can split across devices:

- **Weights** — the parameter matrices themselves (per-layer attention and MLP projections, embeddings, LM head).
- **KV cache** — the per-request, per-layer, per-head key-and-value tensors (grows with context, dominates memory at long context).
- **Activations** — the intermediate tensors produced during a forward pass (per-micro-batch; grows with batch size and sequence length).

Four partitioning axes operate on them:

### Tensor parallel (TP)

**What it splits.** Weight matrices and the KV cache. Each layer's `W ∈ ℝ^{d_out × d_in}` is cut along one dimension; each rank in the TP group holds `W[i-th shard]` and computes with the full input but only its slice of the output (or vice versa, depending on the shard direction).

**What it requires.** At operator depth, two collectives per transformer layer: an **AllReduce** at the output of the attention projection (combines partial results across TP ranks), and an **AllReduce** at the output of the MLP down-projection. For some sharding variants, these decompose into an AllGather + ReduceScatter pair ("sequence parallel") — see chapter 04.

**What it costs.** One AllReduce per layer per micro-batch per rank. The collective cost grows with the number of ranks (TP degree) and shrinks with the per-rank interconnect bandwidth. **TP wants NVLink** — the AllReduce latency on NVLink is in the microsecond-to-low-millisecond range per layer; the AllReduce latency over InfiniBand is 5-30× worse (chapter 04 quantifies). The practical rule: *TP stays inside a node, where NVLink joins the GPUs.*

**Where it shines.** Dense models that fit in a single node's aggregate HBM but exceed one GPU's HBM. Llama-3.1-70B BF16 at ~140 GiB fits 2× H100 (TP=2), 4× H100 (TP=4), or 8× H100 (TP=8); the collective cost grows with TP, so the choice depends on how much HBM you want for the KV pool and batch headroom. Llama-3.1-405B BF16 at ~810 GiB fits 8× H100 (TP=8), with FP8 at ~405 GiB fits 8× H100 (TP=8) comfortably, with INT4 fits 4× H100 — see module 105.

### Pipeline parallel (PP)

**What it splits.** Transformer layers across stages. Stage 0 holds layers `[0, n/K)`, stage 1 holds `[n/K, 2n/K)`, …, stage K-1 holds `[(K-1)n/K, n)`. Each stage runs forward on a micro-batch and passes activations to the next stage.

**What it requires.** Point-to-point communication between adjacent stages — one send and one receive per micro-batch per stage boundary. Not a collective; one-to-one messages. This runs well over InfiniBand / RoCE at the latencies IB offers (~2-10 μs for small messages) because activations are the only payload and they are modest.

**What it costs.** The **bubble** — at the start of a run, stages 1..K-1 are idle while stage 0 fills; at the end, stage 0 is idle while stages drain. For `K` stages and `M` micro-batches, the bubble fraction is approximately `(K - 1) / (M + K - 1)`. Interleaved and 1F1B (one-forward-one-backward, used for training; inference uses the forward-only variant) schedules shrink the bubble but do not eliminate it. Also: the per-stage memory is only `1/K` of the model, but the activations required to pass between stages (for the longest in-flight micro-batch at a stage) still add up.

**Where it shines.** Dense models that **do not fit on a single node** — the full checkpoint is too large even for 8× H100. Llama-3.1-405B BF16 at ~810 GiB needs more than one node of H100; PP spans nodes. MoE at full fidelity (DeepSeek V3 at 1.3 TiB BF16) always needs multi-node — PP is one of the axes composing with EP.

**Where it falters.** Short request batches where the bubble dominates — the overhead eats the gain. Also: pipeline-aware scheduling is finicky; chapter 02 covers the mechanics.

### Expert parallel (EP)

**What it splits.** The experts of a Mixture-of-Experts model across ranks. For an MoE with `E` experts, EP degree `P` places `E / P` experts per rank. Routing sends each token to its top-K experts, which may live on any rank; the result is combined back.

**What it requires.** Two **all-to-all** collectives per MoE layer: a dispatch all-to-all (tokens → their experts' ranks) and a combine all-to-all (expert outputs → the originating ranks). Chapter 05 covers at operator depth.

**What it costs.** The all-to-all cost grows with (number of tokens routed) × (hidden-dim bytes). The routing imbalance cost — if 60 % of tokens pick expert 3 and 2 % pick expert 7, expert 3's rank is overloaded and expert 7's rank is idle. EP-aware batching and capacity-factor tuning (chapter 05) mitigate.

**Where it shines.** All MoE serving at scale. Mixtral 8×7B has 8 experts top-2; DeepSeek V3 has 256 fine-grained + shared experts; Qwen-MoE varies by variant. EP is non-optional for these; the only question is EP degree, how it composes with TP + PP, and the capacity factor.

### Data parallel (DP)

**What it splits.** Requests. `R` replicas each hold a full copy of the model; requests are load-balanced across them. For "DP-attention" at the attention layer specifically, the heads are split across ranks (similar to TP for attention) while the rest of the model is replicated — this is a hybrid used for small models at high concurrency.

**What it requires.** For pure DP, zero inter-replica communication at inference time (training is different). For DP-attention, an AllGather of attention outputs across the attention heads.

**What it costs.** HBM — each replica holds the full model. Fine when the model is small; prohibitive when the model is large.

**Where it shines.** Small models at high concurrency. Llama-3.2-3B at 50,000 QPS wants dozens of replicas, each handling thousands of requests via continuous batching. Also: when a single replica's KV budget is the bottleneck, adding replicas is simpler than growing TP.

## The decision tree

Given a target to serve, run this walk:

```
Inputs:
  - model_params (B parameters)
  - weight_dtype_bytes (2 for BF16, 1 for FP8/INT8, 0.5 for INT4)
  - gpu_hbm_bytes (80 GiB for H100 80GB, 94 GiB for H200, ...)
  - gpus_per_node (typically 8)
  - intranode_interconnect (NVLink / NVSwitch)
  - internode_interconnect (IB / RoCE / Ethernet)
  - target_qps, target_ttft_ms, target_tpot_ms
  - target_context_len
  - is_moe, num_experts (if MoE)
```

### Step 1 — compute model byte-footprint

```
weight_bytes  = model_params * weight_dtype_bytes
kv_budget     = (per-request KV at target_context_len) * (expected concurrent requests)
activation_budget = (batch_size × seq_len × hidden × bytes) × constant
reserve_budget = ~10 % of total for CUDA context, workspace, graph buffers

required_hbm = weight_bytes + kv_budget + activation_budget + reserve_budget
```

(Use the module-104 formula for KV; the module-105 formula for weight bytes at your chosen quantization.)

### Step 2 — pick the topology

```
if is_moe:
    → EP is required. Compose with TP + PP as the dense portion dictates.
    GOTO Step 3 (EP sizing)

if weight_bytes fits on one GPU with headroom for KV + activations:
    if target_qps implies more than one GPU's capacity:
        → DP: N replicas sized by (target_qps / per-replica capacity)
        → Consider DP-attention for very high concurrency on small models
    else:
        → DP = 1 (one replica); you don't need distributed inference
    EXIT

if weight_bytes fits in one node (gpus_per_node × gpu_hbm_bytes):
    → TP within the node, degree = ceil(required_hbm / per-GPU budget)
    → DP = N replicas of the whole TP group if QPS requires it
    EXIT

else:  # doesn't fit in one node
    → PP across nodes; TP within each stage
    → PP degree chosen so each stage fits in one node
    → TP degree chosen so each stage's slice fits across the node's GPUs
    EXIT
```

### Step 3 — EP sizing (when MoE)

```
EP degree bounded above by (num_experts) and by the per-rank capacity
EP degree bounded below by (num_experts * expert_bytes) / fleet_hbm_bytes

If (num_experts * expert_bytes) fits in one node:
    EP within one node, composed with TP for the attention / dense parts
else:
    EP across nodes; dispatch/combine all-to-all goes over IB — be aware of cost
```

Then still: within each EP rank, apply the dense decision tree (TP for the attention + shared layers).

### Step 4 — validate SLO fit

A topology that passes Step 2/3 may still fail SLO. Validate:

- **TTFT.** TP degree raises prefill throughput (more compute, more memory bandwidth) but adds collective overhead per layer. At PP, the first micro-batch's TTFT includes the pipeline fill — this hurts short requests.
- **TPOT.** Decode is memory-bandwidth bound (chapter 103). TP helps (more aggregate HBM bandwidth per token); PP neutral-to-slightly-worse at low batch (bubble dominates); EP neutral at steady state if routing is balanced.
- **Collective-cost floor.** The TP collective cost per layer sets a lower bound on TPOT regardless of GEMM speed. If `TP_collective_cost_per_layer × num_layers > tpot_target`, the topology does not fit the SLO — pick lower TP or a bigger GPU.

### Step 5 — choose the quantization and re-run

Quantization (module 105) changes `weight_dtype_bytes` and the per-GPU footprint. FP8 or INT8 can turn a "needs PP" model into a "fits in one node with TP" model. The topology-selection decision composes directly with quantization — do not fix the topology first and quantize afterward; iterate.

## Worked examples

### Example A — Llama-3.1-8B at 10,000 QPS

```
weight_bytes = 8e9 × 2 = 16 GiB (BF16); or 8 GiB (FP8)
Fits comfortably on one H100 80GB with ≥ 60 GiB left for KV + activations.

Required replicas for 10,000 QPS ≈ (10,000 / per-replica capacity).
If per-replica capacity ≈ 500 QPS at target SLO, need ~20 replicas.

Topology: DP = 20 workers, TP = 1, PP = 1.
Interconnect: nothing beyond load-balancer routing.
```

### Example B — Llama-3.1-70B at 2,000 QPS, 300 ms TTFT

```
weight_bytes = 70e9 × 2 = 140 GiB (BF16) → does not fit on one H100.
Fits on 2× H100 (160 GiB raw, ~130 GiB usable after reserves) with headroom for small batch.
Fits comfortably on 4× H100 or 8× H100.

Choose TP = 4 inside one node:
    per-GPU weight share = 35 GiB → ~45 GiB left for KV + activations.
    4-way AllReduce over NVLink ≈ hundreds of microseconds per layer; verify against chapter 04 numbers.

For 2000 QPS, DP ≈ 2000 / per-replica QPS. Per-replica QPS at TP=4 depends on
batch size and prefill/decode mix; assume 400 QPS → DP = 5 replicas.

Topology: 5 × (TP=4 inside one node) = 20 GPUs total, i.e. 2.5 nodes
(round up to 3 nodes of 8× H100, 24 GPUs, 6 replicas → 2400 QPS headroom).
```

### Example C — Llama-3.1-405B at 1,000 QPS

```
weight_bytes = 405e9 × 2 = 810 GiB (BF16) → does not fit in one 8× H100 node (640 GiB).
With FP8: 405 GiB → fits in one 8× H100 node (TP=8).
With BF16: must span two nodes → PP = 2 (each stage ~405 GiB on an 8× H100 node, TP=8 within each stage).

Topology choice depends on quantization appetite (see module 105 trade-offs):

Option 1 (FP8):  TP = 8 within one node; DP = N replicas.
Option 2 (BF16): PP = 2 × TP = 8 → 16 GPUs per replica; DP = N replicas.

Option 1 has lower collective cost (NVLink-only) and lower TTFT (no pipeline fill);
Option 2 preserves BF16 accuracy. Pick on the quantization vs. accuracy posture (module 105 ch 04).
```

### Example D — DeepSeek V3

```
Total parameters: ~671B. Activated per token: ~37B. 256 fine-grained experts + 1 shared.
Weight bytes BF16: ~1.3 TiB; FP8: ~670 GiB.

Even FP8 does not fit on one 8× H100 node. Need multi-node and EP.

Topology (one plausible shape; vendor and research references list several):
    EP = 32 (experts partitioned across 32 ranks, 8 experts per rank)
    TP = 1 within each rank for the expert FFNs; TP = 8 for the dense attention
    PP = 1 at serving time if memory fits with EP; otherwise PP = 2+

Collectives: EP all-to-all per MoE layer (chapter 05) + TP AllReduce for the attention.
Interconnect: multi-node → IB / RoCE for the EP all-to-all; this is the dominant cost.
```

Verify the specific vendor-recommended shape in the DeepSeek-V3 technical report ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) and the SGLang / vLLM deployment guides. The example above is illustrative; exact numbers rotate.

### Example E — Llama-3.2-3B at 50,000 QPS

```
weight_bytes = 3e9 × 2 = 6 GiB (BF16); fits many times over on one A10 or L4.

QPS target of 50,000 is the dominant constraint.
Per-replica capacity for a 3B model on an L4 might be 1,500 QPS → ~34 replicas.
Alternatively DP-attention within a worker to share the attention across heads
while running continuous batching — mostly useful at very high per-worker concurrency.

Topology: DP = 34 L4s (or equivalent); TP = 1 per replica; PP = 1.
```

## The collectives each topology issues

A quick reference used by chapter 04 to cost each:

| Topology | Per-layer collectives | Per-request collectives |
|---|---|---|
| TP (standard) | 2 × AllReduce (attention out, MLP down) | — |
| TP + sequence-parallel | 2 × AllGather + 2 × ReduceScatter | — |
| PP | — (per-stage) | 1 × send/recv at each stage boundary |
| EP | 2 × all-to-all per MoE layer (dispatch, combine) | — |
| DP | — | — |
| DP-attention | 1 × AllGather per attention block | — |

The collective cost per iteration = (collective_cost × num_layers_with_this_collective). Chapter 04 translates each entry into the latency contribution.

## Common selection errors

### Error 1 — TP across a node boundary over IB

The team picks TP = 16 across two 8× H100 nodes connected by IB. The per-layer AllReduce now goes over IB instead of NVLink. Collective latency rises by 10-30× per layer; TPOT inflates by hundreds of milliseconds; the SLO fails. **TP stays inside a node.**

### Error 2 — PP for a model that fits in one node

The team picks PP = 2 for Llama-3.1-70B because "it's a big model." But 70B BF16 fits in one 8× H100 node with TP = 4. The PP introduces a bubble, raises TTFT at short batches, and adds operational complexity. Choose PP only when the model actually does not fit in one node.

### Error 3 — DP for a model that doesn't fit on one GPU

The team runs Llama-3.1-70B as DP = 10 on 10× H100 — but 70B doesn't fit on one H100. Each replica OOMs on load. DP replicates; it does not shard. The required partitioning for weights that exceed one GPU is TP or PP.

### Error 4 — EP = 1 for an MoE model

The team runs Mixtral-8×7B with no EP ("just treat it as a 46B dense"). The MLP compute now runs all 8 experts for every token instead of routing to 2 — training-time efficiency gained from routing is thrown away. Inference engines usually handle this correctly if you pass the right flag; the error is operator-level, failing to notice.

### Error 5 — ignoring KV + activations in the fit check

The team sizes TP by "weight_bytes / per_GPU_hbm" and leaves no headroom for the KV pool. At long context or high concurrency, the engine OOMs the moment load ramps. **Weights are the floor, not the ceiling** — the fit check must include KV, activations, and reserves.

### Error 6 — ignoring quantization

The team picks PP = 2 for a model that would have fit on one node at FP8. The quantization decision (module 105) is in the loop with the topology decision; iterate.

### Error 7 — fixing topology before measuring SLO

The team picks a topology from a napkin estimate, deploys, and finds TTFT is 2× the SLO. Topology selection must include an SLO-feasibility step; walk Step 4 before committing.

## The boundary to the performance peer

This chapter does not teach:

- How the per-layer AllReduce is implemented (ring vs. tree, warp-level reduction, NVLink SM-copy vs. CE-copy) — see the performance peer and chapter 04 of this module for the operator view.
- How the TP-sharded attention kernel composes with Flash-Attention — kernel craft; `.cu` edit.
- How the MoE dispatch kernel partitions tokens across experts without an all-to-all memcpy for the no-op case — kernel craft.
- How to re-tune the NCCL algorithm selection for a specific message-size regime — the operator sets `NCCL_ALGO`; the peer tunes the algorithm's internals.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This chapter gives you the operator's view: *which topology, with which byte-math, running which collectives*. Chapter 04 names the collective cost; chapter 06 names the common operator-level cliffs; chapter 07 writes the hand-off.

## Summary

- A transformer is partitioned along four axes: TP (weights + KV), PP (layers across stages), EP (experts across ranks), DP (requests across replicas).
- TP wants NVLink and stays inside a node; PP spans nodes when a model doesn't fit in one; EP is non-optional for MoE; DP replicates the model.
- The decision tree is: compute byte-footprint → choose TP inside a node when possible, PP across nodes when not, EP for MoE, DP for throughput → validate against SLO (collective-cost floor, TTFT at pipeline fill, TPOT) → iterate with the quantization decision from module 105.
- Common errors are TP across node boundaries, PP for models that fit in one node, DP for models that don't fit on one GPU, EP ignored for MoE, KV + activations ignored in the fit check, and skipping the SLO-feasibility step.
- The collectives each topology issues are cost-out at chapter 04; the deployment mechanics are at chapter 02; the MoE specifics at chapter 05; the scaling cliffs at chapter 06.
- Kernel-authoring questions (algorithm, warp-level reduction, fused epilogue, Flash-Attention-with-TP) route to the performance peer.
