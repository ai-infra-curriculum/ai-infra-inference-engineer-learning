# Chapter 05 — Long-Context KV Footprint Budgeting

> "Will it fit?" is the first operator question, and it is answered by arithmetic, not by trial and error. This chapter is the arithmetic.

## Motivation

When a product team asks "can we serve this 70B model at 128 k context on an 8× H100 node?", the honest answer comes from a byte-math worksheet, not from `vllm serve` and a stopwatch. Trial-and-error eventually produces the right answer, but it produces it after you have burned engineer-hours on OOM crashes, mysterious `num_gpu_blocks` numbers, and preemption cliffs that only show up at peak load.

This chapter is the arithmetic. By the end you can, in under five minutes with a calculator, estimate:

- Bytes of KV per token for a given model.
- Total KV footprint for a given batch × context length.
- Max concurrent sequences the pool can hold.
- Headroom for prefix cache, swap space, and a safety margin.
- Whether a (model, context, SKU, concurrency) tuple fits — or which axis has to give.

The method is the same regardless of engine. The specific number of `num_gpu_blocks` your engine reports should match the arithmetic to within a few percent; if it does not, something unusual is happening (chapter 01 fragmentation details, or an unusual KV layout).

<!-- needs-research: exact dtype choice for KV at FP8 varies by engine and version (E4M3 vs. E5M2 vs. mixed). Verify in your engine's docs; the arithmetic framework in this chapter is stable regardless of the choice. -->

## The per-token KV formula

The core formula, for one token's KV in one sequence:

```
kv_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

Each factor:

- **2** = K plus V. Two tensors per attention layer.
- **num_layers** = the transformer's layer count. Published in every model card. 32 for Llama-3-8B, 80 for Llama-3-70B, 126 for DeepSeek-V3 (verify current), etc.
- **num_kv_heads** = the number of K/V heads, which for GQA (grouped-query attention) is smaller than the query-head count. Also published in the model config. Classic MHA: `num_kv_heads = num_heads`. GQA with 8:1 ratio: `num_kv_heads = num_heads / 8`.
- **head_dim** = the dimensionality of each head. For most Llama-shape models, 128. For some models, 64 or 256; always check the config.
- **dtype_bytes** = 2 for BF16 / FP16, 1 for FP8, 0.5 for INT4 (quantised KV; where supported).

### Worked examples

**Llama-3-70B** (`num_layers=80`, `num_heads=64`, `num_kv_heads=8` (GQA 8:1), `head_dim=128`), BF16 KV:

```
kv_bytes_per_token = 2 × 80 × 8 × 128 × 2
                   = 327_680 bytes
                   ≈ 320 KiB per token
```

**Llama-3-8B** (`num_layers=32`, `num_kv_heads=8`, `head_dim=128`), BF16:

```
kv_bytes_per_token = 2 × 32 × 8 × 128 × 2
                   = 131_072 bytes
                   = 128 KiB per token
```

**DeepSeek-V3** uses MLA (multi-head latent attention), which radically changes KV footprint — the stored K/V is a compressed latent, not per-head K/V. Treat MLA models as a separate case and verify the engine's reported per-token bytes. The formula above is for standard MHA / GQA.

### The GQA / MQA fan-out effect

The single biggest driver of KV footprint for a given model family is the K/V-to-query head ratio:

- **MHA** (`num_kv_heads = num_heads`): most KV, highest fidelity.
- **GQA** (`num_kv_heads = num_heads / group_size`): KV footprint divided by `group_size`. Common group sizes: 4, 8. Llama-3-70B is GQA 8:1.
- **MQA** (`num_kv_heads = 1`): KV footprint divided by `num_heads`. Aggressively small; some older models (PaLM-shape).

A 70B-class model with MHA would have 8× the per-token KV of its GQA-8:1 counterpart. Model choice has first-order impact on serving math.

### Precision choices

- **BF16 / FP16**: 2 bytes. Default.
- **FP8** (E4M3 or E5M2): 1 byte. Supported in some engines (verify). Halves KV footprint. The accuracy impact of FP8 KV is model- and workload-dependent; see module 105.
- **INT4 KV**: 0.5 bytes with packing. Experimental / limited support. Non-trivial accuracy consideration.

For capacity planning, the typical answer today is "BF16 KV unless the deployment explicitly validated FP8." The byte-math changes by the precision factor.

## Per-sequence KV footprint

For a sequence with `S` total tokens (prompt + output):

```
kv_bytes_per_sequence = S × kv_bytes_per_token
```

For a Llama-3-70B sequence at 4 096 tokens total:

```
4 096 × 320 KiB = 1 310 720 KiB = 1 280 MiB ≈ 1.28 GiB per sequence
```

For a Llama-3-70B sequence at 128 k tokens:

```
131 072 × 320 KiB = 41 943 040 KiB = 40 GiB per sequence
```

Yes. At 128 k context, a *single* 70B sequence costs **40 GiB of KV** on its own. On a single H100 (80 GiB HBM) with weights (140 GiB for 70B BF16 — doesn't fit without sharding), the number is academic; but on a 4× H100 node with TP=4, each GPU holds 10 GiB of KV per 128 k sequence. One sequence takes a notable fraction of each GPU's KV budget.

### Tensor parallelism and KV footprint

Under tensor parallelism (TP=N), the KV cache is **sharded across the N GPUs**, each holding `1/N` of the KV per sequence. The arithmetic becomes:

```
kv_bytes_per_sequence_per_gpu = (S × kv_bytes_per_token) / tp_degree
```

For Llama-3-70B at 128 k and TP=8, each GPU holds 40 GiB / 8 = 5 GiB of KV per sequence. Reasonable.

### Pipeline parallelism

Under pipeline parallelism (PP=N), each stage holds its own layers' KV — effectively dividing `num_layers` by `pp_degree` for the per-stage calculation. Compose with TP as needed; the arithmetic is still per-GPU, just with the right subset of the formula.

### Expert parallelism (MoE)

For MoE models, non-expert layers (attention) are typically TP-sharded; experts are EP-sharded. The KV lives in the attention layers, which are usually TP. The formula above applies with `num_layers` set to the full layer count — expert layers don't add KV.

## The budget calculation

Walking through a realistic capacity plan.

### Scenario

- Model: Llama-3-70B
- GPU: 8× H100 80 GiB (one node)
- Topology: TP=8 (single-node tensor parallel)
- Context target: 32 k tokens (prompt + output)
- Workload: chat, no prefix caching for now, no FP8 KV
- Target: how many concurrent sequences can fit?

### Step 1: HBM budget per GPU

```
HBM per GPU                    = 80 GiB
Weights per GPU (70B BF16 / 8) = 140 GiB / 8 ≈ 17.5 GiB
Activation reserve             ≈ 2–4 GiB (model- and batch-dependent)
Workspace reserve              ≈ 2 GiB (CUDA graphs, pinned buffers)
Runtime overhead               ≈ 1–2 GiB
Available for KV pool          ≈ 80 − 17.5 − 3 − 2 − 2 ≈ 55 GiB
```

Pre-flight: this is a rough estimate. The engine's actual available-for-KV number can be read off the startup log. vLLM reports `num_gpu_blocks` and the block-byte-size, which multiply to the available bytes. The arithmetic should match within 10–20%; discrepancies are usually workspace overhead the engine reserves conservatively.

### Step 2: Per-sequence KV per GPU

```
kv_bytes_per_token         = 2 × 80 × 8 × 128 × 2 = 320 KiB
kv_bytes_per_token_per_gpu = 320 KiB / 8 = 40 KiB
kv_bytes_per_sequence      = 32 768 tokens × 40 KiB = 1 310 720 KiB = 1 280 MiB = 1.25 GiB per GPU
```

### Step 3: Max concurrent sequences

```
max_concurrent_sequences = available_for_KV / kv_bytes_per_sequence_per_gpu
                         = 55 GiB / 1.25 GiB
                         ≈ 44 sequences
```

Round down for safety margin — call it **40 concurrent sequences** at 32 k context.

### Step 4: Headroom

The 44-sequence number is a **ceiling**, not a target. Deduct for:

- **Prefix cache retention** (chapter 03). If 30% of the pool is cache, available for live sequences drops to 70% × 55 ≈ 38.5 GiB, dropping max concurrent to 31.
- **Swap-space reserve** (chapter 01). Not HBM, but if swap is sized for 50% of HBM-pool size, that is 55 × 0.5 = 27.5 GiB of host DRAM, which must be budgeted on the CPU side.
- **Safety margin for prefix growth, bursts, scheduler jitter**. 10–20% headroom typical.

Final defensible number: **~30 concurrent sequences** at 32 k context, with prefix caching on and conservative headroom.

### Step 5: Verify against the engine

```
num_gpu_blocks × block_size × kv_bytes_per_token = available for KV
```

For `block_size=16` and `num_gpu_blocks=90 000` and `kv_bytes_per_token=40 KiB per GPU`:

```
90 000 × 16 × 40 KiB = 90 000 × 640 KiB = 57 600 000 KiB ≈ 55 GiB per GPU
```

Match. If your engine reports much less, something is reserving more than expected (likely activation workspace or CUDA graph buffers for large batch shapes).

## Context-length ladder — the critical table

For a given model + hardware, publish the "fits" table across context lengths:

| Context tokens | KV per sequence per GPU | Max concurrent (naive) | Max concurrent (with 30% prefix cache + 15% headroom) |
|---|---|---|---|
| 4 k | 160 MiB | ~350 | ~210 |
| 8 k | 320 MiB | ~175 | ~105 |
| 16 k | 640 MiB | ~87 | ~52 |
| 32 k | 1.25 GiB | ~44 | ~26 |
| 64 k | 2.5 GiB | ~22 | ~13 |
| 128 k | 5 GiB | ~11 | ~6 |

(Numbers specific to the Llama-3-70B / 8× H100 / TP=8 scenario; verify against your engine.)

This table drives three operator decisions:

1. **Max context the product can promise** at a given concurrency target.
2. **The scaling event**: at what concurrency does this SKU stop meeting demand?
3. **The precision / quantisation decision**: does FP8 KV (half the row) move the needle enough to justify the accuracy work?

## Precision sensitivity — FP8 KV, INT4 KV

Switching from BF16 to FP8 KV halves `kv_bytes_per_token`. Every number in the table doubles. The product team gets 2× the concurrent sequences or 2× the context length (not both).

Accuracy consideration: FP8 KV has measurable quality impact on some models and workloads. Not catastrophic for most Llama-shape chat traffic (benchmarks show single-digit-percent degradation on standard task metrics; see module 105 and the AWQ/FP8 literature in [`resources.md`](resources.md)). Catastrophic for some long-context reasoning traces and some non-English generation. **Treat as an accuracy-gated deployment choice**, not a free switch.

INT4 KV (where supported) halves again. Accuracy impact is model-, workload-, and quantisation-scheme-specific. Must gate through accuracy regression testing per module 105.

## The "does it fit" decision tree

A product request lands: "we want to serve model M at context C with concurrency N."

```
1. Compute kv_bytes_per_token for M at the target precision.
2. Compute kv_bytes_per_sequence = C × that.
3. Add weights, activations, workspace, overhead. Call total `fixed_hbm`.
4. Compute kv_pool_bytes = hbm_per_gpu × tp_degree − fixed_hbm.
5. Compute max_concurrent_sequences = kv_pool_bytes / (kv_bytes_per_sequence × tp_scale).
6. Apply headroom (prefix cache, safety margin): max × 0.6 ≈ practical_max.

If N ≤ practical_max: ship.
If N > practical_max:
    - Can we increase tp_degree? (more GPUs per node, more KV per GPU after sharding)
    - Can we increase pp_degree? (more nodes, more layers' KV spread)
    - Can we lower C? (shorter context — product constraint)
    - Can we quantise KV? (FP8 → doubles capacity; accuracy gate)
    - Can we use prefix caching aggressively? (effective capacity grows if prompts share)
    - Can we deploy tiered store? (chapter 04 — move cache off HBM)
    - Can we accept lower concurrency and scale horizontally? (more replicas)
```

Each branch has cost and operational tradeoffs. The first move for most deployments: tighten the context window. The second move: enable prefix caching. The third move: FP8 KV with accuracy gate. The fourth move: more hardware.

## Common misreadings

- **"Weights dominate HBM."** On small-context deployments, yes. On long-context deployments at 128 k, KV can exceed weights by a factor of 2–3× per request. Capacity planning that only models weights is wrong at the context lengths modern products are shipping.
- **"The engine's reported `num_gpu_blocks` is magic."** It is derived from the exact arithmetic in this chapter plus some workspace reserves. If you can't reproduce it, something is off in your inputs — most commonly the `num_kv_heads` field (GQA vs. MHA confusion).
- **"GQA models have the same KV as MHA."** No — GQA 8:1 has 1/8 the KV. That is the entire *point* of GQA for serving.
- **"FP8 KV is free capacity."** It is 2× capacity at a measurable accuracy cost that must be gated. Module 105 is where that gate lives.
- **"If I buy a bigger GPU the capacity just scales."** HBM scales (H100 → H200 is 1.4× HBM; B100 / B200 scale further; verify). But many long-context serving architectures benefit more from *more parallelism* (TP × PP) than from more memory per GPU; the KV sharding is often what unlocks concurrency, not raw HBM per GPU.

## Summary

- **KV bytes per token = `2 × num_layers × num_kv_heads × head_dim × dtype_bytes`.** Everything else is multiplication.
- **GQA / MQA** is the first-order driver of per-model KV footprint; a GQA-8:1 model has 1/8 the KV of its MHA counterpart.
- **TP shards KV across GPUs**; per-GPU KV is divided by `tp_degree`. PP shards layers; EP leaves attention alone.
- **The budget**: `(HBM per GPU) − weights − activations − workspace − overhead = KV pool`. Divided by per-sequence KV = max concurrent sequences. Apply 30–50% headroom for prefix cache, bursts, and safety.
- **The context-length ladder** is the one-page table every capacity planning conversation starts with.
- **FP8 / INT4 KV** doubles / quadruples capacity but is an accuracy-gated choice (module 105).
- **The decision tree for "does it fit"** — increase TP, lower context, quantise KV, enable prefix caching, deploy tiered store, scale replicas — each option has a cost and a trade-off.
- **Verify against the engine.** `num_gpu_blocks × block_size × kv_bytes_per_token` should match your arithmetic within 10–20%. Discrepancies point at the right thing to investigate.

Chapter 06 moves from "will it fit?" to "is it safe to share?" The cross-tenant sharing of KV cache is where this module's math and this module's policy touch the serving system's trust model.
