# Exercise 05 — Long-Context KV Footprint Budget

**Time:** 2 hours · **Output:** one per-token byte-math worksheet for three models, one per-GPU KV-pool budget worksheet for two SKUs, one max-concurrent table across context lengths, one engine-vs-arithmetic verification, one "does it fit" decision document

## Goal

Make [Chapter 05](../05-long-context-kv-footprint-budgeting.md) concrete by **computing per-token KV footprint** for three real models at BF16 / FP8 / INT4 KV precisions, **budgeting the KV pool** for two GPU SKUs under tensor parallelism, **tabulating max concurrent sequences** across context lengths, **verifying the arithmetic against the engine's reported `num_gpu_blocks`**, and **authoring a "does it fit" decision document** you could hand a product manager.

## Problem statement

Chapters 01–04 operated the KV cache. Chapter 05 (and this exercise) is the step every operator takes *before* operating anything: can the deployment even fit? The arithmetic is simple; the discipline of writing it down and checking it against the engine is the thing that fails in practice.

The output is a worksheet you reuse every time a new model, context target, or SKU enters planning.

## Prerequisites

- Access to the model configs (via Hugging Face model cards) for the three models you choose.
- A running engine (any size GPU is fine — you'll verify arithmetic against it, not run production traffic).
- A spreadsheet or Python notebook for arithmetic.

## Requirements

Deliver five artefacts.

### Artefact A — Per-token KV byte-math for three models

Choose three real models that span the modern design space. Recommended:

- A Llama-shape GQA model (e.g. Llama-3.1-8B, Llama-3.1-70B, or Qwen2.5-72B).
- An MoE model (e.g. Mixtral-8x7B or DeepSeek-V3 if you want to tackle MLA).
- A high-head-dim or MHA model (e.g. an older Llama-2 model, or a specialty model with MHA — verify in model card).

For each model, from the Hugging Face `config.json`, extract:

- `num_hidden_layers` (= `num_layers`)
- `num_attention_heads` (= `num_heads`)
- `num_key_value_heads` (= `num_kv_heads`)
- `hidden_size` and `head_dim` (compute `head_dim` as `hidden_size / num_heads` if not stated)

Compute `kv_bytes_per_token` for each at:

- BF16 KV
- FP8 KV (if supported by your engine)
- INT4 KV (if supported)

If a model uses MLA (DeepSeek), note that the formula does not apply directly; look up the per-token latent dimension from the model card or the DeepSeek-V3 paper and compute the latent-KV byte cost (the paper gives this explicitly — see [`resources.md`](resources.md)).

Submit as a table:

| Model | layers | kv_heads | head_dim | BF16 KV / token | FP8 KV / token | INT4 KV / token |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

### Artefact B — Per-GPU KV-pool budget for two SKUs

Pick two GPU SKUs that span your fleet. Options: H100 80 GiB, H200 141 GiB, A100 80 GiB, L40S 48 GiB, B100 / B200 (verify current spec).

For each SKU, author a budget worksheet:

```
HBM per GPU               = <SKU spec> GiB
Weights per GPU (model X at TP=N) = <computed from model size / TP degree>
Activation reserve        = 2–4 GiB (justify from your engine's logs or a rule of thumb)
Workspace reserve         = 2 GiB
Runtime overhead          = 1–2 GiB
Available for KV pool     = <HBM> − <weights> − <reserves>
```

Pick one of your three models (Artefact A) and three TP degrees (1, 2, 4 or 1, 4, 8 depending on model size).

Submit as a table:

| SKU | Model | TP | HBM | Weights/GPU | Reserves | KV pool / GPU |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

### Artefact C — Max-concurrent table across context lengths

For your chosen model + SKU + TP configuration, publish the concurrency table across context lengths:

| Context tokens | KV / sequence / GPU | Naive max concurrent | With 30% prefix-cache + 15% headroom |
|---|---|---|---|
| 4 k | ... | ... | ... |
| 8 k | ... | ... | ... |
| 16 k | ... | ... | ... |
| 32 k | ... | ... | ... |
| 64 k | ... | ... | ... |
| 128 k | ... | ... | ... |

Where the model config supports longer context (e.g. 1 M-token models), extend the table. Where the model is capped at a shorter context, trim.

Repeat the table once with FP8 KV precision. Comment on which rows this unlocks and which it leaves infeasible.

### Artefact D — Engine-vs-arithmetic verification

Launch your engine with one of the configurations from Artefact C that fits on your actual hardware. Capture:

- The engine's reported `num_gpu_blocks` (vLLM startup log) or the equivalent (TRT-LLM: `max_tokens_in_paged_kv_cache` divided by effective block size; SGLang: equivalent metric).
- The `block_size` the engine is using.
- The compute `num_gpu_blocks × block_size × kv_bytes_per_token_per_gpu` and compare to your Artefact B "available for KV pool" number.

Expect a match within 10–20%. Discrepancies:

- Within 20%: likely activation / workspace reserve variance. Accept.
- Within 50%: investigate. Is your `num_kv_heads` right? Did the engine quantise KV without being asked (some newer engines auto-FP8-KV; check logs)? Is PP in play?
- More than 50%: something is wrong. Dig until it reconciles.

Submit the arithmetic and the measurement side-by-side with a one-paragraph explanation.

### Artefact E — "Does it fit" decision document

A one-page decision doc for a hypothetical product-team request. Pick a plausible request from one of these:

- "We want to serve [your 70B-class model] at 128 k context with 50 concurrent sequences."
- "We want to serve [your MoE model] at 32 k context with 100 concurrent sequences."
- "We want to serve [your MHA model] at 64 k context with 20 concurrent sequences."

Walk through the Chapter 05 decision tree:

1. Compute the request's KV budget demand.
2. Check against your Artefact B / C tables.
3. If it doesn't fit, walk the options: increase TP, lower context, FP8 KV, enable prefix caching, deploy tiered store, scale replicas.
4. Pick the recommended path and defend it. State the trade-offs (e.g. "FP8 KV halves the budget but requires accuracy gating per module 105").
5. State the next-level questions you'd need the product team to answer (e.g. "is 64 k context acceptable instead of 128 k?").

Output is a document a product manager could read without needing an engineer at their shoulder.

## Starter guidance

- **Pull configs from Hugging Face**. `huggingface_hub` CLI or just the `config.json` file in the model repo. Confirm the `num_key_value_heads` field — some older models have GQA but call it `num_kv_heads` or something else.
- **Weights-per-GPU under TP**. `total_params × dtype_bytes / tp_degree`. For Llama-3-70B BF16: `70e9 × 2 bytes = 140 GiB total` → 17.5 GiB / GPU at TP=8.
- **MoE weights are trickier**. The total parameter count includes all experts, but only a fraction is active per step. For capacity planning, use the *total* parameter count — the inactive experts still occupy HBM.
- **MLA (DeepSeek)**: per-token latent dim × 2 (K and V latents) × dtype_bytes × num_layers. Numbers are small compared to standard GQA — one of MLA's design wins.
- **Engine startup log**: on vLLM, the line `# GPU blocks: X, # CPU blocks: Y` is what you compare to. For SGLang, look for radix-cache or token-cache sizing output. For TRT-LLM, the KV-cache config section at startup.
- **A spreadsheet is sufficient**. A Python notebook is a nicer artefact for version control.

## Fallbacks for constrained hardware

The arithmetic exercise does not require specific hardware. For Artefact D (engine verification), use whatever GPU you have — even a 24 GiB consumer card can run a 1B–3B model and let you verify the formula.

## Acceptance criteria

1. **Artefact A** has three models with correct per-token byte-math at BF16, and (where supported) FP8 and INT4. MLA models are handled explicitly, not glossed.
2. **Artefact B** has two SKUs with a reproducible accounting of HBM allocation. Numbers derive from stated inputs, not magic.
3. **Artefact C** has a context-length ladder with ≥ 5 rows and both naive and headroom-adjusted columns. Includes an FP8 variant.
4. **Artefact D** verifies the arithmetic against the engine within 20%; discrepancies are investigated.
5. **Artefact E** fits on one page, has a defensible recommendation, and names the trade-offs.
6. All five artefacts live in version control.

## Common failure modes

- **Wrong `num_kv_heads`**. The single most common bug. GQA models have `num_kv_heads < num_heads`; confusion between the two produces an 8× KV-size error. Verify.
- **MLA mishandled**. If you pick DeepSeek-V3 without reading the MLA paper, your byte-math will be wildly wrong. Either learn MLA or pick another model.
- **Weights-per-GPU wrong under PP**. PP shards layers; weights per GPU drop proportionally with pp_degree. If you combine TP and PP, apply both reductions.
- **Workspace reserve guess too optimistic**. Some models (reasoning models, VLMs) reserve more activation workspace. If your arithmetic disagrees with the engine, this is often the culprit.
- **Engine auto-quantising KV**. Modern engines sometimes default FP8 KV on H100 for certain model configs. Check the startup log before concluding your arithmetic is off.
- **Confusing bytes and bits**. Classic. Double-check.

## Keep it

The three-model byte-math table and the context-length ladder are the capacity-planning artefact you reuse every quarter and whenever a new model lands. Pin to the team wiki.
