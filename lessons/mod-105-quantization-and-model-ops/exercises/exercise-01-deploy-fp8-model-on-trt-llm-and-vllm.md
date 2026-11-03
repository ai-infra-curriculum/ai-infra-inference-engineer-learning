# Exercise 01 — Deploy an FP8 Model on TRT-LLM and vLLM

**Time:** 4 hours · **Output:** one FP8 artefact, two serving deployments (vLLM + TRT-LLM), one memory + throughput table, one logit-diff / sampled-output comparison, one deployment checklist

## Goal

Make [Chapter 01](../01-quantization-formats-and-engine-support.md) (the format × engine matrix) and the FP8 half of [Chapter 02](../02-quantization-algorithms-at-operator-depth.md) concrete by **producing an FP8 variant** of one stock instruction-tuned model, **serving it on both vLLM and TRT-LLM**, and **comparing** memory, throughput, and output distribution against the BF16 baseline. By the end you can state exactly what each engine does with an FP8 artefact and have a reusable comparison harness for the rest of the module.

## Problem statement

Your team is scoping an FP8 migration for one of the fleet's primary models. Before committing, you need credible numbers on three axes: how much HBM FP8 actually saves, how much throughput it actually unlocks, and whether the outputs are close enough to BF16 to pass the chapter-04 gate (which you'll author in exercise 04). The deliverable is a side-by-side comparison and a `go / no-go / needs-more-work` recommendation.

This exercise does **not** go all the way to the full accuracy gate — that is exercise 04. The output here is the input to that gate.

## Prerequisites

- A workstation or lab host with at least one Hopper-generation NVIDIA GPU (H100 80 GB, H200 141 GB, or L40S / Ada SKU for Ada-class FP8). Blackwell will also work for FP8 (not just FP4). Ampere will not — FP8 Tensor Cores are not present and the engines will refuse or fall back.
- CUDA 12.x and matching driver.
- Python 3.10–3.12 with virtual environments.
- A pinned `vllm` install (0.7+ recommended; verify FP8 flag names for your version).
- A pinned `tensorrt-llm` install matching your CUDA / driver (TRT-LLM has rapid releases; pin precisely).
- The model-104 module's engine startup and KV byte-math fluency (you'll compare FP8 vs. BF16 memory using the chapter-05 byte-math from module 104).

## Choose a model

Pick a model that:
- Fits on your chosen GPU in BF16 (so you can baseline).
- Has a published model card and is covered by its licence for your use (see chapter 05).
- Is instruction-tuned (so sampled-output comparison is meaningful).

Recommended choices:

- **Llama-3.1-8B-Instruct** (fits on 1× H100 80 GB in BF16 with room; smallest viable).
- **Qwen2.5-7B-Instruct** or **Mistral-7B-Instruct-v0.3** (alternatives of similar size).
- **Llama-3.1-70B-Instruct** on TP=4 or TP=8 (if you have the hardware).

Record the model and the HF revision SHA. You will need these for chapter-05 lineage.

## Requirements

Deliver five artefacts.

### Artefact A — The FP8 variant

Produce one FP8 variant of the chosen model.

Two paths:

1. **Pre-quantised artefact from Hugging Face.** Models like `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` or `nvidia/Llama-3.1-8B-Instruct-FP8` exist (verify exact names for your release; use the version that is signed and licence-clean). Download, verify checksums.
2. **Quantise yourself with a tool.** Candidates include `llm-compressor` (Neural Magic), `TensorRT-Model-Optimizer` (NVIDIA — this is the TRT-LLM-affiliated tool), or `fbgemm_gpu` depending on the target engine. Pick one; prefer the tool whose output format matches both target engines.

Record:

- The tool version and command line used (if self-quantised).
- The sub-format (E4M3 for weights and activations is the near-universal choice).
- Whether KV cache is FP8 or BF16 in this artefact (independent flag per chapter 01).
- Scale strategy (static vs. dynamic).
- SHA-256 of each file in the artefact.
- Hugging Face revision SHA of the base model.

### Artefact B — vLLM serving deployment

Serve the FP8 artefact on vLLM.

Document:

- The exact launch command (`vllm serve <path-or-id> ...`). Flags to consider: `--quantization fp8` (verify spelling for your version — flag names rotate), `--kv-cache-dtype auto` or `fp8` or `auto_fp8` as applicable, `--max-model-len`, `--gpu-memory-utilization`.
- The startup log — `num_gpu_blocks`, loaded quantisation, loaded KV precision.
- A separate BF16 baseline deployment of the same model with the same `--max-model-len` and `--gpu-memory-utilization`.
- A health-check: send one chat completion and verify response is non-empty.

### Artefact C — TRT-LLM serving deployment

Serve the FP8 artefact on TRT-LLM.

For TRT-LLM you must go through the convert → quantise (if not pre-quantised) → build pipeline:

- `convert_checkpoint.py` (or the current replacement — verify) to produce a TRT-LLM checkpoint.
- `quantize.py` with `--qformat fp8` (or current flag) if the input is not already FP8.
- `trtllm-build` with the appropriate `--max_batch_size`, `--max_input_len`, `--max_seq_len` for your test envelope.

Document:

- The exact pipeline (every command).
- Build time per step.
- The engine's `.engine` file size and SHA.
- The TRT-LLM serving launcher (either direct Python or Triton `tensorrtllm_backend`).
- Health-check: one chat completion through TRT-LLM.

Also build and deploy a BF16 baseline on TRT-LLM for the same model with the same build-time envelope.

### Artefact D — Memory + throughput comparison table

Run a controlled benchmark against each of the four deployments (BF16 vLLM, FP8 vLLM, BF16 TRT-LLM, FP8 TRT-LLM) with the same workload. Suggested workload:

- A fixed prompt set (100–200 prompts) with a mix of short (100 tok) and medium (1 024 tok) prompts.
- Fixed sampling parameters (temperature 0.0 or 0.7 — pick one and document).
- Fixed `max_new_tokens` (256 is a reasonable default).
- Concurrency sweep: 1, 4, 16, 64 concurrent requests (reduce if hardware constrains).

Measure for each (deployment, concurrency) cell:

- **Peak HBM used** (via `nvidia-smi` or the engine's reporting). This is the key memory number.
- **TTFT P50, P99**.
- **TPOT P50**.
- **E2E P50, P99**.
- **Throughput** (requests/sec and output tokens/sec).

Publish as a table. Expected shape: FP8 → half the weight bytes, modest-to-substantial throughput improvement (varies with workload and engine), modest-to-no accuracy drift.

### Artefact E — Logit-diff and sampled-output comparison

For one deployment pair (either vLLM-BF16 vs. vLLM-FP8 or TRT-LLM-BF16 vs. TRT-LLM-FP8), produce the chapter-03 level-1 and level-2 validation:

- **Logit-level**: a fixed 50-prompt set, temperature 0.0, capture top-10 logits for the first 20 generation tokens per prompt. Compute **max abs logit diff**, **mean abs logit diff**, and **top-1 / top-5 agreement rate**. Report per prompt in a table; summary stats at the end.
- **Sampled-output**: on the same 50 prompts (temperature 0.0 → deterministic greedy), compute **exact-match rate** and **edit-distance to the baseline**. Report summary stats and show 3–5 example prompts where outputs differ.

Narrative (≤ 300 words): what the shape of the differences looks like, where they concentrate (prompt type? token position?), and what hypothesis you'd test if the differences are larger than expected.

## Starter guidance

- **Pin everything.** CUDA, driver, Python, vLLM, TRT-LLM, model revision. Record every pin in a one-page `environment.md` you reference from each artefact.
- **Baseline before quantising.** Run the full benchmark on BF16 first. It is both the reference and the sanity check for the benchmark harness itself.
- **Smallest viable model first.** 8B before 70B. Troubleshoot on 8B; replicate on 70B.
- **Use the same seed and the same prompt set across all four deployments.** Any divergence in prompts or seeds destroys the comparison.
- **For TRT-LLM, build on the same hardware you'll serve on.** The `.engine` is architecture-specific; a build on H100 does not run on A100.
- **Check both FP8 weights-and-activations and FP8 KV separately.** Flip one at a time to isolate which contributes what. Common pattern: FP8 weights-and-activations with BF16 KV first; add FP8 KV as a second step.
- **If TRT-LLM build complains about max shapes, pick the envelope that matches your benchmark**, not the model's maximum capacity. Rebuild later if envelope changes.
- **Use the vLLM `benchmark_serving.py` tool** or an equivalent custom harness. Do not try to measure throughput by eye.

## Fallbacks for constrained hardware

- **No Hopper hardware available.** Ada-class GPUs (L40S, L4) also support FP8. If you only have Ampere, this exercise cannot complete as specified — note the constraint and run the exercise on a cloud GPU rental for the FP8 portion only (keep the BF16 baseline local).
- **One Hopper GPU, no TP.** Stay on 8B models; skip the 70B path.
- **No TRT-LLM license / tooling access.** The exercise is still valuable with only vLLM (skip Artefact C and the TRT-LLM half of Artefact D). State the gap in the writeup.
- **Insufficient HBM to run BF16 70B.** Scope down to 8B; don't fake baselines on a smaller model.

## Acceptance criteria

1. **Artefact A** names the FP8 variant source (pre-quantised ID or self-quantisation command), its SHA-256 per file, its sub-format (E4M3 almost always), and whether KV is FP8.
2. **Artefact B** shows a running vLLM FP8 deployment with a documented launch command, startup log, and health-check pass.
3. **Artefact C** shows a running TRT-LLM FP8 deployment with the full convert → quantise → build pipeline documented, `.engine` file hash, and health-check pass.
4. **Artefact D** has HBM, TTFT, TPOT, E2E, and throughput numbers for all four (engine × precision) cells at each concurrency; the FP8 cells use roughly half the weight bytes and show measurable throughput improvement at modest-to-large concurrency.
5. **Artefact E** reports logit-diff and sampled-output metrics on 50 prompts; the summary stats are consistent with "FP8 is close to BF16 within tolerance" (a loose chapter-04 pass band at this stage).
6. All five artefacts live in version control; the `environment.md` is cross-referenced from each.

## Common failure modes

- **Flag name drift.** `--quantization fp8` may be the current spelling in vLLM; it may be something else (`--dtype fp8`?) in your version. Check the help text.
- **KV precision confusion.** FP8 weights-and-activations with BF16 KV cache is a *different* configuration from FP8 everything. Document which you ran.
- **TRT-LLM build-envelope mismatch.** Benchmarks that use longer context than the build's `max_seq_len` silently fail or crash. Set the envelope wide.
- **BF16 baseline uses different `max_model_len` than FP8.** Keep every non-precision flag identical; otherwise you are comparing two different serving configurations, not two different precisions.
- **Confusing per-GPU HBM with total HBM.** On TP=4, each GPU holds 1/4 of weights; the HBM-saved number is per-GPU, not total.
- **Sampling at temperature > 0 for logit comparison.** The comparison is between model distributions; use temperature 0 and seed to isolate the model from the sampler.
- **Comparing output text across different tokenisers.** The two deployments must use the same tokenizer; verify.

## Keep it

The deployment commands, environment manifest, and benchmark harness are reusable through exercises 02–06 and across future quantisation changes. Pin the harness; version the manifest. Exercise 04 (accuracy regression gate) consumes Artefact E's methodology directly.
