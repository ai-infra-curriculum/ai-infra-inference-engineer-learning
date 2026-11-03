# Exercise 02 — Deploy AWQ and GPTQ Weight-Only Serving

**Time:** 4 hours · **Output:** two quantised artefacts (AWQ-INT4 and GPTQ-INT4), two serving deployments, one memory + latency comparison, one output-equivalence table, one AWQ-vs-GPTQ decision writeup

## Goal

Make [Chapter 02](../02-quantization-algorithms-at-operator-depth.md) concrete for the two weight-only INT4 recipes by **producing both an AWQ and a GPTQ variant** of one stock model, **serving each**, and **comparing them** against the BF16 baseline on memory, latency, and output distribution. By the end you can defend an AWQ-vs-GPTQ choice for your workload with numbers.

## Problem statement

Memory-bound deployment of a mid-size LLM on commodity hardware is often an INT4 problem. The two production-ready INT4 recipes — AWQ and GPTQ — produce the same storage shape but via different algorithms with different accuracy signatures. The team needs a reusable comparison between them on your target model and workload, with calibration data documented and the choice defended.

## Prerequisites

- Any NVIDIA GPU with INT4 kernel support (A100, A10, L4, L40S, H100 — effectively any Ampere+). INT4 is weight-only and runs via software dequant through most of these kernels; it does not require FP8-capable hardware.
- CUDA 12.x + driver.
- Python with virtual environments.
- A pinned vLLM install with AWQ and GPTQ kernels enabled (Marlin kernel ideally).
- `autoawq` (or `llm-compressor` with AWQ support) and `auto-gptq` (or `llm-compressor` with GPTQ support) pinned.
- The exercise-01 benchmark harness.
- A calibration dataset (small — see Artefact A).

## Choose a model

Same shape as exercise 01: pick an 8B-class instruct model (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3). Larger if you have the hardware; the exercise is identical.

Record HF revision SHA and licence.

## Requirements

Deliver five artefacts.

### Artefact A — The calibration dataset

Before producing either variant, define the calibration dataset. The chapter-02 guidance applies:

- **Composition** should match production traffic. If you don't have a specific target workload, use a mix: 50% general instruct (dolly / alpaca / open-platypus subset), 25% chat multi-turn, 15% code, 10% math.
- **Size**: 128–512 samples for AWQ; 128–1024 for GPTQ. Use 512 as a default that serves both.
- **Length**: match production prompt length distribution.
- **Pinned**: content-hashed, version-controlled, licence-clean.

Document:

- The dataset identity (name, version, hash).
- The composition (what fraction of each source).
- The licence of each source and the composite licence.
- The sample length distribution (median, p99).

**This dataset is required input for the chapter-05 release contract's `calibration` fields.** Treat it as a first-class artefact.

### Artefact B — The AWQ-INT4 variant

Produce the AWQ-INT4 variant using `autoawq` (or `llm-compressor` with the AWQ recipe).

Document:

- The exact command / script used.
- Config: `w_bit=4`, `q_group_size=128` (default), `zero_point=True` (default — but verify the target kernel supports it).
- The calibration dataset consumed (reference Artefact A).
- Runtime of the quantisation step.
- Output artefact SHA-256 per file.
- The `quantize_config.json` that was produced.

### Artefact C — The GPTQ-INT4 variant

Produce the GPTQ-INT4 variant using `auto-gptq` (or `llm-compressor` with the GPTQ recipe).

Document:

- The exact command / script used.
- Config: `bits=4`, `group_size=128`, `desc_act=False` (unless your target kernel supports `act_order`), `sym=True` or `False` (match the target kernel).
- The calibration dataset consumed (same as Artefact A — pin to the same dataset for apples-to-apples with AWQ).
- Runtime of the quantisation step.
- Output artefact SHA-256 per file.
- The `quantize_config.json`.

### Artefact D — Memory and latency comparison

Serve all three (BF16 baseline, AWQ-INT4, GPTQ-INT4) on the **same** engine (vLLM is the recommended default). Run the exercise-01 benchmark harness against each with the same workload.

Measure for each (variant, concurrency) cell:

- **Peak HBM used** (via `nvidia-smi` or engine report).
- **TTFT P50, P99**.
- **TPOT P50**.
- **E2E P50, P99**.
- **Throughput** (requests/sec, output tokens/sec).

Plus two additional cells for the comparison:

- **Weight memory used** (via engine's weight-loader report — typically ~1/4 of BF16 for INT4).
- **KV memory used** (unchanged — INT4 is weight-only; KV stays BF16 unless flagged separately).

Publish as a table.

Expected shape: INT4 variants show ~75% weight memory reduction, measurable throughput improvement at low concurrency (memory-bound regime), smaller improvement at high concurrency (compute-bound regime), and TTFT impact depending on the dequant-kernel efficiency.

### Artefact E — Output equivalence table and AWQ-vs-GPTQ writeup

For both AWQ and GPTQ variants, produce the chapter-03 level-1 and level-2 comparison against the BF16 baseline (same methodology as exercise 01):

- 50-prompt fixed set with temperature 0.0.
- Logit top-10 capture for the first 20 generation tokens.
- Compute **max abs logit diff**, **mean abs logit diff**, **top-1 / top-5 agreement rate** for each variant.
- Compute **sampled-output exact-match rate** and **edit-distance** against the BF16 baseline for each variant.

Publish per-variant tables.

**AWQ-vs-GPTQ writeup (≤ 500 words):**

- Which variant was closer to BF16 on logit metrics?
- Which was closer on sampled-output?
- Where did they diverge in the same direction (both wrong vs. BF16)?
- Where did they diverge in different directions?
- Which would you recommend for production serving of this model, and why?
- What would change your recommendation (different calibration data, different workload, different engine)?

**This is not a chapter-04 accuracy gate.** It is a developer-shaped comparison that informs the gate's threshold-setting in exercise 04.

## Starter guidance

- **Use `llm-compressor` if you want a single tool for both.** Neural Magic's `llm-compressor` has recipes for AWQ, GPTQ, SmoothQuant, FP8; using one tool keeps the calibration data pipeline identical.
- **Pin the quantisation tool version.** AWQ and GPTQ tooling has had regressions; the tool version is part of the artefact's reproducibility story (chapter 05).
- **Keep calibration data identical between AWQ and GPTQ.** Different data → unfair comparison.
- **Verify the engine kernel supports your chosen config.** The AWQ Marlin kernel in vLLM has specific requirements (symmetric vs. asymmetric; `act_order` vs. not). If the engine refuses to load your artefact, that is usually the cause. The flag-and-config error messages tend to be informative; read them.
- **Measure at both low and high concurrency.** INT4's throughput signature is concurrency-dependent.
- **Sampled-output equivalence depends on tokenizer agreement.** Verify all three variants load the same tokenizer bytes.
- **If GPTQ runtime is long** (hours on a 7B), use a smaller calibration count (128) for the first pass; grow if quality is insufficient.

## Fallbacks for constrained hardware

- **No Ampere+ GPU.** INT4 serving is not practical on Volta / Turing. Note the gap; exercise cannot complete.
- **Single consumer GPU.** 7B / 8B models serve fine; stay at that scale.
- **Insufficient time for GPTQ on 70B.** Scope to 8B; the comparison dynamics are the same.
- **Marlin kernel not available in your vLLM build.** The baseline AWQ / GPTQ kernels work (slower); note the kernel used.

## Acceptance criteria

1. **Artefact A** documents a pinned, content-hashed, licence-clean calibration dataset with composition and length distribution.
2. **Artefacts B and C** are produced artefacts (SHA-256 per file) with full command-line reproducibility.
3. **Artefact D**'s table includes BF16, AWQ, GPTQ across 1 / 4 / 16 / 64 concurrency with the specified metrics; INT4 variants show measurable weight-memory and throughput advantages.
4. **Artefact E** has side-by-side AWQ / GPTQ output-equivalence tables and a ≤ 500-word writeup with a defensible recommendation.
5. All five artefacts live in version control; cross-reference the environment manifest from exercise 01.
6. The calibration dataset's hash and composition appear in both quantised artefacts' `quantize_config.json` (chapter 05 lineage).

## Common failure modes

- **"AWQ and GPTQ gave the same numbers."** They sometimes do, especially on small models with well-behaved weight distributions. State it; don't fabricate differences.
- **Calibration data drift between AWQ and GPTQ.** Pin and verify by hash.
- **Wrong Marlin / exllama kernel for the config.** Symmetric vs. asymmetric and `act_order` matter.
- **Comparing INT4 to FP8 directly without a note.** They are different formats with different trade-offs; stick to BF16 baseline for both unless you specifically want the three-way INT4-vs-FP8-vs-BF16 comparison (which exercise 04 cross-references).
- **Measuring one variant with KV FP8 enabled and the other with BF16.** Pin KV precision to BF16 for this exercise to isolate weight precision.
- **Sampled-output exact-match near 0% at temperature > 0.** Use temperature 0; sampling noise destroys this comparison.

## Keep it

The AWQ and GPTQ variants (and their `quantize_config.json`) are inputs to exercise 04's accuracy gate. The calibration dataset is reusable across future INT4 and INT8 recipes. The AWQ-vs-GPTQ writeup becomes a team doc on which recipe is the default for this model family. Exercises 03 and 04 reference this exercise directly.
