# Exercise 02 — Diffusion Serving with Scheduler Choice

**Time:** 5 hours · **Output:** one diffusion serving replica, one scheduler-sweep report with latency-vs-quality Pareto plot, one operating-point recommendation against a stated SLO, one denoising-batch stress report

## Goal

Make [Chapter 02](../02-diffusion-serving.md) concrete by **serving a diffusion model, sweeping step counts and schedulers, measuring the latency-vs-quality curve, picking an operating point against a stated SLO, and proving the denoising batcher works under concurrent load**.

## Problem statement

Chapter 02 names the step count × scheduler pair as the dominant latency knob in diffusion serving. This exercise turns that into a per-deployment Pareto measurement: for *your* model on *your* hardware, what does the curve look like, and where on it do you operate given a product SLO?

You will deploy a diffusion model, run a scheduler sweep at several step counts, compute a quality metric (CLIP score and / or a small-sample human / reference comparison), plot the Pareto curve, pick the operating point, and then stress the denoising batcher under concurrent traffic to prove throughput holds at your point.

## Prerequisites

- Chapter 02 of this module read.
- A GPU host (24 GB+ VRAM for SDXL-class; 48 GB+ for FLUX.1 / SD3). A single-GPU box is enough; multi-GPU batches multiply throughput but are not required.
- A diffusion model and runtime. Reasonable 2026 candidates (verify current recommendations): Stable Diffusion XL 1.0 (community baseline), FLUX.1-schnell / FLUX.1-dev, Stable Diffusion 3 / 3.5. For runtime: Hugging Face `diffusers` with `StableDiffusionXLPipeline` / `FluxPipeline`, ComfyUI's API mode, or vLLM's / SGLang's diffusion support if available in your pinned release. <!-- needs-research: confirm current vLLM / SGLang diffusion-support status; the mainline inference engines historically did not run diffusion, but 2025–2026 saw experimental support land. -->
- Python 3.11+, `diffusers`, `torch`, `transformers`, `openai-clip` or `open_clip_torch` for CLIP-score measurement, `matplotlib` for plotting.
- A prompt dataset for evaluation — a small subset (100–500 prompts) of a public benchmark (e.g. PartiPrompts, DrawBench, GenAI-Bench) or a product-representative prompt list.

## Requirements

Six deliverables.

### Deliverable A — the deployment manifest

A reproducible script / manifest that stands up the diffusion endpoint. Minimum: a FastAPI service wrapping a `diffusers` pipeline with:

- `POST /v1/images/generations` (OpenAI-shape endpoint; `extra` bag for `scheduler`, `steps`, `guidance_scale`, `seed` per chapter 02).
- Model loaded at startup; pre-warm pass to avoid cold-start.
- Health check endpoint.
- Metrics endpoint exposing `generate_latency_seconds` histogram (labelled by `scheduler` and `steps`), `step_latency_seconds` histogram, and `vae_decode_seconds` histogram.

Document the model, the GPU, the quantization (fp16 / bf16 / fp8 / int8 if any), and the `torch.compile` / `sdpa` / FlashAttention setting. Those pin down what the measurements mean.

### Deliverable B — the scheduler-sweep measurement

For each `(scheduler, steps)` combination in a grid like this, generate K images per combination (K ≥ 20, more if hardware permits) at a fixed resolution (1024×1024), guidance scale 7.0, fixed seed per image across all combinations:

| Scheduler | Steps |
|---|---|
| DDIM | 10, 25, 50 |
| DPM-Solver++ multistep | 10, 20, 25, 50 |
| Euler | 10, 20, 30, 50 |
| Euler-ancestral | 20, 30 |
| LCM / consistency-distilled (if available) | 2, 4, 8 |

For each combination record:

- Mean generate-latency (full pipeline: encode → denoising → VAE decode → save).
- P95 generate-latency.
- Mean step-latency.
- Mean VAE-decode-latency.
- CLIP-score (OpenAI CLIP ViT-L/14 or OpenCLIP ViT-H/14, cosine similarity between image embedding and prompt embedding, averaged across the K images).
- Optional: a small human A/B comparison against a reference (DDIM 50) on a 10-image random subset — note this is inherently subjective; use it for sanity, not as the primary metric.

<!-- needs-research: the LCM / distillation support depends on the base model. SDXL has `LCM-LoRA`; FLUX.1-schnell is natively few-step. Confirm which schedulers are compatible with your chosen base model before running the sweep. -->

### Deliverable C — the Pareto plot and operating-point recommendation

Plot CLIP-score (y) vs. generate-latency (x) for each `(scheduler, steps)` as a point; connect same-scheduler points to show each scheduler's trace.

Define a product SLO, written as:

```
"P95 generate-latency ≤ L seconds at CLIP-score ≥ C against the reference."
```

Pick an L (e.g., 4 s for an interactive endpoint; 1 s for a thumbnail-generator) and a C derived from the DDIM-50 baseline (e.g., within 2 % of the baseline's CLIP score). Identify the operating point(s) that satisfy the SLO on the Pareto frontier.

Write a one-paragraph recommendation: which `(scheduler, steps)` combination the endpoint should default to, and the reasoning against the measured curve.

### Deliverable D — the denoising-batch stress test

Run concurrent request bursts against the endpoint at your chosen operating point:

- Burst size: 1, 2, 4, 8, 16 concurrent requests arriving within 100 ms of each other.
- Measure: per-request latency, total wall-clock to clear the burst, GPU memory high-water, batcher's batch-size selection.

Expectation: as burst size grows, total throughput scales sub-linearly (because the UNet / DiT batches on a `(B, C, H, W)` axis and VRAM caps B). Document:

- The actual batch size the dispatcher achieved.
- The per-request latency increase as a function of batch size.
- The VRAM-bounded B ceiling for your model + hardware.
- Behaviour when the burst exceeds `MAX_BATCH_SIZE` (queueing, latency pile-up).

### Deliverable E — the VAE-decode isolation measurement

Verify that the VAE decode is a non-trivial fraction of the request. Run:

- 4 steps on LCM-distilled model → total latency, step-sum latency, VAE-decode latency. Compute `vae_decode_seconds / total_seconds`.
- 25 steps on DPM-Solver++ → same split.

The 4-step case should show VAE decode as 20–60 % of total; the 25-step case should show it under 10 %. Document the observed ratios. If VAE decode is a clear bottleneck on the short-step path, enable tiled VAE and re-measure; report the delta.

### Deliverable F — the response-path decision

Pick one of the three response-path patterns from chapter 02 (signed URL + blob storage, inline base64, progressive streaming) for your deployment. Justify in two or three sentences. Implement it (a local filesystem "blob store" serving signed URLs via a short-lived HMAC is sufficient — you don't need a real S3).

## Starter guidance

- **Freeze the seed across the sweep.** Same seed = same initial noise = images from different schedulers / step counts are directly comparable. Varying the seed is noise on top of the signal you care about.
- **Warm the pipeline before measuring.** The first request after startup pays compile / cuDNN-autotune / allocator costs that confound the mean. Run 2–5 throwaway requests at each `(scheduler, steps)` before the measured batch.
- **CLIP-score is a signal, not a verdict.** A 2-point CLIP-score difference between DDIM-50 and DPM-Solver-25 is noise on most datasets; a 20-point difference is real. Use it to filter the Pareto frontier, not to pick the single best scheduler.
- **Bench the VAE decode with and without tiling.** Tiled VAE changes peak VRAM and may slightly change per-pixel output; the trade-off is per-deployment.
- **Guidance scale affects step cost.** `guidance_scale > 1.0` doubles the UNet work per step (chapter 02); pick one value for the sweep and hold it fixed.
- **ComfyUI / `diffusers` scheduler names are not identical.** Map between the two carefully; the `DPMSolverMultistepScheduler` in `diffusers` is `DPM++ 2M` in ComfyUI; `EulerDiscreteScheduler` is `Euler`, etc. Cite your mapping.

## Fallbacks for constrained environments

- **Smaller VRAM.** Use SD 1.5 instead of SDXL; the sweep arithmetic is the same at smaller scale and the Pareto shape is similar.
- **No GPU available at all.** You can still run the sweep on CPU for a scaled-down model (e.g. SDXL-Turbo at 512×512 on a modern CPU); latencies will be 50× higher but the *relative shape* of the Pareto curve is preserved. Document that the absolute numbers are not production-usable.
- **No CLIP available.** Use image aesthetic score models (LAION aesthetic predictor) or a visual-similarity-to-reference metric (LPIPS against DDIM-50 baseline). Any quality axis that orders the sweep works for the Pareto exercise.
- **Few prompts.** 50 prompts on one seed-set is enough to see the Pareto shape; 500 is better. Report the sample size with the number.

## Acceptance criteria

1. **Deliverable A** deployment runs; `POST /v1/images/generations` produces an image file / URL.
2. **Deliverable B** sweep table filled in; at least 20 images per `(scheduler, steps)` combination; CLIP-score and latency recorded.
3. **Deliverable C** Pareto plot drawn; an SLO is stated explicitly; a defensible operating-point recommendation written in a paragraph.
4. **Deliverable D** batch-stress numbers captured for 1 / 2 / 4 / 8 / 16 concurrent; batcher behaviour described; VRAM ceiling named.
5. **Deliverable E** VAE-decode fraction measured for a few-step and a many-step run; tiled-VAE delta captured.
6. **Deliverable F** response-path implementation present and justified.

## Common failure modes

- **Scheduler misconfigured for the base model.** Some schedulers (LCM) only work with distilled model weights; applying them to a non-distilled base produces garbage output at any step count. Pin per-model which schedulers are valid.
- **CLIP score drifts because of resolution.** A `resize(512, 512)` before the CLIP forward produces a different score than the native 1024×1024 model output seen by CLIP at 224×224. Hold the CLIP preprocessing fixed across all sweep points.
- **The VAE decode is not what's slow — it's the IO.** Saving the image to disk and uploading to blob storage can dwarf the decode latency on fast hardware. Separate `vae_decode_seconds` from `save_and_upload_seconds` in the histograms.
- **Guidance scale 0 skipped by accident.** Some code paths special-case `guidance_scale == 1.0` to skip the unconditional forward and are 2× faster. Benchmarks at `guidance_scale: 0` or `1.0` do not reflect production.
- **Reproducibility broken by ancestral sampler.** `Euler-ancestral` and similar add noise per step; same seed does not give you the same image. Either hold the ancestral-noise seed too (if the pipeline exposes it) or exclude ancestral samplers from the direct-comparison column.

## Keep it

The deployment manifest, scheduler-sweep report, Pareto plot, batch stress report, VAE-decode measurement, and response-path implementation are the fleet's diffusion artefact. Exercise 06 (server-vs-edge decision) reuses the Pareto report to reason about whether a client-side distilled model is a viable alternative for product features with strict latency SLOs.
