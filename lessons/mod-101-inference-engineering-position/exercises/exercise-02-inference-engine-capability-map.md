# Exercise 02 — Inference Engine Capability Map

**Time:** 2 hours · **Output:** one capability-axes spreadsheet with citations, plus a short written recommendation

## Goal

Turn [Chapter 02](../02-engine-capability-map.md) into a *verified* capability map against today's engine documentation — not what the chapter claims, but what the current releases claim. The deliverable is reusable: when a new engine lands, you'll add a column; when a release changes a capability, you'll edit a cell.

## Problem statement

You are drafting the inference-platform tech-radar entry for engine selection. The platform leads need to see, at a glance:

1. Which engines are first-class candidates today.
2. Which workload shapes each one is best at.
3. Where the engines disagree (and why that matters when upstream behaviour changes).

## Requirements

Deliver three artefacts.

### Artefact A — Capability spreadsheet

A table (spreadsheet, Markdown, or Notion — must be exportable). Rows are the ten capability axes from Chapter 02. Columns are the engines:

- vLLM
- SGLang
- TensorRT-LLM (served via Triton or Dynamo)
- NVIDIA Dynamo (as orchestration layer)
- Triton Inference Server (generic serving)
- llama.cpp
- MLC-LLM
- TGI (historical compare-point)

For each cell:

- State the current support status (first-class / supported / experimental / unsupported / N/A).
- Cite the primary source (release notes, docs page, feature matrix) you verified against.
- Add a one-line nuance when the simple status doesn't tell the whole story (e.g., "FP8 supported on Hopper only; ROCm path lags").

The ten axes to use as rows:

1. Scheduling model (continuous batching, chunked prefill, static)
2. KV-cache primitive (PagedAttention, RadixAttention, contiguous, tiered)
3. Attention / fused kernel path (Flash Attention version, vendor libs)
4. Quantization matrix (FP16 / BF16, FP8, FP4 / NVFP4 / MXFP4, INT8 / INT4, AWQ, GPTQ, SmoothQuant)
5. Parallelism (TP, PP, EP, DP attention, sequence)
6. Disaggregated prefill–decode
7. Multi-adapter serving (static LoRA, dynamic LoRA swap, multi-LoRA)
8. Speculative decoding (draft-target, Medusa, EAGLE, Lookahead, n-gram)
9. API surface (OpenAI-compatible, Triton gRPC, custom)
10. Hardware backends (NVIDIA, AMD, AWS Neuron, Google TPU, Intel Gaudi, CPU, Apple)

### Artefact B — Commercial compare-point summary

A short section (≤ 1 page) placing the proprietary / commercial stacks (NVIDIA NIM, Fireworks, Together, Baseten, Modal, Groq, Cerebras) on three axes only:

- **What's unique** — a single sentence per vendor on the differentiating capability
- **Build-vs-buy fit** — one line on when a self-hosted engine beats using the vendor, and vice versa
- **Exit cost** — one line on how portable workloads are off the vendor

### Artefact C — Written recommendation (≤ 300 words)

Pick one concrete workload shape (make one up if you don't have a real one — be specific). For example: "70 B dense chat model, 4 k average input, 500 token average output, 50 % prefix reuse from a shared system prompt, NVIDIA H200 fleet, OpenAI-compatible API required." Pick the top engine candidate from the spreadsheet and defend the pick in ≤ 300 words. Call out *one* axis on which your pick is weaker than the runner-up and how you plan to mitigate it.

## Starter guidance

- Use primary sources (`resources.md`) not blog posts.
- When the current docs don't say, write `unknown — needs verification` in the cell. Do **not** fill in from the Chapter 02 claims uncritically; the chapter is from a point in time and release notes rotate.
- Separate "the engine supports X on hardware Y" from "the engine supports X on all its hardware backends." The second is almost never true.
- When an engine is "experimental" on an axis, note the release that is expected to land it. If you can't find that information, say so — the gap is itself useful.

## Acceptance criteria

1. Every non-trivial cell has a citation to a current primary source (not Chapter 02).
2. At least three cells differ from a naïve reading of Chapter 02, with a note on *why* (feature landed since; feature was removed; chapter was ambiguous).
3. The commercial compare-point section names a build-vs-buy threshold for each vendor, not just a description.
4. The written recommendation names the runner-up engine and the single axis the pick loses on.
5. A peer teammate who hasn't read Chapter 02 reads the spreadsheet and can pick a reasonable engine for a stated workload in under five minutes.

## Keep it

Commit this to your tech-radar / engine-evaluation-doc location. Add a column when a new engine lands; edit cells when releases change capabilities. Rerun the recommendation against the spreadsheet whenever you take on a new workload class.
