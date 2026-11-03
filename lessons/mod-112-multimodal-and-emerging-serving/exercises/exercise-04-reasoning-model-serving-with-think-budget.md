# Exercise 04 — Reasoning-Model Serving with Think Budget

**Time:** 4 hours · **Output:** one running reasoning-model endpoint with budget enforcement, one interruptible-decode verification, one reasoning-vs-answer channel implementation, one UX-shaped demonstration, one updated SLO document

## Goal

Make [Chapter 04](../04-reasoning-model-serving.md) concrete by **deploying a reasoning-capable model, wiring a thinking-budget parameter through the ingress, implementing interruptible decoding, exposing the reasoning-vs-answer channel split, and proving all of it against a UX-shaped test**.

## Problem statement

Chapter 04 names four pieces that classical text serving does not have: a thinking-budget cap, interruptible decoding whose latency matters, a two-channel output surface, and a product decision about whether to stream the reasoning. This exercise carries each into a working endpoint and a verification report.

You will deploy a reasoning-capable open-weights model, expose the budget parameter in an OpenAI-compatible-ish API, implement the cancellation path that stops a trace within one decode step, publish the reasoning and answer as distinct channels, and demonstrate the result with a developer-facing and a user-facing UX variant.

## Prerequisites

- Chapter 04 of this module read.
- Chapter 02 of module 111 (streaming, cancellation) read.
- A GPU host capable of running a reasoning-capable open-weights model. Reasonable 2026 candidates (verify against your engine's current support): DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1 (if your hardware supports the full MoE), QwQ-32B / 72B, or a recent Qwen3 / Llama-4 reasoning variant. <!-- needs-research: confirm current open-weight reasoning models that are mainline-supported on vLLM / SGLang / TRT-LLM. -->
- An engine that recognises the model's reasoning-transition tokens or that can be configured to emit `reasoning` content distinctly. In 2026 vLLM, SGLang, and TRT-LLM all have some reasoning-mode support — confirm against the version you pin.
- Python 3.11+, the OpenAI client SDK, `anthropic` SDK (to mirror the shape).
- The streaming-proxy from module 111 exercise 02 (optional; chapter 02's cancellation plumbing is reused here).

## Requirements

Six deliverables.

### Deliverable A — the deployment manifest

A reproducible script / manifest that:

- Stands up the reasoning model on the chosen engine with reasoning-content parsing enabled.
- Exposes `POST /v1/chat/completions` (or `/v1/responses` if your engine supports it) with:
  - A `reasoning.budget_tokens` parameter (your field, ingress-level) that clamps to a server-side maximum.
  - Streaming support with typed events for reasoning and answer channels.
- Documents the engine's current flag names (they vary; cite the engine's docs).

Verify: a simple prompt ("What is 17 × 24? Think step by step.") returns reasoning content separate from the final answer, and total output tokens are visible in `usage`.

### Deliverable B — the budget-enforcement verification

Send three requests against a hard prompt (e.g., a math olympiad problem that genuinely benefits from long thinking):

- `budget_tokens = 100` — expectation: the reasoning phase cuts off early; the answer may be weak but is produced.
- `budget_tokens = 2000` — expectation: a fuller reasoning phase; a better answer.
- `budget_tokens = 20000` with a server-side max of 5000 — expectation: clamped to 5000, surfaced in a response header or `usage` field.

For each:

- Capture the reasoning-token count and the answer-token count separately.
- Capture the `finish_reason` and whether the stream closed on `stop`, `length`, or an engine-specific soft-transition code.
- Record the P95 answer quality by eye (not rigorous, but sufficient for the exercise).

Document the shape at the budget boundary — did your engine do a soft transition to the answer (preferred) or a hard stop with reasoning-only output? Chapter 04 called this out; your deliverable confirms which your deployment actually does.

### Deliverable C — interruptible decoding

Add a cancellation hook to the ingress. Two triggers:

1. **Client disconnect.** The SSE / WebSocket connection drops; the engine aborts the request.
2. **Explicit cancel endpoint.** `POST /v1/responses/<id>/cancel` or an equivalent — the client calls it mid-generation; the engine aborts.

Verify:

- Start a reasoning request with `budget_tokens = 20000`.
- Wait 5 seconds (some reasoning has emitted), then cancel.
- Measure wall-clock from "cancel sent" to "engine KV released" (poll `kv_cache_usage` from the engine `/metrics`). Target: under 500 ms.
- Verify the response includes a `finish_reason: "cancelled"` (or equivalent) and the actual tokens generated, not a 500.

Repeat the cancel during the *answer* phase, not just the thinking phase. Both must work; chapter 04's "cancelled reasoning request keeps generating for 30 seconds" failure mode is the thing you are verifying is not present.

### Deliverable D — reasoning-vs-answer channel implementation

Pick one of the two channel shapes (OpenAI `/v1/responses` output items, or Anthropic-style `thinking` content blocks) and implement it end-to-end:

- The streaming event schema is typed correctly: a `reasoning` event stream before the answer stream, or two labelled content blocks with correct start / stop events.
- The non-streaming response places reasoning and answer in separate fields.
- Document in a README which shape you picked and why.

Then write a translation shim: accept the same request in the shape *you did not implement* and translate to / from your canonical shape. This proves the ingress can serve both client worlds without changing the engine.

Verify with a scripted test case against both shapes; assert the correct event sequence and field layout.

### Deliverable E — the UX-shaped demonstration

Build two tiny demo clients that consume the same endpoint:

1. **Developer dashboard client.** Streams the reasoning to a collapsible panel; streams the answer to a main panel. Shows token counts per channel; shows live TTFR, TTFA.
2. **End-user client.** Hides the reasoning entirely; shows "Assistant is thinking…" with a progress indicator; starts streaming the answer when the answer channel opens.

The clients are simple (a terminal UI, a Jupyter notebook, a one-page HTML — pick what you can throw together). The point is to prove the endpoint supports both UX shapes without redeployment.

### Deliverable F — the SLO document update

Extend the module-110 chapter-02 SLO template with a reasoning workload class:

- **TTFR (time to first reasoning token).** P95 target.
- **TTFA (time to first answer token).** P95 target.
- **Per-request wall-clock.** P95 target; higher than chat.
- **Budget-cap-hit rate.** Fraction of requests that hit the server-side maximum; alertable above a threshold.
- **Cancellation-latency.** P95 of "cancel sent" to "engine KV released."

Pick targets consistent with your hardware's measurements from deliverable B. The document is the artefact your SRE partner reviews.

## Starter guidance

- **Confirm the model's transition-token protocol before you build on it.** DeepSeek-R1-style models use `<think>` / `</think>`; some engines strip these before the stream and emit tagged events; others emit the raw tokens and leave parsing to the client. The deliverable-A verification is where you pin down what your engine does.
- **Billing should count reasoning tokens.** The engine reports them; your `usage` object should include them (e.g., `reasoning_tokens` alongside `completion_tokens`). Chapter 04 called this out; the exercise's audit should confirm it on your endpoint.
- **Measure both TTFR and TTFA.** TTFR is classical TTFT; TTFA is TTFR plus `(reasoning_tokens × TPOT)`. Record both on every request.
- **Cancellation tests must cancel during the thinking phase.** The engine's abort hook may be only tested on answer-phase cancels in the engine's own test suite; your exercise should cover both.
- **Clamp the server-side maximum.** A user asking for 1 M reasoning tokens is a denial-of-service in disguise. Pick a reasonable ceiling (e.g., 32k) and enforce it at the ingress.
- **Translation shim is a readability exercise.** The code that maps `thinking` content blocks ↔ `reasoning` output items is 30–50 lines. Write it clean; it is the artefact that proves the ingress can be portable.

## Fallbacks for constrained environments

- **No GPU large enough for a reasoning model.** Use a small distilled reasoning model (DeepSeek-R1-Distill-Qwen-7B or similar) on a 24 GB GPU; the arithmetic is the same.
- **No engine with native reasoning-mode parsing.** Serve a reasoning model via a plain chat endpoint; implement the `<think>` / `</think>` parser at your ingress (a small state machine over the SSE stream). Deliverable D then becomes "my ingress is the reasoning-channel splitter."
- **No `/v1/responses` support on your engine.** Target `/v1/chat/completions` only; implement the reasoning-channel split as a custom field in the response (`{"reasoning": "...", "content": "..."}`) with a documented extension. State the extension in your README.
- **No cancellation hook on the engine.** Minimum viable cancellation: the ingress stops relaying from the engine on client-disconnect, but the engine finishes generating. Document the limitation; cite the engine upstream bug or feature request for a true-abort hook.

## Acceptance criteria

1. **Deliverable A** reasoning endpoint stands up; a two-channel response is produced.
2. **Deliverable B** three budget-settings produce behaviour consistent with chapter 04's soft-vs-hard transition; shape documented.
3. **Deliverable C** cancellation works during both reasoning and answer phases; KV released under 500 ms.
4. **Deliverable D** one channel shape implemented end-to-end; the translation shim passes a scripted compatibility test.
5. **Deliverable E** both UX demos consume the same endpoint cleanly.
6. **Deliverable F** reasoning-class SLO document drafted with concrete targets drawn from your measurements.

## Common failure modes

- **Reasoning tokens not billed.** The engine reports a combined `completion_tokens` and your `usage` block doesn't split them. Operator surprise follows at the first invoice.
- **Soft-transition fails under heavy load.** Under memory pressure the engine's budget check lags; reasoning overshoots by hundreds of tokens before the transition fires. Measure and tune the check interval; chapter 04's `needs-research` note on the exact mechanism depends on the engine.
- **The translation shim loses the tokens for the thinking portion.** Shim maps `reasoning` → `thinking` but drops the content on an error path; client thinks the model answered directly. Scripted compatibility test catches this.
- **Cancellation works on the proxy but not on the engine.** The SSE relay stops; the engine keeps going to completion. Module 111 chapter 02's "cancel-without-release" failure mode; wire the engine abort hook end-to-end.
- **TTFR and TTFA conflated in the dashboard.** The panel labelled "TTFT" is really TTFA for this workload, confusing on-call. Rename explicitly in the dashboard.

## Keep it

The reasoning endpoint, the budget-enforcement verification, the cancellation measurement, the channel implementation and shim, the UX demos, and the reasoning-class SLO document are the fleet's reasoning-model artefact. Module 113 (production safety) will reuse the cancellation plumbing in its auto-rollback runbooks; module 114 (vendor evaluation) will reuse the SLO document for comparing reasoning-provider offerings.
