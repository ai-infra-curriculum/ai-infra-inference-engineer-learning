# Exercise 01 — VLM Serving with Visual-Token Cache

**Time:** 5 hours · **Output:** one running VLM replica, one preprocess sidecar, one visual-token-cache measurement report, one mixed-image-count batching stress report

## Goal

Make [Chapter 01](../01-vlm-serving.md) concrete by **standing up a VLM serving replica, wiring the image-preprocess stage, verifying visual-token prefix-cache hits, and proving the batcher stays stable under a mixed 0 / 1 / 3 / 12-image workload**.

## Problem statement

Chapter 01 names four seams that appear when a text-serving fleet adds vision: image preprocessing, visual-token expansion, visual-token prefix caching, and variable-image-count batching. This exercise carries each into a running system and a measurement artefact your fleet can keep.

You pick a VLM, deploy it on an engine of your choice, confirm the visual-token expansion matches the model's documented encoder behaviour, build (or verify) the prefix-cache path for image reuse, and run a mixed-image-count load that exposes the batcher's behaviour. The deliverable is a per-decision report that another engineer can read and reproduce.

## Prerequisites

- Chapter 01 of this module read.
- A GPU host capable of serving a 7B-class VLM at a usable throughput (one A100 / H100 / L40S is enough; a smaller single-GPU card is enough for a smaller VLM).
- A VLM supported by your chosen engine. Reasonable candidates in 2026 (verify against the engine's current support matrix): Qwen2.5-VL-7B-Instruct, LLaVA-OneVision-7B, Pixtral-12B, InternVL2.5-8B, Molmo-7B.
- An engine with VLM support — vLLM, SGLang, or TRT-LLM (+ Triton) are the three mainline options; pick one your team is likely to run in production.
- Python 3.11+, `httpx` or `openai` client SDK, `Pillow` or `pyvips` for image preprocess, `hashlib` for image hashing, `prometheus-client` for metrics scraping.
- Access to the engine's `/metrics` endpoint (module 110 chapter 03's wiring) so you can see prefix-cache hit counts.

## Requirements

Six deliverables.

### Deliverable A — the deployment manifest

A reproducible runbook (a Docker Compose file, a `kubectl` manifest, or a shell script) that stands up:

- The VLM engine with vision support enabled.
- A preprocess sidecar (a FastAPI or Flask service) that owns fetch / decode / resize / hash of images, exposed on the engine host.
- The OpenAI-compatible `/v1/chat/completions` surface reachable from a client.

Document which engine, which model, and which VLM-support flags you enabled. Cite the engine's VLM docs for the current flag names (they change).

### Deliverable B — the visual-token expansion table

Send single-image requests with a fixed prompt across the detail settings and image sizes:

| Image size | `detail` | Visual tokens measured | Visual tokens predicted | Delta |
|---|---|---|---|---|
| 448×448 | low | — | — | — |
| 896×896 | high | — | — | — |
| 1024×1024 | high | — | — | — |
| 2048×2048 | high | — | — | — |

For "measured," read the engine's prompt-token count from the response's `usage.prompt_tokens` minus the pure-text token count for the same prompt without the image. For "predicted," compute from the model's documented encoder parameters (patch size, tile size, `pixel_shuffle_ratio`) using the chapter-01 formula.

The report is the engineer-facing truth about "how many tokens is an image on this model."

### Deliverable C — the visual-token prefix-cache verification

Send a two-turn conversation where turn 1 uploads an image and turn 2 reuses the same image plus a new text question:

- Turn 1: capture `usage.prompt_tokens`, the engine's prefix-cache hit counter before and after, and the TTFT.
- Turn 2: capture the same. Expectation: the prefix-cache hit counter increases by at least the visual-token count for the reused image; TTFT on turn 2 drops meaningfully (target: 5× lower, verify on your hardware).

Then repeat with:

- Same image with a different hash (re-encode the PNG in a way that changes the byte hash but not the pixel content). Expectation: either the cache still hits (hash is on pixels, not bytes) or it misses (hash is on bytes); either answer is fine, but **the report must state which**.
- Same conversation shape with a completely new image. Expectation: text prefix still hits; image portion is a miss.

Document which engine knob / flag controls the image-hashing behaviour (cite the engine's docs; the current state is uneven across engines, per chapter 01).

### Deliverable D — the per-image / per-request budget enforcement

Implement (or verify) at the ingress:

- A per-image visual-token budget (choose a reasonable value for your model; e.g., 2500 tokens).
- A per-request image-count cap (e.g., 16).

Prove by:

- Submitting a request whose single image would exceed the per-image budget at `detail: high` → expect either an automatic downsample to `low` with a response header indicating the downsample, or a `413 Request Entity Too Large` with a readable error message.
- Submitting a request with more than the image-count cap → expect `400 Bad Request` with a message naming the budget.

Document the exact behaviour on the exceed path; downsample-vs-reject is a product choice and should be deliberate.

### Deliverable E — the mixed-image-count stress test

Replay a traffic mix of:

- 60 % text-only turns (~300 input tokens, 100 output tokens).
- 25 % one-image turns (one 1024×1024 image, ~300 text tokens, 100 output tokens).
- 10 % three-image turns (three 1024×1024 images, ~300 text tokens, 100 output tokens).
- 5 % twelve-image turns (twelve 1024×1024 images, ~300 text tokens, 100 output tokens).

At a sustained QPS that keeps the engine at 60–80 % GPU utilisation. Measure:

- TTFT P50 / P95 / P99 for *each workload class separately*.
- TPOT for each workload class.
- Admission-reject rate.
- KV-cache high-water mark.

Compare against a baseline run of pure text-only traffic at the same QPS. The report should document whether the text-only P95 TTFT degrades when the 12-image traffic is added — this is the "text latency regression on VLM enable" failure mode from chapter 01.

### Deliverable F — the batcher-tuning changes

Based on deliverable E's measurements, identify one engine knob to adjust (chunked-prefill size, `max-num-batched-tokens`, admission classification, separate replica pool) and rerun the stress. Document before / after.

Acceptance: text-only P95 TTFT recovers to within 20 % of the baseline after your tuning. If it does not, document why not and the next thing you would try — a negative result with reasoning is acceptable.

## Starter guidance

- **Hash images in the preprocess sidecar, not in the engine process.** Hashing a 5 MB JPEG is CPU-cheap; running it on the GPU host's main thread is fine as long as it isn't inside the scheduler loop. The sidecar gives you natural overlap with the fetch.
- **Confirm the engine's image-hash contract before building on it.** vLLM, SGLang, and TRT-LLM each handle multi-image prefix cache differently (see chapter 01's `needs-research` note). The deliverable-C report is where you pin down what your pinned version actually does.
- **Keep the image corpus local.** A test that depends on fetching 1024×1024 PNGs from the public internet is slow and flaky; precompute a fixture set of 100 images at known resolutions and point tests at a local URL.
- **Record `usage.prompt_tokens` for every request.** It is the ground truth for the visual-token count after the engine has seen the image. Compare to the chapter-01 formula for drift.
- **Instrument both the ingress and the engine.** The admission-reject rate at the ingress should match the engine's `admission_reject` counter; mismatches mean your ingress is budgeting differently from the engine.
- **Watch for OOM on the 12-image case.** If your model's context is 32k and the 12-image request computes to 25k+ visual tokens, the request can OOM the engine's planner if the admission check is lenient. Tighten the per-request budget first.

## Fallbacks for constrained environments

- **Single-GPU card smaller than 24 GB.** Use a smaller VLM (LLaVA-1.5-7B at 4-bit quantization, or Qwen2-VL-2B). The arithmetic is the same at a smaller scale.
- **No access to a production-grade engine.** Use vLLM via Docker; the VLM support is good enough for the exercise.
- **No metric-scraping setup.** Poll the engine's `/metrics` endpoint with `curl` between test phases; record the counters by hand into a CSV. Less convenient but sufficient for the report.
- **Cannot use a public dataset of images.** Generate a synthetic image corpus with Pillow (a set of 100 differently-coloured gradient-plus-shape images at the needed resolutions). Deliverable C's "same image, different hash" test still works.

## Acceptance criteria

1. **Deliverable A** stands up the VLM replica + preprocess sidecar; a text-plus-image chat request produces a reasonable response.
2. **Deliverable B** visual-token table filled in; predicted-vs-measured within 5 % across the rows.
3. **Deliverable C** prefix-cache verification report names the engine's image-hash contract and shows a TTFT drop on reuse.
4. **Deliverable D** per-image and per-request budgets fire correctly on exceed; error responses are readable.
5. **Deliverable E** mixed-traffic stress measured; the text-only latency regression under 12-image load is characterised quantitatively.
6. **Deliverable F** one tuning change demonstrably recovers the text-only P95 (or a documented negative result with the next-step plan).

## Common failure modes

- **Image hash mismatches between client and engine.** The client sends a `data:` URL, the engine re-encodes before hashing; two clients that encoded the same pixels with different PNG settings don't share a cache entry. Fix by hashing pixel content, not bytes, in the sidecar.
- **The engine treats each turn as a fresh request.** Multi-turn APIs vary on whether they re-tokenize the full conversation each turn. Confirm the per-turn prefix-cache behaviour on your engine; if the engine re-tokenizes from scratch, your text prefix shouldn't hit either.
- **12-image requests silently reduce detail.** Some engines auto-downsample above a threshold without surfacing it. Check `usage.prompt_tokens` against the predicted value to detect silent downsampling.
- **TTFT measured at the client, not the engine.** Client-side TTFT includes sidecar fetch / decode / resize time, which is *not* the engine's TTFT. Report both and name which is which.
- **Preprocess sidecar becomes the bottleneck.** If the sidecar runs CPU-bound on JPEG decode, the engine idles while images queue. Measure sidecar CPU under the deliverable-E traffic mix.

## Keep it

The VLM deployment manifest, the visual-token expansion table, the prefix-cache verification report, and the mixed-traffic stress report are the fleet's VLM artefact. Exercise 05 (agentic workload) and the module 113 canary patterns reuse this deployment as the baseline for VLM-aware rollouts.
