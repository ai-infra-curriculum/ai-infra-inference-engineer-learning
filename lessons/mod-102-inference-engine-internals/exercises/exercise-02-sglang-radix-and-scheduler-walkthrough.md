# Exercise 02 — SGLang Radix and Scheduler Walkthrough

**Time:** 3 hours · **Output:** one annotated source-tour document, one running SGLang server, one RadixAttention behaviour report, one cache-aware-routing probe

## Goal

Make [Chapter 02](../02-sglang-internals-and-operation.md) concrete by walking SGLang's source — focusing on RadixAttention, the zero-overhead batch scheduler, and the router — against a locally running engine. Exercise the engine with a *prefix-heavy* workload where SGLang's design pays off, and verify the behaviour against the chapter's claims.

## Problem statement

Your team is evaluating SGLang against vLLM for a chat workload with a 2–4k shared system prompt and a 1–2k per-request user suffix. The expected win is prefix caching at near-100% hit rate and a flat TTFT regardless of prompt length (since the shared prefix is prefilled once). You must:

1. Launch SGLang locally.
2. Walk its source at the three distinct layers (router, scheduler, RadixAttention allocator).
3. Run a prefix-heavy probe and verify cache-hit telemetry.
4. Run a prefix-churn probe and verify the degradation profile.
5. If you have multiple GPUs available, run a two-worker cache-aware routing probe.
6. Produce a one-page operator cheatsheet.

## Prerequisites

- A workstation with at least one NVIDIA GPU, or a lab machine. Two GPUs preferred for Artefact D.
- Python environment with a recent SGLang release pinned (verify the current install instructions from the SGLang docs; the install command rotates with the attention / kernel backend).
- `git clone` of the matching SGLang source at the same version.

## Requirements

Deliver five artefacts.

### Artefact A — Annotated source tour (markdown, ≤ 2 pages)

Name, for your pinned SGLang version:

1. The server-launch entrypoint (equivalent of `vllm serve`) and the HTTP-handler file.
2. The router / tokenizer-manager class and the function that decides which worker serves a request.
3. The scheduler class and the function implementing one iteration (admission, prefill/decode selection, chunked-prefill interleaving).
4. The RadixAttention data structure: file and class name of the radix tree, the node struct, the `match_prefix` (longest-prefix lookup) function, the `evict` function, and the eviction-policy module.
5. The model runner and worker — including the DP-attention entry point if present.
6. The structured-generation path — which grammar backend (XGrammar / Outlines / native) is wired into the sampling layer, and where constrained-decoding mask-construction lives.

Each item: **file path + function name**, one sentence of what it does. Do not copy the code — map it.

### Artefact B — A running SGLang server for diagnostic use

Launch locally with:

- A small model (1B–7B) that fits on your dev GPU.
- Modest `--context-length`.
- `--max-running-requests` modest (so preemption is reachable).
- Prefix caching / RadixAttention on (verify current flag names).
- `--enable-torch-compile` on, so you can also observe the compile warmup behaviour.
- Metrics endpoint exposed.

Capture: your exact launch command, the server startup log, a `curl` of the metrics endpoint.

### Artefact C — RadixAttention behaviour report

Author a Python probe that runs *three* workload patterns against the server and reports metrics per pattern:

1. **Prefix-heavy** — 50 requests, each with the same ~2k-token system prompt plus a ~200-token unique user suffix. Expected: near-100% prefix-cache hit rate after the first request, flat TTFT regardless of request number, low prefill compute.
2. **Prefix-churn** — 50 requests, each with a *different* ~2k-token prompt (no shared prefix). Expected: ~0% cache-hit rate, TTFT dominated by prefill cost, high prefill compute, eviction counter active.
3. **Mixed** — alternate requests from the two patterns above. Expected: cache-hit rate oscillates; eviction may rise if the churn pattern pushes the pool.

For each pattern, record:

- Mean / P50 / P99 TTFT.
- Mean TPOT.
- Cache-hit rate.
- Radix-tree node count over time (if the engine exports it; otherwise approximate from logs).
- Eviction rate.
- GPU memory used by the KV pool.

Interpret the three patterns against the chapter claims. Does the engine behave the way RadixAttention is supposed to behave?

### Artefact D — Cache-aware routing probe (two-GPU minimum; mark N/A otherwise)

Launch SGLang with two workers (either two GPUs in a TP=1 configuration or a two-worker setup appropriate to your SGLang version). Expose the router metrics.

Run two scenarios:

1. **Session-sticky prefix** — a client that sends 20 requests, each with the same prefix. Verify (via logs or router metrics) that all 20 land on the *same* worker and that the cache stays hot.
2. **Prefix-split** — two concurrent clients, each with a distinct prefix. Verify the router balances them to different workers to keep both caches independently hot.

Report: were your expectations met? If the router did not route cache-stickily, what metric or log tells you why?

If you only have one GPU, mark this artefact N/A and briefly describe how you would run it on a two-GPU setup. Do not fake data.

### Artefact E — One-page operator cheatsheet

A single-page summary:

- Launch flags, grouped by the five layers (API, router, scheduler, RadixAttention, runner).
- The top-10 metrics for SGLang that correspond to each layer.
- Source entry points from Artefact A, one per layer.
- A clear statement of **when SGLang wins over vLLM** (prefix-heavy, structured output, DP-attention MoE) and **when it doesn't** (prefix-churn-dominated workloads).

## Starter guidance

- Pin the version. SGLang evolves fast; a mismatched source clone will produce the wrong map.
- The structured-generation path can be observed with a small probe sending constrained-JSON requests. If you don't need it for this exercise, note it in the source tour and skip the dynamic probe.
- `--enable-torch-compile` warmup is first-call-slow. Give the server a warmup run before measuring.
- If the engine exports a Prometheus `/metrics` endpoint, use it. If it only exports logs, parse logs — but note the drift and open an upstream feature request if observability is thin.
- Fallbacks: CPU-only mode (if supported in your version), ROCm (verify feature parity), cloud spot GPUs for Artefact D.

## Acceptance criteria

1. The source tour names six distinct entry points in the pinned SGLang tree with file paths and function names a `grep` can confirm.
2. The server runs under all probes without crashing.
3. The behaviour report (Artefact C) shows a clear, visible difference between the three patterns on cache-hit rate and TTFT. The prefix-heavy pattern reaches a near-100% cache-hit rate; the prefix-churn pattern does not.
4. The cache-aware routing probe (Artefact D) either demonstrates sticky routing or produces evidence (metrics / logs) explaining why the router behaved differently.
5. The cheatsheet (Artefact E) fits on one page and tells a reader when to pick SGLang over vLLM for a specific workload.

## Common failure modes

- **Faking shared prefixes.** If your "shared prefix" differs by one token across requests (a timestamp, a stray whitespace), the cache will miss. Dump and verify the actual token ids being sent.
- **Confusing warmup cost with steady-state latency.** First call includes compile and graph capture; discard warmup iterations.
- **Expecting cross-worker KV reuse.** SGLang's cache-aware routing is per-worker; cross-worker KV transfer is tiered-KV / disaggregated-serving territory (modules 104, 107).

## Keep it

Pin the cheatsheet to the team wiki. Pair it with the vLLM cheatsheet from exercise 01 so a new on-call engineer has both. Modules 103–104 will reference both when the scheduler and KV allocator get deep treatments.
