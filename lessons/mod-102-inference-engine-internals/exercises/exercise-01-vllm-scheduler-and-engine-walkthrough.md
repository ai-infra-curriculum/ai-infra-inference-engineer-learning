# Exercise 01 — vLLM Scheduler and Engine Walkthrough

**Time:** 3 hours · **Output:** one annotated source-tour document, one running vLLM server, one metric-to-code map, one traffic probe report

## Goal

Make [Chapter 01](../01-vllm-internals-and-operation.md) concrete by *reading the vLLM source bottom-up* against a locally running engine, then correlating the metrics it emits with the specific code paths that produce them. By the end you can open vLLM and point at the scheduler loop, the KV allocator, and the model runner in under five minutes.

## Problem statement

Your team is about to adopt vLLM as the production engine for a new workload. Before you tune it, you need to be sure that *you* — not just a blog post — can map a dashboard regression back to a code path. You will:

1. Launch a vLLM server locally against a small model.
2. Walk the five-layer anatomy in the actual source tree.
3. Run a traffic probe that exercises the scheduler, the prefix cache, and the preemption path.
4. Map every metric vLLM emits during the probe to the code path that emits it.
5. Produce a one-page cheatsheet that lets the next operator onto this engine do the same.

## Prerequisites

- A workstation or lab machine with at least one NVIDIA GPU (any Hopper-era or Ada card will do; even a consumer RTX card for a small model). AMD / Apple-silicon users may follow a reduced-coverage path using CPU-only mode or the ROCm backend — see starter guidance.
- A clean Python environment with a recent vLLM release pinned (`pip install vllm==<pinned-version>`). Record the exact version you used.
- `git clone` of the matching vLLM source at the same version. The source tour requires reading the exact code the server runs — not `main`.

## Requirements

Deliver four artefacts.

### Artefact A — Annotated source tour (markdown, ≤ 2 pages)

A short document that names, for your pinned vLLM version:

1. The entrypoint file for `vllm serve` and the FastAPI handler for `/v1/chat/completions`.
2. The engine class that owns the request queue (`AsyncLLMEngine` or current equivalent) and the method that advances one step.
3. The scheduler class and the function that implements one iteration (admission, prefill/decode selection, preemption).
4. The block manager / allocator class and the functions for `allocate`, `append_slot`, `free`, `swap_out`, `swap_in`, and `fork`.
5. The model runner class and the function that executes the forward pass; the point at which CUDA-graph capture happens; the point at which `enforce_eager` is checked.
6. The worker / executor class — including whether your version uses the Ray executor, the multiprocess executor, or something newer. Name the file.

For each item: give **file path + function name**, plus one sentence on what the code does. Not a copy of the code — a map.

### Artefact B — A running vLLM server launched for diagnostic use

Launch a local server with a small model (e.g., a 1B–7B instruct model that fits on your dev GPU). Choose flags that *provoke* behaviour you can observe:

- `--max-model-len` modest (so you can push prompts up against it).
- `--max-num-seqs` low (say 16) so preemption is reachable.
- `--enable-prefix-caching` on.
- `--enable-chunked-prefill` on.
- `--metrics-port` exposed and scraped locally with `curl` or a tiny Prometheus/Grafana.

Capture: your exact launch command, the engine startup log, and a screenshot of `curl http://localhost:<metrics-port>/metrics | head -50`.

### Artefact C — Traffic probe + metric-to-code map

Author a small Python probe (any `httpx` / `openai`-SDK script will do) that fires three traffic patterns at the server, back to back:

1. **Prefix-heavy traffic** — 20 requests sharing a long system prompt plus a small per-request suffix.
2. **Cold-prefix traffic** — 20 requests with unique, long prompts (no shared prefix).
3. **Overcommitment traffic** — a burst of concurrent requests designed to push `--max-num-seqs` and force preemption.

For each pattern, record the dashboards or metric scrapes. Then build a **metric-to-code map** — a two-column table listing, at minimum:

| Metric name | Code path that emits / updates it |
|---|---|
| `vllm:prefix_cache_hits_total` | file:function from Artefact A |
| `vllm:num_preemptions_total` | file:function |
| `vllm:gpu_cache_usage_perc` | file:function |
| `vllm:time_to_first_token_seconds` | file:function |
| `vllm:time_per_output_token_seconds` | file:function |
| `vllm:num_requests_running` / `vllm:num_requests_waiting` | file:function |
| `vllm:scheduler_running_queue_size` or current scheduler-depth metric | file:function |

Add at least three more metrics *you* found interesting during the probe. If a metric name has changed in your version, use the current one and note the drift.

### Artefact D — One-page operator cheatsheet

A single page (markdown or diagram) that another engineer could use to:

- Launch vLLM with annotated flags grouped by the five layers.
- Pull the top-10 metrics that correspond to each layer.
- Find the five entry points in source (from Artefact A) when a dashboard anomaly needs a code answer.

## Starter guidance

- **Pin the version.** `vllm==<version>` and the matching git tag. Everything in the source tour has to match the running server, or the exercise generates its own confusion.
- **Small models only.** You are learning the engine, not benchmarking. A 1B–7B model keeps the KV footprint small and makes preemption reachable on a dev GPU.
- **Prefer reading over running.** The exercise is a code tour first and a probe second. If you spend more time babysitting the probe than reading source, re-balance.
- **Metrics names rotate.** If a metric from the chapter isn't present in your version, don't fake it — note the drift and find the current one.
- **Use the chapter's five-layer map as the organising principle** of the cheatsheet. "Flags by layer, metrics by layer, source entry points by layer" is the mental model that stays useful for the rest of the module.

## Fallbacks for constrained hardware

- **CPU-only**: vLLM has a CPU backend path. Smaller models, slow steps, but the scheduler and allocator layers behave the same way. Note the backend in your writeup.
- **ROCm (AMD)**: use a model and dtype supported by vLLM's ROCm build. Verify feature parity for prefix caching / chunked prefill / speculative decoding on ROCm before trying to probe them.
- **Cloud GPU**: a 1-hour spot rental on a modest NVIDIA SKU is cheap and avoids local-driver drift. Record the SKU in the writeup.

## Acceptance criteria

1. The source tour (Artefact A) names six distinct entry points in the pinned vLLM tree, with file paths and function names that `grep` can confirm.
2. The server (Artefact B) runs for ≥ 5 minutes under the probe without crashing, and the `/metrics` endpoint responds.
3. The traffic probe reaches each of its three patterns (prefix-heavy, cold-prefix, overcommitment) and the metrics show the expected signature: cache-hit rate climbing on pattern 1, dropping on pattern 2, preemption rate climbing on pattern 3.
4. The metric-to-code map has at least ten rows and every row is a *file:function* that `grep` can confirm, not a vibe.
5. The cheatsheet (Artefact D) fits on one page and is organised around the five layers.
6. A sibling-track teammate (or a peer in your own team) reads the cheatsheet, launches the same server, and makes sense of the metrics without opening Chapter 01.

## Common failure modes

- **Chapter-copying.** The deliverable is *your pinned version's* code paths, not the generic chapter claims. If your source paths don't `grep` cleanly, re-check against your installed version.
- **Benchmarking confusion.** This exercise is not a performance study. Numbers are secondary; the point is diagnostic fluency.
- **Metric-name drift.** If you cannot find a chapter-named metric in your version, your writeup should state so and use the current name.
- **Over-ambitious model choice.** A 70B model on a workstation is a bad fit for this exercise and will spend your time on OOM debugging.

## Keep it

The cheatsheet is a reusable artefact. Pin it to your team wiki as "vLLM: five-layer map, flags, metrics, source entry points." Refresh it on every vLLM upgrade. Exercise 04 (failure-mode drill) and modules 103–104 will reference it directly.
