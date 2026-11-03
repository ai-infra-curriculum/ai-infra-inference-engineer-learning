# Exercise 03 — Prefix-Cache Hit-Rate Measurement on Chat and RAG

**Time:** 4 hours · **Output:** two synthetic workload generators (chat, RAG), two measured hit-rate reports, one TTFT delta analysis, one defended prefix-cache config, one cold-cache cliff demonstration

## Goal

Make [Chapter 03](../03-prefix-caching-for-chat-and-rag.md) concrete by **generating two realistic workloads** (chat with shared system prompt + conversation state; RAG with hot documents + varying queries), **measuring hit-rate and TTFT benefit on both**, demonstrating the **cold-cache cliff** after a worker restart, and defending a prefix-cache configuration against each workload. The output is a config you can take to a product-team conversation and defend with numbers.

## Problem statement

Your team has committed to shipping a chat product (shared system prompt, user conversations) and a RAG product (shared doc corpus, user queries) on the same inference fleet. The product team wants TTFT P50 under 500 ms. Prefix caching is the obvious lever. The question is: how much benefit, at what KV-pool cost, with what cold-start behaviour, under what eviction policy?

This exercise makes you measure your way to a defended answer.

## Prerequisites

- Exercise 02's working prefix-cache setup (single-worker is enough).
- A load generator (locust, `vllm` benchmark_serving, or a short Python script using `httpx`).
- A metrics scrape endpoint (Prometheus or the engine's `/metrics`).
- 2–4 hours of GPU time for the measurements. A 7B model on a single GPU is adequate.

## Requirements

Deliver five artefacts.

### Artefact A — Chat workload generator

Author a load generator that produces a controlled chat workload:

- **Shared system prompt**: ~1 500 tokens, stable across all requests. Keep it block-aligned (pad to 1 504 or 1 536 tokens as needed).
- **Conversation sessions**: N simultaneous sessions, each session is a sequence of turns. For each turn:
  - The prompt is `system_prompt + "\n".join(previous turns) + current user turn`.
  - Each new user turn is ~50 tokens of varied text (sample from a dataset).
  - Each turn's output is bounded by `max_tokens=100`.
- **Arrival pattern**: 10 RPS steady, with each session's turns separated by 2–5 seconds (simulating human typing).
- **Session lifespans**: each session runs 3–10 turns before terminating.
- **Session count**: enough concurrent sessions to drive the engine to a target `max-num-seqs` utilisation (e.g. 20–40 concurrent sessions on a `max-num-seqs=32` engine).

The generator is a reusable artefact; make it configurable (N sessions, turns per session, system prompt file).

### Artefact B — RAG workload generator

A separate generator for RAG:

- **Shared system prompt**: ~200 tokens, stable across all requests, block-aligned.
- **Document corpus**: 1 000 documents of varied length (sample from Wikipedia or similar). Pre-compute token counts.
- **Query shape**: each request is `system_prompt + retrieved_docs + user_query`.
- **Doc selection** follows a Zipf distribution over the 1 000 docs — top 10 docs account for 50% of retrievals, top 100 for 90%. Each query retrieves 3–5 docs (concatenated in a stable, deterministic order).
- **Query**: ~30 tokens, varied.
- **Arrival**: 20 RPS steady.

Make it configurable (Zipf exponent, doc count, docs per query).

### Artefact C — Measured hit-rate reports (markdown + plots, ≤ 3 pages)

Run both workloads against your engine with prefix caching on. For each:

- Run the warm-up (5 minutes) to populate the cache.
- Run the measured phase (10 minutes).
- Scrape metrics every 15 seconds; summarise over the measured phase.

Report, for each workload:

1. **Overall cache-hit rate** (per-request binary — did any block hit — and per-block ratio).
2. **TTFT distribution** split by cache-hit vs. cache-miss. Histogram or box plot preferred.
3. **TTFT P50, P95, P99** overall vs. cache-miss-only vs. cache-hit-only.
4. **Pool occupancy** — fraction of KV pool occupied by cached prefixes vs. live sequences.
5. **(RAG only) Stratified hit-rate by doc popularity.** Group docs into quartiles by retrieval frequency; report hit-rate per quartile.
6. **(Chat only) Hit-rate as a function of conversation turn.** Does turn 2 hit more than turn 1 (yes — the growing conversation prefix is now cached)? Does turn 5 hit more than turn 2? Confirm or refute.

A short narrative (≤ 500 words) interprets the numbers: does prefix caching deliver the TTFT win? For which traffic? What is the ceiling?

### Artefact D — Cold-cache cliff demonstration

Simulate a worker scale-up:

1. Run the chat workload at steady state for 10 minutes with prefix caching on. Record TTFT P99.
2. Restart the engine (`kill -TERM` + relaunch, or `systemctl restart`). The cache is now empty.
3. Resume the same chat workload immediately. Record TTFT at 15-second intervals for 10 minutes.
4. Plot TTFT P99 over time. Expectation: an initial spike as the cache repopulates (requests get cache-miss TTFT), then a decay to steady state within minutes.
5. Measure the **cliff duration** — time until TTFT P99 returns to within 10% of steady-state.
6. Measure the **peak excursion** — how much TTFT P99 rose during the cliff.

Attempt one mitigation:

- **Pre-warm**: before accepting production traffic, replay the top-5 system prompts synthetically to seed the cache. Rerun the restart experiment. Report the new cliff duration and peak excursion.

Document both the baseline cliff and the mitigated cliff.

### Artefact E — Defended prefix-cache configuration

A one-page configuration document that specifies, for your two workloads:

- **The flag set** (enable + eviction policy + any supported knobs).
- **The TTL / LRU decision** for conversation state (chat) and for RAG doc caching.
- **The expected hit-rate and TTFT delta** based on measurement.
- **The KV-pool budget** the cache will consume (as a fraction).
- **The cold-cache mitigation** (pre-warm script, cache-aware routing, or acceptance of the cliff with SLO adjustment).
- **The flag or action that turns prefix caching off** if it starts hurting (criteria: hit rate falls below X, pool occupancy above Y).
- **The invalidation story** for the system prompt and for the document corpus (how a change is deployed without stale cache).

This artefact is the deliverable the product team sees. Keep it crisp.

## Starter guidance

- **Chat prompt dataset.** For user turns, use ShareGPT or a similar dataset. Strip to ~50-token messages; append to the growing conversation.
- **RAG document source.** Wikipedia dumps work; so do CommonCrawl snapshots. Tokenise ahead of time; store per-doc token count.
- **Deterministic doc ordering** (chapter 03) matters. Sort retrieved docs by `(score, doc_id)` so the same doc set produces the same prefix every time.
- **Measure block-level hit rate, not request-level alone.** The TTFT win comes from blocks saved, not from the binary "did any block hit."
- **Run both workloads sequentially**, not concurrently. Mixing them confuses the measurement.
- **For the cold-cliff experiment**, pin a quiet period before the restart so the baseline is stable. Avoid running at the top of the hour if a cron job competes.

## Fallbacks for constrained hardware

- **Smaller model** (1B–3B) if 7B is tight. The hit-rate measurement is model-independent; only the TTFT numbers change.
- **CPU backend** acceptable for Artefacts A and B (generator fidelity) but will produce unrealistic TTFT numbers.
- **Lower RPS** if your rig cannot sustain 20 RPS. Scale down traffic proportionally and note so in writeup.

## Acceptance criteria

1. **Artefacts A and B** are reusable scripts (configurable via CLI or env, documented in a short README section). Running them with default params produces the workload described in the requirements.
2. **Artefact C** reports hit-rate numbers for both workloads: chat achieves ≥ 60% block-level hit-rate after warm-up; RAG achieves ≥ 30% (variable; depends on Zipf exponent). TTFT delta between hit and miss is reported with distribution, not just mean.
3. **Artefact D** shows a visible cliff on the baseline (TTFT P99 excursion at least 30% above steady-state, lasting at least 2 minutes). The mitigation produces a measurable reduction in either excursion or duration.
4. **Artefact E** fits on one page. Every field is populated with a value derived from measurement, not vibes.
5. All five artefacts live in version control.

## Common failure modes

- **System prompt not actually stable.** A timestamp, request-id, or user-name leaked into the system prompt zeros out the hit rate. Fix by inspecting the first 100 tokens of several requests and confirming they are byte-identical.
- **Session routing isn't sticky.** If chat sessions scatter across workers (your test setup has multiple engines), the per-session cache fragments. Confirm single-worker or configure sticky routing.
- **Doc order isn't deterministic.** Set-based retrieval produces different orderings run-to-run. Sort before concatenation.
- **Metric cardinality explosion.** Don't label hit-rate by `tenant_id` if you have 10 000 tenants; aggregate first. Chapter 06 warns about this.
- **Cold-cache cliff masked by lack of load.** If RPS is too low during the restart, the engine stays idle and there is no cliff to measure. Run at the same RPS as the baseline.

## Keep it

Artefacts A and B are reusable workload generators you reuse in exercises 04 and 05 and in module 110 observability drills. Artefact E is the model the product team will reference when scoping chat and RAG features.
