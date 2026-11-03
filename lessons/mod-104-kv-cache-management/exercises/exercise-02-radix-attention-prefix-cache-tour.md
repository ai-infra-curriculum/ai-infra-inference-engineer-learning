# Exercise 02 — RadixAttention Prefix-Cache Tour

**Time:** 3 hours · **Output:** one annotated source-tour, one running engine with prefix-cache metrics, one documented prefix-hit experiment, one cache-aware routing demonstration, one one-page cheatsheet

## Goal

Make [Chapter 02](../02-radix-attention-and-prefix-sharing.md) concrete by locating the prefix-cache data structure in your engine's source, running the engine with prefix caching on, demonstrating prefix sharing with controlled traffic (identical prefixes, then diverging prefixes), measuring the cache-hit rate against the chapter's expectation, and demonstrating cache-aware routing between at least two workers.

## Problem statement

Chapter 03 will have you deploy prefix caching against real workloads. Before that, you need to see the mechanism in isolation — the radix tree (or hash table) that holds cached prefixes, the lookup path, the per-worker vs. fleet-level behaviour, and the cache-aware router that makes multi-worker sharing work.

Pick one engine. SGLang is the recommended default because RadixAttention is first-class and the router ships in the box; vLLM is fine (automatic prefix caching); TRT-LLM works but has fewer operator-facing hooks.

## Prerequisites

- A workstation with at least one GPU (sufficient for a 1B–7B model). For the two-worker routing demonstration, two GPUs (or two processes with CPU-only backends) are needed; see the fallback below.
- A pinned engine version with prefix-cache support (`--enable-prefix-caching` for vLLM; default-on for SGLang; `enable_block_reuse` or equivalent for TRT-LLM).
- Exercise 01's source tour as a prerequisite — the prefix cache sits on top of the block manager you already walked.

## Requirements

Deliver five artefacts.

### Artefact A — Prefix-cache source tour (markdown, ≤ 2 pages)

Author a short document naming, for your pinned engine version:

1. The prefix-cache data structure (radix tree, hash table, prefix index). Which file, which class.
2. The `match_prefix` (or equivalent lookup) function. File / line / signature.
3. The `insert` (or equivalent cache-populate) function.
4. The eviction policy and the function that implements it (LRU, TTL, size-weighted).
5. The point in the scheduler where the lookup is called — how the matched blocks get into the sequence's block table.
6. The point where cached blocks are released or evicted — the interaction with the chapter 01 reference counter.
7. (SGLang) The cache-aware router: file / class / routing function.
8. (vLLM / TRT-LLM) The absence of a built-in router, with notes on which external component owns it in your stack (gateway, Dynamo, LiteLLM-shape).

For each: **file path + function name + one sentence** on behaviour. A map.

### Artefact B — Engine with prefix-cache metrics

Launch a single-worker server with:

- Prefix caching enabled.
- A small model (1B–7B instruct).
- `--max-num-seqs` moderate.
- Verbose logging / metrics on the prefix-cache side. In vLLM, scrape `/metrics` and look for prefix-cache-related counters (names rotate; verify in your version). In SGLang, use the metrics endpoint and the radix-cache stats call. If no metric exists, add a local print in the lookup function logging `(request_id, matched_blocks, matched_tokens, cache_hit_bool)`.

Capture:

- The launch command.
- A ≥ 2-minute run under a probe that sends requests with controlled prefix overlap (see Artefact C).
- The scraped metrics endpoint at start, middle, and end of the run.

### Artefact C — Prefix-hit experiment (markdown, ≤ 2 pages)

Run three controlled traffic shapes and compare the cache-hit metric:

**Phase 1 — No overlap.** Send 50 requests with completely different prompts (sample from a dataset or generate with varied openings). Expectation: cache-hit rate ≈ 0% (after an initial warm-up that leaves nothing cacheable).

**Phase 2 — Full overlap.** Send 50 requests with *identical* 1 000-token prompts (same system prompt + same user turn). Expectation: first request fully misses; subsequent 49 requests have cache-hit on nearly all prefix blocks.

**Phase 3 — Partial overlap.** Send 50 requests with the same 500-token system prompt, then diverging 500-token suffixes. Expectation: cache-hit on the first ~500 tokens of each request, miss on the suffix. Block-level hit rate ≈ 50%.

For each phase document:

1. Per-request cache-hit block count and total block count. Compute per-request hit ratio.
2. Average TTFT across the phase (expect Phase 2 ≪ Phase 1 and Phase 3 somewhere between).
3. Pool-occupancy evolution during the phase (expect Phase 2's cache to grow then plateau; Phase 1's to grow unboundedly if the cache is retaining everything).

Submit the raw numbers plus a short narrative that argues the numbers match the chapter's claims. If any phase deviates from expectation (common: Phase 1 shows unexpected hits because of accidental overlap in sampled prompts), investigate and document.

### Artefact D — Cache-aware routing demonstration

Launch *two* workers behind a router. Three options for the router:

- **SGLang**: use the built-in cache-aware router that ships with the project.
- **vLLM**: launch two vLLM processes and a small Python HTTP router you write that hashes the first 1 024 tokens of each prompt and dispatches by hash. 50 lines of code.
- **Any engine + LiteLLM / Dynamo**: configure cache-aware routing where supported.

Run two experiments:

**Experiment 1 — Prefix-blind routing (control).** Point the router at round-robin. Send 100 requests with the same 1 000-token system prompt. Measure the fleet-level cache-hit rate.

**Experiment 2 — Cache-aware routing.** Point the router at the hash-by-prefix policy. Send the same 100 requests. Measure the fleet-level cache-hit rate.

Expectation: Experiment 1's hit rate is roughly `(N-1) / (2N)` because each worker caches its own share; Experiment 2's hit rate is closer to `(N-1) / N` because all requests hit the same worker. The TTFT delta between the two experiments is the fleet-level benefit of cache-aware routing.

Document:

- The two routing policies.
- The hit-rate numbers (per-worker and aggregate).
- The TTFT delta.
- Any surprises (e.g. the "cache-aware" shard also carries higher load — chapter 02's hotspot concern).

### Artefact E — Prefix-cache cheatsheet (one page)

A single-page summary that another operator could use to:

1. Enable prefix caching correctly on your pinned engine (flag name, defaults, verification).
2. Scrape the right metric to measure hit-rate.
3. Decide whether to turn it off (hit-rate threshold).
4. Know whether cache-aware routing is on, and where it lives in the stack.

Reuse exercise 01's cheatsheet format.

## Starter guidance

- **Prompt generation for Phase 1 (no overlap).** Use a dataset like ShareGPT, Alpaca, or Dolly. Sample 50 random prompts; verify by hashing the first 500 tokens of each and confirming all 50 hashes differ.
- **Prompt padding for Phase 3 (partial overlap).** Pad the shared system prompt to a block boundary (1 536 or 2 048 tokens if `block_size=16`). Chapter 02's "block-aligned prefix" rule matters here; if you forget the padding, hit-rate dips because sub-block partial matches don't count.
- **Metrics may be absent.** vLLM's prefix-cache metrics have evolved; some versions emit nothing. In that case, instrument the `match_prefix` call site locally and emit your own metric. Note the version in your writeup.
- **Two-worker routing on one GPU.** If you have one GPU, run two engine processes on the CPU backend for the routing demo. Mechanics are the same; TTFT numbers won't be realistic but the hit-rate comparison still works.
- **Router can be minimal.** The point is to compare two routing policies, not to build a production router. A 50-line Python HTTP proxy that reads the body and hashes the first N tokens is enough.

## Fallbacks for constrained hardware

- **Single GPU**: run both workers' engines on CPU backend for Artefact D. State so in writeup.
- **No GPU**: do everything on CPU backend with a 125M model.
- **No cache metric in engine**: add a local counter via a 10-line patch to the lookup function.

## Acceptance criteria

1. **Artefact A** names at least seven distinct source entry points with `file:function` references that `grep` can confirm. Includes the match function, the insert function, the eviction policy, and (where applicable) the router.
2. **Artefact B** demonstrates prefix caching is on by a captured metric or log excerpt. Zero-hit baseline before Phase 2 traffic begins.
3. **Artefact C**'s three phases produce hit-rates that match the chapter's claims within reasonable noise (phase 1 < 10%, phase 2 > 80%, phase 3 between 30% and 60%). Deviations are investigated, not hand-waved.
4. **Artefact D** shows a measurable difference between prefix-blind and cache-aware routing (hit-rate delta ≥ 20 percentage points on the controlled workload). TTFT delta is reported with distribution, not just means.
5. **Artefact E** fits on one page.
6. All five artefacts live in version control; the source tour pins to the engine's `git` SHA.

## Common failure modes

- **Prefix caching enabled but not firing.** Verify by sending a known-identical prompt twice and timing TTFT. If identical, caching is off or broken.
- **Tokenization-level mismatch.** Two "identical" prompts can tokenise differently if BOS tokens, special-token insertion, or chat-template formatting varies. Phase 2's "full overlap" starts with identical *token ids*, not identical text — verify.
- **Partial-block hits counted as misses.** Chapter 02's block-aligned-prefix rule means a 1 000-token shared prefix at `block_size=16` yields 62 hit blocks (62 × 16 = 992 tokens) and a non-cached fractional block. If you expected 1 000 ÷ 16 = 62.5, you got 62. That is correct, not a bug.
- **Router over-engineered.** The exercise asks for a routing comparison, not a production router. 50 lines of Python is enough.
- **Cache invalidation between experiments.** If you want each experiment to start from a cold cache, restart the engine (or wait out TTL / fill the cache with disposable prefixes). Document which you did.

## Keep it

Artefact A (source tour) feeds exercise 03's deployment work. Artefact E (cheatsheet) joins the module 104 reference set. Exercise 03 reuses the Phase-2 / Phase-3 traffic generators for larger-scale measurement.
