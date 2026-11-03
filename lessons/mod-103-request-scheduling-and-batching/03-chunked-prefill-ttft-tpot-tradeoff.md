# Chapter 03 — Chunked Prefill and the TTFT / TPOT Trade-off

> Chunked prefill is the single scheduler knob that most directly moves the TTFT ↔ TPOT trade-off. Enabling it is usually right. Choosing a chunk size is where the work is.

## Motivation

Chapter 01 named the central scheduler pain: a long prefill scheduled naively inflates every decode step sharing the same iteration. SARATHI (Agrawal et al., 2023; SARATHI-Serve, OSDI 2024 — see [`resources.md`](resources.md)) showed a clean way out: split a long prefill into fixed-size chunks and interleave those chunks with ongoing decode steps so that no single iteration is dominated by prefill. vLLM, SGLang, and TensorRT-LLM / Triton have all picked up some form of this ("chunked prefill" / "chunked context" / piggyback).

The feature is conceptually simple and operationally subtle. This chapter is the operator's version: when it helps, when it hurts, how to pick a chunk size, and how to tell when the default is wrong.

The specific flag names and defaults rotate between engine releases. The *shape* of the knob is stable; the *name* is not. <!-- needs-research: vLLM has defaulted chunked prefill to on/off differently across releases, and the SGLang / TRT-LLM equivalents have different defaults and chunk-size semantics. Verify the current flag spelling and default against the engine version your fleet runs. -->

## The problem chunked prefill solves

Recall the central dynamic:

> A big prefill scheduled naively is a latency bomb for every decode sequence running at the same time.

Mechanically: in one engine iteration the scheduler builds a batch of mixed prefill tokens + decode tokens and dispatches one forward pass. The wall-clock of that pass is bounded by:

- Compute for prefill tokens (long prompts → many token-positions × attention+MLP work).
- Memory loads for decode tokens (weights + per-sequence KV; dominated by weights for small batches).

For small batches on big models, decode-only iterations are weight-load-bound and relatively quick (tens of ms). A prefill of a 4000-token prompt in the same iteration adds compute that stretches the step to hundreds of ms. Every decode sequence in that iteration pays that stretch — TPOT jitter.

Without a mitigation, the scheduler faces a bad choice:

- **Prefill-first**: don't admit the long prompt until no decode sequences are running. TTFT of the long prompt is terrible (it waits for everyone to finish); goodput is wasted.
- **Prefill inline**: run the whole prefill in one step. TPOT of in-flight decoders spikes; TPOT SLO breached.

## The SARATHI insight

SARATHI (and its serving follow-up SARATHI-Serve; [`resources.md`](resources.md)) observed:

1. Each engine iteration has a budget — call it `max-num-batched-tokens`. In pure-decode this budget is mostly unused (one token per sequence × batch).
2. If you split a long prefill into fixed-size chunks sized to fit in the leftover budget, you can **piggyback** those chunks on decode iterations.
3. The result: no single iteration is dominated by prefill. TPOT variance stays tight. TTFT on the long prompt rises slightly (it takes several iterations to prefill all its chunks), but by far less than the prefill-first alternative.

Formally: given `max-num-batched-tokens = B`, running batch of `R` decode sequences, chunk size `C`, the step processes `R` decode tokens + `C` prefill tokens, with `R + C ≤ B`. The scheduler picks `C` from the remaining budget each iteration.

The win: TPOT variance drops dramatically. The cost: TTFT on long prompts is slightly higher than un-chunked prefill, and short-prompt workloads get no benefit (they already fit in one chunk).

## How chunked prefill lands in each engine

Three engine instantiations with slightly different knob surfaces. Verify against your version's docs.

### vLLM — `--enable-chunked-prefill`

A boolean flag that enables chunk-based scheduling of prefills. The chunk size is governed by `--max-num-batched-tokens` (and in some releases a separate `--max-num-partial-prefills` / long-prefill-threshold knob). As of recent vLLM releases this feature defaults on when supported by the attention backend; **confirm for your version**.

Operational knobs:

- `--enable-chunked-prefill` / disable — the on/off.
- `--max-num-batched-tokens` — the per-step token budget. The chunk fits inside this minus the decode tokens for running sequences.
- `--max-num-partial-prefills N` (release-dependent) — cap on concurrent partial prefills so the scheduler doesn't fragment attention into too many parallel chunk streams.
- `--long-prefill-token-threshold` (release-dependent) — the length above which a prefill is treated as "long" and chunked; shorter prefills may still run as one chunk.

### SGLang — "chunked context" / piggyback

SGLang's scheduler has had chunked-context support for several releases. Server-arg surface typically includes a `--chunked-prefill-size` (or similarly named) knob that sets the chunk size directly. The RadixAttention cache (module 104) composes with chunked prefill — a chunk can hit the prefix cache as cleanly as a full prefill.

### TensorRT-LLM / Triton (via `tensorrtllm_backend`)

In the TRT-LLM batch manager (C++) the knob is often spelled `enable_chunked_context` with `tokens_per_block` and `max_num_tokens` controlling the chunk envelope. The `tensorrtllm_backend` surface exposes these through the Triton `config.pbtxt` backend parameters. The TRT-LLM engine must be built with chunked-context support for the backend to use it.

Across all three the shape is the same: *long prefills decompose into chunks that fit inside the per-step token budget*. The names differ; verify in docs.

## When chunked prefill helps

Four workload shapes where enabling it (and tuning the chunk size) is a clear win:

### 1. Long-prompt chat / RAG with concurrent decode

Shape: P95 prompt ≥ 4000 tokens, concurrent decode traffic, tight TPOT SLO (e.g. ≤ 50 ms).

Un-chunked, each new long prompt causes a wave of TPOT spikes for in-flight sequences. Chunked prefill flattens the wave: TPOT variance drops, P95 TPOT holds the SLO.

### 2. Mixed-length arrival with bursts

Shape: a steady stream of short requests, interrupted periodically by a long request.

Un-chunked, the long request is a periodic latency grenade in the batch. Chunked prefill amortises its cost over multiple iterations; the short-request stream sees no visible disruption.

### 3. High concurrency, limited `max-num-batched-tokens`

Shape: you have a tight `max-num-batched-tokens` cap (perhaps because you need predictable iteration time), but prefills can be arbitrarily long.

Un-chunked, you literally cannot admit a 10k-token prompt without blowing the cap. Chunked prefill is the only way to admit long prompts within a tight step budget.

### 4. Agentic / tool-call-heavy workloads

Shape: long tool-call histories (growing with every turn), relatively short per-turn additions.

Chunked prefill makes the long-history re-prefill (when the prefix cache doesn't fully cover it) manageable without pushing TPOT off-SLO.

## When chunked prefill hurts (or is pointless)

Four shapes where enabling it is at best a wash:

### 1. Short-prompt workloads

Shape: all prompts fit inside a single `max-num-batched-tokens` chunk.

Chunking adds scheduler bookkeeping and (depending on the attention backend) a slightly-less-fused forward kernel path. If prompts are always short, chunked prefill is pure overhead with no benefit. Disable, or leave on and keep the chunk large enough that short prompts always complete in one iteration (which is operationally equivalent).

### 2. Pure-throughput offline batch

Shape: no streaming consumer, no per-request latency SLO, just "process this corpus."

Prefill-first scheduling (or large un-chunked prefills) is better. Chunking's whole purpose is to protect TPOT; if nobody's watching TPOT, don't pay for it.

### 3. Prefill-dominant workloads with low concurrency

Shape: one or two prompts in flight at a time, prompts are long, output short.

Chunking spreads the prefill across iterations with little or no decode piggybacking (there's nothing to piggyback on). You pay a modest overhead with no latency hiding.

### 4. Decode-saturated regimes

Shape: batch is already at `max-num-seqs` with ongoing decoders filling the per-step budget.

There is no leftover budget to chunk a prefill into. New prompts get admitted slowly regardless; chunked prefill becomes irrelevant until decoders retire. In this regime the real knob is `max-num-batched-tokens` or `max-num-seqs`, not the chunk size.

## Picking a chunk size

The chunk size is the knob where operator judgement earns its keep. Think of it as "how many prefill tokens per iteration" when there is leftover budget.

### Rule-of-thumb starting values

- **1024** — a reasonable neutral default. Fits well inside most `max-num-batched-tokens` configs; keeps iteration time predictable; covers P50 prompts in one or two chunks.
- **2048** — if your `max-num-batched-tokens` is 8192+ and TPOT SLO has margin; faster TTFT on long prompts.
- **512** — tight iteration-time budget (e.g. hardware or workload demands sub-20 ms steps); more chunks per prefill but ultra-stable TPOT.
- **≥ 4096** — approaches "un-chunked prefill" semantics. Only useful if `max-num-batched-tokens` is very large and TTFT dominates the SLO story.

### Interaction with `max-num-batched-tokens`

The chunk size is bounded by (budget − decode tokens for current batch). The two knobs co-tune:

- Doubling `max-num-batched-tokens` without changing chunk size → more decode throughput, same TTFT on long prompts.
- Doubling the chunk size without changing `max-num-batched-tokens` → faster TTFT on long prompts, more TPOT jitter (each iteration now has more prefill work).
- Doubling both proportionally → faster TTFT on long prompts, more throughput, more TPOT jitter at the tail.

### Interaction with prefix caching

Prefix caching (module 104) can mean a request's "prefill" is actually only the uncached suffix. If the system prompt is 2000 tokens and cached, a 2500-token prompt is a 500-token prefill — which fits in one chunk regardless. A tight chunk size still won't hurt here; a huge chunk size will not help, because the uncached suffix is already small.

### The right method: sweep

The chunk-size setpoint is empirical, not derivable from first principles. Author a sweep (exercise 03):

- Fix the workload (same reproducible trace used in chapter 02's sweep).
- Fix `max-num-batched-tokens` at the chapter-02 chosen value.
- Vary chunk size over {256, 512, 1024, 2048, 4096}.
- Plot TTFT P95 and TPOT P95 vs. chunk size.
- Pick the point with lowest TTFT P95 subject to TPOT P95 ≤ SLO.

The curve is usually monotonic within the useful range, with a knee somewhere between 512 and 2048 for 70B-class models on 8× H100/H200. Verify empirically.

## Diagnosing a chunked-prefill misconfiguration

Three patterns and the triage move.

### Pattern 1 — TPOT variance is bad, chunked prefill off

Symptom: TPOT P95 regularly exceeds TPOT P50 by 3–10×. Prefill rate correlates with the spikes.

Triage: enable chunked prefill with the default chunk size. Verify TPOT P95 drops. If it does, keep it on; if it doesn't, the regime may be decode-saturated or TPOT jitter has a different cause (worker straggler; module 102 chapter 04).

### Pattern 2 — TTFT bad on long prompts, chunked prefill on with tiny chunk size

Symptom: a 4000-token prompt takes seconds to produce its first token even at low concurrency. TPOT is beautiful.

Triage: raise chunk size (double it; measure). You've over-weighted TPOT stability at the cost of TTFT. In the extreme, with chunk = 128, a 4000-token prompt is ~31 chunks; even on an underloaded engine the TTFT is dominated by iterating 31 times.

### Pattern 3 — TPOT jitter *returned* after enabling chunked prefill with a large chunk size

Symptom: enabled chunked prefill, raised chunk size for TTFT, now TPOT variance is worse than before.

Triage: your chunk size is big enough that an iteration-with-chunk is essentially a prefill-dominated iteration. Lower chunk size, or hold `max-num-batched-tokens` constant and raise chunk size more conservatively.

## Boundary notes

- **The attention kernel supporting chunked prefill** is a performance-peer concern. Flash Attention 2/3, FlashInfer, and the engine-specific "piggyback attention" kernels have evolved specifically to make the mixed prefill-chunk + decode-token forward pass fast. If chunked prefill is slow or incorrect in a given backend, open an issue to the kernel owner; do not patch the kernel yourself.
- **Prefill-decode disaggregation** (DistServe, Splitwise, NVIDIA Dynamo; module 107) is a different mitigation for the same problem — prefill and decode run on different workers, with KV transferred between them. Chunked prefill and disaggregation can coexist (the prefill worker still chunks internally) but at different scales. Pick the one that matches your fleet shape.
- **Prefix caching** composes with chunked prefill (module 104). The two together are the production-defaults for prefix-heavy workloads.

## Summary

- **Chunked prefill (SARATHI-shape)** splits long prefills into chunks sized to fit alongside decode tokens inside the per-step token budget. The result is stable TPOT without sacrificing TTFT by much.
- Modern engines (vLLM, SGLang, TRT-LLM) all support some form of it; the flag name and default rotate. Verify your version.
- **Enabling it is right by default** for long-prompt workloads with concurrent decode and a TPOT SLO. **Not right** for short-prompt workloads, offline batch, prefill-dominant single-stream, or decode-saturated regimes.
- **Chunk size** is the empirical knob. Start at 1024 for 70B-class on H100/H200; sweep {256, 512, 1024, 2048, 4096} against your reproducible trace; pick the lowest TTFT P95 point that still meets TPOT P95 SLO.
- Chunked prefill co-tunes with `max-num-batched-tokens` and composes with prefix caching.
- Diagnose misconfiguration from the TTFT ↔ TPOT trade-off direction: too-small chunk → TTFT suffers; too-large chunk → TPOT jitter returns.

Chapter 04 goes to the other half of the pressure-relief valve set — **preemption**: what happens when admission is overcommitted and the KV pool runs dry.
