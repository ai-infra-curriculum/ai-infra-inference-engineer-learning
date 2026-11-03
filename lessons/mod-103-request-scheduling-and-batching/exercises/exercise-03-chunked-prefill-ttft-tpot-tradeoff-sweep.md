# Exercise 03 — Chunked Prefill TTFT / TPOT Trade-off Sweep

**Time:** 3 hours · **Output:** one chunk-size sweep (TTFT & TPOT vs. chunk size), one Pareto plot overlaid with your SLO, one pinned chunk-size value with reasoning, one risk register

## Goal

Make [Chapter 03](../03-chunked-prefill-ttft-tpot-tradeoff.md) concrete by sweeping chunk size and `max-num-batched-tokens` against a reproducible long-prompt + decode-heavy workload, plotting the TTFT / TPOT trade-off, picking a production chunk size, and naming the specific risks that come with the pick.

## Problem statement

You already have a scheduler config from [exercise 02](exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape.md). The decision whether chunked prefill is on, and at what chunk size, is where TTFT and TPOT trade most directly. This exercise finds the Pareto point for your workload.

Your deliverable is a defended value for the chunk-size knob (or an explicit decision to leave chunked prefill *off*, with evidence) plus the risk notes that go with it.

## Prerequisites

- The engine and load generator from [exercise 02](exercise-02-scheduler-tuning-for-stated-slo-and-traffic-shape.md).
- A workload shape with **at least some long prompts** — if all your P95 prompts are ≤ 1k tokens, chunked prefill is operationally irrelevant for your pool; synthesise a long-prompt variant for the sweep (chapter 03 explains why).
- Access to the engine's chunked-prefill knob and chunk-size equivalent. **Verify the flag spelling for your engine version** — vLLM uses `--enable-chunked-prefill` and governs chunk size via `--max-num-batched-tokens` (and in some releases `--long-prefill-token-threshold` / `--max-num-partial-prefills`); SGLang exposes a `--chunked-prefill-size`-shape knob; TRT-LLM via Triton uses `enable_chunked_context` in the backend config. Verify current spellings against your engine's docs.

## Requirements

Deliver four artefacts.

### Artefact A — Workload variant with long prompts

Starting from the exercise-02 load generator, create a workload variant that:

- Mixes **short prompts (P50 ~500 tokens)** and **long prompts (P95 ~4000–8000 tokens)** in a stated ratio (e.g. 80/20 short/long, or whatever matches your production shape).
- Keeps concurrent decode traffic — the long prompts' impact is only visible against ongoing decoders.
- Streams responses so TTFT and TPOT can be observed per-request-length-bucket.

Document the ratio and the per-bucket length distribution. Your sweep interprets differently if the long-prompt ratio changes.

### Artefact B — Sweep (table)

Run the engine against the Artefact A workload at:

- **chunked prefill off** (baseline).
- **chunked prefill on, chunk size ∈ {256, 512, 1024, 2048, 4096}** (five values).

For each run:

- Hold `max-num-seqs` and `max-num-batched-tokens` at the exercise-02 chosen values.
- 5-minute warmup, 10-minute measured window.
- Record, **split by prompt-length bucket** (short vs. long):
  - TTFT P50, P95, P99.
  - TPOT P50, P95, P99.
  - Preemption rate (overall).
  - Iteration-time distribution (if available).

A single aggregate TTFT P95 across the whole workload hides the signal. Split by bucket.

Secondary sweep (optional but recommended): hold chunked prefill on at your preferred chunk size and vary `max-num-batched-tokens` ∈ {4096, 8192, 16384, 32768} to confirm the co-tuning behaviour from chapter 03.

### Artefact C — Pareto plot and pick

Produce:

- **Plot 1: long-prompt TTFT P95 vs. chunk size.** Overlay the TTFT SLO line.
- **Plot 2: TPOT P95 (across all prompts, since the mixed batch is where TPOT variance shows) vs. chunk size.** Overlay the TPOT SLO line.
- Optionally: **Plot 3: short-prompt TTFT P95 vs. chunk size** (should be flat — short prompts fit in one chunk regardless; a non-flat line is a signal something else is going on).

Then **pick a production chunk size** by applying chapter 03's rule:

> Lowest TTFT P95 (long-prompt bucket) subject to TPOT P95 ≤ SLO.

Document the pick in one paragraph of prose. Compare it to the chunked-prefill-off baseline — if the baseline hits SLO comfortably, chunked prefill may not be worth enabling.

### Artefact D — Risk register

Author a short risk register for the pinned chunk size. Minimum entries:

1. **Workload shift sensitivity.** If the long-prompt ratio grows (e.g. a new RAG workload pushes it from 20% to 60%), does the chunk size still hold? What is the next-pick fallback?
2. **Hardware upgrade sensitivity.** If the pool moves from H100 to H200 (or Hopper to Blackwell), what changes? Chunk size is often robust across GPU generations but iteration times are not; the Pareto shifts.
3. **Engine upgrade sensitivity.** Note the engine version at the time of the sweep. If the engine upgrades across a release that changes chunked-prefill scheduling (verify the release notes), the sweep is stale.
4. **Short-prompt pool regression risk.** If your pool suddenly becomes all short-prompt (chunking becomes pure overhead), the knob should be re-visited.
5. **Interaction with preemption policy.** Chapter 04's note: a sequence preempted mid-prefill in recompute mode loses chunk progress. If your preemption rate is high, chunked prefill composes less well with recompute.

Each risk entry: one sentence on the symptom, one sentence on the detection (which metric), one sentence on the mitigation.

## Starter guidance

- **Make the long-prompt bucket explicit.** The whole point of chunked prefill is to protect decoders from long prompts; your sweep is uninteresting if you don't have long prompts in it.
- **Watch TPOT by prompt mix, not globally.** Pure-decode windows will have great TPOT regardless of chunk size — the signal is in windows that *include* a long prompt mid-prefill.
- **Compare on the same `max-num-batched-tokens` baseline.** Changing both simultaneously conflates variables.
- **Record raw iteration times if your engine exposes them.** Chapter 03's intuition is "iteration-time variance comes from prefill-dominated iterations"; proving it on your workload is the point.
- **The default might be the right answer.** Modern vLLM defaults chunked prefill on with a reasonable chunk size. Your sweep may confirm the default is best. That is a *valid* outcome, not a failure — document it.
- **If the engine does not expose a chunk-size knob directly** (vLLM governs chunk size through `max-num-batched-tokens`), structure the sweep as "`max-num-batched-tokens` ∈ {2048, 4096, 8192, 16384}" with chunked prefill on, holding `max-num-seqs` fixed. Note the engine-specific interpretation.
- **Watch for scheduler-delay interactions.** If your engine has a scheduler-delay knob and it's on, chunk-size sweep results can be noisier. Prefer scheduler-delay off during the sweep.

## Fallbacks for constrained hardware

- **Smaller model.** 7B / 13B still shows the TTFT / TPOT trade-off clearly, often more clearly than 70B because iteration-times are faster and sweep points are cheaper.
- **Shorter sweep.** 3 chunk-size points ({512, 1024, 2048}) + off is enough to see the shape; note the limitation.
- **CPU-only.** Only recommended for the chunked-prefill *logic* verification — the TTFT / TPOT numbers from CPU are not representative. Note that CPU runs validate the policy mechanism, not the production setpoint.

## Acceptance criteria

1. **Artefact A** has a documented long-prompt bucket (≥ 15% of workload, ≥ 4000 tokens P95).
2. **Artefact B**'s table has at least five sweep points, split by prompt-length bucket on TTFT, with preemption rate recorded.
3. **Artefact C**'s plots show the TTFT / TPOT trade-off clearly: TTFT P95 (long-prompt bucket) should fall as chunk size rises; TPOT P95 should rise as chunk size rises. If this signature does not appear, the writeup must explain why (short prompts only, decode-saturated, backend not supporting mixed batching).
4. **Artefact C**'s pick is defended against both SLO lines with margin, and compared to the chunked-prefill-off baseline.
5. **Artefact D**'s risk register has at least five entries, each with symptom / detection / mitigation.
6. The chosen setpoint is pinned alongside the exercise-02 config in version control.

## Common failure modes

- **No long prompts in the workload.** Chunked prefill is a long-prompt mitigation. If your workload is uniformly short, the sweep will be flat — note it and move on.
- **Mixing variables.** Changing chunk size *and* `max-num-batched-tokens` *and* `max-num-seqs` simultaneously is three experiments at once. Hold the other two constant.
- **Global TPOT P95 as the only axis.** Decoders during pure-decode windows are fine. The signal is in windows mixing prefill and decode.
- **Picking the extreme.** Picking chunk size 256 is pure TPOT protection; chunk size 4096 is pure TTFT protection. The right answer is almost always in the middle.
- **Not verifying the chunk-size knob is live.** If your engine version has the flag but no backend implementation, the sweep returns flat results — verify by watching batched-tokens-per-iteration distributions.

## Keep it

The pinned chunk size and the risk register are production artefacts. They should sit next to the exercise-02 config in your serving-pool's repo. The sweep script should be re-runnable on every major engine upgrade — chunked-prefill behaviour is one of the knobs that moves most across releases.
