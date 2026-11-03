# Exercise 02 — Scheduler Tuning for a Stated SLO and Traffic Shape

**Time:** 3 hours · **Output:** one workload / SLO spec, one reproducible load generator, one Pareto sweep (TTFT & TPOT vs. throughput), one defended production config, one regression test

## Goal

Make [Chapter 02](../02-scheduler-configuration-for-slo.md) concrete by taking a **stated** workload spec and SLO, running a Pareto sweep over the two load-bearing scheduler knobs (`max-num-seqs` × `max-num-batched-tokens`), and producing a *defended* scheduler config — version-controlled, SLO-labelled, regression-tested — that an on-call engineer could ship.

## Problem statement

You are the owner of a new serving pool going into production in two weeks. The product team has given you a workload spec and an SLO target. You will:

1. Translate the workload spec and SLO into a first-pass scheduler configuration.
2. Build a reproducible load generator that replays the stated traffic shape.
3. Run a Pareto sweep over the two load-bearing knobs.
4. Pick the Pareto-frontier point with ≥20% SLO headroom and the lower preemption rate.
5. Pin the config in version control with reasoning in comments.
6. Author a regression test that re-runs the sweep's chosen setpoint nightly.

Use the chapter 02 worked example as scaffolding, or substitute your own workload spec if you have one — just make it concrete and written down before you start tuning.

## Prerequisites

- A GPU large enough for the model you tune against. The chapter uses a 70B dense on 8× H200; you can substitute a 7B / 13B model on a smaller GPU and the method stays correct. Document your substitution.
- One of the engines from module 102 (vLLM recommended for simplicity).
- The inner-loop cheatsheet from [exercise 01](exercise-01-continuous-batching-inner-loop-walkthrough.md) handy.
- Access to the engine's `/metrics` endpoint and a lightweight scraper (`curl` loop, or a tiny Prometheus / Grafana — exercise 05 wires this up more thoroughly; for this exercise, raw metric scraping is enough).

## Requirements

Deliver six artefacts.

### Artefact A — Workload / SLO spec (one page, markdown)

Write down the specification *before* touching the engine:

```markdown
## Workload: <pool name>

- **Model:** <model id, dtype, parallelism>
- **Mean RPS:** <N>
- **Peak RPS (P95 over 1 min):** <N>
- **Prompt length distribution:** P50=<N>, P95=<N>, P99=<N> tokens
- **Output length distribution:** P50=<N>, P95=<N>, P99=<N> tokens
- **Workload mix:** <chat / RAG / agent / batch — ratios if mixed>
- **Hit-max fraction:** <% of requests that reach max output length>

## SLO

- **TTFT P95:** <N> ms
- **TPOT P95:** <N> ms
- **E2E P95 (optional):** <N> ms
- **Admission policy:** 429 after queue depth > <N> | queue indefinitely | shed to backup
- **Cost ceiling (optional):** <USD per 1M input tokens> / <USD per 1M output tokens>
```

If you don't have real workload data: use the chapter's `chat-prod-us` worked example verbatim. Otherwise substitute your real numbers.

### Artefact B — First-pass config (one page)

Apply chapter 02's arithmetic to pick starting values for `max-num-seqs`, `max-num-batched-tokens`, scheduler-delay, chunked-prefill on/off, chunk size (if applicable), swap-space, and `--gpu-memory-utilization`. Write out the reasoning in prose — this document is what a reviewer reads before approving your sweep.

Example (abbreviated):

```markdown
- max-num-seqs = 128. Reasoning: dense 70B on 8xH200 TP=8 decode knee is in the 64–256 range; starting mid-range.
- max-num-batched-tokens = 8192. Reasoning: P95 prompt is 4000, chunk size 1024 makes each iteration ~1024 prefill + up to 128 decode tokens = well below budget.
- chunked prefill = on; chunk size = 1024.
- scheduler-delay = off (not bursty enough to justify).
- swap-space = 16 GB.
- gpu-memory-utilization = 0.92.
```

### Artefact C — Reproducible load generator (code)

A Python script (or equivalent) that:

- **Samples prompt lengths from your P50/P95/P99 distribution.** Use at least three distinct prompt-length buckets; do not use a single fixed length.
- **Samples output lengths from your distribution.** Pass `max_tokens` per request accordingly.
- **Simulates the arrival process** — Poisson at mean RPS for the baseline run; add a burst mode of 3× RPS for 1 minute for the stress run.
- **Streams responses (SSE)** and records first-token timestamp (TTFT) and per-token inter-arrival times (ITL, used to compute TPOT) per request.
- **Writes a trace file** — one row per request with (arrival timestamp, prompt tokens, output tokens requested, TTFT, TPOT, E2E, success/failure).

Extensions of your engine's built-in `benchmark_serving.py` are an acceptable base. The script must be re-runnable and deterministic given a fixed seed.

### Artefact D — Pareto sweep (table + two plots)

Run the load generator against the engine at each of the following 16 configurations (or a diagonal 4-point sweep if 16 is cost-prohibitive; document the compromise):

| `max-num-seqs` | `max-num-batched-tokens` ∈ {4096, 8192, 16384, 32768} |
|---|---|
| 64 | sweep |
| 128 | sweep |
| 192 | sweep |
| 256 | sweep |

For each run:

- 5-minute warmup (discarded), 10-minute measured window.
- Record: TTFT P50/P95/P99, TPOT P50/P95/P99, E2E P95, throughput (tokens/sec, requests/sec), preemption rate, GPU cache usage mean/max, queue time P95.

Produce:

- **Table** of all 16 (or 4) configs × all axes.
- **Plot 1: TTFT P95 vs. throughput.** Points for each config. Overlay the TTFT SLO line.
- **Plot 2: TPOT P95 vs. throughput.** Points for each config. Overlay the TPOT SLO line.

### Artefact E — Pinned production config (version-controlled file)

A YAML / JSON / env-var file that captures the chosen Pareto point, with reasoning comments:

```yaml
# serving-config/<pool>.yaml
# Generated by exercise-02 sweep on <date>; sweep trace at <path or URL>.
model: <id>
tp: <N>
max_model_len: <N>
max_num_seqs: <N>
max_num_batched_tokens: <N>
enable_prefix_caching: true
enable_chunked_prefill: true
# Reasoning:
# - max-num-seqs=<N>: Pareto knee for <model> on <hw>
# - max-num-batched-tokens=<N>: TTFT P95 target <N>ms met with <N>% headroom
# - chunked prefill on; chunk size tuned in exercise-03
# - preemption rate at chosen setpoint: <N>%
```

### Artefact F — Regression test

A nightly / weekly job (shell script, Makefile target, CI-callable) that:

1. Launches the engine with the pinned config.
2. Runs the load generator at mean RPS for 10 minutes.
3. Compares TTFT P95, TPOT P95, throughput, and preemption rate against a stored baseline from the Pareto sweep.
4. Fails (exit code 1) if any axis drifts beyond a threshold (e.g. ±10% for latency axes, ±20% for throughput).
5. Writes a dated trace file for later root-cause investigation.

The regression test is the output that outlasts the one-time sweep.

## Starter guidance

- **Spec first; tune second.** If Artefact A is vague, the sweep is wasted effort. Write it down.
- **Use a small model to validate the sweep plumbing** before you spend GPU-hours on the real model. The *shape* of the Pareto curve generalises; the setpoint doesn't.
- **The load generator must be reproducible.** Fixed random seed. Fixed distribution parameters. Record the git SHA of the generator alongside each sweep run.
- **Discard the first minute of every run.** Engine warmup (CUDA graph capture, cache fill) dominates. The chapter's rule-of-thumb is a 5-minute warmup; 1 minute is minimum.
- **Capture metrics continuously, not just at the end.** A clean final P95 can hide a 2-minute mid-run spike. Scrape `/metrics` every 5–10 s during the sweep.
- **Watch preemption and KV cache usage alongside TTFT/TPOT.** Chapter 04's dashboard patterns tell you whether a config's SLO-pass is luck or structural. A config at the SLO line with 95% KV usage is one burst away from falling off.
- **For the plot, use throughput on the x-axis and latency on the y-axis.** Lower-right of the plot is "fast and high-throughput." The Pareto frontier is the outer curve.
- **Compromise honestly.** If you ran a 4-point diagonal sweep instead of a 16-point grid because GPU-hours were tight, say so in Artefact D and name the specific risk.

## Fallbacks for constrained hardware

- **Smaller model.** Substitute 7B or 13B instruct for the chapter's 70B. The Pareto shape is similar; the setpoint will shift.
- **Shorter sweep.** 4 configs on the diagonal (e.g. `(64, 4096), (128, 8192), (192, 16384), (256, 32768)`) is enough to see the Pareto shape. Note the limitation.
- **Shorter runs.** 5-minute measured windows instead of 10. Say so.
- **Cloud spot.** A single H100 or A100 for a 7B / 13B is enough.

## Acceptance criteria

1. **Artefact A** is a filled-out spec with concrete numbers (no "approximate" or "typical" — specific).
2. **Artefact B** reasons from Artefact A to the first-pass config in prose; a reviewer could approve or push back on specific choices.
3. **Artefact C**'s load generator is reproducible — same seed, same output. The trace file is machine-parseable.
4. **Artefact D**'s Pareto plot clearly shows at least four distinct configurations on the plot, with the SLO lines overlaid. The chosen point is identified and justified.
5. **Artefact E**'s pinned config carries reasoning comments. A reviewer unfamiliar with the sweep can read the file and understand *why* each value was chosen.
6. **Artefact F**'s regression test runs end-to-end and fails cleanly if the baseline drifts.

## Common failure modes

- **No-spec tuning.** Starting the sweep without a written spec means you can't tell "pass" from "fail." Write Artefact A first.
- **Fixed-length prompts.** A benchmark with a single prompt length lies about tail latency. The chapter is explicit that distributions must be heterogeneous.
- **No warmup.** The first few minutes of any new config are dominated by CUDA graph capture / prefix cache fill. Discard.
- **No preemption observation.** If every sweep run has zero preemptions, you're not pushing the engine. Try the P95 burst mode.
- **Over-fitting the sweep.** 20% SLO headroom is a floor, not a ceiling. Picking the point with zero margin leaves no room for traffic growth.
- **Pinning without comments.** A config that is "the one that worked" with no reasoning is one engine upgrade away from needing to be re-derived from scratch.

## Keep it

The pinned config (Artefact E) and the regression test (Artefact F) are production artefacts. Pin them to the serving-pool's repo. Exercises 03–05 will tune on top of this baseline, and the final module-103 lab produces the full scheduler-config bundle you ship.
