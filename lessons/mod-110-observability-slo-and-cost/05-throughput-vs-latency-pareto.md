# Chapter 05 — The Throughput-vs-Latency Pareto Curve: Sweeping, Reading, Defending

> Every inference fleet operates somewhere on a trade-off curve: more batching gets more tokens per GPU-hour but makes TTFT and TPOT worse; smaller batching gets cleaner latency but burns GPU. The Pareto curve names the frontier where you cannot improve one without hurting the other, and the operating point is where you choose to sit. A product manager who asks "can we double throughput" is asking you to move along the curve; the right answer is a chart, not a vibe. This chapter builds the chart and the argument.

## Motivation

Three variants of the same conversation:

- *Finance:* "H100 bills are 15 % over plan. Pack more sequences per GPU." You lower `max-num-seqs` and TTFT doubles. Nobody told you TTFT was load-bearing.
- *Product:* "A competitor shows first-token at 150 ms; ours is 500. Fix it." You raise batch limits and lower `max-num-seqs` and cost rises 40 %. Nobody told you cost was load-bearing.
- *Platform:* "The new model is slow. Make it faster." You enable speculative decoding; TPOT improves 2×, but TTFT regresses on short prompts because the drafter costs more than it accepts on short contexts. Nobody told you both regimes mattered.

Each decision was made without the curve. The Pareto curve lets you say: *moving `max-num-seqs` from 128 to 256 raises goodput by 18 % and raises TTFT_P95 from 420 ms to 580 ms — if the TTFT SLO is 500 ms, we lose 10 % of our error budget to pick up 18 % of throughput; here is the math and here is my recommendation*.

By the end of this chapter you can:

- Design the sweep (what to vary, what to hold constant, what traffic shape to use).
- Measure the sweep at operator depth (what metrics to collect, how to avoid common measurement bugs).
- Read the curve (knee, ceiling, cliff — three archetypes).
- Compute **goodput** (throughput at or below the SLO; the number that matters).
- Defend the operating point in a one-page memo your product partner reads.

Chapter 04 produces the cost per operating point; chapter 06 translates the trade-off into business terms; this chapter produces the curve those two depend on.

## The variables and the invariants

A Pareto sweep is a *controlled experiment*. You pick one or two variables to sweep, hold everything else constant, and record how latency and throughput move together.

**Primary sweep variables** (common first choice):

- `max-num-seqs` — the maximum concurrent sequences in the batch (vLLM term; SGLang / TRT-LLM equivalents vary).
- `max-num-batched-tokens` — the maximum tokens per iteration (controls prefill chunk size and the mixed prefill-decode budget).
- Speculative length `k` — if speculative decoding is enabled.
- Replica count (fleet-level sweep).

**Secondary sweep variables** (consider for a second study):

- Prefix-cache enabled / disabled.
- Chunked prefill enabled / disabled.
- Attention backend (FlashAttention vs. FlashInfer vs. engine default).
- Quantisation (fp16 vs. fp8 vs. w8a16).

**Invariants to hold** (critical):

- Traffic shape (prompt-length distribution, output-length distribution, QPS). Use a *replay* of production traffic or a deterministic generator; a random `--input-len X --output-len Y` benchmark is only one point.
- Model, weights, hardware, parallelism (TP / PP / EP).
- Engine version. Even patch versions change scheduler behaviour.
- Warm-up. Every point must include a warm-up phase; chapter 02 of module 102 names the protocol.

## The traffic shape — not optional

The curve shifts materially with prompt-length distribution. A sweep with `--input-len 1024 --output-len 128` is a sweep in one corner of the possibility space; the production curve may have a different knee.

Three traffic-shape options, worst to best:

- **Fixed prompt / output.** `--input-len 1024 --output-len 128`. Easiest; wrong. The resulting curve answers one question (what if all traffic were this shape); production is a mix.
- **Mixed prompt / output from a distribution.** `--dataset sharegpt` (ShareGPT replay) or similar. Better. See the vLLM `benchmark_serving.py` dataset options.
- **Production-replay.** A sampled log of real prompts and real output-length distributions. Best. Requires a load-replay harness (chapter 03 of module 103 or your fleet's existing load generator).

The best practice: produce the curve against production-replay and name the traffic-shape version at the top of the resulting chart.

<!-- needs-research: the exact dataset flags in `vllm/benchmarks/benchmark_serving.py` rotate across releases; the ShareGPT dataset's licence terms are also something to verify before mirroring it. Verify against the pinned engine's benchmark docs. -->

## Metric collection at each point

Each sweep point produces a row of a results table. The row carries:

| Column | Source |
|---|---|
| Config (knob = value) | the sweep input |
| QPS offered | load generator |
| QPS admitted | engine admission counter (chapter 01) |
| TTFT_P50 / P95 / P99 | engine histogram |
| TPOT_P50 / P95 / P99 | engine histogram |
| ITL_P99 | engine histogram |
| Preemption rate | engine counter |
| Admission reject rate | engine counter |
| Cache-hit rate | engine counter |
| Decode tokens / sec | engine counter |
| Prefill tokens / sec | engine counter |
| GPU utilisation (SM_ACTIVE avg) | DCGM |
| HBM used | DCGM |
| NCCL time ratio (if TP > 1) | engine or Nsight |
| Cost per 1M output tokens | derived from chapter 04 |

The sweep script captures a snapshot of every one of these at each point; the chart plots two of them (throughput and latency); the one-page memo cites the rest.

Three measurement disciplines:

- **Steady-state measurement window.** The first 60 – 120 seconds of a load phase are cold / warming; discard. Measure the next 300 – 600 seconds.
- **Separate ramp from measurement.** Don't ramp QPS during measurement. For each sweep point, ramp to steady state and then start the measurement timer.
- **Record the admitted QPS, not the offered QPS.** At saturation the engine rejects admission; the offered QPS is not what the engine actually served.

## The sweep protocol — a worked recipe

A clean sweep for one knob (`max-num-seqs`) at one model (Llama-3.1-70B TP=4) at one traffic shape (chat replay):

```
for max_num_seqs in [8, 16, 32, 64, 128, 192, 256, 384]:
    deploy_replica(max_num_seqs=max_num_seqs)
    warm_up(duration=120s)
    for offered_qps in [10, 20, 40, 60, 80, 100]:
        apply_load(qps=offered_qps, duration=600s)
        wait(60s)                          # steady state
        snapshot = collect_metrics(window=300s)
        record(point=(max_num_seqs, offered_qps), snapshot=snapshot)
    teardown()
```

Produces `8 × 6 = 48` points. Each point takes ~15 minutes end-to-end counting warm-up and settle; the full sweep is 12 hours. Run overnight.

For a two-knob sweep (`max-num-seqs` × `k`), the grid is larger. A reasonable strategy: fix one knob at the baseline, sweep the other; then at the chosen operating point, sweep the second knob. This is faster than a full grid and usually catches the knee for both.

## Reading the curve — three archetypes

Plot `admitted_qps` or `output_tokens_per_second` (fleet-level) on the x-axis and `TTFT_P95` or `TPOT_P95` on the y-axis. Three shapes recur:

### Archetype 1 — the knee

```
  TTFT_P95
   ^
   |                 ___ 
   |              _/
   |          __/
   |        _/
   |      _/
   |__--'
   +---------------------> QPS
          ^
          knee
```

Latency is flat up to a point, then rises sharply. The knee is the capacity of the current batch size.

Operating point: **just below the knee**. Push past it and TTFT runs away; retreat too far and you waste GPU-hours.

### Archetype 2 — the ceiling

```
  TTFT_P95
   ^
   |              ------- (ceiling: admission reject starts)
   |          ___/
   |      ___/
   |__--'
   +---------------------> QPS
```

Latency rises smoothly, then flat-lines at a ceiling because admission rejects. The ceiling is `max-num-seqs × decode_tokens_per_sec / mean_output_length` — the engine's admission ceiling, not an SLO.

Operating point: **below the ceiling**, with admission reject rate well under 1 %.

### Archetype 3 — the cliff

```
  TTFT_P95
   ^
   |              |
   |              |
   |              | (preemption rate spikes)
   |          ___/
   |      ___/
   |__--'
   +---------------------> QPS
```

Latency rises smoothly, then cliff-jumps because the scheduler starts preempting. This is the KV allocator (module 104) running out of room; the fleet is thrashing.

Operating point: **well below the cliff**. If the cliff is close to the business ask, lower `max-num-seqs` or raise HBM budget (which is a different conversation entirely).

The three archetypes are not mutually exclusive — a real fleet often has all three on different sweep lines.

## Goodput — not throughput

The number that matters for an SLO conversation is **goodput**: throughput of requests served *at or below the latency SLO*. A fleet serving 100 QPS at 2 s TTFT is doing worse than a fleet serving 60 QPS at 400 ms TTFT if the SLO is 500 ms. Only the fast fleet produces usable requests.

```
goodput_qps = QPS × fraction_of_requests_meeting_SLO
```

Plot goodput, not throughput, as the x-axis for an SLO conversation. The chart that goes to product and finance has goodput on the x-axis and cost per 1M tokens on the y-axis (chapter 04 feeds this).

Goodput can *decrease* with higher offered QPS: as the fleet saturates, more requests exceed the SLO, so even if raw throughput rises, goodput falls. A goodput-vs-QPS curve has its own knee; the operating-point conversation uses it directly.

## The one-page defense

Your product partner shows up with the question: *"can we double throughput?"* The one-page memo:

```
# Throughput proposal: double chat capacity

## Current operating point
- max-num-seqs = 128, max-num-batched-tokens = 8192
- goodput: 65 QPS (SLO: TTFT_P95 < 500 ms, TPOT_P95 < 50 ms)
- cost: $0.34 per 1M output tokens
- Pareto curve: see panel A

## Proposal interpretation
"Double throughput" could mean one of:
  (A) Double raw QPS at any latency.
  (B) Double goodput (requests meeting the SLO).
  (C) Double revenue-token throughput (chapter 06).

Default assumption: (B). Confirmed with [partner] on [date].

## Options

### Option 1 — raise max-num-seqs to 256
- goodput: 92 QPS (+40%)
- cost: $0.26 per 1M output tokens (-24%)
- TTFT_P95: 580 ms (SLO breach; -10% error budget per week sustained)
- preemption rate: 2% (3× current; marginal thrashing risk)
- recommendation: NOT RECOMMENDED without an SLO relaxation.

### Option 2 — add 40% more replicas, keep max-num-seqs = 128
- goodput: 91 QPS (+40%)
- cost: $0.34 per 1M output tokens (unchanged)
- TTFT_P95: 420 ms (within SLO)
- marginal GPU budget: $220k / month
- recommendation: RECOMMENDED if the $220k is approved.

### Option 3 — enable speculative decoding (chat class)
- goodput: 110 QPS (+69%)
- cost: $0.13 per 1M output tokens (-62%)
- TTFT_P95: 430 ms (within SLO)
- new lifecycle burden: (target, draft) pair, acceptance monitoring (module 108)
- recommendation: STRONGLY RECOMMENDED; implementation 3 – 4 weeks.

## Pareto chart
See panel A: goodput_qps on x, cost_per_1M_output_tokens on y, SLO breach region shaded.

## Risks
- Option 3 requires module 108 chapter 03 monitoring to go live first.
- Option 2 requires module 109 fleet-size review and SKU availability check.
- Option 1 requires SLO re-negotiation.

## Recommendation
Option 3 as primary; Option 2 as fallback if the lifecycle burden is unacceptable in Q[X].

## Appendix — the sweep
Full sweep data: {link to results.csv}
Methodology: {link to this chapter}
```

That is the shape of the memo. It is a page, it names the trade-offs, it quantifies them, it recommends one.

## Three Pareto conversations by role

Same chart, different reading:

- **Engineer:** "The current operating point is 20 % below the knee; we have margin to raise `max-num-seqs` by 50 % if the SLO moves."
- **Product manager:** "The chart says a $0.21 / 1M savings is possible if we accept a 15 % latency regression on long-context — do we want that trade?"
- **Finance partner:** "The Option 2 cost is $220k / month incremental; the Option 3 cost is $0 if the implementation works. The expected value of Option 3 is positive even at a 50 % failure probability."

All three readings are on one chart. Different y-axes for different audiences — latency for engineering, goodput for product, cost for finance — but the underlying sweep is one artifact.

## Common failure modes

- **Fixed-input-length benchmark.** The ShareGPT replay or production replay captures real-mix behaviour; a single length does not.
- **Measuring during warm-up.** The first 60 seconds of each sweep point are polluted by warm-up and CUDA-graph re-trace. Discard.
- **Not isolating the sweep variable.** If `max-num-seqs` and `max-num-batched-tokens` are both changing, the curve is uninterpretable.
- **Confusing offered QPS with admitted QPS.** At saturation, admission rejects; the plotted number is `admitted`.
- **Reporting mean, not percentile.** TPOT_mean hides tail stalls; always report P95 / P99.
- **No warm cache.** The prefix cache (module 104) warms over time; a cold-cache sweep underestimates throughput for repeat-prefix workloads.
- **Running on a shared GPU.** MIG / MPS sharing with another workload gives contaminated numbers. Dedicated hardware for the sweep.
- **Not including goodput.** Raw throughput can go up while goodput goes down. The chart that matters for an SLO conversation always has goodput.
- **Dropping the cost line.** Chapter 04's cost per 1M tokens is the ninth-column number; without it, the memo can't close.
- **Over-fitting to one benchmark run.** Run the sweep twice on different days (traffic / GPU state drift); report the median across runs.

## Summary

- A Pareto sweep is a controlled experiment. Pick one variable, hold the rest constant, use a realistic traffic shape.
- Measure at steady state, after warm-up, with admitted-QPS not offered-QPS.
- Collect the full axis set per point: latency, throughput, cache / preempt / admission, DCGM, cost from chapter 04.
- Read the shape: knee, ceiling, cliff. Each has a different operational meaning.
- Report **goodput** (throughput at or below the SLO), not raw throughput, when talking to product.
- Defend the operating point with a one-page memo: current point, options with quantified trade-offs, recommendation.
- The chart is the artifact. Different roles read different y-axes; the sweep underneath is one.
- Chapter 06 translates the chart into a business-value memo; this chapter produces the chart.
