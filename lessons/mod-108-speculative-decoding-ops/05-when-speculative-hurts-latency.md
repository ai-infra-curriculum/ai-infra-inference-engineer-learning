# Chapter 05 — When Speculative Decoding Hurts Latency: Regimes and Benchmarks

> Speculative decoding is throughput-and-latency positive in some regimes and strictly negative in others. The two regimes where it hurts are legible and bench-mark-able; the fix is almost always a per-class policy switch, not a deep debug. This chapter names the regimes, benches them, and writes the policy.

## Motivation

The team from chapter 01 enables speculative decoding fleet-wide. Three weeks later three on-call pages fire:

1. **Reasoning tenant on holiday.** A tenant doing structured chain-of-thought has `α ≈ 0.38`. Their TPOT at `k=4` reads `58 ms` — above their SLO. With speculative off, the same workload on the same fleet reads `44 ms`. Speculative is net negative for this traffic.

2. **Marketing campaign.** Burst of traffic drives per-replica batch size from 4 (typical) up to 32. Fleet TPOT climbs; `α` is unchanged. The team suspects the draft; the actual story is that at batch=32 the target's decode step is already compute-bound — the "almost free" verify step is no longer almost-free, so the draft's `c_d` has nothing to amortise against. Speculative's net contribution falls to near-zero or slightly negative.

3. **Prefill-heavy workload.** A new product delivers very long prompts (16k tokens) with short generations (50 tokens). Each request's prefill dominates the total latency. Speculative is on, acceptance is fine, but TTFT didn't improve because speculative doesn't touch prefill — and the handful of decode steps the request takes amortises the draft startup cost poorly.

The first two are the classic "speculative hurts" regimes: **low acceptance** and **compute-bound batch**. The third is not quite "hurts" but is "doesn't help enough to be worth the operational cost." All three are detectable with a specific benchmark, and the response is a per-class policy, not a code change.

By the end of this chapter you can:

- State the two regimes where speculative is a net loss and the derivation of the break-even conditions.
- Build a benchmark that catches both regimes against your fleet, with the specific metrics you need and the comparison method.
- Write the fleet's "disable speculative when ..." policy — the predicates that select which requests get speculative and which don't.
- Decide between the three remedies: lower `k`, disable for the class, autoscale differently.

Engine-side support for a per-request "disable speculative" override varies — some engines accept a per-request flag, others only accept a per-fleet config. Verify the engine's documented request-level override surface before building the policy.

## Regime 1 — Acceptance too low

### The condition

Chapter 01's ceiling formula:

```
speedup(α, k, c_d/c_t) ≈ (1 - α^(k+1)) / ((1 - α)(1 + k · c_d / c_t))
```

Setting `speedup < 1` and solving for `α`:

```
α < α_break_even(k, c_d/c_t)
```

For `c_d / c_t = 0.1`, approximate break-even thresholds (illustrative; derive with the formula for your constants):

| `k` | approximate break-even `α` |
|---|---|
| 2 | ~0.21 |
| 3 | ~0.27 |
| 4 | ~0.31 |
| 5 | ~0.34 |

For `c_d / c_t = 0.05`:

| `k` | approximate break-even `α` |
|---|---|
| 2 | ~0.14 |
| 3 | ~0.17 |
| 4 | ~0.20 |
| 5 | ~0.22 |

Interpretation: with a cheap draft (`c_d / c_t = 0.05`), speculative pays whenever `α > ~0.2`. With an expensive draft (`c_d / c_t = 0.2`), speculative needs `α > ~0.4` at low `k` to pay. The break-even is **not** `α = 0.5` — a common misconception. The real threshold is often lower, which is why many deployments see modest wins even on workloads that feel "hard."

But: in practice the ceiling is unreachable. Realised speedup is lower than the formula predicts because of batching dynamics, CUDA-graph re-trace, scheduler decisions, and warm-up overhead not in the formula. A conservative operator sets the "disable" threshold at `α < 1.3 × α_break_even_formula` to leave a safety margin against the un-modeled overhead.

### How it shows up

- Chapter 03's Alert 2 ("per-class acceptance rate collapse") fires.
- Panel 2 (per-class) shows the class in question with `α` below the break-even threshold.
- Panel 1 fleet-level TPOT regresses *when the class's share of traffic rises*.
- Panel 3 (per-tenant) shows the tenants dominated by the class with higher-than-SLO TPOT.

### The remedy

Three options in order of preference:

1. **Disable speculative for the class.** Requests tagged with this workload class bypass speculative at admission; they decode without the draft. TPOT returns to the non-speculative baseline (lower than the speculative-but-low-acceptance TPOT).
2. **Lower `k` for the class.** If the class's `α` is marginal (just above break-even), lower `k` improves the trade-off. Not applicable when `α` is well below break-even.
3. **Request a draft that covers the class.** Chapter 04 Trigger 2 or 4.

The first is the operational response — same day. The second is a short-term adjustment if the class is marginal. The third is a durable fix.

## Regime 2 — Batch already compute-bound

### The condition

Decode is memory-bandwidth bound at small batch sizes (chapter 103): one forward pass reads the target's weights once from HBM and does roughly `batch_size` units of compute per weight read. Up to some batch size `B*`, adding more sequences to the batch adds compute to the same HBM-read; the per-token cost drops.

Above `B*`, decode becomes compute-bound: the HBM-bandwidth is no longer the bottleneck; the GEMM FLOPs are. Adding more sequences no longer helps because the HBM traffic was never the limit at that batch size.

Where this matters for speculative: the speculative-vs-vanilla argument is "the verify pass costs roughly the same HBM traffic as a single-token decode pass, so we get `k`-many token-attempts for one HBM read." Once decode is compute-bound, the verify step is compute-bound too, and the "one HBM read" story weakens — the extra tokens in the verify step cost compute even if they don't cost extra HBM traffic. The draft's `c_d` is pure overhead against a target step that is now priced in FLOPs rather than bandwidth.

Numerically (illustrative — verify on your hardware):

- Llama-3.1-70B on 4× H100 SXM, decode-only: `B* ≈ 8–16` depending on sequence length and quantisation. Below `B*`, speculative helps. Above `B*`, the speculative speedup shrinks and can go negative above `B ≈ 48–64`.
- The exact `B*` depends on GPU HBM bandwidth / FLOPs ratio, quantisation, kernel efficiency, and sequence length. Hopper vs. Blackwell vs. MI300 differ. Measure.

The specific `B*` threshold for a given (target, hardware, quantisation) combination is sometimes published by the engine or hardware vendor and sometimes not; measure on your fleet — the benchmark exercise later in this chapter walks the measurement.

### How it shows up

- Chapter 03's Alert 5 ("draft overhead ratio climb") may fire, but often the signal is more diffuse — `α` is unchanged, TPOT climbs with batch size, and `spec_draft_overhead_ratio` creeps upward.
- The fleet's throughput-per-replica rises (more concurrency means more tokens per iteration from the compute-bound target), but TPOT climbs disproportionately.
- Autoscaling that keys off TPOT thrashes.

### The remedy

Three options:

1. **Lower max batch size for speculative-enabled workloads.** The scheduler is admitting too many sequences into one iteration; cap the per-iteration batch for the speculative path. Trade-off: lower throughput per replica.
2. **Disable speculative above a batch threshold.** The engine runs vanilla decode when batch > `B_disable`. Not every engine exposes this; verify.
3. **Autoscale sooner.** Add replicas earlier so the per-replica batch stays below `B*`. Trade-off: more HBM-bound hardware for the same throughput. Module 109 covers.

Option 1 is the operator's local lever. Option 2 is cleanest if supported. Option 3 is the fleet-level answer.

## Regime 3 — Decode-light requests (long prefill, short generation)

Not strictly "hurts" but worth naming because teams miss it:

### The condition

Request shape: prompt ≫ generation. TTFT is prefill-dominated; speculative doesn't touch prefill. The ~5–20 decode steps the request takes amortise the draft's startup cost (CUDA-graph warmup for the draft shape, scheduler accounting, draft-side KV allocation) poorly.

Example: 16k-token prompt, 50-token generation. TTFT dominates; speculative savings on 50 decode steps are small in absolute terms and worth comparing to the overhead for the request lifecycle.

### How it shows up

- Chapter 03 metrics look fine (`α` is whatever the drafter achieves on the generation).
- End-to-end latency does not fall meaningfully; the user-visible benefit is small.
- Operators sometimes misread this as "speculative doesn't work" when it is doing exactly what it is designed to do on exactly the token range it was designed for.

### The remedy

- **Accept the smaller benefit.** If TTFT is the SLO axis, decode-light requests get negligible benefit from speculative; the fleet still benefits via throughput and the longer-generation requests.
- **Disable speculative for requests with `max_tokens < N`.** Not universally supported by engines; some engines accept per-request flags. Rarely worth the complexity.

Regime 3 is a reminder that speculative is a decode-phase optimisation; its benefit scales with the number of decode steps per request. Chapter 05's benchmark exposes this directly.

## The benchmark

The benchmark produces, for each `(workload_class, batch_size)` pair, four numbers:

- Non-speculative TPOT P50 / P99.
- Speculative TPOT P50 / P99 at the current pair's `k`.
- The pair's `α` on the sample.
- Net speedup = `TPOT_non_speculative / TPOT_speculative`.

The deliverable is a two-dimensional heatmap: workload class on one axis, batch size on the other, cells coloured by net speedup. Cells where speedup < 1 are the "disable" cells.

### Methodology

- **Traffic source.** Representative canary traffic per class — same source chapter 04 uses for baselining. If your fleet classifies in production (chapter 03), use a traced slice.
- **Comparison method.** A/B on the same hardware, same engine version, same config, flipping only the `--enable-speculative` (or equivalent) flag. Alternate 1-minute windows; collect at least 15 minutes of each per class × batch size cell.
- **Batch control.** Harder than it sounds. Continuous batching makes "batch size" an output of the scheduler, not an input. Options:
  - Drive synthetic load at a specific QPS that produces the target batch size on average; measure the resulting batch distribution.
  - Use the engine's benchmark harness (vLLM / SGLang / TRT-LLM each ship a `benchmark_serving.py` or equivalent) which exposes a concurrency knob that correlates with batch.
  - Use the engine's `max-num-seqs` to cap the batch; sweep across values.
- **Statistical rigor.** Report confidence intervals; use enough samples. 1000+ requests per cell is reasonable for stable TPOT P99.
- **Warm-up.** Discard the first N minutes of each regime; let CUDA graphs and the scheduler stabilise.

### Reporting

Three artefacts:

1. **The heatmap.** Workload class × batch size, cells coloured by net speedup. Overlaid with `α` and `c_d / c_t` as secondary annotations.
2. **The "disable" policy.** The predicate set that captures the net-negative cells. Example:
   ```
   Disable speculative when:
     - workload_class in {reasoning_long, structured_json_strict}  (α < 0.35 observed)
     - OR batch_size > 48                                           (compute-bound regime)
     - OR max_tokens < 20                                           (decode-light regime, optional)
   ```
3. **The "monitor and re-run" cadence.** When to re-run the benchmark — on target promotion, on draft retrain, on engine version bump, on a significant workload shift. Chapter 04 Trigger 1 / 5 overlap.

### Pitfall — "it worked in the benchmark"

A common mistake: the benchmark is run under batch=1 or batch=4 conditions that don't represent peak-hour traffic. The benchmark shows speedup; production shows the compute-bound regression. Always benchmark across the realistic batch-size envelope your fleet actually sees, not just the ideal case.

## The policy — how it is enforced

The benchmark produces the policy; the policy is enforced at admission. Three implementation shapes:

### Shape 1 — Admission-time predicate

```python
def should_speculate(request):
    if request.workload_class in DISABLED_CLASSES:
        return False
    if request.max_tokens < 20:
        return False
    # Batch size is scheduler-level; see Shape 2 for compute-bound regime
    return True
```

The admission handler decides at request entry; the engine runs with speculative on or off per request. Not every engine supports per-request speculative — verify.

### Shape 2 — Scheduler-level cap

The scheduler disables speculative when batch > `B_disable`. More natural fit for the compute-bound regime because the condition is a batch-level fact.

Engine support for this is thinner — the engine's scheduler must know to flip. If your engine doesn't support it natively, autoscale (Shape 3) or raise a feature request.

### Shape 3 — Autoscale earlier

Rather than disabling speculative in the compute-bound regime, add replicas to keep each replica's batch size below `B*`. The policy becomes an autoscaling rule, not a request-handling rule. Module 109 writes the autoscaler.

Trade-off: Shape 3 costs more hardware. Shape 1 and 2 preserve the hardware budget at the cost of per-request SLO variation.

### Multi-shape fleets

Mature fleets combine: Shape 1 for the clearly-disabled classes; Shape 3 for the batch-size regime; Shape 2 if the engine supports it as a safety net.

## Three example triage walks

### Walk 1 — Reasoning tenant breaches SLO; speculative is the cause

- Alert 3 fires. The tenant's `α = 0.38`, TPOT P99 = 95 ms vs. their SLO = 60 ms.
- Benchmark cell confirms: `(workload_class = reasoning, batch = 12)` has net speedup = 0.73 (speculative is slower).
- Policy update: add `reasoning` to the disabled set for this `(target, draft)` pair.
- SLO returns within 15 minutes.
- Chapter 04 Trigger 2 raised to the fine-tuning peer with the reasoning-mix request.

### Walk 2 — Marketing campaign saturates one replica

- Fleet TPOT climbs; per-replica batch rises to 48 at peak hours.
- Benchmark shows `(workload_class = chat, batch > 40)` has net speedup = 0.95 — speculative is marginal.
- Short-term: autoscale aggressively (Shape 3) to drop per-replica batch below 32.
- Medium-term: add Shape 2 if the engine supports batch-threshold disable.
- Chapter 03 Alert 5 ("draft overhead ratio climb") refined with a batch-aware threshold.

### Walk 3 — Long-prompt workload disappoints the product team

- Product team: "we enabled speculative and the latency didn't drop." They expected a 2× speedup like the demo.
- Benchmark: for `prompt > 16k, max_tokens < 50`, TTFT is 95 % of E2E latency. Speculative changes TPOT but not TTFT.
- Explanation to product: speculative is a decode-phase optimisation. Their workload is prefill-dominated; the "speedup" is on the small decode portion.
- No code change. A comms change — the fleet publishes "speculative benefits scale with decode-token count; expect modest gains on prefill-heavy workloads" in the product-facing docs.

## The boundary to the peers

This chapter does not teach:

- **Why the target's decode becomes compute-bound at `B*`** — kernel-level FLOP vs. bandwidth analysis. Performance peer and module 103's chapter on compute vs. bandwidth bounds.
- **How to re-tune the fused verify kernel to raise `B*`** — kernel craft. Performance peer.
- **How to train a better drafter to raise `α` on the disabled class** — fine-tuning peer; chapter 04 writes the request.
- **Autoscaler design** — module 109.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) and [`fine-tuning-engineer`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning). This chapter is the operator's view: *when is speculative a net loss, how to measure it, how to disable it as a policy.*

## Summary

- Two regimes where speculative hurts: **acceptance too low** (`α < α_break_even(k, c_d/c_t)`) and **batch already compute-bound** (batch > `B*`, where the target's decode is FLOP-bound and the draft overhead has nothing to amortise against).
- A third regime where it doesn't hurt but underwhelms: decode-light requests (long prefill, short generation) where TTFT dominates E2E and speculative only touches the decode portion.
- The benchmark is a two-dimensional sweep (workload class × batch size) producing a net-speedup heatmap. Cells with speedup < 1 are "disable" cells.
- The policy enforces at admission (per-request class predicate), at the scheduler (batch-threshold disable), or at the autoscaler (keep batch below `B*`). Mature fleets combine shapes.
- The remedy for low-`α` classes is disable or retrain (chapter 04). The remedy for compute-bound batches is lower max batch, batch-threshold disable, or autoscale earlier. The remedy for decode-light is comms and expectation-management.
- Re-run the benchmark on target promotion, draft retrain, engine bump, or workload shift. The policy is not static.
- Kernel craft (why `B*` is what it is; raising `B*` by kernel optimisation) is the performance peer's domain. Draft training to raise `α` is the fine-tuning peer's. This module writes the policy that is enforced in between.
