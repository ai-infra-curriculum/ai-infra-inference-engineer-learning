# Exercise 04 — Preemption Policy Swap-vs-Recompute Drill

**Time:** 2 hours · **Output:** two preemption traces (swap, recompute), one measured break-even analysis, one chosen policy with reasoning, one starvation-guard verification

## Goal

Make [Chapter 04](../04-preemption-policies.md) concrete by **forcing the engine into preemption** under both swap and recompute modes against a reproducible workload, measuring the recovery TTFT and throughput cost, verifying the starvation guard, and picking a production policy with evidence.

## Problem statement

You have the scheduler config from exercise 02 and the chunked-prefill setpoint from exercise 03. Preemption is the next pressure-relief valve. The policy decision — swap vs. recompute, and who-to-evict — has a real cost signature that this exercise forces into the open.

The drill: deliberately overcommit the scheduler, measure what preemption does, and document the policy choice.

## Prerequisites

- The engine, load generator, and metric scraping from exercises 02 and 03.
- Enough CPU DRAM available for `--swap-space` configuration (16 GB minimum per worker is reasonable for a 7B / 13B workload; verify your platform has headroom).
- Access to the engine's preemption-mode knob and swap-space config.

## Requirements

Deliver four artefacts.

### Artefact A — Overcommit workload

Design a workload that forces preemption:

- Low `--max-num-seqs` (e.g. 16 or 32, below your Pareto setpoint).
- Enough concurrent clients at the load generator to push admission pressure (3–5× the running-set cap).
- A mix of short and long prompts — the long prompts are the ones most expensive to preempt and the ones most likely to be evicted on arrival of a newcomer.
- A burst phase: steady traffic for 3 minutes, then a 1-minute burst of long-prompt arrivals, then back to steady.

Document the exact configuration so the drill is reproducible.

### Artefact B — Two preemption traces (swap, recompute)

Run the Artefact A workload twice:

**Run 1: Swap mode.**

- Engine launched with swap enabled (vLLM: `--swap-space 16`; verify the equivalent for your engine).
- Preemption mode: swap (if your engine lets you force the mode; otherwise the behaviour is "swap when space available, else fall back"). Verify by watching metrics.
- Record for the 10-minute run (5-min warmup + 5-min measured):
  - Preemption rate (preemptions per request / per iteration).
  - Swap-out and swap-in event counts.
  - CPU swap-pool usage (`cpu_cache_usage_perc` or equivalent).
  - TTFT P95 on **restored** sequences (track by request-id — the second TTFT after re-admission).
  - E2E latency distribution for preempted vs. not-preempted sequences.
  - Overall throughput (tokens/sec).

**Run 2: Recompute mode.**

- Engine launched to force recompute mode (vLLM: use a very small `--swap-space` so the engine falls through; or an explicit mode flag if your version exposes one). Verify by watching that swap counts stay at zero and recompute counts rise.
- Record the same metrics as Run 1.

Both runs use the same load-generator seed and the same workload spec.

### Artefact C — Break-even analysis

Compute:

| Metric | Run 1 (swap) | Run 2 (recompute) |
|---|---|---|
| Preemption rate | | |
| Mean "cost per preemption" (seconds of GPU-time + PCIe time) | | |
| TTFT P95 on preempted sequences | | |
| E2E P95 on preempted sequences | | |
| Throughput (tokens/sec) | | |
| GPU utilisation during post-preemption restore | | |

Compare:

- Which mode has lower TTFT P95 on preempted requests?
- Which mode has lower throughput cost per preemption event?
- Which mode preserves more sequences without starvation?
- What is the **break-even prompt length** at which recompute = swap cost? (Approximate — compute the swap round-trip time per MB of KV × average KV size per sequence vs. the recompute prefill cost at your model + hardware. Chapter 04 gives the intuition; your numbers make it real.)

Write a one-paragraph verdict: for *this* workload on *this* hardware, which mode wins and why.

### Artefact D — Starvation-guard verification

With the engine still under Artefact A workload, verify (or disprove) that the engine has a starvation guard:

1. Add a tag or priority label to a specific long request so you can track it.
2. Submit it alongside the burst.
3. Watch the request-id in both scheduler logs and the output stream.
4. Record how many times it is preempted before completing.

Observations to make and record:

- Does the sequence complete at all within a reasonable upper bound (e.g. 2× its expected E2E)?
- If preempted multiple times, does the engine escalate its priority to protect it?
- If the engine has no starvation guard, document the risk and prescribe a mitigation (per-request timeout at the API gateway, or application-level retry policy).

For engines that do not expose starvation-guard behaviour clearly in the metrics, this artefact can be a code-review of the engine's selection function (reusing exercise 01 artefact A's source tour): is there a preemption-cap, promotion-on-preemption, or admission-freeze in the code?

## Starter guidance

- **Overcommit by lowering `max-num-seqs`, not by raising RPS.** The former produces a clean, predictable preemption rate; the latter also increases prefill cost and conflates variables.
- **Verify which mode is actually in use.** If you ask for recompute but your engine silently falls back to swap when space is available (or vice versa), your trace is for the wrong mode. Watch the metric counters.
- **Match the swap-space sizing to the workload.** Too-small swap forces recompute even in "swap" mode (hybrid behaviour, chapter 04). Too-large wastes DRAM. For the drill, 16 GB per worker is reasonable; adjust to your model's KV footprint.
- **Measure during the restore, not just at steady state.** The cost of a preemption is paid when the sequence resumes. The dashboard view that matters is the TTFT of the second-admission, by request-id.
- **A per-request trace is more useful than aggregate metrics.** Record request-id alongside TTFT and preemption count; you can slice later.
- **For starvation verification**, if your engine does not surface preemption counts per request, add logging in your local clone — do not upstream. The point is to see what *actually* happens to that one sequence.

## Fallbacks for constrained hardware

- **Smaller model.** The break-even math shifts with model size (bigger model → more KV per sequence → swap more expensive); but the policy shape is the same. Document the model.
- **Shorter runs.** 5-minute measured windows are enough if the preemption rate is high.
- **CPU-only.** Not useful for this drill — swap costs are PCIe, and recompute costs are GPU prefill; neither is representative on CPU.

## Acceptance criteria

1. **Artefact A** describes a workload that *actually* provokes preemption. If the preemption count in either run is zero, revisit the overcommit configuration — the drill failed to engage its own mechanism.
2. **Artefact B** has both runs with per-request restore TTFT recorded. Aggregate preemption counts alone are not enough.
3. **Artefact C**'s break-even analysis names a *specific* prompt length (or prompt-length range) at which recompute becomes cheaper than swap for your model + hardware. "It depends" is not an answer.
4. **Artefact D** either confirms the starvation guard is active (with evidence) or documents its absence with a prescribed mitigation.
5. The policy pick is pinned alongside the exercise-02 config in version control with a comment pointing at the drill evidence.

## Common failure modes

- **No preemption fires.** The whole exercise is uninteresting if the engine never evicts. Lower `max-num-seqs` and increase concurrent clients until the preemption counter moves.
- **Mode confusion.** Many engines have a hybrid "swap when space, else recompute" default. Verify which mode you actually ran in by watching the counters. If your engine does not let you force a mode exclusively, document the hybrid behaviour.
- **Aggregate metrics only.** TTFT P95 across all requests may look fine; what matters is TTFT P95 *of the preempted sequences*. Instrument by request-id.
- **Starvation guard assumed, not verified.** "The engine must have one" is not evidence. Either observe it in a trace or read the code.
- **Pinning swap=16 GB without verification.** Swap size depends on your workload's KV footprint; chapter 04 gives the sizing heuristic. Compute it, don't guess.

## Keep it

The policy pick and the starvation-guard verification are production artefacts. Add them to the serving-pool's runbook. The drill script is re-runnable on every engine upgrade and every major traffic-shape shift. If the engine's eviction-selection policy changes between releases, this drill is the first place you re-measure.
