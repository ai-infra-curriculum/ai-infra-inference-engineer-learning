# Chapter 04 — Authoring an Accuracy Regression Gate

> "The quantised model looks fine" is not a release decision. "Perplexity delta on canary is 0.8% ± 0.3% at 95% CI, under the 2% threshold; task-specific delta on `mt-bench-subset` is 1.1% ± 0.6%, under the 3% threshold" — that is a release decision.

## Motivation

A quantised variant that compresses a 70B model from 140 GiB to 35 GiB and triples throughput is worthless if it also loses 8% on code generation and nobody noticed until a customer did. The accuracy regression gate is the automatic check that catches that case before the swap. It is the single most important deliverable from this module beyond the quantised artefact itself.

By the end of this chapter you can:

- Pick a canary dataset that matches the production workload.
- Pick a perplexity + task-specific metric pair.
- Compute a bootstrap confidence interval so your pass / fail decision has defensible statistics.
- Author the gate as a CI job that blocks the artefact's promotion if it fails.
- Know what to escalate to [`model-evaluation-engineer`](https://github.com/ml-engineering-curriculum/model-evaluation-engineer-learning) and how.

The gate this chapter authors is **narrow by design**: it checks whether a quantised variant of a known model, serving a known workload, regresses within a stated tolerance. It is **not** the full eval-benchmark discipline — that is the model-evaluation-engineer peer, and this chapter cites the boundary explicitly.

<!-- needs-research: specific statistical calibration (bootstrap sample count, significance level, multiple-comparison correction) is a methodology-depth question. This chapter uses defensible defaults; the model-evaluation-engineer peer owns the deeper calibration. -->

## The structural picture — what a release gate is

A release gate is a boolean function that takes an artefact and a reference and returns pass / fail with evidence. For a quantisation change:

```
gate(artefact, reference, canary, thresholds) -> {pass, fail, escalate}
```

- **artefact** — the quantised variant under test (an output of chapter 03's workflow).
- **reference** — the baseline against which to measure (BF16 full-precision, usually).
- **canary** — the dataset + metric pair the comparison runs on.
- **thresholds** — the stated maximum tolerable regression and the escalation band.
- Returns **pass** if the regression is within the pass threshold, **fail** if above the fail threshold, **escalate** if between (needs model-evaluation-engineer review).

The gate blocks promotion. If it fails, the artefact does not advance to production. If it escalates, the artefact pauses and the model-evaluation-engineer peer looks at the signal before anyone decides.

## The canary dataset

### What a canary is

A small (hundreds to a few thousand prompts), fixed, versioned dataset whose composition is a proxy for the production traffic distribution. It is **not** the full eval benchmark suite — it is a daily-regression check. The full benchmark suite is model-evaluation-engineer's territory and runs much less often (per-release, not per-conversion).

### What a canary needs

- **Fixed.** Does not drift between runs. Pinned to a hash. If the canary content changes, the regression number changes for reasons unrelated to the model.
- **Versioned.** When you *do* update the canary (because production traffic shape changed), bump the version so historical regression numbers are comparable only within a version.
- **Representative.** The prompts should cover the task mix of production. If 60% of production traffic is chat, 30% code, 10% math, the canary should mirror that.
- **Small enough to run cheaply.** 500–2000 prompts is typical. Each prompt generates for a bounded number of tokens.
- **Diverse enough to catch regressions.** 2000 prompts of a single type miss regressions in other types.
- **Not in training data.** If the canary is in any model's training data, the model is overfit to it; the gate becomes meaningless.

### Composing a canary

Three ingredients, in roughly these proportions for a general chat model:

1. **Representative-workload samples** (50–70%). Drawn from production logs (scrubbed for PII per security peer's guidance), or hand-curated from the product's actual usage.
2. **Public eval samples** (20–40%). Pulled from established benchmarks (MMLU, HumanEval, GSM8K, MT-Bench, etc.) — the "fingerprint" of general model quality. **Caution:** these can leak into training data; verify. The model-evaluation-engineer peer owns the leakage audit.
3. **Known-failure cases** (5–10%). Historical regressions, model-card warnings, known-hard prompts for this model family. Each new regression caught in production feeds back here (chapter 06 of module 113 is the discipline for this).

### Prompt length distribution

Match production. If production has 10% long-context (> 8 k tokens) prompts, the canary should too. Short-only canaries miss long-context regressions, which quantisation can introduce.

### Store the canary like code

- Version-controlled.
- Content-addressable (hash of the content; detect accidental edits).
- Reproducibility manifest: tokenizer, generation parameters, random seed.
- Clear licence (your organisation's; downstream prompts might have their own).

## The metric pair — perplexity and a task-specific metric

### Perplexity

**Perplexity** = `exp(mean cross-entropy)`. It is the model's **average surprise per token** on a held-out corpus. Lower is better. Perplexity is the single most useful **model-level** metric for a quantisation change because:

- It is cheap to compute (one forward pass; no sampling).
- It is **continuous** — a small quantisation error produces a small perplexity change, so it is a sensitive instrument.
- It is **model-agnostic** — any decoder-only LM has a perplexity on any corpus.
- It is **not game-able** by the quantisation algorithm (the algorithm would have to deliberately improve on calibration perplexity).

**Operator caveats:**

- Perplexity is a scalar; it can hide local degradations.
- Perplexity depends on the tokenizer; comparing perplexities across tokenizers is meaningless. Keep the tokenizer fixed.
- Perplexity depends on the corpus. A model's perplexity on wikitext is not its perplexity on code. Report both.

### A task-specific metric

Perplexity alone is not enough. The quantised variant might have the same perplexity on wikitext but drop 10% on code generation. The gate needs a **second metric** that is sensitive to the task mix in production.

Common choices (pick the one that matches the model's primary workload):

- **Chat / dialogue:** `mt-bench` style LLM-as-judge scoring on a canary of multi-turn prompts (judge the responses with a reference model; measure pairwise win rate against the baseline).
- **Code generation:** HumanEval / HumanEval+ / MBPP pass@1 on a canary subset. Deterministic sampling.
- **Math / reasoning:** GSM8K / MATH subset accuracy.
- **Instruction following:** AlpacaEval 2 / Arena-Hard-Auto on a canary subset.
- **Retrieval / RAG:** workload-specific — needle-in-haystack recall at target context length, or a RAG-answer accuracy score.
- **Classification / structured output:** exact-match or F1 on a labelled canary.

**Rule:** pick one task-specific metric; measure one thing well. More metrics add noise and false alarms if you don't correct for multiple comparisons. The model-evaluation-engineer peer owns the multi-metric methodology.

### The pair, together

- **Perplexity** catches small, broad-spectrum regressions and is cheap.
- **Task-specific metric** catches regressions in the model's actual job.
- Both must pass for the artefact to advance.

## The bootstrap confidence interval

The naive approach — "quantised variant scored 0.73 on canary, baseline scored 0.75, that's a regression of 0.02" — ignores sampling noise. The regression could be noise on a 500-prompt canary; the regression could be real on a 50 000-prompt canary. The bootstrap CI puts an uncertainty band around the number so pass / fail decisions are defensible.

### The procedure

1. **Compute** the metric on the quantised variant and on the baseline, per prompt. You now have two arrays of per-prompt scores (or per-prompt pass/fail), length = canary size.
2. **Pair** them (prompt-by-prompt). This is important — paired variance is smaller than unpaired variance; the test is more sensitive. Compute the **per-prompt delta** (quantised minus baseline, or quantised's score as a fraction of baseline's).
3. **Resample** with replacement from the per-prompt deltas. Pick N (number of bootstrap samples; 1000 or 10000 is typical). For each resample, compute the aggregate statistic (mean delta, or the metric itself).
4. **Report** the 2.5th and 97.5th percentiles of the bootstrap distribution. That is the 95% confidence interval of the mean delta.
5. **Decide**: if the entire CI is below the pass threshold, pass. If the entire CI is above the fail threshold, fail. If the CI straddles any threshold, escalate.

### A concrete gate specification

```yaml
gate: fp8_quantization_release_gate
canary:
  dataset: canary-v3
  size: 1024 prompts
  tokenizer: pinned
  generation:
    temperature: 0.0
    max_new_tokens: 512
    seed: 42
metrics:
  perplexity:
    corpus: wikitext-2-raw-v1 subset of 100 passages
    thresholds:
      pass: delta <= +1%
      escalate: +1% < delta <= +3%
      fail: delta > +3%
  task_specific:
    name: humaneval_pass_at_1
    canary: humaneval-canary (164 problems)
    thresholds:
      pass: delta >= -2 points
      escalate: -5 <= delta < -2 points
      fail: delta < -5 points
bootstrap:
  samples: 1000
  method: paired per-prompt
  ci: 95%
  decision_rule: entire_ci_within_pass_band
```

Both metrics must pass for the gate to pass. Either can escalate; a fail on either fails the gate.

### Thresholds — where to start

The exact numbers are workload-specific. **Starting points** for a weight-only INT4 or FP8 quantisation of a well-behaved base model on a general chat workload:

- **Perplexity delta pass:** +1% (near the noise floor on a canary this size).
- **Perplexity delta fail:** +5% (anything above here is a clear regression).
- **Task-specific delta pass:** 1-2 points (depending on metric scale).
- **Task-specific delta fail:** 5-10 points (depending on metric scale).

**Calibrate on real history:** run the gate on 5–10 historical quantisations of the same model and see where the noise floor is. The model-evaluation-engineer peer owns the calibration deepen.

## The gate as a CI job

### Workflow shape

The gate runs as a CI job on every quantised artefact before it is promoted:

```
artefact-produced
     │
     ├─> level-0/1/3 conversion validation (chapter 03)
     │       │ (pass)
     ├─> accuracy regression gate (this chapter)
     │       │
     │       ├─> pass → promote to staging registry
     │       ├─> escalate → notify model-evaluation-engineer; artefact held
     │       └─> fail → artefact rejected; incident ticket
     │
     └─> (in parallel) performance regression gate (throughput, latency)
```

### Inputs

- The artefact under test (by registry ID).
- The reference baseline (by registry ID — usually the current production BF16 or FP8).
- The canary dataset (by hash — immutable).
- The thresholds spec (in version control alongside the gate).

### Outputs

- The pass / fail / escalate decision.
- The full metric results with bootstrap CIs.
- A machine-readable report (JSON) and a human-readable summary (Markdown).
- Attached to the artefact's lineage record (chapter 05).

### Reproducibility

Every gate run must be reproducible from inputs. That means:

- **Fixed** random seeds for sampling and bootstrapping.
- **Pinned** engine versions, driver versions, kernel versions (even the kernel can affect FP8 results via different fused-op choices).
- **Pinned** GPU model (the gate run on a different SM might produce different results for FP8; verify).

Keep the gate's execution environment in a container image you version alongside the gate.

## When to escalate

Three signatures that mean "do not decide this alone":

### The CI straddles a threshold

The 95% CI on the perplexity delta is `[+0.5%, +2.1%]`; the pass threshold is 1% and the fail threshold is 3%. The CI straddles the pass threshold. Need more samples or a methodology review — escalate.

### Perplexity passes but task metric fails (or vice versa)

Perplexity delta is +0.3% (clear pass); HumanEval delta is −8 points (clear fail). The quantisation preserves average token prediction but destroys code generation specifically. This is a *localised* regression; the model-evaluation-engineer peer looks at it with a wider lens (which subpopulation of prompts caused it, is the pattern consistent across HumanEval+ and MBPP, is this a known pathology).

### Historical anomaly

Metrics are within thresholds, but one metric is wildly out of line with historical quantisations of the same model (e.g. all prior FP8 variants had +0.2% perplexity; this one has +0.9%). The gate passes mechanically but the signal is unusual. Escalate for a sanity check.

### How to escalate well

- Produce a complete artefact for the peer: the canary ID, the metric spec, the pre-registered thresholds, the raw per-prompt results, the bootstrap distribution, the historical comparison.
- Pre-register the hypothesis you suspect (e.g. "activation outliers in layer 25 might be the cause; the Hadamard-rotation remedy might fix it"). Do not require the peer to re-derive the question.
- Do not swap in production on an escalation; keep the current artefact in place until the escalation closes.

## What this gate does *not* do

A list of things that are explicitly out of scope here and belong at the model-evaluation-engineer peer or elsewhere:

- **Full benchmark suite evaluation.** The gate uses a canary subset of 1 or 2 benchmarks. The full-suite run (dozens of benchmarks, hundreds of thousands of prompts) is a model-release decision, not a conversion decision.
- **Leakage audit.** Whether the canary prompts are in the base model's training data is a methodology question that model-evaluation-engineer owns.
- **Benchmark decay.** Standard benchmarks like MMLU have aged; new models score near ceiling and signal is weak. The peer owns the benchmark curation discipline.
- **Statistical rigour calibration.** The bootstrap CI uses defensible defaults; whether 1000 samples is enough, whether the paired test is the right one, whether multiple-comparison correction is needed — the peer owns the deeper calibration.
- **Correlating offline eval with online outcomes.** Does canary passing correlate with user-reported quality? That is an eval-platform question, not a release-gate question.
- **Safety / policy evaluation.** The accuracy gate here is a *utility* gate — does the model still do its job. The *safety* gate — does the quantisation break refusal / content-policy behaviour — is a separate concern that routes to the security peer and the eval peer (see module 113).

## Common misreadings

- **"Lower perplexity means better model."** On the same tokenizer and corpus, yes. Across tokenizers or corpora, no — perplexity is not comparable.
- **"The gate is replaced by the task-specific metric alone."** No — perplexity is cheap, broad-spectrum, and catches regressions the task metric misses. Both or nothing.
- **"Bootstrap is overkill; the mean is enough."** On a 500-prompt canary the sampling noise on a mean can be ±1–2% without a CI; without a CI you'll fail-close on noise.
- **"If it passes the gate, it is safe to ship."** The gate is one axis. Performance regression, memory regression, API-contract regression, safety regression are separate axes.
- **"The gate can replace the chapter-03 conversion validation."** No — chapter-03 isolates *where* the regression came from (conversion vs. quantisation). The gate is a pass / fail on the final artefact.
- **"The thresholds come from the model-evaluation-engineer."** The thresholds are negotiated between this role (serving) and that peer (methodology). This role writes the first draft; the peer calibrates and signs off.

## Summary

- The release gate is a **canary + metric pair + bootstrap CI** that returns **pass / fail / escalate** for a quantised artefact against a reference.
- The **canary** is small (hundreds to thousands), fixed, versioned, representative, and not in the training set.
- The **metric pair** is **perplexity** (cheap, broad-spectrum) and **one task-specific metric** (sensitive to the model's job). Both must pass.
- The **bootstrap CI** turns sampling noise into a defensible pass / fail band. Paired per-prompt resampling; 1000+ bootstrap samples; 95% CI; `entire_ci_within_pass_band` as the decision rule.
- The gate runs as a **CI job** on every quantised artefact; it blocks promotion if it fails and escalates if the CI straddles a threshold.
- **Escalation** targets the model-evaluation-engineer peer with a complete artefact (canary, metrics, thresholds, raw results, historical comparison, hypothesis).
- The gate is **narrow by design**: full-benchmark eval, leakage audit, benchmark-decay, statistical calibration, offline-online correlation, and safety eval all live at the model-evaluation-engineer peer or elsewhere.

Chapter 05 moves from "did this artefact pass the gate" to "is this artefact's metadata and signing fit for the ML-platform peer's registry" — the handshake contract on the artefact itself.
