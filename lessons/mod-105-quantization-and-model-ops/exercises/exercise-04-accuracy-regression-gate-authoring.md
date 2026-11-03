# Exercise 04 — Accuracy Regression Gate Authoring

**Time:** 4 hours · **Output:** one versioned canary dataset, one metric-pair spec, one bootstrap-CI implementation, one CI job definition, one gate run against a real variant, one escalation drill

## Goal

Make [Chapter 04](../04-accuracy-regression-gate.md) concrete by **authoring a real accuracy regression gate** — canary, metric pair, bootstrap CI, thresholds, CI job — and **running it against a real quantised variant** (from exercises 01 or 02). By the end you can hand this gate to the ML-platform peer with every field they need.

## Problem statement

The quantised variants you produced in exercises 01 and 02 are not yet promotable to production because there is no automated check that answers "did this quantisation preserve the model's utility within tolerance?" This exercise builds that check. The gate becomes a mandatory step on every future quantisation of this model family.

## Prerequisites

- A working deployment of the BF16 baseline and at least one quantised variant (from exercise 01 or 02).
- The exercise-01 benchmark harness.
- Python 3.10+ with `numpy`, `scipy`, and a stats library.
- Access to a CI runner (GitHub Actions, GitLab CI, Jenkins, Argo) or a documented local equivalent.
- Chapters 04 and 05 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — The canary dataset (versioned)

Assemble a canary of 500–1000 prompts composed per chapter 04's guidance:

- 50–70% representative-workload samples (synthesise, draw from documented traffic logs with PII scrubbed, or hand-curate).
- 20–40% public eval samples (MMLU subset, HumanEval subset, GSM8K subset, MT-Bench prompts — pick 1–2 benchmarks sized to the target workload).
- 5–10% known-failure cases (if any; otherwise reserve the slots for the first regression).

Store the canary as:

- A JSON Lines file with `{"id": "...", "source": "...", "prompt": "...", "reference_completion": "...", "licence": "..."}` per line.
- A `canary.yaml` manifest with `version`, `total_prompts`, `composition_breakdown`, `tokenizer`, `generation_defaults` (temperature, max_new_tokens, seed), `created_at_utc`, `content_sha256`.
- Version-controlled; content-hashed; immutable after publish.

**Non-negotiable**: every prompt has a documented source and licence; nothing from a sketchy or restricted provenance.

### Artefact B — The metric-pair spec

Pick and specify your two metrics:

- **Metric 1 — perplexity**:
  - Corpus: 100-passage subset of wikitext-2-raw-v1 (or an equivalent well-behaved corpus; document choice).
  - Tokenizer: pinned to the model's tokenizer.
  - Reported as: raw perplexity on reference and candidate, absolute delta, percentage delta.

- **Metric 2 — task-specific**: pick ONE from the chapter-04 options that matches the model's primary workload:
  - For chat / instruct: judged-pairwise win rate on MT-Bench canary subset using a judge model (reference: GPT-4-class judge; document the judge model and the judge rubric).
  - For code: HumanEval `pass@1` with greedy sampling.
  - For math: GSM8K accuracy (exact-match on numeric answer).
  - For instruction-following: AlpacaEval 2 win-rate on canary subset.
  - For your specific product workload: hand-defined (exact-match, F1, or similar) — document.

Document as a YAML spec (chapter 04 shows the shape):

```yaml
metrics:
  perplexity:
    corpus: wikitext-2-raw-v1-100-passages
    corpus_hash_sha256: ...
    thresholds:
      pass_delta_percent: 1.0
      fail_delta_percent: 5.0
  task_specific:
    name: humaneval_pass_at_1
    canary_subset: humaneval_full  # or 50-problem subset
    sampling:
      temperature: 0.0
      max_new_tokens: 512
      seed: 42
    thresholds:
      pass_delta_points: -2.0
      fail_delta_points: -8.0
```

### Artefact C — Bootstrap-CI implementation

Write a reproducible script that, given reference and candidate per-prompt score arrays:

- Computes the per-prompt delta.
- Resamples with replacement (N=1000) from the per-prompt delta array.
- For each resample, computes the aggregate statistic (mean delta or raw delta — document which).
- Reports the 2.5th and 97.5th percentiles → the 95% CI.
- Applies the decision rule: `pass` if the entire CI is within the pass band; `fail` if the entire CI is in the fail band; `escalate` otherwise.
- Writes a machine-readable JSON report and a human-readable Markdown report.

Pin the random seed; the script must be deterministic for the same inputs.

Test the script on synthetic data: generate two score arrays with known mean difference (e.g. 0.1), verify the CI contains 0.1, verify the pass/fail decision matches expectation.

### Artefact D — The CI job definition

Codify the gate as a CI job that:

1. Takes two inputs: `--artefact-id <id>` (the candidate) and `--reference-id <id>` (the baseline).
2. Pulls both artefacts from the registry (or local path for this exercise).
3. Pulls the canary dataset and the metric spec by hash.
4. Spins up a serving process for each artefact (or hits the already-running deployments from exercises 01/02).
5. Runs each canary prompt through each deployment with the pinned sampling config.
6. Computes perplexity on the perplexity corpus for each deployment.
7. Runs the task-specific metric per prompt.
8. Invokes the Artefact C bootstrap script.
9. Emits the pass / fail / escalate verdict.
10. Attaches the full report to the artefact's lineage.

Deliverable: a `.github/workflows/accuracy-gate.yml` (or equivalent for your CI system) that invokes the above as a reproducible pipeline. Supplementary scripts should live alongside.

Document the expected runtime (minutes to tens of minutes) and the required hardware.

### Artefact E — A gate run against a real variant + escalation drill

Run the gate against one quantised variant from exercise 01 or 02 (pick the most interesting — the FP8 or the AWQ-INT4).

Produce:

- The gate run's JSON and Markdown reports.
- A screenshot or paste of the CI run (if run through a CI system).
- The pass / fail / escalate verdict and your interpretation.

**Then run the escalation drill:**

- Deliberately pick a threshold that will cause an **escalate** verdict (e.g. tighten the pass threshold until the CI straddles it).
- Walk through the chapter-04 "how to escalate well" protocol:
  - Produce a complete escalation package: canary ID, metric spec, pre-registered thresholds, raw per-prompt results, bootstrap distribution, historical comparison (if you have one), pre-registered hypothesis.
  - Write a one-page handoff doc you'd send to the model-evaluation-engineer peer.
- Pretend you are the peer; critique your own package. What's missing?

Writeup: ≤ 400 words covering what the gate ran produced, how the escalation was handled, and what you'd change about the gate's thresholds for the next iteration.

## Starter guidance

- **Keep the canary small.** 500–1000 prompts is enough; 10 000 is expensive to run on every gate.
- **Pin the judge model** if using LLM-as-judge. Judges have versions and biases; a judge drift is a confounder.
- **Pin seeds everywhere.** Sampling, bootstrap, random choice.
- **Prefer paired per-prompt resampling** over unpaired resampling (chapter 04). The variance is smaller, the test is more sensitive.
- **Treat the pass / fail thresholds as a first draft.** Calibrate against historical quantisation runs where possible; otherwise be conservative at first and tighten as data arrives.
- **Pre-register everything.** The canary, the metrics, the thresholds, the decision rule — all pre-registered before running the gate against a real artefact. Post-hoc threshold adjustment destroys the gate's credibility.
- **Don't skip the perplexity run.** It is the cheap, broad-spectrum sensor; many regressions it catches would not be caught by the task-specific metric.
- **Make the report self-contained.** The ML-platform peer and model-evaluation-engineer peer should be able to read the Markdown report and understand the decision without talking to you.

## Fallbacks for constrained hardware

- **One GPU, one deployment at a time.** Serial reference and candidate runs are fine; the harness can save intermediate state.
- **No MT-Bench judge access.** Swap MT-Bench for a smaller, cheaper metric (AlpacaEval 2 with a smaller judge, or a greedy-sampled exact-match benchmark). Document the substitution.
- **No CI runner.** Run the pipeline as a shell script locally; document the manual steps. The content of the gate is what matters; CI is the automation.

## Acceptance criteria

1. **Artefact A**'s canary dataset is pinned, hashed, versioned, licensed, and composition-documented; sits under 1000 prompts.
2. **Artefact B**'s metric spec is a YAML with pass / escalate / fail thresholds for both metrics; corpus and canary are hashed.
3. **Artefact C**'s bootstrap script is deterministic given fixed seed, tested on synthetic data, and produces both JSON and Markdown reports.
4. **Artefact D**'s CI job runs end-to-end against a sample input pair, including serving spin-up (or references to existing deployments).
5. **Artefact E**'s gate run produces a real pass / fail / escalate verdict on a real quantised variant; the escalation drill produces a complete handoff package.
6. All five artefacts live in version control; the canary and metric spec are referenced by hash from the gate.

## Common failure modes

- **Threshold shopping.** Running the gate, seeing a fail, loosening the threshold — this invalidates the gate. Pre-register.
- **Canary leakage.** Prompts in the canary are also in the model's training data. The regression number is biased optimistically. Document provenance.
- **Judge drift.** Using a different judge model between reference and candidate runs. Pin the judge.
- **Over-metric.** Three or four metrics without multiple-comparison correction. Pick one task-specific metric; add perplexity. Enough.
- **Noise-floor miscalibration.** Pass threshold at +0.1% perplexity on a 500-prompt canary — sampling noise alone will fail some runs. Calibrate the threshold against the noise floor.
- **Mixing seeds.** Different seeds in reference and candidate runs introduces sampling-noise differences that are not model differences. Pin.
- **Treating the escalation as a failure.** Escalation is a signal, not a decision; the peer looks at it and decides.
- **Running the gate without the chapter-03 conversion validation first.** The chapter-04 gate measures the quantised artefact against a reference; chapter-03 isolates whether the regression came from the format conversion or the quantisation. Run both.

## Keep it

The canary dataset, metric spec, bootstrap script, and CI job are the module's primary operational deliverables. They outlive this exercise and apply to every future quantisation change. The escalation handoff template becomes the standard interface with the model-evaluation-engineer peer. Exercise 05 (release contract) references this exercise's gate output directly.
