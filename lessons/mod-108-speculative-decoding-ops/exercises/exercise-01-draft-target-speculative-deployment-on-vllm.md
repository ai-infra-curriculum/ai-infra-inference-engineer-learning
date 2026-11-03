# Exercise 01 — Draft-Target Speculative Deployment on vLLM

**Time:** 3 hours · **Output:** one running `(target, draft)` vLLM deployment, one per-class acceptance table, one `k` sweep, one per-class recommendation, one `needs-escalation` list

## Goal

Make [Chapter 01](../01-draft-target-speculative-decoding.md) concrete by **standing up a draft-target speculative pair on vLLM**, measuring acceptance rate across three distinct workload classes, sweeping speculative length `k`, and producing a per-class recommendation. The deliverable is a reusable template — the shape of work you repeat on every new target or draft that lands on your platform.

## Problem statement

Your platform is about to adopt speculative decoding on an existing Llama-3.x deployment. The CTO wants a bake-off: three workload samples, a draft-target pair stood up, the per-class acceptance and TPOT measured, and the `k` setting recommended for production. You will produce all of it today.

The pair (adjust to the hardware you have access to):

- Target: `meta-llama/Meta-Llama-3.1-8B-Instruct` (if H100 / A100 budget is tight) or `meta-llama/Meta-Llama-3.1-70B-Instruct` (if a 4× H100 node is available).
- Draft: `meta-llama/Llama-3.2-1B-Instruct` (same tokenizer family; verify engine support).

If you do not have access to the above, substitute the closest same-family pair you can run (e.g. a Qwen target + smaller Qwen draft). The exercise is about the methodology — the specific pair is a stand-in.

## Prerequisites

- Chapter 01 of this module read. Chapter 03 skimmed (you will need the metric names).
- A running vLLM install matching a pinned recent release (verify against the vLLM release notes). Verify the release's speculative-decoding guide — the flag spelling has evolved and the exercise's commands may need to be updated to match.
- Access to a GPU (or GPU node) adequate for the chosen target + draft at BF16 or FP8.
- A small, scrubbed sample of representative prompts from three workload classes:
  - **chat** — ~100 open-ended conversational prompts, ~500 tokens prompt, ~200 tokens generation.
  - **code** — ~100 code-completion prompts, ~800 tokens prompt, ~300 tokens generation.
  - **reasoning** — ~100 chain-of-thought prompts, ~400 tokens prompt, ~500+ tokens generation (longer, more entropy-rich).
  - If you cannot produce real traffic, synthesise representative prompts from an open eval set (MT-Bench chat subset, HumanEval for code, GSM8K / MATH for reasoning). Note the source in the deliverable.
- A simple load-driver — `vllm bench serve` or the engine's `benchmark_serving.py` works; a few `httpx`-based scripts work too.

## Requirements

Three deliverables.

### Deliverable A — a running pair + evidence

1. **Launch the pair.** Verify the current flag names against vLLM's speculative-decoding guide for your pinned release. The pattern (adjust names):

   ```bash
   vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
     --tensor-parallel-size 4 \
     --dtype bfloat16 \
     --speculative-model meta-llama/Llama-3.2-1B-Instruct \
     --num-speculative-tokens 4 \
     --port 8000
   ```

2. **Confirm speculative fired.** Chapter 01 names the four health checks:

   - Startup log mentions speculative + the draft model.
   - Metrics endpoint shows non-zero `*spec*` series after a few test requests. Record the exact metric names your engine release exposes.
   - A/B latency sanity: one request at batch=1 should drop TPOT vs. the non-speculative baseline on a chat prompt. If it doesn't, something didn't engage.
   - Greedy regression: run 20 prompts with speculative on and off at `temperature=0`, verify the outputs are identical (or near-identical modulo tie-breaking).

3. **Attach the startup log and metrics dump** to the deliverable.

### Deliverable B — the acceptance + TPOT measurement

For each of the three workload classes (chat, code, reasoning):

1. Drive the sample through the server for a measurement window long enough to stabilise. Target 500+ completions per class per setting.
2. Collect, for each setting:
   - Acceptance rate `α = spec_tokens_accepted / spec_tokens_proposed` (from the engine's metrics or synthesised from per-request fields).
   - TPOT P50 and P99 (from server-side timing or client-side streaming timestamps).
   - Mean tokens accepted per iteration (`spec_tokens_per_iter` from chapter 03 formula).
3. Record non-speculative baselines (same prompts, speculative disabled) for the same measurements.

Output table (per class):

```
| Class     | Setting         | α     | tokens/iter | TPOT P50 | TPOT P99 | Net speedup |
|-----------|-----------------|-------|-------------|----------|----------|-------------|
| chat      | spec off        | —     | 1.0         | 48 ms    | 92 ms    | 1.00×       |
| chat      | spec on, k=4    | 0.73  | 2.4         | 24 ms    | 44 ms    | 2.00×       |
| code      | spec off        | —     | 1.0         | 50 ms    | 95 ms    | 1.00×       |
| code      | spec on, k=4    | 0.85  | 3.1         | 19 ms    | 36 ms    | 2.63×       |
| reasoning | spec off        | —     | 1.0         | 44 ms    | 85 ms    | 1.00×       |
| reasoning | spec on, k=4    | 0.42  | 1.5         | 61 ms    | 118 ms   | 0.72×       |
```

Numbers are illustrative. Yours will differ.

### Deliverable C — the `k` sweep + per-class recommendation

For each class, repeat Deliverable B at `k ∈ {1, 2, 3, 4, 5, 6, 8}` (or whatever subset your engine supports — note the limit). Produce:

1. A `k`-sweep table per class — `α` and TPOT and net speedup at each `k`.
2. The optimum-`k` per class.
3. A per-class recommendation paragraph:

   ```
   Class: reasoning
   Current observed α: 0.42
   Break-even α at k=4, c_d/c_t≈0.1: ~0.31 (chapter 05 table).
   The class is above break-even but barely. Options:
     - Keep speculative on with k=2 (lower overhead, marginal gain).
     - Disable speculative for this class (chapter 05 policy).
     - Request a draft retrain against a mix with more reasoning (chapter 04 Trigger 2).
   Recommendation: disable speculative for reasoning class on this pair while a retrain
   request is in flight. Re-enable with the next pair.
   ```

### Deliverable D — the `needs-escalation` list

A list of specific numbers or behaviours you could not confidently explain from this exercise alone. For each, name the reference you'd consult. Expected entries:

- "The exact break-even α at our observed `c_d / c_t`. The formula gives the ballpark; the empirical measurement from the sweep is the authority." → chapter 05.
- "Whether the engine's speculative path is bypassed on `logprobs` requests. Not visible from our test harness." → chapter 03 Alert 4 instrumentation; engine docs.
- "Whether to raise or lower `k` dynamically per sequence. Chapter 01 names dynamic-`k`; the engine's current support is release-specific." → vLLM release notes.
- "Whether to request a draft retrain against our reasoning-heavy mix. Chapter 04 Trigger 2 shape." → fine-tuning peer, with the chapter-04 retrain bundle.

## Starter guidance

- **Fix the engine version first.** Pin the vLLM release before starting; the speculative-decoding flag surface has evolved and chapter-01 commands may need updating. Record the version in the deliverable.
- **Health-check gate before measurement.** If the startup log doesn't name speculative or the metrics endpoint is empty of `*spec*` series, do not start measurement — speculative isn't firing. Fix first.
- **Measure non-speculative baselines with the same prompts.** The A/B is paired by prompt sample, not fleet-aggregate; chapter 05's methodology.
- **Keep sampling parameters constant.** Chapter 01 names the sampler-compatibility concern; sampler changes mid-measurement will confuse the signal.
- **Record the SHA of both checkpoints.** Chapter 04's pair record requires it; the eventual production deployment will too.

## Fallbacks for constrained environments

- **Smaller target.** If you cannot run 70B, use 8B. The methodology is the same; the absolute numbers differ.
- **CPU-only draft isn't recommended.** Modern engines expect co-located GPU draft; verify before trying anything else.
- **No real workload traffic.** Use open eval samples (MT-Bench chat subset, HumanEval, GSM8K). Note the substitution. The acceptance curves will not match your production fleet exactly but will exhibit the expected shape.
- **Engine doesn't support `k` sweep natively.** Launch a new instance per `k` and measure each separately. Note the engine's restriction.
- **No access to streaming timestamps.** TTFT and TPOT can be approximated from server-side timing; chapter 03 shows the shape. Note the approximation.

## Acceptance criteria

1. **Deliverable A** has a running pair + the four health-check artefacts attached. Any failed health check is documented with the fix.
2. **Deliverable B** reports `α`, tokens-per-iter, TPOT P50, TPOT P99, and net speedup for each of three classes at the default `k`, with the non-speculative baseline recorded for comparison. Confidence intervals noted (or sample size documented so a reader can derive them).
3. **Deliverable C** reports the `k` sweep for each class, names the optimum per class, and recommends a production setting per class with reasoning that references the chapter-05 break-even analysis.
4. **Deliverable D** names at least four `needs-escalation` items with the reference for each.
5. The entire deliverable is reproducible — anyone with the same engine version, hardware, and traffic sample should be able to re-run it.

## Common failure modes

- **Running the baseline against a different sample than the speculative run.** Not paired; results are not comparable. Chapter 05 methodology demands paired samples.
- **Not noticing a silent bypass.** Chapter 03 Alert 4 — the engine may silently bypass speculative on certain sampler combinations. The `α` metric reports 0 or NaN on bypassed requests; verify.
- **Measuring during warm-up.** CUDA graphs, scheduler state, and prefix caches are not stable in the first minute. Discard warm-up data.
- **Fixing `k` without a sweep.** The whole point of the exercise is the per-class optimum; a single `k` across classes is the chapter-01 Error 3.
- **Comparing TPOT at different batch sizes.** Continuous batching makes batch an output. If the two settings produce different batch distributions on the same load, you are measuring batch, not speculative. Chapter 05's methodology section covers.
- **Not recording the SHAs.** Chapter 04 requires them. Record on the first run.
- **Claiming a pair "works" without the greedy regression check.** If the engine's rejection sampling is subtly wrong, speculative-on and speculative-off produce different outputs at `temperature=0`. Chapter 01's health check 5 catches it.

## Keep it

The three-class acceptance table and the `k`-sweep tables become the deploy-time baseline for chapter 04's pair record. The health-check script becomes the start-up gate for every new pair you promote. The `needs-escalation` list seeds the fleet's documentation gaps. Exercise 03 adds the production monitoring on top of what this exercise manually measured.
