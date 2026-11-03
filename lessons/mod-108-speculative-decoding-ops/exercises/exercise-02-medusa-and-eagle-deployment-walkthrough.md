# Exercise 02 — Medusa and EAGLE Deployment Walkthrough

**Time:** 3 hours · **Output:** one self-speculative deployment (Medusa or EAGLE, whichever your engine supports), a comparison against the exercise-01 draft-target baseline, and a per-class recommendation

## Goal

Make [Chapter 02](../02-self-speculative-schemes.md) concrete by deploying a self-speculative scheme (Medusa or EAGLE) and comparing it directly to the draft-target baseline from [Exercise 01](exercise-01-draft-target-speculative-deployment-on-vllm.md). The deliverable is a side-by-side bake-off that lets you (or the fleet) pick which scheme to carry forward.

## Problem statement

The team has shipped draft-target speculative on the production fleet (exercise 01). An engineer forwards a research note claiming that EAGLE-2 or Medusa (depending on your engine and target) achieves meaningfully higher acceptance on comparable targets. Before adopting, you want evidence on your workload. Build the self-speculative deployment, A/B against the exercise-01 baseline, publish the recommendation.

Choose one scheme to deploy end-to-end:

- **Medusa** — if a Medusa-equipped checkpoint for your target family is available on Hugging Face or published by your target's authors, and your engine supports Medusa decoding (verify current engine support).
- **EAGLE / EAGLE-2 / EAGLE-3** — if a published EAGLE drafter for your target exists (the EAGLE repository tracks releases) and your engine supports EAGLE's feature-level drafting (SGLang is a common choice; vLLM / TRT-LLM support varies by release).

If neither scheme is practical on your fleet right now, document that and run the exercise against **Lookahead decoding** (no training required) as a lighter alternative — note that Lookahead's advantage is strongest on repetitive workloads (code, structured output).

## Prerequisites

- Chapter 02 of this module read.
- Exercise 01 completed, with its baseline acceptance / TPOT table in hand.
- A pinned recent release of SGLang (if EAGLE), vLLM (if Medusa or where supported), or TensorRT-LLM. Verify the engine's self-speculative-decoding guide for the current flag surface.
- Access to the Medusa checkpoint or EAGLE drafter for your target family. Confirm licensing and tokenizer parity before downloading.
- The same workload samples (chat, code, reasoning) from exercise 01.

## Requirements

Four deliverables.

### Deliverable A — a running self-speculative deployment

1. **Fetch the artefact.** For Medusa, this is a single checkpoint that bundles the target weights with the trained Medusa heads. For EAGLE, this is a target checkpoint plus a separately-published EAGLE drafter. Record the SHA(s).

2. **Launch.** Verify flag names against the engine's current docs. For EAGLE on SGLang (adjust to the pinned release):

   ```bash
   python -m sglang.launch_server \
     --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
     --tp 4 \
     --speculative-algorithm EAGLE \
     --speculative-draft-model-path <org>/EAGLE-Llama-3.1-70B \
     --speculative-num-steps 4 \
     --speculative-eagle-topk 10
   ```

   For Medusa on vLLM, the pattern is typically a single checkpoint with Medusa heads loaded via a flag naming the scheme and the number of heads; verify against vLLM's current speculative-decoding guide.

3. **Confirm the scheme fired.** The chapter-01 health checks apply with scheme-specific additions:

   - Startup log mentions "medusa" / "eagle" / "lookahead" (as applicable) and names any drafter / head config.
   - Metrics endpoint shows non-zero `*spec*` series under the new scheme's labels.
   - A/B latency sanity on batch=1 chat (should drop vs. non-speculative).
   - Greedy regression — same prompt set as exercise 01 at `temperature=0`, outputs should match (allowing tie-breaking).
   - For EAGLE's tree variants, verify that the tree parameters (`topk`, number of draft tokens, depth) are the ones you intend.

### Deliverable B — the acceptance + TPOT measurement

Mirror exercise 01's Deliverable B: for each of the three workload classes, measure under the new scheme with the recommended default parameters (`k`, tree parameters for EAGLE):

- `α` (acceptance rate — same metric name across schemes, chapter 02's "monitoring is identical" note).
- Mean tokens committed per iteration.
- TPOT P50 and P99.
- Net speedup vs. non-speculative (same baseline from exercise 01).

Output table (per class):

```
| Class     | Scheme / setting   | α     | tokens/iter | TPOT P50 | TPOT P99 | Speedup |
|-----------|--------------------|-------|-------------|----------|----------|---------|
| chat      | spec off           | —     | 1.0         | 48 ms    | 92 ms    | 1.00×   |
| chat      | draft-target k=4   | 0.73  | 2.4         | 24 ms    | 44 ms    | 2.00×   |
| chat      | EAGLE k=4 topk=10  | 0.81  | 3.2         | 18 ms    | 34 ms    | 2.67×   |
| code      | spec off           | —     | 1.0         | 50 ms    | 95 ms    | 1.00×   |
| code      | draft-target k=4   | 0.85  | 3.1         | 19 ms    | 36 ms    | 2.63×   |
| code      | EAGLE k=4 topk=10  | 0.91  | 3.8         | 15 ms    | 28 ms    | 3.33×   |
| reasoning | spec off           | —     | 1.0         | 44 ms    | 85 ms    | 1.00×   |
| reasoning | draft-target k=4   | 0.42  | 1.5         | 61 ms    | 118 ms   | 0.72×   |
| reasoning | EAGLE k=4 topk=10  | 0.55  | 2.1         | 42 ms    | 82 ms    | 1.05×   |
```

Numbers illustrative. Yours will differ.

### Deliverable C — side-by-side recommendation

A one-page comparison that answers:

1. **For each workload class, which scheme wins?** By `α`, by TPOT P99, by net speedup.
2. **For each class, how large is the gap?** A `0.02` gap on α might not justify the lifecycle churn (chapter 04's cost); a `0.08` gap does.
3. **What is the operational overhead difference?** Draft-target has two artefacts to version (chapter 04). Medusa has one artefact. EAGLE has two. Lookahead has none to version.
4. **What is the engine-coupling risk?** Chapter 02 names this: self-speculative schemes couple more tightly to the engine. If the engine upgrades and regresses EAGLE before draft-target, the fleet's rollback has to account for scheme-switching.
5. **Which scheme to carry into production, and why.** Explicit decision with reasoning.

### Deliverable D — `needs-escalation` list

Expected entries:

- "Whether the EAGLE drafter's SHA on Hugging Face is the one the EAGLE paper claims. The EAGLE repository and the drafter's model card are the authority." → EAGLE repo / HF model card.
- "Whether the Medusa heads bundled with the checkpoint were trained with the same chat template as our workload. Likely yes if from the target's maintainers; verify." → model card.
- "Whether the engine's tree-attention mask has a correctness regression at `k > 5`. Observed outputs identical through `k=5`; `k=6` divergence TBD." → performance peer.
- "Whether to adopt the self-speculative scheme now or wait for a published training recipe we can run ourselves. The in-flight decision couples to chapter 04's lifecycle." → fine-tuning peer; chapter 04 Trigger 5.

## Starter guidance

- **One scheme per deployment.** Don't try to flip between schemes on the same engine instance without a restart; the drafter / head config is a startup-time decision.
- **Keep tree parameters inside the recommended range.** EAGLE's `topk`, `num-draft-tokens`, and depth are inter-dependent. The EAGLE repo publishes sensible defaults; start there and tune only after measuring.
- **Record every parameter.** The pair record (chapter 04) for a self-speculative deployment includes the scheme's parameters. Record them in the deliverable.
- **If Medusa heads are custom-trained for your target** by your org or a collaborator, treat that as a chapter-04 Trigger 5 (architecture upgrade). The fine-tuning peer owns the training; this exercise consumes the output.
- **Match sampler config** between the draft-target baseline and the self-speculative run. A sampler change makes the comparison incoherent.

## Fallbacks for constrained environments

- **No published Medusa checkpoint for your target.** Try EAGLE; the drafter list is often wider.
- **No published EAGLE drafter for your target.** Try Lookahead if your engine supports it. If not, document "no self-speculative scheme is production-ready for this target on this engine" and route back to draft-target with chapter 04 Trigger 5 raised to the fine-tuning peer.
- **Engine version doesn't support the scheme on this target architecture.** Pin an engine version that does; or upgrade.
- **No access to the full chat / code / reasoning samples.** Open eval sets as in exercise 01.
- **Lookahead instead:** acknowledge the limitation — Lookahead is strongest on repetitive patterns. Measure on code and structured-output classes; report that chat / reasoning gains are typically smaller.

## Acceptance criteria

1. **Deliverable A** has a running self-speculative deployment with the four + scheme-specific health checks confirmed.
2. **Deliverable B** reports per-class `α`, tokens-per-iter, TPOT P50 / P99, and net speedup for the self-speculative scheme and the exercise-01 draft-target baseline, all against the same non-speculative baseline. Paired samples; sample size documented.
3. **Deliverable C** is a one-page comparison that names the per-class winner, the magnitude of the gap, the operational-overhead difference, the engine-coupling risk, and the explicit recommendation.
4. **Deliverable D** names at least four `needs-escalation` items with the reference for each.
5. The recommendation in C is consistent with the measurements in B — if the gap is small, the recommendation acknowledges the lifecycle cost (chapter 04); if the gap is large, the recommendation names the migration plan.

## Common failure modes

- **Comparing self-speculative at default parameters to draft-target at tuned parameters** (or vice versa). Both must be measured at their own optima for a fair comparison. If exercise 01's `k` was tuned, do the same tuning here.
- **Ignoring the engine-coupling risk** in the recommendation. Chapter 02 names this; chapter 04 Practice 2 inherits it.
- **Over-weighting a single class.** If the self-speculative scheme wins big on code but loses on reasoning, and your fleet's reasoning volume is 40 %, the picture is more complex than "scheme X wins." Decompose the recommendation per class.
- **Not accounting for the one-artefact-vs-two difference in lifecycle cost.** Medusa / Lookahead have a lifecycle advantage (chapter 04 Practice 3); include it in the comparison.
- **Trusting published acceptance numbers over your own measurement.** Chapter 02 names this: "typically higher" is a hypothesis; your workload may differ. Measure.
- **Running a stale greedy regression suite.** If exercise 01's regression suite was 20 prompts, use the same 20 here (plus whatever new ones you add) so the comparison is apples-to-apples.

## Keep it

The side-by-side table and the per-class recommendation feed chapter 04's pair-promotion decision when self-speculative is being considered. The health-check script extends exercise 01's for scheme-specific checks. The engine-coupling-risk note stays in the fleet's documentation when the scheme is in production.
