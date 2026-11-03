# Exercise 01 — Canary with Response Diffing and Auto-Rollback

**Time:** 6 hours · **Output:** one working canary harness that ramps a candidate model version against a primary, scores response diffs against a semantic-equivalence tolerance, auto-rolls-back on candidate-scoped SLO burn-rate alerts, and demonstrates a full promote-and-rollback cycle end-to-end.

## Goal

Make [Chapter 01](../01-canary-and-shadow-rollouts.md) concrete. Build the canary primitive against a real inference engine, with real traffic (replay or synthesised), real SLO gates, and a real rollback trigger. The exercise is complete when an on-call engineer could use this harness tomorrow against a new candidate model.

## Problem statement

Your platform serves a chat model. A candidate version is ready (same model, different engine flag, or a quantisation change you've shadow-rolled — any realistic change counts). You will route a growing fraction of traffic to the candidate, score every candidate response against the primary under a semantic-equivalence tolerance, watch the inference axis set from module 110 for candidate-scoped SLO burn, and auto-rollback when the gate fails.

The deliverable is not a demo run; it is the harness itself, reusable for your fleet's next real rollout.

## Prerequisites

- Chapters 01, 02, and 04 of this module read.
- Chapter 02 of module 110 (SLO authoring) completed or available as a reference for burn-rate alert design.
- An inference engine stood up with at least two replicas — vLLM, SGLang, or Triton + TRT-LLM. Two distinct model versions available (even if the "candidate" is just the primary with a different flag, as long as responses may differ).
- Prometheus and Grafana running with engine metrics scraped (module 110 chapter 03's stack).
- A traffic router capable of weighted routing per model name — options: a service mesh (Istio, Linkerd), a progressive-delivery controller (Argo Rollouts, Flagger), an application-level router (LiteLLM-shape), or a hand-rolled FastAPI shim. Pick one; justify in the deliverable.
- An embedding model reachable for cosine-similarity scoring (local small model, or a hosted service).

## Requirements

Six deliverables.

### Deliverable A — change-type classification and primitive choice

Document the specific change you are rolling. For the actual exercise pick *one*:

- Scheduler flag change (e.g. `max-num-seqs` tweak) — canary primitive.
- Engine version upgrade on same model weights — canary with widened equivalence tolerance.
- Quantisation change on the same model — shadow-rollout gate + canary (do both, label clearly).

State in writing which primitive the change category calls for (from chapter 01's change-type → primitive map), and defend the picked primitive for your specific case.

### Deliverable B — response-diff scorer

A script or service that takes `(primary_response, candidate_response, workload_class)` and returns a bundle of scores:

- `cosine_similarity_via_embedding` (float).
- `refusal_delta` (boolean or probability, per a lightweight classifier or heuristic).
- `schema_conformance` (per-field conformance if structured output; pass/fail for free text).
- `length_delta_percent` (crude but useful).

Each score comes with a configurable threshold. Publish the thresholds you chose and defend each one in 1–2 sentences. If you know your workload well enough, at least one threshold should be *lower* than you'd find in a generic template — the defence is what separates a real contract from a copy-paste.

### Deliverable C — the canary ramp controller

A controller (a Python script run as a long-lived loop, a Flagger/Argo Rollouts config, or a hand-rolled K8s operator — whatever matches your chosen router) that:

- Reads the ramp schedule (steps, dwell-times, promotion gates) from a YAML / JSON config.
- Enacts weight changes at step boundaries by updating the router's weight table.
- Evaluates promotion gates at each dwell boundary — a gate is a boolean combination of (SLO burn-rate alert not firing) ∧ (response-diff scorer in tolerance) ∧ (no new error class).
- On gate fail, halts the ramp; on burn-rate alert, auto-rollbacks (sets candidate weight to 0).
- Emits a `ramp_state` metric (bounded cardinality — `{step, candidate_version}`) so the ramp itself is observable.

Dwell boundaries must respect *tokens served on the candidate*, not just wall-clock — a 15-minute dwell at 10 QPS is not comparable to the same dwell at 10 000 QPS.

### Deliverable D — candidate-scoped SLO burn-rate alerts

Author the Prometheus `PrometheusRule` (or equivalent) with multi-window multi-burn-rate alerts for your product's SLO targets, scoped per `version` label. At minimum: TTFT P95, TPOT P95, ITL P99, cache-hit, cost-per-1M-tokens, error-rate. Use the SRE workbook's burn-rate ratios adjusted for your SLO window (justify any deviations).

Verify the `version` label is bounded — no more than `active_versions + 1 = 2` distinct values in steady state.

Wire the alert route so `page-rollback` alerts hit the ramp controller as an auto-rollback signal (webhook / API call), not a human page.

### Deliverable E — smoke test and warm-up before the first percent

A pre-flight stage that runs before step 1 of the ramp:

- **Warm-up**: replay a short pre-recorded traffic sample (a few minutes of the previous day's requests) against the candidate replicas; verify TTFT median stabilises before allowing live traffic. CUDA graphs captured; prefix cache populated for representative inputs.
- **Smoke-test fixture**: 50+ `(prompt, params, seed)` pairs with expected-equivalent outputs; every fixture scored by Deliverable B above; the ramp only begins if every fixture is in tolerance.

Capture the pre-flight output — warm-up chart, smoke-test fixture results.

### Deliverable F — full promote-and-rollback demonstration

Run the harness end-to-end twice:

- **Promote case:** a candidate that should pass all gates. Full ramp 0 % → 100 %. Capture the ramp controller's state log and dashboard panels per step.
- **Rollback case:** induce a regression that triggers the auto-rollback path. Options: a scheduler flag that spikes TTFT, a candidate with a bad quantisation, a load pattern that preempts. Document the induction method, the alert that fired, the time from first-burn-sample to candidate-weight-0, and the recovery of the primary SLO.

For the rollback case, write the incident ticket as chapter 04 would have you — a one-page RCA artefact with timeline, metric overlays, and a config-diff section (even if the "diff" is just the induction flag).

## Starter guidance

- **Router choice matters for streaming.** Verify the chosen router respects admission-time version assignment and does not re-evaluate mid-stream. Write a small test that opens a 30-second streaming response and changes the router weight mid-flight; confirm the stream completes on the admitted version.
- **Embedding similarity thresholds are your call.** A quick calibration: run the scorer on two primary-vs-primary runs (same model twice); the distribution of cosine scores gives you the noise floor. Set the candidate-vs-primary threshold a non-trivial distance above it.
- **Per-workload-class dwell matters.** If your traffic is 95 % `chat` and 5 % `long-context`, a 15-minute dwell at step 1 (1 %) sees ~0.05 % of long-context — statistically useless. Either extend the dwell for the step, add a per-class minimum-sample gate, or state explicitly that long-context is not covered at step 1.
- **Cardinality discipline.** Never use `request_id`, `tenant_id`, or `prompt_hash` on hot-path histograms. The `version` label is bounded; keep it that way.
- **Document what you can't test.** If your environment can't actually run two model versions, state so and simulate (both routes go to the same backend; "candidate" is a tagged request). Describe what the production harness would additionally do.

## Fallbacks for constrained environments

- **No Kubernetes:** a FastAPI or Envoy shim with weighted routing is enough; the ramp controller is portable.
- **One model only:** simulate by having the candidate route apply a parameter difference (e.g. `temperature`) so responses actually differ; the diff scorer still exercises.
- **No GPU:** CPU-mode inference on a small model (TinyLlama, Phi) is sufficient to exercise the plumbing; latency SLO targets should be realistic for the hardware, not aspirational.
- **No embedding model:** fall back to normalised edit distance plus schema conformance; document the trade-off in Deliverable B.

## Acceptance criteria

1. **Deliverable A** names the change category, the picked primitive, and the justification.
2. **Deliverable B** runs standalone, takes two responses plus a workload-class tag, and returns the score bundle with thresholds. Defended thresholds published.
3. **Deliverable C** executes a configured ramp, respects promotion gates, enacts weight changes atomically, auto-rolls-back on burn-rate alert, and emits its own ramp-state metrics.
4. **Deliverable D** has `version`-scoped alerts, `page-rollback` routing, and verified bounded-cardinality labels.
5. **Deliverable E** completes before step 1 of every ramp run, with warm-up chart and smoke-test fixture results in the artefacts.
6. **Deliverable F** demonstrates one clean promote and one clean rollback; the rollback run has a one-page RCA attached.
7. The harness is reusable — configured by file, not hard-coded to the specific demo change.

## Common failure modes

- **Byte-equal response diff.** Byte-equality never holds; the canary false-alarms. Use the semantic scorer.
- **Fleet-scoped SLO alert.** Candidate's regression is diluted; alert never fires. Scope by `version`.
- **No warm-up.** Cold candidate's first requests burn TTFT; auto-rollback fires on a false-positive. Warm up.
- **Mid-stream weight flip.** The router re-evaluates mid-stream; response interleaves versions. Verify the router.
- **Per-class blindness.** Low-volume classes don't accumulate samples; their regressions never trigger the gate.

## Keep it

The harness, the scorer, the ramp config, the alert rules, the smoke-test fixtures, the two RCA reports. Exercise 04 will reuse the rollback-case RCA as a worked example. Chapter 05's eval-gate contract integrates directly with this harness as its pre-flight gate.
