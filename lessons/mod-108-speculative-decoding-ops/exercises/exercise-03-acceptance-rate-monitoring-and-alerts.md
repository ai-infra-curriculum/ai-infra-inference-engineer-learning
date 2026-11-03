# Exercise 03 — Acceptance-Rate Monitoring and Alerts

**Time:** 3 hours · **Output:** one Prometheus / Grafana (or equivalent) monitoring stack with speculative metrics, one dashboard with four panels, one alert pack with six alerts, one baseline table, one triage runbook

## Goal

Make [Chapter 03](../03-acceptance-rate-monitoring.md) concrete by **standing up the full monitoring stack** for the deployment from exercise 01 (or 02), slicing by tenant / model_pair / workload_class, baselining the pair, and authoring the alerts that trigger the right lifecycle action (chapter 04) or the right peer escalation (chapter 06).

## Problem statement

Exercise 01 produced a running speculative deployment and a per-class acceptance table. Exercise 02 (if run) compared schemes. Neither produced a monitoring stack — if `α` drifts tomorrow, nothing alerts, nothing dashboards, nothing distinguishes "the draft is stale" from "the engine regressed" from "the tenant changed their workload." Fix that today.

The deliverable is a monitoring stack and an alert set shaped for speculative decoding. If your platform already runs Prometheus / Grafana / OpenTelemetry, add the speculative-specific layer. If not, stand up a minimal local stack with Docker Compose or k3d to exercise the shape; the production rollout follows the same pattern.

## Prerequisites

- Chapter 03 of this module read; chapter 04 skimmed (you need the trigger set to know which alerts route where).
- Exercise 01 (or 02) delivery in hand — a running engine with speculative-decoding metrics exposed.
- A Prometheus-compatible scrape target (the engine's `/metrics` endpoint). Verify the engine's current speculative-metric names; vLLM, SGLang, and TRT-LLM each publish a metrics reference.
- A Grafana instance (or equivalent). A text-based alertmanager-style config is also fine for grading.
- Access to representative traffic — the same chat / code / reasoning samples from exercise 01 — plus a way to tag each request's workload class (chapter 03 names the strategies).

## Requirements

Five deliverables.

### Deliverable A — the metric inventory

A table that maps each chapter-03 metric to what your engine exposes vs. what you must synthesise:

```
| Chapter 03 metric name            | Engine metric name (or "synthesise") | Notes |
|-----------------------------------|---------------------------------------|-------|
| spec_tokens_proposed_total        | vllm:spec_decode_num_draft_tokens     | engine-counter |
| spec_tokens_accepted_total        | vllm:spec_decode_num_accepted_tokens  | engine-counter |
| spec_bonus_tokens_total           | synthesise: compute from a request-log span | engine doesn't label bonus separately |
| spec_requests_total               | engine request counter                | available |
| spec_bypassed_total               | synthesise: tag at admission          | engine doesn't surface |
| spec_draft_latency_seconds        | synthesise from span: draft_start - draft_end | or engine-exposed histogram if available |
| spec_verify_latency_seconds       | synthesise from span: verify_start - verify_end | same |
| spec_num_accepted_per_iter        | engine histogram (if exposed) or synthesise | varies |
| spec_k                            | engine gauge (static) or synthesise from config | available |
```

Fill the table against your pinned engine release. Note the gaps.

### Deliverable B — the recording rules

Prometheus recording rules (or equivalent) that compute the derived metrics from chapter 03:

```yaml
groups:
  - name: speculative_decode
    interval: 30s
    rules:
      - record: spec:acceptance_rate_5m
        expr: |
          sum by (model_pair, workload_class, tenant_tier) (
            rate(spec_tokens_accepted_total[5m])
          )
          /
          sum by (model_pair, workload_class, tenant_tier) (
            rate(spec_tokens_proposed_total[5m])
          )

      - record: spec:tokens_per_iter_5m
        expr: |
          sum by (model_pair, workload_class, tenant_tier) (
            rate(spec_tokens_accepted_total[5m]) + rate(spec_bonus_tokens_total[5m])
          )
          /
          sum by (model_pair, workload_class, tenant_tier) (
            rate(spec_iterations_total[5m])
          )

      - record: spec:bypass_rate_5m
        expr: |
          sum by (model_pair, workload_class, tenant_tier) (rate(spec_bypassed_total[5m]))
          /
          sum by (model_pair, workload_class, tenant_tier) (rate(spec_requests_total[5m]))

      - record: spec:draft_overhead_ratio_5m
        expr: |
          sum by (model_pair) (rate(spec_draft_latency_seconds_sum[5m]))
          /
          sum by (model_pair) (rate(spec_verify_latency_seconds_sum[5m]))
```

Adjust label sets to your cardinality strategy (chapter 03 names the top-N tenants + "other" pattern; use `tenant_tier` rather than raw `tenant` for the hot path).

### Deliverable C — the Grafana dashboard

Four panels, as chapter 03 specifies:

1. **Fleet aggregate speculative health.**
   - `spec:acceptance_rate_5m` (fleet-level, with overlays).
   - `spec:tokens_per_iter_5m`.
   - `spec:bypass_rate_5m`.
   - `spec:draft_overhead_ratio_5m`.
   - TPOT P50 / P99 overlay (from engine's request-latency histogram).
2. **Per-workload-class breakdown.**
   - Rows per workload class, each row: acceptance rate, tokens-per-iter, TPOT P99, request rate.
   - Row-sorted by request rate.
3. **Per-tenant table (tiered).**
   - Top-N tenants + "other" bucket.
   - Columns: QPS, α 24h, α delta vs. baseline, TPOT P99, SLO status.
4. **Per-model-pair comparison.**
   - If one model_pair in production, this panel just shows baseline-vs-current. If multiple, overlay for the canary / promotion window.

Export the dashboard JSON (or equivalent) and include it in the deliverable.

### Deliverable D — the alert pack

The six alerts from chapter 03, authored as Prometheus-rule YAML (or equivalent):

1. **Fleet acceptance rate drift** — threshold 0.08 drift warn, 0.15 drift page; 1h sustained.
2. **Per-class acceptance rate collapse** — `α < 0.3` for 15 m above a QPS floor.
3. **Per-tenant SLO breach via speculative regression** — joint signal: TPOT P99 above SLO AND acceptance 0.1 below baseline.
4. **Bypass rate spike** — `spec:bypass_rate_5m > 0.1` for 10 m.
5. **Draft overhead ratio climb** — `spec:draft_overhead_ratio_5m > 0.3` for 1 h.
6. **Post-deploy acceptance regression** — gated: active only in the 24 h after a `(target, draft)` promotion; `α` of current vs. previous pair.

Each alert must include:

- The `expr` with correct label joins.
- The `for` duration.
- The severity (`warn` / `page`).
- An `annotations.runbook_url` pointing to Deliverable E.
- An `annotations.summary` and `annotations.description` the on-call reads.

### Deliverable E — the triage runbook

A one-page per-alert runbook keyed to the chapter-03 "trigger means" + chapter-04 lifecycle actions + chapter-06 peer-boundary escalation:

```
# Alert: fleet-acceptance-rate-drift

## What this means
Fleet-level α has dropped from baseline. One or more of:
  - A pair-promotion happened and the new pair accepts worse (→ ch 04 rollback / Rule 4).
  - A tenant's workload mix shifted (→ ch 04 Trigger 2; investigate via Panel 2 and Panel 3).
  - The draft is aging against drifting traffic (→ ch 04 Trigger 3; retrain request).
  - The engine regressed (→ ch 06 performance peer; verify engine version).

## Immediate response
1. Check Panel 1: is the drift secular (all classes) or in a single class?
2. If single class: check Panel 2 and Panel 3 to attribute to tenant or workload.
3. If secular and a release was deployed in the past 24h: suspect release → rollback.
4. If secular and no release correlates: raise ch 04 Trigger 3 to the fine-tuning peer.

## Who to page
- On-call engineer first.
- Fine-tuning peer if the response is a retrain request.
- Performance peer if the response is engine-side.

## How to confirm resolution
α returns to within baseline ± 0.03 for 24h.
```

One such page per alert; six total.

## Starter guidance

- **Cardinality discipline first.** Chapter 03's cardinality note is the biggest footgun. Use `tenant_tier` not raw `tenant` on the hot path. Prometheus can be ruined by one careless label.
- **Baseline storage.** The deploy-time baseline table (chapter 04 Rule 1) needs a home — a config map, a `baselines.yaml`, a database. Alerts and dashboards both read it. Set this up now; chapter 04's lifecycle depends on it.
- **Workload classification happens at admission.** The engine doesn't know; your admission layer must tag. Chapter 03 names strategies (client header, server-side heuristic). Pick one and apply consistently.
- **Test the alerts.** An alert without an induced-failure test is a guess. Produce the alert condition (shut off the draft, inject a bad workload) and verify the alert fires with the right annotations.
- **Don't over-alert.** If every drift alerts at warn, nothing alerts in practice. Tune thresholds to your fleet's noise floor; use the chapter-03 "suppression" notes.

## Fallbacks for constrained environments

- **No real multi-tenant setup.** Simulate with `workload_class` labels only; the per-tenant table degenerates to a "per class" view. Document the simplification.
- **Engine doesn't expose per-request speculative data.** Add a middleware shim (FastAPI / Starlette in front of the engine's OpenAI-compatible endpoint) that records request-level fields and emits Prometheus histograms. Chapter 03 names this synthesis path.
- **No alertmanager.** A YAML list of alerts with the same shape and a plan to wire into whatever alerting your platform runs is acceptable for grading.
- **Local-only stack.** Docker Compose with Prometheus + Grafana + the engine is enough; production-fleet wire-up is a follow-up.

## Acceptance criteria

1. **Deliverable A** names every chapter-03 metric, maps it to an engine-exposed counter or marks it `synthesise:`, and documents the pinned engine release the mapping was verified against.
2. **Deliverable B** has recording rules that compute all four derived metrics with correct label joins.
3. **Deliverable C** has four panels showing live data from the exercise-01 deployment plus (if applicable) exercise-02.
4. **Deliverable D** has six alerts, each with `expr` + `for` + severity + `annotations`. Each alert tested by inducing the condition (and documented).
5. **Deliverable E** has one page per alert, following the "what / immediate / who / confirm" shape.
6. The cardinality analysis is explicit: how many series the full label set produces, and the strategy (`tenant_tier` + top-N) that keeps it bounded.

## Common failure modes

- **Labelling every metric with `tenant_id`.** Chapter 03's cardinality note — a 10k-tenant fleet with 20 metrics and 10 label combinations is 2M series. Prometheus will cry; the dashboard will time out.
- **No baseline storage.** Alert on "drift from baseline" without a baseline source → always-firing or never-firing.
- **Alert on raw gauges rather than rates.** `spec_tokens_proposed_total` is a counter; alerting on its absolute value fires every restart. Alert on `rate(...)` or the recording-rule-derived metric.
- **Not testing the alerts.** The three triage walks in chapter 03 all involve the alerts firing with the right annotations. If you can't make yours fire, they aren't production-ready.
- **Confusing "no speculative" (speculative disabled on the engine) with "speculative bypassed" (engine on, but request skipped speculative).** Chapter 03 distinguishes; your dashboard should.
- **Over-granular workload classes.** 20 classes with 5 % traffic each → noisy per-class panels. Chapter 03 recommends 5–10 that materially differ.
- **Ignoring the engine-exposed `k`.** If the engine runs dynamic-`k`, your `k`-sweep intuition from exercise 01 is approximate. Chart `k` as a gauge; correlate with `α`.

## Keep it

The dashboard JSON, the alert pack, and the runbook are the fleet's monitoring artefacts for every subsequent `(target, draft)` deployment. Chapter 04's lifecycle depends on these alerts firing in the right conditions. Exercise 04's lifecycle contract references the dashboard URLs. Exercise 05's benchmark re-uses the metric names.
