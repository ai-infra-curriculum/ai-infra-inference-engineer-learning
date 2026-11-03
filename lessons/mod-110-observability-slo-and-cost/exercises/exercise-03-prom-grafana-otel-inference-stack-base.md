# Exercise 03 — Prometheus / Grafana / OTel Inference Stack — Base

**Time:** 6 hours · **Output:** reproducible manifests for a working Prom + Grafana + OTel stack, four dashboards, a cardinality analysis, a stack-health check-list

## Goal

Make [Chapter 03](../03-prom-grafana-otel-stack.md) concrete by **standing up the full stack** end-to-end: Prometheus Operator scraping DCGM-exporter and the inference engine, OpenTelemetry Collector in the trace path with head + tail sampling, Grafana with four dashboards (fleet, per-class, per-replica, cost), and a documented health check-list that confirms the stack works before a production rollout.

## Problem statement

Exercises 01 and 02 produced metrics, traces, and SLO definitions in isolation. This exercise composes them into the stack your fleet will run every day. The rollout includes DCGM-exporter wiring (chapter 01's GPU-side axes), the engine's `/metrics` with correct relabel discipline, OpenTelemetry Collector with sampling policies, and the four dashboards that render the resulting data.

By the end of this exercise an on-call engineer has a one-stop URL to open during a page.

## Prerequisites

- Chapter 03 of this module read.
- A Kubernetes cluster with GPU nodes (or a local k3d / kind cluster plus a mock engine; the manifests should work on both).
- The NVIDIA GPU Operator installed (module 109 exercise 01 — if not done, follow the GPU Operator quick-start: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/>).
- At least one inference engine replica running and exposing `/metrics`.
- Exercise 01's middleware shim available (if your engine needs synthesis).
- Exercise 02's SLO documents in draft (the Deliverable B PromRule needs their targets).

## Requirements

Five deliverables.

### Deliverable A — the stack manifests

Reproducible Kubernetes manifests (Helm values, kustomize, or raw YAML) for:

```
monitoring/
  prometheus/
    values.yaml             # kube-prometheus-stack Helm values
    service-monitors/
      dcgm-exporter.yaml
      engine.yaml
    prometheus-rules/
      inference-slis.yaml   # recording rules
      inference-slos.yaml   # alerts from exercise 02
  grafana/
    datasources.yaml
    dashboards/
      01-fleet.json
      02-per-class.json
      03-per-replica.json
      04-cost.json
  otel-collector/
    config.yaml             # receivers, processors (batch + tail_sampling), exporters
    deployment.yaml         # Collector pods
```

Three disciplines:

- Manifests must be version-pinned. `kube-prometheus-stack` version, Grafana version, OTel Collector version — all explicit.
- Dashboards are JSON files in git, not clicked-together UI. Grafana's JSON model (<https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/>) is the right format.
- Namespaces, labels, and selectors are consistent. The `ServiceMonitor.labels.release` must match the Prometheus CR's selector; the exercise fails silently if they don't.

### Deliverable B — the scrape config

Working `ServiceMonitor` / `PodMonitor` resources for DCGM and the engine, as in chapter 03. Confirm:

```bash
# DCGM scrape health:
curl -s "http://prometheus:9090/api/v1/query?query=up{job='dcgm-exporter'}" | jq .
# Expect: every GPU node reports up=1.

# Engine scrape health:
curl -s "http://prometheus:9090/api/v1/query?query=up{job='inference-engine'}" | jq .
# Expect: every replica reports up=1.

# Representative DCGM metric:
curl -s "http://prometheus:9090/api/v1/query?query=DCGM_FI_DEV_SM_ACTIVE" | jq '.data.result | length'
# Expect: one series per GPU across the fleet.

# Representative engine latency histogram:
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(inference_ttft_seconds_bucket[5m])) by (le, workload_class)" | jq .
# Expect: non-empty result with the workload_class labels present.
```

Document each check in a markdown table (Deliverable E).

### Deliverable C — the OTel Collector

A Collector config with the chapter-03 processors: `batch` plus `tail_sampling` with policies for errors, slow-TTFT (above a per-class threshold), and head-sampled green traffic.

```yaml
processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: slow-ttft-chat
        type: and
        and:
          and_sub_policy:
            - name: class-chat
              type: string_attribute
              string_attribute:
                key: workload_class
                values: [chat]
            - name: ttft-over-1s
              type: latency
              latency:
                threshold_ms: 1000
      - name: slow-ttft-long
        type: and
        and:
          and_sub_policy:
            - name: class-long
              type: string_attribute
              string_attribute:
                key: workload_class
                values: [long_context]
            - name: ttft-over-20s
              type: latency
              latency:
                threshold_ms: 20000
      - name: sample-rest
        type: probabilistic
        probabilistic:
          sampling_percentage: 1
```

Prove it works:

- Send 1000 requests, 20 of which exceed the per-class slow threshold, and verify ~20 + ~(980 / 100) ≈ 30 traces land in Tempo / Jaeger.
- Send 10 errored requests and verify 10 traces land in Tempo / Jaeger.

### Deliverable D — the four dashboards

The dashboard set from chapter 03, as JSON:

1. **Fleet overview** — per-class TTFT_P95 vs. target, TPOT_P95 vs. target, availability, cost / 1M tokens, burn-rate traffic lights.
2. **Per-workload-class deep dive** — TTFT decomposition (queue + prefill + first_decode stacked), TPOT histogram, ITL heatmap, preemption rate, admission reject, request rate.
3. **Per-replica diagnostic** — one row per replica with in-flight / queue / prefill-TPS / decode-TPS / DCGM (SM_ACTIVE, FB_USED, TENSOR_ACTIVE, DRAM_ACTIVE, NVLink).
4. **Cost** — (stub to be filled by exercise 04) `$ / 1M input tokens`, `$ / 1M output tokens`, per class; utilisation %; idle hours; cache credit $; speculative credit $.

Dashboard 4 can be a stub until exercise 04 lands; dashboards 1 – 3 should render live data.

### Deliverable E — the health check-list

A one-page markdown "stack health" check-list to run after deploy, after upgrade, and once a week:

```
# Stack health check-list

## Prometheus
- [ ] `up{job="prometheus"} == 1` — Prometheus itself is up.
- [ ] `prometheus_tsdb_head_series < <budget>` — cardinality under budget.
- [ ] Top-5 label cardinality via `/api/v1/status/tsdb`: no label with > 1e4 unique values.
- [ ] `rate(prometheus_rule_evaluation_failures_total[5m]) == 0` — recording rules healthy.

## Exporter scrape
- [ ] Every GPU node reports `up{job="dcgm-exporter"} == 1`.
- [ ] Every engine replica reports `up{job="inference-engine"} == 1`.
- [ ] Representative DCGM metric returns one series per GPU.
- [ ] Representative engine metric returns a workload_class-labelled series.

## SLI recording rules
- [ ] `inference:ttft_p95_seconds:class` has a value per class.
- [ ] `inference:availability_ratio:class` returns between 0 and 1.
- [ ] `inference:cache_hit_rate:class` returns between 0 and 1.

## Alerts
- [ ] Alertmanager has routes for each severity (page, ticket).
- [ ] Representative page alert fires in a load-test induced regression.

## OTel
- [ ] Collector processes expected traces/sec; dropped-traces counter = 0.
- [ ] Jaeger/Tempo shows traces; tail-sampled errors visible; head-sampled green visible.
- [ ] Trace tree shape matches the chapter-01 spec (admit → engine.request → queue / prefill / first_token / decode).

## Dashboards
- [ ] Fleet dashboard renders without errors.
- [ ] Per-class dashboard renders for every class in the SLO document.
- [ ] Per-replica dashboard renders for every replica.
- [ ] No panel shows "no data" for a running class/replica.

## Cardinality
- [ ] No metric has `tenant_id` as a label.
- [ ] No metric has `request_id` as a label.
- [ ] `model_name` values match an allow-list.
```

## Starter guidance

- **Start with `kube-prometheus-stack`.** The Helm chart gives you Prometheus Operator + Grafana + AlertManager + default exporters out of the box. Then layer the inference-specific scrape configs on top.
- **Namespaces matter.** Decide where Prometheus lives (`monitoring`), where GPU Operator / DCGM live (`gpu-operator`), where the engine lives (`inference`). Service monitors cross namespaces via `namespaceSelector`. Document the layout.
- **Dashboards in git first.** Build a dashboard in the Grafana UI, export the JSON, commit it. Don't rely on the UI as the source of truth.
- **Pre-compute SLIs.** Every dashboard and alert should query a recording-rule series, not a raw histogram bucket. Faster, consistent.
- **Verify selectors.** `kubectl describe prometheus` shows the `serviceMonitorSelector`; your `ServiceMonitor.labels` must match. #1 silent failure.

## Fallbacks for constrained environments

- **No GPU cluster.** Mock DCGM-exporter with a static metric generator that emits a few `DCGM_FI_*` series; the scrape and dashboard wiring is identical. Document the mock.
- **No real engine.** A mock engine that emits chapter-01 histograms with synthetic distributions is sufficient; swap in the real engine when available.
- **No cloud Tempo.** Grafana Tempo in Docker is the easy path; if even that is heavy, Jaeger all-in-one works as an OTel backend.
- **Local-only stack.** Docker Compose with the full stack (Prometheus, Grafana, OTel Collector, Jaeger, mock engine, DCGM mock) is a valid deliverable; the K8s manifests are a follow-up but should be provided as design.

## Acceptance criteria

1. **Deliverable A** has reproducible manifests with pinned versions and dashboards-as-JSON in git.
2. **Deliverable B** has working scrape with the four verification queries returning expected results.
3. **Deliverable C** has a Collector config with tail-sampling policies that keep errors + slow traffic and head-sample green; verified by induced trace counts.
4. **Deliverable D** has dashboards 1 – 3 rendering live data; dashboard 4 is a stub with correct panels but may lack cost data until exercise 04.
5. **Deliverable E** is a one-page check-list runnable in < 15 minutes.
6. Cardinality analysis explicitly: `prometheus_tsdb_head_series` reported at a sensible number (expect low thousands for a small fleet, under tens of thousands for a mid-size fleet). No label with > 1e4 unique values.

## Common failure modes

- **ServiceMonitor doesn't match Prometheus selector.** Metric endpoints reachable from the pod but no series in Prometheus. Verify labels on both sides.
- **Dashboards query raw buckets instead of recording rules.** Slow dashboards; inconsistent SLI definitions. Query the recording-rule series.
- **OTel head sampling at 100 %.** Volume explodes; Tempo / Jaeger OOMs. Move to tail-based for the error / slow traces plus 1 – 5 % head for green.
- **Cardinality monitoring absent.** The one careless label will cause a Prometheus OOM in production without a smoke-alarm. Add `prometheus_tsdb_head_series` to the fleet overview with a budget.
- **Alerts fire into silence.** AlertManager configured but no receiver; all pages go to `/dev/null`. Verify end-to-end by inducing a known regression.
- **Dashboard panel titles don't name the quantile.** "Latency" on a panel that shows P95 is misleading. Title every panel with the full metric name.
- **Missing per-replica dashboard.** A fleet-wide view hides a single-replica regression. The per-replica dashboard is where you catch a bad GPU.

## Keep it

The manifests, dashboards, Collector config, and health check-list are the inference team's observability substrate. Every subsequent exercise references this stack: exercise 04's cost model fills in dashboard 4; exercise 05's Pareto sweep reads from the metrics; exercise 06's memos cite dashboard URLs. Keep the manifests under version control with a changelog.
