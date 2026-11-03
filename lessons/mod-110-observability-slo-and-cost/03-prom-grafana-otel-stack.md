# Chapter 03 — The Prometheus / Grafana / OpenTelemetry Stack for Inference

> The general observability stack is a Prometheus for metrics, a Grafana for dashboards, an OpenTelemetry collector in front of Jaeger / Tempo / an APM for traces, and a log pipeline for forensic. That stack composes with inference, with four modifications: DCGM-exporter is the GPU observability source and must be scraped at a different cadence than the engine; the engine's `/metrics` exposes histograms with distribution-sensitive buckets that Prometheus's defaults don't match; a request is a trace tree with admission / prefill / decode spans; and logs at 10k QPS × 500 tokens / request will burn your logging bill unless you sample. This chapter wires all four.

## Motivation

A team adopts Prometheus Operator in a day: `helm install kube-prometheus-stack`, point it at the engine's `/metrics`, done. Four days later:

- The DCGM-exporter is up but Prometheus isn't scraping it (the `ServiceMonitor` is in the wrong namespace).
- The engine's TTFT histogram has one bucket between 100 ms and 10 s, so TTFT_P95 is uninformative.
- The Grafana dashboard shows a fleet-level number only; the per-workload-class split from chapter 02 doesn't render because the engine metric isn't labelled by class.
- Jaeger has three traces and 90 % of requests are untraced. The one trace that is captured covers only the HTTP handler, not the admission → prefill → decode critical path.
- The log pipeline ingested 2 TB in a day and the cost dashboard shows the logging bill is now 20 % of the GPU bill.

Every one of those failures is a wiring choice this chapter names. By the end of it you can:

- Deploy Prometheus Operator with `ServiceMonitor` / `PodMonitor` / `PrometheusRule` resources for inference workloads.
- Scrape DCGM-exporter and the engine's `/metrics` with correct relabel discipline.
- Wire an OpenTelemetry collector that captures admission → prefill → decode → stream-close spans.
- Author a trace-sampling policy that is head-based for green traffic and tail-based for errors.
- Author a log-sampling policy that survives high QPS.
- Ship four dashboards (fleet, per-class, per-replica, cost) that chapter 02's SLO review reads from.

## The reference stack

The components:

- **Prometheus** for metrics storage, with the **Prometheus Operator** CRDs (`ServiceMonitor`, `PodMonitor`, `PrometheusRule`, `AlertManager`) — <https://github.com/prometheus-operator/prometheus-operator>.
- **Grafana** for dashboards — <https://grafana.com/docs/>.
- **AlertManager** for alert routing — <https://prometheus.io/docs/alerting/latest/alertmanager/>.
- **DCGM-exporter** for GPU metrics — <https://github.com/NVIDIA/dcgm-exporter>.
- **OpenTelemetry Collector** for trace / metric / log ingestion with pipeline transforms — <https://opentelemetry.io/docs/collector/>.
- **Jaeger / Tempo / an APM** as the trace backend — <https://www.jaegertracing.io/>, <https://grafana.com/docs/tempo/latest/>.
- **A log pipeline** — Loki / Elasticsearch / a cloud native log service. The choice matters less than the sampling discipline below.

The `kube-prometheus-stack` Helm chart (<https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack>) bundles Prometheus Operator + Grafana + AlertManager + exporter CRDs; it is the standard starting point on Kubernetes.

<!-- needs-research: the `kube-prometheus-stack` chart has a release-cadence of weeks and its CRD versions evolve. Pin a version in your deployment and verify against its documented upgrade path. -->

## Scraping DCGM-exporter

DCGM-exporter runs as a DaemonSet (installed by the GPU Operator — see module 109, chapter 01). It exposes `/metrics` on a port (usually `9400`) with the `DCGM_FI_*` field set from chapter 01.

A `ServiceMonitor` to scrape it:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: gpu-operator           # or wherever the GPU Operator installed it
  labels:
    release: prometheus              # match the Prometheus Operator's selector
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  endpoints:
  - port: gpu-metrics
    interval: 30s                    # DCGM metrics change slowly; 30s is enough
    scrapeTimeout: 10s
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_node_name]
      targetLabel: node
    - sourceLabels: [__meta_kubernetes_pod_name]
      targetLabel: pod
```

Three disciplines:

- **30-second scrape interval.** DCGM's sampling cadence is typically 1 s internally but you don't need to scrape every second for dashboards. A 30-second interval cuts Prometheus write amplification by 30× with no diagnostic loss. For high-resolution incident forensic you can lower this on demand.
- **Relabel to add `node` and `pod`.** DCGM-exporter emits one series per `(gpu, pod, node)` triple; without the relabel, downstream queries can't correlate GPU to replica.
- **Verify the Prometheus selector matches.** The `labels.release: prometheus` in the `ServiceMonitor` must match the Prometheus CR's `serviceMonitorSelector`. Mis-matched selectors are the #1 reason a `ServiceMonitor` doesn't scrape; `kubectl -n monitoring get prometheus -o yaml` shows the selector.

Confirm scraping with:

```bash
# Should return non-zero samples for DCGM metrics across all GPU nodes:
curl -s "http://prometheus.monitoring:9090/api/v1/query?query=DCGM_FI_DEV_SM_ACTIVE" \
  | jq '.data.result | length'
```

## Scraping the engine's `/metrics`

Modern inference engines (vLLM, SGLang, TensorRT-LLM via Triton) expose Prometheus `/metrics` at a well-known port. Verify the current port and endpoint against the engine's documentation:

- vLLM — `/metrics` on the HTTP server; metrics documented at <https://docs.vllm.ai/en/latest/serving/metrics.html>.
- SGLang — `/metrics` on the HTTP server; metrics documented at <https://docs.sglang.ai/>.
- TRT-LLM via Triton — port `8002` by default; metrics documented at <https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md>.

<!-- needs-research: metric names, label spellings, and histogram bucket defaults rotate across releases. Verify the specific names and bucket layouts against the pinned engine release, not against this chapter's examples. -->

A `PodMonitor` (one target per replica, since each engine replica is its own pod):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: inference-engine
  namespace: inference
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: inference-engine
  podMetricsEndpoints:
  - port: metrics                     # name defined in the engine's Service
    interval: 15s
    scrapeTimeout: 10s
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_label_workload_class]
      targetLabel: workload_class
    - sourceLabels: [__meta_kubernetes_pod_label_model_pair]
      targetLabel: model_pair
    - sourceLabels: [__meta_kubernetes_pod_name]
      targetLabel: replica
    metricRelabelings:
    - sourceLabels: [__name__]
      regex: "vllm:.*"                # keep engine metrics
      action: keep
    - sourceLabels: [tenant_id]       # drop any accidental high-cardinality labels
      action: labeldrop
```

Three disciplines:

- **15-second scrape interval for engine metrics.** The engine updates per-request; 15 s captures distribution shifts without storing too much.
- **Relabel from pod labels.** `workload_class`, `model_pair`, `replica` need to be present on the metric for chapter 02's class-level SLOs to compute. The convention: the pod's labels carry the identity, the relabel maps them into metric labels.
- **`metricRelabelings` drops high-cardinality labels defensively.** Even if the engine starts emitting `tenant_id`, the drop prevents cardinality collapse. Chapter 01's trap 1.

### Fixing the histogram buckets

The engine's default histogram buckets are usually optimised for HTTP latency (10 ms, 50 ms, 100 ms, 500 ms, 1 s, 5 s, 10 s). Inference latency has a different shape:

- TTFT: tens of ms to tens of seconds, log-spaced.
- TPOT: 1 ms to 500 ms, log-spaced.
- ITL: 1 ms to 2 s, log-spaced.
- E2E: 10 ms to 10 minutes (for long-context + reasoning).

If the engine lets you configure the buckets (vLLM exposes a few config knobs for this; verify current flags), set them. If not, the next-best thing is to compute the derived SLI from the raw spans in OTel and export a *second* histogram with the right buckets from your middleware.

Example sensible bucket sets (log-spaced, 15 – 20 buckets):

```
TTFT buckets (seconds):
  0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 10, 20, 40, 80

TPOT buckets (seconds):
  0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1

ITL buckets (seconds):
  0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3
```

A histogram's P95 at bucket resolution `b` is accurate to within one bucket boundary; log-spaced 15-bucket histograms give ~15 % resolution at any quantile, which is usually enough for SLO alerting.

## Recording rules — pre-computing SLIs

The SLI formulas from chapter 02 should not be recomputed on every dashboard load or alert evaluation. Prometheus's recording-rule facility (`PrometheusRule` CRD) pre-computes them on a schedule:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: inference-slis
  namespace: inference
  labels:
    release: prometheus
spec:
  groups:
  - name: inference_slis
    interval: 30s
    rules:
      # TTFT P95 per class
      - record: inference:ttft_p95_seconds:class
        expr: |
          histogram_quantile(0.95,
            sum by (le, workload_class) (
              rate(vllm:time_to_first_token_seconds_bucket[5m])
            )
          )

      # TPOT P95 per class
      - record: inference:tpot_p95_seconds:class
        expr: |
          histogram_quantile(0.95,
            sum by (le, workload_class) (
              rate(vllm:time_per_output_token_seconds_bucket[5m])
            )
          )

      # Availability — tokens-served / tokens-requested (approximate)
      - record: inference:availability_ratio:class
        expr: |
          sum by (workload_class) (rate(tokens_streamed_total[5m]))
          / clamp_min(
            sum by (workload_class) (rate(tokens_requested_total[5m])),
            1
          )

      # Cache-hit rate per class
      - record: inference:cache_hit_rate:class
        expr: |
          sum by (workload_class) (rate(cache_hit_tokens_total[5m]))
          / clamp_min(
            sum by (workload_class) (rate(cache_attempt_tokens_total[5m])),
            1
          )

      # Preemption rate (events per second) per replica
      - record: inference:preemption_rate:replica
        expr: |
          sum by (replica) (rate(scheduler_preemptions_total[5m]))
```

<!-- needs-research: the exact metric names (`vllm:time_to_first_token_seconds_bucket`, `tokens_streamed_total`, `cache_hit_tokens_total`) are illustrative — verify against your engine's current `/metrics` output. If the engine doesn't expose a metric you need, synthesise it in a middleware shim as described below. -->

Three disciplines:

- **Pre-compute SLIs; alerts and dashboards query the recording-rule series.** Fast queries, consistent definitions, one source of truth.
- **`clamp_min(..., 1)` the denominator.** When traffic is zero on a cold Sunday morning, you divide by zero and get a `NaN`; `clamp_min` makes the result zero, not an error.
- **One naming scheme.** `inference:<slo>:<grouping>` with colons; the convention is a Prometheus-wide `metric_name:aggregation:label_grouping` shape.

## The OpenTelemetry collector — the trace spine

Metrics tell you *that* something is slow; traces tell you *where* in a specific request. For inference, the trace tree you want:

```
client_request                        [span: HTTP server]
  └── admit                           [span: API gateway / middleware]
       └── route_to_replica           [span: load balancer]
            └── engine.request        [span: engine request lifecycle]
                 ├── queue            [span: scheduler queue wait]
                 ├── prefill          [span: prefill forward pass]
                 ├── first_token      [span: time of first-emitted-token]
                 ├── decode_steady    [span: decode loop; may have sub-events per chunk]
                 └── stream_close     [span: HTTP response close]
```

The ingress to this trace tree is the client's HTTP request; the OTel SDK auto-instruments the HTTP framework for the top-level span. The engine spans must be *inside* the engine — vLLM and SGLang have OTel hooks (verify current status). Where the engine does not instrument natively, you add a middleware shim:

```python
# Middleware example — FastAPI in front of an OpenAI-compatible engine
from opentelemetry import trace
from opentelemetry.propagate import extract

tracer = trace.get_tracer("inference-middleware")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    ctx = extract(request.headers)                          # carry upstream trace
    with tracer.start_as_current_span("admit", context=ctx) as admit_span:
        workload_class = classify(request)
        admit_span.set_attribute("workload_class", workload_class)

    with tracer.start_as_current_span("engine.request") as req_span:
        req_span.set_attribute("workload_class", workload_class)
        req_span.set_attribute("model", request.model)

        with tracer.start_as_current_span("queue") as q_span:
            schedule = await engine.admit(request)
            q_span.set_attribute("queue_depth_at_admit", schedule.queue_depth)

        # engine emits sub-spans for prefill / first_token / decode via its own SDK
        async for chunk in engine.stream(schedule):
            yield chunk
```

The OTel Collector sits in the data path, buffers / batches / samples / forwards:

```yaml
# otel-collector-config.yaml (illustrative)
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors-and-slow
        type: and
        and:
          and_sub_policy:
            - name: has-error
              type: status_code
              status_code:
                status_codes: [ERROR]
      - name: slow-ttft
        type: latency
        latency:
          threshold_ms: 1000
      - name: sample-rest
        type: probabilistic
        probabilistic:
          sampling_percentage: 1

exporters:
  otlp/tempo:
    endpoint: tempo.monitoring:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, tail_sampling]
      exporters: [otlp/tempo]
```

The tail-sampling policies:

- **errors-and-slow** — any trace whose root span errored is kept.
- **slow-ttft** — any trace with TTFT > 1 s is kept.
- **sample-rest** — 1 % of everything else is kept (adjust for your traffic volume and forensic needs).

This head-plus-tail pattern is the standard trade-off: full green traffic is too expensive to store; keeping all red / slow traffic gives you forensic coverage where it matters.

## Trace sampling at high QPS — the budget math

A fleet serving 10k QPS, with each trace averaging 20 KB (admission + engine + N chunk spans), produces 200 MB/s of trace data. Over a month that is 500 TB. A real trace backend will cost tens of thousands of dollars a month at that volume if you keep every trace.

The sampling budget:

```
qps = 10000
avg_trace_bytes = 20_000
trace_backend_target_cost = $5000/month
cost_per_tb_month = $50          # illustrative; your provider varies
monthly_budget_bytes = (5000 / 50) × 1 TB = 100 TB

seconds_per_month = 86400 × 30 ≈ 2.6M
monthly_bytes_unsampled = qps × avg_trace_bytes × seconds_per_month
                        ≈ 10000 × 20000 × 2.6M = 5.2 × 10^14 bytes = 520 TB

sampling_rate_target = 100 TB / 520 TB ≈ 0.19 → 20% sampling
```

Three disciplines:

- **Keep 100 % of error / slow traces.** The errors are a tiny fraction of volume (`<1 %`) and 100 % of forensic value.
- **Keep 1 – 5 % of green traffic.** Enough for exemplar / dashboard overlay; cheap enough to afford.
- **Verify the budget with a dry-run.** Before enabling, run the collector with sampling at the target rate against mirrored traffic for a day. Measure actual egress; adjust.

## Logs — sampling at high QPS

The log volume math is similar. Per request:

- 1 access log line (user id, prompt length, output length, latency).
- 0 to N decode-event log lines (if the engine logs per-chunk) — never log per-chunk in production.
- 0 to N error log lines (keep these in full).

At 10k QPS × 300 bytes/log = 3 MB/s = 7.5 TB/month per 1 log/request. Three disciplines:

- **No per-chunk logging in production.** The volume is unmanageable; the signal is in metrics and traces.
- **Sample access logs at a known rate.** 10 % head-sampling for 200-OK, 100 % for 4xx / 5xx. Keep the sample rate per workload class so a 1 %-traffic class isn't drowned out.
- **Log the request-id and the trace-id.** Both. The log is a pointer to the trace; the trace is the forensic artefact.

## The four dashboards

Four dashboards is the minimum useful set. More than four and the on-call doesn't know which to open.

### Dashboard 1 — Fleet overview

The one-screen SLO status.

- Per-class TTFT P95 against target (five rows).
- Per-class TPOT P95 against target (five rows).
- Fleet-level availability (one number per class).
- Fleet-level cost/1M tokens (one number per class, from chapter 04).
- Cache-hit rate per class (overlay on cost panel).
- Burn-rate per SLO per class (traffic-light table).

Audience: on-call engineer at a glance, SLO review once a week.

### Dashboard 2 — Per-workload-class deep dive

Pick a class and drill into:

- TTFT decomposition: queue + prefill + first_decode (stacked area).
- TPOT distribution (heatmap over time).
- ITL distribution (heatmap over time).
- Preemption rate.
- Admission reject rate.
- Request rate.
- Error rate (5xx by sub-category).

Audience: on-call engineer during an active alert; SLO review when a class is burning.

### Dashboard 3 — Per-replica diagnostic

One row per replica, with:

- Replica identity (pod, node, model).
- In-flight sequences, queue depth, prefill-TPS, decode-TPS.
- DCGM: SM_ACTIVE, FB_USED, TENSOR_ACTIVE, DRAM_ACTIVE, NVLink traffic.
- NCCL time per collective (if TP > 1).
- Container restarts, cold-start events.

Audience: diagnosing a single-replica regression; isolating a bad GPU.

### Dashboard 4 — Cost

See chapter 04 for the cost model; the dashboard surfaces:

- $ per 1M input tokens and output tokens (per class, over time).
- Utilisation % (per replica).
- Idle GPU-hours (fleet-level).
- Cache-hit credit $ saved.
- Speculative credit $ saved (if applicable).

Audience: platform owner, finance partner, product partner.

Dashboards 1, 2, 3 are engineer-facing; dashboard 4 is cross-functional. The reasoning behind dashboard 4 is chapter 06 — tying the inference numbers to the business numbers.

## Common failure modes

- **ServiceMonitor doesn't match Prometheus selector.** Metrics are scraped by nothing. The `labels` on the `ServiceMonitor` must match `prometheus.spec.serviceMonitorSelector`.
- **Metric labelled by tenant.** 10k tenants × N metrics explodes series. Label by `tenant_tier` + top-N; drop `tenant_id` at relabel time (chapter 01 trap 1).
- **Default histogram buckets.** `histogram_quantile` returns `NaN` or a bucket-boundary value, which looks like a real number but isn't. Always configure buckets per metric.
- **Head-sampled 10 % traces, zero errors captured.** The errors are 0.1 % of traffic; a 10 % head sample catches 0.01 % of errors. Use tail-based sampling for errors / slow.
- **Prometheus Operator CRs scraped against the Prometheus's `probeSelector` instead of `serviceMonitorSelector`.** The selector names are easy to confuse; verify against `kubectl describe prometheus`.
- **Alert rules in the wrong namespace.** Prometheus Operator's `PrometheusRule` resources need to match the Prometheus's `ruleSelector`. Same confusion as `ServiceMonitor`.
- **Dashboard queries that aren't SLO-aligned.** The dashboard says "latency = 480 ms" and the SLO is "P95 < 500 ms" — but the dashboard is showing the average, not P95. Dashboard panels should name the quantile they show.
- **No cardinality monitor.** The one metric with an unbounded label will one day take down the Prometheus. Alert on `prometheus_tsdb_head_series > budget`.

## Summary

- `kube-prometheus-stack` is a reasonable baseline; the inference-specific wiring is the `ServiceMonitor` / `PodMonitor` / `PrometheusRule` CRs on top.
- Scrape DCGM-exporter at 30 s, engine `/metrics` at 15 s, with relabels that preserve `workload_class` / `model_pair` / `replica`.
- Fix histogram buckets. Default HTTP buckets lose signal for inference latencies.
- Pre-compute SLIs as recording rules. Dashboards and alerts read the pre-computed series.
- OTel collector in the trace path, with tail-based sampling for errors and slow traces plus head sampling for green.
- Log sampling at known rates per class; no per-chunk logs; always carry trace-id in logs.
- Four dashboards: fleet, per-class, per-replica, cost. More than four and nobody reads any.
- Cardinality discipline is the recurring tax. Trap 1 (tenant), trap 2 (request-id), trap 3 (ad-hoc model name) are the three that always catch new teams.
