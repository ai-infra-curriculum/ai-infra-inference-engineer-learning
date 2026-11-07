# Chapter 02 — HPA and KEDA on Inference-Specific Custom Metrics: Signals, Thresholds, Hysteresis

> `HorizontalPodAutoscaler` on `cpu: 70 %` is a safe default for a Flask app. It is wrong for a 70B inference replica. GPU utilisation is wrong too — it reads 100 % during a single prefill *and* during a saturated batch. The right signals are the ones the engine's scheduler emits: queue depth, in-flight requests, prefill-tokens-per-second, and TTFT P99. This chapter wires those signals into the autoscaler and tunes the one knob that matters most on a slow-boot replica: hysteresis.

## Motivation

A platform team ships their 70B replica from chapter 01 with a default HPA:

```yaml
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: {type: Utilization, averageUtilization: 70}
```

Within a week they have three incidents:

1. **A traffic spike at 09:00 drives TTFT P99 from 1.2 s to 11 s.** The HPA sees CPU at 20 % the entire time — the vLLM scheduler is GPU-bound; the Python side barely uses CPU. The HPA never scales up. The pager fires because of the SLO, not because of the HPA.
2. **A traffic dip at 15:00 scales the fleet from 8 replicas to 2 in six minutes.** Each pod took 4 minutes to boot. By 15:30 traffic is back; now the two surviving pods are overloaded, five of the eight pods that were killed had to be re-booted (another four-minute window), and the SLO is breached for 20 minutes.
3. **Someone adds a second HPA metric on `DCGM_FI_DEV_SM_ACTIVE`** to fix issue (1). SM-active reads 1.0 during normal batched traffic, so the fleet is pinned at `maxReplicas: 20` all day. The bill triples.

Every one of these failures is the wrong-signal / wrong-hysteresis story. The right signals are inference-shaped — they move with queue pressure, not with CPU or GPU busyness. The right hysteresis is slow — a replica that takes 4 minutes to boot cannot be scaled down on a 60-second traffic dip. This chapter is how to get both right.

By the end of it you can:

- Name the four inference signals that scale correctly (queue depth, in-flight, prefill-TPS, TTFT P99) and the two signals that scale incorrectly (CPU, GPU utilisation).
- Wire a Prometheus-shape engine metric into the HPA via the Prometheus Adapter or into KEDA via its Prometheus scaler.
- Set scale-up and scale-down behaviour that matches a slow-boot inference replica — fast up, slow down, with a hysteresis window that respects the boot time.
- Reason about multi-metric HPAs — queue depth as the primary, TTFT P99 as the safety net, GPU utilisation only as a *floor* for scale-down.
- Build the alert set that catches a mis-tuned scaler before it causes an SLO breach.

Engine metric names differ per engine and have rotated between releases — the specific series below are confirmed against vLLM's and SGLang's current production-metrics docs, but always cross-check against the engine release your fleet pins. Chapter 03 of module 103 enumerates the full scheduler-metric set this chapter leans on.

## Why CPU and GPU utilisation are wrong signals

### CPU utilisation

The inference container's Python side is doing scheduler bookkeeping, request parsing, and streaming the response — all of which fit in a few percent of a CPU core. The forward pass is on the GPU; the kernel launches cost microseconds of CPU time. A fully saturated vLLM replica may show 10 % CPU; a wholly idle one, 5 %. The *signal-to-noise* ratio is near zero.

CPU-on-the-pod would be the right signal for a web server where the request is CPU-bound. For an LLM replica it is not.

### GPU utilisation (`DCGM_FI_DEV_SM_ACTIVE`)

`SM_ACTIVE` is the fraction of time the GPU's SMs were active. For an LLM replica:

- **Idle:** `SM_ACTIVE = 0` (correct — nothing's happening).
- **One request in flight, decode:** `SM_ACTIVE ≈ 0.9–1.0` on a bandwidth-bound decode kernel. The GPU is "100 % utilised" by the one request.
- **Saturated batch, decode:** `SM_ACTIVE ≈ 0.95–1.0`. The GPU is still at the ceiling — the batch is *producing more tokens per iteration*, but the SM is busy either way.

The signal **cannot distinguish** "one easy request" from "the fleet is overloaded." Any scaler that uses GPU utilisation as a *scale-up* signal will scale too aggressively; any scaler that uses it as a *scale-down* signal will scale too conservatively (SM-active drops only when there is literally no traffic — a situation where the HPA is already at `minReplicas`).

**Where SM-active is useful:** as a *floor* — if `SM_ACTIVE < 0.1` for 15 minutes, there's genuinely nothing happening, and chapter 04's scale-to-near-zero policy can kick in. For reactive scaling it is still the wrong signal.

### GPU memory (`DCGM_FI_DEV_FB_USED`)

Memory is a *budget* for how many concurrent requests fit (KV cache; chapter 104), not a *demand* signal. It is useful as a safety check — *does my replica have HBM headroom?* — but not as a scaler input.

## The four inference signals that scale correctly

### Signal 1 — queue depth (requests waiting to be admitted)

The scheduler holds a queue of requests that arrived but haven't started yet. The queue length is the direct expression of "demand exceeds supply." vLLM exposes it as `vllm:num_requests_waiting` (<https://docs.vllm.ai/en/stable/usage/metrics/>); SGLang as `sglang:num_queue_reqs` (<https://docs.sglang.io/references/production_metrics.html>, requires `--enable-metrics`); TRT-LLM via the Triton metrics for the pending-request queue.

- Rising queue depth → the current replicas can't absorb arrivals → scale up.
- Falling queue depth → there's slack → maybe scale down (with hysteresis).
- Steady-state zero → either the fleet is right-sized, or the fleet is way over-sized; combine with another signal.

Queue depth is *the* primary signal.

### Signal 2 — in-flight requests (`num_requests_running`)

The count of requests currently being decoded on the replica. For a replica with `max-num-seqs=64`, in-flight approaching 64 is "the batch is full and new arrivals will queue." Combined with queue depth, in-flight is the *rate* side of the demand picture (queue depth is the *backlog*).

Fleet-level target: in-flight / max-num-seqs in the rough neighbourhood of 0.7–0.8 at steady state as a starting point, but the right threshold is deployment-specific — it moves with the batch shape, prefill-vs-decode mix, and the SLO, and should be derived empirically. Well below that range the fleet is over-provisioned; much above it, queueing is imminent and TTFT will tail.

### Signal 3 — prefill-tokens-per-second (prefill-TPS)

Prefill is compute-bound and does not share compute with decode on the same tick (chapter 103). The rate at which the replica is processing prefill tokens is a direct expression of ingestion throughput, and the signal rises as new long-context requests arrive. vLLM exposes it via `vllm:prompt_tokens_total` as a rate; most engines emit a similar counter.

Prefill-TPS is especially useful for workloads where request size varies — a chatbot with 100-token prompts has very different scaling than a RAG application with 8K-token prompts. CPU / GPU-utilisation hides that difference; prefill-TPS exposes it.

### Signal 4 — TTFT P99 (the SLO safety net)

TTFT P99 is the SLO signal. If TTFT P99 is above the SLO budget for 5 minutes, scale up *regardless of what the primary signals say* — the user is seeing it. vLLM exposes TTFT as a histogram (`vllm:time_to_first_token_seconds`); SGLang and TRT-LLM expose equivalents.

TTFT P99 is a **lagging** signal — by the time it moves, requests are already slow. Use it as a safety net *in addition to* queue depth, not as the primary. If you only wire TTFT P99, you'll always be scaling in response to pain the user has already felt.

### A summary of the four

| Signal | Primary / Secondary | Direction | Why |
|---|---|---|---|
| Queue depth (`num_requests_waiting`) | Primary | Scale up when > threshold | Leading signal; demand backlog |
| In-flight (`num_requests_running`) | Primary (ratio to `max-num-seqs`) | Scale up when > 0.8 | Rate signal; "almost full" |
| Prefill-TPS (`prompt_tokens_total` rate) | Secondary | Scale up when > per-replica capacity | Captures prompt-size variation |
| TTFT P99 histogram | Safety net | Scale up when > SLO | Lagging; user-visible |

## Prometheus Adapter — HPA on custom metrics

Kubernetes' HPA v2 reads custom metrics from a `custom.metrics.k8s.io` API that an adapter serves. The current standard adapter is `prometheus-adapter` (<https://github.com/kubernetes-sigs/prometheus-adapter>). Install via Helm (`prometheus-community/prometheus-adapter`) and configure a `rules` block that translates a Prometheus series to a custom metric:

```yaml
rules:
- seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    matches: "vllm:num_requests_waiting"
    as: "vllm_num_requests_waiting"
  metricsQuery: 'avg_over_time(<<.Series>>{<<.LabelMatchers>>}[1m])'
```

After the adapter picks that up, you can query the custom-metrics API:

```bash
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/<ns>/pods/*/vllm_num_requests_waiting" \
  | jq .
```

Then wire the HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-llama31-70b
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama31-70b
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric: {name: vllm_num_requests_waiting}
      target:
        type: AverageValue
        averageValue: "5"             # per-replica target: 5 queued requests
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60             # at most 2 pods per minute
    scaleDown:
      stabilizationWindowSeconds: 900 # 15 minutes of hysteresis
      policies:
      - type: Pods
        value: 1
        periodSeconds: 300            # at most 1 pod every 5 minutes
```

The hysteresis is the big-ticket item; the next section defends the specific numbers.

## KEDA — Prometheus scaler as an alternative

KEDA (<https://keda.sh/>) is the ergonomic alternative to Prometheus Adapter + HPA. It wraps the HPA, adds scale-from-zero (critical for chapter 04), and ships Prometheus / Kafka / RabbitMQ / cloud-queue scalers out of the box. For an inference workload a `ScaledObject` looks like:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-llama31-70b
spec:
  scaleTargetRef:
    name: vllm-llama31-70b
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 900                    # 15m scale-down hysteresis
  pollingInterval: 15                    # 15s poll on Prometheus
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:                          # KEDA passes this through to the HPA
        scaleUp:
          stabilizationWindowSeconds: 60
          policies: [{type: Pods, value: 2, periodSeconds: 60}]
        scaleDown:
          stabilizationWindowSeconds: 900
          policies: [{type: Pods, value: 1, periodSeconds: 300}]
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: avg(vllm:num_requests_waiting{deployment="vllm-llama31-70b"})
      threshold: "5"
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket{deployment="vllm-llama31-70b"}[2m])) by (le))
      threshold: "2.0"                   # scale up if TTFT P99 > 2s
```

KEDA's two advantages for inference:

- **Scale-from-zero** — the HPA's `minReplicas: 0` isn't supported; KEDA makes it work. Chapter 04 depends on it.
- **One place to describe multiple triggers** — queue depth as primary and TTFT P99 as safety net is a two-trigger `ScaledObject`. The HPA path works but is uglier.

The Prometheus scaler fields used above (`serverAddress`, `query`, `threshold`) and the `advanced.horizontalPodAutoscalerConfig.behavior` passthrough shape match the current KEDA docs (<https://keda.sh/docs/latest/scalers/prometheus/>, <https://keda.sh/docs/latest/reference/scaledobject-spec/>); cross-check against the pinned KEDA release before adopting verbatim.

## Scale-up vs. scale-down hysteresis

A standard Flask HPA's `scaleUp` and `scaleDown` behaviours are symmetric and fast — the pod boots in seconds, so there's no cost to being wrong. Inference breaks that symmetry twice:

- **Scale-up is slow.** A new replica takes 2–5 minutes to boot (chapter 04). If you under-react, the SLO breaks for minutes. Scale up **aggressively on short windows**.
- **Scale-down is cheap in cost but expensive in blowback.** Killing a replica saves money immediately; booting it back costs 2–5 minutes. If you over-react to a dip, the next spike finds you short. Scale down **conservatively on long windows**.

Operator rule of thumb (verify against your SLO):

```
scaleUp:
  stabilizationWindowSeconds: 60        # 1-minute moving average
  policies: [{type: Pods, value: 2, periodSeconds: 60}]  # 2 pods per minute, max
scaleDown:
  stabilizationWindowSeconds: 900       # 15-minute moving average
  policies: [{type: Pods, value: 1, periodSeconds: 300}]  # 1 pod per 5 min, max
```

Three principles behind those numbers:

1. **Scale-down window ≥ cold-start.** If a replica takes 4 minutes to boot, scale-down's hysteresis must be *longer* than 4 minutes — otherwise a dip removes a replica that will be needed again before it can be re-booted. A 15-minute window on scale-down is a safe starting point for a 2–5 minute cold-start.
2. **Scale-up window = time to react to real demand.** 60 seconds is long enough to filter one-request noise and short enough to catch a sustained spike. On very bursty workloads (sub-minute spikes), consider a shorter window and compensate with a higher queue-depth threshold.
3. **Rate caps on both sides.** Even with hysteresis, cap the per-minute change — scaling by 10 pods in a minute during a brief anomaly over-corrects; scaling down by 5 pods in a minute on a dip makes the subsequent spike worse. "At most 2 up, at most 1 down" is a sane default for a slow-boot fleet.

A deliberate asymmetry — fast-up / slow-down — is the whole shape. If your HPA's `scaleUp` and `scaleDown` have the same `stabilizationWindowSeconds`, you have the Flask-shape, not the inference-shape.

## Multi-metric HPAs — one scaler, many signals

HPA v2 supports multiple metrics; the autoscaler takes the **max** of the computed desired-replica counts. For an inference workload, the useful combination is **primary + safety net**:

```yaml
metrics:
# PRIMARY — queue depth
- type: Pods
  pods:
    metric: {name: vllm_num_requests_waiting}
    target: {type: AverageValue, averageValue: "5"}
# SAFETY NET — TTFT P99 across the deployment
- type: Object
  object:
    describedObject: {apiVersion: v1, kind: Service, name: vllm-llama31-70b}
    metric: {name: vllm_ttft_p99_seconds}
    target: {type: Value, value: "2000m"}       # 2.0 seconds
```

If queue depth drives scale-up, TTFT P99 is quiet (TTFT is downstream of queue depth). If TTFT P99 drives scale-up, something unusual is happening (long prefills, slow GPU, warm-up thrash) and the primary didn't catch it. Either way, scale up.

### Don't add GPU utilisation as a primary

Reiterated for emphasis: `DCGM_FI_DEV_SM_ACTIVE > 0.8` means "GPU is busy," which is the *entire point* of the replica. Using it as a scale-up signal creates run-away growth. Using it as a floor for scale-to-zero (chapter 04) is fine.

### Prefill-TPS as a per-replica-capacity scaler

For heavy-prefill workloads (RAG, long-context chat), the primary scale-up signal is often "the fleet's prefill throughput is near the per-replica ceiling":

```yaml
- type: Pods
  pods:
    metric: {name: vllm_prompt_tokens_per_second}
    target: {type: AverageValue, averageValue: "5000"}
```

The target per-replica number comes from a load test (exercise 02). The threshold *is* the per-replica capacity minus a headroom margin.

## Hysteresis for prefix-cache-heavy workloads

Prefix-cache (chapter 104) is a per-replica asset; a new replica starts cold. If your workload benefits strongly from prefix cache, scaling up with a cold replica incurs a *latency penalty* on its first few hundred requests until the cache fills. Two mitigations:

- **Longer `stabilizationWindowSeconds` on scale-up** — tolerate a small amount of queueing before adding a cold replica. The tradeoff: TTFT P99 may briefly rise during the delay. The win: a sustained spike gets a warm cache sooner than a flashy one gets a cold replica.
- **Dedicated warm pool** (chapter 04) — pre-warm `N_warm` replicas with the common prefix set so scale-up means routing to a pre-warmed replica rather than booting fresh.

The specific tradeoff is workload-dependent; exercise 02 benches on a prefix-heavy workload.

## Alerts the autoscaler must have

A mis-tuned scaler is an SLO breach generator. Three alerts you want firing *before* the breach:

### Alert 1 — scaler pinned at `maxReplicas`

```promql
kube_horizontalpodautoscaler_status_current_replicas{horizontalpodautoscaler="vllm-llama31-70b"}
  == kube_horizontalpodautoscaler_spec_max_replicas{horizontalpodautoscaler="vllm-llama31-70b"}
```

For >15 minutes → the fleet is capped and probably over-saturating; either raise `maxReplicas`, raise the capacity target, or address a downstream constraint (GPU availability, chapter 03 sharing).

### Alert 2 — scaler oscillation (thrashing)

Count scale events in a short window:

```promql
sum(increase(kube_horizontalpodautoscaler_status_desired_replicas{horizontalpodautoscaler="vllm-llama31-70b"}[10m]))
  != 0 and
  rate(kube_horizontalpodautoscaler_status_desired_replicas{horizontalpodautoscaler="vllm-llama31-70b"}[10m]) > 0.5
```

Thrashing means scale-down and scale-up fire in the same window; the hysteresis is too short. The alert pages at design-time; the fix is lengthen scale-down's `stabilizationWindowSeconds`.

### Alert 3 — primary signal and safety-net signal disagree

Queue depth near zero but TTFT P99 above SLO (or vice-versa) means a signal is wrong — either the primary metric isn't reflecting demand (metric broken, scrape stale), or the SLO is being hit by a non-queueing cause (slow prefill on a cold cache, GPU degradation, chapter 05 preemption). Alert and investigate.

### Alert 4 — scaler driven by a stale metric

```promql
time() - prometheus_scrape_last_scrape_timestamp_seconds{job="vllm"} > 120
```

If the scrape is stale, the HPA is reading a frozen number. The HPA won't scale correctly against zero; alert at 2 minutes.

## Common deployment errors

### Error 1 — CPU-utilisation HPA on a GPU workload

Named earlier; the HPA never fires because CPU is near-flat. Fix: remove the CPU trigger, add queue depth.

### Error 2 — symmetric scale-up / scale-down windows

A 2-minute window on both directions on a 4-minute cold-start replica is a thrash machine. Fix: 60 seconds up, 900 seconds down (adjust to your boot time).

### Error 3 — the metric path doesn't resolve

The Prometheus Adapter rule references a series that doesn't exist (metric was renamed in a vLLM upgrade, label changed). The HPA shows `<unknown>` for the metric and sits at `minReplicas` forever. Fix: check the `metricStatus` on the HPA and verify the custom-metrics API returns the series.

### Error 4 — the metric is cumulative, not a rate

A raw `prompt_tokens_total` counter rises monotonically; the HPA on `averageValue: 5000` fires immediately and never scales down. The adapter rule must apply `rate()` or `avg_over_time` on counters. Chapter 02 of module 110 covers Prometheus histogram and counter semantics.

### Error 5 — single-metric HPA with no safety net

Queue depth looks fine; TTFT P99 is 10 s because every request has an 8K prefill. The HPA stays at 2 replicas. Fix: add TTFT P99 as a secondary metric; the HPA's max-of-signals semantics scales up on whichever hits first.

### Error 6 — `minReplicas` lower than the warm pool

Chapter 04 pre-warms replicas against the scaler's `minReplicas`. If `minReplicas: 0` without the warm pool, the first request after idle pays the cold-start. If `minReplicas: 2` while the warm pool is sized for 1, there's an over-provisioning conflict. Fix: coordinate chapter-04's warm pool with the HPA's floor.

## A worked example — rolling the autoscaler onto chapter 01's deployment

Starting from chapter 01's `vllm-llama31-70b` Deployment:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-llama31-70b
  namespace: inference
spec:
  scaleTargetRef:
    name: vllm-llama31-70b
  minReplicaCount: 2
  maxReplicaCount: 20
  cooldownPeriod: 900
  pollingInterval: 15
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60
          policies: [{type: Pods, value: 2, periodSeconds: 60}]
        scaleDown:
          stabilizationWindowSeconds: 900
          policies: [{type: Pods, value: 1, periodSeconds: 300}]
  triggers:
  # PRIMARY — queue depth
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-k8s.monitoring.svc:9090
      query: avg(vllm:num_requests_waiting{deployment="vllm-llama31-70b"})
      threshold: "5"
  # SAFETY NET — TTFT P99
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-k8s.monitoring.svc:9090
      query: histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{deployment="vllm-llama31-70b"}[2m])))
      threshold: "2.0"
  # SECONDARY — in-flight / capacity
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-k8s.monitoring.svc:9090
      query: avg(vllm:num_requests_running{deployment="vllm-llama31-70b"} / 64.0)   # max-num-seqs=64
      threshold: "0.75"
```

Review each number against your workload before shipping; the exercise 02 harness is where you calibrate.

## The boundary to the level-20 peer

This chapter does not teach:

- **Installing Prometheus or Grafana.** Generic observability substrate; `ai-infra-engineer` level 20 or module 110.
- **Writing a Prometheus Adapter Helm values override** at the YAML level divorced from the inference metrics. Level 20.
- **Fundamentals of HPA v2** — what `type: Pods` vs. `type: Object` means at the API shape. Level 20.

This chapter *does* teach: the inference-specific signals, the inference-specific thresholds and hysteresis, the inference-specific alert set, and the pattern for composing multiple triggers against an SLO.

## Summary

- CPU and GPU utilisation are the wrong signals for inference autoscaling. CPU is near-flat; GPU utilisation (`SM_ACTIVE`) cannot distinguish "one easy request" from "fleet saturated."
- The four right signals: **queue depth** (primary, leading), **in-flight / max-num-seqs** (primary, rate), **prefill-TPS** (secondary, for variable prompt-size workloads), **TTFT P99** (safety net, lagging).
- Wire them via Prometheus Adapter → HPA v2, or (preferred) KEDA `ScaledObject` with multiple Prometheus triggers.
- Hysteresis is asymmetric: fast scale-up (60 s window, 2 pods / min), slow scale-down (900 s window, 1 pod / 5 min). The scale-down window must exceed cold-start time.
- Multi-metric HPAs are the right pattern: primary metric for leading, safety-net metric for the SLO.
- Alert on scaler pinned at max, scaler oscillation, primary-vs-safety-net disagreement, and stale metrics.
- Generic Kubernetes HPA concepts are the level-20 peer's; this chapter owns the inference-shaped use.
