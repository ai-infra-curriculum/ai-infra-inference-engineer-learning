# Exercise 02 — HPA on Queue Depth and TTFT P99

**Time:** 4 hours · **Output:** one Prometheus Adapter (or KEDA) wired to engine metrics, one multi-trigger HPA / `ScaledObject`, one load-driven scale-up/scale-down trace, one hysteresis tuning note, one alert pack, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 02](../02-hpa-keda-on-inference-metrics.md) concrete by **wiring a Prometheus-Adapter or KEDA Prometheus scaler** against the engine's queue-depth and TTFT-P99 metrics, authoring a multi-trigger HPA / `ScaledObject` with asymmetric scale-up / scale-down behaviour, and proving on a scripted traffic burst that the scaler responds the way chapter 02 claims — fast up, slow down, no thrash on a dip.

## Problem statement

The chapter is explicit about the failure mode a default HPA creates on a slow-boot inference replica. The exercise is where you make the discipline real on the fleet you serve: pick the right engine metrics, resolve the custom-metrics path, set the right thresholds and behaviours, and verify the scaler against a scripted burst that resembles a 09:00 traffic shape — ramp-up, sustained, dip, re-spike. The deliverable is the thing you hand to the next on-call so they can look at the dashboard at 02 AM and know the scaler will not thrash on a dip.

## Prerequisites

- Chapter 02 of this module read. Chapters 01 and 04 skimmed (you need the substrate assumptions and the cold-start budget that sets scale-down hysteresis).
- **Exercise 01 complete** — the GPU Operator, DCGM, and a `kube-prometheus-stack` (or equivalent Prometheus) are already running. The scaler reads from that Prometheus.
- A running inference deployment emitting Prometheus-shape metrics — vLLM, SGLang, or Triton + TRT-LLM. If you don't have one from exercise 01 of module 102, stand up a vLLM `vllm-llama31-70b` Deployment from chapter 01's worked example, scaled down to whatever hardware you have (an 8B replica on a single H100 / L4 is fine; the methodology is the same).
- Cluster-admin or permissions to install Prometheus Adapter or KEDA.
- A load driver — `vllm bench serve`, a few `httpx` scripts, or `k6` all work. Scripted enough to produce a repeatable burst shape.
- The engine release's production-metrics docs open — for vLLM, <https://docs.vllm.ai/en/stable/usage/metrics/>; for SGLang, <https://docs.sglang.io/references/production_metrics.html>; for Triton, the Triton metrics reference. Metric names rotate between releases; verify against your pinned release.

## Requirements

Six deliverables.

### Deliverable A — the metric inventory

A table that names the chapter-02 signals against what your engine actually exposes, resolved on your pinned engine release:

```
| Chapter-02 signal       | Engine metric (as-emitted)                        | Prom type   | Notes |
|-------------------------|---------------------------------------------------|-------------|-------|
| Queue depth             | vllm:num_requests_waiting                         | gauge       | primary; averaged over 1m in adapter |
| In-flight               | vllm:num_requests_running                         | gauge       | divide by max-num-seqs for ratio |
| Prefill-TPS             | vllm:prompt_tokens_total                          | counter     | needs rate(...) in adapter query |
| TTFT P99                | vllm:time_to_first_token_seconds_bucket           | histogram   | histogram_quantile(0.99, ...) |
| GPU SM-active (floor)   | DCGM_FI_DEV_SM_ACTIVE                             | gauge       | NOT a primary; chapter-02 warns |
```

Fill with the *exact* series names your engine's `/metrics` endpoint returns. If a signal isn't natively emitted, note "synthesise via middleware" and flag it in Deliverable F.

### Deliverable B — the custom-metrics path

Pick **one** path and resolve it end-to-end:

**Option 1 — Prometheus Adapter.** Install `prometheus-community/prometheus-adapter` via helm. Configure a `rules` block matching chapter 02's shape; at minimum, rules for queue depth and TTFT P99:

```yaml
rules:
- seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
  resources: {overrides: {namespace: {resource: "namespace"}, pod: {resource: "pod"}}}
  name: {matches: "vllm:num_requests_waiting", as: "vllm_num_requests_waiting"}
  metricsQuery: 'avg_over_time(<<.Series>>{<<.LabelMatchers>>}[1m])'

- seriesQuery: 'vllm:time_to_first_token_seconds_bucket{namespace!="",pod!=""}'
  resources: {overrides: {namespace: {resource: "namespace"}, pod: {resource: "pod"}}}
  name: {matches: "vllm:time_to_first_token_seconds_bucket", as: "vllm_ttft_p99_seconds"}
  metricsQuery: 'histogram_quantile(0.99, sum by (le) (rate(<<.Series>>{<<.LabelMatchers>>}[2m])))'
```

Verify the custom-metrics API returns the series:

```bash
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/<ns>/pods/*/vllm_num_requests_waiting" | jq .
```

**Option 2 — KEDA.** Install KEDA via helm, author a `ScaledObject` with Prometheus triggers (shape in chapter 02's "A worked example"). KEDA bypasses the Adapter for this purpose — it talks to Prometheus directly.

Attach to the deliverable:

- Install commands with pinned chart versions.
- The adapter `rules` block or the `ScaledObject` manifest.
- Evidence the custom-metrics path resolves — a `kubectl get --raw` output or a KEDA `TriggerAuthentication` / `ScaledObject` status showing the metric read.

### Deliverable C — the HPA / `ScaledObject`

Author the multi-trigger scaler against the deployment from Prerequisites, using chapter 02's "worked example" as the template. Required elements:

- **Primary trigger:** queue depth, average-value target per replica (e.g. 5 waiting requests).
- **Safety-net trigger:** TTFT P99, scalar target (e.g. 2.0 seconds).
- **Secondary trigger:** in-flight-to-capacity ratio (e.g. 0.75 against your `max-num-seqs`).
- **Scale-up behaviour:** short stabilization window (60 s), low rate cap (≤2 pods / minute).
- **Scale-down behaviour:** long stabilization window (≥ cold-start time; 900 s starting point from chapter 02), low rate cap (1 pod / 5 min).
- `minReplicas` set with chapter 04's warm-pool consideration in mind (even if exercise 04 isn't done yet, don't go below 1 for a large-model deploy).
- `maxReplicas` sized to your fleet's upper bound.

Attach the full manifest (HPA v2 or KEDA `ScaledObject`) and the `kubectl get hpa <name> -o yaml` dump showing the metrics resolving (not `<unknown>`).

### Deliverable D — the scripted-traffic scale-up / scale-down trace

Run a repeatable load profile against the deployment and record the scaler's response. Required shape:

```
Phase 1 — baseline    (5 min at base RPS; observe steady-state replica count)
Phase 2 — ramp        (3 min, RPS scales 1× → 4×)
Phase 3 — sustained   (10 min at 4× RPS; observe scale-up to max or to capacity)
Phase 4 — dip         (2 min at 1× RPS; observe NO scale-down kicking in — hysteresis)
Phase 5 — re-spike    (5 min at 4× RPS; observe NO new cold-starts because fleet retained capacity)
Phase 6 — drain       (5 min at ~0 RPS; observe slow scale-down back to minReplicas)
```

For each phase, record (as a time-series or table):

- Observed engine queue depth (`num_requests_waiting`).
- Observed TTFT P99 at the edge.
- HPA's `currentReplicas` and `desiredReplicas`.
- Scale-up / scale-down events (from `kubectl describe hpa` or `kubectl get events`).

Target behaviour to confirm:

- **Phase 2 → 3 transition:** HPA desired climbs within ~60 seconds of queue depth crossing the threshold; new pods are scheduled (even if slow to boot).
- **Phase 4 (dip):** HPA desired holds steady — the stabilization window on scale-down masks the dip.
- **Phase 5 (re-spike):** the fleet absorbs the spike without new cold-starts because capacity was held through the dip.
- **Phase 6 (drain):** HPA desired slowly drops; each step waits ≥ the `periodSeconds` cap.

Attach a screenshot (or exported CSV + plot) of the time-series and a one-paragraph read showing the right-shape behaviour.

### Deliverable E — the hysteresis tuning note

A short (one-page) note explaining the numbers you chose and why — cold-start time, scale-up / scale-down window, rate caps. Expected content:

```
# Scaler Hysteresis Tuning — <deployment>

## Inputs
- Measured cold-start (from exercise 04 or a one-shot measurement): <N> seconds
- Scale-down stabilization window: <≥ cold-start, usually 15 min = 900 s>
- Scale-up stabilization window: <60 s default>
- Scale-up rate cap: <2 pods / minute, justify>
- Scale-down rate cap: <1 pod / 5 min, justify>

## Observations from Deliverable D
- Phase 4 (dip): the scaler did NOT shrink; hysteresis working.
- Phase 5 (re-spike): the fleet absorbed the re-spike with <N> existing replicas; no cold-start was triggered.

## Deviations from chapter-02's starting values
- <if any; e.g. "shortened scale-up window to 45 s because our cold-start is 90 s not 4 min">
- <reasoning anchored in measurement, not guess>

## Review date
- <date>: revisit when cold-start measurement changes by >20% (chapter 04 Trigger).
```

### Deliverable F — the alert pack + `<!-- needs-escalation -->` list

Implement chapter 02's four alerts as Prometheus alerting rules in your `PrometheusRule` / alert manager:

1. **Scaler pinned at max** — `kube_horizontalpodautoscaler_status_current_replicas == kube_horizontalpodautoscaler_spec_max_replicas` for > 15 min.
2. **Scaler oscillation** — scale events in a 10-minute window exceed a threshold (chapter 02 gives the query shape).
3. **Primary vs. safety-net disagreement** — queue depth near zero and TTFT P99 above SLO (or vice versa) for > 5 min.
4. **Stale metric scrape** — `time() - prometheus_scrape_last_scrape_timestamp_seconds{job="<engine>"} > 120`.

Attach the `PrometheusRule` manifest and a test firing for at least one alert (simulated by over-committing a knob — e.g. scaling `maxReplicas: 2` and driving traffic to pin it).

Then produce the `<!-- needs-escalation -->` list — things you couldn't resolve from this exercise alone. Expected entries:

- "The right queue-depth threshold per replica. Chapter 02 names 5 as a starting point; the correct number is workload-specific and depends on the per-replica `max-num-seqs`." → exercise 05 of module 110 (Pareto); chapter 103's admission policy.
- "Whether to include prefill-TPS as a primary trigger. Chapter 02 names it secondary; on heavy-RAG fleets it is arguably primary." → workload profile; benchmark with varied prompt sizes.
- "Scale-up window for non-cold-start-sensitive classes. The 60 s default assumes we tolerate 60 s of queueing; a tighter SLO wants shorter." → SLO authoring (module 110 exercise 02); error-budget math.
- "The right `maxReplicas` ceiling. Chapter 02 doesn't set it; the right number depends on GPU capacity and the budget." → chapter 01 node-autoscaler capacity; chapter 06 multi-region shape.
- "Whether KEDA's `cooldownPeriod` is the same as HPA's `scaleDown.stabilizationWindowSeconds`. Release-dependent; current KEDA versions apply both." → KEDA docs for the pinned release.

## Starter guidance

- **Resolve the custom-metrics path first.** Deliverables C / D / E / F are all blocked on Deliverable B returning real numbers. If `kubectl get --raw` returns `<unknown>` or nothing, debug the adapter (missing rule, missing series, wrong label mapping, stale scrape) before touching the HPA.
- **Use the engine's queue metric, not a derived one.** Chapter 02 names `vllm:num_requests_waiting` / `sglang:num_queue_reqs`. A proxy's queue length is not the engine's queue — different thing. The scaler needs the engine's number.
- **Don't fight with the stabilization window shape.** HPA v2's `scaleDown.stabilizationWindowSeconds` is the rolling max/min over a window; it's how chapter 02's "slow scale-down" is expressed. Match the field name exactly; it's often confused with `periodSeconds`.
- **The dip test (Phase 4) is the hard case.** If scale-down fires during the dip, your window is too short or your rate cap is too high. Longer window, lower cap.
- **Record `desiredReplicas`, not just `currentReplicas`.** The HPA can "want" to shrink while the pod-eviction policy is still waiting; the gap between desired and current is where thrash hides.
- **One deployment at a time.** Multi-tenant scalers with per-tenant metrics can wait for a follow-up; make one deployment's scaler work end-to-end first.
- **Save the load-driver script.** Deliverable D becomes replayable; the next tuning cycle benefits from having "the same burst I ran last time" to rerun against.

## Fallbacks for constrained environments

- **No GPU nodes.** The custom-metrics plumbing is testable against a CPU-shape dummy engine that emits the same metric names (prometheus_client mock). You will not see realistic cold-start; mark Deliverable D's cold-start measurement as "mocked" and use chapter 04's defaults for Deliverable E.
- **Prometheus Adapter and KEDA both unavailable.** The HPA v2 API still supports `object` and `external` metric types with manual metric endpoints; chapter 02's path is harder to replicate without one of the adapters, but not impossible. Document the substitution.
- **Engine doesn't expose queue depth.** Synthesise via middleware (exercise 01 of module 110 shape): a sidecar that tracks in-flight and queued over the API surface. Note the synthesis in Deliverable A.
- **No load-driver infra.** A single-shell `for` loop with `curl` is enough to produce the burst shape; the measurement is less precise but demonstrates scaler behaviour.
- **Shared Prometheus you can't modify.** You can still install Prometheus Adapter or KEDA pointed at it read-only; alert-pack changes may need a change-request. Document the gap.

## Acceptance criteria

1. **Deliverable A** names every chapter-02 signal with the engine's exact metric name (or "synthesise"), the Prometheus type, and the aggregation the adapter / scaler applies.
2. **Deliverable B** shows the custom-metrics API returns real values (or KEDA's `ScaledObject` shows the triggers resolving); `<unknown>` is a fail.
3. **Deliverable C** is a complete HPA / `ScaledObject` manifest with multi-metric triggers and chapter-02-shape asymmetric hysteresis, verified against `kubectl get hpa -o yaml`.
4. **Deliverable D** records a six-phase load trace with the four required observations per phase; the dip (Phase 4) does NOT cause scale-down; the re-spike (Phase 5) is absorbed without new cold-starts.
5. **Deliverable E** is a one-page tuning note that names the inputs and the reasoning; chapter-02 numbers are starting points, not sacred.
6. **Deliverable F** has the four-alert `PrometheusRule` manifest and a demonstrated firing for at least one alert; the `<!-- needs-escalation -->` list names at least four items with references.
7. The scaler is reproducible: anyone with the same deployment, Prometheus, and load-driver script should see the same scale-up / scale-down trace.

## Common failure modes

- **CPU-utilisation HPA left in place.** The chapter-02 Error 1; the fix is to remove the CPU trigger. If this exercise leaves a CPU trigger "as backup," the HPA takes the max of signals — the CPU one is always zero, no harm — but the configuration is misleading. Delete it.
- **Symmetric scale-up / scale-down windows.** The whole shape of the chapter is fast-up / slow-down. A deploy with matching windows is a Flask-shape HPA on an inference workload. Verify Deliverable D's Phase 4 — if scale-down fires, the windows aren't asymmetric enough.
- **Counter metric as an average.** `prompt_tokens_total` is a cumulative counter; without `rate()` it rises forever. Chapter 02 Error 4. Verify every counter-source metric's adapter query applies `rate()` or `irate()`.
- **`maxReplicas` too high.** If the autoscaler can keep scaling but there's no GPU capacity, pods sit Pending and the HPA keeps requesting more — Alert 1 fires on "pinned at max" even though the real fail is upstream. Keep `maxReplicas` matched to node-autoscaler capacity (chapter 01).
- **Burst script with no backoff.** A load driver without client-side jitter creates thundering-herd retries during pod cold-start; the scale-up metric spikes artificially. Add jitter or document the artefact.
- **Phase 4 dip too short to matter.** If your dip is 15 seconds and your stabilization window is 15 minutes, the test proves nothing. Make the dip long enough that a symmetric-window HPA *would* have fired — 2–3 minutes for a 4-minute cold-start.
- **Single-replica deployment.** You can't observe scale-up / scale-down on a `replicas: 1` fixed. Start from `minReplicas: 2` or higher; let the HPA drive the count.
- **Alert pack with no tested firing.** Alerts that have never fired are hope. Simulate at least one; document the firing in Deliverable F.

## Keep it

The custom-metrics wiring (Deliverable B), the HPA / `ScaledObject` manifest (Deliverable C), the burst load profile (Deliverable D), and the alert pack (Deliverable F) become the fleet's autoscaling artefacts. Exercise 03's sharing walkthrough changes how many replicas fit per GPU — re-run the Phase-2-to-5 part of Deliverable D after sharing lands to confirm the scaler still responds correctly. Exercise 04's warm pool composes with `minReplicas` and the stabilization window from this exercise; Deliverable E's hysteresis note is the interface. Exercise 05's spot preemption uses the same scaler to drive reactive scale-out; the Phase-5 re-spike in Deliverable D is a fingerprint of the drain-and-reabsorb shape that chapter 05 relies on.
