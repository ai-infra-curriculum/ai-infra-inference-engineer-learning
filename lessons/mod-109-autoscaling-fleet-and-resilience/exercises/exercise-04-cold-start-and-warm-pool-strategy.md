# Exercise 04 — Cold-Start and Warm-Pool Strategy

**Time:** 4 hours · **Output:** one five-component cold-start measurement, one `N_warm` sizing calculation, one two-tier Deployment (warm pool + reactive scaler) manifest, one pre-pull / pre-stage sidecar, one scale-to-near-zero policy document, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 04](../04-cold-start-and-warm-pools.md) concrete by **measuring cold-start end-to-end** on a representative large-model replica, computing the right warm-pool size `N_warm` for your workload, standing up the two-tier warm-pool + reactive pattern, and publishing the policy document that tells the fleet "when scale-to-zero is right and when it is not." The deliverable is the warm-pool floor every chapter-02 scaler sits on top of and every chapter-05 drain lands into.

## Problem statement

Your fleet from exercises 01–03 has a working scaler on inference metrics. The chapter-04 motivating failure — a 70B on Knative scale-to-zero taking 5 minutes to boot while the client times out at 30 seconds — is the specific thing today's exercise exists to prevent. Measure the cold-start, decide whether scale-to-zero is viable for the deployment, size the warm pool if it isn't, and stand up both tiers so the next 09:00 traffic surge doesn't catch the fleet cold.

## Prerequisites

- Chapter 04 of this module read. Chapters 01, 02, and 05 skimmed (substrate, scaler composition, spot composition).
- **Exercises 01 and 02 complete** — GPU Operator, Prometheus, and a chapter-02-wired scaler on a running deployment. KEDA installed if you want to use scale-to-zero (HPA v2 alone can't go below `minReplicas: 1`).
- A representative large-model deployment on your fleet — the chapter 01 `vllm-llama31-70b` worked example at your hardware scale (70B on 4× H100 if you have it; 8B or 13B on a single GPU is fine, but note the cold-start will be shorter and the warm-pool math scales accordingly).
- Access to your container registry (to measure image-pull time) and to your model store (Hugging Face, S3, GCS, or an on-prem object store).
- Permission to install a privileged DaemonSet (the image pre-pull pattern often needs one).
- A load driver from exercise 02 (same burst script).

## Requirements

Six deliverables.

### Deliverable A — the five-component cold-start measurement

Measure each of chapter 04's five components on a fresh pod. Required method:

1. **Clean cache.** Drain the target node (`kubectl drain`), so the next pod lands on a cold node. Delete any cached weights on the target PVC / hostPath.
2. **Time-stamp each phase.** Instrument the pod's start with a wrapper script or the engine's native timing, capturing:
   - `t0` — pod admit (`containerStatuses` first `startedAt`).
   - `t1` — image pulled (`events` type `Pulled`).
   - `t2` — weights pulled (log line from your weight-fetch step).
   - `t3` — weights loaded into HBM (engine log line, e.g. vLLM's "Loading model weights took..." / SGLang's "Load weight begin"/"end").
   - `t4` — CUDA-graph / kernel warm-up done (engine log line, e.g. vLLM's "Starting profiling..." → "Finished profiling...").
   - `t5` — first-token latency on a warm-up request (measure externally).
3. **Record** each delta: image pull = `t1 - t0`; weight pull = `t2 - t1`; weight load = `t3 - t2`; kernel warm-up = `t4 - t3`; cache warm = measured over the first 1–5 minutes of real traffic (first 100 requests TTFT P99 vs. steady-state TTFT P99).

Output table:

```
| Component           | Measured (cold node) | Measured (warm node, cached) | Chapter-04 illustrative range |
|---------------------|----------------------|------------------------------|-------------------------------|
| 1. Image pull       | <seconds>            | <seconds>                    | 30-180 uncached, 1-5 cached   |
| 2. Weight pull      | <seconds>            | 0 (if local)                 | 60-300 uncached               |
| 3. Weight load      | <seconds>            | <seconds>                    | 20-90 for 70B BF16            |
| 4. Kernel warm-up   | <seconds>            | <seconds>                    | 20-120                        |
| 5. Cache warm (TTFT penalty on first 100 requests) | <percent> | ~0 | 30-50% first 100 reqs |
| TOTAL (cold)        | <sum>                | —                            | 270-600 s (4.5-10 min)        |
| TOTAL (warm node)   | —                    | <sum>                        | 90-180 s (1.5-3 min)          |
```

Attach the raw timestamps / log lines so the measurement is auditable.

### Deliverable B — the `N_warm` sizing calculation

Compute the warm-pool size using chapter 04's starting formula:

```
N_warm = ceil(P95(concurrent_in_flight) / per_replica_capacity)
```

Required inputs:

- **P95 concurrent in-flight** — over the last 7 days (or longest window you have). Use `vllm:num_requests_running` (or engine equivalent) averaged across the fleet; take the 95th percentile.
- **Per-replica capacity** — at the SLO's TTFT target, how many in-flight does one replica sustain? Derive from exercise 02's load-test curve or from an explicit bench.

Then adjust per chapter-04's three signals:

- **Baseline concurrency** — set `N_warm` so baseline fits without reactive scale.
- **Burst-to-warm-up ratio** — if a representative 2× burst takes less time than cold-start, `N_warm` must cover the burst alone.
- **SLO cold-start tolerance** — tight SLO → size larger; lenient SLO → tighter `N_warm`.

Record:

```
# N_warm Sizing — <deployment_name>

## Inputs
- P95(concurrent_in_flight), last 7 days: <N> requests
- Per-replica capacity at SLO: <M> in-flight
- Representative burst: <factor>× over <duration>
- Cold-start (Deliverable A, warm node): <seconds>
- SLO: TTFT P99 < <X> s

## Base formula
N_warm = ceil(N / M) = <base>

## Adjustments
- Burst scenario coverage: +<k> (burst takes less time than cold-start)
- SLO tightness (tight = +1; lenient = +0): +<k>
- Spot churn absorb (compose with exercise 05 — if reactive pool is on spot): +<k>

## Chosen N_warm
<final>

## Review trigger
Re-size when: any input changes >20%, OR SLO changes, OR fleet traffic trend shifts (chapter 02 alert on scaler-pinned-at-max as leading signal).
```

### Deliverable C — the two-tier Deployment manifest

Stand up chapter 04's "worked example" shape: a `vllm-llama31-70b-warm` Deployment with `replicas: N_warm` (on-demand node pool, NOT behind the HPA) and a `vllm-llama31-70b-reactive` Deployment with `replicas: 0` managed by a KEDA `ScaledObject` (on a spot-tolerant pool if you plan exercise 05, else on the same on-demand pool for now).

A shared Service with a label selector matching both Deployments' pods routes traffic across them. Required manifest elements:

- Both Deployments carry the label `app: <model-name>` so the Service covers both.
- The warm pool has `minReplicas` / `maxReplicas` fixed (not managed by the HPA); a `PodDisruptionBudget` with `minAvailable: N_warm - 1` protects it against voluntary disruption.
- The reactive KEDA `ScaledObject` from exercise 02's Deliverable C is re-pointed at the reactive Deployment, with `minReplicaCount: 0`.
- The warm pool's pod spec and the reactive pool's pod spec are **byte-identical** except for the node pool / taint / toleration — same engine version, same weights, same metrics. A drift between the two is a chapter-04 anti-pattern.

Attach:

- Both Deployments.
- The shared Service.
- The `ScaledObject` (updated from exercise 02).
- The `PodDisruptionBudget` for the warm pool.
- Evidence the Service routes to both: `kubectl get endpoints <service>` showing endpoints from both Deployments.

### Deliverable D — the pre-pull / pre-stage sidecar

Attack components 1 and 2 of cold-start with a pre-pull / pre-stage pattern. Pick one of:

**Option 1 — Image pre-pull DaemonSet.** A privileged DaemonSet on every GPU node that pulls the pinned inference image at idle so that new reactive pods on the node pay component 1 near-zero:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: inference-image-prepull
  namespace: inference
spec:
  selector: {matchLabels: {app: inference-image-prepull}}
  template:
    metadata: {labels: {app: inference-image-prepull}}
    spec:
      tolerations:
      - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      initContainers:
      - name: pull
        image: <pinned-inference-image>
        command: ["sh", "-c", "true"]      # just pulling the image is enough
      containers:
      - name: pause
        image: registry.k8s.io/pause:3.9
```

**Option 2 — Weight pre-stage via local PVC or shared RWX.** Pre-populate model weights on a node-local `hostPath` (or a shared ReadWriteMany PVC backed by FSx / Filestore / EFS) so new pods mount the same weights and skip component 2.

Attach:

- The DaemonSet / init-job manifest.
- Before/after measurement: run Deliverable A against a *warmed* node (image pre-pulled, weights pre-staged) and compare to the cold-node measurement.
- A brief note on which option fits your fleet (image pre-pull is simpler; weight pre-stage is more effective if your image is small but the weights are huge).

### Deliverable E — the scale-to-near-zero policy document

A one-page policy that names, per workload class, whether scale-to-zero is used:

```
# Scale-to-Near-Zero Policy — <fleet_name>

## Classification rule (from chapter 04)
- Measured cold-start < 30 s:            Knative / KServe scale-to-zero OK
- 30 s ≤ measured cold-start < 2 minutes: KEDA minReplicas=1 + activator pre-warm OR N_warm=1
- Measured cold-start ≥ 2 minutes:        warm-pool pattern required; no scale-to-zero

## Per-workload-class decisions
| Workload class          | Model     | Measured cold-start | Policy                                 |
|-------------------------|-----------|---------------------|----------------------------------------|
| chat (interactive, big) | 70B BF16  | <X> s (Deliverable A) | warm pool N_warm=<N>; reactive on spot |
| chat (interactive, small)| 8B FP8   | <Y> s                 | KEDA minReplicas=1 with activator       |
| embedding (batch)       | 1B        | <Z> s                 | Knative / KServe scale-to-zero          |
| rerank (bursty)         | 7B        | <W> s                 | N_warm=1 + reactive KEDA                |

## Composition with other chapters
- Warm pool on on-demand nodes always (chapter 05).
- Prefix-cache warm script runs in init-container for prefix-heavy classes (chapter 04 Error 8, chapter 104).
- `maxReplicas` matches cluster-autoscaler capacity; `pool-h100-full` alone cannot be exceeded without triggering chapter 01 Error 4.

## Review date
- Monthly: review per-class cold-start measurements; re-size N_warm if P95 concurrency changed by >20%.
- Quarterly: re-benchmark the Knative/warm-pool threshold against actual client timeouts.
```

### Deliverable F — the `<!-- needs-escalation -->` list

Produce the list — things you couldn't resolve from this exercise alone:

- "The right pre-stage path: node-local `hostPath` vs. `ReadWriteMany` PVC. The trade-off depends on how fast our object store is and how many GPU nodes we run; chapter 04 Error 6 flags shared-PVC saturation." → chapter 04; benchmark on our fleet.
- "Whether to bake weights into the image. Chapter 04 names the trade-off; our image-registry storage cost and model-rotation cadence drive the decision." → platform cost model; module 105 model-ops cadence.
- "Prefix-cache warm script contents. Chapter 104 names the common-prefix identification; we don't have the common-prefix set catalogued yet." → chapter 104; module 110 workload-classification exercise.
- "Predictive scaling cadence. Chapter 04 mentions 08:50 pre-scale for a 09:00 spike; whether cron-triggered scale is the right shape for our traffic pattern is deployment-specific." → module 110 SLO methodology.
- "Whether CUDA graphs should be captured per-serve-shape or pre-persisted. Chapter 04 Error 7 flags re-trace. Engine-release-specific." → vLLM / SGLang / TRT-LLM release notes; exercise 02 of module 102.

## Starter guidance

- **Measure first; policy second.** Deliverable E's classification rule only works if Deliverable A's cold-start measurement is real. Don't estimate; the 30 s / 2 min threshold is chapter 04's, and it has teeth.
- **Drain a node before the cold measurement.** A node that pulled the image yesterday is warm; your "cold-start" measurement will under-report by minutes. `kubectl drain --delete-emptydir-data` is the cleanest path.
- **Instrument the pod, don't eyeball the logs.** A wrapper script around the engine entrypoint, recording timestamps to stdout, is cheaper than piecing it together from `kubectl logs` later. Engines like vLLM and SGLang already log weight-load and warm-up boundaries — grep for those.
- **Keep the two Deployments byte-identical.** Deliverable C's warm and reactive pods differ only in node pool / toleration / spot-vs-on-demand. A drift between their engine versions or args is a landmine; the chapter calls it out and the exercise re-calls it.
- **PodDisruptionBudget on the warm pool.** Without a PDB, a node drain (chapter 05 shape) or a cluster upgrade can evict the warm pool and the next spike pays cold-start. The PDB is the belt + suspenders.
- **Don't skip the pre-stage.** Deliverable D is the biggest single cold-start win; a fleet with no pre-pull pays full image + weight cost every spin-up. A DaemonSet is ~10 lines of YAML; it's not optional.
- **Compose with exercise 05's spot plan.** If exercise 05 will run on spot, your N_warm should include the spot-churn absorption. If exercise 05 hasn't run yet, size for now and leave a review trigger in Deliverable B.

## Fallbacks for constrained environments

- **Can't drain the node.** Measure cold-start on a new node (Karpenter / Cluster Autoscaler scaling up a fresh node). Note that this includes node-boot time, which the chapter-04 measurement doesn't; subtract or document the delta.
- **No KEDA; HPA v2 only.** You can't scale-from-zero. Fallback: `minReplicas: 1` on the reactive HPA, and let the warm pool absorb everything below reactive's capacity. Document the limitation; revisit when KEDA lands.
- **Model is small enough that chapter-04's thresholds are overkill.** For <13B models with sub-30 s cold-start, Deliverable E's policy for that class is "Knative / KServe" — the exercise still produces the measurement and the policy document, just with a shorter warm-pool tier.
- **Container image is enormous (baked-in weights).** Deliverable D's Option 1 (image pre-pull) absorbs the weight-pull cost along with the image. Record the image size and the registry bandwidth; some registries throttle large pulls.
- **No node-local storage; only object store.** Deliverable D's Option 2 is shared RWX; bench the parallel-pod-startup path (chapter 04 Error 6) — some filers saturate at 5–10 concurrent readers.
- **Shared / regulated cluster with no privileged DaemonSets allowed.** Pre-pull via an unprivileged init-container on the warm-pool pods themselves (the image comes down when the first warm pool pod starts; subsequent reactive pods on the same node get the cached image). Note the simplification; cold-start from a brand-new node still pays the pull.

## Acceptance criteria

1. **Deliverable A** records a five-component cold-start table for both a cold node and a warmed node, with the raw timestamps attached; the delta between the two columns is Deliverable D's value.
2. **Deliverable B** computes `N_warm` with inputs documented and adjustments named; the formula and adjustments are inspectable.
3. **Deliverable C** stands up a two-tier Deployment (warm + reactive) with a shared Service, byte-identical pod specs except for the node pool, and a PodDisruptionBudget on the warm tier.
4. **Deliverable D** implements pre-pull and/or pre-stage with a before/after measurement showing the cold-start reduction.
5. **Deliverable E** is a one-page policy document with the chapter-04 threshold instantiated per workload class on the fleet.
6. **Deliverable F** names at least five `<!-- needs-escalation -->` items with references.
7. The warm pool is reproducible — anyone with the same Deployment manifest, PVC configuration, and pre-pull sidecar should get the same cold-start profile.

## Common failure modes

- **Measuring cold-start on a warm node and calling it cold.** Chapter 04 Error 1 masquerading. If the image is in the kubelet's cache or the weights are in local disk, your measurement is wrong. Drain first.
- **`minReplicas: N_warm` on the HPA.** Chapter 04 Error 3. The warm pool becomes subject to scale-down. Keep the warm pool outside the HPA; the reactive tier is the only one the HPA controls.
- **Pre-pull without pre-stage.** Chapter 04 Error 4. Image pull is seconds; weight pull is minutes. Fix: both components or neither.
- **Baked-in image without a tag discipline.** Chapter 04 Error 5. A `:latest` tag hides which weights are deployed; an incident six months later can't be reproduced. Pin.
- **Shared PVC saturation on fleet spin-up.** Chapter 04 Error 6. Twenty pods starting at once reading from one filer; each takes ten minutes instead of two. Fix: prefer node-local PVC primed by a sidecar.
- **CUDA-graph re-trace mid-serve.** Chapter 04 Error 7. Warm-up covers batch shapes up to 32; a shape-64 request arrives; the engine blocks on re-trace. Fix: cover expected shapes at warm-up (compose with module 102 exercise 02).
- **Reactive pods identical to warm *except* the args.** A `max-num-seqs=64` on warm and `max-num-seqs=128` on reactive produces inconsistent latency profiles that your exercise-02 HPA signals cannot untangle. Keep byte-identical.
- **Scaling N_warm by guess.** Deliverable B without data is Deliverable B with a different guess. The P95 of in-flight is a specific number; find it.
- **Policy document with no per-class rows.** Deliverable E's whole point is per-class policy. A one-line policy that says "warm pool for everything" is a fail; a one-line policy that says "scale-to-zero for everything" is a worse fail (chapter 04's motivating story).

## Keep it

The cold-start measurement (Deliverable A), the `N_warm` sizing (Deliverable B), the two-tier Deployment (Deliverable C), and the pre-pull sidecar (Deliverable D) become the fleet's warm-pool artefacts. Exercise 05 moves the reactive tier onto spot and composes the drain protocol on top; the warm tier stays on-demand per chapter-05 discipline. Exercise 06's multi-region shape replicates the entire two-tier structure per region, with each region's `N_warm` sized to absorb its share of the failover (chapter 06's capacity formula). Deliverable E's policy is the artefact an on-call reaches for when a tenant asks "why can't my 70B scale to zero?" — the answer is in the measurement.
