# Chapter 04 — Cold-Start and Warm-Pool Strategy: Scale-to-Near-Zero for Large Models

> A web pod cold-starts in a second. A 70B inference pod cold-starts in minutes. Serverless platforms (Knative, Cloud Run, Fargate) were designed around the first; naive adoption on inference pays the second at every scale-up. The fix is either to pick a serverless platform that budgets for the cold-start (works for small models) or to pre-warm the hot set behind a scaler (works for large models the serverless budget cannot absorb). This chapter builds both.

## Motivation

A platform team ships a Llama-3.1-70B replica on Knative with `min-scale=0` and a 60-second idle timeout. The theory: nights are cheap because the fleet scales to zero; mornings scale back up as traffic arrives.

What actually happens on day one:

- **At 08:57** the first request arrives. The pod is at `min-scale=0`; Knative starts a replica. The container pulls a 20 GiB image from the registry (90 seconds on a cold node — the cache is cold too), downloads 140 GiB of weights from Hugging Face / S3 (120 seconds on a good network, 400 seconds on a bad one), loads the weights into HBM (45 seconds), runs CUDA-graph warm-up (30 seconds), and marks ready at ~5 minutes.
- **Meanwhile at 08:57** the client's first request times out at 30 seconds — Knative's default scale-up budget. Knative returns a 503 to the ingress; the client retries; Knative sees another request and either joins it to the same still-starting pod or starts another one.
- **At 08:58** traffic is now 10 requests and the ingress has eight 503s. A customer complaint arrives.
- **At 09:02** the pod finally serves its first request. The scaler is now "healthy." The team is already in a post-mortem.

Knative's scale-to-zero works beautifully for a small model whose cold-start is 10–20 seconds. It does not work for a 70B whose cold-start is 3–5 minutes. The right design splits the problem: *what makes the cold-start long, which of those components can be shortened, and what does the fleet pre-warm to bound the remaining cost.*

By the end of this chapter you can:

- Enumerate the five components of large-model cold-start — image pull, weight pull, weight load, kernel warm-up, cache warm — and measure each one on your fleet.
- Decide between Knative-shape scale-to-zero (small models) and a custom warm-pool pattern (large models), with a defensible threshold.
- Size the warm pool against traffic variability and the SLO's cold-start budget.
- Compose the warm pool with chapter 02's HPA — `minReplicas` as the warm-pool floor, `maxReplicas` as the spike ceiling, pre-warmers held outside the HPA.
- Deploy the right weight-locality pattern — baked-in, pre-staged on PVC, mmap from object store, in-memory shared cache — against cold-start budget.

Component timings quoted in this chapter are illustrative order-of-magnitude figures. Actual numbers depend on image size, storage tier, network bandwidth, GPU model, and engine; always measure on your own fleet before using any specific figure in a policy.

## The five components of cold-start

A large-model inference pod does five things between "pod is scheduled" and "pod can serve the first request":

### Component 1 — Image pull

The container image contains the engine (vLLM / SGLang / TRT-LLM) + Python + CUDA runtime + Triton kernels + the engine's compiled extensions. Modern inference images are **5–25 GiB** uncompressed depending on which runtimes are bundled (TRT-LLM images are the largest; vLLM's are smaller).

Pull time depends on:

- **Registry locality.** Pulling from a registry in the same region as the node is 10× faster than cross-region.
- **Node cache.** A node that pulled the image yesterday has it cached — subsequent pods on that node skip this step. First-time pulls pay in full.
- **Layer size and count.** Many thin layers parallelise; few fat layers do not. The engine image typically has a few layers covering the thousands of CUDA kernel files.

Typical: **30–180 seconds** uncached, **1–5 seconds** cached.

Mitigations: **image streaming** (gcr.io has zstd:chunked; estargz / soci-snapshotter on AWS); **image pre-pull** via a DaemonSet (a sidecar that pulls the "next" image onto every node at idle); **shared registry mirror**; keep nodes alive long enough to preserve the cache (chapter 01's `consolidateAfter` on Karpenter).

### Component 2 — Weight pull

The model weights are the biggest single artifact. **Llama-3.1-70B BF16 is ~140 GiB**; FP8 is ~70 GiB. The pod pulls from a model store — Hugging Face Hub, S3, GCS, or an on-prem object store.

Pull time depends on:

- **Network bandwidth.** 10 Gbps = ~1 GB/s = 70 seconds for a 70 GiB FP8 model. 100 Gbps = 7 seconds. VPC cross-region = much slower.
- **Storage tier.** S3 Standard vs. S3 Express; GCS Standard vs. GCS multi-region; local SSD cache on the node.
- **Parallelism.** `transformers`'s `from_pretrained` uses one thread by default; `huggingface_hub`'s parallel download is faster; `s5cmd` or `aws s3 cp` with `--max-concurrent-requests` is faster still; `rclone` with many workers can saturate the link.

Typical: **60–300 seconds** uncached, **0 seconds** if the weights are on local PVC or baked into the image.

Mitigations: **pre-stage weights on a shared tier** (an NFS / FSx / Filestore volume that every GPU node mounts, or a per-node local SSD cache primed by a sidecar); **ReadWriteMany PVC** with the weights pre-populated; **weight-streaming** engines that mmap from object store (some engines support reading weights lazily as layers are needed); **baked-in** images (weights in the image — eliminates this component at the cost of making images huge).

### Component 3 — Weight load

Reading weights from disk into HBM. Even with the weights local, moving 140 GiB from host RAM / SSD to HBM takes time:

- PCIe Gen5 x16: **~64 GB/s** peak; **~30–40 seconds** for 140 GiB BF16.
- PCIe Gen4 x16: **~32 GB/s** peak; **~60 seconds** for 140 GiB BF16.
- NVMe-backed PVC adds page-cache / mmap path overhead.
- For sharded (TP) loads, the engine loads each shard on each rank in parallel — the per-rank data is smaller but coordinated.

Typical: **20–90 seconds** for 70B-class models; shorter for FP8 / INT4.

Mitigations: **safetensors** (mmap-efficient) over pickled .pt; **sharded on-disk** so each TP rank reads only its shard; FP8 / FP4 to halve/quarter the bytes; **CUDA Unified Memory** on supported GPUs for lazy transfer (but with runtime cost).

### Component 4 — Kernel / CUDA-graph warm-up

Engines warm up before serving:

- vLLM profiles memory, traces CUDA graphs for common batch shapes, and runs a dummy forward pass to fault in kernels.
- SGLang runs its own warm-up; TensorRT-LLM uses pre-built engines (engine-build is much larger but amortised ahead of serve).

Warm-up time depends on:

- **Model size** (more parameters = longer first forward pass to compile against).
- **Batch shapes** traced (more shapes = more graphs = longer warm-up but faster steady-state).
- **CUDA graphs enabled.** With graphs, there is a re-trace cost at each new shape; without them, warm-up is faster but steady-state is slower.

Typical: **20–120 seconds** depending on model and engine config.

Mitigations: **narrower batch-shape coverage** at warm-up (pay for less, risk re-trace at serve); **disabled CUDA graphs** (lower warm-up cost, worse steady-state); **pre-captured graphs persisted to disk** (some engines support this).

### Component 5 — Cache warm (prefix, speculative-draft)

If the workload benefits from prefix cache (chapter 104) or has a speculative-decoding pair (chapter 108), the pod is *technically* serving at the end of component 4, but its first few hundred requests pay penalties:

- Prefix cache is empty; the TTFT on prefix-cached workloads is 30–50 % worse until the cache fills.
- Speculative decoding's engine-internal state (e.g. EAGLE's feature state for the current sequences) is empty; acceptance may dip for the first batch before stabilising.
- If the deployment runs with torch.compile'd CUDA graphs that were cleared, the first few requests re-trace.

Typical: **first 1–5 minutes** of serving traffic are warm-up; strictly it's not cold-start since the SLO is being met, but the measurement is worse than steady-state.

Mitigations: **synthetic traffic replay** at warm-up (chapter-02 of mod 102 covers the shape); **prefix-set pre-load** via a startup script; **pre-trace** the common shapes the fleet sees.

## The total cold-start budget

Summing — **a BF16 70B on 4× H100** from scratch:

```
image pull (uncached, same-region registry):   60-120 s
weight pull (from fast object store):           120-300 s
weight load (BF16, PCIe Gen4 x16):              60-90 s
CUDA-graph warm-up (vLLM, 8 shapes):            30-90 s
---
total (first-ever pod on a cold node):          270-600 s   (4.5-10 min)

With image cached + weights pre-staged on local SSD:
image pull (cached):                             2-5 s
weight pull (local SSD):                        0 s
weight load (local SSD → HBM):                  60-90 s
CUDA-graph warm-up:                              30-90 s
---
total (warm node, pre-staged weights):          90-180 s   (1.5-3 min)
```

Two orders of magnitude between "cold node" and "warm node" is the whole point. The warm-pool strategy's job is to make sure traffic hits the second column, not the first.

## Knative Serving — when it fits and when it doesn't

Knative Serving (<https://knative.dev/docs/serving/>) is the canonical scale-to-zero platform on Kubernetes. It is the compare-point every custom warm-pool pattern is built against.

### What Knative does well

- **Scale-to-zero** with a configurable idle timeout.
- **Scale-from-zero** with a request-queue: the first request waits while a replica is spinning up, routed by the Activator.
- **Concurrency-based scaling** (as well as RPS-based): a single knob (`containerConcurrency`) targets N in-flight per replica.
- **Rolling revisions** with blue-green and canary semantics (compose with mod 113).

### Where Knative struggles on large models

- **Scale-up budget is tens of seconds, not minutes.** The Activator queues one or a few requests while a replica spins up. On a 5-minute cold-start, the queue depth balloons and clients time out. The default `Revision` timeout is 10 minutes but the client timeout is usually 30–60 s; the mismatch shows up as 503s / 504s to the client.
- **Autoscaler was designed for stateless light pods.** The KPA (Knative Pod Autoscaler) scales based on concurrency with sub-minute reaction; combining with a 4-minute cold-start produces the chapter-02 oscillation shape.
- **Cold-start-optimised ergonomics help small models most.** Features like pre-pulling, smaller images, and scale-to-one-on-first-request fit a FastAPI service naturally.

### The fit rule

Knative Serving is a **good fit** for inference when:

- The model is small (<13B) and cold-start is under 30 seconds on your fleet.
- The traffic shape has long idle periods and small, infrequent bursts.
- The SLO tolerates occasional cold-start latency on the first request after idle.

Knative Serving is a **poor fit** for inference when:

- The model is large (30B+) and cold-start exceeds the client timeout.
- Traffic has frequent bursts where consecutive bursts cannot afford a cold-start each.
- The SLO tolerates no cold-start latency.

The KServe project (built on Knative) adds model-serving-specific knobs that materially shift this fit rule — in particular its **Local Model Cache** feature (<https://kserve.github.io/website/docs/model-serving/generative-inference/modelcache/localmodel>) pre-stages model artifacts to node-local storage and cuts large-model cold-start from the tens of minutes down to roughly a minute. Verify the current KServe release and feature set at <https://kserve.github.io/website/> before choosing between "plain Knative" and "KServe."

### KServe as the inference-shaped wrapper

KServe (<https://kserve.github.io/>) is the model-serving platform on top of Knative. It adds:

- **InferenceService** CRD that bundles the model, the predictor, and the autoscaler.
- **Model-caching** support that pre-stages model artifacts (addresses component 2 above).
- **GPU-aware** `ResourceLimit` defaults.
- **Canary / blue-green** rollout at the model level (compose with mod 113).

For small-to-medium models with scale-to-zero needs, KServe is often the right wrapper around Knative; for large models, the chapter's custom-warm-pool pattern still wins.

## The custom warm-pool pattern

For large models, the robust pattern is a pre-warmed pool decoupled from the HPA's reactive scaling:

```
                         [ Client ]
                             │
                             ▼
                      [ Gateway / LB ]
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    [ Pre-warmed      [ Pre-warmed        [ HPA-scaled
      replica #1 ]      replica #2 ]        replicas, 0..N ]
    (always on,        (always on,         (minReplicas=0, created
     serving)           serving)             in response to spikes)
```

Two tiers:

- **Warm pool.** `N_warm` replicas kept running 24/7 (behind a plain `Deployment` with fixed replicas, *not* behind the HPA). Routes receive these first. Their cold-start happened once; thereafter they are always-warm.
- **Reactive scaler.** Chapter 02's HPA against `minReplicas=0` (via KEDA — HPA v2 cannot go below 1). Scales up when the warm pool is saturated; scales down when demand recedes.

This is **not** the same as "set `minReplicas=N_warm`" — the warm pool is intentionally outside the HPA so that it doesn't shrink on quiet periods. The HPA's floor is zero or one; the warm pool is the constant floor.

### Sizing the warm pool

Three signals shape `N_warm`:

1. **Baseline concurrency.** What's the steady-state concurrent request count at the 10th-percentile hour? That many replicas-worth of in-flight should fit in the warm pool; otherwise even baseline traffic triggers reactive scaling.
2. **Burst to warm-up ratio.** If a 2× traffic spike takes 2 minutes to subside and a cold-start is 3 minutes, the warm pool must absorb the spike entirely (`N_warm ≥ burst_peak / per-replica-capacity`). If the spike takes 10 minutes, the reactive scaler can share the load.
3. **SLO cold-start tolerance.** If the SLO says "TTFT P99 < 2 s always," no cold-start is tolerable and the warm pool covers every spike. If the SLO allows a brief cold-start tail (say, "TTFT P99 < 2 s at steady-state; P99.9 < 10 s"), you can size `N_warm` tighter and rely on reactive scaling to catch the long tail.

A starting formula (verify on your workload):

```
N_warm = ceil(P95(concurrent_in_flight) / per_replica_capacity)
```

Then adjust up for cold-start-sensitive classes, down if cost is tight and the SLO is lenient. Exercise 04 benchmarks the right value.

### Scaling the warm pool itself

The warm pool is static by default — it does not auto-scale. In practice, you update `N_warm` manually (or via a cron) in response to traffic-growth trends over weeks, not minutes. This is a *scale decision* made by the operator, not a reactive signal.

For diurnal workloads where night-time baseline is much lower than day-time, a two-tier warm pool (day-only and night-only) via cron'd `Deployment.replicas` updates is simpler and more reliable than trying to make the HPA do both jobs.

### The pre-pull sidecar

Even with the warm pool, the reactive scaler still boots cold replicas on spikes. Each cold replica pays components 1 (image) and 2 (weights). Mitigations:

- **Image pre-pull DaemonSet** — a privileged DaemonSet that pulls the current image on every GPU node at idle. New pods on that node have component 1 near-zero.
- **Weight pre-stage** — a startup job or sidecar that writes the weights to a node-local `PersistentVolume` (or hostPath) once. New pods mount the same PV; component 2 is near-zero.
- **Shared PVC** (`ReadWriteMany`) backed by Filestore / EFS / NFS. Every pod sees the same weights; slower than node-local but simpler.

Chapter 01's node autoscaler should keep GPU nodes alive long enough (`consolidateAfter: 60m` or longer) so the pre-pull investments survive scale-down events.

### Baking weights into the image — the specific-case discussion

An alternative to pre-staging weights on PVC is to bake them into the container image:

```dockerfile
FROM vllm/vllm-openai:v0.6.3
RUN huggingface-cli download meta-llama/Meta-Llama-3.1-70B-Instruct \
    --local-dir /models/llama31-70b
ENV MODEL_PATH=/models/llama31-70b
```

- Pro: single artefact; weights and engine version pinned together.
- Pro: image streaming (if your registry supports it) can start the container before all layers are pulled.
- Con: image is now **160+ GiB**. Pulls are slow, registry storage is expensive.
- Con: changing the model means a new image; the engine-upgrade and model-upgrade lifecycles are coupled.

For pinned, long-lived deployments the baked-in path can be ergonomic; for frequently rotated models it is painful. The trade-off is fleet-specific.

### Pre-cache warm (component 5)

For prefix-cache-heavy workloads, an additional step in the startup job:

```bash
# After readiness, warm the prefix cache with the common prefix set
for prefix in "${COMMON_PREFIXES[@]}"; do
  curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"${prefix}\", \"max_tokens\": 1}" > /dev/null
done
```

(Or the equivalent in whatever script language your init-container uses.) Chapter 104 covers the prefix-cache and how to identify the common prefix set.

## Cold-start as a scaling input

Chapter 02's HPA works on *current demand*; it has no model of cold-start cost. Two patterns add cold-start awareness:

- **Predictive scaling** — pre-scale up before anticipated traffic (e.g. 08:50 cron-triggered scale to match the 09:00 spike). Works on predictable diurnal / weekly patterns.
- **Buffering** — the per-replica scale-up threshold is tightened (e.g. queue depth = 3 instead of 5) to trigger scale-up earlier, buying time for the cold-start to complete before the SLO breaks.

Both are refinements; the warm pool is the primary defense.

## Composing with chapter 05 (spot resilience)

Spot replicas receive a preemption notice (chapter 05); the warm pool mitigates the resulting churn by covering the drained replica's traffic. Two disciplines:

- **Warm pool on on-demand nodes only.** Spot preemption on the warm pool undermines its purpose — a warm pool that preempts midday has broken the whole model. Chapter 05 recommends the inverse: warm pool on-demand, reactive scale on spot.
- **Pre-warm count accounts for expected spot churn.** If 10 % of spot replicas preempt per hour, `N_warm` should be sized such that the fleet can absorb that rate without dipping to cold-start-on-spike.

## A worked example — a Llama-3.1-70B fleet with a two-replica warm pool

```yaml
# Warm pool: always-on, 2 replicas, NOT behind the HPA
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama31-70b-warm
spec:
  replicas: 2
  # ... same pod spec as chapter 01, on-demand node pool
---
# Reactive scaler: 0..18 replicas, behind KEDA
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama31-70b-reactive
spec:
  replicas: 0   # KEDA manages this
  # ... same pod spec, spot-tolerant node pool
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-llama31-70b-reactive
spec:
  scaleTargetRef:
    name: vllm-llama31-70b-reactive
  minReplicaCount: 0
  maxReplicaCount: 18
  cooldownPeriod: 900
  pollingInterval: 15
  # ... triggers as chapter 02
```

The Service in front routes to **both** Deployments via a label selector that matches both:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama31-70b
spec:
  selector: {app: vllm-llama31-70b}    # label present on both Deployments' pods
  ports: [{port: 80, targetPort: 8000}]
```

Both Deployments' pods carry `app=vllm-llama31-70b` so the Service load-balances across them. The warm pool absorbs baseline + small bursts; the reactive pool handles spikes. (Some fleets prefer a single Deployment with the warm replicas pinned via `minReplicas` and the HPA scaling around them; the two-Deployment pattern is simpler and more explicit but duplicates templating. Check which shape your rollout tooling — Argo Rollouts, Flagger, etc. — composes with before picking.)

## The scale-to-zero policy for small models

For small models (<13B, cold-start under 30 s), the Knative / KServe path is viable:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: small-embedder
spec:
  predictor:
    minReplicas: 0
    maxReplicas: 10
    containerConcurrency: 16
    containers:
    - name: kserve-container
      image: embed-server:<tag>
      resources:
        limits:
          nvidia.com/gpu: 1
```

Scale-to-zero at off-hours; scale-up on the first request. The 20-second cold-start fits within the client timeout; the SLO is a lower tier than the 70B's.

### The threshold

A defensible threshold between "Knative-scale-to-zero" and "warm pool":

```
If measured cold-start < 30 seconds:        Knative / KServe scale-to-zero
If 30 s ≤ measured cold-start < 2 minutes:  Knative/KServe with activator-pre-warm (hold at minReplicas=1) OR warm-pool with N_warm=1
If measured cold-start ≥ 2 minutes:         warm-pool pattern required; scale-to-zero not safe
```

Measure, don't guess.

## Common deployment errors

### Error 1 — Knative scale-to-zero on a 70B

The chapter's motivating failure. The warm-pool pattern or `minReplicas: 2` (not zero) is the fix.

### Error 2 — warm pool sized from guesses

A fleet with `N_warm = 2` serving traffic that routinely hits 4-replica concurrency pages as soon as a spike arrives. Measure baseline concurrency first; size from data.

### Error 3 — warm pool behind the HPA

Setting `minReplicas: N_warm` on the HPA makes the warm pool subject to scale-down during quiet periods. The next scale-up on the next spike pays cold-start. Keep the warm pool *outside* the HPA.

### Error 4 — pre-pull without weight pre-stage

The image is pre-pulled but the weights still come from object store. Component 2 still takes minutes. Fix: both components or neither.

### Error 5 — baked-in image without versioning discipline

Weights baked into an image without a tagging discipline that identifies the (model-version, engine-version) pair leaks into production. Ship with a canonical tag (e.g. `:llama31-70b-bf16-vllm0.6.3`) and a mutation-free policy.

### Error 6 — shared PVC without reader coordination

A `ReadWriteMany` PVC with many pods all reading at pod-start simultaneously saturates the backing filer. Pods see long weight-load times on cluster-wide scale-up. Fix: prefer `ReadWriteOnce` per-node PVC or hostPath with a pre-stage sidecar.

### Error 7 — CUDA-graph re-trace mid-serve

Pod warms up for batch shapes ≤ 32 but serves a shape-64 request; graph re-trace blocks the whole scheduler for seconds. Chapter 02 of mod 102 covers the warm-up shape coverage. Choose shapes that span expected traffic.

### Error 8 — no pre-warm for the prefix cache

The reactive replica is "ready" per readiness probe but its first 50 requests have cold-prefix-cache TTFT. Chapter 02's TTFT P99 metric sees the spike; the HPA scales further; the fleet churns. Fix: a prefix warm-up script before marking ready.

## The boundary to the level-20 peer

This chapter does not teach:

- **Knative installation**, basic Knative concepts, or Istio ingress fundamentals. Those are at `ai-infra-engineer` level 20.
- **Container image fundamentals** — layering, multi-stage builds, registry basics. Level 20 / MLOps peer level 25.
- **General CDN and image-streaming technology** — zstd:chunked, estargz, soci. Level 20 / generic.

This chapter *does* teach: the inference-specific cold-start anatomy, the warm-pool pattern against large-model cold-starts, the Knative fit rule, and the composition with chapter 02's HPA and chapter 05's spot resilience.

## Summary

- Large-model cold-start is five components: **image pull**, **weight pull**, **weight load**, **kernel warm-up**, **cache warm**. Measure each; attack each with a specific mitigation.
- Knative Serving's scale-to-zero fits small models (<13B, cold-start under 30 s). For large models it is a trap — the Activator budget is tens of seconds, the real cold-start is minutes.
- The custom warm-pool pattern: `N_warm` replicas in a separate Deployment, always-on, serving baseline. The reactive scaler sits behind, scaling 0..N on top.
- Size `N_warm` from measured concurrency (P95 of in-flight, divided by per-replica capacity), adjusted up for cold-start-sensitive SLOs.
- Pre-pull sidecars, weight pre-stage (local PVC or shared RWX), image streaming, and FP8 / FP4 all shorten cold-start by cutting the individual components.
- Warm pool on on-demand nodes; reactive scale on spot — chapter 05 composes.
- Prefix-cache warm is a fifth component; a startup script that replays the common prefix set warms it.
- Generic Knative / image-tech is the level-20 peer's; this chapter owns the inference-specific cold-start strategy.
