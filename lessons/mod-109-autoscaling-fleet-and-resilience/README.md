# Module 109 — GPU Fleet Autoscaling and Resilience

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 18

Modules 102–108 made a single replica fast — the engine, the scheduler, the KV allocator, the parallelism topology, the drafter. This module zooms out one level: a *replica* becomes a *fleet* that must scale with demand, share expensive GPUs, survive preemption, and survive the loss of a whole region or a whole provider. All of that is Kubernetes-shaped — but the standard Kubernetes autoscaling toolkit (`HorizontalPodAutoscaler` on CPU utilisation, `Cluster Autoscaler` on generic VMs, `Deployment` with a vanilla rolling update) is wrong for GPU inference in specific, operator-visible ways. This module teaches the inference-specific extensions.

By the end of it you can:

- **Deploy GPU-aware Kubernetes serving** — the NVIDIA GPU Operator (device plugin, driver / toolkit DaemonSets, DCGM-exporter, MIG manager, node-feature-discovery), a `RuntimeClass` that routes to `nvidia-container-runtime`, taints and tolerations that keep CPU pods off GPU nodes, and a GPU-aware node autoscaler (Karpenter or Cluster Autoscaler) that bin-packs replicas against the right GPU SKU rather than the first SKU with a free slot.
- **Author an HPA / KEDA scaler on inference-specific custom metrics** — queue depth, in-flight requests, prefill-tokens-per-second, TTFT P99 — rather than CPU or GPU-utilisation, and reason about the scale-up vs. scale-down hysteresis that keeps a 5-minute-boot replica from thrashing on a 30-second traffic dip.
- **Deploy GPU sharing where appropriate** — MIG for hard isolation (A100 / H100 / H200 only), MPS for latency-tolerant small models (soft sharing, no isolation), DRA (Dynamic Resource Allocation) for flexible per-request scheduling — and reason about when NOT to share (large-model latency-critical workloads, noisy-neighbour-sensitive tenants, HBM-starved GPUs).
- **Design a warm-pool / scale-to-near-zero strategy for cold-start-sensitive workloads** — Knative Serving as a compare-point for small / fast-booting models, custom warmers for large models whose cold-start (pull → load → compile → warm) exceeds a serverless platform's scale-up budget.
- **Build spot resilience** — the ~30-second GPU-spot preemption notice, in-flight-request draining, checkpointed KV where it pays, cross-region failover routing, and provider-fallback for the preemption class your fleet is willing to tolerate.
- **Build multi-region / multi-provider resilience at the serving edge** — DNS / Anycast / global load-balancer patterns at inference-API granularity, provider-fallback via a LiteLLM-shape router, and cost-vs-latency routing that trades a few milliseconds of extra latency for a cheaper or safer backend.
- **Cite the boundary to [`ai-infra-engineer`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) (level 20)** — general Kubernetes, Terraform, cloud base, and generic Day-2 operations live there; this module covers the GPU-specific extensions and the inference-shaped autoscaling / resilience logic on top.

> **Prerequisites:** [Module 102](../mod-102-inference-engine-internals/README.md) (engine startup, warm-up, and metrics surface — the autoscaler's input signal), [Module 103](../mod-103-request-scheduling-and-batching/README.md) (queue depth, admission control — the scaler's primary signals), [Module 104](../mod-104-kv-cache-management/README.md) (KV footprint — the thing that fails first when you over-share a GPU), [Module 107](../mod-107-distributed-inference-topologies/README.md) (a replica is a TP / PP group, not a single pod — the autoscaler must respect the group's atomic shape). The level-20 track covers vanilla Kubernetes, Helm, Terraform, cloud VPC / IAM, generic autoscaling, and generic rolling-update discipline — if any of that is unfamiliar, go there first.

## The core framing

Autoscaling an inference fleet looks like autoscaling any Kubernetes service and *is not*. Three asymmetries drive every design decision in this module:

- **A replica takes minutes to boot, not seconds.** A 70B model on 4× H100 pulls tens of GiB of weights, loads them, runs CUDA-graph warm-up, and compiles kernels for its first few shapes. End-to-end cold-start is commonly 1–5 minutes depending on model, hardware, and image-pull locality. The HPA's default scale-up / scale-down behaviour (seconds-granularity reactive) is wrong at that time scale.
- **The right scaling signal is not CPU or GPU utilisation.** GPU utilisation reads 100 % during a single-request prefill *and* during a saturated batch — the metric doesn't distinguish "one request burning HBM bandwidth" from "the fleet is overloaded." Chapter 02 names the inference-specific signals that do (queue depth, in-flight, prefill-TPS, TTFT P99) and the hysteresis you set on top of them.
- **The GPU is expensive, scarce, and often preemptible.** On-demand H100 capacity is rationed; spot H100 may be a fraction of the price but with a 30-second notice; multi-provider is operationally normal, not optional. Chapters 05 and 06 build the resilience patterns.

Those three asymmetries also set the chapter order: the GPU-aware K8s substrate first (01), the inference-shaped scaler on top of it (02), the sharing modes that change how many replicas fit per GPU (03), the cold-start / warm-pool strategy that bounds scale-up (04), the spot patterns that bound scale-down's cost (05), the multi-region / multi-provider edge that bounds blast radius (06), and the boundary to the generic-K8s peer that keeps this module's scope honest (07).

## Boundary against `ai-infra-engineer` (level 20)

The generic Kubernetes substrate — a cluster, a `Deployment`, a `Service`, a `HorizontalPodAutoscaler` on `cpu`, a `ClusterAutoscaler` on generic VMs, a Terraform module that provisions the VPC, IAM for a cluster, a Helm chart that templates a release — lives at [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning). If your diagnosis lands on *"the pod won't schedule because the CPU request is too high,"* *"the Helm release's values template is wrong,"* *"the cluster's node pools don't match the terraform state,"* or *"the HPA on CPU utilisation is misconfigured in a way that would be wrong for any workload"* — that is the level-20 track's scope. Go there first.

The **GPU-specific extensions** are this module:

- `nvidia.com/gpu` as a schedulable resource (device plugin and GPU Operator installation).
- A GPU-aware node autoscaler that selects the right SKU and respects the atomic shape of a TP group.
- Taints / tolerations / node labels / affinity that keep CPU pods off GPU nodes and route the right replica to the right GPU SKU.
- A `RuntimeClass` that runs the pod under `nvidia-container-runtime` with the right device-plugin injection.
- The HPA / KEDA scaler on inference-specific signals (not the generic HPA on CPU).
- The MIG / MPS / DRA sharing modes that are GPU-specific.
- Cold-start and warm-pool shapes that are specific to large-model-load timings.
- Spot patterns that account for the GPU-preemption notice window and in-flight-request drain.
- Multi-region / multi-provider patterns that respect the GPU-availability asymmetry across regions and clouds.

Rule of thumb: *if the pod would configure the same way for a Flask app, that is the level-20 peer. If the pod configures that way because of the GPU, the HBM, the TP group, the cold-start, or the preemption notice, that is this module.*

Chapter 07 formalises the boundary with a mapping table and a hand-off list.

## Boundary against modules 102, 103, 104, 107

The engine, the scheduler, the KV allocator, and the parallelism topology are *inside* a replica. This module is about *how many replicas, on what hardware, under what signal, with what preemption discipline*. The replica's internal behaviour (chapter 102–107) is a given; this module's levers are external to it. Specifically:

- **Module 102** teaches engine startup — the thing cold-start measures. Chapter 04 reuses the warm-up protocol module 102 names, and treats the time-to-ready as a hard input to the warm-pool sizing.
- **Module 103** teaches scheduler metrics — queue depth, in-flight requests, prefill-TPS, admission reject. Chapter 02 is where those metrics become scaler inputs.
- **Module 104** teaches the KV cache's HBM footprint. Chapter 03's "when not to share" argument and chapter 05's checkpointed-KV argument both lean on module-104 math.
- **Module 107** teaches TP / PP groups. A replica in this module is a *group*, not a pod. Scaling a replica up means scheduling a whole group atomically; chapter 01's bin-packing section and chapter 05's drain section both respect that.

## Boundary against module 110

Module 110 is the observability and SLO methodology that *consumes* this module's scaler signals and *grades* this module's fleet decisions. Chapter 02's scaler-signal set is a subset of chapter-110's metric set; chapter-110's SLO math (error budget, burn rate) is the authority on *whether* to scale at all. The split: this module decides *the mechanism and the policy*; module 110 decides *the SLO contract and the error-budget-driven thresholds*.

## Chapters

1. [**GPU-Aware Kubernetes Serving — Operator, Device Plugin, DCGM, Runtime Class, Node Autoscaler**](01-gpu-aware-kubernetes-serving.md) — the GPU-specific substrate on top of generic Kubernetes. NVIDIA GPU Operator installation, device-plugin injection, DCGM-exporter wiring, `RuntimeClass`, taints / tolerations / node labels, Karpenter / Cluster Autoscaler for GPU bin-packing, TP-group affinity, and the health checks that confirm the substrate is wired right before any inference pod lands.
2. [**HPA and KEDA on Inference-Specific Custom Metrics — Signals, Thresholds, Hysteresis**](02-hpa-keda-on-inference-metrics.md) — why CPU / GPU-utilisation is the wrong signal; the four inference signals that are right (queue depth, in-flight, prefill-TPS, TTFT P99); the Prometheus Adapter / KEDA / Keda-Prometheus bridge; scale-up vs. scale-down hysteresis for a 2-minute-boot replica; the alert set that catches a mis-tuned scaler before an SLO breach.
3. [**GPU Sharing — MIG, MPS, DRA, and When Not to Share**](03-gpu-sharing-mig-mps-dra.md) — MIG hard partitioning (A100 / H100 / H200 only, hardware-isolated slices), MPS soft sharing (one GPU, N processes, no isolation), DRA (Dynamic Resource Allocation — flexible per-request scheduling, Kubernetes-native, moving to GA across releases), and the three regimes where NOT sharing wins (large-model latency-critical, noisy-neighbour-sensitive tenants, HBM-starved topology).
4. [**Cold-Start and Warm-Pool Strategy — Scale-to-Near-Zero for Large Models**](04-cold-start-and-warm-pools.md) — what makes an inference pod's cold-start long (image pull, weight pull, load, CUDA-graph warm-up, prefix-cache warm, speculative-decoder-warm); when Knative Serving's scale-to-zero is right (small / fast-booting models) and when its scale-up budget is too short (large models); the custom-warmer pattern (pre-pulled image, pre-staged weights on a shared tier, pre-warmed replica held hot behind the scaler); the warm-pool sizing math.
5. [**Spot Resilience — Preemption Notice, Drain, Checkpointed KV, Cross-Region Failover**](05-spot-resilience-and-preemption.md) — the GPU-spot preemption notice window (~30s on major clouds; verify against the current docs for the exact provider), the in-flight-request drain protocol (stop accepting new, let the batch finish, hand-off to a replacement), when KV checkpointing pays and when it does not, the failover routing that catches the request a drained replica would have served.
6. [**Multi-Region and Multi-Provider Resilience — Global LB, Provider Fallback, Cost-vs-Latency Routing**](06-multi-region-and-multi-provider-resilience.md) — DNS / Anycast / global load-balancer at inference-API granularity, the LiteLLM-shape provider-fallback router, cost-vs-latency routing (a cheaper provider at the margin, a safer provider at the SLA tail), and the operational trade-offs (per-provider rate limits, data-residency, cross-region cost, telemetry fragmentation).
7. [**Boundary to the Level-20 Base Track — What This Module Owns and What It Defers**](07-boundary-to-ai-infra-engineer.md) — the mapping table from the generic-K8s / Terraform / cloud base scope of `ai-infra-engineer` (level 20) to the GPU-specific extensions this module adds; the hand-off list for ambiguous cases; the three common cross-boundary mistakes.

## Exercises

Each exercise carries a chapter into a reusable artefact — a deployment, a scaler, a sharing config, a warm-pool runbook, a drain-and-failover drill, a provider-fallback router.

1. [**GPU Operator and DCGM Base Install**](exercises/exercise-01-gpu-operator-and-dcgm-base-install.md) — stand up the GPU Operator (device plugin, driver / toolkit DaemonSets, DCGM-exporter, MIG manager) on a GPU-equipped cluster, confirm `nvidia.com/gpu` schedules, land a smoke-test inference pod, and publish the four substrate health checks.
2. [**HPA on Queue Depth and TTFT P99**](exercises/exercise-02-hpa-on-queue-depth-and-ttft-p99.md) — wire the Prometheus Adapter (or KEDA) to a chapter-103 queue-depth metric and a chapter-110 TTFT P99 metric, author the HPA / `ScaledObject` with per-direction hysteresis, and prove it responds correctly to a scripted traffic burst without thrashing on a dip.
3. [**MIG, MPS, and DRA Sharing Walkthrough**](exercises/exercise-03-mig-mps-and-dra-sharing-walkthrough.md) — reshape a GPU node into a MIG partition set, land a replica onto a MIG slice, configure MPS for a small-model co-location, and (where DRA is available on your release) stand up a DRA-based sharing claim; publish the "when NOT to share" cases observed.
4. [**Cold-Start and Warm-Pool Strategy**](exercises/exercise-04-cold-start-and-warm-pool-strategy.md) — measure cold-start for a representative large model end-to-end, author a warm-pool policy that keeps `N_warm` replicas pre-warmed, instrument the scale-up / scale-down budget, and publish the fleet's "when scale-to-zero is right" policy.
5. [**Spot Preemption, Drain, and Failover**](exercises/exercise-05-spot-preemption-drain-and-failover.md) — stand up a spot-backed replica pool, inject a preemption via the cloud's test API or a simulated signal, exercise the drain protocol (stop admission, finish in-flight, cross-replica handoff), and confirm the SLO survives.
6. [**Multi-Region Provider-Fallback Router**](exercises/exercise-06-multi-region-provider-fallback-router.md) — deploy a LiteLLM-shape router in front of two provider backends (self-hosted + external, or two regions), author the fallback policy, inject a region failure, and confirm the latency / error budget hold under fallback.

## Resources

See [`resources.md`](resources.md) for primary / citable references — Kubernetes autoscaling docs, NVIDIA GPU Operator and DCGM references, Karpenter and Cluster Autoscaler docs, Knative Serving documentation, KEDA documentation and scalers reference, Prometheus Adapter reference, MIG / MPS / DRA references, LiteLLM project pointers, cloud-provider spot / preemption docs, and the peer-track pointers.

## Where this fits in the curriculum

- **Before:** [Module 108 — Speculative-Decoding Deployment and Operation](../mod-108-speculative-decoding-ops/README.md). A replica in this module may host a `(target, draft)` pair; chapter 01's bin-packing math and chapter 03's "do not share" case both account for the drafter's HBM budget.
- **After:** [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) (the SLO methodology that grades this module's scaler and resilience decisions), [`mod-113-production-safety-and-rollouts`](../mod-113-production-safety-and-rollouts/README.md) (the canary / blue-green discipline that composes with this module's warm-pool and drain patterns), [`mod-114-cross-functional-inference-leadership`](../mod-114-cross-functional-inference-leadership/README.md) (the build-vs-buy, multi-provider, and vendor-eval decisions this module's chapter 06 feeds into).
- **Base-track hand-off:** every chapter in this module names the explicit level-20 cutoff where you stop writing GPU-aware extensions and route to [`ai-infra-engineer-learning`](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning). Chapter 07 formalises the full mapping.
