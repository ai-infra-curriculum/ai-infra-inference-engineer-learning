# Chapter 07 — Boundary to the Level-20 Base Track: What This Module Owns and What It Defers

> Every chapter in this module has named a cut at the edge: "generic Kubernetes / Terraform / cloud base lives at `ai-infra-engineer` level 20." This chapter formalises that boundary into a mapping table, a hand-off list for ambiguous cases, and the three common cross-boundary mistakes. The goal is that a reader of this module can say — without hesitation — whether a given symptom belongs to this track's troubleshooting path or the level-20 peer's.

## Motivation

A platform team gets paged: *"The inference pod won't schedule — `0/12 nodes available: insufficient nvidia.com/gpu`."* Where does triage begin?

- If the answer is "the device plugin isn't running" — that's chapter 01 of *this module*.
- If the answer is "the cluster has no GPU nodes because the node group was mis-provisioned in Terraform" — that's the level-20 peer's.
- If the answer is "the pod's request exceeds any single node's capacity" — chapter 01 / the level-20 peer / the inference-engineer depending on *why* (TP-group size wrong → this module; typo in `nvidia.com/gpu: 40` where you meant `4` → level-20 generic fix).

The team wastes 20 minutes because no one knows who owns the first look. A clear boundary table — the thing this chapter publishes — would have routed the triage to "run the chapter-01 substrate health checks; one of them will light up red" in two minutes.

By the end of this chapter you can:

- Read a triage pager message and know within seconds whether to route to this track's chapter or the level-20 peer's.
- State for any cluster-level concern (networking, storage, identity, autoscaling base, Helm / Terraform discipline) which track owns the debugging.
- Name the three common cross-boundary mistakes and the tells that signal each.
- Write the hand-off bundle that transfers an ambiguous issue to the right peer without a dropped baton.

## The mapping — what's level 20, what's this module

### Generic Kubernetes (level 20)

The following fall under `ai-infra-engineer` level 20 regardless of whether the workload is GPU:

| Concern | Who owns |
|---|---|
| Kubernetes cluster provisioning (kubeadm, EKS / GKE / AKS install, cluster upgrade) | Level 20 |
| VPC / networking / subnets / CIDR planning / NAT / egress | Level 20 |
| IAM, OIDC, service accounts, RBAC | Level 20 |
| Helm / Kustomize / templating fundamentals | Level 20 |
| Terraform modules for the cluster and node groups | Level 20 |
| Generic Deployments, Services, ConfigMaps, Secrets | Level 20 |
| Generic `HorizontalPodAutoscaler` on CPU / memory | Level 20 |
| Generic `ClusterAutoscaler` on standard VMs | Level 20 |
| Generic Ingress (NGINX / ALB controller / GCE ingress) | Level 20 |
| Generic Pod Security, Network Policy | Level 20 |
| Cluster upgrade / drain discipline (for a node not holding a GPU replica) | Level 20 |
| Generic storage (PV / PVC fundamentals, StorageClasses) | Level 20 |
| MetalLB / cloud load balancer setup | Level 20 |
| Day-2 operations for an unchanged chapter-01 cluster | Level 20 |

### GPU-specific extensions (this module)

The following are inference-shape — this module owns them:

| Concern | Chapter |
|---|---|
| NVIDIA GPU Operator install and configuration | 01 |
| Device plugin advertising `nvidia.com/gpu`, MIG / MPS strategies | 01, 03 |
| DCGM-exporter wiring and inference-relevant metrics | 01 |
| `RuntimeClass` for GPU pods (`nvidia`) | 01 |
| Taints / tolerations / node labels specific to GPU SKUs | 01 |
| Karpenter / Cluster Autoscaler GPU node-pool configuration (SKU-aware, TP-group-aware) | 01 |
| TP / PP group scheduling (LeaderWorkerSet, gang scheduling) | 01 |
| HPA / KEDA on inference-specific signals (queue depth, in-flight, prefill-TPS, TTFT P99) | 02 |
| Scale-up / scale-down hysteresis for slow-boot large models | 02 |
| MIG partitioning and MIG profile reconfiguration | 03 |
| MPS sharing strategy | 03 |
| DRA (Dynamic Resource Allocation) claims for GPUs | 03 |
| "When NOT to share" decision for inference workloads | 03 |
| Cold-start anatomy and mitigations specific to large model loading | 04 |
| Warm-pool pattern (separate Deployment, always-on) | 04 |
| Knative / KServe fit rule for inference | 04 |
| Weight pre-stage patterns (PVC, shared RWX, baked-in) | 04 |
| Spot preemption drain protocol for an inference replica | 05 |
| KV checkpointing cost/benefit | 05 |
| Diversity + fallback shape of spot GPU pools | 05 |
| Multi-region inference failover routing (streaming-aware) | 06 |
| LiteLLM-shape multi-provider router | 06 |
| Cost-vs-latency routing policy | 06 |
| Data-residency / tenant-isolation considerations in provider routing | 06 |
| Any of the above compositions (chapter-02 scaler + chapter-04 warm pool + chapter-05 spot, etc.) | this module |

### Grey zone — ask both

A handful of concerns straddle the boundary:

| Concern | Resolution |
|---|---|
| Persistent volume for model weights | Generic PVC is level 20; the choice of RWX vs. local, size, pre-stage pattern is chapter 04 |
| Pod disruption budget for an inference Deployment | PDB fundamentals level 20; the specific value for a slow-boot inference replica is chapter 02/04/05 |
| Prometheus install | Installing is level 20 / module 110; the inference-specific metric set and alerts is chapter 02 / module 110 |
| Secrets management for model API keys, HF tokens | Secrets fundamentals level 20; the per-tenant isolation discipline is module 106 |
| Service mesh (Istio / Linkerd / Cilium) | Install and basics level 20; the streaming / long-connection / cross-region implications is chapter 06 |

For grey-zone concerns, run the triage test: *"would a Flask service have the same problem, and would the fix be the same?"* If yes → level 20. If the fix is specifically different because it's GPU inference → this module.

## The triage test

A one-sentence rubric:

> *"If the symptom would occur the same way on a Flask-API deployment with no GPU, and the fix would be the same, that is the level-20 peer. If the symptom or the fix is specifically different because of the GPU, the HBM, the TP group, the cold-start, the KV cache, the SLO shape, or the sharing mode, that is this module."*

Three applications:

### Example 1 — pod `Pending` with `insufficient cpu`

A Flask pod would have the same symptom and the same fix (either lower request or add capacity). Level-20 fix: check the node's allocatable CPU, decide between smaller request and larger node.

### Example 2 — pod `Pending` with `insufficient nvidia.com/gpu`

The symptom only occurs for GPU pods, and the fix depends on GPU specifics (is the device plugin advertising? is the SKU right? is the TP group demanding too much? is MIG partitioning cutting the available slots?). This module's chapter 01 / 03.

### Example 3 — HPA scales too aggressively

If it's a Flask HPA on CPU, the fix is "lower the threshold or extend stabilisation window" — level 20.

If it's an inference HPA on queue depth and the issue is "scale-down is too aggressive because the cold-start is 4 minutes and the window is 60 seconds" — chapter 02 of this module.

### Example 4 — pod `OOMKilled`

If it's CPU memory (the pod's `memory` limit), level 20 fix: raise the limit, check for leaks. If it's HBM (the pod runs out of GPU memory loading weights or during inference), this module (chapter 01's sizing, chapter 03's sharing math, chapter 104's KV budget).

## The hand-off bundle

When routing an issue from this track to the level-20 peer (or the reverse), send:

```
Hand-off Bundle: <summary>
Routing: from <src-track> to <dst-track>

Symptom:
  <one sentence>

Evidence:
  <logs / metrics / kubectl output>

What has been ruled out (from <src-track>'s scope):
  - <item 1>
  - <item 2>

Suspected cause (requires <dst-track> depth):
  <one paragraph>

Reproduction:
  <steps, if repro is possible>

Timeline:
  <when it started, how often>
```

Specific fill-ins per direction:

### This module → level-20 peer

If you've traced an issue to a Kubernetes-level concern that isn't inference-shaped:

- Chapter 01 example: *"The GPU Operator is healthy; the device plugin advertises `nvidia.com/gpu`; a smoke-test pod schedules and runs; but the ingress returns 502 on the inference Service. The chapter-01 substrate is confirmed-good. The problem is between the Service and the ingress — RBAC, Network Policy, or ingress-controller config. Routing to level-20 peer."*

### Level-20 peer → this module

If a level-20 peer has confirmed the cluster basics are working but the inference workload specifically isn't:

- Example: *"Cluster, networking, RBAC, Helm chart, ingress all verified against a Flask pod serving on the same node — works fine. The inference Deployment with `nvidia.com/gpu: 4` sits `Pending`. Routing to inference-engineer for chapter-01 substrate check and TP-group sizing."*

The bundle makes the hand-off *resumable* — the next peer doesn't re-walk what's already been ruled out.

## The three common cross-boundary mistakes

### Mistake 1 — "it's a Kubernetes problem" when it's a GPU problem

A team's pod won't schedule; they escalate to the level-20 peer. The level-20 peer finds the node pool is healthy, the Pods are not CPU-constrained, the RBAC is fine. Two hours later someone in this track looks and sees `nvidia.com/gpu: 4` on a pool of 1×GPU nodes. The level-20 peer didn't know to look at the TP-group sizing; this track's check (chapter 01's health checks) would have caught it in minutes.

**Fix:** triage with the chapter-01 health checks *first* for any inference-shape pod, before escalating.

### Mistake 2 — "it's an inference problem" when it's a Kubernetes problem

A team's HPA isn't scaling. They dig into inference metrics, Prometheus Adapter, custom-metrics API, queue-depth signal — and spend hours. The cause: RBAC on the custom-metrics API server. Any HPA on any custom metric would fail; it's not inference-specific. This track wasted time on chapter 02 debugging when level 20 had the answer in minutes.

**Fix:** when a scaler "doesn't scale," check the meta-metrics (is the HPA reading the custom metric at all?) *before* assuming a chapter-02 signal issue.

### Mistake 3 — architectural confusion

A fleet's design review wants to make the HPA scale down to zero. Someone proposes KEDA with scale-from-zero. Someone else proposes Knative. Someone else proposes a warm pool. They're all solving the same problem at different altitudes; the design meeting bogs down because no one has drawn the decision boundary.

**Fix:** chapter 04's fit rule (scale-to-zero only below 30-second cold-start; warm pool above 2 minutes; Knative / KServe in between with `minReplicas=1`). The decision is made from data, not from architectural preference.

## Compose with the other peers

The level-20 boundary is one of several this track composes against. For completeness:

| Peer | Level | Owns |
|---|---|---|
| `ai-infra-junior-engineer` | 10 | Engineering-craft prerequisites (Git, shell, debugging, basic Linux) |
| `ai-infra-engineer` | 20 | Generic K8s / Terraform / cloud base (this chapter's focus) |
| `ai-infra-mlops` | 25 | CI/CD automation, build / release pipelines |
| `ai-infra-ml-platform` | 30 | Self-service ML platform, model registry |
| `ai-infra-sre` | 30 | SRE methodology, SLO math, incident discipline (module 110 integrates) |
| `ai-infra-security` | 35 (peer) | Inference-edge security (chapter 06's data-residency hand-off) |
| `ai-infra-performance` | 35 (peer) | Kernels, profiling, kernel-authoring craft (mod 102–104 defer kernel work here) |
| `ai-infra-architect` / `ai-infra-team-lead` | 40 | Cross-stack architecture, people management, vendor strategy (chapter 06 flags) |

For module 109 specifically, the active hand-offs are **mostly level 20** (the chapter's whole concern). Chapter 06's data-residency routing touches the security peer; chapter 05's drain protocol is mostly internal to this track (the engine provides the drain primitives).

## A worked triage

The pager fires: *"TTFT P99 > 5 s for 10 minutes on `vllm-llama31-70b` in us-east-1."*

Chapter 02's primary scaler should have already scaled up. Check:

1. Is the HPA at its `maxReplicas`? → yes: capacity ceiling hit. Chapter 02 suggests raising `maxReplicas`; chapter 03 suggests "can we share more"; chapter 04 suggests "can the warm pool absorb more"; chapter 06 suggests "can we spill to another region or provider."
2. Is the HPA *trying* to scale and failing? → yes: scheduling problem. Check chapter 01:
   - Any pods `Pending`? If "insufficient `nvidia.com/gpu`" — the autoscaler isn't provisioning GPU nodes. Check Karpenter / CA logs.
   - If "insufficient CPU" on the GPU node — someone put a CPU workload on the GPU node; this is a *level-20* ** taint/toleration misconfig.
3. Is the HPA reading its signals correctly? → check meta-metrics (chapter 02 alert set). If stale, the issue is upstream (Prometheus / scraper); *level-20* concern on the install.
4. Is spot preemption burning the fleet? → check chapter 05 alerts: the preemption rate. If high, more on-demand capacity is needed in a hurry.
5. Is the region's capacity the problem rather than the pool's? → chapter 06 cross-region; is the other region underutilised?

The pager triage walks chapter 02 → chapter 01 → chapter 05 → chapter 06 in that order, not at random. The level-20 peer gets involved at step 2 if the issue is "a CPU pod stole the GPU node's capacity"; otherwise the whole triage lives in this track.

## Summary

- Generic Kubernetes / Terraform / cloud base — the *substrate common to every workload* — lives at `ai-infra-engineer` level 20.
- The GPU-specific extensions — operator, device plugin, DCGM, MIG / MPS / DRA, GPU-aware autoscaler, inference-shaped HPA, warm pool, spot drain, multi-region routing — are this module.
- The triage test: *would a Flask service have the same problem and the same fix?* Yes → level 20. Specifically different because of the GPU / HBM / TP / cold-start / KV / SLO → this module.
- The three common cross-boundary mistakes: *(1)* K8s-blaming when it's a GPU problem, *(2)* GPU-blaming when it's a K8s problem, *(3)* architectural confusion with no decision rubric. All preventable with the module's chapter-specific fit rules.
- Hand-off bundles make routing between tracks resumable — summary, evidence, what's been ruled out, suspected cause, repro, timeline.
- For inference workloads, triage walks this module's chapters *first* (chapter 02 → 01 → 05 → 06), escalating to level 20 only when the substrate is confirmed-good and the issue is non-inference-specific.
- Compose with the broader peer map — `ai-infra-mlops` (25), `ai-infra-ml-platform` (30), `ai-infra-sre` (30), `ai-infra-security` (35), `ai-infra-performance` (35) — each with its own chapter's hand-off touchpoints.
