# Chapter 03 — GPU Sharing: MIG, MPS, DRA, and When Not to Share

> An H100 at $2–4/hr full-time is wasted on a 1B model that never saturates it. Three mechanisms let more than one workload share one GPU — hardware-isolated MIG slices, process-shared MPS, and Kubernetes-native DRA claims. Each has a different isolation guarantee and a different operational envelope. The hard part is not any one of them; it is recognising the regime where sharing wins and the regime where it quietly erodes the SLO.

## Motivation

A platform team runs three workload shapes on one GPU pool:

- **Primary:** Llama-3.1-70B serving at TP=4 on p5 nodes. Latency-critical. Each replica owns four H100s outright.
- **Secondary:** a fleet of 7B embedding, re-rank, and small-LLM workloads. Each one uses maybe 20 GiB of HBM and nowhere near one full GPU's compute. They currently each take one H100; GPU utilisation sits at 15–30 %.
- **Tertiary:** burst-of-small-model batch workloads. Spiky; one-off; small HBM footprint. Currently queued because there's never enough free capacity.

The team's three options:

1. **Co-locate the small models with MPS** on the same H100s — one H100 runs four 7B replicas concurrently. $4/hr instead of $16/hr for the same throughput.
2. **Partition H100s into MIG slices** so each small model gets a hardware-isolated piece of a GPU — the 7B replica gets 1g.10gb or 2g.20gb (an H100 has configurable MIG profiles); batch burst gets a different slice.
3. **DRA (Dynamic Resource Allocation)** — express the resource claim per-request-shape in the Kubernetes API, let the scheduler decide which GPU and which sharing mode.

But the primary workload is a trap: the moment someone co-locates even a small inference pod on a 70B's H100, the 70B's TTFT P99 doubles. The right answer is a hybrid — the 70B pool does not share; the small-model pool does. Picking the mechanism is only half the problem; naming the regime where NOT to share is the other half.

By the end of this chapter you can:

- State what MIG, MPS, and DRA are at operator depth — the isolation guarantee, the engine-side support, and the operational cost of each.
- Deploy each one on a GPU node and verify the pod admitted to the right slice / process / claim.
- Reason about which workload mix justifies sharing and which doesn't — HBM-starved, latency-critical, noisy-neighbour-sensitive, multi-tenant isolation.
- Compose chapter 01's substrate with sharing — the GPU Operator's MIG manager, the device-plugin's MPS and MIG strategies, DRA's `ResourceClaim` shape.

<!-- needs-research: DRA (Dynamic Resource Allocation) is moving between Kubernetes alpha / beta / GA across releases; NVIDIA's DRA driver (`k8s-dra-driver`) is also evolving. The specific API shapes and CRD versions in this chapter should be verified against the Kubernetes release the fleet runs and NVIDIA's DRA driver's release notes. -->

## The three mechanisms at a glance

| Mechanism | Isolation | Granularity | When it fits |
|---|---|---|---|
| **MIG** (Multi-Instance GPU) | Hard — hardware-partitioned, separate SM / HBM / L2 / bandwidth | Fixed set of profiles (A100: 1g.5gb, 2g.10gb, 3g.20gb, 4g.20gb, 7g.40gb; H100 similar with 80GB variants) | Multi-tenant with strict isolation; small LLMs; batch |
| **MPS** (Multi-Process Service) | Soft — processes share one CUDA context on one GPU; no HBM isolation | Flexible per-process | Co-tenant trusted workloads; latency-tolerant; small HBM footprints |
| **DRA** (Dynamic Resource Allocation) | Variable — depends on the DRA driver's partitioning | Per-`ResourceClaim`, scheduler-driven | Fleet-wide flexible scheduling; mixed workload shapes |

### Current support status

- **MIG** — generally available on A100, H100, H200, and newer datacenter GPUs. Managed by the GPU Operator's MIG manager via a `nvidia.com/mig.config` node label. Not available on consumer / older datacenter cards (V100, T4, A10, L4, L40 series — verify your SKU).
- **MPS** — generally available since CUDA 7.5. Device plugin supports an `mps` *sharing strategy*. Not recommended as a default on multi-tenant clusters because of the lack of hard isolation.
- **DRA** — Kubernetes-side moved from alpha to beta over recent releases and is on a GA path; NVIDIA's `k8s-dra-driver` is the official driver. Verify current status against the pinned Kubernetes release and the NVIDIA DRA driver's release notes.

<!-- needs-research: DRA status in the pinned Kubernetes version. KEP-4381 (Dynamic Resource Allocation) covers the API; the NVIDIA DRA driver repo is at <https://github.com/NVIDIA/k8s-dra-driver>. The specific CRD versions (`resource.k8s.io/v1beta1` vs `v1`) and the GA timeline matter for whether the examples below are literally runnable on your cluster. -->

## MIG — hardware-partitioned slices

### What it is

MIG partitions one A100 / H100 / H200 into up to 7 (A100) or 7 (H100) hardware-isolated instances. Each MIG instance is a separate CUDA device as far as the kernel is concerned — separate SMs, separate HBM, separate L2, separate memory bandwidth. Workloads on different MIG instances cannot interfere with each other's performance except through shared cross-node PCIe / NVLink paths.

The MIG *profiles* are fixed:

- **A100 80GB** profiles: `1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`. (`Ng.XXgb` = N GPU slices with XX GiB HBM.)
- **H100 80GB** profiles: `1g.10gb`, `1g.20gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`. (Verify exact profiles against the current H100 reference.)

A single GPU can host *only one layout at a time* — mixing `1g.10gb` and `3g.40gb` on the same card requires one of the predefined profile sets, not an arbitrary combination.

### What it needs

- A100 / H100 / H200 (or newer datacenter card). Not consumer / L4 / L40 / T4.
- MIG mode enabled on the node (requires a GPU reset; the GPU Operator's MIG manager handles it).
- The device plugin configured with the `mixed` or `single` MIG strategy.
- A node label and taints selecting the right slice shape per pod.

### Deployment pattern

**Step 1 — enable MIG mode on the node:**

```yaml
# Node label (set by GPU Operator's MIG manager when nvidia.com/mig.config is set)
labels:
  nvidia.com/mig.config: all-1g.10gb       # or all-2g.20gb, all-3g.40gb, mixed, etc.
```

The GPU Operator's MIG manager reconciles the profile on the node — it drains, resets, reconfigures, un-drains. There is downtime; schedule it.

**Step 2 — the device plugin advertises MIG slices.** With the `mixed` strategy, the kubelet advertises per-profile resources:

```
nvidia.com/mig-1g.10gb: 7
```

**Step 3 — the pod requests the slice:**

```yaml
spec:
  containers:
  - name: embed
    image: embed-server:<tag>
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1
```

### Operational trade-offs

- **Hard isolation.** A noisy neighbour cannot steal SMs or HBM bandwidth. This is MIG's defining feature.
- **Fixed profiles.** You cannot ask for "0.4 of an H100" — you ask for `1g.10gb` (one slice, 10 GiB) or `2g.20gb` (two slices, 20 GiB). HBM-wise this is a hard ceiling.
- **Reset to reconfigure.** Changing MIG profile requires a GPU reset. In practice, you set the profile per-node-group and avoid in-place changes.
- **No NVLink peer-to-peer across MIG slices.** Multi-GPU workloads cannot span MIG instances; MIG is for *fitting more small workloads* per GPU, not for *shrinking a large workload* across slices.
- **Draft-model co-location.** For a draft-target (mod 108) deployment, MIG is a specific no — the draft needs to live on the target's HBM and share bandwidth with it. A MIG split would land the draft in isolation from the target, defeating the point.

### When MIG wins

- **Multi-tenant small models.** Different tenants' 7B / 13B models on different MIG slices on the same H100.
- **Mixed batch and interactive.** A `1g.10gb` for interactive, a `6g.70gb` for the batch workload — hard isolation keeps the interactive's TTFT predictable.
- **Reserved capacity.** Compliance / tenant-isolation requirement; MIG's hardware boundary is auditable in a way MPS isn't.

### When MIG loses

- **Latency-critical big models.** A 70B replica wants a whole GPU (and its NVLink to peers); MIG profiles cap below that.
- **Workloads whose HBM demand varies.** MIG profiles are fixed; a workload that sometimes wants 10 GiB and sometimes 40 GiB cannot stretch into another slice.
- **Fleets with lots of churn.** Reconfiguring MIG requires a reset; a fleet that changes shape weekly spends weekly downtime on resets.

## MPS — process-shared GPU

### What it is

CUDA MPS (Multi-Process Service) runs a background server (`nvidia-cuda-mps-control`) on the node that mediates concurrent CUDA contexts from multiple processes onto one GPU. The GPU appears to each process as if it owned the device; MPS multiplexes kernel launches onto a single CUDA context under the hood.

Reference: <https://docs.nvidia.com/deploy/mps/>.

### What it needs

- The GPU Operator's device plugin configured with the MPS sharing strategy (`sharing.mps.replicas: <N>`).
- Workloads that trust each other — MPS does not provide HBM isolation; a runaway allocation in one process can OOM the others.
- Workloads that are latency-tolerant — MPS reduces context-switch cost but does not eliminate contention.

### Deployment pattern

Configure sharing on the device plugin via a `ConfigMap` the GPU Operator reads:

```yaml
# nvidia-device-plugin-config
version: v1
sharing:
  mps:
    renameByDefault: false
    resources:
    - name: nvidia.com/gpu
      replicas: 4           # one physical GPU appears as four schedulable "gpu" slots
```

Pods now request `nvidia.com/gpu: 1`, and up to four such pods land on one physical GPU — all served through MPS.

### Operational trade-offs

- **No HBM isolation.** Processes share one physical HBM; one OOMing the device OOMs all.
- **Soft compute isolation.** MPS splits SM time across processes based on each process's work; a saturating kernel from one process does slow the others. There is a `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` cap that limits a process's SM share — useful but weak compared to MIG.
- **Lower overhead than context-switching.** Two processes without MPS pay context-switch time; with MPS they interleave. For small, latency-sensitive kernels, the speedup is real.
- **No MIG + MPS on the same GPU** — pick one.

### When MPS wins

- **Small models with low per-request HBM demand** co-located within one tenant / team's trust boundary. Four 1B embedding processes on one H100 is a common pattern.
- **Latency-tolerant batch alongside interactive on the same tenant.** A batch inference pod and an interactive pod from the same team can share with MPS; the SM-share cap keeps the batch from monopolising.
- **Workloads whose GPU demand is bursty.** The spikes rarely coincide; MPS multiplexes efficiently.

### When MPS loses

- **Multi-tenant with isolation requirements.** The lack of HBM isolation is disqualifying.
- **Latency-critical workloads.** Even with the SM cap, contention happens — any shared GPU adds variance.
- **Big models.** A 70B replica with MPS peers is a bad idea for the same reasons as MIG, and worse — no hard isolation means an OOM elsewhere kills your 70B.

## DRA — Kubernetes-native dynamic claims

### What it is

Dynamic Resource Allocation (KEP-4381, <https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/>) extends Kubernetes resource scheduling with a *claim* model — a pod references a `ResourceClaim` that describes the resource it wants in richer terms than `nvidia.com/gpu: 1`. A per-vendor DRA driver (NVIDIA's is <https://github.com/NVIDIA/k8s-dra-driver>) implements the claim's semantics and binds the pod to a specific device / partition.

### What it needs

- Kubernetes version supporting the DRA API at the required maturity (alpha / beta / GA — verify against your release).
- The NVIDIA DRA driver installed on the cluster.
- Pods updated to reference `resourceClaims` rather than (or alongside) `nvidia.com/gpu` requests.

### Deployment pattern

```yaml
# Resource claim template
apiVersion: resource.k8s.io/v1beta1
kind: ResourceClaimTemplate
metadata:
  name: gpu-1g-10gb
spec:
  spec:
    resourceClassName: gpu.nvidia.com
    parametersRef:
      apiGroup: gpu.resource.nvidia.com
      kind: GpuClaimParameters
      name: one-1g-10gb-slice

---
# Claim parameters (vendor-specific CRD; shape per NVIDIA DRA driver release)
apiVersion: gpu.resource.nvidia.com/v1alpha1
kind: GpuClaimParameters
metadata:
  name: one-1g-10gb-slice
spec:
  count: 1
  split: 1g.10gb

---
# Pod
apiVersion: v1
kind: Pod
spec:
  runtimeClassName: nvidia
  resourceClaims:
  - name: gpu
    resourceClaimTemplateName: gpu-1g-10gb
  containers:
  - name: embed
    image: embed-server:<tag>
    resources:
      claims:
      - name: gpu
```

The API shape above is **illustrative** — the exact CRD names and fields are driver-release-specific; always verify against the pinned driver's examples.

<!-- needs-research: the NVIDIA DRA driver's CRD shape (ResourceClaimTemplate, GpuClaimParameters, or whatever the current release calls them) changes release-over-release. See <https://github.com/NVIDIA/k8s-dra-driver> for the current examples. -->

### Operational trade-offs

- **Richer claim semantics.** A pod can ask for "an H100 with ≥ 40 GiB free" or "an MIG slice of size 1g.10gb or 2g.20gb" — expressions the device plugin cannot make.
- **Scheduler-driven.** The DRA driver participates in the scheduler's bind decision; the pod lands where the claim says it can.
- **Vendor-specific CRDs.** The NVIDIA DRA driver's CRDs don't match AMD's or Intel's; portability is at the Kubernetes API, not the driver's parameters.
- **Still-evolving maturity.** Pin the Kubernetes and driver versions; this is the layer where the fewest fleets have long production experience as of this writing.

### When DRA wins

- **Multi-shape fleets** where some pods want full GPUs, some want MIG slices, some want MPS — DRA expresses the mix.
- **Dynamic sharing** — the shape of what's shared can change per-request based on claim parameters.
- **Future-leaning stacks** — DRA is the direction the Kubernetes resource API is heading. If you're building new infra and willing to stay close to the release train, this is where to invest.

### When DRA loses

- **Mature fleets on the device-plugin path.** If MIG + device plugin + MPS strategy already covers your workload, DRA adds complexity without a corresponding gain.
- **Rigid, well-known workload shapes.** If every workload wants the same thing, device-plugin advertising is enough.

## When NOT to share

Sharing is cheap when it works and extremely expensive when it doesn't. Three regimes where NOT sharing is the right call:

### Regime 1 — latency-critical large models

A 70B / 405B / Mixtral replica saturates HBM bandwidth and SM capacity on its own. Co-locating another workload — even a tiny one — adds contention on L2, HBM channels, and NVLink traffic. TTFT P99 variance spikes; the SLO breaks. The 70B pool should request **whole GPUs** (`nvidia.com/gpu: 4` for a TP=4 replica) with **no sharing**.

For draft-target speculative (mod 108), *the draft already shares the target's HBM and bandwidth*; adding external co-tenants on top of that doubles the contention. The target + draft is already a sharing arrangement.

### Regime 2 — noisy-neighbour-sensitive multi-tenant

If tenant A's latency SLO depends on tenant B not spiking, MPS is disqualified (no isolation). MIG works but caps HBM per tenant; DRA with hard-partition claims works but requires the DRA driver on your release. The *failure mode* of sharing without isolation is a per-tenant SLO breach that pages during tenant B's traffic spike; it is operationally unforgivable.

If multi-tenant is on the roadmap, build the sharing layer with hard isolation (MIG preferred) or don't share.

### Regime 3 — HBM-starved topology

A 70B TP=4 replica on 4× H100 80GB uses ~35 GiB per rank for weights, 2 GiB per rank for a draft, and the rest for KV + activations + reserves. If the KV pool is already tight — chapter 104's math — there is no HBM headroom to share. A co-tenant's weights eat the KV pool; the fleet's concurrency drops; throughput falls more than the sharing saves.

Chapter 01's substrate exposes `DCGM_FI_DEV_FB_USED` and `DCGM_FI_DEV_FB_FREE`. Before adopting sharing, verify there is enough HBM headroom for the co-tenant *and* for the primary workload's full working set. The math is unforgiving.

### Regime 4 — collective-bandwidth-sensitive workloads

TP collectives on a 70B replica saturate NVLink. A co-located workload issuing its own collectives on the same NVLink group competes for bandwidth. The 70B's TPOT rises; the co-tenant's work slows. MIG partitions remove NVLink peer-to-peer between slices, so this isn't a MIG problem; MPS co-location on a TP replica is a classic footgun.

## Composing with chapter 01's substrate

The sharing modes compose with the chapter 01 substrate specifically:

- **GPU Operator** installs and manages MIG, device plugin (with MPS strategy), and (in current releases) the DRA driver.
- **Node labels** (`nvidia.com/mig.config`, `nvidia.com/gpu.sharing`) tell the Operator which strategy to apply per node.
- **Taints and tolerations** keep workloads that *don't* share off nodes that do, and vice versa — a 70B replica should not land on a MIG-partitioned node, and a 1B embedding should not land on an un-partitioned H100.
- **Node pools** separate the shared and un-shared pools; chapter 01's Karpenter / Cluster Autoscaler config has one pool per policy. The usual shape:
  - `pool-h100-full` — un-partitioned, taint `inference-tier=latency-critical`.
  - `pool-h100-mig` — partitioned (e.g. `all-1g.10gb` or `all-2g.20gb`), taint `inference-tier=small-model`.
  - `pool-h100-mps` — MPS-shared, taint `inference-tier=batch-tolerant`.

Chapter 02's HPA signals are *per replica*, not per GPU. Scaling with sharing enabled: the HPA scales replicas; the device plugin's advertised capacity (slices or MPS slots) determines how many replicas fit per GPU. The compose: HPA on queue-depth → scale pods → autoscaler on `nvidia.com/mig-1g.10gb` or MPS slots → scale nodes.

## A worked example — hybrid fleet layout

```
Node pool: pool-h100-full
  SKU: p5.48xlarge (8× H100 80GB)
  Sharing: none
  Taint: inference-tier=latency-critical:NoSchedule
  Workloads: 70B, 405B, large-replica TP groups

Node pool: pool-h100-mig
  SKU: p5.48xlarge (8× H100 80GB), MIG-reconfigured to all-2g.20gb (4 slices per GPU, 32 slices per node)
  Sharing: MIG
  Taint: inference-tier=small-model:NoSchedule
  Workloads: 7B, 13B, embedding, re-rank — one slice per replica

Node pool: pool-h100-mps
  SKU: p5.4xlarge or similar (one H100 per node), MPS replicas: 4
  Sharing: MPS
  Taint: inference-tier=batch-tolerant:NoSchedule
  Workloads: trusted-tenant batch; low-latency-tolerance embedding bursts
```

Three pools, three sharing modes, one node autoscaler per pool, chapter-02 HPAs per workload. This is the common shape at mid-fleet size.

## Common deployment errors

### Error 1 — MIG profile change without node drain

Someone changes `nvidia.com/mig.config` on a node that is actively serving. The MIG manager drains and resets. Any in-flight request is killed. Fix: schedule MIG changes with the drain discipline from chapter 05 (and the deployment safety from mod 113).

### Error 2 — MPS as default on multi-tenant

Someone enables MPS on all GPUs for cost reasons. Tenant A OOMs the GPU during a bad request; tenant B's replica dies with it. Fix: MPS only within trust boundaries; between tenants, use MIG or no sharing.

### Error 3 — DRA claim with no fallback

A pod requires a specific DRA claim shape; no node can satisfy; the pod sits Pending. Fix: either widen the claim (OR between slice sizes), or ensure the node pool has nodes that match.

### Error 4 — co-locating a draft and a tenant

Mod 108's draft already shares the target's HBM; adding another co-tenant on top with MPS or DRA creates a three-way contention the SLO can't absorb. Fix: don't share on draft-target replicas.

### Error 5 — MIG on a non-MIG card

Pod requests `nvidia.com/mig-1g.10gb` on a cluster that has only L4 / L40 cards. The pod is Pending forever. Fix: check SKU capability; MIG is datacenter-GPU-only.

### Error 6 — sharing + full-GPU in the same pool

A pool that advertises both `nvidia.com/gpu: 8` (full) and `nvidia.com/mig-2g.20gb: 32` (sliced) on the same card is misconfigured — the device plugin's strategy is single or mixed, not both. Fix: separate pools per sharing mode.

### Error 7 — ignoring NUMA / topology on a shared node

On an 8-GPU node with MIG, slices on different GPUs have different NUMA / NVLink distance. A workload latency-sensitive to the slice placement must bind to specific GPUs. Fix: use topology-aware scheduling (NFD, Kubernetes topology manager).

## The boundary to the level-20 peer

This chapter does not teach:

- **Fundamentals of the device plugin API** — the sidecar protocol, the gRPC shape. Generic K8s extension topic; `ai-infra-engineer` level 20.
- **DRA KEP mechanics** at the Kubernetes-contributor level. Also level 20 / upstream.
- **GPU-hardware internals of SMs, HBM, and NVLink** — the performance peer at level 35 teaches the hardware; this chapter teaches the operator-level use.

This chapter *does* teach: the three sharing modes at operator depth, the composition with chapter 01's substrate, the regimes where NOT sharing wins, and the common deployment errors.

## Summary

- Three sharing modes with distinct guarantees: **MIG** (hardware-isolated hard partitions, datacenter-GPU-only), **MPS** (soft process-sharing, no isolation), **DRA** (Kubernetes-native claims, moving to GA, future-leaning).
- Pick MIG when isolation matters and the workloads fit the fixed profiles. Pick MPS when workloads trust each other and the latency variance is tolerable. Pick DRA when you want flexible per-claim scheduling and your Kubernetes release supports it.
- Do NOT share when: the workload is latency-critical and large (70B / 405B), the tenants are mutually untrusted, HBM is already tight, or TP collectives already saturate the GPU's bandwidth.
- Compose with chapter 01: node pools per sharing mode, labels / taints that route the right workload to the right pool, GPU Operator managing the MIG manager and device plugin strategy.
- Chapter 02's HPA scales replicas; the device plugin's advertised slice count scales *across nodes*.
- Draft-target speculative replicas (mod 108) already share HBM between draft and target; adding external co-tenants on top is a chapter 03 anti-pattern.
- Generic Kubernetes extension / resource API fundamentals route to level 20; GPU-hardware details to the performance peer; this chapter owns the operator-level sharing decision.
