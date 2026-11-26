# Exercise 03 — MIG, MPS, and DRA Sharing Walkthrough

**Time:** 3.5 hours · **Output:** one MIG-partitioned node with a scheduled replica, one MPS-shared node with N co-located replicas, one DRA claim (where supported), one "when NOT to share" decision table, one hybrid node-pool topology diagram, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 03](../03-gpu-sharing-mig-mps-dra.md) concrete by **actually deploying the three sharing modes** — MIG (hard), MPS (soft), DRA (claim-driven) — on a real GPU node, landing a small-model replica onto each, and publishing a decision table that names the regimes where your fleet should NOT share. The deliverable is the hybrid-pool topology you can hand to the next tenant that asks "can we put this on a shared slice?"

## Problem statement

Your fleet from exercise 01 has at least one GPU node. Three new workloads land on your plate:

- **Workload A:** A latency-critical 70B replica (chapter 03's Regime 1 — the "do not share" case). You have to tell the tenant this one owns whole GPUs; the exercise is the evidence that backs the decision.
- **Workload B:** A fleet of 1B–7B embedding, re-rank, and small-LLM replicas (chapter 03's MIG / MPS fit cases). You have to decide which of the small-model cases get MIG slices, which get MPS co-location, and which (if any) get DRA claims.
- **Workload C:** A burst-of-batch inference that comes and goes (the DRA fit case — flexible claim-driven scheduling if your cluster version supports it).

The exercise is where you do the deployment one-of-each, measure the isolation / overhead trade-offs, and write the decision table that governs every subsequent tenant intake.

## Prerequisites

- Chapter 03 of this module read. Chapter 01 skimmed (substrate prerequisite).
- **Exercise 01 complete** — GPU Operator installed, MIG manager enabled (if your hardware supports MIG), DCGM scraping Prometheus.
- At least one **MIG-capable datacenter GPU** (A100 / H100 / H200) on your cluster for the MIG walkthrough; if you only have consumer / L4 / L40 / T4 cards, you'll do the MPS and DRA walks only and document the gap.
- Permission to drain a node and reconfigure its MIG profile. MIG mode changes require a GPU reset; coordinate before starting.
- For the DRA walk, a Kubernetes cluster at the version your DRA driver supports — current `k8s-dra-driver-gpu` (<https://github.com/NVIDIA/k8s-dra-driver-gpu>) requires Kubernetes 1.32+, with GA under `resource.k8s.io/v1` on Kubernetes 1.34+. Verify your cluster version before planning the DRA deliverable.
- A small-model inference image (an embedding server, a 1B instruct model, whatever you have on hand) that can run on a `1g.10gb` or `2g.20gb` MIG slice.

## Requirements

Six deliverables.

### Deliverable A — the inventory and target topology

Before touching the cluster, record what you have and what you'll partition:

```
# Sharing Walkthrough Target Topology — <cluster_name>

## GPU inventory
- Node 1 (<SKU, e.g. p5.48xlarge>): 8× H100 80GB (MIG-capable)
- Node 2 (<SKU, e.g. g5.12xlarge>): 1× A10G (NOT MIG-capable; skip MIG; test MPS/DRA only)
- ...

## Target partitioning
- Node 1 GPUs 0..3: full (no sharing) — Workload A pool
- Node 1 GPUs 4..5: MIG `all-2g.20gb` — Workload B (small LLMs, 4 slices per GPU × 2 GPUs = 8 slices)
- Node 1 GPUs 6..7: MPS `replicas: 4` — Workload B (embedding / rerank)
- Node 2 GPU 0: DRA claim-driven — Workload C

## Kubernetes / driver versions
- Cluster: 1.32.x
- GPU Operator: vX.Y.Z (from exercise 01)
- `k8s-dra-driver-gpu`: vX.Y.Z (if attempting DRA)
```

If the target topology differs from what you can actually stand up (e.g. fewer nodes, no MIG hardware), document the deviation here and adjust subsequent deliverables.

### Deliverable B — MIG walkthrough

On a MIG-capable node, enable MIG mode and partition it:

1. **Label the node** with the MIG profile — chapter 03's `nvidia.com/mig.config: all-2g.20gb` (or whatever matches your target). The GPU Operator's MIG manager picks up the label and reconciles.
2. **Wait for the reconcile** — the node drains (if pods are present), the GPU is reset, MIG profile applied, node returns to Ready. This is multi-minute. Monitor via `kubectl describe node <node>` and `kubectl -n gpu-operator logs <mig-manager-pod>`.
3. **Confirm the slices advertise** — `kubectl describe node <node>` should show `nvidia.com/mig-2g.20gb: <count>` in Capacity and Allocatable.
4. **Deploy a small-model replica** requesting the slice:

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: small-llm-mig
   spec:
     replicas: 1
     selector: {matchLabels: {app: small-llm-mig}}
     template:
       metadata: {labels: {app: small-llm-mig}}
       spec:
         runtimeClassName: nvidia
         tolerations:
         - {key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
         containers:
         - name: engine
           image: <small-model-image>
           resources:
             limits:
               nvidia.com/mig-2g.20gb: 1
   ```

5. **Verify** — `kubectl exec` into the pod and run `nvidia-smi`; the output should show a single MIG instance, not the whole GPU.

Attach:

- The node label command / manifest.
- `kubectl describe node <node>` output showing the MIG slices in Allocatable.
- The Deployment manifest.
- The `nvidia-smi` output from inside the pod proving the MIG slice is what it got.
- A DCGM query showing the MIG instance appears as a separate `UUID` in `DCGM_FI_DEV_SM_ACTIVE`.

### Deliverable C — MPS walkthrough

On a (different) GPU node, configure MPS sharing via the device-plugin ConfigMap:

1. **Apply the device-plugin config** — chapter 03's shape:

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: nvidia-device-plugin-config
     namespace: gpu-operator
   data:
     config.yaml: |
       version: v1
       sharing:
         mps:
           renameByDefault: false
           resources:
           - name: nvidia.com/gpu
             replicas: 4
   ```

2. **Reference the config** from the Operator's `ClusterPolicy` and label the MPS-target node — exact mechanism depends on the Operator release; verify against the Operator sharing-strategies docs for the pinned release.
3. **Wait for the device plugin to re-roll** on the node; a GPU on that node should now advertise `nvidia.com/gpu: 4` (where there was previously 1 per physical GPU).
4. **Deploy 4 small-model replicas** each requesting `nvidia.com/gpu: 1` — all four should land on the same physical GPU.
5. **Verify co-location** — `kubectl get pod -o wide` should show all four pods on the same node; `nvidia-smi` on the node (via a privileged DaemonSet or an SSH session) should show four processes sharing one physical GPU.
6. **Measure contention** — drive load through two of the four replicas simultaneously; capture TTFT and throughput on each; note the degradation vs. running one at a time.

Attach:

- The device-plugin ConfigMap.
- `kubectl describe node <node>` showing `nvidia.com/gpu: N × 4` after MPS.
- The four replica Deployments.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory` output from the node showing four MPS clients.
- A short measurement table showing TTFT / throughput for 1 vs. 2 vs. 4 concurrent replicas.

### Deliverable D — DRA walkthrough (or documented skip)

If your cluster supports the NVIDIA DRA driver:

1. **Install `k8s-dra-driver-gpu`** per the quickstart (<https://github.com/NVIDIA/k8s-dra-driver-gpu/tree/main/demo/specs/quickstart>). Verify the API version your cluster uses — current driver releases on Kubernetes 1.34+ use upstream `resource.k8s.io/v1` with `DeviceClass` / `ResourceClaim` / `ResourceClaimTemplate`; pre-1.34 clusters may still be on `v1beta1`.
2. **Author a `ResourceClaimTemplate`** for a representative claim shape — e.g. "one full H100" or "one MIG slice of profile 2g.20gb." Use the current driver quickstart specs as the template; chapter 03 flagged the vendor CRDs as evolving.
3. **Deploy a pod** that references the `ResourceClaimTemplate` via `resourceClaims` (chapter 03's shape).
4. **Verify binding** — the DRA scheduler extension picks the device; `kubectl describe pod` shows the claim resolved; `nvidia-smi` in the pod shows the expected device.
5. **Try a claim the cluster cannot satisfy** — e.g. a claim for a MIG profile not present; verify the pod sits Pending with a readable reason (DRA binding failures should be observable via `kubectl describe` on the `ResourceClaim`).

If your cluster version is below the DRA driver's minimum (or the policy on your cluster forbids CRD installs), **document the skip explicitly**:

```
Deliverable D — DRA walkthrough: SKIPPED
Reason: cluster at K8s 1.30; `k8s-dra-driver-gpu` requires 1.32+.
Plan: revisit after next cluster upgrade; chapter-03 regime-match analysis in Deliverable E still applies.
```

Skipping is acceptable; vagueness is not.

### Deliverable E — the "when NOT to share" decision table

A one-page table that maps every chapter-03 regime to a decision for your fleet, grounded in the observations from Deliverables B / C:

```
| Workload shape                               | Share? | Mode | Reason (link to chapter-03 regime) |
|---------------------------------------------|--------|------|------------------------------------|
| 70B / 405B / Mixtral latency-critical        | NO     | —    | Regime 1 (HBM+NVLink; co-tenant doubles TPOT variance) |
| 7B–13B single-tenant latency-tolerant        | YES    | MIG `2g.20gb` | 4× density; hardware isolation |
| 1B embedding / rerank, same-tenant trust     | YES    | MPS `replicas: 4` | Bursty; low-HBM; trusted |
| Multi-tenant small models (strict isolation) | YES    | MIG only | Regime 2; MPS disqualified |
| Draft-target speculative replicas            | NO     | —    | Regime 1 (draft already shares target HBM) |
| HBM-starved 70B (chapter 104 math tight)     | NO     | —    | Regime 3; no headroom |
| TP-collective-heavy on same NVLink group     | NO     | —    | Regime 4; NVLink saturation |
| Burst batch (variable shape, needs claim)    | YES    | DRA (if cluster supports) | flexibility |
```

For each YES / NO row, include the *measurement or reasoning* that backs it — chapter-03 says "latency-critical big models don't share"; the row should cite the specific observation (chapter-03 text, your Deliverable C contention measurement, chapter 104's HBM math, etc.).

### Deliverable F — the hybrid node-pool topology diagram + `<!-- needs-escalation -->` list

A diagram (ASCII is fine) showing the hybrid-pool shape from chapter 03's "worked example," instantiated against your fleet:

```
Node pool: pool-h100-full
  Nodes: <count> × p5.48xlarge
  Sharing: none
  Taint: inference-tier=latency-critical:NoSchedule
  Workloads: Workload A (70B, 405B, large-replica TP groups)

Node pool: pool-h100-mig
  Nodes: <count> × p5.48xlarge
  MIG profile: all-2g.20gb (4 slices / GPU × 8 GPUs = 32 slices / node)
  Sharing: MIG
  Taint: inference-tier=small-model:NoSchedule
  Workloads: Workload B — small-model MIG tier

Node pool: pool-h100-mps
  Nodes: <count> × p5.48xlarge or smaller (MPS replicas: 4)
  Sharing: MPS
  Taint: inference-tier=batch-tolerant:NoSchedule
  Workloads: Workload B — small-model MPS tier (same trust boundary)

Node pool: pool-dra (if DRA was deployed)
  Nodes: <count> × (SKU)
  Sharing: DRA (claim-driven)
  Taint: inference-tier=claim-driven:NoSchedule
  Workloads: Workload C
```

Attach the node-autoscaler (Karpenter `NodePool` / Cluster Autoscaler node group) manifests that back each pool. Chapter 01's example is the template; adapt per sharing mode.

Then write the `<!-- needs-escalation -->` list — things you couldn't resolve from this exercise alone:

- "The right MIG profile — `1g.10gb` vs. `2g.20gb` vs. `3g.40gb` — for our small-LLM mix. Depends on each workload's HBM footprint under its max-concurrency." → chapter 104 HBM math; per-workload profiling.
- "Whether MPS's lack of HBM isolation is acceptable for our tenant mix. Chapter-03 Regime 2 names the hard rule; we need legal / security's read on our specific tenant contract." → `ai-infra-security` level 35; tenant contract review.
- "DRA's maturity on our Kubernetes release. The driver's docs claim support; our platform team has no prior production experience." → DRA driver release notes; chapter 04 of module 109 roadmap.
- "Topology / NUMA concerns for MIG slices on an 8-GPU node. Chapter 03 Error 7 flags it; we didn't measure." → NFD topology manager; module 107 distributed-topology discipline.
- "The right cost-accounting unit when GPUs are shared. Is a `2g.20gb` MIG slice billed as 2/7 of a GPU-hour, or by HBM, or flat? Depends on fleet policy." → module 110 cost-per-million-tokens; chapter 114 cross-functional discipline.

## Starter guidance

- **MIG profile change requires a reset. Plan accordingly.** Don't start MIG reconfiguration on a node running your only exercise-02 HPA load test. Coordinate; drain; apply; verify.
- **MPS lands on *one* physical GPU.** The device plugin advertises `replicas: 4` per physical GPU. If your node has 8 GPUs and MPS replicas 4, the kubelet advertises `nvidia.com/gpu: 32` on that node. Chapter 03's wording is terse; the arithmetic matters.
- **DRA driver APIs rotate.** The vendor CRDs changed around the time of upstream GA; chapter 03 flags it. Use the current quickstart examples, not a stale blog post.
- **Measurements, not assertions.** Deliverable E's YES / NO decisions should cite actual contention observations from Deliverables B/C, not just chapter-03 prose. Even a crude TTFT delta ("isolated: 180 ms; co-tenant: 290 ms") makes the decision defensible.
- **Don't co-locate a 70B with anything — even for the test.** Chapter 03's Regime 1 is real; don't confirm it on your production target. Measure on a sandbox GPU.
- **Keep the sharing mode per-pool, not per-node.** Mixing `single` and `mixed` MIG strategies in one pool is an Error 6; separate pools per sharing mode is the clean shape.

## Fallbacks for constrained environments

- **No MIG-capable GPU.** Document skip for Deliverable B. Chapter-03 Regime-match (Deliverable E) still applies — reason about where MIG *would* fit if you had the hardware.
- **DRA driver not supported on your Kubernetes release.** Document skip for Deliverable D explicitly (see template above). The MIG and MPS paths exercise most of the chapter's muscle.
- **Only one GPU node.** You can't exercise the hybrid-pool shape (Deliverable F) literally. Document the topology as *intent* for when you have more nodes; label the single node per deliverable in sequence rather than simultaneously.
- **MPS not supported by your engine.** Some engine releases don't work well under MPS (e.g. CUDA graph capture conflicts). Verify engine support before running Deliverable C; if blocked, note the engine gap and use a simpler process (nvidia-smi + a synthetic kernel) to exercise the MPS shape.
- **Shared / regulated cluster that forbids cluster-wide config changes.** Deliverables B / C may require cluster-admin coordination. Partner with the platform team for a scheduled window; this exercise is not a side activity.

## Acceptance criteria

1. **Deliverable A** records the GPU inventory, target topology, and Kubernetes / driver versions; deviations from the target are documented.
2. **Deliverable B** shows a MIG-partitioned node advertising the right slices, a replica scheduled onto a specific slice, and `nvidia-smi` confirming the pod sees only its slice. (Or documented skip if hardware doesn't support MIG.)
3. **Deliverable C** shows four replicas co-located on one physical GPU via MPS, with a contention measurement showing TTFT / throughput degradation at 1 / 2 / 4 concurrent load.
4. **Deliverable D** either stands up a DRA claim and verifies a pod binding or documents the skip concretely (cluster version, driver, planned follow-up).
5. **Deliverable E** is a decision table with every chapter-03 regime covered and a specific YES / NO backed by observation or reasoning.
6. **Deliverable F** names the hybrid-pool topology (pools, taints, SKUs, sharing modes) and attaches the node-autoscaler manifests; the `<!-- needs-escalation -->` list names at least five items with references.
7. Measurements are reproducible: anyone with the same hardware, same device-plugin config, and same workload image should see the same slice / MPS / DRA behaviour.

## Common failure modes

- **MIG profile change without a drain.** Chapter 03 Error 1. The MIG manager will drain for you; if you're applying MIG by hand, drain first. Any in-flight request on an un-drained node loses its generation.
- **MPS as the default across the cluster.** Chapter 03 Error 2. One tenant's OOM takes down the others' replicas. MPS is in-trust-boundary only.
- **MIG + MPS on the same GPU.** Not supported; chapter 03 names the exclusion. Verify your ConfigMap doesn't set MPS on a MIG-enabled node.
- **DRA claim with no fallback shape.** Chapter 03 Error 3. The pod sits Pending. Make the claim satisfiable before expecting a bind.
- **A `nvidia.com/gpu: 8` request landing on a MIG-partitioned node.** The pool advertises `nvidia.com/mig-Ng.Xgb` not `nvidia.com/gpu`; pods requesting the un-sliced resource can't schedule there. Separate pools (Deliverable F) is the fix.
- **Co-locating a tenant with a draft-target replica.** Chapter 03 Error 4. The draft already shares the target's HBM; adding a co-tenant is three-way contention. Draft-target replicas are always on `pool-h100-full`.
- **MPS replicas without memory accounting.** Setting `replicas: 4` doesn't quarter the HBM; each process can still allocate all of it. Chapter 03 names this — MPS has no HBM isolation. If one pod OOMs the GPU, the others go with it. The decision table (Deliverable E) must flag this.
- **Topology-blind MIG placement.** Two MIG slices on different NUMA nodes of an 8-GPU node have different NVLink distances; a workload sensitive to collective latency should pin. Chapter 03 Error 7 flags it.
- **Decision table as prose, not a table.** Deliverable E has to be scannable; a tenant intake call can't afford a two-page essay. Keep it a table.

## Keep it

The hybrid-pool topology (Deliverable F), the "when NOT to share" decision table (Deliverable E), and the MIG / MPS / DRA walk evidence (Deliverables B / C / D) become the fleet's sharing playbook. Exercise 04's warm pool decision composes with Deliverable F — the warm pool lives on `pool-h100-full` by default; chapter 05's spot pool is `pool-h100-full` or a sibling `pool-h100-spot`, not a MIG / MPS pool. Exercise 05's drain respects the sharing-mode taints; exercise 06's cost-vs-latency routing applies per workload tier, with the tier defined by Deliverable E. When a new tenant asks "can we share?" — point them at Deliverable E before the intake call.
