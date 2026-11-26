# Exercise 01 — GPU Operator and DCGM Base Install

**Time:** 3 hours · **Output:** one running GPU Operator install on a GPU-equipped cluster, one `nvidia-smi` smoke pod, one Prometheus `ServiceMonitor` wiring DCGM, one four-check substrate health report, one `<!-- needs-escalation -->` list

## Goal

Make [Chapter 01](../01-gpu-aware-kubernetes-serving.md) concrete by **standing up the NVIDIA GPU Operator** on a GPU-equipped Kubernetes cluster, confirming every substrate layer is wired, landing a smoke-test inference pod on a GPU node, and publishing the four substrate health checks the chapter demands. The deliverable is the first-day artefact every chapter-02/03/04/05 exercise builds on top of.

## Problem statement

Your platform has just received access to a Kubernetes cluster with GPU nodes (p4d / p5 / a3 / NC-series — pick what you have). Before any chapter-02 scaler, chapter-03 sharing, chapter-04 warm pool, or chapter-05 drain can be meaningfully built, the substrate underneath has to be right. The goal today is to confirm the five substrate layers — driver, container toolkit, device plugin, node labels, DCGM — are all installed, healthy, and observable, so that every downstream exercise can lean on them without re-litigating the base install.

If your cluster already has a GPU Operator install from a previous engagement, treat this exercise as a *verification* pass: confirm every health check passes, document the version pins, and surface any gaps.

## Prerequisites

- Chapter 01 of this module read.
- A Kubernetes cluster you have `cluster-admin` on, with **at least one GPU node** (datacenter-class GPU preferred — A100 / H100 / H200 / L40 — but A10G / L4 works for the install path). Managed (EKS / GKE / AKS) or self-managed both work; note which path you are on before starting.
- `kubectl` and `helm` on the workstation, with cluster credentials configured.
- A Prometheus install reachable from the cluster — `kube-prometheus-stack` on the same cluster is the easy path; a cross-cluster Prometheus works if its service-discovery can reach the GPU nodes.
- Authorisation to install cluster-scoped resources (CRDs, DaemonSets, cluster roles). If you don't have it, acquire it before starting — the Operator install touches all three.
- The GPU Operator **platform support matrix** at <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/platform-support.html> open — you will cross-check against your cluster's Kubernetes version and node OS before picking a release.

## Requirements

Five deliverables.

### Deliverable A — the install plan (one-page document)

Before touching the cluster, write a one-page plan that names the pins:

```
# GPU Operator Install Plan — <cluster_name>

## Cluster facts
- Cloud / distribution: <EKS 1.31 | GKE 1.32 | AKS 1.31 | self-managed kubeadm 1.31 | ...>
- GPU node SKU(s): <p5.48xlarge × 2 | g6.12xlarge × 1 | ...>
- Node OS: <Amazon Linux 2023 | Ubuntu 22.04 | COS | ...>
- Existing driver on node: <pre-installed via AMI | none | version X.Y>
- Target Kubernetes version: <1.31.4>

## Pinned versions
- GPU Operator: <vX.Y.Z> (from platform-support matrix for the K8s version above)
- Device plugin: <managed by Operator>
- Driver: <managed by Operator | pre-installed via AMI → driver.enabled=false>
- DCGM-exporter: <managed by Operator>

## Install path
- helm chart: nvidia/gpu-operator vX.Y.Z
- Values overrides (driver.enabled, mig.strategy, toolkit.enabled, ...)
- Namespace: gpu-operator (created by install)

## Rollback plan
- `helm uninstall --namespace gpu-operator <release>`; confirm DaemonSets removed;
  if driver was Operator-managed, reboot nodes to clear the kernel module.
```

Record the version choices and the reasoning (you will cross-link this to Deliverable E).

### Deliverable B — the install itself

Apply chapter-01's install command, with your chosen pins:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install --wait \
  --generate-name \
  --namespace gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version <pinned-version> \
  --set driver.enabled=<true|false> \      # false if the AMI preinstalls the driver
  --set toolkit.enabled=true \
  --set dcgmExporter.enabled=true \
  --set nodeStatusExporter.enabled=true \
  --set migManager.enabled=<true|false>    # true only on MIG-capable nodes; chapter 03
```

Attach to the deliverable:

- The exact helm command you ran (with the resolved `--set` flags).
- The output of `helm status <release> --namespace gpu-operator`.
- The output of `kubectl -n gpu-operator get pods -o wide` after the install stabilises, showing one driver / toolkit / device-plugin / DCGM-exporter / NFD / validator pod per GPU node.

### Deliverable C — the smoke-test pod

Apply chapter 01's `smoke-cuda` Pod spec against your cluster, unmodified except for a node selector / toleration appropriate to your fleet:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: smoke-cuda
  namespace: gpu-smoke
spec:
  restartPolicy: Never
  runtimeClassName: nvidia
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:<pinned-tag>
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

Attach:

- The exact Pod spec you applied (with the CUDA image tag pinned).
- The pod's logs: `kubectl -n gpu-smoke logs smoke-cuda` — must show `nvidia-smi` output listing the GPU, the driver version, and the CUDA runtime version.
- `kubectl -n gpu-smoke describe pod smoke-cuda` — must show `runtimeClassName: nvidia` and `nvidia.com/gpu: 1` in the limits.

If the pod fails, debug with chapter-01's "If `nvidia-smi` fails" list (missing device plugin, driver/toolkit mismatch, wrong SKU, missing taint toleration) before marking the exercise unblocked.

### Deliverable D — the DCGM scrape

Wire DCGM-exporter into Prometheus. On `kube-prometheus-stack`, the Operator typically ships a `ServiceMonitor` or a `PodMonitor` for the DCGM-exporter; confirm it:

```bash
kubectl -n gpu-operator get servicemonitor,podmonitor
```

If missing, apply one:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nvidia-dcgm-exporter
  namespace: gpu-operator
  labels:
    release: <prometheus-release>            # the label your Prometheus selector requires
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  endpoints:
  - port: gpu-metrics
    interval: 15s
```

Verify in Prometheus (adjust the query for your DCGM-exporter release's metric names):

```promql
count(DCGM_FI_DEV_SM_ACTIVE)
```

The returned count must equal the total number of physical GPUs in the cluster. If zero: the scrape didn't connect. If less than expected: a node's exporter is unhealthy or a DaemonSet didn't roll out — look at the pod list from Deliverable B.

Attach to the deliverable:

- The `ServiceMonitor` / `PodMonitor` manifest (either the Operator-shipped one or your own).
- A screenshot (or query result) from Prometheus showing `count(DCGM_FI_DEV_SM_ACTIVE)` matching the fleet's GPU count.
- A query result for `sum(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) by (Hostname)` showing per-node HBM accounting for every GPU node.

### Deliverable E — the four substrate health checks + `<!-- needs-escalation -->` list

Execute and record chapter 01's four substrate health checks:

```
Health Check 1 — `nvidia.com/gpu` allocatable
  Command: `kubectl get nodes -o json | jq '.items[] | {name:.metadata.name, gpu:.status.allocatable["nvidia.com/gpu"]}'`
  Pass criteria: every GPU node reports > 0; no non-GPU node reports anything.
  Result: <pasted output>
  Status: PASS / FAIL (notes)

Health Check 2 — Operator DaemonSets green
  Command: `kubectl -n gpu-operator get pods -o wide`
  Pass criteria: one driver, toolkit, device-plugin, DCGM-exporter, NFD, validator per GPU node; all Running / Ready.
  Result: <pasted output summary>
  Status: PASS / FAIL (notes)

Health Check 3 — `nvidia-smi` smoke pod
  Command: Deliverable C
  Pass criteria: smoke pod runs to completion and logs show the GPU.
  Result: <pasted logs>
  Status: PASS / FAIL (notes)

Health Check 4 — DCGM visible in Prometheus
  Command: Deliverable D's `count(DCGM_FI_DEV_SM_ACTIVE)` query
  Pass criteria: returned count equals fleet GPU count.
  Result: <pasted result>
  Status: PASS / FAIL (notes)
```

Then write the `<!-- needs-escalation -->` list — things you couldn't confidently resolve from this exercise alone. Expected entries:

- "The exact GPU Operator release that is right for our Kubernetes minor version. The platform-support matrix lists several compatible releases; which one the fleet should pin." → GPU Operator platform-support matrix; your platform-team's upgrade cadence policy.
- "Whether `driver.enabled=false` is the right call on our node AMI. The AMI docs claim the driver is pre-installed; the exact version isn't visible to us." → cloud node AMI release notes; `nvidia-smi` output from a bootstrap DaemonSet.
- "The right MIG strategy — `single` vs. `mixed` — for the chapter-03 exercise. Not resolvable without naming the workload mix." → chapter 03 of this module; exercise 03.
- "The right DCGM profiling scrape interval. 15 s matches our Prometheus baseline; some fleets profile at 5 s." → module 110 observability discipline; DCGM-exporter config options.
- "Whether to install `nvidia-operator-validator` as a separate operand or rely on the embedded one. Release-release-specific." → GPU Operator release notes for the pinned version.

## Starter guidance

- **Pin the Operator version before you install.** The platform-support matrix is the authority; copying someone else's helm command from six months ago is how driver / CUDA mismatches start.
- **If the cluster has an existing Operator install, don't re-install.** Verify first. `helm list -A | grep gpu-operator` and inspect the values; two concurrent installs are a mess.
- **Driver pre-installed on the AMI is the common managed-K8s shape.** EKS GPU AMIs, GKE's autoinstall, AKS GPU-enabled node pools all pre-install the driver. Set `driver.enabled=false` in that case — chapter-01 warns about the conflict.
- **DCGM-exporter scraping is the easiest step to miss.** The Operator installs the exporter DaemonSet; whether Prometheus discovers it is a separate concern. Verify in a PromQL query before marking the exercise done.
- **Save the install plan.** Deliverable A becomes the doc that answers "which version are we on?" during the next upgrade cycle. Keep it with the deployment artefacts, not on a whiteboard.

## Fallbacks for constrained environments

- **No Prometheus on the cluster.** Stand up `kube-prometheus-stack` (<https://github.com/prometheus-operator/kube-prometheus-stack>) via `helm install` as part of the exercise; it is the base-track dependency the rest of the module leans on.
- **No MIG-capable GPUs.** Set `migManager.enabled=false`; the rest of the install proceeds. Chapter 03's MIG section becomes read-only for your fleet; MPS and DRA still land.
- **Managed node pool with Operator pre-installed.** GKE's autoinstall and some EKS add-ons pre-install the full stack. Treat the exercise as a *verification* pass — all four health checks still have to green.
- **Air-gapped cluster.** Mirror the Operator's images to your internal registry and set `imagePullPolicy` / `registry` overrides per the Operator install guide. Allow extra time; the exercise's substance is the same.
- **Shared cluster with other tenants.** Don't change cluster-scoped settings without coordination. If someone else owns the GPU Operator install, limit the exercise to Deliverables C / D / E (verification only) and record who owns the install.

## Acceptance criteria

1. **Deliverable A** is a one-page document naming the exact pins for Operator version, driver policy, DCGM-exporter scrape interval, and rollback strategy.
2. **Deliverable B** shows every Operator-managed DaemonSet landed healthy on every GPU node in the cluster; CrashLoopBackOff on any component is a fail.
3. **Deliverable C** shows the smoke-test pod ran successfully with `nvidia-smi` output listing the device; the pod uses `runtimeClassName: nvidia` and the GPU taint toleration.
4. **Deliverable D** confirms DCGM metrics are scraped into Prometheus — the `count(DCGM_FI_DEV_SM_ACTIVE)` query returns the total GPU count of the cluster.
5. **Deliverable E** records all four substrate health checks with PASS / FAIL status and attached output; the `<!-- needs-escalation -->` list names at least four items with references.
6. The install is reproducible: anyone with the same cluster state, cluster-admin, and the Deliverable A plan could re-run it from scratch.

## Common failure modes

- **Running a dated helm command without re-checking the platform-support matrix.** The Operator release that was right six months ago is not the one that's right today; the driver / Kubernetes compat has rotated. Verify the matrix before pinning.
- **`driver.enabled=true` on an AMI that pre-installs the driver.** The Operator's driver DaemonSet and the AMI's driver fight for the kernel module; pods see `NVML initialization failed`. Fix: `driver.enabled=false` on pre-installed AMIs.
- **Smoke-test pod stuck Pending with `0/N nodes available`.** The device plugin isn't advertising; chapter 01's debug ladder applies. Don't move on until the pod runs.
- **DCGM-exporter running but Prometheus not scraping.** The `ServiceMonitor` label selector doesn't match your Prometheus's `serviceMonitorSelector`. Fix: inspect the Prometheus CR's selector and label the SM to match.
- **Health Check 4 verified only at install time.** DCGM-exporter can silently die on a node restart; verify a scrape freshness query (e.g. `time() - max(DCGM_FI_DEV_SM_ACTIVE) by (Hostname) > 120` should return no series).
- **Mixing MIG strategies on the same node pool.** `single` vs. `mixed` must match the workloads the pool serves. Chapter 03 has the detail; for the install, pick one and record it.
- **Skipping the `<!-- needs-escalation -->` list because "everything worked."** The list is where version drift and policy ambiguity land; a short list is fine, an empty one is suspicious.

## Keep it

The install plan (Deliverable A), the four-check health report (Deliverable E), and the DCGM `ServiceMonitor` (Deliverable D) become the fleet's substrate artefacts. Exercise 02 wires its scaler against the engine metrics that land on *this* Prometheus; exercise 03's MIG / MPS / DRA walkthrough reshapes *this* GPU Operator install; exercise 04's warm pool lands on *this* node autoscaler topology; exercise 05's drain respects *this* taint / toleration contract. Run this exercise first; the rest of the module assumes it is green.
