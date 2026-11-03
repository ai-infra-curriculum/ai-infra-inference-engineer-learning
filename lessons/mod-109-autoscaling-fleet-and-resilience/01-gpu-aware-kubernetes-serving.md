# Chapter 01 — GPU-Aware Kubernetes Serving: Operator, Device Plugin, DCGM, Runtime Class, Node Autoscaler

> A GPU in a Kubernetes pod is not just a bigger CPU request. It needs a device plugin that advertises it, a driver DaemonSet that keeps the kernel module and the container toolkit in sync, a runtime class that routes the container to `nvidia-container-runtime`, taints and tolerations so cheap CPU pods don't land on expensive GPU nodes, and a node autoscaler that knows that a 70B replica wants four GPUs on *one* node, not four nodes with one GPU each. This chapter builds that substrate.

## Motivation

A platform team has a fresh Kubernetes cluster with four `g5.12xlarge` nodes (one A10G per node, close enough for the examples) and three `p4d.24xlarge` nodes (8× A100 each). They want to serve a Llama-3.1-70B replica at TP=4. The team's first attempt — a `Deployment` with `resources.requests.nvidia.com/gpu: 4` and a plain `ClusterAutoscaler` on EKS — fails in six different ways over the course of one afternoon:

1. The pod stays `Pending` with `0/7 nodes available: insufficient nvidia.com/gpu` because the device plugin DaemonSet hasn't been installed. Kubernetes doesn't advertise `nvidia.com/gpu` by default.
2. After installing the device plugin by hand, pods still fail: the `nvidia-container-toolkit` isn't configured on the nodes, so the containers can't see `/dev/nvidia*`.
3. After installing the toolkit, pods still fail on the g5 nodes with "no CUDA-capable device found" — the driver version on the g5 nodes doesn't match the CUDA runtime in the container image.
4. After pinning the driver via the GPU Operator, the 70B pod schedules — on one p4d node fine, but the autoscaler spins up a *second* p4d when a second replica arrives, even though the first p4d still has four A100s free. The default scheduler isn't bin-packing; it's spreading.
5. A CPU-only sidecar pod from a different team lands on a p4d node. The GPU memory isn't contended but the CPU scheduler now has a non-GPU pod squatting on a $25/hr machine.
6. DCGM metrics are nowhere. The team's SRE asks "what's the SM activity on those A100s?" and no one has an answer.

Every one of those failures is prevented by the GPU-aware substrate this chapter lays out. By the end of it you can:

- Install the NVIDIA GPU Operator and verify every DaemonSet it manages is healthy.
- Confirm the device plugin is advertising `nvidia.com/gpu` on every GPU node and only on GPU nodes.
- Confirm DCGM-exporter is scraping to Prometheus and the metric set from module 110 is visible.
- Author a `RuntimeClass` that routes GPU pods through `nvidia-container-runtime` and keep CPU pods off GPU nodes with taints and tolerations.
- Author a Karpenter `NodePool` (or Cluster Autoscaler node group) that bin-packs the right GPU SKU for a stated replica shape — and respects the atomic shape of a TP group.
- Run the four substrate health checks that confirm all of the above is wired right *before* an inference pod is scheduled against it.

<!-- needs-research: component version compatibility matrices (GPU Operator → driver → CUDA → container toolkit → node OS) rotate every quarter. The specific releases named below are illustrative — always pin and verify against the GPU Operator release notes and the pinned Kubernetes version for your cluster. -->

## The substrate layers

A GPU-ready Kubernetes node is a stack of five concerns, each with a specific component responsible for it:

| Layer | Component | Role |
|---|---|---|
| Kernel driver | `nvidia.ko` + `nvidia-fabricmanager` (NVSwitch nodes) | Kernel-side access to the GPU; must match the container toolkit's version |
| Container runtime toolkit | `nvidia-container-toolkit` + `nvidia-container-runtime` | Injects `/dev/nvidia*`, mounts libraries, sets up cgroups, respects `NVIDIA_VISIBLE_DEVICES` |
| Device plugin | `k8s-device-plugin` (DaemonSet) | Advertises `nvidia.com/gpu` to the kubelet; healthchecks the GPU |
| Node labels & topology | `node-feature-discovery` + GPU Operator labels (e.g. `nvidia.com/gpu.product`, `nvidia.com/mig.config`) | Fine-grained scheduling by SKU, memory, MIG profile |
| Observability | `dcgm-exporter` (DaemonSet) | Prometheus-shape GPU metrics (SM activity, HBM usage, ECC errors, NVLink utilisation) |

Three deployment shapes satisfy that stack:

- **NVIDIA GPU Operator** (recommended). One operator installs and manages all five layers via a reconciled `ClusterPolicy`. It is the current path on managed Kubernetes (EKS, GKE, AKS) and self-managed clusters alike. See <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/>.
- **Cloud-provider pre-installed** (fast). Some managed GPU node pools (EKS GPU AMI, GKE GPU node pool with autoinstall, AKS GPU-enabled node pool) pre-install the driver and toolkit, and the device plugin arrives as an add-on. Verify what's included; DCGM-exporter is usually not.
- **Hand-rolled DaemonSets** (deprecated). Historically the components were installed as separate manifests. Still works, but the compatibility matrix is now the operator's job; don't maintain it by hand unless you have a strong reason.

The chapter and exercises assume the Operator path. If you are on a cloud pre-installed path, chapter 01 of `ai-infra-engineer` (level 20) covers the base cluster; this chapter's health checks still apply and will tell you if a layer is missing.

## Installing the GPU Operator

The canonical install is Helm:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait \
  --generate-name \
  --namespace gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version <pinned-version> \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set dcgmExporter.enabled=true \
  --set nodeStatusExporter.enabled=true \
  --set migManager.enabled=true
```

Operational cases:

- **If your node image already ships the driver** (EKS GPU AMI, GKE autoinstall) — set `driver.enabled=false` to avoid a conflict. The Operator will manage the rest.
- **If you want MIG** (chapter 03) — leave `migManager.enabled=true` and label nodes with a MIG profile; the manager will reconfigure the GPUs on first boot.
- **If your cluster is air-gapped** — mirror the images and set the registry overrides; see the Operator install guide.

After install, confirm every component is healthy:

```bash
kubectl -n gpu-operator get pods
# All of:
#  - gpu-operator-xxxxx
#  - nvidia-driver-daemonset-xxxxx   (one per GPU node)
#  - nvidia-container-toolkit-daemonset-xxxxx  (one per GPU node)
#  - nvidia-device-plugin-daemonset-xxxxx     (one per GPU node)
#  - nvidia-dcgm-exporter-xxxxx               (one per GPU node)
#  - nvidia-mig-manager-xxxxx                 (one per MIG-capable GPU node)
#  - gpu-feature-discovery-xxxxx              (one per GPU node)
```

Any CrashLoopBackOff here will mask every downstream failure. Fix at this layer first.

<!-- needs-research: the exact component list and DaemonSet names evolve per Operator release. Verify against the Operator's "component matrix" page in the current release notes. -->

## Advertising `nvidia.com/gpu`

Once the device plugin is up, the kubelet on each GPU node advertises `nvidia.com/gpu` as a schedulable resource:

```bash
kubectl describe node <gpu-node>
# Capacity:
#   nvidia.com/gpu: 8
# Allocatable:
#   nvidia.com/gpu: 8
```

A GPU pod requests the resource like any other:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: smoke-cuda
spec:
  restartPolicy: Never
  runtimeClassName: nvidia                     # chapter section below
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

For a smoke test, `nvidia-smi` from inside the container should list the device and version. If it fails:

- `No devices were found` → device plugin didn't inject; check `/dev/nvidia*` mount.
- `Failed to initialize NVML: Driver/library version mismatch` → driver and container-toolkit versions don't match.
- `Pod stuck Pending with insufficient nvidia.com/gpu` → device plugin not advertising; check the DaemonSet logs.

### A note on `limits` vs. `requests`

GPUs are an *extended resource*. Unlike CPU and memory, you must set `limits` to a non-zero integer; `requests` is implicit (equal to `limits`). Fractional GPU is **not** supported at the device-plugin layer — if you need fractions, that is MIG, MPS, or DRA (chapter 03), not `nvidia.com/gpu: 0.5`.

## The `RuntimeClass`

The container runtime that *actually* understands `/dev/nvidia*` and the GPU driver is `nvidia-container-runtime`, not the default `runc`. Modern GPU Operator installs register a `RuntimeClass` named `nvidia`:

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
```

Any pod that wants a GPU should set `runtimeClassName: nvidia`. On many clusters the default runtime is already `nvidia-container-runtime` (one less footgun), but setting the class explicitly is the robust path and the one that survives a cluster upgrade.

<!-- needs-research: whether the default runtime should be `nvidia` cluster-wide or opt-in via `runtimeClassName` is a judgement call. The Operator's documentation discusses the trade-off. Verify current recommended default against the pinned release's notes. -->

## Taints and tolerations — keeping CPU pods off GPU nodes

A $25/hr p4d node with a CPU-only pod occupying the scheduler is a budget leak. The convention is to **taint** every GPU node and require pods to **tolerate** the taint before scheduling:

```yaml
# node spec (applied by GPU Operator / Karpenter / Terraform / cloud node-pool config)
spec:
  taints:
  - key: nvidia.com/gpu
    value: "true"
    effect: NoSchedule
```

```yaml
# GPU pod spec
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
```

Combined with the `nvidia.com/gpu` request, this gives you a bidirectional contract: *only pods that both request a GPU and tolerate the GPU taint land on a GPU node.* The GPU Operator's own DaemonSets already tolerate the taint (driver, toolkit, device plugin, DCGM-exporter) so the substrate itself is unaffected.

For multi-SKU fleets, add node labels so pods can pin to the SKU they need:

```yaml
# node labels (from gpu-feature-discovery or applied directly)
labels:
  nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3    # or A100-SXM4-80GB, H200, etc.
  nvidia.com/gpu.memory: "81559"                   # in MiB
  node.kubernetes.io/instance-type: p5.48xlarge
```

```yaml
# pod affinity (prefer H100 for a 70B replica)
spec:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values: ["NVIDIA-H100-80GB-HBM3", "NVIDIA-H200"]
```

The alternative (a `nodeSelector` with a single key) is simpler but less expressive; `nodeAffinity` is the pattern for anything with more than one acceptable SKU.

## The node autoscaler — Karpenter and Cluster Autoscaler

A GPU-aware node autoscaler differs from the generic `cluster-autoscaler` in two operator-visible ways:

- **SKU-aware provisioning.** The autoscaler must pick the GPU SKU that matches the pod's node affinity, not just the first SKU in a node pool with free capacity. This is the discipline that keeps a 70B replica from waking up on a cost-wrong machine.
- **TP-group-aware bin-packing.** A TP=4 replica wants four GPUs *on one node* (NVLink-connected) — not four GPUs on four different nodes (PCIe to the NIC, catastrophic for collective bandwidth). The pod spec requests `nvidia.com/gpu: 4` and the autoscaler must size the node accordingly; a pool of 1×GPU nodes will admit the pod to `Pending` forever.

### Karpenter — the current best path on AWS

Karpenter (<https://karpenter.sh/>) is NodePool-based rather than node-group-based. One `NodePool` with the right requirements can host multiple GPU SKUs and bin-pack correctly:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-h100
spec:
  template:
    spec:
      requirements:
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["p5.48xlarge", "p5e.48xlarge"]      # 8× H100 or 8× H200 nodes
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["on-demand"]
      taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-h100
  limits:
    cpu: "1000"
    nvidia.com/gpu: "32"
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 60m                            # keep GPU nodes warm; see chapter 04
```

Two inference-specific choices to notice:

- `consolidateAfter: 60m` — the default is often shorter. GPU boot is slow (chapter 04); consolidating too aggressively creates thrash.
- Separate `NodePool` per SKU is often cleaner than one pool with many SKUs — different SLAs, different pricing, different drain policies (chapter 05 for spot).

Verify the current Karpenter CRD version — `v1` is current at time of writing but the API has evolved from `v1alpha` to `v1beta1` to `v1`. The pattern is stable; the field names rotate.

<!-- needs-research: Karpenter CRD version and field names rotate (`v1alpha5` → `v1beta1` → `v1`). The example above uses the current `v1` shape; verify against <https://karpenter.sh/docs/concepts/nodepools/> for the exact fields and defaults in the pinned release. -->

### Cluster Autoscaler — the broadly supported path

On clouds where Karpenter isn't available (GCP, Azure) or on self-managed clusters, Cluster Autoscaler (<https://github.com/kubernetes/autoscaler>) is the general-purpose option. It operates on node *groups* (ASGs on AWS, instance groups on GCP, VMSS on Azure). The inference-specific disciplines:

- One node group per GPU SKU. Don't mix SKUs in one group — the autoscaler can't choose.
- Scale-from-zero support must be configured per group (node-group tags on AWS, GKE autoscaler config on GCP, VMSS tags on Azure). Without it, the autoscaler won't provision a node to serve a pod when the group has `desired=0`.
- `--scale-down-delay-after-add` and `--scale-down-unneeded-time` should be set much higher than the defaults for GPU groups — chapter 04 covers the sizing.

### Bin-packing a TP group

A single pod with `nvidia.com/gpu: 4` is a single scheduling unit; Kubernetes will land it on one node with 4+ GPUs available. That's the "atomic TP group fits on one node" pattern — the common case for TP≤8 on a single `p5.48xlarge` / `p4d.24xlarge`.

For TP > 8 (two nodes) or for pipeline parallelism across multiple nodes, you have a *group of pods* that must land together on specific topology. Three mechanisms:

- **Kubernetes native: pod anti-affinity + topology spread constraints.** Works for small cases; brittle at scale.
- **JobSet / LeaderWorkerSet.** The current recommended pattern for multi-pod inference replicas. LeaderWorkerSet (`sigs.k8s.io/lws`) is designed for exactly this — a leader + N workers that scale and are rescheduled as a unit. The Kubernetes AI serving community has standardised on it; verify current status and version.
- **Volcano / Kueue / Gang scheduling.** Batch-scheduler-level gang scheduling for groups that must schedule atomically or not at all. Overkill for TP=4 on one node; justified for TP=16 across two nodes with the model loaded in halves.

<!-- needs-research: LeaderWorkerSet is relatively new; the exact repo, release, and K8s API group have been evolving. Verify the current reference at <https://github.com/kubernetes-sigs/lws> and the inference-specific adoption at the K8s AI serving working group. -->

Chapter 107 of this track and chapter 02 of `mod-107-distributed-inference-topologies` cover the TP / PP layout; this chapter's responsibility is naming the mechanism for landing the group on the cluster.

## DCGM — the GPU observability layer

DCGM-exporter (shipped by the Operator) produces a Prometheus-shape metric stream from every GPU on every GPU node. The metrics you care about at the autoscaler layer:

- `DCGM_FI_DEV_SM_ACTIVE` — fraction of time the GPU's SMs were active. Reads 1.0 at single-request prefill *and* at saturated batch; useful for "GPU is doing something" but not for "fleet is overloaded." (Chapter 02 names the inference metrics that are better.)
- `DCGM_FI_DEV_FB_USED` / `DCGM_FI_DEV_FB_FREE` — HBM used / free in MiB. The key signal for "can I fit another replica" and "did the KV allocator run out" (chapter 104).
- `DCGM_FI_DEV_POWER_USAGE` — power draw in W. Secondary signal for utilisation; tracks compute pretty well.
- `DCGM_FI_DEV_XID_ERRORS` and `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` / `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` — hardware errors; module 113 wires these to incident alerts.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` and `DCGM_FI_PROF_DRAM_ACTIVE` — profile-side metrics for tensor-core utilisation and DRAM traffic; useful at operator depth for diagnosing "GPU looks busy but the right pipe isn't firing."

Full reference: <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>.

Wire DCGM-exporter into your Prometheus (via `ServiceMonitor` on Prometheus Operator, scrape config elsewhere) and verify in a quick query:

```promql
# active GPUs count per node
count(DCGM_FI_DEV_SM_ACTIVE) by (Hostname)

# HBM utilisation by node
sum(DCGM_FI_DEV_FB_USED) by (Hostname) / sum(DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) by (Hostname)
```

If these queries return nothing, DCGM isn't scraping; the autoscaler and the dashboards downstream will all be blind. Fix now.

## A worked example — a 70B replica, TP=4, on p5

Putting it together — a complete deployment spec for a Llama-3.1-70B replica (vLLM OpenAI-compatible server):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama31-70b
  labels: {app: vllm-llama31-70b}
spec:
  replicas: 1
  selector: {matchLabels: {app: vllm-llama31-70b}}
  template:
    metadata:
      labels: {app: vllm-llama31-70b}
    spec:
      runtimeClassName: nvidia
      tolerations:
      - key: nvidia.com/gpu
        operator: Equal
        value: "true"
        effect: NoSchedule
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      containers:
      - name: vllm
        image: vllm/vllm-openai:<pinned>
        args:
        - "--model=meta-llama/Meta-Llama-3.1-70B-Instruct"
        - "--tensor-parallel-size=4"
        - "--dtype=bfloat16"
        - "--port=8000"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4
            cpu: "16"
            memory: 128Gi
        env:
        - name: HF_TOKEN
          valueFrom: {secretKeyRef: {name: hf-token, key: token}}
        - name: NCCL_P2P_LEVEL
          value: "NVL"                           # NVLink peer-to-peer; mod-107 context
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: model-cache
          mountPath: /root/.cache/huggingface
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 180              # cold-start-aware; chapter 04
          periodSeconds: 10
          failureThreshold: 60                  # tolerate a long warm-up
      volumes:
      - name: shm
        emptyDir: {medium: Memory, sizeLimit: 32Gi}
      - name: model-cache
        persistentVolumeClaim: {claimName: hf-cache-vllm-llama31-70b}
---
apiVersion: v1
kind: Service
metadata: {name: vllm-llama31-70b}
spec:
  selector: {app: vllm-llama31-70b}
  ports: [{port: 80, targetPort: 8000}]
```

Points of note (every one of them an inference-shape):

- `runtimeClassName: nvidia` and the GPU toleration — the pod can land on GPU nodes and only on GPU nodes.
- `nodeSelector` on `nvidia.com/gpu.product` — H100-only; the autoscaler and the Karpenter `NodePool` must have H100 nodes to satisfy.
- `nvidia.com/gpu: 4` and `tensor-parallel-size=4` — the pod is a TP=4 replica; Kubernetes schedules it on a single node with 4+ H100s.
- `/dev/shm` sized at 32 GiB — NCCL's inter-process comms (TP collectives) use shared memory; the default pod `/dev/shm` is too small and NCCL will crash or stall.
- Model cache on PVC — the alternative (image embeds the weights) bloats image pulls. Chapter 04 revisits the trade-off at the cold-start layer.
- Readiness probe tuned for long warm-up — `initialDelaySeconds: 180` + `failureThreshold: 60 × 10s = 10 minutes` of grace. Default readiness probe timing will mark a vLLM replica unready before it finishes warming.

## The four substrate health checks

Before any production inference deployment lands on a GPU cluster, confirm:

1. **Every GPU node reports `nvidia.com/gpu > 0` in `Allocatable`.** `kubectl describe node <gpu-node>` — if the count is zero, the device plugin isn't advertising; stop and fix.
2. **Every GPU node runs the Operator's DaemonSets healthy.** `kubectl -n gpu-operator get pods -o wide` — one driver + toolkit + device-plugin + DCGM-exporter per GPU node; any CrashLoopBackOff is a substrate failure.
3. **A smoke test (`nvidia-smi` in a pod) lists the devices.** The pod spec at the top of this chapter is the test; run it and confirm.
4. **DCGM metrics are visible in Prometheus.** `count(DCGM_FI_DEV_SM_ACTIVE)` returns a count equal to the fleet's total GPUs; otherwise scraping is broken.

Any of these failing turns subsequent debugging into a nightmare — because every downstream symptom reads as "the inference pod is broken" when it's really the substrate under it. Chapter 02's scaler, chapter 03's sharing, chapter 04's warm pool, and chapters 05–06's resilience all assume (1)–(4) pass.

## Common deployment errors

### Error 1 — mismatched driver and container toolkit versions

The GPU Operator pins both together; a hand-rolled install usually doesn't. Symptom: `Driver/library version mismatch` from inside the container. Fix: use the Operator; or verify the toolkit's version is `<=` the driver's.

### Error 2 — missing `runtimeClassName: nvidia`

On clusters where the default runtime is `runc`, forgetting the runtime class means the container launches *without* GPU device injection — `nvidia.com/gpu` was requested, the pod schedules on a GPU node, but there's no `/dev/nvidia*` inside. Symptom: `No CUDA-capable device is detected` at engine startup even though the pod is on a GPU node.

### Error 3 — insufficient `/dev/shm`

NCCL's TP collectives use shared memory for intra-node comms. The default pod `/dev/shm` is 64 MiB (an inherited `tmpfs` default). Multi-GPU replicas crash at warm-up or hang mid-request. Fix: `emptyDir: {medium: Memory, sizeLimit: 32Gi}` mounted on `/dev/shm`.

### Error 4 — the autoscaler picks the wrong SKU

A 70B replica requires H100 / H200 / A100; a cost-aware autoscaler without node-affinity constraints may land it on a cheaper SKU with less HBM. The pod schedules; the engine OOMs on model load. Fix: nodeAffinity with the required SKU set; chapter 03 covers the sharing modes that may change this.

### Error 5 — CPU pods on GPU nodes

A missing taint on the GPU node, or a `tolerations: [{operator: Exists}]` on an unrelated CPU-only pod, lands cheap pods on expensive hardware. Fix: standard taint on GPU nodes, no over-broad tolerations in generic workloads.

### Error 6 — DCGM-exporter unscraped

The exporter runs but Prometheus isn't scraping it (ServiceMonitor misconfigured, scrape job missing). Downstream, chapter 02's HPA on `vllm_num_requests_waiting` works, but chapter 03's "did the HBM run out" check is blind. Fix: verify in a `promql` query at substrate stand-up, not at incident time.

## The boundary to the level-20 peer

This chapter does not teach:

- **Provisioning a Kubernetes cluster**, VPC / IAM / networking, Helm fundamentals, Terraform modules for a cluster, `kubeadm`-ing a self-managed cluster. All of that is at `ai-infra-engineer` (level 20).
- **Generic workload scheduling** that would be the same on a Flask API — Deployments, Services, Ingress, RBAC, PodSecurity.
- **CI/CD** of a Helm chart or a Kustomize overlay for the deployment. That is the MLOps peer (level 25).

This chapter *does* teach: the GPU-specific slice on top of all of the above, the operator-level reasoning about which SKU, which group, which taint, which runtime class, and the four health checks that confirm the substrate is right.

Chapter 07 completes the mapping table.

## Summary

- The GPU-aware substrate is five layers: driver, container-toolkit, device-plugin, node-labels, DCGM-exporter. Install them via the NVIDIA GPU Operator; hand-rolled installs are deprecated.
- A GPU pod requests `nvidia.com/gpu: N` (integer; fractional GPU is MIG / MPS / DRA — chapter 03), runs under `runtimeClassName: nvidia`, tolerates the `nvidia.com/gpu` taint, and (on multi-SKU fleets) selects the SKU with node affinity.
- The node autoscaler must be GPU-aware. Karpenter on AWS, Cluster Autoscaler elsewhere; one node pool per SKU is the robust shape. For multi-node groups (TP > 8, PP), reach for LeaderWorkerSet or gang scheduling.
- DCGM-exporter is the GPU observability layer the whole fleet depends on. Verify metrics are visible before any downstream work (chapter 02's HPA, chapter 03's sharing math).
- The four substrate health checks (`nvidia.com/gpu` allocatable, Operator DaemonSets green, `nvidia-smi` smoke test, DCGM in Prometheus) are the gate every GPU cluster passes before an inference pod lands.
- Common failure modes (version mismatch, missing `runtimeClassName`, undersized `/dev/shm`, wrong SKU, CPU-on-GPU-node, DCGM unscraped) are all substrate-level and all preventable with the health checks.
- Generic Kubernetes / Terraform / cluster provisioning routes to the level-20 peer; this chapter owns the GPU-specific slice.
