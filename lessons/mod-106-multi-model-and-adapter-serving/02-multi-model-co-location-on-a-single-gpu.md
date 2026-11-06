# Chapter 02 — Multi-Model Co-location on a Single GPU

> "One GPU, one model" is the easy case. The hard case is seventy small models (classifiers, routers, drafters, speech front-ends, embedding models) that together consume less HBM than one 70B — and have no business owning a GPU each.

## Motivation

A platform ends up with a long tail of **small models** — a 350M intent classifier, a 1B embedding model, two 1.5B draft models for speculative decoding (module 108), a Whisper variant, a safety classifier, a few 2–3B specialised fine-tunes. Each one, if deployed on its own GPU, uses perhaps 2–5 % of an H100's HBM and runs at a few percent duty cycle. The fleet becomes expensive and underutilised; cost per inference is nonsense.

Co-locating multiple models on one GPU is the fix. But there is no single mechanism — there are three stacked layers at which co-location can happen, each with different latency, throughput, and isolation properties.

By the end of this chapter you can:

- Enumerate the three co-location mechanisms and name what each guarantees (and does not guarantee).
- Draw the latency / throughput / isolation triangle and place each mechanism on it.
- Decide when to use MIG, when to use MPS, when to use a multi-model engine, when to use K8s DRA, and when to just binpack separate pods.
- Launch and verify a two-model co-location on a single GPU using at least one of the mechanisms.
- Recognise the noisy-neighbour signature of soft sharing and route the fix correctly.

Partition tables, MPS behaviour, and Kubernetes DRA API versions all evolve across driver, plugin, and Kubernetes releases. The citations in this chapter point to current primary sources (NVIDIA MIG User Guide, MPS Service User's Manual, Kubernetes DRA docs), but always confirm against the exact driver / plugin / cluster version your fleet runs before planning a deployment.

## The three layers of co-location

Pick the right layer first — the trade-offs cascade from there.

| Layer | Mechanism | Hardware or OS primitive | HBM share | Compute share | Isolation | Scheduling |
|---|---|---|---|---|---|---|
| **Engine-level** | vLLM / SGLang multi-model server, Triton model-ensemble | None (just process-internal) | Shared HBM pool | Shared SMs, time-sliced by the engine's scheduler | Soft; one noisy model can starve others | Managed by the engine |
| **Hardware-level — hard partition** | **MIG** (Multi-Instance GPU) | Hopper / Ampere / Blackwell MIG partitions | Partitioned HBM (fixed sizes per profile) | Partitioned SMs | Hard; a MIG partition is physically isolated | Each partition appears as a separate CUDA device |
| **Hardware-level — soft sharing** | **MPS** (Multi-Process Service) | NVIDIA MPS daemon | Shared HBM | Shared SMs with CUDA stream multiplexing | Softer than MIG, harder than engine-level | Each client is a separate process; MPS multiplexes |
| **Kubernetes-level** | **DRA** (Dynamic Resource Allocation) and the GPU operator | K8s scheduler | Depends on claimed resource | Depends on claimed resource | Depends on claim | K8s scheduler claims a MIG partition / full GPU / MPS slice per pod |
| **Process-level** | One pod per model, K8s binpack | None (just scheduling) | Full GPU per pod | Full GPU per pod | Full isolation | K8s binpacks pods onto nodes; one GPU per pod |

### Layer 1 — Engine-level (one process, many models)

The engine loads multiple models into one process and routes requests to the right one. **vLLM's `MultiModelEngine` / `OpenAIServer` with `--served-model-name` tags, SGLang's model-router server, Triton Inference Server's model-ensemble and dynamic-batcher**. All are process-internal.

**What this gives you:**

- **Memory efficiency.** One CUDA context, one Python process, one PyTorch allocator, one engine scheduler. No per-model process overhead.
- **Shared KV pool (where supported).** Not every engine lets models share a KV pool, but when they do, the per-tenant / per-model fragmentation of module 104's budget is reduced.
- **Simple ops.** One container, one healthcheck, one metrics endpoint.

**What it does not give you:**

- **Isolation.** One model's traffic spike starves the others. The engine's scheduler is responsible for fairness; if the scheduler does not know about per-model priorities, there is no isolation.
- **Independent versioning.** Both models move together on the engine release. One is tied to the other's upgrade cadence.
- **Different engines.** You cannot run vLLM and SGLang in the same process. If one of the co-located models wants a different engine, you need multiple processes (next layer).

### Layer 2a — MIG (hardware hard partitioning)

**MIG** carves a physical GPU into fixed-size slices, each with its own HBM allocation, SM allocation, L2 slice, memory controllers, and copy engines. Introduced on Ampere (A100) and extended through Hopper (H100, H200) and Blackwell (B100 / B200 / GB200). Each MIG slice appears to CUDA as its own device.

**MIG profiles on H100 80 GB** (indicative; verify against the current `nvidia-smi mig --list-gpu-instance-profiles` for your hardware):

- `1g.10gb` — 1/7 of SMs, 10 GiB HBM. Up to 7 instances.
- `1g.20gb` — 1/7 of SMs, 20 GiB HBM (the "extended" profile). Up to 4 instances.
- `2g.20gb` — 2/7 of SMs, 20 GiB HBM. Up to 3 instances.
- `3g.40gb` — 3/7 of SMs, 40 GiB HBM. Up to 2 instances.
- `4g.40gb` — 4/7 of SMs, 40 GiB HBM. Single instance.
- `7g.80gb` — the full GPU; one instance.

The table above matches the H100 80 GB SXM5 / PCIe entries in the current [NVIDIA MIG User Guide's supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html); an additional `1g.10gb+me` (media-extensions) variant exists for workloads that need the dedicated media engine. Blackwell and GB200 profiles differ — consult the same guide for the hardware generation your fleet runs.

**What this gives you:**

- **Hard isolation.** One partition's noisy workload cannot affect another partition's latency. HBM and SM are physically separated. ECC errors scope to a partition.
- **Independent CUDA devices.** Each partition is a separate device; each can run its own CUDA context, driver mode, and engine process.
- **Deterministic performance.** Benchmarks on a 1g.20gb slice are reproducible across runs regardless of what the other slices do.

**What it does not give you:**

- **Flexibility.** Partition profiles are fixed. You cannot "borrow" SMs from the idle partition next door for a burst.
- **Full-GPU peak throughput.** A 1g.20gb slice has 1/7 the SMs. For throughput-bound workloads, it is strictly 1/7 the peak flops of the full GPU.
- **NVLink.** Partitions do not share NVLink; you cannot build a TP=2 topology across two slices of the same GPU.

MIG is the right primitive for **small models that need latency SLOs** — safety classifiers, embedding models, small draft models — where you need hard isolation but can live with fixed sizes.

### Layer 2b — MPS (hardware soft partitioning)

**MPS** (NVIDIA Multi-Process Service) is a daemon that multiplexes multiple CUDA client processes onto one GPU's SMs. Clients submit work to one CUDA context (owned by the MPS daemon) that interleaves their work. Volta-generation MPS introduced the per-client SM-percentage cap via `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`; CUDA 11.5 added the per-client pinned-device-memory cap via `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` ([NVIDIA MPS environment variables](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html), [CUDA 11.5 release notes](https://developer.nvidia.com/blog/revealing-new-features-in-the-cuda-11-5-toolkit/)).

**What this gives you:**

- **Flexibility.** Clients can share the full GPU; one client can burst to use idle capacity.
- **Independent processes.** Each client runs its own engine; versioning is per-client.
- **Reduced context-switch overhead.** Multiple processes without MPS serialise on the GPU; with MPS they interleave.

**What it does not give you:**

- **Hard isolation.** A noisy client can still affect another's SM scheduling. Memory limits are enforced but SM percentage is a soft hint, not a guarantee (since CUDA 11.5 the thread-percentage cap became a soft ceiling that processes can exceed when peers are idle).
- **ECC scope.** An uncorrectable ECC error takes down the whole GPU.
- **Determinism.** Latency distributions widen under contention.

MPS is the right primitive for **throughput-oriented small-model workloads** where latency variance can be tolerated — bulk embedding, background classification — and for pre-production environments where fast iteration beats hard isolation.

### Layer 3 — Kubernetes DRA

**DRA** (Dynamic Resource Allocation, Kubernetes KEP-4381 — see [`resources.md`](resources.md)) is the K8s API for flexible resource claims that supersedes the earlier device-plugin model. For GPUs, a DRA driver (NVIDIA ships one) lets a pod claim one of:

- A full GPU.
- A specific MIG partition profile.
- An MPS slot.
- A fractional claim (vendor-specific).

K8s schedules the pod to a node that can satisfy the claim; the DRA driver materialises the claim as a visible CUDA device inside the pod.

**What this gives you:**

- **Scheduling flexibility.** The platform decides at pod-admission time whether a workload wants a full GPU, a MIG partition, or an MPS slot. One cluster can serve all three.
- **Standard interface.** The pod spec is K8s-native; the GPU-specific details live in a DRA class.

**What it does not give you:**

- **A new isolation primitive.** DRA is a scheduling abstraction over MIG / MPS / full GPUs; it inherits each primitive's trade-offs.
- **Universal support.** DRA entered alpha in Kubernetes v1.26, was redesigned around structured parameters (KEP-4381) in v1.31, graduated to beta in v1.32, and reached GA in v1.34 with `resource.k8s.io/v1` ([Kubernetes DRA concepts](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), [KEP-4381](https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/4381-dra-structured-parameters)). The NVIDIA DRA driver ([kubernetes-sigs/dra-driver-nvidia-gpu](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu), previously under `NVIDIA/`) ships ComputeDomains (multi-node NVLink) as officially supported while per-GPU allocation is still gated behind an opt-in — verify both your cluster version and the driver feature set before relying on DRA in production; many clusters still run the earlier device-plugin model.

### Layer 4 — Process-level (one pod per model, binpacked)

The simplest option: one pod per model, each requesting a `nvidia.com/gpu: 1` resource; K8s binpacks pods onto nodes. If a model needs less than one GPU, you either accept the waste or combine it with one of the above layers.

**What this gives you:**

- **Full isolation.** One process per model, one GPU per process. No cross-contamination.
- **Independent lifecycle.** Each model has its own container, versioning, rollout.
- **Simple ops.** No MIG partition management, no MPS daemon.

**What it does not give you:**

- **Density.** A 1B model on an H100 wastes 95 % of the HBM.
- **Dynamic reconfiguration.** Rebalancing requires pod restarts.

Use for **large models that fill a GPU anyway**, and as the fallback when the co-location primitives are unavailable or overkill.

## The latency / throughput / isolation triangle

The right choice is a point in a triangle:

- **Latency P99** (how predictable is tail latency under contention) favours MIG > one-pod-per-model > MPS > engine-level.
- **Throughput** (aggregate requests per second from one GPU) favours engine-level > MPS > MIG (fixed partitions limit flops) > one-pod-per-model (no co-location).
- **Isolation** (noisy-neighbour defence, blast radius of a crash, ECC scope) favours one-pod-per-model > MIG > MPS > engine-level.

A workload that *needs* P99 isolation (customer-facing latency SLO on a small model) → MIG. A workload that is throughput-oriented and tolerates variance (batch embedding generation) → engine-level or MPS. A workload that needs a specific engine the co-location primitive does not support → one-pod-per-model.

## Launching co-locations

### Engine-level — vLLM OpenAI server with multiple models

A single vLLM `OpenAIServer` process still serves exactly one base model (plus any LoRA adapters attached to it); to co-locate distinct base models you run one vLLM process per model and front them with an external router (nginx, LiteLLM proxy, llm-d, production-stack) ([vLLM discussion — Run multiple models](https://discuss.vllm.ai/t/run-multiple-models/1181)). For true in-process multi-model co-location, use Triton Inference Server (below) or the one-pod-per-model binpacked layer.

### Engine-level — Triton model-ensemble

Triton Inference Server supports many models in one process natively. Each model has its own `config.pbtxt`; the server exposes each at `/v2/models/<name>/infer`. For LLMs, Triton with the `tensorrtllm_backend` serves one or more TRT-LLM models from one Triton process; the backend handles the inflight batcher per model.

Minimum model repository:

```
/models/
├── classifier-350m/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan           ← TRT engine or other backend artifact
├── llama-3-8b/
│   ├── config.pbtxt
│   └── 1/
│       └── <TRT-LLM engine files>
└── whisper-small/
    ├── config.pbtxt
    └── 1/
        └── <ONNX / TRT engine>
```

Launch: `tritonserver --model-repository=/models`. The three models share one process and one CUDA context; the server routes to each by model name.

**Operator considerations:**

- Each model competes for GPU resources through the shared CUDA context. Triton's dynamic batcher is per-model; it does not know about cross-model priority.
- Model-version updates are per-model (`<name>/<version>/`); you can roll a classifier without disturbing the LLM.
- Model-ensemble (as a feature) also lets you chain models in a pipeline — classifier → LLM — in one request, which is a different use case from co-location.

### Hardware — MIG partition on H100

```
# Enable MIG mode on GPU 0.
sudo nvidia-smi -i 0 -mig 1

# List supported profiles.
nvidia-smi mig --list-gpu-instance-profiles

# Create partitions. Example: 3 × 1g.20gb (fit depends on hardware revision).
sudo nvidia-smi mig -cgi 1g.20gb,1g.20gb,1g.20gb -C

# Verify.
nvidia-smi -L
# -> GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-...)
#   MIG 1g.20gb Device 0 (UUID: MIG-...)
#   MIG 1g.20gb Device 1 (UUID: MIG-...)
#   MIG 1g.20gb Device 2 (UUID: MIG-...)
```

Each `MIG-...` UUID is targetable from `CUDA_VISIBLE_DEVICES=MIG-<UUID>` or via the K8s GPU operator's MIG strategy. Launch a vLLM or Triton process pinned to one MIG and it sees that partition as `cuda:0`.

All H100 SKUs (PCIe 80 / 94 GB and SXM5 80 / 94 GB) are MIG-capable with the profiles above, but older generations are not — the A40 and A10, for example, do not support MIG at all. Always confirm with `nvidia-smi mig --list-gpu-instance-profiles` on the exact SKU and driver your fleet runs ([NVIDIA MIG — supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)).

### Hardware — MPS

```
# Start the MPS daemon.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Set per-client limits (requires CUDA 11.5+ for percentage limits).
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=10240            # MiB

# Launch the client process with the MPS environment set.
python -m vllm.entrypoints.openai.api_server --model classifier --port 8001
```

Each client inherits the SM-percentage and memory-limit environment. MPS multiplexes them.

### K8s DRA

A ResourceClaim example (schema-indicative; verify against your DRA driver's current CRD):

```yaml
apiVersion: resource.k8s.io/v1alpha3
kind: ResourceClaim
metadata:
  name: classifier-gpu
spec:
  resourceClassName: nvidia.com/mig-1g.20gb
---
apiVersion: v1
kind: Pod
metadata:
  name: classifier
spec:
  containers:
    - name: classifier
      image: ...
      resources:
        claims:
          - name: gpu
      env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: $(RESOURCE_CLAIM_UUID)
  resourceClaims:
    - name: gpu
      resourceClaimName: classifier-gpu
```

The DRA API has progressed `v1alpha3` → `v1beta1` (Kubernetes 1.32) → `v1beta2` (1.33) → `v1` (1.34 GA), with field renames along the way — most notably `ResourceClass` → `DeviceClass` and `resourceClassName` → `deviceClassName` inside structured device requests. Treat the snippet above as structural; use `resource.k8s.io/v1` on 1.34+ clusters and verify the exact shape against your cluster's DRA version ([Kubernetes DRA concepts](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/)).

## Common failure modes

### Noisy neighbour on engine-level co-location

Model A has a traffic spike; model B's TTFT P99 doubles. The engine scheduler has no cross-model priority. Mitigation: either (a) move the latency-critical model to a MIG partition, or (b) implement per-model admission control in the engine (not all engines expose this). The structural fix is the layer choice; the quick fix is admission.

### MIG partition "too small"

A 1g.20gb MIG partition holds a 7B BF16 model (14 GiB) with 6 GiB left for KV. The KV budget is too small for the context length. Options: pick a larger partition profile, quantise the model (module 105 reduces weight bytes to leave more KV budget), or scale down the context length. Capacity planning is the chapter-04 byte math; MIG is just the partition.

### MPS crash takes down co-located workloads

An ECC error, an unhandled OOM, or a driver fault on an MPS-shared GPU terminates all clients. There is no partition-level recovery as there is with MIG. Mitigation: use MIG for workloads where blast radius matters; use MPS where it does not.

### DRA claim denies pod scheduling

The pod requests a MIG profile the node does not have carved out. K8s cannot schedule it. Fix: either carve the right partitions on the node ahead of time (the NVIDIA GPU operator supports a `mig-strategy: mixed` mode that pre-creates a mix of profiles), or use a dynamic-MIG mode where partitions are created on demand (newer; verify support).

### Triton model-ensemble routing confusion

Multiple models in one Triton process; one model is updated with a breaking config change; Triton fails to reload only that model but marks the whole process unhealthy. Readiness gate drops the whole pod from the load balancer. Fix: use the Triton explicit model-control API and gate the rollout of individual models; do not rely on auto-reload for breaking changes.

### Hidden PCIe contention

Two models co-located on one GPU each do their own weight-load from host memory, each hitting the same PCIe link. One starves the other during startup. Observable as a cold-start latency spike unrelated to the model's HBM-resident operation. Mitigation: serialise cold-starts; pre-pull weights into page cache; stagger pod starts.

## The boundary to the performance peer

This chapter does not teach:

- How MIG's hardware isolation is implemented in the GPU scheduler.
- How MPS's CUDA stream multiplexing interacts with Tensor Core utilisation.
- Why some batched kernels under-utilise a MIG partition that has fewer SMs than the kernel was tuned for.
- How a kernel should detect it is running under MPS and adapt its grid dimensions.

Those are hardware / kernel questions. If your co-location drill reveals a specific kernel underperforming on a MIG partition relative to the full GPU — not just scaled by the SM ratio but worse — route to [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). The hand-off bundle: SM count on the MIG partition, kernel timing on the partition vs. the full GPU normalised by SM ratio, batch shape, Nsight Compute capture.

## Decision framework — picking the layer

1. **Does the workload need hard P99 isolation under contention?** Yes → **MIG** (if hardware supports; verify profile fits) or **one-pod-per-model**. No → consider softer options.
2. **Does the workload need the full GPU's flops when under load?** Yes → **MPS** or **engine-level** co-location. No → MIG is fine.
3. **Do the co-locating workloads want different engines (vLLM + Triton)?** Yes → separate processes (**MPS** or **one-pod-per-model**). No → **engine-level** possible.
4. **Does the platform have DRA?** Yes → prefer **DRA** to request the right primitive declaratively. No → fall back to the operator / device-plugin configuration.
5. **Is this a long-tail of small models (classifier, embedder, draft-model)?** Yes → **engine-level** (Triton model-ensemble) or **MIG small profiles** are both good; prefer MIG when customer-facing SLOs attach.

## Summary

- Multi-model co-location is a decision across three stacked layers: engine-level (one process, many models), hardware-level (MIG hard partitions or MPS soft sharing), and process-level (one pod per model binpacked).
- The latency / throughput / isolation triangle: MIG gives hard isolation at fixed throughput ceilings; MPS gives soft isolation with flexibility; engine-level gives density but no isolation; one-pod-per-model gives full isolation at density cost.
- Kubernetes DRA is the modern claim API that spans these primitives; it inherits each one's trade-offs.
- Launch mechanics differ: Triton model-ensemble is the canonical multi-model engine pattern; MIG requires partition creation ahead of time; MPS requires the daemon running and per-client limits set.
- Noisy-neighbour and blast-radius questions are structural, not tuning; the fix is the layer choice.
- Kernel-level efficiency on co-located layouts routes to the performance peer.
