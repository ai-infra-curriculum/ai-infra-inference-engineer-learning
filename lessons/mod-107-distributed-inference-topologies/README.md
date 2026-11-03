# Module 107 — Distributed Inference Topologies and Disaggregated Serving

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 18

Modules 102–106 served a model that fit on one GPU, split one tenant's traffic across adapters, and treated the fleet as a flat set of identical workers. The models that pay most of the production inference bill — the frontier dense checkpoints at tens-to-hundreds of billions of parameters, the Mixture-of-Experts checkpoints at hundreds-of-billions-to-trillions of parameters, and the small models that need thousands of parallel workers — do not fit on one GPU, do not run efficiently on a flat fleet, and do not share a single topology. This module is where the fleet acquires shape: tensor-parallel groups within a node, pipeline-parallel groups across nodes, expert-parallel groups across experts, disaggregated prefill and decode worker pools talking over a KV router, and the interconnect (NVLink / InfiniBand / RoCE / RDMA) that makes any of it feasible.

By the end of it you can:

- **Pick a parallelism topology for a given model + fleet shape** — tensor parallel (TP) within a node when the model exceeds one GPU's HBM but fits in a node's aggregate HBM; pipeline parallel (PP) across nodes when it doesn't; expert parallel (EP) for MoE; data parallel (DP) attention for small models at high concurrency. Reason through the decision tree with the model's byte-footprint, batch shape, interconnect bandwidth, and SLO in hand.
- **Deploy TP and PP** on vLLM / SGLang / TensorRT-LLM — launch flags, world-size / rank layout, NCCL requirements, PP micro-batching and bubble overhead, interleaved schedules, and the health checks that confirm the engine is actually sharding (not silently falling back to replication).
- **Deploy disaggregated prefill-decode serving** — DistServe / Splitwise / NVIDIA Dynamo architectures — with a KV router between prefill and decode worker pools, cross-node KV transfer (NIXL / UCX / NCCL-based), separate TP / PP configurations for the two pools, and goodput-optimised placement that keeps TTFT inside its SLO without starving decode throughput.
- **Reason about collective-communication cost at operator depth** — NCCL AllReduce / AllGather / ReduceScatter patterns the engine issues per iteration; bus-bandwidth of NVLink vs. the latency floor of InfiniBand / RoCE / RDMA; the overlap-with-compute discipline that keeps the collective off the critical path. Do all of this without re-implementing NCCL — the kernel craft lives at the performance peer.
- **Deploy MoE serving at operator depth** — expert-parallel topology, token-dispatch and combine kernels, token-routing imbalance across experts, EP-aware batching, and the specific deployment patterns for DeepSeek V3, Mixtral, and Qwen-MoE.
- **Diagnose scaling cliffs** — a straggler rank in the TP group, PCIe-vs-NVLink topology mismatch, cross-NUMA / cross-socket noise, a cross-rack NIC bottleneck, and the placement / affinity fix that recovers the SLO.
- **Cite the boundaries** — to the performance peer (level 35) who implements the kernels and the tensor-ops this module dispatches to, and to `ai-infra-architect` (level 40) who owns multi-cluster / multi-region topology above this fleet.

> **Prerequisites:** [Module 101](../mod-101-inference-engineering-position/README.md) (TP / PP / EP vocabulary, interconnect vocabulary), [Module 102](../mod-102-inference-engine-internals/README.md) (vLLM / SGLang / TRT-LLM internals — the places the distributed executor hooks in), [Module 103](../mod-103-request-scheduling-and-batching/README.md) (continuous batching — the batching discipline composes directly with TP / PP shape), [Module 104](../mod-104-kv-cache-management/README.md) (KV cache as a runtime object — disaggregated serving ships the KV cache across nodes), and [Module 105](../mod-105-quantization-and-model-ops/README.md) (quantization footprints feed the "fits in a node" question that drives TP vs. PP choice).

## The core framing

Distributed inference is a **placement decision**, not a kernel decision. The question "where do the weights, the KV cache, the activations, and the collectives live" has a short list of defensible answers for any given model + fleet shape, and the wrong answer costs more than any quantization decision made at module 105. This module teaches you to make the placement decision — not to re-implement the kernels the placement dispatches to.

The dividing line that recurs: **operate the topology; defer the kernel to the performance peer.** Below.

## Boundary against `ai-infra-performance`

The distributed-inference side has two faces. One is the **kernel implementation** itself — NCCL ring / tree AllReduce, the TP-aware attention and MLP GEMMs, the Flash-Attention-with-TP collectives, the MoE dispatch / combine kernels, the FlashInfer-shaped sparse attention — which the peer at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) owns. If your diagnosis lands on "the AllReduce isn't overlapping with the GEMM epilogue" or "the MoE dispatch is spending 40 % of its time in the all-to-all" or "the TP-sharded MLP's reduce-scatter cost dominates at this batch size," the fix is a `.cu` / `.cuh` / Triton-DSL / CUTLASS edit — route to the performance peer.

The other face is the **production operation** — selecting TP / PP / EP / DP, laying out ranks on the fleet, binding NICs to NUMA domains, choosing between InfiniBand and RoCE, configuring NCCL algorithm hints, deploying DistServe / Splitwise / Dynamo as *topologies*, triaging a straggler rank. That face is this module. The deliverable here is a topology diagram, a launch config, a placement manifest, a NCCL env file, and a runbook — not a kernel edit.

Rule of thumb: *if the fix is to a kernel source file (`.cu` / `.cuh` / NCCL algorithm / CUTLASS epilogue / Flash-Attention variant) or to a collective's warp-level reduction, that is the performance peer. If the fix is to a launch flag, a rank layout, a NIC binding, a KV-router policy, or an expert-parallel group size, that is this module.*

## Boundary against `ai-infra-architect`

The level-40 peer [`ai-infra-architect`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning) owns **multi-cluster and multi-region** topology — the layer above a single serving fleet. If the question is "we have a US-East fleet, a US-West fleet, and an EU fleet; how do requests get routed between them; where does model promotion propagate first; what is the DR / failover posture across regions," that is the architect's scope.

This module stops at a single serving fleet. One cluster, one or many nodes within it, one or many disaggregated pools inside the cluster, one KV router per pool — but one cluster. The handshake to the architect: this module publishes the per-fleet goodput envelope (TP / PP / EP shape, expected RPS, expected TTFT / TPOT), and the architect decides how many such fleets to run and how to route between them.

Rule of thumb: *if the question is "which fleet does this request go to" or "which region holds the warm set" or "how do we fail over from US-East to US-West," that is the architect. If the question is "within this fleet, how are the ranks laid out and which collective they issue," that is this module.*

## Boundary against module 103

Module 103 owns the **scheduler** — iteration-level continuous batching, chunked prefill, preemption, and the admission-control knobs that keep queues sane on *one worker*. This module shapes the fleet of workers underneath the scheduler. The disaggregated prefill-decode architecture in chapter 03 here is a *fleet-level* generalization of the 103 scheduler's prefill-vs-decode mixing: 103 solves it with chunked prefill *inside one worker*; 107 solves it with *separate pools* when the mix at one worker is no longer tractable. Both are correct for their regime; the chapter-03 content here specifies when to graduate from one to the other.

## Boundary against module 104

Module 104 owns the **KV cache as a runtime object inside one worker** — PagedAttention blocks, RadixAttention prefix sharing, tiered KV stores (LMCache, Mooncake). Disaggregated serving in this module ships the KV cache *between workers*: a prefill worker produces a KV cache and a decode worker consumes it. The byte-layout and page semantics are 104's; the cross-node transport, routing, and placement are 107's.

## Boundary against module 106

Module 106 owns multi-model, multi-adapter, multi-tenant serving on a flat fleet. This module introduces *shape* into the fleet. The two compose: a thousand-adapter S-LoRA fleet whose base model is TP=4 — chapter 02 of this module teaches the sharding; chapter 01 of 106 teaches the adapter overlay on top of the sharded base.

## Chapters

1. [**Topology Selection — TP, PP, EP, DP for a Given Model and Fleet Shape**](01-topology-selection-tp-pp-ep-dp.md) — the four axes along which a model is partitioned; the decision tree that picks between them from the model's byte-footprint, batch shape, HBM per GPU, interconnect bandwidth, and SLO; the "small model, big batch" DP-attention case; the "doesn't fit on one node" PP boundary; the "MoE" EP boundary.
2. [**Tensor-Parallel and Pipeline-Parallel Deployment Mechanics**](02-tp-and-pp-deployment-mechanics.md) — how the engine actually shards weights, activations, and the KV cache under TP; the collectives per layer (AllReduce after attention, AllReduce after MLP); PP stage layout, micro-batching, and the bubble; the launch flags for vLLM / SGLang / TRT-LLM; the health checks that confirm sharding (not silent replication).
3. [**Disaggregated Prefill-Decode Serving**](03-disaggregated-prefill-decode.md) — DistServe / Splitwise / NVIDIA Dynamo architectures; the KV router between prefill and decode worker pools; cross-node KV transfer (NIXL, UCX, NCCL-based); separate TP / PP configurations for the two pools; goodput-optimised placement; when to graduate from chunked-prefill to disaggregated.
4. [**Collective-Communication Cost — NCCL at Operator Depth**](04-collective-communication-cost.md) — the NCCL AllReduce / AllGather / ReduceScatter patterns the engine issues per layer; ring vs. tree algorithms; NVLink bus-bandwidth vs. InfiniBand / RoCE latency floor; the overlap-with-compute discipline; `NCCL_DEBUG` and `nccl-tests` as the operator's instruments; where it ends and the performance peer's `.cu` edit begins.
5. [**MoE Serving — Expert-Parallel Topology and Routing Imbalance**](05-moe-and-expert-parallel.md) — the EP topology (experts partitioned across ranks); the dispatch / combine all-to-all collectives; token-routing imbalance and the capacity-factor knob; EP-aware batching; DeepSeek V3 (fine-grained + shared experts), Mixtral (8-expert top-2), Qwen-MoE deployment patterns.
6. [**Scaling Cliffs — Straggler Ranks, PCIe-vs-NVLink Mismatch, Placement**](06-scaling-cliffs-and-placement.md) — the four most common scaling cliffs (straggler rank, PCIe fallback, cross-NUMA noise, cross-rack NIC); how to detect each from per-rank timing and `nccl-tests`; the placement fixes (CPU / GPU affinity, NUMA binding, NIC-to-GPU pairing, rack-local placement); the diagnostic flowchart.
7. [**Boundaries — Performance Peer (L35) and `ai-infra-architect` (L40)**](07-boundaries-to-performance-and-architect.md) — what you ship to the performance peer when the fix is a kernel edit (trace, batch shape, GPU model); what you ship to the architect when the question is multi-cluster; the two-way interface contracts.

## Exercises

The exercises carry each chapter into a hands-on drill you keep as a reusable artefact for the rest of the track:

1. [**Topology Selection Drill per Model Shape**](exercises/exercise-01-topology-selection-drill-per-model-shape.md) — given five model + fleet shapes (small dense + high concurrency, mid-dense fits-one-GPU, mid-dense needs-TP, big dense needs-PP, MoE at serving scale), produce the topology choice, the byte-math behind it, the launch config, and the goodput estimate.
2. [**TP and PP Deployment Walkthrough on vLLM**](exercises/exercise-02-tp-and-pp-deployment-walkthrough-on-vllm.md) — stand up a TP deployment on a single node, a PP deployment across two nodes (or simulate with virtual pipeline stages), measure the per-collective cost, and publish a sharding-vs-replication health-check script.
3. [**Disaggregated Prefill-Decode with Dynamo or SGLang**](exercises/exercise-03-disaggregated-prefill-decode-with-dynamo-or-sglang.md) — deploy a disaggregated prefill-decode fleet with separate prefill and decode pools, a KV router between them, cross-node KV transfer; drive a long-context workload and publish TTFT / TPOT / goodput compared to a non-disaggregated baseline.
4. [**NCCL Collective Cost Walkthrough and Measurement**](exercises/exercise-04-nccl-collective-cost-walkthrough-and-measurement.md) — run `nccl-tests` on your fleet for AllReduce, AllGather, and ReduceScatter at the message sizes a real TP forward pass issues; map the measurements to the engine's observed latency; identify where the fleet is bandwidth-bound, latency-bound, or overlap-limited.
5. [**MoE Expert-Parallel Deployment and Imbalance Triage**](exercises/exercise-05-moe-expert-parallel-deployment-and-imbalance-triage.md) — deploy a Mixtral or DeepSeek-V2 / V3 (or Qwen-MoE) checkpoint with EP, measure per-expert utilization under a mixed workload, and triage a token-routing imbalance scenario; publish the capacity-factor and dispatch-cost table.
6. [**Straggler and Placement Diagnosis Drill**](exercises/exercise-06-straggler-and-placement-diagnosis-drill.md) — inject (or discover) a straggler rank, a PCIe-fallback rank, and a cross-NUMA mis-placement; produce the diagnostic trace for each; apply the placement fix; demonstrate the SLO recovery.

## Resources

See [`resources.md`](resources.md) for primary / citable references — Megatron-LM and the TP / PP papers; DistServe, Splitwise, and NVIDIA Dynamo documentation; the NCCL user guide and `nccl-tests` repository; the Mixtral / DeepSeek-V2 / DeepSeek-V3 / Qwen-MoE technical reports; vLLM, SGLang, and TRT-LLM distributed-serving documentation; the InfiniBand / RoCE / RDMA references; the Linux kernel NUMA documentation; and the peer-track pointers.

## Where this fits in the curriculum

- **Before:** [Module 106 — Multi-Model, Multi-Tenant, and Multi-Adapter Serving](../mod-106-multi-model-and-adapter-serving/README.md). The flat-fleet assumption that let 106 reason about adapters and tenants in isolation does not survive the moment a model needs TP or the fleet goes disaggregated — 107 is where the fleet acquires the shape under 106's adapters.
- **After:** [`mod-108-speculative-decoding-ops`](../mod-108-speculative-decoding-ops/README.md) (speculative decoding across TP-sharded draft and target), [`mod-109-autoscaling-fleet-and-resilience`](../mod-109-autoscaling-fleet-and-resilience/README.md) (fleet-level autoscaling that respects TP / PP / EP boundaries — you cannot scale half a TP group), [`mod-110-observability-slo-and-cost`](../mod-110-observability-slo-and-cost/README.md) (per-rank, per-stage, per-expert metrics the SLO methodology consumes), [`mod-113-production-safety-and-rollouts`](../mod-113-production-safety-and-rollouts/README.md) (canary and blue-green across topology-changing deployments).
- **Peer-track hand-off (performance):** every operator-depth chapter in this module names the explicit kernel-level cutoff where you stop debugging and route the issue to [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). Chapter 07 formalises the hand-off bundle.
- **Peer hand-off (architecture):** when the question grows beyond one cluster — multi-region routing, regional failover, cross-region model promotion — the methodology lives at [`ai-infra-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning). Chapter 07 writes the hand-off interface.
- **Peer hand-off (networking):** if the diagnosis lands on the IB / RoCE fabric itself (subnet manager misconfiguration, SHARP tree health, cable / port errors), the hand-off is to the data-centre networking team. This module teaches you to *recognise* the hand-off (chapter 04 and chapter 06), not to execute it.
