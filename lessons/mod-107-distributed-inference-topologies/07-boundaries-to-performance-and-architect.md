# Chapter 07 — Boundaries: Performance Peer (L35) and `ai-infra-architect` (L40)

> This module operates distributed inference topologies. Two neighbours own the territory on either side — the performance peer implements the kernels and tensor-ops the topology dispatches to; the architect decides which fleets exist and how requests move between them. Both boundaries are frequently crossed in the wrong direction. This chapter writes the fence.

## Motivation

Two confusions recur in distributed-inference triage, and both waste weeks:

1. **Diagnosing "the collective is slow" as a kernel bug** when it is a placement choice (chapter 06). The symptom lands on a Nsight timeline; the operator opens a Jira against the engine project; the engine team closes it with "working as intended on supported topology; your `nvidia-smi topo -m` shows PCIe-only connectivity between ranks 2 and 5."
2. **Diagnosing "cross-region latency is bad" as a fleet-local topology bug** when it is a multi-cluster routing choice. The symptom lands on an SLO burn; the operator reshapes the TP / EP topology; the SLO still burns because every third request still hops across continents.

Both have correct homes. This chapter names them.

By the end of it you can:

- State what this module owns, what the performance peer owns, and what the architect owns — in table form.
- Write the hand-off bundle for the performance peer (what trace, which flags, which measurement, what to try first).
- Write the hand-off bundle for the architect (what per-fleet envelope, what SLO, what capacity ask).
- Avoid the three common cross-boundary mistakes (diagnosing kernel issues from the fleet-plan view; diagnosing cross-cluster issues from the fleet-local view; blaming the engine for a placement cliff).

## The three-way boundary table

| Concern | This module (`ai-infra-inference-engineer`, L35) | `ai-infra-performance` (L35, peer) | `ai-infra-architect` (L40, above) |
|---|---|---|---|
| **Which parallelism topology** | ✅ decides, deploys | — | — (consumes the fleet's advertised envelope) |
| **TP / PP / EP launch flags** | ✅ sets, verifies | — | — |
| **NCCL env + algorithm hints** | ✅ `NCCL_IB_HCA`, `NCCL_P2P_LEVEL`, `NCCL_SOCKET_IFNAME` | ✅ `NCCL_ALGO`, `NCCL_PROTO`, custom plugins | — |
| **NCCL internals** (ring/tree/CollNet algorithm, warp-level reduction) | — | ✅ | — |
| **Flash-Attention + TP collectives (kernel-level)** | — | ✅ | — |
| **DeepEP / FastMoE / grouped-GEMM MoE kernels** | — | ✅ | — |
| **Expert redistribution policy** | ✅ decides policy | — (implements the kernel if needed) | — |
| **Capacity factor tuning** | ✅ | — | — |
| **Routing imbalance diagnosis** | ✅ measures, mitigates (chapter 05) | — (if the fix is a kernel edit) | — |
| **KV transport (NIXL / UCX)** | ✅ enables, binds | ✅ internal tuning | — |
| **GPUDirect RDMA enablement** | ✅ driver + firmware checklist | — | — |
| **Scaling-cliff diagnosis (chapter 06)** | ✅ | — (if the fix is a kernel edit after placement is correct) | — |
| **Node-level placement (NUMA / NIC-to-GPU)** | ✅ | — | — |
| **Rack-local scheduling constraints** | ✅ (within one cluster) | — | — (names the cluster topology) |
| **Which clusters exist** | — | — | ✅ |
| **Cross-cluster request routing** | — (fleet publishes its envelope) | — | ✅ |
| **Multi-region failover posture** | — | — | ✅ |
| **Cross-region model propagation** | — | — | ✅ (methodology) + module 105 (artefact mechanics) |
| **Capacity planning across clusters** | — (publishes envelope) | — | ✅ |
| **Per-fleet goodput envelope** | ✅ publishes | — | — (consumes) |
| **Per-cluster SLO policy** | ✅ enforces within a fleet | — | ✅ sets across clusters |

## The hand-off to the performance peer

When to hand off:

- `nccl-tests` runs match the hardware spec (chapter 04 ballparks). Chapter 06 placement is clean. The engine still spends more time in collectives than theory predicts.
- Overlap-with-compute is not working on the specific message-size regime your workload produces. Nsight shows the collective time appearing additively in iteration time rather than hidden.
- A specific kernel (Flash-Attention-with-TP, DeepEP dispatch, grouped-GEMM for MoE experts) is slower than published benchmarks on the same hardware.
- The engine's auto-tuner picks the wrong algorithm for your message-size curve (e.g. Ring at a size where Tree would win).
- A model's layer kernel has specific issues (sparsity, unusual shapes) that need a kernel edit.

### The hand-off bundle

Package before escalation:

```
# Topology
engine: vllm 0.6.3 / sglang 0.3.4 / trt-llm 0.14 (be specific)
world_size: 16
TP: 8  PP: 1  EP: ... (if MoE)
model: meta-llama/Llama-3.1-70B-Instruct @ <HF revision SHA>
quantisation: BF16 / FP8 / INT4 (module 105 reference)

# Hardware
nodes: 2 × 8× H100 SXM 80GB
interconnect: NVLink intranode (NVSwitch), ConnectX-7 400 Gbps IB internode
driver: 535.129.03  CUDA: 12.4  NCCL: 2.21.5

# Placement verified (chapter 06)
nvidia-smi topo -m: all GPU pairs NV#; no SYS
NCCL_IB_HCA: mapped per-rank
NUMA: each worker pinned to local socket
Rack-local: all 16 GPUs in one rack

# Baseline measurements
nccl-tests all_reduce_perf at 32 MiB (prefill-shape message): <bus-bw> GB/s
nccl-tests all_reduce_perf at 128 KiB (decode-shape message): <bus-bw> GB/s
Published spec on this hardware: <reference link + numbers>
Deficit: X% below spec

# Observed symptom
Workload: <N> QPS, prompts <mean/p99 tokens>, generations <mean/p99 tokens>
TTFT: p50 <ms>, p99 <ms>
TPOT: p50 <ms>, p99 <ms>
Expected TPOT from compute-only math: <ms>
Expected TPOT with collective-cost added (chapter 04 model): <ms>
Observed: <ms> — gap of <ms> is unaccounted for.

# Nsight trace
Attached: ns-rep for a 20-iteration window.
Per-rank timeline shows: <specific observation — e.g. "AllReduce kernel takes 1.2 ms where chapter 04 predicts 0.3 ms">

# What we've already tried
- NCCL_ALGO=Ring forced: same.
- NCCL_PROTO=LL128: same.
- Reduced TP from 8 to 4: TPOT improved by 15 % (collective smaller), still above target.
- `nccl-tests` on a bare node pair: matches spec.

# Specific question
Why is the engine's AllReduce kernel 4× slower than nccl-tests at the same message size?
(Suspicion: the overlap isn't firing on our batch shape.)
```

The performance peer takes this bundle and makes it their first-class task. The operator does not re-write the kernel; the operator defines the test.

### What NOT to send to the performance peer

- "The topology is slow." Too vague.
- A screenshot of a Grafana dashboard. Not the right primitive; traces, measurements, and configs are.
- A request to "re-tune NCCL." If the operator-level NCCL env is right, the kernel-level tuning is the peer's judgment; don't prescribe.
- A request to "optimise this." Direction without a measurement is noise.

## The hand-off to the architect

When to hand off:

- Request rates have grown beyond one cluster's goodput envelope. Need to decide whether to scale this cluster (horizontally within), add another cluster, or add a region.
- A specific tenant has a geographic requirement (data residency, latency) that one cluster cannot satisfy.
- Capacity planning cycle: what hardware to buy for next year, where to put it, which models to pre-position.
- Failover design: if this cluster goes down, where does traffic go? What is the acceptable blast radius?
- Model-launch coordination: a new model needs to be in US-East, EU-West, and Singapore within 24 hours — who owns the propagation?

### The hand-off bundle for the architect

Package the per-fleet envelope:

```
# Fleet identity
cluster: us-east-prod-llama70b-01
status: production, serving
owner-team: inference-platform

# Topology
TP=4 within node, DP=5 replicas → 20× H100 SXM, 2.5 nodes rounded to 3.
Disaggregated: 4 prefill workers (TP=4), 8 decode workers (TP=2), one KV router.
Pool shapes: documented at <link>.

# Goodput envelope (what this fleet can serve inside SLO)
Target SLO: TTFT p99 ≤ 400 ms, TPOT p99 ≤ 50 ms.
At current config, sustained goodput: 1,800 QPS.
Peak goodput (short bursts): 2,400 QPS.
Degradation beyond peak: TTFT inflates to > 1 s; auto-reject at 2,400 QPS.
Measurement window: 7 days, p99 at the whole window.

# Workload profile
Mean prompt: 1,200 tokens. P99: 8,000 tokens.
Mean generation: 300 tokens. P99: 2,000 tokens.
Diurnal pattern: 3× daytime peak over nighttime.
Tenant mix: 60% gold-tier (tight SLO), 40% silver (loose SLO).

# Current utilisation
Average: 68% of goodput envelope.
P99 hour: 92% of envelope.
Running at near-saturation 3 hours per day.

# Available knobs within fleet
- Horizontal scaling: can add 1-2 more replicas before interconnect-cost cliff.
- Quantisation: moving to FP8 would free ~X% capacity; needs module-105 regression gate.
- Model variant: smaller model (Llama-3.1-8B) for silver tier would free ~Y% capacity; needs routing-policy change.
- Disaggregation tuning: pool rebalancing (fewer prefill, more decode) is in-flight.

# Capacity ask
By end-of-quarter, expected request rate is 2,800 QPS.
Options:
A. Add one more replica (same topology): adds ~400 QPS; ~400k USD/year cost; one more node.
B. Add a second cluster in same region (us-east-prod-llama70b-02):
   adds full fleet goodput envelope; ~2M USD/year; multi-cluster routing.
C. Shift silver-tier traffic to a smaller model (Llama-3.1-8B); frees ~500 QPS on this fleet.

# Non-topology questions
- Should we add a US-West fleet for latency to West-coast tenants?
- Compliance: EU tenants' PII must stay in EU — do we need a Frankfurt fleet?
- DR: current failover plan is <X>; is that acceptable for the business continuity SLA?

# Decision needed
Architect to pick A / B / C / combination by <date>, and sign off on the capacity acquisition.
```

The architect takes this envelope and designs across the organization's fleets.

### What NOT to send to the architect

- "We have a scaling cliff." That is a chapter-06 diagnostic, not an architect question.
- "The engine is slow." Not an architect question.
- A low-level kernel benchmark. Noise.
- A multi-cluster question *and* a within-fleet performance regression bundled together — split them; they have different owners.

## The three common cross-boundary mistakes

### Mistake 1 — opening a kernel ticket for a placement cliff

**Shape.** TPOT regressed; operator sees NCCL kernel on Nsight; files a bug against the engine project ("AllReduce slow"). Engine team closes it ("your `nvidia-smi topo -m` shows PCIe-only").

**Fix.** Chapter 06 flow is the gate. Run it before opening any kernel ticket. If chapter 06 shows a placement issue, chapter-06 fix is the resolution; the ticket is to the fleet-infrastructure team (NVLink bridge install, NUMA pinning), not to the engine.

### Mistake 2 — reshaping the topology to fix a multi-cluster latency

**Shape.** EU tenants' p99 TTFT is bad. Operator moves EU traffic to a smaller TP topology to reduce the collective floor. P99 is still bad because every third EU request is going to a US cluster over 100 ms of trans-Atlantic latency.

**Fix.** Chapter-07 escalation: the question isn't within-fleet topology; it's which-cluster routing. Hand off to the architect; the architect scopes an EU cluster or validates that EU-local routing is correct.

### Mistake 3 — the architect micro-managing fleet-local topology

**Shape.** Architect dictates "all production fleets must use TP=8" based on a cross-cluster consistency goal. One fleet's workload would be better served by TP=4 + more replicas; the architect's rule locks in sub-optimal fleet shape.

**Fix.** The three-way table above. Topology within a fleet is this module's decision; the architect provides constraints (budget, SLO, data boundaries), not implementation. Push back politely.

## The scope of multi-cluster — a brief sketch (deferred to the architect)

Multi-cluster adds concerns this module does not teach:

- **Routing policy across clusters.** Latency-based routing, geographic routing, affinity routing, cost-based routing, failover routing — each with trade-offs.
- **Replication posture.** Active-active, active-standby, follow-the-sun; model-weight propagation; configuration synchronization.
- **Data boundaries at the region level.** GDPR, data-residency SLAs; which tenants' traffic can land in which cluster.
- **Capacity shaping across regions.** Pre-positioning capacity; forecast errors; procurement lead times.
- **Model-launch coordination.** New model propagated to all regions atomically or staggered?
- **Cross-region observability.** Correlation across cluster boundaries; distributed-tracing semantics.

Each of these is handled at [`ai-infra-architect-learning`](https://github.com/ai-infra-curriculum/ai-infra-architect-learning). This module's job is to publish the per-fleet envelope the architect's decisions are made against — and to raise questions at the right boundary.

## The scope of kernel craft — a brief sketch (deferred to the performance peer)

Kernel craft adds concerns this module does not teach:

- **Writing / tuning Flash-Attention variants.** FA2, FA3, FA-TP, chunked-prefill-aware FA — all `.cu` files.
- **CUTLASS epilogues for FP8 / FP4 GEMMs.** Fusion opportunities with the collective.
- **NCCL algorithm implementations.** Ring, tree, double-tree, CollNet. The operator chooses `NCCL_ALGO`; the peer writes the algorithm.
- **DeepEP / FastMoE / grouped-GEMM.** Hopper-specific MoE kernels.
- **Overlap-with-compute at kernel level.** Stream management, CUDA graphs, event synchronization.
- **Communication-computation fusion.** The MLP-GEMM-with-ReduceScatter-epilogue family of fused ops.
- **Numerical-stability tuning.** FP8 accumulation; FP4 quantization-aware kernels; mixed-precision softmax for routers.

Each is handled at [`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning). This module's job is to recognize when the diagnosis has crossed into kernel territory and route it, with a complete bundle, to the peer.

## Composition with other modules in this track

- **Module 103 scheduler decisions.** The scheduler composes with topology (chapter 02): continuous batching lives on top of TP; chunked prefill still runs on sharded workers. Scheduler-level fixes are 103, not this module.
- **Module 104 KV-cache.** Chapter 03's disaggregated serving ships the KV cache across workers; the byte-layout is 104's. Fixes to the paged-attention layout or the prefix cache are 104, not this module.
- **Module 105 quantization.** The quantization decision changes the byte-footprint and therefore the topology (chapter 01). Iteration between 105 and 107 is normal.
- **Module 106 multi-model / adapters.** Multi-LoRA composes with a TP-sharded base; the sharding is this module, the adapter overlay is 106.
- **Module 108 speculative decoding.** Draft-target speculative decoding over a TP-sharded draft is both modules' concern; the operator-level deployment is 107 + 108 together.
- **Module 109 autoscaling.** Autoscaling respects TP / EP boundaries — scale replicas, not partial TP groups. The autoscaler policy is 109; the group-shape it must respect is this module.
- **Module 110 observability.** Per-rank, per-expert, per-stage metrics are collected here; the SLO methodology that consumes them is 110.
- **Module 113 rollouts.** Topology changes are rollouts; the canary and rollback framework is 113; the change surface is this module.

## The sign-off a well-run fleet publishes

Every topology-changing deployment ships with a one-page sign-off document that answers:

```
Fleet: <name>
Topology: TP=<n> PP=<n> EP=<n> DP=<n> replicas
Model + revision: <HF ref @ SHA>
Quantisation: <bf16 / fp8 / int4>
Disaggregation: <yes / no + pool shapes>
Hardware: <N × GPU_model, interconnect>

Placement verified (chapter 06):
    nvidia-smi topo -m: <attach>
    NCCL_DEBUG=INFO startup log: <attach>
    nccl-tests all_reduce_perf, alltoall_perf, sendrecv_perf: <attach>
    NUMA + NIC-to-GPU affinity: <per-rank table>

SLO envelope published:
    TTFT target / achieved: <ms> / <ms>
    TPOT target / achieved: <ms> / <ms>
    Goodput at target SLO: <QPS>
    Measurement window: <date range>

Peer-boundary considerations:
    Known kernel-level items with open issues: <list, each with ticket to performance peer>
    Multi-cluster considerations deferred to architect: <list>

Rollback trigger:
    Metric: <specific metric + threshold>
    Procedure: <chapter 113-shaped rollback>
```

This document is signed by the on-call operator for the fleet and the architect for the cluster. It's reviewed quarterly.

## Summary

- This module operates distributed inference topologies at the level of a single serving fleet. The two peers that own the territory on either side are the performance peer (level 35, kernel craft) and the architect (level 40, multi-cluster).
- The three-way table (above) names ownership per concern. The hand-off to performance includes the full topology + placement verification + measurement bundle; the hand-off to architect includes the fleet's goodput envelope + the workload profile + the capacity ask.
- Three common cross-boundary mistakes: opening a kernel ticket when the diagnosis was placement; reshaping topology to fix a multi-cluster latency; the architect micromanaging fleet-local topology.
- Multi-cluster scope (routing, replication, data boundaries, cross-region observability) is explicitly outside this module. Kernel scope (Flash-Attention, NCCL internals, DeepEP, CUTLASS epilogues) is explicitly outside this module.
- Every topology-changing deployment ships a one-page sign-off document that names the topology, the placement verification, the SLO envelope, the open peer-boundary items, and the rollback trigger.
