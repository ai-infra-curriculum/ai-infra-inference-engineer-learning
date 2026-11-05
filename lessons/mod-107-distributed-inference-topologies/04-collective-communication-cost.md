# Chapter 04 — Collective-Communication Cost: NCCL at Operator Depth

> Chapter 01 picked the topology. Chapter 02 named the collectives it issues per layer. This chapter costs them out — bandwidth, latency, where NVLink stops and InfiniBand starts, and the overlap-with-compute discipline that keeps the collective off the critical path. All without opening a `.cu` — that is the performance peer's work.

## Motivation

Three questions reach the operator desk weekly:

1. **"Why does TP=8 give us 2.4× throughput over TP=1 instead of 7×?"** Because every layer has an AllReduce; AllReduce has a bus-bandwidth floor; at some TP degree the collective time dominates the computation time for memory-bound decode. Named correctly, this is "the collective cost crossed the compute cost" — not a kernel bug, not an engine bug, a topology choice.
2. **"If we add a second node, is TP=16 the right move?"** No — because the AllReduce would go over IB instead of NVLink, and the latency jumps by 10-30×. The right move is TP=8 within a node, DP across nodes (or PP + TP, or disaggregation).
3. **"The engine's reported TPOT regressed 15 % after the fleet migration from H100-SXM to H100-PCIe. Why?"** Because the PCIe variant has reduced NVLink bandwidth (and sometimes no NVLink switch); the AllReduce bus-bandwidth per rank drops; TPOT shifts. The GPU is "the same" (same compute, same HBM) but the topology-level characteristic is different.

All three answers require operator-depth NCCL understanding — not kernel-depth. This chapter writes the operator-depth model.

By the end of it you can:

- Enumerate the three collectives TP and MoE issue (AllReduce, AllGather, ReduceScatter — plus all-to-all for EP) and name their communication complexity.
- Reason about NVLink bus-bandwidth vs. InfiniBand / RoCE latency floor.
- Use `nccl-tests` to measure the fleet's actual collective-cost curve and compare to spec.
- Reason about the overlap-with-compute discipline — when the collective lives inside a compute shadow, and when it doesn't.
- Know what you can tune at operator depth (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_P2P_LEVEL`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, env sysconfig) and where the kernel-depth tuning lives (performance peer).

NCCL version matters — algorithm selection and in-network-reduction integration have evolved across releases, so the algorithm choices discussed below are directional rather than exact; verify against the NCCL release notes for your pinned version.

## The three TP collectives and their complexity

For an N-rank group communicating a message of size M bytes per rank, the standard asymptotic bandwidth costs:

| Collective | Data per rank per step | Total data moved | Bus-bandwidth floor (ring algorithm) |
|---|---|---|---|
| AllReduce (ring) | M | 2M × (N-1)/N | `2M × (N-1) / (N × BW)` |
| AllGather | M | M × (N-1)/N | `M × (N-1) / (N × BW)` |
| ReduceScatter | M | M × (N-1)/N | `M × (N-1) / (N × BW)` |
| all-to-all | M × (N-1)/N | M × (N-1)/N (per rank) | `M × (N-1) / (N × BW)` |

(Here `M` is the per-rank payload, `BW` is the achievable bus bandwidth per rank, and the formula ignores the latency component. The important intuition: AllReduce is 2× the one-way cost; AllGather and ReduceScatter are each the one-way cost; and all run at bus-bandwidth / rank — not at the aggregated bandwidth across the group.)

### The two-phase decomposition

AllReduce is often implemented as **ReduceScatter + AllGather**. ReduceScatter sums contributions across ranks into disjoint output chunks; AllGather distributes each rank's chunk to everyone. This is why sequence-parallel can replace an AllReduce with an AllGather + ReduceScatter pair without changing the bytes-per-rank — the same two phases are run, just at a different boundary that saves activation memory.

### The algorithms

NCCL implements two main algorithms:

- **Ring.** Each rank sends/receives to its two neighbours in a ring. Bus-bandwidth is near-optimal at large message sizes.
- **Tree.** Each rank sends/receives to a hierarchical tree. Lower latency at small message sizes (fewer hops to reach every rank).

NCCL's auto-tuner picks ring for large messages and tree for small ones. The crossover is message-size-dependent; `NCCL_ALGO` overrides. **The operator rarely sets `NCCL_ALGO`** — defaults are usually right; if the auto-tuner is making a bad choice at your workload's message-size curve, that's a performance-peer conversation.

The CollNet / SHARP in-network reductions are a third algorithm family — enabled on Mellanox fabrics with SHARP-capable switches. They offload the reduction to the switch, which can improve large-message AllReduce at scale; the operator sets `NCCL_COLLNET_ENABLE=1`, the performance peer tunes. Exact enablement requirements and the realised speed-up have evolved — check the current NCCL and NVIDIA SHARP docs for your fabric before promising a number.

## NVLink vs. InfiniBand — the two regimes

**NVLink (intranode).** NVLink is a point-to-point interconnect with switch (NVSwitch on H100 SXM) that provides a non-blocking any-to-any fabric within a node. Per-GPU bandwidth to the switch:

- NVLink 4 on H100 SXM: 450 GB/s per GPU (bidirectional, aggregated across 18 links).
- NVLink on H100 PCIe (non-SXM): 600 GB/s *total* on the NVLink bridges (two cards per bridge, not any-to-any); if there is no bridge, PCIe-only — much slower.
- NVLink on H200 SXM: 900 GB/s per GPU (NVLink 5 generation on some SKUs; verify).
- NVLink on B100 / B200: higher still.

Latency on NVLink: single-digit microseconds for small messages; sub-millisecond for sizes up to a few MiB.

**InfiniBand / RoCE (internode).** IB is a lossless fabric with RDMA. Per-NIC bandwidth:

- ConnectX-6 (HDR): 200 Gbps (25 GB/s).
- ConnectX-7 (NDR): 400 Gbps (50 GB/s).
- ConnectX-8 (XDR): 800 Gbps (100 GB/s).

RoCE is RDMA-over-Ethernet, same NIC bandwidth, slightly higher latency and more fiddly to configure (requires DCB, PFC for lossless).

Latency on IB: 2-10 μs for small messages; latency per message is dominated by the RDMA write/send setup for small sizes. At large sizes, bandwidth dominates.

**The gap.** NVLink 4 is roughly 10-20× the per-GPU bandwidth of a single ConnectX-7 NIC. If a node has 8 GPUs and 8 NICs (one NIC per GPU, each on the same NUMA / PCIe root complex as its GPU), the aggregate internode bandwidth matches intranode — but *per-pair*, the gap is still ~10×.

### Why TP wants NVLink

TP's AllReduce bus-bandwidth cost per layer scales as `2M(N-1)/N × 1/BW`. At BW = NVLink (450 GB/s), the per-layer cost for a 70B TP=8 forward pass is on the order of hundreds of microseconds. At BW = ConnectX-7 single-NIC (50 GB/s), the per-layer cost rises by ~9× — into the millisecond range per layer. With 80 layers, that's an extra 80 ms per iteration — SLO-destroying.

Multiple NICs per node help, but TP's collectives are per-GPU-pair in the ring, not per-NIC — the fabric-level aggregated bandwidth is not what matters.

### Why PP tolerates IB

PP's send/recv is one-to-one, not an N-wide collective. The payload is one activation per micro-batch per stage boundary. At `(batch × seq × hidden × bytes) ≈ hundreds of MiB` per hand-off, over 50 GB/s IB, that's millisecond-range per hand-off — tolerable at the micro-batch level.

### Why EP all-to-all is sensitive to IB

An EP all-to-all is a true collective: every rank sends to every other rank. For EP=32 across 4 nodes of 8 GPUs, each rank sends `(num_tokens_routed_to_each_rank × hidden × bytes)` to each other rank. The per-rank traffic sums over destinations. For a 1000-token prefill batch with 32 experts top-2 and balanced routing, each rank handles ~63 tokens; the all-to-all moves 63 × hidden × bytes per destination = ~1 MiB per pair. Across 32 ranks, each rank has 31 destinations = ~31 MiB per rank per all-to-all. Two all-to-alls per MoE layer × 61 MoE layers (DeepSeek V3): ~3.8 GiB per rank per forward pass.

At 50 GB/s IB that's ~77 ms per iteration *just* for all-to-all — chapter 05 names mitigations. The sensitivity: EP across IB is tight; EP within one node (NVLink) is comfortable; the design envelope narrows at multi-node.

## Measuring the fleet — `nccl-tests`

The `nccl-tests` repository (NVIDIA / nccl-tests on GitHub) provides canonical micro-benchmarks:

- `all_reduce_perf -b 1K -e 1G -f 2 -g <GPUs-per-node>` — AllReduce sweep from 1 KB to 1 GB.
- `all_gather_perf`, `reduce_scatter_perf` — the AllGather / ReduceScatter sweeps.
- `alltoall_perf` — the all-to-all sweep.
- `sendrecv_perf` — the point-to-point sweep (for PP hand-offs, KV transfer).

Each prints `size`, `time`, `algo-bw`, and `bus-bw`. The **bus-bw** is what you compare to the hardware spec; `algo-bw` is a scaled view that accounts for the algorithm's data-movement multiplier.

### A real measurement set to collect

For every TP / PP / EP topology you intend to deploy, record:

- AllReduce bus-bw at the message sizes a real forward pass uses (per-layer activation bytes). For a 70B TP=8 forward pass with batch 1 at seq 2048 BF16, the per-layer AllReduce message is `1 × 2048 × 8192 × 2 bytes = 32 MiB`. Record bus-bw at 32 MiB.
- AllReduce bus-bw at decode-time message sizes (batch_concurrency × 1 × hidden × bytes) — much smaller, so latency-bound regime.
- AllGather + ReduceScatter at the same sizes (if sequence-parallel is in play).
- Point-to-point sendrecv_perf between the specific nodes that will host the PP boundary.
- All-to-all at the token-count × hidden-dim sizes an MoE layer issues.

Store the measurements alongside your topology plan. When later a scaling cliff shows up, having a baseline lets you tell "the fabric changed" from "the engine changed."

### Expected numbers (rough)

For an 8× H100 SXM node with NVSwitch, NCCL AllReduce at 1 MiB:

- Bus-bw per rank: ~250-350 GB/s (fraction of the 450 GB/s per-GPU-to-switch bandwidth, limited by ring algorithm overhead).

For AllReduce at 1 GiB on the same node: bus-bw at the higher end, 350-430 GB/s per rank. Within 70-90 % of theoretical.

For AllReduce across two nodes over 8 × ConnectX-7 NICs (one per GPU) at 1 MiB: the per-rank bus-bw is much lower — the collective is bottlenecked by the slowest link in the ring, which is the IB hop. Measured bus-bw is typically ~30-45 GB/s per rank (close to single-NIC bandwidth).

**If your measurements are more than 2× below these ballparks, your fleet has a placement or driver problem**, not an engine problem. Chapter 06 names the diagnostic flow. The specific `nccl-tests` numbers for H100 SXM / H200 SXM / B200 and for CX-6/CX-7/CX-8 fabrics rotate across firmware and driver generations — treat the ranges above as directional; your own measurements are authoritative.

## The overlap-with-compute discipline

A collective that runs on a dedicated communication stream can overlap with compute on the main stream — the AllReduce for layer N's attention output runs while layer N's MLP compute starts, hiding the collective cost. Done perfectly, the collective is "free" (contributes nothing to iteration time).

**Why it isn't always perfect:**

- At decode time, the per-layer compute is small (one token's worth of GEMMs); the collective is dominant; there's no shadow to hide in.
- At prefill time with small batches, same story.
- When the collective is on a shared stream with compute (older engine builds), there's no actual overlap — the collective blocks the stream.
- At large TP degree, the collective grows but the compute doesn't — the shadow runs out.

The operator-level signal: measure the per-layer time with and without the collective (NVTX / Nsight Systems trace at the engine). If the collective is overlapped, the per-iteration time is only slightly larger than the compute time; if not, it's compute-time + collective-time additively. See chapter 06 for the diagnostic.

Engine-level implementations vary:

- **vLLM** overlaps the AllReduce with compute via dedicated NCCL streams and CUDA graphs; verify in release notes.
- **SGLang** also overlaps via dedicated streams; the chunked prefill scheduler composes with overlap.
- **TRT-LLM** generates fused kernels that interleave communication and compute at the TensorRT-plan level.

The exact overlap efficiency is kernel-level; the operator diagnoses "is overlap happening" via Nsight Systems, and routes "overlap efficiency low" to the performance peer. Engine-level overlap behaviour has evolved across releases (and often varies by batch regime and build options) — verify against your pinned release's notes rather than assuming the behaviour above.

## Operator-level NCCL knobs

The useful ones (defaults are usually right; most of these are health-check inputs):

- `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` — startup logging. The first few hundred lines show topology discovery, algorithm selection, network path.
- `NCCL_P2P_LEVEL=NVL` — require NVLink for intranode P2P; fail if it would fall back.
- `NCCL_IB_HCA=mlx5_0,mlx5_1,...` — restrict to specific HCAs. Useful for NIC-to-GPU pairing (chapter 06).
- `NCCL_SOCKET_IFNAME=eth0,ib0` — pick the right bootstrap interface.
- `NCCL_COLLNET_ENABLE=1` — enable in-network reductions on SHARP-capable fabrics (requires a supported switch + driver).
- `NCCL_ALGO=Ring` or `Tree` — override the auto-selected algorithm. Rare; usually leave unset.
- `NCCL_PROTO=LL`, `LL128`, `Simple` — override the protocol. Rare; usually leave unset.
- `NCCL_BUFFSIZE=...` — NCCL buffer size per channel. Default is 8 MiB; rarely tune.
- `NCCL_NET_PLUGIN` — load a custom network plugin (e.g. AWS NCCL plugin for EFA). Required on specific clouds.

### `NCCL_IB_GID_INDEX` and RoCE

RoCE requires setting the GID index that NCCL uses. Common on cloud fabrics; `NCCL_IB_GID_INDEX=3` is a common value for RoCEv2. Verify against your cloud's docs.

### `NCCL_IB_TIMEOUT`, `NCCL_IB_RETRY_CNT`

Timeouts for IB. On a flaky fabric, raising these masks the fabric issue — don't; fix the fabric.

## When to route to the performance peer

- Your `nccl-tests` numbers are good; the engine's collective time (from Nsight) is poor. The engine isn't using the collective efficiently — kernel-level issue.
- Overlap efficiency is low (collective time appears additively in iteration time): kernel / graph-fusion issue.
- A specific message-size regime is slow (e.g. 10-100 KB messages); the auto-tuner picked the wrong algorithm: algorithm tuning.
- The collective kernel on an unusual hardware configuration (AMD GPUs with RCCL, custom hardware) is miscompiling: library issue.
- The engine's custom all-to-all kernel for MoE dispatch isn't matching NCCL's expected performance: kernel-level.

What to ship to the performance peer:

- The `nccl-tests` output for the exact topology.
- Nsight Systems trace showing the collective's start/end timestamps and the surrounding compute.
- The engine's version and the exact launch flags.
- GPU model + driver + CUDA + NCCL versions.
- The per-layer collective time distribution over an iteration.

Chapter 07 formalises the hand-off bundle.

## When to route to the networking team

- `nccl-tests` numbers are 2-5× below spec, and the fix isn't an engine config.
- `NCCL_DEBUG` reports IB errors (`WARN NET/IB`).
- `ibv_devinfo` shows HCAs in `Down` state, or ports with CRC errors.
- One node's internode bandwidth is much lower than other nodes' — fabric asymmetry.

What to ship: node names, HCA / port identifiers, `ibv_devinfo` output, `nccl-tests` run across the problematic pair.

## Composition with other chapters

- **Chapter 01 (topology selection).** The collective cost per iteration is a direct input to the SLO-feasibility check in Step 4 of chapter 01's decision tree.
- **Chapter 02 (TP / PP mechanics).** The health checks in chapter 02 (nvidia-smi topo, NCCL_DEBUG, nccl-tests) use this chapter's language.
- **Chapter 03 (disaggregation).** The KV transfer bandwidth is a point-to-point case; `sendrecv_perf` from `nccl-tests` measures it.
- **Chapter 05 (MoE).** The all-to-all cost dominates MoE at multi-node; this chapter's cost model directly feeds chapter 05's design envelope.
- **Chapter 06 (placement).** The common scaling cliffs are placement / affinity errors that show up as `nccl-tests` numbers below spec.

## Summary

- TP issues two AllReduces per layer (vanilla); sequence-parallel decomposes them into AllGather + ReduceScatter. EP issues two all-to-alls per MoE layer. PP uses point-to-point sendrecv. Bus-bandwidth per rank is the cost model; message-size curves pick ring vs. tree algorithm.
- NVLink intranode (hundreds of GB/s per GPU) vs. InfiniBand / RoCE internode (tens of GB/s per NIC) — the ~10× gap is why TP stays inside a node.
- Use `nccl-tests` (`all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, `alltoall_perf`, `sendrecv_perf`) to measure your fleet; store the baseline alongside the topology plan.
- Overlap-with-compute hides collective cost at prefill with large batches; it does not hide decode-time collective cost. Nsight Systems is the diagnostic.
- Operator-level NCCL knobs are environment variables for topology / network binding; algorithm tuning is deferred to the performance peer; fabric issues are deferred to the networking team.
- Route to performance peer when `nccl-tests` is healthy but engine traces show poor collective efficiency; route to networking when `nccl-tests` itself is off-spec.
