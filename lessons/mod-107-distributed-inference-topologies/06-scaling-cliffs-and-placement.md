# Chapter 06 — Scaling Cliffs: Straggler Ranks, PCIe-vs-NVLink Mismatch, Placement

> You picked the topology (chapter 01), deployed it (chapter 02 / 03 / 05), and measured the collectives (chapter 04). Now the SLO slips at scale. The culprit is almost never a kernel bug — it's a placement choice three steps upstream. This chapter writes the diagnostic flow and the placement fixes.

## Motivation

Three things reliably break at scale that did not break at small-scale testing:

1. **One slow rank sets the critical path for the whole TP / EP group.** A ring AllReduce is as fast as the slowest hop; one GPU that runs 10 % slower drops the whole group's throughput by 10 %. The slow rank's cause is usually unglamorous — ECC-recovered DRAM errors, thermal throttling, a bus error on one PCIe lane, a dormant background process.
2. **The intranode interconnect isn't what you thought it was.** On dual-socket boards with asymmetric wiring, two GPUs on the same node can be connected only via PCIe host-memory copy, not NVLink. Collectives through that pair are an order of magnitude slower. `nvidia-smi topo -m` is the diagnostic; most teams don't check it until something breaks.
3. **NIC-to-GPU affinity is wrong for internode traffic.** Node has 8 GPUs and 8 NICs; the NIC for GPU 0 is physically on the socket connected to GPU 4. Cross-socket traffic over QPI / UPI dominates the KV transfer or the EP all-to-all. GPUDirect RDMA on paper, PCIe-over-the-wrong-socket in practice.

None of these are kernel problems. All of them look like kernel problems from above the fleet. This chapter writes the diagnostic flow that keeps the escalation ownership honest.

By the end of it you can:

- Enumerate the four most common scaling cliffs and name the symptom each produces.
- Read `nvidia-smi topo -m`, `lstopo`, `numactl --hardware`, and `ibv_devinfo` well enough to spot a placement mistake.
- Produce a per-rank timing trace from Nsight Systems (or engine-native per-rank timing) and find the straggler.
- Apply the placement fix — CPU / GPU affinity, NUMA binding, NIC-to-GPU pairing, rack-local placement.
- Write the handoff bundle when the diagnosis exceeds operator depth (hardware repair, fabric team, performance peer).

Specific tool versions (nvidia-smi, lstopo, nccl-tests) evolve, and output formats and flags drift with them; treat the ballpark numbers and command-line details below as directional and verify tool output against your fleet's exact hardware.

## The four canonical scaling cliffs

### Cliff 1 — straggler rank

**Symptom.** TP or EP group's iteration time is bimodal — most iterations run at speed, some run 10-20 % slower. Over a sustained window, aggregated throughput drops. One rank's per-step time (visible in Nsight or in engine-native per-rank metrics) is consistently longer than the others'.

**Causes.**

- **ECC-recovered memory errors.** HBM ECC rate above healthy baseline causes error-correction stalls. `nvidia-smi -q -d ECC` reports per-GPU ECC counts; a specific GPU with higher counts than peers is suspect.
- **Thermal throttling.** GPU temperature is high enough that the clock throttles. `nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks_throttle_reasons.active --format=csv` reports clocks and throttle reasons. A GPU persistently throttled is slow.
- **Partial failure.** One SM (streaming multiprocessor) disabled, one HBM stack degraded, one of the NVLink links down. Observable as a GPU reporting "healthy" with slightly reduced max clock or reduced memory bandwidth.
- **Background process.** A monitoring daemon, a stale container, a hung CUDA context from a prior process holds a slice of HBM or SMs. `nvidia-smi` shows an unexpected process.
- **Firmware mismatch.** One GPU has an older VBIOS than peers; subtle behaviour differences accumulate.

**Diagnostic flow.**

1. Collect per-rank iteration time. The engine should expose this; if not, run a short NCCL AllReduce benchmark across the exact ranks (`nccl-tests all_reduce_perf -g <N>`).
2. If one rank is consistently slow:
   - `nvidia-smi -q -d ECC` — check ECC counts.
   - `nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks_throttle_reasons.active --format=csv` — check throttling.
   - `nvidia-smi` — check for unexpected processes.
   - `nvidia-smi -q -d NVLINK` — check NVLink status; look for errors or disabled links.
3. If the slow rank changes over time (not always GPU N), the problem is not hardware; look for scheduler-level or workload-level skew (EP imbalance from chapter 05; chunked-prefill skew from module 103).

**Fix.** 

- Replace the slow GPU (if it's hardware). This is operations, not engine.
- Rebalance the EP routing (if it's expert hotspotting; see chapter 05).
- Rebalance the batching (if it's workload skew; see module 103).

### Cliff 2 — PCIe fallback instead of NVLink

**Symptom.** TP throughput is a fraction of expected. `nccl-tests all_reduce_perf` on the fleet reports bus-bw 5-10× below the NVLink spec. `NCCL_DEBUG=INFO` log shows `via SYS` for some GPU pairs (not `via NVL`).

**Causes.**

- **GPUs on different PCIe root complexes without an NVLink bridge.** Common on dual-socket boards (one socket per PCIe root) where GPUs on socket 0 and socket 1 cannot NVLink-direct. The only path is PCIe → QPI/UPI → PCIe → host memory copy.
- **Mixed GPU generations in the same TP group.** Older-gen GPUs lack NVLink to newer-gen GPUs even on the same board.
- **NVLink bridge missing or cable unseated.** On H100-PCIe boards, NVLink requires an installed bridge card. Easy to miss during install.
- **NVSwitch disabled or in error state.** On H100 SXM / HGX boards, NVSwitch coordinates the NVLink fabric; a failed NVSwitch chip isolates some GPU pairs.

**Diagnostic flow.**

1. Run `nvidia-smi topo -m` on each node. Verify every GPU pair the engine will shard across has `NV#` (not `PIX`, not `PHB`, not `SYS`).
2. For an H100 SXM / HGX node, verify `nvidia-smi nvlink --status` reports all links active.
3. Run `nccl-tests all_reduce_perf -b 1M -e 1G -g 8` on the problem node; compare to published vendor numbers for the same hardware. Deficit of > 2× is placement; deficit of < 20 % is tune-vs-firmware noise.

**Fix.**

- Install the NVLink bridge (for PCIe boards).
- Repair / replace the NVSwitch (for HGX issues) — fleet-maintenance task.
- Change the TP topology to avoid the bad GPU pair (if the pair is unfixable short-term): skip that GPU in the TP group assignment.
- `NCCL_P2P_LEVEL=NVL` forces NCCL to use NVLink only; the communicator fails to build if NVLink isn't available between any pair, which is a loud failure rather than silent degradation.

### Cliff 3 — Cross-NUMA / cross-socket noise

**Symptom.** TP throughput is fine for the "obviously bad" measurements (`nccl-tests` bus-bw matches spec) but the engine's end-to-end TPOT regresses in production. Nsight Systems trace shows CPU-side host-to-device memcpies taking longer than expected; some ranks' kernel launches have higher launch latency; the engine's tokenizer / scheduler on the driver rank shows CPU-side jitter.

**Causes.**

- **GPU 0 is on socket 0; its worker process was placed on socket 1 by the scheduler.** Every memcpy, kernel launch, and host-side scheduler operation crosses QPI/UPI.
- **NUMA memory allocation defaults to interleave.** Host-side buffers are scattered across both sockets; some accesses are cross-socket.
- **CPU-side kernel-launching thread is pinned to the wrong socket.** Rare, but observable.

**Diagnostic flow.**

1. `lstopo` or `lstopo-no-graphics --of txt` on each node. Produces a tree of NUMA domains, PCIe root complexes, GPUs, and NICs. Verify each GPU's PCIe root is annotated with its NUMA domain.
2. `numactl --hardware` — list the NUMA nodes.
3. Check the engine's process placement: `ps -o pid,comm,psr -p <engine-pid>` reports which CPU each thread is on; `taskset -pc <pid>` reports the affinity mask. Each rank's process should be pinned to the socket its GPU is on.
4. Nsight Systems trace shows the host-side kernel launch latency per rank; a rank with 2-3× the launch latency of peers is likely cross-NUMA.

**Fix.**

- Pin each rank's process to its GPU's socket: `numactl --cpunodebind=<N> --membind=<N>`.
- The engine's launcher usually supports this (`VLLM_CPU_OMP_THREADS_BIND`, Ray resource specs with NUMA affinity, or an external launcher like `numactl` wrapping the worker command).
- Use `GPU_BINDING` or equivalent engine flag to pin GPU-to-process at launch.

### Cliff 4 — Cross-rack NIC bottleneck / fabric asymmetry

**Symptom.** `nccl-tests sendrecv_perf` or `alltoall_perf` within a rack shows expected bandwidth; across racks, bandwidth drops by 2-5×. The engine's KV transfer or EP all-to-all performance regresses when a request's two nodes end up on different racks.

**Causes.**

- **Fabric oversubscription.** The ToR (top-of-rack) switch to spine link has lower aggregated bandwidth than the sum of in-rack bandwidth. Normal; the datacentre design assumes most traffic stays in-rack.
- **Fabric asymmetry.** Different racks on different generations of switch, different PFC configurations, different adaptive-routing policies.
- **Multi-tenant traffic.** Another workload saturates the spine; your inference traffic contends.

**Diagnostic flow.**

1. `nccl-tests sendrecv_perf` between every pair of nodes in your fleet; produce a bandwidth matrix. In-rack cells should be at NIC spec; cross-rack cells may be lower (quantify by how much).
2. If cross-rack cells are the culprit, verify with the fabric team that the spine is healthy and your traffic has priority.
3. Check `ibv_devinfo` for port state (`Active`), negotiated speed, and link width on every HCA.

**Fix.**

- **Rack-local placement.** Request that your orchestrator schedule TP / EP groups within one rack. Kubernetes topology-aware scheduling, Slurm topology constraints, Ray node selectors all support this.
- **Fewer internode traffic patterns.** Reduce EP degree if cross-rack bandwidth is the floor; use smaller models; use disaggregation with rack-local pools.
- **Fabric upgrade.** Not an operator task but worth escalating.

## The diagnostic flowchart

Given a scaling-cliff symptom, walk this flow:

```
Topology deployed, SLO slipped?
│
├── Is one rank consistently slower than peers?
│   ├── Yes → Cliff 1 (straggler). nvidia-smi ECC/thermal/throttle.
│   └── No → Continue.
│
├── Did nccl-tests bus-bw match the hardware spec?
│   ├── No, 2-10× below → Cliff 2 (PCIe fallback). nvidia-smi topo -m.
│   └── Yes → Continue.
│
├── Is the regression at the engine level (not NCCL)?
│   ├── Nsight shows cross-NUMA memcpies or high launch latency?
│   │   ├── Yes → Cliff 3 (NUMA). lstopo + numactl.
│   │   └── No → Continue.
│   └── nccl-tests across-node asymmetric?
│       ├── Yes → Cliff 4 (fabric). sendrecv_perf matrix; rack-local placement.
│       └── No → Not covered here. Could be:
│           - Engine config regression (release-notes / A-B test).
│           - Workload-level skew (EP imbalance; chapter 05).
│           - Model-specific issue (route to performance peer).
```

## Tool reference

### `nvidia-smi topo -m`

Prints the GPU-to-GPU topology matrix on a node. Entries:

- `NV#` — NVLink with `#` links.
- `PIX` — Via same PCIe switch.
- `PHB` — Via PCIe host bridge (CPU).
- `PXB` — Via multiple PCIe switches.
- `SYS` — Via QPI/UPI (across sockets); worst.

Also reports GPU-to-NIC proximity when NICs are present.

### `lstopo` (from the `hwloc` package)

Draws the full hardware topology including NUMA nodes, PCIe switches, GPUs, NICs. Essential for internode placement reasoning. `lstopo-no-graphics --of txt` produces an ASCII tree.

### `numactl`

- `numactl --hardware` — list NUMA nodes.
- `numactl --cpunodebind=0 --membind=0 <command>` — pin a process to a NUMA node.
- `numastat -p <pid>` — show per-NUMA memory allocation for a running process.

### `ibv_devinfo`

Lists IB HCAs on a node with port state, speed, and GID count. Essential before running an IB-based fleet.

### `nvidia-smi -q -d ECC`, `-d PERFORMANCE`, `-d NVLINK`

Deep queries for ECC counts, performance throttling reasons, and NVLink link status. Automate scraping into a GPU-health dashboard.

### `dcgmi diag -r 3` (DCGM)

Runs a battery of GPU health checks. Takes minutes; useful before admitting a node into the serving fleet.

### Nsight Systems

NVIDIA's system-level profiler. Produces per-rank, per-stream timelines showing compute kernels, collectives, and host-side events. For scaling-cliff diagnosis, run with a trace that includes CUDA, NVTX, and NCCL on all ranks simultaneously; export; compare per-rank timelines. Nsight Systems command-line flags evolve across releases — verify the exact `--trace=` spelling and available backends against your installed `nsys --help`.

### Engine-native per-rank metrics

Most inference engines expose per-rank timing in their metrics endpoint. vLLM reports per-worker iteration time; SGLang reports similar; TRT-LLM via Triton metrics. Scrape and visualise.

## Placement — the production-time pattern

Four placement decisions a well-run inference fleet makes:

### 1. CPU-GPU-NUMA affinity

Each GPU's worker process pinned to the CPU socket its GPU is on. For 8× H100 on a dual-socket Intel / AMD board: GPUs 0-3 pinned to socket 0, GPUs 4-7 to socket 1 (verify against the specific board's topology). Memory binding to the local socket.

Engine support:

- **vLLM.** Supports `VLLM_HOST_IP`, `VLLM_PORT`; CPU affinity via launcher wrapping (`numactl` + process-wide pinning). Some vLLM worker spawning respects `taskset`.
- **SGLang.** Launcher accepts affinity hints; often done via `numactl` wrapping.
- **TRT-LLM.** MPI launcher respects `mpirun --bind-to numa` / `--map-by ppr:1:node`.

### 2. NIC-to-GPU affinity

Each GPU paired with the NIC on the same PCIe root complex / NUMA domain. For GPUDirect RDMA to work at full bandwidth, the NIC and the GPU must be on the same root. Set `NCCL_IB_HCA` to restrict NCCL to the right HCA for each rank.

Example: for an 8× H100 node with 8 NICs:
```
Rank 0 → GPU 0, NIC mlx5_0 (both on socket 0, root 0)
Rank 1 → GPU 1, NIC mlx5_1 (both on socket 0, root 0)
...
Rank 4 → GPU 4, NIC mlx5_4 (both on socket 1, root 1)
...
```
Then per-rank:
```
NCCL_IB_HCA=mlx5_<rank>
```

### 3. Rack-local placement for internode traffic

Request the orchestrator to schedule TP / EP groups within one rack. Kubernetes:

```yaml
# PodAntiAffinity or TopologySpreadConstraint on rack-topology key
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: inference-serving
          topologyKey: topology.kubernetes.io/rack
```

Verify with your orchestrator's docs; the topology key varies.

### 4. GPU-specific worker pinning

Each worker process bound to a specific GPU ordinal via `CUDA_VISIBLE_DEVICES`. Prevents the engine from accidentally seeing an un-wanted GPU. Standard practice; often done by the engine's launcher.

## Composition with other chapters

- **Chapter 02 (TP / PP mechanics).** The chapter-02 health checks live here as part of the deployment pipeline, not as one-off diagnostics.
- **Chapter 03 (disaggregation).** NIC-to-GPU affinity for KV transfer is critical; without it, GPUDirect RDMA degrades.
- **Chapter 04 (collectives).** The expected `nccl-tests` numbers come from chapter 04; the "below spec" diagnostic in this chapter uses them.
- **Chapter 05 (MoE).** Expert hotspotting in chapter 05 may look like Cliff 1 (straggler rank); the diagnosis distinguishes.

## When to escalate

### Hardware repair ticket

- One GPU persistently slow with ECC errors above baseline, or NVLink links down, or thermal throttling with normal workload.
- NIC in `Down` state or with CRC errors.

### Networking / fabric team

- `nccl-tests sendrecv_perf` across certain node pairs significantly below the single-pair baseline, after confirming NIC-to-GPU affinity is correct.
- IB subnet-manager issues (not operator-level).
- SHARP tree health degraded.
- Chapter 04 covered the IB-vs-TCP fallback; sometimes the fix is "the IB fabric itself is broken on this rack."

### Performance peer

- `nccl-tests` is healthy; the engine still spends > expected time in collectives; Nsight trace shows the collective kernel is slow for the message size.
- The engine falls back to a generic kernel instead of the Hopper-specific DeepEP (chapter 05) and the kernel path isn't exposed as an operator flag.
- Overlap-with-compute is not working on your workload shape; the collective time appears additively instead of hidden.

### Software vendor / upstream

- A specific NCCL version has known issues with your hardware (release notes); upgrade.
- Your pinned engine version has known issues with your topology (release notes); upgrade or patch.

## Common operator-level mistakes

### Mistake 1 — missing a GPU-health gate at scheduler admission

The orchestrator schedules a job on a node whose GPU has latent issues. The job runs; one rank straggles; SLO slips. Fix: node admission runs DCGM diagnostics / NCCL micro-benchmark before accepting the node into the fleet.

### Mistake 2 — reusing stale environment between workloads

The previous workload set `NCCL_IB_HCA=mlx5_0` for all ranks (a bug). The next workload inherits; all ranks share one HCA; cross-node bandwidth is 1/8 of expected. Fix: launcher sets NCCL env freshly per-worker-per-launch; don't rely on inherited environment.

### Mistake 3 — ignoring per-rank metrics in production dashboards

The platform dashboard shows aggregated TPOT. A straggler rank is masked. Fix: per-rank panels on every dashboard; alerts on per-rank-stddev > threshold.

### Mistake 4 — "it works on test, deploy to production" without placement validation

Test fleet has 2 GPUs on one node; production fleet has 16 GPUs across 2 nodes with internode IB. The placement-sensitive paths (EP all-to-all, cross-node KV transfer) weren't exercised in test. Fix: production-shape pre-production fleet; run `nccl-tests` matrix as part of the pre-flight.

### Mistake 5 — assuming the engine handles placement

Some engines do; some don't. Verify empirically; don't assume.

### Mistake 6 — raising timeouts to mask fabric issues

`NCCL_IB_TIMEOUT=25` masks a flaky fabric; the engine stops failing but the fabric is still broken. Fix: don't; escalate to networking.

## Summary

- Four canonical scaling cliffs: straggler rank (hardware-level), PCIe fallback instead of NVLink (topology mismatch), cross-NUMA noise (process pinning), cross-rack fabric asymmetry (orchestrator placement).
- Diagnostic tools: `nvidia-smi topo -m`, `lstopo`, `numactl`, `ibv_devinfo`, `nvidia-smi -q -d ECC/PERFORMANCE/NVLINK`, DCGM, Nsight Systems, engine-native per-rank metrics.
- Four production-time placement decisions: CPU-GPU-NUMA affinity, NIC-to-GPU affinity, rack-local placement, GPU-specific worker pinning.
- Escalate hardware to fleet-maintenance; fabric to networking; kernel-level collective-efficiency to the performance peer.
- Common operator mistakes: missing a health gate, stale environment, no per-rank metrics, placement-free test fleets, assuming the engine does placement, raising timeouts to mask fabric.
