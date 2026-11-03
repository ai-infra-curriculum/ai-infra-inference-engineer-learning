# Exercise 02 — TP and PP Deployment Walkthrough on vLLM

**Time:** 4 hours · **Output:** one TP deployment, one PP deployment (real or simulated), one sharding-vs-replication health-check script, one per-collective cost measurement table, one deployment runbook

## Goal

Make [Chapter 02](../02-tp-and-pp-deployment-mechanics.md) concrete by **standing up a TP deployment on a single node**, **standing up a PP deployment across two nodes** (or simulating it if two nodes aren't available), **measuring the per-collective cost that the forward pass incurs**, and **publishing a health-check script that confirms the engine is actually sharding (not silently replicating)**. By the end you have the launch commands, the health checks, and the measurement harness as reusable artefacts for every TP / PP deployment on your fleet.

## Problem statement

Your team is bringing up a Llama-3.1-70B deployment on a shared research cluster. The deployment plan calls for TP=4 within one node for the primary serving fleet. For a secondary scale-out plan, PP=2 across two nodes is a candidate. You need to prove — not assert — that:

- The TP deployment is actually sharding (each rank holds a weight slice, not a full copy).
- The TP deployment is using NVLink for its collectives, not PCIe fallback.
- The PP deployment hands off activations between nodes over the correct fabric.
- The per-iteration collective cost matches chapter 04's model (within a factor of 2).

The deliverable is the runbook and the measurements, not just "it serves responses."

## Prerequisites

- One node with ≥ 4 NVIDIA GPUs (H100 preferred; A100 / A6000 / 4090 workable with smaller models). NVLink or NVSwitch between the GPUs (verify with `nvidia-smi topo -m` before starting).
- Optional for PP: a second node with the same GPU count, connected to the first via a working network (IB preferred; Ethernet acceptable for the exercise's correctness, not for the latency numbers). If two nodes aren't available, simulate PP with virtual pipeline stages (vLLM supports `pipeline-parallel-size` on a single node; the geometry is same even if the fabric is intranode).
- Pinned `vllm` install with TP + PP support. SGLang is an acceptable substitute (adjust the launch commands); the health-check methodology is the same.
- Base model: a model that requires sharding on your hardware. Llama-3.1-70B BF16 if you have ≥ 4× 40GB GPUs; a smaller model (Qwen2.5-32B, DeepSeek-V2-Lite) if your GPUs are smaller.
- Access to `nccl-tests` (compile from `NVIDIA/nccl-tests` on GitHub).
- Chapter 02 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — The TP deployment

**Launch.** Produce the exact `vllm serve ... --tensor-parallel-size <N>` command. Include:

- Model path and revision.
- `--tensor-parallel-size` (pick 2, 4, or 8 based on hardware).
- `--dtype bfloat16` (or your target quantization).
- `--max-model-len <value>`.
- `--gpu-memory-utilization 0.90`.
- NCCL env: `NCCL_DEBUG=INFO`, `NCCL_P2P_LEVEL=NVL`, `NCCL_IB_HCA=...` if applicable.

**Startup verification (capture the following):**

- Full engine startup log (first ~500 lines).
- `NCCL_DEBUG=INFO` output showing: topology detected, number of channels built, `Using network` (NVLink / IB / Socket), per-GPU `via NVL` or similar.
- Per-worker HBM footprint from the engine's startup log and `nvidia-smi --query-gpu=memory.used --format=csv`.
- The engine's reported `tensor_parallel_size` from its status endpoint. Verify it matches the flag.

**Health check (sharding ≠ replication).** One of the common silent failures is the engine claims TP=N but each rank holds the full model (replication, not sharding). Prove sharding:

- Per-rank HBM should be ≈ `(model_bytes / N) + KV_slice + reserves`. If per-rank HBM ≈ full model bytes, you have replication, not sharding.
- Expected: for a 70B BF16 model with TP=4: ~35 GiB weights per rank + KV + reserves ≈ 60-70 GiB.
- Red flag: ~140 GiB per rank → something is replicated.

### Artefact B — The PP deployment (or simulation)

**Case 1: Two real nodes available.**

Launch across two nodes with `--pipeline-parallel-size 2`. See chapter 02 "vLLM — TP + PP across nodes" for the shape. Document:

- Ray (or `mp`) orchestration setup.
- NCCL environment on each node (IB vs. Ethernet, HCA selection).
- Startup log showing both nodes joined and `world_size = TP × PP`.
- A test request completes and returns tokens.

**Case 2: One node, simulate PP.**

Launch with `--pipeline-parallel-size 2` on a single node (the PP boundary is now within the node, over NVLink or PCIe depending on topology). This is not representative of the IB-over-nodes latency, but it exercises the PP code paths — scheduler, micro-batching, bubble.

Document the deviation from a real two-node setup and what the real setup's latencies would differ by (chapter 04 model).

**Micro-batching observation.** For the PP deployment, send a batch of N requests simultaneously and observe:

- The engine's per-worker log for the micro-batch flow.
- The time between first-request-admitted and first-token-emitted. This includes the pipeline-fill cost.
- The approximate bubble fraction given the batch size and PP degree.

### Artefact C — Sharding-vs-replication health-check script

A short script (Bash + Python; a few dozen lines) that:

1. Reads the engine's startup log and extracts `tensor_parallel_size`, `pipeline_parallel_size`, `world_size`.
2. Queries `nvidia-smi --query-gpu=memory.used --format=csv` on each worker and compares per-rank HBM against expected shard size.
3. Parses `NCCL_DEBUG=INFO` output and verifies:
   - `Using network` is NVLink (for TP) or IB (for PP across nodes).
   - Channel count matches TP degree.
   - No `WARN` entries.
4. Issues a single chat completion to the engine and verifies a response. (Smoke test.)
5. Exits with status 0 if all checks pass; non-zero with a specific error message on failure.

The script is the artefact — the next person who deploys TP on your fleet runs it instead of manually inspecting logs.

### Artefact D — Per-collective cost measurement table

Measure the per-collective time via two complementary paths.

**Path 1 — `nccl-tests` on the actual fleet.**

Run `nccl-tests all_reduce_perf -b 1K -e 1G -f 2 -g <N>` on the TP node. Record `bus-bw` at several message sizes:

- 128 KiB (decode-regime message).
- 4 MiB (prefill-regime message at small batch).
- 32 MiB (prefill-regime message at mid batch).
- 1 GiB (large message, bus-bandwidth asymptote).

Report in a table: message size, time, algo-bw, bus-bw. Compare bus-bw to the NVLink spec for your hardware (chapter 04 ballparks).

**Path 2 — engine-level measurement.**

Capture an Nsight Systems trace (`nsys profile`) of a 10-request serving window. Export the trace; find the NCCL AllReduce kernels in the per-rank timeline. Measure:

- Mean AllReduce duration per transformer layer at decode time (small message, ~128 KiB).
- Mean AllReduce duration per transformer layer at prefill time (larger message, ~4-32 MiB).
- Per-iteration (one forward pass) collective cost summed over layers.

Compare the engine's observed per-layer AllReduce time to the `nccl-tests` bus-bw at the same message size. They should match within a factor of ~2; a larger gap is the "the engine isn't overlapping" signal (route to performance peer).

**Report in a single table:**

```
| Context  | Message size | nccl-tests bus-bw | Engine observed | Expected (ch 04) | Status |
|----------|--------------|-------------------|-----------------|------------------|--------|
| decode   | 128 KiB      | ... GB/s          | ... ms/layer    | ... ms/layer     | ok/off |
| prefill-small | 4 MiB   | ... GB/s          | ... ms/layer    | ... ms/layer     | ok/off |
| prefill-mid   | 32 MiB  | ... GB/s          | ... ms/layer    | ... ms/layer     | ok/off |
| asymptote     | 1 GiB   | ... GB/s          | — (not reached) | spec            | ok/off |
```

### Artefact E — Deployment runbook

A one-page runbook for the next person deploying TP / PP on your fleet:

- Required pre-flight checks: `nvidia-smi topo -m` output, `ibv_devinfo` (if internode), GPU-health DCGM run.
- Required NCCL env variables and their meanings.
- Launch command shapes for TP, PP+TP, with placeholders for the model and sizes.
- Startup log signatures to verify (sharding, NVLink, IB).
- `nccl-tests` sanity command and expected numbers for your hardware.
- Known silent-fallback modes (chapter 02 enumerates five) and how to detect each.
- Rollback: if the deployment fails verification, what command returns the fleet to the previous state.

## Starter guidance

- **Run `nvidia-smi topo -m` first.** If any pair you'll shard across is `SYS`, stop; chapter 06 placement issue; fix before launching.
- **Set `NCCL_P2P_LEVEL=NVL` to catch Cliff 2 (PCIe fallback) early.** The communicator will fail loudly if NVLink isn't available between two ranks — better than silent degradation.
- **The Nsight trace is non-negotiable for Artefact D.** Without it, you cannot distinguish "the collective is slow" from "the collective isn't overlapped." Install `nsys` on the fleet before starting.
- **For PP, send a batch of ≥ 8 requests at once** to actually fill the pipeline. One-request measurements are dominated by the fill and tell you nothing about steady state.
- **Pin everything.** CUDA, driver, NCCL, Python, vLLM, model revision. Record every version in the runbook.
- **The health-check script is as valuable as the measurements.** It's what you run every deploy; the measurement is a one-time baseline.

## Fallbacks for constrained environments

- **Only 2 GPUs available on one node.** Run TP=2; the methodology is unchanged, the numbers are smaller.
- **No multi-node hardware.** Case 2 (simulated PP) is the fallback; call it out in the runbook.
- **No `nccl-tests` installed and cannot compile.** Use the engine's own benchmark harness (vLLM has `benchmarks/`) and compare the engine's reported per-step time to the chapter-04 model directly.
- **vLLM not available; have SGLang.** Substitute the launch command; the health-check script's parsing changes slightly; the measurement methodology is the same.
- **No Nsight Systems.** Use the engine's native per-iteration timing (vLLM's metrics endpoint, SGLang's benchmark output) and infer the collective time by subtracting an estimated compute time. Less precise; call it out.
- **IB fabric unavailable; running over Ethernet / RoCE.** The PP numbers will reflect the fabric; the methodology is unchanged. Record what fabric you ran on.

## Acceptance criteria

1. **Artefact A** is a running TP deployment with the exact launch command, startup log, and per-rank HBM footprint showing sharding (not replication). The engine serves a chat completion end-to-end.
2. **Artefact B** is a running (or clearly-documented-simulated) PP deployment with the micro-batch observation.
3. **Artefact C** is a committed script that runs clean on your deployment and exits 0; a deliberately-broken deployment (e.g. relaunch with `--tensor-parallel-size 1`) causes it to exit non-zero with the correct error.
4. **Artefact D** has the measurement table with both `nccl-tests` and engine-observed numbers. Any `off` entries have an attached note on the likely cause (chapter 02 fallback modes, chapter 04 overlap discussion, or chapter 06 placement cliff).
5. **Artefact E** is a one-page runbook a teammate could follow to reproduce the deployment from scratch.
6. All artefacts live in version control; environment manifest cross-referenced from each.

## Common failure modes

- **Launch succeeds but `NCCL_DEBUG=INFO` shows `Using network Socket`.** IB didn't bind. Chapter 02 Fallback 3. Fix before measuring.
- **Launch succeeds but per-rank HBM shows replication.** Chapter 02 Fallback 4 or a mismatched flag; fix before trusting measurements.
- **`nccl-tests` bus-bw matches spec; engine observed is 3× slower.** Overlap-with-compute not firing on your workload; chapter 04 discussion; route to performance peer (chapter 07) if you've verified everything on this side.
- **`nccl-tests` bus-bw is 5× below spec.** Chapter 06 placement issue; fix before benchmarking the engine.
- **PP bubble dominates at small micro-batches.** Expected; chapter 02 names the formula. Record the batch-size curve.
- **One rank reports slower than peers in Nsight trace.** Chapter 06 Cliff 1 (straggler); diagnose per chapter-06 flow.
- **TTFT measurement includes the first-request-after-cold-start.** Warm-up first; exclude the first N requests from the measurement window.
- **Measurement window too short.** Decode-time collective cost is noisy per step; measure over hundreds of iterations.
- **Mixing TRT-LLM and vLLM launch flags.** Each engine has its own flag spellings; verify against `--help` for the engine you're using.

## Keep it

The health-check script, launch runbook, and measurement table become the production bring-up template for every TP / PP deployment on your fleet. They feed exercise 03 (disaggregated serving uses the same TP / PP mechanics on each pool), exercise 04 (chapter 04 `nccl-tests` matrix extends to the full fleet), and exercise 06 (placement diagnosis uses the same health-check script as the first step). The runbook plugs into the module-113 rollout framework once the fleet is in production.
