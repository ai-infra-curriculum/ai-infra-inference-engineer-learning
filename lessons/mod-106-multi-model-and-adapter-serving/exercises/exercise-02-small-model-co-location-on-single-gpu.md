# Exercise 02 — Small-Model Co-location on a Single GPU

**Time:** 4 hours · **Output:** three co-location deployments (engine-level, MIG, MPS), one noisy-neighbour harness, one latency / throughput / isolation table, one decision write-up

## Goal

Make [Chapter 02](../02-multi-model-co-location-on-a-single-gpu.md) concrete by **co-locating two or more small models on a single GPU via each of the three mechanisms** (engine-level, MIG hard partition, MPS soft sharing), **measuring the same workload across all three**, and **publishing a trade-off table** that a future deployer can use to pick the right mechanism. By the end you can state, with measurements, when each co-location mechanism is worth it.

## Problem statement

Your platform has a long tail of small models — a 1B embedding model, a 1.5B draft model, a Whisper-small, and a small classifier. Each of these on its own GPU is wasteful. You need to co-locate; the question is *which mechanism*. You will build three deployments and run the same noisy-neighbour test against each. The deliverable is a direct comparison and a decision recipe.

## Prerequisites

- One Hopper or Ampere GPU that supports **MIG** (A100, H100, H200, GB200). If MIG is unavailable, you can still complete the engine-level and MPS portions; note the gap.
- CUDA 12.x and matching driver (verify driver supports current MIG and MPS).
- A workstation or lab host where you can enable / disable MIG mode (requires root; may require a reboot or driver-reset).
- A pinned `tritonserver` (Triton Inference Server) install for the engine-level portion.
- Alternatively, a pinned `vllm` with multi-model support (verify your version) or two `vllm` instances for the engine-level portion.
- Two or more small models in a format each mechanism can serve. Suggested set (all run on one H100 easily):
  - **1B embedding model** (e.g. `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` wrapped for Triton; or a BGE-large if you want more HBM pressure).
  - **1.5–3B causal LM** (e.g. `Qwen2.5-1.5B` or `Llama-3.2-3B`).
  - Optional third: a Whisper small / tiny model for ASR.
- Chapter 02 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — Deployment 1: engine-level co-location

Pick one engine-level approach:

**Option A1 — Triton Inference Server with multiple models.** Build a model repository with your two or three models each with their own `config.pbtxt`. Launch Triton; verify each model is reachable at `/v2/models/<name>/infer`.

**Option A2 — Two vLLM processes on the same GPU.** Launch two `vllm serve` processes, each with its own `--port` and its own model, each sized to use a share of HBM via `--gpu-memory-utilization`. Not technically "one process" but it is engine-level co-location without any hardware primitive.

Document for whichever you chose:

- Model repository layout (for Triton) or launch commands (for vLLM).
- `nvidia-smi` output showing both models' HBM footprints on one GPU.
- Health check on each model.

### Artefact B — Deployment 2: MIG hard partition

Enable MIG and create partitions sized for your models:

```
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -cgi <profile_1>,<profile_2>,... -C
nvidia-smi -L     # confirm partition UUIDs
```

Pick profiles whose HBM allocation fits each model with some KV / activation headroom.

Launch each model on its own MIG partition:

```
CUDA_VISIBLE_DEVICES=MIG-<UUID_1> <launch command for model 1> &
CUDA_VISIBLE_DEVICES=MIG-<UUID_2> <launch command for model 2> &
```

Document:

- Partition profile and UUID per model.
- Each model's `nvidia-smi` view (showing it sees only its MIG slice).
- Health check on each.

**Important:** when you're done with the MIG exercise, destroy partitions and disable MIG mode to free the GPU for subsequent exercises:

```
sudo nvidia-smi mig -dgi            # destroy GPU instances
sudo nvidia-smi -i 0 -mig 0
```

### Artefact C — Deployment 3: MPS soft sharing

Start the MPS daemon and launch two clients with per-client limits:

```
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
```

Launch each model as an MPS client with its own SM-percentage and memory limit:

```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=20480 \
    <launch command for model 1> &
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=20480 \
    <launch command for model 2> &
```

Document:

- MPS daemon status (`nvidia-cuda-mps-control` client list).
- Each client's SM / memory limit.
- Health check on each.

Shut down MPS when finished:

```
echo quit | nvidia-cuda-mps-control
```

### Artefact D — Noisy-neighbour harness and measurement table

Design a two-workload harness:

- **Workload "quiet"** on model 1: steady 10 RPS, measured for TTFT P50 / P99 and TPOT P50.
- **Workload "noisy"** on model 2: a burst — ramp from 0 to 100 RPS over 30 seconds, hold 100 RPS for 2 minutes, ramp back.

Run the harness in **three regimes** across the three deployments:

1. **Quiet only.** Baseline — model 1's latency without any contention.
2. **Noisy only.** Baseline — model 2's latency without quiet workload interfering (and vice versa).
3. **Both simultaneously.** The noisy-neighbour scenario — quiet latency during the noisy burst.

Measure, for each (deployment, regime) cell:

- Quiet workload: TTFT P50, TTFT P99, TPOT P50.
- Noisy workload: throughput (requests/sec), TTFT P50.
- (Both regimes) end-to-end latency distribution of the quiet workload at the burst peak.

Publish as a table. Expected shape:

- **Engine-level:** throughput best; noisy workload steals from quiet; quiet P99 spikes during burst.
- **MIG:** hardware isolation holds; quiet workload is unaffected by the noisy burst (within noise); throughput ceiling lower than full-GPU.
- **MPS:** between the two; percentage limits enforced but SM contention still shows.

### Artefact E — Decision write-up

A one-to-two-page write-up answering:

- Which mechanism do you recommend for a **customer-facing latency-SLO model**? Defend from measurements.
- Which mechanism do you recommend for a **throughput-oriented batch model**? Defend.
- Which mechanism do you recommend for the **mixed case** (one latency-SLO model + one batch model co-located)? Defend.
- Which measurement surprised you, if any? What hypothesis would you test next?
- Operational cost of each mechanism (MIG partition management overhead, MPS daemon maintenance, engine-level model-ensemble reload concerns).

## Starter guidance

- **Pin everything.** CUDA, driver, Python, engine versions. Record in `environment.md`.
- **Use the same load generator for all three deployments.** Varying the harness makes comparisons unreliable.
- **Baseline each model solo before co-locating.** Any surprise is easier to attribute if you know the solo profile.
- **For MIG, pick profiles that fit your models with some headroom.** A 10 GiB partition does not fit a 3B BF16 model (6 GiB weights + KV = tight). Use `1g.20gb` or similar.
- **Measure MPS with and without SM-percentage limits.** The "soft isolation" is substantially weaker without the percentage limit; show both.
- **Use short bursts, not sustained peak.** The point is to measure the shape of the degradation, not stress-test the hardware.
- **For Triton, prefer the dynamic batcher per model.** The engine's effective batch sizes affect your throughput measurement.

## Fallbacks for constrained environments

- **No MIG support (consumer GPUs, Turing / earlier generations).** Skip Deployment 2. Note the gap in the write-up; still complete engine-level and MPS.
- **No root / reboot access to enable MIG.** Same as above; use a cloud GPU rental for the MIG portion if cost permits.
- **No Triton install.** Two vLLM processes work as the engine-level approach; also acceptable are two SGLang instances. Note the choice.
- **One small GPU (24 GiB).** Choose smaller models (0.5B, 1B) that fit the MIG partitions available on your hardware.

## Acceptance criteria

1. **Artefact A** has at least two models running on one GPU via the engine-level approach, with a health check and documented memory usage.
2. **Artefact B** has at least two models running on separate MIG partitions, with partition UUIDs and health checks (or an explicit note that MIG is unavailable).
3. **Artefact C** has at least two models running under MPS with per-client limits, with daemon state verified and health checks.
4. **Artefact D** has the quiet / noisy / both measurements across all deployments you built, with a clear table showing the isolation differences.
5. **Artefact E** is a decision write-up with recommendations backed by the measurements, including the operational-cost dimension.
6. All artefacts live in version control; environment manifest shared across them.

## Common failure modes

- **MIG partitioning fails because another process holds the GPU.** Kill all CUDA processes first (`nvidia-smi` shows PID list); if that doesn't work, reboot.
- **MPS daemon dies silently.** Check `nvidia-cuda-mps-control` client list and the daemon log in `CUDA_MPS_LOG_DIRECTORY`.
- **Engine-level "noisy" workload actually under-utilises because of too-small batch size.** Verify the noisy workload is actually saturating; check throughput, not just RPS.
- **Measurement contamination from startup.** Discard the first 30 seconds; models' JIT compile and warm up during that window.
- **MIG partition profile too small.** HBM OOM at engine startup. Pick a larger profile.
- **MPS percentage limits silently ignored.** Older driver, or missing `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` export in the client's actual environment. Verify by running `nvidia-smi -q -d UTILIZATION` during the test.
- **Different `max_batch_size` across deployments.** The engine config affects throughput; pin equally.

## Keep it

The co-location deployments and the noisy-neighbour harness are reusable: exercise 05 (multi-tenant isolation) stresses the same primitives from the tenant-contract angle; module 109's autoscaling exercises build on top of the GPU-sharing choices made here.
