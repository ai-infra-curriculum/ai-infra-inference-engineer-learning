# Exercise 01 — Deploy Multi-LoRA with the S-LoRA Pattern

**Time:** 4 hours · **Output:** one adapter registry, one running multi-LoRA deployment, one per-adapter cold-start / batched-throughput / HBM table, one noisy-batch stress test, one deployment checklist

## Goal

Make [Chapter 01](../01-multi-lora-with-s-lora-and-punica.md) concrete by **standing up a multi-LoRA deployment serving a base model with twenty or more LoRA adapters on a single GPU**, **feeding the engine a real adapter registry**, and **measuring the three load-bearing numbers**: per-adapter cold-start latency, batched-adapter throughput at mixed-adapter batches, and HBM footprint compared to the "one base model per adapter" baseline. By the end you can state exactly what your engine does with adapters under load.

## Problem statement

Your team is scoping a multi-tenant platform where each of twenty-plus customers has their own LoRA adapter on a shared base model. Before committing to the fleet shape, you need credible numbers on three axes: whether S-LoRA / Punica actually batches heterogeneous adapters in your engine, how much HBM you can save vs. twenty separate deployments, and what the first-request-after-load latency penalty looks like. The deliverable is the measurements and a go / no-go on the "one fleet, many adapters" posture.

## Prerequisites

- One Hopper or Ada-class GPU (H100 80 GB preferred; L40S also works) with CUDA 12.x and matching driver.
- Python 3.10–3.12 with virtual environments.
- A pinned `vllm` install with multi-LoRA support, OR a pinned `sglang` install with the multi-LoRA path enabled. (TRT-LLM is harder to drive for this exercise; the ergonomics are not there.)
- Twenty or more LoRA adapters trained on the **same base model**. Options:
  - Pull adapters from Hugging Face (search for `base_model:<your-base>` LoRA checkpoints). Confirm each adapter's `adapter_config.json` names the exact base model + revision.
  - Train twenty quick adapters yourself with PEFT (`peft.LoraConfig`) on different small datasets — different domains, different subsets of a corpus. Rank-16 is fine; targets Q,K,V,O + MLP.
  - Mix: 10 pulled, 10 trained, so you have variety in rank and target modules.
- Chapter 01 of this module read.

## Choose a base model

Pick a base model that:

- Is covered by its licence for your use (see module 105 chapter 05).
- Has published LoRA adapters, or is a common target for fine-tuning (easier to source adapters).
- Fits comfortably on your GPU in BF16 with headroom for the KV pool and adapters.

Recommended choices:

- **Llama-3.1-8B-Instruct** (lots of community LoRAs on HF).
- **Mistral-7B-Instruct-v0.3** or **Qwen2.5-7B-Instruct** (alternatives).
- **Llama-3.2-3B-Instruct** (fits more adapters per GPU).

Record the base model and its HF revision SHA.

## Requirements

Deliver five artefacts.

### Artefact A — The adapter registry

Author a registry (YAML or JSON) that is both the engine's `--lora-modules` input *and* the audit record of what the engine is serving.

Per-adapter fields:

- `adapter_id` — the ID the request references (unique).
- `base_model` and `base_model_revision` — must match for all 20+ adapters.
- `adapter_path` — local path or HF ID.
- `adapter_sha256` — SHA-256 of the adapter.safetensors file.
- `rank`, `alpha`, `target_modules` — read from `adapter_config.json`.
- `dtype` — storage precision.
- `tenant` — a synthetic tenant ID (`customer-01` … `customer-20`).
- `status` — `active`.
- `size_mib` — computed from the files; also check against the chapter-01 formula.

Validation checks before you launch:

1. All adapters' `base_model` matches the chosen base. Reject any that don't.
2. All adapters' target modules are compatible with the engine's LoRA support (verify via the engine's LoRA docs for your version).
3. SHA-256 computed and stored.
4. Rank distribution logged (how many at r=4, r=8, r=16, r=32, r=64).

Commit the registry to version control.

### Artefact B — The running deployment

Launch the engine with multi-LoRA enabled. Example for vLLM:

```
vllm serve <base-model> \
    --enable-lora \
    --max-loras <N_concurrent_in_batch> \
    --max-lora-rank <max_rank_in_registry> \
    --max-cpu-loras <total_in_registry + headroom> \
    --lora-modules <adapter_id>=<path> ... (all 20+ adapters) \
    --max-model-len <chosen_context_len> \
    --gpu-memory-utilization 0.90
```

Document:

- The exact launch command with all 20+ adapters.
- The startup log: `num_gpu_blocks`, `loaded loras`, HBM footprint reported at startup.
- A health-check: send one chat completion specifying one of your adapters by ID; verify a response arrives.
- One cross-adapter sanity test: five requests, five different adapters, interleaved; verify all succeed and the engine does not serialise them.

Pin flag spellings and values; the output of `vllm serve --help` for your pinned version lives alongside the launch command.

### Artefact C — Measurement table — cold-start / throughput / HBM

Measurements across three regimes:

**Regime 1 — Cold-start latency per adapter.**

For each of the 20+ adapters (or a representative subset of 10):

- Restart the engine (or evict the adapter via a dynamic unload endpoint if your engine supports it).
- Send one request to the adapter.
- Measure TTFT.
- Repeat 3 times per adapter; median TTFT.

Expected: the first request after load is slower (adapter must be registered + possibly CUDA-graph-warm-up); subsequent requests are at steady state. Capture both.

**Regime 2 — Batched throughput at heterogeneous-adapter batches.**

Three workload shapes:

- **Homogeneous:** 32 concurrent requests, all using one adapter.
- **Mild heterogeneity:** 32 concurrent requests spread across 8 adapters (4 requests per adapter).
- **Full heterogeneity:** 32 concurrent requests spread across 32 adapters (requires 32+ adapters in your registry, or run with 20 and oversubscribe).

For each shape, measure:

- Aggregate throughput (requests/sec).
- Per-request TTFT P50, P99.
- Per-request TPOT P50.

Expected: homogeneous is fastest (the batched kernel groups by adapter); full heterogeneity shows the ragged-rank penalty if your adapters span ranks. Document which penalty appears.

**Regime 3 — HBM footprint vs. baseline.**

- Multi-LoRA deployment: `nvidia-smi` HBM usage with all 20 adapters loaded + KV pool at a chosen context size.
- Baseline comparison: compute (don't actually deploy) the HBM usage if each adapter were its own worker (base-model bytes × 20 + per-worker KV + per-worker reserves). Use chapter 01's byte-math and module 104's KV byte-math.

Publish the three numbers and the ratio.

### Artefact D — Noisy-batch stress test

Design and run a stress scenario:

- 100-prompt workload, 50 adapters (if you have them), heavy mix at different ranks (if your adapters have mixed ranks).
- Drive at increasing concurrency until the engine's queue saturates.
- Capture: per-adapter TTFT P99 vs. request count per adapter; TPOT distribution; any engine-level warnings in the log.

Narrative (≤ 300 words): what shape the degradation takes, which adapters (if any) suffered disproportionately, and what you'd change if the SLO were tighter.

### Artefact E — Deployment checklist

A one-page checklist for the next person deploying multi-LoRA on this fleet:

- Required engine flags (and why).
- Minimum adapter-registry fields (and why — map each to a failure mode if missing).
- Pre-flight validations (base-model match, target-module compatibility, SHA verification).
- Launch-sanity tests (interleaved-adapter health check, cold-start probe).
- Metrics to watch in the first hour (loaded-LoRAs count, batched-adapter rate, cold-start TTFT, adapter-rejection rate).

## Starter guidance

- **Start with 5 adapters, not 20.** Debug the plumbing on a small set; scale once the health check is reliable.
- **Pin everything.** CUDA, driver, Python, engine, base-model revision, adapter revision list. Record every pin in an `environment.md`.
- **Validate adapter → base-model compatibility before launch.** The engine will reject at load time, but it's slower to iterate that way. A pre-flight check script that confirms every adapter's `adapter_config.json` names the base model you'll serve saves hours.
- **Use the engine's own metrics endpoint.** Don't infer "the batched path is firing" from throughput alone; look for a `lora_batch_size` or `requests_per_lora` metric (name varies by engine).
- **For throughput measurements, use a load generator (`vllm/benchmarks/benchmark_serving.py` or an equivalent).** Hand-timing with `time curl` wastes a day.
- **For heterogeneous-rank batches, ensure your registry actually has mixed ranks.** If all 20 adapters are rank-16, you cannot measure the ragged-rank penalty.
- **Separate "adapter loaded" from "adapter warmed up."** The engine may claim loaded status while the first inference is still slow because of graph compile.

## Fallbacks for constrained environments

- **No 20+ adapters available.** Train 5 fast ones with PEFT on tiny datasets (small-scale fine-tuning is minutes-not-hours); exercise with 5 instead of 20.
- **No multi-LoRA support in your engine version.** Upgrade or move to an engine that supports it (vLLM 0.5+, SGLang recent). Record the gap if you cannot upgrade and run as much of the exercise as you can with "one base + one adapter."
- **No Hopper hardware.** Multi-LoRA works on Ada / Ampere; the kernel's throughput profile differs but the operator work is the same. Note the hardware; the numbers are hardware-specific.
- **No dynamic adapter unload.** Simulate cold-start by restarting the engine between measurements. Document the extra startup cost.

## Acceptance criteria

1. **Artefact A** is a committed adapter registry with ≥ 20 adapters (or the fallback count), every field populated, SHA-256 computed, base-model compatibility verified.
2. **Artefact B** shows a running multi-LoRA engine with the exact launch command, startup log, and interleaved-adapter health-check passing.
3. **Artefact C** has three sub-tables (cold-start per adapter, throughput at three heterogeneity levels, HBM vs. baseline) with the measurement methodology documented.
4. **Artefact D** has the stress test result with a per-adapter TTFT P99 distribution and a narrative on degradation shape.
5. **Artefact E** is a checklist a teammate could follow to redeploy the same configuration from scratch.
6. All five artefacts live in version control; environment manifest cross-referenced from each.

## Common failure modes

- **Target-module name drift.** The adapter targets `up_proj`; the engine wants `w1`. Add an alias map or re-train.
- **`--max-lora-rank` too small.** An adapter with rank-64 fails to load. Set the flag at the max rank in your registry.
- **CUDA-graph recompile on first use of each adapter.** Pre-warm at startup; the engine may support this via `--enable-prefix-caching` + warm requests.
- **Benchmarking across different seeds / prompts per adapter.** Keep the prompt set identical across adapters for the throughput measurement.
- **HBM number is misread.** `nvidia-smi` reports process-wide; the engine's own startup log is more precise.
- **Measurement window too small.** 10 requests per adapter is noisy; measure 100+ for the throughput cells.
- **Confusing "loaded in registry" with "loaded in HBM."** The engine may list all 20 as registered but only hold a subset in HBM at any moment. Verify.

## Keep it

The adapter registry, launch command, measurement harness, and deployment checklist are reusable for exercises 04 (dynamic loader with eviction) and 05 (multi-tenant isolation), and compose with module 105 chapter 06's in-flight-swap primitive when the base model changes under a loaded adapter set.
