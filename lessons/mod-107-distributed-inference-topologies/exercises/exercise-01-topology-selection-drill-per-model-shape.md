# Exercise 01 — Topology Selection Drill per Model Shape

**Time:** 3 hours · **Output:** one topology-selection worksheet per model shape, one byte-math pass per candidate, one launch config, one goodput estimate, one `needs-escalation` list for the items you cannot resolve from the chapter

## Goal

Make [Chapter 01](../01-topology-selection-tp-pp-ep-dp.md) concrete by **walking the TP / PP / EP / DP decision tree for five distinct model + fleet shapes**, producing the byte-math behind each recommendation, the collectives the chosen topology will issue, and a draft launch config. The deliverable is a reusable worksheet — the thing you reach for when the next model lands on your platform.

## Problem statement

Your platform receives five serving requests in one planning cycle. Each has a different model, different QPS, different latency SLO, and different hardware budget. The CTO wants a topology recommendation for each, grounded in byte-math and chapter-04 collective-cost reasoning, by Monday. You will do the walk for all five.

The five shapes (adjust to fit the hardware you actually have access to; the shapes below are illustrative):

| Shape | Model | QPS target | TTFT SLO | TPOT SLO | Context | Hardware pool |
|---|---|---|---|---|---|---|
| A | Llama-3.2-3B-Instruct | 50,000 | 400 ms | 50 ms | 4k | L4 24GB and / or A10 24GB |
| B | Llama-3.1-8B-Instruct | 10,000 | 300 ms | 40 ms | 8k | H100 80GB SXM |
| C | Llama-3.1-70B-Instruct | 2,000 | 400 ms | 60 ms | 8k | H100 80GB SXM, 8-GPU nodes, NVLink intranode, 400 Gbps IB internode |
| D | Llama-3.1-405B-Instruct | 500 | 1,000 ms | 100 ms | 32k | H100 80GB SXM, 8-GPU nodes, 400 Gbps IB internode |
| E | DeepSeek-V3 (or Mixtral-8x7B-Instruct-v0.1 if V3 is out of reach) | 1,000 | 800 ms | 80 ms | 16k | H100 80GB SXM, 8-GPU nodes, 400 Gbps IB internode |

If a specific model or GPU is unavailable to you, swap in the closest available equivalent and record the substitution. The exercise is about the walk, not these exact numbers.

## Prerequisites

- Chapter 01 of this module read; chapter 04 skimmed (for the collective-cost model).
- Access to the module-104 KV byte-math and the module-105 quantization byte-math — you will reach for both.
- A notebook / spreadsheet for the byte-math (not strictly needed, but you will want it).
- Optional: access to actual hardware to validate the paper calculations. If no hardware, the drill produces the paper worksheet; chapter 02's exercise 02 validates the launch.

## Requirements

Deliver five artefacts. Four are per-shape worksheets; one is a cross-shape summary.

### Artefact A — Five topology-selection worksheets (one per shape)

Each worksheet is a Markdown document with the following sections.

**Section 1 — Input summary.** Copy the row from the table into the worksheet. Add any assumptions about workload shape (mean prompt length, mean generation length) if the table underspecifies.

**Section 2 — Byte-math.** For each candidate topology (name at least three per shape where plausible — e.g. DP-only, TP-within-node, TP+PP; or TP-within-node, PP-across-nodes, EP+TP for MoE), compute:

- `weight_bytes = model_params × weight_dtype_bytes` at the quantization you'd target.
- `kv_bytes_per_request = 2 × num_layers × seq_len × num_kv_heads × head_dim × kv_dtype_bytes`.
- Expected concurrent requests at target QPS and SLO; `kv_pool_bytes = concurrent × kv_bytes_per_request × margin`.
- `activation_bytes = batch × seq × hidden × bytes × ~constant` (illustrative; the engine's actual activation footprint varies).
- `reserve_bytes = ~10%` of total.
- `required_hbm_per_rank = (weight_bytes / TP) + (kv_pool_bytes / TP) + activation + reserves`.
- Compare `required_hbm_per_rank` to the GPU's HBM. Reject candidates that don't fit.

Table-shape output per shape:

```
| Candidate         | TP | PP | EP | DP | Per-rank HBM (GiB) | Fits? |
|-------------------|----|----|----|----|---------------------|-------|
| DP only           | 1  | 1  | — | N  | ...                 | yes/no|
| TP within node    | 4  | 1  | — | N  | ...                 | yes/no|
| TP + PP           | 8  | 2  | — | N  | ...                 | yes/no|
| EP + TP (MoE)     | 8  | 1  | 32 | 1  | ...                 | yes/no|
```

**Section 3 — Collectives per topology.** For each candidate that fits, name:

- The collectives per layer (AllReduce / AllGather / ReduceScatter / all-to-all / send-recv).
- The message size per collective (from the byte-math above).
- The expected per-collective latency on the fleet's interconnect — chapter 04 model is enough for a napkin estimate.
- Total collective-time per iteration — compare to the TPOT SLO.

**Section 4 — SLO-feasibility check.**

- Pick the lowest-collective-cost candidate that fits.
- Estimate TTFT: prefill compute time + first-token collective cost + (if PP) pipeline-fill overhead.
- Estimate TPOT: decode compute time + per-iteration collective cost.
- Pass / fail against the shape's SLO.
- If fail: iterate (bigger TP? smaller model? different quantization? different hardware?).

**Section 5 — Recommended topology.** State the choice in one paragraph with the specific `TP=`, `PP=`, `EP=`, `DP=` numbers and the quantization. Name the collectives it will issue and the baseline you'd expect to measure.

**Section 6 — Draft launch config.** Write the actual engine launch command (vLLM, SGLang, or TRT-LLM — pick based on your platform). Include:

- Model, dtype, context length.
- TP / PP / EP flags.
- NCCL env variables (`NCCL_DEBUG`, `NCCL_P2P_LEVEL=NVL` for TP groups, `NCCL_IB_HCA` if applicable).
- GPU-visible / memory-utilization flags.

**Section 7 — Measurements you'd take at bring-up.** Enumerate the chapter-02 health checks and the chapter-04 `nccl-tests` runs you'd execute before admitting the fleet into production.

### Artefact B — Cross-shape summary

One-page document that answers:

- Which shapes picked DP, TP, PP, EP, DP-attention?
- What was the dominant constraint for each (HBM, collective cost, QPS / replica count)?
- Which shape was closest to not fitting? What was the knob (quantization, more GPUs, smaller TP) that saved it?
- For any shape where you concluded "no feasible topology on this hardware," what is the ask: more hardware, different hardware, smaller model, revised SLO?

### Artefact C — `needs-escalation` list

A list of specific numbers you could not confidently fill in from chapter 01 + chapter 04 alone. For each, name what reference you'd consult (vendor doc, pinned engine release notes, nccl-tests on actual hardware, the performance peer).

Examples the list typically contains:

- "The exact collective overhead for TP=8 on H100 SXM at a 32 MiB message. Chapter 04 gives the model; the ballpark; the exact number needs `nccl-tests` on our fleet."
- "The DeepSeek V3 recommended serving topology. Chapter 05 names the shape; the vendor doc has the canonical config."
- "Whether sequence-parallel is actually on in vLLM 0.6 for Llama. Chapter 02 names the flag; the pinned release-notes have the current state."

The list is the honest audit of what the chapter did and did not teach.

## Starter guidance

- **Do the byte-math for weight, KV, activation, and reserves separately.** The error people make is "weight fits; good." Then KV at target concurrency doubles the real footprint.
- **Pick the quantization first and iterate.** FP8 vs BF16 shifts whether a model needs TP vs. PP. Module 105 is in the loop.
- **The decision tree is not a decision oracle — it is a filter.** It eliminates candidates that don't fit; among those that fit, choose the one with the lowest collective-cost on your fleet's interconnect.
- **Shapes A, B, and C are routine.** Shapes D and E are the ones where you'll do the most learning. Spend more time on them.
- **"On paper" is enough for this exercise.** Chapter-02 exercise 02 validates on hardware.

## Fallbacks for constrained environments

- **No access to multi-node hardware.** Do the walk on paper; label the multi-node candidates as "not validated." The methodology is the deliverable.
- **No L4 / A10 fleet for shape A.** Substitute T4 or any small-GPU fleet; the exercise is the walk.
- **DeepSeek V3 out of reach.** Substitute Mixtral or a Qwen-MoE variant; note the substitution and the trade-off (smaller expert count makes EP decisions easier).
- **Unfamiliar with chapter-04 collective-cost model.** Use the ballparks in chapter 04 Section "Expected numbers"; validate on hardware in exercise 04.

## Acceptance criteria

1. **Artefact A** has five worksheets, each with all seven sections populated. Each includes at least three candidate topologies in the byte-math table, with clear fits-or-doesn't-fit calls.
2. **Artefact B** names the dominant constraint per shape and the shape-at-risk, in one page.
3. **Artefact C** has at least five specific `needs-escalation` items with the reference for each.
4. The byte-math arithmetic is right. (This is checkable; have a second-eye pair.)
5. Every launch command compiles against the engine's `--help` (even if not actually run).

## Common failure modes

- **Forgetting KV in the fit check.** Weights pass; the engine OOMs on real workload. Chapter 01 Error 5.
- **Picking TP across node boundaries.** Chapter 01 Error 1 is specific; call it out.
- **Picking PP when the model fits on one node.** Chapter 01 Error 2 is specific.
- **Picking EP=1 for an MoE model.** Chapter 01 Error 4; verify the engine actually does route.
- **Ignoring the SLO-feasibility step.** A topology that passes the HBM filter may still fail SLO.
- **Not iterating with quantization.** Pick a quantization and the topology changes.
- **Confusing a shape with a specific model's config.** For e.g. Mixtral, verify the number of attention heads and the GQA KV-head count — these set the TP ceiling.
- **Picking a topology that assumes the hardware has features it doesn't.** NVLink bridge required for H100 PCIe cards; GPUDirect RDMA requires `nvidia-peermem` loaded; verify the fleet's actual capabilities.
- **"We can always buy more hardware" as the fallback.** Note the capacity ask explicitly for the architect (chapter 07).

## Keep it

The five worksheets become the template for every new model that lands on the platform. The byte-math scripts (if you wrote them) and the launch-config drafts are reusable for exercise 02 (actually run them) and for every production bring-up. The `needs-escalation` list feeds the fleet's documentation gaps.
