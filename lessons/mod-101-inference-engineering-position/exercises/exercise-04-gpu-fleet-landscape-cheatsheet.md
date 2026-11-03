# Exercise 04 — GPU Fleet Landscape Cheatsheet

**Time:** 2 hours · **Output:** a one-page cheatsheet plus three worked fleet-pick scenarios

## Goal

Turn [Chapter 04](../04-gpu-fleet-landscape.md) into a one-page reference you can open during a capital-planning meeting. The deliverable should answer "which GPU, how many, which numeric format, which engine, which parallelism topology?" at a glance.

## Problem statement

Your team is about to be pulled into three capacity decisions this quarter:

1. A new 70 B dense chat model needs serving capacity.
2. A new 400 B+ MoE model (Mixtral / DeepSeek-V3-scale) needs multi-node serving capacity.
3. A long-tail set of smaller models (≤ 13 B) needs cost-optimised capacity.

Each decision will be made in a meeting with procurement and the ML-platform peer. You need the one-page cheatsheet open on the shared screen.

## Requirements

Deliver two artefacts.

### Artefact A — The cheatsheet (one page)

A single page (A4 or Letter), portrait or landscape, containing:

1. **A GPU SKU table** with one row per currently-shippable SKU in each of these families. Minimum rows:
   - NVIDIA Hopper: H100 SXM, H200
   - NVIDIA Ada: L40S, L4
   - NVIDIA Blackwell: B200, GB200 NVL72, GB300 NVL72 (as fielded)
   - AMD: MI300X, MI325X
   - AWS: Inferentia2 (Inf2), Trainium2 (Trn2)
   - Google: TPU v5e, TPU v5p, TPU v6 Trillium / Ironwood (as fielded)

   Columns: HBM capacity (GB), HBM bandwidth (TB/s), peak FP16/BF16 compute, FP8 support (yes/no + hardware or emulation), FP4 support (yes/no), intra-node interconnect (NVLink / Infinity Fabric / ICI / N/A), inter-node interconnect typical (IB / RoCE / EFA / proprietary), engines with first-class support, notes.

   For any value you cannot verify today, write `verify` in the cell — do **not** fill in from memory. The cheatsheet is only useful if it's accurate.

2. **The four operator-depth questions**, prominently placed. Phrase them as a decision loop:
   - Can the model fit? (memory sizing rule of thumb for weights + KV + activations)
   - Will the interconnect carry it? (TP vs. node fabric; PP across nodes; EP for MoE)
   - Does the numeric format match? (FP8 on Hopper+; FP4 on Blackwell+; BF16 everywhere)
   - Does the engine support it as a first-class target?

3. **Rules of thumb** sections (bullet lists, ≤ 5 bullets each):
   - **Memory sizing** — bytes per parameter per numeric format; KV footprint per sequence
   - **Parallelism ceilings** — TP within an NVLink Switch domain only; EP sized to interconnect
   - **Numeric-format fallbacks** — if FP4 doesn't work in the engine today, fall back to FP8; if FP8 regresses accuracy, fall back to BF16

4. **A red-flag list** — three to five items that should block a fleet decision until clarified. Minimum coverage:
   - "FP8" claimed but engine releases don't list hardware support on that SKU
   - TP size proposed larger than the NVLink domain
   - Engine support status "experimental" for a load-bearing feature
   - Procurement timeline that lands the hardware after the model release deadline

### Artefact B — Three worked fleet picks

For each scenario below, deliver a short section (≤ 200 words) that uses the cheatsheet to make the fleet pick:

1. **70 B dense chat model**, 4 k average input, 500 output tokens, 50 % prefix-cache hit on a shared system prompt, 10 k RPS peak, OpenAI-compatible API, ~100 ms P99 TTFT target. Pick: GPU SKU, count, numeric format, engine, parallelism topology. Call out one risk.
2. **400 B+ MoE model** (treat the model as DeepSeek-V3-scale for sizing; verify current parameter / expert counts if you have access). 1 k average input, 500 output, 1 k RPS peak, 500 ms P99 TTFT acceptable. Pick: GPU SKU, count per node, nodes, EP topology, numeric format, engine, parallelism topology. Call out the two most load-bearing risks.
3. **Smaller-model long tail** — ~20 distinct models in the 1 B–13 B range, each ≤ 100 RPS, cost-sensitive. Pick: GPU SKU, co-location strategy (MIG / MPS / DRA — covered in module 109), engine, numeric format. Call out the trade-off against going to a commercial provider instead.

## Starter guidance

- Verify every number against primary sources (NVIDIA / AMD / AWS / Google product pages, engine release notes). The cheatsheet is only useful if it's current.
- When a cell is unverifiable from public sources, write `verify` and keep going. The cheatsheet is an artefact to *maintain*, not a one-shot document.
- Keep the three scenarios realistic — if your pick needs a hardware SKU that isn't shipping yet, say so and name the fallback.
- The red-flag list is the single most useful part of the cheatsheet. Make the items specific enough that a procurement partner can tell when to pause.

## Acceptance criteria

1. The cheatsheet fits on one page (if you need two, cut rules of thumb before you cut SKU rows).
2. Every cell is either (a) cited to a primary source or (b) explicitly marked `verify`.
3. The four operator-depth questions are present and answerable from the SKU table alone.
4. Each of the three scenarios names a specific GPU SKU, count, numeric format, engine, and parallelism topology — not a vibe.
5. Each scenario names at least one risk; the MoE scenario names at least two.

## Keep it

Pin the cheatsheet. Refresh it quarterly or on major hardware launches (new NVIDIA generation, new ROCm MI SKU, new TPU generation). Modules 107 (distributed topologies) and 109 (fleet autoscaling) will reference it directly.
