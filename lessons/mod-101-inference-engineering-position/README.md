# Module 101 — The Inference Engineer Position on the Infra Ladder

> **Role:** AI Inference Engineer · **Level:** L35 (AI Infrastructure family) · **Hours:** 12

The orientation module for the AI Inference Engineer track. By the end of it you can:

- Name what this role owns vs. the five peer tracks (`ai-infra-performance`, `ai-infra-ml-platform`, `ai-infra-sre`, `ai-infra-mlops`, `ai-infra-security`) and the consumer tracks in the AI Engineering family.
- Place every major open-source inference engine (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, Triton Inference Server, llama.cpp, MLC-LLM) and the proprietary / commercial stacks (NVIDIA NIM, Fireworks, Together, Baseten, Modal, Groq, Cerebras) on a shared capability map, including TGI as a historical compare-point.
- Use the inference vocabulary (TTFT, TPOT, ITL, E2E, goodput, cache-hit rate, preemption rate, continuous batching, chunked prefill, prefix caching, PagedAttention, RadixAttention, disaggregated prefill-decode, speculative decoding, TP / PP / EP, LoRA swap, structured output, streaming) and tie each term to the engine primitive that implements it.
- Reason about the current GPU hardware landscape (Hopper, Ada, Blackwell, GB200 / GB300, Vera Rubin; AMD MI300 / MI325; AWS Trainium / Inferentia; Google TPU) at operator-depth — enough to pick a fleet without owning kernel-authoring depth.
- Author a "what this platform does and does not do" charter that routes incoming requests to the right owner.

> **Prerequisite:** [Module 00 — Your Relationship to the Model](../00-your-relationship-to-the-model/README.md). Read that first if you haven't.

## Chapters

1. [**The Inference Engineer's Position on the Ladder**](01-role-scope-on-the-ladder.md) — one-line charter; six peer boundaries (performance / ML platform / SRE / MLOps / security / consumer tracks); routing scenarios.
2. [**The Inference Engine Capability Map**](02-engine-capability-map.md) — ten capability axes; the three serious open-source generalists (vLLM, SGLang, TensorRT-LLM); Dynamo and Triton; llama.cpp and MLC-LLM; TGI historical compare-point; commercial compare-points (NIM, Fireworks, Together, Baseten, Modal, Groq, Cerebras).
3. [**The Inference Vocabulary**](03-inference-vocabulary.md) — latency metrics (TTFT / TPOT / ITL / E2E); throughput metrics (throughput / goodput / cache-hit / preemption); scheduling and batching primitives (continuous batching, chunked prefill, PagedAttention, RadixAttention, disaggregated prefill-decode, speculative decoding); parallelism (TP / PP / EP + DP attention + LoRA swap + model sharding); API primitives (streaming, structured output); term-to-engine-primitive reference table.
4. [**The GPU Fleet Landscape at Operator Depth**](04-gpu-fleet-landscape.md) — NVIDIA Hopper / Ada / Blackwell / GB200 / GB300 / Vera Rubin; AMD MI300 / MI325; AWS Trainium / Inferentia; Google TPU v5p / v6 / Ironwood; the four operator-depth questions (can the model fit, will the interconnect carry it, does the numeric format match, does the engine support it).
5. [**Authoring the Inference Platform Charter**](05-platform-charter.md) — the seven sections (mission, services, non-services, SLOs, protocol contract, change management, peer interfaces); anti-patterns; update cadence.

## Exercises

The exercises land the chapters as a reusable artefact kit you'll keep for the rest of the curriculum:

1. [**Role scope and deferral contract**](exercises/exercise-01-role-scope-and-deferral-contract.md) — a diagram of your scope and a routing table against the five peers and the consumer tracks.
2. [**Inference engine capability map**](exercises/exercise-02-inference-engine-capability-map.md) — place each engine on the ten capability axes against today's docs, with citations.
3. [**Inference metric and term glossary**](exercises/exercise-03-inference-metric-and-term-glossary.md) — a glossary you commit to the team wiki, with each term tied to an engine primitive.
4. [**GPU fleet landscape cheatsheet**](exercises/exercise-04-gpu-fleet-landscape-cheatsheet.md) — one-page cheat sheet of today's accelerators with the four operator-depth questions prefilled.
5. [**Platform charter authoring**](exercises/exercise-05-platform-charter-authoring.md) — a full charter draft against a stated organisation shape.

## Resources

See [`resources.md`](resources.md) for the primary / citable references the chapters draw on.

## Where this fits in the curriculum

- **Before:** [Module 00 — Your Relationship to the Model](../00-your-relationship-to-the-model/README.md)
- **After:** `mod-102-inference-engine-internals` (reading engine source to debug outages)
- **Depends on:** The ladder context in [`CURRICULUM.md`](../../CURRICULUM.md) and the role-boundary map authored here.
