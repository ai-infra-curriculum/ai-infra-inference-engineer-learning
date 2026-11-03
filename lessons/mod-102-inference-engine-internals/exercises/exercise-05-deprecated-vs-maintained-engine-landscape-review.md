# Exercise 05 — Deprecated vs. Maintained Engine Landscape Review

**Time:** 2 hours · **Output:** one TGI maintenance-posture audit, one capability-map placement for maintained alternatives, one TGI → vLLM / SGLang / TRT-LLM translation table, one short migration recommendation

## Goal

Make [Chapter 05](../05-tgi-deprecation-and-maintained-alternatives.md) concrete. Verify TGI's current maintenance posture against primary sources. Place the maintained alternatives (Hugging Face Inference Endpoints, Text-Generation-WebUI) on the module-101 capability map. Produce the TGI → modern-engine translation artefact your team can use when migrating a legacy deployment.

## Problem statement

Your platform team has inherited two legacy TGI deployments. Before kicking off migrations you need:

1. A verified, cited statement of TGI's current maintenance posture (not a vibe).
2. A placement of the maintained alternatives on the same capability axes you placed vLLM / SGLang / TRT-LLM / Dynamo / Triton / llama.cpp / MLC-LLM against in module 101 exercise 02.
3. A concept-and-flag translation table from TGI to the three serious engines, so the migration runbook can be drafted without inventing claims.
4. A short, concrete recommendation for *each of the two deployments* — which engine to migrate to, and why.

## Prerequisites

- Access to the Hugging Face blog, GitHub, and docs; the TGI repo; and the Inference Endpoints documentation. All public, no login needed for most of it.
- The capability-axes spreadsheet from module 101 exercise 02 (or the chapter-02 ten-axis list if you haven't done that exercise).
- A short written sketch of two legacy deployments to migrate. If you don't have real ones, invent two with realistic shapes — e.g., "70B chat model on 4× A100-80G nodes, OpenAI-compatible API, 2k shared system prompt, 10k RPS peak" and "7B instruct model co-located with other workloads on 2× L40S, structured JSON output required."

## Requirements

Deliver four artefacts.

### Artefact A — TGI maintenance-posture audit (≤ 1 page)

Verify, with citations, the current state of:

- The TGI GitHub repository: active? archived? maintenance-only? Last commit date, last release date.
- Hugging Face's own messaging about TGI: blog posts, docs, repo README. Note the date of the most authoritative statement you can find.
- Hugging Face's recommended serving story *today* for self-hosted deployment of models on the HF Hub. (Hint: look at the model page "Deploy" surface and the serving-related docs.)
- Any community-maintained TGI forks that have emerged post-deprecation signal. If none, say so.

For every claim: cite primary source (URL + date accessed). Where a public statement is ambiguous or contradictory, say so — the gap is the useful information.

### Artefact B — Maintained-alternatives capability-map placement

Add two columns to the module 101 exercise 02 spreadsheet (or author a mini-table if you haven't done that exercise):

- **Hugging Face Inference Endpoints**
- **Text-Generation-WebUI**

Place each on the ten axes from module 101 chapter 02:

1. Scheduling model
2. KV-cache primitive
3. Attention / fused kernel path
4. Quantization matrix
5. Parallelism
6. Disaggregated prefill-decode
7. Multi-adapter serving
8. Speculative decoding
9. API surface
10. Hardware backends

For managed products (Inference Endpoints) many of these axes are "controlled by the vendor" rather than operator-tunable; name the vendor control explicitly.

For each cell: primary-source citation (vendor docs). Where a cell truly is "not applicable because managed," say so with the reason.

### Artefact C — TGI → vLLM / SGLang / TRT-LLM translation table

Expand the translation table in [chapter 05](../05-tgi-deprecation-and-maintained-alternatives.md) with the specific knobs from the TGI version your legacy deployments actually run (or, if your deployments are imagined, the most recent widely-deployed TGI version).

Minimum rows:

- Context / sequence-length knobs (`MAX_INPUT_LENGTH`, `MAX_TOTAL_TOKENS`).
- Batching knobs (`MAX_BATCH_PREFILL_TOKENS`, `MAX_BATCH_SIZE`, `WAITING_SERVED_RATIO`).
- Parallelism (`NUM_SHARD`).
- Caching (prefix / paged KV).
- Quantization selection.
- Speculative decoding settings.
- API shape (HF Messages API vs. OpenAI-compatible).
- Model-loading env (`MODEL_ID`, `REVISION`, `TOKENIZER_NAME`).
- Health / metrics endpoints and metric name mapping.

Each row: the TGI concept + flag, the vLLM equivalent (flag or absence thereof), the SGLang equivalent, the TRT-LLM equivalent. Note where a one-to-one translation does *not* exist, with a short note on how to approximate.

### Artefact D — Migration recommendation (≤ 1 page total for both deployments)

For each of the two legacy deployments, produce a short recommendation of the form:

- **Target engine:** vLLM / SGLang / TRT-LLM (one).
- **Why that engine:** two or three sentences grounded in the workload shape (not an engine marketing bullet).
- **Translation summary:** three to five rows pulled from Artefact C relevant to this deployment.
- **Risks and open questions:** two or three items whose answers must be known before the migration cutover.
- **Fallback plan:** what you keep running (TGI pinned to its known-good version) if the migration doesn't meet SLO on first attempt.

## Starter guidance

- **Primary sources only.** No LinkedIn posts, no third-party summaries, no Reddit threads unless they cite primary sources.
- **Date-stamp every citation.** Repo READMEs and docs change; your audit is only trustworthy for the date accessed.
- **Managed products aren't drop-in replacements.** Inference Endpoints is a managed product with a different operator model. Flag that explicitly in Artefact B; do not pretend it has operator knobs it doesn't.
- **The recommendation isn't a decision.** It's a *proposal*. The actual decision runs through module 114's RFC process. The exercise output is a concrete, defensible starting point.
- **When evidence is ambiguous, name the gap.** `<!-- needs-research: ... -->` is the right output when a factual question cannot be settled from public sources. Do not guess.

## Acceptance criteria

1. Every claim in Artefact A has a primary-source URL and a date accessed.
2. The capability-map placement in Artefact B is complete across the ten axes for both products, with vendor-controlled cells explicitly labelled.
3. The translation table (Artefact C) has at least ten rows and every row has at least one real flag or env name per engine (or a labelled "no direct equivalent — approximate by X" cell).
4. Each migration recommendation (Artefact D) names a specific target engine, cites the workload-shape reasons, and lists at least two concrete risks.
5. A sibling-track teammate reading only Artefact C can execute the first-pass flag mapping for a different TGI deployment without re-reading the chapter.

## Common failure modes

- **Treating "deprecated" as "unusable."** TGI continues to run production workloads; the deprecation is about *where new deployments should go*, not an immediate decommission order.
- **Over-crediting managed products.** Inference Endpoints has real operator trade-offs (cost, data-boundary, flexibility) that a flat capability-map placement can hide. Call them out.
- **Inventing flag names.** TGI env-var names rotate over its lifetime; if you cannot verify the one you're citing in the current docs, mark it with `needs-research` rather than guessing.
- **Underwriting the recommendation.** A one-line "we should use vLLM" without workload-shape reasoning is not a recommendation. The two to three sentences in Artefact D should defend against a sceptical peer.

## Extensions (optional)

- Add rows to Artefact C for operational observability — DCGM / Prometheus / OTel wiring, metric name mapping, dashboard migration.
- Extend Artefact B with the other commercial compare-points from module 101 chapter 02 (Fireworks, Together, Baseten, Modal, Groq, Cerebras). The build-vs-buy framing from module 114 will reference this.
- Prototype one of the two migrations in a staging environment using the vLLM / SGLang launch commands you authored in exercises 01–02. The resulting hands-on report is the strongest version of this exercise.

## Keep it

Pin the translation table (Artefact C) to the team wiki alongside the vLLM / SGLang / TRT-LLM cheatsheets from exercises 01–03. Pair the audit (Artefact A) with the tech-radar document module 114 will produce. Refresh on every quarterly engine-landscape review.
