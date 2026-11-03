# Exercise 03 — Inference Metric and Term Glossary

**Time:** 2 hours · **Output:** a team-wiki-ready glossary tied to engine primitives

## Goal

Lift the vocabulary from [Chapter 03](../03-inference-vocabulary.md) into a glossary your team can live with — one that stays accurate as the engines underneath it change. This exercise is the "shared mental model" artefact that modules 102–114 all rely on.

## Problem statement

Your team's dashboards label latency columns inconsistently (one dashboard says "latency P99", another says "TTFT P99", a third says "e2e P99"). Metric names in Prometheus don't match the terms peers use in Slack. New hires spend their first week asking what "goodput" means in your fleet. Fix the vocabulary once; rename the dashboards and the Prometheus metrics to match.

## Requirements

Deliver one artefact — a glossary document (Markdown, Notion page, wiki page) that your team can pin — containing the entries below.

### Required glossary entries

At minimum, define each of these terms with the structure described in "Entry format":

**Latency metrics**

- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- ITL (Inter-Token Latency)
- E2E latency (End-to-End)

**Throughput metrics**

- Throughput
- Goodput
- Cache-hit rate (prefix cache)
- Preemption rate

**Scheduling and KV primitives**

- Continuous batching
- Chunked prefill
- Prefix caching
- PagedAttention
- RadixAttention
- Disaggregated prefill–decode
- Speculative decoding (plus flavours: draft-target, Medusa, EAGLE, Lookahead, n-gram)

**Parallelism primitives**

- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Expert parallelism (EP)
- DP attention
- LoRA swap / multi-LoRA
- Model sharding

**API primitives**

- Streaming
- Structured output / constrained decoding

### Entry format

Each entry must contain:

1. **Term** (bold, with any common synonyms in parentheses)
2. **Definition** — one or two sentences, operator-level, not kernel-level
3. **Unit** — the SI unit or percentage; the time window if applicable; "N/A" if not a numeric metric
4. **Dominated by** — the single most load-bearing contributor (e.g., for TTFT: "prefill cost and queue depth")
5. **Engine primitive** — the concrete scheduler / allocator / API code path the term maps to in at least one engine
6. **Prometheus / exporter name in our fleet** — the metric name as it appears (or should appear) in your Prometheus registry; if there is no metric yet, write `TODO: expose as <proposed-name>`
7. **Who else uses this term** — the peer(s) who are likely to reference it in their own dashboards (e.g., `ai-infra-sre` for SLO work, `ai-eval-engineer` for eval-vs-serving reconciliation)

### Required cross-references

The glossary must also contain:

- **A term-to-engine-primitive reference table** — one row per term, columns `Term` and `Engine primitive(s) that implement it`. (You can borrow the table shape from Chapter 03 but **verify** each engine primitive against today's engine source / docs.)
- **An alias / anti-pattern section** — list labels on your current dashboards that are *not* in the glossary (e.g., "latency_p99 without a scope qualifier") and the rename needed to bring them into line.

## Starter guidance

- The glossary is a *team* artefact, not a textbook. Prefer "operator-level" definitions over mathematically precise ones. If a definition is longer than two sentences, cut.
- Where the term's definition depends on which engine you're running, say so — e.g., "ITL on vLLM reflects stream-pipe behaviour between the engine and the OpenAI-compatible server; the exact flush boundary is a function of the HTTP streaming layer."
- Where a Prometheus metric does not exist in your fleet, propose one. Modules 103 and 110 will come back and implement.
- Cite primary sources from `resources.md` where the definition is contested.

## Acceptance criteria

1. All required terms are present with the full 7-field entry.
2. The term-to-engine-primitive table covers every term in the glossary.
3. The alias / anti-pattern section names at least 3 inconsistent dashboard labels and the rename.
4. A new hire can read the glossary and label a TTFT regression ticket, a TPOT regression ticket, and an ITL regression ticket differently — i.e., the glossary disambiguates these three in practice, not just in theory.
5. A peer from `ai-infra-sre` can read the glossary and plug the SLI definitions into their SRE methodology without rewriting the terms.

## Keep it

Make this the single source of truth for terminology on your platform. Link to it from every runbook, every dashboard description, every Prometheus comment. Update it on every engine upgrade that changes a primitive. Modules 103–110 will each cite entries from it.
