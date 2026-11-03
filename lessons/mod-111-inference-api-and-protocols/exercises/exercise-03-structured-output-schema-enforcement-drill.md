# Exercise 03 — Structured Output Schema Enforcement Drill

**Time:** 4 hours · **Output:** one per-mode correctness and latency table for a non-trivial schema, one per-request routing policy, one set of Pydantic models used across the drill

## Goal

Make [Chapter 03](../03-structured-output-and-constrained-decoding.md) concrete by **running the same JSON Schema through all four enforcement modes** (post-hoc validate, JSON mode, strict JSON Schema, full grammar) against the same engine and measuring the correctness, the latency tax, and the operational footprint of each mode. Publish the per-mode table your fleet uses to route per-request.

## Problem statement

Chapter 03 names the four modes and the trade-off. In practice, which mode your fleet uses depends on the schema's shape, the model's native capability, and the latency budget. Today you measure the four modes on a representative schema, document the trade-offs, and author the per-request routing policy.

You will pick (or author) a non-trivial JSON Schema with nesting, enums, and conditional required-fields; derive the same schema as a Pydantic model and as an EBNF / Lark grammar; run the schema through each enforcement mode on at least one engine; measure correctness (validation pass rate) and latency (TTFT, TPOT, per-mode overhead); and publish the per-mode table.

## Prerequisites

- Chapter 03 of this module read.
- A running inference engine with at least one constrained-decoding backend (vLLM + Outlines or XGrammar; SGLang; TRT-LLM has partial support — verify).
- A representative instruct model — 7B–13B class is enough; the enforcement mechanics are identical at any size.
- Python 3.11+; `pydantic`, `jsonschema`, `httpx`, `outlines` (if driving Outlines directly), `xgrammar` (if available).
- The exercise 01 harness (reuse the response capture mechanics).

## Requirements

Six deliverables.

### Deliverable A — the benchmark schema

Author or pick a JSON Schema with:

- 3+ levels of nesting.
- At least one array of objects.
- At least two enum fields (one short, one long: 2–5 vs. 20+ values).
- At least one conditional field (field X is required iff enum value Y is selected — expressible via `anyOf`).
- At least one numeric field with no bounds (to test the "schema allows too much" trap).
- At least one date or timestamp field (to test grammar vs. schema).

Example (adapt to your domain):

```python
# Pydantic equivalent (source of truth for the schema)
from pydantic import BaseModel, Field
from typing import Literal, Union

class CustomerAddress(BaseModel):
    line1: str
    line2: str | None = None
    city: str
    state: str
    postal_code: str
    country: Literal["US", "CA", "GB", "DE", "FR", "JP", "...20+ values..."]

class LineItem(BaseModel):
    sku: str
    quantity: int
    unit_price_usd: float

class Order(BaseModel):
    order_id: str
    customer_email: str
    customer_name: str
    shipping_address: CustomerAddress
    items: list[LineItem]
    status: Literal["pending", "confirmed", "shipped", "delivered", "cancelled"]
    cancellation_reason: str | None = None  # required iff status == cancelled
    created_at: str   # ISO 8601 date
    total_usd: float
```

Derive:

- `Order.model_json_schema()` as the JSON Schema.
- The same shape as a Lark / EBNF grammar (for the grammar-mode test; Outlines accepts a Lark grammar as a string).

### Deliverable B — the four-mode benchmark

For each mode, run 200 prompts that ask the model to generate an `Order`-shaped output (prompt variation: different synthetic customer names, items, states; keep complexity similar).

Modes:

1. **Post-hoc validate.** Prompt-only JSON instruction, no `response_format`. Parse + validate on the client side; retry up to 2× on failure.
2. **JSON mode.** `response_format: {"type": "json_object"}`. Parse + validate on the client side; retry up to 2× on schema failure.
3. **Strict JSON Schema.** `response_format: {"type": "json_schema", "json_schema": {"name": "order", "schema": Order.model_json_schema(), "strict": true}}`.
4. **Full grammar.** The engine's grammar mode (vLLM's `guided_grammar`, SGLang's `ebnf`, or Outlines directly) with a Lark grammar that matches the Order shape.

Capture per request:

- TTFT and TPOT (chapter 02 of module 110's panels).
- Total latency.
- Output tokens.
- Output bytes.
- Validation outcome (pass / schema violation / parse failure).
- Retry count (modes 1 and 2 only).

### Deliverable C — the per-mode correctness and latency table

Collate into a Markdown table:

```
| Mode                 | Valid-on-first-try | Valid-after-retry | TTFT P50 | TTFT P99 | TPOT mean | Latency tax vs. no-constraint |
|----------------------|--------------------|-------------------|----------|----------|-----------|-------------------------------|
| Post-hoc validate    | 0.84               | 0.97              | 180ms    | 420ms    | 28ms      | ~1.0 (no constraint)          |
| JSON mode            | 0.93               | 0.99              | 185ms    | 435ms    | 29ms      | ~1.03                         |
| Strict JSON Schema   | 1.00               | 1.00              | 310ms    | 620ms    | 32ms      | ~1.14                         |
| Full grammar         | 1.00               | 1.00              | 290ms    | 580ms    | 31ms      | ~1.10                         |
```

(Numbers above are illustrative; measure your own.)

Discipline:

- Report P50 and P99 for every latency number; averages mask the strict-mode tail from FSM compilation on first request per unique schema.
- Separate cold (first request with this schema) from warm (subsequent requests) for strict and grammar modes; chapter 03's FSM compilation cost shows up only on cold.
- Record the FSM compilation time as a separate number.

### Deliverable D — the per-request routing policy

Author a policy for your fleet: given a request, which mode do we run under?

Example:

```
Routing rules:
1. If the client sent response_format == json_schema with strict=true → strict mode (engine must support).
2. If the client sent response_format == json_object → JSON mode (fallback to post-hoc if engine lacks).
3. If the client sent response_format not set / type: text → no constraint.
4. If the schema is tighter than engine's strict mode supports (e.g., requires anyOf OpenAI strict rejects) → fall back to full grammar, emit X-Inference-Fallback: grammar header.
5. If grammar compilation exceeds 2000 ms → return 400 with "schema too complex"; direct the client to simplify.
```

The policy document goes into the ingress docs; chapter 06's ingress enforces it.

### Deliverable E — the trap demonstration

Reproduce the three traps from chapter 03:

1. **Strict-that-isn't.** Pick an engine that advertises strict mode but may fall back to post-hoc. Submit a deliberately hard-to-satisfy schema (short enum, long response). Measure retry / fallback behaviour. Document.
2. **Tokenizer mismatch.** If possible, run the grammar against a different tokenizer than the model's canonical one. Observe drift. (If your engine doesn't allow this, document the test case for a future version.)
3. **Schema allows the string the model can't produce.** Author a schema requiring a 6-significant-digit float; measure TPOT inflation vs. the default schema.

### Deliverable F — cache-and-preload for FSM compilation

Measure FSM compilation cost for your schema; implement (or verify the engine implements) FSM caching so the second request with the same schema skips compilation. For a fleet that sees `K` unique schemas:

- How many unique schemas do you expect per day? (Per-tenant schemas; shared agent schemas.)
- How long does the FSM compile for the median / tail schema?
- What's the fleet's warm-up budget (preload common schemas on startup)?

Publish the caching-and-preload plan.

## Starter guidance

- **Start with the simplest schema, one mode.** Get a single request through strict JSON Schema; verify the output validates; then expand to the four-mode matrix.
- **Use the same model for every mode.** Comparing modes against different models confounds the measurement.
- **Measure cold vs. warm.** Strict mode's FSM compilation is the "surprise" latency; don't average it with warm steady-state.
- **Record raw outputs.** A valid output per mode is also a debugging artefact; the raw bytes show whether a mode truly constrains the output or occasionally slips.
- **Pin the engine version and the backend version.** Outlines and XGrammar have their own release cadences; the numbers you measure are version-specific.

## Fallbacks for constrained environments

- **Only one enforcement mode available.** Report the single mode; note the fallback modes as "not implemented in this engine release." The table is still valuable as a baseline.
- **No grammar-mode support.** Skip the grammar comparison; measure the three modes you have.
- **Expensive to run 200 prompts per mode.** Scale down to 50 per mode; the latency numbers hold at lower sample size, the correctness numbers are noisier.
- **No access to FSM compilation time.** Infer from cold vs. warm TTFT; the delta is approximately the compile time.

## Acceptance criteria

1. **Deliverable A** has a schema meeting all six complexity criteria, in both JSON Schema and Pydantic (and Lark if grammar mode is tested).
2. **Deliverable B** ran ≥ 50 prompts per mode; captured per-request latency and validation outcome.
3. **Deliverable C** is the per-mode table with P50 and P99 for TTFT, separate cold and warm numbers, FSM compile time called out.
4. **Deliverable D** is the per-request routing policy, specific enough to code into the ingress.
5. **Deliverable E** demonstrates at least two of the three traps; the third is either reproduced or documented as unreachable in your stack.
6. **Deliverable F** has the caching-and-preload plan sized against your fleet's schema cardinality.

## Common failure modes

- **Post-hoc retry hides the real failure rate.** If you count "succeeded after 2 retries" as a pass, the latency number lies and the comparison with strict mode is unfair. Report valid-on-first-try separately from valid-after-retry.
- **Schema not compliant with strict-mode subset.** OpenAI's strict mode rejects schemas without `additionalProperties: false`, without `required` listing every property, or with unsupported keywords. The engine either rejects (good) or silently falls back to post-hoc (trap 1). Verify.
- **Benchmarks run against a cold fleet.** FSM compilation on first request shows up in the TTFT numbers and obscures the steady-state comparison. Warm up before measuring.
- **Numeric tokens that exceed the sampler's capability.** A schema allowing arbitrary floats requires the model to emit many digits; some model-tokenizer combinations struggle. TPOT inflates. Trap 3.
- **One schema isn't enough.** The trade-off is schema-dependent. Run a second, simpler schema as a comparison point; publish both.

## Keep it

The schema catalogue, the four-mode benchmark, the per-mode table, the routing policy, the FSM cache plan — these are the fleet's structured-output artefact. Exercise 04 (tool calling) reuses the strict-schema enforcement for tool-call arguments; exercise 06 (ingress) enforces the routing policy; the compatibility harness from exercise 01 gets the per-mode cases.
