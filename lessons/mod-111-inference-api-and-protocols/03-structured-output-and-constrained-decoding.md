# Chapter 03 — Structured Output and Constrained Decoding: JSON Schema, Pydantic, Outlines, XGrammar

> *"Please return JSON"* in the system prompt is a hope. Enforcing it at the sampler is a guarantee. The gap between hope and guarantee is three orders of magnitude of invalid-output rate, four distinct enforcement modes, two sampler-level backends you may deploy, and one latency tax you may be surprised by. This chapter writes the four modes, the two backends, the trade-off, and the per-request routing.

## Motivation

A product team ships an agent that calls the inference API with a prompt instructing the model to return JSON of a specific shape. In testing, 97 % of responses parse. The product ships. In production, 3 % of responses fail to parse — a trailing comma, a double-escaped quote, a halucinated enum value, a nested object with an unexpected key. The agent's downstream pipeline treats a parse failure as an error and retries. Retries cost tokens, cost latency, cost dollars, and sometimes cascade into timeouts that the client surfaces to the user.

The problem is a protocol choice: the API is enforcing structure *in English prose in the system prompt* rather than *at the sampler*. A model is a probability distribution over tokens; a sampler is the thing that converts the distribution into a next token. If the sampler is unconstrained, it will occasionally sample a token that produces invalid JSON. If the sampler is constrained to the subset of tokens consistent with the schema, invalid outputs are *impossible by construction*.

Four enforcement modes exist, in increasing strength:

1. **Post-hoc validate** — model generates freely; the ingress parses and validates; on failure, retry.
2. **JSON mode** — model is instructed and the sampler is lightly constrained to produce syntactically valid JSON (balanced braces, quoted strings, no trailing commas) but not schema-conformant.
3. **Strict JSON Schema** — the sampler is constrained to produce output that validates against a specific JSON Schema at every token (every emitted token keeps the output in the set of "could still become schema-valid").
4. **Full grammar** — the sampler is constrained to produce output that matches a specified context-free grammar (EBNF, Lark, or regex-shaped). JSON Schema is a special case.

The two production-grade sampler-level backends you may deploy:

- **[Outlines](https://github.com/dottxt-ai/outlines)** — a library that compiles a schema or grammar into a finite-state machine, intersects the automaton with the model's logit distribution, and masks disallowed tokens. Integrates with vLLM, SGLang, Transformers, and others.
- **[XGrammar](https://github.com/mlc-ai/xgrammar)** — a GPU-accelerated grammar engine from the MLC team; targets lower per-token overhead for grammar-constrained decoding; integrates with vLLM and MLC-LLM.

By the end of this chapter you can:

- Decide between the four modes per request shape.
- Author a JSON Schema (or Pydantic model) that is enforceable at the sampler.
- Wire Outlines or XGrammar as the backend for sampler-level constraints.
- Measure and reason about the per-token latency tax of each mode.
- Avoid the three common correctness traps that make a "constrained" output not actually constrained.

## The four enforcement modes — side by side

| Mode | Request shape | Enforcement site | Correctness guarantee | Latency tax |
|---|---|---|---|---|
| Post-hoc validate | plain prompt, no `response_format` | ingress parser | none; retry on failure | retry cost on failure (~2× on 3 % failure rate) |
| JSON mode | `response_format: {"type": "json_object"}` | sampler (grammar for syntactic JSON) | syntactically valid JSON; no schema conformance | small (~1-5 % per-token in modern engines) |
| Strict JSON Schema | `response_format: {"type": "json_schema", "json_schema": {...}, "strict": true}` | sampler (schema compiled to FSM) | validates against schema | moderate (~5-20 % per-token depending on schema complexity) |
| Full grammar | engine-specific (`guided_grammar`, `ebnf`, `regex`, ...) | sampler (grammar compiled to FSM) | matches grammar exactly | moderate (~5-20 % per-token depending on grammar complexity) |

Three framing points before the modes:

- **The failure mode of "post-hoc + retry" is correlation.** The requests that fail are not independent — a tricky schema fails on 10 % of requests, and the same request often fails on retry. The expected number of retries is not 1 / (1 - failure_rate); it is higher, because the retries are correlated. A 10 % post-hoc failure rate can produce a 15 % tail-latency inflation.
- **"JSON mode" is weaker than it sounds.** The output is syntactically valid JSON — braces balance, strings are quoted, no trailing commas — but the shape is whatever the model generated. The schema conformance comes from the system prompt, not the sampler. Expect 1-3 % schema violations even with careful prompting.
- **Strict schema and full grammar are the same mechanism, different front end.** Both compile a specification into a finite-state automaton; both mask the sampler's logits per-token. Schema is more approachable (JSON Schema is a standard); grammar is more expressive (EBNF can describe things JSON Schema cannot — a date in a specific format, a programming-language snippet, a chess move in SAN).

## JSON Schema as the enforceable contract

[JSON Schema](https://json-schema.org/) draft 2020-12 is the de-facto standard. OpenAI's structured-outputs feature accepts a subset (see [OpenAI's structured outputs documentation](https://platform.openai.com/docs/guides/structured-outputs) for the exact subset — the authoritative reference drifts with releases). The subset constraints that typically bind:

- Every property must appear in `required`.
- `additionalProperties: false` is mandatory on every object.
- The depth of nesting is bounded (OpenAI's limit as of writing is 5 levels; verify current).
- Only `const`, `enum`, `type`, `properties`, `items`, `additionalProperties`, `required`, `anyOf` are supported in strict mode (and the exact support set drifts).

<!-- needs-research: OpenAI's strict-mode subset of JSON Schema has evolved; verify the current support set against the structured outputs documentation. Engines that advertise OpenAI-compatible strict mode may support a different subset. -->

A compliant strict-mode schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["city", "temperature_c", "conditions"],
  "properties": {
    "city": {"type": "string"},
    "temperature_c": {"type": "number"},
    "conditions": {
      "type": "string",
      "enum": ["sunny", "cloudy", "rainy", "snowy"]
    }
  }
}
```

A non-compliant schema (would be rejected in strict mode by OpenAI; engines may silently fall back to looser enforcement):

```json
{
  "type": "object",
  "properties": {
    "city": {"type": "string"},
    "temperature_c": {"type": "number", "minimum": -100, "maximum": 100},
    "conditions": {"type": "string"}
  }
}
```

The second schema fails three ways: `additionalProperties` not set to `false`, no `required` array, and `minimum`/`maximum` constraints on numbers (not supported in strict mode by all engines).

## Pydantic as schema source

[Pydantic v2](https://docs.pydantic.dev/latest/) models produce JSON Schema via `model.model_json_schema()`. A pattern:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    conditions: Literal["sunny", "cloudy", "rainy", "snowy"]

schema = WeatherReport.model_json_schema()
# {'properties': {...}, 'required': [...], 'type': 'object', ...}
```

Two disciplines:

- **Use `Literal` for enums, not `Enum`.** Pydantic emits `Literal` as a schema `enum`; `Enum` sometimes emits as a `$ref` which some engines reject in strict mode.
- **Set the schema to `additionalProperties: false` explicitly.** Pydantic v2 defaults to `False` for closed models but verify by printing the schema.

The output of `model_json_schema()` is the body of the `json_schema.schema` field in the request:

```python
request = {
    "model": "llama3.1-70b-instruct",
    "messages": [...],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_report",
            "schema": WeatherReport.model_json_schema(),
            "strict": True,
        },
    },
}
```

The client-side payoff: on response, `WeatherReport.model_validate_json(response_content)` parses the string and gets a typed object with Pydantic's error reporting on the (now impossible under strict mode) validation failures.

## Outlines — the schema → FSM compiler

Outlines (<https://github.com/dottxt-ai/outlines>) compiles a schema or grammar into a finite-state machine. At each decode step, the engine's sampler computes next-token logits as usual; Outlines then masks the logits for tokens that would push the output out of the FSM's reachable states. The masked distribution is sampled.

The pattern in a vLLM integration:

```python
# vLLM has native Outlines integration; the invocation shape varies by release.
# Representative:
request_params = {
    "model": "llama3.1-70b-instruct",
    "prompt": "...",
    "guided_json": WeatherReport.model_json_schema(),
    "guided_decoding_backend": "outlines",
}
```

Three correctness properties Outlines provides:

- **The output always validates.** The FSM is derived from the schema; a sample that violates the schema is impossible by construction (modulo FSM-compilation bugs; see traps below).
- **The output always terminates.** Outlines bounds generation to states that can reach a schema-complete terminal.
- **The output respects tokenizer boundaries.** The FSM is defined over the model's tokenizer, not over characters; a schema-reachable byte sequence that doesn't correspond to any token sequence is handled correctly.

Three failure modes:

- **FSM compilation is slow for large schemas.** A deeply nested schema with many enums and `anyOf` branches can take seconds to compile. Cache the compiled FSM per schema (hash the schema → FSM blob). The engine should expose a hook; verify.
- **Unicode / tokenizer mismatch.** A schema that permits characters the tokenizer can't losslessly encode produces an FSM with unreachable states. Rare in practice but a real bug source. If the schema is UTF-8 and the tokenizer is a modern BPE, this doesn't bite.
- **FSM state blowup.** A regex-like constraint with wide quantifiers (`.{1,10000}`) produces a large FSM. Keep bounds tight.

## XGrammar — the GPU-accelerated grammar engine

XGrammar (<https://github.com/mlc-ai/xgrammar>) is a grammar engine designed for lower per-token overhead than CPU-side FSM intersection. It compiles grammars to GPU-resident masks and applies them in the sampler kernel.

The pattern (vLLM integration):

```python
request_params = {
    "model": "llama3.1-70b-instruct",
    "prompt": "...",
    "guided_json": WeatherReport.model_json_schema(),
    "guided_decoding_backend": "xgrammar",
}
```

Three framing points:

- **XGrammar generally has lower per-token overhead than Outlines**, especially for complex grammars, because mask computation is on-GPU. The actual delta depends on model size, grammar complexity, and hardware; benchmark on your fleet before deciding.
- **XGrammar's grammar surface is EBNF-shaped and a strict superset of what JSON Schema expresses.** For a workload that already needs grammar enforcement (not just schema), XGrammar is the natural choice.
- **Engine support is uneven.** vLLM supports both Outlines and XGrammar; SGLang supports its own regex/EBNF; TRT-LLM has its own path (verify against release). The choice of backend is partially an engine choice.

<!-- needs-research: Outlines vs. XGrammar per-token overhead benchmarks are workload- and release-dependent. Measure against your pinned engine and representative schemas rather than citing a cross-fleet ratio. -->

## Full grammar — when schema isn't enough

JSON Schema expresses "an object with these fields." It does not express:

- "A YYYY-MM-DD date." (regex works; schema's `format: "date"` is not sampler-enforceable in most backends.)
- "A valid SQL query against this schema." (requires a context-free grammar.)
- "A chess move in Standard Algebraic Notation." (grammar.)
- "A Python function body matching a signature." (grammar.)

For these, full-grammar modes:

- **vLLM:** `guided_regex`, `guided_grammar` (Lark syntax).
- **SGLang:** `regex`, `ebnf` on the request.
- **TRT-LLM / Triton:** engine-specific; verify.
- **Outlines / XGrammar:** grammars native.

A worked grammar for a date:

```python
grammar = r"""
start: date
date: year "-" month "-" day
year: /\d{4}/
month: /0[1-9]|1[0-2]/
day: /0[1-9]|[12][0-9]|3[01]/
"""

request_params = {
    "prompt": "The date of signing was ",
    "guided_grammar": grammar,
    "max_tokens": 10,
}
```

The grammar is tight — no leading / trailing whitespace, no alternate separators, no invalid calendar dates past the month-day range. The sampler cannot produce `02-30-2024`, `2024/01/15`, or `the fifteenth of January`.

## The latency tax

Constrained decoding is not free. Three overhead sources:

- **FSM compilation.** Pays once per unique schema; cache the compiled FSM. If the fleet sees `K` unique schemas and compiles each in `T` seconds, the fleet's warm-up budget needs `K × T`. Measure, cache, pin.
- **Per-token mask computation.** For each token, compute the set of allowed tokens for the current FSM state and mask the logits. On CPU (Outlines), this is a dictionary lookup and a mask apply; on GPU (XGrammar), the mask is pre-computed and applied in the sampler kernel. The per-token cost is typically single-digit microseconds; it shows up as a 1-20 % TPOT tax depending on schema complexity and model size.
- **Sampler path.** The sampler kernel is modified to apply the mask. For modern engines, this is a fused kernel path; for older ones, it is a Python-side mask-then-resample loop. The latter can be a 2-3x TPOT tax; the former is <5 %.

An inference engineer wires this up and measures. The compatibility harness from chapter 01 gets a case per enforcement mode; the Pareto curve of exercise 05 (module 110) gets a sweep.

## Three correctness traps

### Trap 1 — "strict" that falls back to post-hoc

Some engines advertise strict JSON Schema support but under the hood run post-hoc validate + retry. The symptom: the latency tax is zero (or negative — faster than JSON mode) but the per-request latency distribution has a heavy tail (the retries). Verify by sending a schema that is deliberately hard to satisfy (an enum of three values on a long response); a true strict mode produces a valid output every time; a post-hoc mode produces either a long tail or occasional parse failures.

### Trap 2 — tokenizer mismatch

The FSM is defined over tokens, not characters. If the FSM is built against tokenizer A and the engine runs tokenizer B (common when a quantized checkpoint uses a slightly different tokenizer), the mask applies to the wrong token set and the output drifts. Verify the backend uses the same tokenizer as the model at inference time.

### Trap 3 — the schema allows the string the model can't produce

A schema that permits `"temperature_c": -273.15` (physically plausible, but a float with 6 significant digits may require tokens the FSM reaches only via a narrow path) can cause the sampler to backtrack repeatedly. The symptom: TPOT inflates by 5-10× on a request whose schema contains a wide numeric range. Tighten the schema or use grammar.

## Per-request routing — the production policy

An inference fleet rarely needs *every* request to run under strict schema. The policy is per-request:

- **If the client sends `response_format: {"type": "json_schema", ...}`,** run strict mode; measure the latency tax; expose it in response headers for client awareness.
- **If the client sends `response_format: {"type": "json_object"}`,** run JSON mode (lighter grammar); fall back to post-hoc for engines that don't natively support JSON mode.
- **If the client sends neither,** no constraint; the response is free-form.

The ingress enforces the routing. If an engine does not support a requested mode, the ingress either rejects (`400 Bad Request` with an error that names the unsupported mode) or falls back with a response header that documents the fallback. The compatibility harness tests both paths.

### A note on compound responses

A chat response that includes both reasoning / commentary *and* a structured block is common: "Here's my reasoning: ... And here's the data: {...}." Strict schema on the whole response rejects the reasoning. Two patterns:

- **Separate turns.** First turn is free-form reasoning; second turn is strict-schema data.
- **Schema wrapper.** The schema has a `reasoning` field (free-form string) and a `data` field (strict sub-schema). The whole output is the wrapper object.

The second is operationally simpler; it is the pattern most "reasoning + answer" agents converge on.

## Three failure modes

### Schema compiles slowly, warm-up budget blown

A fleet with many unique schemas sees FSM compilation on every unique first request. The symptom: cold-request TTFT P99 is high; warm-request TTFT P99 is fine. Cache the compiled FSM at the ingress or engine; preload known schemas at startup; set a schema-compilation budget alert (e.g., P99 schema compile > 500 ms is a page).

### Output invalid under "strict" mode

The output doesn't validate. Either the engine is running post-hoc (trap 1), the FSM has a bug (rare), or the schema is itself malformed. Reproduce with a minimal case; file against the engine; add a case to the compatibility harness.

### TPOT inflates on complex schemas

Expected under strict mode; measure per-schema and surface in response headers (`X-Inference-Structured-Overhead-Ms: 123`). Clients can then opt out of strict mode when the schema is tight-enough that post-hoc is cheap.

## Summary

- Four enforcement modes exist, from weakest (post-hoc validate) to strongest (full grammar). Pick per request based on correctness need and latency tolerance.
- JSON Schema is the enforceable contract; strict mode constrains the sampler per-token so invalid output is impossible by construction.
- Pydantic produces JSON Schema; use `Literal` for enums, set `additionalProperties: false`.
- Outlines compiles schema/grammar to a CPU-side FSM; XGrammar is a GPU-accelerated grammar engine with lower per-token overhead; engine support varies.
- Full grammar (regex / EBNF) is the right tool when the output isn't representable in strict JSON Schema (dates, SQL, code, chess moves).
- The latency tax is non-trivial — FSM compilation, per-token mask, sampler path. Measure on your fleet; cache compiled FSMs; expose the overhead in response headers.
- Three correctness traps: strict-that-isn't, tokenizer mismatch, schema-string-the-model-can't-produce.
- Per-request routing based on the client's `response_format` is the production pattern; chapter 06's ingress enforces it.
