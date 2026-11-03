# Chapter 01 — The OpenAI-Compatible API Surface: Endpoints, Shapes, Compliance-Drift Tax

> *OpenAI-compatible* is the phrase that sells the engine. It is also the phrase that pages the on-call engineer at 2 AM when a client's parallel tool call works on provider A and does not work on provider B, same request body, same model, different implementation. This chapter names the compatible surface, the parts that drift, and the compatibility gate that keeps the drift documented rather than discovered in production.

## Motivation

Every serving engine worth deploying announces its API as "OpenAI-compatible." The phrase is useful shorthand: it means the client SDKs (`openai-python`, `openai-node`, LangChain's OpenAI adapters, LlamaIndex's OpenAI adapters, the Vercel AI SDK's OpenAI provider) point at the engine's base URL with an API key and a model name, and most requests work. That portability is real and operationally valuable — a product team can switch between a hosted model, a self-hosted vLLM replica, and a self-hosted SGLang replica without rewriting their client code.

The phrase is also a lie of omission. Three asymmetries make *compatible* a sliding target:

- **OpenAI evolves its own API.** The chat-completions endpoint added structured outputs in 2024, strict JSON mode in 2024, parallel tool calls in 2023, and the `/v1/responses` endpoint in 2024 as a chat-completions successor for stateful / tool-rich / multimodal workloads. Every engine implements some subset of some version. Which subset of which version is a release-note question, not a brand-promise.
- **Engines add their own extensions.** vLLM accepts `guided_json` and `guided_regex` as sampling parameters; SGLang accepts `regex` and `ebnf` on the request; TRT-LLM's Triton backend accepts `response_format` with its own vocabulary. Each extension is a reason to deploy that engine; each extension is a reason a cross-engine client breaks.
- **The reference is under-specified in the places that matter.** The finish-reason enum, the streamed-delta shape for a tool call, the exact bytes emitted when the model refuses, the usage-token object on streamed responses — these are described in English prose, not in a machine-checkable schema. Two implementors read the same prose and produce two different byte sequences. The client that assumes the first is surprised by the second.

An inference engineer does not get to make the drift go away. The engineer gets to write it down. By the end of this chapter you can:

- Name every endpoint in the OpenAI-compatible surface and its primary request / response shape.
- Enumerate the top-ten drift points observed across major engines.
- Author a compatibility test harness that catches drift against pinned engine versions.
- Decide which drifts to normalise in a compatibility shim and which to expose to clients with documentation.

## The endpoint set

The OpenAI-compatible surface is a set of HTTP endpoints, each with a canonical request and response shape. The authoritative reference is the [OpenAI API reference](https://platform.openai.com/docs/api-reference). This chapter cites the shape; it does not recreate the reference.

| Endpoint | Purpose | Request body root | Response body root | Streamed shape |
|---|---|---|---|---|
| `POST /v1/chat/completions` | Multi-turn chat with system / user / assistant / tool roles | `model`, `messages`, `stream`, `tools`, `response_format`, `temperature`, `top_p`, `max_tokens` / `max_completion_tokens` | `id`, `object="chat.completion"`, `choices[].message`, `usage` | `data: {...chat.completion.chunk...}\n\n` SSE stream |
| `POST /v1/completions` | Legacy text completion, single prompt | `model`, `prompt`, `stream`, `max_tokens`, sampling params | `id`, `object="text_completion"`, `choices[].text`, `usage` | `data: {...text_completion...}\n\n` SSE stream |
| `POST /v1/responses` | Successor to chat completions; stateful; richer tool / multimodal surface | `model`, `input`, `instructions`, `tools`, `stream`, `store`, `previous_response_id` | `id`, `object="response"`, `output[]`, `usage` | Typed event stream (`response.created`, `response.output_text.delta`, `response.completed`, ...) |
| `POST /v1/embeddings` | Vector embeddings for a text input | `model`, `input`, `encoding_format` | `object="list"`, `data[].embedding`, `usage` | n/a (non-streaming) |
| `GET /v1/models` | List available models for authorisation key | — | `object="list"`, `data[].id` | n/a |
| `POST /v1/images/generations` | Image generation (where engine supports it) | `model`, `prompt`, `size`, `n`, `response_format` | `data[].url` or `data[].b64_json` | n/a |
| `POST /v1/audio/transcriptions` | Speech-to-text | multipart `file`, `model`, `response_format` | `text` | optional typed event stream |
| `POST /v1/audio/speech` | Text-to-speech | `model`, `input`, `voice`, `response_format` | audio bytes with `Content-Type: audio/*` | optional chunked audio stream |

Two framing points:

- **Not every engine implements every endpoint.** vLLM and SGLang cover chat / completions / embeddings and have partial responses coverage. Triton + TRT-LLM covers chat / completions and relies on the application layer for the rest. Hosted providers cover all of them. Chapter 05 adds MCP as a parallel surface.
- **The chat-completions endpoint is the gravity well.** 90 % of the client traffic a general-purpose inference fleet sees is `POST /v1/chat/completions`, often with streaming and tools. The other endpoints matter, but a compatibility defect on chat-completions is a fleet-visible outage; a defect on `/v1/images` is a product-feature regression.

## The canonical chat-completions shape

Minimal request:

```http
POST /v1/chat/completions HTTP/1.1
Host: inference.example.internal
Authorization: Bearer sk-<redacted>
Content-Type: application/json

{
  "model": "llama3.1-70b-instruct",
  "messages": [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user",   "content": "What is 2+2?"}
  ],
  "temperature": 0.0,
  "max_tokens": 32,
  "stream": false
}
```

Minimal non-streaming response:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1762128000,
  "model": "llama3.1-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "4."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 2,
    "total_tokens": 24
  }
}
```

Minimal streaming response (SSE — chapter 02 goes deep on the framing):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

The `data: [DONE]` sentinel is an OpenAI convention, not an SSE convention; the WHATWG SSE spec has no notion of "end of stream" beyond connection close. Downstream clients commonly key on the sentinel. A compatible engine must emit it; chapter 02 revisits.

## The compliance-drift top ten

Drift is the delta between what the OpenAI reference prose says and what a specific engine's implementation does. The following ten have been the reliable sources of client breakage. The exact set drifts with releases; the categories are stable.

<!-- needs-research: specific parameter names and enum values drift across vLLM / SGLang / TRT-LLM releases. Verify the current behaviour against each engine's release notes and `/v1/chat/completions` reference before writing a shim. -->

### 1. `max_tokens` vs. `max_completion_tokens`

OpenAI deprecated `max_tokens` on reasoning-capable models in favour of `max_completion_tokens` (the former caps only non-reasoning output on `o1` / `o3` lines; the latter caps total generation including reasoning). An engine must decide: accept both, prefer the new one, warn on the old one. Most engines accept both; a few accept only one. A client that sends only the old field against an engine that accepts only the new one gets unbounded generation up to the model's context limit.

### 2. Finish-reason enum

The reference lists `stop`, `length`, `content_filter`, `tool_calls`, `function_call` (deprecated). Engines add their own: `null` during streaming before the final chunk, `error` on an engine-side abort (non-standard), `truncated` on some implementations (non-standard), `abort` on a client-initiated cancellation. A compatibility shim either normalises to the reference set or documents the extensions.

### 3. `tool_calls` vs. `function_call`

The old function-calling protocol (single function) was `choices[0].message.function_call = {name, arguments}`. The new protocol (parallel tools) is `choices[0].message.tool_calls = [{id, type: "function", function: {name, arguments}}, ...]`. Chat completions support both for backward compatibility; the `/v1/responses` endpoint supports only the new shape. Engines vary in which they emit by default.

### 4. Streamed tool-call deltas

Streaming a tool call means emitting incremental `tool_calls[].function.arguments` fragments as JSON strings. The reference shape:

```
delta: {"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}
delta: {"tool_calls":[{"index":0,"function":{"arguments":"{\"city"}}]}
delta: {"tool_calls":[{"index":0,"function":{"arguments":"\":\"SF\"}"}}]}
```

The drift: whether `id`, `type`, and `name` appear only on the first delta or on every delta; whether `arguments` is emitted as a single blob at the end rather than incrementally; whether multiple parallel calls interleave or strictly serialise. A client that assembles arguments by string-concatenating deltas breaks when an engine emits a blob at the end.

### 5. `response_format` enforcement mode

The reference distinguishes:

- `response_format: {"type": "text"}` — no constraint.
- `response_format: {"type": "json_object"}` — the model is instructed to produce JSON; validation is post-hoc and best-effort.
- `response_format: {"type": "json_schema", "json_schema": {...}}` — the model's sampler is constrained to produce output that validates against the schema (strict mode).

Engines vary in which modes they honour, how they enforce `json_schema` (post-hoc validate + retry vs. sampler-level constraint vs. grammar), and what they do when the schema is invalid. Chapter 03 teaches the modes; this chapter notes that the request body is identical across modes and the engine's advertised support list is where you check.

### 6. `logprobs` and `top_logprobs`

The reference exposes per-token log-probabilities. Engines vary on:

- Whether logprobs are available on streamed responses (some only on non-streamed).
- Whether `top_logprobs` can exceed 5 / 10 / 20 (limit is engine- and model-dependent).
- Whether the log-probability values are in natural log or log-base-2 (should be natural log; some older engines ship log-base-2).
- Whether the tokenizer used for the token ids matches the model's canonical tokenizer (chapter 02 of module 102 is relevant here).

### 7. Multimodal content parts

Chat messages with image inputs use:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What's in this image?"},
  {"type": "image_url", "image_url": {"url": "https://..."}}
]}
```

Drift: whether the engine accepts base64 inline (`data:image/png;base64,...`) in addition to HTTPS URLs, whether it supports `image_url` with `detail: "low" | "high" | "auto"`, whether it supports video / audio content parts, and whether it validates the image synchronously or lazily. A client that sends base64 against an engine that only accepts URLs gets a 400.

### 8. `stream_options.include_usage`

Classic chat-completions streaming did not emit the final `usage` object (prompt / completion / total tokens); OpenAI later added `stream_options: {"include_usage": true}` which causes the final SSE chunk to carry `usage`. Engines adopted this at different times. A client that depends on streamed usage for billing against an engine that doesn't emit it has to compute tokens itself.

### 9. Error object shape

The reference error:

```json
{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}
```

The HTTP status: `400` for bad request, `401` for auth, `403` for forbidden, `404` for unknown model, `429` for rate limit (with a `Retry-After` header per chapter 06), `500` / `502` / `503` for server-side. Engines sometimes emit non-standard shapes (`{"detail": "..."}` from FastAPI defaults, `{"error_message": "..."}` from TRT-LLM / Triton). A compatibility shim normalises.

### 10. `seed` and determinism

The reference's `seed` parameter plus `system_fingerprint` is a best-effort determinism contract: same seed, same fingerprint, same model → same output. Engines vary on whether they honour seed, whether they emit `system_fingerprint`, and whether a batched request shares seed state with batch-mates (which can break determinism under continuous batching). A test that assumes deterministic output across engines fails.

## The compatibility test harness

The drift is unavoidable; discovering it in production is not. Every inference fleet should ship a compatibility harness with the engine upgrade:

```python
# pseudo-structure
CASES = [
    ("basic_nonstream",     basic_nonstream_request, basic_nonstream_assertions),
    ("basic_stream",        basic_stream_request,    basic_stream_assertions),
    ("tool_call_single",    tool_call_single_req,    tool_call_single_assertions),
    ("tool_call_parallel",  tool_call_parallel_req,  tool_call_parallel_assertions),
    ("tool_call_streaming", tool_call_streaming_req, tool_call_streaming_assertions),
    ("json_schema_strict",  json_schema_strict_req,  json_schema_strict_assertions),
    ("multimodal_image",    multimodal_image_req,    multimodal_image_assertions),
    ("error_rate_limit",    rate_limit_req,          rate_limit_assertions),
    # ... one case per drift point above
]

for name, req, assertions in CASES:
    for engine in ("vllm", "sglang", "trt_llm"):
        resp = send(engine, req)
        report = assertions(resp)
        record(engine, name, report)
```

Discipline:

- **Pin the engine version.** The harness runs against a specific engine release, not "latest." Drift is a release-note question.
- **One case per drift point.** When a drift is discovered in production, the harness gets a new case so the next upgrade catches it.
- **Capture the full response, not just assert.** The record (full headers, full body, byte-for-byte SSE frames) is the artefact a regression triage reads from.
- **Run against an external reference when possible.** A case that includes the OpenAI-hosted API as a reference target lets the drift be measured against the authoritative implementation, not just inferred from the prose.

The harness lives in-repo next to the serving config; chapter 07 formalises the hand-off from this harness to the application-layer peer's integration tests.

## The compatibility shim — when to normalise, when to document

An ingress sits between the client and the engine (chapter 06 wires it). The shim at the ingress can either:

- **Normalise drift** — translate the request on ingress, translate the response on egress, present a single canonical shape to clients. Works well for stable drifts (error-object shape, `data: [DONE]` sentinel, `max_tokens` aliasing). Breaks subtle semantics (if the engine's `finish_reason` carries an extension that the shim collapses, a client that would use that extension loses it).
- **Expose drift and document** — pass the engine's native response through, publish the drift catalogue, let clients key on the engine identity (via the `model` string or a header). Works well for semantically-rich drifts (logprobs resolution, streamed tool-call shape, structured-output enforcement mode).

Rule of thumb: *normalise what is accidental, document what is essential.* An engine that misspells `finish_reason` is accidental drift — normalise. An engine that emits logprobs at higher resolution than the reference is essential drift — document and expose.

## Three failure modes

### "The client works on OpenAI but not on our vLLM"

Common cause: the client depends on a feature OpenAI emits by default that vLLM does not (streamed `usage`, `logprobs` on streamed responses, strict JSON schema). The harness has a case for each; the drift report says which feature is missing. The fix is one of: enable the feature on vLLM if supported, document the missing feature, or route that client to a different backend.

### "The client works on our vLLM but not on our SGLang"

Common cause: the client depends on a vLLM-specific extension (`guided_json`, `guided_regex`) that SGLang implements differently (`regex`, `ebnf`) or not at all. The harness catches it; chapter 03 teaches the schema-enforcement translation; chapter 07 cites the application-layer peer on which it falls to update clients.

### "The client worked yesterday and doesn't today"

Common cause: an engine upgrade changed a default (parallel tool-calls flag, stream-usage default, finish-reason set). The compatibility harness should have caught it; if it did not, the harness gets a new case. The engine upgrade's canary (module 113) should gate on the harness.

## Summary

- The OpenAI-compatible surface is a set of HTTP endpoints with canonical shapes; chat completions is 90 % of the traffic.
- "Compatible" is drift-prone: parameter aliases, enum extensions, streamed-chunk layouts, tool-call shapes, structured-output enforcement modes, and error-object shapes all vary across engines and across OpenAI releases.
- The top-ten drift list in this chapter is a starting inventory, not an exhaustive one; every fleet adds cases from production incident history.
- A compatibility test harness pinned to engine versions is the mechanism that keeps drift documented rather than discovered in production.
- The ingress shim either *normalises accidental drift* or *exposes and documents essential drift*; both are valid depending on the semantic weight of the field.
- Chapters 02–06 go deep on the sub-surfaces (streaming, structured output, tool calls, MCP, ingress discipline) each with their own drift points to catalogue.
