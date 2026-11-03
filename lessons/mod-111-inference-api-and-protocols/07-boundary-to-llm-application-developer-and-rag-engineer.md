# Chapter 07 — Boundary to `llm-application-developer` and `rag-engineer`: Server Implementation vs. Client Consumption

> The API surface this module writes is the contract two downstream tracks consume. The L25 application-developer peer builds the agent loop, the prompt engineering, the user-facing streaming UI. The L25 RAG-engineer peer builds retrieval pipelines that shape the request body this module's endpoints receive. Both tracks are *consumers* of this module's output. This chapter writes the fence.

## Motivation

Three recurring triage escalations go to the wrong owner:

1. **"The agent's JSON output is malformed."** The product team files against the inference platform. The platform team looks at the request: `response_format` is `{"type": "json_object"}` (JSON mode, not strict schema), the prompt says "return JSON." The engine is doing exactly what was asked. The right fix was for the application to send `{"type": "json_schema", "strict": true, ...}`. The L25 peer's choice, this module's documentation to cite.
2. **"The agent streams fine on OpenAI but flickers on our vLLM."** The platform team investigates: both emit SSE, both emit `data: [DONE]`, but vLLM emits the finish-reason enum differently. The agent's parser assumes OpenAI's set. The right fix is in the agent's parser or in the ingress compatibility shim (chapter 01); the inference engine is behaving correctly for vLLM.
3. **"The RAG prompt costs too many tokens."** The RAG team files against inference. The platform team notes that the per-request token count is correct for the prompt size — the RAG prompt is sending 8k tokens of context per request. The right fix is in the RAG pipeline (chunking strategy, retrieval top-k, context-reranking), not in the inference layer.

Each is a boundary error. This chapter writes the two-way table, the hand-off bundles, and the three recurring mistakes.

By the end of it you can:

- State what this module owns, what `llm-application-developer` owns, and what `rag-engineer` owns — in table form.
- Write the hand-off bundle to each peer (what fields, what trace, what artefact).
- Avoid the three common cross-boundary mistakes.

## The two-way boundary table

The two L25 peers are in the [AI Engineering family](https://github.com/ai-engineering-curriculum), at the application layer. They consume this module's HTTP surface.

| Concern | This module (`ai-infra-inference-engineer`, L35) | `llm-application-developer` (L25) | `rag-engineer` (L25) |
|---|---|---|---|
| HTTP endpoint implementation (chat, embed, images) | ✅ | — (consumer) | — (consumer) |
| SSE / WebSocket transport implementation | ✅ | — (consumer; chooses which) | — (consumer) |
| Compatibility harness across engines | ✅ | — (consumes drift report) | — |
| Streamed-chunk shape, finish-reason enum | ✅ emits | — (parses, must tolerate drift) | — |
| Client-disconnect → engine-cancel wiring | ✅ | — (consumer; sends disconnect by closing connection) | — |
| Back-pressure, token-flush cadence | ✅ | — | — |
| JSON Schema enforcement (post-hoc, JSON mode, strict, grammar) | ✅ (implements) | ✅ (chooses) — picks enforcement mode per workload | — (consumes for retrieved doc extraction, same choice) |
| Constrained-decoding backend (Outlines, XGrammar) | ✅ (wires) | — (uses) | — |
| Pydantic schema → request `response_format` | — (accepts) | ✅ (authors) | ✅ (authors for extraction schemas) |
| Tool declaration shape (description, parameters) | ✅ (accepts, emits results) | ✅ (authors tool list, writes agent loop) | — |
| Agent loop, tool execution | — | ✅ | — |
| `tool_choice` protocol value | ✅ (implements) | ✅ (chooses per request) | — |
| MCP server implementation (this fleet's inference endpoint) | ✅ | — | — |
| MCP client implementation (agent connects to many MCP servers) | — | ✅ | ✅ (if RAG exposes as MCP) |
| MCP capability discovery, session management | ✅ server-side | ✅ client-side | — |
| OAuth 2.1 Resource Server validation | ✅ | — | — |
| OAuth 2.1 client flow (authorization code + PKCE) | — | ✅ | — |
| Ingress auth (API key / OAuth / STS) | ✅ | — | — |
| Idempotency-Key handling | ✅ server-side | ✅ sends per request | — |
| Retry semantics on the client | — | ✅ | ✅ |
| `Retry-After`, `RateLimit`, `X-RateLimit-*` headers | ✅ emits | ✅ respects | — |
| Tenant quotas, admission policy | ✅ | — | — |
| Request-ID / `traceparent` propagation | ✅ ingress-side | ✅ client-side | ✅ |
| System-prompt engineering | — | ✅ | — |
| Prompt library, tool description quality | — | ✅ | — |
| Retrieval chunking, embedding model choice | — | — | ✅ |
| Top-k retrieval, reranking | — | — | ✅ |
| Context-window packing (prompt + retrieved docs) | — | ✅ (orchestrates) | ✅ (provides retrieved docs) |
| Per-request cost attribution | ✅ emits in audit log | ✅ consumes | ✅ consumes |
| User-visible streaming UI (token-by-token render) | — | ✅ | — |
| Agent evaluation / offline testing | — | ✅ | ✅ |

The split is blunt: *this module owns the protocol and the server; the L25 peers own the application semantics and the client*.

## The hand-off to `llm-application-developer`

### When to hand off

- The engine returns valid JSON per the request's enforcement mode but the agent's downstream parser rejects — the issue is the client's parser, not the engine's output.
- The agent's tool-choice protocol value is `"auto"` and the model is not calling the tool as often as expected — the model's calling behaviour is prompt / tool-description quality, an application concern.
- The streaming UI "flickers" (the client's render buffer vs. the engine's per-token stream) — a client-side rendering concern.
- A new tool is being added: the schema, the description, the few-shot examples are authored on the client side; the server exposes the generic `tools[]` array.
- MCP client-side orchestration (which server to call, how to compose capabilities, agent loop implementation).

### The hand-off bundle

Package before escalation:

```
# What we observed
request:
  endpoint:              /v1/chat/completions
  model:                 llama3.1-70b-instruct
  response_format:       {type: json_object}   # or json_schema, grammar, text
  tool_choice:           auto                   # or none, required, named
  parallel_tool_calls:   true
  stream:                true
  request_id:            req_abc123
  traceparent:           00-<trace-id>-<parent>-01

# What the server emitted (verbatim)
first_sse_frame:         data: {...}
nth_sse_frame:           data: {...}
finish_reason:           tool_calls
server-side_latency:
  ttft:                  185 ms
  tpot_mean:             28 ms
  total:                 4.2 s

# What the server verified
compatibility_harness_result: PASS (baseline + all drift cases)
schema_enforcement_mode:      strict
sampler_constraint_backend:   outlines (or xgrammar)
engine_metrics_sample:        [...]

# What we think is a client-layer concern
hypothesis:
  - Client's SSE parser does not handle the finish_reason value we emit.
  - Agent loop's tool-result message is sent without matching tool_call_id.
  - Prompt instructs JSON but response_format is text — enforce via response_format.
```

The bundle is complete: the server's behaviour is documented; the hypothesis is explicit; the L25 peer can reproduce in their own harness.

## The hand-off to `rag-engineer`

### When to hand off

- Per-request latency is high but TTFT / TPOT per token are within SLO — the prompt is long because retrieval is packing it. Retrieval optimisation.
- Response quality is low on retrieval-heavy workloads — the retrieved context's relevance / order / dedup is the lever, not the engine.
- Embedding endpoint load is high — embeddings per retrieval query scale with the retrieval strategy; the retrieval pipeline's batching and caching of embeddings is the lever.
- Context-window overflow — the retrieval pipeline sent more tokens than the model's context limit. The ingress rejects with `400`; the fix is in retrieval context packing.
- RAG-specific structured-output needs (extract structured facts from retrieved documents) — the schema lives on the RAG side; the inference endpoint applies it via `response_format`.

### The hand-off bundle

```
# What we observed
endpoint:                 /v1/chat/completions
model:                    llama3.1-70b-instruct
prompt_tokens:            8_742                 # from usage.prompt_tokens
completion_tokens:        312
total_tokens:             9_054
request_id:               req_def456

# Latency breakdown (chapter 02 of module 110's panels)
ttft:                     1.8 s                 # high because prompt is long
prefill_time:             1.6 s                 # dominates TTFT
tpot_mean:                32 ms                 # steady-state is fine
queue_time:               40 ms                 # queue is healthy

# What the server verified
engine_health:            green
admission_reject_rate:    0
cache_hit_rate:           0.08                  # unusually low for RAG

# Hypothesis
- Retrieval is packing 8k tokens per request; prefix cache hit rate is low because each retrieval pack is near-unique.
- Chunk size / top-k / ranking is the lever: fewer, better chunks would cut prefill time and improve cache locality.
- Alternative: use /v1/responses with "previous_response_id" for stateful retrieval if the pattern fits.
```

## Three common cross-boundary mistakes

### Mistake 1 — Filing a prompt-engineering bug as an engine bug

The agent produces hallucinations; the quality is low; the tool calls are infrequent. The team files against the inference platform ("the model isn't working").

The right triage: the inference platform confirms the engine behaviour (model emitted the tokens the sampler drew; no engine bug). The hand-off bundle documents the exact request and response. The L25 peer owns the prompt-engineering fix.

The mistake: the inference platform tries to improve the output by changing engine config (different sampler, different temperature, different model). It wastes a week and doesn't fix the problem because the problem is upstream of the engine.

### Mistake 2 — Fixing a drift bug by rewriting the client

The agent breaks on engine B but not engine A. The L25 peer files the inference team. The platform team tells the peer "your parser is too strict; fix it in your client."

The right triage: the inference platform first checks the compatibility harness. If the drift is documented and the harness catches it, the fix is either in the ingress shim (normalise the drift) or in the client (handle the drift). The decision belongs to this module per chapter 01.

The mistake: the platform tells the L25 peer to handle the drift without documenting it or adding to the harness. Next client, next peer, same drift, same incident. Every drift gets documented and tested once; every client benefits.

### Mistake 3 — Owning retrieval because it looks like prompt engineering

The RAG prompt is huge; TTFT is bad; the inference team tries to fix it by changing the engine's prefill. Prefill is already correct.

The right triage: the inference platform measures, confirms engine behaviour, writes the hand-off bundle to RAG with the observation that prefill time dominates. The RAG peer owns the fix (chunk size, top-k, reranker).

The mistake: the inference team starts tuning the retrieval pipeline. That is the RAG peer's craft; the inference team doesn't have the data (query distribution, chunk ordering, embedding model choice) to reason about it.

## The L25 track pointers

The two consumer peers:

- [`llm-application-developer-learning`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) — prompt engineering, agent loops, tool orchestration, streaming UI patterns, evaluation.
- [`rag-engineer-learning`](https://github.com/ai-engineering-curriculum/rag-engineer-learning) — retrieval pipeline design, embedding model choice, chunking strategy, reranking, context packing.

Both are in the AI Engineering family, distinct from this AI Infrastructure family. The cross-family boundary is clean: the infrastructure provides the capability; the application consumes it.

Within this module's own peer boundaries:

- The performance peer ([`ai-infra-performance-learning`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning), L35) owns the kernels that produce the tokens this module streams. Chapter 02's TTFT is an engine-side number for them; the ingress discipline is this module's.
- The security peer ([`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning), L35) owns adversarial-defence depth at the API gateway (prompt-injection filtering, model-extraction rate-limit heuristics, supply-chain concerns for model weights). Chapter 06's auth and rate-limit discipline is this module's; the attacker-facing hardening on top is theirs.
- The SRE peer ([`ai-infra-sre-learning`](https://github.com/ai-infra-curriculum/ai-infra-sre-learning), L30) owns the SRE methodology (SLI/SLO/burn-rate math, incident command, postmortem). Module 110 formalises how this module's endpoints get graded.

## Summary

- The API surface this module writes is consumed by two L25 peers: `llm-application-developer` (agent loops, prompt engineering, streaming UI) and `rag-engineer` (retrieval pipelines).
- The split is: *this module owns the protocol and the server; the L25 peers own the application semantics and the client*.
- Hand-off bundle to `llm-application-developer`: full request + response + engine metrics + compatibility harness result + hypothesis that the issue is prompt / agent / client parser.
- Hand-off bundle to `rag-engineer`: latency breakdown + token counts + cache-hit rate + hypothesis that retrieval packing is the lever.
- Three common cross-boundary mistakes: filing prompt-engineering bugs as engine bugs, fixing drift bugs only in the client (without updating the harness), and owning retrieval because it looks like prompt engineering.
- Within the AI Infrastructure family, this module composes with the performance peer (kernels), security peer (adversarial defence), SRE peer (methodology) via clear interfaces named in chapters 01-06 and module 110.
