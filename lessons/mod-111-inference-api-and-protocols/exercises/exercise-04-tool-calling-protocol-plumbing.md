# Exercise 04 — Tool Calling Protocol Plumbing

**Time:** 4 hours · **Output:** one tool-call translator (OpenAI ↔ Anthropic shape), one parallel-call test suite, one streamed-tool-call verification harness

## Goal

Make [Chapter 04](../04-tool-calling-protocol.md) concrete by **implementing a tool-call translator** that accepts an OpenAI-shape request and routes it to either an OpenAI-shape or Anthropic-shape backend, handles parallel tool calls, streams through the tool-use phase, and completes the round-trip without losing a chunk.

## Problem statement

Chapter 04 names the protocol primitives, the parallel path, and the streamed tool-call shape — plus the three dominant provider shapes (OpenAI chat completions, OpenAI `/v1/responses`, Anthropic `/v1/messages`) that a compatibility ingress must translate. In practice, a fleet that serves multiple models or multiple backends ships this translator once and runs it in the ingress.

You will author the translator (both directions), build a parallel-tool-call test suite, implement a streamed tool-call verification harness that asserts the correct per-delta shape, and run a full round-trip across two backend providers (one OpenAI-shape, one Anthropic-shape).

## Prerequisites

- Chapter 04 of this module read.
- At least one OpenAI-shape backend: a self-hosted engine (vLLM / SGLang / TRT-LLM with tool-calling support) or the OpenAI hosted API.
- At least one Anthropic-shape backend: the Anthropic Messages API (hosted) or a self-hosted engine that emits Anthropic-shape responses (rare; most self-hosted engines emit OpenAI shape).
- A model that supports tool calling — most instruct models with `_tool_use` or chat-template support.
- The ingress from exercise 02 as the host for the translator middleware.
- Python 3.11+; `pydantic`, `httpx`, `openai`, `anthropic` SDKs (optional but easier).

## Requirements

Five deliverables.

### Deliverable A — the request translator

A function that takes an OpenAI-shape chat-completions request and emits either:

- The same OpenAI-shape request (passthrough), or
- An Anthropic-shape `/v1/messages` request.

Core translations:

- `messages[]` with roles `system | user | assistant | tool` → Anthropic's `system` (string, top-level) + `messages[]` with roles `user | assistant` and `content[]` arrays of blocks (text, tool_use, tool_result).
- `tools[]` as `{type: function, function: {name, description, parameters}}` → Anthropic's `tools[]` as `{name, description, input_schema}`.
- `tool_choice` values: `"auto"` → `{type: auto}`; `"required"` → `{type: any}`; `{type: function, function: {name}}` → `{type: tool, name}`; `"none"` → omit `tools` (no direct Anthropic equivalent as of writing — verify).
- `parallel_tool_calls: false` → Anthropic's `disable_parallel_tool_use: true` (verify current naming).
- `response_format` is not directly supported in Anthropic Messages as of writing — handle by prompt engineering or extended tool-schema; document the gap.

```python
def openai_to_anthropic(req: dict) -> dict:
    system_parts = [m["content"] for m in req["messages"] if m["role"] == "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    anthropic_messages = []
    for m in req["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []) or []:
                blocks.append({
                    "type": "tool_use",
                    "id": translate_id(tc["id"]),
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                })
            anthropic_messages.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            # Anthropic expects tool_result inside a user-role message.
            anthropic_messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": translate_id(m["tool_call_id"]),
                "content": m["content"],
            }]})
    return {
        "model": req["model"],
        "max_tokens": req.get("max_tokens", 1024),
        "messages": anthropic_messages,
        **({"system": system} if system else {}),
        **({"tools": [translate_tool(t) for t in req["tools"]]} if req.get("tools") else {}),
        **translate_tool_choice(req.get("tool_choice")),
        **translate_stream(req.get("stream")),
    }
```

### Deliverable B — the response translator

A function (or async generator for streaming) that takes an Anthropic-shape response and emits an OpenAI-shape response.

Core translations:

- Anthropic's `content[]` array with `text` and `tool_use` blocks → OpenAI's `message` with `content` string and `tool_calls` array.
- Anthropic's `stop_reason` (`end_turn`, `tool_use`, `max_tokens`, `stop_sequence`) → OpenAI's `finish_reason` (`stop`, `tool_calls`, `length`).
- Anthropic's `usage` (`input_tokens`, `output_tokens`) → OpenAI's `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).

For streaming, the shapes are more different: Anthropic emits typed events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`), OpenAI emits `chat.completion.chunk` SSE frames. The translator assembles Anthropic events into OpenAI deltas.

```python
async def anthropic_stream_to_openai(aiter):
    """Translate Anthropic typed event stream into OpenAI chat.completion.chunk SSE frames."""
    msg_id = None
    current_tool_call = None   # {index, id, name, arguments}
    async for event in aiter:
        if event["type"] == "message_start":
            msg_id = event["message"]["id"]
            yield build_frame(msg_id, {"role": "assistant"}, None)
        elif event["type"] == "content_block_start":
            block = event["content_block"]
            if block["type"] == "text":
                pass  # Nothing until first delta.
            elif block["type"] == "tool_use":
                current_tool_call = {
                    "index": event["index"],
                    "id": openai_id_from(block["id"]),
                    "name": block["name"],
                    "arguments": "",
                }
                yield build_tool_call_frame(msg_id, current_tool_call, first=True)
        elif event["type"] == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "text_delta":
                yield build_frame(msg_id, {"content": delta["text"]}, None)
            elif delta["type"] == "input_json_delta":
                current_tool_call["arguments"] += delta["partial_json"]
                yield build_tool_call_delta_frame(msg_id, current_tool_call)
        elif event["type"] == "message_delta":
            # finish_reason comes from stop_reason here.
            yield build_frame(msg_id, {}, translate_stop(event["delta"]["stop_reason"]))
        elif event["type"] == "message_stop":
            yield "data: [DONE]\n\n"
```

### Deliverable C — the parallel-tool-call test suite

Script a request that prompts the model to call the same tool twice with different inputs:

```python
request = {
    "model": "...",
    "messages": [{"role": "user", "content": "What's the weather in San Francisco and Berlin?"}],
    "tools": [weather_tool_spec],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "stream": False,
}
```

Verify against each backend:

- Both tools are called in the same response (single turn).
- Each call has a unique `tool_call_id`.
- Each call's arguments validate against the tool schema.
- `finish_reason == "tool_calls"`.
- Sending the tool results back (second turn, with both results) produces a coherent final answer.

Then set `parallel_tool_calls: False` and verify the backend emits one tool call per turn (two turns to answer).

### Deliverable D — the streamed tool-call shape verification

For the streamed case, assert:

- Every `tool_calls[]` delta has `index`.
- The first delta for each call carries `id`, `type: "function"`, and `name`.
- Subsequent deltas carry only `index` and incremental `function.arguments`.
- The arguments concatenate (across deltas for that `index`) into valid JSON that validates against the tool schema.
- The `finish_reason: "tool_calls"` frame arrives before `[DONE]`.
- A mid-stream disconnect propagates to the backend abort (reuse exercise 02's disconnect test).

The verification is a parser that reconstructs the tool-call set from the stream and asserts correctness. Fail loudly on any deviation.

### Deliverable E — the drift report

For each backend you tested, record:

- Does it emit `function_call` or `tool_calls`?
- Does `parallel_tool_calls: False` disable parallel calls or is it ignored?
- Does `tool_choice: "required"` reliably force a tool call?
- Does the streamed tool-call shape match chapter 04's delta rules exactly, or does it batch arguments?
- Does cancellation release the KV slot mid-tool-call?
- How does the backend emit `finish_reason` when a tool is called?

Add these as cases to the exercise 01 compatibility harness.

## Starter guidance

- **Start with non-streaming.** Get the request and response translators working against a non-streaming backend; add streaming once the shapes are right.
- **Write the translators as pure functions.** Given an input dict, return an output dict. Easy to unit-test; the integration test is just the pipeline end-to-end.
- **Preserve the `tool_call_id` across translations.** Round-trip: OpenAI client sends a tool-result message with `tool_call_id: "call_abc"`; translator maps to Anthropic `toolu_abc`; response translator maps back. Document the mapping.
- **Test the one-tool-two-calls case explicitly.** Parallel tool calls often work for two different tools; the "same tool, different inputs" case exposes correlation bugs.
- **Use minimal tool schemas.** The schema complexity is tested in exercise 03; here the focus is the protocol plumbing.
- **Cache translator output for a representative request.** It makes a great artefact for the chapter 07 hand-off bundle to the L25 peer.

## Fallbacks for constrained environments

- **One backend available.** Still valuable: implement the translator in both directions, test with mocked responses for the other direction. The translator's logic is the artefact.
- **No parallel-tool-call-capable model.** Test the sequential path (parallel_tool_calls: false) thoroughly; note the parallel case as blocked.
- **No tool-capable model.** This exercise is blocked; defer until a capable model is available in your stack. The compatibility harness from exercise 01 still catches shape issues without a working tool call.

## Acceptance criteria

1. **Deliverable A** request translator handles the four `tool_choice` values, `parallel_tool_calls`, and round-trips `tool_call_id`s across turns.
2. **Deliverable B** response translator handles both non-streaming and streaming; the streaming case produces OpenAI-shape SSE deltas.
3. **Deliverable C** parallel test suite passes against both backends (one of them may be the hosted reference); the sequential path also passes.
4. **Deliverable D** streaming verification passes; every delta meets the shape rules.
5. **Deliverable E** drift report is in the compatibility harness.
6. The translator's test suite has ≥ 10 pure-function tests covering both directions and the edge cases (empty tool_calls, text-only response, message-only tool-result turn).

## Common failure modes

- **Arguments-as-blob emission.** The backend emits full arguments at the end rather than incrementally. Document; the translator can still produce streamed deltas by splitting the blob, but the client UX differs.
- **`tool_call_id` collision across translations.** Mapping `call_abc` → `toolu_abc` → `call_abc` is fine; mapping to a generated ID loses the client's correlation. Preserve.
- **Tool-result message inside assistant vs. user role.** OpenAI uses `role: "tool"`; Anthropic nests `tool_result` blocks inside a `user` message. Mix them up and the backend ignores the tool result.
- **System message in wrong location.** OpenAI puts `system` inside `messages[]`; Anthropic requires it as a top-level `system` field. Mix them up and the system prompt is ignored or dropped.
- **`finish_reason` mapping drift.** Anthropic's `end_turn` vs. OpenAI's `stop`; Anthropic's `tool_use` vs. OpenAI's `tool_calls`. The translator must map cleanly; the compatibility harness catches drift.

## Keep it

The translator, the parallel test suite, and the streamed-shape verification are the fleet's tool-call compatibility artefact. The ingress from exercise 06 hosts the translator; the compatibility harness from exercise 01 inherits the drift cases; chapter 07's hand-off bundle to `llm-application-developer` includes the translator's behaviour notes.
