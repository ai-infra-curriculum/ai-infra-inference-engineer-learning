# Chapter 04 — Tool Calling Protocol: Tool Choice, Parallel Calls, Round-Trip, Streaming

> A chat completion with a tool call is not one response — it is a multi-turn exchange where the model emits a structured intent, the client executes against external systems, and the model continues with the results. The server's job is to emit the intent in a shape the client can parse, accept the results, and continue the stream. Four protocol primitives govern the exchange; three provider-specific shapes make a portable implementation harder than it looks; and the streaming case is where most compatibility bugs live. This chapter writes the server-side contract.

## Motivation

A product team ships an agent that calls a weather tool, a database tool, and a calculator tool. The agent runs against OpenAI's hosted API with the parallel-tool-calls protocol. The product works. The team switches to a self-hosted inference fleet for cost and control; the agent breaks in three specific ways:

- The engine emits the tool call as `function_call` instead of `tool_calls`, breaking the parallel path.
- The engine streams tool-call arguments as one blob at the end instead of incrementally, breaking the agent's progress UI.
- The engine ignores `tool_choice: "required"` and returns a text response, breaking the "must call a tool" flow.

Three bugs, three one-line fixes once named; the on-call engineer who doesn't know the protocol spends an afternoon reproducing them. This chapter is the inventory.

By the end of this chapter you can:

- Describe the tool-call lifecycle end-to-end and the server's job at each step.
- Implement the `tool_choice` protocol in its four values (`auto`, `none`, `required`, named tool).
- Implement parallel tool calls and the round-trip that returns their results.
- Implement streaming during the tool-call phase without losing the "incremental arguments" shape.
- Translate between the OpenAI `tool_calls` shape and the Anthropic `tool_use` shape in a compatibility shim.

## The tool-call lifecycle

The flow across four actors — user, client (agent or SDK), inference server, external tool:

```
user ── prompt ──▶ client
client ── request + tools[] ──▶ server
server ── response with tool_calls[] ──▶ client        (turn 1)
client ── executes tool_calls against external tools
client ── follow-up request with messages + tool results ──▶ server
server ── response with final answer ──▶ client         (turn 2)
client ── response ──▶ user
```

Three framing points:

- **Turn 1 is a model-generated structured intent.** The model has been prompted with the available tools (each tool is a `name`, `description`, and JSON Schema of `parameters`). The model's output is either text or one or more tool-call objects indicating it wants to invoke a tool with specific arguments.
- **The server does not execute the tool.** The server emits the intent; the client (or an agent loop in the client) is responsible for running the tool code. This is a hard boundary — the inference server is a text-in-text-out service, not an RPC router. Chapter 05 (MCP) is where the boundary softens for a specific protocol.
- **Turn 2 continues from where turn 1 left off.** The client sends the message history (including the tool-call message from turn 1 and the tool-result messages for each call) and the model produces the next output: either a final text answer or another round of tool calls.

The round-trip may iterate many times — agents commonly run 5-20 turns before producing a final answer. The server-side job is to make each turn correct; the agent loop is the client's problem.

## The request shape — tools array

The tools are declared on the request:

```json
{
  "model": "llama3.1-70b-instruct",
  "messages": [
    {"role": "user", "content": "What's the weather in San Francisco and Berlin?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "additionalProperties": false,
          "required": ["city"],
          "properties": {
            "city": {"type": "string"}
          }
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto"
}
```

Four points:

- **`parameters` is a JSON Schema (chapter 03).** The same enforcement modes apply — post-hoc, JSON mode, strict, full grammar. OpenAI's tool-call strict mode constrains the generated `arguments` to validate against `parameters`.
- **`description` is prompt engineering.** The model reads it to decide when to invoke the tool. Terse, action-oriented, include the schema's expected fields. This is a client-side concern; chapter 07's boundary note returns to it.
- **`type: "function"` is OpenAI's wrapper object.** The newer `/v1/responses` endpoint uses a flatter shape (`tools[].name`, `tools[].parameters`); chat completions carries the wrapper. Verify the engine's accepted shape.
- **`strict: true` on the function enables sampler-level constraint** on the arguments (requires the engine to route through the structured-output path in chapter 03).

## The response shape — tool_calls array

Non-streaming response with tool calls:

```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"San Francisco\"}"
          }
        },
        {
          "id": "call_def",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"Berlin\"}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Three correctness properties:

- **`content` is `null` (not empty string) on a pure tool-call turn.** Some engines emit `""`; the drift catalogue (chapter 01) captures this.
- **`finish_reason: "tool_calls"`** signals this turn stopped because the model emitted tool calls, not because it reached a stop token. The agent loop keys on this.
- **`arguments` is a JSON *string*, not a JSON object.** The agent parses it. The reason: the API was designed when "function call" was a single-string blob; the array form preserved the shape. Many agents forget and try to use `arguments` as an object.

## The follow-up request — tool results

The client executes the tools and sends the results back in the next request:

```json
{
  "model": "llama3.1-70b-instruct",
  "messages": [
    {"role": "user", "content": "What's the weather in San Francisco and Berlin?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"San Francisco\"}"}},
        {"id": "call_def", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "call_abc", "content": "{\"temperature_c\":17,\"conditions\":\"cloudy\"}"},
    {"role": "tool", "tool_call_id": "call_def", "content": "{\"temperature_c\":9,\"conditions\":\"rainy\"}"}
  ],
  "tools": [...],
  "tool_choice": "auto"
}
```

Three correctness properties:

- **The assistant's prior tool-call message is included verbatim.** The model needs to see what it asked for; the client re-sends.
- **Each tool result is a message with `role: "tool"` and `tool_call_id` matching the call's `id`.** Mismatched IDs break the model's correlation of call-to-result.
- **The result `content` is a string.** Usually JSON; could be plain text. If the result is binary (an image, audio), the application-level peer handles multimodal; the string form is the common case.

## The `tool_choice` protocol

Four values:

| Value | Meaning | Server behaviour |
|---|---|---|
| `"auto"` (default) | Model decides whether to call a tool. | Model may emit text or tool calls per its own reasoning. |
| `"none"` | Model must not call a tool. | Engine suppresses the tool-call path; output is text only. |
| `"required"` | Model must call at least one tool. | Engine constrains the output to be a tool-call sequence (via sampler-level grammar). |
| `{"type": "function", "function": {"name": "get_weather"}}` | Model must call the named tool. | Engine constrains the output to be a call to exactly that tool. |

Server-side implementation:

- **`"auto"`** is the default sampler path; no constraint beyond the system prompt listing the tools.
- **`"none"`** suppresses the tool-call grammar from the sampler path; the engine either doesn't emit the tools in the prompt-side system message (for engines that let you suppress), or masks out the tool-call-opening tokens.
- **`"required"`** is a sampler constraint: the output must begin with the tool-call opening token (or its equivalent in the model's output format — some models emit a JSON wrapper, some emit a `<tool_call>` tag, some use function-call-style tokens). The engine's chat template defines the shape.
- **A named tool** is a stricter constraint: the opening is the named tool's call shape, and the arguments must validate against that tool's `parameters` schema.

Engines vary in how strictly they enforce these. The compatibility harness (chapter 01) gets a case per value.

## Parallel tool calls

The 2023 protocol was one tool call per turn: the model emits one call, the client runs it, the model continues. The 2024 protocol allows one turn to emit multiple calls, which the client runs (often in parallel) and reports back as an array of tool-result messages in the next turn.

The server-side change is small — the `tool_calls` array has multiple entries — but the behaviour change is significant:

- **The model emits all calls before any results are in.** It cannot condition call #2 on the result of call #1 in the same turn; that requires a second turn.
- **Call IDs must be unique within a turn.** The client correlates results by `tool_call_id`; duplicate IDs are an ambiguity.
- **`parallel_tool_calls: false`** on the request body is the opt-out — the model emits at most one call per turn. Engines vary in whether they honour this flag; the drift catalogue tracks.

## Streaming during the tool-call phase

The streaming shape under tool calls is where the subtle bugs live. The model's first tokens may be tool-call opening tokens (not visible text); the engine emits them as incremental `tool_calls[].function.arguments` deltas; the client assembles them into a complete call before executing.

The minimum viable stream for a parallel tool call:

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"San Francisco\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_def","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"city\":\"Berlin\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

```

Client-side assembly:

```python
calls = {}
for frame in stream:
    for tc in frame.get("choices", [{}])[0].get("delta", {}).get("tool_calls", []):
        idx = tc["index"]
        if idx not in calls:
            calls[idx] = {"id": tc.get("id"), "name": tc["function"].get("name"), "arguments": ""}
        calls[idx]["arguments"] += tc["function"].get("arguments", "")
    if frame["choices"][0]["finish_reason"] == "tool_calls":
        break
```

Three correctness properties the server must emit:

- **`index` is present on every tool_call delta.** The client uses `index` to assemble the parallel calls; without it, two deltas look ambiguous.
- **`id`, `type`, `name` appear on the first delta for each call.** Subsequent deltas only need `index` and incremental `arguments`. Some engines re-emit `name` on every delta (harmless); some omit `id` on the first delta (client-breaking).
- **Arguments stream incrementally if the model is capable of partial-JSON emission.** Per chapter 03, strict-mode JSON schema enforcement at the sampler constrains each token; the string incrementally builds toward a schema-valid object. A client UI can show "calling `get_weather` with { city: 'San Fr...'" while the call is forming.

Four failure modes:

### Arguments arrive as one blob at the end

Some engines buffer the entire tool-call arguments object internally and emit it as a single delta at the end. The symptom: the streamed-chunk shape is "name on first delta, then nothing, then a huge arguments string on the final delta." Agents that stream the arguments to a UI show nothing, nothing, then everything.

This is a drift (chapter 01 point 4). The fix is either engine configuration (some engines have a `stream_fine_grained` flag) or a compatibility shim that normalises client-side.

### Interleaving vs. strict-serial

Two parallel calls can interleave their argument deltas (call 0 emits some arguments, call 1 starts and emits some arguments, call 0 resumes). Alternatively the engine may fully serialise (call 0 completes, then call 1 starts).

Both are technically allowed; a client that assumes strict serial breaks on interleaving. The server should document which shape it emits; the client should assemble by `index` and not assume ordering.

### Tool-call phase interrupted by text content

Some models emit prelude text before calling a tool ("I'll check the weather. Let me call the tool...") then the tool call. The streamed deltas carry both:

```
data: {"choices":[{"delta":{"content":"I'll check the weather. "}}]}
data: {"choices":[{"delta":{"content":"Let me call the tool..."}}]}
data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_abc",...}]}}]}
...
```

The client's agent loop must handle both `content` and `tool_calls` deltas in the same response. Most SDKs do; some hand-rolled parsers don't.

### Mid-stream cancellation

A client that disconnects mid-tool-call must trigger the engine's cancel path (chapter 02). The engine must release the KV slot even though the output is a partial tool-call rather than a partial text response. The abort path is identical; the compatibility harness should include a tool-call cancellation case.

## Provider-specific shapes — the translation problem

Three dominant shapes in the market:

### OpenAI (chat completions)

```json
{
  "tool_calls": [
    {"id": "call_abc", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
  ]
}
```

### OpenAI (`/v1/responses`)

```json
{
  "output": [
    {"type": "function_call", "id": "call_abc", "name": "get_weather",
     "arguments": "{\"city\":\"SF\"}"}
  ]
}
```

### Anthropic (`/v1/messages`)

```json
{
  "content": [
    {"type": "tool_use", "id": "toolu_abc", "name": "get_weather",
     "input": {"city": "SF"}}
  ]
}
```

Three deltas:

- **Arguments vs. input, string vs. object.** OpenAI emits `arguments` as a JSON string; Anthropic emits `input` as a JSON object. The compatibility shim parses or re-stringifies.
- **ID prefix convention.** OpenAI's chat completions uses `call_`, `/v1/responses` keeps `call_`, Anthropic uses `toolu_`. The shim preserves or translates.
- **Wrapper shape.** OpenAI chat completions nests under `function`; `/v1/responses` flattens; Anthropic flattens. The shim handles the structure difference.

A compatibility shim (sketch):

```python
def translate_openai_chat_to_anthropic(openai_msg: dict) -> list[dict]:
    """Translate an OpenAI chat-completions assistant message with tool_calls
    into Anthropic Messages content blocks."""
    out = []
    if openai_msg.get("content"):
        out.append({"type": "text", "text": openai_msg["content"]})
    for tc in openai_msg.get("tool_calls", []) or []:
        out.append({
            "type": "tool_use",
            "id": tc["id"].replace("call_", "toolu_", 1),
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        })
    return out
```

The reverse direction is similar; the translation is lossy only if the engine emits a shape with no analogue (e.g., Anthropic's `cache_control` on a tool-use block is OpenAI-foreign).

## Three failure modes

### Finish-reason mismatch

The engine emits `finish_reason: "stop"` on a turn where the model clearly called a tool. The client's agent loop keys on `finish_reason == "tool_calls"` to decide whether to continue; a `"stop"` means "we're done," so the agent never runs the tool.

Common cause: engine uses the base-model finish-reason logic without the tool-call override. Fix: configure the engine's chat template to emit `"tool_calls"` when `tool_calls` is non-empty. The compatibility harness catches it.

### Tool-call-id collision across turns

Call IDs should be unique per request, but some clients re-use them across turns ("call_1" in turn 1, "call_1" in turn 3 for a different tool). The server passes through; the model may be confused about correlation. Enforce uniqueness in the ingress if your fleet sees this.

### Streaming tool-call never terminates

The engine emits deltas forever on a tool-call phase. Common cause: the model's chat template's stop-token logic is wrong for the tool-call path. The engine never sees the "end of tool call" token. Fix: verify the chat template; add a case to the compatibility harness; file against the engine if it's a model-template bug.

## Summary

- Tool calling is a multi-turn exchange: model emits structured intent, client executes externally, model continues with results.
- The server emits tool-call intents, accepts tool-result messages, and continues the stream. It does not execute tools.
- `tool_choice` has four values (`auto`, `none`, `required`, named); engines vary in how strictly they enforce the constrained values.
- Parallel tool calls emit multiple calls per turn, correlated back by `tool_call_id`; `parallel_tool_calls: false` is the opt-out.
- Streaming during the tool-call phase requires `index` on every delta, `id/type/name` on the first, incremental `arguments`. Common drift: arguments as a final blob.
- Provider shapes differ (OpenAI chat vs. `/v1/responses` vs. Anthropic); a compatibility shim at the ingress can translate.
- Three failure modes: finish-reason mismatch, tool-call-id collision, non-terminating stream. Each is a one-config-flag fix once named; the compatibility harness is the mechanism.
