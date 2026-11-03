# Chapter 05 — MCP Server-Side Exposure: Capabilities, Auth, and the Agent-Shaped Client

> The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting AI models to data sources and tools, published by Anthropic in late 2024 and adopted by multiple SDKs since. For an inference engineer, MCP introduces a second server surface alongside the OpenAI-compatible API: not "give me completions for this prompt" but "here are the resources, tools, and prompts I expose; call them." This chapter writes the server-side implementation and the operational implications.

## Motivation

An AI product team builds an agent that reads from a knowledge base, writes to a ticket system, calls an inference endpoint, and queries a search index. In 2023 each of those connections was a bespoke integration — one Python client per external system, hand-written tool schemas, custom auth per vendor. In 2024-2025 the industry converged on MCP: *every* capable system exposes an MCP server, and the agent talks to each via one protocol.

For the inference engineer, this changes the picture in two ways:

- **The inference endpoint may itself be an MCP server.** Instead of (or in addition to) an OpenAI-compatible HTTP API, the fleet exposes an MCP endpoint that advertises `Tools` the agent can call (for instance, "run a chat completion," "generate an embedding," "transcribe audio"). The agent discovers the capability and calls it over the MCP wire protocol.
- **The inference endpoint is one of many MCP servers the agent talks to.** Operational concerns the inference platform used to own (auth, rate limit, request ID) now interleave with similar concerns on the knowledge-base MCP server, the ticket-system MCP server, etc. The ingress discipline from chapter 06 generalises to this heterogeneous landscape.

This chapter focuses on the first: *your* inference fleet exposing an MCP server. The second is a consumer-side concern that belongs to the L25 application-developer peer.

By the end of this chapter you can:

- Describe MCP's wire protocol and its three capability kinds (Resources, Tools, Prompts) in one page.
- Expose an inference endpoint as an MCP Tool.
- Wire the auth model (OAuth 2.1 Resource Server for remote servers).
- Reason about the operational implications when the client is MCP-aware (session lifecycle, long-lived connections, capability discovery) rather than a stateless chat SDK.

## MCP in one page

The [MCP specification](https://modelcontextprotocol.io/specification) is JSON-RPC 2.0 over one of two transports:

- **stdio** — for local MCP servers that run as subprocesses of the client (an IDE plugin, a desktop agent). The client spawns the server; messages flow over stdin/stdout.
- **Streamable HTTP** — for remote MCP servers. The server exposes an HTTP endpoint; the client POSTs JSON-RPC requests; the server responds with JSON-RPC responses, optionally streaming server-to-client notifications via SSE on a single HTTP connection.

<!-- needs-research: MCP transport semantics have evolved (early drafts used an SSE + POST pair; the current spec uses Streamable HTTP, a single endpoint that upgrades to SSE for server→client messages). Verify against the current specification revision before implementing. -->

The three capability kinds:

- **Resources** — read-only data the server exposes (`file://...`, `postgres://table/row`, `knowledge://doc/42`). The client requests a resource by URI; the server returns content. Clients can subscribe to resource changes.
- **Tools** — functions the server exposes that the model can invoke. Each tool has a `name`, `description`, and `inputSchema` (JSON Schema). The model-side protocol is identical to the function-call protocol in chapter 04 — MCP tools *are* the generalization of OpenAI tool calls when the backend is an MCP server.
- **Prompts** — reusable prompt templates the server exposes (`summarize_meeting`, `extract_entities`, `write_bug_report`). The client requests a prompt by name with arguments; the server returns a message list ready to send to a model.

Three framing points:

- **MCP is a client-server protocol between the agent and external capabilities.** The agent is the MCP *client*; the knowledge-base, ticket system, inference endpoint — each is an MCP *server*. The model itself does not speak MCP; the agent surfaces MCP's capabilities as tool calls to the model and interprets the model's tool-call outputs by calling the MCP server.
- **MCP's protocol is not a replacement for OpenAI-compatible APIs.** An inference server can expose both — an OpenAI-compatible HTTP API for direct chat SDK clients, and an MCP endpoint for agents that discover capabilities dynamically. Many fleets expose both.
- **MCP is session-oriented.** Each client-server interaction is a session with capability discovery (`initialize` → `listTools` → `listResources` → `listPrompts`), then a series of calls, then a close. Sessions can be long-lived; the server tracks per-session state.

## The inference-endpoint-as-MCP-server pattern

Pattern: the inference fleet exposes an MCP server that advertises Tools corresponding to the HTTP endpoints — one Tool for chat completions, one for embeddings, one for transcription. An agent that discovers the server sees the capabilities and can call them.

### Capability declaration

```json
// Server's response to listTools:
{
  "tools": [
    {
      "name": "chat_completion",
      "description": "Generate a chat completion from a message list.",
      "inputSchema": {
        "type": "object",
        "additionalProperties": false,
        "required": ["model", "messages"],
        "properties": {
          "model": {"type": "string"},
          "messages": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "required": ["role", "content"],
              "properties": {
                "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                "content": {"type": "string"}
              }
            }
          },
          "temperature": {"type": "number", "minimum": 0, "maximum": 2},
          "max_tokens": {"type": "integer", "minimum": 1}
        }
      }
    },
    {
      "name": "embed_text",
      "description": "Generate embeddings for input text.",
      "inputSchema": {
        "type": "object",
        "additionalProperties": false,
        "required": ["model", "input"],
        "properties": {
          "model": {"type": "string"},
          "input": {"type": "string"}
        }
      }
    }
  ]
}
```

Three correctness properties:

- **`inputSchema` is JSON Schema** — the chapter-03 constraints apply (`additionalProperties: false`, `required`, strict enough to be enforceable).
- **`description` is model-facing.** The agent will prompt a model with these descriptions; terse, actionable, mention when to use the tool.
- **Tool names are stable.** An agent that discovered `chat_completion` yesterday and finds `generate_chat` today has to re-discover; keep the names stable across releases, add new tools rather than rename.

### A call

```json
// Client's call:
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "chat_completion",
    "arguments": {
      "model": "llama3.1-70b-instruct",
      "messages": [
        {"role": "user", "content": "What is 2+2?"}
      ],
      "max_tokens": 32
    }
  }
}

// Server's response:
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [{"type": "text", "text": "4."}],
    "isError": false
  }
}
```

Three properties:

- **The result is a `content` array**, not a chat-completion object. The MCP tool-call result shape is uniform across all tools (text, image, resource reference); the server translates the engine's response into this shape.
- **`isError: false` indicates success.** An error-shaped result (`isError: true`) carries error text in the content; a JSON-RPC error (negative `code`) is reserved for protocol errors.
- **Streaming is possible** via server-to-client notifications (progress updates), but the final result is one JSON-RPC response. Agent-visible token streaming within an MCP tool call is an active area; verify against the current spec.

### A minimal server implementation

```python
# Uses the mcp Python SDK (official). The exact SDK API drifts; verify
# against the pinned release.
from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult
import httpx

server = Server("inference-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="chat_completion",
            description="Generate a chat completion from a message list.",
            inputSchema={
                "type": "object",
                "additionalProperties": False,
                "required": ["model", "messages"],
                "properties": {
                    "model": {"type": "string"},
                    "messages": {"type": "array", "items": {"type": "object"}},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": "integer"},
                },
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "chat_completion":
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://engine.internal/v1/chat/completions",
                json=arguments,
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
    raise ValueError(f"Unknown tool: {name}")

# Transport: Streamable HTTP for remote; stdio for local.
# The SDK's main-entrypoint docs cover both.
```

Four disciplines:

- **Pin the SDK version** — the spec and SDKs drift together.
- **Validate `arguments` against the declared `inputSchema`** inside the handler before forwarding to the engine. A compliant client validates on its side but the server should not trust that.
- **Timeouts and cancellation.** A long inference call needs the same disconnect-and-cancel discipline from chapter 02; MCP's streamable transport should propagate cancellation.
- **Expose the OpenAI-compatible endpoint and the MCP endpoint side by side.** Most fleets will serve both for the foreseeable future.

## The auth model — OAuth 2.1 Resource Server

MCP servers in the local / stdio case run as subprocesses of the client; the client is already a trusted process on the user's machine; auth is implicit. For remote MCP servers (HTTP), the specification mandates OAuth 2.1 as the authorization framework.

The pattern is the "Resource Server" role from OAuth:

- The MCP server is the Resource Server. It validates bearer tokens on incoming requests but does not issue them.
- An Authorization Server (operated by the same org or a third party like Auth0, Okta, Keycloak) issues tokens to clients after they authenticate.
- The client (an agent, an IDE, a desktop app) obtains a token via one of the OAuth 2.1 flows (authorization code + PKCE for interactive clients, client credentials for machine-to-machine).
- Each MCP request carries the token in the `Authorization: Bearer <token>` header.
- The MCP server validates the token (signature, expiration, audience, scope) and extracts identity.

A sketch of the server-side middleware:

```python
from jose import jwt, JWTError  # or an alternative library

AUTH_SERVER_JWKS_URL = "https://auth.example.com/.well-known/jwks.json"
AUDIENCE = "https://inference.example.internal/mcp"
ISSUER = "https://auth.example.com/"

async def authenticate(request) -> Identity:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = header[len("Bearer "):]
    try:
        payload = jwt.decode(
            token,
            fetch_jwks(AUTH_SERVER_JWKS_URL),
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return Identity(
        subject=payload["sub"],
        scopes=set(payload.get("scope", "").split()),
    )
```

Three disciplines:

- **Validate audience.** A token issued for another resource (e.g., the knowledge-base MCP server) must not authenticate the inference MCP server. The `aud` claim must match this server's identifier.
- **Validate issuer.** Only tokens from the expected authorization server are accepted.
- **Enforce scopes per tool.** A `chat_completion` tool may require the `inference:chat` scope; `embed_text` may require `inference:embed`. Reject calls with insufficient scope with a `403`.

### Why OAuth 2.1 and not something simpler

An API-key model (one key per client, server validates the key) is operationally simpler and dominates OpenAI-compatible deployments. For MCP, OAuth was chosen because:

- An agent may talk to many MCP servers; a single OAuth-issued token can gate access to many servers with scope-based isolation.
- Short-lived tokens (minutes to hours) limit the blast radius of a leaked token versus long-lived API keys.
- Standardised flows (authorization code + PKCE) let interactive clients authenticate users; machine-to-machine flows handle non-interactive agents.

The inference fleet's OpenAI-compatible endpoint may still use API keys (chapter 06); the MCP endpoint should use OAuth 2.1. The two can coexist in the ingress.

## Session lifecycle

An MCP session is a long-lived interaction. The lifecycle:

1. **initialize** — client sends protocol version, capabilities, metadata; server responds with its protocol version and declared capabilities.
2. **listTools / listResources / listPrompts** — client discovers what the server exposes.
3. **tools/call, resources/read, prompts/get** — the operational phase; typically many calls per session.
4. **notifications/cancelled** — client aborts an in-flight call.
5. **close** — session end.

Operational implications for the inference platform:

- **Session state.** The server may want to hold per-session state (user identity, rate-limit counters, cached capabilities). Where is it stored? For a stateless MCP server, each request's identity is re-derived from the token; rate-limit counters live in a shared cache (Redis). The MCP server itself is horizontally scalable.
- **Long-lived connections.** A session may span minutes. The HTTP connection carrying it needs the same keepalive discipline from chapter 02 — heartbeats, idle timeouts configured to session length, graceful connection migration if the server pod is drained.
- **Capability discovery is cacheable.** The `listTools` result is stable across requests; clients can cache it. The server may emit a notification on capability change (new tool added, old tool deprecated) and clients re-fetch.
- **Streaming through tool calls.** A long inference call (chat completion with 2k output tokens) inside an MCP tool call needs progress notifications so the agent's UI shows motion. The MCP spec defines progress notifications; use them.

## Three operational implications

### Rate limiting per identity, not per IP

The OpenAI-compatible endpoint rate-limits per API key (chapter 06). The MCP endpoint rate-limits per OAuth subject + scope. The two policies may differ (an interactive user agent sees a tighter per-token-of-work quota; a service account sees a looser per-minute quota). Author the policies separately; share the rate-limit backend (Redis).

### Audit and observability

MCP's session structure is a natural trace scope. Each session carries a session ID; each call within a session is a span; the inference call downstream is a child span. The W3C `traceparent` propagation from chapter 06 extends into MCP's JSON-RPC — the ingress wraps the MCP transport to populate the trace context.

Audit logs on the MCP endpoint should capture: session ID, OAuth subject, scopes, tool name, request shape (not full arguments — those may be sensitive), response shape (not full result), latency, error. Chapter 06's audit discipline carries over.

### Capability drift

MCP servers evolve. A tool's `inputSchema` may add a parameter; a tool may be renamed; a tool may be deprecated. Clients cache capabilities; a schema change without a cache invalidation produces client-visible errors.

Discipline:

- **Add parameters as optional, never required.** Breaks old clients to add required fields.
- **Deprecate before remove.** Emit a `notifications/tools/listChanged` to prompt clients to re-fetch; keep the deprecated tool working for one release.
- **Version the server.** The MCP spec is versioned; the server's own capabilities are versioned via a semver string in the `initialize` response. Clients that depend on specific capabilities can gate on the server version.

## Three failure modes

### Session that outlives the backend connection

The MCP session is long-lived but the inference pod gets drained (module 109 chapter 05's spot preemption flow). The session cleanly closes if the server respects the shutdown signal and emits a close notification; it hangs if the server drops the connection without notification. Wire the shutdown hook.

### Scope leakage through a shared backend

Two tools (`chat_completion` for scope `inference:chat`, `embed_text` for scope `inference:embed`) share a backend engine. A client with only `inference:chat` scope that calls `embed_text` must be rejected at the MCP layer before reaching the engine. Scope enforcement is per-tool, not per-session; check on every call.

### Capability-discovery storm

A fleet upgrade changes capabilities. Every connected client re-fetches `listTools`. If 10k clients are connected and all re-fetch within a second, the MCP endpoint sees a 10k QPS spike on a request type it usually serves at 10 QPS. Rate-limit `listTools` separately from `tools/call`; cache the list response at the ingress.

## Summary

- MCP is a JSON-RPC-based client-server protocol for exposing Resources, Tools, and Prompts to AI agents. The spec is at <https://modelcontextprotocol.io/specification>.
- For an inference engineer, the primary pattern is exposing the inference endpoint as an MCP server with the operational endpoints (chat, embed, transcribe) as Tools.
- The auth model for remote MCP is OAuth 2.1 Resource Server; validate audience, issuer, and scopes per tool.
- MCP sessions are long-lived; the server needs the same connection discipline (heartbeats, idle timeouts, graceful shutdown) as streaming SSE.
- The OpenAI-compatible endpoint and the MCP endpoint can coexist; many fleets expose both.
- Operational implications: per-identity rate limiting, trace propagation into JSON-RPC, audit at session granularity, careful capability evolution.
- Three failure modes: session outliving backend, scope leakage, capability-discovery storm.
- The *client-side* of MCP (agent loops, orchestration, prompt engineering around tools) belongs to the L25 application-developer peer; chapter 07 formalises.
