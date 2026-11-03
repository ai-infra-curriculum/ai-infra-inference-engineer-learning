# Exercise 05 — MCP Server-Side Exposure Walkthrough

**Time:** 4 hours · **Output:** one MCP server exposing an inference endpoint as a Tool, one OAuth 2.1 Resource Server integration, one MCP-aware client round-trip transcript

## Goal

Make [Chapter 05](../05-mcp-server-side-exposure.md) concrete by **standing up an MCP server** that advertises an inference endpoint (chat completion, embedding, or both) as a Tool, implementing capability discovery, wiring OAuth 2.1 token validation, and proving an MCP-aware client can discover, authorise, and invoke the capability end-to-end.

## Problem statement

Chapter 05 names the protocol, the three capability kinds, the auth model, and the operational implications. The exercise builds the smallest realistic server-side MCP endpoint that exposes your inference fleet, wires the auth, and demonstrates the round-trip. It is not a production deployment — it is enough code and configuration to make the chapter's claims concrete.

You will author an MCP server using the official SDK (or a hand-rolled JSON-RPC stack), expose at least one Tool that forwards to the inference engine, implement OAuth 2.1 token validation against a configured authorization server, drive the server from an MCP-aware client, and publish the transcript and runbook.

## Prerequisites

- Chapter 05 of this module read.
- The [MCP specification](https://modelcontextprotocol.io/specification) skimmed — at least the Tools, Transport, and Authorization sections at the current revision.
- A running inference engine exposing OpenAI-compatible chat completions and/or embeddings.
- Python 3.11+; the [`mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk) (verify the current package name and version); `httpx`, `python-jose` (or equivalent JWT library).
- An OAuth 2.1 authorization server reachable from the MCP server. For local testing: a self-hosted [Keycloak](https://www.keycloak.org/) instance in Docker, or [ORY Hydra](https://www.ory.sh/hydra), or the hosted free tier of Auth0 / Okta. Even a hand-rolled token-issuer script is acceptable for proof-of-concept — the goal is to wire validation.
- At least one MCP-aware client: Anthropic's Claude Desktop (for stdio transport) or a custom client script (for streamable HTTP).

<!-- needs-research: MCP SDK APIs, the exact authorization draft section, and current transport semantics evolve. Verify against the pinned specification revision and SDK release before writing the server. -->

## Requirements

Six deliverables.

### Deliverable A — the MCP server skeleton

Author an MCP server exposing at least one Tool (`chat_completion` is the canonical one; add `embed_text` if your fleet exposes embeddings):

```python
# Representative shape; the exact SDK API drifts.
from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult
import httpx, json

server = Server("inference-mcp", version="0.1.0")

INFERENCE_URL = "http://engine.internal:8000/v1/chat/completions"

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="chat_completion",
            description="Generate a chat completion for a conversation.",
            inputSchema={
                "type": "object",
                "additionalProperties": False,
                "required": ["model", "messages"],
                "properties": {
                    "model": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["role", "content"],
                            "properties": {
                                "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4096},
                },
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "chat_completion":
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(INFERENCE_URL, json=arguments)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
    raise ValueError(f"Unknown tool: {name}")
```

Deploy on the Streamable HTTP transport (preferred for remote; stdio is also acceptable for a local demo). Document which transport you used and why.

### Deliverable B — OAuth 2.1 Resource Server validation

Add middleware that validates bearer tokens on every MCP request:

- Fetch the authorization server's JWKS.
- Validate the JWT signature, expiration, audience (`aud` must match this server's identifier), and issuer.
- Extract scopes; enforce `inference:chat` scope for the `chat_completion` tool.
- Reject with JSON-RPC error `-32001` (or equivalent per the spec) on auth failure.

Document:

- The authorization server URL.
- The JWKS endpoint.
- The expected `aud` and `iss` values.
- The scope → tool mapping.
- Cache the JWKS with a reasonable TTL (5 minutes); support JWKS rotation without server restart.

### Deliverable C — the client round-trip

Drive the server with an MCP-aware client. Options:

- **Claude Desktop** — configure the MCP server in Claude Desktop's settings; have Claude call the tool via its interactive UI; capture the request / response logs.
- **Custom client** — write a Python script that opens a session, calls `initialize`, `listTools`, `tools/call`, and prints every message.

Capture the full transcript:

```
→ initialize   {clientInfo: {...}, capabilities: {...}}
← initialize   {serverInfo: {name: "inference-mcp", version: "0.1.0"}, capabilities: {tools: {}}}
→ listTools   {}
← listTools   {tools: [...]}
→ tools/call   {name: "chat_completion", arguments: {...}}
← tools/call   {content: [{type: "text", text: "..."}]}
→ close
```

The transcript is the proof.

### Deliverable D — capability discovery + change notification

If the SDK supports it, implement capability-change notification:

- Add a second tool at runtime (e.g., toggle `embed_text` on and off via an admin endpoint).
- Emit `notifications/tools/listChanged`.
- Verify the client re-fetches `listTools` and picks up the change.

If the SDK doesn't support dynamic tools cleanly, document the workaround (restart the server; the client reconnects).

### Deliverable E — session + cancellation behaviour

Test the operational behaviour from chapter 05:

1. **Session liveness.** Open a session, call `chat_completion`, wait 60 seconds, call again. Verify the session stays alive (heartbeats or keep-alive at the transport level).
2. **Cancellation mid-call.** Start a long chat completion (`max_tokens: 2048`); send `notifications/cancelled`; verify the engine's cancel path fires and KV is released. Reuse the exercise 02 verification approach.
3. **Server drain.** Trigger a graceful shutdown on the MCP server; verify active sessions receive a close notification, not a connection drop.

### Deliverable F — the runbook

Publish a one-page runbook:

- How to start the MCP server (command, config).
- How to point a client at it (URL, auth setup).
- How to add a new tool (schema, handler, scope).
- How to rotate the authorization server's signing keys.
- How to debug a session failure (log location, grep pattern).
- How to grant / revoke scopes for a client.
- How this MCP endpoint coexists with the OpenAI-compatible endpoint on the same ingress.

## Starter guidance

- **Verify the SDK version matches the spec revision.** Both move; incompatibility produces confusing errors.
- **Start with no auth.** Get the server running and the client connecting over plain HTTP; add OAuth validation second. Easier debugging.
- **Use a self-signed JWT issuer for testing.** A 20-line Python script can sign tokens with a test key; configure the server to validate against that key. Much easier than running Keycloak locally.
- **Capture the JSON-RPC wire traffic.** `tcpdump` + a JSON dissector, or the MCP SDK's built-in logging. The wire trace is the forensic artefact.
- **Expose this MCP endpoint on a separate port / path from the OpenAI-compatible one.** They are different surfaces; mixing them in one ingress is possible but harder to reason about.
- **Test with one tool first.** Capability discovery, auth, and session management are all testable with one Tool; add more once the pattern works.

## Fallbacks for constrained environments

- **No MCP-aware client.** The custom Python client with the SDK's client-side helpers is fine; Claude Desktop is a nice demo but not required.
- **No OAuth 2.1 server.** Use a hand-rolled test-key JWT issuer; document the production migration path (swap test issuer for real).
- **SDK missing a feature (dynamic tools, progress notifications, etc.).** Document the gap; test what the SDK supports; note the production-blocking features.
- **No Streamable HTTP support in the SDK yet.** Use stdio transport for the demo; note the production transport required.

## Acceptance criteria

1. **Deliverable A** MCP server runs; `listTools` returns the declared capabilities; `tools/call` forwards to the engine and returns the result.
2. **Deliverable B** OAuth 2.1 validation rejects unauthenticated calls (JSON-RPC error or protocol-equivalent) and accepts calls with a valid token in the correct scope; audience and issuer validation are correct.
3. **Deliverable C** end-to-end transcript captured, showing `initialize` → `listTools` → `tools/call` → response.
4. **Deliverable D** capability-change notification tested, or workaround documented.
5. **Deliverable E** session liveness, cancellation, and drain all behave per chapter 05's expectations.
6. **Deliverable F** runbook published, including the coexistence story with the OpenAI-compatible endpoint.

## Common failure modes

- **Token audience not validated.** The server accepts any token signed by the issuer, including tokens meant for other services. Enforce `aud`.
- **JWKS not cached, hammers the auth server.** Every validation fetches the JWKS; at 100 QPS the auth server sees 100 QPS of JWKS fetches. Cache with a reasonable TTL.
- **Scope mapping in code, not config.** A tool's required scope is hard-coded; adding a tool requires a code change. Externalise the scope → tool map.
- **Session state in process memory.** A pod restart drops all sessions; horizontal scaling is broken. Stateless server + token-carried identity + shared rate-limit cache. (Full session state replication is usually overkill; reconnect-on-restart is fine.)
- **Transport not matching spec revision.** An early spec used SSE + POST pair; current uses Streamable HTTP. Clients and servers at different revisions don't talk. Pin both.
- **Cancellation doesn't propagate.** The MCP spec's cancel notification is received by the server but the server doesn't forward to the engine's abort path. Verify end-to-end.

## Keep it

The MCP server, the OAuth wiring, the round-trip transcript, and the runbook are the fleet's MCP-exposure artefact. The ingress from exercise 06 may host the MCP endpoint next to the OpenAI-compatible endpoint; the compatibility harness from exercise 01 does not cover MCP (different protocol) but a parallel MCP-protocol test suite grows from this exercise. Chapter 07 cites the L25 application-developer peer as the client-side consumer.
