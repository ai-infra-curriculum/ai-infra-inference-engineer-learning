# Chapter 02 — Streaming at Production Quality: SSE, WebSocket, Back-Pressure, Cancellation

> SSE is the HTTP streaming primitive that holds up at scale. WebSocket is the full-duplex alternative for audio and bidirectional protocols. Both carry inference output; neither works correctly if the ingress buffers before it flushes, if the engine ignores client disconnect, or if the token-flush cadence turns a 200 ms TTFT into a 1.2 s TTFT because of a Nagle-shaped batching mistake. This chapter writes the transport discipline.

## Motivation

An on-call page: "chat TTFT P99 is 1.2 s on the vLLM fleet, but the engine's internal `time_to_first_token_seconds` histogram says P99 is 240 ms. The engine is fine. The gap is in the ingress. Where?"

The engine is right, the ingress is wrong, and the gap has one of five causes:

- The reverse proxy (nginx / Envoy) is buffering the response body before flushing.
- The ingress framework (FastAPI / Starlette) is batching writes behind an application-level buffer.
- The streaming middleware is flushing in `k`-token batches instead of per-token, with `k` set too high.
- The TLS termination point is accumulating a TLS record before encrypting (TLS record is up to 16 KiB; a 10 KiB token blob delays the flush until the record fills).
- The engine's own SSE emitter is coalescing tokens inside a `k`-ms window.

Each is a one-config-flag fix once named. The on-call engineer who does not know the five has an afternoon to debug it; the engineer who does has a minute.

By the end of this chapter you can:

- State the SSE framing contract per the WHATWG HTML Living Standard and emit SSE frames correctly.
- Set the flush discipline through the ingress stack (framework → reverse proxy → TLS → client) so TTFT is not inflated.
- Detect a client disconnect mid-stream and cancel the in-flight engine request to recover KV budget.
- Decide between SSE and WebSocket per workload and configure the alternative when SSE is insufficient.
- Diagnose the five causes above with a bisection, not a guess.

## SSE in one page — the WHATWG framing

Server-Sent Events is defined in the [WHATWG HTML Living Standard, "Server-sent events" section](https://html.spec.whatwg.org/multipage/server-sent-events.html). The transport is unidirectional (server → client over a long-lived HTTP response), text-only, UTF-8, and framed by newlines. The response headers:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache, no-transform
Connection: keep-alive
X-Accel-Buffering: no
```

The frames:

- `data: <payload>\n` — the event payload (one or more `data:` lines concatenate with `\n` as separator).
- `event: <name>\n` — the event name (defaults to `message`); the client-side `EventSource` dispatches to a named handler.
- `id: <id>\n` — the Last-Event-ID; the client echoes it on reconnect in the `Last-Event-ID` header so the server can resume.
- `retry: <ms>\n` — the client-side reconnection delay in milliseconds (clients use this when the connection drops).
- `:<comment>\n` — a comment line; servers use these as heartbeats.
- Empty line (`\n`) — frame separator. **This is non-optional.** A frame without a trailing empty line is a frame that may never be dispatched to the client's `onmessage` handler.

Three disciplines to carry:

- **Flush after the empty line.** The ingress must call `flush()` on the response writer after the terminating `\n\n` so the frame leaves the local buffer. A framework that accumulates writes until a size threshold defeats the point.
- **Emit `data: [DONE]\n\n` on completion.** The OpenAI convention (chapter 01) is a sentinel the client keys on; the WHATWG spec has no equivalent, so a reader that watches for connection close works too, but the sentinel is the portable path.
- **No binary.** SSE is UTF-8 text. Audio chunks ride on WebSocket or on `multipart/mixed`, not on SSE.

### A worked SSE frame for chat completions

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1762128000,"model":"llama3.1-70b-instruct","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" answer"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" 4."},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

Each frame is one `data:` line (JSON on one logical line, newlines escaped inside the string) followed by a blank line. The blank line's absence is the single most common framing bug.

### A minimal Python emitter

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json, asyncio

app = FastAPI()

async def chat_stream(req_id: str, model: str, token_source):
    # token_source is an async generator yielding (text, finish_reason) pairs.
    async for token, finish in token_source:
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token} if token else {},
                "finish_reason": finish,
            }],
        }
        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: Request, body: dict):
    token_source = engine_submit(body)  # async generator from the engine client
    return StreamingResponse(
        chat_stream(request_id(request), body["model"], token_source),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

Three correctness properties this snippet gets right:

- The media type is `text/event-stream`. (A `Content-Type: application/json` on a streaming response is wrong — proxies may buffer it.)
- `X-Accel-Buffering: no` disables nginx response buffering in the common deployment where nginx sits in front; chapter below returns to the proxy layer.
- The frame separator is `\n\n`; `json.dumps` with `separators=(',', ':')` keeps the payload one line.

What this snippet does not do: handle client disconnect (next section), emit heartbeats under long idle (section after), or respect flush cadence (section after that).

## Client disconnect and request cancellation

A streaming inference request can take 15 seconds to emit 500 tokens. In 15 seconds the client may give up — the browser tab closed, the mobile app backgrounded, the agent process killed. The server must detect the disconnect and cancel the in-flight engine request. If it doesn't, the engine keeps decoding into a dropped TCP stream, burning HBM for a KV cache no one is reading from, until the context limit or the server times out.

Three detection paths:

- **ASGI `request.is_disconnected()`.** FastAPI / Starlette expose an awaitable that polls the TCP connection state. Check it on every token (cheap) or every N tokens (cheaper).
- **Write-side failure.** The write to the response buffer fails when the client closes. Catch `ConnectionError` / `ClientDisconnect` and treat as cancellation.
- **Keepalive timeout.** If the client stops reading but doesn't close, the TCP keepalive eventually fires; the server-side timeout surfaces as a write failure. Tune the keepalive to a value that matches your acceptable "stuck client" window.

The cancellation path, once detected:

```python
async def chat_stream_with_cancel(req_id, model, token_source, request):
    try:
        async for token, finish in token_source:
            if await request.is_disconnected():
                await token_source.aclose()         # trigger engine-side cancel
                return
            yield build_frame(req_id, model, token, finish)
        yield "data: [DONE]\n\n"
    finally:
        # Belt-and-braces: cancel the engine request on any exit path.
        await token_source.aclose()
```

Three correctness properties:

- `aclose()` on the engine client propagates through to the engine's abort hook (vLLM's `AsyncLLMEngine.abort`, SGLang's abort, TRT-LLM / Triton's `cancel` endpoint). Verify the engine actually frees the KV slot — a cancel that doesn't release KV is a leak.
- The `finally` block guarantees the engine call is torn down on exception paths (schema error, timeout, upstream disconnect).
- Checking `is_disconnected()` on every token is O(TCP syscall) per token. For a 500-token response that is 500 syscalls; usually fine, but if you measure CPU regress, check every 10 tokens or every 100 ms.

### KV release — the back-end contract

The engine's abort is a protocol:

1. The ingress calls the engine client's abort / close method with the request ID.
2. The engine's scheduler removes the sequence from the batch at the next iteration boundary.
3. The KV allocator frees the sequence's blocks (chapter on KV eviction in module 104).
4. The scheduler emits an abort event for metrics (`abort_rate` as an operational axis, module 110 chapter 01).

A buggy engine integration that calls abort but doesn't observe free is a leak. The compatibility harness from chapter 01 should include a cancellation case that asserts KV-usage decreases after an abort.

## Back-pressure — engine ⇢ ingress ⇢ client

An inference engine produces tokens at a rate the client may not consume. The engine's steady-state is ~50-200 tokens/second per sequence; a slow client (a mobile app on a cell network, an agent that pauses to think) may read at 10 tokens/second. The intermediary buffers fill; left unmanaged, the ingress process's memory grows per slow client until it OOMs.

Three back-pressure patterns:

- **Natural TCP back-pressure.** If the ingress writes to a socket whose receive window is full, the write blocks. An `async` ingress that `await`s the write naturally pauses token consumption from the engine. If the engine client also respects its own back-pressure (it should), the engine stops emitting tokens into a buffer no one reads.
- **Bounded queue between engine and ingress.** The ingress keeps a bounded `asyncio.Queue` of pending tokens; when the queue fills, the engine client's producer coroutine blocks. The queue size is a knob: 1 (strict per-token hand-off, lowest memory, highest sync overhead) through 16 (small buffer, forgiving of jitter) through 128 (large buffer, hides slow consumers until the queue fills).
- **Drop with backfill.** For audio or video output where latency-of-last-chunk matters more than every-chunk-delivered, drop oldest-first when the queue fills. Rarely correct for text — a dropped token is a corrupted response.

The default is natural TCP back-pressure with a small bounded queue (≤ 16 entries). A client that reads slowly then pauses the engine; the engine's `max-num-seqs` cap bounds how many such slow sequences can accumulate; the admission layer (module 103) rejects when the fleet runs out of KV. The three mechanisms compose.

## Flush cadence — the hidden TTFT budget

Even with correct SSE framing, correct cancellation, and correct back-pressure, TTFT can be inflated by the flush cadence. The culprits:

### 1. The reverse proxy

**nginx** buffers streaming responses by default (`proxy_buffering on`). The fix is:

```
location /v1/ {
    proxy_pass http://inference-upstream;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    chunked_transfer_encoding on;
}
```

Or per-response, by setting `X-Accel-Buffering: no` on the response (nginx reads this header and bypasses buffering). The `X-Accel-Buffering` convention has been documented in [the nginx X-Accel docs](https://www.nginx.com/resources/wiki/start/topics/examples/x-accel/); verify your nginx version's handling.

**Envoy** does not buffer responses by default and streams through once `response_headers_mode` is set to streaming; verify by inspecting the `envoy.filters.http.buffer` filter is not enabled on the route.

**AWS ALB** buffers by default. Direct NLB or a Kubernetes `Service` of type `LoadBalancer` with session-affinity are the common alternatives for inference workloads; a CloudFront in front of ALB for inference streaming is usually wrong. Verify per-cloud; the specifics drift.

<!-- needs-research: cloud-provider load-balancer streaming behaviour (ALB vs. NLB vs. GCP global LB vs. Azure Front Door) drifts across releases. Confirm against the provider's current docs for the streaming path. -->

### 2. The ingress framework

**FastAPI / Starlette's** `StreamingResponse` writes each chunk from the generator to the response body with an implicit `send()` per yield; the ASGI server (Uvicorn, Hypercorn) relays the chunk to the socket. In the default configuration this is per-chunk; no explicit flush is needed. The pitfall: a middleware that reads the full body before passing it on (gzip middleware with the default compression strategy, logging middleware that captures the body for audit) breaks streaming by buffering. Audit the middleware chain.

**gRPC-translated-to-HTTP** stacks (grpc-gateway, Connect, Triton's gRPC gateway) can introduce their own buffering. Verify the gateway is configured for streaming.

### 3. TLS record coalescing

A TLS record is up to 16 KiB (TLS 1.3). The TLS stack may wait for the record to fill before encrypting and sending. A single token (a few bytes) sits in the TLS buffer. The fix: set the TLS stack's `TCP_NODELAY` and disable Nagle; most modern stacks do this correctly for keep-alive connections, but a hand-rolled TLS termination may not. If you suspect TLS coalescing, test by comparing TTFT with plain-HTTP (dev) and HTTPS (prod) on the same engine; a ≥ 100 ms gap is suspicious.

### 4. Token-flush cadence inside the engine or ingress

Some engines emit SSE in `k`-token batches to reduce ingress CPU. The flag varies: vLLM has `--stream-granularity-tokens` or similar on some releases; SGLang's streaming defaults to per-token but has a batching mode; TRT-LLM / Triton's `tensorrtllm_backend` has a `streaming_batching` config. Verify against the pinned release.

The ingress may also batch — a middleware that aggregates `data:` lines into one `write()` call every 50 ms coalesces frames. For text, per-token is fine; for audio (chapter below) a small coalesce window may actually help by reducing header overhead.

### 5. Nagle and `TCP_NODELAY`

TCP Nagle's algorithm (RFC 896) buffers small writes until an ACK arrives. For a per-token write of a few bytes, Nagle will delay the write by one RTT. `TCP_NODELAY` disables Nagle; most async frameworks set it by default on streaming connections. Verify by inspecting the socket (`getsockopt(TCP_NODELAY)` returns 1). If a middle box (a sidecar, a mesh proxy) terminates and re-establishes the connection with Nagle enabled, the flush delay reappears.

### The bisection

If TTFT is inflated and the engine-side histogram says it shouldn't be:

1. Measure TTFT inside the engine's own emitter (log the wall time at first `data:` frame yield).
2. Measure TTFT at the ingress process's write (wrap the ASGI `send` callable).
3. Measure TTFT at the reverse proxy's egress (nginx `$request_time` plus the first-byte log).
4. Measure TTFT at the client (browser DevTools, a `curl -N` with a timing script).

The step where the number jumps is the culprit. The fix is one of the five bullets above.

## Heartbeats and keep-alive

A long idle (model is thinking on a 1k-token reasoning trace with no interim output) can cause intermediaries to time out the connection. Three defences:

- **SSE comment frames.** Emit `:\n\n` every 10-15 seconds. The client's `EventSource` ignores comments; proxies see traffic.
- **Response headers that disable idle-timeout.** `Connection: keep-alive` on the response; `proxy_read_timeout 600s` on nginx; `idle_timeout` on cloud LBs set to the maximum tolerable.
- **TCP keepalive at the socket level.** `SO_KEEPALIVE` on the socket catches dead peers.

For reasoning models that may think for a minute before emitting the first visible token, a heartbeat emitted during the "thinking" phase is the right path. The OpenAI `/v1/responses` endpoint explicitly emits typed events (`response.in_progress`, `response.output_text.delta`) to make the thinking phase visible; chat completions on reasoning models often emit nothing between request admission and first token. A naive client that assumes "no bytes in 30 seconds means dead" cancels a healthy request.

## WebSocket — when and why

SSE is unidirectional, text-only, HTTP-shaped. For workloads that need:

- **Bidirectional audio streaming** (voice assistants, live transcription with speaker interrupt).
- **Full-duplex token exchange** (an agent that streams user input and model output simultaneously).
- **Binary payloads** on the same connection as text (combined text + image chunks).

WebSocket (RFC 6455) is the right transport. The handshake is an HTTP/1.1 `Upgrade: websocket`; after upgrade the connection is a framed bidirectional byte stream.

### A minimal WebSocket inference endpoint

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/v1/chat/completions/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        request = await ws.receive_json()  # typed request shape per your contract
        token_source = engine_submit(request)
        async for token, finish in token_source:
            await ws.send_json(build_chunk(token, finish))
        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        await token_source.aclose()
    finally:
        await token_source.aclose()
        await ws.close()
```

Three disciplines:

- **Define the message schema per direction.** Client → server (incremental audio bytes, text turns, control messages); server → client (token deltas, speech output chunks, control / error messages). WebSocket is a byte transport; the protocol on top is yours to design.
- **Ping / pong for liveness.** The WebSocket spec defines `Ping` / `Pong` frames; use them for keepalive. Most frameworks handle this.
- **Back-pressure is harder.** TCP back-pressure applies, but there's no explicit flow-control at the WebSocket layer. Design the client / server message protocol to carry application-level flow control (a `pause` / `resume` message, or an ack-per-N-chunks window).

### Where SSE still wins

- **HTTP/2 and HTTP/3 compatibility.** SSE rides on HTTP; a modern HTTP/2 or HTTP/3 stack multiplexes streams natively. WebSocket requires a hijack and is typically a single connection per stream.
- **CDN compatibility.** Many CDNs pass SSE but refuse to proxy WebSocket, or require special configuration.
- **Simpler semantics.** No handshake, no close frames, no bidirectional protocol design.

Rule of thumb: *text out, SSE. Audio in and out, WebSocket. Binary mixed with text, WebSocket.*

## Three failure modes

### Hanging requests that never close

A client disconnected, the ingress didn't detect, the engine kept decoding. In-flight sequences accumulate on the engine until the KV budget is exhausted and admission starts rejecting. The dashboard shows admission-reject rising while the fleet looks healthy. The fix: wire `is_disconnected()` into the token loop; verify abort releases KV.

### TTFT inflated at the proxy

The engine says TTFT P99 is 240 ms, the client sees 1.2 s. The proxy's buffering is on. The fix: `X-Accel-Buffering: no` on the response headers; `proxy_buffering off` in the nginx config; verify Envoy's buffer filter is off on the route.

### Streamed usage missing

A billing path depends on the final `usage` object. The client requests streaming with `stream_options.include_usage: true`; the engine does not emit it. The fix: either enable the feature on the engine (chapter 01's drift point #8), or compute token counts in the ingress by tokenising the request and the aggregated response.

## Summary

- SSE is the primary streaming transport for inference text output; its framing is WHATWG-defined — `text/event-stream`, `data: ...\n\n` frames, `data: [DONE]` as OpenAI's completion sentinel.
- The ingress stack has five places that can inflate TTFT: reverse proxy buffering, framework batching, TLS record coalescing, engine-side token-flush cadence, Nagle. Each is one config flag away from being correct.
- Client disconnect must propagate to engine cancellation; a cancel that doesn't free KV is a leak. The compatibility harness should include a cancellation case.
- Back-pressure composes: TCP back-pressure, a bounded ingress queue, and the engine's admission control together bound memory under slow clients.
- Heartbeats defend against intermediary idle timeouts; reasoning models with long thinking phases need them.
- WebSocket is the right transport for bidirectional audio and binary payloads; SSE wins everywhere else by simplicity and CDN compatibility.
