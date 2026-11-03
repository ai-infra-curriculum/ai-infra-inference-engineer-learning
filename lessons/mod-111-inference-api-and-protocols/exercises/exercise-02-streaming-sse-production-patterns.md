# Exercise 02 — Streaming SSE Production Patterns

**Time:** 4 hours · **Output:** one SSE streaming proxy (code), one adversarial client test suite, one documented flush-discipline runbook for your fleet's ingress stack

## Goal

Make [Chapter 02](../02-streaming-sse-and-websocket.md) concrete by **implementing an SSE streaming proxy** that sits in front of an engine, with correct frame discipline, back-pressure, client-disconnect cancellation, and heartbeats; then prove correctness against a scripted set of adversarial client patterns that model real-world failure shapes.

## Problem statement

Chapter 02 names the five places that inflate TTFT (proxy buffering, framework batching, TLS record coalescing, engine flush cadence, Nagle). The exercise builds a proxy that gets each one right, then stresses it with the client patterns that break naive implementations.

You will author a FastAPI (or similar) proxy that forwards SSE from an engine to a client; configure the reverse proxy / framework / TLS termination stack so flush is per-token; wire the disconnect-and-cancel path with KV release verification; and run an adversarial client suite that models slow readers, early close, mid-stream abort, and idle long-tail behaviour.

## Prerequisites

- Chapter 02 of this module read.
- A running inference engine exposing `/v1/chat/completions` with SSE streaming — any of vLLM, SGLang, Triton + TRT-LLM.
- The compatibility harness from exercise 01 (reuse the SSE parsing).
- Python 3.11+, `httpx` or `aiohttp` as the engine client, `fastapi` + `uvicorn` as the proxy framework.
- A reverse-proxy stack for a realistic ingress — nginx or Envoy running in front of the proxy.
- Access to the engine's metrics endpoint to verify KV release (module 110 chapter 01's `kv_cache_usage_perc` or equivalent).

## Requirements

Six deliverables.

### Deliverable A — the SSE streaming proxy

A FastAPI endpoint that accepts `POST /v1/chat/completions`, forwards to the engine with `stream: true`, and relays SSE frames back. Minimum skeleton:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx, asyncio, time

app = FastAPI()
ENGINE_URL = "http://engine.internal:8000/v1/chat/completions"

async def relay_stream(request: Request, body: dict):
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as client:
        async with client.stream("POST", ENGINE_URL, json=body) as resp:
            if resp.status_code != 200:
                yield build_error_frame(resp)
                return
            async for line in resp.aiter_lines():
                if await request.is_disconnected():
                    # Propagate cancellation: close the engine stream.
                    await resp.aclose()
                    return
                yield line + "\n"
            # Terminator; the engine should emit [DONE], but belt-and-braces.

@app.post("/v1/chat/completions")
async def chat(request: Request, body: dict):
    return StreamingResponse(
        relay_stream(request, body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

Three correctness properties to verify:

- Frames reach the client per-token (not batched).
- Client disconnect propagates to engine abort within one iteration.
- KV cache usage drops after abort (confirm via engine metrics).

### Deliverable B — the flush discipline runbook

Audit your proxy stack and document the flush setting at each layer:

```
Layer            | Setting                                       | Verified
-----------------|-----------------------------------------------|----------
Engine           | per-token SSE emit (verify no --stream-batch) | Y
FastAPI          | StreamingResponse yields per line             | Y
Uvicorn          | ASGI send-per-chunk (default)                 | Y
nginx            | proxy_buffering off; X-Accel-Buffering: no    | Y
TLS termination  | TCP_NODELAY on; no custom record coalescing   | Y
Cloud LB (if any)| streaming passthrough; no response buffering  | Y
```

For each row, the "verified" column cites the config file / flag / measurement that proves it. The runbook is the artefact an on-call engineer reads when TTFT inflates at the ingress.

### Deliverable C — heartbeats for long idle

Implement heartbeat comment frames for reasoning / long-thinking workloads:

```python
async def relay_with_heartbeat(request, body):
    heartbeat = ":\n\n".encode()
    last_data = time.monotonic()
    HEARTBEAT_INTERVAL = 10.0  # seconds
    # ...
    while not done:
        try:
            frame = await asyncio.wait_for(next_frame(), timeout=HEARTBEAT_INTERVAL)
            yield frame
            last_data = time.monotonic()
        except asyncio.TimeoutError:
            yield heartbeat
```

Verify: open a long-idle stream (request that generates a 1 token response but with a `max_tokens` that encourages the engine to take a long time; or use a reasoning model); confirm heartbeat frames arrive every 10 s; confirm intermediaries (nginx, cloud LB) do not time out.

### Deliverable D — the adversarial client test suite

Script five adversarial client patterns:

1. **Slow reader.** The client reads one frame every 500 ms. Verify back-pressure holds — the proxy's memory does not grow unboundedly; the engine eventually pauses (via admission or internal back-pressure).
2. **Early close.** The client reads three frames and closes the connection. Verify the engine cancel fires within one iteration; verify the KV cache for the aborted sequence is freed within 500 ms (measure via engine metrics).
3. **Mid-stream abort.** The client reads 50 frames then issues an abort (TCP RST or a client-SDK cancel). Same verification as early close.
4. **Idle long-tail.** The client is slow by not reading at all for 30 seconds (simulated "tab in background"), then resumes. Verify heartbeat frames do not accumulate out of order; verify the stream resumes correctly.
5. **Network partition.** Simulate a network partition between the proxy and the client (`iptables` drop rule or a tc qdisc). Verify the proxy's TCP keepalive detects the dead peer within the configured timeout; verify the engine cancel fires.

For each pattern, capture:

- Proxy's memory / CPU during the test.
- Engine's KV usage before, during, after.
- The exact point at which the engine's abort hook fired.
- Any residual state (hanging coroutines, leaked connections).

### Deliverable E — bisection of a synthetic TTFT inflation

Induce a TTFT regression at the ingress and bisect it:

1. Enable `proxy_buffering on` in nginx; observe client-visible TTFT jump.
2. Disable; observe TTFT recover.
3. Enable `gzip` middleware in FastAPI with a buffer threshold; observe TTFT regress.
4. Disable; observe recovery.
5. Add a middleware that aggregates writes with a 50ms window; observe regress; verify the five-step bisection from chapter 02 reads the correct stage.

Document the bisection sequence and the per-step TTFT numbers in the runbook.

### Deliverable F — the compatibility extension

Add the following cases to the exercise 01 harness:

- Cancellation test: the harness opens a stream, reads N frames, closes, polls KV metrics to confirm release within a timeout.
- Heartbeat test: a stream that would otherwise be idle for > 10 s; verify heartbeat frames arrive.
- Slow-reader test: the harness throttles its read; verify no OOM, no 500s.
- Mid-stream error test: force the engine to emit an error mid-stream; verify the proxy relays the error chunk and closes cleanly.

## Starter guidance

- **Measure TTFT at every layer.** Instrument the engine (first token out), the proxy (first chunk yielded), the proxy's TCP send (write time), the client (first chunk received). The chapter-02 bisection relies on these.
- **Use `curl -N` as a baseline client.** No fancy SDK; the raw HTTP stream reveals framing bugs immediately.
- **Monitor engine KV during tests.** The KV metric going up during a test and not coming down after abort is the leak signal; the runbook must verify it's clean.
- **Test over HTTPS, not just HTTP.** TLS record coalescing is a real source of inflation; the HTTP-only test may pass while the HTTPS production path fails.
- **Keep heartbeat interval realistic.** 10 s is a reasonable default; 1 s is noise, 60 s may trigger intermediary timeouts.
- **Verify `X-Accel-Buffering: no` is being read by nginx.** Some nginx builds / configurations ignore the header; check the nginx access log or add an explicit `proxy_buffering off` on the location.

## Fallbacks for constrained environments

- **No nginx.** Run the proxy standalone; document the runbook for your actual ingress stack.
- **No access to engine metrics.** Infer KV release from admission behaviour — admit a long-prompt request right after the cancellation; it should be admitted immediately (quick KV availability indicates the aborted sequence's slots were freed).
- **No HTTPS termination in the test harness.** Still valuable; document the TLS caveat in the runbook so the production path gets the extra check.
- **No cloud LB.** Skip the "cloud LB streaming passthrough" row; the four other layers still cover 90 % of the drift.

## Acceptance criteria

1. **Deliverable A** proxy runs; relays SSE per token; propagates cancel to the engine.
2. **Deliverable B** runbook has an entry per layer with a verified flag / config file and a measurement that proves the flush discipline.
3. **Deliverable C** heartbeat works; intermediaries do not time out an idle stream.
4. **Deliverable D** all five adversarial patterns pass; KV release verified; no leaks.
5. **Deliverable E** bisection matches chapter 02 — each induced inflation is readable at the correct layer.
6. **Deliverable F** new harness cases pass against the pinned engine.

## Common failure modes

- **Buffering middleware.** Gzip middleware with default compression strategy buffers the full response. Audit the middleware chain; gzip is usually not useful on text streams and should be disabled.
- **Cancel-without-release.** The engine honours the abort call but holds the KV for several seconds (garbage-collection lag). File against the engine; add a wait + poll to the cancellation verification; document the delay.
- **Trace propagation lost on relay.** The proxy drops the client's `traceparent` when forwarding. The engine's spans are detached from the ingress's. Audit the headers forwarded; whitelist `traceparent` and `x-request-id`.
- **Heartbeat as a `data:` frame.** A `data:\n\n` frame reaches the client's `EventSource.onmessage` handler with an empty payload, which may confuse parsers. Use `:\n\n` (comment); clients ignore it.
- **TLS over HTTP/1.1 vs. HTTP/2.** The two have different coalescing behaviour; measure on the actual production protocol.

## Keep it

The proxy, the runbook, the adversarial test suite, and the harness extensions are the fleet's streaming artefact. Exercise 06 (ingress authoring) layers auth, rate-limit, and idempotency on top of this proxy; exercise 04 (tool calling) adds the mid-stream tool-call-phase correctness tests.
