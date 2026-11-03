# Exercise 01 — TTFT / TPOT / ITL Decomposition and Trace Wiring

**Time:** 4 hours · **Output:** one running engine with decomposed-latency instrumentation, one OpenTelemetry trace that covers admission → prefill → decode → stream-close, one Grafana dashboard with the four latency panels, one histogram-bucket sheet

## Goal

Make [Chapter 01](../01-inference-axis-set.md) concrete by **wiring and emitting** the six latency axes (E2E, TTFT, queue, prefill, TPOT, ITL) against a running engine, correlating them in a single OpenTelemetry trace tree, and rendering them in a dashboard that an on-call engineer can bisect against during a latency regression.

## Problem statement

The chapter named twelve axes; the exercise focuses on the six latency axes because they are the ones you must bisect during an incident. The other six (scheduler state, DCGM, NCCL) are picked up in exercise 03 when you wire the base stack.

Today you will stand up a representative engine, confirm it emits the latency histograms you need, add the OpenTelemetry middleware that produces a trace tree per request, and prove the decomposition works by inducing a known-regression traffic shape (short-prompt vs. long-prompt) and reading the correct bisection off the dashboard and the trace.

## Prerequisites

- Chapter 01 of this module read.
- A running inference engine you can touch — vLLM, SGLang, or Triton + TRT-LLM. If you don't have one from module 102, follow the engine's quick-start to serve a small chat model (1B – 13B class; the decomposition is identical at any size).
- Prometheus and Grafana running (local Docker Compose is fine; a `kube-prometheus-stack` install is better).
- An OpenTelemetry Collector reachable from the engine host; a Jaeger or Tempo backend for trace visualisation.
- Python 3.11+ for middleware shim work; the OpenTelemetry Python SDK (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`).

## Requirements

Four deliverables.

### Deliverable A — the metric inventory

A table that maps each chapter-01 latency axis to what your pinned engine exposes versus what you must synthesise. Fill it against the engine's `/metrics` endpoint, not against this chapter's example names:

```
| Chapter 01 axis         | Engine metric name (or "synthesise")              | Unit    | Shape     | Notes |
|-------------------------|---------------------------------------------------|---------|-----------|-------|
| E2E latency             | {engine:request_duration_seconds | synth}         | seconds | histogram | API-side if engine doesn't expose |
| TTFT                    | {engine:time_to_first_token_seconds | synth}      | seconds | histogram | first-byte in stream |
| Queue time              | {engine:queue_seconds | synth}                    | seconds | histogram | subset of TTFT |
| Prefill time            | {engine:prefill_seconds | synth}                  | seconds | histogram | subset of TTFT |
| TPOT                    | {engine:time_per_output_token_seconds | synth}    | s/token | histogram | first-token to close, averaged |
| ITL                     | {engine:inter_token_seconds | synth}              | s/token | histogram | per-token jitter |
```

Record the exact metric names as of the pinned engine release; note gaps (anything marked `synth` becomes a middleware task below).

### Deliverable B — the middleware shim (if needed)

For any axis marked `synth` in Deliverable A, add middleware in front of the engine that measures it. A representative FastAPI shim:

```python
from fastapi import FastAPI, Request
from prometheus_client import Histogram, make_asgi_app
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import time

TTFT = Histogram("inference_ttft_seconds", "TTFT", 
                 ["workload_class"], 
                 buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 10, 20])
ITL  = Histogram("inference_itl_seconds",  "ITL per token",
                 ["workload_class"],
                 buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5])
# ... TPOT, queue, prefill similarly

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True))
)
tracer = trace.get_tracer("inference-middleware")

app = FastAPI()
app.mount("/metrics", make_asgi_app())

@app.post("/v1/chat/completions")
async def chat(request: Request):
    workload_class = classify(request)       # body heuristic or client header
    t0 = time.monotonic()
    with tracer.start_as_current_span("admit") as admit_span:
        admit_span.set_attribute("workload_class", workload_class)
        # ... forward to engine, handle streaming ...
```

Three disciplines:

- Use `time.monotonic()`, not `time.time()`.
- Set histogram buckets per the chapter-03 recommendations (log-spaced, inference-shaped).
- Label by `workload_class`, never `tenant_id` on hot path.

### Deliverable C — the OpenTelemetry trace tree

A single request produces a span tree. Working trees look like:

```
client_request (HTTP/1.1 POST /v1/chat/completions)
├── admit
│   └── classify_workload
├── engine.request
│   ├── queue
│   ├── prefill
│   │   └── attention_ingest   (optional; from engine if it emits)
│   ├── first_token
│   └── decode_steady
│       └── chunk_* (optional; sampled per 10th chunk)
└── stream_close
```

Verify by:

1. Sending a chat request with a known 500-token output.
2. Opening the resulting trace in Jaeger / Tempo.
3. Confirming the spans cover admission through stream-close and that durations add up to ~E2E.

Export a JSON of one representative trace for the deliverable.

### Deliverable D — the Grafana dashboard

Four panels, one per latency axis, each with `workload_class` in the query:

1. **TTFT panel.** P50 / P95 / P99 overlaid, by class. Vertical line at the chapter-02 SLO target (from exercise 02 if run first; placeholder otherwise).
2. **TTFT decomposition panel.** Stacked area: queue + prefill + first-decode summed to TTFT. Shows where the TTFT budget is spent at any given minute.
3. **TPOT panel.** P50 / P95 / P99, by class, with the SLO line.
4. **ITL distribution panel.** Heatmap over time, by class. Catches preemption spikes that TPOT average hides.

Export the Grafana dashboard JSON for the deliverable.

### Deliverable E — the induced-regression test

Prove the decomposition works by inducing a known pattern and reading it off the dashboard:

- **Test 1 — long prompt.** Send 10 requests with 50k-token prompts; the dashboard should show prefill-time dominating TTFT, queue-time stable.
- **Test 2 — queue saturation.** Send 500 concurrent requests with short prompts; the dashboard should show queue-time climbing, prefill-time stable.
- **Test 3 — preemption.** Over-admit `max-num-seqs` by a factor of 2; the dashboard should show ITL_P99 spiking while TPOT_mean stays flat.

For each test, capture:

- Dashboard screenshot during the test.
- The chapter-01 bisection you read from it ("this looks like a prefill regression because...").
- A representative trace from Jaeger / Tempo supporting the bisection.

## Starter guidance

- **Classification happens at admission, not in the engine.** Engines don't know your workload classes. Either a client header (`X-Workload-Class: chat`) or a body heuristic (short prompt, no tools → chat; tools → agent). Pick one and apply consistently.
- **Buckets matter more than the metric.** A TTFT histogram with HTTP-default buckets has one bucket covering 100 ms – 10 s; `histogram_quantile` returns a bucket boundary, not a real number. Use the chapter-03 bucket sheet.
- **Monotonic clock everywhere.** NTP skew produces negative durations and 20-year spans in trace trees. `time.monotonic()` for Prometheus; the OTel SDK handles its own clock but cross-process spans still carry wall-clock timestamps — document the assumption.
- **Start with one class.** Get `chat` working end-to-end before adding `code` and `long_context`. The multiplication of classes × panels can be confusing to debug.
- **Trace sampling at this stage: 100 %.** Sample everything while wiring; tighten to tail-based sampling in exercise 03.

## Fallbacks for constrained environments

- **No Kubernetes.** Docker Compose with Prometheus + Grafana + OTel Collector + Jaeger is enough; the middleware is portable.
- **Engine doesn't expose fine-grained latency.** The middleware shim is your fallback — you can synthesise TTFT, ITL, queue from the HTTP stream plus timing hooks. Document that the TPOT number comes from middleware and may differ by a few ms from a native engine metric.
- **No OTel Python SDK.** Use the HTTP OTLP endpoint and emit spans with plain `requests.post` to the collector; less ergonomic but testable.
- **No Jaeger / Tempo.** Grafana Tempo in Docker is the easy path; if even that is heavy, use Jaeger's all-in-one Docker image.

## Acceptance criteria

1. **Deliverable A** names every chapter-01 latency axis and maps it to a concrete engine metric or a middleware-synthesised metric; gaps are documented.
2. **Deliverable B** (if any synthesis is needed) exists, is runnable, and emits histograms with log-spaced inference-shaped buckets.
3. **Deliverable C** shows one real trace covering admission → prefill → decode → stream-close, with sub-spans for queue / prefill / first-token.
4. **Deliverable D** has four panels showing live data; each panel's query uses the recording-rule / derived metric name, not a raw histogram bucket.
5. **Deliverable E** runs all three tests, captures the dashboard and trace for each, and reads the correct chapter-01 bisection from the signal.
6. Cardinality analysis: how many series the full label set produces at steady state (should be O(100), not O(10k)); label-set strategy documented.

## Common failure modes

- **TTFT and E2E labelled differently.** A request's TTFT and E2E must be correlatable; use the same label set or a trace-id.
- **Trace spans in different processes have skewed clocks.** Within one process, use the OTel SDK's clock; across processes, document that cross-process span durations are approximate.
- **Sampling too aggressive for induced-regression tests.** Keep 100 % sampling during the tests in Deliverable E; too-low sampling gives you panels with no data.
- **Missing the `workload_class` label.** Every latency histogram should have `workload_class` as a label so chapter 02's SLO slicing works.
- **Over-logging per chunk.** Don't log per-token or per-chunk; the volume is unmanageable. Spans capture what logs would; traces are the forensic surface.
- **Not verifying against a known shape.** A dashboard that looks fine but has never been tested against an induced regression is a dashboard that will fail at 2 AM. Test it in daylight.

## Keep it

The metric inventory, the middleware shim, the Grafana dashboard JSON, and the induced-regression test suite are the fleet's latency-decomposition artefacts. Exercise 02 (SLO authoring) references the dashboard; exercise 03 (stack base) composes the middleware into the production ingress; exercise 05 (Pareto sweep) uses the histograms to characterise operating points.
