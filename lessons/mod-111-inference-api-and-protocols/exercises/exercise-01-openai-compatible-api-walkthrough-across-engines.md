# Exercise 01 — OpenAI-Compatible API Walkthrough Across Engines

**Time:** 4 hours · **Output:** one compatibility harness (code + config), one drift report across two or more engines, one list of normalise-vs-document decisions for your fleet

## Goal

Make [Chapter 01](../01-openai-compatible-api-surface.md) concrete by **exercising the same OpenAI-compatible request across at least two engines** (plus the OpenAI-hosted API as a reference if available), cataloguing the drift in response shape, finish reason, streamed-chunk layout, error object, and tool-call shape, and publishing a compatibility report your fleet uses as a release gate.

## Problem statement

Chapter 01 names the top-ten drift points. In practice, which drifts actually bite your fleet is specific to the engines you run and the clients you serve. Today you build the harness that turns "which drifts matter here" from a 2 AM discovery into a pre-merge check.

You will run a scripted set of requests against two or more engines, capture the full response (headers + body + SSE frames byte-for-byte), diff against a reference shape, and publish a report that names (a) which drifts are present, (b) which should be normalised by an ingress shim, and (c) which must be documented and exposed to clients.

## Prerequisites

- Chapter 01 of this module read.
- At least two of: a vLLM replica, a SGLang replica, a TRT-LLM / Triton replica, an OpenAI-hosted API key (used sparingly — the harness should minimise hosted-API cost). A hosted reference is highly recommended; it is the only way to anchor "what the OpenAI API actually does" against prose that drifts.
- Python 3.11+; `httpx` or `openai` SDK; `pytest` or equivalent runner.
- A representative model available on each engine — a small instruct model (1B–8B class) is enough; the drift shows up regardless of size.

## Requirements

Five deliverables.

### Deliverable A — the request catalogue

A YAML / JSON file listing the test requests. One entry per drift point in chapter 01. Minimum:

```yaml
cases:
  - name: basic_nonstream
    description: "Plain chat completion, no streaming, no tools."
    request:
      endpoint: /v1/chat/completions
      body:
        model: "{model}"
        messages:
          - {role: user, content: "What is 2+2?"}
        max_tokens: 16
        temperature: 0.0
        stream: false
    assertions:
      - response.status_code == 200
      - response.body.object == "chat.completion"
      - "choices[0].finish_reason" in ["stop", "length"]
      - "choices[0].message.role" == "assistant"

  - name: basic_stream
    description: "Chat completion with streaming."
    request: {...stream: true...}
    assertions:
      - frames[0].object == "chat.completion.chunk"
      - frames[-1].data == "[DONE]"
      - all frames have choices[0].delta
      - last meaningful frame has finish_reason != null

  - name: tool_call_single
    description: "Single tool call, non-parallel."
    request: {...tools: [weather]...}
    assertions:
      - choices[0].message.tool_calls has one entry
      - tool_calls[0].function.name == "get_weather"
      - choices[0].finish_reason == "tool_calls"

  - name: tool_call_parallel
    ...
  - name: tool_call_streaming
    ...
  - name: json_schema_strict
    ...
  - name: json_mode
    ...
  - name: multimodal_image
    ...
  - name: logprobs_nonstream
    ...
  - name: logprobs_stream
    ...
  - name: error_rate_limit
    ...
  - name: error_bad_schema
    ...
  - name: stream_cancellation
    description: "Open stream, consume 50 tokens, close abruptly."
    assertions:
      - engine releases KV slot (verify via metrics drop)
      - no partial SSE frames corrupt follow-up requests
```

One case per drift point in chapter 01's top-ten, plus any drifts you've already observed in your fleet. The exact names can match or extend chapter 01's numbering.

### Deliverable B — the harness runner

A Python script that:

1. Loads the request catalogue.
2. For each case and each target engine, sends the request and captures:
   - Response status code.
   - Response headers (verbatim).
   - For non-streaming: response body (parsed JSON + raw bytes).
   - For streaming: every SSE frame (raw bytes + parsed).
   - Wall-clock timings (TTFT if streaming, E2E always).
3. Evaluates assertions and records pass / fail / "drift detected" (assertion passes but a value is engine-specific, e.g., a vendor-extension finish-reason).
4. Writes a structured report (JSON + a Markdown summary).

Sketch:

```python
import httpx, time, json, pathlib, yaml
from dataclasses import dataclass

@dataclass
class Target:
    name: str          # vllm, sglang, trt_llm, openai
    base_url: str
    api_key: str
    default_model: str

def run_case(target: Target, case: dict) -> dict:
    body = resolve_placeholders(case["request"]["body"], target)
    url = target.base_url + case["request"]["endpoint"]
    t0 = time.monotonic()
    if body.get("stream"):
        return run_stream(target, url, body, case, t0)
    with httpx.Client() as client:
        r = client.post(url, json=body, headers=auth(target), timeout=60)
    return {
        "target": target.name, "case": case["name"],
        "status": r.status_code, "headers": dict(r.headers),
        "body": r.json(), "latency_ms": (time.monotonic() - t0) * 1000,
        "pass": evaluate_assertions(case["assertions"], r),
    }
```

### Deliverable C — the drift catalogue

Running the harness produces a matrix: `cases × engines → outcome`. Collate into a Markdown table:

```
| Case                   | vllm 0.7.1 | sglang 0.4.0 | trt-llm 0.14 | openai (ref) | Drift notes                                |
|------------------------|-----------|--------------|--------------|--------------|--------------------------------------------|
| basic_nonstream        | pass      | pass         | pass         | pass         | —                                          |
| basic_stream           | pass      | pass         | drift        | pass         | TRT-LLM omits "object" on chunks           |
| tool_call_parallel     | pass      | drift        | drift        | pass         | sglang emits function_call, trt-llm omits  |
| json_schema_strict     | pass      | drift        | fail         | pass         | sglang post-hoc; trt-llm unsupported       |
| logprobs_stream        | pass      | drift        | fail         | pass         | sglang only exposes top-5                  |
| stream_cancellation    | pass      | pass         | drift        | n/a          | trt-llm doesn't release KV for 500 ms      |
```

Record the exact engine versions. Drift points that look surprising ("the engine's documented behaviour differs from its observed behaviour") get a follow-up ticket.

### Deliverable D — the normalise-vs-document decision list

For each drift, decide:

- **Normalise at the ingress** — a shim rewrites the engine's response into the canonical shape. Document the shim in your API docs ("the fleet emits OpenAI-shape finish reasons regardless of engine").
- **Document as engine-specific** — expose the drift; clients key on the engine identity (via a response header `x-inference-engine: sglang/0.4.0`).
- **File upstream** — the drift is a bug the engine project should fix; file an issue; link from the harness case.
- **Change engine config** — the engine has a flag that controls the drift; set it; verify.

Publish as a short document; chapter 01's "normalise what's accidental, document what's essential" is the rule of thumb.

### Deliverable E — the CI integration (if feasible)

Wire the harness into the engine-upgrade pipeline:

- On every engine config change or upgrade, run the harness against a staging replica.
- The run fails the pipeline on `fail` outcomes; `drift` outcomes produce a warning and require manual confirmation (because drift may be intentional).
- The harness's own version is pinned alongside the engine version.

For bootstrap, a Makefile target is enough; teams with CI pipelines wire it into the pre-merge or canary check.

## Starter guidance

- **Start with one case, two engines.** Get the harness skeleton working end-to-end (one case, two engines, drift detection, report output) before authoring the full catalogue. Expansion is cheap once the shape is right.
- **Capture full bytes, not just parsed JSON.** The report is forensic; a captured SSE byte stream is the thing you grep in six months.
- **Pin engine and SDK versions.** Drift is a release-note question; the harness's value disappears if versions float.
- **Use the hosted OpenAI API sparingly.** Each hosted call costs money; keep the reference requests to one per case, cache the response, re-evaluate only when the reference set changes.
- **Avoid real tenant-shaped requests.** Use synthetic prompts; the harness is pre-production.
- **Write assertions against values you care about.** `response.status_code == 200` is a tautology once auth works; `finish_reason in known_set` is actionable drift.

## Fallbacks for constrained environments

- **Only one engine available.** Still valuable: compare the engine against the OpenAI hosted API as reference. Cost: ~$1 per harness run; cheap.
- **No OpenAI key.** Skip the hosted reference; compare between two self-hosted engines. The drift matrix is smaller but still useful.
- **No tool-capable model.** Skip the tool-call cases; cover streaming and structured-output instead.
- **Network-isolated environment.** All targets must be reachable from the harness host; mock the hosted reference by vendoring a canned response captured from a prior online run.

## Acceptance criteria

1. **Deliverable A** has ≥ 10 cases, covering the chapter 01 top-ten drift points plus at least one case specific to your fleet.
2. **Deliverable B** is runnable, captures full response bytes, and produces a structured report.
3. **Deliverable C** is the matrix of outcomes with explicit version pins and drift notes.
4. **Deliverable D** has an explicit decision per drift (normalise / document / file / config).
5. **Deliverable E** (CI integration) is either wired or has a documented bootstrap path (Makefile target + runbook step).
6. A one-paragraph summary at the top of the report names the three most impactful drifts for your fleet.

## Common failure modes

- **Running against one engine only.** Compatibility is relative; one data point is not a drift signal.
- **Comparing against the English prose of the OpenAI reference.** The reference drifts; the hosted API is the real spec. Measure against it.
- **Normalising aggressively.** A shim that papers over every drift loses semantic signal (an engine's enriched logprobs become OpenAI-reference logprobs). Document the essential drifts.
- **Pinning the harness to "latest."** The harness's own version matters; a harness update can change the pass/fail set, confusing the drift report.
- **Skipping the cancellation and error cases.** These drifts are more operationally important than the happy-path drifts; make sure they're in the catalogue.

## Keep it

The harness, the request catalogue, the drift matrix, and the normalise-vs-document decisions are the fleet's compatibility artefact. Exercise 02 (streaming) extends it with cancellation and back-pressure cases; exercise 04 (tool calling) extends it with the provider-shape translation cases. Chapter 07's hand-off bundle references the harness result.
