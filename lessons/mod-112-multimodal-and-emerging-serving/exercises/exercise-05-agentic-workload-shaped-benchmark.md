# Exercise 05 — Agentic Workload Shaped Benchmark

**Time:** 5 hours · **Output:** one agentic-traffic generator, one baseline chat-traffic comparison report, one scheduler-response characterisation, one per-session observability dashboard, one tuning-recommendation memo

## Goal

Make [Chapter 05](../05-agentic-workloads.md) concrete by **authoring a benchmark harness that plays agentic traffic (tool-interleaved, latency-bursty), measuring the batcher's response on your inference fleet, comparing against a text-only chat workload at matched QPS, and publishing the agentic-vs-chat SLO delta with a tuning recommendation**.

## Problem statement

Chapter 05 argues that agentic traffic is different from chat traffic in three ways: it is prefix-cache-friendly, it is bursty, and the session is a longer-lived unit than the request. This exercise proves the three with measurements against your fleet, and recommends tuning.

You will author a scripted traffic generator that plays agent-shaped traffic (simulated tool-call rounds), replay it at various QPS, compare the scheduler's behaviour against a chat workload at the same QPS, and write a one-page tuning memo based on what you see.

## Prerequisites

- Chapter 05 of this module read.
- A running chat-serving fleet from earlier exercises (any OpenAI-compatible endpoint; vLLM / SGLang / TRT-LLM).
- Chapter 04 of module 104 (prefix cache) understood.
- Chapters 01–03 of module 110 (observability stack) available — agent workloads are observed through the stack, not an ad-hoc one.
- Python 3.11+, `httpx` / `aiohttp` for concurrent clients, `locust` or a bespoke asyncio driver for traffic replay.

## Requirements

Six deliverables.

### Deliverable A — the agentic-traffic generator

A traffic generator that plays sessions with the following shape:

- **Session arrival.** Poisson process at a configurable rate (sessions / minute).
- **Per session:** 10–30 inference requests (sampled from a log-normal distribution with the mean you set), interleaved with simulated tool-call delays (sleep of 500 ms – 5 s between requests).
- **Prompt composition.** Each request has a fixed system prompt (shared across all sessions of a run — this is the prefix-cache target) plus a growing scratchpad that includes prior turns and simulated tool results.
- **Response-size target.** Short per-request (25–150 output tokens; most agent responses are short tool-call dispatches or planning paragraphs).
- **Sub-agent bursts.** On a configurable fraction of sessions, inject a burst where 4–8 requests arrive concurrently from the same session.

The generator should be reproducible (seed-driven), expose configuration (sessions/min, mean steps/session, burst probability, tool-delay distribution), and emit a per-request CSV log with timestamps, token counts, and whether the request was part of a burst.

### Deliverable B — the chat-baseline comparison

A matched chat workload: single-turn conversations at the same QPS (requests / second) as the agent workload, with similar input and output token lengths but no shared system prompt and no tool delays. This is the baseline.

Run both workloads for a ≥ 15-minute window each. Capture:

- Fleet-side: TTFT P50/P95/P99, TPOT P50/P95, prefix-cache hit rate, admission-reject rate, KV-usage over time, preemption count.
- Client-side: per-request wall-clock, per-session wall-clock (agent only), failure count.

The report should answer: "at matched QPS, does the agent workload look cheaper (higher cache-hit, lower average TTFT) or more expensive (more prefill churn, worse tail) than chat?" Chapter 05 claims the former; your measurements verify or refute on your hardware.

### Deliverable C — the burst-handling characterisation

Run the agent workload with sub-agent bursts enabled at a configurable fraction (5 %, 20 %, 50 %) and no per-session concurrency cap at the ingress. Measure:

- Admission-reject rate during bursts.
- Tail TTFT for non-burst requests arriving during burst windows.
- KV-high-water-mark during bursts.

Then enable the per-session concurrency cap at the ingress (e.g., `max_in_flight_per_session = 8`) and repeat. Report before-after.

Expectation: the cap reduces the tail-TTFT impact on non-burst traffic at the cost of serialising the burst within the session. Document the trade-off.

### Deliverable D — the scheduler-response report

Based on deliverables B and C, characterise the scheduler's behaviour under agent traffic. Specifically answer:

- Does the prefill / decode mix look different from chat? Numbers.
- Is chunked-prefill helping under the agent workload? (Compare with and without, if your engine lets you toggle it.)
- Does the preemption rate rise when bursts collide with long-running chat requests (if any)? Mix agent and chat traffic in a single run and measure.
- Does the scheduler batch short agent requests together usefully, or are most requests served singly?

The report should name one specific scheduler knob you would change (or leave alone) based on what you saw.

### Deliverable E — the per-session observability dashboard

Add panels to the module-110 chapter-03 stack (or stand up new ones) for the agent-workload metrics from chapter 05:

- Per-session total duration.
- Steps per session distribution.
- Per-session cost.
- Session abandonment rate.
- Per-step latency histogram (how many agent requests per second are sub-200 ms vs. 200–1000 ms vs. >1 s).
- Burst concurrency over time.

Render against a deliverable-A run and screenshot or export the dashboard JSON.

### Deliverable F — the tuning recommendation memo

A one-page memo that reads like a change proposal to your SRE / scheduler-owner:

- One sentence on what you observed.
- Three sentences on the SLO delta between agent and chat workloads at matched QPS.
- One specific change proposed (per-session cap value, chunked-prefill setting, workload-class routing, autoscaler signal change) with the number.
- The measurement that would validate the change in the next iteration.

Scope the memo so the recipient can read it in one minute and decide.

## Starter guidance

- **Keep the system prompt fixed across all sessions in a run.** This is what makes agent traffic cache-friendly. Any per-session randomness in the prompt destroys the cache-hit signal you are trying to measure.
- **Measure cache hit rate before and after any change.** Chapter 05's "cost-per-session 10× projection" failure mode is prevented by watching this metric; the exercise tests you on using it.
- **Model the sub-agent burst realistically.** A real agent framework (LangGraph, OpenAI Assistants, CrewAI) dispatches sub-tasks in parallel. The generator's burst shape should look like a real framework's dispatch — not all at exactly the same ms, but within a 50–200 ms window.
- **Use at least two concurrent sessions in the generator.** With one session, the cache-hit metric is misleading because all cache hits are from the same session. Agent workloads in production have many sessions sharing a system prompt.
- **Pin the engine version.** Scheduler behaviour drifts across engine releases; your tuning memo pins to a version.
- **Record the session IDs on every request.** The per-session metrics in deliverable E need session-correlated data.

## Fallbacks for constrained environments

- **Small fleet.** Scale the workload down proportionally; the ratios in deliverable B are what matters, not the absolute QPS.
- **No admission-reject visibility on the engine.** Infer from client-side 503s / 429s; the signal is coarser but adequate for the exercise.
- **No burst injection possible.** Use at least two concurrent generator processes against the same fleet and compare serial vs. concurrent client behaviour; you lose some precision in the burst characterisation but not the main report.
- **Mixed fleet unavailable.** Run agent-only and chat-only separately against the same replica; the mixed-traffic deliverable-D question answers against "what would it look like if…" with a small caveat.

## Acceptance criteria

1. **Deliverable A** generator runs; produces agent-shaped sessions with the chapter-05 characteristics; logs are reproducible with a seed.
2. **Deliverable B** agent-vs-chat comparison report captured; the cache-hit delta is quantified.
3. **Deliverable C** burst-handling with and without the per-session cap measured; trade-off articulated.
4. **Deliverable D** scheduler behaviour characterised; one specific tuning knob identified.
5. **Deliverable E** per-session dashboard populated; six panels rendered.
6. **Deliverable F** one-page tuning memo written; one concrete change proposed with a validation metric.

## Common failure modes

- **System prompt drifts between sessions.** A timestamp or UUID in the "fixed" prompt destroys cache reuse; the measurements say agent traffic is no better than chat traffic, contradicting chapter 05.
- **Generator is accidentally too fast.** Real orchestrators wait on tool responses. A loop with no delays dumps requests at the fleet faster than any real agent would; the burst report is uninformative.
- **Cache-hit metric scraped from the wrong label.** vLLM, SGLang, and TRT-LLM all expose a cache-hit metric but the naming differs; a bad label selector in the dashboard shows zero hits and panic ensues.
- **Per-session dashboard built on request ID instead of session ID.** No per-session aggregation possible; the deliverable E dashboard is useless. Propagate session ID in every request and in every log line.
- **Tail TTFT regression masked by average.** The deliverable-B comparison should report P95 and P99 explicitly; a well-behaved mean can hide a bad tail that only appears under bursts.

## Keep it

The traffic generator, the agent-vs-chat comparison report, the burst-handling characterisation, the scheduler-response report, the per-session dashboard, and the tuning memo are the fleet's agentic-workload artefact. Module 113 (rollouts) will reuse the generator for canary load; module 114 (platform leadership) will reuse the memo pattern for inference-platform RFCs.
