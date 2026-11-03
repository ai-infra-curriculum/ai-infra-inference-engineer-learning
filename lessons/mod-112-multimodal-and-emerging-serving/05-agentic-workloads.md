# Chapter 05 — Agentic Workloads at the Serving Edge: Long Sessions, Tool-Call-Heavy Shapes, Bursty Mixes

> One agent session is not one inference request. It is a loop of short completions with long gaps in between — the model thinks, calls a tool, waits, reads the result, thinks again, calls another tool — for minutes. The scheduler's assumption that requests are the unit of work breaks; so does TTFT as a quality signal; so does the GPU-utilisation math from module 110. This chapter is the inference engineer's view of what changes.

## Motivation

By 2026 a large and growing share of inference traffic is *agentic*: the client is an orchestration layer (an LLM application, an agent framework, a coding assistant, an autonomous workflow) that issues many short model calls per user interaction, interleaved with tool invocations that live outside the inference platform. Three things that were simple in classical chat serving become seams:

- **"Request" is a bad unit of work.** A single user asking "fix the failing test in my repo" produces 20–200 model calls over the next ten minutes: tool-call rounds, retries, sub-agent spawns. The fleet's QPS is high, but any individual request is 50–500 tokens. The GPU's prefill / decode mix is dominated by *prefill* because every step is a new request with a new (but similar) prompt. Classical TTFT and TPOT describe the shape poorly.
- **Traffic is latency-bursty, not token-bursty.** A conversational chat workload has smooth per-user token rates. An agent workload has silence (tool runs externally for 2 seconds) and then a burst of fresh requests (five parallel sub-tasks). Admission control must handle the burst without rejecting legitimate traffic; the autoscaler sees different signals than in a chat-only world.
- **The session is the long-lived object; the request is not.** Clients hold open sessions for minutes: streaming responses tied to tool-call cursors, resumable streams, saved reasoning state, retained prefix-cache warmth. Module 111 chapter 02's SSE discipline held the transport open for seconds; chapter 05 holds it open for minutes.

By the end of this chapter you can:

- Distinguish the agent-workload traffic shape from the chat-workload traffic shape, and name the per-shape scheduler / cache / admission knobs that differ.
- Design the per-session state the ingress has to hold (idempotency keys, tool-call cursors, resumable stream state, prefix-cache hints).
- Set the ingress policy for tool-call round-trip timeout, session abandonment, and resumable retries.
- Expose the observability axes that agentic workloads require (per-step latency, tool-success rate, session cost, abandonment) alongside the chat-shape SLOs.
- Reason about the scheduler interaction between a chat workload and an agent workload sharing the same replica.

## The agent traffic shape in one picture

```
t=0.00s  agent -> infra:  prompt-1 (system + user + initial context, 2k tokens)   → stream
t=0.80s  infra -> agent:  final token + tool_call "search_repo(symbol='foo')"
t=0.80s  agent -> tool:   search_repo(symbol='foo')
t=3.20s  tool  -> agent:  [20 matches]                     ← 2.4s gap on the inference fleet
t=3.20s  agent -> infra:  prompt-2 (prompt-1 + tool-response, 4k tokens)         → stream
t=4.10s  infra -> agent:  final token + tool_call "read_file(path=...)"
...
```

The arithmetic looks very different from a chat conversation:

- Many requests per user interaction (20–200), each short (sub-second TTFT, <100 output tokens).
- Long gaps between requests (tools run for 1–10 seconds) — the inference fleet is idle for this user during tool-run.
- Each request's input is similar to the previous one plus the tool result — prefix-cache hit rate should be very high if the engine's cache is sized for it.
- Bursts of parallel sub-agents (an orchestrator spawns five concurrent sub-tasks; the fleet suddenly sees 5× the per-user request rate for a second).

The implication for the scheduler is that it sees many near-identical short requests, which is good for continuous batching (modules 103 and 104 are directly applicable) *if* the prefix cache is doing its job. If the cache is not sized or configured correctly, every request pays a fresh prefill cost on a 4k-token context, and the GPU is prefill-bound for a workload whose tokens-generated-per-second number looks tiny.

## The prefix cache is the dominant cost lever

Agent workloads hit the prefix cache harder than any other shape, and when they hit it they get the biggest discount. Three places to get it right:

- **Hash on the system prompt and the first few turns.** The orchestrator usually sends a fixed system prompt plus a growing conversation history. The cache's longest-common-prefix should be the system prompt plus the stable prefix of the agent's scratchpad. Module 104's prefix cache is designed for exactly this; the agent workload is the ideal traffic for it.
- **Size the cache for the active-user's working set.** If a single agent session's cache footprint is 8k tokens at 2 bytes-per-token-per-layer times 80 layers, that's 1.3 MB per user per layer — small enough to keep per-user caches warm. Module 104 chapter 04 (eviction) is the lever.
- **Beware of randomness that breaks the cache.** A timestamp in the system prompt, a UUID the orchestrator inserts per turn, an injected tool-use nonce — any of these destroys prefix-cache reuse by shifting every byte. Audit the orchestrator's prompt templates with the application-layer peer (`llm-application-developer`, L25).

A healthy agent workload shows:

- `prefix_cache_hit_rate > 90 %` on most requests after the first per-session.
- `tokens_served_from_cache / total_tokens` > 70 %.
- `cost_per_1M_tokens` on the agent workload lower than the chat workload (because of the above).

If any of those break, the orchestrator's prompt templates are the first place to look.

## Latency-bursty vs. token-bursty mix

A chat workload's QPS is smooth: many users, each sending occasional requests. An agent workload's QPS is bursty: a single session can emit 10 requests in 3 seconds (parallel sub-agents), then nothing for 15 seconds. On a shared fleet the mix matters:

- **A fleet autoscaled on QPS** (module 109 chapter 02) sees agent bursts as traffic spikes and scales up, then scales down 60 seconds later when the burst subsides. Churn. The scaler needs a longer averaging window, or a signal that reflects *active sessions* rather than instantaneous QPS.
- **A fleet autoscaled on token-throughput** sees agent traffic as a thin ongoing stream (because the per-request token count is small) and may fail to scale up during a many-session burst.
- **Admission control under burst** (module 103 chapter 05) must reject fairly across sessions rather than first-come-first-served. If one agent spawns 50 parallel sub-agents and the fleet admits them all, other users experience a transient outage.

A reasonable pattern:

- Autoscale on a composite signal: `max(requests_per_second, concurrent_active_sessions)`. Chapter 03 of module 109's scaler signals combines with this.
- Per-session concurrency cap at the ingress (e.g., `max_in_flight_per_session = 8`). Protects the fleet from pathological orchestrators.
- Per-tenant QPS cap (module 111 chapter 06). Prevents one tenant's agent fleet from starving another's.

## The session as a first-class ingress object

A classical HTTP API has no "session" — each request is independent. An agentic serving stack can't afford that: it needs to associate requests into sessions to apply the per-session cap, the prefix-cache hints, the resumable-stream state, and the cost accounting. Three mechanisms to consider:

- **Client-provided session ID.** The orchestrator supplies an `X-Session-Id` header or a `session.id` body field. Easiest; relies on client cooperation. Make it opaque (a UUID) and rotate when the session ends.
- **Stateful endpoint.** OpenAI's `/v1/responses` with `store: true` and `previous_response_id` is a server-maintained session. The server holds the response history keyed by `response.id`; future requests reference the previous ID. More operational surface (a session store, typically a KV-shaped database); gives you resumability and server-side context trimming.
- **Implicit via identity + prompt hash.** Group requests by API key + prompt-prefix hash. Works without client cooperation; fragile when the orchestrator's templates drift.

Default for a greenfield fleet: client-provided session ID, with the stateful endpoint pattern available for products that need resumability.

### What state the ingress holds per session

- **Session concurrency counter.** Decrement on request complete / abort.
- **Prefix-cache hint.** Which replica the session's cache is warm on; use it for sticky routing (module 106 chapter 04's affinity routing).
- **Resumable stream cursor.** If the client retries a streaming request, resume from the last emitted token / event ID rather than regenerating from scratch. Module 111 chapter 06's idempotency keys combine with this.
- **Cost accumulator.** Sum of input and output tokens so far; used by module 110 chapter 04's per-session cost dashboard.
- **Session timeout / idle policy.** Evict the session state after N minutes of inactivity. Longer than a chat session; typically 10–30 minutes.

### Tool-call round-trip policy

The agent calls a tool; the tool runs; the agent returns the result in a follow-up inference request. Two timeouts to set and enforce:

- **Tool-run external timeout.** The agent framework's job, not the inference stack's. The inference stack sees a gap where no request arrives; it should not infer anything from the gap except session idle.
- **Session abandonment timeout.** If no follow-up request arrives in N seconds / minutes after a tool-call completion, evict the session state (and cache hints) to reclaim resources. Typical value: 5–15 minutes. Longer and the session state table grows unbounded; shorter and a slow tool-run causes the follow-up to lose its cache warmth.

## Scheduler behaviour under mixed chat + agent traffic

A fleet that serves both a chat application and an agent application on the same replicas will see the following stress:

- **Prefill / decode mix shift.** Agent traffic is prefill-heavy (short responses after each tool round); chat traffic is decode-heavy (long responses). The scheduler's chunked-prefill / token-budget balance (module 103 chapter 03) must handle both. A dashboard that only shows aggregate numbers hides the per-workload truth.
- **KV preemption.** A long chat response holds KV for many seconds; many short agent requests want fresh KV slots in bursts. The preemption policy either lets agents preempt chat (bad: a chat user sees a stall) or starves agents (bad: agent latency spikes under chat load).
- **Batching window interaction.** Continuous batching is efficient when requests arrive smoothly; the bursty agent pattern dumps 5 requests into the same step and leaves the next 10 steps empty. Short prefill-chunk policies work best.

The clean solution — when traffic volume justifies it — is per-workload replica pools (module 106 chapter 03). Chat traffic on one pool, agent traffic on another; isolate their stress. The cost is more replicas and more fleet complexity; the benefit is clearer SLOs per workload and no cross-class interference.

## Observability axes agentic workloads add

Chapter 01 of module 110 names the latency axis set for chat. Agent workloads add:

- **Per-step latency** — distribution of TTFT across the inference calls in a session; most calls in a well-prefix-cached session are sub-200 ms.
- **Session total duration** — wall-clock from first request to final request; the user-perceived latency.
- **Steps per session** — distribution of inference calls per agent session; multimodal (many short sessions; few long sessions).
- **Tool-call latency contribution** — wall-clock that was *not* on the inference fleet per session; useful for debugging "the agent feels slow" complaints that are actually tool-side.
- **Per-session cost** — sum of input and output tokens per session, converted to dollars via module 110 chapter 04.
- **Session abandonment rate** — sessions where the final step never arrived (user closed the client, orchestrator crashed). Alertable if it spikes; often correlates with product-side errors.

These axes are shaped differently from chat; the dashboards should be per-workload-class (module 110 chapter 03's design) with the agent class having its own row.

## Streaming + resumability for agents

An agent's inference call is usually short (sub-second) but the orchestrator still often requests streaming — the tool-call delta can be acted on as it arrives. For an interrupted stream:

- **The orchestrator retries the same request.** Idempotency key present (module 111 chapter 06). The ingress either replays the cached response (if the original finished) or restarts the generation (if it didn't).
- **The orchestrator asks to resume from an event ID.** Harder to implement correctly; the server has to have persisted the token stream so far. OpenAI's `/v1/responses` with `previous_response_id` is a server-facing resumption mechanism (each response is addressable); a true "stream resume mid-generation" is uncommon in open-weight engines.

Default: idempotency-key-based replay-or-restart, no mid-stream resume. Document the semantics so the orchestrator knows which pattern is safe.

## Three failure modes

### "The agent runs slowly even though the fleet looks idle"

Common cause: cache miss on every agent request because the orchestrator injects a timestamp into the system prompt, so no two requests share a prefix. Prefill cost is 100 % of latency.

Fix: audit the orchestrator's prompt templates; remove per-request randomness from the fixed prefix. Confirm the per-session prefix-cache-hit metric rises after the change.

### "The fleet thrashes when one user spawns many parallel sub-agents"

Common cause: no per-session concurrency cap; 50 parallel sub-agents from one session eat the scheduler's admission slots; other users get rejects.

Fix: ingress-level per-session cap (e.g., 8 concurrent in-flight); per-tenant QPS cap underneath.

### "Cost-per-session is 10× what we projected"

Common cause: prefix-cache hit rate low, agent prompts long, tool retries on failures replay the full context. The per-token cost is right; the token count per session is much higher than the design assumption.

Fix: measure the real session size against the designed one; fix the prompt templates with the application-layer peer; add retry budgets so a failing tool doesn't trigger unbounded context regrowth.

## Summary

- Agent workloads are sequences of short, prefix-cache-friendly inference calls interleaved with tool runs that happen outside the inference fleet.
- Prefix-cache hit rate is the dominant cost lever; broken templates (timestamps, UUIDs, nonces in the prefix) are the first thing to audit on a cost spike.
- Latency-bursty traffic stresses autoscalers tuned for smooth QPS; a composite signal (`max(QPS, concurrent sessions)`) and per-session / per-tenant caps keep the fleet stable.
- Sessions are first-class ingress objects: concurrency counter, cache-affinity hint, resumable stream cursor, cost accumulator, idle timeout.
- Mixed chat + agent traffic should ideally be split across replica pools when volume justifies; otherwise chunked prefill and workload-class-aware admission keep them coexisting.
- The observability axis set extends with per-session metrics (duration, steps, cost, abandonment) alongside the chat SLOs.
