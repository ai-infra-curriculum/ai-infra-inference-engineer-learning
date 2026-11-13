# Chapter 04 — Reasoning-Model Serving: Think Traces, Budget Caps, Interruptible Decoding, Reasoning Channels

> A reasoning model spends most of its output budget on tokens the user never sees. The scheduler's view of the request is "this takes forty seconds and ten thousand decode steps"; the client's view is "I'm waiting." This chapter is where the inference engineer makes those two views coherent — budget caps that fire, cancellation that works, and a reasoning / answer channel split that keeps the streaming UI sane.

## Motivation

Reasoning models (OpenAI's `o1` / `o3` / `o4` lines, Anthropic's extended-thinking Claude 3.7 / 4 models, DeepSeek-R1, QwQ, Gemini with extended thinking) share a serving shape that breaks several text-serving assumptions at once:

- **Most of the output is hidden.** The model generates "thinking" tokens — a long internal monologue — before committing to a visible answer. The hidden-to-visible ratio can be 20× or more on hard prompts. Billing, cost accounting, and the user-facing streaming UI all have to know which tokens are which.
- **Latency is not bounded by response length.** A 50-word answer can take 40 seconds of wall-clock because the model thought for 9000 hidden tokens first. TTFT (time to first *visible* token) is now a very different number from time-to-first-engine-emit.
- **The request is interruptible in a new way.** A user who sees "the assistant is thinking…" for 30 seconds may close the tab or cancel; the serving stack has to stop the trace quickly and release KV. An operator noticing a stuck trace in production (budget exceeded, pathological loop) needs a way to kill it too.
- **The API surface carries two output streams.** The reasoning and the answer are different content kinds on the response — different tagged items on OpenAI's `/v1/responses`, different content blocks on Anthropic's Messages API. The ingress must route them correctly.

By the end of this chapter you can:

- Draw a reasoning request's shape from ingress through hidden decode to visible answer, and name the two user-visible latency SLOs.
- Set and enforce a thinking-budget cap that stops a runaway trace.
- Wire interruptible decoding so a client cancel or operator abort stops the trace within a bounded time.
- Expose the reasoning-vs-answer channel split in your API surface for both OpenAI-compatible and Anthropic-compatible clients.
- Decide per product whether to stream the reasoning to the user or suppress it.

## The reasoning request flow

A reasoning request looks like a chat-completions request with a reasoning-budget parameter. Depending on the provider shape:

OpenAI `/v1/responses` style:

```json
{
  "model": "o1-pro",
  "input": "Prove that the square root of 2 is irrational.",
  "reasoning": { "effort": "medium" },
  "max_output_tokens": 2048,
  "stream": true
}
```

Anthropic Messages style:

```json
{
  "model": "claude-opus-4-reasoning",
  "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}],
  "thinking": { "type": "enabled", "budget_tokens": 10000 },
  "max_tokens": 2048,
  "stream": true
}
```

The engine runs a single autoregressive decode loop, but the output stream is logically two interleaved content kinds:

```
[thinking] Let me set up a proof by contradiction. Suppose sqrt(2) = p/q ...
[thinking] ... (many thousand tokens)
[thinking] This is the contradiction. Therefore sqrt(2) is irrational.
[answer]   Suppose, for contradiction, that √2 = p/q with gcd(p,q)=1. Squaring, ...
```

The transitions between content kinds are driven by special tokens the model was trained to emit (`<think>…</think>` or a model-specific equivalent) or by structured sampling that the engine enforces. The engine recognises the transition and tags each outgoing token with its channel.

The specific transition protocol differs per model family — DeepSeek-R1 uses `<think>` / `</think>` tokens and vLLM exposes them via a `deepseek_r1` reasoning parser ([vLLM reasoning-outputs docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)); OpenAI's reasoning models emit reasoning items as distinct output items on the Responses API rather than inline tokens; Anthropic's extended thinking produces `thinking` content blocks separate from the final `text` content ([Claude streaming docs](https://platform.claude.com/docs/en/build-with-claude/streaming)). Verify the per-model convention before building the channel splitter.

Two user-visible SLOs replace the single text-serving TTFT:

- **Time to first reasoning token (TTFR).** Fast; identical to classical TTFT. This is the "the model is thinking" indicator in the UI.
- **Time to first answer token (TTFA).** Slow; equal to `TTFR + reasoning-token-count × TPOT`. This is "the model is actually answering" — the moment the user starts getting the information they asked for.

Reasoning fleets must track both. A dashboard that only shows classical TTFT misses where the user's real waiting lives.

## The thinking-budget cap

Every reasoning model exposes a knob that bounds the hidden-token count. Shapes vary:

- **Named effort.** `reasoning.effort = "low" | "medium" | "high"` (OpenAI). The provider maps the enum to a model-specific token budget.
- **Explicit token budget.** `thinking.budget_tokens = 10000` (Anthropic). Hard cap on reasoning tokens before the model is forced to answer.
- **Total-output cap.** `max_completion_tokens = 15000` (OpenAI on reasoning models — the parameter supersedes `max_tokens` because it includes reasoning). The engine attributes tokens to reasoning first, answer second.

What happens at the cap:

- **Soft cap (preferred).** When the budget is reached, the engine emits a control token that transitions the stream from reasoning to answer. The model answers with whatever context it has built up. Quality degrades gracefully as the budget shrinks.
- **Hard cap (less preferred).** When the budget is reached, the engine emits a `finish_reason: "length"` and stops. The client gets reasoning-only output, no answer. Useful for operators forcing an abort; bad as a default user experience.

Enforcing the cap at the serving layer, not the model:

```python
# pseudo-code — not an engine's actual sampler
def stop_condition(token, state):
    if state.phase == "reasoning":
        if state.reasoning_tokens >= state.budget:
            # Inject the transition; force the model to answer.
            return INJECT_END_THINK_TOKEN
    return CONTINUE
```

The ingress sets `state.budget` based on the client-provided parameter (clamped to a server-side max — a user requesting a 1 M-token thinking budget on your fleet has to be reined in). Module 103's admission control also needs to know the worst-case output size, which for reasoning requests is `budget + max_answer_tokens` rather than just `max_tokens`.

## Interruptible decoding

A reasoning request may run for 30–120 seconds. Three ways it needs to interrupt:

- **Client disconnect.** The user closed the tab; the HTTP connection drops. Same mechanism as module 111 chapter 02 — the serving proxy detects the TCP close, cancels the engine request, the engine stops decoding and frees KV. Faster propagation matters more here because each wasted second burns more hidden tokens than a classical text request would.
- **Client explicit cancel.** A SDK-level cancel call — often a `POST /v1/responses/<id>/cancel` on providers that expose it. The ingress looks up the request state and sends an abort to the engine. This is the shape most relevant to `/v1/responses` since the request is stateful.
- **Operator abort.** On-call engineer sees a stuck request (a trace running for 10 minutes, a request where the model is clearly looping). A control-plane mechanism — a signal to the engine, a kill on the request ID — stops the trace.

The engine's abort hook is the same mechanism in all three cases; the trigger differs. What matters:

- **Abort-to-free-KV latency should be under one decode step.** Module 111 chapter 02 said this for text; it still applies. The chapter-02 adversarial tests (early close, mid-stream abort) are directly reusable — just with longer-held requests.
- **Partial output is a response, not an error.** When a reasoning request is cancelled, the engine should flush whatever it had and close the stream cleanly. The client sees `finish_reason: "cancelled"` (or an engine-specific equivalent) rather than a 500.
- **Billing should respect the cancel.** The user is billed for the tokens actually generated up to the abort, not the full budget. Module 110 chapter 04's cost accounting needs to see the cancel event.

## The reasoning channel split

The API surface has to carry the two output kinds distinctly. Two provider shapes dominate in 2026:

**OpenAI `/v1/responses` output items.** The response (streaming or not) is a list of typed output items:

```json
{
  "object": "response",
  "output": [
    {"type": "reasoning", "summary": "Set up proof by contradiction; derive gcd=1 vs even-p argument; conclude.", "content": "..."},
    {"type": "message",   "content": [{"type": "output_text", "text": "Suppose √2 = p/q ..."}]}
  ]
}
```

Streaming emits event-typed frames: `response.reasoning.delta`, `response.output_text.delta`, `response.completed`.

**Anthropic Messages content blocks.** The response's `content` array contains blocks of different types:

```json
{
  "role": "assistant",
  "content": [
    {"type": "thinking", "thinking": "Let me set up a proof by contradiction..."},
    {"type": "text",     "text": "Suppose √2 = p/q ..."}
  ]
}
```

Streaming emits `content_block_start` / `content_block_delta` / `content_block_stop` events, with the block index increasing as the stream moves from thinking to text — the documented order on the Claude Messages API is `message_start`, then a `content_block_start` for the `thinking` block with `thinking_delta` and `signature_delta` events, a `content_block_stop`, then a `content_block_start` for the `text` block with `text_delta` events, and finally a `message_stop` ([Claude streaming docs](https://platform.claude.com/docs/en/build-with-claude/streaming)).

On the OpenAI side, the Responses API exposes a `reasoning.summary` parameter (`auto` / `concise` / `detailed`) and returns a natural-language *summary* of the reasoning for most reasoning models rather than the raw trace — a product decision, not an infrastructure one ([OpenAI Responses API announcement](https://openai.com/index/new-tools-and-features-in-the-responses-api/)). Both surfaces evolve; verify event names and whether raw-trace vs. summary is returned against the current provider docs before shipping the splitter.

The inference engineer owns:

- The transition detection inside the engine or the adjacent parser.
- The event framing on the wire (SSE events typed correctly; chapter 02 of module 111 drew the SSE discipline).
- The compatibility shim: a product that depends on `thinking` content blocks against a backend that emits OpenAI-style reasoning items needs translation. Chapter 01 of module 111's "normalise vs. document" rule applies — reasoning channel shape is *essential drift*, document it rather than silently normalise.

## Stream the reasoning, or hide it?

A product decision with operational consequences:

- **Stream reasoning to the user.** The UI shows "thinking…" with a live transcript of the reasoning (collapsed by default, expandable). Pros: builds user trust in long waits; shows that the model is making progress; useful for developer / power-user audiences. Cons: exposes model-internal failure modes (loops, hallucinations in reasoning that are corrected in the answer); uses more bandwidth; raises confidentiality concerns if the reasoning contains sensitive intermediate steps.
- **Suppress reasoning; stream only the answer.** The UI shows a spinner during the reasoning phase and starts streaming text when the answer begins. Pros: cleaner UX; no exposure of model-internal state. Cons: the user gets no feedback for 10–40 seconds; cancellation anxiety is higher.
- **Stream a summary.** The provider (or your ingress) emits a short per-phase summary ("analysing the constraint," "checking the base case") rather than the raw reasoning. OpenAI's reasoning summary is this pattern. Balances feedback and cleanliness.

The inference engineer should expose both as deployment-time options (e.g., per-model or per-API-key) and let product teams choose. The default for a general consumer product is often "summary"; the default for an internal developer tool is often "full reasoning."

## Scheduler implications

Reasoning workloads change the scheduler's assumptions. Three places:

- **Per-request wall-clock is long.** A chat-fleet scheduler tuned for 2-second median requests now has 40-second requests mixed in. Preemption policy (module 103 chapter 04) matters more — a long reasoning request that holds a KV slot blocks shorter follow-ups in a resource-tight fleet.
- **Hidden-to-visible token ratio is a new class.** Workload classification (module 110 chapter 02) should include a "reasoning" class with its own TTFA SLO distinct from classical TTFT.
- **Decode TPOT matters more than TTFT for total latency.** A reasoning request's total time is dominated by decode steps; TPOT regressions that would be invisible on short chat requests are the first thing the reasoning class notices. Chapter 01 of module 110's axis set is directly applicable.

A fleet that adds a reasoning model should expect its P99 end-to-end latency across the fleet to rise; the right response is to split SLOs per class, not to lump them together.

## Three failure modes

### "A cancelled reasoning request keeps generating for 30 more seconds"

Common cause: the cancel event reaches the ingress but the engine's abort hook doesn't check often enough — abort is checked once per *answer* token boundary but the request is still in the *thinking* phase.

Fix: check the abort flag at every decode step, not only at channel transitions. If the engine's scheduler abstracts the two phases differently, instrument both; the exercise-04 verification confirms abort fires during the thinking phase.

### "Reasoning tokens don't decrement the user's token quota"

Common cause: the billing / quota path keys on the `message.content.text` token count, which is the answer-only count. The hidden reasoning is free to the user, expensive to the operator.

Fix: billing and quota must attribute reasoning tokens explicitly. Module 110 chapter 04's cost model needs a reasoning-token line; chapter 06 of module 111's request-shape discipline needs reasoning tokens in the per-request accounting.

### "Streaming UI flashes 'done' before the answer starts"

Common cause: the client treats the `content_block_stop` event on the thinking block as the end of the response, closes the stream, and misses the answer block that follows.

Fix: event-schema alignment with the client SDK; the response is not done until a `response.completed` / `message_stop` / final event fires. Add a compatibility test case to the exercise-01 harness in module 111 for the two-block-stream pattern.

## Summary

- Reasoning models have two output channels — hidden reasoning and visible answer — with different token budgets, different SLOs, and different billing attribution.
- The thinking-budget cap (named `effort`, explicit `budget_tokens`, or total `max_completion_tokens`) is set per-request, clamped server-side, and enforced by the engine as a soft transition (prefer) or a hard stop.
- Interruptible decoding must check the abort flag every decode step; abort propagation under one step; partial output on cancel is a normal response.
- The channel split appears on the wire as OpenAI reasoning items or Anthropic thinking content blocks; the ingress owns correct framing and typed-event emission.
- Streaming reasoning to the user vs. suppressing it vs. summarising it is a product choice the serving platform should expose as a knob.
- Scheduler assumptions tuned for short chat requests need to be re-reviewed when reasoning requests join the fleet; TPOT becomes more important than TTFT for total latency; workload classes split.
