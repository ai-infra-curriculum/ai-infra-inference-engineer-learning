# Exercise 01 — Continuous Batching Inner-Loop Walkthrough

**Time:** 3 hours · **Output:** one annotated source-tour, one running engine with iteration-level logging, one traced iteration document, one one-page inner-loop cheatsheet

## Goal

Make [Chapter 01](../01-continuous-batching-inner-loop.md) concrete by locating the scheduler loop in the actual source of your engine, running the engine with iteration-level logging on, and tracing a single engine step from arrival to decoded token with code-path references. By the end you can point at `schedule()` (or your engine's equivalent) in under 60 seconds and explain who got admitted, who was preempted, and why — on an iteration-by-iteration basis.

## Problem statement

Your team has adopted an engine from module 102 (vLLM, SGLang, or TRT-LLM via Triton) and will spend the rest of module 103 tuning its scheduler against an SLO. Before tuning, every operator on the team must be able to narrate a single scheduler iteration against the source — admission decision, prefill-vs-decode selection, preemption, forward dispatch. This exercise forces that fluency.

Pick one engine (vLLM is the recommended default because its scheduler is the most readable). The chapter applies to all three.

## Prerequisites

- A workstation or lab machine with at least one NVIDIA GPU (any Hopper / Ada-era card works for a small model). CPU-only mode is acceptable for coverage of the scheduler logic itself; see starter guidance.
- A pinned version of your chosen engine (`pip install vllm==<version>` or equivalent), plus a `git clone` of the source at the matching tag. The source tour has to match the running code.
- A tokenizer / Python environment that can submit requests to the engine (any `httpx` / `openai`-SDK script).
- Exercise 01 from [module 102](../../mod-102-inference-engine-internals/exercises/exercise-01-vllm-scheduler-and-engine-walkthrough.md) is a strong prerequisite; the five-layer cheatsheet from that exercise is directly reused here.

## Requirements

Deliver four artefacts.

### Artefact A — Scheduler source-tour (markdown, ≤ 2 pages)

Author a short document that names, for your pinned engine version:

1. The file and function where the engine step loop lives (`AsyncLLMEngine.step()` / `engine_core.step()` / your version's equivalent).
2. The scheduler class and the file it lives in.
3. The function that implements one iteration of the inner loop — the one that returns "which sequences run this step, who is prefilling vs. decoding, who was preempted."
4. The admission helper (the sub-function that pulls from the waiting queue).
5. The prefill vs. decode selection helpers.
6. The preemption helper — identify both the eviction policy choice (who to evict) and the mode (swap vs. recompute).
7. The point in the forward pass where mixed prefill+decode batches are dispatched (the "piggyback" / "mixed batching" path; usually in the model runner or the attention backend wiring).

For each: **file path + function name + one sentence** on what the function does. Not copies of the code. A map.

If your engine does not have a single `schedule()` method but splits logic across multiple files (SGLang, TRT-LLM batch manager), name each piece and how they compose.

### Artefact B — A running engine with iteration-level logging

Launch a local server with a small model (1B–7B instruct) and flags chosen to make the inner loop *observable*:

- `--max-model-len` modest (so prompts can push against it).
- `--max-num-seqs` small (e.g. 8 or 16) so admission decisions and preemption are reachable.
- `--max-num-batched-tokens` modest (e.g. 2048) so you can see prefill chunk effects.
- `--enable-chunked-prefill` on.
- Verbose logging turned up — whatever flag your engine exposes to log per-step scheduler decisions. If none exists, add a lightweight print statement in your local clone of the scheduler (do not upstream it) that logs `(iteration_id, running_count, waiting_count, admitted, prefill_tokens, decode_tokens, preempted)`.

Capture:
- The exact launch command.
- The engine's startup log.
- A head sample of 10 iterations of the scheduler log (10 lines of your log statement).

### Artefact C — One traced iteration (markdown, ≤ 1 page)

Pick one engine step from Artefact B's log that is *interesting* — ideally one where:

- A new request was admitted.
- At least one sequence was in prefill (if chunked prefill, mid-chunk is fine).
- At least one sequence was decoding.
- Bonus: a preemption fired.

Write a one-page trace of that iteration:

1. **Pre-iteration state.** Running count, waiting count, KV usage.
2. **Admission decision.** Which waiting request was admitted (or none, with reason). Which code path made the call (Artefact A reference).
3. **Batch composition.** How many prefill tokens from how many sequences; how many decode tokens from how many sequences; total batched tokens; vs. `max-num-batched-tokens`.
4. **Preemption decision.** If any — who, which mode, which code path.
5. **Forward dispatch.** Which attention path handled the mixed batch (reference Artefact A).
6. **Post-iteration state.** Running count, waiting count, newly-generated tokens.

Explicitly cross-reference each step to a line of code in your pinned engine. "Line 237 of `scheduler.py`" is the discipline.

### Artefact D — Inner-loop cheatsheet (one page)

A single-page summary that another operator could use to:

- Find the scheduler loop in your engine in under 60 seconds.
- Know what every iteration decides and in what order (admission → prefill/decode selection → preemption → forward).
- Map the five load-bearing engine args (`max-num-seqs`, `max-num-batched-tokens`, chunked-prefill-on/off, chunk-size, scheduler-delay) to the code they modulate.

Reuse the module-102 exercise-01 cheatsheet's five-layer map as scaffolding; add this exercise's iteration-level annotations as the layer-3 deep-dive.

## Starter guidance

- **Pin the version.** The source tour's value depends on the paths being accurate. `pip show vllm` → match the `git checkout <tag>`. Changes between vLLM / SGLang releases are substantive.
- **Pick a small model.** The point is to see the loop clearly, not benchmark. A 1B–3B instruct model keeps forward passes fast and iterations frequent, so logging produces useful density of events.
- **Make preemption reachable.** `--max-num-seqs 8` with 20 concurrent clients will provoke admission pressure quickly. If preemption still never fires, add a long-prompt request mid-flight.
- **Instrumentation stays local.** Any logging changes you make to the engine source live in your local clone only. Do not upstream and do not confuse the audit trail.
- **If chunked prefill is a no-op in your version**, note it explicitly in Artefact A — some older versions have the flag but no backend implementation; verify by watching the batched-tokens-per-iteration distribution.
- **Fallback for CPU-only runs**: the scheduler logic runs identically on the CPU backend. The forward is much slower, so each iteration lasts longer and your log output is easier to read. Note the backend in your writeup.

## Fallbacks for constrained hardware

- **CPU-only backend.** vLLM supports CPU; SGLang has limited CPU support; TRT-LLM is GPU-only. If CPU-only, state so and limit yourself to the scheduler layers of the exercise.
- **ROCm.** Supported in vLLM / SGLang for subsets of models; verify feature parity (chunked prefill, prefix caching) before relying on them in a drill.
- **Cloud spot GPU.** A single L4 / A10 / consumer RTX card is enough for a 7B instruct model and all four artefacts.

## Acceptance criteria

1. **Artefact A** names at least seven distinct source entry points, with `file:function` references that `grep` can confirm.
2. **Artefact B**'s engine runs for ≥ 5 minutes under a probe without crashing, and the scheduler log emits ≥ 100 iteration lines.
3. **Artefact C**'s traced iteration references at least four of Artefact A's entry points with specific line numbers, and includes a preemption or (if preemption never fires) a convincing argument from the metric evidence that preemption was not needed.
4. **Artefact D** fits on one page and lets a sibling operator — one who has read chapter 01 but not this exercise — launch the same engine and name every iteration's phases without opening the source tour.
5. All four artefacts live in version control; the engine source tour pins to the engine's `git` SHA.

## Common failure modes

- **Chapter-copying.** Your artefact must reference *your* installed version's source. Chapter 01 is deliberately vague on class names because they rotate; your writeup must be concrete.
- **No preemption in the log.** With `--max-num-seqs 8` and 20 concurrent clients it will fire; if it doesn't, reduce `--max-num-seqs` to 4 or send a long prompt mid-batch.
- **Logging too noisy to read.** Keep your log statement to one line per iteration; dump sampling parameters to a side file if you need them.
- **Benchmarking confusion.** This exercise is not a performance study. The output is diagnostic fluency, not a throughput number.

## Keep it

The inner-loop cheatsheet is a reusable artefact for the rest of module 103 (and 104). Pin it to your team wiki next to the module-102 five-layer map. Refresh on every engine upgrade. Exercises 02–05 will reference it directly.
