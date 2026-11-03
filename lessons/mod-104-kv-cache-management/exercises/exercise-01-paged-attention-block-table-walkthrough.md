# Exercise 01 — PagedAttention Block-Table Walkthrough

**Time:** 3 hours · **Output:** one annotated source-tour, one running engine with block-manager logging, one traced request lifecycle, one block-size trade-off sketch, one one-page cheatsheet

## Goal

Make [Chapter 01](../01-paged-attention-at-operator-depth.md) concrete by locating the block manager in your engine's source, running the engine with block-level logging on, tracing one request's block allocations end-to-end (prefill → decode → release), demonstrating copy-on-write under parallel sampling, and sketching the block-size trade-off. By the end you can point at `allocate` / `append_slot` / `free` in your engine in under 60 seconds and name every physical block a captured request touched.

## Problem statement

Your team has committed to deploying prefix caching, swap-based preemption, and a tiered KV store in the next three modules of this track. All of those features sit on top of the block manager. Before configuring any of them, every operator on the team must be able to narrate the block manager's five-call API against the source and against a running trace. This exercise forces that fluency.

Pick one engine (vLLM is the recommended default; its block manager is the most readable). The exercise applies to all three.

## Prerequisites

- A workstation or lab machine with at least one NVIDIA GPU (any Hopper / Ada-era card works for a small model). CPU-only mode is acceptable for the block-manager logic itself; see starter guidance.
- A pinned version of your chosen engine (`pip install vllm==<version>` or equivalent) plus a `git clone` of the source at the matching tag.
- A tokenizer / Python environment that can submit requests to the engine (`httpx` or `openai` SDK).
- The module 103 exercise 01 inner-loop cheatsheet. The block-manager calls live inside the scheduler loop you already walked.

## Requirements

Deliver five artefacts.

### Artefact A — Block-manager source-tour (markdown, ≤ 2 pages)

Author a short document that names, for your pinned engine version:

1. The block-manager class and the file it lives in.
2. The physical-block pool and the free-block registry — which data structure, which field.
3. The block-table structure — per-sequence, where attached, how indexed.
4. The reference-count table (for copy-on-write).
5. The `allocate(sequence, num_prompt_tokens)` function — file / line / signature.
6. The `append_slot(sequence)` (or equivalent per-step append) function — file / line / signature.
7. The `free(sequence)` function.
8. The `swap_out(sequence)` and `swap_in(sequence)` functions. If your engine uses recompute-only preemption, note that explicitly.
9. The CPU swap-pool initialisation — where the `--swap-space` number becomes a pool.
10. The call site inside the scheduler that invokes `allocate` on admit; the call site that invokes `free` on finish; the call site that invokes `swap_out` on preempt.

For each: **file path + function name + one sentence** on what the function does. A map, not a copy of the code.

If your engine does not have single-named methods (TRT-LLM, SGLang's token-attention layout), name each piece and how they compose.

### Artefact B — A running engine with block-manager logging

Launch a local server with a small model (1B–7B instruct) and flags chosen to make the block manager observable:

- `--block-size 16` (verify flag spelling; the default).
- `--max-num-seqs` moderate (16–32).
- `--max-model-len` modest (so a few requests push against the pool).
- Prefix caching OFF for this exercise (keeps the block picture simple; chapter 02 and exercise 02 turn it back on).
- Verbose logging turned up — whatever flag your engine exposes to log block-manager activity. If none exists, add a lightweight print statement in your local clone of the block manager (do not upstream it) that logs `(request_id, action, block_count, pool_free_before, pool_free_after)` for each `allocate` / `append_slot` / `free` / `swap_out` / `swap_in` event.

Capture:

- The exact launch command.
- The engine's startup log — specifically the line that reports `num_gpu_blocks` and `num_cpu_blocks`.
- A head sample of ≥ 50 block-manager log lines spanning at least one request's full lifecycle.

### Artefact C — One traced request lifecycle (markdown, ≤ 2 pages)

Pick one request from Artefact B's log and narrate its full block lifecycle:

1. **Admission.** The `allocate` call. How many blocks were requested? Which physical block indices were handed out? What was the pool free-count before and after?
2. **Prefill.** If prefill is chunked, which chunks were processed on which steps? Did any trigger an additional `append_slot`? If prefill was in one shot, note it.
3. **Decode.** For every ~16 decode steps, an `append_slot` fires. Trace at least three of them. Did the pool free-count drop by one each time? If yes, consistent with chapter 01. If no, investigate.
4. **Finish.** The `free` call. Which physical blocks were released back to the registry? Did the pool free-count recover exactly to pre-admission?
5. **(If observed) Preemption.** If the request was preempted during its life, trace the `swap_out`: how many blocks moved GPU → CPU; what was the wall-clock duration; which CPU pool indices received them. On resume, trace the `swap_in`.

Explicitly cross-reference each step to a line of code in your pinned engine (reusing Artefact A's references).

### Artefact D — Copy-on-write demonstration

Trigger copy-on-write with a parallel-sampling request (`n=4` samples from one prompt, or the equivalent in your engine's API). Capture:

- The `allocate` for the parent sequence (block count = `ceil(prompt_tokens / 16)`).
- Confirmation that the 4 sampling siblings share the parent's prompt blocks (reference count goes to 4; block table entries point to the same physical blocks).
- The first per-sibling decode step: for each sibling, a copy-on-write event where the trailing shared block is copied to a fresh physical block, the sibling's block table is updated, and the shared block's ref-count drops.
- The final `free` events: once all 4 siblings finish, the parent prompt blocks' ref-counts fall to zero and return to the free pool.

Document each step with log evidence or (where logs don't directly expose ref-counts) with a short Python script that reads the engine's block-manager state (via the engine's API or by printing the ref-count table from your local instrumentation).

If your engine does not support parallel sampling or beam search, note it explicitly in Artefact A and run a two-request duplicate-prefix experiment instead to demonstrate that *without* prefix caching, each request allocates independent blocks even for identical prompts — the counterpoint that motivates chapter 02.

### Artefact E — Block-size trade-off sketch (markdown, 1 page)

A short written sketch (no code required) that answers:

1. For your pinned model + workload, compute `kv_bytes_per_token` (chapter 05 formula).
2. For `block_size ∈ {8, 16, 32, 128}` and your `num_gpu_blocks` as reported in Artefact B, compute how many blocks the pool holds at each size (approximate — the pool size shifts slightly with block size because the per-block metadata overhead changes).
3. For an average sequence length (prompt + output) of 1 024 tokens, compute the expected internal fragmentation per sequence at each block size.
4. Reason about allocator overhead: at each block size, how often does `append_slot` fire per decode step? (Every `block_size`-th step, averaged.)
5. State a recommendation for your workload — "keep 16 because X" or "consider 8 because Y." Reference measured data where available; otherwise mark the recommendation as *hypothesis* and schedule a follow-up measurement.

This artefact does not need to be run as an experiment — it is a worksheet. The point is fluency, not Pareto curves. (Those come in later exercises.)

## Starter guidance

- **Pin the version.** The source tour's value depends on paths being accurate. `pip show vllm` → match the `git checkout <tag>`. Changes between vLLM versions in the block-manager area have been substantive.
- **Use a small model.** 1B–3B instruct keeps forward passes fast and iterations frequent; block-manager events are dense enough to read.
- **Keep prefix caching off** for this exercise. Prefix caching changes what `allocate` does (some blocks already exist); keep that for exercise 02.
- **Make `append_slot` reachable.** Submit at least one request whose output length exceeds `block_size`. A 50-token prompt and `max_tokens=64` guarantees at least three `append_slot` events.
- **Instrumentation stays local.** Any logging changes you make live in your local clone only. Do not upstream.
- **For parallel sampling demo**, use `n=4` in the API; vLLM supports it in the completion API. Verify your engine supports it; if not, see the fallback in Artefact D.
- **For CPU-only runs**, the block-manager code paths are identical; only the attention backend changes. Note the backend in your writeup.

## Fallbacks for constrained hardware

- **CPU-only backend.** vLLM supports CPU; SGLang has limited CPU support; TRT-LLM is GPU-only. If CPU-only, state so and run the exercise with a 125M parameter model (`gpt2`, `pythia-160m`); the block-manager behaviour is observed identically.
- **Small single GPU** (L4 / A10 / consumer card). Perfectly adequate for a 7B instruct model and all five artefacts.
- **ROCm / AMD.** Supported in vLLM / SGLang for subsets of models; verify block-manager feature parity before relying on it in a drill. The code paths are the same; backend differs.

## Acceptance criteria

1. **Artefact A** names at least ten distinct source entry points, with `file:function` references that `grep` can confirm.
2. **Artefact B** reports `num_gpu_blocks` consistent with the Chapter 05 arithmetic for your model / GPU within 10–20%. The log emits ≥ 50 block-manager lines spanning ≥ 1 full request lifecycle.
3. **Artefact C**'s traced lifecycle references at least five of Artefact A's entry points with specific line numbers, and includes `allocate`, at least three `append_slot`s, and `free`, with pool-free-count accounting that balances to zero at the end.
4. **Artefact D** demonstrates copy-on-write with evidence: either engine-side logs showing ref-count ≥ 2 followed by ref-count decrement at write, or a Python-reader script that extracts the same information from engine state.
5. **Artefact E** is one page, has internally-consistent arithmetic, and ends with a concrete recommendation (not a shrug).
6. All five artefacts live in version control; the engine source tour pins to the engine's `git` SHA.

## Common failure modes

- **Chapter-copying.** Your artefact must reference *your* installed version's source. Chapter 01 is deliberately vague on class names because they rotate; your writeup must be concrete.
- **Logging too noisy.** Keep each log line to one record per event; dump sampling parameters to a side file.
- **Confusing block-manager v1 vs. v2 vs. unified** in vLLM. If your version has multiple implementations, name which one is in use (check startup log) and tour that one.
- **Missing swap evidence.** If your workload doesn't trigger preemption, don't fabricate it; state "no preemption observed in this run" and point at where it *would* be logged.
- **No ref-count visibility for copy-on-write.** Some engines don't expose ref-counts in default logs; add a one-line print in your local clone or query the block-manager state via a Python introspection script.

## Keep it

Artefact A (the source tour) and Artefact E (the block-size sketch) are reusable artefacts for the rest of this module and module 110's instrumentation work. Pin them to your team wiki alongside the module 103 inner-loop cheatsheet. Refresh on every engine upgrade. Exercises 02–06 reference this exercise directly.
