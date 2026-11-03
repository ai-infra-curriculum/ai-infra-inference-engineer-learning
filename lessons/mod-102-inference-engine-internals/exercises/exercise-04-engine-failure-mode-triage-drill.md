# Exercise 04 — Engine Failure-Mode Triage Drill

**Time:** 4 hours · **Output:** seven reproduction scripts (or log-replay walkthroughs), one triage runbook per mode, one combined on-call cheatsheet

## Goal

Make [Chapter 04](../04-engine-failure-modes-and-triage.md) concrete by *reproducing* — or, where reproduction isn't feasible on dev hardware, *log-replaying* — each of the seven failure modes, writing the triage move you'd execute, and committing the whole thing as a per-mode runbook you can hand to a new on-call engineer.

## Problem statement

Your inference-platform team has decided that every operator on the on-call rotation must have hands-on triage experience for the common failure modes before taking a shift. The deliverable is a drill book: each of seven modes gets a reproduction (or a log-replay with sources), a diagnosis script, a confirmed-fix, and a prevention note.

The modes, from chapter 04:

1. OOM on long-context
2. Prefix-cache cliff
3. Allocator fragmentation
4. CUDA-graph recompile storm
5. `torch.compile` re-trace
6. NCCL hang on worker crash
7. Slow-rank straggler in TP

## Prerequisites

- One or more GPUs. Modes 6 and 7 need multi-GPU setups to reproduce physically; for them, log-replay from publicly available post-mortems or vendor blogs is an acceptable substitute (see starter guidance).
- A vLLM (or SGLang) deployment you already know how to launch (exercise 01 / 02 output).
- A metrics scrape (even a `watch 'curl -s .../metrics'` loop in a terminal is fine).

## Requirements

Deliver three categories of artefact.

### Artefact A — Seven reproductions (or log-replays)

For each failure mode:

- **Reproduction script or log-replay.** A shell script, Python probe, or documented launch-config + traffic-generator combination that reliably produces the symptom locally. Where reproduction on dev hardware is impractical (NCCL hang, slow-rank straggler), substitute a *log-replay*: pull a public post-mortem, vendor blog, or GitHub issue describing the failure, and annotate the log/trace against chapter 04's claims. Cite sources.
- **Minimum reproduction configuration.** Model, flags, prompt-shape, concurrency — exactly enough to produce the mode without extra noise.
- **Expected metric signature.** The dashboard view that confirms the mode (not just "something is wrong").
- **What you actually saw.** Record the metrics / logs from your run.
- **Any ways the reproduction surprised you.** Reality is instructive.

### Artefact B — Seven runbooks (one page each)

For each mode, author a one-page runbook following this template:

```markdown
# Runbook — <mode name>

## Symptom (what pages / what the dashboard shows)
- metric 1 … metric N (names from your engine's current observability vocabulary)

## First-check metric
- the single most load-bearing metric to look at first

## First hypothesis
- one sentence

## Confirmation steps (≤ 5)
1. …
2. …

## Fix (short term → medium → structural)
- short term: …
- medium: …
- structural: …

## Prevention / regression test
- what to add so this one doesn't recur

## Related failure modes (cascades)
- which modes often follow this one, and how to tell them apart
```

### Artefact C — Combined on-call cheatsheet (one page)

A one-page summary that pulls the seven runbooks into a single triage flow. Minimum:

- A seven-row triage table in the shape of chapter 04's one-page condensed runbook. Augment with the exact metric names from your engine.
- A short "cascades" section: when mode X fires, suspect modes Y and Z next.
- The three top-level moves that come before any mode-specific move: (1) check whether a deploy or config change landed in the last 24h; (2) check the earliest signal in the timeline (not the loudest alert); (3) ping the owner for the suspect layer per the mod-101 role-boundary chapter.

## Starter guidance

Mode-by-mode notes:

- **Mode 1 (OOM):** easy to reproduce. Launch a server with a modest `--max-model-len` and `--max-num-seqs` and send prompts whose combined length × concurrency exceeds the KV pool. Verify the specific error message your engine produces.
- **Mode 2 (prefix-cache cliff):** reproducible with a client that *was* sending a shared prefix and now prepends a per-request timestamp. Measure the before/after cache-hit rate.
- **Mode 3 (fragmentation):** harder to reproduce cleanly. Try a workload with high prefix churn plus small `--max-num-seqs` to see preemption rise while pool usage stays below 100%.
- **Mode 4 (CUDA-graph recompile storm):** reproducible by sending traffic with high batch-shape variability (varying prompt lengths and concurrency) and observing step-time distribution. `--enforce-eager` should make the bimodality disappear — that is the diagnostic test.
- **Mode 5 (torch.compile re-trace):** version-dependent. If your engine's compile integration isn't active, note it as N/A and log-replay from a public post-mortem.
- **Mode 6 (NCCL hang):** requires multi-GPU. Reproduce by sending `kill -9` to one worker during load. If you don't have multi-GPU, log-replay — public NCCL-hang post-mortems from vLLM / PyTorch GitHub issues are an acceptable source.
- **Mode 7 (slow-rank straggler):** requires multi-GPU. If one of your GPUs is thermally throttled or slower than the others, this reproduces for free. If not, log-replay from a performance-engineering blog about per-rank variance.

General guidance:

- **Keep each reproduction small.** A 20-line script is better than a 200-line one.
- **Record versions.** The reproduction is only valid for the specific engine + model + driver version you used.
- **Separate "what the mode looks like" from "what the mode is caused by."** The runbook's symptom section is what pages the on-call; the hypothesis section is what the engineer tests.
- **For log-replays, cite primary sources.** GitHub issues, upstream blog post-mortems, vendor advisories. No invented transcripts.

## Acceptance criteria

1. Each of the seven modes has either a working reproduction script or a cited log-replay. "I didn't get to this one" is not acceptable; "this is log-replay because I don't have multi-GPU" is.
2. Each runbook (Artefact B) fits on one page and follows the template.
3. The triage table (Artefact C) names the current engine's real metric names (not just the chapter's placeholders).
4. For every mode, the "fix" section names at least one short-term knob or action a sleep-deprived on-call engineer could execute in under five minutes.
5. The cascades section lists at least three real cause→symptom chains from your reproductions or log-replays.
6. The whole drill book commits to the team's wiki in a location the on-call rotation actually looks at.

## Common failure modes (of this exercise)

- **Trying to reproduce a hard mode exactly.** NCCL hangs and torch.compile re-traces are often version-specific. If you spend more than 30 minutes trying to reproduce a mode physically, switch to log-replay and move on.
- **Writing runbooks from the chapter without ever running the engine.** The deliverable is grounded by *your* reproductions and *your* metric scrapes. The chapter is the shape; your runbook has the specifics.
- **Ignoring cascades.** The real on-call value is in diagnosing which mode is upstream when multiple fire together. Make sure Artefact C treats this seriously.

## Extensions (optional)

- Add an eighth mode specific to your fleet. Common additions: speculative-decoding acceptance-rate collapse, LoRA adapter load-cache thrashing, streaming SSE back-pressure stalls.
- Automate the triage table — a small script that, given a metric scrape, points at the probable mode. Even a naive classifier is useful as an on-call aide.
- Pair it with a game-day: schedule a session where one team member injects a failure mode (OOM via `--max-model-len=100000` on a small GPU; prefix cliff via a prompt-template diff) and the on-call candidate uses the drill book to triage.

## Keep it

Pin the drill book in the on-call runbook location. Refresh it after every production incident whose root cause lives in the seven modes (or warrants an eighth). Modules 109 (fleet resilience) and 113 (incident response) will reference and extend this artefact.
