# Exercise 06 — Server-vs-Edge Decision Drill

**Time:** 3 hours · **Output:** one one-page server-vs-edge RFC for a chosen product feature, one hands-on on-device runtime walkthrough, one documented RFC defense

## Goal

Make [Chapter 06](../06-edge-and-on-device.md) concrete by **taking a candidate product feature, walking it through the six decision axes, picking a runtime (server or one of the four edge options), and producing a one-page RFC that a product manager and a mobile engineering lead could sign off on**.

## Problem statement

Chapter 06 argues that server-vs-edge is a build-time decision that should be written down in an RFC, not left implicit. This exercise forces you to produce the artefact against a concrete feature. The feature can be a real one from your work, a sketched one from the examples below, or a plausible invented one — the exercise is the discipline of walking the axes, not the specifics of the feature.

You will also put hands on one of the four on-device runtimes for a short walkthrough — enough to have an informed opinion, not enough to deploy production. The walkthrough is paired with the RFC to keep the recommendation honest.

## Prerequisites

- Chapter 06 of this module read.
- A development machine where you can install one of: llama.cpp, Apple MLX (macOS with Apple silicon required), MLC-LLM, or ExecuTorch.
- Access to a stakeholder (a product manager, a mobile engineering lead, a security partner) you can show the RFC to — real sign-off is better than simulated. If you cannot get a real stakeholder, pair with another engineer and swap reviews.
- No specific GPU / hardware requirement.

## Requirements

Five deliverables.

### Deliverable A — the product-feature definition

A one-paragraph description of the feature you will evaluate. Pick one:

- **Private journaling assistant.** User types journal entries in a mobile app; the model summarises and tags them. Target DAU: 50k. Target latency: < 2 s to first draft. Privacy constraint: entries must never leave the device.
- **Offline field-service repair guide.** Technician uses a tablet at a worksite with intermittent connectivity; the model walks them through a diagnostic script. Target DAU: 10k across 500 companies. Latency: < 3 s per turn. Model needs to answer from a bundled knowledge base, not the open web.
- **Voice-first commuter assistant.** User talks to the app while driving; model replies with voice. Target DAU: 500k. Latency constraint: TTFA < 400 ms when on wifi; < 1 s on cellular. Model needs to be interruptible.
- **Code completion in a web IDE.** Browser-based IDE uses an LLM for ghost-text completions. Target DAU: 100k. Latency: < 200 ms per completion. No installation step possible (web only).
- **Educational tutor for primary-school kids.** iPad app for math tutoring. Target DAU: 20k. Latency: < 1 s per reply. Strong moderation and safety constraints; parents want no server-side logging.
- **Your own feature.** Describe it in a paragraph with target DAU, latency SLO, and any constraint that distinguishes it (privacy, offline, cost, device class).

### Deliverable B — the axis walk

A table of the six chapter-06 axes, filled in for the chosen feature:

| Axis | Product constraint | Server answer | Edge answer | Winner |
|---|---|---|---|---|
| Privacy / data-residency | | | | |
| Offline requirement | | | | |
| Cost per user | | | | |
| Latency floor | | | | |
| Model footprint | | | | |
| Device fragmentation and update cadence | | | | |

For the cost row, do the back-of-envelope math: assume a GPU-hour rate (e.g., $2 / h on-demand for an H100 class, $4 on-demand for a B200 class — verify current spot / on-demand numbers for your cloud), an achievable tokens-per-GPU-hour on your model, and the per-user request rate. Compare against the one-time bundle size and per-user update bandwidth for the edge option.

For the device-fragmentation row, name a specific smallest-supported device for the product. Can the model fit (RAM, storage) on *that* device, not the flagship?

### Deliverable C — the runtime walkthrough

Pick one of the four on-device runtimes (llama.cpp, MLX, MLC-LLM, ExecuTorch) that is plausible for your feature. Install it, run a 3B – 7B class model, and capture:

- Install steps (in the artefact) and any gotchas.
- A short prompt / response transcript from the model running locally.
- Measured latency: TTFT, TPOT for a 50-token response.
- RAM / VRAM usage.
- Disk size of the model file.

If you can't run on the device class your product targets (e.g., you don't have an Android phone for an on-device Android app), run on your development machine and document the gap: "my laptop can run a 7B at 20 tok/s; a Pixel 7 at 4-bit quantization will run it at ~5 tok/s" with a cited source for the mobile number, or a `<!-- needs-research: ... -->` marker.

### Deliverable D — the one-page RFC

Fill in the chapter-06 RFC template:

- Title, author, reviewers, status.
- Context (1 paragraph).
- Decision axes (table from deliverable B).
- Runtime choice (if edge) with the pro / con per the four candidates (1 paragraph + one-line-per-option).
- Trade-offs accepted (3–5 bullets).
- Rollout plan (phases, kill criteria, rollback path).
- Open questions (2–4 bullets you don't know the answer to yet).

The whole thing on one printed page. If it spills to two, trim.

### Deliverable E — the defense

Present the RFC to at least one reviewer — a stakeholder if possible, another engineer otherwise — and capture:

- The three strongest objections the reviewer raised.
- Your response to each: either "good point, here's how the RFC changes" (with the edit) or "here's why I'm sticking with the position" (with the reasoning).
- Any open question the reviewer adds that you cannot answer today.

Attach the discussion summary as a follow-on page to the RFC.

## Starter guidance

- **Be honest about the cost math.** The server-vs-edge cost comparison usually tips to edge at high DAU and low per-user QPS, and the other way at low DAU with high per-user QPS. If you've picked a feature where edge is clearly off the table, say so early and spend the ink on the server-side question (which regional presence, which model size).
- **Pick a realistic smallest device.** "Latest iPhone" is not a smallest supported device for a 500k-DAU product. If you're committing to iPhone 12 and up, the on-device model has to fit a 4 GB RAM phone with 4 GB of free app memory — which is a lot tighter than you think.
- **Don't skip the trade-offs-accepted section.** The RFC's value is in naming what you lose. "No server-side logging" is a feature if the product is privacy-first; it is a gaping observability hole if you were counting on it for abuse prevention.
- **Open questions are not a weakness.** A good RFC names what the author does not know. Reviewers will trust the decision more if they can see where the uncertainty is.
- **Keep the walkthrough honest.** If you ran a 3B model on an M1 MacBook and extrapolated to a Pixel 6, say that. The point of the walkthrough is to prevent the recommendation from being paper-only.

## Fallbacks for constrained environments

- **No mobile / iPad / Apple silicon available.** Use llama.cpp on whatever Linux / Windows / Intel-Mac machine you have; the per-platform latency numbers you cite in the RFC come from published community benchmarks (cite them explicitly; mark with `<!-- needs-research: ... -->` if you can't find a strong primary).
- **No real stakeholder to review.** Pair with another curriculum participant or a colleague; do a mock review. Document the exercise as a mock in the deliverable-E summary; it is still worth the discipline.
- **Product feature doesn't match any example.** That's fine; invent one. The exercise is the decision discipline, not the industry relevance of the feature.

## Acceptance criteria

1. **Deliverable A** feature description names DAU, latency SLO, and distinguishing constraint in one paragraph.
2. **Deliverable B** six-axis table filled in with specific numbers (not vague adjectives); cost math shown; smallest-device named.
3. **Deliverable C** runtime walkthrough produced an installable artefact; latency numbers recorded; gap-to-device-class documented.
4. **Deliverable D** one-page RFC complete; fits a single page; rollout plan has phases; kill / rollback criteria named.
5. **Deliverable E** at least one reviewer heard the pitch; three objections captured; your response to each is in the summary.

## Common failure modes

- **"Edge for privacy" without naming the mitigations lost.** The RFC says "we'll run on-device for privacy" and omits that server-side policy enforcement, moderation, and abuse detection are gone. Six months in the product team is surprised.
- **Cost math off by an order of magnitude.** Picking a GPU-hour rate without citing the source, or a tokens-per-GPU-hour that's a vendor headline rather than achievable, makes the server side look either cheaper or more expensive than it is.
- **No smallest-device call.** The RFC says "we'll run the 7B on-device" without saying which device; it works on the flagship iPhone and OOMs on the baseline Android. Name the device class.
- **Runtime chosen for team familiarity, not fit.** An iOS-only product using MLC-LLM because "we know TVM" when MLX fits better; a cross-platform product choosing MLX because the walkthrough was on a Mac. Match runtime to product, not to developer comfort.
- **Rollout plan is one bullet.** "Launch to all users" is not a rollout plan. Phases, canary criteria, and a rollback path are the minimum.

## Keep it

The RFC and the paired runtime walkthrough are the fleet's build-time decision artefact for this feature. Module 114 (cross-functional inference leadership) reuses the RFC template for vendor evaluation and build-vs-buy; the walkthrough serves as the on-device experience the engineer draws on when future features ask the same question.
