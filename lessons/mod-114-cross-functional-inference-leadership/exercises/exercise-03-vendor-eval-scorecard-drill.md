# Exercise 03 — Vendor Eval Scorecard Drill

**Time:** 6 hours · **Output:** a complete three-page vendor-eval artefact (per-vendor detail pages, summary comparison, decision cover memo) for at least three real vendors against the five-axis rubric, including a benchmark on your (or a proxy) traffic sample.

## Goal

Make [Chapter 03](../03-vendor-eval-scorecard.md) concrete. Run a defensible vendor evaluation for a plausible inference workload class and produce the signed-off artefact. The exercise is complete when a sponsor could pick a vendor off your summary comparison and a successor engineer could re-open your evaluation with a clear understanding of why you scored it that way.

## Problem statement

Pick a plausible workload class to evaluate vendors for — ideally one surfaced in your exercise-02 RFC. The archetype candidates: embedding, transcription, long-tail chat on a commodity model, enterprise chat on a dedicated-capacity tier, image generation / diffusion, or a reasoning-model workload.

Evaluate at least three *real* vendors (not synthetic placeholders) from the current managed-inference landscape — representative candidates: Fireworks, Together, Baseten, Modal, NVIDIA NIM, Groq, Cerebras, Replicate, hyperscaler inference offerings. Pick vendors that actually serve the workload class.

Produce the three-page vendor-eval artefact from chapter 03 (per-vendor detail pages, summary comparison, decision cover memo) with a weighting defensible for your workload class.

## Prerequisites

- Chapter 03 read.
- Access to each vendor's public documentation, pricing, SLA, DPA / privacy policy, status page.
- A representative traffic sample from your (or a plausible-hypothetical) workload — anonymised; enough prompts to cover the P50 and P95 request shapes of the workload class.
- An embedding model or scoring utility for the equivalence benchmark where applicable.
- API access to each vendor (sign up for free tiers; invest 1–2 hours in actually running requests). If your workload is compliance-sensitive and signup is not possible, note this and run the benchmark on an adjacent proxy.

## Requirements

Six deliverables.

### Deliverable A — the workload scoping memo

A one-page preamble that scopes the evaluation:

- Which workload class is being evaluated; typical prompt shape; typical output shape; SLO targets; traffic volume; compliance posture.
- The weighting choice (chapter 03 shows two example weightings; yours defends its own by the compliance / cost / performance profile of this workload).
- The three (or more) vendors shortlisted and why each shortlisted (and which plausible ones were excluded from the shortlist with a reason).

### Deliverable B — the per-vendor detail pages

One page per vendor. Chapter 03's structure:

- Vendor name, product, URL, documentation reviewed, eval date, eval owner.
- The five-axis scoring table (SLA, price, feature parity, exit cost, data boundary) with scores 1–5, the weight applied, and a one-sentence evidence pointer per sub-dimension.
- Attached evidence: the benchmark on your traffic (feature parity), the uptime / status-page extraction (SLA), the price quote / effective rate at forecast (price), the DPA summary (data boundary), the exit-contract clauses (exit cost).
- Open questions for the vendor's sales / support contact.

### Deliverable C — the benchmark on your traffic

For each vendor, the feature-parity benchmark that chapter 03 requires:

- Run the representative traffic sample against each vendor's API with equivalent parameters.
- Record TTFT, TPOT, ITL, E2E, error rate, rate-limit hits.
- Score response equivalence against a baseline (your self-host, or an authoritative reference model). Use the response-diff pattern from module 113 chapter 01 — cosine similarity, refusal-rate, schema conformance as applicable.
- Produce a one-page comparison panel per workload class.

If running against real vendors is not possible for your setup, state the constraint explicitly and run the benchmark against public endpoints for comparable models; note the proxy.

### Deliverable D — the summary comparison

One page total. A single table with the vendors as rows and the five axes + weighted total + recommendation as columns. Beneath the table, a short paragraph per axis explaining the weighting choice and any notable per-vendor outliers.

### Deliverable E — the decision cover memo

One page. Chapter 03 structure:

- The workload cells from the exercise-02 RFC this scorecard serves.
- The ranked vendors.
- The specific placement proposed (which cells go to which vendor, if any).
- Sponsor sign-off line (simulate if necessary).
- The "when does this change" triggers that would reopen the scorecard.

### Deliverable F — the DPA / contract review notes

Not a public artefact, but a required file: per-vendor notes on the DPA (data-processing addendum) and the standard contract terms. What's the logging default? What's the training-on-data default? Named subprocessors? Data region flexibility? Termination clauses? Even for an evaluation against public terms-of-service (no enterprise contract read), capture what the public doc says. Mark `<!-- needs-research: contract-level review required before contract signature; this evaluation reads the public terms only -->` where legal review would be needed.

## Starter guidance

- **Pick vendors that actually serve your workload.** A scorecard that scores Groq for embeddings or Cerebras for image generation is scoring on absence; drop those from the shortlist and name why.
- **Evidence over marketing.** Every score needs a pointer to the evidence. Score defended as "they say they're good at this" is a 1-star score waiting to be overturned.
- **The benchmark is the hard part.** Budget 2–3 hours for Deliverable C. Prepare the traffic sample first, then run it through all vendors in sequence, then score.
- **Historical uptime takes work.** Vendor status pages don't always expose aggregate uptime — you often have to extract incident timestamps and reconstruct. If the vendor's status page has <6 months of data, note that as a risk.
- **Read the DPA.** It's often a PDF behind a signup gate. Public privacy policy ≠ DPA; mark the limitation if you only read the public doc.
- **Weight the axes before scoring.** If you weight after scoring, you'll unconsciously weight to produce your preferred answer. Lock the weights before pulling evidence.

## Fallbacks for constrained environments

- **No API keys for one or more vendors:** evaluate against their documentation and a public endpoint for a comparable model; mark the gap; the feature-parity score comes with a caveat.
- **No real traffic sample:** build a synthetic sample of 50–100 prompts covering the workload's prompt-length and complexity distribution; document the synthesis method.
- **No self-host baseline:** benchmark vendors against each other instead of against a baseline; adjust the equivalence scoring to be pairwise.
- **No embedding model for scoring:** use normalised edit distance or a rubric-based manual scoring on a sample; document the choice.
- **Status page archives behind a signup gate:** use publicly archived snapshots (Wayback Machine) where available; name the limitation.

## Acceptance criteria

1. **Deliverable A** scopes the evaluation, defends the weighting choice, and explains the vendor shortlist.
2. **Deliverable B** has one page per vendor; each axis scored with evidence pointers.
3. **Deliverable C** has the benchmark run on at least a plausible-proxy traffic sample against each vendor; latency numbers measured; equivalence scored where applicable.
4. **Deliverable D** has the summary table and the per-axis weighting paragraph.
5. **Deliverable E** has the ranked vendors, the per-cell placement, the sign-off block, and the reopen triggers.
6. **Deliverable F** has per-vendor DPA / contract notes; the limitations of reading public docs only are clearly marked.
7. The weighting was locked *before* scoring (state so in Deliverable A).
8. Every score has an evidence pointer; no scores based on marketing alone.

## Common failure modes

- **Scoring the vendor's blog.** Benchmark on your traffic, not on their benchmark chart.
- **Marketing SLA scored.** Historical uptime + credit schedule + response SLA; the marketing number is a benchmark of self-image.
- **First-year price locked in.** Price the renewal; model the lock-in erosion.
- **DPA unread.** You cannot score data boundary on the privacy page alone.
- **Weighting after scoring.** Lock weights first; defend them in prose; the audit trail matters.
- **Shortlist too large.** Six vendors is six 1-page detail pages; some will be weakly-evaluated. Three is enough for most workload classes.

## Keep it

The three-page artefact, the benchmark data, the DPA notes. The artefact is the input to exercise 02's RFC (if you did them in the reverse order, the RFC references this scorecard). The chapter-01 quarterly migration-roadmap review re-opens the scorecard when triggers fire; keep the raw benchmark data so re-running is cheap.
