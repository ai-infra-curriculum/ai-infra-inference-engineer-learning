# Chapter 03 — Vendor Eval for an Inference Provider: SLA, Price, Feature Parity, Exit Cost, Data Boundary

> Chapter 02's RFC says "evaluate vendors on a scorecard." This chapter builds the scorecard. It is deliberately boring on purpose — a defensible vendor decision is one where the arithmetic and the evidence are obvious enough that any peer or successor can re-run them.

## Motivation

The vendor evaluation is where sponsors get things wrong fastest. The common failure modes are not technical; they are evidential. A team sits through two vendor demos, picks the one with the better slide deck, and signs a 24-month contract on marketing SLA numbers and promotional pricing. Nine months in, the renewal quote rises, a feature the consumer tracks need lands last, an outage drops the business for an afternoon with a credit that covers the pizza order, and the exit — which nobody priced — becomes a quarter of unplanned work.

The scorecard this chapter builds is the defence against that path. It is a short, written artefact (one page summary, one page per vendor's detailed scoring) that reduces the vendor-selection conversation to "we scored it, here are the dimensions, here is the evidence; what question do you want to challenge." Sponsors sign a scorecard. They do not sign a vibe.

By the end of this chapter you can:

- Build a vendor-eval scorecard with five axes — SLA, price, feature parity, exit cost, data boundary — each with sub-dimensions, weights, and evidence requirements.
- Convert a vendor's marketing claims into evidence that lands on your rubric.
- Score a vendor on *your* workloads, not on a benchmark the vendor advertises.
- Write the scorecard as a defensible document a sponsor can sign and a successor can re-open.
- Avoid the five classic vendor-eval traps.

## The five-axis frame

The vendor-eval scorecard uses five axes. Each axis is scored 1–5 with a short written defence; weights sum to a total; the total is advisory, not mechanical. Weights depend on workload — a compliance-sensitive workload weights `data_boundary` heavily; a cost-sensitive workload weights `price` heavily. Name the weights up front.

| Axis | What it measures | Typical sub-dimensions |
|---|---|---|
| **SLA** | Reliability as a *contract* | Marketing uptime; contractual uptime; credit schedule; historical uptime; incident response SLA; mitigation time |
| **Price** | Cost per 1M tokens, *fully loaded* | Base `$ / 1M in/out`; tier thresholds; minimums; overage behaviour; renewal price history; promotional-vs-steady-state |
| **Feature parity** | Does the vendor serve *your* workloads | Model availability; API surface (streaming, structured output, tool calling, MCP); latency / throughput on your prompt distribution; fine-tuning / adapter support; multimodal; reasoning-model surface; speculative decoding support |
| **Exit cost** | Can you leave if the decision reverses | Model-availability portability; prompt / context portability; embedding portability; data return / deletion; fine-tune / adapter weight return; contract termination clauses |
| **Data boundary** | Who sees your prompts and your outputs | Logging default; training-on-customer-data default; data region; subprocessors; encryption posture; access logs; compliance certifications (SOC 2, ISO 27001, HIPAA, FedRAMP where applicable) |

Five axes is not five numbers; each decomposes into evidence. The remaining sections build the sub-rubrics.

## SLA — the contract, not the marketing

Marketing SLA is a slogan. Contractual SLA is a document with a credit schedule. Historical SLA is a distribution the vendor's own status page reveals. All three go on the scorecard.

### Sub-dimensions

- **Marketing uptime.** The number on the vendor's homepage. Record it. It's useful only as a benchmark of the vendor's self-image.
- **Contractual uptime commitment.** The number in the signed contract. Often lower than marketing. Insist on reading it before scoring.
- **Credit schedule.** What the vendor pays if they breach. A typical schedule: 10 % credit for below-99.9 %, 25 % for below-99 %, 100 % for below-95 %. Credits are usually service-credit against the next period, not cash. Price this against your own downtime cost.
- **Historical uptime.** The vendor's status page over the last 12–24 months. Count the incidents; measure the aggregate downtime; compute the achieved uptime against the committed. Vendors rarely hit their own marketing number over 24 months; some miss their contract number too. <!-- needs-research: historical uptime is the vendor's status-page evidence at the time of the eval; vendor status pages drift and incidents are sometimes retroactively reclassified. Capture a dated snapshot. -->
- **Incident response SLA.** First response time, update cadence during an incident, mitigation time. Named in the contract or disclosed on request.
- **Support tiers and escalation path.** Named human contact vs. a ticket queue; direct-engineer access for critical incidents; shared Slack or dedicated channel for enterprise tiers. Who do you reach at 03:00 during a Sev-1.

### Scoring guidance

- **5 — Excellent.** Contractual ≥ 99.95 %, credit schedule is meaningful (≥25 % at ≥99 %, cash credits above a floor), historical matches contract, named support engineer, enterprise channel.
- **3 — Acceptable.** Contractual ≈ 99.9 %, credit covers one incident, historical close to contract, support via ticket queue with SLA.
- **1 — Unacceptable.** No contractual commitment, "best effort," no credits, support only via public channels.

### Common traps

- **The marketing number is the headline.** A contractual 99.9 % with a 24 % historical breach is scored 1, not 4, regardless of marketing.
- **Credits are not cash.** A service-credit against the next period's bill does not pay for the downtime your business took. Score credit schedules by *economic value to the business*, not by percentage.

## Price — fully loaded, including the renewal

Vendors quote price as `$ per 1M input tokens` and `$ per 1M output tokens` (plus sometimes `$ per inference call`, `$ per image`, `$ per audio minute`). The scorecard reflects the whole bill, not the headline.

### Sub-dimensions

- **Base rate per 1M input / output tokens** for *each* model the consumer tracks need. A vendor cheap on their default model and expensive on the flagship you actually use is cheap in demos only.
- **Tier thresholds.** Many vendors have volume breakpoints — $X above 10M tokens / month, $X–10 % above 100M / month. Forecast your traffic and compute the effective rate at the forecasted tier.
- **Minimums.** Monthly or annual commits that bill even if unused. For a workload that varies 5× between peak and off-peak, a high minimum is a steady tax.
- **Overage behaviour.** If you exceed a tier, does the price step down for the overage (good), step up (bad), or stay flat (expected)? Does rate-limiting kick in instead (depends on your SLO).
- **Renewal price history.** The number almost every eval misses. Ask the vendor sales lead for the price history of comparable customers' renewals, or infer from public pricing changes over the last 24 months. Promotional pricing at year 1 that reverts to list price at year 2 is a real-world pattern. <!-- needs-research: public pricing pages change frequently and promotional discounts are often off-list; the scorecard should attach dated screenshots or contract language at eval time. -->
- **Cost of secondary features.** Fine-tuning host fees, adapter host fees, dedicated-instance fees for enterprise customers, logging retention fees, egress fees. Each is an optional add-on that becomes required if you use the feature.

### Scoring guidance

- **5 — Excellent.** Steady-state effective rate (post-tier, post-overage) comfortably below self-host TCO with room; contract has caps on year-over-year increases; no large minimums; secondary features priced in a way that doesn't surprise.
- **3 — Acceptable.** Rate close to self-host TCO at forecasted volume; minimums manageable; renewal trajectory documented in contract.
- **1 — Unacceptable.** Headline rate competitive, effective rate much worse (because of minimums / overage / renewal), or price history shows large jumps at renewal.

### Common traps

- **Scoring the first-year price, not the renewal.** A $0.60 / 1M year-1 price that becomes $1.20 / 1M at renewal is $1.20 / 1M in the TCO for the horizon of the RFC.
- **Ignoring the minimum.** A 100M-tokens/month minimum on a workload that runs 20M/month in off-peak months is paying for 80M/month of nothing.
- **Comparing against the wrong self-host.** Self-host TCO from chapter 02 is the comparison baseline, not the quoted on-demand GPU price.

## Feature parity — on *your* workloads, not benchmarks

A vendor benchmark is a signal that the vendor's engine can hit certain numbers under certain conditions. It is not evidence that the vendor serves *your* workloads well.

### Sub-dimensions

- **Model availability.** Does the vendor host the exact model (name, version, revision) your consumer tracks consume? If they host a close variant (Llama-3.1-70B-Instruct vs. a vendor-specific fine-tune), the eval must verify behaviour on your evals, not theirs.
- **API surface.** OpenAI-compatible completion / chat / responses endpoints; streaming semantics (SSE, chunking, back-pressure); structured output (JSON Schema, Pydantic, Outlines / XGrammar backends); tool calling (parallel tool calls, streamed tool calls); embeddings endpoint; image / audio endpoints; Model Context Protocol server-side if applicable. Module 111 names the surface this role evaluates.
- **Context length actually served.** A vendor's "128k context" claim often comes with a QoS footnote — slower, lower acceptance of max-tokens, higher cost. Measure TTFT at your long-context tail.
- **Latency on your prompt distribution.** The vendor's latency chart is for their chosen prompt shape. Your chat may have a longer system prompt, a specific tool-call pattern, a RAG-sized input. Replay a *representative sample of your real traffic* (anonymised) through the vendor's API and measure TTFT, TPOT, ITL at P50 / P95 / P99.
- **Throughput at your traffic shape.** Same discipline for QPS-like measurements. Burst behaviour matters — vendors often rate-limit by RPM, RPD, TPM, TPD; verify your peak sits comfortably below the ceiling.
- **Multi-model / adapter support.** If your platform routes dozens of models or thousands of adapters (module 106), can the vendor serve the fan-out, or will it force a model-per-workload split?
- **Fine-tuning / custom-model support.** If you have custom fine-tunes, does the vendor host them? At what price? With what cold-start characteristics?
- **Speculative decoding / draft-model support.** If you depend on speculative-decoding gains (module 108), does the vendor's engine expose the equivalent? What is the acceptance-rate observability?
- **Multimodal.** If your workload includes VLM, diffusion, speech (module 112), does the vendor support it, at what parity, with what price.
- **Reasoning-model surface.** If your consumer tracks consume reasoning models, does the vendor expose thinking budgets, interruption, reasoning-vs-answer channels, and the token accounting that drives billing.

### Scoring guidance

- **5 — Excellent.** Every model your consumer tracks consume, plus latency / throughput at your traffic shape that meets or beats your SLO, plus the full API surface. The evidence is a benchmark you ran on your traffic, not a vendor chart.
- **3 — Acceptable.** Models and API surface cover the majority; latency within SLO on the central workload classes; one or two secondary features missing or degraded.
- **1 — Unacceptable.** Core model or API surface missing; latency on your traffic shape doesn't make SLO; fine-tuning / adapter / multimodal gap that forces a workaround.

### The required artefact — the benchmark on your traffic

A feature-parity score without a benchmark on your traffic is marketing, not evidence. The benchmark discipline:

1. Capture a representative, anonymised sample of your real traffic — a mix of workload classes, prompt lengths, output lengths, tool-call shapes. Enough to cover the P50 and P95 shapes of each class. Trace capture lives at module 110 chapter 03.
2. Replay against the vendor's API with the same parameters your production uses. Record TTFT, TPOT, ITL, E2E, error rate, rate-limit hits, token accounting.
3. Compare to your self-host production (or self-host benchmark if you're green-fielding) on the same traffic sample on the same day.
4. Produce a one-page comparison panel per workload class.

That panel is the feature-parity evidence. Without it, the scorecard is advisory at best.

## Exit cost — can you leave

Exit cost is the vendor-eval axis that almost no one asks about at demo time and everyone needs when the decision reverses. The RFC chapter 02 already prices it in TCO; this axis scores *how reversible the decision is*.

### Sub-dimensions

- **Model-availability portability.** If the vendor hosts an open-weights model, the same model is available at a competitor or self-host — exit is cheap. If the vendor hosts a proprietary model only they serve, the consumer tracks must migrate to a different model entirely — exit is expensive.
- **Prompt / context portability.** If the vendor's prompt format, system-prompt semantics, tool-call protocol, or structured-output dialect differs from the OpenAI-compatible baseline, every consumer prompt has to be reshaped at exit. Score the dialect distance.
- **Embedding portability.** If you store vendor-specific embeddings in a vector DB, exit requires re-embedding every record with a new provider. For a 100M-row vector store, that is a real cost.
- **Fine-tune / adapter weight return.** If you paid the vendor to fine-tune a model for you, do you get the weights at exit? Many vendors do not return weights — the fine-tune is theirs. Named in the contract.
- **Data return / deletion.** How are your prompts, outputs, and any stored artefacts (files, assistants, threads) returned and deleted at exit. Timeline. Evidence of deletion.
- **Contract termination clauses.** Notice period. Minimum remaining commit at termination. Any clauses that make early termination punitive.
- **Integration surface.** How many lines of consumer code reference vendor-specific SDK calls, headers, request shapes, response shapes. A thin integration (just the OpenAI-compatible endpoint) has low exit cost; a thick one (vendor SDK, vendor-specific assistants / threads / files API) has high.

### Scoring guidance

- **5 — Excellent.** Open-weights models hosted; OpenAI-compatible API dialect; contract returns all customer-owned artefacts at exit; thin integration; notice period under 30 days; no early-termination penalty.
- **3 — Acceptable.** Mostly portable, with one or two sticky points (fine-tune weights ours but at a cost; some vendor-specific features load-bearing).
- **1 — Unacceptable.** Proprietary model only the vendor serves; vendor-specific dialect; fine-tune weights not returned; punitive termination.

### The rehearsed exit

The score is strongest if you have actually rehearsed the exit path. A paper exit plan is a hypothesis; a tested exit plan is evidence. Chapter 01's quarterly cadence includes a "failover rehearsal" slot; wrapping the vendor-exit path into that rehearsal adds a weeks-of-work guarantee. <!-- needs-research: rehearsing a vendor exit requires coordination with the consumer tracks and a controlled rollback — plan it, don't stumble into it. -->

## Data boundary — who sees your prompts

Data boundary is the compliance and privacy axis. Get it wrong and the cost is not a bill — it is a regulator letter.

### Sub-dimensions

- **Logging default.** Does the vendor log prompts and completions by default? For how long? With what access controls? Can logging be disabled, and does disabling it affect support (i.e. "we can't help you debug without logs")?
- **Training on customer data default.** Does the vendor use your prompts to train their models by default? Opt-out or opt-in? Contractual language. This is the central question for most enterprise contracts.
- **Data region.** In which regions does the vendor process requests? Store logs? Any data crossing a region the compliance team doesn't allow.
- **Subprocessors.** What third parties does the vendor use (hyperscaler for compute, logging provider, analytics). Named in a data-processing addendum (DPA).
- **Encryption posture.** TLS in transit (table stakes). At-rest encryption of logs. Customer-managed keys where applicable.
- **Access logs.** Can you audit who inside the vendor accessed your data. Named-individual access. SOC 2 Type II typically covers this.
- **Compliance certifications.** SOC 2 Type II; ISO 27001; HIPAA BAA if your consumer tracks require it; FedRAMP where applicable; GDPR adequacy for EU workloads; DPF (Data Privacy Framework) for EU-US. The scorecard lists the certifications relevant to *your* consumers.
- **Shared-fleet vs. dedicated.** Shared fleet means your requests batch with other tenants' requests on the same GPUs. Dedicated means your capacity is isolated, often at a cost premium. Compliance teams sometimes require dedicated.

### Scoring guidance

- **5 — Excellent.** Logging opt-out by default or absent; no training on customer data by contract; data region flexible; subprocessors named with DPA; all required certifications current; shared or dedicated as the contract elects.
- **3 — Acceptable.** Logging and training governed by a signed enterprise contract; certifications current; one or two gaps that compliance has accepted.
- **1 — Unacceptable.** Default logging with no opt-out; data potentially used for training; region policy incompatible with compliance; missing required certifications.

### Common traps

- **Marketing claims the consumer-grade default.** The vendor's enterprise tier may have the right data-boundary posture, but the free-tier policy is what shows up at the first customer-support conversation. Score the *tier you will sign*.
- **DPA not read.** The DPA (data-processing addendum) is where the subprocessor list and the data-handling specifics live. If nobody on your side has read it, the scorecard is incomplete.

## The scoring sheet and the summary artefact

The scorecard is a document, not a spreadsheet alone. The structure:

### Per-vendor detail page (one page per vendor)

- Vendor name, product, URL, contract version reviewed, eval date, eval owner.
- The five-axis scoring table with scores, weights, and a one-sentence evidence pointer per sub-dimension.
- Attached evidence: the benchmark run on your traffic (feature parity), the uptime extraction (SLA), the price quote (price), the DPA summary (data boundary), the exit-contract clauses (exit cost).
- Open questions and the vendor-side contacts for each.

### Summary comparison (one page total, sponsor-readable)

- Three to five vendors in a single table.
- Columns: the five axes plus a weighted total and a text "recommendation" cell.
- A short paragraph beneath each column explaining the weighting choice.

### Decision cover memo (one page)

- The workload cells from the chapter-02 RFC that this scorecard serves.
- The ranked vendors.
- The specific placement proposed (which workload cells go to which vendor, if any).
- Sponsor sign-off line.

The three pages together are the vendor-eval artefact. Sponsor signs the cover memo; peer-track leads sign the summary; the detail pages go to the archive. Successor engineer re-opens the archive when a trigger from chapter 02 fires.

## Scoring on *your* weighting, not a generic one

A critical discipline: weights are chosen for *your* workload, not copied from a template. Two illustrative weightings to show the frame:

**Weighting for a cost-sensitive, open-weights workload class** (e.g. free-tier chat on a commodity model):

| Axis | Weight |
|---|---|
| SLA | 15 % |
| Price | 40 % |
| Feature parity | 25 % |
| Exit cost | 10 % |
| Data boundary | 10 % |

**Weighting for a compliance-sensitive enterprise workload class** (e.g. enterprise chat on dedicated capacity):

| Axis | Weight |
|---|---|
| SLA | 25 % |
| Price | 15 % |
| Feature parity | 15 % |
| Exit cost | 20 % |
| Data boundary | 25 % |

The weighting change is not cosmetic. The same vendor can score 5 on data boundary and 2 on price; the first workload class rejects them, the second embraces them. The scorecard works *because* it allows the decomposition from chapter 02 to flow through.

## Five classic vendor-eval traps

- **Demo-day decision.** The team watches two demos and signs. Evidence: zero. Fix: no decision without the per-vendor detail page and the benchmark on your traffic.
- **Marketing-only SLA.** Score copies the vendor's headline uptime. Fix: historical status-page extraction and credit schedule.
- **First-year price locked in.** Scorecard uses the promotional rate. Fix: the renewal history and the contract-caps are the price axis's core.
- **Benchmarks from the vendor's blog.** Feature parity scored on the vendor's own graphs. Fix: replay your traffic.
- **Unread DPA.** Data boundary scored on the vendor's privacy page. Fix: read the DPA; score the subprocessors and the logging default.

## Summary

- Vendor eval is a defensible, signed document. Five axes: SLA, price, feature parity, exit cost, data boundary. Each decomposes into evidence.
- SLA is contract + credit schedule + historical, not marketing uptime.
- Price is fully loaded — tiers, minimums, overage, renewal history, secondary features.
- Feature parity is scored on a benchmark against *your* traffic sample, not a vendor benchmark.
- Exit cost scores portability of models, prompts, embeddings, fine-tunes, data; thin integrations and open-weights models score high; a rehearsed exit path lifts the score.
- Data boundary scores logging default, training-on-data default, region, subprocessors, encryption, access logs, certifications, shared-vs-dedicated. Read the DPA.
- Weights are chosen for the workload, not copied. Different weights can flip the decision on the same vendor.
- The artefact is three pages: per-vendor detail, summary comparison, decision cover memo. Sponsor signs the cover; peer leads sign the summary; detail pages archive for successor re-opens.
- Chapters 04 and 05 move from the vendor-facing scorecard to the peer-facing interface contracts the program also needs.
