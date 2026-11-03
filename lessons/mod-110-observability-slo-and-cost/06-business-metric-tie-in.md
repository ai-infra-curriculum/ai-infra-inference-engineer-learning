# Chapter 06 — Business Metric Tie-In: KPI per TTFT ms, Cost per Revenue Token, Is It Worth the Ask

> Chapters 01 – 05 produced signals, SLOs, dashboards, cost numbers, and a Pareto chart. They are necessary but not sufficient. An infrastructure team that cannot translate a 100 ms TTFT improvement into a dollar value or a KPI lift will lose prioritisation fights to teams that can. This chapter translates. The goal is not to make every inference engineer a product manager — it is to write arguments in the form that product, finance, and engineering leadership will all read, and to resist three common antipatterns in that translation.

## Motivation

Three real-shaped scenarios from inference-platform teams:

- **Engineer-driven:** "Our TTFT_P99 was 1.2 s and I got it to 480 ms." Product says "ok, cool." Six months later, when the team asks for another headcount to continue this work, product says they don't see the impact. The engineer's output didn't have a business shape, so it didn't register in the prioritisation ritual.
- **Executive-driven:** "A competitor shows < 200 ms first-token. We must match them." The team spends a quarter driving TTFT from 400 ms to 190 ms. Shipping increases cost 50 %. Retention doesn't move — the user population never noticed the difference below 400 ms. The competitive-benchmark-chasing was the wrong prioritisation; the team had the signal to detect it (dashboard 4 from chapter 03 showed cost rising with no revenue lift), but no one was asked to read the dashboard in that frame.
- **Partner-driven:** Product asks, "will enabling speculative decoding regress accuracy?" The engineer answers "no, it is mathematically equivalent." Product doesn't believe it (reasonably — "mathematically equivalent" is a technical claim about the sampling distribution, and product doesn't speak that language). The conversation stalls for weeks. The missing piece is a *business-shape* answer: "speculative decoding doesn't change the output distribution; we validated on your eval harness and saw no delta in [KPI]."

Each scenario is a translation failure, not an engineering failure. By the end of this chapter you can:

- Measure product-KPI deltas against infrastructure deltas (TTFT ms, TPOT ms, cost per 1M tokens, cache-hit % change).
- Build the funnel linkage from a technical knob to a dollar value.
- Author the one-page memo that answers *"is this optimisation worth the ask."*
- Recognise the three antipatterns that the discipline exists to resist.
- Cite the specific places where the inference-engineer scope ends and the partner (product analyst, finance partner, applied scientist) takes over.

Chapter 04 produced the cost side; chapter 05 produced the Pareto side; this chapter closes the loop.

## The funnel — how a TTFT ms becomes a dollar

The linkage from infrastructure to business is a funnel. Typical shape:

```
    inference_knob (e.g. max-num-seqs ↑)
        ↓
    inference_metric (TTFT_P95 ↑ or ↓)
        ↓
    product_metric (session-length, bounce-rate, task-completion, retention)
        ↓
    business_metric (revenue, cost, LTV, churn)
```

Each arrow is a *measured* relationship, not an assumed one. The arrow from inference metric to product metric is the hardest; it requires an A/B test, a staged rollout, or a natural experiment. The arrow from product to business is usually owned by a product analyst and already measured; inherit their work.

### Examples of the TTFT → product arrow

Real-shape examples (always cite the source for your fleet; these numbers are illustrative of the *pattern*, not the magnitude):

- **Streaming-chat product:** TTFT below ~400 ms is roughly imperceptible to users; above ~800 ms the session-abandonment rate climbs. The dose-response curve has a plateau and a cliff. Moving from 450 ms to 350 ms *may not* move the KPI; moving from 900 ms to 700 ms *likely will*.
- **Code-completion product:** TTFT below ~200 ms is required to beat the user's tab-trigger threshold (they will have typed the token themselves before the completion arrives). TTFT 300 ms to 200 ms is often transformative; TTFT 190 ms to 150 ms has diminishing returns.
- **Agent / tool-use:** Latency compounds across turns. An agent that averages 4 turns at 2 s each is 8 s total; shaving 500 ms per turn is 2 s off total. The product impact is on time-to-task-completion, which shows up in user retention over a week.
- **Reasoning:** The user expects latency. TTFT matters less; TPOT matters because a 30-second thinking phase at 50 tokens / sec vs. 30 tokens / sec is an extra 20 seconds. The product impact is on willingness-to-wait, measurable by abandonment during generation.

<!-- needs-research: dose-response curves for TTFT / TPOT impacts on product KPIs are highly product- and audience-specific. Papers like Nielsen's classic "response times" numbers (<https://www.nngroup.com/articles/response-times-3-important-limits/>) give human-perception thresholds but not the exact revenue dose-response for an LLM product. Verify against your product's analytics, not a cross-industry rule. -->

### How to measure the arrow

Three techniques, cheapest-to-most-rigorous:

- **Dashboard overlay (weakest).** Plot the inference metric and the product metric on the same time axis for the last quarter. If a known inference regression / improvement visibly moves the product metric, you have a correlation. Not causal, but cheap.
- **Natural experiment.** When a release goes out that moves an inference metric (e.g. deploying chunked prefill drops TTFT 200 ms overnight), measure the product metric before and after with a difference-in-differences against a control cohort. Semi-causal.
- **A/B test.** Route a cohort of users to a config with a different inference metric and measure the product metric. Causal; expensive. Chapter 04 of module 113 (canary methodology) is the framework.

The right technique depends on the stakes. A proposal to migrate to a new provider is worth an A/B; a routine Pareto-point re-tune may only need a dashboard overlay.

## Cost per revenue token

The other side of the arrow is cost. For a product that charges per usage (API resale, per-seat subscription where usage is bounded), define:

```
revenue_per_1M_output_tokens = total_revenue_in_window / (output_tokens_served_in_window / 1e6)
cost_per_1M_output_tokens    = (from chapter 04 model)

gross_margin_per_1M_output_tokens = revenue - cost
margin_ratio                      = (revenue - cost) / revenue
```

Three disciplines:

- **Match the window.** If `revenue` is monthly recognised and `cost` is daily averaged, you get a misleading mismatch.
- **Match the attribution.** If a user's subscription covers both inference and other features, you cannot divide the whole subscription by tokens served; allocate.
- **Report ratio and absolute.** The ratio is a KPI; the absolute dollar is what finance reports to leadership.

For a product that does not charge per usage directly, the equivalent number is `inference_cost_per_active_user`. The conversion:

```
inference_cost_per_active_user = 
    cost_per_1M_output_tokens × mean_tokens_per_user_per_month / 1e6
```

That number compared to average revenue per user (ARPU) tells you what fraction of the user's value goes to inference compute. For most products, if that ratio exceeds ~30 %, the business economics are strained; if it is < 5 %, inference cost is not the bottleneck and optimising it is a lower-ROI activity than the engineering team realises.

## Latency-attributable churn

Latency regressions sometimes don't show in a dashboard overlay but show in the *survival curve* — users exposed to a bad latency week are slightly more likely to leave over the following month. Measuring this:

- Define the latency event: e.g. "any day the user experienced > N requests above the TTFT SLO."
- Build a cohort analysis: users exposed in week W vs. matched users not exposed.
- Measure retention at week W+4, W+8, W+12.
- A statistically significant delta is the latency-attributable churn signal.

This is applied statistics and is typically owned by a product analyst; the inference engineer's job is to produce the exposure signal (which days / users saw bad latency) in a form the analyst can consume. Dashboard 4 from chapter 03 is where this data lives; the exposure flag is a derived series the inference team maintains.

<!-- needs-research: the dose-response and the survival-curve shift from latency are heavily product-specific. The methodology is sound in general; the specific numbers in your product must be measured, not borrowed. -->

## The "is it worth the ask" one-page memo

Every proposed optimisation should fit on a page in this shape. The template:

```
# Memo: Enable speculative decoding on chat workload

## The ask
Enable speculative decoding on the chat workload class with a (Llama-3.1-70B, Llama-3.1-8B) draft pair.

## The expected impact

### Infrastructure
- cost_per_1M_output_tokens: $0.34 → $0.13 (−62%)       [chapter 04 model, assumptions in appendix]
- TTFT_P95: 420 ms → 430 ms (+10 ms, well inside SLO)   [chapter 05 sweep, run 2026-11-01]
- TPOT_P95: 48 ms → 23 ms (−52%)                        [chapter 05 sweep]
- new ops burden: acceptance-rate monitoring, (target, draft) lifecycle (module 108)

### Product
- For chat, TTFT is already well below the user-perception threshold; the TTFT regression is a non-impact.
- TPOT reduction is below the user-perception threshold at this baseline (user reads at ~250 ms / token); non-impact on primary KPI.
- Latency improvement likely imperceptible; cost improvement is the full business case.

### Business
- Fleet cost at current traffic: $520k / month in chat.
- Expected cost at new config: $200k / month in chat (at current traffic).
- Expected monthly savings: $320k.
- Expected annual savings: $3.8M.

## The cost of the ask
- 3-4 weeks of inference-engineer time to deploy and stabilise.
- 2 weeks of lifecycle-contract setup (module 108 chapter 04).
- Ongoing: 1 on-call shift per week absorbs the new monitoring burden.
- Risk: if α drifts (module 108 chapter 03), we can disable the feature and return to baseline. No data-loss risk.

## The alternative asks we are not doing
- Option B: move 30% of chat traffic to Fireworks.
  - Expected savings: $0.34 - $0.90 = negative. Not viable on pure cost.
  - Still valid as a resilience play (module 109 chapter 06); separate memo.
- Option C: hardware upgrade to H200.
  - Expected savings: $0.34 → $0.26 (23%), requires fleet refresh, $400k capex + timeline. Dominated by Option A.

## The decision
Proceed with speculative decoding on chat.
Review at chapter-02 cadence:
  - weekly: acceptance rate, cost.
  - monthly: cost SLO, business-impact validation.

## Appendix
- Pareto sweep data: {link}
- Cost model version: v4 (2026-10-15)
- Acceptance-rate measurement: {link, module 108 ex 03 dashboard}
- Owners: {name} (inference), {name} (SRE), {name} (product)
```

The three disciplines:

- **Infrastructure, product, business — always all three.** Don't ship a memo that only has one.
- **Cite the sources.** Chapter 04 model version, chapter 05 sweep date, module 108 lifecycle. Reproducible, auditable.
- **Name the alternatives you didn't choose.** A memo that only presents one option is doing the reader's thinking for them; naming the other options shows you did the work.

## The three antipatterns

The discipline exists to resist three recurring prioritisation mistakes.

### Antipattern 1 — "optimise the tail to zero"

A team spends months driving TTFT_P99 from 2 s to 1 s. The users on the P99 tail are either (a) long-context users whose SLO is in seconds anyway or (b) users whose network is the bottleneck. Neither case benefits from the engineering work. The cost of the sprint is six engineer-weeks; the KPI impact is undetectable.

The right discipline: before optimising a tail metric, confirm *who* is in the tail and *whether moving them matters*. Chapter 03 dashboard 2 (per-class deep-dive) answers the first question; the business-metric tie-in from this chapter answers the second.

### Antipattern 2 — "ship cheapest provider"

A finance partner sees that Fireworks charges $0.90 / 1M output tokens and the self-hosted fleet charges "$5 / 1M output tokens" (naive math; actual is $0.34 after credits — chapter 04). The partner asks for migration; the team says yes; three months later the migration stalls on data-residency contracts, model-customisation gaps, or SLA mismatches.

The right discipline: present the comparison with all five dimensions — cost, latency, data residency, customisation, SLA. The cheapest provider is cheap because their contract shape is simpler than yours needs to be; the comparison table in chapter 04 is the artifact that makes the trade-off visible.

### Antipattern 3 — "chase the benchmark"

A competitor publishes a model-benchmark score that is 2 points higher on MMLU. The team is asked to migrate to the "better" model, or to retrain. Three months later, the user-visible quality is identical (the 2-point delta doesn't show up in the product's eval harness) and the serving cost is 30 % higher because the new model is bigger.

The right discipline: product-eval first, model-benchmark second. The business metric is on the product's eval set, not on the model-card. Chapter 04 of module 113 (eval gate) is the methodology; this chapter's memo cites it.

Each antipattern shares a shape: *the engineering team optimised the wrong variable because the translation from technical to business was missing.* The memo discipline from this chapter is how that doesn't happen.

## Scope boundaries — what you do, what the partner does

Not every number in the memo is yours to produce. The split:

| Numbers you produce | Numbers your partner produces | Who the partner is |
|---|---|---|
| TTFT / TPOT / ITL / E2E latency (chapter 01) | Dose-response of latency → product KPI | Product analyst |
| Cost per 1M tokens (chapter 04) | Revenue per 1M tokens, ARPU | Finance partner |
| Cache / speculative credit (chapter 04) | ARPU-weighted user segmentation | Product analyst |
| Pareto sweep (chapter 05) | A/B test design / causal attribution | Applied scientist, product analyst |
| Error-budget status (chapter 02) | SLO target negotiation | Product, SRE, you |
| Incident impact in token-seconds / dollars | Incident impact in retention / revenue | Finance, product |

If a number requires a partner input, name them and cite the source. A memo that fabricates a product-KPI delta to justify an infrastructure proposal will not survive its first round of partner review and will damage the team's credibility.

## The weekly ritual

A monthly or weekly cross-functional review where the inference team walks:

- SLO status by class (chapter 02 dashboard).
- Cost per class, delta from last period, attribution to changes (deploys, workload shifts, credit changes) (chapter 04 dashboard).
- Pending asks: what was proposed, what was chosen, what is in-flight (memos from this chapter).
- Pending incidents and their business impact (module 113 feeds this).

The audience is an inference engineer, an SRE, a product manager, a finance partner, and optionally an applied scientist. If a memo is pending, it gets presented in this room; the decision is captured in the memo's change log.

The ritual exists because the memo discipline needs a venue. Without a scheduled recurring conversation, memos are written to no one and prioritisation falls back on ad-hoc email threads.

## Common failure modes

- **Infrastructure memo with no product arrow.** Reads as self-justifying engineering work.
- **Product memo with no infrastructure number.** Reads as armchair wishlist.
- **Fabricated dose-response.** "A 100 ms TTFT improvement is worth $X" without a source is a hallucination; it survives one meeting, not two.
- **Ignoring opportunity cost.** "This saves $200k / month" is one side of the ledger; "and takes 4 engineer-weeks" is the other. Report both.
- **Running only one experiment.** A single A/B is noisy; the dose-response at one magnitude doesn't predict the dose-response at another. Multiple magnitudes if you can afford them.
- **Shipping memos to no audience.** The weekly ritual is where memos land; without it, memos are unread documents and prioritisation drifts.
- **Comparing providers without the full table.** Cost is one column; latency, residency, customisation, SLA are the other four. Chapter 04's comparison table is the right shape.
- **Not naming the alternatives.** A one-option memo looks weaker than it is; naming the alternatives (even the bad ones) is how the reader sees that you have considered them.

## Summary

- Every optimisation lands as a translation from infrastructure to product to business. All three are required.
- The TTFT → product arrow is measured, not assumed. Dashboard overlay, natural experiment, or A/B — pick the technique proportional to the stakes.
- Cost per 1M output tokens compared to revenue per 1M output tokens is a KPI. If inference-cost exceeds ~30 % of ARPU the business model is strained; if it is < 5 % optimisation returns are low.
- The one-page memo is the artifact: infrastructure delta, product delta, business delta, ask cost, alternatives, decision.
- The three antipatterns — tail-optimisation, cheapest-provider, benchmark-chasing — exist because translation to business metrics is skipped.
- Partners own the business arrows; you own the infrastructure arrows. Cite sources both directions.
- The weekly cross-functional ritual is how memos get read. Without it, prioritisation drifts.
- Chapters 01 – 05 produce the data; this chapter turns the data into decisions.
