# Chapter 02 — The Inference-Platform RFC: Build-vs-Buy, TCO Model, Risk Register, Migration Path

> "Should we self-host vLLM on our own H100s, or route to Fireworks?" is not a technical question with a technical answer. It is a program question with a *written, defensible* answer — an RFC. This chapter builds the shape of that RFC.

## Motivation

A platform director walks into the room with a page of the cloud bill and asks: "we spend $800k / month on H100s. Fireworks will sell me the same model for `$X / 1M output tokens`. Why are we not doing that?" The engineer who answers "because we care about performance" has already lost the conversation. The engineer who hands the director a signed RFC — with a TCO model that accounts for utilisation, prefix-cache credit, speculative credit, egress, people cost, exit cost, and risk; a build-vs-buy decision framed against workload classes; a risk register with likelihood and impact; and a migration path between the options — has reframed the conversation into a decision the director can own.

That RFC is the artefact. By the end of this chapter you can:

- Frame a build-vs-buy decision for an inference workload class as an RFC a sponsor can sign.
- Build a defensible total-cost-of-ownership model that integrates module-110's cost-per-1M-tokens number with the full-program overhead.
- Author the risk register for each option with likelihoods, impacts, and mitigations.
- Draw the migration path between options that preserves optionality.
- Pick the right decomposition — workload class, tenant, region, failover tier — so the decision is "hybrid at axis X," not "all-or-nothing."

Chapter 03 then takes the vendor side of the RFC into a formal scorecard; this chapter is the framing.

## The three options, not two

"Build vs buy" is a convenient slogan; the real decision has three options:

| Option | What it means | Typical fit |
|---|---|---|
| **Self-host** | The team runs the engine (vLLM / SGLang / TensorRT-LLM / NVIDIA Dynamo / Triton) on owned or reserved GPUs; the team is on the pager for everything. | High traffic, stable workloads, custom models, data-residency requirements, workloads where unit economics beat the vendor. |
| **Managed inference** | A vendor (Fireworks, Together, Baseten, Modal, NVIDIA NIM, Groq, Cerebras, hyperscaler inference offerings) runs the engine; the team consumes an API. | New workloads, spiky or unpredictable traffic, models the vendor runs better, long-tail models where fleet amortisation doesn't close, teams without 24/7 on-call. |
| **Hybrid** | Some workloads self-hosted, others routed to vendors — often with a cost-aware router (LiteLLM-shape; module 106) that enforces the split. | The common steady-state for a mature platform; the question becomes "where do we draw the line" and the RFC answers it. |

A build-vs-buy RFC that reduces to a binary is almost always wrong. The real decision is *which workload class sits where, and under what conditions does it move*.

## The decomposition that makes the decision tractable

The all-or-nothing RFC fails because an inference platform does not have *one* workload. It has several, and they have different unit economics, different latency budgets, different data-residency constraints, and different reliability pressures. Split the decision along axes that matter:

- **By workload class.** Chat, agent tool-use, long-context document chat, code completion, embedding, re-ranking, image-captioning, speech. Each has a different context-length distribution, output-length distribution, cache-hit profile, and SLO. The unit economics of one class may call for self-host; another, for a vendor.
- **By tenant tier.** Enterprise tenants may require data-residency, dedicated capacity, or a custom contract. Free-tier may tolerate a vendor's shared capacity and a looser SLO. The RFC may reach the split "enterprise self-host, free-tier vendor" without forcing the whole platform into either.
- **By model class.** Flagship models (the top 2–3 by traffic) justify dedicated capacity because amortisation closes; long-tail models (dozens of small or rarely-used models) rarely do. A vendor may run the long-tail cheaper precisely because they amortise across many customers.
- **By region.** US-primary workloads may be cheapest self-hosted in an owned data centre; EU workloads may be cheapest at a vendor that already has an EU region; APAC may route to a hyperscaler to avoid new fleet investment.
- **By reliability tier.** Primary path self-hosted; vendor as the failover for availability (chapter 03 develops the vendor-as-failover scorecard dimension). Or the inverse: vendor primary, self-host as the "cost-escape" path when the vendor bill becomes load-bearing.

A good RFC states its decomposition up front, then evaluates *each cell* of the decomposition on cost, latency, data, reliability, and strategic axes. The output is a table, not a verdict.

## The RFC structure

An inference-platform build-vs-buy RFC fits in 8–15 pages. The structure (author your own template; organisation to taste):

1. **Problem statement.** The forcing function. Cost pressure, SLO pressure, capacity pressure, product requirement, strategic direction. One paragraph.
2. **Current state.** What the platform looks like today — engines, hardware, routing, SLOs, cost per workload class. Reference module 110's cost-per-1M-tokens chapter for the arithmetic.
3. **Options under consideration.** Self-host, managed inference (named vendors; chapter 03 scorecard references for each), hybrid variants. For each: one-paragraph description plus a "what wins" summary.
4. **Decomposition.** The workload / tenant / model / region / reliability-tier matrix. States which cells are in scope for the decision.
5. **TCO model.** The arithmetic. The spreadsheet or `cost-model.yaml` lives in the appendix; the RFC body summarises per-cell.
6. **Risk register.** For each option, named risks with likelihood, impact, owner, mitigation.
7. **Recommendation.** The specific placement for each cell of the decomposition, with "when does this change" triggers. Not a one-liner.
8. **Migration path.** How to move from the current state to the recommended state (or how to stay put with hardening). Chapter 01's migration-plan structure applies here.
9. **Open questions.** What the RFC does *not* resolve; who answers each; by when.
10. **Sign-off.** Platform lead, sponsor (director / VP), peer-track leads whose work the decision touches, finance partner.

The RFC is a proposal, not a presentation. Written prose. Numbered sections. The appendix carries the TCO spreadsheet, the vendor scorecards, and any peer-track interface diffs. No slide decks.

## The TCO model — what it actually contains

Module 110 chapter 04 computed the inference-fleet cost per 1M input / output tokens. That number is an input to TCO, not the output. Total cost of ownership is *everything that scales with running the thing in production over a stated horizon* — hardware, software, people, risk, exit — and is the defensible number to compare against the vendor's quoted price.

### The self-host TCO

A workable self-host TCO over a 12-month horizon:

```
hardware_amortised              (from module 110 chapter 04: GPU_amortised_cost_per_hour × GPU_count × hours_in_horizon)
+ underutilised_capacity        (capacity reserved for headroom / burst / spot-interruption absorption that produces no tokens)
+ warm_pool                     (module 109: scale-to-near-zero reserves)
+ model_weights_storage         (per-region replication, snapshotting)
+ observability                 (Prometheus / Grafana / traces / logs)
+ egress                        (inter-region, to consumer apps)
+ software_licences             (enterprise engine licences, Triton Enterprise, DCGM Enterprise, support contracts — if any)
+ people_cost                   (platform team + fraction of SRE peer + fraction of performance peer)
+ on_call_overhead              (pager cost, incident cost, post-mortem cost — hard to quantify, name anyway)
+ opportunity_cost              (what the team isn't building while it runs the fleet)
+ exit_cost                     (cost to wind this down if the decision reverses — usually small for self-host but non-zero)
=  TCO_self_host
```

Discipline notes:

- **Underutilised capacity is real.** A fleet at 60 % mean utilisation (healthy for inference) is 40 % over-provisioned. That 40 % is in the bill; put it in the TCO.
- **Warm-pool reserves are recurring.** Module 109's scale-to-near-zero pattern keeps replicas warm; those replicas cost money and produce zero tokens. Account for them.
- **People cost is where self-host wins or loses.** A two-engineer platform team that spends 40 % of its time on toil is a very different TCO than a six-engineer team at 10 %. The cost models for the two teams diverge by a factor that matters more than the GPU bill.
- **Exit cost is rarely zero.** Decommissioning a self-hosted fleet has a tail — GPU resale or redeploy, data decommissioning, consumer-track migration — that an RFC should name even if it hopes never to execute it.

### The managed-inference TCO

A workable managed-inference TCO over the same horizon:

```
usage_billed                    (vendor's $ per 1M input + $ per 1M output × forecast traffic)
+ overage_buffer                (vendor pricing often has tier breakpoints; forecast error → budget error)
+ minimum_commits               (some vendor contracts have monthly or annual minimums)
+ data_transit                  (egress from vendor back to app layer; cross-region if applicable)
+ observability_integration     (your own metrics on top of the vendor's; dashboards; cross-correlation plumbing)
+ security_and_compliance       (DLP in front of the vendor, data-boundary audits, contract-level work)
+ integration_people_cost       (a vendor-sourced endpoint still needs the routing layer, the fallback plumbing, the usage-tracking)
+ incident_ownership_premium    (you cannot debug the vendor's stack; you pay for their SLA tier and the time to escalate)
+ lock_in_discount_erosion      (promotional pricing at year 1 that may rise at renewal)
+ exit_cost                     (cost to migrate off if the decision reverses — the central cost here; chapter 03 scores it)
=  TCO_managed
```

Discipline notes:

- **The quoted `$ / 1M tokens` is a floor, not the TCO.** The vendor's price excludes everything from "egress" down.
- **Overage buffer is a real line item.** Traffic forecasts are wrong; vendor pricing tiers mean the first forecast error that crosses a tier costs disproportionately. Name a planned buffer.
- **Exit cost matters a lot.** A vendor that owns the endpoint also owns your consumer tracks' integration. The exit cost is "how much work to point every consumer at the new endpoint and prove parity." Chapter 03 scores this as `exit_cost` on the scorecard.
- **Lock-in is a risk, not a cost line.** Price it in the risk register, not the TCO. But pricing it zero is a mistake.

### The hybrid TCO

Hybrid TCO is the sum of the per-cell TCOs plus the cost of running the hybrid machinery itself:

```
Σ (TCO_self_host[cell]) for cells in self-host
+ Σ (TCO_managed[cell])  for cells in managed
+ router_cost                   (the cost-aware router from module 106 — compute, operations, latency overhead)
+ cross_option_observability    (metrics cross-correlated between self-host and vendor paths)
+ failover_rehearsal_cost       (the discipline to prove the failover works before an incident forces it)
=  TCO_hybrid
```

The point of computing hybrid TCO: the router / observability / failover-rehearsal costs are *real* and sometimes they close the gap that made hybrid look attractive on raw per-cell arithmetic. An RFC that says "hybrid is best" without pricing the hybrid machinery itself is understating the cost.

## The risk register

Every option has risks. The RFC's job is to name them, score them, and show the mitigations — not to hide them under the cost arithmetic.

A workable scoring axis: each risk has `likelihood` (low / medium / high over the horizon), `impact` (what it costs if it happens — latency, cost, availability, reputation, compliance), `owner` (who watches it), and `mitigation` (the specific thing that reduces it). A one-line example row:

> *"Risk: vendor renews with a 40 % price increase. Likelihood: medium. Impact: $300k / year or forced migration back to self-host. Owner: platform lead. Mitigation: 12-month renewal notice clause; keep a 1-week-to-restart self-host warm path (see migration path)."*

### Representative risk classes

For self-host:

- **Capacity.** Fleet can't scale to peak; SLO burns during a spike. Mitigation: warm pool (module 109); cost-aware router with vendor-overflow (hybrid).
- **People.** Key engineer departs; on-call quality drops. Mitigation: runbook discipline (module 113 chapter 03); shadow rotations (chapter 01); documented architecture.
- **Hardware obsolescence.** Current GPUs become uncompetitive; cost per 1M tokens drifts upward. Mitigation: hardware refresh roadmap on quarterly cadence (chapter 01).
- **Software supply chain.** Engine upstream makes a breaking change; the pinned version can't absorb a security fix. Mitigation: engine version-tracking (chapter 05 interface with eval peers), explicit upgrade-window planning.
- **Driver / firmware.** A GPU driver update introduces an XID regression; fleet degrades. Mitigation: canary a driver update on a single node pool; back-out plan (module 113 chapter 03's XID runbook).

For managed inference:

- **SLA.** Vendor outage; consumer-track availability tied to the vendor's. Mitigation: multi-vendor router; self-host fallback; contractual SLA + credit terms.
- **Price.** Vendor raises price; TCO blows out. Mitigation: multi-vendor contract; self-host fallback; contract clause on price-change notice.
- **Feature parity drift.** Vendor lags on a feature the consumer tracks need (reasoning models, structured output, tool-calling). Mitigation: feature-parity scorecard (chapter 03); contract clause on feature-release cadence.
- **Data boundary.** Vendor logs or trains on customer data by default; or data crosses a region the compliance team doesn't allow. Mitigation: data-boundary clause in contract; DLP layer; chapter 03's data-boundary scorecard dimension.
- **Exit cost.** Vendor relationship ends; consumer tracks must migrate. Mitigation: abstraction behind the router (module 106); per-vendor model-name map; rehearsed exit.
- **Lock-in.** Vendor-proprietary features become load-bearing (vendor's fine-tunes, vendor-specific tool-call semantics). Mitigation: discipline to keep the integration thin; feature-parity watch.

For hybrid:

- **Split discipline.** The split drifts over time — a workload meant to be at the vendor migrates to self-host silently, or vice versa. Mitigation: router metrics; monthly review of per-cell traffic.
- **Observability gap.** Metrics on self-host and vendor paths are not comparable; a cross-path incident is un-triageable. Mitigation: unified SLO definition (chapter 05 interface with SRE peer); cross-path dashboards.
- **Failover drift.** The failover path rots because it is rarely exercised. Mitigation: scheduled failover rehearsal (quarterly), chapter 01's quarterly cadence.

## The migration path

The RFC does not just pick an option; it specifies *how to get there from the current state* in a way that preserves optionality.

Three qualitative patterns:

- **Current-state is self-host; recommendation is partial managed.** The migration carves a specific workload class out, runs a shadow / canary / hybrid-weight routing (module 113 chapter 01), and ramps traffic to the vendor only after the chapter-03 scorecard thresholds are met. Rollback is a routing weight for the duration of the ramp; permanent rollback becomes a capacity re-expansion on the self-host side.
- **Current-state is vendor; recommendation is partial self-host.** The migration provisions capacity (hardware, engine, model artefacts), warms the pool, runs the chapter-03 scorecard in reverse ("is self-host at least as good on these dimensions"), and ramps traffic off the vendor. Rollback is reversing the routing weight; the self-host capacity is a cost sunk against the discovery.
- **Current-state and recommendation are both hybrid, but at different split points.** The migration moves a cell at a time, under the router's weight control, with per-cell SLO and cost evidence before promoting each.

In all three patterns:

- The **rollback branch** is explicit — at what stage is rollback cheap, at what stage does it become a second migration. Preserve the cheap rollback for as long as the evidence is still accumulating.
- The **consumer-track communication** mirrors the chapter-06 deprecation runbook of module 113 even if the change is not a deprecation — the consumer tracks still see a routing or versioning change on an axis they consume.
- The **sign-off gates** are on the evidence, not on the calendar. "We promised to migrate by end of Q2" is a schedule; "we migrate to the next cell when the scorecard shows green on dimensions A–G for two weeks" is a gate.

## The "when does this change" triggers

An RFC that reads as a permanent verdict is already stale. Name the conditions under which the decision reverses. Representative triggers:

- **Price delta compresses.** If the vendor's price drops 30 % or our self-host cost rises 30 %, revisit the decomposition (likely re-evaluate cells near the boundary).
- **SLO changes.** If the product commits to a tighter TTFT than the vendor can serve, self-host wins more cells. If a new workload class emerges with a long-context profile the vendor serves better, vendor wins more cells.
- **Feature parity shifts.** A vendor ships a feature the consumer tracks want that self-host lags on (or vice versa).
- **Hardware refresh.** A new SKU materially shifts self-host unit economics (Blackwell lands; FP4 serving at throughput self-host could not previously reach).
- **Incident evidence.** A vendor's SLA breach that costs the business more than its mitigation; a self-host incident that proves the on-call rotation is undersized.
- **Strategic direction.** A corporate decision (data sovereignty, cost discipline, partnership) that reframes the acceptable options.

Each trigger has an owner (platform lead for cost and SLO, eval / peer leads for feature parity, sponsor for strategic). When a trigger fires, the RFC is re-opened — not the decision executed by reflex.

## Common failure modes in build-vs-buy RFCs

- **The unit-cost shootout.** The RFC compares `$ / 1M tokens` on one axis and picks the lower number. The vendor wins because egress, observability, and exit cost weren't counted; the next quarter reveals the TCO gap.
- **The strategic abstract.** The RFC argues "we must own our infra" or "we must leverage vendors" without numbers. Sponsor signs, work starts, no gate fires anywhere. The outcome is not the chosen option; it is heroics in whichever direction the team drifts.
- **The all-or-nothing binary.** The RFC asks "self-host or vendor" without a decomposition. One answer loses everywhere and wins nowhere; the hybrid that would have been obvious is not evaluated.
- **The exit-cost zero.** The RFC does not price the exit. A year later, the vendor decision becomes a trap because migrating the consumer tracks off the vendor is a quarter of effort nobody budgeted.
- **The wishful SLA.** The vendor's marketing SLA (e.g. "99.9 % uptime") is quoted verbatim without the credit schedule and without the real-world historical uptime the vendor's status page shows. Chapter 03 scores this properly.

## A worked structure — abbreviated

A sketch of the recommendation section of a plausible RFC, to show what "defensible" looks like concretely (this is a structural example — the numbers are not claims about any specific vendor or workload; fill in your own):

> **Recommendation.** Move the long-tail `transcription` and `small-embedding` workload classes to a managed inference provider (chapter 03 scorecard picks the top-scoring vendor). Keep the flagship chat, agent, and long-context chat workloads self-hosted on current H100 capacity. Retain enterprise-tier chat on dedicated self-host capacity in the primary region to satisfy data-residency. Add a vendor-overflow failover for free-tier chat via the cost-aware router (module 106), capped at 10 % of peak traffic.
>
> **TCO delta.** [Reference appendix: self-host cells sum to X, managed cells sum to Y, hybrid overhead Z, net savings vs. current state ≈ N %. Horizon 12 months.]
>
> **Decision reverses if:** vendor price on the moved workloads rises >25 % at renewal; enterprise contract requires moved workloads to be self-hosted; a self-host hardware refresh lowers self-host unit cost by >30 % on the moved workloads.

That section is one paragraph in the RFC body; the appendix carries the arithmetic. A sponsor can sign it; a successor engineer can re-open it when a trigger fires.

## Summary

- Build-vs-buy is never binary. The three options are self-host, managed, and hybrid; the right question is *which cells of the decomposition sit where*.
- Decompose by workload class, tenant, model, region, and reliability tier. Evaluate each cell; produce a table, not a verdict.
- TCO is more than the quoted price. Self-host: hardware + underutilised + warm pool + storage + observability + egress + licences + people + on-call + opportunity + exit. Managed: usage + overage buffer + commits + transit + observability-integration + security + integration-people + incident-ownership + discount-erosion + exit. Hybrid: per-cell sum plus router + cross-observability + failover-rehearsal.
- Risk register names each risk per option with likelihood, impact, owner, mitigation. Representative classes: capacity, people, hardware obsolescence, SLA, price, feature parity, data boundary, exit cost, lock-in, split discipline.
- Migration path specifies how to get from current state to recommended state with preserved rollback, consumer-track communication, and evidence-based gates. It is the chapter-01 migration-plan shape, specialised.
- Name the triggers that reverse the decision: price delta, SLO shift, feature parity, hardware refresh, incident evidence, strategic direction. An RFC without triggers is already stale.
- Chapter 03 formalises the vendor-eval scorecard this chapter's RFC references for every managed-inference cell.
