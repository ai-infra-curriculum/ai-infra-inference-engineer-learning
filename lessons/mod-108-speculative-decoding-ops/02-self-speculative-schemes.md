# Chapter 02 — Self-Speculative Schemes: Medusa, EAGLE, Lookahead Decoding

> Chapter 01's draft-target scheme requires a second checkpoint, a second tokenizer-matched model, and a second forward pass per iteration. The three schemes in this chapter remove one or more of those costs by making the target drafters *for itself* — with multi-head prediction (Medusa), feature-level drafting (EAGLE / EAGLE-2 / EAGLE-3), or an n-gram cache (Lookahead).

## Motivation

A team deploying speculative decoding on Llama-3.1-70B hits two operational walls with the draft-target scheme from chapter 01:

1. **The 1B draft's acceptance rate is `α ≈ 0.65`** on their traffic. Chapter 01's `k`-sweep lands at `k = 3`, giving a ~1.9× speedup. The team wants more — the published numbers for the 70B class suggest 2.5–3× is achievable on similar workloads.
2. **The draft is a second lifecycle to manage.** The target gets upgraded; the draft needs re-distilled; the versioning contract (chapter 04) adds process cost. Smaller teams want one artefact to ship.

Both walls motivate the schemes in this chapter. Medusa adds auxiliary prediction heads to the target itself — no second checkpoint, same tokenizer by construction, higher acceptance in many settings. EAGLE drafts at the *feature* level (second-to-last hidden state) with a tiny auxiliary network — significantly higher acceptance than draft-target in the published numbers, with a modest training overhead. Lookahead decoding needs no training at all — it uses recent n-gram statistics from the sequence itself.

By the end of this chapter you can:

- State what each scheme is at operator depth — the "where does the proposal come from" diagram.
- State what each scheme *costs you to deploy* — a checkpoint with Medusa heads, a trained EAGLE drafter, nothing, or a different engine entirely.
- Deploy the one your engine supports and your workload warrants.
- Reason about when each one beats draft-target and when it does not.

<!-- needs-research: engine support for self-speculative schemes rotates. Medusa and EAGLE adoption spans vLLM, SGLang, TensorRT-LLM, and NVIDIA's TensorRT-LLM Model Optimizer; the exact matrix of (scheme × engine × target architecture) is release-specific. Verify the current support matrix against the pinned release of the engine your fleet runs. -->

## Medusa — multi-head prediction

**What it is.** Medusa (Cai et al., 2024 — see [`resources.md`](resources.md)) attaches *k* additional prediction heads to the target model. Each head `h_i` is trained to predict the token at position `t + i` from the hidden state at position `t`. At inference time, one target forward pass produces predictions for the next token (the normal LM head) and for the next `k` tokens (the Medusa heads), all in parallel. Those `k + 1` candidates are then verified via tree-based attention — the target's next forward pass checks all tree branches at once, and the longest accepted prefix wins.

**What it needs.**

- A target checkpoint with Medusa heads trained. The heads are small (one or two linear layers each), trained on top of the frozen target or jointly with a short fine-tune. Training is the fine-tuning peer's deliverable; chapter 04 writes the hand-off.
- Engine support for Medusa-style multi-head decoding and tree-attention verification. vLLM has shipped Medusa support; TRT-LLM has shipped Medusa; SGLang's primary focus is EAGLE but check the matrix.

**What it costs.**

- The Medusa heads add a small number of parameters to the target (per head: one or two linear layers of `d_model → vocab`; `k = 4` heads adds <1 % to a 70B target's parameter count). HBM-side, trivial.
- Draft "cost" `c_d / c_t` is **near zero** — the heads run in parallel with the LM head on the same forward pass. There is no autoregressive draft loop. This is Medusa's key operational advantage over draft-target.

**Where it shines.**

- When your target can be re-shipped with heads. If you own the target's lifecycle (your org trains it, or it's an open checkpoint you can distil heads onto), Medusa adds low-overhead speculative without a second checkpoint.
- When draft-target's `c_d / c_t` is uncomfortably high (small-model-on-tiny-draft overhead doesn't amortise well on your batch shape). Medusa's near-zero draft cost widens the operational envelope.

**Where it falters.**

- Requires training. If you cannot train or distil a Medusa-equipped target, this scheme is unavailable to you.
- Head quality is the ceiling on `α` — if the heads are weak, acceptance plateaus below what a well-matched draft-target pair would achieve.
- The engine must support Medusa-shape decoding. Older or less-feature-rich engines may not.

**Deployment pattern.**

```bash
# vLLM (verify flags against pinned release)
vllm serve some-org/Llama-70B-Medusa-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 4
```

The specific flag spelling evolves; verify against the vLLM speculative-decoding guide. The engine's startup log should name "Medusa" or the equivalent. The metrics in chapter 03 apply identically — Medusa-shaped acceptance rate is still `α = accepted_tokens / proposed_tokens`.

<!-- needs-research: Medusa + Llama-70B checkpoints are available from several labs; the authoritative current catalogue is on Hugging Face. Verify a specific checkpoint, its training recipe, and the engine's documented support before adopting. -->

## EAGLE — feature-level drafting

**What it is.** EAGLE (Li et al., 2024) runs the drafter at the **feature level** rather than the token level. The draft network takes the target's second-to-last hidden state (not the softmax output) as input, plus the previous token, and predicts the next feature. A tiny auto-regressive network — one or two transformer blocks — runs over features, and only the last step projects to a token prediction. EAGLE-2 adds dynamic tree expansion based on the drafter's own uncertainty. EAGLE-3 extends further (see the EAGLE-3 paper for the specific changes).

The key insight: features are smoother and more predictable than tokens, so a tiny feature-level drafter can achieve acceptance rates that a token-level drafter of the same parameter budget cannot. Reported acceptance rates and speedups are higher than draft-target on comparable settings in the EAGLE paper and its extensions.

**What it needs.**

- A trained EAGLE drafter paired with the target. The EAGLE repo publishes drafters for popular open targets (Llama, Qwen, Vicuna); training a custom EAGLE drafter is the fine-tuning peer's deliverable.
- Engine support for feature-level drafting + tree verification. SGLang has EAGLE as a first-class speculative-algorithm. vLLM has shipped EAGLE support (verify version). TensorRT-LLM supports EAGLE via the Executor API and Model Optimizer.

**What it costs.**

- A separate (small) checkpoint — the EAGLE drafter — paired with the target. HBM-side it's smaller than a 1B draft-target drafter (EAGLE's auto-regressive module is a few layers; typically a few tens of MiB to a few hundred MiB).
- The target's forward pass must expose its second-to-last hidden state to the drafter. Engines that support EAGLE do this; others don't.
- Training cost (fine-tuning peer's problem).

**Where it shines.**

- When published acceptance rates for your target family exist and are higher than draft-target by enough margin to matter. The EAGLE repository tracks the numbers; verify on your workload.
- When the engine supports tree-decoding well — EAGLE-2 and EAGLE-3's dynamic tree variants are a significant contributor to the gains.
- When the drafter's HBM footprint is a concern (EAGLE drafters are typically smaller than token-level drafters at comparable `α`).

**Where it falters.**

- Not every target architecture has a published EAGLE drafter. If yours doesn't, training one is a fine-tuning-peer task that may be significant work.
- Engine coupling is tighter than draft-target — the engine must plumb the feature-level handoff. A new engine version may regress EAGLE support before token-level draft-target support.

**Deployment pattern.**

```bash
# SGLang (verify flags against pinned release)
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp 4 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path some-org/EAGLE-Llama-3.1-70B \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 10 \
  --speculative-num-draft-tokens 64
```

The parameters above are illustrative — EAGLE's tree parameters (`topk`, total draft tokens, tree depth) all affect the measured acceptance and the per-iteration cost. SGLang's docs document the tuning surface; verify against the pinned release.

<!-- needs-research: EAGLE's tree-parameter defaults and the SGLang / vLLM flag names rotate with releases. Verify the current recommended configuration from the EAGLE repository and the engine's speculative-decoding guide. -->

## Lookahead decoding — n-gram cache, no training

**What it is.** Lookahead decoding (Fu et al., 2024) proposes next-tokens from an n-gram cache built from the sequence itself. The idea: in generation, the model often repeats or quasi-repeats recent spans (code with similar patterns, structured output, formatting). A short-horizon cache of recent n-grams gives free candidates. The target verifies a batch of candidates per iteration, as with Medusa / EAGLE, under a specific attention mask.

**What it needs.**

- Engine support for Lookahead's specific decoding path and the attention mask it requires.
- No training. No second checkpoint. No distillation. This is Lookahead's defining operational feature.

**What it costs.**

- Zero additional HBM for drafter weights.
- A small amount of per-sequence state (the n-gram cache). Scales with context, bounded.
- Engine-side implementation complexity (the attention mask is specific).

**Where it shines.**

- Highly repetitive workloads — code generation (same identifiers reused), structured output (JSON key repetition), templated prose. Reported acceptance rates in the Lookahead paper and reference implementation are substantial on code tasks.
- Workloads where you cannot or will not train a drafter and cannot adopt a Medusa / EAGLE checkpoint.

**Where it falters.**

- Open-ended generation (chat, free-form creative, reasoning). The n-gram cache's hit rate on genuinely novel content is low; acceptance plateaus below what a model-based drafter can achieve.
- Engine support is thinner than Medusa / EAGLE. Verify.

**Deployment pattern.**

Lookahead's reference implementation ships with Hugging Face transformers and has partial integrations into other engines; the exact deployment pattern depends on your engine. Verify against the engine's speculative-decoding guide and the Lookahead reference repo.

<!-- needs-research: Lookahead's integration into vLLM / SGLang / TRT-LLM is less mature than Medusa / EAGLE as of this writing; check each engine's docs for current support. The reference implementation is primarily published alongside the paper's repo. -->

## Choosing between the schemes

A decision sketch, keyed to the "what do I own and what does my engine support":

```
Do you own the target's training pipeline (can you ship a Medusa-equipped checkpoint)?
├── Yes → Medusa is operationally cheapest if your engine supports it.
│         Draft cost c_d ≈ 0; one artefact to ship.
└── No  → Continue.

Is there a published EAGLE drafter for your target, and does your engine support it?
├── Yes → EAGLE often beats draft-target on α for comparable deployment effort.
└── No  → Continue.

Is your workload highly repetitive (code, structured output), and does your engine
support Lookahead?
├── Yes → Lookahead avoids training entirely; try it.
└── No  → Fall back to draft-target (chapter 01); it is the most broadly supported.
```

The decision is not permanent. All four schemes (chapter 01's draft-target plus the three here) share the same acceptance-rate instrumentation (chapter 03) and the same "when does it hurt" analysis (chapter 05). A/B them under the same monitoring; keep the one that produces the best measured `TPOT` and `cost / 1M tokens` on your workload.

## Operational differences from draft-target

The three schemes in this chapter share some but not all of chapter 01's deployment ergonomics:

| Concern | Draft-target (ch 01) | Medusa | EAGLE | Lookahead |
|---|---|---|---|---|
| Separate checkpoint required | yes (draft) | no (heads bundled with target) | yes (EAGLE drafter) | no |
| Tokenizer match concern | yes | n/a (same model) | yes (EAGLE drafter trained paired) | n/a (n-gram cache operates in target vocab) |
| Draft-step cost `c_d / c_t` | ~0.05–0.20 | ~0 (parallel heads) | small (tiny feature net) | ~0 (lookup) |
| Training cost | high (distilling a draft) | medium (training heads) | medium (training the EAGLE drafter) | zero |
| HBM footprint (per rank) | ~1–7B weights + KV | heads only | tiny | zero |
| Lifecycle complexity (ch 04) | two artefacts; must version together | one artefact | two artefacts (target + EAGLE drafter) | one artefact |
| Engine support breadth | broad | broad on major engines | SGLang / vLLM / TRT-LLM; verify | narrower |
| Acceptance rate on open workloads | moderate (`α ≈ 0.6–0.8`) | typically higher than draft-target | typically highest of the four | high on repetitive, low on open-ended |

<!-- needs-research: the specific "typically higher" / "typically highest" phrasing is a summary of the EAGLE and Medusa papers' reported numbers on common benchmarks; your workload may differ. Treat the ordering as a starting hypothesis, not a conclusion — measure. -->

## Monitoring is identical

Regardless of scheme, the metric set from chapter 03 applies: tokens proposed, tokens accepted, acceptance rate, draft overhead, bypass counters. The schemes differ in *where the proposals come from* and *how they are trained*; the thing you measure is the same number.

That symmetry simplifies lifecycle: your draft-target monitoring stack carries forward to Medusa and EAGLE without redesign. Chapter 03's alerts are scheme-agnostic.

## The boundary to the peers

This chapter does not teach:

- **How to train Medusa heads.** Fine-tuning peer; chapter 04 writes the hand-off.
- **How to train an EAGLE drafter.** Fine-tuning peer; the EAGLE repo documents the recipe but execution is the peer's task.
- **How to implement tree attention in the engine's kernel.** Performance peer.
- **How the engine fuses the Medusa heads with the target's LM head.** Performance peer.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) and [`fine-tuning-engineer`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning). This chapter is the operator's view: *given a checkpoint and an engine that supports it, how do I turn each scheme on, how do I verify it fired, and when do I prefer one over another.*

## Summary

- Three alternatives to chapter 01's draft-target: **Medusa** (multi-head heads on the target itself, near-zero draft cost, requires a Medusa-trained target), **EAGLE** (feature-level drafter paired with the target, typically highest acceptance in published numbers, requires a trained EAGLE drafter), **Lookahead** (n-gram cache, no training, best on repetitive workloads).
- The decision tree is "what do I own + what does my engine support": Medusa if you can re-ship the target; EAGLE if a published drafter exists; Lookahead for repetitive workloads; draft-target as the broadly-supported fallback.
- All four schemes share the same monitoring (chapter 03), lifecycle discipline (chapter 04 — modified: Medusa / Lookahead have one artefact, draft-target / EAGLE have two), and "when it hurts" analysis (chapter 05).
- Engine support for self-speculative schemes rotates faster than for draft-target; always verify the current support matrix against the pinned release.
- Kernel work on tree attention, fused heads, and the specific EAGLE drafter forward pass routes to the performance peer. Training work on Medusa heads and EAGLE drafters routes to the fine-tuning peer.
