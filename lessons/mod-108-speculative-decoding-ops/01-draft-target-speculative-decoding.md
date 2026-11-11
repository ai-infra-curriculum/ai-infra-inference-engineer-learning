# Chapter 01 — Draft-Target Speculative Decoding: Mechanics, Sizing, and Deployment

> Decode is memory-bandwidth bound. Reading the target's weights to produce one token is almost as expensive as reading them to verify `k` candidates. If a cheap drafter can propose a plausible `k` tokens, the target can accept-or-reject them in approximately one target step. The whole scheme pays or fails on one number: the acceptance rate.

## Motivation

A platform team runs Llama-3.1-70B at TP=4 behind continuous batching. TPOT is 50 ms at batch=1, 55 ms at batch=4, and ~70 ms at batch=16 — decode is memory-bandwidth bound, the batch barely moves the needle, and the team wants a lower TPOT without buying more HBM-bandwidth per GPU.

They enable speculative decoding with Llama-3.2-1B as the draft, speculative length `k=4`. On their internal chat workload, acceptance rate measures `α ≈ 0.72`; TPOT drops from 50 ms to ~22 ms at batch=1. On a code-completion workload, acceptance measures `α ≈ 0.85`; TPOT drops to ~17 ms. On a reasoning workload with high-entropy decoding, acceptance measures `α ≈ 0.41`; TPOT **rises** to ~58 ms because the draft overhead no longer pays. The team's deployment is three decisions, not one: *which drafter, which `k`, and which workload classes do we even enable speculative on?*

This chapter walks each decision. By the end of it you can:

- Draw the draft-and-verify loop on a whiteboard — the draft forward passes, the target single-pass verification over `k` candidates, the rejection sampling (or longest-prefix match for greedy / top-1), the KV commit-vs-rollback.
- Pick a drafter for a given target — tokenizer parity (required), distribution closeness (determines `α`), parameter ratio (determines draft-step cost), and HBM budget (determines whether the draft co-locates on the target's workers).
- Size the speculative length `k` from the measured acceptance rate.
- Deploy a draft-target pair on vLLM, SGLang, and (where applicable) TensorRT-LLM — the launch flags, the KV-cache impact, the sampling-parameter restrictions, and the health check that confirms speculative is actually firing (not silently falling back to per-token decode).
- State the TTFT-vs-throughput trade-off and the two load regimes where speculative pays vs. hurts (chapter 05 completes the "hurts" side).

The engine-specific flag names and command-line ergonomics below rotate between releases; the deployment section at the end of this chapter cites the current primary-source guides for vLLM, SGLang, and TensorRT-LLM.

## The draft-and-verify loop

A standard decode step runs:

```
loop:
    input_ids ← current_sequence
    logits   ← target.forward(input_ids)       # one HBM pass over weights
    next_tok ← sample(logits)
    sequence.append(next_tok)
```

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023 — see [`resources.md`](resources.md)) replaces the loop body with:

```
loop:
    # 1. Draft proposes k tokens autoregressively (fast — small model)
    drafts = []
    for i in 1..k:
        d_logits = draft.forward(input_ids + drafts)
        drafts.append(sample(d_logits))        # k forward passes over the draft

    # 2. Target verifies all k candidates in ONE forward pass
    #    (feed input_ids + drafts; get logits at each of the k+1 positions)
    t_logits = target.forward(input_ids + drafts)   # one HBM pass over target weights

    # 3. Accept-or-reject per position (speculative sampling for stochastic;
    #    longest-prefix match for greedy / top-1)
    accepted = speculative_sample(t_logits, drafts)
    sequence.extend(accepted)                   # 1 to k tokens committed

    # 4. If all k accepted, the target's extra logits at position k+1 give a
    #    bonus token "for free" (one extra token per step when the drafter is perfect)
    if len(accepted) == k:
        sequence.append(sample(t_logits[k+1]))
```

The two savings:

- **At most 1 target forward pass per iteration**, regardless of how many tokens are accepted. Decode cost is dominated by that one HBM pass over the target's weights.
- **A bonus token at `k+1`** when all drafts are accepted — the target's final-position logits are "free" because the forward pass that verifies positions `1..k` also computes position `k+1`.

The one cost:

- **`k` draft forward passes per iteration.** The draft is small, so each pass is cheap — but it is `k`-many passes and they are *autoregressive* (each one depends on the previous one's sample), so they do not parallelise within the one iteration.

### Rejection and sampling-equivalence

For greedy decoding (top-1), verification is trivial: longest prefix where `argmax(target_logits[i]) == drafts[i]` is accepted.

For stochastic sampling, speculative decoding uses **rejection sampling** to preserve the target's distribution exactly:

- Compare `p_target(token)` and `p_draft(token)` at each position.
- Accept with probability `min(1, p_target / p_draft)`.
- On the first rejection, re-sample from the "residual" distribution `max(0, p_target - p_draft)` normalized.
- All subsequent drafts are discarded; the loop proceeds from the resampled token.

This is Leviathan et al.'s result: *the output distribution is identical to vanilla sampling from the target.* Speculative decoding does not change what the model generates in expectation — it only changes how many target-forward-passes it costs.

Engines vary in exactly which sampling-parameter combinations are supported when speculative is enabled — vLLM's speculative-decoding documentation, for example, notes that pipeline parallelism, beam search, and some `logprobs`-related configurations are incompatible with speculative in current releases (see <https://docs.vllm.ai/en/latest/features/speculative_decoding/>). Verify the engine's documented sampler-compatibility matrix against the pinned release before deployment.

### Tree-based verification (variants to be aware of)

The standard scheme verifies a *linear* `k`-token chain. SpecInfer (Miao et al., 2023) and several engines verify a **tree** of candidates — the drafter proposes multiple branches, each of length ≤ `k`, and the target verifies all branches in one pass under a block-diagonal / tree-structured attention mask. Tree verification raises the effective acceptance rate at a fixed `k` (more candidates means more chances to find an accepted path) at the cost of a wider verify batch.

Engine support rotates:

- vLLM has shipped linear verification and (as of recent releases) tree-based variants under the speculative-decoding guide; verify the flag spelling.
- SGLang supports several speculative schemes including EAGLE's tree variant (chapter 02).
- TRT-LLM supports speculative via the Executor API and the engine-build-time flags.

Tree-based verification is a **performance-peer** concern at kernel depth (the tree-attention mask is a `.cuh` concern); the **operator** turns it on with a flag and measures acceptance. Chapter 06 writes the boundary.

## Picking a drafter

The drafter must be a strict mechanical fit to the target. The two hard requirements and two soft ones:

### Hard: tokenizer parity

The drafter and target **must share a tokenizer** (same vocab, same special tokens, same byte-pair / BPE / SentencePiece configuration). A tokenizer mismatch makes the drafts untranslatable into target token IDs — the engine either refuses to load the pair or produces garbage. In practice this means the drafter is from the same model family:

| Target family | Common drafter choices |
|---|---|
| Llama-3.x | smaller Llama-3.x variant (3.2-1B, 3.2-3B, 3.1-8B as draft for 70B / 405B) |
| Qwen-2.5 | smaller Qwen-2.5 variant |
| Mistral / Mixtral | smaller Mistral variant |
| DeepSeek | DeepSeek-R1-Distill-Qwen for Qwen targets; DeepSeek draft for DeepSeek targets |
| Gemma | smaller Gemma variant |

Cross-family drafters (e.g. a Llama drafter for a Qwen target) require tokenizer-remapping and are rarely production-viable. Verify the engine's documented support matrix for your pair.

### Hard: the engine must support the pair

Not every engine supports every draft-target configuration. vLLM, SGLang, and TRT-LLM each publish a speculative-decoding guide; check:

- Is the specific target architecture supported for speculative?
- Is the draft architecture allowed to differ from the target architecture, or must it be the same family?
- Is quantization supported on both sides (e.g. FP8 target + BF16 draft, or must they match)?
- Does the engine support the KV-cache sharing / separate layout you expect?

### Soft: distribution closeness (determines `α`)

The closer the drafter's output distribution is to the target's, the higher the acceptance rate. The two common strategies:

- **Smaller same-family instruct checkpoint.** The 1B / 3B / 8B variants of Llama-3.x are instruction-tuned on data the 70B / 405B saw. They are not identical distributions, but they are close. Expected acceptance on a well-matched workload is roughly `α ∈ [0.6, 0.85]`.
- **Distilled draft (trained specifically to match the target).** A draft trained by distillation from the target — same architecture family, trained with a KL-divergence or feature-matching loss against the target's outputs — reaches higher `α`. This is the fine-tuning peer's deliverable (chapter 04); the operator consumes it.

Specific reported acceptance rates for popular target-draft pairs rotate as checkpoints are released and refined; treat any `α` cited here as illustrative and measure on your workload.

### Soft: parameter ratio (determines `c_d / c_t`)

The draft's step cost `c_d` relative to the target's `c_t` sets the floor acceptance rate. For a draft that is `r` times smaller than the target in parameters, roughly:

```
c_d / c_t ≈ r × (bandwidth_utilization_ratio) × (compute_utilization_ratio)
```

For a 1B draft on a 70B target, `r ≈ 0.014` — on paper, the draft costs about 1.4 % of a target step. In practice, overhead (CUDA-graph re-trace at the smaller model shape, kernel launch overhead that doesn't amortise across the small forward pass, extra scheduler accounting) pushes this to 5–15 %. For a 7B draft on a 70B target, the ratio is 10 % on paper, maybe 15–25 % after overhead.

Larger draft → higher `α` but higher `c_d`; the sweet spot is workload-specific. Exercise 01 walks the sweep.

### HBM budget: does the draft co-locate on the target's workers?

Two deployment shapes:

- **Co-located draft.** The draft weights live in the same HBM as the target's shards (sharing the TP group or sharded separately). This is the common path. The draft's HBM footprint reduces the KV budget — a 1B BF16 draft is ~2 GiB per TP rank; subtract that from the chapter-104 KV pool math.
- **Separate draft worker.** Rare at production scale (adds a hop, adds complexity). Used when the draft is CPU-only (unusual for modern targets) or when the draft's TP topology differs enough to warrant separation. Not covered further in this chapter.

For co-located drafts, the fit check is: *does `(weight_bytes_target / TP) + (weight_bytes_draft / TP_draft) + kv_pool + activations + reserves` fit in HBM at the target concurrency?* If not, either reduce the KV pool (lowers concurrency), use a smaller draft (lowers `α`), quantize the draft (small loss in `α`, saves HBM — verify on your workload), or add replicas (more hardware).

### A worked sizing

Llama-3.1-70B BF16 target on 4× H100 80GB SXM (TP=4):

- Target weight bytes: `70e9 × 2 = 140 GiB` → `35 GiB per rank`.
- Draft candidate: Llama-3.2-1B BF16. Weight bytes: `1.2e9 × 2 = 2.4 GiB` → co-located on the TP=4 group, so `0.6 GiB per rank` (if sharded) or `2.4 GiB per rank` (if replicated across the group; small models often replicate).
- Remaining HBM per rank for KV + activations + reserves: `80 - 35 - 2.4 ≈ 42.6 GiB` (if draft replicated) vs. `80 - 35 - 0.6 ≈ 44.4 GiB` (if draft sharded).
- At chapter-104 Llama-3.1-70B KV math (~160 KiB per token across the model), 42 GiB of KV budget per rank × 4 ranks ≈ 168 GiB total ≈ ~1.1M tokens of KV. The draft's draft-KV adds a small further line item that the engine typically manages separately.

Fits comfortably. For the 405B target the arithmetic is tighter; the 1B / 3B draft is still cheap, but the KV pool is already under more pressure.

## Setting the speculative length `k`

The ceiling speed-up at acceptance rate `α` and draft-step cost ratio `c_d/c_t`, over speculative length `k`:

```
speedup(α, k, c_d/c_t) ≈ (1 - α^(k+1)) / ((1 - α)(1 + k · c_d / c_t))
```

Properties:

- At `α = 0`: ratio is `1 / (1 + k · c_d / c_t)` — strictly worse than vanilla. Speculative never helps when nothing is accepted.
- At `α = 1`: ratio is `(k + 1) / (1 + k · c_d / c_t)` — approaches `k + 1` as `c_d → 0`. The bonus token is the `+1`.
- The derivative wrt `k` is 0 at some finite `k*(α)`; above `k*`, the overhead dominates and the speedup drops.

For a sanity sketch (illustrative — verify on your workload):

| `α` | `c_d / c_t = 0.1` | `c_d / c_t = 0.05` |
|---|---|---|
| 0.5 | optimal `k ≈ 2–3`, speedup ~1.6× | optimal `k ≈ 3`, speedup ~1.8× |
| 0.7 | optimal `k ≈ 3–4`, speedup ~2.2× | optimal `k ≈ 4–5`, speedup ~2.6× |
| 0.85 | optimal `k ≈ 4–6`, speedup ~3.2× | optimal `k ≈ 6–8`, speedup ~4.0× |

Operator rule: **pick `k` from `α`**, not from a default. Run the acceptance measurement (chapter 03 instruments it) over a representative traffic sample, then set `k` to the measured optimum. Re-measure on drift.

Many engines expose a dynamic / adaptive `k` that adjusts per-sequence based on recent acceptance history. If available, prefer it over a fixed `k` — the per-sequence acceptance variance is often large.

vLLM ships a "Dynamic Speculative Decoding" feature that can vary `num_speculative_tokens` by batch size (see <https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/>) and SGLang documents an "Adaptive Speculative Decoding" flow; the specific flag names and algorithms differ between engines and rotate between releases, so verify against the pinned-release docs.

## The throughput-vs-TTFT trade-off

Speculative decoding is **a decode-phase optimisation**. The two things to watch at the SLO boundary:

### TTFT

Prefill is unchanged by speculative decoding — the first-token latency is still "prefill the prompt through the target." Speculative only kicks in once decode starts. If your TTFT is dominated by prefill (long prompts), speculative doesn't change TTFT.

Where it *can* raise TTFT: the first decode step after prefill may incur CUDA-graph re-trace overhead if the speculative scheme uses a different graph shape than non-speculative. Mitigate with warm-up at startup (chapter 02 of mod-102); engines document the warm-up protocol.

### TPOT (and ITL)

Speculative decoding directly lowers TPOT when acceptance is high. The practical ceiling is roughly:

- Linear verification: `TPOT_spec ≈ TPOT_baseline / E[accepted + bonus]`, where `E[accepted + bonus]` is the mean tokens committed per iteration.
- At `α = 0.7, k = 4`: `E[accepted + bonus] ≈ 2.7`, so `TPOT_spec ≈ 0.37 × TPOT_baseline`.

In practice the realised ratio is worse because of draft-step overhead not fully in the ceiling formula and batching dynamics (speculative changes the batch's "effective" token count per iteration, which the scheduler sees as higher throughput and may admit more work into).

### Throughput

Per-replica throughput rises if acceptance pays. Fleet-level throughput rises less than per-replica because the autoscaler may now admit more work per replica — the net effect on QPS depends on how the fleet is sized.

### The scheduler interaction (chapter 05 completes)

At high batch sizes, decode becomes compute-bound rather than memory-bandwidth-bound (chapter 103). The "one HBM pass of the target" argument that makes speculative pay in the first place weakens — the target's single-pass is no longer the "almost free" cost; it is itself constrained by compute. In that regime, the draft's `c_d` is pure overhead with less to amortise against.

Chapter 05 names this regime and the second regime (acceptance too low). This chapter sets up the two numbers to watch; chapter 05 benchmarks the thresholds.

## Deployment walkthrough

Engine-specific; verify flag names against the release.

### vLLM

The vLLM speculative-decoding guide (<https://docs.vllm.ai/>) documents the current recommended flags. The pattern (verify against the pinned release):

```bash
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --speculative-model meta-llama/Llama-3.2-1B-Instruct \
  --num-speculative-tokens 4 \
  --use-v2-block-manager \
  --disable-log-requests
```

Engine-side things to confirm:

- The engine's startup log reports speculative-decoding is enabled and names the draft model.
- The engine exposes speculative metrics (chapter 03 enumerates). If the metrics endpoint doesn't show `*spec*` series, speculative didn't actually turn on.
- Sampler compatibility — some flag combinations (specific `logprobs`, specific `top_p` + `temperature` combinations) may fall back to non-speculative silently. The engine's documentation should state the restrictions.

The vLLM speculative-decoding API has evolved across releases — recent versions consolidate the settings behind a single `--speculative-config` JSON object (e.g. `--speculative-config '{"method": "draft_model", "model": "...", "num_speculative_tokens": 4}'`) rather than the separate `--speculative-model` / `--num-speculative-tokens` flags shown above. Always verify against the pinned-release docs at <https://docs.vllm.ai/en/latest/features/speculative_decoding/>.

### SGLang

SGLang supports speculative decoding including EAGLE-based schemes (chapter 02). The pattern:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp 4 \
  --speculative-algorithm EAGLE \
  --speculative-draft meta-llama/Llama-3.2-1B-Instruct \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 10
```

Verify the current flag set against the SGLang docs. The EAGLE variant (and its tree-decoding) is a stronger integration than a raw draft-target; chapter 02 covers.

### TensorRT-LLM

TRT-LLM supports speculative via the engine-build pipeline. The target and draft are converted and built together (or separately, depending on the version). Verify the current pattern in the TRT-LLM examples directory and the Triton `tensorrtllm_backend` configuration.

TRT-LLM selects the speculative mode at engine-build time — the `trtllm-build` command accepts `--speculative_decoding_mode` with values including `draft_tokens_external`, `lookahead_decoding`, `medusa`, `explicit_draft_tokens`, and `eagle` (see <https://nvidia.github.io/TensorRT-LLM/latest/features/speculative-decoding.html>). The exact `trtllm-build` incantation and the Triton `instance_group` / `kv_cache_type` settings should be lifted from the TRT-LLM `examples/` directory for the release the fleet runs.

### Health checks that confirm speculative is firing

Before admitting a draft-target deployment into production, verify:

1. **Startup log** names the draft model and the speculative-decoding backend.
2. **Metrics endpoint** exposes a non-zero `spec_tokens_proposed` after a few test requests (chapter 03 lists the full set).
3. **A/B latency sanity.** With one request at batch=1, median TPOT should drop vs. the non-speculative baseline on a workload where you expect `α > 0.5`. If it doesn't drop, something didn't engage.
4. **No fallback.** If the engine exposes a `spec_bypassed` or `spec_disabled_for_request` counter, verify it is near-zero for the workload class you tested. Some engines bypass speculative on `logprobs=True`, structured output, or specific sampler combinations; the health check surfaces this.
5. **Output equivalence.** Rejection sampling (for stochastic) and longest-prefix-match (for greedy) are supposed to preserve the target's output distribution. A regression suite comparing speculative vs. non-speculative outputs on a frozen prompt set (greedy decoding) should produce identical tokens; stochastic decoding should produce distributionally equivalent samples. If the outputs diverge in the greedy case, something is wrong — route to the performance peer (chapter 06).

## Common deployment errors

### Error 1 — tokenizer mismatch

The team picks "a small model" as the draft without checking tokenizer parity. The engine either refuses to load or produces garbage. Verify the tokenizer-equality constraint from the engine docs before picking the draft.

### Error 2 — unverified acceptance

The team ships speculative decoding without measuring `α` on real traffic. The deployment "works" (no errors) but net latency is unchanged or worse because the workload's acceptance is below the break-even point. Chapter 03 instruments; chapter 05 benchmarks. Measure before enabling.

### Error 3 — fixed `k` across workload classes

The team sets `k = 4` globally. The chat workload's acceptance is `0.72`; the code workload's is `0.85`; the reasoning workload's is `0.4`. The chat is near-optimal, the code is under-shot, the reasoning is net-negative. Enable speculative *per workload class* with per-class `k` (chapter 03 defines the workload class; chapter 05 benchmarks the per-class threshold).

### Error 4 — draft that doesn't fit the HBM

The team adds a 7B draft to a 70B TP=4 deployment that was already running near the HBM ceiling. The KV pool shrinks; concurrency falls; queue latency rises. The speculative speedup is real but it's eaten by the queue. Size the draft against the HBM budget *including* the KV pool and the activations.

### Error 5 — speculative with incompatible sampler

The team enables speculative on an endpoint that uses `logprobs=True` or structured output. The engine silently bypasses speculative on these requests. The `α` metric looks fine; the end-to-end latency doesn't move. Verify sampler compatibility and the engine's bypass counters.

### Error 6 — leaking draft version into the target promotion

The team upgrades the target to `target_v2` but leaves `draft_v1` in the config. The draft's acceptance drops because the target's distribution shifted. The lifecycle contract (chapter 04) prevents this: *you promote `(target, draft)` as a pair, or you don't promote.*

## The boundary to the peers

This chapter does not teach:

- **The fused draft-and-verify kernel** — tree-attention masks, bonus-token-handling kernels, TP-aware draft forward pass with reduced collective footprint. Performance peer; chapter 06 writes the hand-off.
- **How to distil a drafter** — the loss, the data mix, the preference-tuning for the drafter. Fine-tuning peer; chapter 04 writes the retrain request.
- **How to tune NCCL for the extra collective traffic introduced by a TP-sharded draft** — the operator sets the env; the peer tunes the algorithm.

Those live at [`ai-infra-performance`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) and [`fine-tuning-engineer`](https://github.com/ai-infra-curriculum/fine-tuning-engineer-learning). This chapter is the operator's view: *which drafter, which `k`, what the engine flag is, which health check confirms it fired.*

## Summary

- Speculative decoding runs a cheap drafter for `k` tokens then verifies all `k` with one target forward pass. If the drafter is right, you get `≥1` and up to `k+1` tokens per iteration; if it's wrong, you get `1` and paid `k` draft steps as overhead.
- The scheme preserves the target's output distribution exactly under rejection sampling (stochastic) or longest-prefix match (greedy). It changes the step cost, not the output.
- Pick a drafter with the **same tokenizer** as the target; the engine must support the pair; the parameter ratio sets the draft-step cost `c_d / c_t`; the distribution closeness sets the acceptance rate `α`.
- Set `k` from the measured `α`. Prefer adaptive-`k` when the engine supports it; otherwise sweep (exercise 01) and freeze per workload class.
- Deploy via the engine's documented speculative-decoding flags. Verify fired via the startup log, the metrics endpoint, the A/B latency check, and the sampler-compatibility bypass counters.
- The six deployment errors — tokenizer mismatch, unverified acceptance, fixed `k` across workloads, over-fat draft that eats the KV pool, incompatible sampler bypassing silently, draft-target version drift — are all operator-level and all preventable with chapter 03 and chapter 04.
- Kernel-level questions (fused verify, tree attention, bonus-token kernel) route to the performance peer. Draft-training questions route to the fine-tuning peer. Chapter 06 writes the hand-off.
