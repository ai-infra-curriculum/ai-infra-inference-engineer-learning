# Chapter 05 — TGI Deprecation and the Maintained Alternatives

> You will meet TGI in the wild for years to come. This chapter gives you the compare-point depth you need to translate a TGI-shaped world into a vLLM / SGLang / TRT-LLM one.

## Motivation

Hugging Face **Text Generation Inference (TGI)** was, for several years, one of the four major open-source LLM serving engines alongside vLLM, SGLang, and TensorRT-LLM. A lot of production fleets, a lot of blog posts, a lot of runbooks, a lot of your peer-track colleagues' muscle memory were built against TGI.

Hugging Face has since signalled that TGI's maintenance posture has changed and that recommended serving for new deployments points elsewhere. The `huggingface/text-generation-inference` GitHub repository was archived on 2026-03-21 and placed in maintenance mode; its README now points new deployments to vLLM, SGLang, and (for local / on-device use) llama.cpp or MLX. See <https://github.com/huggingface/text-generation-inference> for the authoritative statement.

This chapter does three things:

1. Characterise the deprecation state at a compare-point level — enough that you can read a TGI-era runbook, understand what the author meant, and translate it.
2. Place the maintained alternatives Hugging Face points to today (HF Inference Endpoints) and the community-maintained serving UIs (Text-Generation-WebUI) on the capability map from module 101.
3. Give you a translation table from a TGI-shaped operator vocabulary to the vLLM / SGLang / TRT-LLM world so you don't get burned when porting a legacy deployment.

This chapter is at **compare-point depth**, not **operational depth**. You should not be launching new TGI fleets. You should be able to read a TGI deployment, understand what it was doing, and migrate it.

## The TGI deprecation signal — what to track

Hugging Face has moved its primary serving story toward:

- **Hugging Face Inference Endpoints** — their managed, cloud-hosted serving product. This is where "the HF way to serve a model" now points for most teams.
- **vLLM** and **TRT-LLM** as the recommended self-hosted serving engines when Inference Endpoints doesn't fit.
- **TGI** itself in a maintenance-only / archived posture — the GitHub repository was archived on 2026-03-21 and its README states the project will accept only "pull requests for minor bug fixes, documentation improvements and lightweight maintenance tasks" going forward.

What this means operationally for an existing fleet:

- **Do not accept new upstream behaviour.** Pin to a known-good TGI version. Treat TGI releases after the deprecation signal as high-risk even if they are published.
- **Audit the fleet for migration.** For every TGI deployment, name the destination engine. For most production workloads that is vLLM; for NVIDIA-only shops with hot models it may be TRT-LLM; for prefix-heavy workloads it may be SGLang.
- **Keep the TGI runbooks** — but mark them as **translation layer**, not the current state of the art. A new on-call engineer should learn vLLM / SGLang / TRT-LLM first and TGI second.

## Where TGI sits on the capability map

Placing TGI against the ten axes from module 101 chapter 02:

| Axis | TGI's historical position |
|---|---|
| Scheduling model | Continuous batching |
| KV-cache primitive | Paged KV cache (TGI-specific implementation) |
| Attention / kernels | Flash Attention; TGI's own fused kernels |
| Quantization matrix | BF16/FP16; INT8, INT4, AWQ, GPTQ, EETQ depending on version; FP8 on later releases |
| Parallelism | TP; PP via `shards`; limited EP support |
| Disaggregated P/D | Not a first-class feature |
| Multi-adapter | LoRA via adapters; multi-LoRA with varying maturity across versions |
| Speculative decoding | Draft-target, Medusa — version-dependent |
| API surface | HF Messages API, OpenAI-compatible endpoint, native TGI HTTP + gRPC |
| Hardware backends | NVIDIA primary; AMD / Gaudi / Neuron paths at varying maturity |

The *shape* of TGI matches vLLM's much more than TRT-LLM's — Python-defined serving with runtime continuous batching, no compile step, OpenAI-compatible endpoints. The key operational differences in the TGI era:

- A different observability vocabulary (TGI metric names are not vLLM metric names).
- A different flag surface — `MAX_INPUT_LENGTH`, `MAX_TOTAL_TOKENS`, `MAX_BATCH_PREFILL_TOKENS`, `NUM_SHARD`, `WAITING_SERVED_RATIO` and friends (verify current names against the version you have in production).
- A different scheduler shape — TGI's own continuous-batching implementation, not vLLM's.

## The maintained alternatives

### Hugging Face Inference Endpoints — the managed path

A managed, cloud-hosted serving product from Hugging Face. Teams point it at a model on the HF Hub and get an endpoint. The underlying engine is HF-managed and has shifted over time (TGI historically; now a mix in which vLLM is the default engine for new endpoints with SGLang and the archived-but-still-selectable TGI available as alternates — verify the current engine list against the Inference Endpoints docs for your account).

- **Capability map placement** — the full axes above don't fully apply because the engine is managed; you consume *endpoints*, not *engine knobs*. Available knobs are high-level: instance type, autoscaling bounds, model and revision.
- **Build-vs-buy fit** — Inference Endpoints wins when: the team wants fast time-to-endpoint, the HF Hub model catalogue is the source of truth, and operational depth on serving engines is not a differentiator. Loses when: cost-per-token at scale matters, custom engine knobs matter, data-boundary / enclave requirements rule out managed services.
- **Exit cost** — moderate. Workloads are *conceptually* portable (open-weight models, OpenAI-compatible API) but migrating involves re-tuning for the self-hosted engine, re-running the eval gate, and operating a new engine.

Operator stance: a credible build-vs-buy target. Module 114's RFC is where this gets a formal decision for your fleet.

### Text-Generation-WebUI — the community serving / UI compare-point

Text-Generation-WebUI (oobabooga's project) is a community-maintained web UI *and* multi-backend serving tool for running open-weight LLMs. It is **not** a datacentre-scale production inference engine; it is a desktop / small-fleet tool that unifies backends like llama.cpp, Transformers, ExLlamaV2 and others behind one UI.

- **Capability map placement** — serves at the LLM-interaction-UI layer plus the local-serving layer. Not a replacement for vLLM / SGLang / TRT-LLM at scale.
- **Build-vs-buy fit** — right choice for internal "model-exploration" or "prompt-engineering" terminals, hobby deployments, and single-GPU self-hosted LLM users. Wrong choice for an inference platform serving an application fleet.
- **Exit cost** — low; backends are swappable, model artifacts are standard HF or GGUF formats.

Operator stance: compare-point only. Know it exists so when a consumer-track engineer says "we use ooba locally for prompt dev" you understand what they mean.

### Note on other community serving stacks

Several other open-source projects — various forks of TGI, serving wrappers over Hugging Face Transformers, local tools like LM Studio, Ollama — occupy adjacent niches. Mostly they are:

- **Edge / local inference** (llama.cpp wrappers, Ollama) — the module 101 chapter 02 placement for llama.cpp applies.
- **Framework-glue projects** (various Transformers-based `generate()` servers) — not production inference engines; they'll show up in demos.
- **Community TGI forks** — several vendor- or community-maintained forks exist (for example DeepInfra's, Preemo's, and H2O's `open-text-generation-inference`); if one becomes the de facto successor for a given fleet, treat it as its own compare-point rather than as a continuation of upstream TGI's support story.

The main compare-points you must know are Inference Endpoints (managed path) and Text-Generation-WebUI (community UI). Everything else slots into "edge / local" or "framework glue" and is covered where appropriate.

## Translation table — TGI concept → vLLM / SGLang / TRT-LLM equivalent

A short dictionary for when you port a TGI runbook or interview an engineer whose mental model is TGI-shaped:

| TGI concept | vLLM equivalent | SGLang equivalent | TRT-LLM equivalent |
|---|---|---|---|
| `MAX_INPUT_LENGTH` | `--max-model-len` (combined with output) | `--context-length` | `--max_input_len` compiled in |
| `MAX_TOTAL_TOKENS` | `--max-model-len` | `--max-total-tokens` or equivalent (verify current) | `--max_seq_len` compiled in |
| `MAX_BATCH_PREFILL_TOKENS` | `--max-num-batched-tokens` + chunked prefill | `--max-prefill-tokens` / chunked-prefill size | `--max_num_tokens` + chunked context |
| `NUM_SHARD` | `--tensor-parallel-size` | `--tensor-parallel-size` | `--tp_size` at build time |
| `WAITING_SERVED_RATIO` | implicit in continuous-batching scheduler policy | implicit in `--schedule-policy` | implicit in in-flight batching policy |
| TGI paged KV cache | PagedAttention block table | RadixAttention tree | TRT-LLM paged KV cache |
| HF Messages API | `/v1/chat/completions` | `/v1/chat/completions` | Triton ensemble + OpenAI-compatible layer |
| TGI `/metrics` names | `vllm:*` metric namespace | SGLang `/metrics` names | Triton + TRT-LLM backend metrics |
| TGI model-loading env (`MODEL_ID`, `REVISION`) | `--model`, `--revision` | `--model-path`, `--revision` | checkpoint convert step reads these |

The semantics are mostly preserved; the spelling isn't. When migrating a runbook, do it column by column and verify each row against the current engine's docs.

## Migration playbook from a TGI deployment to vLLM (worked outline)

Not a full runbook (that is a module-113 artifact), but the shape of the migration:

1. **Characterise the current TGI deployment.** Capture: model id, revision, quantization, launch env (`NUM_SHARD`, `MAX_INPUT_LENGTH`, `MAX_TOTAL_TOKENS`, `MAX_BATCH_PREFILL_TOKENS`, speculative-decoding settings, LoRA adapters). Capture traffic shape (prompt-length distribution, output-length distribution, concurrency, prefix locality). Capture current SLOs (TTFT P99, TPOT, availability).
2. **Pick the target engine.** Default to vLLM unless the workload argues for SGLang (prefix-heavy, DP-attention MoE) or TRT-LLM (max NVIDIA throughput on a stable high-volume model). Confirm the target engine supports the model, the numeric format, and the required features.
3. **Translate the launch surface.** Use the table above to map TGI env → target engine flags. Validate the translation on a single-GPU staging box first.
4. **Reproduce the traffic probe.** Replay the characterised traffic shape against the new engine. Measure TTFT, TPOT, cache-hit, preemption, cost per 1M tokens. Compare to TGI baseline.
5. **Run the eval gate.** Confirm accuracy parity on your canary dataset. If quantization changed in the migration, this is non-optional (mod-105).
6. **Canary / shadow / blue-green.** Roll out via your module-113 release discipline, with an SLO-based auto-rollback.
7. **Decommission TGI.** Grace period for consumers, migration guide, decommission date. Keep the TGI containers around for a rollback window; delete when the rollback window closes.

The migration is boring when done well. Pattern-match it to your canonical model-swap runbook; it is not fundamentally different.

## When to stay on TGI anyway

A small number of situations argue for not migrating off TGI immediately:

- **The fleet is end-of-life soon.** A product retiring this quarter doesn't need a migration.
- **A specific TGI feature is a drop-in blocker** and the vLLM / SGLang / TRT-LLM equivalent is not yet shipped at parity for your model. Rare, but real; verify before committing to migration.
- **Vendor commitment.** If the organisation has a contractual TGI-shaped relationship with Hugging Face (Inference Endpoints or similar), the migration path may be *to* that managed product rather than off TGI the open-source engine.

In every other case: migrate. A deprecated engine accumulates risk every quarter it stays in production.

## Boundary notes

- **The migration is yours.** TGI → vLLM is an inference-platform decision, not an ML-platform decision. The ML-platform peer provides the model artifact; the inference team runs the migration.
- **The decision to retire TGI as a *recommendation* at Hugging Face is a vendor action**, not a technical judgement you own. Your job is to track it, communicate it to peers and consumers, and plan the migration.
- **New community forks of TGI**, if any emerge, are a tech-radar question (module 114). Do not treat an unknown fork as a continuation of TGI's support story without evidence.

## Summary

- TGI is in a deprecated / maintenance-only posture. Confirm the exact classification against Hugging Face's current statement; do not launch new TGI fleets.
- Hugging Face Inference Endpoints is the managed-path alternative; Text-Generation-WebUI is the community compare-point at the UI / small-fleet layer. Neither replaces vLLM / SGLang / TRT-LLM at datacentre serving scale.
- The TGI → vLLM / SGLang / TRT-LLM translation is mostly direct — the shape of the serving loop is preserved, but the flag surface and metric vocabulary are not. Translate by table, verify against docs.
- Existing TGI fleets should migrate on a planned cadence (not a panic cadence). Stay on TGI only when there is a specific, time-boxed reason.
- Compare-point depth is enough. You are not operating TGI anymore; you are *reading it* when you meet it.

This closes module 102. The exercises take each chapter into a hands-on walkthrough that produces artefacts you keep for the rest of the track.
