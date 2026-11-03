# Chapter 03 — Model-Format Conversion and Round-Trip Validation

> A model artefact in production has crossed at least two format boundaries since it left training. Every boundary is a place where bits moved and outputs might have changed. The job is to know which boundaries matter and how to prove the crossing was clean.

## Motivation

The Hugging Face `safetensors` checkpoint your team downloaded is not what runs in production. It gets quantised (chapter 02), converted into whatever packaging your engine wants (`safetensors` with a config, GGUF, a TensorRT-LLM compiled engine, or ONNX), copied to the registry, loaded by the worker, and served. At each step, bits move, dtypes cast, kernels differ, and the output distribution can drift in ways that pass a `diff` on the config but fail a `diff` on the logits.

This chapter is the operator discipline for the format layer: the four formats you will encounter, the conversion recipes between them, and the validation discipline that proves the conversion did not change outputs beyond a stated tolerance.

<!-- needs-research: conversion tool chains rotate (TensorRT-LLM converter paths, HF Optimum ONNX exporter, llama.cpp convert.py vs. convert_hf_to_gguf.py). Verify the specific CLI invocation against the tool version. -->

## The four formats that matter in production

### `safetensors` — the Hugging Face native format

The current standard for distributing model weights. Replaces `pytorch_model.bin` (which was pickle-based and had a code-execution security issue; see [`resources.md`](resources.md)).

**Layout:**
- A single file (or sharded across multiple files) with a JSON header followed by raw tensor data.
- The JSON header lists each tensor's name, dtype, shape, and byte offset.
- `safetensors` is **memory-mappable and zero-copy** — the engine can `mmap` the file and point the GPU DMA engine at the right offset without a `malloc + memcpy` round trip. This is why it loads fast.
- No code execution path (unlike pickle). Safer by design.

**Operator-visible files in a HF-shape repo:**

```
<model-dir>/
├── config.json           ← model architecture (hidden_size, num_layers, …)
├── tokenizer.json        ← tokenizer (if included)
├── tokenizer_config.json
├── generation_config.json ← default generation parameters
├── model-00001-of-00008.safetensors
├── model-00002-of-00008.safetensors
├── …
└── model.safetensors.index.json ← maps tensor names → shard files
```

For a quantised variant you also typically see `quantize_config.json` (AWQ / GPTQ metadata — bits, group size, zero-point policy) or `hf_quant_config.json` (newer unified naming — verify per tooling version).

**Who consumes safetensors directly:** vLLM, SGLang, Hugging Face Transformers, Text Generation Inference. Essentially every PyTorch-native inference engine.

### GGUF — the llama.cpp family format

A single-file packaging format designed for the llama.cpp runtime (which also serves many derivatives — Ollama, LM Studio, text-generation-webui, llama-cpp-python).

**Layout:**
- A single `.gguf` file containing weights, metadata (model name, architecture, hyperparameters), tokenizer, and chat template.
- Supports a wide family of quantisation types (`Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`, `BF16`, etc.) — note that these are **different** quantisation schemes from AWQ / GPTQ, with their own calibration and layout.
- Designed for memory-mapping and streaming load on resource-constrained hardware.

**When operators meet GGUF:**
- CPU-side inference (laptop demos, edge, offline eval harnesses).
- Fallback serving on hardware without GPU support.
- Compatibility with the llama.cpp ecosystem (Ollama is a common consumer).
- Some engines are growing GGUF loader support (vLLM has partial support in recent releases — verify).

**What converts to / from GGUF:**
- `convert_hf_to_gguf.py` (in the llama.cpp repo) converts a HF `safetensors` checkpoint to GGUF `F16` or `BF16`.
- `llama-quantize` (compiled binary) re-quantises a GGUF-F16 to any of the Q-formats.
- There is no first-class "GGUF → safetensors" tool; the round-trip is lossy for Q-format inputs because the dequantised tensors have been rounded.

### TensorRT-LLM compiled engines

A **compiled binary engine** produced by the TRT-LLM build system. Takes a quantised weight set + a kernel-selection manifest and emits a fully-fused, GPU-architecture-specific engine (a `.engine` file).

**Pipeline:**

```
HF safetensors ──convert──> TRT-LLM checkpoint ──quantize──> TRT-LLM quantised checkpoint ──build──> .engine file
                              (JSON + weights)                (JSON + quant weights)              (binary, SM-arch-specific)
```

Each step is a separate CLI invocation; TRT-LLM's docs organise them as `convert_checkpoint.py` → `quantize.py` → `trtllm-build`.

**Operator-visible characteristics:**
- The `.engine` file is **GPU-architecture-specific** (SM 80 vs. SM 90 vs. SM 100 — Ampere vs. Hopper vs. Blackwell). A single engine artefact is not portable between GPU generations.
- The engine encodes build-time choices (max batch size, max input length, max output length, parallelism topology, kernel choice). Changing any of these requires a rebuild.
- Loaded by TRT-LLM Python or by the Triton `tensorrtllm_backend`.
- Build time is substantial (tens of minutes for a 70B-class model); plan the CI job for it, not a per-request build.

**When operators meet TRT-LLM engines:**
- Production serving on NVIDIA hardware when squeezing the last 10–20% of throughput out of the kernel stack matters.
- When Transformer-Engine-fused FP8 or FP4 kernels are the specific win.
- When Triton Inference Server is the deployment target.

### ONNX

A cross-framework model-exchange format. In the LLM serving world, ONNX is **less common** than the first three — most production LLM serving on NVIDIA uses safetensors + vLLM/SGLang or TRT-LLM engines; most on AMD / Intel uses ONNX Runtime + ONNX.

**Layout:**
- A `.onnx` file (protobuf-encoded graph + weight tensors) plus optional external-data files for weights above a size threshold.
- The graph is a static computational graph; dynamic shapes are supported but with constraints.
- Weights can be stored in any supported dtype (BF16, FP16, FP8, INT8, INT4).

**When operators meet ONNX:**
- ONNX Runtime deployment (CPU, GPU, DirectML, CoreML, mobile).
- Cross-framework export (training in PyTorch, inference in ORT or Triton with ONNX backend).
- Edge / mobile serving (ONNX Runtime Mobile, ONNX Web).
- Some vendors (Microsoft, AMD, Intel) publish ONNX-first optimised inference stacks.

**Conversion:**
- HF Optimum's ONNX exporter is the common path from `safetensors` to ONNX for LLMs.
- The exporter handles the KV cache structure, decoder-only vs. encoder-decoder patterns, and the `past_key_values` plumbing.
- For LLMs above ~7B, ONNX export can be fragile; verify end-to-end before committing a workflow to it.

<!-- needs-research: ONNX LLM serving maturity varies strongly by model size and architecture. HF Optimum documentation is the authoritative reference for supported models — verify. -->

## The conversion matrix

| From \ To | safetensors | GGUF | TRT-LLM engine | ONNX |
|---|---|---|---|---|
| **safetensors** | — (shard / re-shard) | `convert_hf_to_gguf.py` (llama.cpp) | `convert_checkpoint.py` → `quantize.py` → `trtllm-build` | HF Optimum ONNX export |
| **GGUF** | not standard; de-quantise via llama.cpp then re-export | — (re-quantise with `llama-quantize`) | not supported | not supported |
| **TRT-LLM engine** | not supported (compiled binary, lossy) | not supported | — (rebuild with new flags) | not supported |
| **ONNX** | HF Optimum reverse (fragile; verify) | not supported | ORT → TRT engine via NVIDIA TRT (older path) | — (optimise with `onnxoptimizer`) |

**Operator rule:** `safetensors` is the hub. Everything flows out of `safetensors`; most things do not flow back in without loss. Keep the `safetensors` artefact as the lineage root (chapter 05 bakes this into the release contract).

## The round-trip validation discipline

A conversion is not clean because the file produced loads without errors. The engine *reading* the file might silently cast, pad, re-tile, or quantise in ways that change the output. The discipline below gates every conversion.

### Level 0 — the format loads

The weakest test: the target engine starts and serves one request without crashing.

```
# vLLM against the new artefact
vllm serve <artefact-path> --max-model-len 2048
# then: curl a trivial prompt and check the response is non-empty
```

Passing level 0 is table stakes. It does not prove anything about correctness.

### Level 1 — logit-level diff on a fixed prompt set

The strongest test: fix a seed, fix a prompt set, deterministic sampling (greedy or seeded top-k), capture the top-k logits per generation step for the first N tokens, diff against a reference artefact's logits.

Measure:

- **Max absolute logit difference** — the largest magnitude disagreement, per prompt. Not usually the right summary (one outlier dominates).
- **Mean absolute logit difference** — per prompt, averaged across tokens.
- **Top-k agreement** — for `k ∈ {1, 5, 10}`, the fraction of generation steps where the top-k token set matches the reference. For a BF16 → FP8 conversion, top-1 agreement at > 99.5% on well-behaved prompts is a reasonable floor; exact thresholds depend on model and workload.
- **KL divergence** — the KL divergence between the reference and the candidate's output distribution per generation step. More principled than top-k but more work to compute.

The *reference* is the pre-conversion artefact run on the same engine with the same seed. If the engine changed too (e.g. HF Transformers → vLLM), that is a two-dimensional comparison and both axes have to be controlled.

### Level 2 — sampled-output equivalence

Run the artefact and the reference on the same prompt set with a fixed seed and a sampling strategy that matches production (temperature, top-p, top-k). Compare:

- **Exact-match rate** — tokens or completions that match character-for-character. Only useful for very short outputs or near-deterministic sampling.
- **Edit-distance / BLEU / ROUGE** — standard text-similarity metrics. A 1–2% delta is typical for a well-behaved conversion; 10%+ is a red flag.
- **Semantic equivalence** — "does a judge model (or a human reviewer) rate the two outputs equivalent?" Noisier; useful on a smaller sample for high-value prompts.

Sampled-output equivalence has to budget for sampling noise; see chapter 04's bootstrap-CI discipline.

### Level 3 — perplexity on a held-out eval set

Compute perplexity on a held-out corpus (wikitext, c4, or a workload-matched set):

```
PPL(model, corpus) = exp( mean over tokens of -log P(token | context) )
```

Lower is better. A clean conversion has a perplexity delta of < 1–3% on standard corpora; larger deltas warrant investigation. Perplexity is the single most useful **model-level** sanity check for a quantisation change and is a core input to the chapter-04 accuracy gate.

### Level 4 — task-specific eval

The strongest test but the slowest and the most expensive: run a canary task suite (chat accuracy on a chat benchmark, code accuracy on a code benchmark, math accuracy on a math benchmark). This is where the chapter-04 accuracy gate's `task_specific_metric` lives; chapter 04 goes deeper.

## Pathology catalogue

### "The config.json looks right but outputs are nonsense"

The most common conversion failure. Common causes:

- **Dtype mismatch.** The checkpoint claims BF16 but a quantiser cast it to FP16 silently, and some tensors overflowed. Verify `metadata` on the safetensors header.
- **Weight permutation / transposition.** Some engines expect `[in_features, out_features]`; others expect the transpose. If the kernel reads the wrong shape, it still runs but produces garbage.
- **Rope / positional-embedding mismatch.** The target engine's rope implementation differs from the source (base frequency, scaling type — linear vs. dynamic vs. yarn). Common when porting long-context models.
- **Tokeniser mismatch.** The source and target engines use different tokenizers (common when converting between HF and llama.cpp — llama.cpp has its own tokeniser implementation). Garbage-in-garbage-out.

**Debug ladder:** compare config.json; compare the first weight tensor's statistics (mean, std, max abs); run the two engines with the same input tokens and compare the embedding-layer output; compare layer-0 attention output; isolate the first divergence.

### "Perplexity doubled on wikitext"

The quantisation introduced too much error. Common causes:

- **Calibration data mismatch** (chapter 02).
- **Wrong group size / granularity** (chapter 02).
- **Zero-point / symmetric mismatch** between the quantiser and the engine kernel. The engine assumes symmetric; the quantiser emitted asymmetric. The engine reads the scales but ignores the zero-points. Outputs degrade.
- **Mis-identified first / last layer.** First and last layer often need to stay at higher precision; some tooling defaults don't preserve them.

### "Works on the test rig, fails under TRT-LLM build"

The TRT-LLM converter rejected or silently converted something.

- **Dtype combinations not supported by the targeted SM arch.** E.g. FP8 on SM 80 (Ampere). The converter either errors or silently casts to BF16.
- **Max shapes mismatch.** The build-time `max_input_len`, `max_output_len`, `max_batch_size` have to accommodate the serving target; if they're too small, the engine builds but fails on larger requests.
- **Parallelism mismatch.** The weight sharding assumes TP=`N`, the build assumes TP=`M`. The engine builds (if `M` divides `N` symmetrically or vice-versa) but the per-rank weight layout is wrong.

Rebuild with verbose logs (`--log_level=verbose` for `trtllm-build`); the converter prints what it did to each tensor.

### "GGUF-quantised model produces drift the HF-quantised model does not"

GGUF's `Q4_K_M` etc. are **different quantisation algorithms** from AWQ / GPTQ. The llama.cpp Q-formats use k-quants, i-quants, and other variants with their own calibration (or no calibration). A GGUF-Q4_K_M of a model is not the same bit-for-bit as an AWQ-INT4 variant of that model. Round-tripping between formats re-quantises; the drift compounds.

**Operator rule:** do not treat GGUF as a storage format for AWQ / GPTQ output. If GGUF is the target, re-quantise from BF16 using llama-cpp-native tooling; the AWQ / GPTQ variants and the GGUF variants are separate artefacts.

### "ONNX export OOM'd on a 70B model"

ONNX export loads the entire model graph into memory on the exporting host. On models above ~7B parameters, this is memory-intensive. Options:

- Use HF Optimum's **external-data** mode (weights stored outside the `.onnx` graph file).
- Export shard-by-shard where the tooling supports it.
- Export on a host with sufficient RAM.
- Reconsider whether ONNX is the right target format for this model size.

## Lineage and provenance

Every conversion step must leave a trail. The chapter-05 release contract bakes this into the artefact's metadata; here is the data to collect at conversion time:

- **Source artefact** — the exact `safetensors` revision (git SHA for the HF repo, or your internal registry ID).
- **Conversion tool version** — `pip show`-equivalent for the converter, or the git SHA of the TRT-LLM / llama.cpp repo used.
- **Conversion flags** — the exact command line.
- **Calibration dataset identity** — a hash of the calibration data (or a reference to a stored, versioned dataset).
- **Quantisation config** — `quantize_config.json` or equivalent (bits, group size, calibration-dataset reference).
- **Output artefact hash** — SHA-256 of the produced file(s), to detect accidental corruption later.
- **Validation results** — the level-0 to level-3 pass/fail with numbers.

Store the trail alongside the artefact. Chapter 05 is the handshake with the ML-platform peer on where this lives.

## The CI shape for conversion

A healthy conversion workflow is a CI job that:

1. **Pulls** the source artefact by ID from the registry (not from a transient path).
2. **Pins** every tool version (converter, Python, CUDA, driver).
3. **Runs** the conversion with recorded flags.
4. **Validates** at level 0 (loads), level 1 (logit diff), and level 3 (perplexity delta) on a fixed canary.
5. **Hashes** the output artefact.
6. **Writes** the lineage record alongside.
7. **Publishes** to the registry — only if validation passed.

Level 2 (sampled-output) and level 4 (task eval) often run as a separate downstream gate — chapter 04 is that gate.

**Why CI matters here:** conversion is slow and expensive, and the artefact is a shared production dependency. A one-off script on an engineer's laptop produces an artefact nobody else can reproduce; the ML-platform peer rejects it in the release contract. Automate from day one.

## Common misreadings

- **"If it loads, it's correct."** Loading is level 0. Correctness is level 1–4. Many pathologies pass level 0.
- **"Round-tripping safetensors → GGUF → safetensors preserves the model."** No — Q-format GGUF is lossy; the de-quantised safetensors is a different model.
- **"TRT-LLM engines are portable."** They are architecture-specific. An engine built for H100 does not run on B200 without a rebuild.
- **"ONNX is the universal exchange format."** For small CV models, yes. For large LLMs, maturity is uneven; verify before relying on it.
- **"Logit diff is overkill; perplexity is enough."** Perplexity is a corpus-average scalar; it can hide local divergences that matter (e.g. a specific token class consistently drops in probability). Logit diff is the operator's level-1 regression test.
- **"The chapter-04 accuracy gate is enough; I can skip conversion validation."** No — conversion validation isolates *where* the drift was introduced (format conversion vs. quantisation vs. engine kernel). The chapter-04 gate is a go / no-go; conversion validation is the diagnostic.

## Summary

- The four production formats are **safetensors** (hub), **GGUF** (llama.cpp family), **TRT-LLM engines** (compiled binary, architecture-specific), and **ONNX** (cross-framework; less common for LLM serving).
- The conversion matrix is one-way-dominant: `safetensors` flows out; most formats do not cleanly flow back in. Treat `safetensors` as the lineage root.
- **Round-trip validation has four levels**: load, logit-level diff, sampled-output equivalence, perplexity + task eval. Run levels 0, 1, 3 in CI on every conversion; levels 2 and 4 feed the chapter-04 accuracy gate.
- The common pathologies are **dtype mismatch**, **permutation / layout mismatch**, **tokeniser mismatch**, **zero-point handling mismatch**, **GGUF re-quantisation drift**, **TRT-LLM build parameter mismatches**, and **ONNX export fragility on large models**.
- Every conversion must record **source artefact, tool version, conversion flags, calibration dataset identity, output hash, and validation results**. This is the lineage trail; chapter 05 is the handshake with the ML-platform peer.
- Automate from day one. A one-off conversion is a one-off problem.

Chapter 04 moves from "did the format conversion preserve the model" to "did the quantisation preserve its utility" — the accuracy regression gate, the discipline that turns level-1 logit diffs and level-3 perplexity into a defensible release decision.
