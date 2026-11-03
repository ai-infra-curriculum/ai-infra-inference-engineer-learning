# Exercise 03 — TensorRT-LLM Engine Build and Triton Serve

**Time:** 4 hours · **Output:** one convert/quantize/build/serve pipeline run, one Triton model repository, one build-vs-serve-shape audit, one operator cheatsheet

## Goal

Make [Chapter 03](../03-trt-llm-and-triton-and-dynamo.md) concrete by running the full TRT-LLM pipeline end-to-end against a small model: convert the checkpoint, (optionally) quantize it, build an engine artifact, serve it through Triton's `tensorrtllm_backend`, and verify that the compiled envelope matches the serving-time envelope. You should leave the exercise able to defend the operator trade-off between the TRT-LLM path and the Python-defined (vLLM / SGLang) path.

## Problem statement

Your team is deciding whether to add TRT-LLM as a second production engine (alongside vLLM) for a hot, high-volume model where throughput ceiling matters. Before you make that call, you need direct, hands-on experience with the compile-ahead workflow — including what it costs operationally when a flag drifts or a model updates. You will:

1. Produce a converted + (optionally) quantized TRT-LLM checkpoint.
2. Build a `.engine` artifact with a specific TP degree, batch envelope, and context envelope.
3. Package it into a Triton model repository with a working `tensorrtllm_backend` config.
4. Launch Triton, serve the model, and verify basic generation behaviour.
5. Audit the mapping from build-time flags to runtime behaviour.
6. (Optional) Run a quantization accuracy sanity check if you built an FP8 or INT-quantized engine.
7. Author an operator cheatsheet.

## Prerequisites

- A workstation or lab machine with at least one NVIDIA GPU of Ampere+ compute capability. Hopper strongly preferred if you want to exercise FP8; Blackwell for FP4.
- Access to the NVIDIA TRT-LLM container images or a working local install of a pinned TRT-LLM version (CUDA, cuDNN, TensorRT, Python, NCCL versions must match the TRT-LLM release — this is the single most common source of friction).
- Triton Inference Server image matching the TRT-LLM backend version (NVIDIA ships paired versions; mismatching them is a reliable way to lose an afternoon).
- A small open-weight model that TRT-LLM's current convert scripts support — e.g., a 1B–8B instruct model from a supported family.

## Requirements

Deliver six artefacts.

### Artefact A — Convert step

Run the convert step for your chosen model + TP degree. Example (verify exact script and flags against your TRT-LLM release — the convert tooling moves):

```bash
python convert_checkpoint.py \
    --model_dir ./llama-3-8b-instruct-hf \
    --output_dir ./llama-3-8b-trtllm-tp1 \
    --dtype bfloat16 \
    --tp_size 1
```

Capture: the exact command, the convert log, and the resulting directory listing. Note: how big are the per-rank shards? Is the config.json inside the output directory present and consistent with your flags?

### Artefact B — (Optional) Quantize step

Pick one of:

- Skip (serve BF16/FP16 directly). Note why.
- FP8 (Hopper+) via TensorRT Model Optimizer. Supply a small calibration dataset (a few hundred representative prompts).
- INT8 / AWQ / GPTQ if your model + release path supports it.

Record: the quantization recipe, the calibration data origin, and the resulting checkpoint size. Compare to the pre-quantization size.

### Artefact C — Build step

Run `trtllm-build` against the (possibly quantized) checkpoint. Choose an envelope that constrains the serving behaviour so you can see what the compile-time flags *do*:

```bash
trtllm-build \
    --checkpoint_dir ./llama-3-8b-trtllm-tp1 \
    --output_dir   ./llama-3-8b-trtllm-tp1-engine \
    --gemm_plugin  <bf16|fp8|...> \
    --max_batch_size 32 \
    --max_input_len 4096 \
    --max_seq_len   8192 \
    --max_num_tokens 8192 \
    --use_paged_context_fmha enable \
    --paged_kv_cache enable \
    --tp_size 1
```

Capture: the build command, the build log (how long did it take?), the resulting engine file size, and the `config.json` next to the engine (what settings are baked in?).

### Artefact D — Triton model repository and launch

Build a Triton model repository with the TRT-LLM backend wiring. The canonical layout uses an ensemble of `preprocessing` + `tensorrt_llm` + `postprocessing` or the `tensorrt_llm_bls` BLS variant. Adapt from the reference configs in the `tensorrtllm_backend` repo; do not try to write from scratch.

Capture: the repository tree (`tree model_repository/`), the key `config.pbtxt` for the `tensorrt_llm` model (annotated with the parameters you edited), and the Triton launch command (image tag, mounted volumes, GPU count, ports).

Then launch Triton and verify it comes up healthy:

- `curl http://localhost:8000/v2/health/ready` returns 200.
- `curl http://localhost:8000/v2/models/<ensemble-name>` returns the model config.

### Artefact E — Serve-time behaviour verification

Hit the endpoint with a small test suite:

1. A short-prompt request (< 100 tokens in, < 100 out). Record TTFT, E2E.
2. A prompt at the edge of `--max_input_len`. Does it accept? Does the output quality look sensible?
3. A prompt *past* `--max_input_len`. Does Triton reject cleanly? With what error?
4. A concurrent burst past `--max_batch_size`. Where does the queue/overflow show up?
5. A multi-turn chat session that exercises prefix caching (if `enable_kv_cache_reuse` is on).

For FP8 or INT-quantized builds: run a tiny accuracy sanity check — ask the model a handful of factual or arithmetic prompts and compare qualitatively against the BF16 reference. This is *not* an eval gate (mod-105 territory); it is a reality check that the quantization build did not catastrophically regress.

### Artefact F — One-page operator cheatsheet

A single-page summary a sibling engineer can use to:

- Reproduce the convert → build → serve pipeline against a new model.
- Know which TRT-LLM flags are *build-time* (pinned) and which are *runtime* (editable in Triton config).
- Know the four operator-depth questions for an engine (from mod-101 chapter 04) applied to this build: hardware target, numeric path, parallelism topology, shape envelope.
- Know where in the pipeline to look when a specific symptom appears (convert warning, build mismatch, Triton config drift, serve-time rejection).

## Starter guidance

- **Version alignment is the whole game.** TRT-LLM, Triton, the `tensorrtllm_backend`, CUDA, cuDNN, TensorRT, NCCL, and the Python stack must all match a known-good combination. Use NVIDIA's reference container when possible.
- **TP=1 is fine for this exercise.** You are learning the pipeline, not benchmarking at scale. Save multi-GPU TP builds for the module-107 track.
- **Compile time matters.** A 70B FP8 build takes meaningfully longer than an 8B BF16 build. Budget wall-clock time.
- **The accuracy sanity check is *not* a release gate.** Mod-105 will cover the real gate; this is just "did FP8 destroy the model?"
- **Record every version.** The deliverables are useless if you don't know what versions produced them.
- **Fallbacks:** if you cannot access an NVIDIA GPU for this exercise, build on a cloud spot VM for an hour. There is no CPU-only path for TRT-LLM; this is explicitly an NVIDIA-only pipeline.

## Acceptance criteria

1. The convert, build, and Triton launch all complete without errors.
2. The resulting engine serves a request end-to-end and returns coherent output.
3. The build-vs-runtime audit in Artefact F clearly labels at least six flags as either build-pinned or runtime-editable, with evidence (either the flag's documentation or its location in `config.json` vs. `config.pbtxt`).
4. The max-input-len edge test (Artefact E, item 3) actually triggers a Triton-level rejection or truncation, demonstrating that the compile-time envelope is enforced.
5. The cheatsheet (Artefact F) fits on one page and lets a reader reproduce the pipeline for a new model.
6. If you built a quantized engine, the sanity-check prompts in Artefact E produce reasonable output. (If they do not, document it — a failed quantization sanity check is itself a useful finding, and leads into the mod-105 discussion.)

## Common failure modes

- **Version mismatch between TRT-LLM, Triton, and backend.** Produces cryptic C++ errors on model load. Fix by rebuilding on NVIDIA's reference image.
- **Convert step shards don't match build step `--tp_size`.** Produces an error on build load. Re-convert.
- **Backend `config.pbtxt` references an engine path that doesn't exist, or wrong model repo layout.** Triton will log the error on startup; do not just chase the symptom in Python.
- **Default calibration data for FP8.** Default-provided calibration data (e.g., a generic text corpus) is better than nothing but is *not* a production recipe. For this exercise it's fine; call it out in Artefact B.
- **Treating the build as reproducible across hardware generations.** It isn't. The engine is pinned to the GPU generation it was built on; building on an H200 and running on an L40S will fail.

## Keep it

The pipeline you produce in this exercise is a *reference* for the mod-105 (quantization) and mod-107 (multi-GPU engines, Dynamo) exercises. Keep the model repository and the per-step command log in version control. Refresh when you upgrade TRT-LLM.
