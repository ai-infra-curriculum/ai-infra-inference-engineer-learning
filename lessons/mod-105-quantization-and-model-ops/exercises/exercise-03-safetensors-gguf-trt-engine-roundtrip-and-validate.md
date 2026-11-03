# Exercise 03 — safetensors → GGUF → TRT-LLM Engine Round-Trip and Validate

**Time:** 4 hours · **Output:** three serving artefacts (safetensors, GGUF, TRT-LLM engine), one conversion log, one round-trip validation table, one pathology-isolation drill, one one-page cheatsheet

## Goal

Make [Chapter 03](../03-model-format-conversion-and-roundtrip.md) concrete by **converting one model across three production formats** (`safetensors` → GGUF → TRT-LLM engine), **serving each**, and **running the four-level validation** on each format. By the end you can state exactly what each conversion did to the model and have a reusable validation harness for the module.

## Problem statement

A model your team ships is distributed in three formats: `safetensors` for the primary vLLM / SGLang serving fleet, GGUF for a desktop / Ollama consumer path, and a TRT-LLM engine for the latency-optimised NVIDIA-Triton fleet. The three formats must produce near-identical outputs (within a stated tolerance) because the team cannot defend a position where "the Ollama users get a different model." This exercise is the round-trip discipline that establishes that defensible position.

## Prerequisites

- A GPU host that can serve all three formats (an H100 or A100 covers everything; smaller GPUs cover subsets).
- CPU host or GPU host for GGUF (llama.cpp runs on both).
- The exercise-01 benchmark harness.
- Clone of `llama.cpp` repo (pinned) for GGUF conversion.
- TRT-LLM install for engine build.
- vLLM for safetensors serving.
- A pinned base model (BF16 safetensors). Use the same model as exercise 01 unless there is a reason to switch.

## Requirements

Deliver five artefacts.

### Artefact A — Format inventory

For your chosen base model, document the starting state:

- Source: Hugging Face revision SHA.
- Format: `safetensors` (BF16).
- Files + sizes + per-file SHA-256.
- Tokenizer file(s).
- Config: `config.json` contents (summary).
- Licence.

This is the lineage root (chapter 05 reference).

### Artefact B — The three conversions

Produce three serving artefacts from the same source:

#### B.1 — `safetensors` serving

(Already present from Artefact A — document the serving command for completeness.)

- Launch command (vLLM or SGLang; pick one and document).
- Startup log.
- Health-check.

#### B.2 — GGUF conversion

Convert `safetensors` → GGUF using the `llama.cpp` tooling.

- Command: `python convert_hf_to_gguf.py <path> --outfile <out.gguf> --outtype bf16` (verify exact spelling against your `llama.cpp` revision).
- Record build time, output file size, SHA-256.
- Optionally also produce a `Q4_K_M` quantised GGUF via `llama-quantize <in.gguf> <out.gguf> Q4_K_M`. Keep the BF16 GGUF as the format-fidelity target; the Q4_K_M is a secondary quantised comparison.
- Serve the BF16 GGUF with `llama-server` or an equivalent (`llama-cpp-python` server, Ollama with the raw GGUF). Document the serving command.
- Health-check.

#### B.3 — TRT-LLM engine conversion

Convert `safetensors` → TRT-LLM engine (BF16 only for this exercise; keep the quantisation question orthogonal — that is exercises 01 and 02).

- `convert_checkpoint.py` (BF16 target, matching TP for your hardware).
- `trtllm-build` with an envelope (`max_batch_size`, `max_input_len`, `max_seq_len`) that covers the benchmark.
- Record each step's command, duration, output file sizes, SHA-256.
- Serve either via direct TRT-LLM Python API or via Triton `tensorrtllm_backend`. Document.
- Health-check.

### Artefact C — Round-trip validation table

Run the chapter-03 four-level validation on each format.

For each format and each level, run against the same fixed prompt set (reuse the exercise-01 prompt set):

**Level 0 — loads**
- Pass / fail (binary).

**Level 1 — logit-level diff** (against `safetensors` as reference)
- Max abs logit diff.
- Mean abs logit diff.
- Top-1 agreement rate.
- Top-5 agreement rate.

Note: for GGUF via llama.cpp, extracting logits requires the `--log-probs` or equivalent flag; verify. For TRT-LLM, logits are directly available via the engine API.

**Level 2 — sampled-output equivalence** (against `safetensors` as reference)
- Exact-match rate.
- Mean edit-distance (character-level) per prompt.
- Pick 3 prompts where outputs differ most; cite the differences.

**Level 3 — perplexity delta**
- Compute perplexity on a 100-passage subset of wikitext-2-raw-v1.
- Report per format: raw perplexity, absolute delta, percentage delta.

Publish as a 3 × 4 table (format × level), with cells containing the metric values and any pass/fail notes.

### Artefact D — Pathology isolation drill

The round-trip will usually surface at least one discrepancy. Pick the biggest one and run the chapter-03 debug ladder:

1. Compare `config.json` across the three deployments.
2. Compare first-layer weight statistics (mean, std, max abs) if accessible.
3. Compare attention-layer-0 output on a short fixed prompt (where accessible — this often requires engine-side introspection; document any blockers).
4. Isolate the first divergence point.

If there is no significant discrepancy to isolate, instead run a **deliberate** pathology:

- Convert a model with mismatched tokenizer settings (e.g. HF tokenizer vs. llama.cpp's tokenizer with a known difference).
- Observe the divergence; isolate it to tokenisation.
- Document the diagnostic path.

Writeup: ≤ 400 words covering (a) what the discrepancy was, (b) which conversion step introduced it, (c) how you ruled out alternatives, (d) what the remediation is or would be.

### Artefact E — Round-trip cheatsheet

A one-page cheatsheet that captures the operator-facing content of this exercise:

- The three conversion commands you'd actually run.
- The validation harness invocation.
- The tolerance thresholds you'd use as a go / no-go for each format (if a reasonable default emerged from the exercise).
- The top three pathologies and their diagnostics.

This is the artefact your team's docs page should point at.

## Starter guidance

- **Pin tool versions.** `llama.cpp` git SHA, TRT-LLM version, vLLM version, Python package versions.
- **Keep the base model identical.** The three conversions should start from the same `safetensors` revision; otherwise you are comparing format against model change.
- **Use the same tokenizer across formats where possible.** llama.cpp's tokenizer is implemented in C++ and can differ from HF's in edge cases (special tokens, BOS / EOS handling, byte-fallback behaviour). Document any divergence.
- **Set sampling to temperature 0** for the validation runs (deterministic, repeatable).
- **For GGUF, strongly prefer the BF16 variant as the format-fidelity target.** The Q-formats are *also* quantisations; mixing format-conversion with quantisation makes isolation harder. Keep them separate.
- **TRT-LLM build is slow (minutes to tens of minutes on larger models).** Start the build early.
- **Use the vLLM `benchmark_serving.py` harness and extend it for logit capture.** The engine exposes a `/v1/completions` with `logprobs=K` which is the hook.

## Fallbacks for constrained hardware

- **No TRT-LLM-capable GPU.** Skip Artefact B.3 and the TRT-LLM row of Artefact C; state the gap. The two-format comparison still has value.
- **No GPU at all for GGUF.** llama.cpp runs on CPU; GGUF serving on CPU is slow but functional.
- **Small model only.** Keep to 7B / 8B; dynamics are the same.

## Acceptance criteria

1. **Artefact A** documents the lineage root (source revision, SHA, files, licence).
2. **Artefact B** has three serving deployments, each with command + health-check pass.
3. **Artefact C**'s table is complete across (format × level); numbers are consistent with "the format-conversion preserves outputs within noise" — or if not, Artefact D investigates.
4. **Artefact D** either isolates a real discrepancy (preferred) or executes a deliberate pathology drill; writeup ≤ 400 words with clear diagnostic path.
5. **Artefact E** fits on one page and includes commands, validation harness, thresholds, and top-three pathologies.
6. All five artefacts live in version control; cross-reference exercise 01's environment manifest.

## Common failure modes

- **Tokenizer mismatch inflating logit-diff.** Trace the first-token embedding; if it differs for a tokenizer-only reason, the "logit diff" is a tokenizer diff. Fix the tokenizer or document.
- **GGUF BF16 is not actually BF16 on the serving path.** llama.cpp sometimes dequantises to FP32 internally; the output can still differ from the HF reference by tiny amounts. Treat the delta's magnitude as the signal.
- **TRT-LLM `max_input_len` too small**, causing the benchmark to silently truncate prompts. Validate.
- **Comparing wikitext perplexity on a tokenizer-divergent setup.** Perplexity is tokenizer-dependent; the three deployments must tokenize wikitext identically for the comparison to be meaningful.
- **Sampled-output exact-match at temperature > 0.** Pin to temperature 0.
- **Skipping Level 0 because "of course it loads."** Level 0 catches silent dtype mismatches; always run it.

## Keep it

The validation harness is the module's reusable backbone. The cheatsheet (Artefact E) is the team-wiki document. Exercises 04 and 05 reference this exercise's validation methodology. The pathology drill (Artefact D) is the template for future "the formats aren't matching anymore" incident response.
