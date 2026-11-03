# Chapter 05 — Release Contract with the ML-Platform Peer

> An artefact without a lineage pointer and a signing hook is a 70 GiB blob somebody has to remember the provenance of. In a year it is a liability.

## Motivation

Chapters 01–04 produced a quantised artefact that passes an accuracy gate. Chapters 03 and 04 collected the lineage data as a side effect. This chapter is the handshake with the ML-platform peer at [`ai-infra-ml-platform`](https://github.com/ai-infra-curriculum/ai-infra-ml-platform-learning): what fields the artefact must carry, what metadata is required, where lineage points, how signing works, what the registry promises back.

The chapter is written from the inference-engineer side of the contract. It defines what *this role* owes the ML-platform peer and what it must *require* from that peer. The ML-platform peer's track owns the registry service itself; this chapter is how we interface with it.

By the end you can:

- Specify the artefact format precisely enough that the ML-platform peer can store it.
- Enumerate the metadata fields and defend each one.
- Point at the lineage trail (source checkpoint → quant recipe → calibration data → tool versions).
- Compute and attach a quantisation-config hash.
- Declare the licence in a form that composes with the base model's licence.
- Hook into the signing workflow (sign the artefact, verify at serving-time load).
- Run the "what breaks if a field is missing" audit so the contract has teeth.

<!-- needs-research: specific registry implementations vary (MLflow, Weights & Biases, Vertex AI Model Registry, SageMaker Model Registry, Databricks Unity Catalog, HuggingFace Hub, in-house). The contract below is implementation-agnostic; the specific field names and APIs will adapt. -->

## The structural picture — contract as code

The contract is a specification, versioned alongside the artefact format, that both sides implement:

- **Inference engineer** (this role) implements the *producer* side: writes the metadata, computes the hashes, signs the artefact.
- **ML-platform peer** implements the *consumer* side: validates the metadata, stores by hash, serves to the registry's downstream consumers.
- **Both** agree on the schema and the signing scheme.

A new field or a schema change requires a version bump and a sign-off on both sides. "Contract as code" means the schema lives in a repo both sides can PR against.

## The artefact format

An artefact is the deployable unit the registry holds. For a quantised LLM, it is a **directory** (or a tarball) containing:

```
<artefact-root>/
├── artefact.yaml           ← the metadata manifest (this contract's primary deliverable)
├── artefact.sig            ← signature over artefact.yaml and all file hashes
├── model/
│   ├── config.json
│   ├── tokenizer.json      ← and other tokenizer files
│   ├── generation_config.json
│   ├── model-00001-of-NN.safetensors
│   ├── model-00002-of-NN.safetensors
│   ├── …
│   ├── model.safetensors.index.json
│   ├── quantize_config.json  ← or hf_quant_config.json, per tooling
│   └── chat_template.jinja   ← if applicable
└── validation/
    ├── gate_report.json    ← the chapter-04 gate's output
    ├── gate_report.md      ← human-readable version
    ├── conversion_log.json ← the chapter-03 conversion lineage
    └── canary_hash.txt     ← SHA-256 of the canary dataset used
```

**For TRT-LLM compiled engines**, replace the `model/` safetensors shards with the `.engine` file(s) and the `config.json` from `trtllm-build`. The rest of the structure is identical.

**For GGUF artefacts**, replace the `model/` contents with the `.gguf` file. The rest of the structure is identical.

The artefact is **immutable once published**. Any change (even a metadata correction) produces a new artefact with a new ID.

## The metadata manifest — `artefact.yaml`

The manifest is the one file both sides agree to parse. Field spec:

```yaml
schema_version: "1.0"             # bump on breaking change

# ─────────────── identity ───────────────
id: "org/model-family/variant/version"   # fully-qualified; registry-chosen scheme
display_name: "Llama-3-70B-Instruct FP8 E4M3 v2"
family: "llama-3-70b-instruct"           # for grouping / lineage
variant: "fp8-e4m3-static"               # the quantisation flavour
version: "2"                             # semver or integer, per registry convention

# ─────────────── artefact contents ───────────────
artefact_format: "safetensors"           # safetensors | gguf | trtllm-engine | onnx
file_manifest:
  - path: "model/config.json"
    sha256: "..."
    size_bytes: ...
  - path: "model/model-00001-of-08.safetensors"
    sha256: "..."
    size_bytes: ...
  - …                                     # every file, no exceptions

total_size_bytes: ...                    # sum of file_manifest sizes
sha256_of_manifest: "..."                # a hash the signature is computed over

# ─────────────── model identity ───────────────
model_card_slice:
  base_model_hf_id: "meta-llama/Llama-3.1-70B-Instruct"
  base_model_revision: "<git-sha-of-HF-repo>"
  base_model_license: "llama-3.1-community-license"
  base_model_license_url: "https://..."
  parameters: 70_553_706_496              # count, not category
  architecture: "LlamaForCausalLM"
  context_length: 128000
  intended_use: ["chat", "instruct", "general"]
  known_limitations:
    - "..."

# ─────────────── quantisation config ───────────────
quantization:
  format: "fp8"
  sub_format: "e4m3"
  method: "static_calibration"            # or "dynamic_awq" | "static_gptq" | ...
  weight_bits: 8
  activation_bits: 8
  kv_cache_bits: 8                        # separate flag; may be null if BF16 KV
  granularity:
    weight: "per_tensor"                  # or per_channel | per_group:128 | block:16 ...
    activation: "per_tensor"              # or per_token | per_channel ...
    kv: "per_head"
  calibration:
    dataset_id: "canary-calibration-v4"
    dataset_hash_sha256: "..."
    sample_count: 512
    sample_max_tokens: 2048
  config_hash_sha256: "..."               # hash of the above sub-tree; see "quant-config hash" below

# ─────────────── lineage ───────────────
lineage:
  source_artefact_id: "org/llama-3-70b-instruct/bf16/v1"  # the parent BF16 variant
  conversion_tool: "tensorrt-model-optimizer"
  conversion_tool_version: "0.19.0"
  conversion_git_sha: "<git-sha-if-sourced-from-repo>"
  conversion_command: "python quantize.py --model_dir=... --qformat=fp8 ..."
  conversion_timestamp_utc: "2026-11-01T14:22:08Z"
  conversion_environment:
    python: "3.11.9"
    cuda: "12.6"
    driver: "550.90.07"
    torch: "2.5.1"
    tensorrt_llm: "0.14.0"
  parent_lineage_pointer: "org/llama-3-70b-instruct/bf16/v1 → upstream HF revision <sha>"

# ─────────────── validation ───────────────
validation:
  gate_name: "fp8_quantization_release_gate"
  gate_version: "3"
  gate_result: "pass"                     # pass | fail | escalate
  canary_id: "canary-v3"
  canary_hash_sha256: "..."
  metrics:
    perplexity:
      corpus: "wikitext-2-raw-v1-100-passages"
      reference_value: 4.521
      candidate_value: 4.556
      delta_percent: 0.774
      ci_95:
        low_percent: 0.42
        high_percent: 1.10
      threshold_pass_percent: 1.0
      threshold_fail_percent: 3.0
    task_specific:
      name: "mt_bench_subset_judged_pairwise"
      reference_score: 7.84
      candidate_score: 7.72
      delta_points: -0.12
      ci_95:
        low: -0.33
        high: 0.09
      threshold_pass_points: -0.5
      threshold_fail_points: -1.5
  gate_run_timestamp_utc: "2026-11-01T15:48:33Z"
  gate_run_hardware: "H100 80GB SXM5"
  gate_run_engine: "vllm-0.8.4"

# ─────────────── compatibility ───────────────
compatibility:
  supported_engines:
    - name: "vllm"
      min_version: "0.8.0"
    - name: "trt-llm"
      min_version: "0.14.0"
    - name: "sglang"
      min_version: "0.3.5"
  required_hardware:
    nvidia_sm_min: 90                     # Hopper or newer for FP8
    amd_gfx_min: null
  required_memory_gb:
    bf16: 140                             # for reference
    this_variant: 70

# ─────────────── distribution / policy ───────────────
distribution:
  licence: "llama-3.1-community-license"
  licence_compatible_with_base: true
  export_restrictions: []                 # or ["US-EAR"], etc.
  attribution: "Modified from meta-llama/Llama-3.1-70B-Instruct, quantised by <org>."
  internal_tags: ["chat", "primary", "canary-tier"]

# ─────────────── signing ───────────────
signing:
  algorithm: "ed25519"                    # or ecdsa-p256 | sigstore-cosign | ...
  signer_identity: "quant-pipeline-signer@<org>"
  signature_file: "artefact.sig"
  key_fingerprint: "SHA256:..."
  trust_anchor: "<ref to the signing CA / sigstore config>"
```

Fields can be optional per registry convention. The ones that are **non-negotiable** in this role's half of the contract are called out below.

## The quantisation-config hash

The `quantization.config_hash_sha256` is the signature of the quantisation decision — bits, granularity, calibration dataset ID + hash, method. It is computed by:

1. Canonicalising the `quantization` subtree (sorted keys, UTF-8 encoding, newline-normalised).
2. Hashing with SHA-256.

**Why it matters:**

- **Reproducibility.** Two artefacts with the same config hash produced from the same source should be byte-identical (if the tooling is deterministic). Mismatch means the quantisation configuration drifted even if the human-readable name didn't change.
- **Audit.** When a regression lands in production and the hypothesis is "the calibration dataset changed," the config hash either confirms or refutes it in one line.
- **Dedup.** Registry can refuse to store an artefact whose config hash already exists for the same source — prevents accidental re-quantisations with no changes.

**Common failure:** the calibration dataset's content changed but the dataset_id stayed the same. The config hash doesn't update because the manifest content didn't update. Fix: include the dataset's content hash in the config, not just the ID.

## The lineage pointer

Every artefact must point back to its source. The chain:

```
upstream HF revision
   │ (download)
BF16 safetensors in registry
   │ (quantise with recipe R, calibration data C)
Quantised safetensors in registry
   │ (convert to TRT-LLM engine)
TRT-LLM .engine in registry
```

Each arrow has:
- A **source artefact ID** (what we started from).
- A **conversion step** (tool + version + flags + hash of inputs).
- An **output artefact ID** (what we produced).

Lineage is **reconstructable** — given the final artefact, you can walk back to the HF revision in O(depth) hops. The registry owns the storage; this contract owns the field format.

**Why operators care:**

- **Rollback.** "Which BF16 did this FP8 come from" is a one-line query, not a dig through pipeline logs.
- **Supply-chain security.** When an upstream HF revision is retracted or re-published, you need to know every artefact derived from it.
- **Compliance.** Licence inheritance (next section) depends on this chain being complete.

## The licence field and inheritance

A quantised variant is a **derivative work** of the base model under most licences. The licence field has three parts:

- `distribution.licence` — the licence this artefact is distributed under.
- `distribution.licence_compatible_with_base` — a boolean the producer asserts.
- `model_card_slice.base_model_license` — the base model's licence, redundantly recorded.

**The compatibility assertion is this role's responsibility to make correctly.** If the base model is under the Llama Community Licence, the derivative is governed by it; if under Apache 2.0, usually the derivative can be redistributed under Apache 2.0 but the derivative-work attribution must be included. For some commercial-use models, the licence prohibits certain downstream deployments; this chapter does not resolve the legal question but the contract carries the fields for the legal / compliance review to check.

**Signalling to the registry:**
- `licence` present and non-empty → registry accepts.
- `licence_compatible_with_base: false` → registry rejects or requires explicit override (defer to the ML-platform peer's policy).
- `export_restrictions` populated → registry adds a gating step before distribution to restricted regions.

**Boundary:** licence interpretation is a legal / compliance question that is explicitly out of scope for this track (see CURRICULUM.md). This contract carries the data for that review; it does not substitute for it.

## The signing hook

An artefact without a signature is an artefact anyone can replace. A signed artefact survives supply-chain attacks and accidental swaps.

### What signs what

- The producer (this role's CI pipeline) computes a hash of the manifest (and every file listed in `file_manifest`).
- A signing key (held by the pipeline, not by a human) signs the hash.
- The signature is written to `artefact.sig` alongside the manifest.
- The registry stores the artefact and the signature together.

### What verifies what

- When a worker loads an artefact, it:
  1. Reads the manifest.
  2. Recomputes the hash of every file listed in `file_manifest` and verifies it matches.
  3. Verifies the signature over the manifest hash against the configured trust anchor.
- Any failure aborts the load. The worker does not serve an un-verifiable artefact.

### Signing-scheme options

- **Static key-pair signing** (ed25519 / ecdsa-p256) — simple, works offline, requires key management.
- **Sigstore / cosign** — keyless signing via OIDC identity; integrates with Fulcio and Rekor transparency log; see sigstore.dev.
- **In-TOTO attestations** — rich provenance metadata; often paired with Sigstore.

The ML-platform peer owns the signing-scheme decision at the org level; this contract carries the fields for whichever scheme is chosen.

### Why operators care

- **Supply chain.** A compromised registry upstream of the worker should not become a compromised worker. Verification at load time catches it.
- **Rollbacks.** A signed artefact's integrity is self-contained; it does not depend on a trusted registry link.
- **Audit.** `signer_identity` tells you which pipeline produced the artefact. Combined with lineage, this is a complete provenance trail.

**Peer boundary:** adversarial-defence depth around signing (threat modelling, key rotation, trust-anchor compromise response) is a security peer concern — [`ai-infra-security-learning`](https://github.com/ai-infra-curriculum/ai-infra-security-learning). This chapter requires that signing be in place and specifies the on-the-wire fields; it does not design the trust model.

## The "what breaks if a field is missing" audit

Every field in the manifest should answer the question: *what fails if this is missing or wrong?* If the answer is "nothing concrete," the field is over-engineered. Spot-check:

| Field | What breaks if missing / wrong |
|---|---|
| `id` | Registry can't address the artefact. Nothing works. |
| `file_manifest` with per-file hashes | Can't verify integrity; a corrupted file loads and produces garbage. |
| `quantization.config_hash_sha256` | Can't dedup, can't audit calibration drift. |
| `lineage.source_artefact_id` | Can't trace a regression back to upstream changes. |
| `validation.gate_result` | Registry can't enforce the chapter-04 gate; artefact promotes without validation. |
| `compatibility.supported_engines.min_version` | Workers may try to load an incompatible artefact; silent kernel failure. |
| `compatibility.required_hardware.nvidia_sm_min` | Artefact lands on Ampere SKU; FP8 kernels unsupported; worker crashes at load. |
| `distribution.licence` | Legal / compliance exposure; registry may reject. |
| `signing.signature_file` | No supply-chain guarantee. |

**The audit as a drill:** for each field, write one sentence of "what breaks if missing." If multiple fields have the same answer, consolidate. If a field has no answer, remove it.

## The contract as a boundary

This contract is the **operational interface** between this role and the ML-platform peer. The things *this role owes*:

- Correctly populated manifest.
- Computed hashes and signatures.
- Chapter-04 gate result attached.
- Honest licence and compatibility declarations.
- Reproducible build (so the registry can verify by rebuilding).

The things *this role requires back*:

- Immutable storage once accepted (no silent re-writes).
- Addressable by ID (stable URLs or registry endpoints).
- Load-time retrieval of the manifest without downloading the full artefact (for cheap preflight checks).
- Policy enforcement at promotion time (reject un-gated artefacts; reject licence-incompatible artefacts; reject un-signed artefacts).
- A rollback primitive (promote a prior version in O(seconds), not O(minutes)).
- An audit log of who promoted what, when.

Any point where the two sides disagree on these responsibilities is a contract-review conversation.

## Common misreadings

- **"The registry will enforce whatever we ask."** The registry enforces what it is coded to enforce. The contract has to specify explicitly or the enforcement doesn't happen.
- **"A hash in the filename is enough."** Hashes in filenames are easily forged (rename the file). Hash the content and sign the manifest; verify at load.
- **"Lineage is a nice-to-have."** It is the single most important field for post-incident investigation. Treat it as non-negotiable.
- **"The licence doesn't matter for internal deployments."** It does: internal distribution is still distribution; the base model's licence still applies.
- **"Signing is cosmetic since we trust our own CI."** Trust is binary and transitive. If CI is compromised, everything downstream is compromised unless there's a verification step.
- **"The ML-platform peer owns the metadata schema; this role just fills it in."** Partially true. The schema is co-owned; the serving-side fields (quantisation, compatibility, validation) are this role's to specify. The ML-platform peer owns the fields the registry service needs (identity, storage, distribution).
- **"The chapter-04 gate result is just a boolean."** It is a complete record (metrics, CI, thresholds, canary ID). The registry stores it; downstream (incident response, future quant decisions) reads it.

## Summary

- The release contract is **manifest + hashes + lineage + licence + signing**, co-owned with the ML-platform peer, versioned as code.
- The **artefact** is an immutable directory containing model files, `artefact.yaml` manifest, `artefact.sig` signature, and validation outputs.
- The manifest's **non-negotiable fields** are identity, file-level hashes, model-card slice, quantisation config, config hash, lineage pointer, validation (gate result), compatibility (engine + hardware), licence, and signing metadata.
- The **quantisation-config hash** captures the full quantisation recipe including calibration data hash. It is the reproducibility and dedup primitive.
- The **lineage pointer** chains the artefact back to its upstream HF revision in O(depth) hops; it is the rollback and compliance primitive.
- The **licence field** is this role's assertion; legal interpretation is out of scope.
- The **signing hook** is required for supply-chain integrity. Scheme choice is a peer decision; the contract carries whichever fields are needed.
- The audit question for every field is "what breaks if missing." If nothing concrete breaks, the field is over-engineered.
- The contract is a **boundary**: this role owes manifest + hashes + gate result + signatures; the ML-platform peer owes immutable storage, policy enforcement at promotion, retrieval, rollback primitive, and audit log.

Chapter 06 moves from "we have a signed, gated, lineage-traced artefact" to "how do we swap the fleet onto it without dropping requests" — the serving-side mechanics of model swap.
