# Exercise 05 — ML-Platform Handoff Contract Drill

**Time:** 3 hours · **Output:** one versioned contract schema, one filled-in manifest for a real artefact, one signing + verification demonstration, one missing-field audit, one handoff dry run

## Goal

Make [Chapter 05](../05-release-contract-with-ml-platform.md) concrete by **drafting the contract as code**, **filling it in for one real quantised artefact**, **wiring the signing + verification path end-to-end**, and **running a handoff dry run** with the ML-platform peer (or in their stead a teammate playing the peer role). By the end you can hand the ML-platform peer a complete artefact with every field they need and no field they can't defend.

## Problem statement

The quantised artefact you produced in exercise 01 or 02, and the gate you built in exercise 04, do not yet have a path into the model registry. The ML-platform peer will not accept an artefact without a complete manifest, a lineage pointer, a signing hook, and licence declarations. This exercise closes that gap.

The output is reusable: the contract schema and the handoff drill template become the standard interface for every future quantised artefact.

## Prerequisites

- One quantised artefact from exercise 01 (FP8) or exercise 02 (AWQ or GPTQ).
- The gate output from exercise 04.
- The conversion log from exercise 03 (or an equivalent lineage trail).
- Access to a signing key or Sigstore / cosign tooling.
- Access to the ML-platform peer (recommended) or a teammate willing to play the role.
- Chapter 05 of this module read.

## Requirements

Deliver five artefacts.

### Artefact A — The contract schema (as code)

Author the contract schema as a JSON Schema (or OpenAPI schema, or Protobuf, depending on your ecosystem) in a repo both sides can PR against.

Required field groups (from chapter 05):

- **Identity**: `id`, `display_name`, `family`, `variant`, `version`.
- **Artefact contents**: `artefact_format`, `file_manifest` (array of `path`, `sha256`, `size_bytes`), `total_size_bytes`, `sha256_of_manifest`.
- **Model identity (model-card slice)**: base model HF ID + revision, base model licence, parameters, architecture, context length, intended use, known limitations.
- **Quantisation config**: `format`, `sub_format`, `method`, `weight_bits`, `activation_bits`, `kv_cache_bits`, `granularity`, `calibration` subtree (dataset ID + hash + sample count + sample length), `config_hash_sha256`.
- **Lineage**: source artefact ID, conversion tool + version + git SHA, conversion command, conversion timestamp, conversion environment (python / CUDA / driver / torch / tool), parent lineage pointer.
- **Validation**: gate name + version, gate result, canary ID + hash, metrics + CIs + thresholds, gate run timestamp + hardware + engine.
- **Compatibility**: supported engines + min versions, required hardware (SM min), required memory GB.
- **Distribution**: licence, licence_compatible_with_base, export restrictions, attribution, internal tags.
- **Signing**: algorithm, signer identity, signature file, key fingerprint, trust anchor.

Schema deliverable:

- Checked into a shared repo (not scattered across your laptop).
- Versioned (`schema_version: "1.0"`).
- Has example values and field descriptions for every field.
- Has validation rules (regex for IDs, enum for enum fields, required-field markers).

### Artefact B — The filled-in manifest for one real artefact

Take one quantised artefact (from exercises 01 or 02) and produce its complete `artefact.yaml` per the schema.

Every field must be populated with real values from your actual artefact and exercise outputs. **No placeholder TBDs.**

- Identity: from your naming convention.
- File manifest: computed by walking the artefact directory and SHA-256ing each file.
- Model-card slice: from the base model's HF model card.
- Quantisation config: from your exercise 01 / 02 recipe.
- `config_hash_sha256`: computed by canonicalising the `quantization` subtree and hashing.
- Lineage: from the environment manifest and the exercise 03 conversion log (where applicable).
- Validation: from the exercise 04 gate output.
- Compatibility: from the engine docs for the format you produced.
- Distribution: from the licence research and your organisation's attribution standards.
- Signing: computed in Artefact C.

Attach the filled-in manifest to the artefact directory.

### Artefact C — Signing + verification demonstration

Wire up the signing hook end-to-end.

- Pick a signing scheme (ed25519 for a simple key-pair, or Sigstore / cosign for keyless signing with OIDC — both are acceptable; document choice).
- Generate or acquire the signing key (do not use a production key).
- Compute `sha256_of_manifest` by canonicalising the manifest (sorted keys, UTF-8, LF line endings) and hashing.
- Sign the hash. Write `artefact.sig` to the artefact root.
- Record the signer identity, key fingerprint, and trust anchor in the manifest's `signing` block.

Then wire up the verification path:

- Write a small `verify.py` (or equivalent) that:
  1. Reads the manifest.
  2. Recomputes `sha256` for every file in `file_manifest` and verifies it matches.
  3. Recomputes `sha256_of_manifest` and verifies it matches.
  4. Verifies `artefact.sig` against the manifest hash using the recorded public key / trust anchor.
  5. Exits 0 on success, non-zero on failure.
- Run `verify.py` on the complete artefact; show it passes.
- **Then run three failure drills:**
  1. Mutate one byte of one weight file. Re-run `verify.py`; show it fails at step 2.
  2. Mutate one byte of the manifest (e.g. change a licence field). Re-run; show it fails at step 3 or 4.
  3. Mutate the signature file. Re-run; show it fails at step 4.

Document each drill with the command and the failure output.

### Artefact D — Missing-field audit

For every field in the contract schema, write one sentence: *what breaks if this field is missing or wrong?*

Format as a table (field name, what breaks).

For each row, categorise the impact:
- **Catastrophic** (worker can't load; registry rejects; data loss).
- **Operational** (observability gap; auditability gap; rollback slowed).
- **Compliance** (licence exposure; signing bypass).
- **Cosmetic** (nice-to-have).

Identify any fields rated **cosmetic**. Either defend why they're in the schema or delete them.

### Artefact E — Handoff dry run

Simulate the handoff conversation with the ML-platform peer (or a teammate playing the role).

Prepare:

- The complete artefact (manifest + files + signature).
- The schema (Artefact A).
- The verify script and the three failure drills.
- The missing-field audit.

Conduct the handoff:

1. Present the artefact.
2. Walk through the schema with the peer; agree on any field clarifications.
3. Demonstrate `verify.py` passing and the three failure drills.
4. Walk through the missing-field audit; agree on any cosmetic deletions.
5. Agree on the registry endpoint, how the peer promotes the artefact, and how rollback works.
6. Agree on what happens if the artefact fails verification at serving-time load.
7. Agree on the SLA for registry retrieval (how fast must the serving fleet be able to pull from the registry).

Produce a one-page handoff summary document that captures the agreed-upon decisions, open items, and sign-off from both sides.

## Starter guidance

- **The schema is the contract.** If it isn't in the schema, don't rely on it. Everything the two sides agree to goes in writing.
- **Canonicalise before hashing.** Different YAML libraries emit different byte sequences for the same content; a canonical form (sorted keys, specific indentation, specific line endings) is required for reproducible hashes.
- **Pin everything in `lineage`.** Future-you reading this manifest a year from now will not remember which Python, which CUDA, which tool. Pin.
- **Prefer Sigstore / cosign to raw key pairs if your org has the infrastructure.** Keyless signing via OIDC removes key-management burden. If not, ed25519 is simple and works.
- **Keep the licence block honest.** If you're not sure the licence is compatible, flag it (`licence_compatible_with_base: false`) and route to legal. Don't guess.
- **Make verify.py the inverse of sign.py.** They should be symmetric; a round-trip proves correctness.
- **The handoff dry run with a real peer surfaces issues the schema alone can't.** Prefer a real peer or teammate; a solo paper exercise misses the "the peer's registry doesn't support field X" conversation.

## Fallbacks for constrained environments

- **No real ML-platform peer available.** A teammate who has not been part of the exercise can play the role; a thoughtful critic is more valuable than a yes-person.
- **No Sigstore infrastructure.** Local ed25519 key pair is fine for the drill.
- **No real registry available.** Simulate with a directory + a Python helper that validates manifests before "accepting" them.

## Acceptance criteria

1. **Artefact A** is a complete JSON Schema (or equivalent) checked into a repo, version-pinned, with every field documented and validated.
2. **Artefact B** is a filled-in manifest for a real artefact — no placeholders, every hash computed, every field defensible.
3. **Artefact C** demonstrates signing and verification end-to-end, including the three failure drills.
4. **Artefact D** has a per-field audit with impact category; any cosmetic field is justified or removed.
5. **Artefact E** is a handoff summary with real decisions, open items, and dual sign-off (or dual role-play sign-off).
6. All five artefacts live in version control; the artefact's manifest cross-references the gate output (exercise 04) and the conversion log (exercise 03) by hash.

## Common failure modes

- **"We'll figure out the signing later."** This is the shortcut everyone takes. Future-you lives with it. Do it now.
- **Non-canonical hashing.** Two YAML emitters produce two hashes for the "same" content; the signature doesn't verify; nobody can tell why. Pick a canonical form and enforce it.
- **Licence over-claiming.** Marking a non-compatible licence as compatible to avoid the legal review. Don't.
- **Field explosion.** Adding fields the peer did not ask for. The schema is a minimal interface, not a bucket.
- **Not running verification at serving load.** The signing is pointless if the worker doesn't verify. The chapter-06 model-swap uses this; exercise 06 confirms it.
- **Skipping the missing-field audit.** This is where you discover the schema has fields nobody can defend.
- **Dry-run with yourself only.** Blind spots stay blind. Use a real peer.

## Keep it

The contract schema, the manifest template, the signing + verify scripts, and the handoff summary template are the standard interface with the ML-platform peer. They apply to every future quantised artefact and compose with modules 106 (adapter / LoRA artefacts) and 108 (speculative-decoding draft artefacts), where the same shape of contract covers different artefact types. Exercise 06 references the signing and verification path directly.
