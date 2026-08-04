# Phase Marker Pipeline Fix Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Close the six remaining production-evidence trust-boundary findings while preserving a read-only planning pipeline and offline-only verification.

**Architecture:** Producer artifacts bind canonical base identity, canonical split members, immutable per-example evidence, and safe tensor contents. Consumers independently recompute every aggregate and lineage edge before model imports or output creation, while approval readiness is derived from structured workload counts rather than prose.

**Tech Stack:** Python 3.12, pytest, PyTorch safe deserialization, Transformers/PEFT CPU integration, vLLM-compatible mocked APIs, canonical JSON/JSONL/TSV hashing.

## Global Constraints

- Use RED/GREEN TDD for every production change.
- Do not load a real large checkpoint or use GPU, scheduler, or network resources.
- Preserve the pre-existing untracked `artifacts/` directory.
- Pilot seed is 42; confirmatory seeds are 101, 202, and 303.
- Production gates reject plumbing-only evidence and operator-selected substitutes.
- Create separate fix-round-3 commit(s); do not amend existing commits.

---

### Task 1: Canonical PEFT adapter identity

**Files:** Modify `phase_marker/training.py`; test `tests/phase_marker/test_training.py` and selector tests.

**Interfaces:** Training saves `adapter_config.json` with canonical `base_model_name_or_path` and pinned `revision`, even when model weights came from a local snapshot. Selector compatibility remains strict.

- [x] Add a CPU tiny-Qwen2 PEFT integration test through the actual training/build configuration path.
- [x] Run it RED and record the snapshot path leaking into saved adapter metadata.
- [x] Normalize the saved PEFT metadata to the canonical frozen identity.
- [x] Run the integration and selector acceptance tests GREEN.

### Task 2: Canonical selection and behavior evidence

**Files:** Modify `phase_marker/behavior.py`, `phase_marker/pipeline.py`; test behavior and pipeline modules.

**Interfaces:** Selection validates canonical split/validation bytes before tokenizer load; candidate manifests cover checkpoints exactly; evidence JSONL binds per-example score/logprob contributions and is independently recomputed. Production behavior consumes only canonical sibling `test.jsonl` with recomputed membership.

- [x] Add forged split, substituted validation/test, omitted/extra candidate, and pre-loader tests; run RED.
- [x] Add deterministic per-example evidence and manifest base identity assertions; run RED.
- [x] Implement canonical split/member validation, exact candidate coverage, evidence persistence, and consumer recomputation.
- [x] Run focused behavior/pipeline tests GREEN.

### Task 3: Recomputed manual audit evidence

**Files:** Modify `phase_marker/statistics.py`, `phase_marker/pipeline.py`; test statistics and pipeline modules.

**Interfaces:** Audit accepts exact TSV columns with parseable labels, reproduces the deterministic stratified generation sample, recomputes automatic correctness and disagreement metrics, and gates independently recompute the same facts.

- [x] Add missing-label-column, invalid/flipped-label, forged-metric, and sample-substitution tests; run RED.
- [x] Implement exact schema/sample/scoring validation and aggregate recomputation.
- [x] Run Task 13 and focused audit tests GREEN.

### Task 4: Safe immutable capture publication

**Files:** Modify `phase_marker/activations.py`; test `tests/phase_marker/test_activations.py`.

**Interfaces:** Capture rejects existing output immediately, safely validates tensor batches fully before model imports, writes into a new staging directory, then atomically renames to the absent destination.

- [x] Add corrupt tensor and existing-output loader-untouched tests; run RED.
- [x] Implement safe deserialization, semantic tensor validation, and immutable staged publication.
- [x] Run activation tests GREEN.

### Task 5: Safe immutable intervention publication

**Files:** Modify `phase_marker/interventions.py`; test `tests/phase_marker/test_interventions.py`.

**Interfaces:** Intervention validates activation tensor and all batch tensors/pairs before model imports, rejects existing output immediately, and publishes via an absent staging directory and atomic rename.

- [x] Add stale/missing activation tensor, corrupt batch, and existing-output loader-untouched tests; run RED.
- [x] Implement safe tensor/manifest/batch/pair validation and immutable staged publication.
- [x] Run intervention tests GREEN.

### Task 6: Structured approval workload and release verification

**Files:** Modify `phase_marker/pipeline.py`, `README.md`, ignored Task 12 report; test pipeline module.

**Interfaces:** Approval metadata is schema-versioned structured counts and separate GPU-hour estimates; readiness cross-checks exact job counts, commands, expected outputs, totals, and explicitly excluded mechanism work.

- [x] Add missing/inconsistent/undercounted workload tests; run RED.
- [x] Implement structured approval parsing, validation, and serialization; run GREEN.
- [x] Update README/report and correct the finding count.
- [x] Run all requested focused/integration/full verification and dry-run nonmutation.
- [x] Stage authorized files and create a separate non-amended commit.
