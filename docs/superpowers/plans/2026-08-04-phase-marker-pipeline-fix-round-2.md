# Phase Marker Pipeline Fix Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all seven re-review findings with executable, provenance-bound local contracts and no real model, GPU, scheduler, or network action.

**Architecture:** Keep the pipeline read-only and stage-lazy. Producer CLIs validate immutable envelopes before side effects, share frozen base-model plus LoRA semantics, and emit schema-v1 artifacts consumed by strict downstream gates.

**Tech Stack:** Python 3.12, argparse, pytest, vLLM/PEFT-compatible mocked APIs, canonical JSON/JSONL/TSV hashing.

## Global Constraints

- Write and observe a failing focused test before each production change.
- Pilot seed is exactly 42; confirmatory seeds are exactly 101, 202, 303.
- Production evidence must never accept `plumbing_only` artifacts.
- No real model, tokenizer, GPU, scheduler, or network operation.
- Preserve the untracked `artifacts/` directory.

---

### Task 1: Lazy stage commands and synthetic seed binding

**Files:** Modify `phase_marker/pipeline.py`; test `tests/phase_marker/test_pipeline.py`.

**Interfaces:** `_commands_for_stage(...) -> tuple[str, ...]` constructs only `stage`; synthetic reads and validates the preregistered seed before returning a command.

- [x] Add parametrized positive stage tests and a preregistered-seed mismatch test.
- [x] Run the focused tests and record the eager dictionary `KeyError`/mismatch RED.
- [x] Replace the command dictionary with lazy stage branches and bind the synthetic seed.
- [x] Run focused and pipeline tests GREEN.

### Task 2: Audit and analysis path contract

**Files:** Modify `phase_marker/statistics.py`, `phase_marker/pipeline.py`; test `tests/phase_marker/test_statistics.py`, `tests/phase_marker/test_pipeline.py`.

**Interfaces:** `analyze --audit-manifest PATH` is optional; absent value resolves the unique audit-bound kind sibling. Audit reads TSV labels and may publish beside them only when no output conflicts exist.

- [x] Add an exact emitted-command 300-row end-to-end test using `--generations raw-generations` and `--manual-audit audit/manual-labels.tsv`.
- [x] Run it RED and capture the incorrect directory/manifest resolution.
- [x] Implement unique-kind behavior resolution, optional audit manifest, TSV consistency, strict audit schema/hash/count validation, and nonconflicting publication.
- [x] Run command-level and statistics tests GREEN.

### Task 3: Validation-only checkpoint selection and LoRA evaluation

**Files:** Modify `phase_marker/behavior.py`, `phase_marker/pipeline.py`; test `tests/phase_marker/test_behavior.py`, `tests/phase_marker/test_pipeline.py`.

**Interfaces:** `behavior select` evaluates all declared candidates on validation only and emits a schema-v1 selection. Ranking key is strict accuracy descending, mean gold-answer logprob descending, checkpoint step ascending. `VLLMGenerationBackend` loads the frozen base revision with `enable_lora=True` and supplies one `LoRARequest` for the selected adapter.

- [x] Add selection ranking/provenance tests and mocked vLLM base/revision/LoRARequest tests.
- [x] Run them RED against the adapter-as-model implementation.
- [x] Implement `select`, adapter compatibility validation, and LoRARequest-backed generation.
- [x] Add selection commands/outputs to approval-bound pipeline jobs and require the full 6/18 manifest matrix.
- [x] Run behavior and pipeline tests GREEN.

### Task 4: Validate before side effects

**Files:** Modify `phase_marker/behavior.py`, `phase_marker/activations.py`, `phase_marker/interventions.py`; test their corresponding test modules.

**Interfaces:** Every production runner completes schema, parent, file, hash, and allowlist validation before tokenizer/model loader calls or output creation. Pair rows include `recipient_batch_hash` and `donor_batch_hash`.

- [x] Add monkeypatch tests proving loaders/writers remain untouched for stale behavior lineage, stale capture parents/batch, missing pair files, stale pair hashes, and unknown methods.
- [x] Run tests RED.
- [x] Move all validation ahead of loading/writing and reject unknown methods explicitly.
- [x] Run focused tests GREEN.

### Task 5: Documentation, report, and release verification

**Files:** Modify `README.md` and ignored Task 12 report.

**Interfaces:** Operator docs show selection, audit TSV, optional audit manifest, approval workload, and production evidence constraints.

- [x] Update README and report with the exact contracts and RED/GREEN evidence.
- [x] Run producer-focused tests, every successful stage gate, exact Task13 fixture, full offline suite, py_compile, diff checks, and dry-run nonmutation.
- [x] Stage only authorized files and create a separate fix-round-2 commit without amending prior commits.
