# Phase Marker Pipeline Fix Round 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Close the five fourth-review findings with independently replayable selection evidence, canonical behavior membership, scope-separated approvals, and shared activation validation.

**Architecture:** Persist raw evidence rather than trusted aggregates, and make every consumer replay the scorer/token accounting. Separate experiment and mechanism authorization types so no approval can cross scopes; share one activation tensor validator between capture artifacts and intervention preflight.

**Tech Stack:** Python 3.12, pytest, PyTorch safe loading, PEFT CPU integration, canonical JSON/JSONL hashing.

## Global Constraints

- RED before production changes; rerun focused tests after each slice.
- No large checkpoint, GPU, scheduler, or network action.
- Preserve the pre-existing untracked `artifacts/` directory.
- Create a separate commit; never amend prior Task 12 commits.

---

### Task 1: Replayable atomic selector evidence

**Files:** Modify `phase_marker/behavior.py`, tests in `test_behavior.py` and `test_pipeline.py`.

- [x] Add forged scorer/token accounting, duplicate/missing evidence, and partial-output tests; run RED.
- [x] Persist canonical example, raw completion, complete scorer result, ordered gold tokens/pieces/logprobs, and tokenizer binding.
- [x] Atomically publish manifest/evidence as one absent directory transaction and replay all evidence in consumers.
- [x] Run behavior/pipeline tests GREEN.

### Task 2: Canonical test membership and multiplicity

**Files:** Modify `phase_marker/pipeline.py`, `phase_marker/statistics.py`; test both modules.

- [x] Add self-hashed validation/operator substitute and record multiplicity tests; run RED.
- [x] Recompute canonical split/test bytes and exact example/question/checkpoint/cell/completion multiplicities.
- [x] Run behavior-parent tests GREEN.

### Task 3: Separate mechanism approval

**Files:** Modify `phase_marker/pipeline.py`, `README.md`; test pipeline.

- [x] Add cross-scope and count/hash/output mismatch tests; run RED.
- [x] Add frozen `MechanismApprovalMetadata` and require it only for capture/intervene.
- [x] Keep general approval explicitly mechanism-excluded and planning output not ready without mechanism approval.
- [x] Run pipeline tests GREEN.

### Task 4: Shared activation semantic validation

**Files:** Modify `phase_marker/activations.py`, `phase_marker/interventions.py`; test both.

- [x] Add scalar, empty, nonfinite, wrong-shape/name/dtype tests and run RED.
- [x] Expose one canonical safe activation artifact validator and reuse it before intervention model imports.
- [x] Run activation/intervention tests GREEN.

### Task 5: Output accounting and release verification

**Files:** Modify `phase_marker/pipeline.py`, `README.md`, ignored Task 12 report.

- [x] Include selection manifest and evidence JSONL in every expected-output contract.
- [x] Update docs/report with exactly five round-4 findings.
- [x] Run every requested focused/integration/full/static/nonmutation check.
- [x] Stage authorized files and commit separately.
