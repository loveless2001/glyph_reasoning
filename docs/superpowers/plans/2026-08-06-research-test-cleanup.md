# Research Test and Harness Cleanup Implementation Plan

> **For agentic workers:** Execute inline in the isolated cleanup worktree. Do
> not dispatch subagents. This is a deletion/simplification pass, not TDD; the
> user explicitly requested that product-development ceremony be removed.

**Goal:** Preserve scientific validity checks while removing the product-style
Modal test and safety-harness bloat.

**Architecture:** Keep content-addressed experiment plans, scientific manifests,
canonical receipts, and ordinary immutable publication. Replace exhaustive
adversarial filesystem/concurrency handling with a single-operator workflow
using standard Python filesystem operations and a compact integration boundary.

**Tech Stack:** Python 3.12, pytest, Modal 1.3.5, pathlib, shutil, JSON, SHA-256.

## Global constraints

- Work only in `.worktrees/research-test-cleanup` on
  `cleanup/research-test-suite` until the live source-bound run finishes.
- Do not change scoring, splits, training, behavior, statistics, activation, or
  intervention semantics.
- Do not run network, GPU, model-download, or external Modal actions.
- Do not add replacement tests for deleted implementation-private branches.
- Preserve unrelated untracked `artifacts/`, `model_cards/`, and `paper/` on
  `main`.

---

### Task 1: Remove transient process documentation

**Files:**
- Delete: `docs/superpowers/plans/2026-08-04-phase-marker-pipeline-fix-round-2.md`
- Delete: `docs/superpowers/plans/2026-08-04-phase-marker-pipeline-fix-round-3.md`
- Delete: `docs/superpowers/plans/2026-08-04-phase-marker-pipeline-fix-round-4.md`
- Delete: `docs/superpowers/plans/2026-08-05-stage-a-cpu-preflight.md`
- Delete: `docs/superpowers/plans/2026-08-05-stage-a-namespace-directories.md`
- Delete: `docs/superpowers/plans/2026-08-06-stage-a-modal-listdir-not-found.md`
- Delete: corresponding superseded repair specs under `docs/superpowers/specs/`
- Keep: original mechanism design, Modal pilot design/plan, and this cleanup
  design/plan.

- [ ] Delete the listed repair-history documents with `apply_patch`.
- [ ] Confirm retained docs contain no links to deleted repair plans with
  `rg -n 'pipeline-fix-round|stage-a-cpu-preflight|stage-a-namespace-directories|stage-a-modal-listdir' docs README.md`.
- [ ] Commit as `docs: remove transient repair plans`.

### Task 2: Replace the Modal test gauntlet with a compact boundary

**Files:**
- Rewrite: `tests/phase_marker/test_modal_plan.py`
- Rewrite: `tests/phase_marker/test_modal_artifacts.py`
- Rewrite: `tests/phase_marker/test_modal_adapter.py`

**Retained cases:**

1. `test_plan_is_deterministic_and_binds_source_inputs_and_budget`
2. `test_stage_a_action_binds_smoke_cache_and_resume_mode`
3. `test_input_bundle_detects_changed_bytes`
4. `test_nested_checkpoint_tree_promotes_without_overwrite`
5. `test_symlink_or_special_output_is_rejected`
6. `test_cpu_smoke_validates_bundle_cache_and_source`
7. `test_stage_a_fresh_path_runs_training_selection_then_finalizes`
8. `test_stage_a_failure_preserves_completed_receipts_for_resume`
9. `test_modal_functions_request_two_h100_workers_and_cpu_finalizer`

- [ ] Extract only fixtures needed by these nine cases; remove fake classes and
  mutation helpers not used by them.
- [ ] Rewrite the three files so each retained case tests an external research
  contract, not private helper call order.
- [ ] Confirm the compact Modal suite collects exactly nine tests.
- [ ] Commit as `test: reduce Modal suite to research contracts`.

### Task 3: Simplify Stage A orchestration and recovery

**Files:**
- Modify: `modal_phase_marker.py`
- Modify: `phase_marker/modal_artifacts.py`

**Interfaces retained:**

- `run_stage_a_local(...) -> dict[str, object]`
- `execute_pilot_job(...) -> dict[str, object]`
- `promote_validated_output(...) -> Path`
- `validate_job_receipt_payload(...) -> dict[str, object]`
- `validate_stage_a_summary(...) -> dict[str, object]`

- [ ] Remove authenticated lease expiry, inode snapshot, quarantine-tree, and
  recovery RPC machinery (`_RecoveryStat` through `_recover_stage_a_orphans`,
  recovery payload validation, and the recovery Modal function).
- [ ] Make resume accept only complete canonical producer/receipt pairs; report
  partial output plainly and require a new run identity instead of mutating
  orphan namespaces.
- [ ] Replace custom `renameat2`/inode verification with destination existence
  checks plus `os.rename` under the single-operator contract.
- [ ] Replace rollback exception hierarchies and multi-phase cleanup hooks with
  direct failure receipts and never-overwrite checks.
- [ ] Remove imports, dataclasses, constants, and helpers left unreferenced by
  the simplified path.
- [ ] Compile both modules and run the nine-test compact Modal boundary.
- [ ] Commit as `refactor: simplify research Modal harness`.

### Task 4: Retain the scientific suite and remove redundant plumbing cases

**Files:**
- Review and trim: `tests/phase_marker/test_pipeline.py`
- Review and trim: `tests/phase_marker/test_behavior.py`
- Review and trim: `tests/phase_marker/test_materialize.py`
- Keep focused modules for scoring, prompts, splits, training, statistics,
  activations, interventions, synthetic data, token audit, and traces.

- [ ] Remove cases that only duplicate schema-type rejection already covered by
  their owning module or assert exact internal call order.
- [ ] Keep at least one positive and one scientifically meaningful rejection
  case for each retained research stage.
- [ ] Collect the retained suite and record per-file counts in the commit body.
- [ ] Commit as `test: focus suite on scientific validity`.

### Task 5: Make the repository describe the actual workflow

**Files:**
- Modify: `README.md`

- [ ] Replace the approval-gate narrative with a concise workflow: materialize
  data, run the six Stage A arms, select checkpoints on validation, score held-out
  behavior, then run mechanism analyses separately.
- [ ] Document the compact verification command and clarify that tests are
  offline research checks, not proof that Modal/GPU execution works.
- [ ] Remove references to deleted test ceremony and superseded commands.
- [ ] Commit as `docs: describe the research workflow`.

### Task 6: Verify the cleanup without rebuilding the gauntlet

**Files:** none unless verification finds a direct cleanup defect.

- [ ] Run `python -m py_compile` on all `phase_marker/*.py` files and the two
  Modal entrypoints.
- [ ] Run the compact Modal boundary.
- [ ] Run retained scientific tests offline.
- [ ] Run `python -m phase_marker.modal_plan run-id` and `plan` locally and
  confirm both remain side-effect free.
- [ ] Run `git diff --check`, inspect line/test-count reductions, and confirm
  `git status --short` contains only intended cleanup changes.
- [ ] Commit any direct verification corrections separately; do not add broad
  regression tests.
