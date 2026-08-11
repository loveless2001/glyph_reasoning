# Codex Session Audit — Worktrees, Modal Run State, Garbage Marks

Date: 2026-08-06 | Auditor: Claude | Scope: `.worktrees/`, Modal volume `phase-marker-pilot-runs-v1`, untracked dirs

## Modal experiment status: TRAINING DONE, SELECTION/FINALIZE NOT RUN

Current-source run namespace (matches `main` @ 6a0d2ca, verified via `phase_marker.modal_plan run-id`):

```
pilot-s42-cfg-f112fda5-split-3b15bcc3-src-8ada607e4c2e-plan-d783c5c1...
```

- All 6 arm train receipts present (`receipts/canonical/train/{semantic,glyph,dot,random,direct,filler}.json`), each `exit_status: 0`, `validated: true`, `promoted: true`, H100, ~8.6 min/arm, finished 2026-08-05 ~18:57 UTC.
- Checkpoints present for all 6 arms under `artifacts/phase-marker/checkpoints/pilot/seed-42/`.
- Smoke receipt present.
- **Missing:** `receipts/canonical/selection/` and `stage-a-summary.json` — checkpoint selection + finalizer never ran (quota exhausted). Stage A is resumable: canonical train pairs are complete, resume should skip training and run selection only.

**CRITICAL ORDERING:** run identity is source-bound. Merging `cleanup/research-test-suite` changes the src hash → new run id → all 6 arms would retrain. **Resume/finish Stage A from `main`'s current code BEFORE merging the cleanup branch.**

## Cleanup branch (`.worktrees/research-test-cleanup`, branch `cleanup/research-test-suite`)

6 commits ahead of main, 0 behind, tree clean. NOT garbage — real work. -13,252/+619 lines (deletes 5,727-line `test_modal_adapter.py`, 3,437-line `test_modal_artifacts.py`, etc.; replaces with 269-line `test_modal.py`).

Plan `docs/superpowers/plans/2026-08-06-research-test-cleanup.md` task status:
- Task 1 (remove transient repair plans): DONE (d6c4751)
- Task 2 (compact Modal suite): DONE (01a7ce1)
- Task 3 (simplify Modal harness): DONE (7084d80)
- Task 4 (trim scientific suite: test_pipeline/test_behavior/test_materialize): **NOT DONE** — no commit
- Task 5 (README rewrite): DONE (5190b64)
- Task 6 (verification): partially verified now: `py_compile` all modules OK; offline scientific suite **323 passed**. Compact Modal boundary (`test_modal.py`) not yet run — needs env with `modal` installed (system python3 import failed despite pip metadata; likely broken install, re-verify).

## GARBAGE — marked for cleanup (not yet deleted)

### Local worktrees (branches fully merged into main, 0 ahead)
1. `.worktrees/stage-a-cpu-preflight` + branch `fix/stage-a-cpu-preflight` — includes untracked `artifacts/` (21M) byte-identical to main's `artifacts/` (verified `diff -rq`)
2. `.worktrees/stage-a-modal-listdir-not-found` + branch `fix/stage-a-modal-listdir-not-found`
3. `.worktrees/stage-a-namespace-directories` + branch `fix/stage-a-namespace-directories`

Removal (after user confirm):
```bash
git worktree remove --force .worktrees/stage-a-cpu-preflight
git worktree remove .worktrees/stage-a-modal-listdir-not-found
git worktree remove .worktrees/stage-a-namespace-directories
git branch -d fix/stage-a-cpu-preflight fix/stage-a-modal-listdir-not-found fix/stage-a-namespace-directories
```

### Stale Modal run namespaces (9 of 10, superseded source hashes; partial = failed per protocol)
Under `phase-marker-pilot-runs-v1:/runs/`, all `pilot-s42-cfg-f112fda5-split-3b15bcc3-*` EXCEPT `src-8ada607e4c2e-plan-d783c5c1...`:
`src-5538744c83b5`, `src-6480d1d564cf`, `src-00d65fadcd55`, `src-838e16bcb57c`, `src-7f3927853290`, `src-add344975af3`, `src-fe4da7f8e798`, `src-8dfb425024a3`, `src-dc9267a1d49f`

Some contain partial `artifacts/` (checkpoint bytes — likely the bulk of volume storage). Delete via `modal volume rm phase-marker-pilot-runs-v1 /runs/<name> -r` after user confirm. **Do NOT delete `src-8ada607e4c2e...` — it holds the completed training.**

### Keep (explicitly not garbage)
- Untracked on main: `artifacts/` (input bundle for current run — hashes match train receipts), `model_cards/`, `paper/` — cleanup plan mandates preserving these.
- `.worktrees/research-test-cleanup` — unmerged real work.

## Recommended sequence
1. Resume Stage A (selection + finalize) from main's current code; confirm `stage-a-summary.json` lands.
2. Finish Task 4 + run compact Modal boundary test in an env with working `modal`.
3. Merge `cleanup/research-test-suite`.
4. Execute garbage removal (worktrees, branches, stale volume namespaces).

## Unresolved questions
- System python3 `import modal` fails though pip metadata shows modal 1.3.5 — which env did codex use to launch? (`~/venvs/` exists, uninspected.)
- Whether stale namespaces' partial checkpoints have any salvage value (protocol says no — partial = failed).
