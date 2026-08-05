# Stage A Modal `listdir` Not-Found Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let fresh Stage A preflight treat a real Modal missing producer directory as empty while preserving fail-closed handling for every non-not-found listing error.

**Architecture:** Keep `_list_volume_files_optional` as the single optional-list compatibility boundary. Extend only its caught exception tuple, and make one existing full Stage A integration test use a test client that reproduces Modal 1.3.5's missing-path exception so the entire preflight-to-summary flow protects the contract.

**Tech Stack:** Python 3.12, pytest, Modal 1.3.5 Volume client semantics, existing Stage A adapter fakes.

## Global Constraints

- Preserve all experiment arms, commands, model revision, dependency evidence, approval semantics, GPU resources, resume behavior, spend limits, and the mandatory stop before behavior.
- Return an empty optional listing only for `FileNotFoundError` and `modal.exception.NotFoundError`.
- Continue propagating permission, authentication, conflict, data-loss, transport, service, and every other non-not-found exception.
- Leave `_read_volume_file_optional` unchanged because Modal `Volume.read_file()` already converts backend not-found errors to `FileNotFoundError`.
- Do not add broad exception normalization, clean up remote state, mutate a Modal Volume, use `--resume`, schedule a GPU, or reuse the prior smoke or Stage A approval as part of this repair.
- Use the maintained offline test boundary: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q tests`; never use bare `pytest -q`, which collects the networked top-level `infer_test.py`.

---

### Task 1: Normalize only Modal missing-directory listings

**Files:**
- Modify: `tests/phase_marker/test_modal_adapter.py:2729` (add a real-style missing-directory Stage A client)
- Modify: `tests/phase_marker/test_modal_adapter.py:3675` (exercise the complete fresh Stage A flow and non-not-found propagation)
- Modify: `modal_phase_marker.py:2278-2282` (catch the exact Modal not-found exception)
- Reference: `docs/superpowers/specs/2026-08-06-stage-a-modal-listdir-not-found-design.md`

**Interfaces:**
- Consumes: `_list_volume_files_optional(client: VolumeClient, path: str) -> tuple[object, ...]`, `StageARunsClient`, and `FakeModalNotFoundError`.
- Produces: `ModalMissingStageARunsClient`, a test-only `StageARunsClient` subclass whose `listdir(path: str, *, recursive: bool = False) -> list[SimpleNamespace]` raises `FakeModalNotFoundError` only when the inherited stored-file listing is empty.
- Preserves: all production signatures, `_read_volume_file_optional`, namespace validation, output/recovery validation, tag ordering, GPU mapping, finalization, and fresh/resume semantics.

- [ ] **Step 1: Add the real-style Stage A listing client**

Add this helper immediately after `StageARunsClient` and before `DirectoryListingStageARunsClient`:

```python
class ModalMissingStageARunsClient(StageARunsClient):
    """Stage A client with Modal's missing-directory listing behavior."""

    def listdir(
        self, path: str, *, recursive: bool = False,
    ) -> list[SimpleNamespace]:
        entries = super().listdir(path, recursive=recursive)
        if not entries:
            raise FakeModalNotFoundError("No such file or directory")
        return entries
```

This helper must not replace or globally change `RecordingVolume`; other tests retain their existing fake contracts.

- [ ] **Step 2: Make the complete fresh Stage A regression reproduce Modal's exception**

Rename:

```python
def test_stage_a_validates_all_training_before_selection_and_stops(...):
```

to:

```python
def test_stage_a_with_modal_missing_producer_roots_validates_and_stops(...):
```

In that test, replace:

```python
    runs = StageARunsClient({}, events)
```

with:

```python
    runs = ModalMissingStageARunsClient({}, events)
```

Keep all existing assertions. They already prove six training calls, six selection calls, one finalizer call, tags before the first GPU-map call, twelve total map calls, no behavior call, ordered canonical validation, and a summary with `stopped_before_behavior is True`.

- [ ] **Step 3: Add a fail-closed non-not-found regression**

Add this focused test beside the renamed Stage A regression:

```python
def test_optional_volume_listing_propagates_non_not_found(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if service or permission failures were treated as empty state."""

    class PermissionDeniedVolume:
        def listdir(
            self, path: str, *, recursive: bool = False,
        ) -> list[SimpleNamespace]:
            assert path == "/runs/approved"
            assert recursive is True
            raise PermissionError("listing denied")

    with pytest.raises(PermissionError, match="listing denied"):
        imported_adapter._list_volume_files_optional(
            PermissionDeniedVolume(), "/runs/approved"
        )
```

This test deliberately supplies no `read_file` method because the unit under test must perform only a listing.

- [ ] **Step 4: Run the focused tests and confirm the integration case fails for the intended reason**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py \
  -k 'modal_missing_producer_roots or optional_volume_listing_propagates_non_not_found'
```

Expected before the production change: the Stage A integration case fails at the first absent producer listing with `FakeModalNotFoundError: No such file or directory`; the `PermissionError` propagation case passes. No production edit may occur until this RED result is captured.

- [ ] **Step 5: Extend only the optional-list not-found tuple**

Change `_list_volume_files_optional` to:

```python
def _list_volume_files_optional(client: VolumeClient, path: str) -> tuple[object, ...]:
    try:
        return tuple(client.listdir(path, recursive=True))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return ()
```

Do not modify `_read_volume_file_optional` or catch `Exception`, `modal.exception.Error`, or any other Modal exception base class.

- [ ] **Step 6: Run the focused tests and confirm both pass**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py \
  -k 'modal_missing_producer_roots or optional_volume_listing_propagates_non_not_found'
```

Expected: `2 passed`, with no collection error or warning.

- [ ] **Step 7: Run the complete adapter suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py
```

Expected: all adapter tests pass.

- [ ] **Step 8: Run the full maintained offline suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q tests
```

Expected: all tests pass, with only the repository's previously known real-cache skip if it remains applicable.

- [ ] **Step 9: Inspect the patch for exact scope and failure ordering**

Run:

```bash
git diff --check
git diff -- modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
```

Confirm the diff contains only the test client, renamed/instrumented integration regression, fail-closed regression, and exact two-exception catch. Confirm `_read_volume_file_optional`, application tags, remote calls, resources, approval logic, and resume logic are unchanged.

- [ ] **Step 10: Commit the focused repair**

Run:

```bash
git add modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
git commit -m "fix: handle Modal missing producer listings"
```

- [ ] **Step 11: Obtain task and whole-branch reviews before relaunch**

Give each reviewer the approved design, this plan, the implementation report with exact RED/GREEN/full-suite output, and the complete implementation diff. Require checks for exact exception scope, fail-closed non-not-found behavior, test realism, unchanged read behavior, unchanged side-effect ordering, and absence of launch/approval/resource/resume changes. Resolve every Critical or Important finding before continuing.

- [ ] **Step 12: Derive new identities and stop at the approval boundary**

After review and integration, derive the new source hash, plan digest, run ID, and exact CPU smoke action locally. Confirm they differ from source `add344975af3...` and its invalidated approvals. Do not invoke Modal until the user explicitly approves the newly rendered CPU smoke command. After independently validating that receipt, derive a new fresh Stage A action with the reviewed model-cache artifact and request separate approval for that exact GPU command.
