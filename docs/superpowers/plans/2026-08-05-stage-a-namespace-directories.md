# Stage A Namespace Directory Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept Modal's explicit directory entries for the three approved Stage A mutable/evidence namespaces while preserving fail-closed rejection of every other run-namespace path.

**Architecture:** Keep `_preflight_stage_a_namespace` as the single canonical-path gate. Represent each approved ignored namespace as an exact normalized root, then accept only that root or a slash-delimited descendant; extend the adapter test fixture locally so tests can reproduce Modal's mixture of directory and file entries without changing the default fake-volume contract.

**Tech Stack:** Python 3.11, pytest, Modal Volume-compatible `listdir` entries, existing phase-marker plan and adapter helpers.

## Global Constraints

- Preserve all experiment arms, commands, model revision, dependency evidence, approval semantics, GPU resources, resume behavior, budget, and the mandatory stop before behavior.
- Permit exactly `/runs/<run-id>/attempts`, `/runs/<run-id>/receipts/attempts`, and `/runs/<run-id>/receipts/smoke`, plus slash-delimited descendants of those roots.
- Normalize every listed path to one leading slash before comparing it.
- Do not trust or ignore a path merely because its entry type is `directory`.
- Continue rejecting unrecognized files and directories, including siblings beneath `receipts` and unexpected top-level run namespaces.
- Do not clean up remote state, mutate a volume, use `--resume`, schedule a GPU, or reuse the prior smoke or Stage A approval as part of this repair.

---

### Task 1: Make Stage A namespace preflight compatible with explicit directory entries

**Files:**
- Modify: `tests/phase_marker/test_modal_adapter.py:2729` (add a real-style Stage A listing fake)
- Modify: `tests/phase_marker/test_modal_adapter.py:4667` (add focused accepted and rejected namespace tests)
- Modify: `modal_phase_marker.py:1197-1245` (replace trailing-slash prefixes with exact approved roots)
- Reference: `docs/superpowers/specs/2026-08-05-stage-a-namespace-directories-design.md`

**Interfaces:**
- Consumes: `_preflight_stage_a_namespace(plan: PilotPlan, *, resume: bool, runs_client: VolumeClient) -> bytes | None` and `StageARunsClient(files: dict[str, bytes], events: list[tuple[object, ...]])`.
- Produces: `DirectoryListingStageARunsClient`, a test-only `StageARunsClient` subclass whose `listdir(path: str, *, recursive: bool = False) -> list[SimpleNamespace]` returns explicitly supplied directory entries alongside stored file entries.
- Preserves: `_preflight_stage_a_namespace`'s signature, summary behavior, expected canonical-path/ancestor logic, and exception behavior.

- [ ] **Step 1: Add a test-only fake that reproduces Modal's explicit directory entries**

Add this helper immediately after `StageARunsClient`:

```python
class DirectoryListingStageARunsClient(StageARunsClient):
    """Stage A client that includes explicit Modal-style directory entries."""

    def __init__(
        self,
        files: dict[str, bytes],
        events: list[tuple[object, ...]],
        *,
        directory_paths: tuple[str, ...],
    ) -> None:
        super().__init__(files, events)
        self.directory_paths = directory_paths

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        file_entries = super().listdir(path, recursive=recursive)
        prefix = path.rstrip("/") + "/"
        directory_entries = [
            SimpleNamespace(path=remote_path, type="directory")
            for remote_path in self.directory_paths
            if remote_path == path or remote_path.startswith(prefix)
        ]
        return sorted(
            [*directory_entries, *file_entries],
            key=lambda entry: (entry.path, entry.type),
        )
```

- [ ] **Step 2: Write acceptance tests for each approved exact root and descendant**

Add this parameterized regression beside the existing fresh-run namespace tests. It deliberately combines an explicit directory entry with an existing file under `receipts/smoke`, proving that entry type is not the acceptance criterion:

```python
@pytest.mark.parametrize(
    "relative_path",
    (
        "attempts",
        "attempts/train-attempt",
        "receipts/attempts",
        "receipts/attempts/train-attempt.json",
        "receipts/smoke",
        "receipts/smoke/archive",
    ),
)
def test_stage_a_namespace_accepts_approved_modal_directory_entries(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    relative_path: str,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    run_root = f"/runs/{plan.run_id}"
    _cache_files, _cache_manifest, smoke = _stage_a_test_dependency_evidence(plan)
    smoke_path = f"{run_root}/receipts/smoke/{smoke['artifact_id']}.json"
    runs = DirectoryListingStageARunsClient(
        {smoke_path: (canonical_json(smoke) + "\n").encode("utf-8")},
        [],
        directory_paths=(f"{run_root}/{relative_path}",),
    )

    assert imported_adapter._preflight_stage_a_namespace(
        plan, resume=False, runs_client=runs
    ) is None
```

- [ ] **Step 3: Write rejection tests for unapproved directory siblings**

Add this parameterized fail-closed regression beside the acceptance test:

```python
@pytest.mark.parametrize("relative_path", ("receipts/unexpected", "unexpected"))
def test_stage_a_namespace_rejects_unapproved_modal_directory_entries(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    relative_path: str,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    run_root = f"/runs/{plan.run_id}"
    runs = DirectoryListingStageARunsClient(
        {},
        [],
        directory_paths=(f"{run_root}/{relative_path}",),
    )

    with pytest.raises(
        ValueError,
        match=rf"unexpected canonical Stage A path: .*/{re.escape(relative_path)}$",
    ):
        imported_adapter._preflight_stage_a_namespace(
            plan, resume=False, runs_client=runs
        )
```

- [ ] **Step 4: Run the focused tests and confirm the new acceptance cases fail for the intended reason**

Run:

```bash
pytest -q tests/phase_marker/test_modal_adapter.py \
  -k 'stage_a_namespace_accepts_approved_modal_directory_entries or stage_a_namespace_rejects_unapproved_modal_directory_entries'
```

Expected before the production change: the three exact-root acceptance cases fail with `ValueError: unexpected canonical Stage A path`; descendant acceptance cases and both rejection cases pass. This mixed RED result is required because the existing trailing-slash implementation already accepts descendants and rejects unapproved siblings.

- [ ] **Step 5: Replace trailing-slash prefixes with exact approved roots**

In `_preflight_stage_a_namespace`, replace `ignored_prefixes` and its predicate with:

```python
    ignored_roots = (
        f"{run_root}/attempts",
        f"{run_root}/receipts/attempts",
        f"{run_root}/receipts/smoke",
    )
```

Then, after the existing one-leading-slash normalization, use:

```python
        if path == run_root or any(
            path == root or path.startswith(root + "/")
            for root in ignored_roots
        ):
            continue
```

Do not change the `expected` set, ancestor/descendant comparison, summary handling, or error paths.

- [ ] **Step 6: Run the focused tests and confirm all eight cases pass**

Run:

```bash
pytest -q tests/phase_marker/test_modal_adapter.py \
  -k 'stage_a_namespace_accepts_approved_modal_directory_entries or stage_a_namespace_rejects_unapproved_modal_directory_entries'
```

Expected: `8 passed`, with no deselected test failing during collection.

- [ ] **Step 7: Run the complete adapter suite**

Run:

```bash
pytest -q tests/phase_marker/test_modal_adapter.py
```

Expected: all adapter tests pass.

- [ ] **Step 8: Run the full offline suite**

Run:

```bash
pytest -q
```

Expected: all tests pass, with only the repository's previously known skip if it remains applicable.

- [ ] **Step 9: Inspect the patch for scope and policy preservation**

Run:

```bash
git diff --check
git diff -- modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
```

Confirm the diff contains only the local listing fake, the eight regression cases, and the exact-root predicate repair. Confirm there is no experiment, approval, resource, resume, volume-mutation, or launch-command change.

- [ ] **Step 10: Commit the focused repair**

Run:

```bash
git add modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
git commit -m "fix: accept approved Stage A directory roots"
```

- [ ] **Step 11: Obtain an independent two-stage review before any Modal relaunch**

Give the reviewer the design, this plan, and the repair commit. Require the reviewer to check first that all approved and rejected path cases match the spec, then check code quality and test isolation. If review finds an issue, add a failing regression where applicable, make the smallest correction, rerun Steps 6 through 9, and commit that correction separately.

- [ ] **Step 12: Derive fresh launch identities and stop at the approval boundary**

Using the repository's existing planning/status commands, derive and report the new source hash, plan digest, run ID, and exact CPU smoke action digest from the reviewed commit. Confirm they differ from the invalidated identities. Do not invoke Modal until the user explicitly approves the newly rendered smoke command; after a validated smoke receipt exists, derive a new Stage A action digest and request separate approval for that exact GPU command.
