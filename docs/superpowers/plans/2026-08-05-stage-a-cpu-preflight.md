# Stage A CPU Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Stage A's full dependency revalidation from the memory-constrained local Modal coordinator into a CPU-only mounted-volume function that completes before any GPU scheduling.

**Architecture:** Declare a read-only CPU preflight function in the existing Stage A app and reuse `validate_stage_a_remote_dependencies`, whose model-cache hashes are streamed in 1 MiB chunks. Return a compact evidence object to the local coordinator, validate it as untrusted data, then retain the existing output inspection, recovery, training, selection, and finalization graph.

**Tech Stack:** Python 3.12, dataclasses, hashlib/JSON/pathlib, pytest, Modal 1.3.5, Modal Volumes.

## Global Constraints

- Preserve the six arms, seed `42`, exact experiment commands, Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`, concurrency, four-hour GPU timeouts, and USD 1,000 operator-acknowledged ceiling.
- Preserve the mandatory stop after Stage A; do not launch behavior or mechanism stages.
- Perform complete bundle, model-cache, run-provenance, and smoke validation before requesting an H100 or accepting an H200 upgrade.
- The dependency preflight function is CPU-only, has `retries=0`, and mounts input, model, and run volumes read-only.
- Model-cache hashes must use the existing 1 MiB streaming implementation; no local fallback may download or join model shards.
- Treat remote evidence as untrusted and fail closed on missing fields, extra fields, type errors, or identity mismatches.
- Validate the operator's run ID, budget acknowledgement, plan digest, action digest, smoke receipt ID, and model-cache ID before the CPU remote call.
- Do not mutate a volume, create a lease, write a receipt, load model weights, apply app tags, or schedule a GPU during dependency preflight.
- Preserve initial-launch and explicit-resume semantics and never overwrite canonical output.
- Preserve untracked `artifacts/`, `model_cards/`, `paper/`, and all unrelated user changes.
- Do not run a Modal command while implementing this plan. A changed source digest requires a newly approved CPU smoke and a separately approved Stage A command after implementation review.
- No Git push, PR, publication, checkpoint upload, or later experiment stage is authorized.

## File Responsibility Map

- Modify `modal_phase_marker.py`: declare the CPU preflight resources and wrapper, serialize compact dependency evidence, validate returned evidence, and rewire `run_stage_a_local` plus the `run-stage-a` entrypoint.
- Modify `tests/phase_marker/test_modal_adapter.py`: test the CPU-only declaration, mounted-path wrapper, strict evidence validation, call ordering, failure behavior, and unchanged Stage A orchestration.
- Reuse `phase_marker/modal_artifacts.py` unchanged: `validate_stage_a_remote_dependencies` remains the single mounted-path validator and `_file_sha256` remains the streaming hash implementation.

---

### Task 1: Declare the CPU Preflight and Its Compact Contract

**Files:**
- Modify: `modal_phase_marker.py:40-430`
- Test: `tests/phase_marker/test_modal_adapter.py:600-1200`

**Interfaces:**
- Consumes: `validate_stage_a_remote_dependencies(*, plan_payload, approval_payload, input_root, model_root, run_root) -> tuple[InputBundle, ModelCacheManifest, dict[str, object]]`.
- Produces: `preflight_stage_a_remote(remote_payload: Mapping[str, object]) -> dict[str, object]` with exactly `schema_version`, `run_id`, `bundle_id`, `bundle_manifest_artifact_id`, `model_cache_artifact_id`, `smoke_receipt_artifact_id`, and `smoke_receipt`.
- Produces: `PREFLIGHT_VOLUMES`, with `/inputs`, `/model-cache`, and `/runs` all read-only.

- [ ] **Step 1: Extend the fake-Modal declaration test and write a failing wrapper test**

Add `preflight_stage_a_remote` to the exact Stage A function set asserted by
`test_modal_graph_is_dedicated_and_bounded`, add the preflight declaration
assertions to `test_stage_a_job_resources_and_mount_permissions_are_exact`,
and add this focused wrapper test near the other CPU wrapper tests:

```python
def test_cpu_dependency_preflight_wrapper_returns_compact_evidence(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(
        pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    )
    plan_payload = modal_plan.pilot_plan_payload(plan)
    _files, manifest_payload, smoke = _stage_a_test_dependency_evidence(plan)
    manifest = modal_artifacts.parse_model_cache_manifest_payload(manifest_payload)
    bundle = build_input_bundle(pilot_repo)
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id=str(smoke["artifact_id"]),
        model_cache_artifact_id=manifest.artifact_id,
    )
    calls: list[dict[str, object]] = []

    def validate(**kwargs: object) -> tuple[object, object, dict[str, object]]:
        calls.append(dict(kwargs))
        return bundle, manifest, dict(smoke)

    monkeypatch.setattr(
        imported_adapter, "validate_stage_a_remote_dependencies", validate
    )

    result = imported_adapter.preflight_stage_a_remote.local(
        {"plan": plan_payload, "approval": approval}
    )

    assert set(result) == {
        "schema_version",
        "run_id",
        "bundle_id",
        "bundle_manifest_artifact_id",
        "model_cache_artifact_id",
        "smoke_receipt_artifact_id",
        "smoke_receipt",
    }
    assert result["schema_version"] == 1
    assert result["run_id"] == plan.run_id
    assert result["bundle_id"] == plan.bundle_id
    assert result["bundle_manifest_artifact_id"] == plan.bundle_manifest_artifact_id
    assert result["model_cache_artifact_id"] == manifest.artifact_id
    assert result["smoke_receipt_artifact_id"] == smoke["artifact_id"]
    assert result["smoke_receipt"] == smoke
    assert calls == [{
        "plan_payload": plan_payload,
        "approval_payload": approval,
        "input_root": Path("/inputs"),
        "model_root": Path("/model-cache"),
        "run_root": Path("/runs"),
    }]
```

In the fake-Modal resource assertions, require:

```python
preflight = declarations["preflight_stage_a_remote"]
assert "gpu" not in preflight
assert preflight["cpu"] == 2.0
assert preflight["memory"] == 8_192
assert preflight["timeout"] == 7_200
assert preflight["retries"] == 0
assert preflight["volumes"]["/inputs"].read_only is True
assert preflight["volumes"]["/model-cache"].read_only is True
assert preflight["volumes"]["/runs"].read_only is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py::test_modal_graph_is_dedicated_and_bounded \
  tests/phase_marker/test_modal_adapter.py::test_stage_a_job_resources_and_mount_permissions_are_exact \
  tests/phase_marker/test_modal_adapter.py::test_cpu_dependency_preflight_wrapper_returns_compact_evidence
```

Expected: FAIL because `preflight_stage_a_remote` is not declared and absent from the Stage A app's remote-function set.

- [ ] **Step 3: Add the minimal CPU resource declaration and wrapper**

Add constants and a dedicated read-only mount map:

```python
STAGE_A_PREFLIGHT_CPU = 2.0
STAGE_A_PREFLIGHT_MEMORY_MIB = 8_192
STAGE_A_PREFLIGHT_TIMEOUT_SECONDS = 7_200

PREFLIGHT_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
    "/runs": runs_volume.read_only(),
}
```

Declare the function before the GPU functions:

```python
@app.function(
    image=cpu_image,
    cpu=STAGE_A_PREFLIGHT_CPU,
    memory=STAGE_A_PREFLIGHT_MEMORY_MIB,
    timeout=STAGE_A_PREFLIGHT_TIMEOUT_SECONDS,
    retries=0,
    volumes=PREFLIGHT_VOLUMES,
)
def preflight_stage_a_remote(
    remote_payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(remote_payload, Mapping) or set(remote_payload) != {
        "plan", "approval"
    }:
        raise ValueError("remote Stage A preflight payload fields are invalid")
    plan_payload = remote_payload["plan"]
    approval_payload = remote_payload["approval"]
    if not isinstance(plan_payload, Mapping) or not isinstance(
        approval_payload, Mapping
    ):
        raise ValueError("remote Stage A preflight payload values are invalid")
    bundle, manifest, smoke = validate_stage_a_remote_dependencies(
        plan_payload=plan_payload,
        approval_payload=approval_payload,
        input_root=JOB_INPUT_MOUNT_ROOT,
        model_root=JOB_MODEL_MOUNT_ROOT,
        run_root=JOB_RUN_MOUNT_ROOT,
    )
    return {
        "schema_version": 1,
        "run_id": plan_payload["run_id"],
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_artifact_id": plan_payload[
            "bundle_manifest_artifact_id"
        ],
        "model_cache_artifact_id": manifest.artifact_id,
        "smoke_receipt_artifact_id": smoke["artifact_id"],
        "smoke_receipt": smoke,
    }
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2.

Expected: both tests PASS; the declaration contains no `gpu` key and all three mounts are read-only.

- [ ] **Step 5: Run the complete adapter suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py
```

Expected: PASS with no Modal RPC because all adapter tests use the fake Modal module.

- [ ] **Step 6: Commit the isolated remote contract**

```bash
git add modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
git diff --cached --check
git commit -m "feat: add CPU Stage A dependency preflight"
```

---

### Task 2: Rewire the Local Coordinator and Remove Local Model Reads

**Files:**
- Modify: `modal_phase_marker.py:590-930`
- Modify: `modal_phase_marker.py:2620-2700`
- Test: `tests/phase_marker/test_modal_adapter.py:2460-2800`
- Test: `tests/phase_marker/test_modal_adapter.py:3420-4440`

**Interfaces:**
- Consumes: `dependency_function: RemoteFunction`, invoked once as `dependency_function.remote({"plan": plan_payload, "approval": validated_approval})`.
- Produces: `_validate_stage_a_dependency_evidence(plan: PilotPlan, approval_payload: Mapping[str, object], payload: object) -> StageADependencyEvidence`.
- Changes: `run_stage_a_local` removes `inputs_client` and `model_client`, requires `dependency_function`, and keeps `runs_client` for compact run-state revalidation.

- [ ] **Step 1: Add a CPU preflight fake and convert the dependency fixture**

Add this fake beside `StageAFinalizer`:

```python
class StageADependencyFunction:
    def __init__(
        self, result: object, events: list[tuple[object, ...]] | None = None
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.calls: list[object] = []

    def remote(self, payload: object) -> object:
        self.calls.append(payload)
        self.events.append(("dependency-preflight",))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result
```

Replace the local model/input clients returned by `_stage_a_dependency_kwargs`
with a compact result and the fake callable:

```python
result = {
    "schema_version": 1,
    "run_id": plan.run_id,
    "bundle_id": plan.bundle_id,
    "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
    "model_cache_artifact_id": model_cache_artifact_id,
    "smoke_receipt_artifact_id": smoke_id,
    "smoke_receipt": smoke,
}
return {
    "dependency_function": StageADependencyFunction(result),
    "smoke_receipt_artifact_id": smoke_id,
    "model_cache_artifact_id": model_cache_artifact_id,
    "approval_payload": approval,
}
```

Keep the smoke and provenance bytes in `runs.files` so existing run-namespace
inspection tests retain realistic permitted state. Do not build or return a
local model Volume client.

- [ ] **Step 2: Write failing ordering and fail-closed tests**

Add:

```python
def test_stage_a_cpu_preflight_completes_before_tags_or_gpu(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(
        pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    )
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, events)
    kwargs = _stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False)
    dependency = kwargs["dependency_function"]
    assert isinstance(dependency, StageADependencyFunction)
    dependency.events = events
    training, training_results = _publishing_stage_a_function(
        plan, "train", events, runs
    )
    selection, selection_results = _publishing_stage_a_function(
        plan, "selection", events, runs
    )
    finalizer = StageAFinalizer(
        _stage_a_summary(plan, training_results, selection_results), events
    )
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda *_args, **_kwargs: events.append(("tags",)),
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )

    imported_adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=False,
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        runs_client=runs,
        **kwargs,
    )

    assert events.index(("dependency-preflight",)) < events.index(("tags",))
    assert events.index(("dependency-preflight",)) < events.index(
        ("train", plan.jobs[0].arm)
    )
```

Add this parameterized failure test:

```python
@pytest.mark.parametrize(
    "fault",
    (
        "exception",
        "extra-field",
        "wrong-schema-type",
        "wrong-run",
        "wrong-bundle",
        "wrong-model",
        "wrong-smoke",
    ),
)
def test_stage_a_rejects_invalid_cpu_preflight_evidence(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    plan = _build_plan(
        pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    )
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, events)
    kwargs = _stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False)
    original = kwargs["dependency_function"]
    assert isinstance(original, StageADependencyFunction)
    result: object
    if fault == "exception":
        result = RuntimeError("CPU preflight failed")
    else:
        assert isinstance(original.result, dict)
        result = dict(original.result)
        if fault == "extra-field":
            result["unexpected"] = True
        elif fault == "wrong-schema-type":
            result["schema_version"] = True
        elif fault == "wrong-run":
            result["run_id"] = "wrong-run"
        elif fault == "wrong-bundle":
            result["bundle_id"] = "0" * 64
        elif fault == "wrong-model":
            result["model_cache_artifact_id"] = "0" * 64
        else:
            result["smoke_receipt_artifact_id"] = "0" * 64
    dependency = StageADependencyFunction(result, events)
    kwargs["dependency_function"] = dependency
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )

    with pytest.raises((RuntimeError, ValueError)):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
            **kwargs,
        )

    assert len(dependency.calls) == 1
    assert tags == []
    assert training.calls == []
    assert selection.calls == []
    assert finalizer.calls == []
```

Also extend the existing mismatched-run and invalid-approval tests to assert
that `dependency_function.calls == []`, proving local approval validation
precedes the CPU RPC.

Update `test_run_stage_a_entrypoint_forwards_explicit_resume_and_prints_summary`
so its expected coordinator arguments contain:

```python
"dependency_function": imported_adapter.preflight_stage_a_remote,
```

and no longer contain `inputs_client` or `model_client`.

- [ ] **Step 3: Run the new coordinator tests and verify RED**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py::test_stage_a_cpu_preflight_completes_before_tags_or_gpu \
  tests/phase_marker/test_modal_adapter.py::test_stage_a_rejects_invalid_cpu_preflight_evidence
```

Expected: FAIL because `run_stage_a_local` still requires local input/model clients and calls `preflight_stage_a_dependencies`.

- [ ] **Step 4: Implement strict compact-evidence validation**

Add the local validator:

```python
def _validate_stage_a_dependency_evidence(
    plan: PilotPlan,
    approval_payload: Mapping[str, object],
    payload: object,
) -> StageADependencyEvidence:
    expected_fields = {
        "schema_version",
        "run_id",
        "bundle_id",
        "bundle_manifest_artifact_id",
        "model_cache_artifact_id",
        "smoke_receipt_artifact_id",
        "smoke_receipt",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("Stage A dependency preflight result is invalid")
    smoke = payload["smoke_receipt"]
    if not isinstance(smoke, Mapping):
        raise ValueError("Stage A dependency preflight result is invalid")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["run_id"] != plan.run_id
        or payload["bundle_id"] != plan.bundle_id
        or payload["bundle_manifest_artifact_id"]
        != plan.bundle_manifest_artifact_id
        or payload["model_cache_artifact_id"]
        != approval_payload["model_cache_artifact_id"]
        or payload["smoke_receipt_artifact_id"]
        != approval_payload["smoke_receipt_artifact_id"]
    ):
        raise ValueError("Stage A dependency preflight identity is invalid")
    validated_smoke = _validate_successful_smoke_receipt(
        smoke,
        plan=plan,
        artifact_id=str(payload["smoke_receipt_artifact_id"]),
        model_cache_artifact_id=str(payload["model_cache_artifact_id"]),
    )
    return StageADependencyEvidence(
        bundle_id=plan.bundle_id,
        model_cache_artifact_id=str(payload["model_cache_artifact_id"]),
        smoke_receipt_artifact_id=str(payload["smoke_receipt_artifact_id"]),
        bundle_manifest_artifact_id=plan.bundle_manifest_artifact_id,
        smoke_receipt=validated_smoke,
    )
```

Change `_validate_successful_smoke_receipt` from a `None` return to
`dict[str, object]` and append `return dict(smoke)` after its existing checks.
Update its existing success test to assert the returned copy. Do not relax any
receipt checks.

- [ ] **Step 5: Rewire `run_stage_a_local` in approval-first order**

Change its signature to require `dependency_function: RemoteFunction` and
remove `inputs_client` and `model_client`. Immediately after operator and
explicit-boolean validation:

```python
plan_payload = pilot_plan_payload(plan)
validated_approval = validate_action_approval_payload(
    plan_payload=plan_payload,
    approval_payload=approval_payload,
    action="run-stage-a",
    resume=resume,
    smoke_receipt_artifact_id=str(smoke_receipt_artifact_id),
    model_cache_artifact_id=str(model_cache_artifact_id),
)
dependency_result = dependency_function.remote(
    {"plan": plan_payload, "approval": validated_approval}
)
dependency_evidence = _validate_stage_a_dependency_evidence(
    plan, validated_approval, dependency_result
)
```

Then retain `_preflight_stage_a_outputs`, plan printing, tagging, recovery,
training, selection, and finalization in their current order. Remove the old
call to `preflight_stage_a_dependencies`; do not replace it with local reads.

In the `run-stage-a` local entrypoint, pass:

```python
dependency_function=preflight_stage_a_remote,
```

and remove `inputs_client=inputs_volume` and `model_client=model_volume`.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the command from Step 3, then run all orchestration tests containing
`stage_a`:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py -k 'stage_a'
```

Expected: PASS, with dependency preflight before tags/GPU calls and unchanged
initial/resume behavior.

- [ ] **Step 7: Remove the obsolete local model-cache preflight path**

Delete `preflight_stage_a_dependencies` and remove imports used only by that
function, including `parse_model_cache_manifest_payload` if `rg` confirms no
remaining reference in `modal_phase_marker.py`. Keep shared batch-read helpers
that are still used for compact receipts, summaries, or evidence downloads.

Delete the old local dependency-preflight tests that directly mutate a fake
model Volume. Their security coverage is replaced by:

- mounted-path validator tests in `tests/phase_marker/test_modal_artifacts.py`;
- the Task 1 wrapper contract test; and
- strict untrusted-result tests from Task 2.

Verify there is no local Stage A model read:

```bash
rg -n "preflight_stage_a_dependencies|inputs_client=inputs_volume|model_client=model_volume" \
  modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
```

Expected: no matches.

- [ ] **Step 8: Run the complete adapter suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py
```

Expected: PASS with no real Modal RPC.

- [ ] **Step 9: Commit the coordinator repair**

```bash
git add modal_phase_marker.py tests/phase_marker/test_modal_adapter.py
git diff --cached --check
git commit -m "fix: move Stage A preflight to CPU remote"
```

---

### Task 3: Verify the Repair and Prepare the Approval Boundary

**Files:**
- Verify: `modal_phase_marker.py`
- Verify: `phase_marker/modal_artifacts.py`
- Verify: `tests/phase_marker/test_modal_adapter.py`
- Verify: all `tests/`

**Interfaces:**
- Consumes: the committed Task 1 and Task 2 changes.
- Produces: offline verification evidence and a newly derived CPU smoke command; it performs no Modal execution.

- [ ] **Step 1: Verify the memory-safe hash path and CPU/GPU boundary statically**

Run:

```bash
rg -n "def preflight_stage_a_remote|validate_stage_a_remote_dependencies|def _file_sha256|gpu=|PREFLIGHT_VOLUMES" \
  modal_phase_marker.py phase_marker/modal_artifacts.py
```

Confirm from the matched definitions that the preflight wrapper has no GPU,
all preflight mounts are read-only, the mounted validator reaches
`validate_model_cache_manifest`, and `_file_sha256` reads 1 MiB chunks.

- [ ] **Step 2: Run the full offline suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q tests
```

Expected: all tests PASS; the one real-cache test may SKIP when the complete
offline Qwen snapshot is unavailable.

- [ ] **Step 3: Check repository scope**

Run:

```bash
git status --short
git log -4 --oneline --decorate
git diff --check HEAD~2..HEAD
```

Expected: only the two planned implementation commits follow the design/plan
commits; untracked `artifacts/`, `model_cards/`, and `paper/` remain untouched.

- [ ] **Step 4: Review before any external command**

Use `superpowers:requesting-code-review` to inspect the committed diff against
the approved design. Resolve every critical, important, or minor correctness
finding with a fresh failing test and a scoped commit, then rerun the full
offline suite.

- [ ] **Step 5: Derive the new approval identities offline**

Run the repository's existing offline planner and action-command generator for
the committed source. Record the new source hash, plan digest, run ID, and
exact CPU smoke command. Do not execute `modal run`.

Present the exact CPU smoke command for fresh user approval. Only after the
approved smoke succeeds and its downloaded receipt is independently validated
may a new Stage A action digest and exact GPU command be derived and presented
for separate approval.
