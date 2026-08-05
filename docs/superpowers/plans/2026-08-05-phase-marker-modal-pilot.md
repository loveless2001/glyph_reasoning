# Phase-Marker Modal Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated, artifact-bound Modal launcher that runs seed-42 training and validation-only checkpoint selection for all six phase-marker arms, then stops before behavior evaluation.

**Architecture:** Keep experiment planning and artifact rules in Modal-independent Python modules, with `modal_phase_marker.py` acting only as the compute resource and entrypoint adapter and `modal_phase_marker_inspect.py` acting as the standalone zero-compute inspection adapter. Stage immutable inputs and the exact Qwen snapshot on separate volumes, execute each arm in an isolated ephemeral workspace, and promote validated outputs through attempt directories into immutable canonical paths.

**Tech Stack:** Python 3.12, dataclasses, hashlib/JSON/pathlib/subprocess, pytest, PyTorch 2.9.0, Transformers 4.57.3, PEFT 0.18.0, vLLM 0.13.0, Modal 1.3.5, CUDA 12.8, Modal Volumes.

## Global Constraints

- Pilot run kind and excluded seed are exactly `pilot` and `42`.
- Arm order is exactly `semantic`, `glyph`, `dot`, `random`, `direct`, `filler`.
- Base model is exactly `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Emit exactly six training and six validation-only selection commands; emit no executable behavior, confirmatory, synthetic, capture, or intervention command.
- Use `gpu="H100"`, one visible GPU per job, four-hour job timeouts, at most two active containers, and zero application retries.
- Approval metadata is exactly 24 training GPU-hours, 24 selection GPU-hours, 72 later behavior GPU-hours, 120 total maximum GPU-hours, USD 600 estimated full-pilot spend, and USD 1,000 ceiling.
- Stage A estimate is USD 250; its worst-case GPU allocation is 48 H100-hours and approximately USD 189.56 at USD 3.9492 per hour.
- Preserve the existing `modal_app.py`, `glyph-reasoning-vol`, and all unrelated user changes.
- Use dedicated volumes `phase-marker-pilot-inputs-v1`, `phase-marker-pilot-model-cache-v1`, and `phase-marker-pilot-runs-v1`.
- Freeze the Modal environment as `main`, include it in plan/action identities,
  and emit `--env main` in every command. The operator must independently
  confirm the active profile and workspace match the reviewed account; the
  repository makes no live-account claim.
- Input and model volumes are read-only inside GPU jobs; only the run volume is writable.
- Never overwrite a canonical input, model snapshot, checkpoint, selection, receipt, or summary.
- Receipts and logs live outside hashed checkpoint and selection directories.
- Ordinary tests and `python -m phase_marker.modal_plan plan` are offline and perform no Modal/Hugging Face calls or remote writes; no `modal run` command is described as offline.
- Do not execute `stage-inputs`, `cache-model`, CPU remote smoke, `run-stage-a`, deployment, or a GPU command while implementing this plan.
- The implementation handoff must request fresh authorization separately for exact input upload/cache commands and the exact Stage A H100 command.
- Never include local Modal credentials, `.modal.toml`, tokens, `.git`, arbitrary artifacts, or model outputs in an image or input bundle; a later authentication need must use a named Modal Secret without recording its value.
- Treat the USD 1,000 value as an operator-acknowledged environment budget, not as an application-enforced account hard stop.
- No Git push, PR, dataset publication, checkpoint upload, or later behavior/mechanism launch is authorized by implementing this plan.

## File Responsibility Map

- Create `phase_marker/modal_plan.py`: frozen workload, approval, identity, and exact command planning; no Modal import or writes.
- Create `phase_marker/modal_artifacts.py`: input allowlist, source/bundle hashes, workspaces, receipts, attempt publication, and model-cache manifests.
- Create `modal_phase_marker.py`: thin Modal 1.3.5 compute images, volumes, remote functions, and compute operator entrypoints; it imports only pure inspection helpers and defines no status or evidence-download function.
- Create `phase_marker/modal_inspection.py`: Modal-independent status validation and descriptor-safe, atomic no-replace local evidence publication helpers.
- Create `modal_phase_marker_inspect.py`: standalone zero-compute, read-only runs-volume status and evidence-download entrypoints; it must not import the compute adapter.
- Create `requirements-modal-phase-marker.in`: human-reviewed direct runtime pins.
- Create `requirements-modal-phase-marker.txt`: compiled transitive lock with hashes; no floating dependency specifiers.
- Create `tests/phase_marker/test_modal_plan.py`: plan, workload, budget, and identity tests.
- Create `tests/phase_marker/test_modal_artifacts.py`: allowlist, staging, receipt, promotion, and model-cache tests.
- Create `tests/phase_marker/test_modal_adapter.py`: fake-Modal import tests and end-to-end fake Stage A orchestration.
- Modify `phase_marker/token_audit.py`: expose one public snapshot validator without changing existing validation behavior.
- Modify `README.md`: document local dry run, external-action commands as examples only, and approval boundaries.

---

### Task 1: Frozen Pilot Workload Planner

**Files:**
- Create: `phase_marker/modal_plan.py`
- Create: `tests/phase_marker/test_modal_plan.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `ApprovalMetadata`, `build_command_manifest`, `canonical_json`, and caller-supplied source/dependency hashes.
- Produces: `StageAResources`, `PilotJob`, `PilotPlan`, `build_stage_a_resources()`, `build_pilot_plan(config_path: Path, artifact_root: Path, source_hash: str, dependency_lock_hash: str) -> PilotPlan`, and `pilot_plan_payload(plan: PilotPlan) -> dict[str, object]`.

- [ ] **Step 1: Write failing workload and exclusion tests**

```python
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from phase_marker.modal_plan import build_pilot_plan, build_stage_a_resources


def test_stage_a_resources_are_the_approved_envelope():
    resources = build_stage_a_resources()
    assert resources.hardware == "H100"
    assert resources.timeout_seconds == 14_400
    assert resources.max_containers == 2
    assert resources.training_gpu_hours == 24
    assert resources.selection_gpu_hours == 24
    assert resources.behavior_gpu_hours == 72
    assert resources.max_gpu_hours == 120
    assert resources.stage_a_estimated_spend_usd == 250
    assert resources.estimated_spend_usd == 600
    assert resources.spend_cap_usd == 1_000
    with pytest.raises(FrozenInstanceError):
        resources.hardware = "A100"  # type: ignore[misc]


def test_pilot_plan_contains_only_six_training_and_six_selection_commands(
    prepared_artifacts: Path,
):
    plan = build_pilot_plan(
        Path("configs/phase-marker-qwen25-7b.toml"),
        prepared_artifacts,
        source_hash="1" * 64,
        dependency_lock_hash="2" * 64,
    )
    assert [job.arm for job in plan.jobs] == [
        "semantic", "glyph", "dot", "random", "direct", "filler"
    ]
    assert {job.seed for job in plan.jobs} == {42}
    assert len([job.training_command for job in plan.jobs]) == 6
    assert len([job.selection_command for job in plan.jobs]) == 6
    serialized = "\n".join(
        [*(job.training_command for job in plan.jobs),
         *(job.selection_command for job in plan.jobs)]
    )
    assert "--kind confirmatory" not in serialized
    assert "phase_marker.behavior run" not in serialized
    assert "phase_marker.activations" not in serialized
    assert "phase_marker.interventions" not in serialized
```

Reuse the reviewed split/materialization fixture helpers from
`tests/phase_marker/test_pipeline.py` to build `prepared_artifacts`; do not
invent a weaker manifest fixture.

- [ ] **Step 2: Run the new module and confirm RED**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_plan.py
```

Expected: collection fails because `phase_marker.modal_plan` does not exist.

- [ ] **Step 3: Implement frozen resource and plan types**

```python
@dataclass(frozen=True)
class StageAResources:
    hardware: str = "H100"
    timeout_seconds: int = 14_400
    max_containers: int = 2
    training_gpu_hours: float = 24.0
    selection_gpu_hours: float = 24.0
    behavior_gpu_hours: float = 72.0
    max_gpu_hours: float = 120.0
    stage_a_estimated_spend_usd: float = 250.0
    estimated_spend_usd: float = 600.0
    spend_cap_usd: float = 1_000.0

    def approval(self) -> ApprovalMetadata:
        return ApprovalMetadata(
            hardware="1x Modal H100 or automatic H200 upgrade",
            max_duration_hours=self.max_gpu_hours,
            training_gpu_hours=self.training_gpu_hours,
            selection_gpu_hours=self.selection_gpu_hours,
            behavior_gpu_hours=self.behavior_gpu_hours,
            spend_cap_usd=self.spend_cap_usd,
            estimated_spend_usd=self.estimated_spend_usd,
            workload_schema_version=1,
            training_jobs=6,
            checkpoint_selection_jobs=6,
            behavior_evaluation_jobs=1,
            manual_audit_rows=300,
            statistics_jobs=1,
            mechanism_jobs_excluded=True,
        )


@dataclass(frozen=True)
class PilotJob:
    arm: str
    seed: int
    model_revision: str
    training_command: str
    selection_command: str
    expected_outputs: tuple[str, ...]


@dataclass(frozen=True)
class PilotPlan:
    schema_version: int
    kind: str
    seed: int
    config_hash: str
    split_artifact_id: str
    materialization_artifact_ids: tuple[str, ...]
    model_revision: str
    source_hash: str
    dependency_lock_hash: str
    resources: StageAResources
    jobs: tuple[PilotJob, ...]
    run_id: str
```

`build_pilot_plan` must call the existing `build_command_manifest` with the
resource envelope's `ApprovalMetadata`, validate all six returned dictionaries,
read the already validated split/materialization IDs, and reject any extra or
missing job. Use the existing `QWEN25_7B_TOKENIZER_REVISION`; do not duplicate
the revision literal in implementation code.

- [ ] **Step 4: Add mutation and canonical-payload coverage**

Test wrong seed, reordered arms, missing approval, changed model revision,
non-SHA source/lock values, and a monkeypatched command manifest containing a
behavior command. Assert `pilot_plan_payload` contains only JSON-compatible
lists/scalars and reproduces through `canonical_json` plus `json.loads`.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_plan.py tests/phase_marker/test_pipeline.py
./.venv/bin/python -m py_compile phase_marker/modal_plan.py
git diff --check
```

Expected: all focused tests pass; compilation and diff check exit zero.

Commit:

```bash
git add phase_marker/modal_plan.py tests/phase_marker/test_modal_plan.py
git commit -m "feat: plan immutable Modal pilot workload"
```

---

### Task 2: Content-Addressed Source and Input Bundles

**Files:**
- Create: `phase_marker/modal_artifacts.py`
- Modify: `phase_marker/modal_plan.py`
- Modify: `tests/phase_marker/test_modal_plan.py`
- Create: `tests/phase_marker/test_modal_artifacts.py`

**Interfaces:**
- Consumes: repository root, approved artifact root, `sha256_json`, and `canonical_json`.
- Produces: `BundleFile`, `InputBundle`, a Modal-independent `VolumeClient` `Protocol`, `SOURCE_INCLUDE_PATHS`, `INPUT_ALLOWLIST`, `require_clean_tracked_status(status: str) -> None`, `hash_source_tree(repo_root: Path) -> str`, `build_input_bundle(repo_root: Path) -> InputBundle`, and `validate_bundle_at_root(bundle: InputBundle, root: Path) -> None`.
- CLI: `python -m phase_marker.modal_plan plan --repo-root PATH --config PATH --artifact-root PATH --dependency-lock PATH` prints canonical plan JSON; the sibling `run-id` subcommand prints only the full canonical run ID.

- [ ] **Step 1: Write failing allowlist, dirtiness, and hash tests**

```python
def test_input_bundle_is_exact_and_content_addressed(repo_fixture: Path):
    bundle = build_input_bundle(repo_fixture)
    assert tuple(item.path for item in bundle.files) == INPUT_ALLOWLIST
    assert all(len(item.sha256) == 64 and item.size > 0 for item in bundle.files)
    assert bundle.bundle_id == sha256_json({
        "schema_version": 1,
        "files": [asdict(item) for item in bundle.files],
        "artifact_ids": list(bundle.artifact_ids),
    })


def test_bundle_rejects_extra_or_changed_files(repo_fixture: Path):
    bundle = build_input_bundle(repo_fixture)
    target = repo_fixture / "artifacts/phase-marker/training-data/glyph.jsonl"
    target.write_text(target.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bundle file hash mismatch"):
        validate_bundle_at_root(bundle, repo_fixture)


@pytest.mark.parametrize("status", [" M phase_marker/modal_plan.py\n", "D  README.md\n"])
def test_tracked_dirty_status_is_rejected(status: str):
    with pytest.raises(ValueError, match="tracked source changes"):
        require_clean_tracked_status(status)
```

Also prove `?? artifacts/` is ignored by `require_clean_tracked_status`, while
path traversal such as `../secret` and absolute allowlist paths are rejected.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_artifacts.py tests/phase_marker/test_modal_plan.py
```

Expected: imports or assertions fail because bundle interfaces are missing.

- [ ] **Step 3: Implement exact file manifests and hashing**

Define the allowlist as the configuration; the four split JSONL files plus
split manifest; and `<arm>.jsonl` plus `<arm>.manifest.json` for every frozen
arm. Sort nothing implicitly: construct the tuple in canonical protocol order.

```python
@dataclass(frozen=True)
class BundleFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class InputBundle:
    schema_version: int
    bundle_id: str
    files: tuple[BundleFile, ...]
    artifact_ids: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

`hash_source_tree` includes only `phase_marker/**/*.py` and
`modal_phase_marker.py`, excluding `__pycache__`, `.pyc`, `.git`, `.venv`,
artifacts, and the three legacy top-level scripts. It hashes relative path plus
file SHA-256 using `sha256_json`. Keep the approved config under `config_hash`
and the compiled requirements under `dependency_lock_hash`; tests must prove a
change to each identity changes only its intended hash before recomputing the
derived run ID.

- [ ] **Step 4: Bind source, lock, and bundle IDs into the plan**

Change the interface to
`build_pilot_plan(config_path: Path, artifact_root: Path, *, bundle: InputBundle, source_hash: str, dependency_lock_hash: str) -> PilotPlan`
rather than independently rediscovering input artifact IDs. The split ID and
six arm IDs in the plan must equal the IDs parsed from the bundle's exact
manifest files. Derive:

```python
run_id = (
    f"pilot-s42-cfg-{config_hash[:8]}-split-{split_id[:8]}-"
    f"src-{source_hash[:12]}"
)
```

Reject any noncanonical run ID, duplicate artifact ID, missing manifest field,
or bundle whose file hashes change between the first and second read.

- [ ] **Step 5: Add the pure run-ID CLI**

Add `argparse` `plan` and `run-id` subcommands that resolve all paths beneath
`--repo-root`, hashes the tracked source tree and dependency lock, builds the
exact bundle and plan, and print canonical JSON or only `plan.run_id`,
respectively. Test `main(argv)` with `capsys`; both paths must make no Modal,
Hugging Face, network, subprocess, or filesystem-write call.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_plan.py tests/phase_marker/test_modal_artifacts.py
./.venv/bin/python -m py_compile phase_marker/modal_plan.py phase_marker/modal_artifacts.py
git diff --check
```

Commit:

```bash
git add phase_marker/modal_plan.py phase_marker/modal_artifacts.py \
  tests/phase_marker/test_modal_plan.py tests/phase_marker/test_modal_artifacts.py
git commit -m "feat: bind Modal pilot inputs and source"
```

---

### Task 3: Isolated Workspaces, Receipts, and Atomic Promotion

**Files:**
- Modify: `phase_marker/modal_artifacts.py`
- Modify: `tests/phase_marker/test_modal_artifacts.py`

**Interfaces:**
- Consumes: `InputBundle`, a read-only bundle root, baked code root, run-volume root, `PilotJob`, stage, arm, and subprocess environment.
- Produces: `AttemptReceipt`, `create_attempt_id() -> str`, `prepare_ephemeral_workspace(*, code_root: Path, input_root: Path, run_root: Path, bundle: InputBundle, stage: str, arm: str, attempt_id: str, canonical_training_root: Path | None = None) -> Path`, `run_exact_command(command: str, *, workspace: Path, log_path: Path, env: Mapping[str, str]) -> int`, `write_attempt_receipt(run_root: Path, receipt: AttemptReceipt) -> Path`, and `promote_validated_output(source: Path, attempt_root: Path, canonical_root: Path, receipt: AttemptReceipt) -> Path`.

- [ ] **Step 1: Write failing workspace and command-boundary tests**

```python
def test_workspace_recreates_exact_repository_paths(tmp_path: Path, bundle):
    workspace = prepare_ephemeral_workspace(
        code_root=tmp_path / "code",
        input_root=tmp_path / "inputs",
        run_root=tmp_path / "runs",
        bundle=bundle,
        stage="train",
        arm="glyph",
        attempt_id="attempt-1",
    )
    assert (workspace / ".venv/bin/python").is_symlink()
    assert (workspace / "phase_marker").is_symlink()
    assert (workspace / "configs/phase-marker-qwen25-7b.toml").is_file()
    assert (workspace / "artifacts/phase-marker/training-data/glyph.jsonl").is_file()


def test_exact_command_uses_no_shell(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0))
    run_exact_command(
        "./.venv/bin/python -m phase_marker.training train --arm glyph",
        workspace=tmp_path, log_path=tmp_path / "train.log", env={"A": "B"},
    )
    assert calls[0][0][:4] == ["./.venv/bin/python", "-m", "phase_marker.training"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == tmp_path
```

Reject commands whose executable is not `./.venv/bin/python`, contain shell
operators, name a different arm/seed, or address paths outside the ephemeral
workspace.

- [ ] **Step 2: Write failing promotion and receipt tests**

```python
def test_failed_attempt_never_promotes(tmp_path: Path):
    receipt = receipt_fixture(exit_status=1, validated=False)
    with pytest.raises(ValueError, match="validated successful receipt"):
        promote_validated_output(
            tmp_path / "output", tmp_path / "attempts/a1",
            tmp_path / "canonical/glyph", receipt,
        )
    assert not (tmp_path / "canonical/glyph").exists()


def test_receipt_and_log_are_outside_hashed_output(tmp_path: Path):
    receipt_path = write_attempt_receipt(tmp_path / "runs", receipt_fixture())
    assert "/receipts/" in receipt_path.as_posix()
    assert "/checkpoints/" not in receipt_path.as_posix()
```

Add tests for canonical-output refusal, byte-identical attempt copy, one-time
rename, rescheduled executions producing distinct UUID attempt IDs, and receipt
artifact ID recomputation.

- [ ] **Step 3: Implement fail-closed execution records**

```python
@dataclass(frozen=True)
class AttemptReceipt:
    schema_version: int
    run_id: str
    bundle_id: str
    stage: str
    arm: str
    seed: int
    attempt_id: str
    command: str
    command_hash: str
    source_hash: str
    dependency_lock_hash: str
    model_cache_artifact_id: str
    requested_gpu: str
    observed_gpu: str | None
    started_at: str
    finished_at: str
    elapsed_seconds: float
    timeout_seconds: int
    exit_status: int
    validated: bool
    promoted: bool
    expected_outputs: tuple[str, ...]
    output_hashes: tuple[str, ...]
    failure_reason: str | None
    artifact_id: str
```

Compute `artifact_id` from every field except itself. Write receipts atomically
as canonical JSON. `promote_validated_output` copies into an attempt directory
on the run filesystem and uses `Path.replace` only after validation; it refuses
an existing canonical path even when bytes match.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_artifacts.py
./.venv/bin/python -m py_compile phase_marker/modal_artifacts.py
git diff --check
```

Commit:

```bash
git add phase_marker/modal_artifacts.py tests/phase_marker/test_modal_artifacts.py
git commit -m "feat: quarantine and promote Modal attempts"
```

---

### Task 4: Exact Qwen Model-Cache Manifest

**Files:**
- Modify: `phase_marker/token_audit.py`
- Modify: `phase_marker/modal_artifacts.py`
- Modify: `tests/phase_marker/test_materialize.py`
- Modify: `tests/phase_marker/test_modal_artifacts.py`

**Interfaces:**
- Consumes: an exact local snapshot directory and the pinned revision.
- Produces: `validate_pinned_qwen_tokenizer_snapshot(snapshot: Path) -> None`, `ModelCacheFile`, `ModelCacheManifest`, `build_model_cache_manifest(snapshot: Path) -> ModelCacheManifest`, and `validate_model_cache_manifest(snapshot: Path, manifest: ModelCacheManifest) -> None`.

- [ ] **Step 1: Expose the existing tokenizer validator with behavior-preserving tests**

Rename the internal snapshot preflight to a public function and make
`_load_cached_tokenizer` call it:

```python
def validate_pinned_qwen_tokenizer_snapshot(snapshot: Path) -> None:
    """Fail before Transformers unless snapshot is the pinned Qwen JSON-BPE layout."""
```

Keep every existing malformed-config/tokenizer test green, and add a direct
test proving the public function makes no Transformers import.

- [ ] **Step 2: Write failing model-index and shard tests**

```python
def test_model_cache_manifest_binds_every_index_shard(qwen_snapshot: Path):
    manifest = build_model_cache_manifest(qwen_snapshot)
    assert manifest.model_revision == QWEN25_7B_TOKENIZER_REVISION
    assert {item.path for item in manifest.files} >= {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    validate_model_cache_manifest(qwen_snapshot, manifest)


def test_model_cache_rejects_missing_or_unindexed_shards(qwen_snapshot: Path):
    (qwen_snapshot / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="model shard"):
        build_model_cache_manifest(qwen_snapshot)
```

Also reject an invalid/non-object index, empty shard, path traversal in
`weight_map`, unindexed `*.safetensors`, wrong snapshot revision path, changed
bytes after manifest creation, and missing generation metadata.

- [ ] **Step 3: Implement cache file and manifest hashing**

```python
@dataclass(frozen=True)
class ModelCacheFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelCacheManifest:
    schema_version: int
    model_id: str
    model_revision: str
    files: tuple[ModelCacheFile, ...]
    artifact_id: str
```

Resolve all file names from the model index and add the exact config,
generation, and tokenizer files. Hash symlink targets by reading their bytes;
never trust link names as content hashes. The manifest artifact ID hashes the
schema, identity, revision, and ordered file records.

- [ ] **Step 4: Run real offline probe and focused suite**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_materialize.py tests/phase_marker/test_modal_artifacts.py
./.venv/bin/python -m py_compile phase_marker/token_audit.py phase_marker/modal_artifacts.py
git diff --check
```

The conditional real-cache test may skip if all four Qwen weight shards are not
locally cached; tokenizer-only real probes must pass. No test may download the
missing shard.

- [ ] **Step 5: Commit**

```bash
git add phase_marker/token_audit.py phase_marker/modal_artifacts.py \
  tests/phase_marker/test_materialize.py tests/phase_marker/test_modal_artifacts.py
git commit -m "feat: certify pinned Modal model cache"
```

---

### Task 5: Locked Modal Image and Resource Graph

**Files:**
- Create: `requirements-modal-phase-marker.in`
- Create: `requirements-modal-phase-marker.txt`
- Create: `modal_phase_marker.py`
- Create: `tests/phase_marker/test_modal_adapter.py`

**Interfaces:**
- Consumes: Modal SDK 1.3.5 and constants/interfaces from Tasks 1–4.
- Produces: dedicated `app`, `gpu_image`, `cpu_image`, three named volumes, a testable `RemoteFunction` `Protocol`, `apply_approved_app_tags(plan: PilotPlan) -> None`, and inert resource-decorated function declarations with no invoked remote side effects.

- [ ] **Step 1: Write direct pins and compile the exact dependency lock**

Create `requirements-modal-phase-marker.in` with these human-reviewed direct
runtime pins:

```text
accelerate==1.12.0
datasets==4.4.2
einops==0.8.1
huggingface-hub==0.36.0
modal==1.3.5
numpy==2.2.6
peft==0.18.0
protobuf==6.33.2
safetensors==0.7.0
sentencepiece==0.2.1
statsmodels==0.14.6
tokenizers==0.22.2
torch==2.9.0
transformers==4.57.3
vllm==0.13.0
```

Compile the complete transitive graph for the production interpreter and image
platform; do not permit the Modal image build to resolve an unlocked package:

```bash
uv pip compile requirements-modal-phase-marker.in \
  --output-file requirements-modal-phase-marker.txt \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --generate-hashes
```

If required package metadata is not already cached, stop and obtain approval
for the network read before running this compile command. Add a test that every
non-comment requirement in the compiled file has an exact `==` pin, package
names are unique, and at least one continuation contains `--hash=sha256:`.
Assert `dependency_lock_hash` is the SHA-256 of the compiled lock and changes
when its bytes change without changing `source_hash` or `config_hash`. The image
installs only `requirements-modal-phase-marker.txt`; `pip freeze` may be
recorded as additional evidence but is not the dependency lock.

- [ ] **Step 2: Write a fake Modal module and failing graph test**

Build a small `FakeModal` in the test that records `App`, image-chain, volume,
function-decorator, and local-entrypoint calls. Load `modal_phase_marker.py` with
`monkeypatch.setitem(sys.modules, "modal", fake_modal)`.

```python
def test_modal_graph_is_dedicated_and_bounded(imported_adapter):
    assert imported_adapter.app.name == "phase-marker-pilot-stage-a"
    assert imported_adapter.BASE_IMAGE == (
        "nvidia/cuda@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    )
    assert imported_adapter.VOLUME_NAMES == (
        "phase-marker-pilot-inputs-v1",
        "phase-marker-pilot-model-cache-v1",
        "phase-marker-pilot-runs-v1",
    )
    assert imported_adapter.GPU == "H100"
    assert imported_adapter.GPU_TIMEOUT_SECONDS == 14_400
    assert imported_adapter.MAX_GPU_CONTAINERS == 2
    assert "glyph-reasoning-vol" not in imported_adapter.source_text
    assert "/vol/work" not in imported_adapter.source_text
```

The fake must permit lazy `Volume.from_name` declarations but fail if import
attempts authentication, hydration, deployment, client RPC, or a remote call.

Test `apply_approved_app_tags` separately: it must call `app.set_tags` only
after plan/run-ID validation and preserve `experiment`, `run-kind`, and `seed`
while adding the full `run-id`. The read-only `status` path must never call it.

- [ ] **Step 3: Implement image, app, and volume declarations**

```python
APP_NAME = "phase-marker-pilot-stage-a"
BASE_IMAGE = (
    "nvidia/cuda@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
)
GPU = "H100"
GPU_TIMEOUT_SECONDS = 14_400
MAX_GPU_CONTAINERS = 2
VOLUME_NAMES = (
    "phase-marker-pilot-inputs-v1",
    "phase-marker-pilot-model-cache-v1",
    "phase-marker-pilot-runs-v1",
)

app = modal.App(
    APP_NAME,
    tags={"experiment": "phase-marker", "run-kind": "pilot", "seed": "42"},
    include_source=False,
)
inputs_volume = modal.Volume.from_name(VOLUME_NAMES[0], create_if_missing=True)
model_volume = modal.Volume.from_name(VOLUME_NAMES[1], create_if_missing=True)
runs_volume = modal.Volume.from_name(VOLUME_NAMES[2], create_if_missing=True)

gpu_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .pip_install_from_requirements("requirements-modal-phase-marker.txt")
    .add_local_dir("phase_marker", "/opt/glyph_reasoning/phase_marker", copy=True)
    .add_local_file(
        "modal_phase_marker.py", "/opt/glyph_reasoning/modal_phase_marker.py", copy=True
    )
    .add_local_file(
        "requirements-modal-phase-marker.txt",
        "/opt/glyph_reasoning/requirements-modal-phase-marker.txt",
        copy=True,
    )
    .run_commands(
        "mkdir -p /opt/glyph_reasoning/.venv/bin",
        "ln -sf /usr/local/bin/python /opt/glyph_reasoning/.venv/bin/python",
    )
)
cpu_image = gpu_image
```

The digest is the Docker Hub linux/amd64 manifest behind
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`; retain the tag only in a comment
for human readability. Assert the image declaration contains no mutable tag.

Use `.read_only()` on input/model volumes in GPU decorators because that is the
Modal 1.3.5 API installed in this workspace. Do not use the newer
`with_mount_options` spelling.

- [ ] **Step 4: Keep offline planning outside the Modal adapter**

Do not add a Modal `plan` entrypoint: `modal run` hydrates an app even for local
entrypoints. Test the Task 2 pure CLI while the fake Modal module records the
adapter import; the Python CLI must print canonical plan JSON with zero Modal
imports or fake calls, while importing the adapter may only create inert local
declarations and must perform zero client RPCs.

- [ ] **Step 5: Run adapter tests and commit**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_plan.py tests/phase_marker/test_modal_artifacts.py \
  tests/phase_marker/test_modal_adapter.py
./.venv/bin/python -m py_compile modal_phase_marker.py
git diff --check
```

Commit:

```bash
git add requirements-modal-phase-marker.in requirements-modal-phase-marker.txt \
  modal_phase_marker.py \
  tests/phase_marker/test_modal_adapter.py
git commit -m "feat: define bounded Modal pilot app"
```

---

### Task 6: Explicit Input Staging and CPU Model Cache

**Files:**
- Modify: `modal_phase_marker.py`
- Modify: `phase_marker/modal_artifacts.py`
- Modify: `tests/phase_marker/test_modal_adapter.py`
- Modify: `tests/phase_marker/test_modal_artifacts.py`

**Interfaces:**
- Consumes: `PilotPlan`, `InputBundle`, local repository, input/model volumes, and exact Qwen revision.
- Produces: `stage_inputs_local(bundle: InputBundle, volume: VolumeClient, *, approved_run_id: str, plan: PilotPlan, budget_acknowledged: bool) -> dict[str, object]`, `cache_model_to_volume(*, plan_payload: Mapping[str, object], cache_root: Path, volume: VolumeClient) -> dict[str, object]`, remote `cache_model_remote(plan_payload: dict[str, object]) -> dict[str, object]`, remote `smoke_remote(plan_payload: dict[str, object]) -> dict[str, object]`, and local `cache-model` and `smoke` entrypoints.

- [ ] **Step 1: Write failing idempotent upload tests**

Use a fake volume with `batch_upload`, `listdir`, and `read_file` recorders:

```python
def test_stage_inputs_uploads_only_allowlisted_bundle(adapter, fake_volume, bundle, plan):
    result = adapter.stage_inputs_local(
        bundle,
        fake_volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )
    assert [call.remote_path for call in fake_volume.put_calls] == [
        f"/bundles/{bundle.bundle_id}/{item.path}" for item in bundle.files
    ] + [f"/bundles/{bundle.bundle_id}/bundle-manifest.json"]
    assert result["uploaded"] is True


def test_byte_identical_restaging_is_noop(
    adapter, populated_fake_volume, bundle, plan,
):
    result = adapter.stage_inputs_local(
        bundle,
        populated_fake_volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )
    assert result == {"bundle_id": bundle.bundle_id, "uploaded": False}
    assert populated_fake_volume.put_calls == []
```

Reject a preexisting mismatched byte, any remote path outside the bundle ID,
missing budget acknowledgement, or a plan/bundle identity mismatch.

- [ ] **Step 2: Write failing CPU cache tests**

Monkeypatch `huggingface_hub.snapshot_download` to populate a temporary fake
snapshot and assert:

```python
assert kwargs == {
    "repo_id": "Qwen/Qwen2.5-7B-Instruct",
    "revision": QWEN25_7B_TOKENIZER_REVISION,
    "cache_dir": "/model-cache/hub",
}
```

Prove the cache function downloads to a temporary revision-addressed path,
builds and validates `ModelCacheManifest`, refuses conflicting canonical cache,
promotes once, writes the manifest beside rather than inside the hashed
snapshot, and calls `model_volume.commit()` only after validation.

Also test `smoke_remote` with fake read-only input/model mounts and a writable
run volume. It must import the locked runtime modules, validate source hash,
input bundle, and model-cache manifest without loading model weights, write one
content-addressed CPU smoke receipt, and make no GPU-decorated call.

- [ ] **Step 3: Implement staging and CPU cache functions**

Use `inputs_volume.batch_upload()` from the local client for the exact bundle.
Declare cache remote function with CPU/memory/timeout and only the writable model
volume:

```python
@app.function(
    image=cpu_image,
    cpu=4.0,
    memory=32_768,
    timeout=7_200,
    retries=0,
    volumes={"/model-cache": model_volume},
)
def cache_model_remote(plan_payload: dict[str, object]) -> dict[str, object]:
    return cache_model_to_volume(
        plan_payload=plan_payload,
        cache_root=Path("/model-cache"),
        volume=model_volume,
    )
```

The implementation imports `snapshot_download` inside the function, validates
the plan payload before any download, writes an attempt receipt on failure, and
commits only a complete canonical cache plus separate manifest.

Declare `smoke_remote` on `cpu_image` with CPU/memory/timeout, zero retries,
read-only input/model mounts, and the writable run volume. Its result is the
validated smoke receipt; a failed import or hash check persists a failed
receipt and re-raises.

- [ ] **Step 4: Add local entrypoints without invoking them**

`stage-inputs` prints the exact bundle ID, file count, destination, and budget
acknowledgement before calling `stage_inputs_local`. `cache-model` prints the
revision, CPU timeout, model destination, and source/lock hashes before calling
`cache_model_remote.remote`. `smoke` prints its CPU resources and exact checks,
then calls `smoke_remote.remote` and reports the receipt path. Require
`--approved-run-id` to equal the plan's full run ID and an explicit budget
acknowledgement for all three mutating/compute entrypoints. After those checks
and before the first remote action, call `apply_approved_app_tags(plan)` so the
Modal app is attributable to the full run ID.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_artifacts.py tests/phase_marker/test_modal_adapter.py
./.venv/bin/python -m py_compile modal_phase_marker.py phase_marker/modal_artifacts.py
git diff --check
```

Commit:

```bash
git add modal_phase_marker.py phase_marker/modal_artifacts.py \
  tests/phase_marker/test_modal_artifacts.py tests/phase_marker/test_modal_adapter.py
git commit -m "feat: stage Modal inputs and model cache"
```

---

### Task 7: H100 Training, Selection, and Mandatory Stop

**Files:**
- Modify: `modal_phase_marker.py`
- Modify: `phase_marker/modal_artifacts.py`
- Modify: `tests/phase_marker/test_modal_adapter.py`
- Modify: `tests/phase_marker/test_modal_artifacts.py`

**Interfaces:**
- Consumes: approved serialized `PilotPlan`, canonical bundle/model cache, one `PilotJob`, and three volumes.
- Produces: remote `run_training_job(job_payload: dict[str, object]) -> dict[str, object]`, remote `run_selection_job(job_payload: dict[str, object]) -> dict[str, object]`, remote `finalize_stage_a_remote(plan_payload: Mapping[str, object], receipts: Sequence[Mapping[str, object]]) -> dict[str, object]`, `run_stage_a_local(plan: PilotPlan, *, approved_run_id: str, budget_acknowledged: bool, resume: bool, training_function: RemoteFunction, selection_function: RemoteFunction, finalizer_function: RemoteFunction, runs_client: VolumeClient) -> dict[str, object]`, and the `run-stage-a` entrypoint.

- [ ] **Step 1: Write failing resource and stage-order tests**

Assert fake decorators record both GPU functions with:

```python
{
    "gpu": "H100",
    "timeout": 14_400,
    "max_containers": 2,
    "retries": 0,
    "ephemeral_disk": 80 * 1024,
}
```

Assert input/model mounts use `.read_only()`, run volume is writable, and each
function sees one job payload. The fake orchestrator test must prove all six
training results are validated before the first selection remote call.

```python
def test_training_failure_prevents_every_selection(fake_adapter, plan):
    fake_adapter.training_results["dot"] = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="dot"):
        fake_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=fake_adapter.training_function,
            selection_function=fake_adapter.selection_function,
            finalizer_function=fake_adapter.finalizer_function,
            runs_client=fake_adapter.runs_client,
        )
    assert fake_adapter.selection_calls == []
```

- [ ] **Step 2: Write failing exact-command and promotion tests**

For each arm, inject a fake subprocess and producer validator. Assert training
executes `job.training_command` byte-for-byte, selection executes
`job.selection_command` byte-for-byte, `CUDA_VISIBLE_DEVICES` exposes one GPU,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_CACHE` points to the
validated read-only snapshot.

Prove validation failure writes an unpromoted receipt, canonical paths remain
absent, and success promotes only the producer directory while storing
receipt/log separately.

Add resume tests with two valid canonical training arms and one quarantined
failed attempt. Revalidate the two canonical producer manifests and receipts,
schedule only the four missing training arms, and ignore rather than reuse or
delete the failed attempt. A corrupt canonical receipt or producer manifest
must abort the whole resume before any remote call.

- [ ] **Step 3: Implement the GPU job boundary**

```python
GPU_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
    "/runs": runs_volume,
}


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=14_400,
    startup_timeout=1_200,
    max_containers=2,
    retries=0,
    ephemeral_disk=80 * 1024,
    volumes=GPU_VOLUMES,
)
def run_training_job(job_payload: dict[str, object]) -> dict[str, object]:
    return _execute_job("train", job_payload)
```

Define selection with the same resources and
`return _execute_job("select", job_payload)`.
Inside `_execute_job`: validate plan/job/bundle/cache before importing model
packages, create a fresh UUID attempt and ephemeral workspace, execute with
`shell=False`, call existing pipeline producer/consumer validators, observe GPU
identity through a bounded `nvidia-smi --query-gpu=name --format=csv,noheader`
call, require exactly one visible CUDA device and
`torch.cuda.is_bf16_supported()` before the experiment command, write
receipt/log, promote once, commit, and return canonical receipt JSON.

Catch exceptions only to persist a failed receipt and log; re-raise afterward.
Never turn an exception into a successful map result.

- [ ] **Step 4: Implement ordered Stage A orchestration and summary**

`run_stage_a_local` must:

1. require exact `approved_run_id` and environment-budget acknowledgement,
   then call `apply_approved_app_tags(plan)` before the first remote action;
2. re-run local preflight; with `resume=False`, reject any canonical output;
   with `resume=True`, revalidate every existing canonical receipt and producer
   manifest and abort before remote work on any mismatch;
3. derive a frozen-order resume plan containing only missing canonical training
   arms, print it for the operator, and invoke `run_training_job.map` only for
   those arms;
4. require six successful validated training receipts across existing and new
   outputs;
5. call `runs_volume.reload()` before selection;
6. revalidate existing selections, derive the missing selection arms whose
   training parents are valid, and invoke `run_selection_job.map` only for them
   in frozen order;
7. require six successful validated selection receipts across existing and new
   outputs;
8. call `runs_volume.reload()` before finalization;
9. call CPU-only `finalize_stage_a_remote` with the run volume mounted,
   re-run the existing behavior prerequisite gate as read-only validation; and
10. publish a Stage A summary whose `next_command` is inert data and whose
    `stopped_before_behavior` is exactly `true`.

The summary schema must reject an executable callback, confirmation seeds, or
mechanism approval. No code path calls the behavior command.

Decorate `finalize_stage_a_remote` with `cpu_image`, bounded CPU/memory/timeout,
zero retries, read-only input/model mounts, and the writable run volume. It
must not request a GPU or import/load model weights.

- [ ] **Step 5: Run fake end-to-end Stage A tests**

Test all six arms through fake remote functions, a training error, a selection
error, a corrupt canonical output, a mismatched run ID, initial-mode refusal of
existing outputs, a partial valid explicit resume, and a successful summary.
Assert exactly 12 GPU calls plus one CPU finalizer call and zero behavior calls
in the clean successful case; assert the resume case calls only missing arms
and never deletes or overwrites an existing attempt or canonical output.

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q \
  tests/phase_marker/test_modal_adapter.py tests/phase_marker/test_modal_artifacts.py \
  tests/phase_marker/test_pipeline.py tests/phase_marker/test_training.py \
  tests/phase_marker/test_behavior.py
./.venv/bin/python -m py_compile modal_phase_marker.py phase_marker/modal_artifacts.py
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add modal_phase_marker.py phase_marker/modal_artifacts.py \
  tests/phase_marker/test_modal_adapter.py tests/phase_marker/test_modal_artifacts.py
git commit -m "feat: orchestrate staged Modal pilot"
```

---

### Task 8: Operator Status, Evidence Retrieval, Documentation, and Final Local Gate

**Files:**
- Create: `phase_marker/modal_inspection.py`
- Create: `modal_phase_marker_inspect.py`
- Modify: `modal_phase_marker.py`
- Modify: `README.md`
- Modify: `tests/phase_marker/test_modal_adapter.py`
- Modify: `tests/phase_marker/test_modal_artifacts.py`
- Modify: `tests/phase_marker/test_modal_plan.py`

**Interfaces:**
- Consumes: run ID, volume client, canonical receipts/summary.
- Produces: read-only `status_local(volume: VolumeClient, *, run_id: str) -> dict[str, object]`, explicit `download_evidence_local(volume: VolumeClient, *, run_id: str, destination: Path) -> tuple[Path, ...]`, `status` and `download-evidence` entrypoints, and final approval-ready command manifest.

- [ ] **Step 1: Write failing read-only status tests**

```python
def test_status_reads_receipts_without_remote_mutation(adapter, populated_volume):
    result = adapter.status_local(populated_volume, run_id=RUN_ID)
    assert result["training"] == {arm: "complete" for arm in ARMS}
    assert result["selection"] == {arm: "complete" for arm in ARMS}
    assert result["stopped_before_behavior"] is True
    assert populated_volume.write_calls == []
    assert populated_volume.remote_calls == []
```

Cover partial, failed, hash-mismatched, unknown-run, and complete states. Status
must never infer completion from a directory name without validating the receipt
and canonical producer manifest.

- [ ] **Step 2: Write failing evidence-download tests**

The allowlist is receipts, logs, Stage A summary, six run manifests, six adapter
configs, six selection manifests, and six selection evidence JSONLs. Assert
checkpoint weights, raw model cache, `.modal.toml`, and arbitrary volume files
cannot be downloaded by this entrypoint. Refuse an existing local destination.

- [ ] **Step 3: Implement status and download entrypoints**

Implement the entrypoints in the standalone `modal_phase_marker_inspect.py`
adapter, which declares only the non-creating read-only runs volume and imports
no compute image, GPU function, or compute adapter. Keep validation and local
publication in `phase_marker/modal_inspection.py`; the compute adapter may
import only pure helpers from that module. Use Modal 1.3.5 `Volume.listdir` and
`Volume.read_file` through small injectable adapters. Materialize downloads in
a temporary sibling directory, recompute every advertised hash, perform a final
source allowlist relist, and publish with an atomic no-replace directory rename.
Refuse any pre-existing destination, including a dangling symlink, and preserve
a destination created concurrently. `status` is read-only; `download-evidence`
is an explicit local write initiated by its own command.

- [ ] **Step 4: Document exact non-executing operator flow**

Add a README section that distinguishes:

```bash
# Local only: derive the full content-bound identity without Modal or network.
PHASE_MARKER_RUN_ID="$(./.venv/bin/python -m phase_marker.modal_plan run-id \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt)"

# Local only: print the canonical plan; no Modal import, remote call, write, model load, or GPU.
./.venv/bin/python -m phase_marker.modal_plan plan \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt

# The plan JSON contains only the exact digest-bound stage-inputs, cache-model,
# and smoke actions as inert strings. Execute none without separate fresh approval.

# Local only, and only after reviewing exact successful cache/smoke artifact IDs:
./.venv/bin/python -m phase_marker.modal_plan stage-a-action \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt \
  --smoke-receipt-artifact-id '<REVIEWED_SMOKE_RECEIPT_ARTIFACT_ID>' \
  --model-cache-artifact-id '<REVIEWED_MODEL_CACHE_ARTIFACT_ID>' \
  --fresh

# The local command above prints the exact H100 action as inert data. It is not
# included in this checked-in plan. Use --resume only after status/recovery review.

# Read-only remote inspection after an authorized run.
modal run --env main modal_phase_marker_inspect.py::status --run-id "$PHASE_MARKER_RUN_ID"

# Explicit local evidence materialization through the standalone zero-compute
# inspector. The destination must not already exist.
modal run --env main modal_phase_marker_inspect.py::download-evidence \
  --run-id "$PHASE_MARKER_RUN_ID" \
  --destination artifacts/phase-marker-evidence/"$PHASE_MARKER_RUN_ID"
```

Explain Stage A's 48-hour/$250 envelope, full pilot $600/$1,000 boundary, two
concurrent H100 jobs, automatic H200 compatibility, mandatory behavior stop,
and separate mechanism approval. Do not insert a real run ID until the final
source and lock hashes are frozen.

- [ ] **Step 5: Run fresh full local verification**

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/pytest -q tests
./.venv/bin/python -m py_compile \
  phase_marker/modal_plan.py phase_marker/modal_artifacts.py \
  phase_marker/modal_inspection.py modal_phase_marker.py \
  modal_phase_marker_inspect.py
git diff --check
git status --short
```

Expected: the maintained suite passes, compilation/diff check exit zero, and
only the preserved untracked experiment `artifacts/` plus intentional task
changes appear before commit. Do not use bare `pytest -q`, which collects the
pre-existing top-level `infer_test.py` and performs an import-time Hub lookup.

- [ ] **Step 6: Generate but do not execute the final action manifest**

Run the local planner and save its canonical JSON under `/tmp` only. Verify it
contains the exact digest-bound input-upload, CPU cache, and CPU smoke commands,
but explicitly withholds the Stage A command pending reviewed cache/smoke
artifact IDs and an explicit fresh/resume mode. Verify six training and six
selection jobs; the full run ID and plan digest; H100/time/concurrency; USD 250
Stage A estimate; USD 600 full estimate; USD 1,000 ceiling; and no
behavior/mechanism execution command.

Do not contact Modal, create volumes, upload data, cache the model, build the
remote image, deploy, or launch a GPU in this step.

- [ ] **Step 7: Commit and prepare approval handoff**

```bash
git add README.md docs/superpowers/plans/2026-08-05-phase-marker-modal-pilot.md \
  modal_phase_marker.py modal_phase_marker_inspect.py phase_marker/config.py \
  phase_marker/modal_artifacts.py phase_marker/modal_inspection.py \
  phase_marker/modal_plan.py tests/phase_marker/test_modal_adapter.py \
  tests/phase_marker/test_modal_artifacts.py tests/phase_marker/test_modal_plan.py
git commit -m "docs: finalize Modal pilot operator flow"
```

Do not stage the preserved untracked `artifacts/` tree.

The handoff must report test counts, branch/commit, input/model/run identities,
exact proposed external commands, and the remaining budget. Request fresh
authorization first for input upload plus CPU cache/smoke. After those artifacts
are inspected, request a second fresh authorization for the exact H100 Stage A
command. Do not combine either authorization with later behavior or mechanism
work.

---

## Plan Completion Gate

Before claiming implementation complete, verify every task commit and map each
design requirement to passing evidence:

- Dedicated app and volumes; legacy launcher untouched.
- Exact content-addressed source, lock, input, and model identities.
- Six seed-42 training and six validation-only selection jobs.
- H100, one GPU, four-hour timeouts, two containers, zero application retries.
- Immutable attempts, canonical promotion, separate receipts/logs.
- CPU model download and GPU read-only cache.
- Mandatory stop before behavior and separate mechanism boundary.
- USD 250 Stage A estimate, USD 600 full estimate, USD 1,000 acknowledged budget.
- Offline local suite green and no external actions taken during implementation.

Any failed item keeps the launcher unready for a remote smoke or GPU approval
request.
