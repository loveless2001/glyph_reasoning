from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

import modal_phase_marker as adapter
from phase_marker.config import ExperimentConfig
from phase_marker.modal_artifacts import (
    _tree_hashes,
    atomic_publish_directory_noreplace,
    build_input_bundle,
    hash_source_tree,
    source_tree_relative_paths,
    validate_bundle_at_root,
)
from phase_marker.modal_plan import (
    action_approval_payload,
    approved_stage_a_action_manifest,
    build_pilot_plan,
)
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_HASH = "1" * 64
LOCK_HASH = "2" * 64
SMOKE_ID = "3" * 64
CACHE_ID = "4" * 64


@pytest.fixture
def pilot_repo(tmp_path: Path) -> Path:
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / CONFIG_PATH, config_path)
    config = ExperimentConfig.load(config_path)
    artifact_root = tmp_path / "artifacts/phase-marker"
    _write_split(artifact_root, config)
    _write_materializations(artifact_root, config)
    return tmp_path


@pytest.fixture
def plan(pilot_repo: Path):
    return build_pilot_plan(
        pilot_repo / CONFIG_PATH,
        pilot_repo / "artifacts/phase-marker",
        bundle=build_input_bundle(pilot_repo),
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )


def test_plan_is_deterministic_and_binds_research_workload(plan, pilot_repo: Path) -> None:
    rebuilt = build_pilot_plan(
        pilot_repo / CONFIG_PATH,
        pilot_repo / "artifacts/phase-marker",
        bundle=build_input_bundle(pilot_repo),
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )

    assert rebuilt == plan
    assert [job.arm for job in plan.jobs] == [
        "semantic", "glyph", "dot", "random", "direct", "filler"
    ]
    assert plan.resources.hardware == "H100"
    assert plan.resources.max_containers == 2
    assert plan.resources.stage_a_estimated_spend_usd == 250
    assert plan.resources.spend_cap_usd == 1_000


def test_stage_a_action_binds_dependencies_and_resume_mode(plan) -> None:
    fresh = approved_stage_a_action_manifest(
        plan,
        smoke_receipt_artifact_id=SMOKE_ID,
        model_cache_artifact_id=CACHE_ID,
        resume=False,
    )
    resumed = approved_stage_a_action_manifest(
        plan,
        smoke_receipt_artifact_id=SMOKE_ID,
        model_cache_artifact_id=CACHE_ID,
        resume=True,
    )

    assert fresh["approval"]["resume"] is False
    assert resumed["approval"]["resume"] is True
    assert fresh["approval"]["approval_digest"] != resumed["approval"]["approval_digest"]
    assert SMOKE_ID in fresh["external_action"]
    assert CACHE_ID in fresh["external_action"]


def test_input_bundle_detects_changed_bytes(pilot_repo: Path) -> None:
    bundle = build_input_bundle(pilot_repo)
    path = pilot_repo / "artifacts/phase-marker/training-data/glyph.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle file hash mismatch"):
        validate_bundle_at_root(bundle, pilot_repo)


def test_source_hash_changes_with_python_source(tmp_path: Path) -> None:
    for relative in source_tree_relative_paths(REPO_ROOT):
        source = REPO_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    before = hash_source_tree(tmp_path)
    target = tmp_path / "phase_marker/scoring.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert hash_source_tree(tmp_path) != before


def test_nested_checkpoint_tree_publishes_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "attempt" / "checkpoint"
    (source / "checkpoint-100").mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "checkpoint-100/trainer_state.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "canonical" / "semantic"

    result = atomic_publish_directory_noreplace(
        source, destination, create_parents=True
    )

    assert result == destination
    assert (destination / "checkpoint-100/trainer_state.json").read_text() == "{}\n"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        atomic_publish_directory_noreplace(replacement, destination)


def test_symlink_output_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    external = tmp_path / "external"
    external.write_text("bytes", encoding="utf-8")
    (root / "adapter").symlink_to(external)

    with pytest.raises(ValueError, match="regular files only"):
        _tree_hashes(root)


def test_modal_functions_request_h100_workers_and_cpu_finalizer() -> None:
    def spec(function):
        inner = next(
            value
            for name, value in vars(function).items()
            if name.startswith("_sync_original")
        )
        return inner._spec

    assert spec(adapter.run_training_job).gpus == "H100"
    assert spec(adapter.run_selection_job).gpus == "H100"
    assert spec(adapter.finalize_stage_a_remote).gpus is None
    assert spec(adapter.finalize_stage_a_remote).cpu == 2.0


class _Remote:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[object] = []

    def remote(self, payload):
        self.calls.append(payload)
        return self.result


class _MapRemote:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.payloads: list[dict[str, object]] = []

    def map(self, payloads):
        self.payloads = list(payloads)
        return [
            {"stage": self.stage, "arm": payload["job"]["arm"]}
            for payload in self.payloads
        ]


def _run_stage_a(monkeypatch, plan, *, existing_training=None):
    existing_training = {} if existing_training is None else existing_training
    evidence = adapter.StageADependencyEvidence(
        bundle_id=plan.bundle_id,
        model_cache_artifact_id=CACHE_ID,
        smoke_receipt_artifact_id=SMOKE_ID,
        bundle_manifest_artifact_id=plan.bundle_manifest_artifact_id,
        smoke_receipt={"validated": True},
    )
    monkeypatch.setattr(adapter, "_validate_stage_a_dependency_evidence", lambda *a: evidence)
    monkeypatch.setattr(
        adapter,
        "_preflight_stage_a_outputs",
        lambda *a, **k: adapter.StageAPreflight(existing_training, {}, None, ()),
    )
    monkeypatch.setattr(adapter, "apply_approved_app_tags", lambda *a, **k: None)
    monkeypatch.setattr(
        adapter,
        "validate_job_receipt_payload",
        lambda receipt_payload, **kwargs: dict(receipt_payload),
    )
    monkeypatch.setattr(
        adapter,
        "_revalidate_completed_stage_outputs",
        lambda plan, stage, runs_client, expected: dict(expected),
    )
    monkeypatch.setattr(
        adapter,
        "_revalidate_resume_selections",
        lambda plan, resume, runs_client, existing: dict(existing),
    )
    monkeypatch.setattr(adapter, "validate_stage_a_summary", lambda summary, **kwargs: dict(summary))
    training = _MapRemote("train")
    selection = _MapRemote("selection")
    finalizer = _Remote({"stopped_before_behavior": True})
    dependency = _Remote({"validated": True})
    approval = action_approval_payload(
        plan,
        action="run-stage-a",
        resume=bool(existing_training),
        smoke_receipt_artifact_id=SMOKE_ID,
        model_cache_artifact_id=CACHE_ID,
    )
    result = adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=bool(existing_training),
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        dependency_function=dependency,
        runs_client=object(),
        smoke_receipt_artifact_id=SMOKE_ID,
        model_cache_artifact_id=CACHE_ID,
        approval_payload=approval,
    )
    return result, training, selection


def test_stage_a_runs_training_selection_then_stops(monkeypatch, plan) -> None:
    result, training, selection = _run_stage_a(monkeypatch, plan)

    assert [payload["job"]["arm"] for payload in training.payloads] == [
        job.arm for job in plan.jobs
    ]
    assert [payload["job"]["arm"] for payload in selection.payloads] == [
        job.arm for job in plan.jobs
    ]
    assert result == {"stopped_before_behavior": True}


def test_stage_a_resume_skips_durable_training(monkeypatch, plan) -> None:
    completed = {"semantic": {"stage": "train", "arm": "semantic"}}
    result, training, _selection = _run_stage_a(
        monkeypatch, plan, existing_training=completed
    )

    assert [payload["job"]["arm"] for payload in training.payloads] == [
        "glyph", "dot", "random", "direct", "filler"
    ]
    assert result["stopped_before_behavior"] is True
