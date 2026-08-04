from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import shutil
import shlex
from typing import Callable

import pytest

import phase_marker.modal_plan as modal_plan
from phase_marker.modal_artifacts import build_input_bundle
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
SOURCE_HASH = "1" * 64
LOCK_HASH = "2" * 64


@pytest.fixture
def prepared_artifacts(tmp_path: Path) -> Path:
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(CONFIG_PATH, config_path)
    config = ExperimentConfig.load(config_path)
    artifact_root = tmp_path / "artifacts/phase-marker"
    _write_split(artifact_root, config)
    _write_materializations(artifact_root, config)
    return artifact_root


def _plan(prepared_artifacts: Path) -> modal_plan.PilotPlan:
    repo_root = prepared_artifacts.parent.parent
    return modal_plan.build_pilot_plan(
        repo_root / CONFIG_PATH,
        prepared_artifacts,
        bundle=build_input_bundle(repo_root),
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )


def test_stage_a_resources_are_the_approved_envelope() -> None:
    resources = modal_plan.build_stage_a_resources()

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
    assert resources.approval().training_jobs == 6
    assert resources.approval().checkpoint_selection_jobs == 6
    with pytest.raises(FrozenInstanceError):
        resources.hardware = "A100"  # type: ignore[misc]


def test_pilot_plan_contains_only_six_training_and_six_selection_commands(
    prepared_artifacts: Path,
) -> None:
    plan = _plan(prepared_artifacts)

    assert [job.arm for job in plan.jobs] == [
        "semantic", "glyph", "dot", "random", "direct", "filler"
    ]
    assert {job.seed for job in plan.jobs} == {42}
    assert len([job.training_command for job in plan.jobs]) == 6
    assert len([job.selection_command for job in plan.jobs]) == 6
    assert plan.model_revision == QWEN25_7B_TOKENIZER_REVISION
    assert plan.run_id == "pilot-s42-cfg-{}-split-{}-src-{}".format(
        plan.config_hash[:8], plan.split_artifact_id[:8], SOURCE_HASH[:12]
    )
    serialized = "\n".join(
        [*(job.training_command for job in plan.jobs),
         *(job.selection_command for job in plan.jobs)]
    )
    assert "--kind confirmatory" not in serialized
    assert "phase_marker.behavior run" not in serialized
    assert "phase_marker.activations" not in serialized
    assert "phase_marker.interventions" not in serialized


def test_pilot_plan_commands_are_portable_workspace_relative_bytes(
    prepared_artifacts: Path,
) -> None:
    """Would fail if a frozen remote job retained its host worktree prefix."""
    plan = _plan(prepared_artifacts)
    host_prefix = prepared_artifacts.parent.parent.as_posix()

    for job in plan.jobs:
        training = shlex.split(job.training_command)
        selection = shlex.split(job.selection_command)
        assert host_prefix not in job.training_command
        assert host_prefix not in job.selection_command
        assert training[training.index("--config") + 1] == (
            "configs/phase-marker-qwen25-7b.toml"
        )
        assert training[training.index("--data") + 1] == (
            f"artifacts/phase-marker/training-data/{job.arm}.jsonl"
        )
        assert training[training.index("--output-dir") + 1] == (
            f"artifacts/phase-marker/checkpoints/pilot/seed-42/{job.arm}"
        )
        assert selection[selection.index("--split-manifest") + 1] == (
            "artifacts/phase-marker/splits/manifest.json"
        )
        assert selection[selection.index("--training-manifest") + 1] == (
            f"artifacts/phase-marker/checkpoints/pilot/seed-42/{job.arm}/run-manifest.json"
        )
        assert selection[selection.index("--output") + 1] == (
            f"artifacts/phase-marker/checkpoint-selections/pilot/seed-42/{job.arm}"
        )


def test_pilot_plan_rejects_missing_or_extra_manifest_jobs(
    monkeypatch: pytest.MonkeyPatch, prepared_artifacts: Path,
) -> None:
    real_manifest = modal_plan.build_command_manifest

    def missing(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        return real_manifest(*args, **kwargs)[:-1]

    monkeypatch.setattr(modal_plan, "build_command_manifest", missing)
    with pytest.raises(ValueError):
        _plan(prepared_artifacts)

    def extra(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        jobs = real_manifest(*args, **kwargs)
        return (*jobs, jobs[0])

    monkeypatch.setattr(modal_plan, "build_command_manifest", extra)
    with pytest.raises(ValueError):
        _plan(prepared_artifacts)


ManifestMutation = Callable[[list[dict[str, object]]], None]


@pytest.mark.parametrize(
    ("mutate",),
    [
        (lambda jobs: jobs[0].update(seed=101),),
        (lambda jobs: jobs.__setitem__(0, jobs[1]),),
        (
            lambda jobs: jobs[0].update(
                approval_ready=False,
                approval=None,
                missing_approval_fields=["hardware"],
            ),
        ),
        (
            lambda jobs: jobs[0].update(model_revision="3" * 40),
        ),
        (
            lambda jobs: jobs[0].update(
                selection_command="./.venv/bin/python -m phase_marker.behavior run"
            ),
        ),
    ],
)
def test_pilot_plan_rejects_mutated_manifest_contract(
    monkeypatch: pytest.MonkeyPatch,
    prepared_artifacts: Path,
    mutate: ManifestMutation,
) -> None:
    real_manifest = modal_plan.build_command_manifest

    def mutated(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        jobs = [dict(job) for job in real_manifest(*args, **kwargs)]
        mutate(jobs)
        return tuple(jobs)

    monkeypatch.setattr(modal_plan, "build_command_manifest", mutated)

    with pytest.raises(ValueError):
        _plan(prepared_artifacts)


@pytest.mark.parametrize(
    "extra_command",
    [
        pytest.param("phase_marker.synthetic build", id="synthetic"),
        pytest.param("phase_marker.activations capture", id="capture"),
        pytest.param("phase_marker.interventions run", id="intervention"),
    ],
)
def test_pilot_plan_rejects_an_added_excluded_command(
    monkeypatch: pytest.MonkeyPatch,
    prepared_artifacts: Path,
    extra_command: str,
) -> None:
    real_manifest = modal_plan.build_command_manifest

    def mutated(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        jobs = [dict(job) for job in real_manifest(*args, **kwargs)]
        jobs[0]["selection_command"] += f"; ./.venv/bin/python -m {extra_command}"
        return tuple(jobs)

    monkeypatch.setattr(modal_plan, "build_command_manifest", mutated)

    with pytest.raises(ValueError):
        _plan(prepared_artifacts)


@pytest.mark.parametrize(
    ("source_hash", "dependency_lock_hash"),
    [
        ("not-a-sha", LOCK_HASH),
        (SOURCE_HASH, "not-a-sha"),
    ],
)
def test_pilot_plan_rejects_non_sha_identity_inputs(
    prepared_artifacts: Path, source_hash: str, dependency_lock_hash: str,
) -> None:
    with pytest.raises(ValueError):
        modal_plan.build_pilot_plan(
            prepared_artifacts.parent.parent / CONFIG_PATH,
            prepared_artifacts,
            bundle=build_input_bundle(prepared_artifacts.parent.parent),
            source_hash=source_hash,
            dependency_lock_hash=dependency_lock_hash,
        )


def test_pilot_plan_payload_round_trips_as_canonical_json(
    prepared_artifacts: Path,
) -> None:
    payload = modal_plan.pilot_plan_payload(_plan(prepared_artifacts))

    def assert_json_compatible(value: object) -> None:
        if isinstance(value, dict):
            assert all(isinstance(key, str) for key in value)
            for nested in value.values():
                assert_json_compatible(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_json_compatible(nested)
        else:
            assert value is None or isinstance(value, (str, int, float, bool))

    assert_json_compatible(payload)
    assert json.loads(canonical_json(payload)) == payload


def test_approval_action_manifest_contains_only_inert_approved_boundaries(
    prepared_artifacts: Path,
) -> None:
    """Would fail if the handoff omitted a gate or embedded later experiment work."""
    plan = _plan(prepared_artifacts)

    manifest = modal_plan.approval_action_manifest(plan)

    assert manifest["run_id"] == plan.run_id
    assert manifest["bundle_id"] == plan.bundle_id
    assert manifest["model_revision"] == plan.model_revision
    assert manifest["training_job_count"] == 6
    assert manifest["selection_job_count"] == 6
    assert manifest["resources"] == {
        "hardware": "H100",
        "timeout_seconds": 14_400,
        "max_containers": 2,
        "stage_a_estimated_spend_usd": 250.0,
        "estimated_spend_usd": 600.0,
        "spend_cap_usd": 1_000.0,
    }
    assert manifest["external_actions"] == {
        "stage_inputs": (
            "modal run modal_phase_marker.py::stage-inputs --approved-run-id "
            '"$PHASE_MARKER_RUN_ID" --acknowledge-budget-usd 1000'
        ),
        "cache_model": (
            "modal run modal_phase_marker.py::cache-model --approved-run-id "
            '"$PHASE_MARKER_RUN_ID" --acknowledge-budget-usd 1000'
        ),
        "smoke": (
            "modal run modal_phase_marker.py::smoke --approved-run-id "
            '"$PHASE_MARKER_RUN_ID" --acknowledge-budget-usd 1000'
        ),
        "run_stage_a": (
            "modal run modal_phase_marker.py::run-stage-a --approved-run-id "
            '"$PHASE_MARKER_RUN_ID" --acknowledge-budget-usd 1000'
        ),
    }
    serialized = canonical_json(manifest)
    assert "phase_marker.behavior run" not in serialized
    assert "phase_marker.activations" not in serialized
    assert "phase_marker.interventions" not in serialized


def test_plan_cli_prints_approval_action_manifest(
    prepared_artifacts: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = prepared_artifacts.parent.parent
    lock = repo_root / "requirements-modal-phase-marker.txt"
    lock.write_text("example==1\n", encoding="utf-8")

    modal_plan.main([
        "plan",
        "--repo-root", str(repo_root),
        "--config", str(CONFIG_PATH),
        "--artifact-root", "artifacts/phase-marker",
        "--dependency-lock", lock.name,
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["action_manifest"]["run_id"] == payload["run_id"]
    assert set(payload["action_manifest"]["external_actions"]) == {
        "stage_inputs", "cache_model", "smoke", "run_stage_a",
    }
