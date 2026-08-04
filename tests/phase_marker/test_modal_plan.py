from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Callable

import pytest

import phase_marker.modal_plan as modal_plan
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
SOURCE_HASH = "1" * 64
LOCK_HASH = "2" * 64


@pytest.fixture
def prepared_artifacts(tmp_path: Path) -> Path:
    config = ExperimentConfig.load(CONFIG_PATH)
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    return tmp_path


def _plan(prepared_artifacts: Path) -> modal_plan.PilotPlan:
    return modal_plan.build_pilot_plan(
        CONFIG_PATH,
        prepared_artifacts,
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
            CONFIG_PATH,
            prepared_artifacts,
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
