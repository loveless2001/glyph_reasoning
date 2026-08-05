from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
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
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    assert plan.run_label == "pilot-s42-cfg-{}-split-{}-src-{}".format(
        plan.config_hash[:8], plan.split_artifact_id[:8], SOURCE_HASH[:12]
    )
    assert plan.run_id == f"{plan.run_label}-plan-{plan.plan_digest}"
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
    assert manifest["schema_version"] == 2
    assert manifest["plan_digest"] == plan.plan_digest
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
    actions = manifest["external_actions"]
    assert isinstance(actions, dict)
    assert set(actions) == {"stage_inputs", "cache_model", "smoke"}
    for key, action, entrypoint in (
        ("stage_inputs", "stage-inputs", "stage-inputs"),
        ("cache_model", "cache-model", "cache-model"),
        ("smoke", "smoke", "smoke"),
    ):
        approval = modal_plan.action_approval_payload(plan, action=action)
        assert shlex.split(str(actions[key])) == [
            "modal", "run", "--env", "main",
            f"modal_phase_marker.py::{entrypoint}",
            "--approved-run-id", plan.run_id,
            "--acknowledge-budget-usd", "1000",
            "--repo-root", ".",
            "--approved-plan-digest", plan.plan_digest,
            "--approved-action-digest", approval["approval_digest"],
        ]
    assert manifest["withheld_actions"] == {
        "run_stage_a": {
            "status": "withheld-pending-reviewed-dependencies",
            "action": "run-stage-a",
            "hardware": "H100",
            "command_included": False,
            "required_evidence": [
                "smoke_receipt_artifact_id",
                "model_cache_artifact_id",
                "resume",
            ],
            "reason": (
                "review the exact successful CPU smoke receipt and model-cache "
                "manifest before deriving a Stage A action approval"
            ),
        }
    }
    serialized = canonical_json(manifest)
    assert "modal_phase_marker.py::run-stage-a" not in serialized
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
        "stage_inputs", "cache_model", "smoke",
    }
    assert payload["action_manifest"]["withheld_actions"]["run_stage_a"][
        "command_included"
    ] is False


@pytest.mark.parametrize(("resume", "mode_flag"), ((False, "--fresh"), (True, "--resume")))
def test_stage_a_action_is_derived_only_from_reviewed_dependency_ids(
    prepared_artifacts: Path,
    capsys: pytest.CaptureFixture[str],
    resume: bool,
    mode_flag: str,
) -> None:
    repo_root = prepared_artifacts.parent.parent
    lock = repo_root / "requirements-modal-phase-marker.txt"
    lock.write_text("example==1\n", encoding="utf-8")
    smoke_id = "3" * 64
    cache_id = "4" * 64

    modal_plan.main([
        "stage-a-action",
        "--repo-root", str(repo_root),
        "--config", str(CONFIG_PATH),
        "--artifact-root", "artifacts/phase-marker",
        "--dependency-lock", lock.name,
        "--smoke-receipt-artifact-id", smoke_id,
        "--model-cache-artifact-id", cache_id,
        mode_flag,
    ])

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "approval-ready-after-reviewed-dependencies"
    assert payload["resume"] is resume
    assert payload["smoke_receipt_artifact_id"] == smoke_id
    assert payload["model_cache_artifact_id"] == cache_id
    command = shlex.split(payload["external_action"])
    assert command[:5] == [
        "modal", "run", "--env", "main", "modal_phase_marker.py::run-stage-a"
    ]
    assert command[command.index("--approved-plan-digest") + 1] == payload[
        "plan_digest"
    ]
    assert command[command.index("--approved-action-digest") + 1] == payload[
        "approval"
    ]["approval_digest"]
    assert command[command.index("--smoke-receipt-artifact-id") + 1] == smoke_id
    assert command[command.index("--model-cache-artifact-id") + 1] == cache_id
    assert ("--resume" in command) is resume


def test_checked_in_operator_surfaces_withhold_executable_stage_a_command() -> None:
    """Would fail if the H100 boundary reappeared before reviewed dependencies."""
    forbidden = "modal run modal_phase_marker.py::run-stage-a"
    for relative in (
        "README.md",
        "docs/superpowers/plans/2026-08-05-phase-marker-modal-pilot.md",
    ):
        assert forbidden not in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_machine_namespace_binds_the_full_canonical_plan_digest(
    prepared_artifacts: Path,
) -> None:
    """Would fail if changed workload bytes could alias a truncated run label."""
    plan = _plan(prepared_artifacts)

    assert len(plan.plan_digest) == 64
    assert plan.run_label == "pilot-s42-cfg-{}-split-{}-src-{}".format(
        plan.config_hash[:8], plan.split_artifact_id[:8], plan.source_hash[:12]
    )
    assert plan.run_id == f"{plan.run_label}-plan-{plan.plan_digest}"
    assert modal_plan.pilot_plan_digest(plan) == plan.plan_digest
    assert plan.canonical_dependency_lock_path == "requirements-modal-phase-marker.txt"
    assert plan.modal_environment == "main"


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda plan: replace(plan, config_hash="a" * 64), id="config"),
        pytest.param(lambda plan: replace(plan, split_artifact_id="b" * 64), id="split"),
        pytest.param(
            lambda plan: replace(
                plan,
                materialization_artifact_ids=("c" * 64, *plan.materialization_artifact_ids[1:]),
            ),
            id="materialization",
        ),
        pytest.param(lambda plan: replace(plan, bundle_id="d" * 64), id="bundle"),
        pytest.param(
            lambda plan: replace(plan, modal_environment="other"),
            id="modal-environment",
        ),
        pytest.param(
            lambda plan: replace(plan, bundle_manifest_artifact_id="1" * 64),
            id="bundle-manifest",
        ),
        pytest.param(
            lambda plan: replace(
                plan,
                bundle_files=(
                    replace(plan.bundle_files[0], size=plan.bundle_files[0].size + 1),
                    *plan.bundle_files[1:],
                ),
            ),
            id="bundle-files",
        ),
        pytest.param(lambda plan: replace(plan, source_hash="e" * 64), id="source"),
        pytest.param(lambda plan: replace(plan, dependency_lock_hash="f" * 64), id="lock"),
        pytest.param(
            lambda plan: replace(plan, canonical_dependency_lock_path="locks/other.txt"),
            id="lock-path",
        ),
        pytest.param(lambda plan: replace(plan, model_revision="0" * 40), id="model"),
        pytest.param(
            lambda plan: replace(
                plan,
                jobs=(replace(plan.jobs[0], training_command=plan.jobs[0].training_command + " "), *plan.jobs[1:]),
            ),
            id="jobs",
        ),
        pytest.param(
            lambda plan: replace(plan, resources=replace(plan.resources, timeout_seconds=1)),
            id="resources",
        ),
    ],
)
def test_full_plan_digest_isolates_every_workload_identity(
    prepared_artifacts: Path,
    mutation: Callable[[modal_plan.PilotPlan], modal_plan.PilotPlan],
) -> None:
    """Would fail if any approved workload field were omitted from plan identity."""
    plan = _plan(prepared_artifacts)

    assert modal_plan.pilot_plan_digest(mutation(plan)) != plan.plan_digest


def test_action_approval_digests_cannot_cross_actions_or_stage_a_evidence(
    prepared_artifacts: Path,
) -> None:
    """Would fail if approval for one side effect authorized a distinct boundary."""
    plan = _plan(prepared_artifacts)
    stage = modal_plan.action_approval_digest(plan, action="stage-inputs")
    cache = modal_plan.action_approval_digest(plan, action="cache-model")
    smoke = modal_plan.action_approval_digest(plan, action="smoke")
    initial = modal_plan.action_approval_digest(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id="3" * 64,
        model_cache_artifact_id="4" * 64,
    )
    resumed = modal_plan.action_approval_digest(
        plan,
        action="run-stage-a",
        resume=True,
        smoke_receipt_artifact_id="3" * 64,
        model_cache_artifact_id="4" * 64,
    )
    changed_smoke = modal_plan.action_approval_digest(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id="5" * 64,
        model_cache_artifact_id="4" * 64,
    )

    assert len({stage, cache, smoke, initial, resumed, changed_smoke}) == 6
    with pytest.raises(ValueError, match="Stage A evidence"):
        modal_plan.action_approval_digest(plan, action="run-stage-a")
    with pytest.raises(ValueError, match="not accept Stage A evidence"):
        modal_plan.action_approval_digest(
            plan, action="smoke", smoke_receipt_artifact_id="3" * 64
        )


def test_modal_environment_changes_plan_and_action_approval_identities(
    prepared_artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an approval could cross the explicit Modal environment."""
    plan = _plan(prepared_artifacts)
    approval_id = modal_plan.action_approval_digest(plan, action="stage-inputs")
    monkeypatch.setattr(modal_plan, "MODAL_ENVIRONMENT", "other")
    changed = _plan(prepared_artifacts)

    assert changed.plan_digest != plan.plan_digest
    assert modal_plan.action_approval_digest(
        changed, action="stage-inputs"
    ) != approval_id
    assert modal_plan.action_approval_payload(
        changed, action="stage-inputs"
    )["modal_environment"] == "other"


@pytest.mark.parametrize("suffix", [" ", "  ", "\t"])
def test_pilot_plan_rejects_command_bytes_with_equivalent_shell_tokenization(
    monkeypatch: pytest.MonkeyPatch,
    prepared_artifacts: Path,
    suffix: str,
) -> None:
    """Would fail if alternate quoting or whitespace shared command approval."""
    real_manifest = modal_plan.build_command_manifest

    def mutated(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        jobs = [dict(job) for job in real_manifest(*args, **kwargs)]
        jobs[0]["command"] = str(jobs[0]["command"]) + suffix
        return tuple(jobs)

    monkeypatch.setattr(modal_plan, "build_command_manifest", mutated)
    with pytest.raises(ValueError, match="approved form"):
        _plan(prepared_artifacts)


def test_cli_rejects_an_alternate_dependency_lock_path(
    prepared_artifacts: Path,
) -> None:
    """Would fail if an arbitrary same-byte lock path shared plan approval."""
    repo_root = prepared_artifacts.parent.parent
    alternate = repo_root / "alternate-lock.txt"
    alternate.write_text("example==1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical dependency lock"):
        modal_plan.main([
            "run-id",
            "--repo-root", str(repo_root),
            "--config", str(CONFIG_PATH),
            "--artifact-root", "artifacts/phase-marker",
            "--dependency-lock", alternate.name,
        ])


def test_cli_rejects_a_symlink_at_the_canonical_dependency_lock_path(
    prepared_artifacts: Path,
) -> None:
    """Would fail if the canonical lock pathname could redirect its trusted read."""
    repo_root = prepared_artifacts.parent.parent
    external = repo_root / "same-lock-bytes.txt"
    external.write_text("example==1\n", encoding="utf-8")
    canonical = repo_root / "requirements-modal-phase-marker.txt"
    canonical.unlink(missing_ok=True)
    canonical.symlink_to(external)

    with pytest.raises(ValueError, match="symlinked|nonregular"):
        modal_plan.main([
            "run-id",
            "--repo-root", str(repo_root),
            "--config", str(CONFIG_PATH),
            "--artifact-root", "artifacts/phase-marker",
            "--dependency-lock", canonical.name,
        ])
