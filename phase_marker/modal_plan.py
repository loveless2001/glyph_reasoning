"""Pure, immutable planning for the excluded seed-42 Modal pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Sequence

from phase_marker.modal_artifacts import InputBundle, build_input_bundle, hash_source_tree, validate_bundle_at_root
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.pipeline import (
    ApprovalMetadata,
    build_command_manifest,
)
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


_PILOT_KIND = "pilot"
_EXPECTED_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
_PORTABLE_CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
_PORTABLE_ARTIFACT_ROOT = Path("artifacts/phase-marker")
_SHA256_LENGTH = 64
_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "arm",
        "seed",
        "model_id",
        "model_revision",
        "approval_required",
        "approval_ready",
        "missing_approval_fields",
        "approval",
        "estimated_gpu_hours",
        "command",
        "selection_command",
        "expected_outputs",
    }
)


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
    bundle_id: str
    resources: StageAResources
    jobs: tuple[PilotJob, ...]
    run_id: str
    local_repo_root: Path


def build_stage_a_resources() -> StageAResources:
    """Return the frozen Stage A resource and approval envelope."""
    return StageAResources()


def build_pilot_plan(
    config_path: Path,
    artifact_root: Path,
    *,
    bundle: InputBundle,
    source_hash: str,
    dependency_lock_hash: str,
) -> PilotPlan:
    """Build an immutable, approval-bound plan without launching any commands."""
    if not _is_sha256(source_hash) or not _is_sha256(dependency_lock_hash):
        raise ValueError("source and dependency lock hashes must be lowercase sha256 values")

    root, config_path, artifact_root = _approved_pilot_paths(config_path, artifact_root)
    _reject_duplicate_artifact_ids(bundle.artifact_ids)
    validate_bundle_at_root(bundle, root)
    bundle_split_id, *bundle_materialization_ids = _bundle_artifact_ids(bundle, artifact_root)
    config = ExperimentConfig.load(config_path)
    if config.pilot_seed != 42:
        raise ValueError("pilot plan requires the frozen seed 42")
    resources = build_stage_a_resources()
    approval = resources.approval()
    manifest_jobs = build_command_manifest(
        config,
        _PORTABLE_ARTIFACT_ROOT,
        kind=_PILOT_KIND,
        seeds=(config.pilot_seed,),
        config_path=_PORTABLE_CONFIG_PATH,
        approval=approval,
    )
    jobs = _validate_manifest_jobs(
        manifest_jobs,
        config,
        approval,
        resources,
        _PORTABLE_CONFIG_PATH,
        _PORTABLE_ARTIFACT_ROOT,
    )
    config_hash = hashlib.sha256(
        canonical_json(asdict(config)).encode("utf-8")
    ).hexdigest()
    run_id = (
        f"pilot-s{config.pilot_seed}-cfg-{config_hash[:8]}"
        f"-split-{bundle_split_id[:8]}-src-{source_hash[:12]}"
    )
    _validate_run_id(run_id, config_hash, bundle_split_id, source_hash)
    return PilotPlan(
        schema_version=1,
        kind=_PILOT_KIND,
        seed=config.pilot_seed,
        config_hash=config_hash,
        split_artifact_id=bundle_split_id,
        materialization_artifact_ids=tuple(bundle_materialization_ids),
        model_revision=QWEN25_7B_TOKENIZER_REVISION,
        source_hash=source_hash,
        dependency_lock_hash=dependency_lock_hash,
        bundle_id=bundle.bundle_id,
        resources=resources,
        jobs=jobs,
        run_id=run_id,
        local_repo_root=root,
    )


def pilot_plan_payload(plan: PilotPlan) -> dict[str, object]:
    """Return a canonical-JSON-ready representation with lists, not tuples."""
    return {
        "schema_version": plan.schema_version,
        "kind": plan.kind,
        "seed": plan.seed,
        "config_hash": plan.config_hash,
        "split_artifact_id": plan.split_artifact_id,
        "materialization_artifact_ids": list(plan.materialization_artifact_ids),
        "model_revision": plan.model_revision,
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "bundle_id": plan.bundle_id,
        "resources": {
            "hardware": plan.resources.hardware,
            "timeout_seconds": plan.resources.timeout_seconds,
            "max_containers": plan.resources.max_containers,
            "training_gpu_hours": plan.resources.training_gpu_hours,
            "selection_gpu_hours": plan.resources.selection_gpu_hours,
            "behavior_gpu_hours": plan.resources.behavior_gpu_hours,
            "max_gpu_hours": plan.resources.max_gpu_hours,
            "stage_a_estimated_spend_usd": plan.resources.stage_a_estimated_spend_usd,
            "estimated_spend_usd": plan.resources.estimated_spend_usd,
            "spend_cap_usd": plan.resources.spend_cap_usd,
        },
        "jobs": [
            {
                "arm": job.arm,
                "seed": job.seed,
                "model_revision": job.model_revision,
                "training_command": job.training_command,
                "selection_command": job.selection_command,
                "expected_outputs": list(job.expected_outputs),
            }
            for job in plan.jobs
        ],
        "run_id": plan.run_id,
    }


def approval_action_manifest(plan: PilotPlan) -> dict[str, object]:
    """Return exact external commands as inert, approval-gated handoff data."""
    if not isinstance(plan, PilotPlan):
        raise TypeError("approval action manifest requires a PilotPlan")
    prefix = (
        "modal run modal_phase_marker.py::{entrypoint} --approved-run-id "
        '"$PHASE_MARKER_RUN_ID" --acknowledge-budget-usd 1000'
    )
    return {
        "schema_version": 1,
        "run_id": plan.run_id,
        "bundle_id": plan.bundle_id,
        "model_revision": plan.model_revision,
        "training_job_count": len(plan.jobs),
        "selection_job_count": len(plan.jobs),
        "resources": {
            "hardware": plan.resources.hardware,
            "timeout_seconds": plan.resources.timeout_seconds,
            "max_containers": plan.resources.max_containers,
            "stage_a_estimated_spend_usd": plan.resources.stage_a_estimated_spend_usd,
            "estimated_spend_usd": plan.resources.estimated_spend_usd,
            "spend_cap_usd": plan.resources.spend_cap_usd,
        },
        "external_actions": {
            "stage_inputs": prefix.format(entrypoint="stage-inputs"),
            "cache_model": prefix.format(entrypoint="cache-model"),
            "smoke": prefix.format(entrypoint="smoke"),
            "run_stage_a": prefix.format(entrypoint="run-stage-a"),
        },
        "approval_required": True,
        "stopped_before_behavior": True,
        "mechanism_approval_included": False,
    }


def _validate_manifest_jobs(
    manifest_jobs: tuple[dict[str, object], ...],
    config: ExperimentConfig,
    approval: ApprovalMetadata,
    resources: StageAResources,
    config_path: Path,
    artifact_root: Path,
) -> tuple[PilotJob, ...]:
    if len(manifest_jobs) != len(_EXPECTED_ARMS):
        raise ValueError("pilot command manifest must contain exactly six jobs")

    expected_approval = asdict(approval)
    jobs: list[PilotJob] = []
    for expected_arm, item in zip(_EXPECTED_ARMS, manifest_jobs, strict=True):
        if not isinstance(item, dict) or set(item) != _MANIFEST_FIELDS:
            raise ValueError("pilot command manifest job fields are invalid")
        if item["kind"] != _PILOT_KIND:
            raise ValueError("pilot command manifest kind is invalid")
        if item["arm"] != expected_arm:
            raise ValueError("pilot command manifest arms are not in frozen order")
        if item["seed"] != config.pilot_seed:
            raise ValueError("pilot command manifest seed is invalid")
        if item["model_id"] != config.model_id:
            raise ValueError("pilot command manifest model identity is invalid")
        if item["model_revision"] != QWEN25_7B_TOKENIZER_REVISION:
            raise ValueError("pilot command manifest revision is invalid")
        if (
            item["approval_required"] is not True
            or item["approval_ready"] is not True
            or item["missing_approval_fields"] != []
            or item["approval"] != expected_approval
        ):
            raise ValueError("pilot command manifest approval is incomplete")
        if item["estimated_gpu_hours"] != resources.training_gpu_hours / len(_EXPECTED_ARMS):
            raise ValueError("pilot command manifest training estimate is invalid")

        training_command = _required_command(item.get("command"), "training")
        selection_command = _required_command(item.get("selection_command"), "selection")
        expected_training, expected_selection = _expected_commands(
            config_path, artifact_root, config.pilot_seed, expected_arm
        )
        if shlex.split(training_command) != expected_training:
            raise ValueError("pilot training command is not the approved form")
        if shlex.split(selection_command) != expected_selection:
            raise ValueError("pilot selection command is not the approved form")
        expected_outputs = item["expected_outputs"]
        if (
            not isinstance(expected_outputs, list)
            or expected_outputs
            != _expected_outputs(artifact_root, config.pilot_seed, expected_arm)
        ):
            raise ValueError("pilot command manifest expected outputs are invalid")
        jobs.append(
            PilotJob(
                arm=expected_arm,
                seed=config.pilot_seed,
                model_revision=QWEN25_7B_TOKENIZER_REVISION,
                training_command=training_command,
                selection_command=selection_command,
                expected_outputs=tuple(expected_outputs),
            )
        )
    return tuple(jobs)


def _required_command(value: object, command_type: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"pilot {command_type} command is invalid")
    return value


def _expected_commands(
    config_path: Path, artifact_root: Path, seed: int, arm: str
) -> tuple[list[str], list[str]]:
    output_dir = artifact_root / "checkpoints" / _PILOT_KIND / f"seed-{seed}" / arm
    manifest = output_dir / "run-manifest.json"
    selection_output = (
        artifact_root / "checkpoint-selections" / _PILOT_KIND / f"seed-{seed}" / arm
    )
    return (
        [
            "./.venv/bin/python", "-m", "phase_marker.training", "train",
            "--config", str(config_path), "--arm", arm, "--seed", str(seed),
            "--data", str(artifact_root / "training-data" / f"{arm}.jsonl"),
            "--output-dir", str(output_dir), "--manifest", str(manifest),
        ],
        [
            "./.venv/bin/python", "-m", "phase_marker.behavior", "select",
            "--config", str(config_path), "--kind", _PILOT_KIND, "--seed", str(seed),
            "--arm", arm, "--split-manifest",
            str(artifact_root / "splits" / "manifest.json"), "--validation-examples",
            str(artifact_root / "splits" / "validation.jsonl"), "--training-manifest",
            str(manifest), "--backend", "vllm", "--output", str(selection_output),
        ],
    )


def _expected_outputs(artifact_root: Path, seed: int, arm: str) -> list[str]:
    output_dir = artifact_root / "checkpoints" / _PILOT_KIND / f"seed-{seed}" / arm
    selection_output = (
        artifact_root / "checkpoint-selections" / _PILOT_KIND / f"seed-{seed}" / arm
    )
    return [
        str(output_dir / "adapter_config.json"),
        str(output_dir / "adapter_model.safetensors"),
        str(output_dir / "run-manifest.json"),
        str(selection_output / "manifest.json"),
        str(selection_output / "evidence.jsonl"),
    ]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _approved_pilot_paths(config_path: Path, artifact_root: Path) -> tuple[Path, Path, Path]:
    resolved_config = Path(config_path).resolve()
    if resolved_config.parent.name != "configs":
        raise ValueError("config path must be the approved configuration")
    repo_root = resolved_config.parent.parent
    approved_config = (repo_root / "configs/phase-marker-qwen25-7b.toml").resolve()
    if resolved_config != approved_config:
        raise ValueError("config path must be the approved configuration")
    approved_artifact_root = (repo_root / "artifacts/phase-marker").resolve()
    if Path(artifact_root).resolve() != approved_artifact_root:
        raise ValueError("artifact root must be the approved artifact root")
    return repo_root, approved_config, approved_artifact_root


def _bundle_artifact_ids(bundle: InputBundle, artifact_root: Path) -> tuple[str, ...]:
    relative_manifests = (
        "splits/manifest.json",
        *(f"training-data/{arm}.manifest.json" for arm in _EXPECTED_ARMS),
    )
    values: list[str] = []
    for relative_path in relative_manifests:
        path = Path(artifact_root) / relative_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"bundle manifest is invalid: {path}") from error
        artifact_id = payload.get("artifact_id") if isinstance(payload, dict) else None
        if not _is_sha256(artifact_id):
            raise ValueError(f"bundle manifest artifact_id is missing or malformed: {path}")
        values.append(str(artifact_id))
    actual = tuple(values)
    if actual != bundle.artifact_ids:
        raise ValueError("bundle artifact IDs do not match exact manifest files")
    return actual


def _reject_duplicate_artifact_ids(artifact_ids: tuple[str, ...]) -> None:
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("duplicate artifact ID in bundle")


def _validate_run_id(
    run_id: str, config_hash: str, split_id: str, source_hash: str
) -> None:
    expected = (
        f"pilot-s42-cfg-{config_hash[:8]}-split-{split_id[:8]}-src-{source_hash[:12]}"
    )
    if run_id != expected:
        raise ValueError("pilot run ID is noncanonical")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_beneath_repo(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"path escapes --repo-root: {value}") from error
    return resolved


def _require_cli_pilot_paths(
    repo_root: Path, config_path: Path, artifact_root: Path
) -> tuple[Path, Path]:
    approved_config = (repo_root / "configs/phase-marker-qwen25-7b.toml").resolve()
    if config_path != approved_config:
        raise ValueError("config path must be the --repo-root approved configuration")
    approved_artifact_root = (repo_root / "artifacts/phase-marker").resolve()
    if artifact_root != approved_artifact_root:
        raise ValueError("artifact root must be the --repo-root approved artifact root")
    return approved_config, approved_artifact_root


def main(argv: Sequence[str] | None = None) -> None:
    """Print an immutable pilot plan or its canonical run ID, entirely offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run-id"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--repo-root", required=True)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--artifact-root", required=True)
        subparser.add_argument("--dependency-lock", required=True)
    arguments = parser.parse_args(argv)
    repo_root = Path(arguments.repo_root).resolve()
    config_path = _resolve_beneath_repo(repo_root, arguments.config)
    artifact_root = _resolve_beneath_repo(repo_root, arguments.artifact_root)
    config_path, artifact_root = _require_cli_pilot_paths(
        repo_root, config_path, artifact_root
    )
    dependency_lock = _resolve_beneath_repo(repo_root, arguments.dependency_lock)
    plan = build_pilot_plan(
        config_path,
        artifact_root,
        bundle=build_input_bundle(repo_root),
        source_hash=hash_source_tree(repo_root),
        dependency_lock_hash=_file_sha256(dependency_lock),
    )
    if arguments.command == "run-id":
        print(plan.run_id)
    else:
        payload = pilot_plan_payload(plan)
        payload["action_manifest"] = approval_action_manifest(plan)
        print(canonical_json(payload))


if __name__ == "__main__":
    main()
