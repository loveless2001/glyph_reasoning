"""Pure, immutable planning for the excluded seed-42 Modal pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import shlex

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.pipeline import (
    ApprovalMetadata,
    _validate_materializations,
    _validate_split_manifest,
    build_command_manifest,
)
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


_PILOT_KIND = "pilot"
_EXPECTED_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
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
    resources: StageAResources
    jobs: tuple[PilotJob, ...]
    run_id: str


def build_stage_a_resources() -> StageAResources:
    """Return the frozen Stage A resource and approval envelope."""
    return StageAResources()


def build_pilot_plan(
    config_path: Path,
    artifact_root: Path,
    source_hash: str,
    dependency_lock_hash: str,
) -> PilotPlan:
    """Build an immutable, approval-bound plan without launching any commands."""
    if not _is_sha256(source_hash) or not _is_sha256(dependency_lock_hash):
        raise ValueError("source and dependency lock hashes must be lowercase sha256 values")

    config = ExperimentConfig.load(config_path)
    resources = build_stage_a_resources()
    approval = resources.approval()
    manifest_jobs = build_command_manifest(
        config,
        artifact_root,
        kind=_PILOT_KIND,
        seeds=(config.pilot_seed,),
        config_path=config_path,
        approval=approval,
    )
    jobs = _validate_manifest_jobs(
        manifest_jobs, config, approval, resources, config_path, artifact_root
    )
    split = _validate_split_manifest(artifact_root, config)
    materialization_ids = _validate_materializations(
        artifact_root, config, split.artifact_id
    )
    config_hash = hashlib.sha256(
        canonical_json(asdict(config)).encode("utf-8")
    ).hexdigest()
    run_id = (
        f"pilot-s{config.pilot_seed}-cfg-{config_hash[:8]}"
        f"-split-{split.artifact_id[:8]}-src-{source_hash[:12]}"
    )
    return PilotPlan(
        schema_version=1,
        kind=_PILOT_KIND,
        seed=config.pilot_seed,
        config_hash=config_hash,
        split_artifact_id=split.artifact_id,
        materialization_artifact_ids=materialization_ids,
        model_revision=QWEN25_7B_TOKENIZER_REVISION,
        source_hash=source_hash,
        dependency_lock_hash=dependency_lock_hash,
        resources=resources,
        jobs=jobs,
        run_id=run_id,
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
