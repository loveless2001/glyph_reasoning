"""Pure, immutable planning for the excluded seed-42 Modal pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shlex

from phase_marker.modal_artifacts import (
    BundleFile,
    InputBundle,
    build_input_bundle,
    hash_source_tree,
    read_bundle_files_at_root,
    read_regular_file_at,
)
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
_CANONICAL_DEPENDENCY_LOCK_PATH = "requirements-modal-phase-marker.txt"
_APPROVED_ACTIONS = frozenset({"stage-inputs", "cache-model", "smoke", "run-stage-a"})
_SHA256_LENGTH = 64
MODAL_ENVIRONMENT = "main"
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
    canonical_dependency_lock_path: str
    modal_environment: str
    bundle_id: str
    bundle_manifest_artifact_id: str
    bundle_files: tuple[BundleFile, ...]
    resources: StageAResources
    jobs: tuple[PilotJob, ...]
    run_label: str
    plan_digest: str
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
    bundle_files = read_bundle_files_at_root(bundle, root)
    bundle_split_id, *bundle_materialization_ids = _bundle_artifact_ids(
        bundle, bundle_files
    )
    config = ExperimentConfig.from_toml_bytes(
        bundle_files[_PORTABLE_CONFIG_PATH.as_posix()]
    )
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
    run_label = (
        f"pilot-s{config.pilot_seed}-cfg-{config_hash[:8]}"
        f"-split-{bundle_split_id[:8]}-src-{source_hash[:12]}"
    )
    plan = PilotPlan(
        schema_version=1,
        kind=_PILOT_KIND,
        seed=config.pilot_seed,
        config_hash=config_hash,
        split_artifact_id=bundle_split_id,
        materialization_artifact_ids=tuple(bundle_materialization_ids),
        model_revision=QWEN25_7B_TOKENIZER_REVISION,
        source_hash=source_hash,
        dependency_lock_hash=dependency_lock_hash,
        canonical_dependency_lock_path=_CANONICAL_DEPENDENCY_LOCK_PATH,
        modal_environment=MODAL_ENVIRONMENT,
        bundle_id=bundle.bundle_id,
        bundle_manifest_artifact_id=hashlib.sha256(
            (canonical_json(asdict(bundle)) + "\n").encode("utf-8")
        ).hexdigest(),
        bundle_files=bundle.files,
        resources=resources,
        jobs=jobs,
        run_label=run_label,
        plan_digest="",
        run_id="",
        local_repo_root=root,
    )
    plan_digest = pilot_plan_digest(plan)
    run_id = f"{run_label}-plan-{plan_digest}"
    plan = replace(plan, plan_digest=plan_digest, run_id=run_id)
    _validate_run_id(plan)
    return plan


def pilot_plan_digest(plan: PilotPlan) -> str:
    """Hash every canonical workload byte used to authorize this pilot."""
    if not isinstance(plan, PilotPlan):
        raise TypeError("pilot plan digest requires a PilotPlan")
    return hashlib.sha256(
        canonical_json(
            {
                "schema_version": plan.schema_version,
                "kind": plan.kind,
                "seed": plan.seed,
                "config_hash": plan.config_hash,
                "split_artifact_id": plan.split_artifact_id,
                "materialization_artifact_ids": list(
                    plan.materialization_artifact_ids
                ),
                "model_revision": plan.model_revision,
                "source_hash": plan.source_hash,
                "dependency_lock": {
                    "path": plan.canonical_dependency_lock_path,
                    "sha256": plan.dependency_lock_hash,
                },
                "modal_environment": plan.modal_environment,
                "bundle_id": plan.bundle_id,
                "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
                "bundle_files": [asdict(item) for item in plan.bundle_files],
                "resources": asdict(plan.resources),
                "jobs": [asdict(job) for job in plan.jobs],
            }
        ).encode("utf-8")
    ).hexdigest()


def action_approval_digest(
    plan: PilotPlan,
    *,
    action: str,
    resume: bool | None = None,
    smoke_receipt_artifact_id: str | None = None,
    model_cache_artifact_id: str | None = None,
) -> str:
    """Derive one non-transferable approval identity for an external action."""
    _validate_run_id(plan)
    if action not in _APPROVED_ACTIONS:
        raise ValueError("external action is not approved")
    payload: dict[str, object] = {
        "schema_version": 1,
        "plan_digest": plan.plan_digest,
        "action": action,
        "modal_environment": plan.modal_environment,
    }
    evidence = (resume, smoke_receipt_artifact_id, model_cache_artifact_id)
    if action == "run-stage-a":
        if (
            not isinstance(resume, bool)
            or not _is_sha256(smoke_receipt_artifact_id)
            or not _is_sha256(model_cache_artifact_id)
        ):
            raise ValueError("Stage A evidence identities and resume mode are required")
        payload.update(
            {
                "resume": resume,
                "smoke_receipt_artifact_id": smoke_receipt_artifact_id,
                "model_cache_artifact_id": model_cache_artifact_id,
                "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
            }
        )
    elif any(value is not None for value in evidence):
        raise ValueError("non-Stage-A actions do not accept Stage A evidence")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def action_approval_payload(
    plan: PilotPlan,
    *,
    action: str,
    resume: bool | None = None,
    smoke_receipt_artifact_id: str | None = None,
    model_cache_artifact_id: str | None = None,
) -> dict[str, object]:
    """Return the exact approval envelope passed across a remote boundary."""
    digest = action_approval_digest(
        plan,
        action=action,
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": action,
        "plan_digest": plan.plan_digest,
        "modal_environment": plan.modal_environment,
        "approval_digest": digest,
    }
    if action == "run-stage-a":
        payload.update(
            {
                "resume": resume,
                "smoke_receipt_artifact_id": smoke_receipt_artifact_id,
                "model_cache_artifact_id": model_cache_artifact_id,
                "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
            }
        )
    return payload


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
        "canonical_dependency_lock_path": plan.canonical_dependency_lock_path,
        "modal_environment": plan.modal_environment,
        "bundle_id": plan.bundle_id,
        "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
        "bundle_files": [asdict(item) for item in plan.bundle_files],
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
        "run_label": plan.run_label,
        "plan_digest": plan.plan_digest,
        "run_id": plan.run_id,
    }


def approval_action_manifest(plan: PilotPlan) -> dict[str, object]:
    """Return pre-GPU commands while withholding Stage A until evidence review."""
    if not isinstance(plan, PilotPlan):
        raise TypeError("approval action manifest requires a PilotPlan")
    _validate_run_id(plan)

    return {
        "schema_version": 2,
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "bundle_id": plan.bundle_id,
        "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
        "modal_environment": plan.modal_environment,
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
            "stage_inputs": _modal_action_command(
                plan,
                entrypoint="stage-inputs",
                approval=action_approval_payload(plan, action="stage-inputs"),
            ),
            "cache_model": _modal_action_command(
                plan,
                entrypoint="cache-model",
                approval=action_approval_payload(plan, action="cache-model"),
            ),
            "smoke": _modal_action_command(
                plan,
                entrypoint="smoke",
                approval=action_approval_payload(plan, action="smoke"),
            ),
        },
        "withheld_actions": {
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
        },
        "approval_required": True,
        "stopped_before_behavior": True,
        "mechanism_approval_included": False,
    }


def approved_stage_a_action_manifest(
    plan: PilotPlan,
    *,
    smoke_receipt_artifact_id: str,
    model_cache_artifact_id: str,
    resume: bool,
) -> dict[str, object]:
    """Derive the exact inert H100 action only from reviewed dependency IDs."""
    if not isinstance(plan, PilotPlan):
        raise TypeError("Stage A action manifest requires a PilotPlan")
    _validate_run_id(plan)
    approval = action_approval_payload(
        plan,
        action="run-stage-a",
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    extra = [
        "--smoke-receipt-artifact-id", smoke_receipt_artifact_id,
        "--model-cache-artifact-id", model_cache_artifact_id,
    ]
    if resume:
        extra.append("--resume")
    return {
        "schema_version": 1,
        "status": "approval-ready-after-reviewed-dependencies",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "resume": resume,
        "smoke_receipt_artifact_id": smoke_receipt_artifact_id,
        "model_cache_artifact_id": model_cache_artifact_id,
        "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
        "modal_environment": plan.modal_environment,
        "approval": approval,
        "resources": {
            "hardware": plan.resources.hardware,
            "timeout_seconds": plan.resources.timeout_seconds,
            "max_containers": plan.resources.max_containers,
            "stage_a_estimated_spend_usd": (
                plan.resources.stage_a_estimated_spend_usd
            ),
            "spend_cap_usd": plan.resources.spend_cap_usd,
        },
        "external_action": _modal_action_command(
            plan,
            entrypoint="run-stage-a",
            approval=approval,
            extra_arguments=extra,
        ),
        "approval_required": True,
        "stopped_before_behavior": True,
        "mechanism_approval_included": False,
    }


def _modal_action_command(
    plan: PilotPlan,
    *,
    entrypoint: str,
    approval: Mapping[str, object],
    extra_arguments: Sequence[str] = (),
) -> str:
    return shlex.join(
        [
            "modal", "run", "--env", plan.modal_environment,
            f"modal_phase_marker.py::{entrypoint}",
            "--approved-run-id", plan.run_id,
            "--acknowledge-budget-usd", "1000",
            "--repo-root", ".",
            "--approved-plan-digest", plan.plan_digest,
            "--approved-action-digest", str(approval["approval_digest"]),
            *extra_arguments,
        ]
    )


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
        if training_command != shlex.join(expected_training):
            raise ValueError("pilot training command is not the approved form")
        if selection_command != shlex.join(expected_selection):
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


def _bundle_artifact_ids(
    bundle: InputBundle, bundle_files: Mapping[str, bytes],
) -> tuple[str, ...]:
    relative_manifests = (
        "splits/manifest.json",
        *(f"training-data/{arm}.manifest.json" for arm in _EXPECTED_ARMS),
    )
    values: list[str] = []
    for relative_path in relative_manifests:
        bundle_path = f"{_PORTABLE_ARTIFACT_ROOT.as_posix()}/{relative_path}"
        try:
            content = bundle_files[bundle_path]
            payload = json.loads(content.decode("utf-8"))
        except (KeyError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"bundle manifest is invalid: {bundle_path}") from error
        artifact_id = payload.get("artifact_id") if isinstance(payload, dict) else None
        if not _is_sha256(artifact_id):
            raise ValueError(
                f"bundle manifest artifact_id is missing or malformed: {bundle_path}"
            )
        values.append(str(artifact_id))
    actual = tuple(values)
    if actual != bundle.artifact_ids:
        raise ValueError("bundle artifact IDs do not match exact manifest files")
    return actual


def _reject_duplicate_artifact_ids(artifact_ids: tuple[str, ...]) -> None:
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("duplicate artifact ID in bundle")


def _validate_run_id(plan: PilotPlan) -> None:
    expected_label = (
        f"pilot-s42-cfg-{plan.config_hash[:8]}-split-{plan.split_artifact_id[:8]}"
        f"-src-{plan.source_hash[:12]}"
    )
    expected_digest = pilot_plan_digest(plan)
    if (
        plan.canonical_dependency_lock_path != _CANONICAL_DEPENDENCY_LOCK_PATH
        or plan.modal_environment != MODAL_ENVIRONMENT
        or plan.run_label != expected_label
        or plan.plan_digest != expected_digest
        or plan.run_id != f"{expected_label}-plan-{expected_digest}"
    ):
        raise ValueError("pilot run ID is noncanonical")


def _file_sha256(path: Path) -> str:
    target = Path(path)
    return hashlib.sha256(
        read_regular_file_at(
            target.parent, target.name, label="dependency lock file"
        )
    ).hexdigest()


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
    for command in ("plan", "run-id", "stage-a-action"):
        subparser = subcommands.add_parser(command)
        subparser.add_argument("--repo-root", required=True)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--artifact-root", required=True)
        subparser.add_argument("--dependency-lock", required=True)
        if command == "stage-a-action":
            subparser.add_argument("--smoke-receipt-artifact-id", required=True)
            subparser.add_argument("--model-cache-artifact-id", required=True)
            mode = subparser.add_mutually_exclusive_group(required=True)
            mode.add_argument("--fresh", action="store_false", dest="resume")
            mode.add_argument("--resume", action="store_true", dest="resume")
    arguments = parser.parse_args(argv)
    repo_root = Path(arguments.repo_root).resolve()
    config_path = _resolve_beneath_repo(repo_root, arguments.config)
    artifact_root = _resolve_beneath_repo(repo_root, arguments.artifact_root)
    config_path, artifact_root = _require_cli_pilot_paths(
        repo_root, config_path, artifact_root
    )
    dependency_lock = _resolve_beneath_repo(repo_root, arguments.dependency_lock)
    canonical_dependency_lock = repo_root / _CANONICAL_DEPENDENCY_LOCK_PATH
    approved_dependency_lock = canonical_dependency_lock.resolve()
    if dependency_lock != approved_dependency_lock:
        raise ValueError("dependency lock must use the exact canonical dependency lock path")
    plan = build_pilot_plan(
        config_path,
        artifact_root,
        bundle=build_input_bundle(repo_root),
        source_hash=hash_source_tree(repo_root),
        dependency_lock_hash=_file_sha256(canonical_dependency_lock),
    )
    if arguments.command == "run-id":
        print(plan.run_id)
    elif arguments.command == "stage-a-action":
        print(canonical_json(approved_stage_a_action_manifest(
            plan,
            smoke_receipt_artifact_id=arguments.smoke_receipt_artifact_id,
            model_cache_artifact_id=arguments.model_cache_artifact_id,
            resume=arguments.resume,
        )))
    else:
        payload = pilot_plan_payload(plan)
        payload["action_manifest"] = approval_action_manifest(plan)
        print(canonical_json(payload))


if __name__ == "__main__":
    main()
