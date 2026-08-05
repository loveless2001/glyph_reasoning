"""Inert Modal resource declarations for the approved phase-marker pilot."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import modal

from phase_marker.io import canonical_json
from phase_marker.modal_artifacts import (
    InputBundle,
    VolumeClient,
    build_input_bundle,
    cache_model_to_volume,
    execute_pilot_job,
    finalize_stage_a,
    hash_source_tree,
    load_promotion_lease_payload,
    load_attempt_receipt_payload,
    parse_model_cache_manifest_payload,
    read_bundle_files_at_root,
    read_regular_file_at,
    require_clean_tracked_status,
    run_cpu_smoke,
    validate_canonical_job_output,
    validate_canonical_job_semantics,
    validate_action_approval_payload,
    validate_job_receipt_payload,
    validate_stage_a_remote_dependencies,
    load_validated_canonical_stage_a_receipts,
    validate_stage_a_summary,
    sha256_json,
)
from phase_marker.modal_inspection import (
    _decode_json_object,
    _normalized_listed_paths,
    download_evidence_local,
    status_local,
)
from phase_marker.modal_plan import (
    MODAL_ENVIRONMENT,
    PilotPlan,
    action_approval_payload,
    build_pilot_plan,
    pilot_plan_digest,
    pilot_plan_payload,
)
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


APP_NAME = "phase-marker-pilot-stage-a"
STAGE_INPUTS_APP_NAME = "phase-marker-pilot-stage-inputs"
CACHE_MODEL_APP_NAME = "phase-marker-pilot-cache-model"
# Preserve the certified runtime provenance name while isolating the selected
# smoke graph in a distinct App object.
SMOKE_APP_NAME = APP_NAME
# linux/amd64 manifest for nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04.
BASE_IMAGE = (
    "nvidia/cuda@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
)
GPU = "H100"
GPU_TIMEOUT_SECONDS = 14_400
MAX_GPU_CONTAINERS = 2
GPU_STARTUP_TIMEOUT_SECONDS = 1_200
GPU_EPHEMERAL_DISK_MIB = 80 * 1024
MODEL_CACHE_CPU = 4.0
MODEL_CACHE_MEMORY_MIB = 32_768
MODEL_CACHE_TIMEOUT_SECONDS = 7_200
SMOKE_CPU = 2.0
SMOKE_MEMORY_MIB = 8_192
SMOKE_TIMEOUT_SECONDS = 900
VOLUME_NAMES = (
    "phase-marker-pilot-inputs-v1",
    "phase-marker-pilot-model-cache-v1",
    "phase-marker-pilot-runs-v1",
)
_BASE_TAGS = {"experiment": "phase-marker", "run-kind": "pilot", "seed": "42"}
_SHA256_CHARS = frozenset("0123456789abcdef")
_PILOT_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
_RUN_ID_PATTERN = re.compile(
    r"pilot-s42-cfg-[0-9a-f]{8}-split-[0-9a-f]{8}-src-[0-9a-f]{12}"
    r"-plan-[0-9a-f]{64}"
)
_TRAINING_MANIFEST_FIELDS = frozenset({
    "kind", "arm", "seed", "model_id", "model_revision", "tokenizer_revision",
    "config_hash", "dataset_path", "dataset_hash", "data_artifact_id",
    "parent_hashes", "data_parent_hashes", "arguments", "environment",
    "checkpoints", "saved_artifacts", "output_hash",
})
_SELECTION_MANIFEST_FIELDS = frozenset({
    "schema_version", "kind", "config_hash", "run_kind", "arm", "seed",
    "selected_on", "evidence_scope", "origin_verification", "backend",
    "model_id", "model_revision", "criterion", "split_artifact_id",
    "split_manifest_hash", "validation_examples_file",
    "validation_examples_hash", "training_manifest_file",
    "training_manifest_hash", "materialization_artifact_id", "candidates",
    "evidence_file", "evidence_hash", "selected_path",
    "selected_checkpoint_hash", "selected_step", "parent_hashes", "completed",
    "artifact_id",
})
_SELECTION_CRITERION = {
    "primary": "maximize_strict_validation_exact_answer_accuracy",
    "tie_break_1": "higher_mean_gold_answer_logprob",
    "tie_break_2": "earliest_checkpoint_step",
}
_SMOKE_RECEIPT_FIELDS = frozenset({
    "schema_version", "stage", "hardware", "run_id", "source_hash",
    "dependency_lock_hash", "canonical_dependency_lock_path", "bundle_id",
    "bundle_manifest_artifact_id", "bundle_files", "modal_environment",
    "plan_digest", "config_hash", "split_artifact_id",
    "materialization_artifact_ids", "model_revision",
    "model_cache_artifact_id", "imports", "validated", "failure_reason",
    "modal_app_id", "modal_app_name", "modal_function_name",
    "modal_function_call_id", "modal_input_id", "python_version",
    "torch_version", "cuda_runtime_version", "cuda_driver_version",
    "artifact_id",
})
CODE_ROOT = Path("/opt/glyph_reasoning")
INPUT_MOUNT_ROOT = Path("/mnt/inputs")
MODEL_MOUNT_ROOT = Path("/mnt/model")
RUN_MOUNT_ROOT = Path("/mnt/runs")
JOB_INPUT_MOUNT_ROOT = Path("/inputs")
JOB_MODEL_MOUNT_ROOT = Path("/model-cache")
JOB_RUN_MOUNT_ROOT = Path("/runs")
LOCKED_RUNTIME_IMPORTS = (
    "accelerate",
    "datasets",
    "einops",
    "huggingface_hub",
    "modal",
    "numpy",
    "peft",
    "google.protobuf",
    "safetensors",
    "sentencepiece",
    "statsmodels",
    "tokenizers",
    "torch",
    "transformers",
    "vllm",
)


def _ignore_unhashed_phase_source(path: Path) -> bool:
    """Exclude every package path not represented by ``hash_source_tree``."""
    candidate = Path(path)
    return candidate.suffix != ".py" or "__pycache__" in candidate.parts


@runtime_checkable
class RemoteFunction(Protocol):
    """Small callable boundary implemented by Modal function handles."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Invoke the declared function remotely."""

    def map(self, payloads: object) -> object:
        """Invoke the declared function once per ordered payload."""


@dataclass(frozen=True)
class InputStagingPlan:
    """Immutable result of local-byte and current remote-state staging preflight."""

    bundle_id: str
    bundle_root: str
    upload_items: tuple[tuple[str, bytes], ...]
    upload_required: bool


@dataclass(frozen=True)
class StageAPreflight:
    training: dict[str, dict[str, object]]
    selection: dict[str, dict[str, object]]
    summary: dict[str, object] | None
    recoveries: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StageADependencyEvidence:
    """Exact staged identities required before any Stage A side effect."""

    bundle_id: str
    model_cache_artifact_id: str
    smoke_receipt_artifact_id: str
    bundle_manifest_artifact_id: str
    smoke_receipt: dict[str, object]


app = modal.App(
    APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
stage_inputs_app = modal.App(
    STAGE_INPUTS_APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
cache_model_app = modal.App(
    CACHE_MODEL_APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
smoke_app = modal.App(
    SMOKE_APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
inputs_volume = modal.Volume.from_name(
    VOLUME_NAMES[0], environment_name=MODAL_ENVIRONMENT, create_if_missing=False
)
model_volume = modal.Volume.from_name(
    VOLUME_NAMES[1], environment_name=MODAL_ENVIRONMENT, create_if_missing=False
)
runs_volume = modal.Volume.from_name(
    VOLUME_NAMES[2], environment_name=MODAL_ENVIRONMENT, create_if_missing=False
)
cache_model_volume = modal.Volume.from_name(
    VOLUME_NAMES[1], environment_name=MODAL_ENVIRONMENT, create_if_missing=True
)
smoke_runs_volume = modal.Volume.from_name(
    VOLUME_NAMES[2], environment_name=MODAL_ENVIRONMENT, create_if_missing=True
)

GPU_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
    "/runs": runs_volume,
}
FINALIZER_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
    "/runs": runs_volume,
}
RECOVERY_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/runs": runs_volume,
}

gpu_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .pip_install_from_requirements("requirements-modal-phase-marker.txt")
    .add_local_dir(
        "phase_marker",
        "/opt/glyph_reasoning/phase_marker",
        copy=True,
        ignore=_ignore_unhashed_phase_source,
    )
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
    .workdir("/opt/glyph_reasoning")
)
cpu_image = gpu_image


@cache_model_app.function(
    image=cpu_image,
    cpu=MODEL_CACHE_CPU,
    memory=MODEL_CACHE_MEMORY_MIB,
    timeout=MODEL_CACHE_TIMEOUT_SECONDS,
    retries=0,
    volumes={"/model-cache": cache_model_volume},
)
def cache_model_remote(remote_payload: dict[str, object]) -> dict[str, object]:
    """Populate the pinned model cache without allocating a GPU."""
    plan_payload, _approval = _validated_remote_action_payload(
        remote_payload, action="cache-model"
    )
    return cache_model_to_volume(
        plan_payload=plan_payload,
        cache_root=Path("/model-cache"),
        volume=cache_model_volume,
    )


@smoke_app.function(
    image=cpu_image,
    cpu=SMOKE_CPU,
    memory=SMOKE_MEMORY_MIB,
    timeout=SMOKE_TIMEOUT_SECONDS,
    retries=0,
    volumes={
        "/mnt/inputs": inputs_volume.read_only(),
        "/mnt/model": model_volume.read_only(),
        "/mnt/runs": smoke_runs_volume,
    },
)
def smoke_remote(remote_payload: dict[str, object]) -> dict[str, object]:
    """Run the locked CPU-only preflight without constructing model weights."""
    plan_payload, _approval = _validated_remote_action_payload(
        remote_payload, action="smoke"
    )
    return run_cpu_smoke(
        plan_payload=plan_payload,
        code_root=CODE_ROOT,
        input_root=INPUT_MOUNT_ROOT,
        model_root=MODEL_MOUNT_ROOT,
        run_root=RUN_MOUNT_ROOT,
        volume=smoke_runs_volume,
        runtime_imports=LOCKED_RUNTIME_IMPORTS,
        execution_provenance=_collect_modal_execution_provenance("smoke_remote"),
    )


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
    """Execute one immutable training arm inside the approved GPU boundary."""
    return _execute_job("train", job_payload)


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
def run_selection_job(job_payload: dict[str, object]) -> dict[str, object]:
    """Execute one immutable selection arm inside the approved GPU boundary."""
    return _execute_job("selection", job_payload)


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=8_192,
    timeout=900,
    retries=0,
    volumes=FINALIZER_VOLUMES,
)
def finalize_stage_a_remote(remote_payload: Mapping[str, object]) -> dict[str, object]:
    """Validate Stage A prerequisites and publish an inert stop summary."""
    if not isinstance(remote_payload, Mapping) or set(remote_payload) != {
        "plan", "approval", "receipts",
    }:
        raise ValueError("remote finalizer payload fields are invalid")
    plan_payload = remote_payload["plan"]
    approval = remote_payload["approval"]
    receipts = remote_payload["receipts"]
    if (
        not isinstance(plan_payload, Mapping)
        or not isinstance(approval, Mapping)
        or not isinstance(receipts, Sequence)
        or isinstance(receipts, (str, bytes))
    ):
        raise ValueError("remote finalizer payload values are invalid")
    validate_action_approval_payload(
        plan_payload=plan_payload,
        approval_payload=approval,
        action="run-stage-a",
        resume=approval.get("resume"),
        smoke_receipt_artifact_id=approval.get("smoke_receipt_artifact_id"),
        model_cache_artifact_id=approval.get("model_cache_artifact_id"),
    )
    runs_volume.reload()
    canonical_receipts = load_validated_canonical_stage_a_receipts(
        plan_payload=plan_payload,
        approval_payload=approval,
        input_root=JOB_INPUT_MOUNT_ROOT,
        model_root=JOB_MODEL_MOUNT_ROOT,
        run_root=JOB_RUN_MOUNT_ROOT,
    )
    supplied = tuple(receipts)
    if len(supplied) != len(canonical_receipts) or any(
        not isinstance(item, Mapping) or dict(item) != canonical
        for item, canonical in zip(supplied, canonical_receipts, strict=True)
    ):
        raise ValueError("coordinator receipts do not match canonical Stage A bytes")
    return finalize_stage_a(
        plan_payload=plan_payload,
        receipts=canonical_receipts,
        input_root=Path("/inputs"),
        model_root=Path("/model-cache"),
        run_root=Path("/runs"),
        volume=runs_volume,
        execution_provenance=_collect_modal_execution_provenance(
            "finalize_stage_a_remote"
        ),
        stage_a_approval=approval,
    )


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=8_192,
    timeout=900,
    retries=0,
    volumes=RECOVERY_VOLUMES,
)
def recover_stage_a_orphans_remote(
    remote_payload: Mapping[str, object],
) -> dict[str, object]:
    """Quarantine exact resume residue without adopting or deleting its bytes."""
    if not isinstance(remote_payload, Mapping) or set(remote_payload) != {
        "plan", "approval", "recoveries",
    }:
        raise ValueError("remote recovery payload fields are invalid")
    plan_payload = remote_payload["plan"]
    approval = remote_payload["approval"]
    recoveries = remote_payload["recoveries"]
    if (
        not isinstance(plan_payload, Mapping)
        or not isinstance(approval, Mapping)
        or not isinstance(recoveries, Sequence)
        or isinstance(recoveries, (str, bytes))
    ):
        raise ValueError("remote recovery payload values are invalid")
    validate_action_approval_payload(
        plan_payload=plan_payload,
        approval_payload=approval,
        action="run-stage-a",
        resume=approval.get("resume"),
        smoke_receipt_artifact_id=approval.get("smoke_receipt_artifact_id"),
        model_cache_artifact_id=approval.get("model_cache_artifact_id"),
    )
    result = _recover_stage_a_orphans(
        plan_payload=plan_payload,
        recoveries=recoveries,
        input_root=Path("/inputs") / "bundles" / str(plan_payload["bundle_id"]),
        run_mount_root=Path("/runs"),
    )
    runs_volume.commit()
    return result


def _validated_remote_action_payload(
    payload: object, *, action: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate authorization before a direct callable can touch mounts or compute."""
    if not isinstance(payload, Mapping) or set(payload) != {"plan", "approval"}:
        raise ValueError("remote action payload fields are invalid")
    plan = payload["plan"]
    approval = payload["approval"]
    if not isinstance(plan, Mapping) or not isinstance(approval, Mapping):
        raise ValueError("remote action payload values are invalid")
    validated = validate_action_approval_payload(
        plan_payload=plan,
        approval_payload=approval,
        action=action,
    )
    return dict(plan), validated


def _execute_job(stage: str, job_payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(job_payload, dict) or set(job_payload) != {
        "plan", "job", "approval",
    }:
        raise ValueError("remote job payload fields are invalid")
    plan = job_payload["plan"]
    job = job_payload["job"]
    approval = job_payload["approval"]
    if (
        not isinstance(plan, Mapping)
        or not isinstance(job, Mapping)
        or not isinstance(approval, Mapping)
    ):
        raise ValueError("remote job payload values are invalid")
    validate_action_approval_payload(
        plan_payload=plan,
        approval_payload=approval,
        action="run-stage-a",
        resume=approval.get("resume") if isinstance(approval, Mapping) else None,
        smoke_receipt_artifact_id=(
            approval.get("smoke_receipt_artifact_id")
            if isinstance(approval, Mapping)
            else None
        ),
        model_cache_artifact_id=(
            approval.get("model_cache_artifact_id")
            if isinstance(approval, Mapping)
            else None
        ),
    )
    runs_volume.reload()
    if stage == "selection":
        load_validated_canonical_stage_a_receipts(
            plan_payload=plan,
            approval_payload=approval,
            input_root=JOB_INPUT_MOUNT_ROOT,
            model_root=JOB_MODEL_MOUNT_ROOT,
            run_root=JOB_RUN_MOUNT_ROOT,
            stages=("train",),
        )
    else:
        validate_stage_a_remote_dependencies(
            plan_payload=plan,
            approval_payload=approval,
            input_root=JOB_INPUT_MOUNT_ROOT,
            model_root=JOB_MODEL_MOUNT_ROOT,
            run_root=JOB_RUN_MOUNT_ROOT,
        )
    function_name = "run_training_job" if stage == "train" else "run_selection_job"
    return execute_pilot_job(
        stage=stage,
        plan_payload=plan,
        job_payload=job,
        code_root=CODE_ROOT,
        input_root=Path("/inputs"),
        model_root=Path("/model-cache"),
        run_root=Path("/runs"),
        volume=runs_volume,
        execution_provenance=_collect_modal_execution_provenance(function_name),
        stage_a_approval=approval,
    )


def _collect_modal_execution_provenance(function_name: str) -> dict[str, object]:
    """Collect exact Modal invocation and locked runtime versions inside a call."""
    if function_name not in {
        "run_training_job", "run_selection_job", "smoke_remote",
        "finalize_stage_a_remote",
    }:
        raise ValueError("Modal provenance function name is invalid")
    runtime_versions: list[dict[str, str]] = []
    for module_name in LOCKED_RUNTIME_IMPORTS:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"runtime version is unavailable: {module_name}")
        runtime_versions.append({"module": module_name, "version": version})
    torch = importlib.import_module("torch")
    torch_version = getattr(torch, "__version__", None)
    torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    driver = (
        "not-observed-cpu"
        if function_name in {"smoke_remote", "finalize_stage_a_remote"}
        else subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    )
    runtime_app = smoke_app if function_name == "smoke_remote" else app
    runtime_app_name = SMOKE_APP_NAME if function_name == "smoke_remote" else APP_NAME
    app_id = getattr(runtime_app, "app_id", None)
    function_call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    values = (app_id, function_call_id, input_id, torch_version, torch_cuda, driver)
    if any(not isinstance(value, str) or not value for value in values):
        raise RuntimeError("Modal execution provenance is unavailable")
    return {
        "modal_app_id": app_id,
        "modal_app_name": runtime_app_name,
        "modal_function_name": function_name,
        "modal_function_call_id": function_call_id,
        "modal_input_id": input_id,
        "python_version": sys.version,
        "torch_version": torch_version,
        "cuda_runtime_version": torch_cuda,
        "cuda_driver_version": driver,
        "runtime_versions": runtime_versions,
    }


def run_stage_a_local(
    plan: PilotPlan,
    *,
    approved_run_id: str,
    budget_acknowledged: bool,
    resume: bool,
    training_function: RemoteFunction,
    selection_function: RemoteFunction,
    finalizer_function: RemoteFunction,
    recovery_function: RemoteFunction | None = None,
    runs_client: VolumeClient,
    inputs_client: VolumeClient | None = None,
    model_client: VolumeClient | None = None,
    smoke_receipt_artifact_id: str | None = None,
    model_cache_artifact_id: str | None = None,
    approval_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run the approved Stage A graph in frozen order, stopping before behavior."""
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    if not isinstance(resume, bool):
        raise TypeError("resume must be an explicit boolean")
    if (
        inputs_client is None
        or model_client is None
        or not isinstance(approval_payload, Mapping)
    ):
        raise ValueError("Stage A requires validated dependency clients and approval")
    dependency_evidence = preflight_stage_a_dependencies(
        plan,
        inputs_client=inputs_client,
        model_client=model_client,
        runs_client=runs_client,
        smoke_receipt_artifact_id=str(smoke_receipt_artifact_id),
        model_cache_artifact_id=str(model_cache_artifact_id),
    )
    validated_approval = validate_action_approval_payload(
        plan_payload=pilot_plan_payload(plan),
        approval_payload=approval_payload,
        action="run-stage-a",
        resume=resume,
        smoke_receipt_artifact_id=dependency_evidence.smoke_receipt_artifact_id,
        model_cache_artifact_id=dependency_evidence.model_cache_artifact_id,
    )
    existing = _preflight_stage_a_outputs(plan, resume=resume, runs_client=runs_client)
    if existing.recoveries and recovery_function is None:
        raise ValueError("Stage A resume recovery requires its approved CPU callable")
    if existing.summary is not None and (
        existing.summary.get("smoke_receipt_artifact_id")
        != dependency_evidence.smoke_receipt_artifact_id
        or existing.summary.get("model_cache_artifact_id")
        != dependency_evidence.model_cache_artifact_id
        or existing.summary.get("bundle_manifest_artifact_id")
        != dependency_evidence.bundle_manifest_artifact_id
    ):
        raise ValueError(
            "completed Stage A summary does not bind the approved dependencies"
        )
    plan_payload = pilot_plan_payload(plan)
    missing_training = tuple(job for job in plan.jobs if job.arm not in existing.training)
    print(
        canonical_json(
            {
                "operation": "run-stage-a",
                "run_id": plan.run_id,
                "resume": resume,
                "plan_digest": plan.plan_digest,
                "approval_digest": validated_approval["approval_digest"],
                "smoke_receipt_artifact_id": dependency_evidence.smoke_receipt_artifact_id,
                "model_cache_artifact_id": dependency_evidence.model_cache_artifact_id,
                "bundle_manifest_artifact_id": (
                    dependency_evidence.bundle_manifest_artifact_id
                ),
                "modal_environment": plan.modal_environment,
                "recoveries": list(existing.recoveries),
                "missing_training": [
                    {"arm": job.arm, "command": job.training_command}
                    for job in missing_training
                ],
                "missing_selection": [
                    {"arm": job.arm, "command": job.selection_command}
                    for job in plan.jobs
                    if job.arm not in existing.selection
                ],
                "resources": {
                    "hardware": plan.resources.hardware,
                    "timeout_seconds": plan.resources.timeout_seconds,
                    "max_containers": plan.resources.max_containers,
                    "training_gpu_hours": plan.resources.training_gpu_hours,
                    "selection_gpu_hours": plan.resources.selection_gpu_hours,
                    "stage_a_estimated_spend_usd": (
                        plan.resources.stage_a_estimated_spend_usd
                    ),
                    "spend_cap_usd": plan.resources.spend_cap_usd,
                },
            }
        ),
        flush=True,
    )
    if existing.summary is not None and not existing.recoveries:
        return dict(existing.summary)
    apply_approved_app_tags(
        plan, approval_payload=validated_approval, action="run-stage-a"
    )
    if existing.recoveries:
        assert recovery_function is not None
        recovery_result = recovery_function.remote(
            {
                "plan": plan_payload,
                "approval": validated_approval,
                "recoveries": list(existing.recoveries),
            }
        )
        if (
            not isinstance(recovery_result, Mapping)
            or set(recovery_result) != {"schema_version", "run_id", "quarantined"}
            or recovery_result.get("schema_version") != 1
            or recovery_result.get("run_id") != plan.run_id
            or recovery_result.get("quarantined") != list(existing.recoveries)
        ):
            raise ValueError("Stage A recovery result is invalid")
        runs_client.reload()
        recovered = _preflight_stage_a_outputs(
            plan, resume=True, runs_client=runs_client
        )
        if (
            recovered.recoveries
            or recovered.training != existing.training
            or recovered.selection != existing.selection
            or recovered.summary != existing.summary
        ):
            raise ValueError("Stage A recovery did not preserve canonical state")
        existing = recovered
        missing_training = tuple(
            job for job in plan.jobs if job.arm not in existing.training
        )
        if existing.summary is not None:
            return dict(existing.summary)

    training_receipts = dict(existing.training)
    training_payloads = tuple(
        _remote_job_payload(plan_payload, job, validated_approval)
        for job in missing_training
    )
    if training_payloads:
        results = list(training_function.map(training_payloads))
        if len(results) != len(missing_training):
            raise RuntimeError("training map returned an incomplete result set")
        for job, payload, result in zip(
            missing_training, training_payloads, results, strict=True
        ):
            if not isinstance(result, Mapping):
                raise TypeError(f"training remote returned a non-object for {job.arm}")
            training_receipts[job.arm] = validate_job_receipt_payload(
                receipt_payload=result,
                plan_payload=plan_payload,
                job_payload=payload["job"],
                stage="train",
                stage_a_approval=validated_approval,
            )
    _require_complete_receipt_matrix(training_receipts, plan, "training")

    runs_client.reload()
    training_receipts = _revalidate_completed_stage_outputs(
        plan,
        stage="train",
        runs_client=runs_client,
        expected=training_receipts,
    )
    existing_selection = _revalidate_resume_selections(
        plan, resume=resume, runs_client=runs_client, existing=existing.selection
    )
    missing_selection = tuple(
        job for job in plan.jobs if job.arm not in existing_selection
    )
    selection_receipts = dict(existing_selection)
    selection_payloads = tuple(
        _remote_job_payload(plan_payload, job, validated_approval)
        for job in missing_selection
    )
    if selection_payloads:
        results = list(selection_function.map(selection_payloads))
        if len(results) != len(missing_selection):
            raise RuntimeError("selection map returned an incomplete result set")
        for job, payload, result in zip(
            missing_selection, selection_payloads, results, strict=True
        ):
            if not isinstance(result, Mapping):
                raise TypeError(f"selection remote returned a non-object for {job.arm}")
            selection_receipts[job.arm] = validate_job_receipt_payload(
                receipt_payload=result,
                plan_payload=plan_payload,
                job_payload=payload["job"],
                stage="selection",
                stage_a_approval=validated_approval,
            )
    _require_complete_receipt_matrix(selection_receipts, plan, "selection")

    runs_client.reload()
    selection_receipts = _revalidate_completed_stage_outputs(
        plan,
        stage="selection",
        runs_client=runs_client,
        expected=selection_receipts,
    )
    ordered_training = tuple(training_receipts[job.arm] for job in plan.jobs)
    ordered_selection = tuple(selection_receipts[job.arm] for job in plan.jobs)
    summary = finalizer_function.remote(
        {
            "plan": plan_payload,
            "approval": validated_approval,
            "receipts": [*ordered_training, *ordered_selection],
        }
    )
    if not isinstance(summary, Mapping):
        raise TypeError("Stage A finalizer returned a non-object")
    return validate_stage_a_summary(
        summary,
        plan_payload=plan_payload,
        training_receipts=ordered_training,
        selection_receipts=ordered_selection,
        stage_a_approval=validated_approval,
    )


def _remote_job_payload(
    plan_payload: dict[str, object],
    job: object,
    approval_payload: Mapping[str, object],
) -> dict[str, object]:
    payload = asdict(job)
    payload["expected_outputs"] = list(payload["expected_outputs"])
    return {"plan": plan_payload, "job": payload, "approval": dict(approval_payload)}


def preflight_stage_a_dependencies(
    plan: PilotPlan,
    *,
    inputs_client: VolumeClient,
    model_client: VolumeClient,
    runs_client: VolumeClient,
    smoke_receipt_artifact_id: str,
    model_cache_artifact_id: str,
) -> StageADependencyEvidence:
    """Revalidate staged inputs, cache manifest, and exact successful smoke."""
    _validate_tag_plan(plan)
    if not _is_sha256(smoke_receipt_artifact_id) or not _is_sha256(
        model_cache_artifact_id
    ):
        raise ValueError("Stage A dependency artifact identities are invalid")
    # Modal volume reads are snapshot-based.  Cross an explicit reload barrier
    # before validating the exact dependency bytes named by this approval.
    inputs_client.reload()
    model_client.reload()
    runs_client.reload()
    bundle = build_input_bundle(_plan_repo_root(plan))
    staging = preflight_inputs_local(
        bundle,
        inputs_client,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )
    if staging.upload_required:
        raise ValueError("Stage A requires the exact staged input bundle")

    snapshot_root = (
        "/canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        f"{plan.model_revision}"
    )
    manifest_path = (
        "/canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        f"{plan.model_revision}.manifest.json"
    )
    manifest_bytes = _read_volume_file_optional(model_client, manifest_path)
    if manifest_bytes is None:
        raise ValueError("Stage A model-cache manifest is missing")
    manifest_payload = _decode_json_object(manifest_bytes, "model-cache manifest")
    manifest = parse_model_cache_manifest_payload(manifest_payload)
    if manifest.artifact_id != model_cache_artifact_id:
        raise ValueError("Stage A model-cache artifact identity is invalid")
    expected_model_paths = {
        f"{snapshot_root}/{item.path}" for item in manifest.files
    }
    listed_model_paths = _normalized_listed_paths(
        _list_volume_files_optional(model_client, snapshot_root), snapshot_root
    )
    if listed_model_paths != expected_model_paths:
        raise ValueError("Stage A model-cache file set does not match its manifest")
    for item in manifest.files:
        remote_path = f"{snapshot_root}/{item.path}"
        content = _read_volume_file_optional(model_client, remote_path)
        if (
            content is None
            or len(content) != item.size
            or _sha256_bytes(content) != item.sha256
        ):
            raise ValueError(f"Stage A model-cache file hash mismatch: {item.path}")

    smoke_path = (
        f"/runs/{plan.run_id}/receipts/smoke/"
        f"{smoke_receipt_artifact_id}.json"
    )
    smoke_bytes = _read_volume_file_optional(runs_client, smoke_path)
    if smoke_bytes is None:
        raise ValueError("Stage A smoke receipt is missing")
    smoke = _decode_json_object(smoke_bytes, "smoke receipt")
    _validate_successful_smoke_receipt(
        smoke,
        plan=plan,
        artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    provenance_path = (
        f"/runs/{plan.run_id}/provenance/input-bundle-manifest.json"
    )
    provenance_bytes = _read_volume_file_optional(runs_client, provenance_path)
    expected_manifest = (canonical_json(asdict(bundle)) + "\n").encode("utf-8")
    if (
        provenance_bytes != expected_manifest
        or _sha256_bytes(expected_manifest) != plan.bundle_manifest_artifact_id
        or smoke.get("bundle_manifest_artifact_id")
        != plan.bundle_manifest_artifact_id
    ):
        raise ValueError("Stage A input bundle provenance is invalid")
    return StageADependencyEvidence(
        bundle_id=bundle.bundle_id,
        model_cache_artifact_id=model_cache_artifact_id,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        bundle_manifest_artifact_id=plan.bundle_manifest_artifact_id,
        smoke_receipt=smoke,
    )


def _validate_successful_smoke_receipt(
    smoke: Mapping[str, object],
    *,
    plan: PilotPlan,
    artifact_id: str,
    model_cache_artifact_id: str,
) -> None:
    unsigned = dict(smoke)
    receipt_artifact_id = unsigned.pop("artifact_id", None)
    imports = smoke.get("imports")
    provenance_fields = (
        "modal_app_id",
        "modal_app_name",
        "modal_function_name",
        "modal_function_call_id",
        "modal_input_id",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
    )
    if (
        set(smoke) != _SMOKE_RECEIPT_FIELDS
        or receipt_artifact_id != artifact_id
        or receipt_artifact_id != sha256_json(unsigned)
        or smoke.get("schema_version") != 1
        or smoke.get("stage") != "smoke"
        or smoke.get("hardware") != "CPU"
        or smoke.get("run_id") != plan.run_id
        or smoke.get("plan_digest") != plan.plan_digest
        or smoke.get("config_hash") != plan.config_hash
        or smoke.get("split_artifact_id") != plan.split_artifact_id
        or smoke.get("materialization_artifact_ids")
        != list(plan.materialization_artifact_ids)
        or smoke.get("source_hash") != plan.source_hash
        or smoke.get("dependency_lock_hash") != plan.dependency_lock_hash
        or smoke.get("canonical_dependency_lock_path")
        != plan.canonical_dependency_lock_path
        or smoke.get("bundle_id") != plan.bundle_id
        or smoke.get("bundle_manifest_artifact_id")
        != plan.bundle_manifest_artifact_id
        or smoke.get("bundle_files")
        != [asdict(item) for item in plan.bundle_files]
        or smoke.get("modal_environment") != plan.modal_environment
        or smoke.get("model_revision") != plan.model_revision
        or smoke.get("model_cache_artifact_id") != model_cache_artifact_id
        or smoke.get("validated") is not True
        or smoke.get("failure_reason") is not None
        or smoke.get("modal_app_name") != SMOKE_APP_NAME
        or smoke.get("modal_function_name") != "smoke_remote"
        or any(
            not isinstance(smoke.get(field), str)
            or not smoke.get(field)
            or "\n" in str(smoke.get(field))
            or "\r" in str(smoke.get(field))
            for field in provenance_fields
        )
        or not isinstance(imports, list)
        or len(imports) != len(LOCKED_RUNTIME_IMPORTS)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"module", "version"}
            or item.get("module") != module
            or not isinstance(item.get("version"), str)
            or not item.get("version")
            for module, item in zip(
                LOCKED_RUNTIME_IMPORTS,
                imports if isinstance(imports, list) else [],
                strict=True,
            )
        )
    ):
        raise ValueError("Stage A smoke receipt identity is invalid")


def _preflight_stage_a_outputs(
    plan: PilotPlan,
    *,
    resume: bool,
    runs_client: VolumeClient,
    now: datetime | None = None,
) -> StageAPreflight:
    summary_bytes = _preflight_stage_a_namespace(
        plan, resume=resume, runs_client=runs_client
    )
    existing: dict[str, dict[str, dict[str, object]]] = {
        "train": {},
        "selection": {},
    }
    recoveries: list[dict[str, object]] = []
    plan_payload = pilot_plan_payload(plan)
    observed_at = _validated_utc_now(now)
    for stage in ("train", "selection"):
        for job in plan.jobs:
            receipt_path = _volume_canonical_receipt_path(plan.run_id, stage, job.arm)
            producer_path = _volume_producer_path(plan.run_id, stage, job.arm)
            lock_path = _volume_promotion_lock_path(plan.run_id, stage, job.arm)
            receipt = _read_volume_file_optional(runs_client, receipt_path)
            entries = _list_volume_files_optional(runs_client, producer_path)
            lock = _read_volume_file_optional(runs_client, lock_path)
            producer_present = bool(entries)
            lock_present = lock is not None
            lease: dict[str, object] | None = None
            if lock_present:
                assert lock is not None
                lease = _validated_expired_promotion_lease(
                    lock,
                    plan=plan,
                    stage=stage,
                    arm=job.arm,
                    now=observed_at,
                )
            if receipt is None and not producer_present and not lock_present:
                continue
            if not resume:
                raise FileExistsError(
                    f"canonical Stage A output already exists for {stage}/{job.arm}; use --resume"
                )
            if receipt is not None and not producer_present:
                raise ValueError(
                    f"canonical output is incomplete for {stage}/{job.arm}"
                )
            if receipt is None and not lock_present:
                if producer_present:
                    raise ValueError(
                        f"orphan producer lacks an authenticated expired lease for "
                        f"{stage}/{job.arm}"
                    )
                continue
            if receipt is None:
                assert lease is not None
                recoveries.append(
                    _stage_a_orphan_recovery_spec(
                        plan,
                        stage=stage,
                        arm=job.arm,
                        producer_present=producer_present,
                        receipt_present=False,
                        lock_present=lock_present,
                        move_producer=producer_present,
                        lease=lease,
                    )
                )
                continue
            validated_receipt = _validate_volume_canonical_output(
                plan_payload=plan_payload,
                job=job,
                stage=stage,
                receipt_bytes=receipt,
                entries=entries,
                producer_path=producer_path,
                runs_client=runs_client,
                local_input_root=Path(plan.local_repo_root),
            )
            existing[stage][job.arm] = validated_receipt
            if lock_present:
                assert lease is not None
                _validate_recovery_lease_receipt_owner(
                    lease, validated_receipt
                )
                recoveries.append(
                    _stage_a_orphan_recovery_spec(
                        plan,
                        stage=stage,
                        arm=job.arm,
                        producer_present=True,
                        receipt_present=True,
                        lock_present=True,
                        move_producer=False,
                        lease=lease,
                    )
                )
    if not set(existing["selection"]).issubset(existing["train"]):
        raise ValueError("canonical selection is missing its canonical training parent")
    summary: dict[str, object] | None = None
    if summary_bytes is not None:
        if len(existing["train"]) != 6 or len(existing["selection"]) != 6:
            raise ValueError("canonical Stage A summary has an incomplete receipt matrix")
        try:
            payload = json.loads(summary_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("canonical Stage A summary is invalid") from error
        if not isinstance(payload, Mapping):
            raise ValueError("canonical Stage A summary is invalid")
        summary = validate_stage_a_summary(
            payload,
            plan_payload=plan_payload,
            training_receipts=tuple(
                existing["train"][job.arm] for job in plan.jobs
            ),
            selection_receipts=tuple(
                existing["selection"][job.arm] for job in plan.jobs
            ),
        )
    return StageAPreflight(
        training=existing["train"],
        selection=existing["selection"],
        summary=summary,
        recoveries=tuple(recoveries),
    )


def _validated_utc_now(value: datetime | None) -> datetime:
    observed = datetime.now(timezone.utc) if value is None else value
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("Stage A recovery clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _validated_expired_promotion_lease(
    content: bytes,
    *,
    plan: PilotPlan,
    stage: str,
    arm: str,
    now: datetime,
) -> dict[str, object]:
    lease = load_promotion_lease_payload(content)
    if (
        lease.get("run_id") != plan.run_id
        or lease.get("plan_digest") != plan.plan_digest
        or lease.get("stage") != stage
        or lease.get("arm") != arm
    ):
        raise ValueError("promotion lease owner identity does not match the plan")
    try:
        expires = datetime.fromisoformat(str(lease["recover_after"]))
    except (TypeError, ValueError) as error:
        raise ValueError("promotion lease expiry is invalid") from error
    expires = expires.astimezone(timezone.utc)
    if expires > now:
        raise FileExistsError(
            f"live promotion lease exists for {stage}/{arm} until "
            f"{lease['recover_after']}"
        )
    return lease


def _validate_recovery_lease_receipt_owner(
    lease: Mapping[str, object], receipt: Mapping[str, object],
) -> None:
    expected = {
        "owner_attempt_id": receipt.get("attempt_id"),
        "owner_receipt_artifact_id": receipt.get("artifact_id"),
        "owner_stage_a_action_digest": receipt.get("stage_a_action_digest"),
        "modal_app_id": receipt.get("modal_app_id"),
        "modal_app_name": receipt.get("modal_app_name"),
        "modal_function_name": receipt.get("modal_function_name"),
        "modal_function_call_id": receipt.get("modal_function_call_id"),
        "modal_input_id": receipt.get("modal_input_id"),
    }
    if any(lease.get(name) != value for name, value in expected.items()):
        raise ValueError("promotion lease is not owned by the canonical receipt")


def _require_expired_recovery_lease(
    lease: Mapping[str, object], *, now: datetime | None = None,
) -> None:
    """Recheck expiry at the remote mutation boundary, independent of preflight."""
    observed = _validated_utc_now(now)
    try:
        recover_after = datetime.fromisoformat(
            str(lease["recover_after"])
        ).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("promotion lease recovery deadline is invalid") from error
    if recover_after > observed:
        raise FileExistsError(
            "live promotion lease cannot be recovered before "
            f"{lease['recover_after']}"
        )


def _preflight_stage_a_namespace(
    plan: PilotPlan, *, resume: bool, runs_client: VolumeClient,
) -> bytes | None:
    run_root = f"/runs/{plan.run_id}"
    entries = _list_volume_files_optional(runs_client, run_root)
    producer_roots = {
        _volume_producer_path(plan.run_id, stage, job.arm)
        for stage in ("train", "selection")
        for job in plan.jobs
    }
    promotion_locks = {
        _volume_promotion_lock_path(plan.run_id, stage, job.arm)
        for stage in ("train", "selection")
        for job in plan.jobs
    }
    receipt_paths = {
        _volume_canonical_receipt_path(plan.run_id, stage, job.arm)
        for stage in ("train", "selection")
        for job in plan.jobs
    }
    summary_path = f"{run_root}/stage-a-summary.json"
    provenance_path = f"{run_root}/provenance/input-bundle-manifest.json"
    expected = {
        *producer_roots, *receipt_paths, *promotion_locks, summary_path,
        provenance_path,
    }
    ignored_prefixes = (
        f"{run_root}/attempts/",
        f"{run_root}/receipts/attempts/",
        f"{run_root}/receipts/smoke/",
    )
    summary_seen = False
    for entry in entries:
        raw_path = entry if isinstance(entry, str) else getattr(entry, "path", None)
        if not isinstance(raw_path, str):
            raise ValueError("Stage A namespace contains an invalid path")
        path = "/" + raw_path.lstrip("/")
        if path == run_root or any(path.startswith(prefix) for prefix in ignored_prefixes):
            continue
        if path == summary_path:
            summary_seen = True
            continue
        if any(
            path == target
            or path.startswith(target.rstrip("/") + "/")
            or target.startswith(path.rstrip("/") + "/")
            for target in expected
        ):
            continue
        raise ValueError(f"unexpected canonical Stage A path: {path}")
    if not summary_seen:
        return None
    if not resume:
        raise FileExistsError("canonical Stage A summary already exists; use --resume")
    summary = _read_volume_file_optional(runs_client, summary_path)
    if summary is None:
        raise ValueError("canonical Stage A summary disappeared during preflight")
    return summary


def _revalidate_resume_selections(
    plan: PilotPlan,
    *,
    resume: bool,
    runs_client: VolumeClient,
    existing: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not resume:
        return dict(existing)
    plan_payload = pilot_plan_payload(plan)
    revalidated: dict[str, dict[str, object]] = {}
    for job in plan.jobs:
        if job.arm not in existing:
            continue
        receipt_path = _volume_canonical_receipt_path(
            plan.run_id, "selection", job.arm
        )
        producer_path = _volume_producer_path(plan.run_id, "selection", job.arm)
        receipt = _read_volume_file_optional(runs_client, receipt_path)
        entries = _list_volume_files_optional(runs_client, producer_path)
        if receipt is None or not entries:
            raise ValueError(
                f"canonical output changed during resume for selection/{job.arm}"
            )
        current = _validate_volume_canonical_output(
            plan_payload=plan_payload,
            job=job,
            stage="selection",
            receipt_bytes=receipt,
            entries=entries,
            producer_path=producer_path,
            runs_client=runs_client,
            local_input_root=Path(plan.local_repo_root),
        )
        if current["artifact_id"] != existing[job.arm]["artifact_id"]:
            raise ValueError(
                f"canonical output changed during resume for selection/{job.arm}"
            )
        revalidated[job.arm] = current
    return revalidated


def _revalidate_completed_stage_outputs(
    plan: PilotPlan,
    *,
    stage: str,
    runs_client: VolumeClient,
    expected: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Read every completed arm from the reloaded volume and bind its receipt."""
    if stage not in {"train", "selection"}:
        raise ValueError("completed Stage A stage is invalid")
    _require_complete_receipt_matrix(expected, plan, stage)
    plan_payload = pilot_plan_payload(plan)
    revalidated: dict[str, dict[str, object]] = {}
    for job in plan.jobs:
        receipt_path = _volume_canonical_receipt_path(plan.run_id, stage, job.arm)
        producer_path = _volume_producer_path(plan.run_id, stage, job.arm)
        receipt = _read_volume_file_optional(runs_client, receipt_path)
        entries = _list_volume_files_optional(runs_client, producer_path)
        if receipt is None or not entries:
            raise ValueError(
                f"canonical output is missing after {stage} reload for {job.arm}"
            )
        current = _validate_volume_canonical_output(
            plan_payload=plan_payload,
            job=job,
            stage=stage,
            receipt_bytes=receipt,
            entries=entries,
            producer_path=producer_path,
            runs_client=runs_client,
            local_input_root=Path(plan.local_repo_root),
        )
        if current["artifact_id"] != expected[job.arm]["artifact_id"]:
            raise ValueError(
                f"canonical receipt artifact does not match {stage} return for {job.arm}"
            )
        revalidated[job.arm] = current
    return revalidated


def _validate_volume_canonical_output(
    *,
    plan_payload: dict[str, object],
    job: object,
    stage: str,
    receipt_bytes: bytes,
    entries: tuple[object, ...],
    producer_path: str,
    runs_client: VolumeClient,
    local_input_root: Path,
) -> dict[str, object]:
    try:
        receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical receipt is invalid") from error
    if not isinstance(receipt_payload, Mapping):
        raise ValueError("canonical receipt is invalid")
    files = _read_volume_producer_files(
        entries=entries,
        producer_path=producer_path,
        runs_client=runs_client,
    )
    payload = asdict(job)
    payload["expected_outputs"] = list(payload["expected_outputs"])
    validated = validate_canonical_job_output(
        receipt_payload=receipt_payload,
        producer_files=files,
        plan_payload=plan_payload,
        job_payload=payload,
        stage=stage,
    )
    training_files: dict[str, bytes] | None = None
    if stage == "selection":
        training_path = _volume_producer_path(
            str(plan_payload["run_id"]), "train", str(payload["arm"])
        )
        training_entries = _list_volume_files_optional(runs_client, training_path)
        if not training_entries:
            raise ValueError("canonical selection is missing its semantic training parent")
        training_files = _read_volume_producer_files(
            entries=training_entries,
            producer_path=training_path,
            runs_client=runs_client,
        )
    validate_canonical_job_semantics(
        stage=stage,
        producer_files=files,
        canonical_training_files=training_files,
        plan_payload=plan_payload,
        job_payload=payload,
        local_input_root=local_input_root,
    )
    return validated


def _read_volume_producer_files(
    *,
    entries: tuple[object, ...],
    producer_path: str,
    runs_client: VolumeClient,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    prefix = producer_path.rstrip("/") + "/"
    for entry in entries:
        raw_path = entry if isinstance(entry, str) else getattr(entry, "path", None)
        if not isinstance(raw_path, str):
            raise ValueError("canonical producer listing contains an invalid path")
        path = "/" + raw_path.lstrip("/")
        kind = _remote_entry_kind(entry)
        if kind == "directory":
            continue
        if kind not in {"file", "unspecified"} or not path.startswith(prefix):
            raise ValueError("canonical producer listing contains an invalid path")
        relative = path[len(prefix):]
        if not relative or relative in files:
            raise ValueError("canonical producer listing contains an invalid path")
        content = _read_volume_file_optional(runs_client, path)
        if content is None:
            raise ValueError("canonical producer file disappeared during validation")
        files[relative] = content
    return files


def _require_complete_receipt_matrix(
    receipts: Mapping[str, Mapping[str, object]], plan: PilotPlan, label: str,
) -> None:
    expected = tuple(job.arm for job in plan.jobs)
    if tuple(arm for arm in expected if arm in receipts) != expected or len(receipts) != 6:
        raise RuntimeError(f"{label} receipt matrix is incomplete")


def _volume_canonical_receipt_path(run_id: str, stage: str, arm: str) -> str:
    return f"/runs/{run_id}/receipts/canonical/{stage}/{arm}.json"


def _volume_producer_path(run_id: str, stage: str, arm: str) -> str:
    kind = "checkpoints" if stage == "train" else "checkpoint-selections"
    return f"/runs/{run_id}/artifacts/phase-marker/{kind}/pilot/seed-42/{arm}"


def _volume_promotion_lock_path(run_id: str, stage: str, arm: str) -> str:
    producer = PurePosixPath(_volume_producer_path(run_id, stage, arm))
    return (producer.parent / f".{arm}.promotion.lock").as_posix()


def _stage_a_orphan_recovery_spec(
    plan: PilotPlan,
    *,
    stage: str,
    arm: str,
    producer_present: bool,
    receipt_present: bool,
    lock_present: bool,
    move_producer: bool,
    lease: Mapping[str, object],
) -> dict[str, object]:
    if (
        stage not in {"train", "selection"}
        or arm not in {job.arm for job in plan.jobs}
        or not isinstance(producer_present, bool)
        or not isinstance(receipt_present, bool)
        or not isinstance(lock_present, bool)
        or not isinstance(move_producer, bool)
        or lock_present is not True
        or not isinstance(lease, Mapping)
        or (receipt_present and not producer_present)
        or move_producer is not (producer_present and not receipt_present)
    ):
        raise ValueError("Stage A orphan recovery state is invalid")
    return {
        "schema_version": 1,
        "stage": stage,
        "arm": arm,
        "producer_path": _volume_producer_path(plan.run_id, stage, arm),
        "receipt_path": _volume_canonical_receipt_path(plan.run_id, stage, arm),
        "lock_path": _volume_promotion_lock_path(plan.run_id, stage, arm),
        "quarantine_root": (
            f"/runs/{plan.run_id}/attempts/orphan-recovery-"
            f"{stage}-{arm}-{uuid.uuid4().hex}"
        ),
        "producer_present": producer_present,
        "receipt_present": receipt_present,
        "lock_present": lock_present,
        "move_producer": move_producer,
        "move_lock": lock_present,
        "lock_artifact_id": lease.get("artifact_id"),
        "lease_owner_attempt_id": lease.get("owner_attempt_id"),
        "lease_modal_function_call_id": lease.get("modal_function_call_id"),
        "lease_modal_input_id": lease.get("modal_input_id"),
    }


def _validated_stage_a_recovery_specs(
    *,
    plan_payload: Mapping[str, object],
    recoveries: Sequence[object],
) -> tuple[dict[str, object], ...]:
    fields = {
        "schema_version", "stage", "arm", "producer_path", "receipt_path",
        "lock_path", "quarantine_root", "producer_present", "receipt_present",
        "lock_present", "move_producer", "move_lock",
        "lock_artifact_id", "lease_owner_attempt_id",
        "lease_modal_function_call_id", "lease_modal_input_id",
    }
    run_id = plan_payload.get("run_id")
    jobs = plan_payload.get("jobs")
    if (
        not isinstance(run_id, str)
        or _RUN_ID_PATTERN.fullmatch(run_id) is None
        or not isinstance(jobs, list)
        or not recoveries
        or len(recoveries) > 12
    ):
        raise ValueError("Stage A recovery plan is invalid")
    approved_arms = {
        item.get("arm")
        for item in jobs
        if isinstance(item, Mapping) and isinstance(item.get("arm"), str)
    }
    normalized: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    quarantine_roots: set[str] = set()
    for item in recoveries:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("Stage A recovery record fields are invalid")
        stage = item.get("stage")
        arm = item.get("arm")
        if stage not in {"train", "selection"} or arm not in approved_arms:
            raise ValueError("Stage A recovery identity is invalid")
        assert isinstance(stage, str) and isinstance(arm, str)
        identity = (stage, arm)
        if identity in identities:
            raise ValueError("Stage A recovery identity is duplicated")
        identities.add(identity)
        expected_producer = _volume_producer_path(run_id, stage, arm)
        expected_receipt = _volume_canonical_receipt_path(run_id, stage, arm)
        expected_lock = _volume_promotion_lock_path(run_id, stage, arm)
        quarantine = item.get("quarantine_root")
        quarantine_pattern = re.compile(
            rf"/runs/{re.escape(run_id)}/attempts/orphan-recovery-"
            rf"{re.escape(stage)}-{re.escape(arm)}-[0-9a-f]{{32}}"
        )
        boolean_fields = (
            "producer_present", "receipt_present", "lock_present",
            "move_producer", "move_lock",
        )
        if (
            item.get("schema_version") != 1
            or item.get("producer_path") != expected_producer
            or item.get("receipt_path") != expected_receipt
            or item.get("lock_path") != expected_lock
            or not isinstance(quarantine, str)
            or quarantine_pattern.fullmatch(quarantine) is None
            or quarantine in quarantine_roots
            or any(not isinstance(item.get(name), bool) for name in boolean_fields)
            or item.get("lock_present") is not True
            or item.get("move_lock") != item.get("lock_present")
            or not _is_sha256(item.get("lock_artifact_id"))
            or any(
                not isinstance(item.get(name), str)
                or not item.get(name)
                or "\n" in str(item.get(name))
                or "\r" in str(item.get(name))
                for name in (
                    "lease_owner_attempt_id",
                    "lease_modal_function_call_id",
                    "lease_modal_input_id",
                )
            )
            or (
                item.get("receipt_present") is True
                and item.get("producer_present") is not True
            )
            or item.get("move_producer")
            is not (
                item.get("producer_present") is True
                and item.get("receipt_present") is False
            )
        ):
            raise ValueError("Stage A recovery record identity is invalid")
        quarantine_roots.add(quarantine)
        normalized.append(dict(item))
    return tuple(normalized)


@dataclass(frozen=True)
class _RecoveryStat:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass
class _RecoveryFile:
    parent_fd: int
    name: str
    snapshot: _RecoveryStat
    content: bytes

    def close(self) -> None:
        os.close(self.parent_fd)


@dataclass
class _RecoveryTree:
    parent_fd: int
    name: str
    root_fd: int
    snapshots: dict[str, _RecoveryStat]
    files: dict[str, bytes]

    def close(self) -> None:
        os.close(self.root_fd)
        os.close(self.parent_fd)


@dataclass
class _ValidatedLocalRecovery:
    spec: dict[str, object]
    producer: _RecoveryTree | None
    receipt: _RecoveryFile | None
    lease: _RecoveryFile
    training: _RecoveryTree | None

    def close(self) -> None:
        for item in (self.producer, self.receipt, self.lease, self.training):
            if item is not None:
                item.close()


_RECOVERY_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
)
_RECOVERY_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _recovery_stat(value: os.stat_result) -> _RecoveryStat:
    return _RecoveryStat(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _same_recovery_inode(left: _RecoveryStat, right: _RecoveryStat) -> bool:
    return (
        left.device,
        left.inode,
        stat.S_IFMT(left.mode),
    ) == (
        right.device,
        right.inode,
        stat.S_IFMT(right.mode),
    )


def _same_recovery_snapshot_after_rename(
    left: _RecoveryStat, right: _RecoveryStat,
) -> bool:
    """Compare stable metadata while allowing rename(2)'s ctime update."""
    return (
        left.device,
        left.inode,
        left.mode,
        left.size,
        left.modified_ns,
    ) == (
        right.device,
        right.inode,
        right.mode,
        right.size,
        right.modified_ns,
    )


def _recovery_volume_parts(volume_path: object) -> tuple[str, ...]:
    if not isinstance(volume_path, str):
        raise ValueError("Stage A recovery path is invalid")
    path = PurePosixPath(volume_path)
    parts = path.parts
    if (
        not path.is_absolute()
        or len(parts) < 2
        or parts[0] != "/"
        or any(part in {"", ".", ".."} or "/" in part for part in parts[1:])
    ):
        raise ValueError("Stage A recovery path is invalid")
    return tuple(parts[1:])


def _open_recovery_directory(parent_fd: int, name: str, label: str) -> int:
    listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(listed.st_mode):
        raise ValueError(f"{label} contains a non-directory path component")
    try:
        child_fd = os.open(name, _RECOVERY_DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"{label} contains a non-directory path component"
            ) from error
        raise
    if not _same_recovery_inode(
        _recovery_stat(listed), _recovery_stat(os.fstat(child_fd))
    ):
        os.close(child_fd)
        raise ValueError(f"{label} changed during descriptor traversal")
    return child_fd


def _open_recovery_parent(
    mount_fd: int, volume_path: object, *, missing_ok: bool = False,
) -> tuple[int, str] | None:
    parts = _recovery_volume_parts(volume_path)
    current_fd = os.dup(mount_fd)
    try:
        for component in parts[:-1]:
            try:
                child_fd = _open_recovery_directory(
                    current_fd, component, "Stage A recovery path"
                )
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise ValueError("Stage A recovery path disappeared") from None
            os.close(current_fd)
            current_fd = child_fd
        result = (current_fd, parts[-1])
        current_fd = -1
        return result
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _stat_recovery_path_optional(
    mount_fd: int, volume_path: object,
) -> _RecoveryStat | None:
    opened = _open_recovery_parent(mount_fd, volume_path, missing_ok=True)
    if opened is None:
        return None
    parent_fd, name = opened
    try:
        try:
            return _recovery_stat(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            )
        except FileNotFoundError:
            return None
    finally:
        os.close(parent_fd)


def _stat_recovery_entry_optional(
    parent_fd: int, name: str,
) -> _RecoveryStat | None:
    try:
        return _recovery_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except FileNotFoundError:
        return None


def _read_recovery_file_at(
    parent_fd: int, name: str, *, label: str,
) -> tuple[bytes, _RecoveryStat]:
    listed = _recovery_stat(
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    )
    if not stat.S_ISREG(listed.mode):
        raise ValueError(f"{label} is not a regular file")
    try:
        file_fd = os.open(name, _RECOVERY_FILE_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"{label} is not a regular file") from error
        raise
    try:
        opened = _recovery_stat(os.fstat(file_fd))
        if opened != listed:
            raise ValueError(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        if _recovery_stat(os.fstat(file_fd)) != listed or len(content) != listed.size:
            raise ValueError(f"{label} changed while it was read")
        current = _recovery_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if current != listed:
            raise ValueError(f"{label} changed while it was read")
        return content, listed
    finally:
        os.close(file_fd)


def _snapshot_recovery_tree(
    directory_fd: int,
    *,
    label: str,
    prefix: str = "",
    read_files: bool,
) -> tuple[dict[str, _RecoveryStat], dict[str, bytes]]:
    before = _recovery_stat(os.fstat(directory_fd))
    snapshots: dict[str, _RecoveryStat] = {prefix: before}
    files: dict[str, bytes] = {}
    names = sorted(os.listdir(directory_fd))
    for name in names:
        if name in {"", ".", ".."} or "/" in name:
            raise ValueError(f"{label} contains an invalid entry")
        relative = f"{prefix}/{name}" if prefix else name
        listed = _recovery_stat(
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if stat.S_ISDIR(listed.mode):
            child_fd = _open_recovery_directory(directory_fd, name, label)
            try:
                child_snapshots, child_files = _snapshot_recovery_tree(
                    child_fd,
                    label=label,
                    prefix=relative,
                    read_files=read_files,
                )
            finally:
                os.close(child_fd)
            snapshots.update(child_snapshots)
            files.update(child_files)
        elif stat.S_ISREG(listed.mode):
            if read_files:
                content, stable = _read_recovery_file_at(
                    directory_fd, name, label=label
                )
                files[relative] = content
                snapshots[relative] = stable
            else:
                snapshots[relative] = listed
        else:
            raise ValueError(
                f"{label} must contain only regular files and directories"
            )
    if sorted(os.listdir(directory_fd)) != names or _recovery_stat(
        os.fstat(directory_fd)
    ) != before:
        raise ValueError(f"{label} changed while it was inspected")
    return snapshots, files


def _open_recovery_tree(
    mount_fd: int, volume_path: object, *, label: str,
) -> _RecoveryTree:
    opened = _open_recovery_parent(mount_fd, volume_path)
    assert opened is not None
    parent_fd, name = opened
    root_fd = -1
    try:
        root_fd = _open_recovery_directory(parent_fd, name, label)
        snapshots, files = _snapshot_recovery_tree(
            root_fd, label=label, read_files=True
        )
        if not files:
            raise ValueError(f"{label} contains no regular files")
        return _RecoveryTree(parent_fd, name, root_fd, snapshots, files)
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _open_recovery_file(
    mount_fd: int, volume_path: object, *, label: str,
) -> _RecoveryFile:
    opened = _open_recovery_parent(mount_fd, volume_path)
    assert opened is not None
    parent_fd, name = opened
    try:
        content, snapshot = _read_recovery_file_at(
            parent_fd, name, label=label
        )
        return _RecoveryFile(parent_fd, name, snapshot, content)
    except BaseException:
        os.close(parent_fd)
        raise


def _verify_recovery_tree(tree: _RecoveryTree, *, label: str) -> None:
    namespace = _stat_recovery_entry_optional(tree.parent_fd, tree.name)
    root = tree.snapshots[""]
    if namespace is None or namespace != root:
        raise ValueError(f"{label} identity changed after validation")
    _verify_recovery_tree_contents(tree, label=label)


def _verify_recovery_tree_contents(tree: _RecoveryTree, *, label: str) -> None:
    snapshots, _files = _snapshot_recovery_tree(
        tree.root_fd, label=label, read_files=False
    )
    current_root = snapshots.pop("")
    expected = dict(tree.snapshots)
    expected_root = expected.pop("")
    # Renaming a directory legitimately updates the directory inode's ctime;
    # its identity and every descendant snapshot must otherwise remain exact.
    if (
        not _same_recovery_inode(expected_root, current_root)
        or snapshots != expected
    ):
        raise ValueError(f"{label} changed after validation")


def _verify_recovery_file(item: _RecoveryFile, *, label: str) -> None:
    content, snapshot = _read_recovery_file_at(item.parent_fd, item.name, label=label)
    if snapshot != item.snapshot or content != item.content:
        raise ValueError(f"{label} changed after validation")


def _create_recovery_quarantine(
    mount_fd: int, volume_path: object,
) -> tuple[int, str, int]:
    """Create one exact quarantine directory through no-follow descriptors."""
    parts = _recovery_volume_parts(volume_path)
    current_fd = os.dup(mount_fd)
    quarantine_fd = -1
    try:
        for component in parts[:-1]:
            try:
                child_fd = _open_recovery_directory(
                    current_fd, component, "Stage A quarantine path"
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                child_fd = _open_recovery_directory(
                    current_fd, component, "Stage A quarantine path"
                )
            os.close(current_fd)
            current_fd = child_fd
        name = parts[-1]
        if _stat_recovery_entry_optional(current_fd, name) is not None:
            raise FileExistsError("Stage A orphan quarantine root already exists")
        os.mkdir(name, 0o700, dir_fd=current_fd)
        listed = _recovery_stat(
            os.stat(name, dir_fd=current_fd, follow_symlinks=False)
        )
        quarantine_fd = _open_recovery_directory(
            current_fd, name, "Stage A quarantine root"
        )
        if listed != _recovery_stat(os.fstat(quarantine_fd)):
            raise OSError("Stage A quarantine root identity changed during creation")
        result = (current_fd, name, quarantine_fd)
        current_fd = -1
        quarantine_fd = -1
        return result
    finally:
        if quarantine_fd >= 0:
            os.close(quarantine_fd)
        if current_fd >= 0:
            os.close(current_fd)


def _validate_local_recovery_state(
    *,
    plan_payload: Mapping[str, object],
    spec: Mapping[str, object],
    mount_fd: int,
    input_root: Path,
) -> _ValidatedLocalRecovery:
    paths = {
        "producer": spec["producer_path"],
        "receipt": spec["receipt_path"],
        "lease": spec["lock_path"],
    }
    observed = {
        name: _stat_recovery_path_optional(mount_fd, path) is not None
        for name, path in paths.items()
    }
    if (
        observed["producer"] is not spec["producer_present"]
        or observed["receipt"] is not spec["receipt_present"]
        or observed["lease"] is not spec["lock_present"]
    ):
        raise ValueError("Stage A recovery state changed after preflight")

    producer: _RecoveryTree | None = None
    receipt: _RecoveryFile | None = None
    lease: _RecoveryFile | None = None
    training: _RecoveryTree | None = None
    try:
        if observed["producer"]:
            producer = _open_recovery_tree(
                mount_fd, paths["producer"], label="orphan producer"
            )
        if observed["receipt"]:
            receipt = _open_recovery_file(
                mount_fd, paths["receipt"], label="canonical recovery receipt"
            )
        lease = _open_recovery_file(
            mount_fd, paths["lease"], label="promotion lease"
        )
        lease_payload = load_promotion_lease_payload(lease.content)
        _require_expired_recovery_lease(lease_payload)
        lease_identity = {
            "run_id": plan_payload.get("run_id"),
            "plan_digest": plan_payload.get("plan_digest"),
            "stage": spec["stage"],
            "arm": spec["arm"],
            "producer_path": spec["producer_path"],
            "receipt_path": spec["receipt_path"],
            "lease_path": spec["lock_path"],
            "artifact_id": spec["lock_artifact_id"],
            "owner_attempt_id": spec["lease_owner_attempt_id"],
            "modal_function_call_id": spec["lease_modal_function_call_id"],
            "modal_input_id": spec["lease_modal_input_id"],
        }
        if any(
            lease_payload.get(name) != value
            for name, value in lease_identity.items()
        ):
            raise ValueError("promotion lease changed after preflight")

        stage = str(spec["stage"])
        arm = str(spec["arm"])
        jobs = plan_payload["jobs"]
        assert isinstance(jobs, list)
        matching_jobs = [
            dict(item)
            for item in jobs
            if isinstance(item, Mapping) and item.get("arm") == arm
        ]
        if len(matching_jobs) != 1:
            raise ValueError("Stage A recovery job is not singular")
        job_payload = matching_jobs[0]
        if stage == "selection" and producer is not None:
            training = _open_recovery_tree(
                mount_fd,
                _volume_producer_path(str(plan_payload["run_id"]), "train", arm),
                label="canonical recovery training parent",
            )
        if producer is not None:
            validate_canonical_job_semantics(
                stage=stage,
                producer_files=producer.files,
                canonical_training_files=(
                    None if training is None else training.files
                ),
                plan_payload=plan_payload,
                job_payload=job_payload,
                local_input_root=input_root,
            )
        if receipt is not None:
            if producer is None:
                raise ValueError("canonical recovery receipt lacks its producer")
            receipt_payload = _decode_json_object(
                receipt.content, "canonical recovery receipt"
            )
            validated_receipt = validate_canonical_job_output(
                receipt_payload=receipt_payload,
                producer_files=producer.files,
                plan_payload=plan_payload,
                job_payload=job_payload,
                stage=stage,
            )
            _validate_recovery_lease_receipt_owner(
                lease_payload, validated_receipt
            )
        assert lease is not None
        return _ValidatedLocalRecovery(
            dict(spec), producer, receipt, lease, training
        )
    except BaseException:
        for item in (producer, receipt, lease, training):
            if item is not None:
                item.close()
        raise


def _recover_stage_a_orphans(
    *,
    plan_payload: Mapping[str, object],
    recoveries: Sequence[object],
    input_root: Path,
    run_mount_root: Path,
) -> dict[str, object]:
    """Validate and move each exact residue into a fresh, never-reused namespace."""
    specs = _validated_stage_a_recovery_specs(
        plan_payload=plan_payload, recoveries=recoveries
    )
    try:
        mount_fd = os.open(Path(run_mount_root), _RECOVERY_DIRECTORY_FLAGS)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("Stage A run mount is not a regular directory") from error
        raise
    validated: list[_ValidatedLocalRecovery] = []
    try:
        for spec in specs:
            if (
                _stat_recovery_path_optional(
                    mount_fd, spec["quarantine_root"]
                )
                is not None
            ):
                raise FileExistsError(
                    "Stage A orphan quarantine root already exists"
                )
            validated.append(
                _validate_local_recovery_state(
                    plan_payload=plan_payload,
                    spec=spec,
                    mount_fd=mount_fd,
                    input_root=Path(input_root),
                )
            )

        quarantined: list[dict[str, object]] = []
        for recovery in validated:
            spec = recovery.spec
            quarantine_parent_fd, quarantine_name, quarantine_fd = (
                _create_recovery_quarantine(
                    mount_fd, spec["quarantine_root"]
                )
            )
            try:
                if recovery.producer is not None:
                    _verify_recovery_tree(
                        recovery.producer, label="orphan producer"
                    )
                if recovery.receipt is not None:
                    _verify_recovery_file(
                        recovery.receipt, label="canonical recovery receipt"
                    )
                if recovery.training is not None:
                    _verify_recovery_tree(
                        recovery.training,
                        label="canonical recovery training parent",
                    )
                _verify_recovery_file(recovery.lease, label="promotion lease")
                _require_expired_recovery_lease(
                    load_promotion_lease_payload(recovery.lease.content)
                )

                if spec["move_producer"] is True:
                    assert recovery.producer is not None
                    os.rename(
                        recovery.producer.name,
                        "producer",
                        src_dir_fd=recovery.producer.parent_fd,
                        dst_dir_fd=quarantine_fd,
                    )
                    source_after = _stat_recovery_entry_optional(
                        recovery.producer.parent_fd, recovery.producer.name
                    )
                    destination = _stat_recovery_entry_optional(
                        quarantine_fd, "producer"
                    )
                    if source_after is not None:
                        raise OSError(
                            "orphan producer quarantine was not atomic"
                        )
                    if (
                        destination is None
                        or not _same_recovery_snapshot_after_rename(
                            recovery.producer.snapshots[""], destination
                        )
                    ):
                        raise OSError(
                            "orphan producer identity changed during quarantine"
                        )
                    _verify_recovery_tree_contents(
                        recovery.producer, label="quarantined producer"
                    )

                # Recheck every non-moved dependency immediately before removing
                # the authenticated lease that guards this canonical namespace.
                if recovery.producer is not None:
                    if spec["move_producer"] is True:
                        _verify_recovery_tree_contents(
                            recovery.producer, label="quarantined producer"
                        )
                    else:
                        _verify_recovery_tree(
                            recovery.producer, label="canonical producer"
                        )
                if recovery.receipt is not None:
                    _verify_recovery_file(
                        recovery.receipt, label="canonical recovery receipt"
                    )
                if recovery.training is not None:
                    _verify_recovery_tree(
                        recovery.training,
                        label="canonical recovery training parent",
                    )
                _verify_recovery_file(recovery.lease, label="promotion lease")
                _require_expired_recovery_lease(
                    load_promotion_lease_payload(recovery.lease.content)
                )

                if spec["move_lock"] is True:
                    os.rename(
                        recovery.lease.name,
                        "promotion.lock",
                        src_dir_fd=recovery.lease.parent_fd,
                        dst_dir_fd=quarantine_fd,
                    )
                    source_after = _stat_recovery_entry_optional(
                        recovery.lease.parent_fd, recovery.lease.name
                    )
                    try:
                        moved_content, moved_snapshot = _read_recovery_file_at(
                            quarantine_fd,
                            "promotion.lock",
                            label="quarantined promotion lease",
                        )
                    except FileNotFoundError as error:
                        raise OSError(
                            "promotion lease identity changed during quarantine"
                        ) from error
                    if source_after is not None:
                        raise OSError(
                            "promotion lease quarantine was not atomic"
                        )
                    if (
                        not _same_recovery_snapshot_after_rename(
                            recovery.lease.snapshot, moved_snapshot
                        )
                        or moved_content != recovery.lease.content
                    ):
                        raise OSError(
                            "promotion lease identity changed during quarantine"
                        )

                # The quarantine directory itself must still be the directory
                # created for this exact recovery record.
                namespace = _stat_recovery_entry_optional(
                    quarantine_parent_fd, quarantine_name
                )
                if namespace is None or not _same_recovery_inode(
                    namespace, _recovery_stat(os.fstat(quarantine_fd))
                ):
                    raise OSError(
                        "Stage A quarantine root identity changed during recovery"
                    )
                quarantined.append(dict(spec))
            finally:
                os.close(quarantine_fd)
                os.close(quarantine_parent_fd)
        return {
            "schema_version": 1,
            "run_id": plan_payload["run_id"],
            "quarantined": quarantined,
        }
    finally:
        for recovery in reversed(validated):
            recovery.close()
        os.close(mount_fd)


def _read_volume_file_optional(client: VolumeClient, path: str) -> bytes | None:
    try:
        return b"".join(client.read_file(path))
    except FileNotFoundError:
        return None


def _list_volume_files_optional(client: VolumeClient, path: str) -> tuple[object, ...]:
    try:
        return tuple(client.listdir(path, recursive=True))
    except FileNotFoundError:
        return ()


def apply_approved_app_tags(
    plan: PilotPlan,
    *,
    approval_payload: Mapping[str, object],
    action: str,
) -> None:
    """Attach a validated full run identity immediately before an approved run."""
    run_tag = _modal_run_tag(plan)
    validate_action_approval_payload(
        plan_payload=pilot_plan_payload(plan),
        approval_payload=approval_payload,
        action=action,
        resume=approval_payload.get("resume") if action == "run-stage-a" else None,
        smoke_receipt_artifact_id=(
            approval_payload.get("smoke_receipt_artifact_id")
            if action == "run-stage-a"
            else None
        ),
        model_cache_artifact_id=(
            approval_payload.get("model_cache_artifact_id")
            if action == "run-stage-a"
            else None
        ),
    )
    action_app = {
        "stage-inputs": stage_inputs_app,
        "cache-model": cache_model_app,
        "smoke": smoke_app,
        "run-stage-a": app,
    }.get(action)
    if action_app is None:
        raise ValueError("Modal app tag action is invalid")
    action_app.set_tags({**_BASE_TAGS, "run-id": run_tag})


def _validate_operator_action_approval(
    plan: PilotPlan,
    *,
    action: str,
    approved_plan_digest: str,
    approved_action_digest: str,
    resume: bool | None = None,
    smoke_receipt_artifact_id: str | None = None,
    model_cache_artifact_id: str | None = None,
) -> dict[str, object]:
    """Bind typed operator input to one exact plan and one exact external action."""
    if approved_plan_digest != plan.plan_digest:
        raise ValueError("full approved plan digest must exactly match the plan")
    expected = action_approval_payload(
        plan,
        action=action,
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    if approved_action_digest != expected["approval_digest"]:
        raise ValueError("exact action approval digest must match the plan and action")
    return validate_action_approval_payload(
        plan_payload=pilot_plan_payload(plan),
        approval_payload=expected,
        action=action,
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )


def _create_authorized_volume(
    name: str,
    *,
    plan_payload: Mapping[str, object],
    approval_payload: Mapping[str, object],
    action: str,
) -> object:
    """Create exactly one action-owned volume only after digest validation."""
    validate_action_approval_payload(
        plan_payload=plan_payload,
        approval_payload=approval_payload,
        action=action,
        resume=approval_payload.get("resume") if action == "run-stage-a" else None,
        smoke_receipt_artifact_id=(
            approval_payload.get("smoke_receipt_artifact_id")
            if action == "run-stage-a"
            else None
        ),
        model_cache_artifact_id=(
            approval_payload.get("model_cache_artifact_id")
            if action == "run-stage-a"
            else None
        ),
    )
    expected_name = {
        "stage-inputs": VOLUME_NAMES[0],
        "cache-model": VOLUME_NAMES[1],
        "smoke": VOLUME_NAMES[2],
    }.get(action)
    if name != expected_name:
        raise ValueError("volume creation target is not authorized for this action")
    volume = modal.Volume.from_name(
        name, environment_name=MODAL_ENVIRONMENT, create_if_missing=True
    )
    volume.hydrate()
    return volume


def stage_inputs_local(
    bundle: InputBundle,
    volume: VolumeClient,
    *,
    approved_run_id: str,
    plan: PilotPlan,
    budget_acknowledged: bool,
) -> dict[str, object]:
    """Compose the read-only staging preflight and narrow upload apply boundary."""
    staging_plan = preflight_inputs_local(
        bundle,
        volume,
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    return _apply_input_staging_plan(staging_plan, volume)


def preflight_inputs_local(
    bundle: InputBundle,
    volume: VolumeClient,
    *,
    approved_run_id: str,
    plan: PilotPlan,
    budget_acknowledged: bool,
) -> InputStagingPlan:
    """Validate local identity and all current remote state without writing."""
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    if plan.bundle_id != bundle.bundle_id:
        raise ValueError("plan and bundle identity do not match")
    repo_root = _plan_repo_root(plan)
    bundle_files = read_bundle_files_at_root(bundle, repo_root)

    bundle_root = f"/bundles/{bundle.bundle_id}"
    upload_bytes = {
        f"{bundle_root}/{item.path}": bundle_files[item.path]
        for item in bundle.files
    }
    upload_bytes[f"{bundle_root}/bundle-manifest.json"] = (
        canonical_json(asdict(bundle)) + "\n"
    ).encode("utf-8")
    _validate_local_upload_bytes(bundle, upload_bytes, bundle_root)

    try:
        entries = volume.listdir(bundle_root, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        entries = []
    existing_paths = _listed_file_paths(entries, bundle_root, set(upload_bytes))
    if existing_paths:
        if existing_paths != set(upload_bytes):
            raise FileExistsError("conflicting remote bundle is incomplete")
        for remote_path, expected in upload_bytes.items():
            actual = b"".join(volume.read_file(remote_path))
            if actual != expected:
                raise FileExistsError(f"conflicting remote bundle byte: {remote_path}")
        return InputStagingPlan(
            bundle_id=bundle.bundle_id,
            bundle_root=bundle_root,
            upload_items=tuple(upload_bytes.items()),
            upload_required=False,
        )

    return InputStagingPlan(
        bundle_id=bundle.bundle_id,
        bundle_root=bundle_root,
        upload_items=tuple(upload_bytes.items()),
        upload_required=True,
    )


def _apply_input_staging_plan(
    staging_plan: InputStagingPlan, volume: VolumeClient,
) -> dict[str, object]:
    if not isinstance(staging_plan, InputStagingPlan):
        raise TypeError("input staging apply requires an immutable preflight plan")
    if not staging_plan.upload_required:
        return {"bundle_id": staging_plan.bundle_id, "uploaded": False}
    with volume.batch_upload() as batch:
        for remote_path, content in staging_plan.upload_items:
            batch.put_file(io.BytesIO(content), remote_path)
    return {"bundle_id": staging_plan.bundle_id, "uploaded": True}





@stage_inputs_app.local_entrypoint(name="stage-inputs")
def stage_inputs(
    approved_run_id: str,
    acknowledge_budget_usd: float = 0.0,
    repo_root: str = ".",
    approved_plan_digest: str = "",
    approved_action_digest: str = "",
) -> None:
    """Explicitly stage one immutable bundle after full operator approval."""
    bundle, plan = _build_operator_context(Path(repo_root))
    budget_acknowledged = _acknowledge_operator_budget(acknowledge_budget_usd)
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    approval = _validate_operator_action_approval(
        plan,
        action="stage-inputs",
        approved_plan_digest=approved_plan_digest,
        approved_action_digest=approved_action_digest,
    )
    writable_volume: object | None = None
    staging_plan = preflight_inputs_local(
        bundle,
        inputs_volume,
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    if staging_plan.upload_required:
        writable_volume = _create_authorized_volume(
            VOLUME_NAMES[0],
            plan_payload=pilot_plan_payload(plan),
            approval_payload=approval,
            action="stage-inputs",
        )
        staging_plan = preflight_inputs_local(
            bundle,
            writable_volume,
            approved_run_id=approved_run_id,
            plan=plan,
            budget_acknowledged=budget_acknowledged,
        )
    print(canonical_json(_staging_plan_payload(staging_plan, plan, approval)))
    if not staging_plan.upload_required:
        print(canonical_json({"bundle_id": bundle.bundle_id, "uploaded": False}))
        return
    assert writable_volume is not None
    apply_approved_app_tags(
        plan, approval_payload=approval, action="stage-inputs"
    )
    result = _apply_input_staging_plan(staging_plan, writable_volume)
    print(canonical_json(result))


@cache_model_app.local_entrypoint(name="cache-model")
def cache_model(
    approved_run_id: str,
    acknowledge_budget_usd: float = 0.0,
    repo_root: str = ".",
    approved_plan_digest: str = "",
    approved_action_digest: str = "",
) -> None:
    """Explicitly invoke the pinned CPU cache population boundary."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    budget_acknowledged = _acknowledge_operator_budget(acknowledge_budget_usd)
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    approval = _validate_operator_action_approval(
        plan,
        action="cache-model",
        approved_plan_digest=approved_plan_digest,
        approved_action_digest=approved_action_digest,
    )
    print(canonical_json({
        "operation": "cache-model",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": approval["approval_digest"],
        "model_revision": plan.model_revision,
        "cpu": MODEL_CACHE_CPU,
        "memory_mib": MODEL_CACHE_MEMORY_MIB,
        "timeout_seconds": MODEL_CACHE_TIMEOUT_SECONDS,
        "destination": f"{VOLUME_NAMES[1]}:/model-cache/canonical",
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "budget_acknowledged_usd": 1_000.0,
    }))
    _create_authorized_volume(
        VOLUME_NAMES[1],
        plan_payload=pilot_plan_payload(plan),
        approval_payload=approval,
        action="cache-model",
    )
    apply_approved_app_tags(
        plan, approval_payload=approval, action="cache-model"
    )
    result = cache_model_remote.remote(
        {"plan": pilot_plan_payload(plan), "approval": approval}
    )
    _print_remote_result(result)


@smoke_app.local_entrypoint(name="smoke")
def smoke(
    approved_run_id: str,
    acknowledge_budget_usd: float = 0.0,
    repo_root: str = ".",
    approved_plan_digest: str = "",
    approved_action_digest: str = "",
) -> None:
    """Explicitly invoke the CPU-only preflight and report its receipt path."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    budget_acknowledged = _acknowledge_operator_budget(acknowledge_budget_usd)
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    approval = _validate_operator_action_approval(
        plan,
        action="smoke",
        approved_plan_digest=approved_plan_digest,
        approved_action_digest=approved_action_digest,
    )
    print(canonical_json({
        "operation": "smoke",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": approval["approval_digest"],
        "hardware": "CPU",
        "cpu": SMOKE_CPU,
        "memory_mib": SMOKE_MEMORY_MIB,
        "timeout_seconds": SMOKE_TIMEOUT_SECONDS,
        "checks": [
            "locked-imports",
            "source-hash",
            "dependency-lock-hash",
            "input-bundle",
            "model-cache",
        ],
        "budget_acknowledged_usd": 1_000.0,
    }))
    _create_authorized_volume(
        VOLUME_NAMES[2],
        plan_payload=pilot_plan_payload(plan),
        approval_payload=approval,
        action="smoke",
    )
    apply_approved_app_tags(plan, approval_payload=approval, action="smoke")
    result = smoke_remote.remote(
        {"plan": pilot_plan_payload(plan), "approval": approval}
    )
    _print_remote_result(result)


@app.local_entrypoint(name="run-stage-a")
def run_stage_a(
    approved_run_id: str,
    acknowledge_budget_usd: float = 0.0,
    resume: bool = False,
    repo_root: str = ".",
    approved_plan_digest: str = "",
    approved_action_digest: str = "",
    smoke_receipt_artifact_id: str = "",
    model_cache_artifact_id: str = "",
) -> None:
    """Explicitly run training, selection, and CPU finalization only."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    budget_acknowledged = _acknowledge_operator_budget(acknowledge_budget_usd)
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    approval = _validate_operator_action_approval(
        plan,
        action="run-stage-a",
        approved_plan_digest=approved_plan_digest,
        approved_action_digest=approved_action_digest,
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    result = run_stage_a_local(
        plan,
        approved_run_id=approved_run_id,
        budget_acknowledged=budget_acknowledged,
        resume=resume,
        training_function=run_training_job,
        selection_function=run_selection_job,
        finalizer_function=finalize_stage_a_remote,
        recovery_function=recover_stage_a_orphans_remote,
        runs_client=runs_volume,
        inputs_client=inputs_volume,
        model_client=model_volume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
        approval_payload=approval,
    )
    print(canonical_json(result))


def _build_operator_context(repo_root: Path) -> tuple[InputBundle, PilotPlan]:
    root = Path(repo_root).resolve()
    status_result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    require_clean_tracked_status(status_result.stdout)
    bundle = build_input_bundle(root)
    lock_bytes = read_regular_file_at(
        root,
        "requirements-modal-phase-marker.txt",
        label="compiled Modal dependency lock",
    )
    plan = build_pilot_plan(
        root / "configs/phase-marker-qwen25-7b.toml",
        root / "artifacts/phase-marker",
        bundle=bundle,
        source_hash=hash_source_tree(root),
        dependency_lock_hash=_sha256_bytes(lock_bytes),
    )
    return bundle, plan


def _print_remote_result(result: object) -> None:
    if not isinstance(result, dict):
        raise TypeError("remote boundary returned a non-object result")
    print(canonical_json(result))


def _staging_plan_payload(
    staging_plan: InputStagingPlan,
    plan: PilotPlan,
    approval_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "operation": "stage-inputs",
        "action": "upload" if staging_plan.upload_required else "no-op",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": approval_payload["approval_digest"],
        "bundle_id": staging_plan.bundle_id,
        "file_count": len(staging_plan.upload_items),
        "destination": f"{VOLUME_NAMES[0]}:{staging_plan.bundle_root}",
        "remote_files": [
            {
                "path": remote_path,
                "size": len(content),
                "sha256": _sha256_bytes(content),
            }
            for remote_path, content in staging_plan.upload_items
        ],
        "budget_acknowledged_usd": 1_000.0,
    }


def _validate_tag_plan(plan: PilotPlan) -> None:
    if not isinstance(plan, PilotPlan):
        raise TypeError("app tags require a PilotPlan")
    if plan.schema_version != 1 or plan.kind != "pilot" or plan.seed != 42:
        raise ValueError("app tags require the approved pilot plan")
    hashes = (
        plan.config_hash,
        plan.split_artifact_id,
        *plan.materialization_artifact_ids,
        plan.source_hash,
        plan.dependency_lock_hash,
        plan.bundle_id,
    )
    if not hashes or any(not _is_sha256(value) for value in hashes):
        raise ValueError("app tags require validated plan hashes")
    expected_run_label = (
        f"pilot-s42-cfg-{plan.config_hash[:8]}"
        f"-split-{plan.split_artifact_id[:8]}-src-{plan.source_hash[:12]}"
    )
    if (
        plan.canonical_dependency_lock_path != "requirements-modal-phase-marker.txt"
        or plan.run_label != expected_run_label
        or plan.plan_digest != pilot_plan_digest(plan)
        or plan.run_id != f"{expected_run_label}-plan-{plan.plan_digest}"
    ):
        raise ValueError("app tags require the canonical run ID")
    if (
        plan.resources.hardware != GPU
        or plan.resources.timeout_seconds != GPU_TIMEOUT_SECONDS
        or plan.resources.max_containers != MAX_GPU_CONTAINERS
        or len(plan.jobs) != 6
        or any(job.seed != plan.seed for job in plan.jobs)
    ):
        raise ValueError("app tags require the approved resource and job envelope")


def _modal_run_tag(plan: PilotPlan) -> str:
    """Encode the full plan identity within Modal's 63-character tag limit."""
    _validate_tag_plan(plan)
    encoded = base64.urlsafe_b64encode(
        bytes.fromhex(plan.plan_digest)
    ).decode("ascii").rstrip("=")
    value = f"s42-{encoded}"
    if len(value) > 63 or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("app run tag is invalid")
    return value


def _validate_operator_approval(
    *, approved_run_id: str, plan: PilotPlan, budget_acknowledged: bool,
) -> None:
    if budget_acknowledged is not True:
        raise ValueError("explicit USD 1000 budget acknowledgement is required")
    if not isinstance(approved_run_id, str) or approved_run_id != plan.run_id:
        raise ValueError("full approved run ID must exactly match the plan")
    _validate_tag_plan(plan)


def _acknowledge_operator_budget(value: float) -> bool:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) != 1_000.0
    ):
        raise ValueError("explicit USD 1000 budget acknowledgement is required")
    return True


def _plan_repo_root(plan: PilotPlan) -> Path:
    root = Path(plan.local_repo_root).resolve()
    expected = root / "configs/phase-marker-qwen25-7b.toml"
    if not expected.is_file():
        raise ValueError("pilot plan repository configuration is not approved")
    try:
        argv = shlex.split(plan.jobs[0].training_command)
        config_path = Path(argv[argv.index("--config") + 1])
    except (IndexError, ValueError) as error:
        raise ValueError("pilot plan lacks its approved repository configuration") from error
    if config_path != Path("configs/phase-marker-qwen25-7b.toml"):
        raise ValueError("pilot plan repository configuration is not approved")
    return root


def _validate_local_upload_bytes(
    bundle: InputBundle, upload_bytes: dict[str, bytes], bundle_root: str,
) -> None:
    for item in bundle.files:
        content = upload_bytes[f"{bundle_root}/{item.path}"]
        if len(content) != item.size or _sha256_bytes(content) != item.sha256:
            raise ValueError(f"bundle file changed while staging: {item.path}")


def _listed_file_paths(
    entries: object, bundle_root: str, expected_files: set[str],
) -> set[str]:
    allowed_directories = {
        parent.as_posix()
        for path in expected_files
        for parent in PurePosixPath(path).parents
        if parent.as_posix().startswith(bundle_root)
    }
    found: set[str] = set()
    for entry in entries:
        raw_path = entry if isinstance(entry, str) else getattr(entry, "path", None)
        if not isinstance(raw_path, str):
            raise ValueError("remote bundle listing contains an invalid path")
        path = "/" + raw_path.lstrip("/")
        if path != bundle_root and not path.startswith(bundle_root + "/"):
            raise ValueError("remote path is outside the bundle ID")
        if path in allowed_directories:
            if _remote_entry_kind(entry) == "directory":
                continue
            raise FileExistsError(f"conflicting remote bundle path: {path}")
        if path not in expected_files:
            raise FileExistsError(f"conflicting remote bundle path: {path}")
        if _remote_entry_kind(entry) not in {"file", "unspecified"}:
            raise FileExistsError(f"conflicting remote bundle path: {path}")
        found.add(path)
    return found




def _remote_entry_kind(entry: object) -> str:
    if isinstance(entry, str):
        return "unspecified"
    kind = getattr(entry, "type", None)
    name = getattr(kind, "name", kind)
    if isinstance(name, str):
        normalized = name.lower()
        if normalized in {"file", "directory"}:
            return normalized
    return "unspecified" if kind is None else "other"


def _sha256_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in _SHA256_CHARS for character in value
    )
