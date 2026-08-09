"""Inert Modal resource declarations for the approved phase-marker pilot."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
MODEL_CACHE_CPU = 4.0
MODEL_CACHE_MEMORY_MIB = 32_768
MODEL_CACHE_TIMEOUT_SECONDS = 7_200
SMOKE_CPU = 2.0
SMOKE_MEMORY_MIB = 8_192
SMOKE_TIMEOUT_SECONDS = 900
STAGE_A_PREFLIGHT_CPU = 2.0
STAGE_A_PREFLIGHT_MEMORY_MIB = 8_192
STAGE_A_PREFLIGHT_TIMEOUT_SECONDS = 7_200
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
PREFLIGHT_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
    "/runs": runs_volume.read_only(),
}
FINALIZER_VOLUMES = {
    "/inputs": inputs_volume.read_only(),
    "/model-cache": model_volume.read_only(),
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
    """Validate Stage A dependencies and return compact identity evidence."""
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


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=14_400,
    startup_timeout=1_200,
    max_containers=2,
    retries=0,
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
    image=gpu_image,
    gpu="H100",
    timeout=14_400,
    startup_timeout=1_200,
    max_containers=2,
    retries=0,
    volumes=GPU_VOLUMES,
)
def run_behavior_remote(remote_payload: Mapping[str, object]) -> dict[str, object]:
    """Score one durable slice (or merge) of the held-out behavior matrix.

    Custody chain: the stage-a summary must hash-verify, its receipts must
    match the summary's recorded ids, and every canonical producer tree must
    byte-match its receipt before the summary's own next_command is executed.
    Each cell publishes its slice never-overwrite so interruptions cost one
    cell, not the run; "merge" reassembles the frozen single output tree.
    """
    import hashlib
    import shlex
    import shutil
    import subprocess
    import tempfile
    import time

    from phase_marker.io import sha256_json

    if not isinstance(remote_payload, Mapping) or set(remote_payload) != {
        "run_id", "expected_summary_artifact_id", "action", "cell_index",
    }:
        raise ValueError("remote behavior payload fields are invalid")
    run_id = remote_payload["run_id"]
    expected_summary = remote_payload["expected_summary_artifact_id"]
    action = remote_payload["action"]
    cell_index = remote_payload["cell_index"]
    if not isinstance(run_id, str) or "/" in run_id or not run_id:
        raise ValueError("behavior run id is invalid")
    if action not in {"cell", "merge"} or (
        action == "cell" and not isinstance(cell_index, int)
    ):
        raise ValueError("behavior action is invalid")

    runs_volume.reload()
    started = time.monotonic()
    run_root = JOB_RUN_MOUNT_ROOT / "runs" / run_id
    summary_path = run_root / "stage-a-summary.json"
    summary = json.loads(summary_path.read_text())
    unsigned = dict(summary)
    summary_artifact_id = unsigned.pop("artifact_id")
    if summary_artifact_id != sha256_json(unsigned):
        raise ValueError("stage-a summary artifact id does not verify")
    if expected_summary and summary_artifact_id != expected_summary:
        raise ValueError("stage-a summary does not match the operator-pinned id")
    if summary.get("stopped_before_behavior") is not True or summary.get("run_id") != run_id:
        raise ValueError("stage-a summary stop contract is invalid")

    arms = ("semantic", "glyph", "dot", "random", "direct", "filler")
    bundle_id: str | None = None
    for stage, expected_ids in (
        ("train", summary["training_receipt_ids"]),
        ("selection", summary["selection_receipt_ids"]),
    ):
        for arm, expected_id in zip(arms, expected_ids, strict=True):
            receipt = json.loads(
                (run_root / "receipts" / "canonical" / stage / f"{arm}.json").read_text()
            )
            if receipt["artifact_id"] != expected_id or receipt["arm"] != arm:
                raise ValueError(f"canonical {stage}/{arm} receipt does not match the summary")
            bundle_id = bundle_id or str(receipt["bundle_id"])
            kind = "checkpoints" if stage == "train" else "checkpoint-selections"
            producer = run_root / "artifacts" / "phase-marker" / kind / "pilot" / "seed-42" / arm
            for relative, expected_hash in zip(
                receipt["expected_outputs"], receipt["output_hashes"], strict=True
            ):
                digest = hashlib.sha256((producer / relative).read_bytes()).hexdigest()
                if digest != expected_hash:
                    raise ValueError(f"canonical bytes changed for {stage}/{arm}: {relative}")

    next_command = summary["next_command"]
    argv = shlex.split(next_command, posix=True)
    if argv[:4] != ["./.venv/bin/python", "-m", "phase_marker.behavior", "run"]:
        raise ValueError("stage-a summary next_command is not the behavior run")
    output_root_position = argv.index("--output-root") + 1

    os.environ["HF_HUB_CACHE"] = str(JOB_MODEL_MOUNT_ROOT / "canonical")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    bundle = JOB_INPUT_MOUNT_ROOT / "bundles" / str(bundle_id)
    canonical = run_root / "artifacts" / "phase-marker"
    cells_root = canonical / "behavior-cells"
    behavior_target = canonical / "raw-generations" / "pilot"
    receipt_target = run_root / "receipts" / "canonical" / "behavior.json"

    if action == "cell":
        cell_target = cells_root / f"cell-{cell_index:02d}"
        if cell_target.is_dir():
            return {
                "schema_version": 1, "run_id": run_id, "action": "cell",
                "cell_index": cell_index, "skipped": True,
            }
        view_output = "artifacts/phase-marker/behavior-cells-staging"
        argv = [*argv, "--cell-indices", str(cell_index)]
        argv[output_root_position] = view_output
    else:
        if behavior_target.exists() or receipt_target.exists():
            raise FileExistsError("behavior output already exists; refusing to overwrite")
        cell_dirs = sorted(cells_root.glob("cell-*")) if cells_root.is_dir() else []
        if not cell_dirs:
            raise ValueError("behavior merge requires published cell slices")
        view_output = "artifacts/phase-marker/raw-generations/pilot"
        argv[0:4] = ["./.venv/bin/python", "-m", "phase_marker.behavior", "merge"]
        argv = [item for item in argv if item != "--allow-test-backend"]
        argv[argv.index("--output-root") + 1] = view_output
        argv.extend(["--cell-roots", *(str(path) for path in cell_dirs)])

    with tempfile.TemporaryDirectory(prefix="phase-marker-behavior-") as temporary:
        view = Path(temporary)
        (view / "configs").mkdir(parents=True)
        (view / "configs/phase-marker-qwen25-7b.toml").symlink_to(
            bundle / "configs/phase-marker-qwen25-7b.toml"
        )
        artifacts = view / "artifacts/phase-marker"
        artifacts.mkdir(parents=True)
        for name, source in (
            ("splits", bundle / "artifacts/phase-marker/splits"),
            ("training-data", bundle / "artifacts/phase-marker/training-data"),
            ("checkpoints", canonical / "checkpoints"),
            ("checkpoint-selections", canonical / "checkpoint-selections"),
        ):
            if not source.is_dir():
                raise ValueError(f"behavior source is missing: {name}")
            (artifacts / name).symlink_to(source, target_is_directory=True)
        (view / ".venv/bin").mkdir(parents=True)
        (view / ".venv/bin/python").symlink_to(CODE_ROOT / ".venv/bin/python")
        (view / "phase_marker").symlink_to(CODE_ROOT / "phase_marker", target_is_directory=True)

        completed = subprocess.run(argv, cwd=view, timeout=13_500)
        if completed.returncode != 0:
            raise RuntimeError(
                f"behavior {action} failed with exit {completed.returncode}"
            )

        produced = view / view_output
        if not produced.is_dir():
            raise ValueError(f"behavior {action} produced no output tree")
        output_hashes: dict[str, str] = {}
        for path in sorted(p for p in produced.rglob("*") if p.is_file()):
            relative = path.relative_to(produced).as_posix()
            output_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not output_hashes:
            raise ValueError(f"behavior {action} produced an empty output tree")
        target = cell_target if action == "cell" else behavior_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(produced, target, symlinks=False)

    result: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "action": action,
        "cell_index": cell_index,
        "output_file_count": len(output_hashes),
        "elapsed_seconds": time.monotonic() - started,
        "skipped": False,
    }
    if action == "merge":
        receipt = {
            "schema_version": 1,
            "stage": "behavior",
            "run_id": run_id,
            "stage_a_summary_artifact_id": summary_artifact_id,
            "command": shlex.join(argv),
            "elapsed_seconds": result["elapsed_seconds"],
            "output_hashes": output_hashes,
            "execution_provenance": _collect_modal_execution_provenance(
                "run_behavior_remote"
            ),
        }
        receipt["artifact_id"] = sha256_json(receipt)
        receipt_target.write_text(canonical_json(receipt))
        result["behavior_receipt_artifact_id"] = receipt["artifact_id"]
    runs_volume.commit()
    return result


@app.local_entrypoint(name="run-behavior")
def run_behavior(
    stage_a_run_id: str,
    expected_summary_artifact_id: str = "",
    total_cells: int = 26,
) -> None:
    """Run held-out behavior scoring cell by cell, then merge, resumably."""
    base = {
        "run_id": stage_a_run_id,
        "expected_summary_artifact_id": expected_summary_artifact_id,
    }
    for result in run_behavior_remote.map(
        [
            {**base, "action": "cell", "cell_index": index}
            for index in range(total_cells)
        ]
    ):
        print(canonical_json(result))
    print(
        canonical_json(
            run_behavior_remote.remote(
                {**base, "action": "merge", "cell_index": None}
            )
        )
    )


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
        "finalize_stage_a_remote", "run_behavior_remote",
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
    dependency_function: RemoteFunction,
    runs_client: VolumeClient,
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
    if not isinstance(approval_payload, Mapping):
        raise ValueError("Stage A requires validated approval")
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
    existing = _preflight_stage_a_outputs(plan, resume=resume, runs_client=runs_client)
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
    if existing.summary is not None:
        return dict(existing.summary)
    apply_approved_app_tags(
        plan, approval_payload=validated_approval, action="run-stage-a"
    )
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


def _validate_stage_a_dependency_evidence(
    plan: PilotPlan,
    approval_payload: Mapping[str, object],
    payload: object,
) -> StageADependencyEvidence:
    """Validate the untrusted compact result of the CPU dependency preflight."""
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


def _validate_successful_smoke_receipt(
    smoke: Mapping[str, object],
    *,
    plan: PilotPlan,
    artifact_id: str,
    model_cache_artifact_id: str,
) -> dict[str, object]:
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
        or type(smoke["schema_version"]) is not int
        or smoke["schema_version"] != 1
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
    return dict(smoke)


def _preflight_stage_a_outputs(
    plan: PilotPlan,
    *,
    resume: bool,
    runs_client: VolumeClient,
) -> StageAPreflight:
    summary_bytes = _preflight_stage_a_namespace(
        plan, resume=resume, runs_client=runs_client
    )
    existing: dict[str, dict[str, dict[str, object]]] = {
        "train": {},
        "selection": {},
    }
    plan_payload = pilot_plan_payload(plan)
    for stage in ("train", "selection"):
        for job in plan.jobs:
            receipt_path = _volume_canonical_receipt_path(
                plan.run_id, stage, job.arm
            )
            producer_path = _volume_producer_path(plan.run_id, stage, job.arm)
            receipt = _read_volume_file_optional(runs_client, receipt_path)
            entries = _list_volume_files_optional(runs_client, producer_path)
            producer_present = bool(entries)
            receipt_present = receipt is not None
            if not producer_present and not receipt_present:
                continue
            if not resume:
                raise FileExistsError(
                    f"canonical Stage A output already exists for "
                    f"{stage}/{job.arm}; use --resume"
                )
            if producer_present is not receipt_present:
                raise ValueError(
                    f"partial canonical output for {stage}/{job.arm}; "
                    "start a new content-bound run"
                )
            assert receipt is not None
            existing[stage][job.arm] = _validate_volume_canonical_output(
                plan_payload=plan_payload,
                job=job,
                stage=stage,
                receipt_bytes=receipt,
                entries=entries,
                producer_path=producer_path,
                runs_client=runs_client,
                local_input_root=Path(plan.local_repo_root),
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
    receipt_paths = {
        _volume_canonical_receipt_path(plan.run_id, stage, job.arm)
        for stage in ("train", "selection")
        for job in plan.jobs
    }
    summary_path = f"{run_root}/stage-a-summary.json"
    provenance_path = f"{run_root}/provenance/input-bundle-manifest.json"
    expected = {*producer_roots, *receipt_paths, summary_path, provenance_path}
    ignored_roots = (
        f"{run_root}/attempts",
        f"{run_root}/receipts/attempts",
        f"{run_root}/receipts/smoke",
    )
    summary_seen = False
    for entry in entries:
        raw_path = entry if isinstance(entry, str) else getattr(entry, "path", None)
        if not isinstance(raw_path, str):
            raise ValueError("Stage A namespace contains an invalid path")
        path = "/" + raw_path.lstrip("/")
        if path == run_root or any(
            path == root or path.startswith(root + "/")
            for root in ignored_roots
        ):
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
    """Read every completed arm through fresh Volume RPCs and bind its receipt."""
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
                f"canonical output is missing after {stage} publication for {job.arm}"
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


def _read_volume_file_optional(client: VolumeClient, path: str) -> bytes | None:
    try:
        return b"".join(client.read_file(path))
    except FileNotFoundError:
        return None


def _list_volume_files_optional(client: VolumeClient, path: str) -> tuple[object, ...]:
    try:
        return tuple(client.listdir(path, recursive=True))
    except (FileNotFoundError, modal.exception.NotFoundError):
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
        dependency_function=preflight_stage_a_remote,
        runs_client=runs_volume,
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
