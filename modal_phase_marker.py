"""Inert Modal resource declarations for the approved phase-marker pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
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
    require_clean_tracked_status,
    run_cpu_smoke,
    validate_canonical_job_output,
    validate_canonical_job_semantics,
    validate_job_receipt_payload,
    validate_stage_a_summary,
    validate_bundle_at_root,
)
from phase_marker.modal_plan import (
    PilotPlan,
    build_pilot_plan,
    pilot_plan_payload,
)


APP_NAME = "phase-marker-pilot-stage-a"
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
CODE_ROOT = Path("/opt/glyph_reasoning")
INPUT_MOUNT_ROOT = Path("/mnt/inputs")
MODEL_MOUNT_ROOT = Path("/mnt/model")
RUN_MOUNT_ROOT = Path("/mnt/runs")
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


app = modal.App(
    APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
inputs_volume = modal.Volume.from_name(VOLUME_NAMES[0], create_if_missing=True)
model_volume = modal.Volume.from_name(VOLUME_NAMES[1], create_if_missing=True)
runs_volume = modal.Volume.from_name(VOLUME_NAMES[2], create_if_missing=True)

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


@app.function(
    image=cpu_image,
    cpu=MODEL_CACHE_CPU,
    memory=MODEL_CACHE_MEMORY_MIB,
    timeout=MODEL_CACHE_TIMEOUT_SECONDS,
    retries=0,
    volumes={"/model-cache": model_volume},
)
def cache_model_remote(plan_payload: dict[str, object]) -> dict[str, object]:
    """Populate the pinned model cache without allocating a GPU."""
    return cache_model_to_volume(
        plan_payload=plan_payload,
        cache_root=Path("/model-cache"),
        volume=model_volume,
    )


@app.function(
    image=cpu_image,
    cpu=SMOKE_CPU,
    memory=SMOKE_MEMORY_MIB,
    timeout=SMOKE_TIMEOUT_SECONDS,
    retries=0,
    volumes={
        "/mnt/inputs": inputs_volume.read_only(),
        "/mnt/model": model_volume.read_only(),
        "/mnt/runs": runs_volume,
    },
)
def smoke_remote(plan_payload: dict[str, object]) -> dict[str, object]:
    """Run the locked CPU-only preflight without constructing model weights."""
    return run_cpu_smoke(
        plan_payload=plan_payload,
        code_root=CODE_ROOT,
        input_root=INPUT_MOUNT_ROOT,
        model_root=MODEL_MOUNT_ROOT,
        run_root=RUN_MOUNT_ROOT,
        volume=runs_volume,
        runtime_imports=LOCKED_RUNTIME_IMPORTS,
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
def finalize_stage_a_remote(
    plan_payload: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate Stage A prerequisites and publish an inert stop summary."""
    return finalize_stage_a(
        plan_payload=plan_payload,
        receipts=receipts,
        input_root=Path("/inputs"),
        model_root=Path("/model-cache"),
        run_root=Path("/runs"),
        volume=runs_volume,
    )


def _execute_job(stage: str, job_payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(job_payload, dict) or set(job_payload) != {"plan", "job"}:
        raise ValueError("remote job payload fields are invalid")
    plan = job_payload["plan"]
    job = job_payload["job"]
    if not isinstance(plan, Mapping) or not isinstance(job, Mapping):
        raise ValueError("remote job payload values are invalid")
    return execute_pilot_job(
        stage=stage,
        plan_payload=plan,
        job_payload=job,
        code_root=CODE_ROOT,
        input_root=Path("/inputs"),
        model_root=Path("/model-cache"),
        run_root=Path("/runs"),
        volume=runs_volume,
    )


def run_stage_a_local(
    plan: PilotPlan,
    *,
    approved_run_id: str,
    budget_acknowledged: bool,
    resume: bool,
    training_function: RemoteFunction,
    selection_function: RemoteFunction,
    finalizer_function: RemoteFunction,
    runs_client: VolumeClient,
) -> dict[str, object]:
    """Run the approved Stage A graph in frozen order, stopping before behavior."""
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    if not isinstance(resume, bool):
        raise TypeError("resume must be an explicit boolean")
    existing = _preflight_stage_a_outputs(plan, resume=resume, runs_client=runs_client)
    plan_payload = pilot_plan_payload(plan)
    missing_training = tuple(job for job in plan.jobs if job.arm not in existing.training)
    print(
        canonical_json(
            {
                "operation": "run-stage-a",
                "run_id": plan.run_id,
                "resume": resume,
                "missing_training_arms": [job.arm for job in missing_training],
            }
        )
    )
    apply_approved_app_tags(plan)

    training_receipts = dict(existing.training)
    training_payloads = tuple(
        _remote_job_payload(plan_payload, job) for job in missing_training
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
        _remote_job_payload(plan_payload, job) for job in missing_selection
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
        plan_payload, (*ordered_training, *ordered_selection)
    )
    if not isinstance(summary, Mapping):
        raise TypeError("Stage A finalizer returned a non-object")
    return validate_stage_a_summary(
        summary,
        plan_payload=plan_payload,
        training_receipts=ordered_training,
        selection_receipts=ordered_selection,
    )


def _remote_job_payload(
    plan_payload: dict[str, object], job: object,
) -> dict[str, object]:
    payload = asdict(job)
    payload["expected_outputs"] = list(payload["expected_outputs"])
    return {"plan": plan_payload, "job": payload}


def _preflight_stage_a_outputs(
    plan: PilotPlan, *, resume: bool, runs_client: VolumeClient,
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
            receipt_path = _volume_canonical_receipt_path(plan.run_id, stage, job.arm)
            producer_path = _volume_producer_path(plan.run_id, stage, job.arm)
            receipt = _read_volume_file_optional(runs_client, receipt_path)
            entries = _list_volume_files_optional(runs_client, producer_path)
            if receipt is None and not entries:
                continue
            if not resume:
                raise FileExistsError(
                    f"canonical Stage A output already exists for {stage}/{job.arm}; use --resume"
                )
            if receipt is None or not entries:
                raise ValueError(
                    f"canonical output is incomplete for {stage}/{job.arm}"
                )
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
    expected = {*producer_roots, *receipt_paths, summary_path}
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


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=MAX_GPU_CONTAINERS,
    retries=0,
    volumes={
        "/mnt/inputs": inputs_volume.read_only(),
        "/mnt/model": model_volume.read_only(),
        "/mnt/runs": runs_volume,
    },
)
def gpu_resources() -> dict[str, object]:
    """Return declaration metadata if explicitly invoked in a later task."""
    return status()


@app.function(
    image=cpu_image,
    timeout=300,
    max_containers=MAX_GPU_CONTAINERS,
    volumes={"/mnt/runs": runs_volume.read_only()},
)
def status_resources() -> dict[str, object]:
    """Return resource status without mutating application metadata."""
    return status()


def apply_approved_app_tags(plan: PilotPlan) -> None:
    """Attach a validated full run identity immediately before an approved run."""
    _validate_tag_plan(plan)
    app.set_tags({**_BASE_TAGS, "run-id": plan.run_id})


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
    validate_bundle_at_root(bundle, repo_root)

    bundle_root = f"/bundles/{bundle.bundle_id}"
    upload_bytes = {
        f"{bundle_root}/{item.path}": (repo_root / item.path).read_bytes()
        for item in bundle.files
    }
    upload_bytes[f"{bundle_root}/bundle-manifest.json"] = (
        canonical_json(asdict(bundle)) + "\n"
    ).encode("utf-8")
    _validate_local_upload_bytes(bundle, upload_bytes, bundle_root)

    try:
        entries = volume.listdir(bundle_root, recursive=True)
    except FileNotFoundError:
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


@app.local_entrypoint()
def status() -> dict[str, object]:
    """Return the static, read-only declaration envelope without client calls."""
    return {
        "app": APP_NAME,
        "gpu": GPU,
        "max_gpu_containers": MAX_GPU_CONTAINERS,
        "volumes": list(VOLUME_NAMES),
    }


@app.local_entrypoint(name="stage-inputs")
def stage_inputs(
    repo_root: str,
    approved_run_id: str,
    budget_acknowledged: bool = False,
) -> None:
    """Explicitly stage one immutable bundle after full operator approval."""
    bundle, plan = _build_operator_context(Path(repo_root))
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    staging_plan = preflight_inputs_local(
        bundle,
        inputs_volume,
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    print(canonical_json(_staging_plan_payload(staging_plan, plan)))
    if not staging_plan.upload_required:
        print(canonical_json({"bundle_id": bundle.bundle_id, "uploaded": False}))
        return
    apply_approved_app_tags(plan)
    result = _apply_input_staging_plan(staging_plan, inputs_volume)
    print(canonical_json(result))


@app.local_entrypoint(name="cache-model")
def cache_model(
    repo_root: str,
    approved_run_id: str,
    budget_acknowledged: bool = False,
) -> None:
    """Explicitly invoke the pinned CPU cache population boundary."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    print(canonical_json({
        "operation": "cache-model",
        "run_id": plan.run_id,
        "model_revision": plan.model_revision,
        "cpu": MODEL_CACHE_CPU,
        "memory_mib": MODEL_CACHE_MEMORY_MIB,
        "timeout_seconds": MODEL_CACHE_TIMEOUT_SECONDS,
        "destination": f"{VOLUME_NAMES[1]}:/model-cache/canonical",
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "budget_acknowledged_usd": 1_000.0,
    }))
    apply_approved_app_tags(plan)
    result = cache_model_remote.remote(pilot_plan_payload(plan))
    _print_remote_result(result)


@app.local_entrypoint(name="smoke")
def smoke(
    repo_root: str,
    approved_run_id: str,
    budget_acknowledged: bool = False,
) -> None:
    """Explicitly invoke the CPU-only preflight and report its receipt path."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    _validate_operator_approval(
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
    print(canonical_json({
        "operation": "smoke",
        "run_id": plan.run_id,
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
    apply_approved_app_tags(plan)
    result = smoke_remote.remote(pilot_plan_payload(plan))
    _print_remote_result(result)


@app.local_entrypoint(name="run-stage-a")
def run_stage_a(
    repo_root: str,
    approved_run_id: str,
    budget_acknowledged: bool = False,
    resume: bool = False,
) -> None:
    """Explicitly run training, selection, and CPU finalization only."""
    _bundle, plan = _build_operator_context(Path(repo_root))
    result = run_stage_a_local(
        plan,
        approved_run_id=approved_run_id,
        budget_acknowledged=budget_acknowledged,
        resume=resume,
        training_function=run_training_job,
        selection_function=run_selection_job,
        finalizer_function=finalize_stage_a_remote,
        runs_client=runs_volume,
    )
    print(canonical_json(result))


def _build_operator_context(repo_root: Path) -> tuple[InputBundle, PilotPlan]:
    root = Path(repo_root).resolve()
    status_result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    require_clean_tracked_status(status_result.stdout)
    bundle = build_input_bundle(root)
    lock_path = root / "requirements-modal-phase-marker.txt"
    if not lock_path.is_file():
        raise ValueError("compiled Modal dependency lock is missing")
    plan = build_pilot_plan(
        root / "configs/phase-marker-qwen25-7b.toml",
        root / "artifacts/phase-marker",
        bundle=bundle,
        source_hash=hash_source_tree(root),
        dependency_lock_hash=_sha256_bytes(lock_path.read_bytes()),
    )
    return bundle, plan


def _print_remote_result(result: object) -> None:
    if not isinstance(result, dict):
        raise TypeError("remote boundary returned a non-object result")
    print(canonical_json(result))


def _staging_plan_payload(
    staging_plan: InputStagingPlan, plan: PilotPlan,
) -> dict[str, object]:
    return {
        "operation": "stage-inputs",
        "action": "upload" if staging_plan.upload_required else "no-op",
        "run_id": plan.run_id,
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
    expected_run_id = (
        f"pilot-s42-cfg-{plan.config_hash[:8]}"
        f"-split-{plan.split_artifact_id[:8]}-src-{plan.source_hash[:12]}"
    )
    if plan.run_id != expected_run_id:
        raise ValueError("app tags require the canonical run ID")
    if (
        plan.resources.hardware != GPU
        or plan.resources.timeout_seconds != GPU_TIMEOUT_SECONDS
        or plan.resources.max_containers != MAX_GPU_CONTAINERS
        or len(plan.jobs) != 6
        or any(job.seed != plan.seed for job in plan.jobs)
    ):
        raise ValueError("app tags require the approved resource and job envelope")


def _validate_operator_approval(
    *, approved_run_id: str, plan: PilotPlan, budget_acknowledged: bool,
) -> None:
    if budget_acknowledged is not True:
        raise ValueError("explicit USD 1000 budget acknowledgement is required")
    if not isinstance(approved_run_id, str) or approved_run_id != plan.run_id:
        raise ValueError("full approved run ID must exactly match the plan")
    _validate_tag_plan(plan)


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
