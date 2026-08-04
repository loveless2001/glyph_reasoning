"""Inert Modal resource declarations for the approved phase-marker pilot."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import modal

from phase_marker.modal_plan import PilotPlan


APP_NAME = "phase-marker-pilot-stage-a"
# linux/amd64 manifest for nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04.
BASE_IMAGE = (
    "nvidia/cuda@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
)
GPU = "H100"
GPU_TIMEOUT_SECONDS = 14_400
MAX_GPU_CONTAINERS = 2
VOLUME_NAMES = (
    "phase-marker-pilot-inputs-v1",
    "phase-marker-pilot-model-cache-v1",
    "phase-marker-pilot-runs-v1",
)
_BASE_TAGS = {"experiment": "phase-marker", "run-kind": "pilot", "seed": "42"}
_SHA256_CHARS = frozenset("0123456789abcdef")


@runtime_checkable
class RemoteFunction(Protocol):
    """Small callable boundary implemented by Modal function handles."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Invoke the declared function remotely."""


app = modal.App(
    APP_NAME,
    tags=_BASE_TAGS,
    include_source=False,
)
inputs_volume = modal.Volume.from_name(VOLUME_NAMES[0], create_if_missing=True)
model_volume = modal.Volume.from_name(VOLUME_NAMES[1], create_if_missing=True)
runs_volume = modal.Volume.from_name(VOLUME_NAMES[2], create_if_missing=True)

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


@app.local_entrypoint()
def status() -> dict[str, object]:
    """Return the static, read-only declaration envelope without client calls."""
    return {
        "app": APP_NAME,
        "gpu": GPU,
        "max_gpu_containers": MAX_GPU_CONTAINERS,
        "volumes": list(VOLUME_NAMES),
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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in _SHA256_CHARS for character in value
    )
