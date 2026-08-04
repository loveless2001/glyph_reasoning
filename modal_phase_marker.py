"""Inert Modal resource declarations for the approved phase-marker pilot."""

from __future__ import annotations

from dataclasses import asdict
import io
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import Protocol, runtime_checkable

import modal

from phase_marker.io import canonical_json
from phase_marker.modal_artifacts import (
    InputBundle,
    VolumeClient,
    build_input_bundle,
    cache_model_to_volume,
    hash_source_tree,
    require_clean_tracked_status,
    run_cpu_smoke,
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
    """Upload exactly one approved bundle, or prove its remote bytes are identical."""
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
        return {"bundle_id": bundle.bundle_id, "uploaded": False}

    with volume.batch_upload() as batch:
        for remote_path, content in upload_bytes.items():
            batch.put_file(io.BytesIO(content), remote_path)
    return {"bundle_id": bundle.bundle_id, "uploaded": True}


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
    print(canonical_json({
        "operation": "stage-inputs",
        "run_id": plan.run_id,
        "bundle_id": bundle.bundle_id,
        "file_count": len(bundle.files) + 1,
        "destination": f"{VOLUME_NAMES[0]}:/bundles/{bundle.bundle_id}",
        "budget_acknowledged_usd": 1_000.0,
    }))
    apply_approved_app_tags(plan)
    result = stage_inputs_local(
        bundle,
        inputs_volume,
        approved_run_id=approved_run_id,
        plan=plan,
        budget_acknowledged=budget_acknowledged,
    )
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
    try:
        argv = shlex.split(plan.jobs[0].training_command)
        config_index = argv.index("--config") + 1
        config_path = Path(argv[config_index])
    except (IndexError, ValueError) as error:
        raise ValueError("pilot plan lacks its approved repository configuration") from error
    if not config_path.is_absolute():
        raise ValueError("pilot plan repository configuration must be absolute")
    root = config_path.resolve().parent.parent
    expected = (root / "configs/phase-marker-qwen25-7b.toml").resolve()
    if config_path.resolve() != expected:
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
