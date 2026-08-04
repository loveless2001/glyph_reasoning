"""Content-addressed, Modal-independent inputs for the phase-marker pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import chdir, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Protocol
import uuid

from phase_marker.config import REQUIRED_MODEL_ID
from phase_marker.io import canonical_json, sha256_json
from phase_marker.token_audit import (
    QWEN25_7B_TOKENIZER_REVISION,
    validate_pinned_qwen_tokenizer_snapshot,
)


SOURCE_INCLUDE_PATHS = ("phase_marker/**/*.py", "modal_phase_marker.py")
_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
_SPLITS = ("train", "validation", "test", "exclusions")
_ARTIFACT_ROOT = "artifacts/phase-marker"
INPUT_ALLOWLIST = (
    "configs/phase-marker-qwen25-7b.toml",
    *(f"{_ARTIFACT_ROOT}/splits/{split}.jsonl" for split in _SPLITS),
    f"{_ARTIFACT_ROOT}/splits/manifest.json",
    *(
        path
        for arm in _ARMS
        for path in (
            f"{_ARTIFACT_ROOT}/training-data/{arm}.jsonl",
            f"{_ARTIFACT_ROOT}/training-data/{arm}.manifest.json",
        )
    ),
)
_MANIFEST_PATHS = (
    f"{_ARTIFACT_ROOT}/splits/manifest.json",
    *(f"{_ARTIFACT_ROOT}/training-data/{arm}.manifest.json" for arm in _ARMS),
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_PILOT_KIND = "pilot"
_PILOT_SEED = 42
_WORKSPACE_METADATA = "workspace-metadata.json"
_DEFAULT_EPHEMERAL_JOB_ROOT = Path("/tmp/phase-marker-pilot")
_MODEL_CACHE_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_PINNED_QWEN_MODEL_METADATA = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
}
_PINNED_QWEN_GENERATION_METADATA = {
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
}
_PLAN_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "seed",
        "config_hash",
        "split_artifact_id",
        "materialization_artifact_ids",
        "model_revision",
        "source_hash",
        "dependency_lock_hash",
        "bundle_id",
        "resources",
        "jobs",
        "run_id",
    }
)
_PLAN_JOB_FIELDS = frozenset(
    {
        "arm",
        "seed",
        "model_revision",
        "training_command",
        "selection_command",
        "expected_outputs",
    }
)
_PLAN_RESOURCE_FIELDS = frozenset(
    {
        "hardware",
        "timeout_seconds",
        "max_containers",
        "training_gpu_hours",
        "selection_gpu_hours",
        "behavior_gpu_hours",
        "max_gpu_hours",
        "stage_a_estimated_spend_usd",
        "estimated_spend_usd",
        "spend_cap_usd",
    }
)
_EXPECTED_PLAN_RESOURCES = {
    "hardware": "H100",
    "timeout_seconds": 14_400,
    "max_containers": 2,
    "training_gpu_hours": 24.0,
    "selection_gpu_hours": 24.0,
    "behavior_gpu_hours": 72.0,
    "max_gpu_hours": 120.0,
    "stage_a_estimated_spend_usd": 250.0,
    "estimated_spend_usd": 600.0,
    "spend_cap_usd": 1_000.0,
}
_EXPECTED_PLAN_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")


class VolumeClient(Protocol):
    """The small volume boundary used by later Modal adapters."""

    def commit(self) -> None:
        """Make prior volume writes durable."""


class _CachePublicationRollbackError(RuntimeError):
    """A visible cache publication could not be restored to quarantine."""


class _JobPublicationRollbackError(RuntimeError):
    """A visible job publication could not be restored to its attempt."""


@dataclass(frozen=True)
class BundleFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class InputBundle:
    schema_version: int
    bundle_id: str
    files: tuple[BundleFile, ...]
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelCacheFile:
    """One content-addressed file in the immutable pinned model cache."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelCacheManifest:
    """Exact local Qwen model cache required by a later Modal adapter."""

    schema_version: int
    model_id: str
    model_revision: str
    files: tuple[ModelCacheFile, ...]
    artifact_id: str


@dataclass(frozen=True)
class AttemptReceipt:
    """An immutable, content-addressed record of one execution attempt."""

    schema_version: int
    run_id: str
    bundle_id: str
    stage: str
    arm: str
    seed: int
    attempt_id: str
    command: str
    command_hash: str
    source_hash: str
    dependency_lock_hash: str
    model_cache_artifact_id: str
    requested_gpu: str
    observed_gpu: str | None
    started_at: str
    finished_at: str
    elapsed_seconds: float
    timeout_seconds: int
    exit_status: int
    validated: bool
    promoted: bool
    expected_outputs: tuple[str, ...]
    output_hashes: tuple[str, ...]
    failure_reason: str | None
    artifact_id: str

    def recomputed_artifact_id(self) -> str:
        """Return the receipt identity from every immutable field but itself."""
        return sha256_json(_receipt_payload(self, include_artifact_id=False))


def create_attempt_id() -> str:
    """Allocate a fresh namespace for an execution or rescheduled execution."""
    return str(uuid.uuid4())


def prepare_ephemeral_workspace(
    *,
    code_root: Path,
    input_root: Path,
    run_root: Path,
    bundle: InputBundle,
    stage: str,
    arm: str,
    attempt_id: str,
    canonical_training_root: Path | None = None,
) -> Path:
    """Recreate only approved repository paths under a one-attempt workspace."""
    if stage not in {"train", "selection"}:
        raise ValueError("workspace stage is invalid")
    if arm not in _ARMS:
        raise ValueError("workspace arm is invalid")
    if not _is_path_identity(attempt_id):
        raise ValueError("attempt ID must be a single path component")

    code = Path(code_root).resolve()
    inputs = Path(input_root).resolve()
    validate_bundle_at_root(bundle, inputs)
    python = code / ".venv/bin/python"
    package = code / "phase_marker"
    if not python.is_file() or not package.is_dir():
        raise ValueError("code root lacks the approved Python runtime or package")
    if stage == "selection" and canonical_training_root is None:
        raise ValueError("selection workspace requires a canonical training root")
    if stage == "train" and canonical_training_root is not None:
        raise ValueError("training workspace cannot bind canonical training output")

    attempt_root = Path(run_root).resolve() / "attempts" / attempt_id
    if attempt_root.exists():
        raise FileExistsError("attempt workspace already exists")
    workspace = attempt_root / "workspace"
    workspace.mkdir(parents=True)
    _symlink_exact_path(python, workspace / ".venv/bin/python")
    _symlink_exact_path(package, workspace / "phase_marker", directory=True)
    adapter = code / "modal_phase_marker.py"
    if adapter.is_file():
        _symlink_exact_path(adapter, workspace / "modal_phase_marker.py")
    for item in bundle.files:
        destination = workspace / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / item.path, destination)
        destination.chmod(0o444)
    if canonical_training_root is not None:
        training = Path(canonical_training_root).resolve()
        if not training.is_dir():
            raise ValueError("canonical training root is missing")
        training_records = _source_output_records(training)
        if not training_records:
            raise ValueError("canonical training root is empty")
        target = (
            workspace / _ARTIFACT_ROOT / "checkpoints" / _PILOT_KIND
            / f"seed-{_PILOT_SEED}" / arm
        )
        shutil.copytree(training, target, copy_function=shutil.copy2)
        if _source_output_records(target) != training_records:
            raise ValueError("ephemeral training parent copy does not match canonical bytes")
        _make_tree_read_only(target)
    _write_workspace_metadata(workspace, attempt_id, stage, arm)
    return workspace


def run_exact_command(
    command: str,
    *,
    workspace: Path,
    log_path: Path,
    env: Mapping[str, str],
    durable_attempt_root: Path | None = None,
) -> int:
    """Run only a frozen pilot command from its isolated workspace, without a shell."""
    argv = _approved_command_argv(command)
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("workspace is missing")
    workspace_attempt = _validate_workspace_metadata(root, argv)
    attempt_root = (
        workspace_attempt
        if durable_attempt_root is None
        else Path(durable_attempt_root).resolve()
    )
    if (
        attempt_root.name != workspace_attempt.name
        or attempt_root.parent.name != "attempts"
    ):
        raise ValueError("durable attempt root does not match ephemeral workspace")
    log = Path(log_path).resolve()
    logs_root = (attempt_root / "logs").resolve()
    if (
        log == logs_root
        or not _is_within(log, logs_root)
        or _is_within(log, root)
    ):
        raise ValueError("log path must remain outside the ephemeral workspace")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise ValueError("subprocess environment must contain string keys and values")

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        result = subprocess.run(
            argv,
            cwd=root,
            env=dict(env),
            shell=False,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return int(result.returncode)


def write_attempt_receipt(run_root: Path, receipt: AttemptReceipt) -> Path:
    """Atomically persist a verified receipt outside attempt outputs and checkpoints."""
    _validate_receipt(receipt)
    root = Path(run_root).resolve()
    receipt_path = root / "receipts" / f"{receipt.attempt_id}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise FileExistsError("attempt receipt already exists")
    payload = canonical_json(_receipt_payload(receipt, include_artifact_id=True)) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=receipt_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        # link(2) is an atomic create-if-absent operation, preserving receipts forever.
        os.link(temporary, receipt_path)
    except FileExistsError as error:
        raise FileExistsError("attempt receipt already exists") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt_path


def promote_validated_output(
    source: Path,
    attempt_root: Path,
    canonical_root: Path,
    receipt: AttemptReceipt,
) -> Path:
    """Copy an accepted output into its attempt namespace, then atomically promote it."""
    _validate_receipt(receipt)
    if receipt.exit_status != 0 or not receipt.validated:
        raise ValueError("promotion requires a validated successful receipt")
    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise ValueError("promotion source is missing or not a directory")
    target = Path(canonical_root).resolve()
    root = Path(attempt_root).resolve()
    _validate_promotion_paths(source_path, root, target, receipt)
    if target.exists():
        raise FileExistsError("canonical output already exists")
    if _filesystem_device(root) != _filesystem_device(target.parent):
        raise ValueError("attempt and canonical destinations must share the same filesystem")
    _validate_receipt_outputs(source_path, receipt)

    staged = root / "promotion-staging"
    if staged.exists():
        raise FileExistsError("attempt promotion staging already exists")
    shutil.copytree(source_path, staged, copy_function=shutil.copy2)
    if _tree_hashes(source_path) != _tree_hashes(staged):
        raise ValueError("attempt output copy does not match source bytes")

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.promotion.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError("canonical promotion is already in progress") from error
    try:
        if target.exists():
            raise FileExistsError("canonical output already exists")
        staged.replace(target)
    except Exception as error:
        try:
            lock.unlink(missing_ok=True)
        except Exception as lock_error:
            try:
                _restore_failed_job_promotion(
                    error=lock_error,
                    published=False,
                    target=target,
                    attempt_root=root,
                    lock=lock,
                )
            except _JobPublicationRollbackError:
                raise
            error.add_note(
                "promotion lock cleanup also failed: "
                f"{type(lock_error).__name__}: {lock_error}"
            )
        raise
    try:
        lock.unlink(missing_ok=True)
    except Exception as error:
        _restore_failed_job_promotion(
            error=error,
            published=True,
            target=target,
            attempt_root=root,
            lock=lock,
        )
        raise
    return target


def _restore_failed_job_promotion(
    *,
    error: Exception,
    published: bool,
    target: Path,
    attempt_root: Path,
    lock: Path,
) -> None:
    failures: list[str] = []
    if published:
        try:
            target.replace(Path(attempt_root) / "failed-promotion")
        except Exception as rollback_error:
            failures.append(
                "producer rollback failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
    if lock.exists():
        try:
            lock.replace(Path(attempt_root) / "failed-promotion.lock")
        except Exception as rollback_error:
            failures.append(
                "lock quarantine failed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            )
    if failures:
        compound = _JobPublicationRollbackError(
            "job publication rollback failed; refusing to commit ambiguous state"
        )
        compound.add_note(f"original failure: {type(error).__name__}: {error}")
        for failure in failures:
            compound.add_note(failure)
        raise compound from error


def execute_pilot_job(
    *,
    stage: str,
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
    code_root: Path,
    input_root: Path,
    model_root: Path,
    run_root: Path,
    volume: VolumeClient,
    ephemeral_root: Path = _DEFAULT_EPHEMERAL_JOB_ROOT,
    environ: Mapping[str, str] | None = None,
    producer_validator: Callable[..., None] | None = None,
    bf16_probe: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Run one frozen pilot producer and publish only its validated output tree."""
    if stage not in {"train", "selection"}:
        raise ValueError("pilot job stage is invalid")
    _validate_pilot_plan_payload(plan_payload)
    job = _validate_stage_job_payload(stage, plan_payload, job_payload)

    code = Path(code_root).resolve()
    if hash_source_tree(code) != plan_payload["source_hash"]:
        raise ValueError("pilot job source hash does not match the plan")
    lock = code / "requirements-modal-phase-marker.txt"
    if not lock.is_file() or _file_sha256(lock) != plan_payload["dependency_lock_hash"]:
        raise ValueError("pilot job dependency lock hash does not match the plan")

    bundle_root = Path(input_root) / "bundles" / str(plan_payload["bundle_id"])
    bundle = load_input_bundle(bundle_root / "bundle-manifest.json")
    if bundle.bundle_id != plan_payload["bundle_id"]:
        raise ValueError("pilot job bundle identity does not match the plan")
    validate_bundle_at_root(bundle, bundle_root)

    snapshot, cache_manifest = _validated_model_cache(model_root)

    attempt_id = create_attempt_id()
    run_mount = Path(run_root).resolve()
    run = run_mount / "runs" / str(plan_payload["run_id"])
    training_root = run / _producer_relative_path("train", str(job["arm"]))
    attempt_root = run / "attempts" / attempt_id
    if attempt_root.exists():
        raise FileExistsError("fresh pilot attempt namespace already exists")
    command = str(job["training_command"] if stage == "train" else job["selection_command"])
    started = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    log_path = attempt_root / "logs" / f"{stage}.log"
    canonical = run / _producer_relative_path(stage, str(job["arm"]))
    command_env = dict(os.environ if environ is None else environ)
    ephemeral = Path(ephemeral_root).resolve()
    observed_gpu: str | None = None
    exit_status = 1
    published = False
    attempt_receipt_path: Path | None = None
    canonical_receipt_path: Path | None = None
    failed_records: tuple[tuple[str, str], ...] = ()
    try:
        if (
            ephemeral == run_mount
            or ephemeral.is_relative_to(run_mount)
            or run_mount.is_relative_to(ephemeral)
        ):
            raise ValueError("ephemeral job root must be outside the run volume")
        ephemeral.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"phase-marker-{stage}-{job['arm']}-", dir=ephemeral
        ) as temporary:
            local_attempts = Path(temporary)
            workspace = local_attempts / "attempts" / attempt_id / "workspace"
            producer = workspace / _producer_relative_path(stage, str(job["arm"]))
            try:
                prepared = prepare_ephemeral_workspace(
                    code_root=code,
                    input_root=bundle_root,
                    run_root=local_attempts,
                    bundle=bundle,
                    stage=stage,
                    arm=str(job["arm"]),
                    attempt_id=attempt_id,
                    canonical_training_root=(
                        training_root if stage == "selection" else None
                    ),
                )
                if prepared.resolve() != workspace.resolve():
                    raise ValueError("pilot attempt workspace path is noncanonical")
                _require_one_visible_cuda_device(command_env)
                observed_gpu = _observe_gpu_name()
                if not _is_approved_observed_gpu(observed_gpu):
                    raise RuntimeError(
                        "pilot job requires exactly one observed H100 or H200 GPU"
                    )
                supports_bf16 = (
                    _torch_bf16_supported if bf16_probe is None else bf16_probe
                )
                if supports_bf16() is not True:
                    raise RuntimeError("pilot job requires BF16 support")
                command_env.update(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_CACHE": str(snapshot.parents[2]),
                    }
                )
                exit_status = run_exact_command(
                    command,
                    workspace=workspace,
                    log_path=log_path,
                    env=command_env,
                    durable_attempt_root=attempt_root,
                )
                if exit_status != 0:
                    raise RuntimeError(
                        f"{stage} job for {job['arm']} exited with status {exit_status}"
                    )
                validator = (
                    _validate_job_producer
                    if producer_validator is None
                    else producer_validator
                )
                if producer_validator is None:
                    with _offline_model_cache(snapshot.parents[2]):
                        validator(stage, producer, plan_payload, job)
                else:
                    validator(stage, producer, plan_payload, job)
                records = _source_output_records(producer)
                if not records:
                    raise ValueError("pilot job producer output is empty")
                receipt = _job_attempt_receipt(
                    stage=stage,
                    plan_payload=plan_payload,
                    job=job,
                    attempt_id=attempt_id,
                    command=command,
                    cache_artifact_id=cache_manifest.artifact_id,
                    observed_gpu=observed_gpu,
                    started=started,
                    elapsed_seconds=max(0.0, time.monotonic() - started_clock),
                    exit_status=0,
                    validated=True,
                    promoted=True,
                    records=records,
                    failure_reason=None,
                )
                promote_validated_output(producer, attempt_root, canonical, receipt)
                published = True
                attempt_receipt_path = _write_attempt_receipt_in_namespace(run, receipt)
                canonical_receipt_path = _link_canonical_receipt(
                    run, receipt, attempt_receipt_path
                )
                volume.commit()
                return _receipt_payload(receipt, include_artifact_id=True)
            except Exception:
                if producer.is_dir():
                    try:
                        failed_records = _source_output_records(producer)
                    except Exception:
                        failed_records = ()
                raise
    except Exception as error:
        if isinstance(error, _JobPublicationRollbackError):
            _append_failure_log(log_path, error)
            raise
        try:
            _quarantine_failed_job_publication(
                attempt_root=attempt_root,
                canonical=canonical,
                published=published,
                attempt_receipt_path=attempt_receipt_path,
                canonical_receipt_path=canonical_receipt_path,
            )
            _append_failure_log(log_path, error)
            failed = _job_attempt_receipt(
                stage=stage,
                plan_payload=plan_payload,
                job=job,
                attempt_id=attempt_id,
                command=command,
                cache_artifact_id=cache_manifest.artifact_id,
                observed_gpu=observed_gpu,
                started=started,
                elapsed_seconds=max(0.0, time.monotonic() - started_clock),
                exit_status=exit_status,
                validated=False,
                promoted=False,
                records=failed_records,
                failure_reason=f"{type(error).__name__}: {error}",
            )
            _write_attempt_receipt_in_namespace(run, failed)
            volume.commit()
        except Exception as persistence_error:
            error.add_note(
                "pilot job failure persistence also failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        raise


def _job_attempt_receipt(
    *,
    stage: str,
    plan_payload: Mapping[str, object],
    job: Mapping[str, object],
    attempt_id: str,
    command: str,
    cache_artifact_id: str,
    observed_gpu: str | None,
    started: datetime,
    elapsed_seconds: float,
    exit_status: int,
    validated: bool,
    promoted: bool,
    records: tuple[tuple[str, str], ...],
    failure_reason: str | None,
) -> AttemptReceipt:
    receipt = AttemptReceipt(
        schema_version=1,
        run_id=str(plan_payload["run_id"]),
        bundle_id=str(plan_payload["bundle_id"]),
        stage=stage,
        arm=str(job["arm"]),
        seed=int(job["seed"]),
        attempt_id=attempt_id,
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=str(plan_payload["source_hash"]),
        dependency_lock_hash=str(plan_payload["dependency_lock_hash"]),
        model_cache_artifact_id=cache_artifact_id,
        requested_gpu="H100",
        observed_gpu=observed_gpu,
        started_at=started.isoformat(),
        finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed_seconds,
        timeout_seconds=14_400,
        exit_status=exit_status,
        validated=validated,
        promoted=promoted,
        expected_outputs=tuple(path for path, _ in records),
        output_hashes=tuple(digest for _, digest in records),
        failure_reason=failure_reason,
        artifact_id="",
    )
    return AttemptReceipt(
        **{**asdict(receipt), "artifact_id": receipt.recomputed_artifact_id()}
    )


def _append_failure_log(log_path: Path, error: Exception) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(f"\n{type(error).__name__}: {error}\n".encode("utf-8"))


def _quarantine_failed_job_publication(
    *,
    attempt_root: Path,
    canonical: Path,
    published: bool,
    attempt_receipt_path: Path | None,
    canonical_receipt_path: Path | None,
) -> None:
    """Remove this attempt's uncommitted canonical names without deleting evidence."""
    if not published and attempt_receipt_path is None and canonical_receipt_path is None:
        return
    quarantine = Path(attempt_root) / "failed-publication"
    quarantine.mkdir(parents=True, exist_ok=False)
    if canonical_receipt_path is not None:
        canonical_receipt_path.replace(quarantine / "canonical-receipt.json")
    if attempt_receipt_path is not None:
        attempt_receipt_path.replace(quarantine / "success-receipt.json")
    if published:
        canonical.replace(quarantine / "producer")


def _validate_stage_job_payload(
    stage: str,
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(job_payload, Mapping) or set(job_payload) != _PLAN_JOB_FIELDS:
        raise ValueError("pilot job payload fields are invalid")
    normalized = dict(job_payload)
    outputs = normalized.get("expected_outputs")
    if isinstance(outputs, tuple):
        normalized["expected_outputs"] = list(outputs)
    jobs = plan_payload["jobs"]
    assert isinstance(jobs, list)
    matches = [item for item in jobs if isinstance(item, Mapping) and item.get("arm") == normalized.get("arm")]
    if len(matches) != 1 or dict(matches[0]) != normalized:
        raise ValueError("pilot job payload does not match the approved plan")
    command = normalized["training_command"] if stage == "train" else normalized["selection_command"]
    expected = (
        _workspace_training_command(str(normalized["arm"]))
        if stage == "train"
        else _workspace_selection_command(str(normalized["arm"]))
    )
    if shlex.split(str(command), posix=True) != expected:
        raise ValueError("pilot job command does not match the approved plan")
    return normalized


def _observe_gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        raise RuntimeError("pilot job requires exactly one observed H100 or H200 GPU")
    return names[0]


def _is_approved_observed_gpu(name: object) -> bool:
    if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
        return False
    tokens = name.upper().replace("-", " ").split()
    return sum(token in {"H100", "H200"} for token in tokens) == 1


def _require_one_visible_cuda_device(environ: Mapping[str, str]) -> None:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    devices = [] if visible is None else [item.strip() for item in visible.split(",")]
    if len(devices) != 1 or not devices[0] or devices[0] == "-1":
        raise RuntimeError("pilot job requires exactly one visible CUDA device")


def _torch_bf16_supported() -> bool:
    import torch

    return bool(torch.cuda.is_bf16_supported())


def _validate_job_producer(
    stage: str,
    producer: Path,
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
    *,
    replay_tokenizer: bool = True,
) -> None:
    outputs = job_payload["expected_outputs"]
    assert isinstance(outputs, list)
    selected = outputs[:3] if stage == "train" else outputs[3:]
    root = _producer_relative_path(stage, str(job_payload["arm"]))
    for value in selected:
        path = Path(str(value))
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError("pilot job expected output escapes its producer root") from error
        if not (producer / relative).is_file():
            raise ValueError(f"pilot job expected output is missing: {relative.as_posix()}")

    producer_path = Path(producer).resolve()
    relative_producer = _producer_relative_path(stage, str(job_payload["arm"]))
    workspace = producer_path.parents[len(relative_producer.parts) - 1]
    if (workspace / relative_producer).resolve() != producer_path:
        raise ValueError("pilot job producer is outside its approved workspace path")

    from phase_marker.config import ExperimentConfig
    from phase_marker.pipeline import (
        _validate_materializations,
        _validate_split_manifest,
        _validate_training_runs,
    )

    artifact_root = Path(_ARTIFACT_ROOT)
    config_path = Path("configs/phase-marker-qwen25-7b.toml")
    identity = frozenset({(int(job_payload["seed"]), str(job_payload["arm"]))})
    with chdir(workspace):
        config = ExperimentConfig.load(config_path)
        if sha256_json(asdict(config)) != plan_payload["config_hash"]:
            raise ValueError("pilot job producer configuration does not match the plan")
        split = _validate_split_manifest(artifact_root, config)
        materialization_ids = _validate_materializations(
            artifact_root, config, split.artifact_id
        )
        if (
            split.artifact_id != plan_payload["split_artifact_id"]
            or list(materialization_ids)
            != plan_payload["materialization_artifact_ids"]
        ):
            raise ValueError("pilot producer input lineage does not match the plan")
        expected_materializations = dict(
            zip(config.arms, materialization_ids, strict=True)
        )
        training_ids = _validate_training_runs(
            artifact_root,
            config,
            split.artifact_id,
            kind="pilot",
            seeds=(42,),
            expected_materializations=expected_materializations,
            expected_identities=identity,
        )
        if len(training_ids) != 1:
            raise ValueError("pilot training producer validation was not singular")
        if stage == "selection":
            from phase_marker.behavior import (
                _load_checkpoint_selections,
                _validate_production_behavior_inputs,
            )

            selections = _load_checkpoint_selections(
                (relative_producer / "manifest.json",),
                config,
                "pilot",
                (42,),
                allow_test=False,
                expected_identities=identity,
                replay_tokenizer=replay_tokenizer,
            )
            _validate_production_behavior_inputs(
                artifact_root / "splits/manifest.json",
                split.artifact_id,
                config,
                selections,
            )


def validate_canonical_job_semantics(
    *,
    stage: str,
    producer_files: Mapping[str, bytes],
    canonical_training_files: Mapping[str, bytes] | None,
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
    local_input_root: Path,
) -> None:
    """Re-run per-identity producer/consumer semantics in an isolated local view."""
    _validate_pilot_plan_payload(plan_payload)
    job = _validate_stage_job_payload(stage, plan_payload, job_payload)
    local = Path(local_input_root).resolve()
    bundle = build_input_bundle(local)
    if bundle.bundle_id != plan_payload["bundle_id"]:
        raise ValueError("local resume inputs no longer match the approved bundle")
    validate_bundle_at_root(bundle, local)
    if stage == "selection" and not canonical_training_files:
        raise ValueError("canonical selection lacks its semantic training parent")
    if stage == "train" and canonical_training_files is not None:
        raise ValueError("training semantic validation received an unexpected parent")

    with tempfile.TemporaryDirectory(prefix="phase-marker-resume-") as temporary:
        workspace = Path(temporary)
        for relative in INPUT_ALLOWLIST:
            source = local / relative
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        producer = workspace / _producer_relative_path(stage, str(job["arm"]))
        _write_semantic_file_view(producer, producer_files)
        if canonical_training_files is not None:
            training = workspace / _producer_relative_path("train", str(job["arm"]))
            _write_semantic_file_view(training, canonical_training_files)
        _validate_job_producer(
            stage,
            producer,
            plan_payload,
            job,
            replay_tokenizer=False,
        )


def _write_semantic_file_view(root: Path, files: Mapping[str, bytes]) -> None:
    if not isinstance(files, Mapping) or not files:
        raise ValueError("canonical semantic producer files are missing")
    for relative, content in files.items():
        candidate = PurePosixPath(relative)
        if (
            not isinstance(relative, str)
            or not relative
            or candidate.is_absolute()
            or "." in candidate.parts
            or ".." in candidate.parts
            or not isinstance(content, bytes)
        ):
            raise ValueError("canonical semantic producer file record is invalid")
        destination = Path(root).joinpath(*candidate.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _validated_model_cache(model_root: Path) -> tuple[Path, ModelCacheManifest]:
    snapshot = (
        Path(model_root).resolve()
        / "canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    manifest = load_model_cache_manifest(
        snapshot.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
    )
    validate_model_cache_manifest(snapshot, manifest)
    return snapshot, manifest


@contextmanager
def _offline_model_cache(cache_root: Path) -> object:
    values = {
        "HF_HUB_CACHE": str(Path(cache_root).resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    missing = object()
    previous: dict[str, object] = {
        name: os.environ.get(name, missing) for name in values
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                assert isinstance(value, str)
                os.environ[name] = value


def _write_attempt_receipt_in_namespace(run_root: Path, receipt: AttemptReceipt) -> Path:
    _validate_receipt(receipt)
    path = Path(run_root) / "receipts" / "attempts" / f"{receipt.attempt_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(_receipt_payload(receipt, include_artifact_id=True)) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError("attempt receipt already exists") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _link_canonical_receipt(
    run_root: Path, receipt: AttemptReceipt, attempt_receipt: Path,
) -> Path:
    path = (
        Path(run_root) / "receipts" / "canonical" / receipt.stage / f"{receipt.arm}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(attempt_receipt, path)
    except FileExistsError as error:
        raise FileExistsError("canonical receipt already exists") from error
    return path


def load_attempt_receipt_payload(payload: Mapping[str, object]) -> AttemptReceipt:
    """Parse one receipt mapping and revalidate its content-addressed identity."""
    fields = frozenset(AttemptReceipt.__dataclass_fields__)
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("attempt receipt fields are invalid")
    normalized = dict(payload)
    for name in ("expected_outputs", "output_hashes"):
        value = normalized[name]
        if not isinstance(value, (list, tuple)):
            raise ValueError("attempt receipt output records are invalid")
        normalized[name] = tuple(value)
    try:
        receipt = AttemptReceipt(**normalized)
    except TypeError as error:
        raise ValueError("attempt receipt fields are invalid") from error
    _validate_receipt(receipt)
    return receipt


def validate_job_receipt_payload(
    *,
    receipt_payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    """Fail closed unless a successful receipt binds the exact plan, job, and stage."""
    _validate_pilot_plan_payload(plan_payload)
    job = _validate_stage_job_payload(stage, plan_payload, job_payload)
    receipt = load_attempt_receipt_payload(receipt_payload)
    command = str(job["training_command"] if stage == "train" else job["selection_command"])
    if (
        receipt.run_id != plan_payload["run_id"]
        or receipt.bundle_id != plan_payload["bundle_id"]
        or receipt.stage != stage
        or receipt.arm != job["arm"]
        or receipt.seed != job["seed"]
        or receipt.command != command
        or receipt.command_hash != hashlib.sha256(command.encode("utf-8")).hexdigest()
        or receipt.source_hash != plan_payload["source_hash"]
        or receipt.dependency_lock_hash != plan_payload["dependency_lock_hash"]
        or receipt.requested_gpu != "H100"
        or receipt.observed_gpu is None
        or not _is_approved_observed_gpu(receipt.observed_gpu)
        or receipt.timeout_seconds != 14_400
        or receipt.exit_status != 0
        or receipt.validated is not True
        or receipt.promoted is not True
        or receipt.failure_reason is not None
    ):
        raise ValueError(f"{stage} receipt does not match approved job {job['arm']}")
    required = (
        {"adapter_config.json", "adapter_model.safetensors", "run-manifest.json"}
        if stage == "train"
        else {"manifest.json", "evidence.jsonl"}
    )
    if not required.issubset(set(receipt.expected_outputs)):
        raise ValueError(f"{stage} receipt producer outputs are incomplete for {job['arm']}")
    return _receipt_payload(receipt, include_artifact_id=True)


def validate_canonical_job_output(
    *,
    receipt_payload: Mapping[str, object],
    producer_files: Mapping[str, bytes],
    plan_payload: Mapping[str, object],
    job_payload: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    """Revalidate one canonical receipt, complete producer tree, and producer manifest."""
    validated = validate_job_receipt_payload(
        receipt_payload=receipt_payload,
        plan_payload=plan_payload,
        job_payload=job_payload,
        stage=stage,
    )
    receipt = load_attempt_receipt_payload(validated)
    if not isinstance(producer_files, Mapping) or not producer_files:
        raise ValueError("canonical producer files are missing")
    actual: list[tuple[str, str]] = []
    for path, content in producer_files.items():
        candidate = PurePosixPath(path)
        if (
            not isinstance(path, str)
            or not path
            or candidate.is_absolute()
            or "." in candidate.parts
            or ".." in candidate.parts
            or not isinstance(content, bytes)
        ):
            raise ValueError("canonical producer file record is invalid")
        actual.append((path, hashlib.sha256(content).hexdigest()))
    actual_records = tuple(sorted(actual))
    receipt_records = tuple(
        zip(receipt.expected_outputs, receipt.output_hashes, strict=True)
    )
    if actual_records != receipt_records:
        raise ValueError("canonical producer files do not match their receipt")

    manifest_name = "run-manifest.json" if stage == "train" else "manifest.json"
    try:
        manifest = json.loads(producer_files[manifest_name].decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("canonical producer manifest is missing or invalid") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("canonical producer manifest is missing or invalid")
    expected_kind = (
        "phase_marker_training_run"
        if stage == "train"
        else "phase_marker_checkpoint_selection"
    )
    job = _validate_stage_job_payload(stage, plan_payload, job_payload)
    if (
        manifest.get("kind") != expected_kind
        or manifest.get("arm") != job["arm"]
        or manifest.get("seed") != job["seed"]
        or manifest.get("config_hash") != plan_payload["config_hash"]
        or manifest.get("model_revision") != plan_payload["model_revision"]
        or (
            stage == "train"
            and manifest.get("tokenizer_revision") != plan_payload["model_revision"]
        )
        or (stage == "selection" and manifest.get("run_kind") != "pilot")
        or (stage == "selection" and manifest.get("completed") is not True)
    ):
        raise ValueError("canonical producer manifest identity is invalid")
    return validated


_STAGE_A_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "run_id",
        "training_receipt_ids",
        "selection_receipt_ids",
        "behavior_gate_checked_artifact_ids",
        "next_command",
        "stopped_before_behavior",
        "artifact_id",
    }
)


def validate_stage_a_summary(
    summary: Mapping[str, object],
    *,
    plan_payload: Mapping[str, object],
    training_receipts: tuple[Mapping[str, object], ...],
    selection_receipts: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Validate the compact, inert Stage A stop envelope."""
    _validate_pilot_plan_payload(plan_payload)
    if not isinstance(summary, Mapping) or set(summary) != _STAGE_A_SUMMARY_FIELDS:
        raise ValueError("Stage A summary fields are invalid")
    training_ids = [receipt.get("artifact_id") for receipt in training_receipts]
    selection_ids = [receipt.get("artifact_id") for receipt in selection_receipts]
    checked = summary["behavior_gate_checked_artifact_ids"]
    if (
        summary["schema_version"] != 1
        or summary["stage"] != "stage-a"
        or summary["run_id"] != plan_payload["run_id"]
        or summary["training_receipt_ids"] != training_ids
        or summary["selection_receipt_ids"] != selection_ids
        or not isinstance(checked, list)
        or not all(_is_sha256(value) for value in checked)
        or not isinstance(summary["next_command"], str)
        or not summary["next_command"]
        or summary["stopped_before_behavior"] is not True
        or not _is_sha256(summary["artifact_id"])
    ):
        raise ValueError("Stage A summary identity or stop contract is invalid")
    unsigned = dict(summary)
    artifact_id = unsigned.pop("artifact_id")
    if artifact_id != sha256_json(unsigned):
        raise ValueError("Stage A summary artifact ID is invalid")
    return dict(summary)


def finalize_stage_a(
    *,
    plan_payload: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    input_root: Path,
    model_root: Path,
    run_root: Path,
    volume: VolumeClient,
    behavior_gate: Callable[..., Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate the complete Stage A matrix and publish a mandatory stop summary."""
    _validate_pilot_plan_payload(plan_payload)
    snapshot, cache_manifest = _validated_model_cache(model_root)
    jobs = plan_payload["jobs"]
    assert isinstance(jobs, list)
    received = tuple(receipts)
    if len(received) != 12:
        raise ValueError("Stage A finalization requires exactly twelve receipts")
    training = tuple(
        validate_job_receipt_payload(
            receipt_payload=received[index],
            plan_payload=plan_payload,
            job_payload=job,
            stage="train",
        )
        for index, job in enumerate(jobs)
    )
    selection = tuple(
        validate_job_receipt_payload(
            receipt_payload=received[index + 6],
            plan_payload=plan_payload,
            job_payload=job,
            stage="selection",
        )
        for index, job in enumerate(jobs)
    )
    artifact_ids = [
        receipt["artifact_id"] for receipt in (*training, *selection)
    ]
    if len(set(artifact_ids)) != 12:
        raise ValueError("Stage A finalization receipt identities are not unique")
    if any(
        receipt["model_cache_artifact_id"] != cache_manifest.artifact_id
        for receipt in (*training, *selection)
    ):
        raise ValueError("Stage A receipts do not bind the validated model cache")

    gate_function = (
        _run_behavior_prerequisite_gate if behavior_gate is None else behavior_gate
    )
    with _offline_model_cache(snapshot.parents[2]):
        gate = gate_function(
            plan_payload=plan_payload,
            input_root=Path(input_root),
            model_root=Path(model_root),
            run_root=Path(run_root),
        )
    if not isinstance(gate, Mapping) or set(gate) != {
        "passed", "checked_artifact_ids", "commands",
    }:
        raise ValueError("behavior prerequisite gate result is invalid")
    checked = gate["checked_artifact_ids"]
    commands = gate["commands"]
    if (
        gate["passed"] is not True
        or not isinstance(checked, list)
        or not all(_is_sha256(value) for value in checked)
        or not isinstance(commands, list)
        or len(commands) != 1
        or not isinstance(commands[0], str)
        or not commands[0]
    ):
        raise ValueError("behavior prerequisite gate did not pass exactly once")

    summary: dict[str, object] = {
        "schema_version": 1,
        "stage": "stage-a",
        "run_id": plan_payload["run_id"],
        "training_receipt_ids": [receipt["artifact_id"] for receipt in training],
        "selection_receipt_ids": [receipt["artifact_id"] for receipt in selection],
        "behavior_gate_checked_artifact_ids": list(checked),
        "next_command": commands[0],
        "stopped_before_behavior": True,
    }
    summary["artifact_id"] = sha256_json(summary)
    validated = validate_stage_a_summary(
        summary,
        plan_payload=plan_payload,
        training_receipts=training,
        selection_receipts=selection,
    )
    run = Path(run_root).resolve() / "runs" / str(plan_payload["run_id"])
    run.mkdir(parents=True, exist_ok=True)
    summary_path = run / "stage-a-summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("existing Stage A summary is invalid") from error
        if not isinstance(existing, Mapping):
            raise ValueError("existing Stage A summary is invalid")
        revalidated = validate_stage_a_summary(
            existing,
            plan_payload=plan_payload,
            training_receipts=training,
            selection_receipts=selection,
        )
        if revalidated != validated:
            raise ValueError("existing Stage A summary conflicts with current validation")
        return revalidated

    _write_canonical_json_exclusive(summary_path, validated)
    try:
        volume.commit()
    except Exception as error:
        try:
            quarantine = run / "attempts" / f"finalizer-{create_attempt_id()}"
            quarantine.mkdir(parents=True, exist_ok=False)
            summary_path.replace(quarantine / "stage-a-summary.json")
            volume.commit()
        except Exception as persistence_error:
            error.add_note(
                "Stage A summary rollback also failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        raise
    return validated


def _write_canonical_json_exclusive(
    path: Path, payload: Mapping[str, object]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(payload) + "\n")
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError("immutable JSON destination already exists") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _run_behavior_prerequisite_gate(
    *,
    plan_payload: Mapping[str, object],
    input_root: Path,
    model_root: Path,
    run_root: Path,
) -> dict[str, object]:
    """Run the existing pilot behavior gate against a temporary read-only path view."""
    expected_cache_root = Path(model_root).resolve() / "canonical"
    if (
        os.environ.get("HF_HUB_CACHE") != str(expected_cache_root)
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise ValueError("behavior prerequisite gate requires the validated offline cache")
    from phase_marker.config import ExperimentConfig
    from phase_marker.modal_plan import StageAResources
    from phase_marker.pipeline import _run_gate

    bundle = Path(input_root) / "bundles" / str(plan_payload["bundle_id"])
    canonical = (
        Path(run_root) / "runs" / str(plan_payload["run_id"])
        / "artifacts" / "phase-marker"
    )
    with tempfile.TemporaryDirectory(prefix="phase-marker-finalize-") as temporary:
        view = Path(temporary)
        config = view / "configs/phase-marker-qwen25-7b.toml"
        config.parent.mkdir(parents=True)
        config.symlink_to(bundle / "configs/phase-marker-qwen25-7b.toml")
        artifacts = view / "artifacts/phase-marker"
        artifacts.mkdir(parents=True)
        for name, source in (
            ("splits", bundle / "artifacts/phase-marker/splits"),
            ("training-data", bundle / "artifacts/phase-marker/training-data"),
            ("checkpoints", canonical / "checkpoints"),
            ("checkpoint-selections", canonical / "checkpoint-selections"),
        ):
            if not source.is_dir():
                raise ValueError(f"behavior prerequisite source is missing: {name}")
            (artifacts / name).symlink_to(source, target_is_directory=True)
        with chdir(view):
            loaded = ExperimentConfig.load(Path("configs/phase-marker-qwen25-7b.toml"))
            result = _run_gate(
                "behavior",
                loaded,
                Path("artifacts/phase-marker"),
                kind="pilot",
                seeds=(42,),
                config_path=Path("configs/phase-marker-qwen25-7b.toml"),
                approval=StageAResources().approval(),
            )
    if not result.passed:
        raise ValueError(f"behavior prerequisite gate failed: {result.reason}")
    return {
        "passed": True,
        "checked_artifact_ids": list(result.checked_artifact_ids),
        "commands": list(result.commands),
    }


def _receipt_payload(receipt: AttemptReceipt, *, include_artifact_id: bool) -> dict[str, object]:
    payload = asdict(receipt)
    payload["expected_outputs"] = list(receipt.expected_outputs)
    payload["output_hashes"] = list(receipt.output_hashes)
    if not include_artifact_id:
        payload.pop("artifact_id")
    return payload


def _write_workspace_metadata(workspace: Path, attempt_id: str, stage: str, arm: str) -> None:
    payload = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "stage": stage,
        "arm": arm,
        "workspace_name": "workspace",
        "allowed_argv": _workspace_command(stage, arm),
    }
    path = workspace.parent / _WORKSPACE_METADATA
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
    path.chmod(0o444)


def _validate_workspace_metadata(workspace: Path, argv: list[str]) -> Path:
    metadata_path = workspace.parent / _WORKSPACE_METADATA
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("workspace metadata is missing or invalid") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "attempt_id", "stage", "arm", "workspace_name", "allowed_argv",
    }:
        raise ValueError("workspace metadata is missing or invalid")
    attempt_id = payload["attempt_id"]
    stage = payload["stage"]
    arm = payload["arm"]
    allowed_argv = payload["allowed_argv"]
    if (
        payload["schema_version"] != 1
        or payload["workspace_name"] != "workspace"
        or not _is_path_identity(attempt_id)
        or stage not in {"train", "selection"}
        or arm not in _ARMS
        or workspace.name != "workspace"
        or workspace.parent.name != attempt_id
        or not isinstance(allowed_argv, list)
        or not all(isinstance(token, str) for token in allowed_argv)
        or allowed_argv != _workspace_command(stage, arm)
    ):
        raise ValueError("workspace metadata is missing or invalid")
    if argv != allowed_argv:
        raise ValueError("command does not match the prepared workspace command")
    return workspace.parent


def _workspace_command(stage: str, arm: str) -> list[str]:
    if stage == "train":
        return _workspace_training_command(arm)
    if stage == "selection":
        return _workspace_selection_command(arm)
    raise ValueError("workspace stage is invalid")


def _validate_receipt(receipt: AttemptReceipt) -> None:
    if receipt.schema_version != 1:
        raise ValueError("receipt schema version is invalid")
    if not receipt.run_id or not _is_path_identity(receipt.attempt_id):
        raise ValueError("receipt identity is invalid")
    if receipt.stage not in {"train", "selection"} or receipt.arm not in _ARMS:
        raise ValueError("receipt stage or arm is invalid")
    if receipt.seed != _PILOT_SEED:
        raise ValueError("receipt seed is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            receipt.bundle_id,
            receipt.command_hash,
            receipt.source_hash,
            receipt.dependency_lock_hash,
            receipt.model_cache_artifact_id,
            receipt.artifact_id,
            *receipt.output_hashes,
        )
    ):
        raise ValueError("receipt hash fields are invalid")
    output_records_valid = _valid_output_records(
        receipt.expected_outputs, receipt.output_hashes
    ) or (
        receipt.validated is False
        and receipt.expected_outputs == ()
        and receipt.output_hashes == ()
    )
    if (
        not isinstance(receipt.command, str)
        or not receipt.command
        or not isinstance(receipt.requested_gpu, str)
        or not receipt.requested_gpu
        or (receipt.observed_gpu is not None and not isinstance(receipt.observed_gpu, str))
        or not isinstance(receipt.started_at, str)
        or not isinstance(receipt.finished_at, str)
        or not isinstance(receipt.elapsed_seconds, (float, int))
        or isinstance(receipt.elapsed_seconds, bool)
        or receipt.elapsed_seconds < 0
        or not isinstance(receipt.timeout_seconds, int)
        or isinstance(receipt.timeout_seconds, bool)
        or receipt.timeout_seconds <= 0
        or not isinstance(receipt.exit_status, int)
        or isinstance(receipt.exit_status, bool)
        or not isinstance(receipt.validated, bool)
        or not isinstance(receipt.promoted, bool)
        or not output_records_valid
        or (receipt.validated is False and receipt.promoted is True)
        or (receipt.failure_reason is not None and not isinstance(receipt.failure_reason, str))
    ):
        raise ValueError("receipt fields are invalid")
    if receipt.artifact_id != receipt.recomputed_artifact_id():
        raise ValueError("receipt artifact ID does not match its fields")


def _valid_output_records(paths: tuple[str, ...], hashes: tuple[str, ...]) -> bool:
    """Output path/hash records are parallel, source-relative tuples in stable order."""
    if not paths or len(paths) != len(hashes) or len(set(paths)) != len(paths):
        return False
    for path in paths:
        if not isinstance(path, str) or not path:
            return False
        candidate = PurePosixPath(path)
        if (
            candidate.is_absolute()
            or "." in candidate.parts
            or ".." in candidate.parts
        ):
            return False
    return True


def _validate_promotion_paths(
    source: Path, attempt_root: Path, canonical_root: Path, receipt: AttemptReceipt,
) -> None:
    if attempt_root.name != receipt.attempt_id or attempt_root.parent.name != "attempts":
        raise ValueError("attempt destination does not match receipt identity")
    run_root = attempt_root.parent.parent
    producer = _producer_relative_path(receipt.stage, receipt.arm)
    try:
        source_workspace = source.parents[len(producer.parts) - 1]
    except IndexError as error:
        raise ValueError("promotion source does not match receipt identity") from error
    source_attempt = source_workspace.parent
    expected_source = (source_attempt / "workspace" / producer).resolve()
    expected_canonical = (run_root / producer).resolve()
    if (
        source_workspace.name != "workspace"
        or source_attempt.name != receipt.attempt_id
        or source_attempt.parent.name != "attempts"
        or source != expected_source
    ):
        raise ValueError("promotion source does not match receipt identity")
    if canonical_root != expected_canonical:
        raise ValueError("canonical destination does not match receipt stage and arm")


def _producer_relative_path(stage: str, arm: str) -> Path:
    if stage == "train":
        kind = "checkpoints"
    elif stage == "selection":
        kind = "checkpoint-selections"
    else:
        raise ValueError("receipt stage is invalid")
    return Path(_ARTIFACT_ROOT) / kind / _PILOT_KIND / f"seed-{_PILOT_SEED}" / arm


def _validate_receipt_outputs(source: Path, receipt: AttemptReceipt) -> None:
    expected = tuple(zip(receipt.expected_outputs, receipt.output_hashes, strict=True))
    actual = _source_output_records(source)
    if actual != expected:
        raise ValueError("receipt output hash or complete source file set does not match")


def _source_output_records(root: Path) -> tuple[tuple[str, str], ...]:
    """Return every regular producer file as sorted source-relative path/hash records."""
    records: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("producer output must contain only regular files and directories")
        if path.is_file():
            records.append((path.relative_to(root).as_posix(), _file_sha256(path)))
    return tuple(records)


def _make_tree_read_only(root: Path) -> None:
    """Make an already-validated local copy immutable to the producer command."""
    path = Path(root)
    for candidate in sorted(path.rglob("*"), reverse=True):
        candidate.chmod(0o555 if candidate.is_dir() else 0o444)
    path.chmod(0o555)


def _filesystem_device(path: Path) -> int:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("filesystem root is missing")
        candidate = parent
    return candidate.stat().st_dev


def _approved_command_argv(command: str) -> list[str]:
    if not isinstance(command, str) or not command:
        raise ValueError("approved command is required")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as error:
        raise ValueError("approved command is malformed") from error
    for arm in _ARMS:
        if argv == _workspace_training_command(arm) or argv == _workspace_selection_command(arm):
            return argv
    raise ValueError("command is not an approved command")


def _workspace_training_command(arm: str) -> list[str]:
    output = f"{_ARTIFACT_ROOT}/checkpoints/{_PILOT_KIND}/seed-{_PILOT_SEED}/{arm}"
    return [
        "./.venv/bin/python", "-m", "phase_marker.training", "train",
        "--config", "configs/phase-marker-qwen25-7b.toml", "--arm", arm,
        "--seed", str(_PILOT_SEED), "--data", f"{_ARTIFACT_ROOT}/training-data/{arm}.jsonl",
        "--output-dir", output, "--manifest", f"{output}/run-manifest.json",
    ]


def _workspace_selection_command(arm: str) -> list[str]:
    training = f"{_ARTIFACT_ROOT}/checkpoints/{_PILOT_KIND}/seed-{_PILOT_SEED}/{arm}"
    output = f"{_ARTIFACT_ROOT}/checkpoint-selections/{_PILOT_KIND}/seed-{_PILOT_SEED}/{arm}"
    return [
        "./.venv/bin/python", "-m", "phase_marker.behavior", "select",
        "--config", "configs/phase-marker-qwen25-7b.toml", "--kind", _PILOT_KIND,
        "--seed", str(_PILOT_SEED), "--arm", arm, "--split-manifest",
        f"{_ARTIFACT_ROOT}/splits/manifest.json", "--validation-examples",
        f"{_ARTIFACT_ROOT}/splits/validation.jsonl", "--training-manifest",
        f"{training}/run-manifest.json", "--backend", "vllm", "--output", output,
    ]


def _symlink_exact_path(source: Path, destination: Path, *, directory: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=directory)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_path_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and Path(value).name == value
    )


def _tree_hashes(root: Path) -> tuple[tuple[str, int, str], ...]:
    records: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("attempt output must contain regular files only")
        records.append((path.relative_to(root).as_posix(), path.stat().st_size, _file_sha256(path)))
    return tuple(records)


def require_clean_tracked_status(status: str) -> None:
    """Reject tracked changes while permitting the approved untracked artifact tree."""
    for line in status.splitlines():
        if not line:
            continue
        if line.startswith("?? ") and line[3:] == "artifacts/":
            continue
        if line.startswith("  "):
            continue
        raise ValueError("tracked source changes are not permitted for the pilot")


def hash_source_tree(repo_root: Path) -> str:
    """Hash precisely the Python source staged by the future Modal adapter."""
    root = Path(repo_root).resolve()
    paths: list[Path] = []
    package = root / "phase_marker"
    if package.is_dir():
        paths.extend(
            path
            for path in package.rglob("*.py")
            if "__pycache__" not in path.parts and path.is_file()
        )
    adapter = root / "modal_phase_marker.py"
    if adapter.is_file():
        paths.append(adapter)
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)}
        for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix())
    ]
    return sha256_json(records)


def build_model_cache_manifest(snapshot: Path) -> ModelCacheManifest:
    """Build an exact, content-addressed manifest for one pinned local Qwen snapshot."""
    root = _pinned_qwen_snapshot_root(snapshot)
    validate_pinned_qwen_tokenizer_snapshot(root)
    _validate_pinned_qwen_model_metadata(root)
    paths = _model_cache_paths(root)
    files = tuple(_model_cache_file(root, path) for path in paths)
    manifest = ModelCacheManifest(
        schema_version=1,
        model_id=REQUIRED_MODEL_ID,
        model_revision=QWEN25_7B_TOKENIZER_REVISION,
        files=files,
        artifact_id=_model_cache_artifact_id(
            schema_version=1,
            model_id=REQUIRED_MODEL_ID,
            model_revision=QWEN25_7B_TOKENIZER_REVISION,
            files=files,
        ),
    )
    # Re-read every record to fail closed if a cache changes while being described.
    validate_model_cache_manifest(root, manifest)
    return manifest


def validate_model_cache_manifest(snapshot: Path, manifest: ModelCacheManifest) -> None:
    """Fail closed unless manifest metadata and every pinned cache byte still match."""
    root = _pinned_qwen_snapshot_root(snapshot)
    validate_pinned_qwen_tokenizer_snapshot(root)
    _validate_pinned_qwen_model_metadata(root)
    _validate_model_cache_manifest_shape(manifest)
    expected_paths = _model_cache_paths(root)
    actual_paths = tuple(item.path for item in manifest.files)
    if actual_paths != expected_paths:
        raise ValueError("model cache file paths do not match the exact pinned cache")
    for item in manifest.files:
        actual = _model_cache_file(root, item.path)
        if actual.size != item.size or actual.sha256 != item.sha256:
            raise ValueError(f"model cache file hash mismatch: {item.path}")


def load_model_cache_manifest(path: Path) -> ModelCacheManifest:
    """Read and validate the shape and content identity of a cache manifest."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("model cache manifest is missing or invalid") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "model_id", "model_revision", "files", "artifact_id",
    }:
        raise ValueError("model cache manifest is missing or invalid")
    files_payload = payload["files"]
    if not isinstance(files_payload, list) or any(
        not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}
        for item in files_payload
    ):
        raise ValueError("model cache manifest is missing or invalid")
    try:
        manifest = ModelCacheManifest(
            schema_version=payload["schema_version"],
            model_id=payload["model_id"],
            model_revision=payload["model_revision"],
            files=tuple(
                ModelCacheFile(
                    path=item["path"], size=item["size"], sha256=item["sha256"]
                )
                for item in files_payload
            ),
            artifact_id=payload["artifact_id"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("model cache manifest is missing or invalid") from error
    _validate_model_cache_manifest_shape(manifest)
    return manifest


def cache_model_to_volume(
    *,
    plan_payload: Mapping[str, object],
    cache_root: Path,
    volume: VolumeClient,
) -> dict[str, object]:
    """Populate and immutably publish the exact pinned Qwen cache on CPU."""
    root = Path(cache_root).resolve()
    attempt_id = create_attempt_id()
    attempt_root = root / "attempts" / "cache-model" / attempt_id
    try:
        _validate_pilot_plan_payload(plan_payload)
        canonical = (
            root
            / "canonical"
            / "models--Qwen--Qwen2.5-7B-Instruct"
            / "snapshots"
            / QWEN25_7B_TOKENIZER_REVISION
        )
        manifest_path = canonical.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
        existing = _existing_model_cache_result(canonical, manifest_path)
        if existing is not None:
            return existing

        # Keep this import at the CPU execution boundary. Importing this module is offline.
        from huggingface_hub import snapshot_download

        downloaded = Path(
            snapshot_download(
                repo_id=REQUIRED_MODEL_ID,
                revision=QWEN25_7B_TOKENIZER_REVISION,
                cache_dir=str(root / "hub"),
            )
        )
        hub_root = (root / "hub").resolve()
        resolved_download = downloaded.resolve()
        if not _is_within(resolved_download, hub_root):
            raise ValueError("downloaded snapshot escaped the model cache hub")
        source_manifest = build_model_cache_manifest(resolved_download)

        staged_model_root = (
            attempt_root
            / "publication"
            / "models--Qwen--Qwen2.5-7B-Instruct"
        )
        staged = (
            staged_model_root
            / "snapshots"
            / QWEN25_7B_TOKENIZER_REVISION
        )
        staged.mkdir(parents=True)
        for item in source_manifest.files:
            destination = staged / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_download / item.path, destination, follow_symlinks=True)
        staged_manifest = build_model_cache_manifest(staged)
        if staged_manifest != source_manifest:
            raise ValueError("staged model cache does not match downloaded bytes")
        validate_model_cache_manifest(staged, staged_manifest)
        staged_manifest_path = staged.parent / (
            f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
        )
        _write_quarantined_model_cache_manifest(
            staged_manifest_path, staged_manifest
        )
        quarantined_manifest = load_model_cache_manifest(staged_manifest_path)
        validate_model_cache_manifest(staged, quarantined_manifest)

        canonical_model_root = canonical.parents[1]
        canonical_model_root.parent.mkdir(parents=True, exist_ok=True)
        lock = canonical_model_root.parent / ".cache-promotion.lock"
        lock_acquired = False
        published = False
        try:
            lock.touch(exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                "canonical model cache promotion is already in progress"
            ) from error
        lock_acquired = True
        try:
            if canonical_model_root.exists() or canonical.exists() or manifest_path.exists():
                raise FileExistsError("canonical model cache already exists")
            staged_model_root.replace(canonical_model_root)
            published = True
            _cache_publication_hook(
                "after-publication",
                canonical_model_root=canonical_model_root,
                attempt_root=attempt_root,
            )
            _cache_publication_hook(
                "during-final-validation",
                canonical_snapshot=canonical,
                canonical_manifest=manifest_path,
            )
            published_manifest = load_model_cache_manifest(manifest_path)
            validate_model_cache_manifest(canonical, published_manifest)
            _release_cache_promotion_lock(lock)
            lock_acquired = False
            volume.commit()
        except Exception as error:
            _restore_failed_cache_publication(
                error=error,
                published=published,
                canonical_model_root=canonical_model_root,
                attempt_root=attempt_root,
                lock=lock,
                lock_acquired=lock_acquired,
            )
            raise

        return _model_cache_result(
            canonical, manifest_path, published_manifest, cached=True
        )
    except Exception as error:
        try:
            _write_cache_attempt_receipt(
                attempt_root,
                plan_payload=plan_payload,
                attempt_id=attempt_id,
                error=error,
            )
            if not isinstance(error, _CachePublicationRollbackError):
                volume.commit()
        except Exception as persistence_error:
            error.add_note(
                "cache-model receipt persistence also failed: "
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        raise


def _write_quarantined_model_cache_manifest(
    manifest_path: Path, manifest: ModelCacheManifest,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError("quarantined model cache manifest already exists")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=manifest_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(asdict(manifest)) + "\n")
        _cache_publication_hook(
            "during-manifest-publication",
            temporary_manifest=temporary,
            manifest_path=manifest_path,
        )
        os.link(temporary, manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rollback_cache_publication(
    canonical_model_root: Path, rollback_model_root: Path,
) -> None:
    rollback_model_root.parent.mkdir(parents=True, exist_ok=True)
    if rollback_model_root.exists():
        raise FileExistsError("cache rollback quarantine already exists")
    canonical_model_root.replace(rollback_model_root)
    if canonical_model_root.exists():
        raise OSError("canonical model cache remains visible after rollback")


def _release_cache_promotion_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def _quarantine_cache_promotion_lock(lock: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("failed promotion lock quarantine already exists")
    lock.replace(destination)
    if lock.exists():
        raise OSError("promotion lock remains visible after quarantine")


def _restore_failed_cache_publication(
    *,
    error: Exception,
    published: bool,
    canonical_model_root: Path,
    attempt_root: Path,
    lock: Path,
    lock_acquired: bool,
) -> None:
    restoration_errors: list[tuple[str, Exception]] = []
    if published and canonical_model_root.exists():
        rollback_model_root = (
            attempt_root
            / "rolled-back-publication"
            / "models--Qwen--Qwen2.5-7B-Instruct"
        )
        try:
            _rollback_cache_publication(
                canonical_model_root, rollback_model_root
            )
        except Exception as rollback_error:
            restoration_errors.append(("rollback", rollback_error))
    if lock_acquired and lock.exists():
        try:
            _release_cache_promotion_lock(lock)
        except Exception as release_error:
            try:
                _quarantine_cache_promotion_lock(
                    lock, attempt_root / "failed-promotion.lock"
                )
            except Exception as quarantine_error:
                restoration_errors.append(("lock quarantine", quarantine_error))
            else:
                error.add_note(
                    "promotion lock release failed and was quarantined: "
                    f"{type(release_error).__name__}: {release_error}"
                )
    if restoration_errors:
        compound = _CachePublicationRollbackError(
            "cache publication rollback failed; refusing to commit poisoned canonical state"
        )
        for kind, restoration_error in restoration_errors:
            compound.add_note(
                f"{kind} error: {type(restoration_error).__name__}: {restoration_error}"
            )
        raise compound from error


def _cache_publication_hook(stage: str, **context: object) -> None:
    """Injectable local failure boundary; production execution is a no-op."""


def _existing_model_cache_result(
    canonical: Path, manifest_path: Path,
) -> dict[str, object] | None:
    if not canonical.exists() and not manifest_path.exists():
        return None
    if not canonical.is_dir() or not manifest_path.is_file():
        raise ValueError("canonical model cache conflicts with an incomplete publication")
    try:
        manifest = load_model_cache_manifest(manifest_path)
        validate_model_cache_manifest(canonical, manifest)
    except (OSError, ValueError) as error:
        raise ValueError("canonical model cache conflicts with its manifest") from error
    return _model_cache_result(canonical, manifest_path, manifest, cached=False)


def _model_cache_result(
    canonical: Path,
    manifest_path: Path,
    manifest: ModelCacheManifest,
    *,
    cached: bool,
) -> dict[str, object]:
    return {
        "model_revision": manifest.model_revision,
        "artifact_id": manifest.artifact_id,
        "snapshot_path": str(canonical),
        "manifest_path": str(manifest_path),
        "cached": cached,
    }


def _validate_pilot_plan_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _PLAN_PAYLOAD_FIELDS:
        raise ValueError("pilot plan payload fields are invalid")
    hashes = (
        payload["config_hash"],
        payload["split_artifact_id"],
        payload["source_hash"],
        payload["dependency_lock_hash"],
        payload["bundle_id"],
    )
    materialization_ids = payload["materialization_artifact_ids"]
    resources = payload["resources"]
    jobs = payload["jobs"]
    if (
        payload["schema_version"] != 1
        or payload["kind"] != "pilot"
        or payload["seed"] != _PILOT_SEED
        or payload["model_revision"] != QWEN25_7B_TOKENIZER_REVISION
        or not all(_is_sha256(value) for value in hashes)
        or not isinstance(materialization_ids, list)
        or len(materialization_ids) != len(_EXPECTED_PLAN_ARMS)
        or not all(_is_sha256(value) for value in materialization_ids)
        or len(set(materialization_ids)) != len(materialization_ids)
        or not isinstance(resources, Mapping)
        or set(resources) != _PLAN_RESOURCE_FIELDS
        or dict(resources) != _EXPECTED_PLAN_RESOURCES
        or not isinstance(jobs, list)
        or len(jobs) != len(_EXPECTED_PLAN_ARMS)
    ):
        raise ValueError("pilot plan payload identity or resource envelope is invalid")
    for arm, job in zip(_EXPECTED_PLAN_ARMS, jobs, strict=True):
        if (
            not isinstance(job, Mapping)
            or set(job) != _PLAN_JOB_FIELDS
            or job["arm"] != arm
            or job["seed"] != _PILOT_SEED
            or job["model_revision"] != QWEN25_7B_TOKENIZER_REVISION
            or not isinstance(job["training_command"], str)
            or not job["training_command"]
            or not isinstance(job["selection_command"], str)
            or not job["selection_command"]
            or not isinstance(job["expected_outputs"], list)
            or len(job["expected_outputs"]) != 5
            or not all(
                isinstance(path, str) and path for path in job["expected_outputs"]
            )
        ):
            raise ValueError("pilot plan payload jobs are invalid")
    expected_run_id = (
        f"pilot-s42-cfg-{str(payload['config_hash'])[:8]}"
        f"-split-{str(payload['split_artifact_id'])[:8]}"
        f"-src-{str(payload['source_hash'])[:12]}"
    )
    if payload["run_id"] != expected_run_id:
        raise ValueError("pilot plan payload run ID is invalid")


def _write_cache_attempt_receipt(
    attempt_root: Path,
    *,
    plan_payload: Mapping[str, object],
    attempt_id: str,
    error: Exception,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "cache-model",
        "attempt_id": attempt_id,
        "run_id": plan_payload.get("run_id") if isinstance(plan_payload, Mapping) else None,
        "model_revision": (
            plan_payload.get("model_revision") if isinstance(plan_payload, Mapping) else None
        ),
        "validated": False,
        "promoted": False,
        "failure_reason": f"{type(error).__name__}: {error}",
    }
    payload["artifact_id"] = sha256_json(payload)
    attempt_root.mkdir(parents=True, exist_ok=True)
    path = attempt_root / "receipt.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
    return path


def _pinned_qwen_snapshot_root(snapshot: Path) -> Path:
    root = Path(snapshot)
    if (
        root.name != QWEN25_7B_TOKENIZER_REVISION
        or root.parent.name != "snapshots"
        or root.parent.parent.name != "models--Qwen--Qwen2.5-7B-Instruct"
    ):
        raise ValueError(f"pinned Qwen snapshot path is not bound to the exact revision: {root}")
    if root.is_symlink():
        raise ValueError(f"pinned Qwen snapshot directory must not be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"pinned Qwen snapshot directory is missing: {root}")
    return root


def _model_cache_paths(snapshot: Path) -> tuple[str, ...]:
    shards = _model_index_shards(snapshot)
    found_shards = tuple(
        sorted(
            path.relative_to(snapshot).as_posix()
            for path in snapshot.rglob("*.safetensors")
            if path.is_file() or path.is_symlink()
        )
    )
    if found_shards != shards:
        raise ValueError("model cache contains unindexed or missing model shards")
    return tuple(sorted((*_MODEL_CACHE_REQUIRED_FILES, *shards)))


def _validate_pinned_qwen_model_metadata(snapshot: Path) -> None:
    config = _read_model_metadata_object(snapshot / "config.json", "model")
    generation = _read_model_metadata_object(snapshot / "generation_config.json", "generation")
    if any(config.get(key) != value for key, value in _PINNED_QWEN_MODEL_METADATA.items()):
        raise ValueError("pinned Qwen model metadata is invalid")
    if any(
        generation.get(key) != value
        for key, value in _PINNED_QWEN_GENERATION_METADATA.items()
    ):
        raise ValueError("pinned Qwen generation metadata is invalid")


def _read_model_metadata_object(path: Path, kind: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"pinned Qwen {kind} metadata is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"pinned Qwen {kind} metadata is invalid")
    return payload


def _model_index_shards(snapshot: Path) -> tuple[str, ...]:
    index_path = snapshot / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"model index is missing or invalid: {index_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("model index must be an object")
    metadata = payload.get("metadata")
    weight_map = payload.get("weight_map")
    total_size = metadata.get("total_size") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size <= 0
    ):
        raise ValueError("model index metadata is invalid")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("model index weight_map is invalid")
    shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("model index weight_map is invalid")
        if not isinstance(shard_name, str) or not _is_model_shard_path(shard_name):
            raise ValueError("model shard path is invalid")
        shards.add(shard_name)
    return tuple(sorted(shards))


def _is_model_shard_path(value: str) -> bool:
    candidate = PurePosixPath(value)
    return (
        bool(value)
        and not candidate.is_absolute()
        and "." not in candidate.parts
        and ".." not in candidate.parts
        and candidate.suffix == ".safetensors"
    )


def _model_cache_file(snapshot: Path, relative_path: str) -> ModelCacheFile:
    if relative_path not in _MODEL_CACHE_REQUIRED_FILES and not _is_model_shard_path(relative_path):
        raise ValueError(f"model cache file path is invalid: {relative_path}")
    path = snapshot / relative_path
    if not path.is_file():
        if relative_path.endswith(".safetensors"):
            raise ValueError(f"model shard is missing or not a regular file: {relative_path}")
        raise ValueError(f"required model cache file is missing: {relative_path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"model cache file cannot be resolved: {relative_path}") from error
    if not resolved.is_file():
        raise ValueError(f"model cache file is not a regular file: {relative_path}")
    if path.is_symlink() and not _is_within(resolved, snapshot.parent.parent.resolve()):
        raise ValueError(f"model shard symlink is outside the pinned cache: {relative_path}")
    size = resolved.stat().st_size
    if size <= 0:
        if relative_path.endswith(".safetensors"):
            raise ValueError(f"model shard is empty: {relative_path}")
        raise ValueError(f"required model cache file is empty: {relative_path}")
    return ModelCacheFile(relative_path, size, _file_sha256(resolved))


def _model_cache_artifact_id(
    *, schema_version: int, model_id: str, model_revision: str, files: tuple[ModelCacheFile, ...],
) -> str:
    return sha256_json(
        {
            "schema_version": schema_version,
            "model_id": model_id,
            "model_revision": model_revision,
            "files": [asdict(item) for item in files],
        }
    )


def _validate_model_cache_manifest_shape(manifest: ModelCacheManifest) -> None:
    if not isinstance(manifest, ModelCacheManifest):
        raise ValueError("model cache manifest is invalid")
    if manifest.schema_version != 1:
        raise ValueError("model cache schema version is invalid")
    if manifest.model_id != REQUIRED_MODEL_ID or manifest.model_revision != QWEN25_7B_TOKENIZER_REVISION:
        raise ValueError("model cache identity is invalid")
    paths = tuple(item.path for item in manifest.files)
    if (
        not manifest.files
        or paths != tuple(sorted(paths))
        or len(paths) != len(set(paths))
        or any(
            not isinstance(item, ModelCacheFile)
            or not isinstance(item.size, int)
            or isinstance(item.size, bool)
            or item.size <= 0
            or not _is_sha256(item.sha256)
            for item in manifest.files
        )
    ):
        raise ValueError("model cache file metadata is invalid")
    expected_id = _model_cache_artifact_id(
        schema_version=manifest.schema_version,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        files=manifest.files,
    )
    if manifest.artifact_id != expected_id:
        raise ValueError("model cache artifact ID is noncanonical")


def build_input_bundle(repo_root: Path) -> InputBundle:
    """Describe exactly the approved config and input artifacts, without writes."""
    root = Path(repo_root).resolve()
    _validate_allowlist_paths(INPUT_ALLOWLIST)
    files = tuple(_bundle_file(root, relative_path) for relative_path in INPUT_ALLOWLIST)
    artifact_ids = tuple(_manifest_artifact_id(root / relative_path) for relative_path in _MANIFEST_PATHS)
    _reject_duplicate_artifact_ids(artifact_ids)
    bundle = InputBundle(
        schema_version=1,
        bundle_id=sha256_json(
            {
                "schema_version": 1,
                "files": [asdict(item) for item in files],
                "artifact_ids": list(artifact_ids),
            }
        ),
        files=files,
        artifact_ids=artifact_ids,
    )
    # A second read closes the race between manifest parsing and the returned bundle.
    validate_bundle_at_root(bundle, root)
    return bundle


def validate_bundle_at_root(bundle: InputBundle, root: Path) -> None:
    """Fail closed unless this root still exactly matches the bundle description."""
    _validate_bundle_shape(bundle)
    resolved_root = Path(root).resolve()
    for item in bundle.files:
        path = resolved_root / item.path
        if not path.is_file():
            raise ValueError(f"bundle file is missing: {item.path}")
        if path.stat().st_size != item.size or _file_sha256(path) != item.sha256:
            raise ValueError(f"bundle file hash mismatch: {item.path}")


def load_input_bundle(path: Path) -> InputBundle:
    """Read a staged input-bundle manifest and validate its content identity."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("input bundle manifest is missing or invalid") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "bundle_id", "files", "artifact_ids",
    }:
        raise ValueError("input bundle manifest is missing or invalid")
    files_payload = payload["files"]
    artifact_ids = payload["artifact_ids"]
    if (
        not isinstance(files_payload, list)
        or any(
            not isinstance(item, Mapping) or set(item) != {"path", "size", "sha256"}
            for item in files_payload
        )
        or not isinstance(artifact_ids, list)
    ):
        raise ValueError("input bundle manifest is missing or invalid")
    try:
        bundle = InputBundle(
            schema_version=payload["schema_version"],
            bundle_id=payload["bundle_id"],
            files=tuple(
                BundleFile(path=item["path"], size=item["size"], sha256=item["sha256"])
                for item in files_payload
            ),
            artifact_ids=tuple(artifact_ids),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("input bundle manifest is missing or invalid") from error
    _validate_bundle_shape(bundle)
    return bundle


def run_cpu_smoke(
    *,
    plan_payload: Mapping[str, object],
    code_root: Path,
    input_root: Path,
    model_root: Path,
    run_root: Path,
    volume: VolumeClient,
    runtime_imports: tuple[str, ...],
) -> dict[str, object]:
    """Validate the locked CPU preflight and persist one content-addressed receipt."""
    run_id = plan_payload.get("run_id") if isinstance(plan_payload, Mapping) else None
    receipt_namespace = run_id if _is_path_identity(run_id) else "invalid-plan"
    receipt_root = Path(run_root) / "runs" / receipt_namespace / "receipts" / "smoke"
    imported: list[dict[str, object]] = []
    model_cache_artifact_id: str | None = None
    try:
        _validate_pilot_plan_payload(plan_payload)
        if not runtime_imports or len(set(runtime_imports)) != len(runtime_imports):
            raise ValueError("locked runtime import list is invalid")
        for name in runtime_imports:
            if not isinstance(name, str) or not name:
                raise ValueError("locked runtime import list is invalid")
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
            imported.append(
                {"module": name, "version": version if isinstance(version, str) else None}
            )

        code = Path(code_root).resolve()
        if hash_source_tree(code) != plan_payload["source_hash"]:
            raise ValueError("CPU smoke source hash does not match the plan")
        lock = code / "requirements-modal-phase-marker.txt"
        if not lock.is_file() or _file_sha256(lock) != plan_payload["dependency_lock_hash"]:
            raise ValueError("CPU smoke dependency lock hash does not match the plan")

        bundle_root = Path(input_root) / "bundles" / str(plan_payload["bundle_id"])
        bundle = load_input_bundle(bundle_root / "bundle-manifest.json")
        if bundle.bundle_id != plan_payload["bundle_id"]:
            raise ValueError("CPU smoke bundle identity does not match the plan")
        validate_bundle_at_root(bundle, bundle_root)

        snapshot = (
            Path(model_root)
            / "canonical"
            / "models--Qwen--Qwen2.5-7B-Instruct"
            / "snapshots"
            / QWEN25_7B_TOKENIZER_REVISION
        )
        manifest_path = snapshot.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
        manifest = load_model_cache_manifest(manifest_path)
        validate_model_cache_manifest(snapshot, manifest)
        model_cache_artifact_id = manifest.artifact_id

        receipt = _cpu_smoke_receipt_payload(
            plan_payload=plan_payload,
            imported=imported,
            model_cache_artifact_id=model_cache_artifact_id,
            validated=True,
            failure_reason=None,
        )
        receipt_path = _write_content_addressed_receipt(receipt_root, receipt)
        volume.commit()
        return {
            "validated": True,
            "artifact_id": receipt["artifact_id"],
            "receipt_path": str(receipt_path),
        }
    except Exception as error:
        receipt = _cpu_smoke_receipt_payload(
            plan_payload=plan_payload,
            imported=imported,
            model_cache_artifact_id=model_cache_artifact_id,
            validated=False,
            failure_reason=f"{type(error).__name__}: {error}",
        )
        _write_content_addressed_receipt(receipt_root, receipt)
        volume.commit()
        raise


def _cpu_smoke_receipt_payload(
    *,
    plan_payload: Mapping[str, object],
    imported: list[dict[str, object]],
    model_cache_artifact_id: str | None,
    validated: bool,
    failure_reason: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "smoke",
        "hardware": "CPU",
        "run_id": plan_payload.get("run_id"),
        "source_hash": plan_payload.get("source_hash"),
        "dependency_lock_hash": plan_payload.get("dependency_lock_hash"),
        "bundle_id": plan_payload.get("bundle_id"),
        "model_revision": plan_payload.get("model_revision"),
        "model_cache_artifact_id": model_cache_artifact_id,
        "imports": imported,
        "validated": validated,
        "failure_reason": failure_reason,
    }
    payload["artifact_id"] = sha256_json(payload)
    return payload


def _write_content_addressed_receipt(
    receipt_root: Path, payload: Mapping[str, object],
) -> Path:
    artifact_id = payload.get("artifact_id")
    if not _is_sha256(artifact_id):
        raise ValueError("receipt artifact ID is invalid")
    expected = sha256_json({key: value for key, value in payload.items() if key != "artifact_id"})
    if artifact_id != expected:
        raise ValueError("receipt artifact ID does not match its fields")
    root = Path(receipt_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    content = canonical_json(dict(payload)) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError("content-addressed receipt conflicts with existing bytes")
        return path
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError("content-addressed receipt conflicts with existing bytes")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _bundle_file(root: Path, relative_path: str) -> BundleFile:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"required input bundle file is missing: {relative_path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"required input bundle file is empty: {relative_path}")
    return BundleFile(relative_path, size, _file_sha256(path))


def _manifest_artifact_id(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid bundle manifest: {path}") from error
    value = payload.get("artifact_id") if isinstance(payload, dict) else None
    if not _is_sha256(value):
        raise ValueError(f"bundle manifest artifact_id is missing or malformed: {path}")
    return str(value)


def _validate_bundle_shape(bundle: InputBundle) -> None:
    if bundle.schema_version != 1:
        raise ValueError("bundle schema version is invalid")
    paths = tuple(item.path for item in bundle.files)
    _validate_allowlist_paths(paths)
    if paths != INPUT_ALLOWLIST:
        raise ValueError("bundle file paths must match the exact input allowlist")
    if any(
        not isinstance(item.size, int) or isinstance(item.size, bool) or item.size <= 0
        or not _is_sha256(item.sha256)
        for item in bundle.files
    ):
        raise ValueError("bundle file metadata is invalid")
    if len(bundle.artifact_ids) != len(_MANIFEST_PATHS):
        raise ValueError("bundle artifact IDs are incomplete")
    if not all(_is_sha256(value) for value in bundle.artifact_ids):
        raise ValueError("bundle artifact IDs are invalid")
    _reject_duplicate_artifact_ids(bundle.artifact_ids)
    expected_bundle_id = sha256_json(
        {
            "schema_version": bundle.schema_version,
            "files": [asdict(item) for item in bundle.files],
            "artifact_ids": list(bundle.artifact_ids),
        }
    )
    if bundle.bundle_id != expected_bundle_id:
        raise ValueError("bundle ID is noncanonical")


def _validate_allowlist_paths(paths: tuple[str, ...]) -> None:
    for value in paths:
        candidate = PurePosixPath(value)
        if not value or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("bundle file paths must be relative and traversal-free")


def _reject_duplicate_artifact_ids(artifact_ids: tuple[str, ...]) -> None:
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("duplicate artifact ID in bundle")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS
