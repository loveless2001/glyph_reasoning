"""Content-addressed, Modal-independent inputs for the phase-marker pilot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
from typing import Protocol
import uuid

from phase_marker.io import canonical_json, sha256_json


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


class VolumeClient(Protocol):
    """The small volume boundary used by later Modal adapters."""

    def commit(self) -> None:
        """Make prior volume writes durable."""


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
        target = (
            workspace / _ARTIFACT_ROOT / "checkpoints" / _PILOT_KIND
            / f"seed-{_PILOT_SEED}" / arm
        )
        _symlink_exact_path(training, target, directory=True)
    _write_workspace_metadata(workspace, attempt_id, stage, arm)
    return workspace


def run_exact_command(
    command: str,
    *,
    workspace: Path,
    log_path: Path,
    env: Mapping[str, str],
) -> int:
    """Run only a frozen pilot command from its isolated workspace, without a shell."""
    argv = _approved_command_argv(command)
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("workspace is missing")
    attempt_root = _validate_workspace_metadata(root, argv)
    log = Path(log_path).resolve()
    logs_root = (attempt_root / "logs").resolve()
    if log == logs_root or not _is_within(log, logs_root):
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
    finally:
        lock.unlink(missing_ok=True)
    return target


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
        or not _valid_output_records(receipt.expected_outputs, receipt.output_hashes)
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
    expected_source = (attempt_root / "workspace" / producer).resolve()
    expected_canonical = (run_root / producer).resolve()
    if source != expected_source:
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
