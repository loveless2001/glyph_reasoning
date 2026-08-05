"""Modal-independent validation and export for read-only Stage A inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tempfile

from phase_marker.io import canonical_json
from phase_marker.modal_artifacts import (
    INPUT_ALLOWLIST,
    LOCKED_RUNTIME_MODULES,
    VolumeClient,
    load_attempt_receipt_payload,
    atomic_publish_directory_noreplace,
    sha256_json,
)
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


__all__ = ("download_evidence_local", "status_local")

APP_NAME = "phase-marker-pilot-stage-a"
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
LOCKED_RUNTIME_IMPORTS = LOCKED_RUNTIME_MODULES


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

def status_local(volume: VolumeClient, *, run_id: str) -> dict[str, object]:
    """Read and validate one Stage A namespace without mutating remote state."""
    _require_canonical_run_id(run_id)
    run_root = f"/runs/{run_id}"
    entries = _list_volume_files_optional(volume, run_root)
    if not entries:
        raise ValueError(f"unknown run ID: {run_id}")

    paths = _normalized_listed_paths(entries, run_root)
    failed, attempt_identities, attempt_errors = _inspect_attempt_receipts(
        volume, run_id=run_id, paths=paths
    )
    states = {
        "train": {
            arm: ("failed" if ("train", arm) in failed else "pending")
            for arm in _PILOT_ARMS
        },
        "selection": {
            arm: ("failed" if ("selection", arm) in failed else "pending")
            for arm in _PILOT_ARMS
        },
    }
    receipt_ids: dict[str, dict[str, str]] = {"train": {}, "selection": {}}
    receipt_elapsed: dict[str, dict[str, float]] = {"train": {}, "selection": {}}
    receipt_payloads: dict[str, dict[str, dict[str, object]]] = {
        "train": {},
        "selection": {},
    }
    shared_identities: list[tuple[str, str, tuple[str, ...]]] = []
    lineage_identities: list[tuple[str, str, tuple[str, ...]]] = []
    errors: list[str] = list(attempt_errors)
    for stage in ("train", "selection"):
        for arm in _PILOT_ARMS:
            receipt_path = _volume_canonical_receipt_path(run_id, stage, arm)
            producer_path = _volume_producer_path(run_id, stage, arm)
            receipt = _read_volume_file_optional(volume, receipt_path)
            producer_entries = _list_volume_files_optional(volume, producer_path)
            if receipt is None and not producer_entries:
                continue
            if receipt is None or not producer_entries:
                states[stage][arm] = "invalid"
                errors.append(f"{stage}/{arm}: incomplete canonical evidence")
                continue
            try:
                validated, shared_identity, lineage_identity = (
                    _validate_status_canonical_output(
                    volume=volume,
                    run_id=run_id,
                    stage=stage,
                    arm=arm,
                    receipt_bytes=receipt,
                    producer_path=producer_path,
                    entries=producer_entries,
                    )
                )
            except (OSError, TypeError, ValueError) as error:
                states[stage][arm] = "invalid"
                errors.append(f"{stage}/{arm}: {error}")
                continue
            states[stage][arm] = "complete"
            receipt_ids[stage][arm] = str(validated["artifact_id"])
            receipt_elapsed[stage][arm] = float(validated["elapsed_seconds"])
            receipt_payloads[stage][arm] = validated
            shared_identities.append((stage, arm, shared_identity))
            lineage_identities.append((stage, arm, lineage_identity))

    shared_values = {identity for _stage, _arm, identity in shared_identities}
    lineage_values = {identity for _stage, _arm, identity in lineage_identities}
    if len(shared_values) > 1 or len(lineage_values) > 1:
        errors.append("canonical outputs disagree on pilot identity")
        for stage, arm, _identity in shared_identities:
            states[stage][arm] = "invalid"
            receipt_ids[stage].pop(arm, None)
            receipt_payloads[stage].pop(arm, None)
    canonical_shared = next(iter(shared_values)) if len(shared_values) == 1 else None
    attempt_values = set(attempt_identities)
    if (
        len(attempt_values) > 1
        or (
            canonical_shared is not None
            and any(identity != canonical_shared for identity in attempt_values)
        )
    ):
        errors.append("attempt receipts disagree with the canonical pilot identity")
    dependency_error: str | None = None
    advertised_values = shared_values | attempt_values
    if len(advertised_values) == 1:
        approved_shared = next(iter(advertised_values))
        smoke_id = approved_shared[5]
        smoke_path = f"{run_root}/receipts/smoke/{smoke_id}.json"
        smoke_bytes = _read_volume_file_optional(volume, smoke_path)
        try:
            if smoke_bytes is None:
                raise ValueError("receipt-referenced smoke receipt is missing")
            validated_smoke = _validate_smoke_receipt(
                smoke_bytes,
                relative_path=f"receipts/smoke/{smoke_id}.json",
                run_id=run_id,
                approved_shared=approved_shared,
            )
            provenance_bytes = _read_volume_file_optional(
                volume, f"{run_root}/provenance/input-bundle-manifest.json"
            )
            _validate_bundle_provenance(
                provenance_bytes,
                smoke=validated_smoke,
            )
        except (TypeError, ValueError) as error:
            dependency_error = str(error)
            errors.append(f"dependencies: {dependency_error}")
    for arm in _PILOT_ARMS:
        if (
            states["selection"][arm] == "complete"
            and states["train"][arm] != "complete"
        ):
            states["selection"][arm] = "invalid"
            receipt_ids["selection"].pop(arm, None)
            receipt_payloads["selection"].pop(arm, None)
            errors.append(f"selection/{arm}: canonical training parent is not complete")

    all_complete = all(
        states[stage][arm] == "complete"
        for stage in ("train", "selection")
        for arm in _PILOT_ARMS
    )
    summary_path = f"{run_root}/stage-a-summary.json"
    summary_bytes = _read_volume_file_optional(volume, summary_path)
    summary_state = "pending"
    stopped = False
    if summary_bytes is not None:
        try:
            validated_summary = _validate_status_summary(
                summary_bytes,
                run_id=run_id,
                training_receipts=[
                    receipt_payloads["train"].get(arm) for arm in _PILOT_ARMS
                ],
                selection_receipts=[
                    receipt_payloads["selection"].get(arm) for arm in _PILOT_ARMS
                ],
            )
            if not all_complete:
                raise ValueError("summary names an incomplete canonical receipt matrix")
            if canonical_shared is None:
                raise ValueError("summary lacks one canonical pilot identity")
            if dependency_error is not None:
                raise ValueError(
                    f"summary dependencies are invalid: {dependency_error}"
                )
        except (TypeError, ValueError) as error:
            summary_state = "invalid"
            errors.append(f"summary: {error}")
        else:
            summary_state = "complete"
            stopped = True

    return {
        "run_id": run_id,
        "training": states["train"],
        "selection": states["selection"],
        "summary": summary_state,
        "stopped_before_behavior": stopped,
        "valid": not errors,
        "errors": errors,
    }


def download_evidence_local(
    volume: VolumeClient, *, run_id: str, destination: Path,
) -> tuple[Path, ...]:
    """Atomically download only the compact, approved Stage A evidence set."""
    target = Path(destination)
    if _local_path_exists_nofollow(target):
        raise FileExistsError(f"evidence destination already exists: {target}")
    status_result = status_local(volume, run_id=run_id)
    if (
        status_result["valid"] is not True
        or status_result["summary"] != "complete"
        or status_result["stopped_before_behavior"] is not True
    ):
        raise ValueError("download requires validated complete Stage A evidence")

    run_root = f"/runs/{run_id}"
    summary_path = f"{run_root}/stage-a-summary.json"
    summary_bytes = _read_volume_file_optional(volume, summary_path)
    if summary_bytes is None:
        raise ValueError("Stage A summary disappeared before download")
    summary_payload = _decode_json_object(summary_bytes, "Stage A summary")
    smoke_receipt_artifact_id = summary_payload.get("smoke_receipt_artifact_id")
    if not _is_sha256(smoke_receipt_artifact_id):
        raise ValueError("Stage A summary smoke receipt identity is invalid")
    smoke_id = str(smoke_receipt_artifact_id)
    paths = _normalized_listed_paths(
        _list_volume_files_optional(volume, run_root), run_root
    )
    selected = tuple(
        path for path in sorted(paths)
        if _evidence_relative_path(
            run_id, path, smoke_receipt_artifact_id=smoke_id
        ) is not None
    )
    contents: dict[str, bytes] = {}
    for remote_path in selected:
        content = _read_volume_file_optional(volume, remote_path)
        if content is None:
            raise ValueError(f"evidence file disappeared during download: {remote_path}")
        relative = _evidence_relative_path(
            run_id,
            remote_path,
            smoke_receipt_artifact_id=smoke_id,
        )
        assert relative is not None
        contents[relative] = content
    _validate_downloaded_advertised_hashes(contents)
    _revalidate_download_snapshot(
        volume,
        run_id=run_id,
        selected_paths=selected,
        contents=contents,
        smoke_receipt_artifact_id=smoke_id,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        for relative, content in contents.items():
            path = temporary.joinpath(*PurePosixPath(relative).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if _sha256_bytes(path.read_bytes()) != _sha256_bytes(content):
                raise ValueError(f"downloaded evidence hash mismatch: {relative}")
        _assert_evidence_allowlist_unchanged(
            volume,
            run_id=run_id,
            selected_paths=selected,
            smoke_receipt_artifact_id=smoke_id,
        )
        _publish_directory_noreplace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return tuple(
        target.joinpath(*PurePosixPath(path).parts) for path in sorted(contents)
    )


def _local_path_exists_nofollow(path: Path) -> bool:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return False
    return True


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory while preserving every existing name."""
    atomic_publish_directory_noreplace(source, destination)

def _require_canonical_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("status requires a canonical run ID")


def _normalized_listed_paths(entries: Sequence[object], root: str) -> set[str]:
    normalized_root = "/" + root.lstrip("/")
    paths: set[str] = set()
    for entry in entries:
        raw = entry if isinstance(entry, str) else getattr(entry, "path", None)
        if not isinstance(raw, str):
            raise ValueError("run listing contains an invalid path")
        path = "/" + raw.lstrip("/")
        if path != normalized_root and not path.startswith(normalized_root + "/"):
            raise ValueError("run listing escaped its canonical namespace")
        if _remote_entry_kind(entry) in {"file", "unspecified"}:
            paths.add(path)
        elif _remote_entry_kind(entry) != "directory":
            raise ValueError("run listing contains an unsupported entry")
    return paths


def _decode_json_object(content: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object")
    return payload


def _inspect_attempt_receipts(
    volume: VolumeClient, *, run_id: str, paths: set[str],
) -> tuple[set[tuple[str, str]], tuple[tuple[str, ...], ...], tuple[str, ...]]:
    prefix = f"/runs/{run_id}/receipts/attempts/"
    failed: set[tuple[str, str]] = set()
    identities: list[tuple[str, ...]] = []
    errors: list[str] = []
    for path in sorted(path for path in paths if path.startswith(prefix)):
        relative = path[len(prefix):]
        if not relative or "/" in relative or not relative.endswith(".json"):
            errors.append(f"attempt receipt path is invalid: {path}")
            continue
        content = _read_volume_file_optional(volume, path)
        if content is None:
            errors.append(f"attempt receipt disappeared: {path}")
            continue
        try:
            receipt = load_attempt_receipt_payload(
                _decode_json_object(content, "attempt receipt")
            )
            _validate_attempt_receipt_job_identity(
                receipt, run_id=run_id, filename=relative
            )
        except (TypeError, ValueError) as error:
            errors.append(f"attempt receipt {path}: {error}")
            continue
        identities.append(_receipt_shared_identity(receipt))
        if (
            receipt.validated is False
            and receipt.promoted is False
        ):
            failed.add((receipt.stage, receipt.arm))
    return failed, tuple(identities), tuple(errors)


def _validate_attempt_receipt_job_identity(
    receipt: object, *, run_id: str, filename: str,
) -> None:
    attempt_id = getattr(receipt, "attempt_id", None)
    if filename != f"{attempt_id}.json":
        raise ValueError("attempt receipt filename does not match its attempt ID")
    _validate_receipt_job_identity(receipt, run_id=run_id)


def _validate_receipt_job_identity(receipt: object, *, run_id: str) -> None:
    stage = getattr(receipt, "stage", None)
    arm = getattr(receipt, "arm", None)
    source_hash = getattr(receipt, "source_hash", None)
    config_hash = getattr(receipt, "config_hash", None)
    split_artifact_id = getattr(receipt, "split_artifact_id", None)
    plan_digest = getattr(receipt, "plan_digest", None)
    if (
        getattr(receipt, "run_id", None) != run_id
        or stage not in {"train", "selection"}
        or arm not in _PILOT_ARMS
        or getattr(receipt, "seed", None) != 42
        or getattr(receipt, "requested_gpu", None) != "H100"
        or getattr(receipt, "timeout_seconds", None) != 14_400
        or not _is_sha256(source_hash)
        or not _run_id_binds_source(run_id, source_hash)
        or not _is_sha256(config_hash)
        or f"-cfg-{str(config_hash)[:8]}-" not in run_id
        or not _is_sha256(split_artifact_id)
        or f"-split-{str(split_artifact_id)[:8]}-" not in run_id
        or not _is_sha256(plan_digest)
        or not run_id.endswith(f"-plan-{plan_digest}")
        or getattr(receipt, "model_revision", None)
        != QWEN25_7B_TOKENIZER_REVISION
    ):
        raise ValueError("attempt receipt identity is not an approved pilot job")
    expected_command = _status_expected_command(str(stage), str(arm))
    if (
        getattr(receipt, "command", None) != expected_command
        or getattr(receipt, "command_hash", None)
        != _sha256_bytes(expected_command.encode("utf-8"))
    ):
        raise ValueError("attempt receipt command is not the approved exact job")
    validated = getattr(receipt, "validated", None)
    promoted = getattr(receipt, "promoted", None)
    exit_status = getattr(receipt, "exit_status", None)
    failure_reason = getattr(receipt, "failure_reason", None)
    failure_stage = getattr(receipt, "failure_stage", None)
    observed_gpu = getattr(receipt, "observed_gpu", None)
    if validated is True:
        required_outputs = (
            {"adapter_config.json", "adapter_model.safetensors", "run-manifest.json"}
            if stage == "train"
            else {"manifest.json", "evidence.jsonl"}
        )
        if (
            promoted is not True
            or exit_status != 0
            or failure_reason is not None
            or failure_stage is not None
            or not _is_approved_observed_gpu(observed_gpu)
            or not required_outputs.issubset(
                set(getattr(receipt, "expected_outputs", ()))
            )
        ):
            raise ValueError("successful attempt receipt state is invalid")
    elif (
        validated is not False
        or promoted is not False
        or not isinstance(failure_reason, str)
        or not failure_reason
        or failure_stage not in {
            "workspace-setup",
            "runtime-validation",
            "command",
            "producer-validation",
            "promotion",
            "post-promotion-validation",
            "receipt-publication",
            "commit",
        }
        or (
            exit_status == 0
            and failure_stage not in {
                "producer-validation", "promotion", "post-promotion-validation",
                "receipt-publication", "commit",
            }
        )
        or (
            observed_gpu is not None
            and not _is_approved_observed_gpu(observed_gpu)
        )
    ):
        raise ValueError("failed attempt receipt state is invalid")


def _run_id_binds_source(run_id: str, source_hash: object) -> bool:
    return bool(
        _is_sha256(source_hash)
        and run_id.endswith(f"-src-{str(source_hash)[:12]}-plan-{run_id[-64:]}")
        and _is_sha256(run_id[-64:])
    )


def _is_approved_observed_gpu(value: object) -> bool:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        return False
    tokens = value.upper().replace("-", " ").split()
    return sum(token in {"H100", "H200"} for token in tokens) == 1


def _receipt_shared_identity(receipt: object) -> tuple[str, ...]:
    values = (
        getattr(receipt, "bundle_id", None),
        getattr(receipt, "bundle_manifest_artifact_id", None),
        getattr(receipt, "source_hash", None),
        getattr(receipt, "dependency_lock_hash", None),
        getattr(receipt, "model_cache_artifact_id", None),
        getattr(receipt, "smoke_receipt_artifact_id", None),
        getattr(receipt, "plan_digest", None),
        getattr(receipt, "config_hash", None),
        getattr(receipt, "split_artifact_id", None),
        *getattr(receipt, "materialization_artifact_ids", ()),
    )
    if (
        not all(_is_sha256(value) for value in values)
        or getattr(receipt, "modal_environment", None) != "main"
    ):
        raise ValueError("attempt receipt shared identity is invalid")
    return (*tuple(str(value) for value in values), "main")


def _validate_status_canonical_output(
    *,
    volume: VolumeClient,
    run_id: str,
    stage: str,
    arm: str,
    receipt_bytes: bytes,
    producer_path: str,
    entries: tuple[object, ...],
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    receipt = load_attempt_receipt_payload(
        _decode_json_object(receipt_bytes, "canonical receipt")
    )
    _validate_receipt_job_identity(receipt, run_id=run_id)
    expected_command = _status_expected_command(stage, arm)
    required_outputs = (
        {"adapter_config.json", "adapter_model.safetensors", "run-manifest.json"}
        if stage == "train"
        else {"manifest.json", "evidence.jsonl"}
    )
    if (
        receipt.run_id != run_id
        or receipt.stage != stage
        or receipt.arm != arm
        or receipt.seed != 42
        or receipt.command != expected_command
        or receipt.command_hash != _sha256_bytes(expected_command.encode("utf-8"))
        or receipt.requested_gpu != "H100"
        or receipt.observed_gpu is None
        or not any(model in receipt.observed_gpu.upper() for model in ("H100", "H200"))
        or receipt.timeout_seconds != 14_400
        or receipt.exit_status != 0
        or receipt.validated is not True
        or receipt.promoted is not True
        or receipt.failure_reason is not None
        or receipt.failure_stage is not None
        or not required_outputs.issubset(receipt.expected_outputs)
    ):
        raise ValueError("canonical receipt does not match its Stage A identity")
    producer_files = _read_volume_producer_files(
        entries=entries,
        producer_path=producer_path,
        runs_client=volume,
    )
    actual = tuple(
        sorted(
            (path, _sha256_bytes(content))
            for path, content in producer_files.items()
        )
    )
    advertised = tuple(
        zip(receipt.expected_outputs, receipt.output_hashes, strict=True)
    )
    if actual != advertised:
        raise ValueError("canonical producer bytes do not match their receipt")
    manifest_name = "run-manifest.json" if stage == "train" else "manifest.json"
    manifest_bytes = producer_files[manifest_name]
    manifest = _decode_json_object(manifest_bytes, "producer manifest")
    config_hash = manifest.get("config_hash")
    model_revision = manifest.get("model_revision")
    if (
        not _is_sha256(config_hash)
        or model_revision != QWEN25_7B_TOKENIZER_REVISION
        or not _is_sha256(receipt.bundle_id)
        or not _is_sha256(receipt.source_hash)
        or not _is_sha256(receipt.dependency_lock_hash)
        or receipt.config_hash != config_hash
        or receipt.model_revision != model_revision
        or not run_id.endswith(f"-plan-{receipt.plan_digest}")
        or f"-cfg-{str(config_hash)[:8]}-" not in run_id
        or not _run_id_binds_source(run_id, receipt.source_hash)
    ):
        raise ValueError("canonical producer content identity is invalid")
    split_id: object
    if stage == "train":
        if set(manifest) != _TRAINING_MANIFEST_FIELDS:
            raise ValueError("training producer manifest schema is invalid")
        parents = manifest.get("data_parent_hashes")
        split_id = (
            parents[0] if isinstance(parents, list) and len(parents) == 1 else None
        )
        if (
            manifest.get("kind") != "phase_marker_training_run"
            or manifest.get("arm") != arm
            or manifest.get("seed") != 42
            or manifest.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
            or manifest.get("tokenizer_revision") != model_revision
            or manifest.get("dataset_path")
            != f"artifacts/phase-marker/training-data/{arm}.jsonl"
            or not _is_sha256(manifest.get("data_artifact_id"))
            or manifest.get("parent_hashes") != [manifest.get("data_artifact_id")]
            or not _is_sha256(split_id)
            or split_id != receipt.split_artifact_id
            or manifest.get("data_artifact_id")
            != receipt.materialization_artifact_ids[_PILOT_ARMS.index(arm)]
            or f"-split-{str(split_id)[:8]}-" not in run_id
        ):
            raise ValueError("training producer manifest semantic identity is invalid")
    else:
        training_path = _volume_producer_path(run_id, "train", arm)
        training_manifest_bytes = _read_volume_file_optional(
            volume, f"{training_path}/run-manifest.json"
        )
        if training_manifest_bytes is None:
            raise ValueError("selection lacks its canonical training manifest")
        training_manifest = _decode_json_object(
            training_manifest_bytes, "canonical training manifest"
        )
        training_parents = training_manifest.get("data_parent_hashes")
        split_id = (
            training_parents[0]
            if isinstance(training_parents, list) and len(training_parents) == 1
            else None
        )
        materialization_id = training_manifest.get("data_artifact_id")
        training_manifest_hash = _sha256_bytes(training_manifest_bytes)
        unsigned_manifest = dict(manifest)
        selection_artifact_id = unsigned_manifest.pop("artifact_id", None)
        if (
            set(manifest) != _SELECTION_MANIFEST_FIELDS
            or selection_artifact_id != sha256_json(unsigned_manifest)
            or manifest.get("schema_version") != 1
            or manifest.get("kind") != "phase_marker_checkpoint_selection"
            or manifest.get("run_kind") != "pilot"
            or manifest.get("arm") != arm
            or manifest.get("seed") != 42
            or manifest.get("selected_on") != "validation"
            or manifest.get("evidence_scope") != "experiment"
            or manifest.get("origin_verification")
            != "execution_receipt_or_rerun_required"
            or manifest.get("backend") != "vllm"
            or manifest.get("model_id") != "Qwen/Qwen2.5-7B-Instruct"
            or manifest.get("criterion") != _SELECTION_CRITERION
            or manifest.get("split_artifact_id") != split_id
            or manifest.get("training_manifest_hash") != training_manifest_hash
            or manifest.get("materialization_artifact_id") != materialization_id
            or split_id != receipt.split_artifact_id
            or materialization_id
            != receipt.materialization_artifact_ids[_PILOT_ARMS.index(arm)]
            or manifest.get("parent_hashes")
            != [split_id, materialization_id, training_manifest_hash]
            or manifest.get("training_manifest_file")
            != f"artifacts/phase-marker/checkpoints/pilot/seed-42/{arm}/run-manifest.json"
            or manifest.get("validation_examples_file")
            != "artifacts/phase-marker/splits/validation.jsonl"
            or manifest.get("evidence_file")
            != (
                "artifacts/phase-marker/checkpoint-selections/pilot/"
                f"seed-42/{arm}/evidence.jsonl"
            )
            or manifest.get("evidence_hash")
            != _sha256_bytes(producer_files["evidence.jsonl"])
            or not _is_sha256(manifest.get("split_manifest_hash"))
            or not _is_sha256(manifest.get("validation_examples_hash"))
            or not _is_sha256(manifest.get("selected_checkpoint_hash"))
            or not isinstance(manifest.get("candidates"), list)
            or not manifest.get("candidates")
            or manifest.get("completed") is not True
        ):
            raise ValueError("selection producer manifest semantic identity is invalid")
    shared_identity = _receipt_shared_identity(receipt)
    lineage_identity = (
        str(config_hash), str(model_revision), str(split_id)
    )
    return (
        _decode_json_object(receipt_bytes, "canonical receipt"),
        shared_identity,
        lineage_identity,
    )


def _status_expected_command(stage: str, arm: str) -> str:
    if arm not in _PILOT_ARMS:
        raise ValueError("status arm is invalid")
    training_root = f"artifacts/phase-marker/checkpoints/pilot/seed-42/{arm}"
    if stage == "train":
        argv = [
            "./.venv/bin/python", "-m", "phase_marker.training", "train",
            "--config", "configs/phase-marker-qwen25-7b.toml", "--arm", arm,
            "--seed", "42", "--data",
            f"artifacts/phase-marker/training-data/{arm}.jsonl",
            "--output-dir", training_root,
            "--manifest", f"{training_root}/run-manifest.json",
        ]
    elif stage == "selection":
        selection_root = (
            f"artifacts/phase-marker/checkpoint-selections/pilot/seed-42/{arm}"
        )
        argv = [
            "./.venv/bin/python", "-m", "phase_marker.behavior", "select",
            "--config", "configs/phase-marker-qwen25-7b.toml", "--kind", "pilot",
            "--seed", "42", "--arm", arm, "--split-manifest",
            "artifacts/phase-marker/splits/manifest.json", "--validation-examples",
            "artifacts/phase-marker/splits/validation.jsonl", "--training-manifest",
            f"{training_root}/run-manifest.json", "--backend", "vllm",
            "--output", selection_root,
        ]
    else:
        raise ValueError("status stage is invalid")
    return shlex.join(argv)


def _validate_status_summary(
    content: bytes,
    *,
    run_id: str,
    training_receipts: list[dict[str, object] | None],
    selection_receipts: list[dict[str, object] | None],
) -> dict[str, object]:
    summary = _decode_json_object(content, "Stage A summary")
    fields = {
        "schema_version", "stage", "run_id", "plan_digest",
        "stage_a_action_digest", "stage_a_resume",
        "modal_environment",
        "smoke_receipt_artifact_id", "model_cache_artifact_id",
        "bundle_manifest_artifact_id",
        "receipt_approval_history", "training_receipt_ids",
        "selection_receipt_ids", "behavior_gate_checked_artifact_ids",
        "elapsed_gpu_seconds", "finalizer_provenance", "next_command",
        "stopped_before_behavior", "artifact_id",
    }
    if (
        len(training_receipts) != len(_PILOT_ARMS)
        or len(selection_receipts) != len(_PILOT_ARMS)
        or any(
            not isinstance(receipt, Mapping)
            for receipt in (*training_receipts, *selection_receipts)
        )
    ):
        raise ValueError("Stage A summary receipt matrix is incomplete")
    training = tuple(dict(receipt) for receipt in training_receipts if receipt is not None)
    selection = tuple(
        dict(receipt) for receipt in selection_receipts if receipt is not None
    )
    receipt_matrix = (*training, *selection)
    training_ids = [str(receipt["artifact_id"]) for receipt in training]
    selection_ids = [str(receipt["artifact_id"]) for receipt in selection]
    training_elapsed = [receipt.get("elapsed_seconds") for receipt in training]
    selection_elapsed = [receipt.get("elapsed_seconds") for receipt in selection]
    checked = summary.get("behavior_gate_checked_artifact_ids")
    elapsed_values = [*training_elapsed, *selection_elapsed]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in elapsed_values
    ):
        raise ValueError("Stage A summary receipt elapsed time is invalid")
    expected_elapsed = {
        "training": math.fsum(float(value) for value in training_elapsed),
        "selection": math.fsum(float(value) for value in selection_elapsed),
        "total": math.fsum(float(value) for value in elapsed_values),
    }
    plan_digests = {receipt.get("plan_digest") for receipt in receipt_matrix}
    smoke_ids = {
        receipt.get("smoke_receipt_artifact_id") for receipt in receipt_matrix
    }
    cache_ids = {
        receipt.get("model_cache_artifact_id") for receipt in receipt_matrix
    }
    bundle_manifest_ids = {
        receipt.get("bundle_manifest_artifact_id") for receipt in receipt_matrix
    }
    modal_environments = {
        receipt.get("modal_environment") for receipt in receipt_matrix
    }
    if (
        len(plan_digests) != 1
        or len(smoke_ids) != 1
        or len(cache_ids) != 1
        or len(bundle_manifest_ids) != 1
        or modal_environments != {"main"}
        or not _is_sha256(next(iter(plan_digests), None))
        or not _is_sha256(next(iter(smoke_ids), None))
        or not _is_sha256(next(iter(cache_ids), None))
        or not _is_sha256(next(iter(bundle_manifest_ids), None))
    ):
        raise ValueError("Stage A summary receipt approval identity is invalid")
    plan_digest = str(next(iter(plan_digests)))
    smoke_id = str(next(iter(smoke_ids)))
    cache_id = str(next(iter(cache_ids)))
    bundle_manifest_id = str(next(iter(bundle_manifest_ids)))
    summary_resume = summary.get("stage_a_resume")
    if not isinstance(summary_resume, bool):
        raise ValueError("Stage A summary approval mode is invalid")
    expected_action_digest = _status_stage_a_action_digest(
        plan_digest=plan_digest,
        resume=summary_resume,
        smoke_receipt_artifact_id=smoke_id,
        model_cache_artifact_id=cache_id,
        bundle_manifest_artifact_id=bundle_manifest_id,
        modal_environment="main",
    )
    expected_approval_history = _status_receipt_approval_history(receipt_matrix)
    provenance = summary.get("finalizer_provenance")
    provenance_fields = {
        "modal_app_id", "modal_app_name", "modal_function_name",
        "modal_function_call_id", "modal_input_id", "python_version",
        "torch_version", "cuda_runtime_version", "cuda_driver_version",
        "runtime_versions",
    }
    runtime_versions = (
        provenance.get("runtime_versions") if isinstance(provenance, Mapping) else None
    )
    if (
        set(summary) != fields
        or summary.get("schema_version") != 1
        or summary.get("stage") != "stage-a"
        or summary.get("run_id") != run_id
        or summary.get("plan_digest") != plan_digest
        or not run_id.endswith(f"-plan-{plan_digest}")
        or summary.get("stage_a_action_digest") != expected_action_digest
        or summary.get("modal_environment") != "main"
        or summary.get("smoke_receipt_artifact_id") != smoke_id
        or summary.get("model_cache_artifact_id") != cache_id
        or summary.get("bundle_manifest_artifact_id") != bundle_manifest_id
        or summary.get("receipt_approval_history") != expected_approval_history
        or summary.get("training_receipt_ids") != training_ids
        or summary.get("selection_receipt_ids") != selection_ids
        or not isinstance(checked, list)
        or not all(_is_sha256(value) for value in checked)
        or summary.get("elapsed_gpu_seconds") != expected_elapsed
        or not isinstance(provenance, Mapping)
        or set(provenance) != provenance_fields
        or provenance.get("modal_app_name") != APP_NAME
        or provenance.get("modal_function_name") != "finalize_stage_a_remote"
        or any(
            not isinstance(provenance.get(field), str)
            or not provenance.get(field)
            or "\n" in str(provenance.get(field))
            or "\r" in str(provenance.get(field))
            for field in provenance_fields - {"runtime_versions"}
        )
        or not isinstance(runtime_versions, list)
        or len(runtime_versions) != len(LOCKED_RUNTIME_IMPORTS)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"module", "version"}
            or item.get("module") != module
            or not isinstance(item.get("version"), str)
            or not item.get("version")
            for module, item in zip(
                LOCKED_RUNTIME_IMPORTS,
                runtime_versions if isinstance(runtime_versions, list) else [],
                strict=True,
            )
        )
        or not isinstance(summary.get("next_command"), str)
        or not summary.get("next_command")
        or summary.get("stopped_before_behavior") is not True
        or not _is_sha256(summary.get("artifact_id"))
    ):
        raise ValueError("Stage A summary identity or stop contract is invalid")
    unsigned = dict(summary)
    artifact_id = unsigned.pop("artifact_id")
    if artifact_id != sha256_json(unsigned):
        raise ValueError("Stage A summary artifact ID is invalid")
    return summary


def _status_stage_a_action_digest(
    *,
    plan_digest: str,
    resume: bool,
    smoke_receipt_artifact_id: str,
    model_cache_artifact_id: str,
    bundle_manifest_artifact_id: str,
    modal_environment: str,
) -> str:
    if (
        not _is_sha256(plan_digest)
        or not isinstance(resume, bool)
        or not _is_sha256(smoke_receipt_artifact_id)
        or not _is_sha256(model_cache_artifact_id)
        or not _is_sha256(bundle_manifest_artifact_id)
        or modal_environment != "main"
    ):
        raise ValueError("Stage A approval identity is invalid")
    return sha256_json(
        {
            "schema_version": 1,
            "plan_digest": plan_digest,
            "action": "run-stage-a",
            "modal_environment": modal_environment,
            "resume": resume,
            "smoke_receipt_artifact_id": smoke_receipt_artifact_id,
            "model_cache_artifact_id": model_cache_artifact_id,
            "bundle_manifest_artifact_id": bundle_manifest_artifact_id,
        }
    )


def _status_receipt_approval_history(
    receipts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, bool], list[str]] = {}
    for receipt in receipts:
        digest = receipt.get("stage_a_action_digest")
        resume = receipt.get("stage_a_resume")
        artifact_id = receipt.get("artifact_id")
        if (
            not _is_sha256(digest)
            or not isinstance(resume, bool)
            or not _is_sha256(artifact_id)
        ):
            raise ValueError("Stage A receipt approval history is invalid")
        expected = _status_stage_a_action_digest(
            plan_digest=str(receipt.get("plan_digest")),
            resume=resume,
            smoke_receipt_artifact_id=str(
                receipt.get("smoke_receipt_artifact_id")
            ),
            model_cache_artifact_id=str(
                receipt.get("model_cache_artifact_id")
            ),
            bundle_manifest_artifact_id=str(
                receipt.get("bundle_manifest_artifact_id")
            ),
            modal_environment=str(receipt.get("modal_environment")),
        )
        if (
            digest != expected
        ):
            raise ValueError("Stage A receipt approval history is invalid")
        grouped.setdefault((str(digest), resume), []).append(str(artifact_id))
    return [
        {
            "stage_a_action_digest": digest,
            "stage_a_resume": resume,
            "receipt_artifact_ids": sorted(artifact_ids),
        }
        for (digest, resume), artifact_ids in sorted(grouped.items())
    ]


def _revalidate_download_snapshot(
    volume: VolumeClient,
    *,
    run_id: str,
    selected_paths: tuple[str, ...],
    contents: Mapping[str, bytes],
    smoke_receipt_artifact_id: str,
) -> None:
    """Re-read the full receipt-advertised producer set before local publication."""
    receipts: dict[str, list[dict[str, object]]] = {
        "train": [],
        "selection": [],
    }
    shared: set[tuple[str, ...]] = set()
    lineage: set[tuple[str, ...]] = set()
    for stage in ("train", "selection"):
        for arm in _PILOT_ARMS:
            receipt_path = _volume_canonical_receipt_path(run_id, stage, arm)
            relative_receipt = receipt_path.removeprefix(f"/runs/{run_id}/")
            receipt_bytes = _read_volume_file_optional(volume, receipt_path)
            if receipt_bytes is None or receipt_bytes != contents.get(relative_receipt):
                raise ValueError("canonical receipt changed during download")
            producer_path = _volume_producer_path(run_id, stage, arm)
            entries = _list_volume_files_optional(volume, producer_path)
            if not entries:
                raise ValueError("canonical producer changed during download")
            validated, shared_identity, lineage_identity = (
                _validate_status_canonical_output(
                    volume=volume,
                    run_id=run_id,
                    stage=stage,
                    arm=arm,
                    receipt_bytes=receipt_bytes,
                    producer_path=producer_path,
                    entries=entries,
                )
            )
            receipts[stage].append(validated)
            shared.add(shared_identity)
            lineage.add(lineage_identity)
    if len(shared) != 1 or len(lineage) != 1:
        raise ValueError("canonical pilot identity changed during download")
    summary_path = f"/runs/{run_id}/stage-a-summary.json"
    summary = _read_volume_file_optional(volume, summary_path)
    if summary is None or summary != contents.get("stage-a-summary.json"):
        raise ValueError("Stage A summary changed during download")
    _validate_status_summary(
        summary,
        run_id=run_id,
        training_receipts=list(receipts["train"]),
        selection_receipts=list(receipts["selection"]),
    )
    for remote_path in selected_paths:
        relative = _evidence_relative_path(
            run_id,
            remote_path,
            smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        )
        if relative is None:
            raise ValueError("download selection changed during validation")
        current = _read_volume_file_optional(volume, remote_path)
        if current is None or current != contents.get(relative):
            raise ValueError(f"evidence changed during download: {remote_path}")


def _assert_evidence_allowlist_unchanged(
    volume: VolumeClient,
    *,
    run_id: str,
    selected_paths: tuple[str, ...],
    smoke_receipt_artifact_id: str,
) -> None:
    """Relist the exact source set at the final atomic-publication barrier."""
    relisted_paths = _normalized_listed_paths(
        _list_volume_files_optional(volume, f"/runs/{run_id}"),
        f"/runs/{run_id}",
    )
    relisted_selected = tuple(
        path for path in sorted(relisted_paths)
        if _evidence_relative_path(
            run_id,
            path,
            smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        ) is not None
    )
    if relisted_selected != selected_paths:
        raise ValueError("evidence allowlist changed during download")


def _evidence_relative_path(
    run_id: str,
    remote_path: str,
    *,
    smoke_receipt_artifact_id: str | None = None,
) -> str | None:
    prefix = f"/runs/{run_id}/"
    if not remote_path.startswith(prefix):
        return None
    relative = remote_path[len(prefix):]
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        return None
    parts = candidate.parts
    if relative == "stage-a-summary.json":
        return relative
    if relative == "provenance/input-bundle-manifest.json":
        return relative
    if (
        len(parts) == 4
        and parts[:2] == ("receipts", "canonical")
        and parts[2] in {"train", "selection"}
        and parts[3] in {f"{arm}.json" for arm in _PILOT_ARMS}
    ):
        return relative
    if (
        len(parts) == 3
        and parts[:2] == ("receipts", "attempts")
        and parts[2].endswith(".json")
        and _safe_evidence_component(parts[2][:-5])
    ):
        return relative
    if (
        len(parts) == 3
        and parts[:2] == ("receipts", "smoke")
        and parts[2] == f"{smoke_receipt_artifact_id}.json"
        and _is_sha256(smoke_receipt_artifact_id)
    ):
        return relative
    if (
        len(parts) == 4
        and parts[0] == "attempts"
        and _safe_evidence_component(parts[1])
        and parts[2] == "logs"
        and parts[3] in {"train.log", "selection.log"}
    ):
        return relative
    if (
        len(parts) == 7
        and parts[:3] == ("artifacts", "phase-marker", "checkpoints")
        and parts[3] == "pilot"
        and parts[4] == "seed-42"
        and parts[5] in _PILOT_ARMS
        and parts[6] in {"adapter_config.json", "run-manifest.json"}
    ):
        return relative
    if (
        len(parts) == 7
        and parts[:3] == ("artifacts", "phase-marker", "checkpoint-selections")
        and parts[3] == "pilot"
        and parts[4] == "seed-42"
        and parts[5] in _PILOT_ARMS
        and parts[6] in {"manifest.json", "evidence.jsonl"}
    ):
        return relative
    return None


def _safe_evidence_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value


def _validate_smoke_receipt(
    content: bytes,
    *,
    relative_path: str,
    run_id: str,
    approved_shared: tuple[str, ...],
) -> dict[str, object]:
    smoke = _decode_json_object(content, "smoke receipt")
    unsigned = dict(smoke)
    artifact_id = unsigned.pop("artifact_id", None)
    imports = smoke.get("imports")
    bundle_files = smoke.get("bundle_files")
    artifact_ids = [
        smoke.get("split_artifact_id"),
        *(smoke.get("materialization_artifact_ids") or []),
    ]
    bundle_payload = {
        "schema_version": 1,
        "files": bundle_files,
        "artifact_ids": artifact_ids,
    }
    bundle_manifest_payload = {
        "schema_version": 1,
        "bundle_id": smoke.get("bundle_id"),
        "files": bundle_files,
        "artifact_ids": artifact_ids,
    }
    if (
        len(approved_shared) != 10 + len(_PILOT_ARMS)
        or set(smoke) != _SMOKE_RECEIPT_FIELDS
        or smoke.get("schema_version") != 1
        or smoke.get("stage") != "smoke"
        or smoke.get("hardware") != "CPU"
        or smoke.get("run_id") != run_id
        or smoke.get("plan_digest") != approved_shared[6]
        or smoke.get("config_hash") != approved_shared[7]
        or smoke.get("split_artifact_id") != approved_shared[8]
        or smoke.get("materialization_artifact_ids")
        != list(approved_shared[9:-1])
        or smoke.get("source_hash") != approved_shared[2]
        or smoke.get("dependency_lock_hash") != approved_shared[3]
        or smoke.get("canonical_dependency_lock_path")
        != "requirements-modal-phase-marker.txt"
        or smoke.get("bundle_id") != approved_shared[0]
        or smoke.get("bundle_manifest_artifact_id") != approved_shared[1]
        or smoke.get("modal_environment") != approved_shared[-1]
        or not isinstance(bundle_files, list)
        or len(bundle_files) != len(INPUT_ALLOWLIST)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"path", "size", "sha256"}
            for item in bundle_files
        )
        or tuple(item["path"] for item in bundle_files) != INPUT_ALLOWLIST
        or smoke.get("bundle_id") != sha256_json(bundle_payload)
        or smoke.get("bundle_manifest_artifact_id")
        != _sha256_bytes(
            (canonical_json(bundle_manifest_payload) + "\n").encode("utf-8")
        )
        or smoke.get("model_revision") != QWEN25_7B_TOKENIZER_REVISION
        or smoke.get("model_cache_artifact_id") != approved_shared[4]
        or artifact_id != approved_shared[5]
        or not isinstance(imports, list)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"module", "version"}
            or not isinstance(item.get("module"), str)
            or (
                item.get("version") is not None
                and not isinstance(item.get("version"), str)
            )
            for item in imports
        )
        or tuple(item["module"] for item in imports) != LOCKED_RUNTIME_IMPORTS
        or any(
            not isinstance(item.get("version"), str)
            or not item.get("version")
            for item in imports
        )
        or smoke.get("modal_app_name") != APP_NAME
        or smoke.get("modal_function_name") != "smoke_remote"
        or any(
            not isinstance(smoke.get(field), str)
            or not smoke.get(field)
            or "\n" in str(smoke.get(field))
            or "\r" in str(smoke.get(field))
            for field in {
                "modal_app_id", "modal_function_call_id", "modal_input_id",
                "python_version", "torch_version", "cuda_runtime_version",
                "cuda_driver_version",
            }
        )
        or smoke.get("validated") is not True
        or smoke.get("failure_reason") is not None
        or PurePosixPath(relative_path).stem != artifact_id
        or artifact_id != sha256_json(unsigned)
    ):
        raise ValueError("smoke receipt identity mismatch")
    return smoke


def _validate_bundle_provenance(
    content: bytes | None, *, smoke: Mapping[str, object],
) -> dict[str, object]:
    if content is None:
        raise ValueError("input bundle provenance is missing")
    payload = _decode_json_object(content, "input bundle provenance")
    expected = {
        "schema_version": 1,
        "bundle_id": smoke.get("bundle_id"),
        "files": smoke.get("bundle_files"),
        "artifact_ids": [
            smoke.get("split_artifact_id"),
            *(smoke.get("materialization_artifact_ids") or []),
        ],
    }
    if (
        payload != expected
        or _sha256_bytes(content) != smoke.get("bundle_manifest_artifact_id")
        or content != (canonical_json(expected) + "\n").encode("utf-8")
    ):
        raise ValueError("input bundle provenance bytes are invalid")
    return payload


def _validate_downloaded_advertised_hashes(contents: Mapping[str, bytes]) -> None:
    summary = contents.get("stage-a-summary.json")
    if summary is None:
        raise ValueError("downloaded evidence lacks the Stage A summary")
    summary_payload = _decode_json_object(summary, "downloaded Stage A summary")
    unsigned_summary = dict(summary_payload)
    advertised_summary = unsigned_summary.pop("artifact_id", None)
    if advertised_summary != sha256_json(unsigned_summary):
        raise ValueError("downloaded Stage A summary hash mismatch")
    run_id = summary_payload.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("downloaded Stage A summary run identity is invalid")
    canonical_receipts: dict[tuple[str, str], object] = {}
    canonical_shared: set[tuple[str, ...]] = set()
    for stage in ("train", "selection"):
        for arm in _PILOT_ARMS:
            receipt_path = f"receipts/canonical/{stage}/{arm}.json"
            receipt_content = contents.get(receipt_path)
            if receipt_content is None:
                raise ValueError(f"downloaded evidence lacks {receipt_path}")
            receipt = load_attempt_receipt_payload(
                _decode_json_object(receipt_content, "downloaded canonical receipt")
            )
            _validate_receipt_job_identity(receipt, run_id=run_id)
            if getattr(receipt, "stage", None) != stage or getattr(receipt, "arm", None) != arm:
                raise ValueError("downloaded canonical receipt path identity mismatch")
            canonical_receipts[(stage, arm)] = receipt
            canonical_shared.add(_receipt_shared_identity(receipt))
    if len(canonical_shared) != 1:
        raise ValueError("downloaded canonical receipts disagree on pilot identity")
    approved_shared = next(iter(canonical_shared))
    referenced_smoke_id = summary_payload.get("smoke_receipt_artifact_id")
    smoke_paths = {
        path for path in contents if path.startswith("receipts/smoke/")
    }
    expected_smoke_path = f"receipts/smoke/{referenced_smoke_id}.json"
    if (
        referenced_smoke_id != approved_shared[5]
        or smoke_paths != {expected_smoke_path}
    ):
        raise ValueError("downloaded evidence lacks its exact approved smoke receipt")
    smoke_content = contents.get(expected_smoke_path)
    if smoke_content is None:
        raise ValueError("downloaded evidence lacks its exact approved smoke receipt")
    validated_smoke = _validate_smoke_receipt(
        smoke_content,
        relative_path=expected_smoke_path,
        run_id=run_id,
        approved_shared=approved_shared,
    )
    _validate_bundle_provenance(
        contents.get("provenance/input-bundle-manifest.json"),
        smoke=validated_smoke,
    )
    attempt_receipts: dict[str, object] = {}
    for path, content in contents.items():
        if path.startswith("receipts/attempts/"):
            receipt = load_attempt_receipt_payload(
                _decode_json_object(content, "downloaded attempt receipt")
            )
            _validate_attempt_receipt_job_identity(
                receipt,
                run_id=run_id,
                filename=PurePosixPath(path).name,
            )
            if _receipt_shared_identity(receipt) != approved_shared:
                raise ValueError("downloaded attempt receipt pilot identity mismatch")
            attempt_receipts[receipt.attempt_id] = receipt
        elif path.startswith("receipts/smoke/"):
            _validate_smoke_receipt(
                content,
                relative_path=path,
                run_id=run_id,
                approved_shared=approved_shared,
            )
    for path in contents:
        parts = PurePosixPath(path).parts
        if len(parts) == 4 and parts[0] == "attempts" and parts[2] == "logs":
            receipt = attempt_receipts.get(parts[1])
            expected_stage = parts[3].removesuffix(".log")
            if receipt is None or getattr(receipt, "stage", None) != expected_stage:
                raise ValueError("downloaded log lacks its bound attempt receipt")
    for stage in ("train", "selection"):
        for arm in _PILOT_ARMS:
            receipt = canonical_receipts[(stage, arm)]
            producer_kind = (
                "checkpoints" if stage == "train" else "checkpoint-selections"
            )
            producer_root = (
                f"artifacts/phase-marker/{producer_kind}/pilot/seed-42/{arm}"
            )
            advertised = dict(
                zip(receipt.expected_outputs, receipt.output_hashes, strict=True)
            )
            evidence_names = (
                ("adapter_config.json", "run-manifest.json")
                if stage == "train"
                else ("manifest.json", "evidence.jsonl")
            )
            for name in evidence_names:
                content = contents.get(f"{producer_root}/{name}")
                if content is None or advertised.get(name) != _sha256_bytes(content):
                    raise ValueError(
                        f"downloaded evidence hash mismatch: {stage}/{arm}/{name}"
                    )


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
