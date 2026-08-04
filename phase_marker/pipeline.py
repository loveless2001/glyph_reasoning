"""Read-only stage gates and approval-bound phase-marker command manifests.

This module never launches a subprocess, loads a tokenizer/model, creates an
artifact directory, or writes a command manifest.  It validates immutable
artifacts and prints/returns commands as data for an operator to review.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.splits import question_hash
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


STAGES = (
    "splits",
    "render",
    "tokenize",
    "train",
    "behavior",
    "audit",
    "statistics",
    "synthetic",
    "capture",
    "intervene",
)
CANONICAL_ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
CANONICAL_PILOT_SEED = 42
CANONICAL_CONFIRMATORY_SEEDS = (101, 202, 303)
DEFAULT_CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
_SHA256_LENGTH = 64
_APPROVAL_FIELDS = (
    "hardware",
    "max_duration_hours",
    "estimated_gpu_hours",
    "spend_cap_usd",
    "estimated_spend_usd",
    "evaluation_workload",
)
_BEHAVIOR_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "evidence_scope",
        "backend",
        "config_hash",
        "run_kind",
        "seeds",
        "split_artifact_id",
        "split_manifest_hash",
        "materialization_artifact_ids",
        "checkpoint_artifact_ids",
        "checkpoint_manifest_hashes",
        "checkpoint_manifests",
        "examples_file",
        "examples_hash",
        "records_file",
        "records_hash",
        "row_count",
        "record_hashes",
        "exclusions",
        "parent_hashes",
        "completed",
        "artifact_id",
    }
)


@dataclass(frozen=True)
class GateResult:
    stage: str
    passed: bool
    reason: str
    checked_hashes: tuple[str, ...]
    next_commands: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checked_hashes", tuple(self.checked_hashes))
        object.__setattr__(self, "next_commands", tuple(self.next_commands))


@dataclass(frozen=True)
class ApprovalMetadata:
    hardware: str
    max_duration_hours: float
    estimated_gpu_hours: float
    spend_cap_usd: float
    estimated_spend_usd: float
    evaluation_workload: str

    def __post_init__(self) -> None:
        if not self.hardware.strip() or not self.evaluation_workload.strip():
            raise ValueError("approval hardware and evaluation workload must be nonempty")
        numeric = (
            self.max_duration_hours,
            self.estimated_gpu_hours,
            self.spend_cap_usd,
            self.estimated_spend_usd,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            for value in numeric
        ):
            raise ValueError("approval duration, GPU-hours, and spend values must be positive")
        if self.estimated_gpu_hours > self.max_duration_hours:
            raise ValueError("estimated GPU-hours cannot exceed the maximum duration")
        if self.estimated_spend_usd > self.spend_cap_usd:
            raise ValueError("estimated spend cannot exceed the spend cap")


class GateFailure(ValueError):
    """An immutable prerequisite is absent, malformed, or stale."""


def validate_run_request(
    kind: str, seeds: Sequence[int], config: ExperimentConfig
) -> GateResult:
    """Enforce the excluded pilot and exact confirmatory seed partitions."""
    if (
        tuple(config.arms) != CANONICAL_ARMS
        or config.pilot_seed != CANONICAL_PILOT_SEED
        or tuple(config.confirmatory_seeds) != CANONICAL_CONFIRMATORY_SEEDS
    ):
        return GateResult(
            "run_request",
            False,
            "configuration drifts from the frozen protocol arms or seeds",
            (),
            (),
        )
    normalized = tuple(seeds)
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in normalized):
        return GateResult("run_request", False, "seeds must be integers", (), ())
    if kind == "pilot":
        expected = (config.pilot_seed,)
        reason = f"pilot runs require exactly the excluded seed {config.pilot_seed}"
    elif kind == "confirmatory":
        expected = tuple(config.confirmatory_seeds)
        reason = (
            "confirmatory runs require exactly seeds "
            + ",".join(str(seed) for seed in config.confirmatory_seeds)
            + " and must never include the pilot seed"
        )
    else:
        return GateResult(
            "run_request", False, f"unknown run kind {kind!r}", (), ()
        )
    if normalized != expected:
        return GateResult("run_request", False, reason, (), ())
    return GateResult(
        "run_request", True, f"valid {kind} request", (_config_hash(config),), ()
    )


def build_command_manifest(
    config: ExperimentConfig,
    artifact_root: Path,
    *,
    kind: str,
    seeds: Sequence[int],
    config_path: Path = DEFAULT_CONFIG_PATH,
    arms: Sequence[str] | None = None,
    approval: ApprovalMetadata | None = None,
) -> tuple[dict[str, object], ...]:
    """Return exact training commands and expected outputs as inert data."""
    request = validate_run_request(kind, seeds, config)
    if not request.passed:
        raise ValueError(request.reason)
    requested_arms = tuple(config.arms if arms is None else arms)
    if requested_arms != tuple(config.arms):
        raise ValueError("command manifests require all six configured arms in frozen order")
    jobs: list[dict[str, object]] = []
    for seed in seeds:
        for arm in requested_arms:
            data = artifact_root / "training-data" / f"{arm}.jsonl"
            output_dir = artifact_root / "checkpoints" / kind / f"seed-{seed}" / arm
            manifest = output_dir / "run-manifest.json"
            selection_output = (
                artifact_root / "checkpoint-selections" / kind / f"seed-{seed}" / f"{arm}.json"
            )
            arguments = (
                "./.venv/bin/python",
                "-m",
                "phase_marker.training",
                "train",
                "--config",
                str(config_path),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--data",
                str(data),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest),
            )
            jobs.append(
                {
                    "kind": kind,
                    "arm": arm,
                    "seed": seed,
                    "model_id": config.model_id,
                    "model_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "approval_required": True,
                    "approval_ready": approval is not None,
                    "missing_approval_fields": (
                        [] if approval is not None else list(_APPROVAL_FIELDS)
                    ),
                    "approval": None if approval is None else asdict(approval),
                    "estimated_gpu_hours": (
                        None if approval is None else approval.estimated_gpu_hours
                    ),
                    "command": shlex.join(arguments),
                    "selection_command": shlex.join((
                        "./.venv/bin/python", "-m", "phase_marker.behavior", "select",
                        "--config", str(config_path), "--kind", kind, "--seed", str(seed),
                        "--arm", arm, "--split-manifest",
                        str(artifact_root / "splits" / "manifest.json"),
                        "--validation-examples", str(artifact_root / "splits" / "validation.jsonl"),
                        "--training-manifest", str(manifest), "--backend", "vllm",
                        "--output", str(selection_output),
                    )),
                    "expected_outputs": [
                        str(output_dir / "adapter_config.json"),
                        str(output_dir / "adapter_model.safetensors"),
                        str(manifest),
                        str(selection_output),
                    ],
                }
            )
    return tuple(jobs)


def run_gate(
    stage: str, config: ExperimentConfig, artifact_root: Path
) -> GateResult:
    """Validate a confirmatory stage without executing its returned commands."""
    return _run_gate(
        stage,
        config,
        Path(artifact_root),
        kind="confirmatory",
        seeds=config.confirmatory_seeds,
        config_path=DEFAULT_CONFIG_PATH,
        approval=None,
    )


def _run_gate(
    stage: str,
    config: ExperimentConfig,
    artifact_root: Path,
    *,
    kind: str,
    seeds: Sequence[int],
    config_path: Path,
    approval: ApprovalMetadata | None,
) -> GateResult:
    if stage not in STAGES:
        return GateResult(stage, False, f"unknown stage {stage!r}", (), ())
    request = validate_run_request(kind, seeds, config)
    if not request.passed:
        return GateResult(stage, False, request.reason, (), ())
    checked = [_config_hash(config)]
    try:
        if stage in {"render", "tokenize", "train", "behavior"}:
            split = _validate_split_manifest(artifact_root, config)
            checked.append(split.artifact_id)
        else:
            split = None

        if stage == "train":
            assert split is not None
            checked.extend(
                _validate_materializations(artifact_root, config, split.artifact_id)
            )
        elif stage == "behavior":
            assert split is not None
            expected_materializations: Mapping[str, str] | None = None
            if not (artifact_root / "adapter.json").is_file():
                materialization_hashes = _validate_materializations(
                    artifact_root, config, split.artifact_id
                )
                checked.extend(materialization_hashes)
                expected_materializations = dict(
                    zip(config.arms, materialization_hashes, strict=True)
                )
            checked.extend(
                _validate_training_runs(
                    artifact_root,
                    config,
                    split.artifact_id,
                    kind=kind,
                    seeds=tuple(seeds),
                    expected_materializations=expected_materializations,
                )
            )
            checked.extend(
                _validate_checkpoint_selections(
                    artifact_root, config, split.artifact_id,
                    kind=kind, seeds=tuple(seeds),
                    expected_materializations=expected_materializations,
                )
            )
        elif stage in {"audit", "statistics", "capture"}:
            behavior = _validate_behavior_manifest(
                artifact_root, config, kind=kind, seeds=tuple(seeds)
            )
            checked.append(behavior)
            if stage == "statistics":
                checked.append(
                    _validate_audit_manifest(
                        artifact_root,
                        behavior,
                        config,
                        kind=kind,
                        seeds=tuple(seeds),
                    )
                )
            elif stage == "capture":
                checked.append(_validate_synthetic_manifest(artifact_root, config))
                checked.extend(_validate_capture_inputs(artifact_root, config))
        elif stage == "intervene":
            checked.append(_validate_activation_manifest(artifact_root, config))
            checked.extend(_validate_intervention_inputs(artifact_root, config))
        elif stage == "synthetic":
            checked.append(
                _validate_synthetic_preregistration(
                    artifact_root, config, expected_seed=tuple(seeds)[0]
                )
            )
        # ``splits`` is the sole root stage. Synthetic counts are preregistered.
    except (GateFailure, OSError, UnicodeError) as error:
        return GateResult(stage, False, str(error), tuple(checked), ())

    if stage in {"train", "behavior", "capture", "intervene"} and approval is None:
        return GateResult(
            stage,
            False,
            f"{stage} prerequisites passed but approval metadata is missing",
            tuple(dict.fromkeys(checked)),
            (),
        )
    commands = _commands_for_stage(
        stage,
        config,
        artifact_root,
        kind=kind,
        seeds=tuple(seeds),
        config_path=config_path,
        approval=approval,
    )
    return GateResult(
        stage,
        True,
        f"{stage} prerequisites passed; commands are printed data only",
        tuple(dict.fromkeys(checked)),
        commands,
    )


@dataclass(frozen=True)
class _LoadedManifest:
    path: Path
    payload: Mapping[str, Any]
    artifact_id: str


def _validate_split_manifest(
    artifact_root: Path, config: ExperimentConfig
) -> _LoadedManifest:
    path = _first_file(
        (artifact_root / "splits" / "manifest.json", artifact_root / "split.json"),
        "split manifest",
    )
    payload = _read_object(path, "split manifest")
    _require_completion(payload, path, canonical=path.name == "manifest.json")
    _require_config(payload, config, path)
    artifact_id = _artifact_id(payload, path, allow_label=path.name == "split.json")
    if "overlap_count" in payload and payload["overlap_count"] != 0:
        raise GateFailure(f"split manifest overlap count is not zero: {path}")

    source_counts = payload.get("source_counts")
    canonical_root = path.parent if path.name == "manifest.json" else None
    if canonical_root is not None:
        if not isinstance(source_counts, Mapping):
            raise GateFailure(f"split manifest missing source counts: {path}")
        split_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
        seen_live: dict[str, str] = {}
        for split_name in ("train", "validation", "test", "exclusions"):
            split_path = canonical_root / f"{split_name}.jsonl"
            if not split_path.is_file():
                raise GateFailure(f"missing split completion output: {split_path}")
            rows = _jsonl_rows(split_path)
            split_rows[split_name] = rows
            for row in rows:
                _validate_split_row(row, split_name, split_path)
                if split_name != "exclusions":
                    row_hash = str(row["question_hash"])
                    prior = seen_live.get(row_hash)
                    if prior is not None and prior != split_name:
                        raise GateFailure(
                            f"normalized question hash overlap between {prior} and {split_name}"
                        )
                    seen_live[row_hash] = split_name
            expected = source_counts.get(split_name)
            if not isinstance(expected, Mapping):
                raise GateFailure(f"split manifest missing {split_name} source counts")
            actual = dict(sorted(Counter(_row_source(row, split_path) for row in rows).items()))
            normalized_expected = {str(key): value for key, value in expected.items()}
            if actual != normalized_expected:
                raise GateFailure(f"split {split_name} row/count/exclusion mismatch")
        accounting = payload.get("source_pool_accounting")
        if not isinstance(accounting, Mapping):
            raise GateFailure("split manifest missing source-pool exclusion accounting")
        values = tuple(accounting.get(key) for key in ("input_rows", "parsed", "parse_exclusions"))
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise GateFailure("split manifest has malformed exclusion accounting")
        if values[0] != values[1] + values[2]:
            raise GateFailure("split manifest exclusion accounting mismatch")
        provenance = payload.get("parse_exclusion_provenance")
        if not isinstance(provenance, list) or len(provenance) != values[2]:
            raise GateFailure("split manifest parse exclusion provenance mismatch")
        train_counts = Counter(str(row["source"]) for row in split_rows["train"])
        if train_counts.get("svamp", 0) != 0:
            raise GateFailure("split protocol retains SVAMP in training")
        validation_counts = Counter(
            str(row["source"]) for row in split_rows["validation"]
        )
        if validation_counts != Counter({"gsm8k": 300, "math": 300}):
            raise GateFailure(
                "split protocol requires exactly 300 GSM8K and 300 MATH validation rows"
            )
        test_counts = Counter(str(row["source"]) for row in split_rows["test"])
        if test_counts != Counter({"gsm8k": 1319, "svamp": 1000, "math": 5000}):
            raise GateFailure(
                "split protocol requires full official tests: GSM8K 1319, SVAMP 1000, MATH 5000"
            )
        datasets = payload.get("datasets")
        input_lineage = payload.get("input_lineage")
        if not isinstance(datasets, list) or not isinstance(input_lineage, Mapping):
            raise GateFailure("split manifest dataset/input lineage is malformed")
        expected_artifact_id = sha256_json(
            {
                "config": asdict(config),
                "train": list(split_rows["train"]),
                "validation": list(split_rows["validation"]),
                "test": list(split_rows["test"]),
                "exclusions": list(split_rows["exclusions"]),
                "datasets": datasets,
                "input_lineage": dict(input_lineage),
                "source_pool_accounting": dict(accounting),
                "parse_exclusion_provenance": provenance,
            }
        )
        if artifact_id != expected_artifact_id:
            raise GateFailure(
                f"split artifact hash does not match recomputed producer artifact: {path}"
            )
    return _LoadedManifest(path, payload, artifact_id)


def _validate_split_row(
    row: Mapping[str, Any], split_name: str, path: Path
) -> None:
    required = {"source", "split", "example_id", "question", "answer", "question_hash"}
    if set(row) != required:
        raise GateFailure(f"split row schema mismatch: {path}")
    for field in required:
        if not isinstance(row[field], str):
            raise GateFailure(f"split row {field} is malformed: {path}")
    recorded_split = str(row["split"])
    if split_name == "exclusions":
        if not recorded_split.startswith("excluded_"):
            raise GateFailure(f"split exclusion row has invalid split marker: {path}")
    elif recorded_split != split_name:
        raise GateFailure(f"split row split marker mismatch: {path}")
    expected_hash = question_hash(str(row["source"]), str(row["question"]))
    if row["question_hash"] != expected_hash:
        raise GateFailure(f"split row normalized question hash mismatch: {path}")


def _validate_materializations(
    artifact_root: Path, config: ExperimentConfig, split_hash: str
) -> tuple[str, ...]:
    root = artifact_root / "training-data"
    checked: list[str] = []
    row_counts: set[int] = set()
    semantic_hashes: set[str] = set()
    for arm in config.arms:
        path = root / f"{arm}.manifest.json"
        payload = _read_required_object(path, f"{arm} materialization manifest")
        _require_completion(payload, path, canonical=True)
        _require_kind(payload, "phase_marker_training_data", path)
        _require_config(payload, config, path)
        artifact_id = _artifact_id(payload, path)
        parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise GateFailure(f"materialization metadata is malformed: {path}")
        if parents != (split_hash,) or metadata.get("parent_split_hash") != split_hash:
            raise GateFailure(f"materialization parent split hash mismatch: {path}")
        if metadata.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION:
            raise GateFailure(f"materialization tokenizer revision mismatch: {path}")
        exclusions = metadata.get("exclusions")
        if not isinstance(exclusions, list) or exclusions:
            raise GateFailure(f"materialization exclusion mismatch: {path}")
        data_path = root / f"{arm}.jsonl"
        rows = _jsonl_rows_required(data_path, f"{arm} materialization output")
        row_count = payload.get("row_count")
        if row_count != len(rows):
            raise GateFailure(f"materialization row count mismatch: {path}")
        row_hashes = metadata.get("row_hashes")
        if row_hashes != [sha256_json(row) for row in rows]:
            raise GateFailure(f"materialization row hash mismatch: {path}")
        expected_artifact_id = sha256_json(
            {
                "arm": arm,
                "config_hash": _config_hash(config),
                "parent_split_hash": split_hash,
                "row_hashes": row_hashes,
                "metadata": dict(metadata),
            }
        )
        if artifact_id != expected_artifact_id:
            raise GateFailure(
                f"materialization artifact id does not match producer payload: {path}"
            )
        semantic_hash = metadata.get("semantic_dataset_hash")
        if not isinstance(semantic_hash, str) or not semantic_hash:
            raise GateFailure(f"materialization semantic hash is missing: {path}")
        if arm in {"semantic", "glyph", "dot", "random"}:
            semantic_hashes.add(semantic_hash)
        row_counts.add(len(rows))
        checked.append(artifact_id)
    if len(row_counts) != 1:
        raise GateFailure("materialization row count mismatch across arms")
    if len(semantic_hashes) != 1:
        raise GateFailure("marker-only materialization semantic hash mismatch")
    return tuple(checked)


def _validate_training_runs(
    artifact_root: Path,
    config: ExperimentConfig,
    split_hash: str,
    *,
    kind: str,
    seeds: tuple[int, ...],
    expected_materializations: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    fallback = artifact_root / "adapter.json"
    if fallback.is_file():
        candidates = (fallback,)
    else:
        canonical = artifact_root / "checkpoints" / kind
        candidates = tuple(sorted(canonical.glob("seed-*/*/run-manifest.json")))
        if not candidates:
            candidates = tuple(sorted((artifact_root / "checkpoints").glob("**/run-manifest.json")))
    if not candidates:
        raise GateFailure("missing training run manifests")

    expected = {(seed, arm) for seed in seeds for arm in config.arms}
    observed: set[tuple[int, str]] = set()
    checked: list[str] = []
    for path in candidates:
        payload = _read_object(path, "training run manifest")
        _require_completion(payload, path, canonical=path.name == "run-manifest.json")
        _require_kind(payload, "phase_marker_training_run", path)
        _require_config(payload, config, path)
        recorded_kind = payload.get("run_kind")
        if recorded_kind is not None and recorded_kind != kind:
            raise GateFailure(f"training run kind mismatch: {path}")
        seed = payload.get("seed")
        arm = payload.get("arm")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise GateFailure(f"training run seed is malformed: {path}")
        if not isinstance(arm, str) or arm not in config.arms:
            raise GateFailure(f"training run arm is malformed: {path}")
        if kind == "confirmatory" and seed == config.pilot_seed:
            raise GateFailure(f"confirmatory training run mixes pilot seed: {path}")
        parent_split = payload.get("parent_split_hash")
        data_parents = payload.get("data_parent_hashes")
        if parent_split is None and isinstance(data_parents, list) and len(data_parents) == 1:
            parent_split = data_parents[0]
        if parent_split != split_hash:
            raise GateFailure(f"training run parent split hash mismatch: {path}")
        if expected_materializations is not None:
            expected_data = expected_materializations[arm]
            data_parents = _string_list(
                payload.get("parent_hashes"), "parent hashes", path
            )
            if (
                payload.get("data_artifact_id") != expected_data
                or data_parents != (expected_data,)
            ):
                raise GateFailure(
                    f"training run materialization parent mismatch: {path}"
                )
        saved = payload.get("saved_artifacts")
        required_saved = {"adapter", "tokenizer", "trainer_state"}
        if path.name == "run-manifest.json":
            if not isinstance(saved, list) or not required_saved.issubset(set(saved)):
                raise GateFailure(f"training run completion markers are incomplete: {path}")
            if not _is_sha256(payload.get("output_hash")):
                raise GateFailure(f"training run output completion hash is missing: {path}")
            if (
                payload.get("model_revision") != QWEN25_7B_TOKENIZER_REVISION
                or payload.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION
            ):
                raise GateFailure(f"training run model/tokenizer revision mismatch: {path}")
            data_path = artifact_root / "training-data" / f"{arm}.jsonl"
            if payload.get("dataset_path") != str(data_path):
                raise GateFailure(f"training run dataset path mismatch: {path}")
            if payload.get("dataset_hash") != sha256_json(
                data_path.read_bytes().hex()
            ):
                raise GateFailure(f"training run dataset hash mismatch: {path}")
            for required_file in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer_config.json",
                "trainer_state.json",
            ):
                if not (path.parent / required_file).is_file():
                    raise GateFailure(
                        f"training run completion file is missing: {path.parent / required_file}"
                    )
            if payload.get("output_hash") != _directory_hash(path.parent):
                raise GateFailure(f"training run output tree hash mismatch: {path}")
        elif saved is not None and (
            not isinstance(saved, list) or not required_saved.issubset(set(saved))
        ):
            raise GateFailure(f"training run completion markers are incomplete: {path}")
        identity = (seed, arm)
        if identity in observed:
            raise GateFailure(f"duplicate training run manifest for seed/arm {identity!r}")
        observed.add(identity)
        checked.append(_artifact_id(payload, path, allow_derived=True, allow_label=True))

    # A flat compatibility manifest is validated for stale lineage before the
    # complete matrix check, preserving a useful fail-closed diagnostic.
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise GateFailure(f"training run seed/arm completion mismatch: missing={missing}, extra={extra}")
    return tuple(checked)


def _validate_behavior_manifest(
    artifact_root: Path,
    config: ExperimentConfig,
    *,
    kind: str,
    seeds: tuple[int, ...],
) -> str:
    path = _first_file(
        (
            artifact_root / "raw-generations" / kind / "manifest.json",
            artifact_root / "raw-generations" / "manifest.json",
            artifact_root / "behavior.json",
        ),
        "behavior manifest",
    )
    payload = _read_object(path, "behavior manifest")
    if set(payload) != _BEHAVIOR_MANIFEST_FIELDS:
        raise GateFailure(f"behavior manifest schema_version 1 fields mismatch: {path}")
    if (
        payload.get("schema_version") != 1
        or payload.get("evidence_scope") != "experiment_candidate"
        or payload.get("backend") != "vllm"
    ):
        raise GateFailure(f"production behavior gate rejects test/plumbing evidence: {path}")
    _require_completion(payload, path, canonical=False)
    _require_kind(payload, "phase_marker_behavior_generations", path)
    _require_config(payload, config, path)
    _require_run_identity(payload, path, kind, seeds, config)
    _require_nonnegative_count(payload, path)
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list) or exclusions:
        raise GateFailure(f"behavior exclusion mismatch: {path}")
    records_file = payload.get("records_file")
    if not isinstance(records_file, str) or not records_file:
        raise GateFailure(f"behavior records file is missing: {path}")
    records_path = path.parent / records_file
    rows = _jsonl_rows_required(records_path, "behavior records")
    if not rows or len(rows) != payload.get("row_count"):
        raise GateFailure(f"behavior row count mismatch: {path}")
    if payload.get("records_hash") != _sha256_file(records_path):
        raise GateFailure(f"behavior records file hash mismatch: {path}")
    if payload.get("record_hashes") != [sha256_json(row) for row in rows]:
        raise GateFailure(f"behavior record hashes mismatch: {path}")
    examples_file = payload.get("examples_file")
    if not isinstance(examples_file, str) or not Path(examples_file).is_file():
        raise GateFailure(f"behavior examples completion file is missing: {path}")
    if payload.get("examples_hash") != _sha256_file(Path(examples_file)):
        raise GateFailure(f"behavior examples hash mismatch: {path}")
    split = _validate_split_manifest(artifact_root, config)
    if payload.get("split_artifact_id") != split.artifact_id:
        raise GateFailure(f"behavior split parent mismatch: {path}")
    if payload.get("split_manifest_hash") != _sha256_file(split.path):
        raise GateFailure(f"behavior split manifest byte hash mismatch: {path}")
    checkpoint_paths = payload.get("checkpoint_manifests")
    checkpoint_hashes = payload.get("checkpoint_manifest_hashes")
    checkpoint_ids = payload.get("checkpoint_artifact_ids")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint_paths, checkpoint_hashes, checkpoint_ids)
    ):
        raise GateFailure(f"behavior checkpoint lineage maps are malformed: {path}")
    assert isinstance(checkpoint_paths, Mapping)
    assert isinstance(checkpoint_hashes, Mapping)
    assert isinstance(checkpoint_ids, Mapping)
    if set(checkpoint_paths) != set(checkpoint_hashes) or set(checkpoint_paths) != set(checkpoint_ids):
        raise GateFailure(f"behavior checkpoint lineage key mismatch: {path}")
    for key, value in checkpoint_paths.items():
        if not isinstance(value, str) or not Path(value).is_file():
            raise GateFailure(f"behavior checkpoint manifest is missing for {key}: {value}")
        if checkpoint_hashes[key] != _sha256_file(Path(value)):
            raise GateFailure(f"behavior checkpoint manifest hash mismatch for {key}")
    parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
    expected_parents = (split.artifact_id, *(str(checkpoint_ids[key]) for key in checkpoint_ids))
    if parents != expected_parents:
        raise GateFailure(f"behavior parent hashes mismatch: {path}")
    artifact_id = payload.get("artifact_id")
    unsigned = dict(payload)
    unsigned.pop("artifact_id")
    if artifact_id != sha256_json(unsigned):
        raise GateFailure(f"behavior artifact hash mismatch: {path}")
    return str(artifact_id)


def _validate_checkpoint_selections(
    artifact_root: Path,
    config: ExperimentConfig,
    split_hash: str,
    *,
    kind: str,
    seeds: tuple[int, ...],
    expected_materializations: Mapping[str, str] | None,
) -> tuple[str, ...]:
    paths = tuple(sorted((artifact_root / "checkpoint-selections" / kind).glob("seed-*/*.json")))
    expected_count = len(seeds) * len(config.arms)
    if len(paths) != expected_count:
        raise GateFailure(
            f"checkpoint selection matrix is incomplete: expected {expected_count}, found {len(paths)}"
        )
    from phase_marker.behavior import (
        _load_checkpoint_selections,
        _validate_production_behavior_inputs,
    )

    try:
        selections = _load_checkpoint_selections(
            paths, config, kind, seeds, allow_test=False
        )
    except (ValueError, FileNotFoundError) as error:
        raise GateFailure(str(error)) from error
    split_manifest = artifact_root / "splits" / "manifest.json"
    validation_examples = artifact_root / "splits" / "validation.jsonl"
    try:
        _validate_production_behavior_inputs(
            split_manifest, split_hash, config, selections
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        raise GateFailure(str(error)) from error
    checked: list[str] = []
    for (seed, arm), selection in selections.items():
        expected_training = (
            artifact_root / "checkpoints" / kind / f"seed-{seed}" / arm / "run-manifest.json"
        )
        if (
            selection.get("split_artifact_id") != split_hash
            or selection.get("split_manifest_hash") != _sha256_file(split_manifest)
            or selection.get("validation_examples_file") != str(validation_examples)
            or selection.get("validation_examples_hash") != _sha256_file(validation_examples)
            or selection.get("training_manifest_file") != str(expected_training)
            or selection.get("training_manifest_hash") != _sha256_file(expected_training)
            or (
                expected_materializations is not None
                and selection.get("materialization_artifact_id") != expected_materializations[arm]
            )
        ):
            raise GateFailure(f"checkpoint selection lineage mismatch for seed={seed}, arm={arm}")
        checked.append(str(selection["artifact_id"]))
    return tuple(checked)


def _validate_audit_manifest(
    artifact_root: Path,
    behavior_hash: str,
    config: ExperimentConfig,
    *,
    kind: str,
    seeds: tuple[int, ...],
) -> str:
    path = _first_file(
        (
            artifact_root / "audit" / kind / "manifest.json",
            artifact_root / "audit" / "manifest.json",
            artifact_root / "audit.json",
        ),
        "audit manifest",
    )
    payload = _read_object(path, "audit manifest")
    expected_fields = {
        "schema_version", "kind", "evidence_scope", "config_hash", "run_kind", "seeds",
        "behavior_artifact_id", "behavior_manifest_hash", "labels_file", "labels_hash",
        "row_count", "source_counts", "disagreements", "total", "rate", "passed",
        "parent_hashes", "completed", "artifact_id",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise GateFailure(f"audit manifest must use the exact schema-v1 envelope: {path}")
    if payload.get("evidence_scope") != "experiment":
        raise GateFailure(f"production statistics gate rejects plumbing audit evidence: {path}")
    _require_completion(payload, path, canonical=False)
    _require_kind(payload, "phase_marker_manual_audit", path)
    _require_config(payload, config, path)
    _require_run_identity(payload, path, kind, seeds, config)
    parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
    if parents != (behavior_hash,) or payload.get("behavior_artifact_id") != behavior_hash:
        raise GateFailure(f"audit parent behavior hash mismatch: {path}")
    behavior_path = _first_file(
        (
            artifact_root / "raw-generations" / kind / "manifest.json",
            artifact_root / "raw-generations" / "manifest.json",
            artifact_root / "behavior.json",
        ),
        "behavior manifest",
    )
    if payload.get("behavior_manifest_hash") != _sha256_file(behavior_path):
        raise GateFailure(f"audit behavior-manifest byte hash mismatch: {path}")
    labels_file = payload.get("labels_file")
    if not isinstance(labels_file, str) or not Path(labels_file).is_file():
        raise GateFailure(f"audit label file is missing: {path}")
    labels_path = Path(labels_file)
    if payload.get("labels_hash") != _sha256_file(labels_path):
        raise GateFailure(f"audit label file hash mismatch: {path}")
    with labels_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or any(not row.get("generation_id") or not row.get("source") for row in rows):
        raise GateFailure(f"audit labels TSV is malformed: {labels_path}")
    source_counts = Counter(row["source"] for row in rows)
    if len(rows) != 300 or source_counts != Counter({"gsm8k": 100, "svamp": 100, "math": 100}):
        raise GateFailure("audit requires exactly 300 labels and 100 each for gsm8k, svamp, and math")
    behavior_payload = _read_object(behavior_path, "behavior manifest")
    records_file = behavior_payload.get("records_file")
    if not isinstance(records_file, str):
        raise GateFailure(f"behavior records file is missing: {behavior_path}")
    behavior_rows = _jsonl_rows_required(
        behavior_path.parent / records_file, "behavior records"
    )
    behavior_generations = {
        row.get("generation_id"): row.get("source") for row in behavior_rows
    }
    label_ids = [row["generation_id"] for row in rows]
    if len(label_ids) != len(set(label_ids)) or any(
        behavior_generations.get(row["generation_id"]) != row["source"]
        for row in rows
    ):
        raise GateFailure(
            "audit labels must uniquely bind source-matched behavior generation IDs"
        )
    if payload.get("row_count") != 300 or payload.get("source_counts") != dict(source_counts):
        raise GateFailure("audit manifest must bind the exact 300-row source counts")
    if payload.get("passed") is not True:
        raise GateFailure(f"audit completion marker did not pass: {path}")
    disagreements = payload.get("disagreements")
    total = payload.get("total")
    if not isinstance(disagreements, int) or total != 300:
        raise GateFailure(f"audit row counts are malformed: {path}")
    rate = payload.get("rate")
    if not isinstance(rate, (int, float)) or rate != disagreements / total or rate > 0.01:
        raise GateFailure(f"audit row/count disagreement mismatch: {path}")
    unsigned = dict(payload)
    artifact_id = unsigned.pop("artifact_id", None)
    if artifact_id != sha256_json(unsigned):
        raise GateFailure(f"audit artifact hash mismatch: {path}")
    return str(artifact_id)


def _validate_synthetic_manifest(
    artifact_root: Path, config: ExperimentConfig
) -> str:
    path = _first_file(
        (artifact_root / "synthetic" / "manifest.json", artifact_root / "synthetic.json"),
        "synthetic manifest",
    )
    payload = _read_object(path, "synthetic manifest")
    expected_fields = {
        "schema_version", "kind", "seed", "counts", "family_counts", "split_counts",
        "parameter_overlap", "exact_scorer_agreement", "evidence_scope", "backend",
        "config_hash", "preregistration_hash", "completed", "data_hashes", "artifact_id",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise GateFailure(f"synthetic manifest must use the exact schema-v1 envelope: {path}")
    if payload.get("evidence_scope") != "experiment" or payload.get("backend") != "production":
        raise GateFailure(f"production gate rejects plumbing-only synthetic evidence: {path}")
    _require_completion(payload, path, canonical=path.name == "manifest.json")
    _require_kind(payload, "phase_marker_synthetic_four_state_suite", path)
    _require_config(payload, config, path)
    overlap = payload.get("parameter_overlap")
    if not isinstance(overlap, Mapping) or any(value != 0 for value in overlap.values()):
        raise GateFailure(f"synthetic split exclusion/overlap mismatch: {path}")
    agreement = payload.get("exact_scorer_agreement")
    if not isinstance(agreement, Mapping) or agreement.get("agreeing") != agreement.get("total"):
        raise GateFailure(f"synthetic scorer completion mismatch: {path}")
    counts = payload.get("split_counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise GateFailure(f"synthetic row counts are malformed: {path}")
    if path.name == "manifest.json":
        data_hashes = payload.get("data_hashes")
        if not isinstance(data_hashes, Mapping) or set(data_hashes) != set(counts):
            raise GateFailure(f"synthetic data-file hashes are malformed: {path}")
        for split_name, expected in counts.items():
            split_path = path.parent / f"{split_name}.jsonl"
            rows = _jsonl_rows_required(split_path, "synthetic split")
            if len(rows) != expected:
                raise GateFailure(f"synthetic {split_name} row count mismatch")
            if data_hashes[split_name] != _sha256_file(split_path):
                raise GateFailure(f"synthetic {split_name} file hash mismatch")
    artifact_id = payload.get("artifact_id")
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    if artifact_id != sha256_json(unsigned):
        raise GateFailure(f"synthetic artifact hash mismatch: {path}")
    return str(artifact_id)


def _validate_synthetic_preregistration(
    artifact_root: Path,
    config: ExperimentConfig,
    *,
    expected_seed: int,
) -> str:
    path = artifact_root / "synthetic-preregistration.json"
    payload = _read_required_object(path, "synthetic preregistration")
    expected = {
        "schema_version", "kind", "config_hash", "seed", "counts", "family_balance",
        "conditions", "workspace_lengths", "protocol_hash",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise GateFailure(f"synthetic preregistration must use the exact schema-v1 envelope: {path}")
    if payload.get("kind") != "phase_marker_synthetic_preregistration":
        raise GateFailure(f"synthetic preregistration kind mismatch: {path}")
    _require_config(payload, config, path)
    if payload.get("seed") != expected_seed:
        raise GateFailure(
            f"synthetic preregistration seed does not match requested seed: {path}"
        )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != {"train", "validation", "test"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts.values()
    ):
        raise GateFailure(f"synthetic preregistration counts are malformed: {path}")
    if payload.get("family_balance") != [
        "modular_chain", "affine_chain", "two_source_numeric_composition",
        "string_transformation_composition",
    ]:
        raise GateFailure(f"synthetic preregistration family balance mismatch: {path}")
    if set(payload.get("conditions", ())) != {
        "glyph", "dot", "repeated_glyph", "permuted_glyph", "random_symbol", "no_slot"
    } or set(payload.get("workspace_lengths", ())) != {12, 16, 64}:
        raise GateFailure(f"synthetic preregistration conditions or lengths mismatch: {path}")
    protocol = {key: value for key, value in payload.items() if key != "protocol_hash"}
    if payload.get("protocol_hash") != sha256_json(protocol):
        raise GateFailure(f"synthetic preregistration protocol hash mismatch: {path}")
    return _sha256_file(path)


def _validate_activation_manifest(artifact_root: Path, config: ExperimentConfig) -> str:
    path = _first_file(
        (artifact_root / "activations" / "manifest.json", artifact_root / "capture.json"),
        "activation manifest",
    )
    payload = _read_object(path, "activation manifest")
    expected_fields = {
        "schema_version", "kind", "evidence_scope", "backend", "config_hash", "model_id",
        "model_revision", "mode", "example_ids", "conditions", "layers", "positions",
        "parent_hashes", "tensor_file", "tensor_hash", "tensors", "checkpoint_artifact_id",
        "validation_selection_artifact_id", "behavior_artifact_id", "synthetic_artifact_id",
        "tokenized_batch_artifact_id", "tokenized_batch_manifest_hash", "completed", "artifact_id",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != 1:
        raise GateFailure(f"activation manifest must use the exact schema-v1 envelope: {path}")
    if payload.get("evidence_scope") != "experiment" or payload.get("backend") != "hf":
        raise GateFailure(f"production gate rejects plumbing-only activation evidence: {path}")
    _require_completion(payload, path, canonical=path.name == "manifest.json")
    _require_kind(payload, "phase_marker_selected_activations", path)
    _require_config(payload, config, path)
    if payload.get("model_id") != config.model_id or payload.get("model_revision") != QWEN25_7B_TOKENIZER_REVISION:
        raise GateFailure(f"activation model identity mismatch: {path}")
    parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
    expected_parents = tuple(
        str(payload[key])
        for key in (
            "validation_selection_artifact_id", "tokenized_batch_artifact_id",
            "checkpoint_artifact_id", "behavior_artifact_id", "synthetic_artifact_id",
        )
    )
    if parents != expected_parents or any(not _is_sha256(value) for value in parents):
        raise GateFailure(f"activation manifest parents mismatch: {path}")
    tensor_name = payload.get("tensor_file")
    tensor_hash = payload.get("tensor_hash")
    if not isinstance(tensor_name, str) or not isinstance(tensor_hash, str):
        raise GateFailure(f"activation completion marker is missing: {path}")
    tensor_path = path.parent / tensor_name
    if not tensor_path.is_file():
        raise GateFailure(f"missing activation tensor: {tensor_path}")
    if _sha256_file(tensor_path) != tensor_hash:
        raise GateFailure(f"activation tensor hash mismatch: {tensor_path}")
    artifact_id = payload.get("artifact_id")
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    if artifact_id != sha256_json(unsigned):
        raise GateFailure(f"activation artifact hash mismatch: {path}")
    return str(artifact_id)


def _validate_capture_inputs(artifact_root: Path, config: ExperimentConfig) -> tuple[str, ...]:
    selection = _validate_stage_parent(
        artifact_root / "capture-selection.json", "capture selection", config,
        {"schema_version", "kind", "config_hash", "selected_on", "artifact_id"},
    )
    if selection["payload"].get("selected_on") != "validation":
        raise GateFailure("capture selection must be selected_on=validation")
    checkpoint = _validate_stage_parent(
        artifact_root / "capture-checkpoint.json", "capture checkpoint", config,
        {
            "schema_version", "kind", "config_hash", "model_id", "model_revision",
            "checkpoint_path", "artifact_id",
        },
    )
    if (
        checkpoint["payload"].get("model_id") != config.model_id
        or checkpoint["payload"].get("model_revision") != QWEN25_7B_TOKENIZER_REVISION
        or not Path(str(checkpoint["payload"].get("checkpoint_path"))).is_dir()
    ):
        raise GateFailure("capture checkpoint model identity or path mismatch")
    batch = _validate_stage_parent(
        artifact_root / "capture-batch" / "manifest.json", "capture batch", config,
        {
            "schema_version", "kind", "config_hash", "batch_file", "batch_hash",
            "layers", "positions", "artifact_id",
        },
    )
    batch_file = batch["payload"].get("batch_file")
    if not isinstance(batch_file, str):
        raise GateFailure("capture batch path is malformed")
    batch_path = Path(batch_file)
    if not batch_path.is_file() or batch["payload"].get("batch_hash") != _sha256_file(batch_path):
        raise GateFailure("capture batch file is missing or hash-mismatched")
    return tuple(str(item["artifact_id"]) for item in (selection, checkpoint, batch))


def _validate_intervention_inputs(artifact_root: Path, config: ExperimentConfig) -> tuple[str, ...]:
    selection = _validate_stage_parent(
        artifact_root / "intervention-selection.json", "intervention selection", config,
        {"schema_version", "kind", "config_hash", "selected_on", "artifact_id"},
    )
    if selection["payload"].get("selected_on") != "validation":
        raise GateFailure("intervention selection must be selected_on=validation")
    checkpoint = _validate_stage_parent(
        artifact_root / "capture-checkpoint.json", "intervention checkpoint", config,
        {
            "schema_version", "kind", "config_hash", "model_id", "model_revision",
            "checkpoint_path", "artifact_id",
        },
    )
    if checkpoint["payload"].get("model_id") != config.model_id or checkpoint["payload"].get("model_revision") != QWEN25_7B_TOKENIZER_REVISION:
        raise GateFailure("intervention checkpoint model identity mismatch")
    pairs = _validate_stage_parent(
        artifact_root / "aligned-pairs" / "manifest.json", "aligned pairs", config,
        {
            "schema_version", "kind", "config_hash", "rows_file", "rows_hash",
            "row_count", "row_hashes", "artifact_id",
        },
    )
    rows_file = pairs["payload"].get("rows_file")
    if not isinstance(rows_file, str):
        raise GateFailure("aligned-pairs rows path is malformed")
    rows_path = (artifact_root / "aligned-pairs") / rows_file
    rows = _jsonl_rows_required(rows_path, "aligned pairs")
    required = {
        "pair_id", "recipient_id", "donor_id", "recipient_batch_path", "donor_batch_path",
        "recipient_batch_hash", "donor_batch_hash",
        "target_token_ids", "method", "layer", "positions", "norm_match", "control_name",
    }
    if (
        not rows
        or any(set(row) != required for row in rows)
        or pairs["payload"].get("rows_hash") != _sha256_file(rows_path)
        or pairs["payload"].get("row_count") != len(rows)
        or pairs["payload"].get("row_hashes") != [sha256_json(row) for row in rows]
    ):
        raise GateFailure("aligned-pairs rows schema, count, or hashes mismatch")
    allowed = {
        "residual_patch": {"donor", "random_donor"},
        "ablate": {"zero", "validation_mean", "within_batch_shuffle", "matched_non_marker_position"},
        "kv_transplant": {"donor"},
    }
    for row in rows:
        method = row["method"]
        control = row["control_name"]
        if method not in allowed or control not in allowed[method]:
            raise GateFailure("aligned-pairs method/control allowlist mismatch")
        for path_field, hash_field in (
            ("recipient_batch_path", "recipient_batch_hash"),
            ("donor_batch_path", "donor_batch_hash"),
        ):
            batch_path = Path(str(row[path_field]))
            if not batch_path.is_file() or row[hash_field] != _sha256_file(batch_path):
                raise GateFailure("aligned-pairs batch path or hash mismatch")
    return tuple(str(item["artifact_id"]) for item in (selection, checkpoint, pairs))


def _validate_stage_parent(
    path: Path, label: str, config: ExperimentConfig, fields: set[str]
) -> Mapping[str, object]:
    payload = _read_required_object(path, label)
    if set(payload) != fields or payload.get("schema_version") != 1:
        raise GateFailure(f"{label} must use its exact schema-v1 envelope: {path}")
    _require_config(payload, config, path)
    unsigned = dict(payload)
    artifact_id = unsigned.pop("artifact_id", None)
    if artifact_id != sha256_json(unsigned):
        raise GateFailure(f"{label} artifact hash mismatch: {path}")
    return {"payload": payload, "artifact_id": artifact_id}


def _commands_for_stage(
    stage: str,
    config: ExperimentConfig,
    artifact_root: Path,
    *,
    kind: str,
    seeds: tuple[int, ...],
    config_path: Path,
    approval: ApprovalMetadata | None,
) -> tuple[str, ...]:
    if stage == "train":
        return tuple(
            str(job["command"])
            for job in build_command_manifest(
                config,
                artifact_root,
                kind=kind,
                seeds=seeds,
                config_path=config_path,
                approval=approval,
            )
        )
    if stage == "splits":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.splits", "build",
            "--config", str(config_path), "--traces", "data/sft_final.jsonl",
            "--unified", "data/unified_dataset.jsonl", "--output-root",
            str(artifact_root / "splits"),
        )),)
    if stage == "render":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.traces", "audit",
            "--input", "data/sft_final.jsonl", "--output",
            str(artifact_root / "trace-audit.jsonl"),
        )),)
    if stage == "tokenize":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.token_audit", "materialize",
            "--config", str(config_path), "--limit", "2455", "--output-root",
            str(artifact_root / "training-data"),
        )),)
    if stage == "behavior":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.behavior", "run",
            "--config", str(config_path), "--kind", kind, "--seeds",
            *(str(seed) for seed in seeds), "--split-manifest",
            str(artifact_root / "splits" / "manifest.json"), "--examples",
            str(artifact_root / "splits" / "test.jsonl"), "--checkpoint-manifests",
            *(
                str(artifact_root / "checkpoint-selections" / kind / f"seed-{seed}" / f"{arm}.json")
                for seed in seeds for arm in config.arms
            ), "--backend", "vllm", "--output-root",
            str(artifact_root / "raw-generations" / kind),
        )),)
    if stage == "audit":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.statistics", "audit",
            "--config", str(config_path), "--kind", kind, "--seeds",
            *(str(seed) for seed in seeds), "--generations",
            str(artifact_root / "raw-generations" / kind), "--manual-labels",
            str(artifact_root / "audit" / "manual-labels.tsv"), "--output-root",
            str(artifact_root / "audit" / kind),
        )),)
    if stage == "statistics":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.statistics", "analyze",
            "--config", str(config_path), "--generations",
            str(artifact_root / "raw-generations"), "--manual-audit",
            str(artifact_root / "audit" / "manual-labels.tsv"), "--audit-manifest",
            str(artifact_root / "audit" / kind / "manifest.json"), "--output-root",
            str(artifact_root / "analysis"),
        )),)
    if stage == "synthetic":
        preregistration = _read_required_object(
            artifact_root / "synthetic-preregistration.json", "synthetic preregistration"
        )
        if preregistration.get("seed") != seeds[0]:
            raise GateFailure("synthetic preregistration seed does not match emitted seed")
        value = preregistration.get("counts")
        if not isinstance(value, Mapping):
            raise GateFailure("synthetic preregistration counts are malformed")
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.synthetic", "build",
            "--config", str(config_path), "--preregistration",
            str(artifact_root / "synthetic-preregistration.json"), "--seed", str(seeds[0]),
            "--train", str(value["train"]), "--validation", str(value["validation"]),
            "--test", str(value["test"]), "--backend", "production", "--output-root",
            str(artifact_root / "synthetic"),
        )),)
    if stage == "capture":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.activations", "capture",
            "--config", str(config_path), "--mode", "teacher_forced",
            "--validation-selection-manifest", str(artifact_root / "capture-selection.json"),
            "--tokenized-batch-manifest", str(artifact_root / "capture-batch" / "manifest.json"),
            "--tokenized-batch", str(artifact_root / "capture-batch" / "batch.pt"),
            "--model-id", config.model_id, "--model-revision", QWEN25_7B_TOKENIZER_REVISION,
            "--checkpoint-manifest", str(artifact_root / "capture-checkpoint.json"),
            "--behavior-manifest", str(artifact_root / "raw-generations" / kind / "manifest.json"),
            "--synthetic-manifest", str(artifact_root / "synthetic" / "manifest.json"),
            "--backend", "hf", "--output-root", str(artifact_root / "activations"),
        )),)
    if stage == "intervene":
        return (shlex.join((
            "./.venv/bin/python", "-m", "phase_marker.interventions", "run",
            "--config", str(config_path), "--validation-selection-manifest",
            str(artifact_root / "intervention-selection.json"), "--aligned-pairs-manifest",
            str(artifact_root / "aligned-pairs" / "manifest.json"), "--activation-manifest",
            str(artifact_root / "activations" / "manifest.json"), "--checkpoint-manifest",
            str(artifact_root / "capture-checkpoint.json"), "--model-id", config.model_id,
            "--model-revision", QWEN25_7B_TOKENIZER_REVISION, "--backend", "hf",
            "--output-root", str(artifact_root / "interventions"),
        )),)
    raise GateFailure(f"unknown stage {stage!r}")


def _read_required_object(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise GateFailure(f"missing {label}: {path}")
    return _read_object(path, label)


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GateFailure(f"malformed {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise GateFailure(f"malformed {label}, expected a JSON object: {path}")
    return payload


def _first_file(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise GateFailure(f"missing {label}; checked " + ", ".join(str(path) for path in paths))


def _require_completion(payload: Mapping[str, Any], path: Path, *, canonical: bool) -> None:
    marker = payload.get("completed", payload.get("complete"))
    if marker is False:
        raise GateFailure(f"completion marker is false: {path}")
    if marker is None and not canonical:
        raise GateFailure(f"missing completion marker: {path}")
    if marker is not None and marker is not True:
        raise GateFailure(f"malformed completion marker: {path}")


def _require_config(payload: Mapping[str, Any], config: ExperimentConfig, path: Path) -> None:
    expected = _config_hash(config)
    if payload.get("config_hash") != expected:
        raise GateFailure(f"config hash mismatch: {path}")


def _require_kind(payload: Mapping[str, Any], expected: str, path: Path) -> None:
    if payload.get("kind") != expected:
        raise GateFailure(f"manifest kind mismatch, expected {expected}: {path}")


def _require_run_identity(
    payload: Mapping[str, Any],
    path: Path,
    kind: str,
    seeds: tuple[int, ...],
    config: ExperimentConfig,
) -> None:
    if payload.get("run_kind") != kind:
        raise GateFailure(f"run kind mismatch: {path}")
    recorded = payload.get("seeds")
    if not isinstance(recorded, list) or tuple(recorded) != seeds:
        raise GateFailure(f"run seed mismatch: {path}")
    if kind == "confirmatory" and config.pilot_seed in recorded:
        raise GateFailure(f"confirmatory manifest mixes pilot seed: {path}")


def _require_nonnegative_count(payload: Mapping[str, Any], path: Path) -> None:
    count = payload.get("row_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise GateFailure(f"row count is malformed: {path}")


def _validate_manifest_data_files(payload: Mapping[str, Any], path: Path) -> None:
    data_files = payload.get("data_files")
    if not isinstance(data_files, list) or not data_files:
        raise GateFailure(f"missing data-file completion markers: {path}")
    rows = 0
    for value in data_files:
        if not isinstance(value, str) or not value:
            raise GateFailure(f"malformed data-file completion marker: {path}")
        rows += len(_jsonl_rows_required(path.parent / value, "behavior data file"))
    if rows != payload.get("row_count"):
        raise GateFailure(f"behavior row count mismatch: {path}")


def _artifact_id(
    payload: Mapping[str, Any],
    path: Path,
    *,
    allow_derived: bool = False,
    allow_label: bool = False,
) -> str:
    value = payload.get("artifact_id", payload.get("hash"))
    if _is_sha256(value):
        return str(value)
    if allow_label and isinstance(value, str) and value:
        return value
    if allow_derived:
        return sha256_json(dict(payload))
    raise GateFailure(f"manifest artifact hash is missing or malformed: {path}")


def _string_list(value: object, label: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise GateFailure(f"manifest {label} are malformed: {path}")
    return tuple(value)


def _jsonl_rows_required(path: Path, label: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise GateFailure(f"missing {label}: {path}")
    return _jsonl_rows(path)


def _jsonl_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateFailure(f"malformed JSONL row {path}:{line_number}") from error
            if not isinstance(row, Mapping):
                raise GateFailure(f"malformed JSONL row {path}:{line_number}")
            rows.append(row)
    return tuple(rows)


def _row_source(row: Mapping[str, Any], path: Path) -> str:
    source = row.get("source")
    if not isinstance(source, str) or not source:
        raise GateFailure(f"split row has malformed source: {path}")
    return source


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _config_hash(config: ExperimentConfig) -> str:
    return sha256_json(asdict(config))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    records = [
        {
            "path": str(candidate.relative_to(path)),
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.name != "run-manifest.json"
    ]
    return sha256_json(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry_run = commands.add_parser("dry-run", help="print the confirmatory plan without writes")
    dry_run.add_argument("--config", type=Path, required=True)
    dry_run.add_argument("--artifact-root", type=Path, required=True)

    gate = commands.add_parser("gate", help="validate prerequisites and print commands")
    gate.add_argument("--stage", choices=STAGES, required=True)
    gate.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    gate.add_argument("--seeds", type=int, nargs="+", required=True)
    gate.add_argument("--config", type=Path, required=True)
    gate.add_argument("--artifact-root", type=Path, required=True)
    _add_approval_arguments(gate)

    listed = commands.add_parser("commands", help="serialize exact training commands")
    listed.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    listed.add_argument("--arms", nargs="+", required=True)
    listed.add_argument("--seeds", type=int, nargs="+", required=True)
    listed.add_argument("--config", type=Path, required=True)
    listed.add_argument("--artifact-root", type=Path, required=True)
    _add_approval_arguments(listed)
    return parser


def _add_approval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hardware")
    parser.add_argument("--max-duration-hours", type=float)
    parser.add_argument("--estimated-gpu-hours", type=float)
    parser.add_argument("--spend-cap-usd", type=float)
    parser.add_argument("--estimated-spend-usd", type=float)
    parser.add_argument("--evaluation-workload")


def _approval_from_arguments(arguments: argparse.Namespace) -> ApprovalMetadata | None:
    values = tuple(
        getattr(arguments, field.replace("_usd", "_usd"), None)
        for field in _APPROVAL_FIELDS
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        return None
    return ApprovalMetadata(
        hardware=arguments.hardware,
        max_duration_hours=arguments.max_duration_hours,
        estimated_gpu_hours=arguments.estimated_gpu_hours,
        spend_cap_usd=arguments.spend_cap_usd,
        estimated_spend_usd=arguments.estimated_spend_usd,
        evaluation_workload=arguments.evaluation_workload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = ExperimentConfig.load(arguments.config)
    if arguments.command == "dry-run":
        jobs = build_command_manifest(
            config,
            arguments.artifact_root,
            kind="confirmatory",
            seeds=config.confirmatory_seeds,
            config_path=arguments.config,
            approval=None,
        )
        print(
            canonical_json(
                {
                    "mode": "dry-run",
                    "read_only": True,
                    "models_loaded": False,
                    "commands_executed": False,
                    "config_hash": _config_hash(config),
                    "arms": list(config.arms),
                    "seeds": list(config.confirmatory_seeds),
                    "jobs": list(jobs),
                }
            )
        )
        return 0
    if arguments.command == "commands":
        try:
            jobs = build_command_manifest(
                config,
                arguments.artifact_root,
                kind=arguments.kind,
                seeds=tuple(arguments.seeds),
                arms=tuple(arguments.arms),
                config_path=arguments.config,
                approval=_approval_from_arguments(arguments),
            )
        except ValueError as error:
            print(canonical_json({"passed": False, "reason": str(error), "jobs": []}))
            return 1
        ready = all(job["approval_ready"] is True for job in jobs)
        print(
            canonical_json(
                {
                    "passed": ready,
                    "approval_ready": ready,
                    "reason": (
                        "approval metadata complete"
                        if ready
                        else "approval metadata is missing"
                    ),
                    "jobs": list(jobs),
                }
            )
        )
        return 0 if ready else 1
    result = _run_gate(
        arguments.stage,
        config,
        arguments.artifact_root,
        kind=arguments.kind,
        seeds=tuple(arguments.seeds),
        config_path=arguments.config,
        approval=_approval_from_arguments(arguments),
    )
    print(canonical_json(asdict(result)))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
