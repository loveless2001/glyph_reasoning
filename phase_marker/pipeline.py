"""Read-only stage gates and approval-bound phase-marker command manifests.

This module never launches a subprocess, loads a tokenizer/model, creates an
artifact directory, or writes a command manifest.  It validates immutable
artifacts and prints/returns commands as data for an operator to review.
"""

from __future__ import annotations

import argparse
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
DEFAULT_CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
_SHA256_LENGTH = 64


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


class GateFailure(ValueError):
    """An immutable prerequisite is absent, malformed, or stale."""


def validate_run_request(
    kind: str, seeds: Sequence[int], config: ExperimentConfig
) -> GateResult:
    """Enforce the excluded pilot and exact confirmatory seed partitions."""
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
                    "estimated_gpu_hours": None,
                    "command": shlex.join(arguments),
                    "expected_outputs": [
                        str(output_dir / "adapter_config.json"),
                        str(output_dir / "adapter_model.safetensors"),
                        str(manifest),
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
    )


def _run_gate(
    stage: str,
    config: ExperimentConfig,
    artifact_root: Path,
    *,
    kind: str,
    seeds: Sequence[int],
    config_path: Path,
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
        elif stage == "intervene":
            checked.append(_validate_activation_manifest(artifact_root))
        # ``splits`` and ``synthetic`` are root stages.  The validated frozen
        # ExperimentConfig is their sole immutable parent.
    except (GateFailure, OSError, UnicodeError) as error:
        return GateResult(stage, False, str(error), tuple(checked), ())

    commands = _commands_for_stage(
        stage,
        config,
        artifact_root,
        kind=kind,
        seeds=tuple(seeds),
        config_path=config_path,
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
        for split_name in ("train", "validation", "test", "exclusions"):
            split_path = canonical_root / f"{split_name}.jsonl"
            if not split_path.is_file():
                raise GateFailure(f"missing split completion output: {split_path}")
            rows = _jsonl_rows(split_path)
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
    return _LoadedManifest(path, payload, artifact_id)


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
    _require_completion(payload, path, canonical=False)
    _require_kind(payload, "phase_marker_behavior_generations", path)
    _require_config(payload, config, path)
    _require_run_identity(payload, path, kind, seeds, config)
    _require_nonnegative_count(payload, path)
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list) or exclusions:
        raise GateFailure(f"behavior exclusion mismatch: {path}")
    _validate_manifest_data_files(payload, path)
    return _artifact_id(payload, path, allow_derived=True)


def _validate_audit_manifest(
    artifact_root: Path,
    behavior_hash: str,
    config: ExperimentConfig,
    *,
    kind: str,
    seeds: tuple[int, ...],
) -> str:
    path = _first_file(
        (artifact_root / "audit" / "manifest.json", artifact_root / "audit.json"),
        "audit manifest",
    )
    payload = _read_object(path, "audit manifest")
    _require_completion(payload, path, canonical=False)
    _require_kind(payload, "phase_marker_manual_audit", path)
    _require_config(payload, config, path)
    _require_run_identity(payload, path, kind, seeds, config)
    parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
    if behavior_hash not in parents:
        raise GateFailure(f"audit parent behavior hash mismatch: {path}")
    if payload.get("passed") is not True:
        raise GateFailure(f"audit completion marker did not pass: {path}")
    disagreements = payload.get("disagreements")
    total = payload.get("total")
    if not isinstance(disagreements, int) or not isinstance(total, int) or total <= 0:
        raise GateFailure(f"audit row counts are malformed: {path}")
    rate = payload.get("rate")
    if not isinstance(rate, (int, float)) or rate != disagreements / total or rate > 0.01:
        raise GateFailure(f"audit row/count disagreement mismatch: {path}")
    return _artifact_id(payload, path, allow_derived=True)


def _validate_synthetic_manifest(
    artifact_root: Path, config: ExperimentConfig
) -> str:
    path = _first_file(
        (artifact_root / "synthetic" / "manifest.json", artifact_root / "synthetic.json"),
        "synthetic manifest",
    )
    payload = _read_object(path, "synthetic manifest")
    _require_completion(payload, path, canonical=path.name == "manifest.json")
    _require_kind(payload, "phase_marker_synthetic_four_state_suite", path)
    if "config_hash" in payload:
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
        for split_name, expected in counts.items():
            rows = _jsonl_rows_required(path.parent / f"{split_name}.jsonl", "synthetic split")
            if len(rows) != expected:
                raise GateFailure(f"synthetic {split_name} row count mismatch")
    return _artifact_id(payload, path, allow_derived=True)


def _validate_activation_manifest(artifact_root: Path) -> str:
    path = _first_file(
        (artifact_root / "activations" / "manifest.json", artifact_root / "capture.json"),
        "activation manifest",
    )
    payload = _read_object(path, "activation manifest")
    _require_completion(payload, path, canonical=path.name == "manifest.json")
    _require_kind(payload, "phase_marker_selected_activations", path)
    parents = _string_list(payload.get("parent_hashes"), "parent hashes", path)
    if not parents:
        raise GateFailure(f"activation manifest has no parents: {path}")
    tensor_name = payload.get("tensor_file")
    tensor_hash = payload.get("tensor_hash")
    if not isinstance(tensor_name, str) or not isinstance(tensor_hash, str):
        raise GateFailure(f"activation completion marker is missing: {path}")
    tensor_path = path.parent / tensor_name
    if not tensor_path.is_file():
        raise GateFailure(f"missing activation tensor: {tensor_path}")
    if _sha256_file(tensor_path) != tensor_hash:
        raise GateFailure(f"activation tensor hash mismatch: {tensor_path}")
    return _artifact_id(payload, path)


def _commands_for_stage(
    stage: str,
    config: ExperimentConfig,
    artifact_root: Path,
    *,
    kind: str,
    seeds: tuple[int, ...],
    config_path: Path,
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
            )
        )
    commands: dict[str, tuple[str, ...]] = {
        "splits": (
            shlex.join(
                (
                    "./.venv/bin/python", "-m", "phase_marker.splits", "build",
                    "--config", str(config_path), "--traces", "data/sft_final.jsonl",
                    "--unified", "data/unified_dataset.jsonl", "--output-root",
                    str(artifact_root / "splits"),
                )
            ),
        ),
        "render": (
            shlex.join(
                ("./.venv/bin/python", "-m", "phase_marker.traces", "audit", "--input", "data/sft_final.jsonl", "--output", str(artifact_root / "trace-audit.jsonl"))
            ),
        ),
        "tokenize": (
            shlex.join(
                ("./.venv/bin/python", "-m", "phase_marker.token_audit", "materialize", "--config", str(config_path), "--limit", "2455", "--output-root", str(artifact_root / "training-data"))
            ),
        ),
        "behavior": tuple(
            shlex.join(
                ("./.venv/bin/python", "-m", "phase_marker.behavior", "run", "--config", str(config_path), "--kind", kind, "--seed", str(seed), "--artifact-root", str(artifact_root))
            )
            for seed in seeds
        ),
        "audit": (
            shlex.join(("./.venv/bin/python", "-m", "phase_marker.statistics", "audit", "--artifact-root", str(artifact_root))),
        ),
        "statistics": (
            shlex.join(("./.venv/bin/python", "-m", "phase_marker.statistics", "analyze", "--config", str(config_path), "--generations", str(artifact_root / "raw-generations"), "--manual-audit", str(artifact_root / "audit" / "manual-labels.tsv"), "--output-root", str(artifact_root / "analysis"))),
        ),
        "synthetic": (
            shlex.join(("./.venv/bin/python", "-m", "phase_marker.synthetic", "build", "--seed", str(seeds[0]), "--train", "100", "--validation", "20", "--test", "20", "--output-root", str(artifact_root / "synthetic"))),
        ),
        "capture": (
            shlex.join(("./.venv/bin/python", "-m", "phase_marker.activations", "capture", "--config", str(config_path), "--artifact-root", str(artifact_root))),
        ),
        "intervene": (
            shlex.join(("./.venv/bin/python", "-m", "phase_marker.interventions", "run", "--config", str(config_path), "--artifact-root", str(artifact_root))),
        ),
    }
    return commands[stage]


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

    listed = commands.add_parser("commands", help="serialize exact training commands")
    listed.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    listed.add_argument("--arms", nargs="+", required=True)
    listed.add_argument("--seeds", type=int, nargs="+", required=True)
    listed.add_argument("--config", type=Path, required=True)
    listed.add_argument("--artifact-root", type=Path, required=True)
    return parser


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
            )
        except ValueError as error:
            print(canonical_json({"passed": False, "reason": str(error), "jobs": []}))
            return 1
        print(canonical_json({"passed": True, "jobs": list(jobs)}))
        return 0
    result = _run_gate(
        arguments.stage,
        config,
        arguments.artifact_root,
        kind=arguments.kind,
        seeds=tuple(arguments.seeds),
        config_path=arguments.config,
    )
    print(canonical_json(asdict(result)))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
