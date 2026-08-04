from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import subprocess

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.pipeline import GateResult, main, run_gate, validate_run_request
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig.load(CONFIG_PATH)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _write_split(root: Path, config: ExperimentConfig, *, artifact_id: str = "a" * 64) -> None:
    split_root = root / "splits"
    rows = {
        "train": ({"source": "gsm8k", "question_hash": "train-q"},),
        "validation": ({"source": "gsm8k", "question_hash": "validation-q"},),
        "test": ({"source": "svamp", "question_hash": "test-q"},),
        "exclusions": ({"source": "svamp", "question_hash": "excluded-q"},),
    }
    for name, values in rows.items():
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / f"{name}.jsonl").write_text(
            "".join(canonical_json(value) + "\n" for value in values),
            encoding="utf-8",
        )
    _write_json(
        split_root / "manifest.json",
        {
            "artifact_id": artifact_id,
            "config_hash": sha256_json(asdict(config)),
            "overlap_count": 0,
            "source_pool_accounting": {
                "input_rows": 2,
                "parsed": 1,
                "parse_exclusions": 1,
            },
            "parse_exclusion_provenance": ["line-2|phase_markers"],
            "source_counts": {
                "train": {"gsm8k": 1},
                "validation": {"gsm8k": 1},
                "test": {"svamp": 1},
                "exclusions": {"svamp": 1},
            },
        },
    )


def _write_materializations(root: Path, config: ExperimentConfig) -> None:
    config_hash = sha256_json(asdict(config))
    split_hash = "a" * 64
    shared_semantic_hash = "b" * 64
    for index, arm in enumerate(config.arms):
        row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Final answer: 1"}]}
        data_path = root / "training-data" / f"{arm}.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
        artifact_id = sha256_json({"arm": arm, "index": index})
        _write_json(
            data_path.with_suffix(".manifest.json"),
            {
                "artifact_id": artifact_id,
                "kind": "phase_marker_training_data",
                "config_hash": config_hash,
                "parent_hashes": [split_hash],
                "row_count": 1,
                "metadata": {
                    "semantic_dataset_hash": (
                        shared_semantic_hash if arm in {"semantic", "glyph", "dot", "random"} else sha256_json(arm)
                    ),
                    "row_hashes": [sha256_json(row)],
                    "exclusions": [],
                    "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "parent_split_hash": split_hash,
                },
            },
        )


def _write_training_runs(root: Path, config: ExperimentConfig) -> None:
    config_hash = sha256_json(asdict(config))
    for seed in config.confirmatory_seeds:
        for arm_index, arm in enumerate(config.arms):
            materialization = json.loads(
                (root / "training-data" / f"{arm}.manifest.json").read_text(encoding="utf-8")
            )
            _write_json(
                root
                / "checkpoints"
                / "confirmatory"
                / f"seed-{seed}"
                / arm
                / "run-manifest.json",
                {
                    "kind": "phase_marker_training_run",
                    "arm": arm,
                    "seed": seed,
                    "model_id": config.model_id,
                    "model_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "config_hash": config_hash,
                    "data_artifact_id": materialization["artifact_id"],
                    "parent_hashes": [materialization["artifact_id"]],
                    "data_parent_hashes": ["a" * 64],
                    "saved_artifacts": ["adapter", "tokenizer", "trainer_state"],
                    "output_hash": sha256_json({"seed": seed, "arm": arm, "index": arm_index}),
                },
            )


def test_gate_result_is_frozen() -> None:
    result = GateResult("train", True, "ready", ("a" * 64,), ("command",))
    with pytest.raises(FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kind", "seeds", "passed"),
    [
        ("pilot", (42,), True),
        ("pilot", (101,), False),
        ("pilot", (42, 101), False),
        ("confirmatory", (101, 202, 303), True),
        ("confirmatory", (101, 202), False),
        ("confirmatory", (42,), False),
        ("confirmatory", (101, 202, 303, 42), False),
        ("exploratory", (42,), False),
    ],
)
def test_run_request_enforces_exact_kind_seed_partition(
    config: ExperimentConfig, kind: str, seeds: tuple[int, ...], passed: bool
) -> None:
    assert validate_run_request(kind=kind, seeds=seeds, config=config).passed is passed


def test_train_gate_validates_real_split_and_materialization_manifests(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)

    result = run_gate("train", config, tmp_path)

    assert result.passed
    assert len(result.next_commands) == 18
    assert all("phase_marker.training train" in command for command in result.next_commands)
    assert all(seed in " ".join(result.next_commands) for seed in ("101", "202", "303"))


def test_train_gate_fails_closed_on_row_count_mismatch(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    manifest_path = tmp_path / "training-data" / "glyph.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["row_count"] = 2
    _write_json(manifest_path, payload)

    result = run_gate("train", config, tmp_path)

    assert not result.passed
    assert "row count" in result.reason


def test_behavior_gate_rejects_stale_adapter_parent_before_completeness(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_json(
        tmp_path / "split.json",
        {
            "hash": "split-a",
            "config_hash": sha256_json(asdict(config)),
            "completed": True,
        },
    )
    _write_json(
        tmp_path / "adapter.json",
        {
            "hash": "adapter-a",
            "kind": "phase_marker_training_run",
            "config_hash": sha256_json(asdict(config)),
            "parent_split_hash": "split-b",
            "arm": "glyph",
            "seed": 101,
            "run_kind": "confirmatory",
            "completed": True,
        },
    )

    result = run_gate("behavior", config, tmp_path)

    assert not result.passed
    assert "parent split hash" in result.reason


def test_behavior_gate_rejects_stale_materialization_parent_in_training_run(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    _write_training_runs(tmp_path, config)
    manifest_path = (
        tmp_path
        / "checkpoints"
        / "confirmatory"
        / "seed-101"
        / "glyph"
        / "run-manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["data_artifact_id"] = "0" * 64
    payload["parent_hashes"] = ["0" * 64]
    _write_json(manifest_path, payload)

    result = run_gate("behavior", config, tmp_path)

    assert not result.passed
    assert "materialization parent" in result.reason


def test_behavior_gate_requires_training_completion_markers(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    _write_training_runs(tmp_path, config)
    manifest_path = (
        tmp_path
        / "checkpoints"
        / "confirmatory"
        / "seed-101"
        / "glyph"
        / "run-manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("saved_artifacts")
    _write_json(manifest_path, payload)

    result = run_gate("behavior", config, tmp_path)

    assert not result.passed
    assert "completion markers" in result.reason


@pytest.mark.parametrize(
    "stage",
    ("render", "tokenize", "train", "behavior", "audit", "statistics", "capture", "intervene"),
)
def test_dependent_gates_fail_closed_when_manifests_are_missing(
    tmp_path: Path, config: ExperimentConfig, stage: str
) -> None:
    result = run_gate(stage, config, tmp_path)
    assert not result.passed
    assert "missing" in result.reason


def test_gate_rejects_unknown_stage_without_mutating_root(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    root = tmp_path / "absent"
    result = run_gate("launch-everything", config, root)
    assert not result.passed
    assert "unknown stage" in result.reason
    assert not root.exists()


def test_dry_run_lists_all_arms_and_confirmatory_seeds_without_side_effects(
    tmp_path: Path,
    config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "must-not-exist"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run attempted execution or model loading")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr("phase_marker.training._cached_model_snapshot", forbidden)
    monkeypatch.setattr("phase_marker.token_audit._load_cached_tokenizer", forbidden)

    assert main(
        [
            "dry-run",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            str(root),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    jobs = payload["jobs"]
    assert {(job["arm"], job["seed"]) for job in jobs} == {
        (arm, seed) for arm in config.arms for seed in config.confirmatory_seeds
    }
    assert all(job["approval_required"] is True for job in jobs)
    assert all(job["expected_outputs"] for job in jobs)
    assert all(job["command"].startswith("./.venv/bin/python -m phase_marker.training train ") for job in jobs)
    assert not root.exists()


def test_gate_cli_prints_only_and_never_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "absent"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("gate attempted to execute a command")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert main(
        [
            "gate",
            "--stage",
            "train",
            "--kind",
            "pilot",
            "--seeds",
            "42",
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            str(root),
        ]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["next_commands"] == []
    assert not root.exists()


def test_statistics_gate_rejects_audit_with_wrong_config_hash(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    generation_root = tmp_path / "raw-generations" / "confirmatory"
    generation_root.mkdir(parents=True)
    (generation_root / "records.jsonl").write_text(
        canonical_json({"generation_id": "g-1"}) + "\n", encoding="utf-8"
    )
    behavior = {
        "artifact_id": "c" * 64,
        "kind": "phase_marker_behavior_generations",
        "config_hash": sha256_json(asdict(config)),
        "run_kind": "confirmatory",
        "seeds": [101, 202, 303],
        "row_count": 1,
        "exclusions": [],
        "data_files": ["records.jsonl"],
        "completed": True,
    }
    _write_json(generation_root / "manifest.json", behavior)
    _write_json(
        tmp_path / "audit" / "manifest.json",
        {
            "artifact_id": "d" * 64,
            "kind": "phase_marker_manual_audit",
            "config_hash": "0" * 64,
            "run_kind": "confirmatory",
            "seeds": [101, 202, 303],
            "parent_hashes": [behavior["artifact_id"]],
            "disagreements": 0,
            "total": 100,
            "rate": 0.0,
            "passed": True,
            "completed": True,
        },
    )

    result = run_gate("statistics", config, tmp_path)

    assert not result.passed
    assert "config hash mismatch" in result.reason
