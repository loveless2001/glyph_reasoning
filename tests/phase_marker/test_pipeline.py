from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
from types import SimpleNamespace

import pytest
import phase_marker.pipeline as pipeline_module

from phase_marker.config import ExperimentConfig
from phase_marker.behavior import _selection_generation
from phase_marker.scoring import score_generation
from phase_marker.io import canonical_json, sha256_json
from phase_marker.pipeline import (
    ApprovalMetadata,
    MechanismApprovalMetadata,
    GateResult,
    build_command_manifest,
    main,
    run_gate,
    validate_run_request,
)
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION
from phase_marker.splits import question_hash


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig.load(CONFIG_PATH)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _split_row(source: str, split: str, index: int) -> dict[str, str]:
    question = f"{source} {split} protocol question {index}"
    return {
        "source": source,
        "split": split,
        "example_id": f"{source}-{split}-{index}",
        "question": question,
        "answer": str(index),
        "question_hash": question_hash(source, question),
    }


def _write_split(root: Path, config: ExperimentConfig) -> str:
    split_root = root / "splits"
    rows = {
        "train": (_split_row("gsm8k", "train", 0),),
        "validation": tuple(
            _split_row(source, "validation", index)
            for source in ("gsm8k", "math")
            for index in range(300)
        ),
        "test": tuple(
            _split_row(source, "test", index)
            for source, count in (("gsm8k", 1319), ("svamp", 1000), ("math", 5000))
            for index in range(count)
        ),
        "exclusions": (_split_row("svamp", "excluded_svamp", 0),),
    }
    for name, values in rows.items():
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / f"{name}.jsonl").write_text(
            "".join(canonical_json(value) + "\n" for value in values),
            encoding="utf-8",
        )
    datasets = [
        {
            "source": source,
            "dataset_id": dataset_id,
            "config": dataset_config,
            "requested_split": requested_split,
            "revision": revision,
        }
        for source, dataset_id, dataset_config, requested_split, revision in (
            ("gsm8k", "gsm8k", "main", "train", "1" * 40),
            ("gsm8k", "gsm8k", "main", "test", "1" * 40),
            ("svamp", "ChilleD/SVAMP", None, "train", "2" * 40),
            ("math", "EleutherAI/hendrycks_math", "all", "train", "3" * 40),
            ("math", "EleutherAI/hendrycks_math", "all", "test", "3" * 40),
        )
    ]
    input_lineage = {
        "traces": {"path": "data/sft_final.jsonl", "sha256": "4" * 64},
        "unified": {"path": "data/unified_dataset.jsonl", "sha256": "5" * 64},
    }
    accounting = {"input_rows": 1, "parsed": 1, "parse_exclusions": 0}
    parse_provenance: list[str] = []
    artifact_id = sha256_json(
        {
            "config": asdict(config),
            **{name: list(values) for name, values in rows.items()},
            "datasets": datasets,
            "input_lineage": input_lineage,
            "source_pool_accounting": accounting,
            "parse_exclusion_provenance": parse_provenance,
        }
    )
    _write_json(
        split_root / "manifest.json",
        {
            "artifact_id": artifact_id,
            "config_hash": sha256_json(asdict(config)),
            "datasets": datasets,
            "input_lineage": input_lineage,
            "overlap_count": 0,
            "source_pool_accounting": accounting,
            "parse_exclusion_provenance": parse_provenance,
            "source_counts": {
                name: dict(sorted(Counter(row["source"] for row in values).items()))
                for name, values in rows.items()
            },
        },
    )
    return artifact_id


def _write_materializations(root: Path, config: ExperimentConfig) -> None:
    config_hash = sha256_json(asdict(config))
    split_hash = json.loads(
        (root / "splits" / "manifest.json").read_text(encoding="utf-8")
    )["artifact_id"]
    shared_semantic_hash = "b" * 64
    for index, arm in enumerate(config.arms):
        row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Final answer: 1"}]}
        data_path = root / "training-data" / f"{arm}.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
        metadata = {
            "semantic_dataset_hash": (
                shared_semantic_hash if arm in {"semantic", "glyph", "dot", "random"} else sha256_json(arm)
            ),
            "row_hashes": [sha256_json(row)],
            "exclusions": [],
            "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
            "parent_split_hash": split_hash,
        }
        artifact_id = sha256_json(
            {
                "arm": arm,
                "config_hash": config_hash,
                "parent_split_hash": split_hash,
                "row_hashes": metadata["row_hashes"],
                "metadata": metadata,
            }
        )
        _write_json(
            data_path.with_suffix(".manifest.json"),
            {
                "artifact_id": artifact_id,
                "kind": "phase_marker_training_data",
                "config_hash": config_hash,
                "parent_hashes": [split_hash],
                "row_count": 1,
                "metadata": metadata,
            },
        )


def _write_training_runs(root: Path, config: ExperimentConfig) -> None:
    config_hash = sha256_json(asdict(config))
    split_hash = json.loads(
        (root / "splits" / "manifest.json").read_text(encoding="utf-8")
    )["artifact_id"]
    for seed in config.confirmatory_seeds:
        for arm in config.arms:
            materialization = json.loads(
                (root / "training-data" / f"{arm}.manifest.json").read_text(encoding="utf-8")
            )
            output_dir = root / "checkpoints" / "confirmatory" / f"seed-{seed}" / arm
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "adapter_config.json").write_text(
                canonical_json({
                    "base_model_name_or_path": config.model_id,
                    "revision": QWEN25_7B_TOKENIZER_REVISION,
                }) + "\n", encoding="utf-8"
            )
            (output_dir / "adapter_model.safetensors").write_bytes(
                f"fixture adapter {seed} {arm}".encode()
            )
            (output_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (output_dir / "trainer_state.json").write_text("{}\n", encoding="utf-8")
            checkpoint_dir = output_dir / "checkpoint-100"
            checkpoint_dir.mkdir()
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer_config.json",
                "trainer_state.json",
            ):
                source = output_dir / filename
                target = checkpoint_dir / filename
                target.write_bytes(source.read_bytes())
            checkpoint_records = [
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(checkpoint_dir.rglob("*"))
                if path.is_file()
            ]
            checkpoint_hash = sha256_json(checkpoint_records)
            output_records = [
                {"path": str(path.relative_to(output_dir)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(output_dir.rglob("*"))
                if path.is_file() and path.name != "run-manifest.json"
            ]
            data_path = root / "training-data" / f"{arm}.jsonl"
            _write_json(
                output_dir / "run-manifest.json",
                {
                    "kind": "phase_marker_training_run", "arm": arm, "seed": seed,
                    "model_id": config.model_id, "model_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
                    "config_hash": config_hash, "dataset_path": str(data_path),
                    "dataset_hash": sha256_json(data_path.read_bytes().hex()),
                    "data_artifact_id": materialization["artifact_id"],
                    "parent_hashes": [materialization["artifact_id"]],
                    "data_parent_hashes": [split_hash],
                    "arguments": ["fixture"],
                    "environment": {"fixture": True},
                    "checkpoints": [
                        {"path": "checkpoint-100", "hash": checkpoint_hash}
                    ],
                    "saved_artifacts": ["adapter", "tokenizer", "trainer_state"],
                    "output_hash": sha256_json(output_records),
                },
            )


def _write_checkpoint_selections(root: Path, config: ExperimentConfig) -> None:
    split_manifest = root / "splits" / "manifest.json"
    split_id = json.loads(split_manifest.read_text(encoding="utf-8"))["artifact_id"]
    validation_examples = root / "splits" / "validation.jsonl"
    for seed in config.confirmatory_seeds:
        for arm in config.arms:
            training_path = root / "checkpoints" / "confirmatory" / f"seed-{seed}" / arm / "run-manifest.json"
            checkpoint_path = training_path.parent / "checkpoint-100"
            checkpoint_records = [
                {"path": str(path.relative_to(checkpoint_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(checkpoint_path.rglob("*"))
                if path.is_file() and path.name != "run-manifest.json"
            ]
            checkpoint_hash = sha256_json(checkpoint_records)
            materialization_id = json.loads(
                (root / "training-data" / f"{arm}.manifest.json").read_text(encoding="utf-8")
            )["artifact_id"]
            training_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()
            candidate = {
                "path": str(checkpoint_path), "checkpoint_hash": checkpoint_hash, "step": 100,
                "strict_accuracy": 0.0, "mean_gold_answer_logprob": -1.0, "row_count": 600,
            }
            selection_path = (
                root / "checkpoint-selections" / "confirmatory" / f"seed-{seed}" / arm / "manifest.json"
            )
            evidence_path = selection_path.parent / "evidence.jsonl"
            validation_rows = [json.loads(line) for line in validation_examples.read_text().splitlines()]
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text("".join(canonical_json({
                "dataset": row["source"], "example_id": row["example_id"],
                "question_hash": row["question_hash"], "gold_answer": row["answer"],
                "checkpoint_id": checkpoint_hash, "checkpoint_path": str(checkpoint_path),
                "raw_greedy_completion": "Final answer: definitely_not_the_gold_answer",
                "scorer_inputs": {"source": row["source"], "gold_answer": row["answer"]},
                "scorer_outputs": asdict(score_generation(_selection_generation(
                    __import__("phase_marker.splits", fromlist=["DatasetExample"]).DatasetExample(**row),
                    "Final answer: definitely_not_the_gold_answer", arm, seed, str(checkpoint_path),
                ))),
                "gold_continuation": f"\n{config.final_delimiter} {row['answer']}",
                "gold_token_ids": [0],
                "gold_token_pieces": [f"\n{config.final_delimiter} {row['answer']}"],
                "gold_token_logprobs": [-1.0],
                "gold_answer_logprob_contribution": -1.0,
                "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
                "tokenizer_snapshot_hash": QWEN25_7B_TOKENIZER_REVISION,
            }) + "\n" for index, row in enumerate(validation_rows)), encoding="utf-8")
            payload = {
                "schema_version": 1, "kind": "phase_marker_checkpoint_selection",
                "evidence_scope": "experiment", "backend": "vllm",
                "model_id": config.model_id, "model_revision": QWEN25_7B_TOKENIZER_REVISION,
                "config_hash": sha256_json(asdict(config)), "run_kind": "confirmatory",
                "arm": arm, "seed": seed, "selected_on": "validation",
                "criterion": {
                    "primary": "maximize_strict_validation_exact_answer_accuracy",
                    "tie_break_1": "higher_mean_gold_answer_logprob",
                    "tie_break_2": "earliest_checkpoint_step",
                },
                "split_artifact_id": split_id,
                "split_manifest_hash": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
                "validation_examples_file": str(validation_examples),
                "validation_examples_hash": hashlib.sha256(validation_examples.read_bytes()).hexdigest(),
                "training_manifest_file": str(training_path), "training_manifest_hash": training_hash,
                "materialization_artifact_id": materialization_id, "candidates": [candidate],
                "evidence_file": str(evidence_path),
                "evidence_hash": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                "selected_path": str(checkpoint_path), "selected_checkpoint_hash": checkpoint_hash,
                "selected_step": 100,
                "parent_hashes": [split_id, materialization_id, training_hash], "completed": True,
            }
            payload["artifact_id"] = sha256_json(payload)
            _write_json(
                selection_path,
                payload,
            )


def _write_behavior_and_audit(root: Path, config: ExperimentConfig) -> tuple[Path, Path]:
    split_id = _write_split(root, config)
    generation_root = root / "raw-generations" / "confirmatory"
    generation_root.mkdir(parents=True)
    records = [
        {
            "generation_id": f"audit:{source}:{index}", "source": source,
            "question_hash": sha256_json(f"{source}:{index}"), "training_arm": "glyph",
            "seed": 101, "prompt_condition": "glyph", "gold_answer": "1",
            "extracted_answer": "1", "score": {
                "generation_id": f"audit:{source}:{index}", "source": source,
                "question_hash": sha256_json(f"{source}:{index}"), "training_arm": "glyph",
                "seed": 101, "prompt_condition": "glyph", "gold_answer": "1",
                "extracted_answer": "1", "normalized_gold": "1",
                "normalized_prediction": "1", "correct": True, "parse_error": None,
                "equivalence_reason": "numeric_equivalent",
            },
        }
        for source in ("gsm8k", "svamp", "math") for index in range(100)
    ]
    records_path = generation_root / "records.jsonl"
    records_path.write_text(
        "".join(canonical_json(row) + "\n" for row in records), encoding="utf-8"
    )
    split_manifest = root / "splits" / "manifest.json"
    examples_path = root / "splits" / "test.jsonl"
    behavior = {
        "schema_version": 1, "kind": "phase_marker_behavior_generations",
        "evidence_scope": "experiment_candidate", "backend": "vllm",
        "config_hash": sha256_json(asdict(config)), "run_kind": "confirmatory",
        "seeds": [101, 202, 303], "split_artifact_id": split_id,
        "split_manifest_hash": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "materialization_artifact_ids": {}, "checkpoint_artifact_ids": {},
        "checkpoint_manifest_hashes": {}, "checkpoint_manifests": {},
        "examples_file": str(examples_path),
        "examples_hash": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "records_file": records_path.name,
        "records_hash": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "row_count": len(records), "record_hashes": [sha256_json(row) for row in records],
        "exclusions": [], "parent_hashes": [split_id], "completed": True,
    }
    behavior["artifact_id"] = sha256_json(behavior)
    behavior_path = generation_root / "manifest.json"
    _write_json(behavior_path, behavior)
    labels_path = root / "audit" / "manual-labels.tsv"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(
        "generation_id\tsource\tquestion_hash\ttraining_arm\tseed\tprompt_condition\tgold_answer\textracted_answer\tauto_correct\tmanual_correct\n"
        + "".join(
            f"audit:{source}:{index}\t{source}\t{sha256_json(f'{source}:{index}')}\tglyph\t101\tglyph\t1\t1\ttrue\ttrue\n"
            for source in ("gsm8k", "svamp", "math") for index in range(100)
        ),
        encoding="utf-8",
    )
    audit = {
        "schema_version": 1, "kind": "phase_marker_manual_audit",
        "evidence_scope": "experiment", "config_hash": sha256_json(asdict(config)),
        "run_kind": "confirmatory", "seeds": [101, 202, 303],
        "behavior_artifact_id": behavior["artifact_id"],
        "behavior_manifest_hash": hashlib.sha256(behavior_path.read_bytes()).hexdigest(),
        "labels_file": str(labels_path),
        "labels_hash": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "row_count": 300, "source_counts": {"gsm8k": 100, "svamp": 100, "math": 100},
        "disagreements": 0, "total": 300, "rate": 0.0, "passed": True,
        "parent_hashes": [behavior["artifact_id"]], "completed": True,
    }
    audit["artifact_id"] = sha256_json(audit)
    audit_path = root / "audit" / "confirmatory" / "manifest.json"
    _write_json(audit_path, audit)
    return behavior_path, audit_path


def _stub_behavior_gate(
    monkeypatch: pytest.MonkeyPatch, behavior_path: Path
) -> None:
    artifact = json.loads(behavior_path.read_text(encoding="utf-8"))["artifact_id"]
    monkeypatch.setattr(
        pipeline_module, "_validate_behavior_manifest", lambda *_, **__: artifact
    )


def test_gate_result_is_frozen() -> None:
    result = GateResult("train", True, "ready", ("a" * 64,), ("command",))
    with pytest.raises(FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


def test_gitignore_keeps_checkpoint_manifests_but_ignores_model_payloads() -> None:
    manifest = "artifacts/phase-marker/checkpoints/confirmatory/seed-101/glyph/run-manifest.json"
    tensor = "artifacts/phase-marker/checkpoints/confirmatory/seed-101/glyph/adapter_model.safetensors"
    manifest_check = subprocess.run(
        ["git", "check-ignore", "--no-index", manifest], capture_output=True, text=True
    )
    tensor_check = subprocess.run(
        ["git", "check-ignore", "--no-index", tensor], capture_output=True, text=True
    )
    assert manifest_check.returncode == 1
    assert tensor_check.returncode == 0


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


@pytest.mark.parametrize(
    "drifted",
    (
        lambda value: replace(value, pilot_seed=7),
        lambda value: replace(value, confirmatory_seeds=(11, 22, 33)),
        lambda value: replace(value, arms=tuple(reversed(value.arms))),
        lambda value: replace(value, arms=value.arms[:-1]),
    ),
)
def test_run_request_rejects_protocol_drift_even_when_config_is_self_consistent(
    config: ExperimentConfig, drifted
) -> None:
    changed = drifted(config)
    result = validate_run_request(
        "pilot", (changed.pilot_seed,), changed
    )
    assert not result.passed
    assert "frozen protocol" in result.reason


def test_command_manifest_is_planning_only_without_approval_metadata(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    jobs = build_command_manifest(
        config,
        tmp_path,
        kind="pilot",
        seeds=(42,),
    )
    assert jobs
    assert all(job["approval_ready"] is False for job in jobs)
    assert all(job["missing_approval_fields"] for job in jobs)


def test_command_manifest_binds_operator_supplied_approval_metadata(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    approval = ApprovalMetadata(
        hardware="1x NVIDIA H100 80GB",
        max_duration_hours=9.0,
        training_gpu_hours=4.5,
        selection_gpu_hours=1.0,
        behavior_gpu_hours=2.0,
        spend_cap_usd=25.0,
        estimated_spend_usd=18.0,
        workload_schema_version=1, training_jobs=6, checkpoint_selection_jobs=6,
        behavior_evaluation_jobs=1, manual_audit_rows=300, statistics_jobs=1,
        mechanism_jobs_excluded=True,
    )
    jobs = build_command_manifest(
        config,
        tmp_path,
        kind="pilot",
        seeds=(42,),
        approval=approval,
    )
    assert all(job["approval_ready"] is True for job in jobs)
    assert all(job["approval"] == asdict(approval) for job in jobs)
    assert all("phase_marker.behavior select" in job["selection_command"] for job in jobs)
    assert all(
        any(path.endswith(f"/{job['arm']}/manifest.json") for path in job["expected_outputs"])
        and any(path.endswith(f"/{job['arm']}/evidence.jsonl") for path in job["expected_outputs"])
        for job in jobs
    )


def test_command_manifest_rejects_approval_workload_undercount(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    approval = ApprovalMetadata(
        hardware="1x H100", max_duration_hours=4, training_gpu_hours=1,
        selection_gpu_hours=1, behavior_gpu_hours=1, spend_cap_usd=10,
        estimated_spend_usd=8, workload_schema_version=1, training_jobs=5,
        checkpoint_selection_jobs=6, behavior_evaluation_jobs=1,
        manual_audit_rows=300, statistics_jobs=1, mechanism_jobs_excluded=True,
    )
    with pytest.raises(ValueError, match="counts"):
        build_command_manifest(config, tmp_path, kind="pilot", seeds=(42,), approval=approval)


@pytest.mark.parametrize("stage", pipeline_module.STAGES)
def test_each_successful_stage_constructs_only_its_requested_commands(
    tmp_path: Path, config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    artifact = "a" * 64
    monkeypatch.setattr(pipeline_module, "_validate_split_manifest", lambda *_: SimpleNamespace(artifact_id=artifact))
    monkeypatch.setattr(pipeline_module, "_validate_materializations", lambda *_: (artifact,) * 6)
    monkeypatch.setattr(pipeline_module, "_validate_training_runs", lambda *_, **__: (artifact,))
    monkeypatch.setattr(pipeline_module, "_validate_checkpoint_selections", lambda *_, **__: (artifact,))
    monkeypatch.setattr(pipeline_module, "_validate_behavior_manifest", lambda *_, **__: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_audit_manifest", lambda *_, **__: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_synthetic_manifest", lambda *_: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_capture_inputs", lambda *_: (artifact,))
    monkeypatch.setattr(pipeline_module, "_validate_activation_manifest", lambda *_: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_intervention_inputs", lambda *_: (artifact,))
    monkeypatch.setattr(
        pipeline_module,
        "_validate_synthetic_preregistration",
        lambda *_, **__: artifact,
    )
    _write_json(
        tmp_path / "synthetic-preregistration.json",
        {"seed": 42, "counts": {"train": 8, "validation": 4, "test": 4}},
    )
    approval = ApprovalMetadata(
        hardware="1x H100", max_duration_hours=4, training_gpu_hours=1,
        selection_gpu_hours=1, behavior_gpu_hours=1,
        spend_cap_usd=10, estimated_spend_usd=8, workload_schema_version=1,
        training_jobs=6, checkpoint_selection_jobs=6, behavior_evaluation_jobs=1,
        manual_audit_rows=300, statistics_jobs=1, mechanism_jobs_excluded=True,
    )
    mechanism_approval = MechanismApprovalMetadata(
        schema_version=1, hardware="1x H100", capture_jobs=1, intervention_jobs=1,
        capture_gpu_hours=1, intervention_gpu_hours=1, max_duration_hours=3,
        estimated_spend_usd=8, spend_cap_usd=10,
        selection_parent_hash=artifact, activation_parent_hash=artifact,
        command_count=2, expected_output_count=4,
    )

    result = pipeline_module._run_gate(
        stage, config, tmp_path, kind="pilot", seeds=(42,), config_path=CONFIG_PATH,
        approval=approval, mechanism_approval=mechanism_approval,
    )

    assert result.passed is True
    assert result.next_commands


def test_general_approval_cannot_authorize_mechanism_command(
    tmp_path: Path, config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = "a" * 64
    monkeypatch.setattr(pipeline_module, "_validate_behavior_manifest", lambda *_, **__: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_synthetic_manifest", lambda *_: artifact)
    monkeypatch.setattr(pipeline_module, "_validate_capture_inputs", lambda *_: (artifact,))
    approval = ApprovalMetadata(
        hardware="1x H100", max_duration_hours=4, training_gpu_hours=1,
        selection_gpu_hours=1, behavior_gpu_hours=1, spend_cap_usd=10,
        estimated_spend_usd=8, workload_schema_version=1, training_jobs=6,
        checkpoint_selection_jobs=6, behavior_evaluation_jobs=1,
        manual_audit_rows=300, statistics_jobs=1, mechanism_jobs_excluded=True,
    )

    result = pipeline_module._run_gate(
        "capture", config, tmp_path, kind="pilot", seeds=(42,),
        config_path=CONFIG_PATH, approval=approval,
    )

    assert result.passed is False
    assert "separate mechanism approval" in result.reason
    assert result.next_commands == ()


def test_synthetic_command_rejects_preregistered_seed_mismatch(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    preregistration = {
        "schema_version": 1, "kind": "phase_marker_synthetic_preregistration",
        "config_hash": sha256_json(asdict(config)), "seed": 101,
        "counts": {"train": 8, "validation": 4, "test": 4},
        "family_balance": [
            "modular_chain", "affine_chain", "two_source_numeric_composition",
            "string_transformation_composition",
        ],
        "conditions": ["glyph", "dot", "repeated_glyph", "permuted_glyph", "random_symbol", "no_slot"],
        "workspace_lengths": [12, 16, 64],
    }
    preregistration["protocol_hash"] = sha256_json(preregistration)
    _write_json(
        tmp_path / "synthetic-preregistration.json",
        preregistration,
    )
    result = pipeline_module._run_gate(
        "synthetic", config, tmp_path, kind="pilot", seeds=(42,),
        config_path=CONFIG_PATH, approval=None,
    )
    assert result.passed is False
    assert "seed" in result.reason


def test_emitted_audit_and_analysis_commands_use_tsv_and_bound_manifest(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    audit = shlex.split(
        pipeline_module._commands_for_stage(
            "audit", config, tmp_path, kind="confirmatory", seeds=(101, 202, 303),
            config_path=CONFIG_PATH, approval=None,
        )[0]
    )
    analysis = shlex.split(
        pipeline_module._commands_for_stage(
            "statistics", config, tmp_path, kind="confirmatory", seeds=(101, 202, 303),
            config_path=CONFIG_PATH, approval=None,
        )[0]
    )
    assert audit[audit.index("--manual-labels") + 1].endswith("audit/manual-labels.tsv")
    assert audit[audit.index("--output-root") + 1].endswith("audit/confirmatory")
    assert analysis[analysis.index("--generations") + 1].endswith("raw-generations")
    assert analysis[analysis.index("--manual-audit") + 1].endswith("audit/manual-labels.tsv")
    assert analysis[analysis.index("--audit-manifest") + 1].endswith(
        "audit/confirmatory/manifest.json"
    )


def test_train_gate_validates_real_split_and_materialization_manifests(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)

    result = run_gate("train", config, tmp_path)

    assert not result.passed
    assert "approval metadata" in result.reason
    assert result.next_commands == ()


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


def test_train_gate_recomputes_split_artifact_id_from_protocol_rows(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    manifest_path = tmp_path / "splits" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_id"] = "0" * 64
    _write_json(manifest_path, payload)

    result = run_gate("train", config, tmp_path)

    assert not result.passed
    assert "artifact" in result.reason and "recomputed" in result.reason


def test_train_gate_recomputes_question_hashes_and_split_disjointness(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    train_path = tmp_path / "splits" / "train.jsonl"
    row = json.loads(train_path.read_text(encoding="utf-8"))
    row["question_hash"] = "f" * 64
    train_path.write_text(canonical_json(row) + "\n", encoding="utf-8")

    result = run_gate("train", config, tmp_path)

    assert not result.passed
    assert "question hash" in result.reason


def test_train_gate_recomputes_materialization_artifact_id(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    manifest_path = tmp_path / "training-data" / "glyph.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_id"] = "0" * 64
    _write_json(manifest_path, payload)

    result = run_gate("train", config, tmp_path)

    assert not result.passed
    assert "materialization artifact" in result.reason


def test_behavior_gate_recomputes_training_output_tree_hash(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    _write_training_runs(tmp_path, config)
    adapter = (
        tmp_path
        / "checkpoints"
        / "confirmatory"
        / "seed-101"
        / "glyph"
        / "adapter_model.safetensors"
    )
    adapter.write_bytes(b"forged adapter bytes")

    result = run_gate("behavior", config, tmp_path)

    assert not result.passed
    assert "output tree hash" in result.reason


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


def test_behavior_gate_requires_complete_validation_selection_matrix(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    _write_training_runs(tmp_path, config)
    _write_checkpoint_selections(tmp_path, config)
    missing = tmp_path / "checkpoint-selections" / "confirmatory" / "seed-303" / "filler" / "manifest.json"
    missing.unlink()

    result = run_gate("behavior", config, tmp_path)

    assert result.passed is False
    assert "selection" in result.reason


def test_behavior_gate_rejects_omitted_declared_checkpoint_candidate(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    _write_split(tmp_path, config)
    _write_materializations(tmp_path, config)
    _write_training_runs(tmp_path, config)
    _write_checkpoint_selections(tmp_path, config)
    path = tmp_path / "checkpoint-selections/confirmatory/seed-101/glyph/manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidates"] = []
    payload["artifact_id"] = sha256_json(
        {key: value for key, value in payload.items() if key != "artifact_id"}
    )
    _write_json(path, payload)

    result = run_gate("behavior", config, tmp_path)

    assert result.passed is False
    assert "candidate" in result.reason


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
    assert all(job["approval_ready"] is False for job in jobs)
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
    tmp_path: Path, config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_path, audit_path = _write_behavior_and_audit(tmp_path, config)
    _stub_behavior_gate(monkeypatch, behavior_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["config_hash"] = "0" * 64
    audit.pop("artifact_id")
    audit["artifact_id"] = sha256_json(audit)
    _write_json(audit_path, audit)

    result = run_gate("statistics", config, tmp_path)

    assert not result.passed
    assert "config hash mismatch" in result.reason


@pytest.mark.parametrize("tamper", ("plumbing", "one-row", "forged-artifact"))
def test_statistics_gate_rejects_nonproduction_or_forged_audit(
    tmp_path: Path, config: ExperimentConfig, tamper: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    behavior_path, audit_path = _write_behavior_and_audit(tmp_path, config)
    _stub_behavior_gate(monkeypatch, behavior_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if tamper == "plumbing":
        payload["evidence_scope"] = "plumbing_only"
        payload["artifact_id"] = sha256_json({key: value for key, value in payload.items() if key != "artifact_id"})
    elif tamper == "one-row":
        payload["row_count"] = payload["total"] = 1
        payload["source_counts"] = {"gsm8k": 1}
        payload["artifact_id"] = sha256_json({key: value for key, value in payload.items() if key != "artifact_id"})
    else:
        payload["artifact_id"] = "f" * 64
    _write_json(audit_path, payload)

    result = run_gate("statistics", config, tmp_path)

    assert result.passed is False
    assert any(word in result.reason for word in ("production", "300", "artifact"))


def test_statistics_gate_rejects_label_not_bound_to_behavior_generation(
    tmp_path: Path, config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_path, audit_path = _write_behavior_and_audit(tmp_path, config)
    _stub_behavior_gate(monkeypatch, behavior_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    labels_path = Path(payload["labels_file"])
    labels_path.write_text(
        labels_path.read_text(encoding="utf-8").replace(
            "audit:gsm8k:0\tgsm8k", "unbound:gsm8k:0\tgsm8k", 1
        ),
        encoding="utf-8",
    )
    payload["labels_hash"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    payload["artifact_id"] = sha256_json(
        {key: value for key, value in payload.items() if key != "artifact_id"}
    )
    _write_json(audit_path, payload)

    result = run_gate("statistics", config, tmp_path)

    assert result.passed is False
    assert "behavior generation" in result.reason


def test_statistics_gate_rejects_flipped_label_with_forged_claimed_metrics(
    tmp_path: Path, config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    behavior_path, audit_path = _write_behavior_and_audit(tmp_path, config)
    _stub_behavior_gate(monkeypatch, behavior_path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    labels_path = Path(payload["labels_file"])
    labels_path.write_text(
        labels_path.read_text(encoding="utf-8").replace("\ttrue\ttrue\n", "\ttrue\tfalse\n", 1),
        encoding="utf-8",
    )
    payload["labels_hash"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    payload["artifact_id"] = sha256_json(
        {key: value for key, value in payload.items() if key != "artifact_id"}
    )
    _write_json(audit_path, payload)

    result = run_gate("statistics", config, tmp_path)

    assert result.passed is False
    assert "row counts" in result.reason
