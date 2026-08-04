from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import phase_marker.statistics as statistics_module

from phase_marker.schema import ScoreRecord
from phase_marker.statistics import (
    AuditGateError,
    AuditResult,
    CoefficientSummary,
    ContrastSpec,
    Interval,
    ModelSummary,
    UnpairedComparisonError,
    apply_audit_gate,
    build_contrast_results,
    effect_is_inconclusive,
    fit_hierarchical_logit,
    generate_manual_audit_template,
    load_score_records,
    paired_bootstrap_delta,
    read_manual_audit_tsv,
    write_confirmatory_outputs,
)
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


def test_behavior_root_resolves_exactly_one_kind_manifest(tmp_path):
    manifest = tmp_path / "raw-generations" / "confirmatory" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"kind":"fixture"}\n', encoding="utf-8")

    path, payload = statistics_module._load_behavior_envelope(tmp_path / "raw-generations")

    assert path == manifest
    assert payload == {"kind": "fixture"}


def test_analyze_parser_accepts_optional_explicit_audit_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="behavior manifest"):
        statistics_module.main(
            (
                "analyze", "--config", "configs/phase-marker-qwen25-7b.toml",
                "--generations", str(tmp_path / "raw-generations"),
                "--manual-audit", str(tmp_path / "audit" / "manual-labels.tsv"),
                "--audit-manifest", str(tmp_path / "audit" / "confirmatory" / "manifest.json"),
                "--output-root", str(tmp_path / "analysis"),
            )
        )


def test_exact_task13_audit_and_analysis_paths_succeed_on_faithful_fixture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    config_path = Path("configs/phase-marker-qwen25-7b.toml")
    config = ExperimentConfig.load(config_path)
    config_hash = sha256_json(asdict(config))
    scores = [
        _score(
            source, f"{source}-q{question}", seed, arm, prompt,
            (question + seed + cell) % 3 != 0,
        )
        for source in ("gsm8k", "svamp", "math")
        for question in range(25)
        for seed in (101, 202, 303)
        for cell, (arm, prompt) in enumerate((
            ("semantic", "neutral"), ("glyph", "glyph"),
            ("glyph", "dot"), ("dot", "dot"),
        ))
    ]
    parent = "a" * 64
    rows = [_envelope_from_score(score, parent) for score in scores]
    generation_root = tmp_path / "raw-generations" / "confirmatory"
    generation_root.mkdir(parents=True)
    records_path = generation_root / "records.jsonl"
    records_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    examples = tmp_path / "splits" / "test.jsonl"
    examples.parent.mkdir()
    examples.write_text(canonical_json({"fixture": True}) + "\n", encoding="utf-8")
    split_manifest = examples.parent / "manifest.json"
    split_manifest.write_text(canonical_json({"artifact_id": parent}) + "\n", encoding="utf-8")
    behavior = {
        "schema_version": 1, "kind": "phase_marker_behavior_generations",
        "evidence_scope": "experiment_candidate", "backend": "vllm",
        "config_hash": config_hash, "run_kind": "confirmatory", "seeds": [101, 202, 303],
        "split_artifact_id": parent,
        "split_manifest_hash": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "materialization_artifact_ids": {}, "checkpoint_artifact_ids": {},
        "checkpoint_manifest_hashes": {}, "checkpoint_manifests": {},
        "examples_file": str(examples), "examples_hash": hashlib.sha256(examples.read_bytes()).hexdigest(),
        "records_file": records_path.name, "records_hash": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "row_count": len(rows), "record_hashes": [sha256_json(row) for row in rows],
        "exclusions": [], "parent_hashes": [parent], "completed": True,
    }
    behavior["artifact_id"] = sha256_json(behavior)
    (generation_root / "manifest.json").write_text(canonical_json(behavior) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        __import__("phase_marker.pipeline", fromlist=["_validate_behavior_manifest"]),
        "_validate_behavior_manifest", lambda *_, **__: behavior["artifact_id"],
    )
    labels = tmp_path / "audit" / "manual-labels.tsv"
    selected = generate_manual_audit_template(scores, labels, seed=20260804)
    by_id = {score.generation_id: score.correct for score in selected}
    with labels.open(encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle, delimiter="\t"))
        fieldnames = handle.seek(0) or list(label_rows[0])
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in label_rows:
            row["manual_correct"] = str(by_id[row["generation_id"]]).lower()
            writer.writerow(row)

    assert statistics_module.main((
        "audit", "--config", str(config_path), "--kind", "confirmatory", "--seeds", "101", "202", "303",
        "--generations", str(tmp_path / "raw-generations"), "--manual-labels", str(labels),
        "--output-root", str(tmp_path / "audit" / "confirmatory"),
    )) == 0
    audit_manifest = json.loads(
        (tmp_path / "audit" / "confirmatory" / "manifest.json").read_text(encoding="utf-8")
    )
    assert audit_manifest["evidence_scope"] == "experiment"
    assert statistics_module.main((
        "analyze", "--config", str(config_path),
        "--generations", str(tmp_path / "raw-generations"),
        "--manual-audit", str(labels), "--output-root", str(tmp_path / "analysis"),
    )) == 0
    assert (tmp_path / "analysis" / "manifest.json").is_file()


def test_paired_bootstrap_aligns_full_keys_and_uses_local_seeded_rng():
    left = [
        _score("gsm8k", "a", 101, "glyph", "glyph", True),
        _score("gsm8k", "b", 101, "glyph", "glyph", True),
        _score("gsm8k", "c", 101, "glyph", "glyph", False),
    ]
    right = [
        _score("gsm8k", "c", 101, "semantic", "neutral", False),
        _score("gsm8k", "a", 101, "semantic", "neutral", False),
        _score("gsm8k", "b", 101, "semantic", "neutral", True),
    ]

    np.random.seed(19)
    expected_next_global_draw = np.random.random()
    np.random.seed(19)
    first = paired_bootstrap_delta(left, right, seed=7, draws=1_000)
    observed_next_global_draw = np.random.random()
    second = paired_bootstrap_delta(left, right, seed=7, draws=1_000)

    assert first == second
    assert first.point == pytest.approx(1 / 3)
    assert first.draws == 1_000
    assert first.seed == 7
    assert observed_next_global_draw == expected_next_global_draw


def test_paired_bootstrap_rejects_unaligned_or_duplicate_analysis_rows():
    left = [_score("gsm8k", "a", 101, "glyph", "glyph", True)]
    right = [_score("gsm8k", "b", 101, "semantic", "neutral", True)]
    with pytest.raises(UnpairedComparisonError, match="paired keys"):
        paired_bootstrap_delta(left, right, seed=7)

    duplicate = [left[0], _score("gsm8k", "a", 101, "glyph", "glyph", False)]
    with pytest.raises(UnpairedComparisonError, match="duplicate"):
        paired_bootstrap_delta(duplicate, left, seed=7)


def test_scored_envelope_loader_keeps_adapter_and_decoding_seeds_separate(tmp_path):
    path = tmp_path / "sampled.jsonl"
    rows = [
        _scored_envelope(
            completion_index=index,
            adapter_seed=101,
            decoding_seed=101 + index,
        )
        for index in range(5)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = load_score_records(path)

    assert len(loaded) == 5
    assert [record.generation_id for record in loaded] == [
        f"completion-{index}" for index in range(5)
    ]
    assert [record.seed for record in loaded] == [101] * 5
    assert [record.adapter_seed for record in loaded] == [101] * 5
    assert [record.decoding_seed for record in loaded] == [101, 102, 103, 104, 105]
    assert [record.completion_index for record in loaded] == [0, 1, 2, 3, 4]
    assert all(record.evaluation_kind == "sampled" for record in loaded)


def test_scored_envelope_loader_rejects_adapter_seed_lineage_mismatch(tmp_path):
    path = tmp_path / "mismatched.jsonl"
    row = _scored_envelope(completion_index=0, adapter_seed=101, decoding_seed=101)
    row["provenance"]["adapter_seed"] = 202
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter seed"):
        load_score_records(path)


def test_mixed_model_rejects_sampled_completions_as_pseudoreplicated_seed_rows(tmp_path):
    path = tmp_path / "sampled.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                _scored_envelope(
                    completion_index=index,
                    adapter_seed=101,
                    decoding_seed=101 + index,
                )
            )
            + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    sampled = load_score_records(path)

    with pytest.raises(ValueError, match="duplicate analysis cell"):
        fit_hierarchical_logit([*_model_records(), *sampled])


def test_primary_paired_analysis_rejects_even_one_sampled_observation(tmp_path):
    path = tmp_path / "partial-sampled.jsonl"
    path.write_text(
        json.dumps(
            _scored_envelope(
                completion_index=0,
                adapter_seed=101,
                decoding_seed=101,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    sampled = load_score_records(path)[0]

    with pytest.raises(UnpairedComparisonError, match="primary greedy"):
        paired_bootstrap_delta([sampled], [sampled], seed=7)


def test_manual_audit_template_is_exactly_300_stable_source_stratified_rows(tmp_path):
    records = [
        _score(
            source,
            f"question-{index}",
            (101, 202, 303)[index % 3],
            ("semantic", "glyph", "dot", "random")[index % 4],
            ("neutral", "glyph", "dot", "headings")[index % 4],
            index % 3 != 0,
            generation_id=f"{source}-{index}",
        )
        for source in ("gsm8k", "svamp", "math")
        for index in range(110)
    ]
    first_path = tmp_path / "audit-first.tsv"
    second_path = tmp_path / "audit-second.tsv"

    generate_manual_audit_template(records, first_path, seed=20260804)
    generate_manual_audit_template(list(reversed(records)), second_path, seed=20260804)

    with first_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 300
    assert Counter(row["source"] for row in rows) == {
        "gsm8k": 100,
        "svamp": 100,
        "math": 100,
    }
    assert all(row["manual_correct"] == "" for row in rows)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_manual_audit_ingestion_and_gate_block_only_above_one_percent(tmp_path):
    auto = [
        _score(
            "gsm8k",
            f"question-{index}",
            101,
            "glyph",
            "glyph",
            True,
            generation_id=f"audit-{index}",
        )
        for index in range(100)
    ]
    completed = tmp_path / "completed.tsv"
    completed.write_text(
        "generation_id\tsource\tquestion_hash\ttraining_arm\tseed\tprompt_condition\tgold_answer\textracted_answer\tauto_correct\tmanual_correct\n"
        + "".join(
            f"audit-{index}\tgsm8k\tquestion-{index}\tglyph\t101\tglyph\t1\t1\ttrue\t{'false' if index == 0 else 'true'}\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    manual = read_manual_audit_tsv(completed)

    exactly_one_percent = apply_audit_gate(auto, manual, threshold=0.01)
    assert exactly_one_percent == AuditResult(True, 1, 100, 0.01, 0.01)

    manual["audit-1"] = False
    above_one_percent = apply_audit_gate(auto, manual, threshold=0.01)
    assert not above_one_percent.passed
    assert above_one_percent.disagreements == 2
    assert above_one_percent.rate == 0.02


def test_audit_gate_rejects_missing_or_extra_manual_labels():
    auto = [_score("gsm8k", "a", 101, "glyph", "glyph", True, generation_id="a")]
    with pytest.raises(UnpairedComparisonError, match="audit generation IDs"):
        apply_audit_gate(auto, {"other": True})


def test_hierarchical_logit_fits_required_fixed_and_random_effects():
    records = _model_records()

    summary = fit_hierarchical_logit(records)

    assert summary.formula == (
        "correct ~ C(training_arm) * C(prompt_condition) + C(source)"
    )
    assert "Intercept" in summary.coefficients
    assert any(":" in name for name in summary.coefficients)
    assert all(np.isfinite(value.estimate) for value in summary.coefficients.values())
    assert all(value.posterior_sd > 0 for value in summary.coefficients.values())
    assert summary.diagnostics["algorithm"] == "variational_bayes"
    assert summary.diagnostics["random_intercepts"] == ["question_hash", "seed"]
    assert summary.diagnostics["optimizer_success"] is summary.converged


def test_contrasts_preserve_seed_rows_scope_holm_and_apply_inconclusive_rule():
    records = _contrast_records()
    contrasts = (
        ContrastSpec("glyph-v-semantic", "glyph", "glyph", "semantic", "neutral"),
        ContrastSpec("glyph-v-dot", "glyph", "glyph", "dot", "dot", secondary=True),
        ContrastSpec("glyph-prompt-v-dot-prompt", "glyph", "glyph", "glyph", "dot", secondary=True),
    )

    results = build_contrast_results(records, contrasts, bootstrap_seed=11, draws=1_000)

    assert [seed for seed, _ in results[0].per_seed_deltas] == [101, 202, 303]
    assert results[0].holm_adjusted_p is None
    assert all(result.holm_adjusted_p is not None for result in results[1:])
    assert results[0].interval.draws == 1_000
    assert effect_is_inconclusive(Interval(0.019, 0.01, 0.03, 10_000, 7))
    assert effect_is_inconclusive(Interval(0.2, -0.01, 0.3, 10_000, 7))
    assert not effect_is_inconclusive(Interval(0.2, 0.1, 0.3, 10_000, 7))


def test_confirmatory_outputs_label_two_uncertainties_and_are_machine_readable(tmp_path):
    synthetic_records = [
        record for record in _contrast_records() if record.seed in (101, 202)
    ]
    results = build_contrast_results(
        synthetic_records,
        (ContrastSpec("glyph-v-semantic", "glyph", "glyph", "semantic", "neutral"),),
        bootstrap_seed=13,
        draws=500,
    )
    model = ModelSummary(
        formula="correct ~ arm * prompt + source",
        coefficients={"Intercept": CoefficientSummary(0.1, 0.2, -0.292, 0.492)},
        converged=True,
        diagnostics={"optimizer_success": True},
    )
    auto = _audit_records()
    manual = _manual_labels(auto, disagreements=3)

    paths = write_confirmatory_outputs(
        tmp_path, results, model, auto, manual, synthetic=True
    )

    markdown = paths["markdown"].read_text(encoding="utf-8")
    latex = paths["latex"].read_text(encoding="utf-8")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "Evaluation-sample 95% paired bootstrap CI" in markdown
    assert "Three-seed variation" in markdown
    assert "Synthetic/test-only analysis" in markdown
    assert "Evaluation-sample 95\\% paired bootstrap CI" in latex
    assert "Three-seed variation" in latex
    assert "Synthetic/test-only analysis" in latex
    assert summary["synthetic_smoke"] is True
    assert summary["experiment_outcomes"] is False
    assert summary["analysis_mode"] == "synthetic_test_only"
    assert summary["audit"]["disagreements"] == 3
    assert len(summary["contrasts"][0]["per_seed_deltas"]) == 2
    assert summary["contrasts"][0]["interval"]["draws"] == 500
    assert paths["model_diagnostics"].exists()
    assert paths["audit_status"].exists()


def test_failed_audit_writes_status_but_blocks_all_confirmatory_tables(tmp_path):
    model = ModelSummary("formula", {}, True, {"optimizer_success": True})
    auto = _audit_records()
    manual = _manual_labels(auto, disagreements=4)

    with pytest.raises(AuditGateError, match="blocks confirmatory tables"):
        write_confirmatory_outputs(tmp_path, (), model, auto, manual, synthetic=True)

    assert (tmp_path / "audit-status.json").exists()
    assert not (tmp_path / "contrast-table.md").exists()
    assert not (tmp_path / "contrast-table.tex").exists()
    assert not (tmp_path / "summary.json").exists()


def test_confirmatory_outputs_reject_passing_but_undersized_audit(tmp_path):
    model = ModelSummary("formula", {}, True, {"optimizer_success": True})
    auto = [
        _score(
            "gsm8k",
            "only-question",
            101,
            "glyph",
            "glyph",
            True,
            generation_id="only-audit-row",
        )
    ]

    with pytest.raises(AuditGateError, match="exactly 300"):
        write_confirmatory_outputs(
            tmp_path,
            (),
            model,
            auto,
            {"only-audit-row": True},
            synthetic=True,
        )

    assert not (tmp_path / "contrast-table.md").exists()


def test_confirmatory_outputs_reject_unstratified_300_row_audit(tmp_path):
    model = ModelSummary("formula", {}, True, {"optimizer_success": True})
    auto = _audit_records({"gsm8k": 150, "math": 150})

    with pytest.raises(AuditGateError, match="100 rows for each"):
        write_confirmatory_outputs(
            tmp_path,
            (),
            model,
            auto,
            _manual_labels(auto),
            synthetic=True,
        )

    assert not (tmp_path / "contrast-table.md").exists()


def test_non_synthetic_outputs_require_exactly_10000_bootstrap_draws(tmp_path):
    results = build_contrast_results(
        _contrast_records(),
        (ContrastSpec("glyph-v-semantic", "glyph", "glyph", "semantic", "neutral"),),
        bootstrap_seed=13,
        draws=9_999,
    )
    auto = _audit_records()

    with pytest.raises(ValueError, match="exactly 10,000"):
        write_confirmatory_outputs(
            tmp_path,
            results,
            ModelSummary("formula", {}, True, {"optimizer_success": True}),
            auto,
            _manual_labels(auto),
        )

    assert not (tmp_path / "contrast-table.md").exists()


def test_non_synthetic_outputs_require_three_confirmatory_adapter_seeds(tmp_path):
    two_seed_records = [
        record for record in _contrast_records() if record.seed in (101, 202)
    ]
    results = build_contrast_results(
        two_seed_records,
        (ContrastSpec("glyph-v-semantic", "glyph", "glyph", "semantic", "neutral"),),
        bootstrap_seed=13,
        draws=10_000,
    )
    auto = _audit_records()

    with pytest.raises(ValueError, match="adapter seeds.*101, 202, 303"):
        write_confirmatory_outputs(
            tmp_path,
            results,
            ModelSummary("formula", {}, True, {"optimizer_success": True}),
            auto,
            _manual_labels(auto),
        )

    assert not (tmp_path / "contrast-table.md").exists()


def test_non_synthetic_outputs_reject_empty_contrasts_and_remove_stale_artifacts(
    tmp_path,
):
    stale_names = (
        "contrast-table.md",
        "contrast-table.tex",
        "summary.json",
        "model-diagnostics.json",
    )
    for name in stale_names:
        (tmp_path / name).write_text("stale confirmatory output\n", encoding="utf-8")
    auto = _audit_records()

    with pytest.raises(ValueError, match="nonempty declared contrast set"):
        write_confirmatory_outputs(
            tmp_path,
            (),
            ModelSummary("formula", {}, True, {"optimizer_success": True}),
            auto,
            _manual_labels(auto),
        )

    assert (tmp_path / "audit-status.json").exists()
    assert all(not (tmp_path / name).exists() for name in stale_names)


def _score(
    source: str,
    question_hash: str,
    seed: int,
    training_arm: str,
    prompt_condition: str,
    correct: bool,
    *,
    generation_id: str | None = None,
) -> ScoreRecord:
    identifier = generation_id or (
        f"{source}:{question_hash}:{seed}:{training_arm}:{prompt_condition}"
    )
    prediction = "1" if correct else "0"
    return ScoreRecord(
        generation_id=identifier,
        source=source,
        question_hash=question_hash,
        training_arm=training_arm,
        seed=seed,
        prompt_condition=prompt_condition,
        gold_answer="1",
        extracted_answer=prediction,
        normalized_gold="1",
        normalized_prediction=prediction,
        correct=correct,
        parse_error=None,
        equivalence_reason="numeric_equivalent" if correct else "numeric_mismatch",
    )


def _envelope_from_score(score: ScoreRecord, parent_hash: str) -> dict[str, object]:
    decoding = {
        "seed": score.seed, "adapter_seed": score.seed, "checkpoint": "/checkpoints/adapter",
        "run_kind": "production", "config_hash": "c" * 64,
        "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
        "split_artifact_id": parent_hash, "split_parent_hashes": [],
        "max_tokens": 64, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0,
        "n": 1, "completion_index": None, "completion_token_logprobs": [-0.1],
        "evaluation_kind": "primary", "perturbation": None,
    }
    return {
        "generation_id": score.generation_id, "source": score.source,
        "question_hash": score.question_hash, "gold_answer": score.gold_answer,
        "training_arm": score.training_arm, "seed": score.seed,
        "checkpoint": "/checkpoints/adapter", "prompt_condition": score.prompt_condition,
        "prompt_hash": "d" * 64, "raw_prompt": "solve",
        "raw_completion": f"Final answer: {score.extracted_answer}",
        "prompt_token_ids": [1], "completion_token_ids": [2], "decoding": decoding,
        "parent_hashes": [parent_hash], "prompt_token_count": 1, "completion_token_count": 1,
        "provenance": {
            "run_kind": "production", "adapter_seed": score.seed, "config_hash": "c" * 64,
            "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
            "split_artifact_id": parent_hash, "split_parent_hashes": [],
            "checkpoint": "/checkpoints/adapter", "parent_hashes": [parent_hash],
        },
        "score": asdict(score),
    }
def _model_records() -> list[ScoreRecord]:
    records = []
    for source_index, source in enumerate(("gsm8k", "svamp", "math")):
        for question_index in range(4):
            for seed_index, seed in enumerate((101, 202, 303)):
                for arm_index, arm in enumerate(("semantic", "glyph")):
                    for prompt_index, prompt in enumerate(("neutral", "glyph")):
                        correct = (
                            question_index
                            + seed_index
                            + source_index
                            + arm_index
                            + prompt_index
                        ) % 5 < 2
                        records.append(
                            _score(
                                source,
                                f"{source}-q{question_index}",
                                seed,
                                arm,
                                prompt,
                                correct,
                            )
                        )
    return records


def _contrast_records() -> list[ScoreRecord]:
    records = []
    cells = (
        ("semantic", "neutral"),
        ("glyph", "glyph"),
        ("glyph", "dot"),
        ("dot", "dot"),
    )
    for source_index, source in enumerate(("gsm8k", "svamp")):
        for question_index in range(8):
            for seed_index, seed in enumerate((101, 202, 303)):
                baseline = (source_index + question_index + seed_index) % 4
                for arm, prompt in cells:
                    bonus = {
                        ("semantic", "neutral"): 0,
                        ("glyph", "glyph"): 1,
                        ("glyph", "dot"): 0,
                        ("dot", "dot"): -1,
                    }[(arm, prompt)]
                    records.append(
                        _score(
                            source,
                            f"{source}-q{question_index}",
                            seed,
                            arm,
                            prompt,
                            baseline + bonus >= 2,
                        )
                    )
    return records


def _scored_envelope(
    *, completion_index: int, adapter_seed: int, decoding_seed: int
) -> dict[str, object]:
    score = _score(
        "gsm8k",
        "question-a",
        adapter_seed,
        "glyph",
        "glyph",
        completion_index % 2 == 0,
        generation_id=f"completion-{completion_index}",
    )
    parent_hash = "a" * 64
    split_parent_hash = "b" * 64
    decoding = {
        "seed": decoding_seed,
        "adapter_seed": adapter_seed,
        "checkpoint": "/checkpoints/glyph",
        "run_kind": "production",
        "config_hash": "c" * 64,
        "tokenizer_revision": "Qwen/Qwen2.5-7B-Instruct@revision",
        "split_artifact_id": parent_hash,
        "split_parent_hashes": [split_parent_hash],
        "max_tokens": 64,
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
        "completion_index": completion_index,
        "completion_token_logprobs": [-0.1],
        "evaluation_kind": "sampled",
        "perturbation": None,
    }
    return {
        "generation_id": score.generation_id,
        "source": score.source,
        "question_hash": score.question_hash,
        "gold_answer": score.gold_answer,
        "training_arm": score.training_arm,
        "seed": adapter_seed,
        "checkpoint": "/checkpoints/glyph",
        "prompt_condition": score.prompt_condition,
        "prompt_hash": "d" * 64,
        "raw_prompt": "Solve the problem",
        "raw_completion": f"Final answer: {score.extracted_answer}",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3],
        "decoding": decoding,
        "parent_hashes": [parent_hash, split_parent_hash],
        "prompt_token_count": 2,
        "completion_token_count": 1,
        "provenance": {
            "run_kind": "production",
            "adapter_seed": adapter_seed,
            "config_hash": "c" * 64,
            "tokenizer_revision": "Qwen/Qwen2.5-7B-Instruct@revision",
            "split_artifact_id": parent_hash,
            "split_parent_hashes": [split_parent_hash],
            "checkpoint": "/checkpoints/glyph",
            "parent_hashes": [parent_hash, split_parent_hash],
        },
        "score": asdict(score),
    }


def _audit_records(
    counts: dict[str, int] | None = None,
) -> list[ScoreRecord]:
    counts = counts or {"gsm8k": 100, "math": 100, "svamp": 100}
    return [
        _score(
            source,
            f"audit-question-{index}",
            (101, 202, 303)[index % 3],
            "glyph",
            "glyph",
            index % 2 == 0,
            generation_id=f"audit:{source}:{index}",
        )
        for source, count in counts.items()
        for index in range(count)
    ]


def _manual_labels(
    records: list[ScoreRecord], *, disagreements: int = 0
) -> dict[str, bool]:
    return {
        record.generation_id: (
            not record.correct if index < disagreements else record.correct
        )
        for index, record in enumerate(records)
    }
