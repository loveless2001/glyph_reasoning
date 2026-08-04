from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict
import json

import numpy as np
import pytest

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


def test_scored_envelope_loader_preserves_all_five_sampled_completion_seeds(tmp_path):
    path = tmp_path / "sampled.jsonl"
    rows = []
    for completion_index in range(5):
        score = _score(
            "gsm8k",
            "question-a",
            42 + completion_index,
            "glyph",
            "glyph",
            completion_index % 2 == 0,
            generation_id=f"completion-{completion_index}",
        )
        rows.append(
            {
                "generation_id": score.generation_id,
                "raw_completion": f"Final answer: {int(score.correct)}",
                "decoding": {"completion_index": completion_index, "n": 1},
                "score": asdict(score),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded = load_score_records(path)

    assert len(loaded) == 5
    assert [record.generation_id for record in loaded] == [
        f"completion-{index}" for index in range(5)
    ]
    assert [record.seed for record in loaded] == [42, 43, 44, 45, 46]


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
        "generation_id\tmanual_correct\n"
        + "".join(
            f"audit-{index}\t{'false' if index == 0 else 'true'}\n"
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
    results = build_contrast_results(
        _contrast_records(),
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
    audit = AuditResult(True, 3, 300, 0.01, 0.01)

    paths = write_confirmatory_outputs(tmp_path, results, model, audit, synthetic=True)

    markdown = paths["markdown"].read_text(encoding="utf-8")
    latex = paths["latex"].read_text(encoding="utf-8")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "Evaluation-sample 95% paired bootstrap CI" in markdown
    assert "Three-seed variation" in markdown
    assert "Evaluation-sample 95\\% paired bootstrap CI" in latex
    assert "Three-seed variation" in latex
    assert summary["synthetic_smoke"] is True
    assert summary["experiment_outcomes"] is False
    assert len(summary["contrasts"][0]["per_seed_deltas"]) == 3
    assert paths["model_diagnostics"].exists()
    assert paths["audit_status"].exists()


def test_failed_audit_writes_status_but_blocks_all_confirmatory_tables(tmp_path):
    model = ModelSummary("formula", {}, True, {"optimizer_success": True})
    failed = AuditResult(False, 4, 300, 4 / 300, 0.01)

    with pytest.raises(AuditGateError, match="blocks confirmatory tables"):
        write_confirmatory_outputs(tmp_path, (), model, failed)

    assert (tmp_path / "audit-status.json").exists()
    assert not (tmp_path / "contrast-table.md").exists()
    assert not (tmp_path / "contrast-table.tex").exists()
    assert not (tmp_path / "summary.json").exists()


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
