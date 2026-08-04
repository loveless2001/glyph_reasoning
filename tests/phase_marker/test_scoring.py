from collections import Counter

import pytest

from phase_marker.schema import GenerationRecord, ScoreRecord
from phase_marker.scoring import (
    answers_equivalent,
    extract_final_answer,
    normalize_answer,
    score_generation,
    select_audit_sample,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 3 = 5\nFinal answer: 5", "5"),
        ("Final answer: -3/4", "-3/4"),
        ("Reasoning mentions 9.81 but has no delimiter", None),
        ("Final answer: 1,200", "1,200"),
    ],
)
def test_extract_final_answer_requires_delimiter(text, expected):
    assert extract_final_answer(text) == expected


def test_extract_final_answer_uses_the_last_complete_delimited_line():
    text = "Final answer: 3\nrevision\nFinal answer: 5\nmore reasoning 9"
    assert extract_final_answer(text) == "5"


def test_numeric_equivalence_handles_fraction_decimal_and_percent():
    assert answers_equivalent("gsm8k", "3/4", "0.75")
    assert answers_equivalent("svamp", "25%", "0.25")
    assert not answers_equivalent("gsm8k", "9.81", "5")


def test_math_normalization_handles_latex_fraction_box_and_finite_set():
    predicted = r"\boxed{\left\{2, \dfrac{ 1 }{ 2}\right\}}"
    gold = r"\left\{\tfrac{1}{2}, 2\right\}"
    assert normalize_answer("math", predicted) == "{1/2,2}"
    assert answers_equivalent("math", predicted, gold)


def test_score_generation_records_missing_delimiter_without_numeric_fallback():
    record = _generation("gsm8k", "5", "Reasoning: 2 + 3 = 5")
    score = score_generation(record)

    assert score.extracted_answer is None
    assert score.normalized_prediction is None
    assert not score.correct
    assert score.parse_error == "missing_final_delimiter"
    assert score.equivalence_reason == "missing_prediction"


def test_score_generation_records_normalized_values_and_numeric_reason():
    score = score_generation(_generation("gsm8k", "0.75", "Final answer: 3/4"))

    assert score.normalized_gold == "3/4"
    assert score.normalized_prediction == "3/4"
    assert score.correct
    assert score.parse_error is None
    assert score.equivalence_reason == "numeric_equivalent"


@pytest.fixture
def score_records():
    return [
        _score(source, f"{source}-{number}")
        for source in ("gsm8k", "svamp", "math")
        for number in range(4)
    ]


def test_audit_sample_is_stable_and_source_stratified(score_records):
    first = select_audit_sample(score_records, per_source=2, seed=20260804)
    second = select_audit_sample(
        list(reversed(score_records)), per_source=2, seed=20260804
    )
    assert [row.generation_id for row in first] == [row.generation_id for row in second]
    assert Counter(row.source for row in first) == {"gsm8k": 2, "svamp": 2, "math": 2}


def test_audit_sample_retains_all_available_rows_when_source_has_under_quota():
    records = [_score("gsm8k", "g-1"), _score("svamp", "s-1"), _score("svamp", "s-2")]
    selected = select_audit_sample(records, per_source=2, seed=20260804)

    assert [row.generation_id for row in selected if row.source == "gsm8k"] == ["g-1"]
    assert Counter(row.source for row in selected) == {"gsm8k": 1, "svamp": 2}


def test_audit_sample_rejects_duplicate_generation_ids():
    records = [_score("gsm8k", "duplicate"), _score("svamp", "duplicate")]

    with pytest.raises(ValueError, match="generation_id values must be unique"):
        select_audit_sample(records, per_source=1, seed=20260804)


def _generation(source: str, gold_answer: str, raw_completion: str) -> GenerationRecord:
    return GenerationRecord(
        generation_id="generation-1",
        source=source,
        question_hash="question-hash",
        gold_answer=gold_answer,
        training_arm="glyph",
        seed=101,
        checkpoint="checkpoint-1",
        prompt_condition="baseline",
        prompt_hash="prompt-hash",
        raw_prompt="Question",
        raw_completion=raw_completion,
        prompt_token_ids=(1,),
        completion_token_ids=(2,),
        decoding={"temperature": 0.0},
        parent_hashes=("parent-hash",),
    )


def _score(source: str, generation_id: str) -> ScoreRecord:
    return ScoreRecord(
        generation_id=generation_id,
        source=source,
        question_hash=f"question-{generation_id}",
        training_arm="glyph",
        seed=101,
        prompt_condition="baseline",
        gold_answer="1",
        extracted_answer="1",
        normalized_gold="1",
        normalized_prediction="1",
        correct=True,
        parse_error=None,
        equivalence_reason="numeric_equivalent",
    )
