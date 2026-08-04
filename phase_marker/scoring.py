"""Strict final-answer extraction, normalization, and scoring."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re
from typing import Sequence

from phase_marker.schema import GenerationRecord, ScoreRecord


FINAL_LINE = re.compile(r"(?im)^Final answer:\s*(.+?)\s*$")
_LATEX_FRACTION = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


def extract_final_answer(text: str, delimiter: str = "Final answer:") -> str | None:
    """Return the last non-empty line marked with *delimiter*, if present."""
    pattern = FINAL_LINE if delimiter == "Final answer:" else re.compile(
        rf"(?im)^{re.escape(delimiter)}\s*(.+?)\s*$"
    )
    matches = list(pattern.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def normalize_answer(source: str, answer: str) -> str:
    """Normalize a dataset answer without inferring an answer from reasoning."""
    if source.lower() == "math":
        return _normalize_math(answer)

    numeric = _numeric_value(answer)
    return _format_fraction(numeric) if numeric is not None else answer.strip()


def answers_equivalent(source: str, predicted: str, gold: str) -> bool:
    """Compare answers under the source-specific deterministic normalization."""
    return _compare_answers(source, predicted, gold)[0]


def score_generation(record: GenerationRecord) -> ScoreRecord:
    """Score a raw generation using only its explicitly marked final answer."""
    extracted = extract_final_answer(record.raw_completion)
    normalized_gold = normalize_answer(record.source, record.gold_answer)
    if extracted is None:
        return ScoreRecord(
            generation_id=record.generation_id,
            source=record.source,
            question_hash=record.question_hash,
            training_arm=record.training_arm,
            seed=record.seed,
            prompt_condition=record.prompt_condition,
            gold_answer=record.gold_answer,
            extracted_answer=None,
            normalized_gold=normalized_gold,
            normalized_prediction=None,
            correct=False,
            parse_error="missing_final_delimiter",
            equivalence_reason="missing_prediction",
        )

    normalized_prediction = normalize_answer(record.source, extracted)
    correct, reason = _compare_answers(record.source, extracted, record.gold_answer)
    parse_error = _prediction_parse_error(record.source, extracted)
    return ScoreRecord(
        generation_id=record.generation_id,
        source=record.source,
        question_hash=record.question_hash,
        training_arm=record.training_arm,
        seed=record.seed,
        prompt_condition=record.prompt_condition,
        gold_answer=record.gold_answer,
        extracted_answer=extracted,
        normalized_gold=normalized_gold,
        normalized_prediction=normalized_prediction,
        correct=correct,
        parse_error=parse_error,
        equivalence_reason=reason,
    )


def select_audit_sample(
    records: Sequence[ScoreRecord], per_source: int, seed: int
) -> list[ScoreRecord]:
    """Hash-select up to ``per_source`` records for each source deterministically."""
    if per_source < 1:
        raise ValueError("per_source must be positive")

    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source].append(record)

    selected: list[ScoreRecord] = []
    for source in sorted(grouped):
        ranked = sorted(
            grouped[source],
            key=lambda record: (
                _audit_rank(seed, source, record.generation_id),
                record.generation_id,
            ),
        )
        selected.extend(ranked[:per_source])
    return selected


def _compare_answers(source: str, predicted: str, gold: str) -> tuple[bool, str]:
    normalized_prediction = normalize_answer(source, predicted)
    normalized_gold = normalize_answer(source, gold)
    prediction_number = _numeric_value(normalized_prediction)
    gold_number = _numeric_value(normalized_gold)

    if prediction_number is not None and gold_number is not None:
        return (
            prediction_number == gold_number,
            "numeric_equivalent" if prediction_number == gold_number else "numeric_mismatch",
        )
    if source.lower() in {"gsm8k", "svamp"} and prediction_number is None:
        return False, "unparseable_prediction"
    return (
        normalized_prediction == normalized_gold,
        "normalized_equivalent"
        if normalized_prediction == normalized_gold
        else "normalized_mismatch",
    )


def _prediction_parse_error(source: str, prediction: str) -> str | None:
    if source.lower() in {"gsm8k", "svamp"} and _numeric_value(prediction) is None:
        return "unparseable_prediction"
    return None


def _numeric_value(value: str) -> Fraction | Decimal | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        try:
            return Decimal(cleaned[:-1]) / Decimal(100)
        except InvalidOperation:
            return None
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _format_fraction(value: Fraction | Decimal) -> str:
    if isinstance(value, Decimal):
        value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(value)


def _normalize_math(answer: str) -> str:
    value = re.sub(r"\s+", "", answer)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\{", "{").replace(r"\}", "}")
    value = _unwrap_outer_box(value)
    while True:
        converted = _LATEX_FRACTION.sub(r"\1/\2", value)
        if converted == value:
            break
        value = converted

    if value.startswith("{") and value.endswith("}"):
        members = value[1:-1].split(",")
        if all(members):
            return "{" + ",".join(sorted(_normalize_math(member) for member in members)) + "}"

    numeric = _numeric_value(value)
    return _format_fraction(numeric) if numeric is not None else value


def _unwrap_outer_box(value: str) -> str:
    prefix = r"\boxed{"
    while value.startswith(prefix) and value.endswith("}"):
        depth = 0
        closing_index = None
        for index, character in enumerate(value[len(prefix) - 1 :], start=len(prefix) - 1):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break
        if closing_index != len(value) - 1:
            return value
        value = value[len(prefix) : -1]
    return value


def _audit_rank(seed: int, source: str, generation_id: str) -> str:
    payload = f"{seed}\0{source}\0{generation_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
