"""Immutable records shared by phase-marker pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


@dataclass(frozen=True)
class PhaseSpan:
    name: Literal["guideline", "plan", "step", "takeaway"]
    body: str


@dataclass(frozen=True)
class CanonicalTrace:
    trace_id: str
    source: str
    question: str
    answer: str
    phases: tuple[PhaseSpan, PhaseSpan, PhaseSpan, PhaseSpan]


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    kind: str
    config_hash: str
    parent_hashes: tuple[str, ...]
    row_count: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    source: str
    question_hash: str
    gold_answer: str
    training_arm: str
    seed: int
    checkpoint: str
    prompt_condition: str
    prompt_hash: str
    raw_prompt: str
    raw_completion: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    decoding: Mapping[str, object]
    parent_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ScoreRecord:
    generation_id: str
    source: str
    question_hash: str
    training_arm: str
    seed: int
    prompt_condition: str
    gold_answer: str
    extracted_answer: str | None
    normalized_gold: str
    normalized_prediction: str | None
    correct: bool
    parse_error: str | None
    equivalence_reason: str


@dataclass(frozen=True)
class InterventionRecord:
    intervention_id: str
    recipient_id: str
    donor_id: str
    method: str
    control_name: str
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    source_positions: tuple[int, ...] | None
    control_source_hash: str | None
    baseline_target_logprob: float
    intervened_target_logprob: float
    baseline_target_rank: int
    intervened_target_rank: int
    baseline_donor_target_rank: int | None
    intervened_donor_target_rank: int | None
    baseline_correct: bool
    intervened_correct: bool
    parent_hashes: tuple[str, ...]
