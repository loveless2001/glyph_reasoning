"""Confirmatory paired statistics, parser-audit gating, and result tables.

The ``smoke`` command uses synthetic scores only.  Its artifacts validate the
analysis machinery and must never be interpreted as experiment outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, read_jsonl, sha256_json
from phase_marker.schema import ScoreRecord
from phase_marker.scoring import select_audit_sample


_BEHAVIOR_ENVELOPE_FIELDS = {
    "schema_version", "kind", "evidence_scope", "backend", "config_hash", "run_kind",
    "seeds", "split_artifact_id", "split_manifest_hash", "materialization_artifact_ids",
    "checkpoint_artifact_ids", "checkpoint_manifest_hashes", "checkpoint_manifests",
    "examples_file", "examples_hash", "records_file", "records_hash", "row_count",
    "record_hashes", "exclusions", "parent_hashes", "completed", "artifact_id",
}
_AUDIT_ENVELOPE_FIELDS = {
    "schema_version", "kind", "evidence_scope", "config_hash", "run_kind", "seeds",
    "behavior_artifact_id", "behavior_manifest_hash", "labels_file", "labels_hash",
    "row_count", "source_counts", "disagreements", "total", "rate", "passed",
    "parent_hashes", "completed", "artifact_id",
}


PAIR_KEY_FIELDS = ("source", "question_hash", "seed")
MODEL_FORMULA = "correct ~ C(training_arm) * C(prompt_condition) + C(source)"
MODEL_RANDOM_INTERCEPTS = {
    "question_hash": "0 + C(question_hash)",
    "seed": "0 + C(seed)",
}
AUDIT_FIELDS = (
    "generation_id",
    "source",
    "question_hash",
    "training_arm",
    "seed",
    "prompt_condition",
    "gold_answer",
    "extracted_answer",
    "auto_correct",
    "manual_correct",
)
_NORMAL_95 = 1.959963984540054
CONFIRMATORY_ADAPTER_SEEDS = (101, 202, 303)
CONFIRMATORY_BOOTSTRAP_DRAWS = 10_000


class UnpairedComparisonError(ValueError):
    """Raised when a declared paired analysis cannot be aligned one-to-one."""


class AuditGateError(RuntimeError):
    """Raised when parser disagreement blocks confirmatory table creation."""


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    draws: int
    seed: int


@dataclass(frozen=True)
class CoefficientSummary:
    estimate: float
    posterior_sd: float
    low: float
    high: float


@dataclass(frozen=True)
class ModelSummary:
    formula: str
    coefficients: Mapping[str, CoefficientSummary]
    converged: bool
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    disagreements: int
    total: int
    rate: float
    threshold: float


@dataclass(frozen=True)
class ScoredObservation:
    """A score plus distinct adapter and within-run decoding provenance."""

    score: ScoreRecord
    adapter_seed: int
    decoding_seed: int
    completion_index: int | None
    evaluation_kind: str

    @property
    def generation_id(self) -> str:
        return self.score.generation_id

    @property
    def source(self) -> str:
        return self.score.source

    @property
    def question_hash(self) -> str:
        return self.score.question_hash

    @property
    def training_arm(self) -> str:
        return self.score.training_arm

    @property
    def seed(self) -> int:
        return self.score.seed

    @property
    def prompt_condition(self) -> str:
        return self.score.prompt_condition

    @property
    def gold_answer(self) -> str:
        return self.score.gold_answer

    @property
    def extracted_answer(self) -> str | None:
        return self.score.extracted_answer

    @property
    def correct(self) -> bool:
        return self.score.correct


AnalysisRecord = ScoreRecord | ScoredObservation


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    left_training_arm: str
    left_prompt_condition: str
    right_training_arm: str
    right_prompt_condition: str
    secondary: bool = False


@dataclass(frozen=True)
class ContrastResult:
    spec: ContrastSpec
    interval: Interval
    per_seed_deltas: tuple[tuple[int, float], ...]
    across_seed_mean: float
    across_seed_sd: float | None
    inconclusive: bool
    paired_p: float
    holm_adjusted_p: float | None


def load_score_records(path: Path) -> list[ScoredObservation]:
    """Load Task-7 scored envelopes without conflating adapter and decoding seeds."""
    records: list[ScoredObservation] = []
    generation_ids: set[str] = set()
    for line_number, row in enumerate(read_jsonl(path), start=1):
        candidate = row.get("score")
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{path}:{line_number}: scored envelope lacks a score object")
        record = _score_from_mapping(candidate, path=path, line_number=line_number)
        if record.generation_id in generation_ids:
            raise ValueError(
                f"{path}:{line_number}: duplicate generation_id {record.generation_id!r}"
            )
        generation_ids.add(record.generation_id)
        for field in (
            "generation_id",
            "source",
            "question_hash",
            "training_arm",
            "seed",
            "prompt_condition",
        ):
            if row.get(field) != getattr(record, field):
                raise ValueError(
                    f"{path}:{line_number}: envelope {field} does not match score"
                )
        decoding = row.get("decoding")
        provenance = row.get("provenance")
        if not isinstance(decoding, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError(
                f"{path}:{line_number}: scored envelope lacks decoding or provenance"
            )
        adapter_seeds = (
            row.get("seed"),
            record.seed,
            provenance.get("adapter_seed"),
            decoding.get("adapter_seed"),
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in adapter_seeds
        ) or len(set(adapter_seeds)) != 1:
            raise ValueError(
                f"{path}:{line_number}: generation, score, provenance, and decoding "
                "adapter seed values must match"
            )
        decoding_seed = decoding.get("seed")
        if not isinstance(decoding_seed, int) or isinstance(decoding_seed, bool):
            raise ValueError(f"{path}:{line_number}: decoding seed must be an integer")
        evaluation_kind = decoding.get("evaluation_kind")
        if evaluation_kind not in {"primary", "sampled", "perturbation"}:
            raise ValueError(f"{path}:{line_number}: invalid evaluation kind")
        completion_index = decoding.get("completion_index")
        if evaluation_kind == "sampled":
            if (
                not isinstance(completion_index, int)
                or isinstance(completion_index, bool)
                or completion_index not in range(5)
            ):
                raise ValueError(
                    f"{path}:{line_number}: sampled completion index must be 0 through 4"
                )
        elif completion_index is not None:
            raise ValueError(
                f"{path}:{line_number}: non-sampled rows cannot have a completion index"
            )
        records.append(
            ScoredObservation(
                score=record,
                adapter_seed=record.seed,
                decoding_seed=decoding_seed,
                completion_index=completion_index,
                evaluation_kind=str(evaluation_kind),
            )
        )
    return records


def paired_bootstrap_delta(
    left: Sequence[AnalysisRecord],
    right: Sequence[AnalysisRecord],
    seed: int,
    draws: int = 10_000,
) -> Interval:
    """Estimate left-minus-right accuracy with a paired row bootstrap."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("draws must be a positive integer")
    left_by_key = _unique_pair_rows(left, "left")
    right_by_key = _unique_pair_rows(right, "right")
    if not left_by_key:
        raise UnpairedComparisonError("paired comparison requires at least one row")
    if set(left_by_key) != set(right_by_key):
        left_only = sorted(set(left_by_key).difference(right_by_key))[:3]
        right_only = sorted(set(right_by_key).difference(left_by_key))[:3]
        raise UnpairedComparisonError(
            "left and right paired keys differ: "
            f"left_only={left_only!r}, right_only={right_only!r}"
        )

    keys = sorted(left_by_key)
    differences = np.asarray(
        [
            int(left_by_key[key].correct) - int(right_by_key[key].correct)
            for key in keys
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=float)
    # Bound transient memory while preserving one resample per immutable row.
    batch_size = max(1, min(draws, 1_000_000 // len(differences)))
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        indices = rng.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return Interval(
        point=float(differences.mean()),
        low=float(low),
        high=float(high),
        draws=draws,
        seed=seed,
    )


def fit_hierarchical_logit(records: Sequence[AnalysisRecord]) -> ModelSummary:
    """Fit the pre-registered Bayesian binomial mixed model by VB."""
    _validate_model_records(records)
    data = pd.DataFrame(
        {
            "correct": [int(record.correct) for record in records],
            "training_arm": [record.training_arm for record in records],
            "prompt_condition": [record.prompt_condition for record in records],
            "source": [record.source for record in records],
            "question_hash": [record.question_hash for record in records],
            "seed": [record.seed for record in records],
        }
    )
    model = BinomialBayesMixedGLM.from_formula(
        MODEL_FORMULA, MODEL_RANDOM_INTERCEPTS, data
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit_vb(
            fit_method="BFGS",
            minim_opts={"maxiter": 300},
            scale_fe=True,
            verbose=False,
        )

    coefficients = {
        name: CoefficientSummary(
            estimate=float(mean),
            posterior_sd=float(sd),
            low=float(mean - _NORMAL_95 * sd),
            high=float(mean + _NORMAL_95 * sd),
        )
        for name, mean, sd in zip(model.exog_names, result.fe_mean, result.fe_sd)
    }
    optimizer = result.optim_retvals
    success = bool(optimizer.get("success", False))
    jacobian = optimizer.get("jac")
    diagnostics: dict[str, object] = {
        "algorithm": "variational_bayes",
        "optimizer": "BFGS",
        "optimizer_success": success,
        "optimizer_status": int(optimizer.get("status", -1)),
        "optimizer_message": str(optimizer.get("message", "")),
        "iterations": int(optimizer.get("nit", 0)),
        "function_evaluations": int(optimizer.get("nfev", 0)),
        "objective": float(optimizer.get("fun", math.nan)),
        "gradient_l2": (
            float(np.linalg.norm(np.asarray(jacobian, dtype=float)))
            if jacobian is not None
            else None
        ),
        "n_observations": len(records),
        "random_intercepts": list(MODEL_RANDOM_INTERCEPTS),
        "random_log_sd": {
            name: {
                "estimate": float(mean),
                "posterior_sd": float(sd),
            }
            for name, mean, sd in zip(
                model.vcp_names, result.vcp_mean, result.vcp_sd
            )
        },
        "warnings": [str(item.message) for item in caught],
    }
    return ModelSummary(MODEL_FORMULA, coefficients, success, diagnostics)


def generate_manual_audit_template(
    records: Sequence[AnalysisRecord],
    path: Path,
    *,
    seed: int,
) -> tuple[AnalysisRecord, ...]:
    """Write the fixed 100-per-source, 300-row manual audit TSV."""
    sources = sorted({record.source for record in records})
    if len(sources) != 3:
        raise ValueError(
            f"manual audit requires exactly three sources, found {sources!r}"
        )
    selected = tuple(select_audit_sample(records, per_source=100, seed=seed))
    counts = {source: sum(row.source == source for row in selected) for source in sources}
    if counts != {source: 100 for source in sources} or len(selected) != 300:
        raise ValueError(
            "manual audit requires at least 100 unique records for each source"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, delimiter="\t")
            writer.writeheader()
            for record in selected:
                writer.writerow(
                    {
                        "generation_id": record.generation_id,
                        "source": record.source,
                        "question_hash": record.question_hash,
                        "training_arm": record.training_arm,
                        "seed": record.seed,
                        "prompt_condition": record.prompt_condition,
                        "gold_answer": record.gold_answer,
                        "extracted_answer": record.extracted_answer or "",
                        "auto_correct": str(record.correct).lower(),
                        "manual_correct": "",
                    }
                )
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return selected


def read_manual_audit_tsv(path: Path) -> dict[str, bool]:
    """Ingest a completed audit TSV as generation-ID to manual correctness."""
    labels: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"generation_id", "manual_correct"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("manual audit TSV lacks required columns")
        for line_number, row in enumerate(reader, start=2):
            generation_id = row["generation_id"].strip()
            if not generation_id or generation_id in labels:
                raise ValueError(
                    f"{path}:{line_number}: missing or duplicate generation_id"
                )
            labels[generation_id] = _parse_bool_label(
                row["manual_correct"], path, line_number
            )
    if not labels:
        raise ValueError("manual audit TSV contains no labels")
    return labels


def apply_audit_gate(
    auto_scores: Sequence[AnalysisRecord],
    manual_scores: Sequence[AnalysisRecord] | Mapping[str, bool],
    threshold: float = 0.01,
) -> AuditResult:
    """Pass at or below the disagreement threshold; block only above it."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    auto = _unique_generation_labels(auto_scores, "automatic")
    if isinstance(manual_scores, Mapping):
        manual: dict[str, bool] = {}
        for generation_id, value in manual_scores.items():
            if not isinstance(generation_id, str) or not generation_id:
                raise ValueError("manual generation IDs must be non-empty strings")
            if not isinstance(value, bool):
                raise ValueError("manual labels must be booleans")
            manual[generation_id] = value
    else:
        manual = _unique_generation_labels(manual_scores, "manual")
    if not auto:
        raise ValueError("audit gate requires at least one score")
    if set(auto) != set(manual):
        automatic_only = sorted(set(auto).difference(manual))[:3]
        manual_only = sorted(set(manual).difference(auto))[:3]
        raise UnpairedComparisonError(
            "automatic and manual audit generation IDs differ: "
            f"automatic_only={automatic_only!r}, manual_only={manual_only!r}"
        )
    disagreements = sum(auto[key] != manual[key] for key in auto)
    rate = disagreements / len(auto)
    return AuditResult(
        passed=rate <= threshold,
        disagreements=disagreements,
        total=len(auto),
        rate=rate,
        threshold=threshold,
    )


def effect_is_inconclusive(interval: Interval) -> bool:
    """Apply the pre-registered two-point and zero-spanning stop rule."""
    return abs(interval.point) < 0.02 or interval.low <= 0 <= interval.high


def build_contrast_results(
    records: Sequence[AnalysisRecord],
    contrasts: Sequence[ContrastSpec],
    *,
    bootstrap_seed: int,
    draws: int = 10_000,
) -> tuple[ContrastResult, ...]:
    """Evaluate declared contrasts and Holm-adjust declared secondaries only."""
    names = [contrast.name for contrast in contrasts]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("contrast names must be non-empty and unique")
    results: list[ContrastResult] = []
    for contrast in contrasts:
        left = [
            record
            for record in records
            if record.training_arm == contrast.left_training_arm
            and record.prompt_condition == contrast.left_prompt_condition
        ]
        right = [
            record
            for record in records
            if record.training_arm == contrast.right_training_arm
            and record.prompt_condition == contrast.right_prompt_condition
        ]
        interval = paired_bootstrap_delta(
            left, right, seed=bootstrap_seed, draws=draws
        )
        left_by_key = _unique_pair_rows(left, "left")
        right_by_key = _unique_pair_rows(right, "right")
        keys = sorted(left_by_key)
        seeds = sorted({key[2] for key in keys})
        per_seed = tuple(
            (
                seed,
                float(
                    np.mean(
                        [
                            int(left_by_key[key].correct)
                            - int(right_by_key[key].correct)
                            for key in keys
                            if key[2] == seed
                        ]
                    )
                ),
            )
            for seed in seeds
        )
        seed_deltas = np.asarray([delta for _, delta in per_seed], dtype=float)
        differences = np.asarray(
            [
                int(left_by_key[key].correct) - int(right_by_key[key].correct)
                for key in keys
            ],
            dtype=int,
        )
        discordant = differences[differences != 0]
        paired_p = (
            float(binomtest(int(np.sum(discordant > 0)), len(discordant), 0.5).pvalue)
            if len(discordant)
            else 1.0
        )
        results.append(
            ContrastResult(
                spec=contrast,
                interval=interval,
                per_seed_deltas=per_seed,
                across_seed_mean=float(seed_deltas.mean()),
                across_seed_sd=(
                    float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else None
                ),
                inconclusive=effect_is_inconclusive(interval),
                paired_p=paired_p,
                holm_adjusted_p=None,
            )
        )
    return _apply_secondary_holm(results)


def write_confirmatory_outputs(
    output_root: Path,
    results: Sequence[ContrastResult],
    model: ModelSummary,
    auto_scores: Sequence[AnalysisRecord],
    manual_scores: Sequence[AnalysisRecord] | Mapping[str, bool],
    *,
    synthetic: bool = False,
) -> dict[str, Path]:
    """Recompute the fixed audit, then write gated analysis artifacts."""
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": output_root / "contrast-table.md",
        "latex": output_root / "contrast-table.tex",
        "summary": output_root / "summary.json",
        "model_diagnostics": output_root / "model-diagnostics.json",
        "audit_status": output_root / "audit-status.json",
    }
    audit = apply_audit_gate(auto_scores, manual_scores, threshold=0.01)
    source_counts = Counter(record.source for record in auto_scores)
    analysis_mode = "synthetic_test_only" if synthetic else "confirmatory"
    population_error: str | None = None
    if len(auto_scores) != 300:
        population_error = (
            f"confirmatory audit requires exactly 300 matched identities, "
            f"found {len(auto_scores)}"
        )
    elif source_counts != Counter({"gsm8k": 100, "math": 100, "svamp": 100}):
        population_error = (
            "confirmatory audit requires exactly 100 rows for each of "
            f"gsm8k, math, and svamp; found {dict(sorted(source_counts.items()))!r}"
        )
    protocol_error = _confirmatory_protocol_error(results) if not synthetic else None
    reason = population_error or protocol_error
    if reason is None and not audit.passed:
        reason = (
            f"audit disagreement {audit.rate:.4f} exceeds {audit.threshold:.4f}"
        )
    _atomic_json(
        paths["audit_status"],
        {
            **asdict(audit),
            "population_valid": population_error is None,
            "protocol_valid": protocol_error is None,
            "analysis_eligible": audit.passed and reason is None,
            "analysis_mode": analysis_mode,
            "source_counts": dict(sorted(source_counts.items())),
            "reason": reason,
        },
    )
    if population_error is not None:
        _remove_confirmatory_outputs(paths)
        raise AuditGateError(population_error + " and blocks confirmatory tables")
    if not audit.passed:
        _remove_confirmatory_outputs(paths)
        raise AuditGateError(
            f"audit disagreement {audit.rate:.4f} exceeds {audit.threshold:.4f} "
            "and blocks confirmatory tables"
        )
    if protocol_error is not None:
        _remove_confirmatory_outputs(paths)
        raise ValueError(protocol_error)

    _atomic_text(paths["markdown"], _render_markdown(results, synthetic))
    _atomic_text(paths["latex"], _render_latex(results, synthetic))
    _atomic_json(paths["model_diagnostics"], _model_payload(model))
    _atomic_json(
        paths["summary"],
        {
            "analysis_mode": analysis_mode,
            "synthetic_smoke": synthetic,
            "experiment_outcomes": not synthetic,
            "audit": asdict(audit),
            "model": _model_payload(model),
            "contrasts": [asdict(result) for result in results],
        },
    )
    return paths


def _remove_confirmatory_outputs(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if name != "audit_status":
            path.unlink(missing_ok=True)


def _confirmatory_protocol_error(
    results: Sequence[ContrastResult],
) -> str | None:
    if not results:
        return "non-synthetic confirmatory outputs require a nonempty declared contrast set"
    for result in results:
        if result.interval.draws != CONFIRMATORY_BOOTSTRAP_DRAWS:
            return (
                "non-synthetic confirmatory intervals require exactly 10,000 "
                f"bootstrap draws; {result.spec.name!r} has {result.interval.draws}"
            )
        seeds = tuple(seed for seed, _ in result.per_seed_deltas)
        if seeds != CONFIRMATORY_ADAPTER_SEEDS:
            return (
                "non-synthetic confirmatory seed summaries require adapter seeds "
                f"(101, 202, 303); {result.spec.name!r} has {seeds!r}"
            )
    return None


def _unique_pair_rows(
    records: Sequence[AnalysisRecord], label: str
) -> dict[tuple[str, str, int], AnalysisRecord]:
    rows: dict[tuple[str, str, int], AnalysisRecord] = {}
    for record in records:
        if isinstance(record, ScoredObservation) and record.evaluation_kind != "primary":
            raise UnpairedComparisonError(
                f"{label} primary greedy analysis cannot include "
                f"{record.evaluation_kind!r} observations"
            )
        key = (record.source, record.question_hash, record.seed)
        if key in rows:
            raise UnpairedComparisonError(
                f"{label} contains duplicate paired key {key!r}; rows may not be averaged"
            )
        rows[key] = record
    return rows


def _unique_generation_labels(
    records: Sequence[AnalysisRecord], label: str
) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for record in records:
        if record.generation_id in labels:
            raise UnpairedComparisonError(
                f"{label} audit contains duplicate generation ID {record.generation_id!r}"
            )
        labels[record.generation_id] = record.correct
    return labels


def _validate_model_records(records: Sequence[AnalysisRecord]) -> None:
    if not records:
        raise ValueError("hierarchical model requires score records")
    generation_ids = [record.generation_id for record in records]
    if len(generation_ids) != len(set(generation_ids)):
        raise ValueError("hierarchical model generation IDs must be unique")
    analysis_cells = [
        (
            record.source,
            record.question_hash,
            record.seed,
            record.training_arm,
            record.prompt_condition,
        )
        for record in records
    ]
    if len(analysis_cells) != len(set(analysis_cells)):
        raise ValueError(
            "hierarchical model contains a duplicate analysis cell; sampled "
            "completions cannot be treated as independent adapter-seed rows"
        )
    non_primary = [
        record.evaluation_kind
        for record in records
        if isinstance(record, ScoredObservation) and record.evaluation_kind != "primary"
    ]
    if non_primary:
        raise ValueError(
            "hierarchical model accepts primary greedy observations only; "
            f"found {sorted(set(non_primary))!r}"
        )
    dimensions = {
        "training arms": {record.training_arm for record in records},
        "prompt conditions": {record.prompt_condition for record in records},
        "datasets": {record.source for record in records},
        "question hashes": {record.question_hash for record in records},
        "seeds": {record.seed for record in records},
        "outcomes": {record.correct for record in records},
    }
    for name, values in dimensions.items():
        if len(values) < 2:
            raise ValueError(f"hierarchical model requires at least two {name}")


def _apply_secondary_holm(
    results: Sequence[ContrastResult],
) -> tuple[ContrastResult, ...]:
    secondary = sorted(
        (
            (index, result.paired_p)
            for index, result in enumerate(results)
            if result.spec.secondary
        ),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(secondary)
    for rank, (index, p_value) in enumerate(secondary):
        candidate = min(1.0, p_value * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return tuple(
        replace(result, holm_adjusted_p=adjusted[index])
        if index in adjusted
        else result
        for index, result in enumerate(results)
    )


def _render_markdown(
    results: Sequence[ContrastResult], synthetic: bool
) -> str:
    lines = []
    if synthetic:
        lines.extend(
            [
                "> **Synthetic/test-only analysis:** these values are not experiment outcomes.",
                "",
            ]
        )
    lines.extend(
        [
            "| Contrast | Delta | Evaluation-sample 95% paired bootstrap CI | Three-seed variation | Holm-adjusted p (declared secondary only) | Interpretation |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        seed_text = _seed_variation_text(result)
        holm = (
            f"{result.holm_adjusted_p:.4g}"
            if result.holm_adjusted_p is not None
            else "not applied"
        )
        interpretation = "inconclusive" if result.inconclusive else "directional"
        lines.append(
            f"| {result.spec.name} | {100 * result.interval.point:.2f} pp | "
            f"[{100 * result.interval.low:.2f}, {100 * result.interval.high:.2f}] pp | "
            f"{seed_text} | {holm} | {interpretation} |"
        )
    return "\n".join(lines) + "\n"


def _render_latex(results: Sequence[ContrastResult], synthetic: bool) -> str:
    lines = []
    if synthetic:
        lines.append(
            "% Synthetic/test-only analysis; these values are not experiment outcomes."
        )
    lines.extend(
        [
            "\\begin{tabular}{lrrrrl}",
            "\\toprule",
            "Contrast & Delta & Evaluation-sample 95\\% paired bootstrap CI & Three-seed variation & Holm $p$ & Interpretation \\\\",
            "\\midrule",
        ]
    )
    for result in results:
        holm = (
            f"{result.holm_adjusted_p:.4g}"
            if result.holm_adjusted_p is not None
            else "--"
        )
        interpretation = "inconclusive" if result.inconclusive else "directional"
        lines.append(
            f"{_latex_escape(result.spec.name)} & {100 * result.interval.point:.2f} pp & "
            f"[{100 * result.interval.low:.2f}, {100 * result.interval.high:.2f}] pp & "
            f"{_latex_escape(_seed_variation_text(result))} & {holm} & {interpretation} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _seed_variation_text(result: ContrastResult) -> str:
    deltas = ", ".join(
        f"{seed}: {100 * delta:.2f} pp" for seed, delta in result.per_seed_deltas
    )
    sd = "n/a" if result.across_seed_sd is None else f"{100 * result.across_seed_sd:.2f} pp"
    return f"mean {100 * result.across_seed_mean:.2f} pp; SD {sd}; {deltas}"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _model_payload(model: ModelSummary) -> dict[str, object]:
    return {
        "formula": model.formula,
        "coefficients": {
            name: asdict(coefficient)
            for name, coefficient in model.coefficients.items()
        },
        "converged": model.converged,
        "diagnostics": dict(model.diagnostics),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _parse_bool_label(value: str, path: Path, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"{path}:{line_number}: manual_correct must be true or false"
    )


def _score_from_mapping(
    row: Mapping[str, object], *, path: Path, line_number: int
) -> ScoreRecord:
    fields = tuple(ScoreRecord.__dataclass_fields__)
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{path}:{line_number}: score lacks fields {missing!r}")
    values = {field: row[field] for field in fields}
    if not isinstance(values["seed"], int) or isinstance(values["seed"], bool):
        raise ValueError(f"{path}:{line_number}: score seed must be an integer")
    if not isinstance(values["correct"], bool):
        raise ValueError(f"{path}:{line_number}: score correct must be a boolean")
    for field in (
        "generation_id",
        "source",
        "question_hash",
        "training_arm",
        "prompt_condition",
        "gold_answer",
        "normalized_gold",
        "equivalence_reason",
    ):
        if not isinstance(values[field], str) or not values[field]:
            raise ValueError(f"{path}:{line_number}: score {field} must be non-empty")
    for field in ("extracted_answer", "normalized_prediction", "parse_error"):
        if values[field] is not None and not isinstance(values[field], str):
            raise ValueError(f"{path}:{line_number}: score {field} has invalid type")
    return ScoreRecord(**values)  # type: ignore[arg-type]


def _synthetic_score(
    source: str,
    question_index: int,
    seed: int,
    arm: str,
    prompt: str,
    correct: bool,
) -> ScoreRecord:
    question_hash = f"synthetic-{source}-q{question_index}"
    generation_id = f"{source}:{question_index}:{seed}:{arm}:{prompt}"
    prediction = "1" if correct else "0"
    return ScoreRecord(
        generation_id=generation_id,
        source=source,
        question_hash=question_hash,
        training_arm=arm,
        seed=seed,
        prompt_condition=prompt,
        gold_answer="1",
        extracted_answer=prediction,
        normalized_gold="1",
        normalized_prediction=prediction,
        correct=correct,
        parse_error=None,
        equivalence_reason="numeric_equivalent" if correct else "numeric_mismatch",
    )


def _smoke_records() -> list[ScoreRecord]:
    records = []
    sources = ("gsm8k", "svamp", "math")
    arms = ("semantic", "glyph", "dot")
    prompts = ("neutral", "glyph", "dot")
    for source_index, source in enumerate(sources):
        for question_index in range(4):
            for seed_index, seed in enumerate((101, 202, 303)):
                for arm_index, arm in enumerate(arms):
                    for prompt_index, prompt in enumerate(prompts):
                        score = (
                            2 * question_index
                            + seed_index
                            + source_index
                            + arm_index
                            + prompt_index
                            + (1 if arm == "glyph" and prompt == "glyph" else 0)
                        )
                        records.append(
                            _synthetic_score(
                                source,
                                question_index,
                                seed,
                                arm,
                                prompt,
                                score % 7 < 3,
                            )
                        )
    return records


def _smoke_audit_records() -> list[ScoreRecord]:
    return [
        _synthetic_score(source, index, 101, "glyph", "glyph", index % 3 != 0)
        for source in ("gsm8k", "svamp", "math")
        for index in range(100)
    ]


def _run_smoke(output_root: Path) -> int:
    records = _smoke_records()
    contrasts = (
        ContrastSpec(
            "glyph-glyph-v-semantic-neutral",
            "glyph",
            "glyph",
            "semantic",
            "neutral",
        ),
        ContrastSpec(
            "glyph-glyph-v-dot-dot",
            "glyph",
            "glyph",
            "dot",
            "dot",
            secondary=True,
        ),
        ContrastSpec(
            "glyph-glyph-v-glyph-dot",
            "glyph",
            "glyph",
            "glyph",
            "dot",
            secondary=True,
        ),
    )
    results = build_contrast_results(
        records, contrasts, bootstrap_seed=20260804, draws=10_000
    )
    model = fit_hierarchical_logit(records)
    audit_records = _smoke_audit_records()
    selected = generate_manual_audit_template(
        audit_records, output_root / "manual-audit.tsv", seed=20260804
    )
    audit = apply_audit_gate(
        selected,
        {record.generation_id: record.correct for record in selected},
        threshold=0.01,
    )
    paths = write_confirmatory_outputs(
        output_root,
        results,
        model,
        selected,
        {record.generation_id: record.correct for record in selected},
        synthetic=True,
    )
    print("synthetic smoke only: no experiment outcomes")
    print(
        "audit gate: "
        f"{'PASS' if audit.passed else 'BLOCKED'} "
        f"({audit.disagreements}/{audit.total} disagreements; "
        f"rate={audit.rate:.4f}, threshold={audit.threshold:.4f})"
    )
    print(f"model converged: {str(model.converged).lower()}")
    print(f"wrote analysis artifacts to {output_root}")
    print("artifacts: " + ", ".join(sorted(path.name for path in paths.values())))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    audit.add_argument("--seeds", type=int, nargs="+", required=True)
    audit.add_argument("--generations", type=Path, required=True)
    audit.add_argument("--manual-labels", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--allow-test-evidence", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--generations", type=Path, required=True)
    analyze.add_argument("--manual-audit", type=Path, required=True)
    analyze.add_argument("--audit-manifest", type=Path)
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--allow-test-evidence", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "smoke":
        return _run_smoke(arguments.output_root)
    if arguments.command == "audit":
        return _run_audit(arguments)
    if arguments.command == "analyze":
        return _run_analyze(arguments)
    raise AssertionError("unreachable")


def _run_audit(arguments: argparse.Namespace) -> int:
    config = ExperimentConfig.load(arguments.config)
    behavior_path, behavior = _load_behavior_envelope(arguments.generations)
    _validate_analysis_parent(
        behavior_path, behavior, config, arguments.allow_test_evidence,
        kind=arguments.kind, seeds=tuple(arguments.seeds)
    )
    records_path = behavior_path.parent / str(behavior["records_file"])
    records = load_score_records(records_path)
    selected = tuple(select_audit_sample(records, per_source=100, seed=20260804))
    source_counts = Counter(record.source for record in selected)
    if len(selected) != 300 or source_counts != Counter(
        {"gsm8k": 100, "svamp": 100, "math": 100}
    ):
        raise ValueError("manual audit requires exactly 300 labels and 100 per source")
    manual = read_manual_audit_tsv(arguments.manual_labels)
    result = apply_audit_gate(selected, manual, threshold=0.01)
    _prepare_audit_output_root(arguments.output_root, arguments.manual_labels)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "phase_marker_manual_audit",
        "evidence_scope": "experiment" if behavior["evidence_scope"] != "plumbing_only" else "plumbing_only",
        "config_hash": sha256_json(asdict(config)),
        "run_kind": arguments.kind,
        "seeds": list(arguments.seeds),
        "behavior_artifact_id": behavior["artifact_id"],
        "behavior_manifest_hash": _statistics_file_hash(behavior_path),
        "labels_file": str(arguments.manual_labels),
        "labels_hash": _statistics_file_hash(arguments.manual_labels),
        "row_count": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "disagreements": result.disagreements,
        "total": result.total,
        "rate": result.rate,
        "passed": result.passed,
        "parent_hashes": [behavior["artifact_id"]],
        "completed": True,
    }
    manifest["artifact_id"] = sha256_json(manifest)
    _atomic_json(arguments.output_root / "manifest.json", manifest)
    print(canonical_json(manifest))
    return 0 if result.passed else 1


def _run_analyze(arguments: argparse.Namespace) -> int:
    config = ExperimentConfig.load(arguments.config)
    behavior_path, behavior = _load_behavior_envelope(arguments.generations)
    kind = behavior.get("run_kind")
    seeds = behavior.get("seeds")
    if not isinstance(kind, str) or not isinstance(seeds, list):
        raise ValueError("behavior manifest lacks run identity")
    _validate_analysis_parent(
        behavior_path, behavior, config, arguments.allow_test_evidence,
        kind=kind, seeds=tuple(seeds)
    )
    audit_path = _resolve_audit_manifest(
        arguments.manual_audit, kind, arguments.audit_manifest
    )
    audit = _statistics_read_object(audit_path, "audit manifest")
    if (
        set(audit) != _AUDIT_ENVELOPE_FIELDS
        or audit.get("schema_version") != 1
        or audit.get("kind") != "phase_marker_manual_audit"
        or audit.get("evidence_scope") != (
            "plumbing_only" if arguments.allow_test_evidence else "experiment"
        )
        or audit.get("config_hash") != sha256_json(asdict(config))
        or audit.get("behavior_artifact_id") != behavior.get("artifact_id")
        or audit.get("labels_file") != str(arguments.manual_audit)
        or audit.get("labels_hash") != _statistics_file_hash(arguments.manual_audit)
        or audit.get("completed") is not True
        or audit.get("passed") is not True
    ):
        raise ValueError("manual audit manifest lineage or completion mismatch")
    unsigned_audit = dict(audit)
    audit_id = unsigned_audit.pop("artifact_id", None)
    if audit_id != sha256_json(unsigned_audit):
        raise ValueError("manual audit artifact hash mismatch")
    records = load_score_records(behavior_path.parent / str(behavior["records_file"]))
    primary = [record for record in records if record.evaluation_kind == "primary"]
    contrasts = (
        ContrastSpec("glyph-glyph-v-semantic-neutral", "glyph", "glyph", "semantic", "neutral"),
        ContrastSpec("glyph-glyph-v-dot-dot", "glyph", "glyph", "dot", "dot", secondary=True),
        ContrastSpec("glyph-glyph-v-glyph-dot", "glyph", "glyph", "glyph", "dot", secondary=True),
    )
    results = build_contrast_results(
        primary, contrasts, bootstrap_seed=20260804, draws=CONFIRMATORY_BOOTSTRAP_DRAWS
    )
    model = fit_hierarchical_logit(primary)
    manual = read_manual_audit_tsv(arguments.manual_audit)
    by_id = {record.generation_id: record for record in records}
    audited = [by_id[generation_id] for generation_id in manual]
    paths = write_confirmatory_outputs(
        arguments.output_root,
        results,
        model,
        audited,
        manual,
        synthetic=behavior.get("evidence_scope") == "plumbing_only",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "phase_marker_confirmatory_analysis",
        "evidence_scope": behavior["evidence_scope"],
        "config_hash": sha256_json(asdict(config)),
        "run_kind": kind,
        "seeds": seeds,
        "behavior_artifact_id": behavior["artifact_id"],
        "audit_artifact_id": audit_id,
        "output_hashes": {
            name: _statistics_file_hash(path) for name, path in sorted(paths.items())
        },
        "parent_hashes": [behavior["artifact_id"], audit_id],
        "completed": True,
    }
    manifest["artifact_id"] = sha256_json(manifest)
    _atomic_json(arguments.output_root / "manifest.json", manifest)
    print(canonical_json(manifest))
    return 0


def _load_behavior_envelope(generations: Path) -> tuple[Path, Mapping[str, object]]:
    if not generations.is_dir():
        path = generations
    elif (generations / "manifest.json").is_file():
        path = generations / "manifest.json"
    else:
        candidates = tuple(sorted(generations.glob("*/manifest.json")))
        if len(candidates) != 1:
            raise ValueError(
                "generations root must contain exactly one audit-bound kind manifest"
            )
        path = candidates[0]
    return path, _statistics_read_object(path, "behavior manifest")


def _resolve_audit_manifest(
    manual_audit: Path, kind: object, explicit: Path | None
) -> Path:
    if explicit is not None:
        return explicit
    if not isinstance(kind, str) or not kind:
        raise ValueError("cannot infer audit manifest without a run kind")
    candidates = tuple(
        path for path in (
            manual_audit.parent / "manifest.json",
            manual_audit.parent / kind / "manifest.json",
        ) if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError("manual audit path does not resolve exactly one audit manifest")
    return candidates[0]


def _prepare_audit_output_root(output_root: Path, labels_path: Path) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    existing = {path.name for path in output_root.iterdir()}
    allowed = labels_path.parent == output_root and existing == {labels_path.name}
    if not allowed:
        raise FileExistsError(f"refusing to overwrite audit output: {output_root}")


def _validate_analysis_parent(
    behavior_path: Path,
    behavior: Mapping[str, object],
    config: ExperimentConfig,
    allow_test: bool,
    *,
    kind: str,
    seeds: tuple[object, ...],
) -> None:
    expected_seeds = (42,) if kind == "pilot" else (101, 202, 303)
    if (
        set(behavior) != _BEHAVIOR_ENVELOPE_FIELDS
        or behavior.get("schema_version") != 1
        or behavior.get("kind") != "phase_marker_behavior_generations"
        or behavior.get("config_hash") != sha256_json(asdict(config))
        or behavior.get("run_kind") != kind
        or tuple(behavior.get("seeds", ())) != expected_seeds
        or seeds != expected_seeds
        or behavior.get("completed") is not True
    ):
        raise ValueError("behavior manifest config/run lineage mismatch")
    if behavior.get("evidence_scope") == "plumbing_only" and not allow_test:
        raise ValueError("production analysis rejects plumbing-only behavior evidence")
    expected_backend = (
        "tiny-fixture" if behavior.get("evidence_scope") == "plumbing_only" else "vllm"
    )
    if behavior.get("backend") != expected_backend:
        raise ValueError("behavior manifest backend/evidence scope mismatch")
    unsigned = dict(behavior)
    artifact_id = unsigned.pop("artifact_id", None)
    if artifact_id != sha256_json(unsigned):
        raise ValueError("behavior manifest artifact hash mismatch")
    records_path = behavior_path.parent / str(behavior.get("records_file"))
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        not rows
        or behavior.get("records_hash") != _statistics_file_hash(records_path)
        or behavior.get("row_count") != len(rows)
        or behavior.get("record_hashes") != [sha256_json(row) for row in rows]
    ):
        raise ValueError("behavior records count or hashes mismatch")
    examples_path = Path(str(behavior.get("examples_file")))
    if behavior.get("examples_hash") != _statistics_file_hash(examples_path):
        raise ValueError("behavior examples hash mismatch")
    manifests = behavior.get("checkpoint_manifests")
    manifest_hashes = behavior.get("checkpoint_manifest_hashes")
    if (
        not isinstance(manifests, Mapping)
        or not isinstance(manifest_hashes, Mapping)
        or set(manifests) != set(manifest_hashes)
    ):
        raise ValueError("behavior checkpoint lineage maps mismatch")
    for key, value in manifests.items():
        if (
            not isinstance(value, str)
            or manifest_hashes[key] != _statistics_file_hash(Path(value))
        ):
            raise ValueError("behavior checkpoint manifest hash mismatch")


def _statistics_read_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {label}: {path}") from None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _statistics_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
