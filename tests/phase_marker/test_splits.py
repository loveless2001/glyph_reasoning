from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.schema import CanonicalTrace, PhaseSpan
from phase_marker.splits import (
    DatasetCacheMiss,
    DatasetExample,
    SplitBundle,
    SplitOverlapError,
    assert_disjoint_splits,
    build_split_bundle,
    main,
    question_hash,
)


TEST_CONFIG = ExperimentConfig(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    pilot_seed=42,
    confirmatory_seeds=(101, 202, 303),
    phase_markers=("🜞", "🜆", "🜂", "🜃"),
    final_delimiter="Final answer:",
    arms=("semantic", "glyph"),
)


def example(source: str, question: str, answer: str, *, split: str = "train") -> DatasetExample:
    return DatasetExample(
        source=source,
        split=split,
        example_id=f"{source}-{question}",
        question=question,
        answer=answer,
        question_hash=question_hash(source, question),
    )


def trace(question: str, answer: str = "1") -> CanonicalTrace:
    phases = tuple(
        PhaseSpan(name=name, body=name)
        for name in ("guideline", "plan", "step", "takeaway")
    )
    return CanonicalTrace(
        trace_id=f"trace-{question}",
        source="legacy",
        question=question,
        answer=answer,
        phases=phases,  # type: ignore[arg-type]
    )


class InMemoryLoader:
    """Small complete dataset adapter; split logic remains unmocked."""

    def __init__(self, rows: Mapping[tuple[str, str | None, str], Sequence[Mapping[str, object]]]):
        self.rows = rows

    def load(
        self, dataset_id: str, config: str | None, split: str, revision: str
    ) -> Sequence[Mapping[str, object]]:
        assert revision == "main"
        return self.rows[(dataset_id, config, split)]


class MissingDatasetLoader:
    def load(
        self, dataset_id: str, config: str | None, split: str, revision: str
    ) -> Sequence[Mapping[str, object]]:
        raise DatasetCacheMiss(dataset_id, revision)


@pytest.fixture
def source_traces() -> list[CanonicalTrace]:
    return [trace("gsm trace"), trace("math trace"), trace("svamp trace"), trace("not found")]


@pytest.fixture
def unified_rows() -> list[dict[str, object]]:
    return [
        {"id": "g-trace", "source": "gsm8k", "question": "gsm trace", "answer": "1"},
        {"id": "m-trace", "source": "math", "question": "math trace", "answer": "1"},
        {"id": "s-trace", "source": "svamp", "question": "svamp trace", "answer": "1"},
        {"id": "amb-g", "source": "gsm8k", "question": "ambiguous", "answer": "1"},
        {"id": "amb-m", "source": "math", "question": "ambiguous", "answer": "1"},
    ]


@pytest.fixture
def fake_loader() -> InMemoryLoader:
    gsm_train = [{"question": "gsm trace", "answer": "#### 1"}] + [
        {"question": f"gsm validation {index}", "answer": f"#### {index}"}
        for index in range(301)
    ]
    math_train = [{"problem": "math trace", "solution": "\\boxed{1}"}] + [
        {"problem": f"math validation {index}", "solution": f"\\boxed{{{index}}}"}
        for index in range(301)
    ]
    svamp = [
        {"ID": f"svamp-{index}", "Body": f"svamp body {index}", "Question": "what?", "Answer": index}
        for index in range(1000)
    ]
    return InMemoryLoader(
        {
            ("gsm8k", "main", "train"): gsm_train,
            ("gsm8k", "main", "test"): [{"question": "gsm official test", "answer": "#### 4"}],
            ("ChilleD/SVAMP", None, "train"): svamp,
            ("EleutherAI/hendrycks_math", "all", "train"): math_train,
            ("EleutherAI/hendrycks_math", "all", "test"): [
                {"problem": "math official test", "solution": "\\boxed{5}"}
            ],
        }
    )


def test_overlap_gate_rejects_same_question_with_whitespace_changes():
    train = [example("gsm8k", "How many?\n", "5")]
    test = [example("gsm8k", "  How   many? ", "5", split="test")]

    with pytest.raises(SplitOverlapError, match="gsm8k"):
        assert_disjoint_splits(SplitBundle(train=train, validation=[], test=test, exclusions=[]))


def test_svamp_is_never_retained_for_training(source_traces, unified_rows, fake_loader):
    bundle = build_split_bundle(TEST_CONFIG, fake_loader, source_traces, unified_rows)

    assert not any(row.source == "svamp" for row in bundle.train)
    assert sum(row.source == "svamp" for row in bundle.test) == 1000


def test_build_selects_exactly_300_unused_rows_per_validation_source(
    source_traces, unified_rows, fake_loader
):
    bundle = build_split_bundle(TEST_CONFIG, fake_loader, source_traces, unified_rows)

    assert sum(row.source == "gsm8k" for row in bundle.validation) == 300
    assert sum(row.source == "math" for row in bundle.validation) == 300
    assert "gsm trace" not in {row.question for row in bundle.validation}
    assert "math trace" not in {row.question for row in bundle.validation}
    assert [row.question_hash for row in bundle.validation if row.source == "gsm8k"] == sorted(
        row.question_hash for row in bundle.validation if row.source == "gsm8k"
    )


def test_build_records_unmatched_and_ambiguous_source_recovery(fake_loader):
    traces = [trace("ambiguous"), trace("unknown")]
    rows = [
        {"id": "a-g", "source": "gsm8k", "question": "ambiguous", "answer": "1"},
        {"id": "a-m", "source": "math", "question": "ambiguous", "answer": "1"},
    ]

    bundle = build_split_bundle(TEST_CONFIG, fake_loader, traces, rows)

    assert {(row.question, row.split) for row in bundle.exclusions} >= {
        ("ambiguous", "excluded_ambiguous"),
        ("unknown", "excluded_unmatched"),
    }


def test_cache_miss_names_the_dataset_and_revision():
    with pytest.raises(DatasetCacheMiss, match=r"gsm8k@main"):
        build_split_bundle(TEST_CONFIG, MissingDatasetLoader(), [], [])


def test_cli_cache_miss_writes_no_partial_manifest(tmp_path):
    traces = tmp_path / "traces.jsonl"
    unified = tmp_path / "unified.jsonl"
    output_root = tmp_path / "splits"
    traces.write_text("", encoding="utf-8")
    unified.write_text("", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                'model_id = "Qwen/Qwen2.5-7B-Instruct"',
                "pilot_seed = 42",
                "confirmatory_seeds = [101, 202, 303]",
                'phase_markers = ["🜞", "🜆", "🜂", "🜃"]',
                'final_delimiter = "Final answer:"',
                'arms = ["semantic", "glyph"]',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match=r"gsm8k@main"):
        main(
            [
                "build",
                "--config",
                str(config),
                "--traces",
                str(traces),
                "--unified",
                str(unified),
                "--output-root",
                str(output_root),
            ],
            loader=MissingDatasetLoader(),
        )

    assert not output_root.exists()
