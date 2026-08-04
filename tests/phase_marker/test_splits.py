from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import sha256_json
from phase_marker.schema import CanonicalTrace, PhaseSpan
from phase_marker.splits import (
    DatasetCacheMiss,
    DatasetExample,
    OfflineDatasetLoader,
    SplitBundle,
    SplitOverlapError,
    assert_disjoint_splits,
    build_split_bundle,
    main,
    parse_trace_pool,
    question_hash,
    write_split_bundle,
    _dataset_specs,
    _validation_rows,
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


class FakeDownloadConfig:
    def __init__(self, *, local_files_only: bool):
        self.local_files_only = local_files_only


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
            ("ChilleD/SVAMP", None, "all"): svamp,
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


def test_validation_selection_relabels_copies_without_mutating_train_candidates():
    candidates = tuple(
        example("gsm8k", f"candidate {index}", str(index))
        for index in range(300)
    )
    original_records = tuple(
        (
            row.source,
            row.split,
            row.example_id,
            row.question,
            row.answer,
            row.question_hash,
        )
        for row in candidates
    )
    originals_by_id = {row.example_id: row for row in candidates}

    selected = _validation_rows(candidates, set(), "gsm8k")

    assert {row.split for row in selected} == {"validation"}
    assert {
        row.example_id: (
            row.source,
            row.example_id,
            row.question,
            row.answer,
            row.question_hash,
        )
        for row in selected
    } == {
        row.example_id: (
            row.source,
            row.example_id,
            row.question,
            row.answer,
            row.question_hash,
        )
        for row in candidates
    }
    assert all(row is not originals_by_id[row.example_id] for row in selected)
    assert tuple(
        (
            row.source,
            row.split,
            row.example_id,
            row.question,
            row.answer,
            row.question_hash,
        )
        for row in candidates
    ) == original_records


def test_write_bundle_serializes_and_hashes_relabelled_validation_rows(tmp_path):
    candidates = tuple(
        example("gsm8k", f"artifact candidate {index}", str(index))
        for index in range(300)
    )
    selected = _validation_rows(candidates, set(), "gsm8k")
    bundle = SplitBundle(validation=(selected[0],))
    output_root = tmp_path / "splits"
    frozen_specs = _dataset_specs(
        gsm8k_revision="a" * 40,
        svamp_revision="a" * 40,
        math_revision="a" * 40,
    )
    lineage = {
        "traces": {"sha256": "a" * 64, "path": "data/sft_final.jsonl"},
        "unified": {"sha256": "b" * 64, "path": "data/unified_dataset.jsonl"},
    }
    accounting = {"input_rows": 0, "parsed": 0, "parse_exclusions": 0}

    write_split_bundle(
        output_root,
        TEST_CONFIG,
        bundle,
        dataset_specs=frozen_specs,
        input_lineage=lineage,
        source_pool_accounting=accounting,
        parse_exclusion_provenance=(),
    )

    validation_rows = [
        json.loads(line)
        for line in (output_root / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    artifact_payload = {
        "config": asdict(TEST_CONFIG),
        "train": [],
        "validation": validation_rows,
        "test": [],
        "exclusions": [],
        "datasets": list(frozen_specs),
        "input_lineage": lineage,
        "source_pool_accounting": accounting,
        "parse_exclusion_provenance": [],
    }
    train_labeled_payload = {
        **artifact_payload,
        "validation": [{**row, "split": "train"} for row in validation_rows],
    }

    assert [row["split"] for row in validation_rows] == ["validation"]
    assert manifest["artifact_id"] == sha256_json(artifact_payload)
    assert manifest["artifact_id"] != sha256_json(train_labeled_payload)


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


def test_offline_loader_aggregates_all_pinned_math_configs_in_frozen_order(monkeypatch):
    revision = "a" * 40
    calls = []

    def load_dataset(dataset_id, config, **kwargs):
        calls.append((dataset_id, config, kwargs))
        return ({"config": config, "row": 0}, {"config": config, "row": 1})

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(DownloadConfig=FakeDownloadConfig, load_dataset=load_dataset),
    )

    rows = OfflineDatasetLoader().load(
        "EleutherAI/hendrycks_math", "all", "test", revision
    )

    expected_configs = (
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    )
    assert [config for _, config, _ in calls] == list(expected_configs)
    assert all(dataset_id == "EleutherAI/hendrycks_math" for dataset_id, _, _ in calls)
    assert all(kwargs["split"] == "test" for _, _, kwargs in calls)
    assert all(kwargs["revision"] == revision for _, _, kwargs in calls)
    assert all(
        kwargs["download_config"].local_files_only is True for _, _, kwargs in calls
    )
    assert rows == tuple(
        {"config": config, "row": row}
        for config in expected_configs
        for row in (0, 1)
    )


def test_offline_loader_keeps_non_math_all_requests_to_one_call(monkeypatch):
    calls = []

    def load_dataset(dataset_id, config, **kwargs):
        calls.append((dataset_id, config, kwargs))
        return ({"question": "one"},)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(DownloadConfig=FakeDownloadConfig, load_dataset=load_dataset),
    )

    rows = OfflineDatasetLoader().load("gsm8k", "main", "train", "b" * 40)

    assert rows == ({"question": "one"},)
    assert len(calls) == 1
    assert calls[0][0:2] == ("gsm8k", "main")


def test_offline_loader_concatenates_all_pinned_svamp_rows_in_split_order(monkeypatch):
    revision = "d" * 40
    calls = []

    def load_dataset(dataset_id, config, **kwargs):
        calls.append(
            (
                dataset_id,
                config,
                kwargs["split"],
                kwargs["revision"],
                kwargs["download_config"],
            )
        )
        count = 700 if kwargs["split"] == "train" else 300
        return tuple(
            {"ID": f"{kwargs['split']}-{index}"}
            for index in range(count)
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(DownloadConfig=FakeDownloadConfig, load_dataset=load_dataset),
    )

    rows = OfflineDatasetLoader().load("ChilleD/SVAMP", None, "all", revision)

    assert [(dataset_id, config, split, pinned_revision) for dataset_id, config, split, pinned_revision, _ in calls] == [
        ("ChilleD/SVAMP", None, "train", revision),
        ("ChilleD/SVAMP", None, "test", revision),
    ]
    assert calls[0][4] is calls[1][4]
    assert calls[0][4].local_files_only is True
    assert len(rows) == 1000
    assert [row["ID"] for row in rows[:2]] == ["train-0", "train-1"]
    assert [row["ID"] for row in rows[698:702]] == [
        "train-698",
        "train-699",
        "test-0",
        "test-1",
    ]
    assert rows[-1]["ID"] == "test-299"


def test_offline_loader_normalizes_svamp_constituent_failure(monkeypatch):
    revision = "e" * 40

    def load_dataset(dataset_id, config, **kwargs):
        del dataset_id, config
        if kwargs["split"] == "test":
            raise FileNotFoundError("test split not cached")
        return ({"ID": "train-0"},)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(DownloadConfig=FakeDownloadConfig, load_dataset=load_dataset),
    )

    with pytest.raises(DatasetCacheMiss, match=rf"ChilleD/SVAMP@{revision}"):
        OfflineDatasetLoader().load("ChilleD/SVAMP", None, "all", revision)


def test_offline_loader_normalizes_math_config_failure(monkeypatch):
    def load_dataset(dataset_id, config, **kwargs):
        del dataset_id, kwargs
        if config == "geometry":
            raise FileNotFoundError("not cached")
        return ({"config": config},)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(DownloadConfig=FakeDownloadConfig, load_dataset=load_dataset),
    )

    with pytest.raises(
        DatasetCacheMiss,
        match=r"EleutherAI/hendrycks_math@cccccccccccccccccccccccccccccccccccccccc",
    ):
        OfflineDatasetLoader().load(
            "EleutherAI/hendrycks_math", "all", "train", "c" * 40
        )


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


def test_frozen_publication_requires_immutable_dataset_specs_and_complete_input_lineage(tmp_path):
    output_root = tmp_path / "splits"
    bundle = SplitBundle()
    lineage = {
        "traces": {"sha256": "a" * 64, "path": "data/sft_final.jsonl"},
        "unified": {"sha256": "b" * 64, "path": "data/unified_dataset.jsonl"},
    }
    mutable_specs = (
        {"source": "gsm8k", "dataset_id": "gsm8k", "config": "main", "requested_split": "train", "revision": "main"},
        {"source": "gsm8k", "dataset_id": "gsm8k", "config": "main", "requested_split": "test", "revision": "main"},
        {"source": "svamp", "dataset_id": "ChilleD/SVAMP", "config": None, "requested_split": "all", "revision": "main"},
        {"source": "math", "dataset_id": "EleutherAI/hendrycks_math", "config": "all", "requested_split": "train", "revision": "main"},
        {"source": "math", "dataset_id": "EleutherAI/hendrycks_math", "config": "all", "requested_split": "test", "revision": "main"},
    )

    with pytest.raises(ValueError, match="immutable"):
        write_split_bundle(
            output_root,
            TEST_CONFIG,
            bundle,
            dataset_specs=mutable_specs,
            input_lineage=lineage,
        )

    frozen_specs = tuple({**spec, "revision": "a" * 40} for spec in mutable_specs)
    write_split_bundle(
        output_root,
        TEST_CONFIG,
        bundle,
        dataset_specs=frozen_specs,
        input_lineage=lineage,
        source_pool_accounting={"input_rows": 0, "parsed": 0, "parse_exclusions": 0},
        parse_exclusion_provenance=(),
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"] == list(frozen_specs)
    assert manifest["input_lineage"] == lineage


def test_parse_trace_pool_accounts_for_each_parse_failure_with_reason_and_line(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Problem:\nOne plus one?"},
                            {
                                "role": "assistant",
                                "content": (
                                    "🜞 Guideline: add\n🜆 Plan: add\n🜂 Step: 1+1=2\n"
                                    "🜃 Takeaway: two\n🝞 Final answer: 2"
                                ),
                            },
                        ]
                    }
                ),
                json.dumps({"messages": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    traces, exclusions, accounting = parse_trace_pool(trace_path)

    assert len(traces) == 1
    assert [(row.example_id, row.split) for row in exclusions] == [
        ("line-2", "excluded_parse_invalid_messages")
    ]
    assert accounting == {"input_rows": 2, "parsed": 1, "parse_exclusions": 1}


def test_same_source_duplicate_unified_questions_are_ambiguous(fake_loader):
    duplicate_trace = trace("duplicate question")
    rows = [
        {"id": "duplicate-1", "source": "gsm8k", "question": "duplicate question", "answer": "1"},
        {"id": "duplicate-2", "source": "gsm8k", "question": "duplicate question", "answer": "1"},
    ]

    bundle = build_split_bundle(TEST_CONFIG, fake_loader, [duplicate_trace], rows)

    assert not any(row.question == "duplicate question" for row in bundle.train)
    ambiguous = [row for row in bundle.exclusions if row.split == "excluded_ambiguous"]
    assert len(ambiguous) == 1
    assert "gsm8k:duplicate-1" in ambiguous[0].example_id
    assert "gsm8k:duplicate-2" in ambiguous[0].example_id


def test_frozen_publication_rejects_parse_exclusion_provenance_mismatch(tmp_path):
    output_root = tmp_path / "splits"
    frozen_specs = (
        {"source": "gsm8k", "dataset_id": "gsm8k", "config": "main", "requested_split": "train", "revision": "a" * 40},
        {"source": "gsm8k", "dataset_id": "gsm8k", "config": "main", "requested_split": "test", "revision": "a" * 40},
        {"source": "svamp", "dataset_id": "ChilleD/SVAMP", "config": None, "requested_split": "all", "revision": "a" * 40},
        {"source": "math", "dataset_id": "EleutherAI/hendrycks_math", "config": "all", "requested_split": "train", "revision": "a" * 40},
        {"source": "math", "dataset_id": "EleutherAI/hendrycks_math", "config": "all", "requested_split": "test", "revision": "a" * 40},
    )
    lineage = {
        "traces": {"sha256": "a" * 64, "path": "data/sft_final.jsonl"},
        "unified": {"sha256": "b" * 64, "path": "data/unified_dataset.jsonl"},
    }
    mismatched_bundle = SplitBundle(
        exclusions=[
            DatasetExample(
                source="legacy",
                split="excluded_parse_final_marker",
                example_id="line-2",
                question="",
                answer="",
                question_hash=question_hash("legacy", ""),
            )
        ]
    )

    with pytest.raises(ValueError, match="parse exclusion provenance"):
        write_split_bundle(
            output_root,
            TEST_CONFIG,
            mismatched_bundle,
            dataset_specs=frozen_specs,
            input_lineage=lineage,
            source_pool_accounting={"input_rows": 1, "parsed": 0, "parse_exclusions": 1},
            parse_exclusion_provenance=("line-2|invalid_messages",),
        )

    assert not output_root.exists()
