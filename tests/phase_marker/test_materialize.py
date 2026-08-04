from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import phase_marker.token_audit as token_audit_module
from phase_marker.config import ExperimentConfig
from phase_marker.schema import CanonicalTrace, PhaseSpan
from phase_marker.splits import question_hash
from phase_marker.token_audit import (
    QWEN25_7B_TOKENIZER_REVISION,
    SplitLineageUnavailable,
    _load_cached_tokenizer,
    _load_frozen_training_traces,
    main,
    materialize_training_arms,
)

from tests.phase_marker.test_token_audit import FaithfulFakeTokenizer


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        pilot_seed=42,
        confirmatory_seeds=(101, 202, 303),
        phase_markers=("🜞", "🜆", "🜂", "🜃"),
        final_delimiter="Final answer:",
        arms=("semantic", "glyph", "dot", "random", "direct", "filler"),
    )


@pytest.fixture
def traces() -> list[CanonicalTrace]:
    phases = tuple(
        PhaseSpan(name=name, body=f"{name} body")
        for name in ("guideline", "plan", "step", "takeaway")
    )
    return [
        CanonicalTrace(
            trace_id="trace-1",
            source="gsm8k",
            question="What is 2 + 3?",
            answer="5",
            phases=phases,  # type: ignore[arg-type]
        ),
        CanonicalTrace(
            trace_id="trace-2",
            source="math",
            question="What is 4 + 4?",
            answer="8",
            phases=phases,  # type: ignore[arg-type]
        ),
    ]


def _write_published_train_split(
    tmp_path: Path, traces: list[CanonicalTrace]
) -> None:
    recovered_sources = {"trace-1": "gsm8k", "trace-2": "math"}
    split_root = tmp_path / "splits"
    split_root.mkdir()
    rows = [
        {
            "source": recovered_sources[trace.trace_id],
            "split": "train",
            "example_id": trace.trace_id,
            "question": trace.question,
            "answer": trace.answer,
            "question_hash": question_hash(
                recovered_sources[trace.trace_id], trace.question
            ),
        }
        for trace in traces
    ]
    (split_root / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (split_root / "manifest.json").write_text(
        json.dumps({"artifact_id": "f" * 64}) + "\n", encoding="utf-8"
    )


@pytest.fixture
def materialized(tmp_path: Path, config: ExperimentConfig, traces: list[CanonicalTrace]):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    (split_root / "manifest.json").write_text(
        json.dumps({"artifact_id": "f" * 64}) + "\n", encoding="utf-8"
    )
    return materialize_training_arms(
        config, traces, FaithfulFakeTokenizer(), tmp_path / "training-data"
    )


def test_materialized_marker_arms_share_semantic_hashes(materialized):
    hashes = {
        arm: manifest.metadata["semantic_dataset_hash"]
        for arm, manifest in materialized.items()
        if arm in {"semantic", "glyph", "dot", "random"}
    }

    assert len(set(hashes.values())) == 1


def test_materialization_writes_six_real_jsonl_datasets_and_lineage_manifest(
    tmp_path: Path, materialized
):
    output_root = tmp_path / "training-data"

    assert set(materialized) == {"semantic", "glyph", "dot", "random", "direct", "filler"}
    for arm, manifest in materialized.items():
        rows = [json.loads(line) for line in (output_root / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()]
        saved = json.loads((output_root / f"{arm}.manifest.json").read_text(encoding="utf-8"))
        assert len(rows) == 2
        assert saved["artifact_id"] == manifest.artifact_id
        assert saved["parent_hashes"] == ["f" * 64]
        assert saved["metadata"]["row_hashes"]
        assert saved["metadata"]["neutral_delimiter"] == ". . ."
        assert saved["metadata"]["tokenizer_revision"] == QWEN25_7B_TOKENIZER_REVISION
        assert saved["metadata"]["local_frequency_label"] == "local_corpus_frequency_proxy"


def test_materialization_refuses_to_invent_missing_frozen_split_lineage(
    tmp_path: Path, config: ExperimentConfig, traces: list[CanonicalTrace]
):
    with pytest.raises(SplitLineageUnavailable, match="frozen split manifest"):
        materialize_training_arms(
            config, traces, FaithfulFakeTokenizer(), tmp_path / "training-data"
        )


def test_manifest_binds_the_resolved_cached_qwen_revision(
    tmp_path: Path, config: ExperimentConfig, traces: list[CanonicalTrace]
):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    (split_root / "manifest.json").write_text(
        json.dumps({"artifact_id": "e" * 64}) + "\n", encoding="utf-8"
    )
    tokenizer = FaithfulFakeTokenizer()
    tokenizer.init_kwargs["revision"] = "main"

    manifests = materialize_training_arms(config, traces, tokenizer, tmp_path / "training-data")

    assert manifests["dot"].metadata["tokenizer_revision"] == QWEN25_7B_TOKENIZER_REVISION


def test_cached_tokenizer_loader_uses_exact_filesystem_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen2.5-7B-Instruct"
        / "snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[object, dict[str, object]]] = []
    sentinel = object()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda source, **kwargs: (
            calls.append((source, kwargs)), sentinel
        )[1]
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "ignored"))

    loaded = _load_cached_tokenizer("Qwen/Qwen2.5-7B-Instruct")

    assert loaded is sentinel
    assert calls == [(str(snapshot), {"local_files_only": True})]


@pytest.mark.parametrize("snapshot_state", ("missing", "missing-config", "missing-assets"))
def test_cached_tokenizer_loader_rejects_incomplete_snapshot_before_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snapshot_state: str
):
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen2.5-7B-Instruct"
        / "snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    if snapshot_state != "missing":
        snapshot.mkdir(parents=True)
    if snapshot_state == "missing-config":
        (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    elif snapshot_state == "missing-assets":
        (snapshot / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *_args, **_kwargs: pytest.fail(
            "transformers loader was called"
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="pinned tokenizer snapshot"):
        _load_cached_tokenizer("Qwen/Qwen2.5-7B-Instruct")


def test_materialization_rejects_a_subset_of_the_canonical_six_arms(
    tmp_path: Path, config: ExperimentConfig, traces: list[CanonicalTrace]
):
    subset = replace(config, arms=("semantic",))

    with pytest.raises(ValueError, match="exactly the canonical six arms"):
        materialize_training_arms(subset, traces, FaithfulFakeTokenizer(), tmp_path / "training-data")

    assert not (tmp_path / "training-data").exists()


def test_materialization_rejects_unresolved_or_mismatched_qwen_provenance(
    tmp_path: Path, config: ExperimentConfig, traces: list[CanonicalTrace]
):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    (split_root / "manifest.json").write_text(
        json.dumps({"artifact_id": "d" * 64}) + "\n", encoding="utf-8"
    )
    tokenizer = FaithfulFakeTokenizer()
    tokenizer.init_kwargs = {"_commit_hash": "b" * 40, "revision": "main"}

    with pytest.raises(ValueError, match="resolved tokenizer revision"):
        materialize_training_arms(config, traces, tokenizer, tmp_path / "training-data")

    assert not (tmp_path / "training-data").exists()


def test_frozen_trace_loader_preserves_frozen_order_and_identity(
    tmp_path: Path, traces: list[CanonicalTrace], monkeypatch: pytest.MonkeyPatch
):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    rows = [
        {
            "example_id": "trace-2",
            "source": "math",
            "question": "What is 4 + 4?",
            "answer": "8",
            "split": "train",
        },
        {
            "example_id": "trace-1",
            "source": "gsm8k",
            "question": "What is 2 + 3?",
            "answer": "5",
            "split": "train",
        },
    ]
    (split_root / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr("phase_marker.token_audit.parse_trace_pool", lambda _: (tuple(traces), (), {}))

    selected = _load_frozen_training_traces(tmp_path / "training-data")

    assert [trace.trace_id for trace in selected] == ["trace-2", "trace-1"]


def test_frozen_trace_loader_uses_published_example_identity_and_recovered_source(
    tmp_path: Path, traces: list[CanonicalTrace], monkeypatch: pytest.MonkeyPatch
):
    canonical = [
        replace(traces[1], source="legacy"),
        replace(traces[0], source="legacy"),
    ]
    _write_published_train_split(tmp_path, canonical)
    monkeypatch.setattr(
        token_audit_module,
        "parse_trace_pool",
        lambda _: (tuple(canonical), (), {}),
    )

    selected = _load_frozen_training_traces(tmp_path / "training-data")

    assert [trace.trace_id for trace in selected] == ["trace-2", "trace-1"]
    assert [trace.source for trace in selected] == ["legacy", "legacy"]


def test_materialize_cli_without_limit_consumes_every_frozen_train_row(
    tmp_path: Path,
    config: ExperimentConfig,
    traces: list[CanonicalTrace],
    monkeypatch: pytest.MonkeyPatch,
):
    canonical = [replace(trace, source="legacy") for trace in traces]
    _write_published_train_split(tmp_path, canonical)
    monkeypatch.setattr(
        token_audit_module,
        "parse_trace_pool",
        lambda _: (tuple(canonical), (), {}),
    )
    monkeypatch.setattr(
        token_audit_module,
        "_load_cached_tokenizer",
        lambda _: FaithfulFakeTokenizer(),
    )
    output_root = tmp_path / "training-data"

    main(
        [
            "materialize",
            "--config",
            "configs/phase-marker-qwen25-7b.toml",
            "--output-root",
            str(output_root),
        ]
    )

    for arm in config.arms:
        manifest = json.loads(
            (output_root / f"{arm}.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["row_count"] == 2
        assert manifest["parent_hashes"] == ["f" * 64]
        assert manifest["metadata"]["parent_split_hash"] == "f" * 64


def test_materialize_cli_rejects_limit_larger_than_frozen_train(
    tmp_path: Path, traces: list[CanonicalTrace], monkeypatch: pytest.MonkeyPatch
):
    canonical = [replace(trace, source="legacy") for trace in traces]
    _write_published_train_split(tmp_path, canonical)
    monkeypatch.setattr(
        token_audit_module,
        "parse_trace_pool",
        lambda _: (tuple(canonical), (), {}),
    )

    with pytest.raises(
        SystemExit,
        match=r"frozen training split provides 2 canonical traces, need --limit 3",
    ):
        main(
            [
                "materialize",
                "--config",
                "configs/phase-marker-qwen25-7b.toml",
                "--limit",
                "3",
                "--output-root",
                str(tmp_path / "training-data"),
            ]
        )

    assert not (tmp_path / "training-data").exists()


@pytest.mark.parametrize(
    "row",
    [
        {
            "source": "gsm8k",
            "question": "What is 2 + 3?",
            "answer": "5",
            "split": "train",
        },
        {
            "example_id": "trace-1",
            "source": "gsm8k",
            "question": "What is 2 + 3?",
            "answer": "wrong",
            "split": "train",
        },
    ],
)
def test_frozen_trace_loader_rejects_missing_or_mismatched_identity(
    tmp_path: Path,
    traces: list[CanonicalTrace],
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, str],
):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    (split_root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr("phase_marker.token_audit.parse_trace_pool", lambda _: (tuple(traces), (), {}))

    with pytest.raises(ValueError, match="frozen train row"):
        _load_frozen_training_traces(tmp_path / "training-data")


def test_frozen_trace_loader_rejects_duplicate_stable_ids(
    tmp_path: Path, traces: list[CanonicalTrace], monkeypatch: pytest.MonkeyPatch
):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    row = {
        "example_id": "trace-1",
        "source": "gsm8k",
        "question": "What is 2 + 3?",
        "answer": "5",
        "split": "train",
    }
    (split_root / "train.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("phase_marker.token_audit.parse_trace_pool", lambda _: (tuple(traces), (), {}))

    with pytest.raises(ValueError, match="duplicate"):
        _load_frozen_training_traces(tmp_path / "training-data")
