from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.schema import CanonicalTrace, PhaseSpan
from phase_marker.token_audit import SplitLineageUnavailable, materialize_training_arms

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
        assert saved["metadata"]["tokenizer_revision"] == "fake-qwen-tokenizer-revision"
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
    tokenizer.revision = None
    tokenizer.name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer.init_kwargs = {"revision": "main"}

    manifests = materialize_training_arms(config, traces, tokenizer, tmp_path / "training-data")

    assert manifests["dot"].metadata["tokenizer_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
