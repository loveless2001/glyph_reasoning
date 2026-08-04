from dataclasses import asdict

import pytest

from phase_marker.behavior import (
    _vllm_sampling_parameters,
    EvaluationCell,
    FakeGenerationBackend,
    GenerationOutput,
    GenerationRequest,
    build_generation_requests,
    build_behavior_matrix,
    records_from_outputs,
    serialize_generation_record,
)
from phase_marker.config import ExperimentConfig
from phase_marker.schema import ArtifactManifest
from phase_marker.splits import DatasetExample


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
def split_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id="a" * 64,
        kind="phase_marker_splits",
        config_hash="b" * 64,
        parent_hashes=("c" * 64,),
        row_count=3,
        metadata={"sampled_test_hashes": ("question-1",)},
    )


def test_matrix_contains_preregistered_cells_only(config, split_manifest):
    cells = build_behavior_matrix(config, split_manifest)
    assert {
        (cell.training_arm, cell.prompt_condition)
        for cell in cells
        if cell.kind == "primary"
    } == {
        (arm, prompt)
        for arm in ("semantic", "glyph", "dot", "random")
        for prompt in ("neutral", "glyph", "dot", "headings")
    }
    assert {
        (cell.training_arm, cell.prompt_condition)
        for cell in cells
        if cell.kind == "sampled"
    } == {
        ("semantic", "neutral"),
        ("glyph", "neutral"),
        ("glyph", "glyph"),
        ("glyph", "dot"),
    }
    assert {
        cell.perturbation for cell in cells if cell.kind == "perturbation"
    } == {"delete", "cluster", "displace", "permute", "dot_replace", "unseen_replace"}
    assert all(cell.decoding_name == "greedy" for cell in cells if cell.kind != "sampled")
    assert all(cell.decoding_name == "sampled" for cell in cells if cell.kind == "sampled")


def test_records_preserve_ordered_raw_outputs_and_are_independently_rescorable():
    cell = EvaluationCell("primary", "glyph", "glyph", None, "greedy")
    examples = (
        DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1"),
        DatasetExample("gsm8k", "test", "two", "2 + 2", "4", "question-2"),
    )
    requests = (
        GenerationRequest(
            "generation-1", "prompt 1", (11, 12), 64, {"seed": 42, "checkpoint": "ckpt"}
        ),
        GenerationRequest(
            "generation-2", "prompt 2", (21, 22), 64, {"seed": 42, "checkpoint": "ckpt"}
        ),
    )
    outputs = (
        GenerationOutput("generation-1", "work\nFinal answer: 2", (31, 32), (-0.1, -0.2)),
        GenerationOutput("generation-2", "work\nFinal answer: 4", (41, 42), (-0.3, -0.4)),
    )

    records = records_from_outputs(cell, examples, requests, outputs, ("parent" * 8,))

    assert [record.generation_id for record in records] == ["generation-1", "generation-2"]
    assert records[0].raw_prompt == "prompt 1"
    assert records[0].raw_completion == "work\nFinal answer: 2"
    assert records[0].prompt_token_ids == (11, 12)
    assert records[0].completion_token_ids == (31, 32)
    assert records[0].decoding["completion_token_logprobs"] == [-0.1, -0.2]
    assert records[0].parent_hashes == ("parent" * 8,)
    assert asdict(records[0])["gold_answer"] == "2"

    persisted = serialize_generation_record(records[0])
    assert persisted["prompt_token_count"] == 2
    assert persisted["completion_token_count"] == 2
    assert persisted["raw_prompt"] == "prompt 1"


def test_records_reject_missing_duplicate_or_reordered_output_ids():
    cell = EvaluationCell("primary", "glyph", "glyph", None, "greedy")
    example = DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1")
    request = GenerationRequest(
        "generation-1", "prompt", (1,), 64, {"seed": 42, "checkpoint": "ckpt"}
    )
    duplicate = GenerationOutput("generation-1", "Final answer: 2", (2,), ())
    with pytest.raises(ValueError, match="unique"):
        records_from_outputs(cell, (example, example), (request, request), (duplicate, duplicate), ())
    with pytest.raises(ValueError, match="order"):
        records_from_outputs(
            cell,
            (example,),
            (request,),
            (GenerationOutput("other", "Final answer: 2", (2,), ()),),
            (),
        )


def test_sampled_cells_expand_to_five_deterministic_independent_completions(config):
    cell = EvaluationCell("sampled", "glyph", "glyph", None, "sampled")
    examples = (
        DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1"),
        DatasetExample("gsm8k", "test", "two", "2 + 2", "4", "question-2"),
    )

    requests = build_generation_requests(cell, examples, config)

    assert len(requests) == 10
    assert [request.generation_id for request in requests] == [
        f"sampled:glyph:glyph:base:sampled:{example.example_id}:completion-{index}"
        for example in examples
        for index in range(5)
    ]
    assert [request.decoding["completion_index"] for request in requests] == [
        index for _ in examples for index in range(5)
    ]
    assert all(request.decoding["n"] == 1 for request in requests)
    assert all(request.decoding["temperature"] == 0.7 for request in requests)
    assert all(request.decoding["top_p"] == 0.95 for request in requests)
    assert requests == build_generation_requests(cell, examples, config)

    outputs = FakeGenerationBackend().generate(requests)
    records = records_from_outputs(cell, examples, requests, outputs, ("parent" * 8,))
    assert len(records) == 10
    assert [record.generation_id for record in records] == [
        request.generation_id for request in requests
    ]
    assert [record.gold_answer for record in records] == ["2"] * 5 + ["4"] * 5


def test_vllm_batching_ignores_per_completion_provenance_not_sampling_settings(config):
    cell = EvaluationCell("sampled", "glyph", "glyph", None, "sampled")
    examples = (
        DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1"),
        DatasetExample("gsm8k", "test", "two", "2 + 2", "4", "question-2"),
    )

    parameters = _vllm_sampling_parameters(build_generation_requests(cell, examples, config))

    assert parameters == {
        "seed": 42,
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
    }
