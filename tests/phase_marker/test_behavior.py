from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from phase_marker.behavior import (
    _vllm_sampling_parameters,
    _validate_production_request,
    build_provenance_envelope,
    EvaluationCell,
    FakeGenerationBackend,
    GenerationOutput,
    GenerationRequest,
    VLLMGenerationBackend,
    build_generation_requests,
    build_behavior_matrix,
    records_from_outputs,
    serialize_generation_record,
    main as behavior_main,
)
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.schema import ArtifactManifest, GenerationRecord
from phase_marker.splits import DatasetExample
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


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
            "generation-1",
            "prompt 1",
            (11, 12),
            64,
            {"seed": 42, "adapter_seed": 42, "checkpoint": "ckpt"},
        ),
        GenerationRequest(
            "generation-2",
            "prompt 2",
            (21, 22),
            64,
            {"seed": 42, "adapter_seed": 42, "checkpoint": "ckpt"},
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



def test_records_reject_missing_duplicate_or_reordered_output_ids():
    cell = EvaluationCell("primary", "glyph", "glyph", None, "greedy")
    example = DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1")
    request = GenerationRequest(
        "generation-1",
        "prompt",
        (1,),
        64,
        {"seed": 42, "adapter_seed": 42, "checkpoint": "ckpt"},
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

    requests = build_generation_requests(
        cell,
        examples,
        config,
        checkpoint="fake://glyph",
        adapter_seed=42,
        fake=True,
    )

    assert len(requests) == 10
    assert [request.generation_id for request in requests] == [
        f"sampled:glyph:glyph:base:sampled:{example.example_id}:completion-{index}"
        for example in examples
        for index in range(5)
    ]
    assert [request.decoding["completion_index"] for request in requests] == [
        index for _ in examples for index in range(5)
    ]
    assert [request.decoding["seed"] for request in requests] == [
        42 + index for _ in examples for index in range(5)
    ]
    assert all(request.decoding["n"] == 1 for request in requests)
    assert all(request.decoding["temperature"] == 0.7 for request in requests)
    assert all(request.decoding["top_p"] == 0.95 for request in requests)
    assert requests == build_generation_requests(
        cell,
        examples,
        config,
        checkpoint="fake://glyph",
        adapter_seed=42,
        fake=True,
    )

    sampling_parameters = [_vllm_sampling_parameters((request,)) for request in requests]
    assert [parameters["seed"] for parameters in sampling_parameters] == [
        42 + index for _ in examples for index in range(5)
    ]

    outputs = FakeGenerationBackend().generate(requests)
    records = records_from_outputs(cell, examples, requests, outputs, ("parent" * 8,))
    assert len(records) == 10
    assert [record.generation_id for record in records] == [
        request.generation_id for request in requests
    ]
    assert [record.gold_answer for record in records] == ["2"] * 5 + ["4"] * 5
    assert [record.seed for record in records] == [42] * 10


def test_vllm_sampling_parameters_retain_the_request_sample_seed(config):
    cell = EvaluationCell("sampled", "glyph", "glyph", None, "sampled")
    examples = (
        DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1"),
        DatasetExample("gsm8k", "test", "two", "2 + 2", "4", "question-2"),
    )

    parameters = _vllm_sampling_parameters(
        (
            build_generation_requests(
                cell,
                examples,
                config,
                checkpoint="fake://glyph",
                adapter_seed=42,
                fake=True,
            )[0],
        )
    )

    assert parameters == {
        "seed": 42,
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
    }


def test_production_request_construction_rejects_fake_tokenizer_or_checkpoint(config, split_manifest):
    cell = EvaluationCell("primary", "glyph", "glyph", None, "greedy")
    examples = (DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1"),)
    with pytest.raises(ValueError, match="tokenizer revision"):
        build_generation_requests(
            cell,
            examples,
            config,
            checkpoint="/checkpoints/glyph",
            tokenize=lambda _: (1,),
            tokenizer_revision="wrong-revision",
            split_manifest=split_manifest,
            adapter_seed=101,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        build_generation_requests(
            cell,
            examples,
            config,
            checkpoint="unconfigured://checkpoint",
            tokenize=lambda _: (1,),
            tokenizer_revision=QWEN25_7B_TOKENIZER_REVISION,
            split_manifest=split_manifest,
            adapter_seed=101,
        )


def test_production_request_construction_requires_explicit_adapter_seed(config, split_manifest):
    split_manifest = replace(split_manifest, config_hash=sha256_json(asdict(config)))
    cell = EvaluationCell("primary", "glyph", "glyph", None, "greedy")
    example = DatasetExample("gsm8k", "test", "one", "1 + 1", "2", "question-1")
    with pytest.raises(ValueError, match="adapter seed"):
        build_generation_requests(
            cell,
            (example,),
            config,
            checkpoint="/checkpoints/glyph",
            tokenize=lambda _: (1,),
            tokenizer_revision=QWEN25_7B_TOKENIZER_REVISION,
            split_manifest=split_manifest,
        )


def test_vllm_preflight_rejects_fake_or_unpinned_requests_before_import():
    request = GenerationRequest(
        "generation-1",
        "prompt",
        (1,),
        64,
        {
            "seed": 42,
            "adapter_seed": 42,
            "checkpoint": "fake://glyph",
            "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
            "run_kind": "production",
        },
    )
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_production_request(request)
    with pytest.raises(ValueError, match="tokenizer revision"):
        _validate_production_request(
            replace(
                request,
                decoding={
                    **request.decoding,
                    "checkpoint": "/checkpoints/glyph",
                    "tokenizer_revision": "wrong-revision",
                },
            )
        )


def test_vllm_backend_binds_every_request_checkpoint_to_its_model_before_import():
    with pytest.raises(ValueError, match="real checkpoint"):
        VLLMGenerationBackend("fake://glyph", adapter_seed=42)
    with pytest.raises(ValueError, match="real checkpoint"):
        VLLMGenerationBackend("unconfigured://checkpoint", adapter_seed=42)
    with pytest.raises(ValueError, match="adapter seed"):
        VLLMGenerationBackend("/checkpoints/glyph-final")

    backend = VLLMGenerationBackend("/checkpoints/glyph-final", adapter_seed=42)
    shared = {
        "seed": 42,
        "adapter_seed": 42,
        "run_kind": "production",
        "config_hash": "a" * 64,
        "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
        "split_artifact_id": "b" * 64,
        "split_parent_hashes": ("c" * 64,),
    }
    matching = GenerationRequest(
        "matching",
        "prompt",
        (1,),
        64,
        {**shared, "checkpoint": "/checkpoints/glyph-final"},
    )
    mismatched = GenerationRequest(
        "mismatched",
        "prompt",
        (1,),
        64,
        {**shared, "checkpoint": "/checkpoints/other-final"},
    )
    with pytest.raises(ValueError, match="backend model"):
        backend.generate((matching, mismatched))
    wrong_seed = replace(
        matching,
        generation_id="wrong-seed",
        decoding={**matching.decoding, "adapter_seed": 101},
    )
    with pytest.raises(ValueError, match="backend adapter seed"):
        backend.generate((matching, wrong_seed))


def test_provenance_envelope_rejects_empty_or_mismatched_production_lineage(
    config, split_manifest
):
    split_manifest = replace(
        split_manifest, config_hash=sha256_json(asdict(config))
    )
    record = GenerationRecord(
        generation_id="generation-1",
        source="gsm8k",
        question_hash="question-1",
        gold_answer="2",
        training_arm="glyph",
        seed=42,
        checkpoint="/checkpoints/glyph",
        prompt_condition="glyph",
        prompt_hash="prompt-1",
        raw_prompt="prompt",
        raw_completion="Final answer: 2",
        prompt_token_ids=(1,),
        completion_token_ids=(2,),
        decoding={
            "run_kind": "production",
            "adapter_seed": 42,
            "config_hash": sha256_json(asdict(config)),
            "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
            "split_artifact_id": split_manifest.artifact_id,
            "split_parent_hashes": split_manifest.parent_hashes,
        },
        parent_hashes=(split_manifest.artifact_id, *split_manifest.parent_hashes),
    )

    provenance = build_provenance_envelope(record, config, split_manifest)
    persisted = serialize_generation_record(record, provenance)
    assert persisted["provenance"] == asdict(provenance)
    assert persisted["prompt_token_count"] == 1
    assert persisted["completion_token_count"] == 1
    with pytest.raises(ValueError, match="checkpoint"):
        serialize_generation_record(
            record, replace(provenance, checkpoint="/checkpoints/other")
        )

    with pytest.raises(ValueError, match="parent"):
        build_provenance_envelope(replace(record, parent_hashes=()), config, split_manifest)
    with pytest.raises(ValueError, match="config hash"):
        build_provenance_envelope(
            replace(record, decoding={**record.decoding, "config_hash": "wrong"}),
            config,
            split_manifest,
        )
    with pytest.raises(ValueError, match="adapter seed"):
        build_provenance_envelope(replace(record, seed=101), config, split_manifest)


def test_run_cli_tiny_fixture_emits_versioned_plumbing_manifest(
    tmp_path: Path, config: ExperimentConfig
) -> None:
    config_path = Path("configs/phase-marker-qwen25-7b.toml")
    config_hash = sha256_json(asdict(config))
    split_path = tmp_path / "split-manifest.json"
    split_payload = {
        "artifact_id": "a" * 64,
        "config_hash": config_hash,
        "overlap_count": 0,
        "source_counts": {"test": {"gsm8k": 1}},
    }
    split_path.write_text(canonical_json(split_payload) + "\n", encoding="utf-8")
    examples_path = tmp_path / "examples.jsonl"
    example = {
        "source": "gsm8k",
        "split": "test",
        "example_id": "fixture-1",
        "question": "What is 1 + 1?",
        "answer": "2",
        "question_hash": "q" * 64,
    }
    examples_path.write_text(canonical_json(example) + "\n", encoding="utf-8")
    selections: list[str] = []
    for arm in config.arms:
        payload = {
            "schema_version": 1,
            "kind": "phase_marker_checkpoint_selection",
            "config_hash": config_hash,
            "run_kind": "pilot",
            "arm": arm,
            "seed": 42,
            "selected_on": "validation",
            "checkpoint_path": f"/fixture/{arm}",
            "training_manifest_hash": sha256_json({"training": arm}),
            "materialization_artifact_id": sha256_json({"materialization": arm}),
            "completed": True,
        }
        payload["artifact_id"] = sha256_json(payload)
        selection_path = tmp_path / f"{arm}.selection.json"
        selection_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        selections.append(str(selection_path))
    output_root = tmp_path / "behavior-output"

    assert behavior_main(
        [
            "run",
            "--config",
            str(config_path),
            "--kind",
            "pilot",
            "--seeds",
            "42",
            "--split-manifest",
            str(split_path),
            "--examples",
            str(examples_path),
            "--checkpoint-manifests",
            *selections,
            "--backend",
            "tiny-fixture",
            "--allow-test-backend",
            "--output-root",
            str(output_root),
        ]
    ) == 0

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "phase_marker_behavior_generations"
    assert manifest["evidence_scope"] == "plumbing_only"
    assert manifest["backend"] == "tiny-fixture"
    assert manifest["completed"] is True
    assert manifest["row_count"] > 0
    assert (output_root / "records.jsonl").is_file()


def test_run_cli_rejects_test_backend_without_explicit_flag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="allow-test-backend"):
        behavior_main(
            [
                "run",
                "--config",
                "configs/phase-marker-qwen25-7b.toml",
                "--kind",
                "pilot",
                "--seeds",
                "42",
                "--split-manifest",
                str(tmp_path / "missing-split.json"),
                "--examples",
                str(tmp_path / "missing-examples.jsonl"),
                "--checkpoint-manifests",
                str(tmp_path / "missing-checkpoint.json"),
                "--backend",
                "tiny-fixture",
                "--output-root",
                str(tmp_path / "must-not-exist"),
            ]
        )
    assert not (tmp_path / "must-not-exist").exists()
