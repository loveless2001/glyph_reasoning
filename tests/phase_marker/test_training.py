from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

import pytest

import phase_marker.training as training
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION
from phase_marker.training import (
    TruncatedAnswerError,
    _build_parser,
    build_lora_config,
    build_run_manifest,
    build_training_arguments,
    main,
    tokenize_assistant_only,
    verify_confirmatory_output,
)


EXAMPLE = {
    "example_id": "example-1",
    "messages": [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": "Reasoning.\nFinal answer: 5"},
    ],
}


class FaithfulChatTokenizer:
    """Small chat-template tokenizer with distinct structural token ids."""

    def __init__(self) -> None:
        self.name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        self.init_kwargs = {"_commit_hash": "a09a35458c702b33eeacc393d103063234e8bc28"}
        self._tokens = {
            "<bos>": 1,
            "<|user|>": 2,
            "<|assistant|>": 3,
            "<|end|>": 4,
        }

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert not tokenize
        rendered = "<bos>" + "".join(
            f"<|{message['role']}|>{message['content']}<|end|>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        pieces = re.findall(r"<[^>]+>|.", value, flags=re.DOTALL)
        result: list[int] = []
        for piece in pieces:
            if piece not in self._tokens:
                self._tokens[piece] = len(self._tokens) + 1
            result.append(self._tokens[piece])
        return result


@pytest.fixture
def fake_chat_tokenizer() -> FaithfulChatTokenizer:
    return FaithfulChatTokenizer()


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


def write_materialized_arm(
    root: Path,
    config: ExperimentConfig,
    *,
    arm: str = "glyph",
    parent_hash: str = "f" * 64,
) -> tuple[Path, Path, dict[str, object]]:
    data = root / f"{arm}.jsonl"
    data.write_text(canonical_json(EXAMPLE) + "\n", encoding="utf-8")
    row_hashes = [sha256_json(EXAMPLE)]
    metadata = {
        "row_hashes": row_hashes,
        "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
        "parent_split_hash": parent_hash,
    }
    config_hash = sha256_json(asdict(config))
    artifact_id = sha256_json(
        {
            "arm": arm,
            "config_hash": config_hash,
            "parent_split_hash": parent_hash,
            "row_hashes": row_hashes,
            "metadata": metadata,
        }
    )
    payload: dict[str, object] = {
        "artifact_id": artifact_id,
        "kind": "phase_marker_training_data",
        "config_hash": config_hash,
        "parent_hashes": [parent_hash],
        "row_count": 1,
        "metadata": metadata,
    }
    manifest_path = root / f"{arm}.manifest.json"
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return data, manifest_path, payload


def test_user_tokens_are_masked(fake_chat_tokenizer):
    encoded = tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=128)

    boundary = encoded["assistant_start"]
    assistant_end = len(encoded["input_ids"]) - 1
    assert set(encoded["labels"][:boundary]) == {-100}
    assert encoded["labels"][boundary:assistant_end] == encoded["input_ids"][
        boundary:assistant_end
    ]
    assert boundary == len(
        fake_chat_tokenizer.encode(
            fake_chat_tokenizer.apply_chat_template(
                EXAMPLE["messages"][:1], tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )
    )


def test_assistant_end_template_tokens_are_masked(fake_chat_tokenizer):
    encoded = tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=128)

    assistant_end = len(encoded["input_ids"]) - 1
    assert encoded["input_ids"][assistant_end:] == [
        fake_chat_tokenizer._tokens["<|end|>"]
    ]
    assert encoded["labels"][assistant_end:] == [-100]


def test_tokenization_rejects_a_truncated_final_answer(fake_chat_tokenizer):
    with pytest.raises(TruncatedAnswerError, match="example-1"):
        tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=20)


def test_user_final_span_cannot_hide_truncated_assistant_answer(fake_chat_tokenizer):
    duplicated = {
        "example_id": "duplicate-final-span",
        "messages": [
            {"role": "user", "content": "Repeat Final answer: 5 exactly."},
            {"role": "assistant", "content": "Reasoning.\nFinal answer: 5"},
        ],
    }
    prefix = fake_chat_tokenizer.apply_chat_template(
        duplicated["messages"][:1], tokenize=False, add_generation_prompt=True
    )
    cutoff = len(fake_chat_tokenizer.encode(prefix, add_special_tokens=False)) + len(
        fake_chat_tokenizer.encode(
            "Reasoning.\nFinal answer: ", add_special_tokens=False
        )
    )

    with pytest.raises(TruncatedAnswerError, match="duplicate-final-span"):
        tokenize_assistant_only(duplicated, fake_chat_tokenizer, max_length=cutoff)


def test_lora_and_training_arguments_are_arm_invariant(config):
    left = build_training_arguments(config, "glyph", 101, Path("/tmp/glyph"))
    right = build_training_arguments(config, "semantic", 101, Path("/tmp/semantic"))

    assert left.learning_rate == right.learning_rate == 2e-4
    assert left.gradient_accumulation_steps * left.per_device_train_batch_size == 16
    assert left.lr_scheduler_type.value == right.lr_scheduler_type.value == "cosine"
    assert left.num_train_epochs == right.num_train_epochs == 1
    assert left.bf16 is right.bf16 is True
    assert left.save_steps == right.save_steps == 100
    assert left.save_total_limit is right.save_total_limit is None
    lora = build_lora_config(config)
    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.lora_dropout == 0.05
    assert set(lora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_run_manifest_binds_dataset_config_environment_and_checkpoints(config, tmp_path):
    data, _, materialization = write_materialized_arm(tmp_path, config)
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"smoke-adapter")
    validation_checkpoint = output_dir / "checkpoint-200"
    validation_checkpoint.mkdir()
    (validation_checkpoint / "adapter_model.safetensors").write_bytes(
        b"validation-adapter"
    )

    manifest = build_run_manifest(
        config,
        "glyph",
        101,
        data,
        ["--arm", "glyph"],
        output_dir=output_dir,
    )

    assert manifest["arm"] == "glyph"
    assert manifest["seed"] == 101
    assert manifest["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert manifest["dataset_hash"] == sha256_json(data.read_bytes().hex())
    assert manifest["config_hash"] == sha256_json(asdict(config))
    assert manifest["data_artifact_id"] == materialization["artifact_id"]
    assert manifest["parent_hashes"] == [materialization["artifact_id"]]
    assert manifest["data_parent_hashes"] == ["f" * 64]
    assert manifest["arguments"] == ["--arm", "glyph"]
    assert {"torch", "transformers", "peft"} <= set(manifest["environment"])
    assert manifest["environment"]["pytorch"] == manifest["environment"]["torch"]
    assert "cuda" in manifest["environment"]
    assert manifest["checkpoints"] == [
        {
            "path": "checkpoint-100",
            "hash": "fd1a47870fcd9b229e1b5baa194f9b4ab62d45b542e6991be8b59bb10deeebad",
        },
        {
            "path": "checkpoint-200",
            "hash": "ba1971644fbfca18cf2aff0ad3948323e1f715fb2aed7609d91ca4f17a4518aa",
        },
    ]


def test_materialization_manifest_rejects_mislabeled_arm(config, tmp_path):
    data, _, _ = write_materialized_arm(tmp_path, config, arm="glyph")

    with pytest.raises(ValueError, match="arm"):
        build_run_manifest(config, "semantic", 101, data, [])


def test_materialization_manifest_rejects_changed_jsonl_rows(config, tmp_path):
    data, _, _ = write_materialized_arm(tmp_path, config)
    data.write_text(
        canonical_json(
            {
                **EXAMPLE,
                "messages": [
                    EXAMPLE["messages"][0],
                    {"role": "assistant", "content": "Final answer: 6"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row hashes"):
        build_run_manifest(config, "glyph", 101, data, [])


def test_materialization_manifest_rejects_parent_mismatch(config, tmp_path):
    data, manifest_path, payload = write_materialized_arm(tmp_path, config)
    payload["parent_hashes"] = ["e" * 64]
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parent"):
        build_run_manifest(config, "glyph", 101, data, [])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("kind", "kind"),
        ("config", "config hash"),
        ("tokenizer", "tokenizer revision"),
        ("artifact", "artifact id"),
    ],
)
def test_materialization_manifest_rejects_invalid_identity_fields(
    config, tmp_path, mutation, message
):
    data, manifest_path, payload = write_materialized_arm(tmp_path, config)
    if mutation == "kind":
        payload["kind"] = "untrusted_data"
    elif mutation == "config":
        payload["config_hash"] = "0" * 64
    elif mutation == "tokenizer":
        payload["metadata"]["tokenizer_revision"] = "0" * 40
    else:
        payload["artifact_id"] = "0" * 64
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_run_manifest(config, "glyph", 101, data, [])


def test_confirmatory_output_hash_mismatch_fails(config, tmp_path):
    output_dir = tmp_path / "confirmatory"
    output_dir.mkdir()
    (output_dir / "run-manifest.json").write_text(
        canonical_json(
            {
                "config_hash": sha256_json(asdict(config)),
                "output_hash": "e" * 64,
                "seed": 101,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output hash"):
        verify_confirmatory_output(output_dir, config, 101)


def test_confirmatory_checks_the_cli_manifest_path(config, tmp_path):
    output_dir = tmp_path / "confirmatory"
    output_dir.mkdir()
    manifest_path = tmp_path / "requested-manifest.json"
    manifest_path.write_text(
        canonical_json({"config_hash": "f" * 64, "seed": 101}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="config hash"):
        verify_confirmatory_output(output_dir, config, 101, manifest_path)


def test_train_cli_requires_all_lineage_arguments():
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", "configs/phase-marker-qwen25-7b.toml"])

    assert error.value.code == 2


def test_train_cli_rejects_a_max_length_override():
    parser = _build_parser()
    arguments = [
        "train",
        "--config",
        "configs/phase-marker-qwen25-7b.toml",
        "--arm",
        "glyph",
        "--seed",
        "101",
        "--data",
        "/tmp/glyph.jsonl",
        "--output-dir",
        "/tmp/output",
        "--manifest",
        "/tmp/run-manifest.json",
        "--max-length",
        "1024",
    ]

    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)

    assert error.value.code == 2


def test_training_arguments_reject_multi_process_world_size(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="world size must be exactly 1"):
        build_training_arguments(config, "glyph", 101, tmp_path / "run")


def test_train_preflight_rejects_multiple_visible_cuda_devices(
    config, tmp_path, monkeypatch
):
    data, _, _ = write_materialized_arm(tmp_path, config)
    arguments = _build_parser().parse_args(
        [
            "train",
            "--config",
            "configs/phase-marker-qwen25-7b.toml",
            "--arm",
            "glyph",
            "--seed",
            "101",
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path / "output"),
            "--manifest",
            str(tmp_path / "run-manifest.json"),
        ]
    )
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "is_torch_bf16_gpu_available", lambda: True)
    monkeypatch.setattr(training.torch.cuda, "device_count", lambda: 2)

    def fail_at_model_boundary(_model_id):
        raise AssertionError("multi-device preflight crossed the model boundary")

    monkeypatch.setattr(training, "_cached_model_snapshot", fail_at_model_boundary)

    with pytest.raises(ValueError, match="exactly one visible CUDA device"):
        training._train(arguments, [])
