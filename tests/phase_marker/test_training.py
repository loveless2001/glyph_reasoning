from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.training import (
    TruncatedAnswerError,
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
        self._tokens = {"<bos>": 1, "<|user|>": 2, "<|assistant|>": 3}

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert not tokenize
        rendered = "<bos>" + "".join(
            f"<|{message['role']}|>{message['content']}" for message in messages
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


def test_user_tokens_are_masked(fake_chat_tokenizer):
    encoded = tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=128)

    boundary = encoded["assistant_start"]
    assert set(encoded["labels"][:boundary]) == {-100}
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert boundary == len(
        fake_chat_tokenizer.encode(
            fake_chat_tokenizer.apply_chat_template(
                EXAMPLE["messages"][:1], tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )
    )


def test_tokenization_rejects_a_truncated_final_answer(fake_chat_tokenizer):
    with pytest.raises(TruncatedAnswerError, match="example-1"):
        tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=20)


def test_lora_and_training_arguments_are_arm_invariant(config):
    left = build_training_arguments(config, "glyph", 101, Path("/tmp/glyph"))
    right = build_training_arguments(config, "semantic", 101, Path("/tmp/semantic"))

    assert left.learning_rate == right.learning_rate == 2e-4
    assert left.gradient_accumulation_steps * left.per_device_train_batch_size == 16
    assert left.lr_scheduler_type.value == right.lr_scheduler_type.value == "cosine"
    assert left.num_train_epochs == right.num_train_epochs == 1
    assert left.bf16 is right.bf16 is True
    assert left.save_steps == right.save_steps == 100
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
    data = tmp_path / "glyph.jsonl"
    data.write_text(canonical_json(EXAMPLE) + "\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"smoke-adapter")

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
    assert manifest["arguments"] == ["--arm", "glyph"]
    assert {"torch", "transformers", "peft"} <= set(manifest["environment"])
    assert manifest["environment"]["pytorch"] == manifest["environment"]["torch"]
    assert "cuda" in manifest["environment"]
    assert manifest["checkpoints"] == [
        {
            "path": "checkpoint-100",
            "hash": "fd1a47870fcd9b229e1b5baa194f9b4ab62d45b542e6991be8b59bb10deeebad",
        }
    ]


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
