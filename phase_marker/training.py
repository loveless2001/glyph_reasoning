"""Assistant-only LoRA training with immutable run lineage."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available

from phase_marker.config import ExperimentConfig, REQUIRED_FINAL_DELIMITER
from phase_marker.io import canonical_json, read_jsonl, sha256_json
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MAX_SEQUENCE_LENGTH = 2048


class TruncatedAnswerError(ValueError):
    """A max-length cutoff would remove the final delimiter or answer."""


def tokenize_assistant_only(
    example: Mapping[str, object], tokenizer: object, max_length: int
) -> dict[str, list[int]]:
    """Render one chat and mask every token before the assistant response."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(
            "example messages must contain a user turn and an assistant turn"
        )
    assistant = messages[-1]
    if not isinstance(assistant, Mapping) or assistant.get("role") != "assistant":
        raise ValueError("the final message must be the assistant response")
    assistant_content = assistant.get("content")
    if not isinstance(assistant_content, str):
        raise ValueError("assistant content must be a string")
    if assistant_content.count(REQUIRED_FINAL_DELIMITER) != 1:
        raise ValueError("assistant response must contain exactly one final-answer delimiter")
    final_span = assistant_content[assistant_content.index(REQUIRED_FINAL_DELIMITER) :]
    if not final_span.removeprefix(REQUIRED_FINAL_DELIMITER).strip():
        raise ValueError("assistant response must contain an answer after the final delimiter")

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(apply_template) or not callable(encode):
        raise TypeError("tokenizer must provide apply_chat_template and encode")

    rendered = apply_template(messages, tokenize=False, add_generation_prompt=False)
    rendered_prefix = apply_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    empty_assistant = dict(assistant)
    empty_assistant["content"] = ""
    rendered_empty_assistant = apply_template(
        [*messages[:-1], empty_assistant],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not all(
        isinstance(value, str)
        for value in (rendered, rendered_prefix, rendered_empty_assistant)
    ):
        raise TypeError("chat template must render text when tokenize=False")
    input_ids = _integer_ids(encode(rendered, add_special_tokens=False))
    prefix_ids = _integer_ids(encode(rendered_prefix, add_special_tokens=False))
    empty_assistant_ids = _integer_ids(
        encode(rendered_empty_assistant, add_special_tokens=False)
    )
    if input_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "assistant generation prefix is not a token prefix of the rendered chat"
        )
    suffix_width = _common_suffix_width(
        input_ids,
        empty_assistant_ids,
        minimum_prefix_width=len(prefix_ids),
    )
    if suffix_width < 1:
        raise ValueError("chat template must include an assistant end-of-turn suffix")
    assistant_start = len(prefix_ids)
    assistant_end = len(input_ids) - suffix_width
    if assistant_end <= assistant_start:
        raise ValueError("assistant content must tokenize to at least one token")

    assistant_ids = input_ids[assistant_start:assistant_end]
    final_span_ids = _integer_ids(encode(final_span, add_special_tokens=False))
    final_span_offsets = _window_offsets(assistant_ids, final_span_ids)
    if not final_span_offsets:
        raise ValueError("final-answer token span is not inside assistant content")
    final_span_end = assistant_start + final_span_offsets[-1] + len(final_span_ids)
    labels = [-100] * len(input_ids)
    labels[assistant_start:assistant_end] = input_ids[assistant_start:assistant_end]
    truncated_input_ids = input_ids[:max_length]
    truncated_labels = labels[:max_length]
    example_id = str(example.get("example_id", "<unknown>"))
    if final_span_end > max_length:
        raise TruncatedAnswerError(
            f"{example_id}: max_length={max_length} truncates the final delimiter or answer"
        )

    # assistant_start is intentionally included for mask audits and removed before Trainer input.
    return {
        "input_ids": truncated_input_ids,
        "attention_mask": [1] * len(truncated_input_ids),
        "labels": truncated_labels,
        "assistant_start": assistant_start,  # type: ignore[dict-item]
    }


def build_lora_config(config: ExperimentConfig) -> LoraConfig:
    """Return the arm-invariant LoRA configuration for every experiment run."""
    _require_supported_model(config)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )


def build_training_arguments(
    config: ExperimentConfig, arm: str, seed: int, output_dir: Path
) -> TrainingArguments:
    """Return the fixed matched-training arguments, varying only lineage fields."""
    _validate_run_identity(config, arm, seed)
    _require_single_process_world_size()
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=True,
        fp16=False,
        use_cpu=not is_torch_bf16_gpu_available(),
        logging_steps=5,
        logging_first_step=True,
        save_steps=100,
        save_total_limit=None,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=20,
        gradient_checkpointing=True,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )


def build_run_manifest(
    config: ExperimentConfig,
    arm: str,
    seed: int,
    data: Path,
    arguments: Sequence[str],
    *,
    output_dir: Path | None = None,
    model_revision: str = QWEN25_7B_TOKENIZER_REVISION,
) -> dict[str, object]:
    """Bind fixed configuration, JSONL bytes, runtime versions, and checkpoints."""
    _validate_run_identity(config, arm, seed)
    if not data.is_file():
        raise FileNotFoundError(data)
    if model_revision != QWEN25_7B_TOKENIZER_REVISION:
        raise ValueError("resolved model revision does not match the pinned Qwen revision")
    materialization = _validate_materialization_manifest(data, config, arm)
    checkpoints = _checkpoint_lineage(output_dir) if output_dir is not None else []
    environment = {
        "torch": torch.__version__,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
    }
    manifest: dict[str, object] = {
        "kind": "phase_marker_training_run",
        "arm": arm,
        "seed": seed,
        "model_id": config.model_id,
        "model_revision": model_revision,
        "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
        "config_hash": sha256_json(asdict(config)),
        "dataset_path": str(data),
        "dataset_hash": sha256_json(data.read_bytes().hex()),
        "data_artifact_id": materialization["artifact_id"],
        "parent_hashes": [materialization["artifact_id"]],
        "data_parent_hashes": list(materialization["parent_hashes"]),
        "arguments": list(arguments),
        "environment": environment,
        "checkpoints": checkpoints,
        "saved_artifacts": ["adapter", "tokenizer", "trainer_state"],
    }
    if output_dir is not None:
        manifest["output_hash"] = _directory_hash(output_dir)
    return manifest


def verify_confirmatory_output(
    output_dir: Path,
    config: ExperimentConfig,
    seed: int,
    manifest_path: Path | None = None,
) -> None:
    """Reject confirmatory output whose recorded lineage or bytes do not match."""
    if seed not in config.confirmatory_seeds:
        return
    candidates: list[Path] = []
    if manifest_path is not None and manifest_path.is_file():
        candidates.append(manifest_path)
    canonical_manifest = output_dir / "run-manifest.json"
    if canonical_manifest.is_file() and canonical_manifest not in candidates:
        candidates.append(canonical_manifest)

    if output_dir.exists() and any(output_dir.iterdir()) and not candidates:
        raise ValueError("confirmatory output directory has no run manifest for hash verification")

    expected_config_hash = sha256_json(asdict(config))
    excluded = {candidate.resolve() for candidate in candidates}
    for candidate in candidates:
        payload = _read_manifest(candidate)
        if payload.get("config_hash") != expected_config_hash:
            raise ValueError("confirmatory output directory has a different config hash")
        if payload.get("seed") != seed:
            raise ValueError("confirmatory output directory has a different seed")
        output_hash = payload.get("output_hash")
        if output_hash is not None and output_hash != _directory_hash(
            output_dir, excluded=excluded
        ):
            raise ValueError("confirmatory output directory has an output hash mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "tokenize-smoke":
        return _tokenize_smoke(
            arguments.config, arguments.data, arguments.limit, arguments.max_length
        )
    if arguments.command == "train":
        return _train(arguments, list(argv) if argv is not None else sys.argv[1:])
    parser.error("a command is required")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser(
        "tokenize-smoke", help="tokenizer-only offline smoke gate"
    )
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--data", type=Path, required=True)
    smoke.add_argument("--limit", type=int, required=True)
    smoke.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)

    train = commands.add_parser("train", help="train one matched LoRA adapter")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--arm", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    return parser


def _tokenize_smoke(config_path: Path, data: Path, limit: int, max_length: int) -> int:
    if limit < 1:
        raise ValueError("smoke limit must be positive")
    config = ExperimentConfig.load(config_path)
    from transformers import AutoTokenizer

    snapshot = _cached_model_snapshot(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    _require_resolved_revision(tokenizer, "tokenizer")
    rows = []
    for index, row in enumerate(read_jsonl(data), start=1):
        if len(rows) == limit:
            break
        labeled = dict(row)
        labeled.setdefault("example_id", f"smoke-{index}")
        rows.append(tokenize_assistant_only(labeled, tokenizer, max_length))
    if len(rows) != limit:
        raise ValueError(
            f"SMOKE requested {limit} examples but data contains only {len(rows)}"
        )
    if not all(
        row["assistant_start"] > 0
        and set(row["labels"][: int(row["assistant_start"])]) == {-100}
        for row in rows
    ):
        raise AssertionError("SMOKE assistant-only label mask check failed")
    print(
        f"SMOKE tokenize-only: {len(rows)}/{limit} examples; "
        "user labels masked; final delimiters retained; model weights not loaded"
    )
    return 0


def _train(arguments: argparse.Namespace, command_arguments: list[str]) -> int:
    config = ExperimentConfig.load(arguments.config)
    _validate_run_identity(config, arguments.arm, arguments.seed)
    _validate_materialization_manifest(arguments.data, config, arguments.arm)
    verify_confirmatory_output(
        arguments.output_dir, config, arguments.seed, arguments.manifest
    )
    if arguments.manifest.exists() or (
        arguments.output_dir / "run-manifest.json"
    ).exists():
        raise FileExistsError("refusing to overwrite an immutable completed run manifest")
    _require_single_process_world_size()
    if not torch.cuda.is_available() or not is_torch_bf16_gpu_available():
        raise RuntimeError("LoRA training requires a BF16-capable CUDA device")
    if torch.cuda.device_count() != 1:
        raise ValueError(
            "LoRA training requires exactly one visible CUDA device for effective batch size 16"
        )
    training_arguments = build_training_arguments(
        config, arguments.arm, arguments.seed, arguments.output_dir
    )

    from datasets import Dataset
    from peft import get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
    )

    snapshot = _cached_model_snapshot(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    _require_resolved_revision(tokenizer, "tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_rows: list[dict[str, list[int]]] = []
    for index, row in enumerate(read_jsonl(arguments.data), start=1):
        labeled = dict(row)
        labeled.setdefault("example_id", f"row-{index}")
        encoded = tokenize_assistant_only(labeled, tokenizer, MAX_SEQUENCE_LENGTH)
        tokenized_rows.append(
            {key: value for key, value in encoded.items() if key != "assistant_start"}
        )
    if not tokenized_rows:
        raise ValueError("training data must contain at least one example")

    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    _require_resolved_revision(model, "model")
    model = get_peft_model(model, build_lora_config(config))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=Dataset.from_list(tokenized_rows),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )
    trainer.train()
    trainer.save_model(str(arguments.output_dir))
    tokenizer.save_pretrained(arguments.output_dir)
    trainer.save_state()

    manifest = build_run_manifest(
        config,
        arguments.arm,
        arguments.seed,
        arguments.data,
        command_arguments,
        output_dir=arguments.output_dir,
        model_revision=QWEN25_7B_TOKENIZER_REVISION,
    )
    canonical_path = arguments.output_dir / "run-manifest.json"
    _write_manifest_immutable(canonical_path, manifest)
    if arguments.manifest.resolve() != canonical_path.resolve():
        _write_manifest_immutable(arguments.manifest, manifest)
    return 0


def _integer_ids(value: object) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("tokenizer.encode must return a sequence of integers")
    result = list(value)
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in result
    ):
        raise TypeError("tokenizer.encode must return a sequence of integers")
    return result


def _window_offsets(values: Sequence[int], window: Sequence[int]) -> list[int]:
    if not window or len(window) > len(values):
        return []
    width = len(window)
    return [
        index
        for index in range(len(values) - width + 1)
        if list(values[index : index + width]) == list(window)
    ]


def _common_suffix_width(
    left: Sequence[int], right: Sequence[int], *, minimum_prefix_width: int
) -> int:
    maximum = min(
        len(left) - minimum_prefix_width,
        len(right) - minimum_prefix_width,
    )
    width = 0
    while width < maximum and left[-(width + 1)] == right[-(width + 1)]:
        width += 1
    return width


def _require_supported_model(config: ExperimentConfig) -> None:
    if config.model_id != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError(
            "LoRA configuration is pinned to Qwen/Qwen2.5-7B-Instruct"
        )


def _validate_run_identity(config: ExperimentConfig, arm: str, seed: int) -> None:
    _require_supported_model(config)
    if arm not in config.arms:
        raise ValueError(f"unknown configured training arm: {arm}")
    if seed not in (config.pilot_seed, *config.confirmatory_seeds):
        raise ValueError(f"seed {seed} is not a configured pilot or confirmatory seed")


def _require_single_process_world_size() -> None:
    observed: list[int] = []
    for variable in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        raw = os.environ.get(variable)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{variable} must be an integer") from error
        if value < 1:
            raise ValueError(f"{variable} must be positive")
        observed.append(value)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        observed.append(torch.distributed.get_world_size())
    if any(value != 1 for value in observed):
        raise ValueError("training world size must be exactly 1 for effective batch size 16")


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hash(path: Path, *, excluded: set[Path] | None = None) -> str:
    exclusions = excluded or set()
    if not path.exists():
        return sha256_json([])
    records = [
        {"path": str(candidate.relative_to(path)), "sha256": _file_hash(candidate)}
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and candidate.resolve() not in exclusions
        and candidate.name != "run-manifest.json"
    ]
    return sha256_json(records)


def _checkpoint_lineage(output_dir: Path) -> list[dict[str, str]]:
    if not output_dir.exists():
        return []
    return [
        {"path": checkpoint.name, "hash": _directory_hash(checkpoint)}
        for checkpoint in sorted(output_dir.glob("checkpoint-*"))
        if checkpoint.is_dir()
    ]


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run manifest: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"run manifest must be a JSON object: {path}")
    return payload


def _validate_materialization_manifest(
    data: Path, config: ExperimentConfig, arm: str
) -> Mapping[str, Any]:
    if not data.is_file():
        raise FileNotFoundError(data)
    if data.stem != arm:
        raise ValueError(
            f"materialized data arm {data.stem!r} does not match requested arm {arm!r}"
        )
    manifest_path = data.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"materialization manifest required beside training data: {manifest_path}"
        )
    payload = _read_manifest(manifest_path)
    if payload.get("kind") != "phase_marker_training_data":
        raise ValueError("materialization manifest has an invalid kind")
    config_hash = sha256_json(asdict(config))
    if payload.get("config_hash") != config_hash:
        raise ValueError("materialization manifest has a different config hash")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("materialization manifest metadata must be an object")
    if metadata.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION:
        raise ValueError("materialization manifest has a different tokenizer revision")
    expected_row_hashes = [sha256_json(row) for row in read_jsonl(data)]
    row_hashes = metadata.get("row_hashes")
    if row_hashes != expected_row_hashes:
        raise ValueError("materialization manifest row hashes do not match the JSONL data")
    if payload.get("row_count") != len(expected_row_hashes):
        raise ValueError("materialization manifest row count does not match the JSONL data")

    parent_hashes = payload.get("parent_hashes")
    parent_split_hash = metadata.get("parent_split_hash")
    if (
        not isinstance(parent_hashes, list)
        or len(parent_hashes) != 1
        or not _is_sha256(parent_hashes[0])
        or parent_split_hash != parent_hashes[0]
    ):
        raise ValueError("materialization manifest parent lineage is inconsistent")
    artifact_id = payload.get("artifact_id")
    expected_artifact_id = sha256_json(
        {
            "arm": arm,
            "config_hash": config_hash,
            "parent_split_hash": parent_split_hash,
            "row_hashes": expected_row_hashes,
            "metadata": dict(metadata),
        }
    )
    if not _is_sha256(artifact_id) or artifact_id != expected_artifact_id:
        raise ValueError("materialization manifest artifact id does not match its arm or metadata")
    return payload


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _write_manifest_immutable(path: Path, manifest: Mapping[str, object]) -> None:
    serialized = canonical_json(dict(manifest)) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite immutable run manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _require_resolved_revision(value: object, kind: str) -> None:
    init_kwargs = getattr(value, "init_kwargs", None)
    candidates: set[str] = set()
    if isinstance(init_kwargs, Mapping):
        commit_hash = init_kwargs.get("_commit_hash")
        if isinstance(commit_hash, str):
            candidates.add(commit_hash)
        for candidate in init_kwargs.values():
            if isinstance(candidate, str):
                candidates.update(_snapshot_revisions(candidate))
    config = getattr(value, "config", None)
    commit_hash = getattr(config, "_commit_hash", None)
    if isinstance(commit_hash, str):
        candidates.add(commit_hash)
    name_or_path = getattr(value, "name_or_path", None)
    if isinstance(name_or_path, str):
        candidates.update(_snapshot_revisions(name_or_path))
    if candidates != {QWEN25_7B_TOKENIZER_REVISION}:
        raise ValueError(
            f"resolved {kind} revision {sorted(candidates)!r} does not match "
            f"{QWEN25_7B_TOKENIZER_REVISION}"
        )


def _cached_model_snapshot(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            model_id,
            revision=QWEN25_7B_TOKENIZER_REVISION,
            local_files_only=True,
        )
    )
    if _snapshot_revisions(str(snapshot)) != {QWEN25_7B_TOKENIZER_REVISION}:
        raise ValueError("cached model snapshot does not match the pinned Qwen revision")
    return snapshot


def _snapshot_revisions(value: str) -> set[str]:
    parts = Path(value).parts
    return {
        parts[index + 1]
        for index, part in enumerate(parts[:-1])
        if part == "snapshots" and len(parts[index + 1]) == 40
    }


if __name__ == "__main__":
    raise SystemExit(main())
