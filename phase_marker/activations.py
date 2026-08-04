"""Bounded activation, attention, logit-lens, and phase-probe utilities.

The capture path is deliberately inference-only.  It resolves the Qwen causal
LM layout explicitly, installs read-only hooks on requested decoder layers,
copies only requested sequence positions, and removes every hook in ``finally``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from phase_marker.io import canonical_json, sha256_json
from phase_marker.config import ExperimentConfig, REQUIRED_MODEL_ID


MAX_ATTENTION_EXAMPLES = 16
MAX_ATTENTION_SEQUENCE = 256
CAPTURE_MODES = frozenset(("teacher_forced", "free_generation"))
_CAPTURE_PARENT_FIELDS = {
    "validation_selection": {"schema_version", "kind", "config_hash", "selected_on", "artifact_id"},
    "checkpoint": {
        "schema_version", "kind", "config_hash", "model_id", "model_revision",
        "checkpoint_path", "artifact_id",
    },
    "tokenized_batch": {
        "schema_version", "kind", "config_hash", "batch_file", "batch_hash", "layers",
        "positions", "artifact_id",
    },
    "behavior": {
        "schema_version", "kind", "evidence_scope", "backend", "config_hash", "run_kind",
        "seeds", "split_artifact_id", "split_manifest_hash", "materialization_artifact_ids",
        "checkpoint_artifact_ids", "checkpoint_manifest_hashes", "checkpoint_manifests",
        "examples_file", "examples_hash", "records_file", "records_hash", "row_count",
        "record_hashes", "exclusions", "parent_hashes", "completed", "artifact_id",
    },
    "synthetic": {
        "schema_version", "kind", "seed", "counts", "family_counts", "split_counts",
        "parameter_overlap", "exact_scorer_agreement", "evidence_scope", "backend",
        "config_hash", "preregistration_hash", "completed", "data_hashes", "artifact_id",
    },
}
_BATCH_METADATA_KEYS = frozenset(
    (
        "example_ids",
        "conditions",
        "parent_hashes",
        "capture_mode",
        "mode",
        "split",
        "workspace_prompt",
        "workspace_prompts",
    )
)


@dataclass(frozen=True)
class CaptureSpec:
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    capture_residual: bool = True
    capture_attention: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "positions", tuple(self.positions))
        if not self.layers or any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in self.layers
        ):
            raise ValueError("layers must contain nonnegative integer layer indices")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicates")
        if not self.positions or any(
            not isinstance(position, int) or isinstance(position, bool) or position < 0
            for position in self.positions
        ):
            raise ValueError("positions must contain nonnegative integer sequence positions")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must not contain duplicates")
        if not self.capture_residual and not self.capture_attention:
            raise ValueError("at least one capture type must be enabled")


@dataclass(frozen=True)
class ActivationBatch:
    example_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    residual: torch.Tensor
    attention: torch.Tensor
    parent_hashes: tuple[str, ...]


@dataclass(frozen=True)
class LogitLensBatch:
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    ranks: torch.Tensor
    parent_hashes: tuple[str, ...]


@dataclass(frozen=True)
class PhaseProbe:
    weight: torch.Tensor
    bias: torch.Tensor
    source_condition: str
    layer: int
    seed: int


@dataclass(frozen=True)
class ProbeMetrics:
    accuracy: float
    macro_f1: float
    source_condition: str
    target_condition: str
    layer: int


@dataclass
class _CaptureSession:
    residual: dict[int, torch.Tensor]
    attention: dict[int, torch.Tensor]


class _QwenCausalLMAdapter:
    """Resolve only the pinned Qwen-style causal-LM module topology."""

    def __init__(self, model: nn.Module) -> None:
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        unembedding = getattr(model, "lm_head", None)
        if not isinstance(layers, (nn.ModuleList, list, tuple)) or not all(
            isinstance(layer, nn.Module) for layer in layers
        ):
            raise TypeError("Qwen causal LM must expose decoder layers at model.model.layers")
        if not isinstance(norm, nn.Module):
            raise TypeError("Qwen causal LM must expose final RMSNorm at model.model.norm")
        if not isinstance(unembedding, nn.Module):
            raise TypeError("Qwen causal LM must expose unembedding at model.lm_head")
        self.layers = layers
        self.norm = norm
        self.unembedding = unembedding


@contextmanager
def capture_context(
    model: nn.Module, spec: CaptureSpec
) -> Iterator[_CaptureSession]:
    """Install read-only selected-position hooks and always remove them."""
    adapter = _QwenCausalLMAdapter(model)
    _validate_layer_indices(spec.layers, len(adapter.layers))
    session = _CaptureSession(residual={}, attention={})
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_index: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            hidden = _hidden_from_layer_output(output)
            _validate_positions(spec.positions, hidden.shape[1])
            if spec.capture_residual:
                selected = hidden[:, spec.positions, :].permute(1, 0, 2)
                session.residual[layer_index] = _compact_copy(selected)
            if spec.capture_attention:
                weights = _attention_from_layer_output(output)
                if weights is not None:
                    session.attention[layer_index] = _select_attention(
                        weights, spec.positions
                    )

        return hook

    try:
        for layer_index in spec.layers:
            handles.append(adapter.layers[layer_index].register_forward_hook(make_hook(layer_index)))
        yield session
    finally:
        for handle in handles:
            handle.remove()


def capture_selected_states(
    model: nn.Module, batch: Mapping[str, object], spec: CaptureSpec
) -> ActivationBatch:
    """Run one teacher-forced-style forward pass and return bounded captures."""
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping of model inputs and capture metadata")
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("batch input_ids must be a rank-2 tensor")
    batch_size, sequence_length = input_ids.shape
    _validate_positions(spec.positions, sequence_length)
    if spec.capture_attention:
        if batch_size > MAX_ATTENTION_EXAMPLES or sequence_length > MAX_ATTENTION_SEQUENCE:
            raise ValueError(
                "attention capture is limited to "
                f"{MAX_ATTENTION_EXAMPLES} examples and sequence length {MAX_ATTENTION_SEQUENCE}"
            )
        split = batch.get("split")
        if split is not None and split != "validation":
            raise ValueError("attention capture is restricted to the aligned validation subset")

    example_ids = _string_metadata(
        batch.get("example_ids"), batch_size, prefix="example"
    )
    conditions = _string_metadata(
        batch.get("conditions"), batch_size, prefix="unknown", repeat_single=True
    )
    parent_hashes = _parent_hashes(batch.get("parent_hashes"))
    model_inputs = {
        key: value for key, value in batch.items() if key not in _BATCH_METADATA_KEYS
    }
    if spec.capture_attention:
        model_inputs["output_attentions"] = True

    with torch.no_grad():
        with capture_context(model, spec) as session:
            outputs = model(**model_inputs)

    if spec.capture_residual:
        missing = [layer for layer in spec.layers if layer not in session.residual]
        if missing:
            raise RuntimeError(f"decoder hooks did not capture residuals for layers {missing}")
        residual = torch.stack([session.residual[layer] for layer in spec.layers])
    else:
        residual = torch.empty(0)

    if spec.capture_attention:
        returned_attentions = getattr(outputs, "attentions", None)
        if returned_attentions is not None:
            for layer in spec.layers:
                if layer not in session.attention:
                    session.attention[layer] = _select_attention(
                        returned_attentions[layer], spec.positions
                    )
        missing = [layer for layer in spec.layers if layer not in session.attention]
        if missing:
            raise RuntimeError(
                "model did not return requested attention tensors for layers "
                f"{missing}; use an eager attention implementation that supports output_attentions"
            )
        attention = torch.stack([session.attention[layer] for layer in spec.layers])
    else:
        attention = torch.empty(0)

    return ActivationBatch(
        example_ids=example_ids,
        conditions=conditions,
        layers=spec.layers,
        positions=spec.positions,
        residual=residual,
        attention=attention,
        parent_hashes=parent_hashes,
    )


def apply_logit_lens(
    model: nn.Module,
    activation_batch: ActivationBatch,
    candidate_token_ids: Sequence[int] | torch.Tensor | None = None,
) -> LogitLensBatch:
    """Apply final RMSNorm and unembedding, then rank declared candidates."""
    _validate_activation_batch(activation_batch, require_residual=True)
    adapter = _QwenCausalLMAdapter(model)
    device = _module_device(adapter.unembedding)
    with torch.no_grad():
        residual = activation_batch.residual.to(device)
        normalized = adapter.norm(residual)
        logits = adapter.unembedding(normalized).float()
        all_logprobs = logits.log_softmax(dim=-1)
        if candidate_token_ids is None:
            token_ids = logits.argmax(dim=-1)
            logprobs = all_logprobs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            ranks = torch.ones_like(token_ids, dtype=torch.long)
        else:
            token_ids = torch.as_tensor(candidate_token_ids, dtype=torch.long, device=device)
            if not token_ids.numel():
                raise ValueError("candidate_token_ids must be nonempty")
            if token_ids.min().item() < 0 or token_ids.max().item() >= logits.shape[-1]:
                raise IndexError("candidate token id is outside the model vocabulary")
            prefix = logits.shape[:-1]
            if token_ids.ndim == 1:
                expanded = token_ids.view(*((1,) * len(prefix)), -1).expand(
                    *prefix, token_ids.numel()
                )
                candidate_logits = logits.gather(-1, expanded)
                logprobs = all_logprobs.gather(-1, expanded)
                ranks = 1 + (
                    logits.unsqueeze(-2) > candidate_logits.unsqueeze(-1)
                ).sum(dim=-1)
            elif token_ids.ndim == 2 and tuple(token_ids.shape) == prefix[1:]:
                token_ids = token_ids.unsqueeze(0).expand(prefix)
                candidate_logits = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                logprobs = all_logprobs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                ranks = 1 + (logits > candidate_logits.unsqueeze(-1)).sum(dim=-1)
            elif token_ids.ndim == 3 and tuple(token_ids.shape) == prefix:
                candidate_logits = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                logprobs = all_logprobs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                ranks = 1 + (logits > candidate_logits.unsqueeze(-1)).sum(dim=-1)
            elif token_ids.ndim == 3 and tuple(token_ids.shape[:2]) == prefix[1:]:
                token_ids = token_ids.unsqueeze(0).expand(prefix[0], *token_ids.shape)
                candidate_logits = logits.gather(-1, token_ids)
                logprobs = all_logprobs.gather(-1, token_ids)
                ranks = 1 + (
                    logits.unsqueeze(-2) > candidate_logits.unsqueeze(-1)
                ).sum(dim=-1)
            elif token_ids.ndim == 4 and tuple(token_ids.shape[:3]) == prefix:
                candidate_logits = logits.gather(-1, token_ids)
                logprobs = all_logprobs.gather(-1, token_ids)
                ranks = 1 + (
                    logits.unsqueeze(-2) > candidate_logits.unsqueeze(-1)
                ).sum(dim=-1)
            else:
                raise ValueError(
                    "candidate_token_ids must be global [candidates], known targets "
                    "[positions, examples], or aligned candidates [positions, examples, candidates]"
                )

    return LogitLensBatch(
        token_ids=_compact_copy(token_ids),
        logprobs=_compact_copy(logprobs),
        ranks=_compact_copy(ranks),
        parent_hashes=activation_batch.parent_hashes,
    )


def fit_phase_probe(
    train: ActivationBatch, validation: ActivationBatch, seed: int
) -> PhaseProbe:
    """Validation-select a ridge-linear phase classifier without test access."""
    _validate_probe_pair(train, validation)
    source_condition = _single_condition(train.conditions, label="training")
    if _single_condition(validation.conditions, label="validation") != source_condition:
        raise ValueError("training and validation captures must use the same source condition")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    candidates: list[tuple[float, float, int, float, torch.Tensor, torch.Tensor]] = []
    regularizations = (1e-4, 1e-2, 1.0, 100.0)
    for layer_offset, layer in enumerate(train.layers):
        train_x, train_y = _probe_rows(train, layer_offset)
        validation_x, validation_y = _probe_rows(validation, layer_offset)
        for regularization in regularizations:
            weight, bias = _fit_ridge(train_x, train_y, len(train.positions), regularization)
            predictions = validation_x.double() @ weight.T + bias
            predicted_labels = predictions.argmax(dim=-1)
            accuracy, macro_f1 = _classification_metrics(
                predicted_labels, validation_y, len(train.positions)
            )
            candidates.append(
                (
                    accuracy,
                    macro_f1,
                    -layer_offset,
                    -regularization,
                    weight,
                    bias,
                )
            )
    best = max(candidates, key=lambda item: item[:4])
    selected_offset = -best[2]
    return PhaseProbe(
        weight=_compact_copy(best[4]).float(),
        bias=_compact_copy(best[5]).float(),
        source_condition=source_condition,
        layer=train.layers[selected_offset],
        seed=seed,
    )


def evaluate_phase_probe(probe: PhaseProbe, test: ActivationBatch) -> ProbeMetrics:
    """Evaluate a frozen probe on another format without refitting."""
    _validate_activation_batch(test, require_residual=True)
    target_condition = _single_condition(test.conditions, label="test")
    try:
        layer_offset = test.layers.index(probe.layer)
    except ValueError as error:
        raise ValueError(f"test captures do not include selected layer {probe.layer}") from error
    test_x, test_y = _probe_rows(test, layer_offset)
    if test_x.shape[1] != probe.weight.shape[1]:
        raise ValueError("probe and test capture hidden sizes differ")
    with torch.no_grad():
        predictions = test_x.float() @ probe.weight.float().T + probe.bias.float()
        predicted_labels = predictions.argmax(dim=-1)
    accuracy, macro_f1 = _classification_metrics(
        predicted_labels, test_y, probe.weight.shape[0]
    )
    return ProbeMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        source_condition=probe.source_condition,
        target_condition=target_condition,
        layer=probe.layer,
    )


def save_activation_artifact(
    activation_batch: ActivationBatch,
    output_root: Path,
    *,
    mode: str,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write compact tensors and a provenance-bound JSON manifest."""
    _validate_activation_batch(activation_batch, require_residual=False)
    if mode not in CAPTURE_MODES:
        raise ValueError(f"mode must be one of {sorted(CAPTURE_MODES)}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tensors = {
        "residual": _compact_copy(activation_batch.residual),
        "attention": _compact_copy(activation_batch.attention),
    }
    tensor_path = output_root / "selected-states.pt"
    _torch_save_atomic(tensor_path, tensors)
    tensor_hash = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "kind": "phase_marker_selected_activations",
        "mode": mode,
        "example_ids": list(activation_batch.example_ids),
        "conditions": list(activation_batch.conditions),
        "layers": list(activation_batch.layers),
        "positions": list(activation_batch.positions),
        "parent_hashes": list(activation_batch.parent_hashes),
        "tensor_file": tensor_path.name,
        "tensor_hash": tensor_hash,
        "tensors": {
            name: {"shape": list(value.shape), "dtype": _dtype_name(value.dtype)}
            for name, value in tensors.items()
        },
    }
    if extra_metadata is not None:
        manifest["metadata"] = dict(extra_metadata)
    manifest["artifact_id"] = sha256_json(manifest)
    _write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def load_and_validate_activation_artifact(
    manifest_path: Path,
    *,
    expected_checkpoint_id: str | None = None,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor]]:
    """Safely load and fully validate a canonical activation tensor artifact."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("activation manifest must be an object")
    tensor_file = manifest.get("tensor_file")
    if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
        raise ValueError("activation tensor filename is malformed")
    tensor_path = manifest_path.parent / tensor_file
    if (
        not tensor_path.is_file()
        or manifest.get("tensor_hash") != hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    ):
        raise ValueError("activation tensor path or hash mismatch")
    if expected_checkpoint_id is not None and manifest.get("checkpoint_artifact_id") != expected_checkpoint_id:
        raise ValueError("activation/checkpoint lineage mismatch")
    if manifest.get("mode") not in CAPTURE_MODES:
        raise ValueError("activation capture mode mismatch")
    raw = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or set(raw) != {"residual", "attention"}:
        raise ValueError("activation tensor names must be exactly residual and attention")
    tensors: dict[str, torch.Tensor] = {}
    metadata = manifest.get("tensors")
    if not isinstance(metadata, Mapping) or set(metadata) != set(raw):
        raise ValueError("activation tensor metadata mismatch")
    layers = manifest.get("layers")
    positions = manifest.get("positions")
    example_ids = manifest.get("example_ids")
    conditions = manifest.get("conditions")
    if not all(isinstance(value, list) and value for value in (layers, positions, example_ids, conditions)):
        raise ValueError("activation dimensions must be nonempty")
    if len(example_ids) != len(conditions):
        raise ValueError("activation example/condition dimensions mismatch")
    prefix = (len(layers), len(positions), len(example_ids))
    for name, value in raw.items():
        if (
            not isinstance(value, torch.Tensor) or value.numel() == 0
            or value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            or not torch.isfinite(value).all().item()
            or value.ndim != (4 if name == "residual" else 5)
            or tuple(value.shape[:3]) != prefix
        ):
            raise ValueError(f"activation {name} tensor semantics mismatch")
        declared = metadata[name]
        if not isinstance(declared, Mapping) or declared.get("shape") != list(value.shape) or declared.get("dtype") != _dtype_name(value.dtype):
            raise ValueError(f"activation {name} metadata mismatch")
        tensors[name] = value
    return manifest, tensors


def _hidden_from_layer_output(output: object) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("Qwen decoder layer output must contain rank-3 hidden states")
    return hidden


def _attention_from_layer_output(output: object) -> torch.Tensor | None:
    if not isinstance(output, (tuple, list)):
        return None
    for value in output[1:]:
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return value
    return None


def _select_attention(weights: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor) or weights.ndim != 4:
        raise TypeError("attention weights must have shape [batch, heads, query, key]")
    _validate_positions(positions, weights.shape[-2])
    selected = weights[:, :, positions, :].permute(2, 0, 1, 3)
    return _compact_copy(selected)


def _validate_layer_indices(layers: tuple[int, ...], layer_count: int) -> None:
    for layer in layers:
        if layer >= layer_count:
            raise IndexError(f"layer {layer} is outside decoder layer count {layer_count}")


def _validate_positions(positions: tuple[int, ...], sequence_length: int) -> None:
    for position in positions:
        if position >= sequence_length:
            raise IndexError(
                f"position {position} is outside actual sequence length {sequence_length}"
            )


def _string_metadata(
    value: object,
    count: int,
    *,
    prefix: str,
    repeat_single: bool = False,
) -> tuple[str, ...]:
    if value is None:
        return tuple(f"{prefix}-{index}" for index in range(count))
    if isinstance(value, str):
        if repeat_single:
            return (value,) * count
        if count == 1:
            return (value,)
        raise ValueError(f"{prefix} metadata must have one value per example")
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{prefix} metadata must be a sequence of strings")
    result = tuple(value)
    if len(result) != count or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{prefix} metadata must contain {count} nonempty strings")
    return result


def _parent_hashes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError("parent_hashes must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("parent_hashes must contain nonempty strings")
    return result


def _validate_activation_batch(batch: ActivationBatch, *, require_residual: bool) -> None:
    if len(batch.example_ids) != len(batch.conditions):
        raise ValueError("example_ids and conditions must have equal length")
    if require_residual or batch.residual.numel():
        expected_prefix = (len(batch.layers), len(batch.positions), len(batch.example_ids))
        if batch.residual.ndim != 4 or tuple(batch.residual.shape[:3]) != expected_prefix:
            raise ValueError(
                "residual must have shape [layers, positions, examples, hidden_size]"
            )
    if batch.attention.numel():
        expected_prefix = (len(batch.layers), len(batch.positions), len(batch.example_ids))
        if batch.attention.ndim != 5 or tuple(batch.attention.shape[:3]) != expected_prefix:
            raise ValueError(
                "attention must have shape [layers, positions, examples, heads, key_length]"
            )


def _validate_probe_pair(train: ActivationBatch, validation: ActivationBatch) -> None:
    _validate_activation_batch(train, require_residual=True)
    _validate_activation_batch(validation, require_residual=True)
    if train.layers != validation.layers:
        raise ValueError("training and validation captures must include identical layers")
    if train.positions != validation.positions:
        raise ValueError("training and validation captures must use identical phase positions")
    if train.residual.shape[-1] != validation.residual.shape[-1]:
        raise ValueError("training and validation capture hidden sizes differ")


def _single_condition(conditions: tuple[str, ...], *, label: str) -> str:
    distinct = set(conditions)
    if len(distinct) != 1:
        raise ValueError(f"{label} captures must contain exactly one format condition")
    return next(iter(distinct))


def _probe_rows(batch: ActivationBatch, layer_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    states = batch.residual[layer_offset].detach().cpu().float()
    positions, examples, hidden_size = states.shape
    rows = states.reshape(positions * examples, hidden_size)
    labels = torch.arange(positions, dtype=torch.long).repeat_interleave(examples)
    return rows, labels


def _fit_ridge(
    rows: torch.Tensor, labels: torch.Tensor, classes: int, regularization: float
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rows.double()
    y = torch.nn.functional.one_hot(labels, num_classes=classes).double()
    design = torch.cat((x, torch.ones(x.shape[0], 1, dtype=x.dtype)), dim=1)
    penalty = torch.eye(design.shape[1], dtype=x.dtype) * regularization
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coefficients[:-1].T, coefficients[-1]


def _classification_metrics(
    predictions: torch.Tensor, labels: torch.Tensor, classes: int
) -> tuple[float, float]:
    predictions = predictions.detach().cpu()
    labels = labels.detach().cpu()
    accuracy = float((predictions == labels).float().mean().item())
    f1_values: list[float] = []
    for class_index in range(classes):
        true_positive = int(((predictions == class_index) & (labels == class_index)).sum())
        false_positive = int(((predictions == class_index) & (labels != class_index)).sum())
        false_negative = int(((predictions != class_index) & (labels == class_index)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return accuracy, sum(f1_values) / len(f1_values)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _compact_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_save_atomic(path: Path, payload: Mapping[str, torch.Tensor]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(dict(payload), handle)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(dict(payload)) + "\n")
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class _SmokeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, *, output_attentions: bool = False
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = hidden_states + torch.tanh(self.projection(hidden_states))
        if not output_attentions:
            return (hidden_states,)
        attention = (hidden_states @ hidden_states.transpose(-1, -2)).softmax(-1).unsqueeze(1)
        return hidden_states, attention


class _SmokeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + 1e-6) * self.weight


class _SmokeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(23, 8)
        self.layers = nn.ModuleList(_SmokeLayer(8) for _ in range(3))
        self.norm = _SmokeRMSNorm(8)

    def forward(self, input_ids: torch.Tensor, *, output_attentions: bool = False) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            layer_outputs = layer(hidden, output_attentions=output_attentions)
            hidden = layer_outputs[0]
            if output_attentions:
                attentions.append(layer_outputs[1])
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden),
            attentions=tuple(attentions) if output_attentions else None,
        )


class _SmokeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _SmokeBackbone()
        self.lm_head = nn.Linear(8, 23, bias=False)

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        outputs = self.model(**kwargs)
        return SimpleNamespace(
            logits=self.lm_head(outputs.last_hidden_state), attentions=outputs.attentions
        )


def _smoke(output_root: Path) -> dict[str, object]:
    torch.manual_seed(101)
    model = _SmokeCausalLM().eval()
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.long
    )
    model_inputs = {"input_ids": input_ids}
    spec = CaptureSpec(
        layers=(0, 2),
        positions=(3, 7),
        capture_residual=True,
        capture_attention=True,
    )
    with torch.no_grad():
        baseline = model(**model_inputs).logits
        with capture_context(model, spec):
            observed = model(**model_inputs, output_attentions=True).logits
    logits_equal = torch.equal(baseline, observed)
    if not logits_equal:
        raise RuntimeError("read-only capture changed smoke-model logits")
    batch: dict[str, object] = {
        **model_inputs,
        "example_ids": ("smoke:0", "smoke:1"),
        "conditions": ("glyph", "glyph"),
        "parent_hashes": ("1" * 64, "2" * 64),
        "split": "validation",
    }
    activation = capture_selected_states(model, batch, spec)
    manifest = save_activation_artifact(
        activation,
        output_root,
        mode="teacher_forced",
        extra_metadata={
            "baseline_capture_logits_equal": logits_equal,
            "model": "tiny_local_torch_qwen_layout",
            "device": "cpu",
        },
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture bounded phase-marker activations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run the tiny local torch capture smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    capture = subparsers.add_parser("capture", help="capture from provenance-bound inputs")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--mode", choices=tuple(sorted(CAPTURE_MODES)), required=True)
    capture.add_argument("--validation-selection-manifest", type=Path, required=True)
    capture.add_argument("--tokenized-batch-manifest", type=Path, required=True)
    capture.add_argument("--tokenized-batch", type=Path, required=True)
    capture.add_argument("--model-id", required=True)
    capture.add_argument("--model-revision", required=True)
    capture.add_argument("--checkpoint-manifest", type=Path, required=True)
    capture.add_argument("--behavior-manifest", type=Path, required=True)
    capture.add_argument("--synthetic-manifest", type=Path, required=True)
    capture.add_argument("--backend", choices=("hf", "tiny-fixture"), required=True)
    capture.add_argument("--allow-test-backend", action="store_true")
    capture.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        manifest = _smoke(args.output_root)
    else:
        destination = args.output_root
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite capture output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix=f".{destination.name}-staging-"
        ) as temporary:
            staging = Path(temporary) / "publish"
            args.output_root = staging
            manifest = _run_capture(args)
            staging.replace(destination)
    print(canonical_json(manifest))
    return 0


def _run_capture(args: argparse.Namespace) -> dict[str, object]:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite capture output: {args.output_root}")
    if args.backend == "tiny-fixture" and not args.allow_test_backend:
        raise ValueError("tiny-fixture requires explicit --allow-test-backend")
    config = ExperimentConfig.load(args.config)
    config_hash = sha256_json(config.__dict__)
    if args.model_id != REQUIRED_MODEL_ID:
        raise ValueError("capture model id does not match the frozen configuration")
    parents = {
        name: _load_capture_parent(
            path, name, config_hash, allow_fixture=args.backend == "tiny-fixture"
        )
        for name, path in (
            ("validation_selection", args.validation_selection_manifest),
            ("checkpoint", args.checkpoint_manifest),
            ("behavior", args.behavior_manifest),
            ("synthetic", args.synthetic_manifest),
        )
    }
    if parents["validation_selection"].get("selected_on") != "validation":
        raise ValueError("capture selection must be selected_on=validation")
    if args.backend == "hf" and (
        parents["behavior"].get("evidence_scope") != "experiment_candidate"
        or parents["behavior"].get("backend") != "vllm"
        or parents["synthetic"].get("evidence_scope") != "experiment"
        or parents["synthetic"].get("backend") != "production"
    ):
        raise ValueError("production capture requires production behavior and synthetic evidence")
    if args.backend == "hf":
        _validate_capture_bound_files(
            parents["behavior"], args.behavior_manifest, "behavior"
        )
        _validate_capture_bound_files(
            parents["synthetic"], args.synthetic_manifest, "synthetic"
        )
    checkpoint = parents["checkpoint"]
    if checkpoint.get("model_id") != args.model_id or checkpoint.get("model_revision") != args.model_revision:
        raise ValueError("capture checkpoint model identity mismatch")
    batch_manifest = _load_capture_parent(
        args.tokenized_batch_manifest, "tokenized_batch", config_hash,
        allow_fixture=args.backend == "tiny-fixture",
    )
    if batch_manifest.get("batch_file") != str(args.tokenized_batch):
        raise ValueError("tokenized batch path mismatch")
    if batch_manifest.get("batch_hash") != hashlib.sha256(args.tokenized_batch.read_bytes()).hexdigest():
        raise ValueError("tokenized batch hash mismatch")
    batch = None
    if args.backend == "hf":
        batch = _load_and_validate_capture_batch(
            args.tokenized_batch, batch_manifest, mode=args.mode
        )
    if args.backend == "tiny-fixture":
        manifest = _smoke(args.output_root)
    else:  # model loading happens only after every immutable input has passed validation
        from transformers import AutoModelForCausalLM

        checkpoint_path = checkpoint.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_dir():
            raise ValueError("capture checkpoint path is missing")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, revision=args.model_revision, local_files_only=True
        ).eval()
        assert batch is not None
        layers = tuple(batch_manifest.get("layers", ()))
        positions = tuple(batch_manifest.get("positions", ()))
        captured = capture_selected_states(model, batch, CaptureSpec(layers, positions))
        manifest = save_activation_artifact(captured, args.output_root, mode=args.mode)
    tensor_path = args.output_root / str(manifest["tensor_file"])
    envelope: dict[str, object] = {
        **manifest,
        "schema_version": 1,
        "evidence_scope": "plumbing_only" if args.backend == "tiny-fixture" else "experiment",
        "backend": args.backend,
        "config_hash": config_hash,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "checkpoint_artifact_id": checkpoint["artifact_id"],
        "validation_selection_artifact_id": parents["validation_selection"]["artifact_id"],
        "behavior_artifact_id": parents["behavior"]["artifact_id"],
        "synthetic_artifact_id": parents["synthetic"]["artifact_id"],
        "tokenized_batch_artifact_id": batch_manifest["artifact_id"],
        "tokenized_batch_manifest_hash": hashlib.sha256(args.tokenized_batch_manifest.read_bytes()).hexdigest(),
        "tensor_hash": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "completed": True,
    }
    envelope["parent_hashes"] = [
        envelope["validation_selection_artifact_id"], envelope["tokenized_batch_artifact_id"],
        envelope["checkpoint_artifact_id"], envelope["behavior_artifact_id"],
        envelope["synthetic_artifact_id"],
    ]
    envelope.pop("artifact_id", None)
    envelope["artifact_id"] = sha256_json(envelope)
    _write_json_atomic(args.output_root / "manifest.json", envelope)
    return envelope


def _load_and_validate_capture_batch(
    path: Path, manifest: Mapping[str, object], *, mode: str
) -> Mapping[str, object]:
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batch, Mapping):
        raise ValueError("tokenized batch must contain a mapping")
    input_ids = batch.get("input_ids")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or input_ids.shape[0] < 1
        or input_ids.shape[1] < 1
    ):
        raise ValueError("tokenized batch input_ids must be nonempty rank-2 torch.long")
    for field in ("attention_mask", "position_ids"):
        value = batch.get(field)
        if value is not None and (
            not isinstance(value, torch.Tensor)
            or value.shape != input_ids.shape
            or value.dtype != torch.long
        ):
            raise ValueError(f"tokenized batch {field} must match input_ids")
    layers = manifest.get("layers")
    positions = manifest.get("positions")
    if (
        not isinstance(layers, list) or not layers
        or any(not isinstance(value, int) or value < 0 for value in layers)
        or not isinstance(positions, list) or not positions
        or any(not isinstance(value, int) or value < 0 or value >= input_ids.shape[1] for value in positions)
    ):
        raise ValueError("tokenized batch layers or positions are malformed")
    if manifest.get("mode", mode) != mode:
        raise ValueError("tokenized batch capture mode mismatch")
    for field in ("example_ids", "conditions"):
        values = batch.get(field)
        if not isinstance(values, (list, tuple)) or len(values) != input_ids.shape[0] or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"tokenized batch {field} must align with batch rows")
    return batch


def _load_capture_parent(
    path: Path, label: str, config_hash: str, *, allow_fixture: bool
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{label} parent must be a schema-v1 object")
    if payload.get("config_hash") != config_hash:
        raise ValueError(f"{label} parent config hash mismatch")
    expected_fields = _CAPTURE_PARENT_FIELDS.get(label)
    if not allow_fixture and expected_fields is not None and set(payload) != expected_fields:
        raise ValueError(f"{label} parent fields do not match the exact schema")
    artifact_id = payload.get("artifact_id")
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or artifact_id != sha256_json(unsigned)
    ):
        raise ValueError(f"{label} parent artifact id is malformed")
    if payload.get("evidence_scope") == "plumbing_only" and label in {"behavior", "synthetic"}:
        raise ValueError(f"production capture rejects plumbing-only {label} evidence")
    return payload


def _validate_capture_bound_files(
    payload: Mapping[str, object], manifest_path: Path, label: str
) -> None:
    if label == "behavior":
        bindings = (
            (manifest_path.parent / str(payload["records_file"]), payload["records_hash"]),
            (Path(str(payload["examples_file"])), payload["examples_hash"]),
        )
        for path, expected_hash in bindings:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"behavior bound file is missing or stale: {path}")
        records_path = bindings[0][0]
        rows = [
            json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            not rows
            or payload.get("row_count") != len(rows)
            or payload.get("record_hashes") != [sha256_json(row) for row in rows]
        ):
            raise ValueError("behavior bound record count or hashes mismatch")
        manifests = payload.get("checkpoint_manifests")
        hashes = payload.get("checkpoint_manifest_hashes")
        if not isinstance(manifests, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("behavior checkpoint bindings are malformed")
        for key, value in manifests.items():
            path = Path(str(value))
            if key not in hashes or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
                raise ValueError("behavior checkpoint manifest is missing or stale")
        return
    data_hashes = payload.get("data_hashes")
    if not isinstance(data_hashes, Mapping):
        raise ValueError("synthetic data hashes are malformed")
    for split, expected_hash in data_hashes.items():
        path = manifest_path.parent / f"{split}.jsonl"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"synthetic bound file is missing or stale: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
