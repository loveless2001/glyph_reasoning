"""Causal residual and KV-cache interventions for aligned Qwen-style batches."""

from __future__ import annotations

import argparse
import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from phase_marker.io import canonical_json, sha256_json
from phase_marker.schema import InterventionRecord


_METADATA_KEYS = frozenset(
    {
        "example_ids",
        "recipient_id",
        "donor_id",
        "parent_hashes",
        "target_token_ids",
        "matched_positions",
    }
)


class AlignmentError(ValueError):
    """Raised when donor and recipient states cannot be causally aligned."""


@dataclass(frozen=True)
class InterventionSpec:
    method: str
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    norm_match: bool
    target_token_ids: tuple[int, ...]
    control_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "target_token_ids", tuple(self.target_token_ids))
        if len(self.layers) != 1:
            raise ValueError("an intervention invocation must specify exactly one layer")
        if not isinstance(self.layers[0], int) or isinstance(self.layers[0], bool) or self.layers[0] < 0:
            raise ValueError("layers must contain a nonnegative integer layer")
        if not self.positions or any(
            not isinstance(position, int) or isinstance(position, bool) or position < 0
            for position in self.positions
        ):
            raise ValueError("positions must contain nonnegative integer sequence positions")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must not contain duplicates")
        if not self.target_token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in self.target_token_ids
        ):
            raise ValueError("target_token_ids must contain nonnegative token ids")
        if not self.method or not self.control_name:
            raise ValueError("method and control_name must be nonempty")


@dataclass(frozen=True)
class InterventionResult:
    records: tuple[InterventionRecord, ...]
    baseline_logits: torch.Tensor
    intervened_logits: torch.Tensor

    @property
    def record(self) -> InterventionRecord:
        if len(self.records) != 1:
            raise ValueError("record convenience access requires exactly one record")
        return self.records[0]


def replace_positions(
    recipient: torch.Tensor,
    donor: torch.Tensor,
    *,
    positions: Sequence[int],
    norm_match: bool = False,
) -> torch.Tensor:
    """Clone ``recipient`` and replace only selected sequence rows."""
    if not isinstance(recipient, torch.Tensor) or not isinstance(donor, torch.Tensor):
        raise TypeError("recipient and donor residuals must be tensors")
    if recipient.ndim != 3 or donor.ndim != 3:
        raise ValueError("residuals must have shape [batch, sequence, hidden]")
    if recipient.shape != donor.shape:
        raise AlignmentError("recipient and donor residual shapes must match")
    if recipient.dtype != donor.dtype:
        raise AlignmentError("recipient and donor residual dtype must match")
    if recipient.device != donor.device:
        raise AlignmentError("recipient and donor residual device must match")
    selected = tuple(positions)
    _validate_positions(selected, recipient.shape[1])
    replacement = donor[:, selected, :]
    if norm_match:
        target = recipient[:, selected, :]
        target_mean = target.mean(dim=-1, keepdim=True)
        target_centered = target - target_mean
        source_centered = replacement - replacement.mean(dim=-1, keepdim=True)
        source_norm = source_centered.norm(dim=-1, keepdim=True)
        target_norm = target_centered.norm(dim=-1, keepdim=True)
        epsilon = torch.finfo(source_centered.dtype).eps
        if torch.any((source_norm <= epsilon) & (target_norm > epsilon)):
            raise AlignmentError(
                "cannot norm-match a zero-centered source to a nonzero recipient norm"
            )
        scaled = torch.where(
            source_norm > epsilon,
            source_centered * target_norm / source_norm.clamp_min(
                epsilon
            ),
            torch.zeros_like(source_centered),
        )
        replacement = scaled + target_mean
    result = recipient.clone()
    result[:, selected, :] = replacement
    return result


def patch_residual_positions(
    model: nn.Module,
    recipient_batch: Mapping[str, object],
    donor_batch: Mapping[str, object],
    spec: InterventionSpec,
) -> InterventionResult:
    """Patch one decoder layer and one selected token region from an aligned donor."""
    _validate_operation(
        spec,
        operation="residual_patch",
        controls=frozenset(("donor", "random_donor")),
    )
    layer = _decoder_layer(model, spec.layers[0])
    recipient_inputs, recipient_ids, recipient_parents = _batch_parts(recipient_batch)
    donor_inputs, donor_ids, donor_parents = _batch_parts(donor_batch)
    _validate_aligned_inputs(recipient_inputs, donor_inputs)
    batch_size = _batch_size(recipient_inputs)
    target_ids = _target_ids(spec.target_token_ids, batch_size, "recipient")
    donor_target_ids = _batch_target_ids(donor_batch, batch_size, "donor")
    source_donor_ids = donor_ids
    source_donor_targets = donor_target_ids
    if spec.control_name == "random_donor":
        source_donor_ids = _roll_tuple(donor_ids)
        source_donor_targets = donor_target_ids.roll(1, dims=0)

    with torch.no_grad():
        baseline_logits = _logits(model(**recipient_inputs))
        donor_state = _capture_layer_output(model, layer, donor_inputs)
        source_holder: dict[str, torch.Tensor] = {}

        def patch_hook(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> object:
            recipient_state = _hidden(output)
            source = _control_source(
                spec.control_name,
                recipient_state,
                donor_state,
                spec.positions,
                recipient_batch,
                validation_mean=None,
            )
            source_holder["source"] = source
            patched = replace_positions(
                recipient_state, source, positions=spec.positions, norm_match=spec.norm_match
            )
            return _with_hidden(output, patched)

        handle = layer.register_forward_hook(patch_hook)
        try:
            intervened_logits = _logits(model(**recipient_inputs))
        finally:
            handle.remove()
    if "source" not in source_holder:
        raise RuntimeError("residual intervention hook did not run")
    return _result(
        baseline_logits,
        intervened_logits,
        spec,
        recipient_ids,
        source_donor_ids,
        recipient_parents + donor_parents,
        target_ids,
        source_donor_targets,
    )


def ablate_positions(
    model: nn.Module,
    batch: Mapping[str, object],
    spec: InterventionSpec,
    validation_mean: torch.Tensor | None = None,
) -> InterventionResult:
    """Apply a zero, validation-mean, shuffle, or matched-position control."""
    _validate_operation(
        spec,
        operation="ablate",
        controls=frozenset(
            ("zero", "validation_mean", "within_batch_shuffle", "matched_non_marker_position")
        ),
    )
    layer = _decoder_layer(model, spec.layers[0])
    model_inputs, recipient_ids, parents = _batch_parts(batch)
    batch_size = _batch_size(model_inputs)
    target_ids = _target_ids(spec.target_token_ids, batch_size, "recipient")
    source_holder: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        baseline_logits = _logits(model(**model_inputs))

        def ablation_hook(
            _module: nn.Module, _inputs: tuple[object, ...], output: object
        ) -> object:
            recipient_state = _hidden(output)
            source = _control_source(
                spec.control_name,
                recipient_state,
                None,
                spec.positions,
                batch,
                validation_mean,
            )
            source_holder["selected"] = source[:, spec.positions, :].detach().clone()
            patched = replace_positions(
                recipient_state, source, positions=spec.positions, norm_match=spec.norm_match
            )
            return _with_hidden(output, patched)

        handle = layer.register_forward_hook(ablation_hook)
        try:
            intervened_logits = _logits(model(**model_inputs))
        finally:
            handle.remove()
    if "selected" not in source_holder:
        raise RuntimeError("ablation intervention hook did not run")
    if spec.control_name == "within_batch_shuffle":
        donor_ids = tuple(
            f"control:within_batch_shuffle:{source_id}"
            for source_id in _roll_tuple(recipient_ids)
        )
    else:
        donor_ids = (f"control:{spec.control_name}",) * batch_size
    source_positions = (
        tuple(batch["matched_positions"])
        if spec.control_name == "matched_non_marker_position"
        else None
    )
    control_source_hashes = (
        tuple(
            _effective_tensor_hash(source_holder["selected"][index])
            for index in range(batch_size)
        )
        if spec.control_name == "validation_mean"
        else (None,) * batch_size
    )
    return _result(
        baseline_logits,
        intervened_logits,
        spec,
        recipient_ids,
        donor_ids,
        parents,
        target_ids,
        None,
        source_positions=source_positions,
        control_source_hashes=control_source_hashes,
    )


def transplant_cache_rows(
    recipient_cache: object,
    donor_cache: object,
    *,
    positions: Sequence[int],
    layers: Sequence[int] | None = None,
) -> object:
    """Clone a DynamicCache-compatible recipient and transplant selected KV rows."""
    recipient_pairs = _cache_pairs(recipient_cache)
    donor_pairs = _cache_pairs(donor_cache)
    if len(recipient_pairs) != len(donor_pairs):
        raise AlignmentError("cache layer counts differ")
    selected_layers = tuple(range(len(recipient_pairs))) if layers is None else tuple(layers)
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layers must select unique cache layers")
    for layer_index in selected_layers:
        if layer_index < 0 or layer_index >= len(recipient_pairs):
            raise IndexError(f"cache layer {layer_index} is outside layer count {len(recipient_pairs)}")
    for recipient_pair, donor_pair in zip(recipient_pairs, donor_pairs, strict=True):
        _validate_cache_pair(recipient_pair, donor_pair)
    cloned = copy.deepcopy(recipient_cache)
    cloned_pairs = _cache_pairs(cloned)
    selected_positions = tuple(positions)
    for layer_index in selected_layers:
        recipient_key, recipient_value = cloned_pairs[layer_index]
        donor_key, donor_value = donor_pairs[layer_index]
        _validate_positions(selected_positions, recipient_key.shape[-2])
        recipient_key[:, :, selected_positions, :] = donor_key[:, :, selected_positions, :]
        recipient_value[:, :, selected_positions, :] = donor_value[:, :, selected_positions, :]
    return cloned


def transplant_kv_positions(
    model: nn.Module,
    recipient_batch: Mapping[str, object],
    donor_batch: Mapping[str, object],
    spec: InterventionSpec,
) -> InterventionResult:
    """Transplant aligned prefix KV rows, then score the recipient query token."""
    _validate_operation(spec, operation="kv_transplant", controls=frozenset(("donor",)))
    recipient_inputs, recipient_ids, recipient_parents = _batch_parts(recipient_batch)
    donor_inputs, donor_ids, donor_parents = _batch_parts(donor_batch)
    _validate_aligned_inputs(recipient_inputs, donor_inputs)
    batch_size = _batch_size(recipient_inputs)
    target_ids = _target_ids(spec.target_token_ids, batch_size, "recipient")
    donor_target_ids = _batch_target_ids(donor_batch, batch_size, "donor")
    recipient_prefix, recipient_query = _prefix_and_query(recipient_inputs)
    donor_prefix, _donor_query = _prefix_and_query(donor_inputs)
    with torch.no_grad():
        recipient_outputs = model(**recipient_prefix, use_cache=True)
        donor_outputs = model(**donor_prefix, use_cache=True)
        recipient_cache = getattr(recipient_outputs, "past_key_values", None)
        donor_cache = getattr(donor_outputs, "past_key_values", None)
        if recipient_cache is None or donor_cache is None:
            raise TypeError("model must return DynamicCache-compatible past_key_values")
        baseline_cache = copy.deepcopy(recipient_cache)
        transplanted_cache = transplant_cache_rows(
            recipient_cache,
            donor_cache,
            positions=spec.positions,
            layers=spec.layers,
        )
        baseline_logits = _logits(
            model(**recipient_query, past_key_values=baseline_cache, use_cache=True)
        )
        intervened_logits = _logits(
            model(**recipient_query, past_key_values=transplanted_cache, use_cache=True)
        )
    return _result(
        baseline_logits,
        intervened_logits,
        spec,
        recipient_ids,
        donor_ids,
        recipient_parents + donor_parents,
        target_ids,
        donor_target_ids,
    )


def _decoder_layer(model: nn.Module, index: int) -> nn.Module:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise TypeError("Qwen causal LM must expose decoder layers at model.model.layers")
    if index >= len(layers):
        raise IndexError(f"layer {index} is outside decoder layer count {len(layers)}")
    return layers[index]


def _validate_operation(
    spec: InterventionSpec, *, operation: str, controls: frozenset[str]
) -> None:
    if spec.method != operation:
        raise ValueError(f"{operation} operation requires method={operation!r}")
    if spec.control_name not in controls:
        raise ValueError(
            f"{operation} operation does not support control={spec.control_name!r}"
        )


def _roll_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return values[-1:] + values[:-1]


def _capture_layer_output(
    model: nn.Module, layer: nn.Module, model_inputs: Mapping[str, object]
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        captured["hidden"] = _hidden(output).detach().clone()

    handle = layer.register_forward_hook(hook)
    try:
        model(**model_inputs)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("donor residual capture hook did not run")
    return captured["hidden"]


def _control_source(
    control_name: str,
    recipient: torch.Tensor,
    donor: torch.Tensor | None,
    positions: tuple[int, ...],
    batch: Mapping[str, object],
    validation_mean: torch.Tensor | None,
) -> torch.Tensor:
    if control_name in {"donor", "patch", "kv_transplant"}:
        if donor is None:
            raise ValueError("donor control requires donor residuals")
        return donor
    if control_name == "zero":
        return torch.zeros_like(recipient)
    if control_name == "validation_mean":
        if validation_mean is None:
            raise ValueError("validation_mean control requires validation_mean")
        mean = validation_mean.to(device=recipient.device, dtype=recipient.dtype)
        if mean.ndim == 1 and mean.shape[0] == recipient.shape[-1]:
            return mean.view(1, 1, -1).expand_as(recipient)
        if mean.ndim == 3 and mean.shape == recipient.shape:
            return mean
        raise AlignmentError("validation_mean must have shape [hidden] or match residuals")
    if control_name == "within_batch_shuffle":
        if recipient.shape[0] < 2:
            raise AlignmentError("within-batch shuffle requires at least two examples")
        return recipient.roll(1, dims=0)
    if control_name == "random_donor":
        if donor is None or donor.shape[0] < 2:
            raise AlignmentError("random donor control requires at least two donor examples")
        return donor.roll(1, dims=0)
    if control_name == "matched_non_marker_position":
        matched = batch.get("matched_positions")
        if not isinstance(matched, Sequence) or isinstance(matched, (str, bytes)):
            raise AlignmentError("matched non-marker control requires matched_positions")
        matched_positions = tuple(matched)
        if len(matched_positions) != len(positions):
            raise AlignmentError("matched_positions must align one-to-one with positions")
        _validate_positions(matched_positions, recipient.shape[1])
        source = recipient.clone()
        source[:, positions, :] = recipient[:, matched_positions, :]
        return source
    raise ValueError(f"unknown intervention control {control_name!r}")


def _batch_parts(
    batch: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(batch, Mapping):
        raise TypeError("intervention batch must be a mapping")
    model_inputs = {key: value for key, value in batch.items() if key not in _METADATA_KEYS}
    batch_size = _batch_size(model_inputs)
    raw_ids = batch.get("example_ids")
    if isinstance(raw_ids, str):
        ids = (raw_ids,)
    elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (bytes, bytearray)):
        ids = tuple(raw_ids)
    else:
        explicit = batch.get("recipient_id") or batch.get("donor_id")
        ids = (explicit,) if isinstance(explicit, str) else ()
    if len(ids) != batch_size or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("example_ids must contain one nonempty id per example")
    raw_parents = batch.get("parent_hashes", ())
    if isinstance(raw_parents, str):
        parents = (raw_parents,)
    elif isinstance(raw_parents, Sequence) and not isinstance(raw_parents, (bytes, bytearray)):
        parents = tuple(raw_parents)
    else:
        raise TypeError("parent_hashes must be a sequence of strings")
    if any(not isinstance(value, str) or not value for value in parents):
        raise ValueError("parent_hashes must contain nonempty strings")
    return model_inputs, ids, parents


def _batch_size(model_inputs: Mapping[str, object]) -> int:
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("batch input_ids must have shape [batch, sequence]")
    return input_ids.shape[0]


def _validate_aligned_inputs(
    recipient: Mapping[str, object], donor: Mapping[str, object]
) -> None:
    recipient_ids = recipient.get("input_ids")
    donor_ids = donor.get("input_ids")
    assert isinstance(recipient_ids, torch.Tensor) and isinstance(donor_ids, torch.Tensor)
    if recipient_ids.shape != donor_ids.shape:
        raise AlignmentError("donor and recipient input sequences must be aligned")
    if recipient_ids.dtype != donor_ids.dtype or recipient_ids.device != donor_ids.device:
        raise AlignmentError("donor and recipient input ids must share dtype and device")
    for field in ("attention_mask", "position_ids", "cache_position"):
        recipient_value = recipient.get(field)
        donor_value = donor.get(field)
        if (recipient_value is None) != (donor_value is None):
            raise AlignmentError(f"{field} presence must match for donor and recipient")
        if recipient_value is None:
            continue
        if not isinstance(recipient_value, torch.Tensor) or not isinstance(
            donor_value, torch.Tensor
        ):
            raise AlignmentError(f"{field} must be a tensor on both aligned batches")
        if recipient_value.shape != donor_value.shape:
            raise AlignmentError(f"{field} shapes differ")
        if recipient_value.dtype != donor_value.dtype:
            raise AlignmentError(f"{field} dtypes differ")
        if recipient_value.device != donor_value.device:
            raise AlignmentError(f"{field} devices differ")
        if recipient_value.device != recipient_ids.device:
            raise AlignmentError(f"{field} must share the input_ids device")
        if field == "cache_position":
            if recipient_value.ndim != 1 or recipient_value.shape[0] != recipient_ids.shape[1]:
                raise AlignmentError("cache_position must have shape [sequence]")
        elif recipient_value.shape != recipient_ids.shape:
            raise AlignmentError(f"{field} must align exactly with input_ids shape")
        if not torch.equal(recipient_value, donor_value):
            raise AlignmentError(f"{field} values differ")


def _batch_target_ids(
    batch: Mapping[str, object], batch_size: int, label: str
) -> torch.Tensor:
    value = batch.get("target_token_ids")
    if value is None:
        raise ValueError(f"{label}_batch['target_token_ids'] is required")
    return _target_ids(value, batch_size, label)


def _target_ids(value: object, batch_size: int, label: str) -> torch.Tensor:
    ids = torch.as_tensor(value, dtype=torch.long)
    if ids.ndim != 1 or ids.numel() != batch_size:
        raise AlignmentError(f"{label} target_token_ids must contain one token per example")
    if ids.min().item() < 0:
        raise ValueError(f"{label} target token ids must be nonnegative")
    return ids


def _hidden(output: object) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("decoder layer output must contain rank-3 hidden states")
    return hidden


def _with_hidden(output: object, hidden: torch.Tensor) -> object:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


def _logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("causal LM output must expose logits [batch, sequence, vocab]")
    return logits


def _result(
    baseline: torch.Tensor,
    intervened: torch.Tensor,
    spec: InterventionSpec,
    recipient_ids: tuple[str, ...],
    donor_ids: tuple[str, ...],
    parent_hashes: tuple[str, ...],
    target_ids: torch.Tensor,
    donor_target_ids: torch.Tensor | None,
    *,
    source_positions: tuple[int, ...] | None = None,
    control_source_hashes: tuple[str | None, ...] | None = None,
) -> InterventionResult:
    baseline_logprobs, baseline_ranks, baseline_correct = _score_target(baseline, target_ids)
    intervened_logprobs, intervened_ranks, intervened_correct = _score_target(
        intervened, target_ids
    )
    baseline_donor_ranks = (
        None if donor_target_ids is None else _score_target(baseline, donor_target_ids)[1]
    )
    intervened_donor_ranks = (
        None if donor_target_ids is None else _score_target(intervened, donor_target_ids)[1]
    )
    if len(recipient_ids) != len(donor_ids) or len(recipient_ids) != target_ids.numel():
        raise AlignmentError("record provenance must align one-to-one with batch examples")
    if control_source_hashes is None:
        control_source_hashes = (None,) * len(recipient_ids)
    if len(control_source_hashes) != len(recipient_ids):
        raise AlignmentError("control source hashes must align one-to-one with records")
    records: list[InterventionRecord] = []
    for index, (recipient_id, donor_id) in enumerate(
        zip(recipient_ids, donor_ids, strict=True)
    ):
        recipient_target_id = int(target_ids[index].item())
        donor_target_id = (
            None if donor_target_ids is None else int(donor_target_ids[index].item())
        )
        control_source_hash = control_source_hashes[index]
        payload = {
            "recipient_id": recipient_id,
            "donor_id": donor_id,
            "method": spec.method,
            "control_name": spec.control_name,
            "layers": spec.layers,
            "positions": spec.positions,
            "source_positions": source_positions,
            "control_source_hash": control_source_hash,
            "norm_match": spec.norm_match,
            "recipient_target_token_id": recipient_target_id,
            "donor_target_token_id": donor_target_id,
            "parent_hashes": parent_hashes,
        }
        records.append(
            InterventionRecord(
                intervention_id=sha256_json(payload),
                recipient_id=recipient_id,
                donor_id=donor_id,
                method=spec.method,
                control_name=spec.control_name,
                layers=spec.layers,
                positions=spec.positions,
                source_positions=source_positions,
                control_source_hash=control_source_hash,
                baseline_target_logprob=float(baseline_logprobs[index].item()),
                intervened_target_logprob=float(intervened_logprobs[index].item()),
                baseline_target_rank=int(baseline_ranks[index].item()),
                intervened_target_rank=int(intervened_ranks[index].item()),
                baseline_donor_target_rank=(
                    None
                    if baseline_donor_ranks is None
                    else int(baseline_donor_ranks[index].item())
                ),
                intervened_donor_target_rank=(
                    None
                    if intervened_donor_ranks is None
                    else int(intervened_donor_ranks[index].item())
                ),
                baseline_correct=bool(baseline_correct[index].item()),
                intervened_correct=bool(intervened_correct[index].item()),
                parent_hashes=parent_hashes,
            )
        )
    return InterventionResult(
        records=tuple(records),
        baseline_logits=baseline.detach().cpu().clone(),
        intervened_logits=intervened.detach().cpu().clone(),
    )


def _effective_tensor_hash(value: torch.Tensor) -> str:
    effective = value.detach().contiguous().cpu()
    header = canonical_json(
        {
            "dtype": str(effective.dtype).removeprefix("torch."),
            "shape": list(effective.shape),
        }
    ).encode("utf-8")
    raw_values = effective.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw_values).hexdigest()


def _score_target(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_logits = logits[:, -1, :].float()
    ids = target_ids.to(next_logits.device)
    if ids.max().item() >= next_logits.shape[-1]:
        raise IndexError("target token id is outside model vocabulary")
    selected = next_logits.gather(1, ids[:, None]).squeeze(1)
    logprobs = next_logits.log_softmax(dim=-1).gather(1, ids[:, None]).squeeze(1)
    ranks = 1 + (next_logits > selected[:, None]).sum(dim=-1)
    return (
        logprobs.detach().cpu(),
        ranks.detach().cpu(),
        (next_logits.argmax(dim=-1) == ids).detach().cpu(),
    )


def _validate_positions(positions: tuple[int, ...], sequence_length: int) -> None:
    if not positions:
        raise ValueError("positions must be nonempty")
    for position in positions:
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("positions must contain nonnegative integers")
        if position >= sequence_length:
            raise IndexError(
                f"position {position} is outside actual sequence length {sequence_length}"
            )


def _cache_pairs(cache: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = getattr(cache, "layers", None)
    if isinstance(layers, Sequence):
        pairs = []
        for layer in layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError("DynamicCache layers must expose tensor keys and values")
            pairs.append((key, value))
        return pairs
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if isinstance(keys, Sequence) and isinstance(values, Sequence):
        if len(keys) != len(values):
            raise AlignmentError("cache key/value layer counts differ")
        if not all(isinstance(value, torch.Tensor) for value in (*keys, *values)):
            raise TypeError("DynamicCache key_cache/value_cache must contain tensors")
        return list(zip(keys, values, strict=True))
    raise TypeError("cache must be DynamicCache-compatible")


def _validate_cache_pair(
    recipient: tuple[torch.Tensor, torch.Tensor], donor: tuple[torch.Tensor, torch.Tensor]
) -> None:
    for label, recipient_tensor, donor_tensor in (
        ("key", recipient[0], donor[0]),
        ("value", recipient[1], donor[1]),
    ):
        if recipient_tensor.ndim != 4 or donor_tensor.ndim != 4:
            raise AlignmentError(f"cache {label} dimensions must be rank four")
        if recipient_tensor.shape[0] != donor_tensor.shape[0]:
            raise AlignmentError("cache batch sizes differ")
        if recipient_tensor.shape[1] != donor_tensor.shape[1]:
            raise AlignmentError("cache attention heads differ")
        if recipient_tensor.shape[-1] != donor_tensor.shape[-1]:
            raise AlignmentError("cache head dimensions differ")
        if recipient_tensor.shape[-2] != donor_tensor.shape[-2]:
            raise AlignmentError("cache sequence alignment differs")
        if recipient_tensor.dtype != donor_tensor.dtype:
            raise AlignmentError("cache dtype differs")
        if recipient_tensor.device != donor_tensor.device:
            raise AlignmentError("cache device differs")
    if recipient[0].shape != recipient[1].shape or donor[0].shape != donor[1].shape:
        raise AlignmentError("cache key/value dimensions differ")


def _prefix_and_query(
    model_inputs: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    input_ids = model_inputs["input_ids"]
    assert isinstance(input_ids, torch.Tensor)
    if input_ids.shape[1] < 2:
        raise AlignmentError("KV transplantation requires prefix plus query token")
    prefix: dict[str, object] = {}
    query: dict[str, object] = {}
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == input_ids.shape[1]:
            prefix[key] = value[:-1].clone()
            query[key] = value[-1:].clone()
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == input_ids.shape[1]:
            prefix[key] = value[:, :-1].clone()
            query[key] = value.clone() if key == "attention_mask" else value[:, -1:].clone()
        else:
            prefix[key] = value
            query[key] = value
    return prefix, query


class _SmokeLayer(nn.Module):
    def __init__(self, hidden_size: int, *, mix: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.mix = mix

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        if self.mix:
            hidden = hidden + hidden.cumsum(dim=1) / hidden.shape[1]
        return (hidden + torch.tanh(self.projection(hidden)),)


class _SmokeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 2)
        self.layers = nn.ModuleList((_SmokeLayer(2, mix=False), _SmokeLayer(2, mix=True)))
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


class _SmokeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _SmokeBackbone()
        self.lm_head = nn.Linear(2, 17, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()
            embeddings = {
                1: (1.0, 3.0),
                2: (2.0, 4.0),
                3: (1.0, 1.0),
                4: (2.0, 2.0),
                7: (4.0, 1.0),
                8: (5.0, 2.0),
                9: (1.0, 5.0),
                10: (2.0, 6.0),
                11: (1.0, 7.0),
                12: (2.0, 8.0),
            }
            for token_id, value in embeddings.items():
                self.model.embed_tokens.weight[token_id] = torch.tensor(value)
            self.lm_head.weight.zero_()
            self.lm_head.weight[4, 0] = 1.0

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(self.model(**kwargs).last_hidden_state))


def _smoke(output_root: Path) -> dict[str, object]:
    torch.manual_seed(101)
    model = _SmokeModel().eval()
    recipient = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "example_ids": ("smoke:recipient",),
        "parent_hashes": ("1" * 64,),
        "target_token_ids": torch.tensor([4]),
        "matched_positions": (0, 1),
    }
    donor = {
        "input_ids": torch.tensor([[1, 2, 7, 8, 5, 6]]),
        "example_ids": ("smoke:donor",),
        "parent_hashes": ("2" * 64,),
        "target_token_ids": torch.tensor([11]),
    }
    selected = patch_residual_positions(
        model, recipient, donor, InterventionSpec("residual_patch", (0,), (2, 3), False, (4,), "donor")
    )
    random_recipient = {
        **recipient,
        "input_ids": recipient["input_ids"].repeat(2, 1),
        "example_ids": ("smoke:recipient-a", "smoke:recipient-b"),
        "target_token_ids": torch.tensor([4, 4]),
    }
    random_donor = {
        **random_recipient,
        "input_ids": torch.tensor(
            [[1, 2, 9, 10, 5, 6], [1, 2, 11, 12, 5, 6]]
        ),
        "example_ids": ("smoke:random-a", "smoke:random-b"),
        "parent_hashes": ("3" * 64,),
    }
    random_control = patch_residual_positions(
        model,
        random_recipient,
        random_donor,
        InterventionSpec(
            "residual_patch", (0,), (2, 3), False, (4, 4), "random_donor"
        ),
    )
    matched = ablate_positions(
        model,
        recipient,
        InterventionSpec("ablate", (0,), (2, 3), False, (4,), "matched_non_marker_position"),
    )
    layer = model.model.layers[0]
    random_recipient_state = _capture_layer_output(
        model, layer, {"input_ids": random_recipient["input_ids"]}
    )
    random_donor_state = _capture_layer_output(
        model, layer, {"input_ids": random_donor["input_ids"]}
    )
    random_source = _control_source(
        "random_donor",
        random_recipient_state,
        random_donor_state,
        (2, 3),
        random_recipient,
        None,
    )
    recipient_state = _capture_layer_output(
        model, layer, {"input_ids": recipient["input_ids"]}
    )
    matched_source = _control_source(
        "matched_non_marker_position",
        recipient_state,
        None,
        (2, 3),
        recipient,
        None,
    )
    random_delta = sum(
        record.intervened_target_logprob - record.baseline_target_logprob
        for record in random_control.records
    ) / len(random_control.records)
    metrics = {
        "selected_target_logprob_delta": selected.record.intervened_target_logprob
        - selected.record.baseline_target_logprob,
        "random_control_target_logprob_delta": random_delta,
        "non_marker_control_target_logprob_delta": matched.record.intervened_target_logprob
        - matched.record.baseline_target_logprob,
        "random_control_replacement_changed": not torch.equal(
            random_source[:, (2, 3), :], random_recipient_state[:, (2, 3), :]
        ),
        "non_marker_control_replacement_changed": not torch.equal(
            matched_source[:, (2, 3), :], recipient_state[:, (2, 3), :]
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "records.jsonl"
    records = tuple(
        record
        for result in (selected, random_control, matched)
        for record in result.records
    )
    records_path.write_text(
        "".join(canonical_json(asdict(record)) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "kind": "phase_marker_intervention_smoke",
        "model": "tiny_local_torch_qwen_layout",
        "device": "cpu",
        "records_file": records_path.name,
        "metrics": metrics,
        "parent_hashes": ["1" * 64, "2" * 64, "3" * 64],
    }
    manifest["artifact_id"] = sha256_json(manifest)
    (output_root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded phase-marker interventions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run a tiny local CPU intervention smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _smoke(args.output_root)
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
