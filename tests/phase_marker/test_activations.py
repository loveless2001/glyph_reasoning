from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import hashlib
import json

import pytest
import torch
from torch import nn

from phase_marker.activations import (
    ActivationBatch,
    CaptureSpec,
    apply_logit_lens,
    capture_context,
    capture_selected_states,
    evaluate_phase_probe,
    fit_phase_probe,
    save_activation_artifact,
)
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.variance_epsilon) * self.weight


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.raise_on_forward = False

    def forward(
        self, hidden_states: torch.Tensor, *, output_attentions: bool = False
    ) -> tuple[torch.Tensor, ...]:
        if self.raise_on_forward:
            raise RuntimeError("tiny layer failure")
        hidden_states = hidden_states + torch.tanh(self.projection(hidden_states))
        if not output_attentions:
            return (hidden_states,)
        scores = hidden_states @ hidden_states.transpose(-1, -2)
        attention = scores.softmax(dim=-1).unsqueeze(1)
        return hidden_states, attention


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(TinyDecoderLayer(hidden_size) for _ in range(layers))
        self.norm = TinyRMSNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> SimpleNamespace:
        del attention_mask
        hidden_states = self.embed_tokens(input_ids)
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            outputs = layer(hidden_states, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                attentions.append(outputs[1])
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden_states),
            attentions=tuple(attentions) if output_attentions else None,
        )


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = TinyBackbone(vocab_size=13, hidden_size=8, layers=3)
        self.lm_head = nn.Linear(8, 13, bias=False)

    def forward(self, **kwargs: object) -> SimpleNamespace:
        outputs = self.model(**kwargs)
        return SimpleNamespace(
            logits=self.lm_head(outputs.last_hidden_state),
            attentions=outputs.attentions,
        )


@pytest.fixture
def tiny_causal_lm() -> TinyCausalLM:
    model = TinyCausalLM()
    model.eval()
    return model


@pytest.fixture
def tiny_batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "example_ids": ("synthetic:a", "synthetic:b"),
        "conditions": ("glyph", "glyph"),
        "parent_hashes": ("a" * 64, "b" * 64),
    }


def test_capture_returns_only_requested_layers_and_positions(tiny_causal_lm, tiny_batch):
    spec = CaptureSpec(layers=(0, 2), positions=(3, 7))

    captured = capture_selected_states(tiny_causal_lm, tiny_batch, spec)

    assert captured.residual.shape == (2, 2, 2, 8)
    assert captured.layers == (0, 2)
    assert captured.positions == (3, 7)
    assert captured.example_ids == ("synthetic:a", "synthetic:b")
    assert captured.parent_hashes == ("a" * 64, "b" * 64)


def test_capture_does_not_change_logits(tiny_causal_lm, tiny_batch):
    model_inputs = {key: tiny_batch[key] for key in ("input_ids", "attention_mask")}
    baseline = tiny_causal_lm(**model_inputs).logits

    with capture_context(tiny_causal_lm, CaptureSpec(layers=(1,), positions=(2,))):
        observed = tiny_causal_lm(**model_inputs).logits

    torch.testing.assert_close(baseline, observed, rtol=0, atol=0)


def test_capture_context_removes_hooks_when_forward_raises(tiny_causal_lm, tiny_batch):
    layer = tiny_causal_lm.model.layers[1]
    layer.raise_on_forward = True
    model_inputs = {key: tiny_batch[key] for key in ("input_ids", "attention_mask")}

    with pytest.raises(RuntimeError, match="tiny layer failure"):
        with capture_context(
            tiny_causal_lm, CaptureSpec(layers=(1,), positions=(2,))
        ):
            tiny_causal_lm(**model_inputs)

    assert not layer._forward_hooks


def test_capture_rejects_selected_position_outside_actual_sequence(tiny_causal_lm, tiny_batch):
    with pytest.raises(IndexError, match="position 8.*sequence length 8"):
        capture_selected_states(
            tiny_causal_lm, tiny_batch, CaptureSpec(layers=(0,), positions=(8,))
        )
    assert all(not layer._forward_hooks for layer in tiny_causal_lm.model.layers)


def test_attention_capture_keeps_only_selected_query_positions(tiny_causal_lm, tiny_batch):
    captured = capture_selected_states(
        tiny_causal_lm,
        tiny_batch,
        CaptureSpec(
            layers=(0, 2),
            positions=(1, 6),
            capture_residual=False,
            capture_attention=True,
        ),
    )

    assert captured.residual.numel() == 0
    assert captured.attention.shape == (2, 2, 2, 1, 8)


def test_logit_lens_applies_final_norm_and_records_candidate_ranks(tiny_causal_lm):
    with torch.no_grad():
        tiny_causal_lm.lm_head.weight.zero_()
        tiny_causal_lm.lm_head.weight[4, 0] = 2.0
        tiny_causal_lm.lm_head.weight[5, 0] = 1.0
        tiny_causal_lm.lm_head.weight[6, 0] = -1.0
    activation = ActivationBatch(
        example_ids=("synthetic:a",),
        conditions=("glyph",),
        layers=(2,),
        positions=(3,),
        residual=torch.tensor([[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]),
        attention=torch.empty(0),
        parent_hashes=("a" * 64,),
    )

    lens = apply_logit_lens(tiny_causal_lm, activation, candidate_token_ids=(4, 5, 6))

    assert lens.token_ids.tolist() == [4, 5, 6]
    assert lens.ranks[0, 0, 0].tolist() == [1, 2, 13]
    assert lens.logprobs.shape == (1, 1, 1, 3)
    assert lens.parent_hashes == activation.parent_hashes


def test_logit_lens_accepts_known_target_token_per_phase_and_example(tiny_causal_lm):
    with torch.no_grad():
        tiny_causal_lm.lm_head.weight.zero_()
        tiny_causal_lm.lm_head.weight[4, 0] = 2.0
        tiny_causal_lm.lm_head.weight[6, 0] = -2.0
    activation = ActivationBatch(
        example_ids=("synthetic:a",),
        conditions=("glyph",),
        layers=(2,),
        positions=(3, 7),
        residual=torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
              [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]
        ),
        attention=torch.empty(0),
        parent_hashes=("a" * 64,),
    )

    lens = apply_logit_lens(
        tiny_causal_lm, activation, candidate_token_ids=torch.tensor([[4], [6]])
    )

    assert lens.token_ids.shape == lens.ranks.shape == lens.logprobs.shape == (1, 2, 1)
    assert lens.token_ids.tolist() == [[[4], [6]]]
    assert lens.ranks.tolist() == [[[1], [1]]]


def test_activation_artifact_manifest_binds_mode_shapes_ids_and_parents(
    tiny_causal_lm, tiny_batch, tmp_path: Path
):
    captured = capture_selected_states(
        tiny_causal_lm, tiny_batch, CaptureSpec(layers=(0,), positions=(3,))
    )

    manifest = save_activation_artifact(captured, tmp_path, mode="teacher_forced")
    payload = torch.load(tmp_path / "selected-states.pt", weights_only=True)

    assert manifest["mode"] == "teacher_forced"
    assert manifest["example_ids"] == ["synthetic:a", "synthetic:b"]
    assert manifest["layers"] == [0]
    assert manifest["positions"] == [3]
    assert manifest["parent_hashes"] == ["a" * 64, "b" * 64]
    assert manifest["tensors"]["residual"] == {
        "shape": [1, 1, 2, 8],
        "dtype": "float32",
    }
    assert tuple(payload["residual"].shape) == (1, 1, 2, 8)
    assert (tmp_path / "manifest.json").is_file()


def test_capture_cli_rejects_tiny_backend_without_explicit_opt_in(tmp_path: Path):
    from phase_marker.activations import main

    with pytest.raises(ValueError, match="allow-test-backend"):
        main(
            (
                "capture", "--config", "configs/phase-marker-qwen25-7b.toml",
                "--mode", "teacher_forced", "--validation-selection-manifest", "missing.json",
                "--tokenized-batch-manifest", "missing.json", "--tokenized-batch", "missing.pt",
                "--model-id", "Qwen/Qwen2.5-7B-Instruct", "--model-revision", "deadbeef",
                "--checkpoint-manifest", "missing.json", "--behavior-manifest", "missing.json",
                "--synthetic-manifest", "missing.json", "--backend", "tiny-fixture",
                "--output-root", str(tmp_path),
            )
        )


def test_capture_cli_tiny_fixture_emits_plumbing_only_envelope(tmp_path: Path):
    from phase_marker.activations import main

    config_path = Path("configs/phase-marker-qwen25-7b.toml")
    config = ExperimentConfig.load(config_path)
    config_hash = sha256_json(asdict(config))

    def parent(name: str, **values: object) -> Path:
        payload = {"schema_version": 1, "kind": name, "config_hash": config_hash, **values}
        payload["artifact_id"] = sha256_json(payload)
        path = tmp_path / f"{name}.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return path

    batch_path = tmp_path / "batch.pt"
    batch_path.write_bytes(b"tiny fixture batch")
    selection = parent("selection", selected_on="validation")
    checkpoint = parent(
        "checkpoint", model_id=config.model_id, model_revision="deadbeef", checkpoint_path="/fixture"
    )
    behavior = parent("behavior", evidence_scope="experiment_candidate")
    synthetic = parent("synthetic", evidence_scope="experiment")
    batch = parent(
        "tokenized_batch", batch_file=str(batch_path),
        batch_hash=hashlib.sha256(batch_path.read_bytes()).hexdigest(), layers=[0], positions=[1]
    )
    output = tmp_path / "capture"
    assert main(
        (
            "capture", "--config", str(config_path), "--mode", "teacher_forced",
            "--validation-selection-manifest", str(selection), "--tokenized-batch-manifest", str(batch),
            "--tokenized-batch", str(batch_path), "--model-id", config.model_id,
            "--model-revision", "deadbeef", "--checkpoint-manifest", str(checkpoint),
            "--behavior-manifest", str(behavior), "--synthetic-manifest", str(synthetic),
            "--backend", "tiny-fixture", "--allow-test-backend", "--output-root", str(output),
        )
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["evidence_scope"] == "plumbing_only"
    assert manifest["backend"] == "tiny-fixture"


def _probe_batch(condition: str, *, informative: bool) -> ActivationBatch:
    residual = torch.zeros(2, 2, 4, 2)
    residual[0, :, :, 0] = 1.0
    if informative:
        residual[1, 0, :, 0] = 2.0
        residual[1, 1, :, 1] = 2.0
    return ActivationBatch(
        example_ids=tuple(f"{condition}:{index}" for index in range(4)),
        conditions=(condition,) * 4,
        layers=(0, 2),
        positions=(3, 7),
        residual=residual,
        attention=torch.empty(0),
        parent_hashes=("c" * 64,),
    )


def test_phase_probe_selects_on_validation_and_transfers_without_refit():
    train = _probe_batch("glyph", informative=True)
    validation = _probe_batch("glyph", informative=True)
    dot_test = _probe_batch("dot", informative=True)
    unseen_test = _probe_batch("unseen_symbol", informative=True)

    probe = fit_phase_probe(train, validation, seed=101)
    original_weight = probe.weight.clone()
    original_bias = probe.bias.clone()
    dot_metrics = evaluate_phase_probe(probe, dot_test)
    unseen_metrics = evaluate_phase_probe(probe, unseen_test)

    assert probe.layer == 2
    assert probe.source_condition == "glyph"
    assert dot_metrics.accuracy == dot_metrics.macro_f1 == 1.0
    assert unseen_metrics.accuracy == unseen_metrics.macro_f1 == 1.0
    assert dot_metrics.target_condition == "dot"
    assert unseen_metrics.target_condition == "unseen_symbol"
    torch.testing.assert_close(probe.weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(probe.bias, original_bias, rtol=0, atol=0)
