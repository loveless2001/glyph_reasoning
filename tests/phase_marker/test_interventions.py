from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from phase_marker.interventions import (
    AlignmentError,
    InterventionSpec,
    ablate_positions,
    patch_residual_positions,
    replace_positions,
    transplant_cache_rows,
    transplant_kv_positions,
)


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.raise_on_call: int | None = None
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        self.calls += 1
        if self.raise_on_call == self.calls:
            raise RuntimeError("intervention forward failure")
        mixed = hidden_states + hidden_states.cumsum(dim=1) / hidden_states.shape[1]
        return (mixed + torch.tanh(self.projection(mixed)),)


class TinyNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 6)
        self.layers = nn.ModuleList(TinyLayer(6) for _ in range(2))
        self.norm = TinyNorm()

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(6, 17, bias=False)

    def forward(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(self.model(**kwargs).last_hidden_state))


class FakeDynamicCache:
    def __init__(self, layers: list[SimpleNamespace]) -> None:
        self.layers = layers


def fake_cache(
    *, length: int = 6, batch: int = 1, heads: int = 2, width: int = 3,
    dtype: torch.dtype = torch.float32,
) -> FakeDynamicCache:
    layers = []
    for index in range(2):
        base = torch.arange(batch * heads * length * width, dtype=dtype).reshape(
            batch, heads, length, width
        ) + 100 * index
        layers.append(SimpleNamespace(keys=base.clone(), values=(base + 0.5).clone()))
    return FakeDynamicCache(layers)


class TinyCacheLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(13, 4)
        self.head = nn.Linear(4, 13, bias=False)
        torch.manual_seed(19)
        nn.init.normal_(self.embed.weight)
        nn.init.normal_(self.head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: FakeDynamicCache | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise AlignmentError("query position_ids were not sliced with input_ids")
        hidden = self.embed(input_ids)
        if past_key_values is None:
            values = hidden[:, None, :, :]
            cache = FakeDynamicCache(
                [SimpleNamespace(keys=values.clone(), values=values.clone()) for _ in range(2)]
            )
            context = hidden.cumsum(dim=1)
        else:
            cache = past_key_values
            context = hidden + sum(
                layer.values.sum(dim=-2).squeeze(1)[:, None, :]
                for layer in past_key_values.layers
            )
        return SimpleNamespace(
            logits=self.head(context),
            past_key_values=cache if use_cache else None,
        )


@pytest.fixture
def recipient_batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long),
        "example_ids": ("recipient:one",),
        "parent_hashes": ("r" * 64,),
        "target_token_ids": torch.tensor([4]),
        "matched_positions": (0, 1),
    }


@pytest.fixture
def donor_batch() -> dict[str, object]:
    return {
        "input_ids": torch.tensor([[6, 5, 4, 3, 2, 1]], dtype=torch.long),
        "example_ids": ("donor:one",),
        "parent_hashes": ("d" * 64,),
        "target_token_ids": torch.tensor([7]),
    }


def spec(**overrides: object) -> InterventionSpec:
    values = {
        "method": "residual_patch",
        "layers": (0,),
        "positions": (2, 3),
        "norm_match": False,
        "target_token_ids": (4,),
        "control_name": "donor",
    }
    values.update(overrides)
    return InterventionSpec(**values)


def test_intervention_spec_is_frozen_and_rejects_unauditable_multi_layer_calls():
    intervention = spec()
    with pytest.raises(FrozenInstanceError):
        intervention.method = "zero"  # type: ignore[misc]
    with pytest.raises(ValueError, match="exactly one layer"):
        spec(layers=(0, 1))


def test_residual_patch_changes_only_selected_positions_bit_for_bit():
    recipient = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3)
    donor = recipient.neg()

    patched = replace_positions(recipient, donor, positions=(2, 5))

    assert torch.equal(patched[:, [0, 1, 3, 4]], recipient[:, [0, 1, 3, 4]])
    assert torch.equal(patched[:, [2, 5]], donor[:, [2, 5]])
    assert torch.equal(recipient, torch.arange(36, dtype=torch.float32).reshape(2, 6, 3))


def test_norm_matching_preserves_recipient_selected_row_mean_and_centered_norm():
    recipient = torch.tensor([[[10.0, 12.0, 14.0], [1.0, 2.0, 3.0]]])
    donor = torch.tensor([[[2.0, 8.0, 5.0], [9.0, 8.0, 7.0]]])

    patched = replace_positions(recipient, donor, positions=(0,), norm_match=True)

    assert patched[:, 0].mean(dim=-1) == recipient[:, 0].mean(dim=-1)
    torch.testing.assert_close(
        (patched[:, 0] - patched[:, 0].mean(dim=-1, keepdim=True)).norm(dim=-1),
        (recipient[:, 0] - recipient[:, 0].mean(dim=-1, keepdim=True)).norm(dim=-1),
    )
    assert torch.equal(patched[:, 1], recipient[:, 1])


def test_patch_records_metrics_transfer_ranks_and_exact_provenance_without_mutation(
    recipient_batch, donor_batch
):
    model = TinyCausalLM().eval()
    recipient_before = recipient_batch["input_ids"].clone()
    donor_before = donor_batch["input_ids"].clone()

    result = patch_residual_positions(model, recipient_batch, donor_batch, spec())

    assert result.record.recipient_id == "recipient:one"
    assert result.record.donor_id == "donor:one"
    assert result.record.layers == (0,)
    assert result.record.positions == (2, 3)
    assert result.record.control_name == "donor"
    assert result.record.parent_hashes == ("r" * 64, "d" * 64)
    assert result.record.baseline_donor_target_rank is not None
    assert result.record.intervened_donor_target_rank is not None
    assert result.record.intervention_id
    assert not torch.equal(result.baseline_logits, result.intervened_logits)
    assert torch.equal(recipient_batch["input_ids"], recipient_before)
    assert torch.equal(donor_batch["input_ids"], donor_before)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_ablation_zero_and_validation_mean_share_record_path(recipient_batch):
    model = TinyCausalLM().eval()
    zero = ablate_positions(
        model, recipient_batch, spec(method="ablate", control_name="zero")
    )
    mean = ablate_positions(
        model,
        recipient_batch,
        spec(method="ablate", control_name="validation_mean"),
        validation_mean=torch.full((6,), 0.25),
    )

    assert zero.record.donor_id == "control:zero"
    assert zero.record.baseline_donor_target_rank is None
    assert zero.record.intervened_donor_target_rank is None
    assert mean.record.control_name == "validation_mean"
    assert zero.record.parent_hashes == mean.record.parent_hashes == ("r" * 64,)
    assert not torch.equal(zero.intervened_logits, mean.intervened_logits)


def test_shuffle_and_random_donor_controls_use_the_same_intervention_path():
    model = TinyCausalLM().eval()
    recipient = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "example_ids": ("recipient:a", "recipient:b"),
        "parent_hashes": ("r" * 64,),
        "target_token_ids": torch.tensor([4, 5]),
    }
    donor = {
        "input_ids": torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]]),
        "example_ids": ("donor:a", "donor:b"),
        "parent_hashes": ("d" * 64,),
        "target_token_ids": torch.tensor([7, 8]),
    }
    shuffled = ablate_positions(
        model,
        recipient,
        spec(
            method="ablate",
            positions=(1,),
            target_token_ids=(4, 5),
            control_name="within_batch_shuffle",
        ),
    )
    random_donor = patch_residual_positions(
        model,
        recipient,
        donor,
        spec(
            positions=(1,),
            target_token_ids=(4, 5),
            control_name="random_donor",
        ),
    )

    assert shuffled.record.control_name == "within_batch_shuffle"
    assert random_donor.record.control_name == "random_donor"
    assert not torch.equal(shuffled.baseline_logits, shuffled.intervened_logits)
    assert not torch.equal(random_donor.baseline_logits, random_donor.intervened_logits)


def test_hooks_are_removed_when_intervened_forward_raises(recipient_batch, donor_batch):
    model = TinyCausalLM().eval()
    model.model.layers[0].raise_on_call = 3

    with pytest.raises(RuntimeError, match="intervention forward failure"):
        patch_residual_positions(model, recipient_batch, donor_batch, spec())

    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_cache_transplant_clones_recipient_and_changes_only_selected_rows():
    recipient = fake_cache()
    donor = fake_cache()
    donor.layers[1].keys.add_(1000)
    donor.layers[1].values.add_(2000)
    recipient_before = recipient.layers[1].values.clone()

    transplanted = transplant_cache_rows(recipient, donor, positions=(1, 4), layers=(1,))

    assert transplanted is not recipient
    assert torch.equal(recipient.layers[1].values, recipient_before)
    assert torch.equal(transplanted.layers[1].values[:, :, [1, 4]], donor.layers[1].values[:, :, [1, 4]])
    assert torch.equal(transplanted.layers[1].values[:, :, [0, 2, 3, 5]], recipient.layers[1].values[:, :, [0, 2, 3, 5]])
    assert torch.equal(transplanted.layers[0].values, recipient.layers[0].values)


@pytest.mark.parametrize(
    ("recipient", "donor", "message"),
    [
        (fake_cache(length=6), fake_cache(length=5), "sequence alignment"),
        (fake_cache(batch=1), fake_cache(batch=2), "batch"),
        (fake_cache(heads=2), fake_cache(heads=3), "heads"),
        (fake_cache(width=3), fake_cache(width=4), "dimensions"),
        (fake_cache(dtype=torch.float32), fake_cache(dtype=torch.float64), "dtype"),
        (FakeDynamicCache(fake_cache().layers[:1]), fake_cache(), "layer counts"),
    ],
)
def test_cache_transplant_rejects_incompatible_caches(recipient, donor, message):
    with pytest.raises(AlignmentError, match=message):
        transplant_cache_rows(recipient, donor, positions=(1,), layers=(0,))


def test_public_kv_transplant_preserves_inputs_and_records_donor_answer_transfer():
    model = TinyCacheLM().eval()
    recipient = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "example_ids": ("recipient:cache",),
        "parent_hashes": ("a" * 64,),
        "target_token_ids": torch.tensor([2]),
    }
    donor = {
        "input_ids": torch.tensor([[5, 6, 7, 4]], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "example_ids": ("donor:cache",),
        "parent_hashes": ("b" * 64,),
        "target_token_ids": torch.tensor([9]),
    }
    recipient_before = recipient["input_ids"].clone()

    result = transplant_kv_positions(
        model,
        recipient,
        donor,
        spec(method="kv_transplant", layers=(1,), positions=(1,), target_token_ids=(2,)),
    )

    assert result.record.method == "kv_transplant"
    assert result.record.baseline_donor_target_rank is not None
    assert result.record.intervened_donor_target_rank is not None
    assert result.record.parent_hashes == ("a" * 64, "b" * 64)
    assert not torch.equal(result.baseline_logits, result.intervened_logits)
    assert torch.equal(recipient["input_ids"], recipient_before)


def test_smoke_cli_writes_provenance_bound_metrics(tmp_path: Path):
    from phase_marker.interventions import main

    assert main(("smoke", "--output-root", str(tmp_path))) == 0
    manifest = __import__("json").loads((tmp_path / "manifest.json").read_text())

    assert manifest["kind"] == "phase_marker_intervention_smoke"
    assert manifest["model"] == "tiny_local_torch_qwen_layout"
    assert manifest["device"] == "cpu"
    assert manifest["metrics"]["selected_target_logprob_delta"] != 0.0
    assert (tmp_path / "records.jsonl").is_file()
    records = [
        __import__("json").loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert [record["control_name"] for record in records] == [
        "donor",
        "random_donor",
        "matched_non_marker_position",
    ]
