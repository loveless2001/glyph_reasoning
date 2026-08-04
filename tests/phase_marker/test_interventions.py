from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
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
from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json


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

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
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
        cache_position: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: FakeDynamicCache | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise AlignmentError("query position_ids were not sliced with input_ids")
        if cache_position is not None and cache_position.shape != (input_ids.shape[1],):
            raise AlignmentError("cache_position was not sliced with input_ids")
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


def test_norm_matching_rejects_zero_centered_source_for_nonzero_recipient_norm():
    recipient = torch.tensor([[[1.0, 2.0, 3.0]]])
    donor = torch.tensor([[[5.0, 5.0, 5.0]]])

    with pytest.raises(AlignmentError, match="zero-centered source"):
        replace_positions(recipient, donor, positions=(0,), norm_match=True)


def test_norm_matching_allows_both_centered_norms_to_be_zero():
    recipient = torch.tensor([[[2.0, 2.0, 2.0]]])
    donor = torch.tensor([[[5.0, 5.0, 5.0]]])

    patched = replace_positions(recipient, donor, positions=(0,), norm_match=True)

    assert torch.equal(patched, recipient)


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


def test_validation_mean_hash_binds_effective_tensor_and_prevents_id_collision(
    recipient_batch
):
    model = TinyCausalLM().eval()
    first_source = torch.full((6,), 0.25, dtype=torch.float64)
    second_source = torch.full((6,), 0.75, dtype=torch.float64)
    first_before = first_source.clone()
    second_before = second_source.clone()
    intervention = spec(
        method="ablate",
        positions=(2,),
        control_name="validation_mean",
    )

    first = ablate_positions(model, recipient_batch, intervention, first_source)
    second = ablate_positions(model, recipient_batch, intervention, second_source)
    repeated = ablate_positions(model, recipient_batch, intervention, first_source.clone())

    assert not torch.equal(first.intervened_logits, second.intervened_logits)
    assert first.record.intervention_id != second.record.intervention_id
    assert first.record.intervention_id == repeated.record.intervention_id
    assert first.record.control_source_hash == (
        "809fda63375cdcd1d10698ab638ae000e439b14969ad0949c56eb52ac9d5ef06"
    )
    assert first.record.control_source_hash != second.record.control_source_hash
    assert first.record.source_positions is None
    assert torch.equal(first_source, first_before)
    assert torch.equal(second_source, second_before)


def test_matched_positions_are_inspectable_and_prevent_id_collision(recipient_batch):
    model = TinyCausalLM().eval()
    input_before = recipient_batch["input_ids"].clone()
    first_batch = {**recipient_batch, "matched_positions": (0,)}
    second_batch = {**recipient_batch, "matched_positions": (1,)}
    intervention = spec(
        method="ablate",
        positions=(2,),
        control_name="matched_non_marker_position",
    )

    first = ablate_positions(model, first_batch, intervention)
    second = ablate_positions(model, second_batch, intervention)
    repeated = ablate_positions(model, first_batch, intervention)

    assert not torch.equal(first.intervened_logits, second.intervened_logits)
    assert first.record.intervention_id != second.record.intervention_id
    assert first.record.intervention_id == repeated.record.intervention_id
    assert first.record.source_positions == (0,)
    assert second.record.source_positions == (1,)
    assert first.record.control_source_hash is None
    assert torch.equal(recipient_batch["input_ids"], input_before)


def test_controls_without_external_tensor_sources_record_none_source_fields(
    recipient_batch, donor_batch
):
    model = TinyCausalLM().eval()
    donor = patch_residual_positions(model, recipient_batch, donor_batch, spec())
    zero = ablate_positions(
        model,
        recipient_batch,
        spec(method="ablate", control_name="zero"),
    )

    assert donor.record.source_positions is None
    assert donor.record.control_source_hash is None
    assert zero.record.source_positions is None
    assert zero.record.control_source_hash is None


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

    assert {record.control_name for record in shuffled.records} == {
        "within_batch_shuffle"
    }
    assert {record.control_name for record in random_donor.records} == {"random_donor"}
    assert not torch.equal(shuffled.baseline_logits, shuffled.intervened_logits)
    assert not torch.equal(random_donor.baseline_logits, random_donor.intervened_logits)


def test_batch_interventions_emit_one_record_per_actual_random_donor_pair():
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

    result = patch_residual_positions(
        model,
        recipient,
        donor,
        spec(
            positions=(1,),
            target_token_ids=(4, 5),
            control_name="random_donor",
        ),
    )

    assert [record.recipient_id for record in result.records] == [
        "recipient:a",
        "recipient:b",
    ]
    assert [record.donor_id for record in result.records] == ["donor:b", "donor:a"]
    expected_donor_ranks = []
    for row, donor_target in zip(
        result.baseline_logits[:, -1, :], (8, 7), strict=True
    ):
        expected_donor_ranks.append(
            1 + int((row > row[donor_target]).sum().item())
        )
    assert [record.baseline_donor_target_rank for record in result.records] == (
        expected_donor_ranks
    )
    with pytest.raises(ValueError, match="exactly one record"):
        _ = result.record
    assert result.baseline_logits.shape[0] == result.intervened_logits.shape[0] == 2


def test_within_batch_shuffle_records_each_rolled_source_identity():
    model = TinyCausalLM().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "example_ids": ("recipient:a", "recipient:b"),
        "parent_hashes": ("r" * 64,),
        "target_token_ids": torch.tensor([4, 5]),
    }

    result = ablate_positions(
        model,
        batch,
        spec(
            method="ablate",
            positions=(1,),
            target_token_ids=(4, 5),
            control_name="within_batch_shuffle",
        ),
    )

    assert [record.donor_id for record in result.records] == [
        "control:within_batch_shuffle:recipient:b",
        "control:within_batch_shuffle:recipient:a",
    ]
    assert all(record.baseline_donor_target_rank is None for record in result.records)


def test_intervention_ids_bind_norm_targets_and_actual_pairing(recipient_batch, donor_batch):
    model = TinyCausalLM().eval()
    base = patch_residual_positions(model, recipient_batch, donor_batch, spec())
    normed = patch_residual_positions(
        model, recipient_batch, donor_batch, spec(norm_match=True)
    )
    other_recipient_target = patch_residual_positions(
        model, recipient_batch, donor_batch, spec(target_token_ids=(5,))
    )
    other_donor = {**donor_batch, "target_token_ids": torch.tensor([8])}
    other_donor_target = patch_residual_positions(
        model, recipient_batch, other_donor, spec()
    )

    ids = {
        base.record.intervention_id,
        normed.record.intervention_id,
        other_recipient_target.record.intervention_id,
        other_donor_target.record.intervention_id,
    }
    assert len(ids) == 4


def test_public_operations_reject_misleading_method_or_control_labels(
    recipient_batch, donor_batch
):
    residual_model = TinyCausalLM().eval()
    with pytest.raises(ValueError, match="residual_patch.*method"):
        patch_residual_positions(
            residual_model, recipient_batch, donor_batch, spec(method="kv_transplant")
        )
    with pytest.raises(ValueError, match="ablate.*control"):
        ablate_positions(
            residual_model,
            recipient_batch,
            spec(method="ablate", control_name="donor"),
        )

    cache_model = TinyCacheLM().eval()
    recipient = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "example_ids": ("recipient:cache",),
        "parent_hashes": ("a" * 64,),
        "target_token_ids": torch.tensor([2]),
    }
    donor = {
        "input_ids": torch.tensor([[5, 6, 7, 4]]),
        "example_ids": ("donor:cache",),
        "parent_hashes": ("b" * 64,),
        "target_token_ids": torch.tensor([9]),
    }
    with pytest.raises(ValueError, match="kv_transplant.*control"):
        transplant_kv_positions(
            cache_model,
            recipient,
            donor,
            spec(method="kv_transplant", control_name="zero", positions=(1,)),
        )


@pytest.mark.parametrize("field", ["attention_mask", "position_ids"])
def test_donor_recipient_alignment_checks_optional_sequence_tensors(
    recipient_batch, donor_batch, field
):
    model = TinyCausalLM().eval()
    recipient = {**recipient_batch, field: torch.tensor([[0, 1, 1, 1, 1, 1]])}
    donor = {**donor_batch, field: torch.tensor([[1, 1, 1, 1, 1, 1]])}

    with pytest.raises(AlignmentError, match=field):
        patch_residual_positions(model, recipient, donor, spec())


def test_donor_recipient_alignment_requires_optional_tensor_presence_on_both_sides(
    recipient_batch, donor_batch
):
    model = TinyCausalLM().eval()
    recipient = {**recipient_batch, "attention_mask": torch.ones(1, 6, dtype=torch.long)}

    with pytest.raises(AlignmentError, match="attention_mask.*presence"):
        patch_residual_positions(model, recipient, donor_batch, spec())


def test_optional_alignment_tensors_must_share_the_input_device(
    recipient_batch, donor_batch
):
    model = TinyCausalLM().eval()
    meta_mask = torch.ones(1, 6, dtype=torch.long, device="meta")
    recipient = {**recipient_batch, "attention_mask": meta_mask}
    donor = {**donor_batch, "attention_mask": meta_mask.clone()}

    with pytest.raises(AlignmentError, match="attention_mask.*input_ids device"):
        patch_residual_positions(model, recipient, donor, spec())


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
        "cache_position": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "example_ids": ("recipient:cache",),
        "parent_hashes": ("a" * 64,),
        "target_token_ids": torch.tensor([2]),
    }
    donor = {
        "input_ids": torch.tensor([[5, 6, 7, 4]], dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "cache_position": torch.tensor([0, 1, 2, 3], dtype=torch.long),
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
    assert manifest["metrics"]["random_control_replacement_changed"] is True
    assert manifest["metrics"]["non_marker_control_replacement_changed"] is True
    assert abs(manifest["metrics"]["random_control_target_logprob_delta"]) <= 1e-7
    assert abs(manifest["metrics"]["non_marker_control_target_logprob_delta"]) <= 1e-7
    assert (tmp_path / "records.jsonl").is_file()
    records = [
        __import__("json").loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert [record["control_name"] for record in records] == [
        "donor",
        "random_donor",
        "random_donor",
        "matched_non_marker_position",
    ]


def test_run_cli_rejects_tiny_backend_without_explicit_opt_in(tmp_path: Path):
    from phase_marker.interventions import main

    with pytest.raises(ValueError, match="allow-test-backend"):
        main(
            (
                "run", "--config", "configs/phase-marker-qwen25-7b.toml",
                "--validation-selection-manifest", "missing.json",
                "--aligned-pairs-manifest", "missing.json",
                "--activation-manifest", "missing.json", "--checkpoint-manifest", "missing.json",
                "--model-id", "Qwen/Qwen2.5-7B-Instruct", "--model-revision", "deadbeef",
                "--backend", "tiny-fixture", "--output-root", str(tmp_path / "absent-output"),
            )
        )


def test_run_rejects_existing_output_before_any_input_or_loader(tmp_path: Path):
    from phase_marker.interventions import main

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output"):
        main((
            "run", "--config", "missing.toml",
            "--validation-selection-manifest", "missing.json",
            "--aligned-pairs-manifest", "missing.json",
            "--activation-manifest", "missing.json", "--checkpoint-manifest", "missing.json",
            "--model-id", "Qwen/Qwen2.5-7B-Instruct", "--model-revision", "deadbeef",
            "--backend", "hf", "--output-root", str(output),
        ))


def test_run_cli_tiny_fixture_emits_plumbing_only_envelope(tmp_path: Path):
    from phase_marker.interventions import main

    config_path = Path("configs/phase-marker-qwen25-7b.toml")
    config = ExperimentConfig.load(config_path)
    config_hash = sha256_json(asdict(config))

    def parent(name: str, **values: object) -> Path:
        payload = {"schema_version": 1, "kind": name, "config_hash": config_hash, **values}
        payload["artifact_id"] = sha256_json(payload)
        path = tmp_path / f"{name}.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return path

    selection = parent("selection", selected_on="validation")
    activation = parent("activation", evidence_scope="experiment")
    checkpoint = parent(
        "checkpoint", model_id=config.model_id, model_revision="deadbeef", checkpoint_path="/fixture"
    )
    recipient_path = tmp_path / "recipient.pt"
    donor_path = tmp_path / "donor.pt"
    recipient_path.write_bytes(b"recipient")
    donor_path.write_bytes(b"donor")
    row = {
        "pair_id": "pair-1", "recipient_id": "r", "donor_id": "d",
        "recipient_batch_path": str(recipient_path), "donor_batch_path": str(donor_path),
        "recipient_batch_hash": hashlib.sha256(recipient_path.read_bytes()).hexdigest(),
        "donor_batch_hash": hashlib.sha256(donor_path.read_bytes()).hexdigest(),
        "target_token_ids": [4], "method": "residual_patch", "layer": 0,
        "positions": [2, 3], "norm_match": False, "control_name": "donor",
    }
    rows_path = tmp_path / "pairs.jsonl"
    rows_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    pairs = parent(
        "aligned_pairs", rows_file=rows_path.name,
        rows_hash=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        row_count=1, row_hashes=[sha256_json(row)],
    )
    output = tmp_path / "interventions"
    assert main(
        (
            "run", "--config", str(config_path), "--validation-selection-manifest", str(selection),
            "--aligned-pairs-manifest", str(pairs), "--activation-manifest", str(activation),
            "--checkpoint-manifest", str(checkpoint), "--model-id", config.model_id,
            "--model-revision", "deadbeef", "--backend", "tiny-fixture",
            "--allow-test-backend", "--output-root", str(output),
        )
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["evidence_scope"] == "plumbing_only"
    assert manifest["backend"] == "tiny-fixture"

    row["method"] = "mystery_patch"
    rows_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    pairs_payload = json.loads(pairs.read_text(encoding="utf-8"))
    pairs_payload["rows_hash"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    pairs_payload["row_hashes"] = [sha256_json(row)]
    pairs_payload.pop("artifact_id")
    pairs_payload["artifact_id"] = sha256_json(pairs_payload)
    pairs.write_text(canonical_json(pairs_payload) + "\n", encoding="utf-8")
    rejected_output = tmp_path / "unknown-output"
    with pytest.raises(ValueError, match="method"):
        main(
            (
                "run", "--config", str(config_path), "--validation-selection-manifest", str(selection),
                "--aligned-pairs-manifest", str(pairs), "--activation-manifest", str(activation),
                "--checkpoint-manifest", str(checkpoint), "--model-id", config.model_id,
                "--model-revision", "deadbeef", "--backend", "tiny-fixture",
                "--allow-test-backend", "--output-root", str(rejected_output),
            )
        )
    assert not rejected_output.exists()
