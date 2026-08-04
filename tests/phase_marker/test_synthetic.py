from __future__ import annotations

from collections.abc import Sequence
import json

import pytest

from phase_marker.synthetic import (
    SplitCounts,
    affine_example,
    generate_synthetic_suite,
    modular_example,
    numeric_composition_example,
    render_workspace,
    string_composition_example,
)


class FaithfulWorkspaceTokenizer:
    """Small reversible tokenizer that encodes the rendered workspace text."""

    _pieces = {
        "🜞": (101, 102, 103),
        "🜆": (104, 105, 106),
        "🜂": (107, 108, 109),
        "🜃": (110, 111, 112),
        ". . .": (201, 202, 203),
        ".": (401,),
        "♠♠♠": (301, 302, 303),
        "♣♣♣": (304, 305, 306),
        "♥♥♥": (307, 308, 309),
        "♦♦♦": (310, 311, 312),
    }

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if not value:
            return []
        return [token for piece in value.split("|") for token in self._pieces[piece]]

    def decode(self, token_ids: Sequence[int], **_: object) -> str:
        pieces: list[str] = []
        offset = 0
        while offset < len(token_ids):
            for piece, ids in self._pieces.items():
                if tuple(token_ids[offset : offset + len(ids)]) == ids:
                    pieces.append(piece)
                    offset += len(ids)
                    break
            else:  # pragma: no cover - protects the fake itself
                raise ValueError(f"unknown token at {offset}")
        return "|".join(pieces)


@pytest.fixture
def fake_tokenizer() -> FaithfulWorkspaceTokenizer:
    return FaithfulWorkspaceTokenizer()


def test_affine_chain_intermediates_are_exact():
    row = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))

    assert row.intermediates == (6, 11, 33, 29)
    assert row.answer == "29"


def test_other_families_have_four_hand_checked_states():
    modular = modular_example(x=5, modulus=11, operations=(("add", 8), ("mul", 3), ("sub", 7), ("add", 5)))
    numeric = numeric_composition_example(left=4, right=6, operations=(3, 2, 5))
    string = string_composition_example("ab", "cd", ("reverse", "upper", "-", "wrap"))

    assert modular.intermediates == (2, 6, 10, 4)
    assert modular.answer == "4"
    assert numeric.intermediates == (7, 12, 19, 95)
    assert numeric.answer == "95"
    assert string.intermediates == ("ba", "CD", "ba-CD", "[ba-CD]")
    assert string.answer == "[ba-CD]"


def test_parameter_tuples_do_not_cross_splits_and_generation_is_deterministic():
    counts = SplitCounts(train=100, validation=20, test=20)
    bundle = generate_synthetic_suite(101, counts)

    assert bundle == generate_synthetic_suite(101, counts)
    assert bundle.train.parameter_hashes.isdisjoint(bundle.validation.parameter_hashes)
    assert bundle.train.parameter_hashes.isdisjoint(bundle.test.parameter_hashes)
    assert bundle.validation.parameter_hashes.isdisjoint(bundle.test.parameter_hashes)
    assert len(bundle.train.examples) == 100
    assert set(bundle.manifest["family_counts"]) == {
        "affine_chain",
        "modular_chain",
        "two_source_numeric_composition",
        "string_transformation_composition",
    }


@pytest.mark.parametrize(("total_tokens", "region_width"), ((12, 3), (16, 4), (64, 16)))
def test_workspace_has_four_aligned_regions_from_actual_tokenization(
    fake_tokenizer, total_tokens, region_width
):
    example = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))

    prompt = render_workspace(example, "glyph", total_tokens, fake_tokenizer)

    assert tuple(fake_tokenizer.encode(prompt.text, add_special_tokens=False)) == prompt.token_ids
    assert len(prompt.regions) == 4
    assert [region.end - region.start for region in prompt.regions] == [region_width] * 4
    assert [region.marker_position for region in prompt.regions] == [
        region.start for region in prompt.regions
    ]
    assert len(prompt.token_ids) == total_tokens


@pytest.mark.parametrize(
    ("condition", "first_region"),
    (
        ("glyph", (101, 102, 103, 401)),
        ("dot", (201, 202, 203, 401)),
        ("repeated_glyph", (101, 102, 103, 401)),
        ("permuted_glyph", (104, 105, 106, 401)),
        ("random_symbol", (301, 302, 303, 401)),
    ),
)
def test_workspace_condition_uses_marker_sequence_then_shared_neutral_filler(
    fake_tokenizer, condition, first_region
):
    example = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))

    prompt = render_workspace(example, condition, 16, fake_tokenizer)

    assert prompt.token_ids[:4] == first_region
    assert all(prompt.token_ids[region.end - 1] == 401 for region in prompt.regions)


def test_workspace_rejects_marker_wider_than_an_actual_region(fake_tokenizer):
    fake_tokenizer._pieces["🜞"] = (101, 102, 103, 104)
    example = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))

    with pytest.raises(ValueError, match="exceeds region width 3"):
        render_workspace(example, "glyph", 12, fake_tokenizer)


def test_no_slot_has_no_workspace_regions(fake_tokenizer):
    example = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))

    prompt = render_workspace(example, "no_slot", 16, fake_tokenizer)

    assert prompt.regions == ()
    assert prompt.token_ids == ()
    assert prompt.text == ""


def test_build_smoke_writes_family_counts_zero_overlap_and_exact_scorer_evidence(tmp_path, capsys):
    from phase_marker.synthetic import main

    output_root = tmp_path / "synthetic"
    assert main(
        [
            "build",
            "--seed",
            "101",
            "--train",
            "8",
            "--validation",
            "4",
            "--test",
            "4",
            "--output-root",
            str(output_root),
        ]
    ) == 0

    reported = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert reported["parameter_overlap"] == {"train_test": 0, "train_validation": 0, "validation_test": 0}
    assert manifest["exact_scorer_agreement"] == {"agreeing": 16, "total": 16}
    assert manifest["family_counts"] == {
        "affine_chain": 4,
        "modular_chain": 4,
        "two_source_numeric_composition": 4,
        "string_transformation_composition": 4,
    }
    glyph_12 = manifest["workspace_layouts"]["12"]["glyph"]
    dot_16 = manifest["workspace_layouts"]["16"]["dot"]
    random_64 = manifest["workspace_layouts"]["64"]["random_symbol"]
    assert glyph_12["actual_token_count"] == 12
    assert glyph_12["region_widths"] == [3, 3, 3, 3]
    assert len(glyph_12["token_ids"]) == 12
    assert dot_16["actual_token_count"] == 16
    assert dot_16["region_widths"] == [4, 4, 4, 4]
    assert len(dot_16["regions"]) == 4
    assert random_64["actual_token_count"] == 64
    assert random_64["region_widths"] == [16, 16, 16, 16]
    assert len((output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 8
