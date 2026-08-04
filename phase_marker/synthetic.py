"""Deterministic four-state synthetic mechanism tasks and aligned workspaces.

This module is intentionally self-contained: it uses tokenizer interfaces only
to make the rendered workspace auditable, and never loads a model or executes
behavioral evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Literal

from phase_marker.config import REQUIRED_PHASE_MARKERS
from phase_marker.io import canonical_json, sha256_json, write_jsonl_atomic
from phase_marker.prompts import UNSEEN_MARKERS
from phase_marker.scoring import answers_equivalent


Family = Literal[
    "modular_chain",
    "affine_chain",
    "two_source_numeric_composition",
    "string_transformation_composition",
]

FAMILIES: tuple[Family, ...] = (
    "modular_chain",
    "affine_chain",
    "two_source_numeric_composition",
    "string_transformation_composition",
)
WORKSPACE_LENGTHS = frozenset((12, 16, 64))
NEUTRAL_FILLER = "."
WORKSPACE_CONDITIONS = frozenset(
    {"glyph", "dot", "repeated_glyph", "permuted_glyph", "random_symbol", "no_slot"}
)


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    test: int

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("split counts must be nonnegative integers")
        if not any(values):
            raise ValueError("at least one synthetic split must be nonempty")


@dataclass(frozen=True)
class SyntheticExample:
    example_id: str
    family: Family
    parameters: Mapping[str, object]
    parameter_hash: str
    question: str
    intermediates: tuple[int | str, int | str, int | str, int | str]
    answer: str


@dataclass(frozen=True)
class SyntheticSplit:
    examples: tuple[SyntheticExample, ...]
    parameter_hashes: frozenset[str]


@dataclass(frozen=True)
class WorkspaceRegion:
    index: int
    start: int
    end: int
    marker_position: int


@dataclass(frozen=True)
class WorkspacePrompt:
    example_id: str
    condition: str
    total_tokens: int
    text: str
    token_ids: tuple[int, ...]
    regions: tuple[WorkspaceRegion, ...]


@dataclass(frozen=True)
class SyntheticBundle:
    train: SyntheticSplit
    validation: SyntheticSplit
    test: SyntheticSplit
    manifest: Mapping[str, object]


def affine_example(
    x: int, operations: tuple[tuple[str, int], tuple[str, int], tuple[str, int], tuple[str, int]],
) -> SyntheticExample:
    """Build one four-step affine chain with its exact four state values."""
    states = _numeric_states(x, operations)
    parameters = {"x": x, "operations": operations}
    return _example(
        "affine_chain",
        parameters,
        (
            f"Start with {x}. Apply { _format_operations(operations) } in order. "
            "Report the value after the fourth operation."
        ),
        states,
    )


def modular_example(
    x: int,
    modulus: int,
    operations: tuple[tuple[str, int], tuple[str, int], tuple[str, int], tuple[str, int]],
) -> SyntheticExample:
    """Build one chain that reduces modulo ``modulus`` after every operation."""
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    states = _numeric_states(x, operations, modulus=modulus)
    parameters = {"x": x, "modulus": modulus, "operations": operations}
    return _example(
        "modular_chain",
        parameters,
        (
            f"Start with {x} modulo {modulus}. Apply { _format_operations(operations) } in order, "
            f"reducing modulo {modulus} after every operation. Report the fourth value."
        ),
        states,
    )


def numeric_composition_example(
    left: int, right: int, operations: tuple[int, int, int]
) -> SyntheticExample:
    """Build a four-state two-source numeric composition task.

    The state sequence is ``left + delta``, ``right * factor``, their sum,
    and that sum multiplied by ``final_factor``.
    """
    delta, factor, final_factor = operations
    if factor == 0 or final_factor == 0:
        raise ValueError("numeric composition factors must be nonzero")
    state_1 = left + delta
    state_2 = right * factor
    state_3 = state_1 + state_2
    state_4 = state_3 * final_factor
    parameters = {"left": left, "right": right, "operations": operations}
    return _example(
        "two_source_numeric_composition",
        parameters,
        (
            f"Source A is {left} and source B is {right}. Add {delta} to source A; multiply source B "
            f"by {factor}; add those two results; then multiply the sum by {final_factor}. "
            "Report the fourth value."
        ),
        (state_1, state_2, state_3, state_4),
    )


def string_composition_example(
    left: str, right: str, operations: tuple[str, str, str, str]
) -> SyntheticExample:
    """Build a four-state exact string transformation/composition task."""
    left_transform, right_transform, separator, final_transform = operations
    state_1 = _transform_string(left, left_transform)
    state_2 = _transform_string(right, right_transform)
    state_3 = state_1 + separator + state_2
    state_4 = _final_string_transform(state_3, final_transform)
    parameters = {"left": left, "right": right, "operations": operations}
    return _example(
        "string_transformation_composition",
        parameters,
        (
            f"Transform {left!r} with {left_transform}; transform {right!r} with {right_transform}; "
            f"join the results with {separator!r}; then apply {final_transform}. "
            "Report the exact fourth string."
        ),
        (state_1, state_2, state_3, state_4),
    )


def generate_synthetic_suite(seed: int, counts: SplitCounts) -> SyntheticBundle:
    """Generate deterministic parameter-disjoint train/validation/test splits."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    used_hashes: set[str] = set()
    splits: dict[str, SyntheticSplit] = {}
    for split_name, count in (
        ("train", counts.train),
        ("validation", counts.validation),
        ("test", counts.test),
    ):
        examples: list[SyntheticExample] = []
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            example = _unique_generated_example(family, rng, used_hashes)
            example = SyntheticExample(
                example_id=f"synthetic:{split_name}:{index:05d}:{example.parameter_hash[:12]}",
                family=example.family,
                parameters=example.parameters,
                parameter_hash=example.parameter_hash,
                question=example.question,
                intermediates=example.intermediates,
                answer=example.answer,
            )
            examples.append(example)
            used_hashes.add(example.parameter_hash)
        splits[split_name] = SyntheticSplit(tuple(examples), frozenset(row.parameter_hash for row in examples))

    return SyntheticBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        manifest=_manifest(seed, counts, splits),
    )


def render_workspace(
    example: SyntheticExample, condition: str, total_tokens: int, tokenizer: object
) -> WorkspacePrompt:
    """Render four equal, tokenizer-audited workspace regions for one condition."""
    if condition not in WORKSPACE_CONDITIONS:
        raise ValueError(f"unknown workspace condition {condition!r}")
    if total_tokens not in WORKSPACE_LENGTHS:
        raise ValueError(f"total_tokens must be one of {sorted(WORKSPACE_LENGTHS)}")
    if condition == "no_slot":
        return WorkspacePrompt(example.example_id, condition, total_tokens, "", (), ())

    filler_ids = _encode(tokenizer, NEUTRAL_FILLER)
    if len(filler_ids) != 1:
        raise ValueError("the neutral filler must encode to exactly one token")
    region_width = total_tokens // 4
    markers = _markers_for_condition(condition)
    pieces: list[str] = []
    expected_ids: list[int] = []
    regions: list[WorkspaceRegion] = []
    for index, marker in enumerate(markers):
        marker_ids = _encode(tokenizer, marker)
        if not marker_ids:
            raise ValueError(f"workspace marker for {condition!r} encodes to no tokens")
        if len(marker_ids) > region_width:
            raise ValueError(
                f"workspace marker for {condition!r} uses {len(marker_ids)} tokens, "
                f"which exceeds region width {region_width}"
            )
        start = len(expected_ids)
        expected_ids.extend(marker_ids)
        expected_ids.extend(filler_ids * (region_width - len(marker_ids)))
        pieces.extend((marker, *(NEUTRAL_FILLER for _ in range(region_width - len(marker_ids)))) )
        regions.append(WorkspaceRegion(index, start, start + region_width, start))

    text = _render_token_exact_text(tokenizer, pieces, tuple(expected_ids))
    token_ids = tuple(_encode(tokenizer, text))
    if token_ids != tuple(expected_ids):  # defensive: _render_token_exact_text is the boundary proof
        raise ValueError("workspace text does not round-trip to the intended token layout")
    return WorkspacePrompt(example.example_id, condition, total_tokens, text, token_ids, tuple(regions))


def _example(
    family: Family,
    parameters: Mapping[str, object],
    question: str,
    intermediates: tuple[int | str, int | str, int | str, int | str],
) -> SyntheticExample:
    parameter_hash = sha256_json({"family": family, "parameters": parameters})
    return SyntheticExample(
        example_id=f"synthetic:unassigned:{parameter_hash[:12]}",
        family=family,
        parameters=parameters,
        parameter_hash=parameter_hash,
        question=question,
        intermediates=intermediates,
        answer=str(intermediates[-1]),
    )


def _numeric_states(
    initial: int,
    operations: Sequence[tuple[str, int]],
    *,
    modulus: int | None = None,
) -> tuple[int, int, int, int]:
    if len(operations) != 4:
        raise ValueError("numeric chains require exactly four operations")
    value = initial
    states: list[int] = []
    for operator, operand in operations:
        if operator == "add":
            value += operand
        elif operator == "sub":
            value -= operand
        elif operator == "mul":
            value *= operand
        else:
            raise ValueError(f"unknown numeric operation {operator!r}")
        if modulus is not None:
            value %= modulus
        states.append(value)
    return (states[0], states[1], states[2], states[3])


def _format_operations(operations: Sequence[tuple[str, int]]) -> str:
    return ", then ".join(f"{operator} {operand}" for operator, operand in operations)


def _transform_string(value: str, transform: str) -> str:
    if transform == "reverse":
        return value[::-1]
    if transform == "upper":
        return value.upper()
    if transform == "lower":
        return value.lower()
    if transform == "rotate_left":
        return value[1:] + value[:1]
    raise ValueError(f"unknown string transform {transform!r}")


def _final_string_transform(value: str, transform: str) -> str:
    if transform == "wrap":
        return f"[{value}]"
    if transform == "reverse":
        return value[::-1]
    if transform == "upper":
        return value.upper()
    raise ValueError(f"unknown final string transform {transform!r}")


def _unique_generated_example(
    family: Family, rng: random.Random, used_hashes: set[str]
) -> SyntheticExample:
    while True:
        row = _generated_example(family, rng)
        if row.parameter_hash not in used_hashes:
            return row


def _generated_example(family: Family, rng: random.Random) -> SyntheticExample:
    if family == "modular_chain":
        return modular_example(
            rng.randrange(0, 51),
            rng.choice((11, 13, 17, 19)),
            (("add", rng.randrange(1, 10)), ("mul", rng.randrange(2, 6)), ("sub", rng.randrange(1, 10)), ("add", rng.randrange(1, 10))),
        )
    if family == "affine_chain":
        return affine_example(
            rng.randrange(-12, 13),
            (("mul", rng.randrange(2, 6)), ("add", rng.randrange(-9, 10)), ("mul", rng.randrange(2, 6)), ("sub", rng.randrange(-9, 10))),
        )
    if family == "two_source_numeric_composition":
        return numeric_composition_example(
            rng.randrange(-20, 21), rng.randrange(-20, 21), (rng.randrange(-9, 10), rng.randrange(2, 6), rng.randrange(2, 6))
        )
    if family == "string_transformation_composition":
        return string_composition_example(
            _random_word(rng),
            _random_word(rng),
            (rng.choice(("reverse", "upper", "rotate_left")), rng.choice(("reverse", "upper", "lower")), rng.choice(("-", "/", "+")), rng.choice(("wrap", "reverse", "upper"))),
        )
    raise AssertionError(f"unhandled family {family!r}")


def _random_word(rng: random.Random) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(4))


def _manifest(seed: int, counts: SplitCounts, splits: Mapping[str, SyntheticSplit]) -> dict[str, object]:
    all_examples = tuple(row for split in splits.values() for row in split.examples)
    overlap = {
        "train_validation": len(splits["train"].parameter_hashes & splits["validation"].parameter_hashes),
        "train_test": len(splits["train"].parameter_hashes & splits["test"].parameter_hashes),
        "validation_test": len(splits["validation"].parameter_hashes & splits["test"].parameter_hashes),
    }
    family_counts = dict(sorted(Counter(row.family for row in all_examples).items()))
    agreeing = sum(answers_equivalent("synthetic", row.answer, str(row.intermediates[-1])) for row in all_examples)
    return {
        "kind": "phase_marker_synthetic_four_state_suite",
        "seed": seed,
        "counts": asdict(counts),
        "family_counts": family_counts,
        "split_counts": {name: len(split.examples) for name, split in splits.items()},
        "parameter_overlap": overlap,
        "exact_scorer_agreement": {"agreeing": agreeing, "total": len(all_examples)},
    }


def _markers_for_condition(condition: str) -> tuple[str, str, str, str]:
    if condition == "glyph":
        return REQUIRED_PHASE_MARKERS
    if condition == "dot":
        return (". . .",) * 4
    if condition == "repeated_glyph":
        return (REQUIRED_PHASE_MARKERS[0],) * 4
    if condition == "permuted_glyph":
        return (
            REQUIRED_PHASE_MARKERS[1],
            REQUIRED_PHASE_MARKERS[0],
            REQUIRED_PHASE_MARKERS[3],
            REQUIRED_PHASE_MARKERS[2],
        )
    if condition == "random_symbol":
        return UNSEEN_MARKERS
    raise ValueError(f"condition {condition!r} does not have workspace markers")


def _encode(tokenizer: object, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide encode")
    token_ids = encode(text, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("tokenizer.encode must return a sequence of integer token IDs")
    return tuple(token_ids)


def _render_token_exact_text(tokenizer: object, pieces: Sequence[str], expected_ids: tuple[int, ...]) -> str:
    """Find text whose *full* tokenizer result is exactly the planned layout."""
    for separator in ("", " ", "\n", "|"):
        candidate = separator.join(pieces)
        try:
            if _encode(tokenizer, candidate) == expected_ids:
                return candidate
        except (KeyError, ValueError):
            continue
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            candidate = decode(list(expected_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            candidate = decode(list(expected_ids))
        if isinstance(candidate, str) and _encode(tokenizer, candidate) == expected_ids:
            return candidate
    raise ValueError("tokenizer cannot render an exact workspace token layout")


def _write_bundle(output_root: Path, bundle: SyntheticBundle) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite synthetic output: {output_root}")
    output_root.mkdir(parents=True)
    try:
        for name, split in (("train", bundle.train), ("validation", bundle.validation), ("test", bundle.test)):
            write_jsonl_atomic(output_root / f"{name}.jsonl", (asdict(row) for row in split.examples))
        (output_root / "manifest.json").write_text(canonical_json(bundle.manifest) + "\n", encoding="utf-8")
    except BaseException:
        for path in output_root.iterdir():
            path.unlink()
        output_root.rmdir()
        raise


class _SyntheticSmokeTokenizer:
    """Reversible synthetic-only tokenizer with Task 5's measured marker widths."""

    _pieces = {
        "🜞": (1001, 1002, 1003),
        "🜆": (1004, 1005, 1006),
        "🜂": (1007, 1008, 1009),
        "🜃": (1010, 1011, 1012),
        ". . .": (2001, 2002, 2003),
        ".": (3001,),
        "♠♠♠": (4001, 4002, 4003),
        "♣♣♣": (4004, 4005, 4006),
        "♥♥♥": (4007, 4008, 4009),
        "♦♦♦": (4010, 4011, 4012),
    }

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("synthetic smoke tokenizer has no special tokens")
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
            else:  # pragma: no cover - malformed production-free smoke IDs
                raise ValueError(f"unknown synthetic smoke token at {offset}")
        return "|".join(pieces)


def _smoke_workspace_layouts(example: SyntheticExample) -> dict[str, object]:
    tokenizer = _SyntheticSmokeTokenizer()
    layouts: dict[str, object] = {}
    for total_tokens in sorted(WORKSPACE_LENGTHS):
        by_condition: dict[str, object] = {}
        for condition in ("glyph", "dot", "repeated_glyph", "permuted_glyph", "random_symbol"):
            prompt = render_workspace(example, condition, total_tokens, tokenizer)
            by_condition[condition] = {
                "actual_token_count": len(prompt.token_ids),
                "token_ids": list(prompt.token_ids),
                "region_widths": [region.end - region.start for region in prompt.regions],
                "regions": [asdict(region) for region in prompt.regions],
            }
        layouts[str(total_tokens)] = by_condition
    return layouts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a synthetic phase-marker mechanism suite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="write deterministic synthetic-only artifacts")
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--train", type=int, required=True)
    build.add_argument("--validation", type=int, required=True)
    build.add_argument("--test", type=int, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "build":  # pragma: no cover - argparse currently has one command
        raise ValueError(f"unknown command {args.command!r}")
    bundle = generate_synthetic_suite(args.seed, SplitCounts(args.train, args.validation, args.test))
    representative = next(
        row for split in (bundle.train, bundle.validation, bundle.test) for row in split.examples
    )
    bundle = SyntheticBundle(
        train=bundle.train,
        validation=bundle.validation,
        test=bundle.test,
        manifest={
            **bundle.manifest,
            "workspace_layouts": _smoke_workspace_layouts(representative),
        },
    )
    _write_bundle(args.output_root, bundle)
    print(canonical_json(bundle.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
