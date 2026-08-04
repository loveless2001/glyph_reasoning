"""Canonical inference prompts and focused phase-marker perturbations."""

from __future__ import annotations

from dataclasses import dataclass


PROMPT_TEMPLATE = """Solve the problem carefully.{format_span}

Problem:
{question}

End with exactly one line of the form `Final answer: <answer>`.
"""

# Frozen Task 5 replacement symbols.  Under cached Qwen revision
# a09a35458c702b33eeacc393d103063234e8bc28 they tokenize as exactly three
# repeated tokens each: [144867]*3, [144646]*3, [76709]*3, [144126]*3.
# Task 5 also audited their local-corpus status; this constant makes no claim
# about their pretraining rarity.
UNSEEN_MARKERS = ("♠♠♠", "♣♣♣", "♥♥♥", "♦♦♦")


@dataclass(frozen=True)
class MarkerSet:
    guideline: str
    plan: str
    step: str
    takeaway: str

    def __post_init__(self) -> None:
        values = (self.guideline, self.plan, self.step, self.takeaway)
        if any(not value for value in values) or len(set(values)) != 4:
            raise ValueError("marker sets require four distinct nonempty markers")


def format_span(condition: str, marker_set: MarkerSet) -> str:
    """Return the sole variable span in the canonical inference template."""
    markers = _markers_for(condition, marker_set)
    if condition == "neutral":
        return ""
    if condition == "headings" or condition == "delete":
        return _phase_lines(("", "", "", ""))
    if condition in {"glyph", "dot", "dot_replace", "unseen_replace", "permute"}:
        return _phase_lines(markers)
    if condition == "cluster":
        return (
            "\n\nUse exactly these reasoning phases, in order:\n"
            f"{''.join(markers)} Guideline, Plan, Step, Takeaway"
        )
    if condition == "displace":
        return (
            "\n\nUse exactly these reasoning phases, in order:\n"
            f"Guideli{markers[0]}ne\nPl{markers[1]}an\nSt{markers[2]}ep\nTake{markers[3]}away"
        )
    raise ValueError(f"unknown prompt condition {condition!r}")


def render_prompt(question: str, condition: str, marker_set: MarkerSet) -> str:
    """Render one canonical prompt, varying only its declared format span."""
    return PROMPT_TEMPLATE.format(question=question, format_span=format_span(condition, marker_set))


def render_perturbation(question: str, perturbation: str, marker_set: MarkerSet) -> str:
    """Render a focused glyph perturbation through the canonical template."""
    if perturbation not in {
        "delete",
        "cluster",
        "displace",
        "permute",
        "dot_replace",
        "unseen_replace",
    }:
        raise ValueError(f"unknown perturbation {perturbation!r}")
    return render_prompt(question, perturbation, marker_set)


def _markers_for(condition: str, marker_set: MarkerSet) -> tuple[str, str, str, str]:
    markers = (marker_set.guideline, marker_set.plan, marker_set.step, marker_set.takeaway)
    if condition in {"glyph", "cluster", "displace"}:
        return markers
    if condition in {"dot", "dot_replace"}:
        return (".", ".", ".", ".")
    if condition == "permute":
        return (markers[1], markers[0], markers[3], markers[2])
    if condition == "unseen_replace":
        return UNSEEN_MARKERS
    if condition in {"headings", "delete", "neutral"}:
        return ("", "", "", "")
    raise ValueError(f"unknown prompt condition {condition!r}")


def _phase_lines(markers: tuple[str, str, str, str]) -> str:
    prefix = "\n\nUse exactly these reasoning phases, in order:\n"
    labels = ("Guideline", "Plan", "Step", "Takeaway")
    return prefix + "\n".join(
        f"{marker}{' ' if marker else ''}{label}" for marker, label in zip(markers, labels)
    )
