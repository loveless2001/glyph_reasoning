from phase_marker.prompts import (
    MarkerSet,
    format_span,
    render_perturbation,
    render_prompt,
)


TEST_MARKERS = MarkerSet("G", "P", "S", "T")


def test_primary_prompts_differ_only_in_declared_format_span():
    rendered = {
        name: render_prompt("What is 2+3?", name, TEST_MARKERS)
        for name in ("neutral", "glyph", "dot", "headings")
    }
    projections = {
        text.replace(format_span(name, TEST_MARKERS), "") for name, text in rendered.items()
    }
    assert projections == {
        "Solve the problem carefully.\n\nProblem:\nWhat is 2+3?\n\n"
        "End with exactly one line of the form `Final answer: <answer>`.\n"
    }


def test_glyph_perturbations_change_only_the_glyph_format_span():
    question = "What is 2+3?"
    base = render_prompt(question, "glyph", TEST_MARKERS)
    for perturbation in (
        "delete",
        "cluster",
        "displace",
        "permute",
        "dot_replace",
        "unseen_replace",
    ):
        rendered = render_perturbation(question, perturbation, TEST_MARKERS)
        assert rendered.replace(
            format_span(perturbation, TEST_MARKERS), ""
        ) == base.replace(format_span("glyph", TEST_MARKERS), "")


def test_focused_perturbations_have_distinct_marker_layouts():
    assert "G Guideline" not in format_span("delete", TEST_MARKERS)
    assert "GPST" in format_span("cluster", TEST_MARKERS)
    assert "GuideliGne" in format_span("displace", TEST_MARKERS)
    assert format_span("permute", TEST_MARKERS).index("P") < format_span(
        "permute", TEST_MARKERS
    ).index("G")
    assert "G Guideline" not in format_span("dot_replace", TEST_MARKERS)
    assert "G Guideline" not in format_span("unseen_replace", TEST_MARKERS)


def test_unseen_replacement_uses_the_frozen_task5_width_matched_symbols():
    rendered = format_span("unseen_replace", TEST_MARKERS)
    assert "♠♠♠ Guideline" in rendered
    assert "♣♣♣ Plan" in rendered
    assert "♥♥♥ Step" in rendered
    assert "♦♦♦ Takeaway" in rendered
