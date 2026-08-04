import json

import pytest

from phase_marker.traces import (
    TraceParseError,
    parse_legacy_trace,
    recover_question,
    render_training_example,
    semantic_projection,
    main,
)


LEGACY = {
    "messages": [
        {"role": "user", "content": "Solve carefully.\n\nProblem:\nWhat is 2+3?\n"},
        {"role": "assistant", "content": (
            "🜞 Guideline:\nUse arithmetic.\n"
            "🜆 Plan:\nAdd the terms.\n"
            "🜂 Step:\n2+3=5.\n"
            "🜃 Takeaway:\nThe sum is five.\n"
            "🝞 Final answer: 5"
        )},
    ]
}


def test_parse_treats_final_glyph_as_delimiter_not_phase():
    trace = parse_legacy_trace(LEGACY)

    assert [phase.name for phase in trace.phases] == ["guideline", "plan", "step", "takeaway"]
    assert trace.question == "What is 2+3?"
    assert trace.answer == "5"


def test_marker_only_arms_have_identical_semantics():
    trace = parse_legacy_trace(LEGACY)

    outputs = [
        render_training_example(trace, arm, 101, 512)["messages"][1]["content"]
        for arm in ("semantic", "glyph", "dot", "random")
    ]

    assert len({semantic_projection(output) for output in outputs}) == 1
    assert all("🝞" not in output for output in outputs)
    assert all(output.count("Final answer:") == 1 for output in outputs)


def test_renderers_cover_direct_and_deterministic_filler_lengths():
    trace = parse_legacy_trace(LEGACY)

    direct = render_training_example(trace, "direct", 101, 512)["messages"][1]["content"]
    filler = render_training_example(trace, "filler", 101, 512)["messages"][1]["content"]

    assert direct == "Final answer: 5"
    assert filler.endswith("Final answer: 5")
    assert filler == render_training_example(trace, "filler", 101, 512)["messages"][1]["content"]
    assert filler.split("Final answer:")[0].strip(".\n").count(".") == 0
    assert 0 < filler.split("Final answer:")[0].count(".") <= 512


def test_parser_rejects_repeated_phase_marker_and_empty_answer():
    repeated = {
        **LEGACY,
        "messages": [*LEGACY["messages"]],
    }
    repeated["messages"][1] = {
        "role": "assistant",
        "content": LEGACY["messages"][1]["content"].replace("🜂 Step:", "🜆 Step:"),
    }
    empty_answer = {
        **LEGACY,
        "messages": [*LEGACY["messages"]],
    }
    empty_answer["messages"][1] = {
        "role": "assistant",
        "content": LEGACY["messages"][1]["content"].replace("Final answer: 5", "Final answer:  "),
    }

    with pytest.raises(TraceParseError, match="phase_markers"):
        parse_legacy_trace(repeated)
    with pytest.raises(TraceParseError, match="empty_answer"):
        parse_legacy_trace(empty_answer)


def test_parser_rejects_a_second_final_answer_delimiter():
    repeated_delimiter = {
        **LEGACY,
        "messages": [*LEGACY["messages"]],
    }
    repeated_delimiter["messages"][1] = {
        "role": "assistant",
        "content": LEGACY["messages"][1]["content"].replace(
            "Final answer: 5", "Final answer: 5\nFinal answer: 6"
        ),
    }

    with pytest.raises(TraceParseError, match="repeated_final_delimiter"):
        parse_legacy_trace(repeated_delimiter)


def test_projection_preserves_multiline_phase_content_that_begins_with_dot():
    dotted_content = {
        **LEGACY,
        "messages": [*LEGACY["messages"]],
    }
    dotted_content["messages"][1] = {
        "role": "assistant",
        "content": LEGACY["messages"][1]["content"].replace(
            "Use arithmetic.", "Use arithmetic.\n. This is semantic content."
        ),
    }
    trace = parse_legacy_trace(dotted_content)
    semantic = render_training_example(trace, "semantic", 101, 512)["messages"][1]["content"]
    dot = render_training_example(trace, "dot", 101, 512)["messages"][1]["content"]

    assert semantic_projection(dot) == semantic
    assert ". This is semantic content." in semantic_projection(dot)


def test_custom_neutral_delimiter_is_structural_and_semantically_neutral():
    trace = parse_legacy_trace(LEGACY)
    semantic = render_training_example(trace, "semantic", 101, 512)["messages"][1]["content"]
    rendered = render_training_example(
        trace, "dot", 101, 512, neutral_delimiter="§"
    )["messages"][1]["content"]

    assert rendered.count("\n§\n") == 3
    assert rendered.startswith("§\n")
    assert semantic_projection(rendered, neutral_delimiter="§") == semantic


def test_recover_question_requires_problem_delimiter():
    assert recover_question("Solve carefully.\n\nProblem:\nWhat is 2+3?\n") == "What is 2+3?"
    with pytest.raises(TraceParseError, match="missing_problem"):
        recover_question("Solve carefully.")


def test_audit_accounts_for_parsed_and_excluded_rows(tmp_path, capsys):
    input_path = tmp_path / "legacy.jsonl"
    output_path = tmp_path / "audit.jsonl"
    invalid = {"messages": [{"role": "user", "content": "No question heading"}, LEGACY["messages"][1]]}
    input_path.write_text("\n".join(json.dumps(row) for row in (LEGACY, invalid)) + "\n", encoding="utf-8")

    main(["audit", "--input", str(input_path), "--output", str(output_path)])

    assert json.loads(capsys.readouterr().out) == {"missing_problem": 1, "parsed": 1}
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"line_number": 1, "status": "parsed", "trace_id": parse_legacy_trace(LEGACY).trace_id},
        {"line_number": 2, "reason": "missing_problem", "status": "excluded"},
    ]
