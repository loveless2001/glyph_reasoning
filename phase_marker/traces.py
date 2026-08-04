"""Canonical parsing and deterministic rendering of legacy phase traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from phase_marker.io import canonical_json, write_jsonl_atomic
from phase_marker.schema import CanonicalTrace, PhaseSpan


PHASE_SPECS = (
    ("guideline", "🜞", "Guideline:"),
    ("plan", "🜆", "Plan:"),
    ("step", "🜂", "Step:"),
    ("takeaway", "🜃", "Takeaway:"),
)
FINAL_GLYPH = "🝞"
FINAL_DELIMITER = "Final answer:"
DOT_DELIMITER = "."


class TraceParseError(ValueError):
    """A legacy row cannot be represented as a canonical trace."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def recover_question(user_content: str) -> str:
    """Recover the exact question after the final legacy ``Problem:`` heading."""
    delimiter = "Problem:"
    if delimiter not in user_content:
        raise TraceParseError("missing_problem")
    question = user_content.rsplit(delimiter, 1)[1].strip()
    if not question:
        raise TraceParseError("empty_question")
    return question


def parse_legacy_trace(row: Mapping[str, object]) -> CanonicalTrace:
    """Parse one strict five-glyph legacy row without treating ``🝞`` as a phase."""
    messages = _messages(row)
    user = _message_content(messages, 0, "user")
    assistant = _message_content(messages, 1, "assistant")
    question = recover_question(user)

    glyphs = tuple(spec[1] for spec in PHASE_SPECS)
    if any(assistant.count(glyph) != 1 for glyph in glyphs):
        raise TraceParseError("phase_markers")
    if assistant.count(FINAL_GLYPH) != 1:
        raise TraceParseError("final_marker")
    positions = [assistant.index(glyph) for glyph in glyphs]
    final_position = assistant.index(FINAL_GLYPH)
    if positions != sorted(positions) or positions[-1] >= final_position:
        raise TraceParseError("phase_markers")

    phases: list[PhaseSpan] = []
    boundaries = [*positions[1:], final_position]
    for (name, glyph, heading), start, end in zip(PHASE_SPECS, positions, boundaries):
        prefix = f"{glyph} {heading}"
        if not assistant.startswith(prefix, start):
            raise TraceParseError("phase_heading")
        body = assistant[start + len(prefix) : end].strip()
        if not body:
            raise TraceParseError("empty_phase")
        phases.append(PhaseSpan(name=name, body=body))

    final = assistant[final_position + len(FINAL_GLYPH) :].strip()
    if not final.startswith(FINAL_DELIMITER):
        raise TraceParseError("missing_final_delimiter")
    answer = final[len(FINAL_DELIMITER) :].strip()
    if not answer:
        raise TraceParseError("empty_answer")

    source = _source(row)
    trace_id = _trace_id(source, question)
    return CanonicalTrace(
        trace_id=trace_id,
        source=source,
        question=question,
        answer=answer,
        phases=(phases[0], phases[1], phases[2], phases[3]),
    )


def render_training_example(
    trace: CanonicalTrace, arm: str, seed: int, max_filler_tokens: int
) -> dict[str, object]:
    """Render one arm with exactly one literal final-answer delimiter."""
    if arm not in {"semantic", "glyph", "dot", "random", "direct", "filler"}:
        raise ValueError(f"unknown training arm: {arm}")
    if max_filler_tokens < 1:
        raise ValueError("max_filler_tokens must be positive")

    if arm == "direct":
        assistant = _final_answer(trace.answer)
    elif arm == "filler":
        length = _filler_length(trace, seed, max_filler_tokens)
        assistant = f"{'.' * length}\n{_final_answer(trace.answer)}"
    else:
        markers = _markers_for_arm(trace, arm, seed)
        body_lines = [
            f"{marker}{phase.body}" if marker else phase.body
            for marker, phase in zip(markers, trace.phases)
        ]
        assistant = "\n".join([*body_lines, _final_answer(trace.answer)])

    return {
        "messages": [
            {"role": "user", "content": trace.question},
            {"role": "assistant", "content": assistant},
        ]
    }


def semantic_projection(rendered_assistant: str) -> str:
    """Remove only renderer boundary markers for cross-arm semantic comparisons."""
    marker_pattern = "|".join(re.escape(spec[1]) for spec in PHASE_SPECS)
    return re.sub(rf"(?m)^(?:{marker_pattern}|{re.escape(DOT_DELIMITER)})\s*", "", rendered_assistant)


def _messages(row: Mapping[str, object]) -> Sequence[object]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise TraceParseError("invalid_messages")
    return messages


def _message_content(messages: Sequence[object], index: int, role: str) -> str:
    message = messages[index]
    if not isinstance(message, Mapping) or message.get("role") != role:
        raise TraceParseError("invalid_messages")
    content = message.get("content")
    if not isinstance(content, str):
        raise TraceParseError("invalid_messages")
    return content


def _source(row: Mapping[str, object]) -> str:
    source = row.get("source", "legacy")
    if not isinstance(source, str) or not source.strip():
        raise TraceParseError("invalid_source")
    return source.strip()


def _trace_id(source: str, question: str) -> str:
    normalized = f"{source.casefold().strip()}\n{_normalize_whitespace(question)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _final_answer(answer: str) -> str:
    return f"{FINAL_DELIMITER} {answer}"


def _markers_for_arm(trace: CanonicalTrace, arm: str, seed: int) -> tuple[str, str, str, str]:
    if arm == "semantic":
        return ("", "", "", "")
    if arm == "glyph":
        return tuple(f"{glyph} " for _, glyph, _ in PHASE_SPECS)  # type: ignore[return-value]
    if arm == "dot":
        return (f"{DOT_DELIMITER} ",) * 4
    identities = [glyph for _, glyph, _ in PHASE_SPECS]
    deterministic = hashlib.sha256(f"{trace.trace_id}{seed}".encode("utf-8")).digest()
    ordered = [glyph for _, glyph in sorted(zip(deterministic[:4], identities))]
    return tuple(f"{glyph} " for glyph in ordered)  # type: ignore[return-value]


def _filler_length(trace: CanonicalTrace, seed: int, max_filler_tokens: int) -> int:
    trace_matched = min(max_filler_tokens, len(" ".join(phase.body for phase in trace.phases).split()))
    choices = (4, 16, 64, trace_matched)
    digest = hashlib.sha256(f"{trace.trace_id}{seed}filler".encode("utf-8")).digest()
    return min(max_filler_tokens, choices[digest[0] % len(choices)])


def _audit(input_path: Path, output_path: Path) -> Counter[str]:
    records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TraceParseError("invalid_row")
                trace = parse_legacy_trace(row)
            except json.JSONDecodeError:
                reason = "invalid_json"
                records.append({"line_number": line_number, "status": "excluded", "reason": reason})
            except TraceParseError as error:
                records.append({"line_number": line_number, "status": "excluded", "reason": error.code})
            else:
                records.append({"line_number": line_number, "status": "parsed", "trace_id": trace.trace_id})
                counts["parsed"] += 1
                continue
            counts[records[-1]["reason"]] += 1
    write_jsonl_atomic(output_path, records)
    return counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "audit":
        print(canonical_json(dict(sorted(_audit(arguments.input, arguments.output).items()))))


if __name__ == "__main__":
    main()
