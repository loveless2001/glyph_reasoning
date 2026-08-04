from __future__ import annotations

from collections.abc import Sequence

from phase_marker.token_audit import audit_marker_set, select_neutral_delimiter


class FaithfulFakeTokenizer:
    """Tiny tokenizer double with the HF methods consumed by the audit code."""

    revision = "fake-qwen-tokenizer-revision"

    def __init__(self) -> None:
        self._encodings = {
            "🜞": [41, 42],
            "🜆": [43, 44],
            "🜂": [45, 46],
            "🜃": [47, 48],
            ".": [3],
            "|": [4],
            "§": [5, 6],
            "·": [11],
            "•": [12],
            "🟦": [7, 8],
            "🟥": [13, 14],
            "🔶": [15, 16],
            "🔷": [17, 18],
            "♠": [9, 10],
            "♣": [19, 20],
            "♥": [21, 22],
            "♦": [23, 24],
        }
        self._vocabulary = {
            ".": 3,
            "|": 4,
            "§": 5,
            "·": 11,
            "•": 12,
            "🟦": 7,
            "🟥": 13,
            "🔶": 15,
            "🔷": 17,
            "♠": 9,
            "♣": 19,
            "♥": 21,
            "♦": 23,
        }

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(self._encodings[value])

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [f"tok-{token_id}" for token_id in ids]

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)


def test_neutral_delimiter_matches_glyph_token_width():
    tokenizer = FaithfulFakeTokenizer()
    glyph_width = len(tokenizer.encode("🜞", add_special_tokens=False))

    audit = audit_marker_set(tokenizer, [".", "|", "§"])
    selected = select_neutral_delimiter(audit, target_width=glyph_width)

    assert selected == "§"
    assert len(tokenizer.encode(selected, add_special_tokens=False)) == glyph_width


def test_audit_records_token_and_unicode_measurements_without_pretraining_claims():
    tokenizer = FaithfulFakeTokenizer()

    row = audit_marker_set(tokenizer, ["§"])[0]

    assert row.symbol == "§"
    assert row.codepoints == ("U+00A7",)
    assert row.utf8_hex == "c2a7"
    assert row.token_ids == (5, 6)
    assert row.token_strings == ("tok-5", "tok-6")
    assert row.token_count == 2
    assert row.vocabulary_member is True
    assert row.local_corpus_count == 0
