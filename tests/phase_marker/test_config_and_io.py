from pathlib import Path

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json


def test_config_locks_confirmatory_seeds():
    config = ExperimentConfig.load(Path("configs/phase-marker-qwen25-7b.toml"))
    assert config.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.pilot_seed == 42
    assert config.confirmatory_seeds == (101, 202, 303)
    assert config.phase_markers == ("🜞", "🜆", "🜂", "🜃")
    assert config.final_delimiter == "Final answer:"


def test_canonical_hash_ignores_mapping_insertion_order():
    left = {"seed": 101, "arm": "glyph"}
    right = {"arm": "glyph", "seed": 101}
    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_id", "Qwen/Qwen2.5-3B-Instruct"),
        ("phase_markers", '["A", "B", "C", "D"]'),
        ("final_delimiter", "Answer:"),
        ("final_delimiter", "Final 🝞 answer:"),
    ],
)
def test_config_rejects_protocol_values_that_are_not_frozen(tmp_path, field, value):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        "\n".join(
            [
                'model_id = "Qwen/Qwen2.5-7B-Instruct"',
                "pilot_seed = 42",
                "confirmatory_seeds = [101, 202, 303]",
                'phase_markers = ["🜞", "🜆", "🜂", "🜃"]',
                'final_delimiter = "Final answer:"',
                'arms = ["semantic", "glyph", "dot", "random", "direct", "filler"]',
            ]
        ).replace(
            {
                "model_id": 'model_id = "Qwen/Qwen2.5-7B-Instruct"',
                "phase_markers": 'phase_markers = ["🜞", "🜆", "🜂", "🜃"]',
                "final_delimiter": 'final_delimiter = "Final answer:"',
            }[field],
            f"{field} = {value if field == 'phase_markers' else repr(value)}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ExperimentConfig.load(config_path)
