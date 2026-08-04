from pathlib import Path

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
