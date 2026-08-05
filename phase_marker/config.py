"""Typed, validated configuration for the phase-marker experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


ALLOWED_ARMS = frozenset({"semantic", "glyph", "dot", "random", "direct", "filler"})
REQUIRED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REQUIRED_PHASE_MARKERS = ("🜞", "🜆", "🜂", "🜃")
REQUIRED_FINAL_DELIMITER = "Final answer:"
LEGACY_FINAL_DELIMITER = "🝞"


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str
    pilot_seed: int
    confirmatory_seeds: tuple[int, ...]
    phase_markers: tuple[str, str, str, str]
    final_delimiter: str
    arms: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with path.open("rb") as handle:
            content = handle.read()
        return cls.from_toml_bytes(content)

    @classmethod
    def from_toml_bytes(cls, content: bytes) -> "ExperimentConfig":
        """Parse one already-authenticated TOML byte snapshot."""
        if not isinstance(content, bytes):
            raise TypeError("experiment config content must be bytes")
        try:
            raw = tomllib.loads(content.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as error:
            raise ValueError("experiment config is invalid TOML") from error

        config = cls(
            model_id=_require_string(raw, "model_id"),
            pilot_seed=_require_int(raw, "pilot_seed"),
            confirmatory_seeds=_require_int_tuple(raw, "confirmatory_seeds"),
            phase_markers=_require_marker_tuple(raw),
            final_delimiter=_require_string(raw, "final_delimiter"),
            arms=_require_string_tuple(raw, "arms"),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.model_id != REQUIRED_MODEL_ID:
            raise ValueError(f"model_id must be {REQUIRED_MODEL_ID!r}")
        if self.phase_markers != REQUIRED_PHASE_MARKERS:
            raise ValueError(f"phase_markers must be {REQUIRED_PHASE_MARKERS!r}")
        if LEGACY_FINAL_DELIMITER in self.final_delimiter:
            raise ValueError("final delimiter must not contain the legacy 🝞 marker")
        if self.final_delimiter != REQUIRED_FINAL_DELIMITER:
            raise ValueError(f"final_delimiter must be {REQUIRED_FINAL_DELIMITER!r}")
        if not self.arms or any(arm not in ALLOWED_ARMS for arm in self.arms):
            raise ValueError(f"unknown experiment arm in {self.arms!r}")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("experiment arms must be unique")
        all_seeds = (self.pilot_seed, *self.confirmatory_seeds)
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("pilot and confirmatory seeds must be unique")
        if any(marker in self.final_delimiter for marker in self.phase_markers):
            raise ValueError("final delimiter must not contain a phase glyph")


def _require_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_string_tuple(raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(value)


def _require_int_tuple(raw: dict[str, object], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{key} must be an array of integers")
    return tuple(value)


def _require_marker_tuple(raw: dict[str, object]) -> tuple[str, str, str, str]:
    markers = _require_string_tuple(raw, "phase_markers")
    if len(markers) != 4:
        raise ValueError("phase_markers must contain exactly four glyphs")
    return (markers[0], markers[1], markers[2], markers[3])
