"""Offline tokenizer audit and deterministic phase-marker training data materialization."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import string

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, read_jsonl, sha256_json, write_jsonl_atomic
from phase_marker.schema import ArtifactManifest, CanonicalTrace
from phase_marker.splits import parse_trace_pool
from phase_marker.traces import render_training_example, semantic_projection


DOT_CANDIDATES = (".", "|", "§", "·", "•", ". . .")
EMOJI_CONTROLS = ("🟦", "🟥", "🔶", "🔷")
RANDOM_SYMBOL_CANDIDATES = ("♠", "♣", "♥", "♦")
LOCAL_FREQUENCY_LABEL = "local_corpus_frequency_proxy"
QWEN25_7B_TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class TokenWidthMismatch(ValueError):
    """A control cannot match the configured phase-marker token width."""


class SplitLineageUnavailable(ValueError):
    """No approved frozen split lineage is available for training data output."""


@dataclass(frozen=True)
class TokenAuditRow:
    symbol: str
    codepoints: tuple[str, ...]
    utf8_hex: str
    token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    token_count: int
    vocabulary_member: bool
    local_corpus_count: int


def audit_marker_set(
    tokenizer: object,
    symbols: Sequence[str],
    *,
    local_corpus: Sequence[str] = (),
) -> list[TokenAuditRow]:
    """Measure tokenization and local-corpus occurrence for each candidate symbol.

    ``local_corpus_count`` is a corpus-local proxy only.  It says nothing
    about pretraining frequency.
    """
    vocabulary = _vocabulary(tokenizer)
    corpus = "".join(local_corpus)
    rows: list[TokenAuditRow] = []
    for symbol in symbols:
        if not symbol:
            raise ValueError("token audit symbols must be nonempty")
        token_ids = tuple(_encode(tokenizer, symbol))
        token_strings = tuple(_token_strings(tokenizer, token_ids))
        rows.append(
            TokenAuditRow(
                symbol=symbol,
                codepoints=tuple(f"U+{ord(character):04X}" for character in symbol),
                utf8_hex=symbol.encode("utf-8").hex(),
                token_ids=token_ids,
                token_strings=token_strings,
                token_count=len(token_ids),
                vocabulary_member=symbol in vocabulary,
                local_corpus_count=corpus.count(symbol),
            )
        )
    return rows


def select_neutral_delimiter(audit: Sequence[TokenAuditRow], target_width: int) -> str:
    """Select the first audited neutral candidate with the required width."""
    if target_width < 1:
        raise ValueError("target_width must be positive")
    matches = [row.symbol for row in audit if row.token_count == target_width]
    if not matches:
        measured = ", ".join(f"{row.symbol!r}={row.token_count}" for row in audit)
        raise TokenWidthMismatch(
            f"no audited neutral delimiter has token width {target_width}; measured {measured}"
        )
    return matches[0]


def materialize_training_arms(
    config: ExperimentConfig,
    traces: Sequence[CanonicalTrace],
    tokenizer: object,
    output_root: Path,
) -> dict[str, ArtifactManifest]:
    """Emit all configured training arms only with approved frozen split lineage."""
    split_hash = _parent_split_hash(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite materialized training data: {output_root}")
    if not traces:
        raise ValueError("cannot materialize an empty canonical training trace set")
    if len({trace.trace_id for trace in traces}) != len(traces):
        raise ValueError("canonical training traces must have unique trace ids")

    corpus = _canonical_training_corpus(traces)
    symbols = (*config.phase_markers, *DOT_CANDIDATES, *EMOJI_CONTROLS, *RANDOM_SYMBOL_CANDIDATES)
    audit = audit_marker_set(tokenizer, symbols, local_corpus=corpus)
    audit_by_symbol = {row.symbol: row for row in audit}
    glyph_widths = {audit_by_symbol[symbol].token_count for symbol in config.phase_markers}
    if len(glyph_widths) != 1:
        measured = {symbol: audit_by_symbol[symbol].token_count for symbol in config.phase_markers}
        raise TokenWidthMismatch(f"configured glyph markers do not share a token width: {measured}")
    target_width = next(iter(glyph_widths))
    delimiter_audit = [audit_by_symbol[symbol] for symbol in DOT_CANDIDATES]
    neutral_delimiter = select_neutral_delimiter(delimiter_audit, target_width)
    _require_random_marker_widths(config, audit_by_symbol, target_width)

    output_root.mkdir(parents=True)
    config_hash = sha256_json(asdict(config))
    manifests: dict[str, ArtifactManifest] = {}
    try:
        for arm in config.arms:
            rows = [
                render_training_example(
                    trace,
                    arm,
                    config.pilot_seed,
                    512,
                    neutral_delimiter=neutral_delimiter,
                )
                for trace in traces
            ]
            row_hashes = [sha256_json(row) for row in rows]
            semantic_hash = _semantic_dataset_hash(rows, neutral_delimiter)
            filler_lengths = _filler_length_counts(rows) if arm == "filler" else {}
            metadata = {
                "semantic_dataset_hash": semantic_hash,
                "row_hashes": row_hashes,
                "exclusions": [],
                "filler_length_counts": filler_lengths,
                "tokenizer_revision": _tokenizer_revision(tokenizer, config.model_id),
                "parent_split_hash": split_hash,
                "neutral_delimiter": neutral_delimiter,
                "target_marker_token_width": target_width,
                "token_audit": [asdict(row) for row in audit],
                "local_frequency_label": LOCAL_FREQUENCY_LABEL,
            }
            artifact_id = sha256_json(
                {
                    "arm": arm,
                    "config_hash": config_hash,
                    "parent_split_hash": split_hash,
                    "row_hashes": row_hashes,
                    "metadata": metadata,
                }
            )
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                kind="phase_marker_training_data",
                config_hash=config_hash,
                parent_hashes=(split_hash,),
                row_count=len(rows),
                metadata=metadata,
            )
            write_jsonl_atomic(output_root / f"{arm}.jsonl", rows)
            _write_manifest(output_root / f"{arm}.manifest.json", manifest)
            manifests[arm] = manifest
    except BaseException:
        for path in output_root.iterdir():
            path.unlink()
        output_root.rmdir()
        raise
    return manifests


def _encode(tokenizer: object, symbol: str) -> Sequence[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must provide encode")
    token_ids = encode(symbol, add_special_tokens=False)
    if not isinstance(token_ids, Sequence) or not all(isinstance(token_id, int) for token_id in token_ids):
        raise TypeError("tokenizer.encode must return token ids")
    return token_ids


def _token_strings(tokenizer: object, token_ids: Sequence[int]) -> Sequence[str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise TypeError("tokenizer must provide convert_ids_to_tokens")
    strings = convert(list(token_ids))
    if not isinstance(strings, Sequence) or not all(isinstance(value, str) for value in strings):
        raise TypeError("tokenizer.convert_ids_to_tokens must return token strings")
    return strings


def _vocabulary(tokenizer: object) -> Mapping[str, object]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TypeError("tokenizer must provide get_vocab")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping):
        raise TypeError("tokenizer.get_vocab must return a mapping")
    return vocabulary


def _parent_split_hash(output_root: Path) -> str:
    manifest_path = output_root.parent / "splits" / "manifest.json"
    if not manifest_path.is_file():
        raise SplitLineageUnavailable(
            f"frozen split manifest required at {manifest_path}; refusing to invent parent split lineage"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SplitLineageUnavailable(f"invalid frozen split manifest: {manifest_path}") from error
    artifact_id = payload.get("artifact_id") if isinstance(payload, Mapping) else None
    if not isinstance(artifact_id, str) or len(artifact_id) != 64 or any(
        character not in string.hexdigits for character in artifact_id
    ):
        raise SplitLineageUnavailable(f"frozen split manifest has no valid artifact_id: {manifest_path}")
    return artifact_id.lower()


def _canonical_training_corpus(traces: Sequence[CanonicalTrace]) -> tuple[str, ...]:
    return tuple(
        value
        for trace in traces
        for value in (trace.question, trace.answer, *(phase.body for phase in trace.phases))
    )


def _require_random_marker_widths(
    config: ExperimentConfig, audit_by_symbol: Mapping[str, TokenAuditRow], target_width: int
) -> None:
    random_widths = {symbol: audit_by_symbol[symbol].token_count for symbol in config.phase_markers}
    mismatches = {symbol: width for symbol, width in random_widths.items() if width != target_width}
    if mismatches:
        raise TokenWidthMismatch(
            f"random-marker permutation cannot match target width {target_width}: {mismatches}"
        )


def _semantic_dataset_hash(rows: Sequence[Mapping[str, object]], neutral_delimiter: str) -> str:
    projected: list[dict[str, str]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError("rendered training row does not contain two messages")
        user, assistant = messages
        if not isinstance(user, Mapping) or not isinstance(assistant, Mapping):
            raise ValueError("rendered training messages must be mappings")
        question = user.get("content")
        content = assistant.get("content")
        if not isinstance(question, str) or not isinstance(content, str):
            raise ValueError("rendered training message content must be strings")
        projected.append(
            {"question": question, "assistant": semantic_projection(content, neutral_delimiter=neutral_delimiter)}
        )
    return sha256_json(projected)


def _filler_length_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        messages = row["messages"]
        assert isinstance(messages, list)
        assistant = messages[1]
        assert isinstance(assistant, Mapping)
        content = assistant["content"]
        assert isinstance(content, str)
        prefix = content.split("Final answer:", 1)[0].strip()
        counts[str(len(prefix))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _tokenizer_revision(tokenizer: object, model_id: str) -> str:
    explicit = getattr(tokenizer, "revision", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        commit_hash = init_kwargs.get("_commit_hash")
        if isinstance(commit_hash, str) and commit_hash:
            return commit_hash
    if model_id == "Qwen/Qwen2.5-7B-Instruct":
        return QWEN25_7B_TOKENIZER_REVISION
    if isinstance(init_kwargs, Mapping):
        revision = init_kwargs.get("revision")
        if isinstance(revision, str) and revision:
            return revision
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path
    return "unknown"


def _write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    path.write_text(canonical_json(asdict(manifest)) + "\n", encoding="utf-8")


def _load_cached_tokenizer(model_id: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, local_files_only=True)


def _load_frozen_training_traces(output_root: Path) -> tuple[CanonicalTrace, ...]:
    split_root = output_root.parent / "splits"
    train_path = split_root / "train.jsonl"
    if not train_path.is_file():
        raise SplitLineageUnavailable(
            f"frozen training split required at {train_path}; refusing to materialize unapproved traces"
        )
    selected = {
        (str(row["source"]), str(row["question"]))
        for row in read_jsonl(train_path)
        if row.get("split") == "train"
    }
    traces, _, _ = parse_trace_pool(Path("data/sft_final.jsonl"))
    # The split builder uses official-source labels while legacy traces retain their original
    # source. Match the frozen question set and let the manifest remain the lineage authority.
    questions = {question for _, question in selected}
    return tuple(trace for trace in traces if trace.question in questions)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--limit", type=int, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "materialize":
        if arguments.limit < 1:
            raise SystemExit("--limit must be positive")
        config = ExperimentConfig.load(arguments.config)
        traces = _load_frozen_training_traces(arguments.output_root)[: arguments.limit]
        if len(traces) != arguments.limit:
            raise SystemExit(
                f"frozen training split provides {len(traces)} canonical traces, need --limit {arguments.limit}"
            )
        tokenizer = _load_cached_tokenizer(config.model_id)
        manifests = materialize_training_arms(config, traces, tokenizer, arguments.output_root)
        print(canonical_json({arm: manifest.artifact_id for arm, manifest in manifests.items()}))


if __name__ == "__main__":
    main()
