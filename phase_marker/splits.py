"""Frozen, contamination-free source, validation, and benchmark splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Protocol
import unicodedata

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, read_jsonl, sha256_json, write_jsonl_atomic
from phase_marker.schema import CanonicalTrace
from phase_marker.traces import TraceParseError, parse_legacy_trace


GSM8K_DATASET = ("gsm8k", "main", "main")
SVAMP_DATASET = ("ChilleD/SVAMP", None, "main")
MATH_DATASET = ("EleutherAI/hendrycks_math", "all", "main")
VALIDATION_PER_SOURCE = 300
SVAMP_TEST_SIZE = 1000


class SplitOverlapError(ValueError):
    """A normalized source/question hash occurs in more than one live split."""


class DatasetCacheMiss(RuntimeError):
    """An immutable benchmark revision is unavailable in the local cache."""

    def __init__(self, dataset_id: str, revision: str):
        self.dataset_id = dataset_id
        self.revision = revision
        super().__init__(f"missing cached dataset {dataset_id}@{revision}")


@dataclass(frozen=True)
class DatasetExample:
    source: str
    split: str
    example_id: str
    question: str
    answer: str
    question_hash: str


@dataclass(frozen=True)
class SplitBundle:
    train: tuple[DatasetExample, ...] = ()
    validation: tuple[DatasetExample, ...] = ()
    test: tuple[DatasetExample, ...] = ()
    exclusions: tuple[DatasetExample, ...] = ()

    def __post_init__(self) -> None:
        for field in ("train", "validation", "test", "exclusions"):
            object.__setattr__(self, field, tuple(getattr(self, field)))


class DatasetLoader(Protocol):
    def load(
        self, dataset_id: str, config: str | None, split: str, revision: str
    ) -> Sequence[Mapping[str, object]]: ...


class OfflineDatasetLoader:
    """Hugging Face loader that never attempts a network request."""

    def load(
        self, dataset_id: str, config: str | None, split: str, revision: str
    ) -> Sequence[Mapping[str, object]]:
        # These process-wide flags protect Hub resolution in addition to datasets'
        # local-files-only download configuration.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import datasets

            rows = datasets.load_dataset(
                dataset_id,
                config,
                split=split,
                revision=revision,
                download_config=datasets.DownloadConfig(local_files_only=True),
            )
        except Exception as error:
            raise DatasetCacheMiss(dataset_id, revision) from error
        return tuple(dict(row) for row in rows)


def normalize_question(text: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(text.split())).strip().casefold()


def question_hash(source: str, question: str) -> str:
    payload = f"{source}\0{normalize_question(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_bundle(
    config: ExperimentConfig,
    loader: DatasetLoader,
    source_traces: Sequence[CanonicalTrace],
    unified_rows: Sequence[Mapping[str, object]],
) -> SplitBundle:
    """Build the full benchmark bundle before emitting any manifest files."""
    del config  # The frozen config is an explicit pipeline dependency and manifest parent.
    official = _load_official_rows(loader)
    recovered, exclusions = _recover_training_traces(source_traces, unified_rows)

    train = tuple(
        _example(source, "train", trace.trace_id, trace.question, trace.answer)
        for source, trace in recovered
        if source != "svamp"
    )
    exclusions.extend(
        _example("svamp", "excluded_svamp", trace.trace_id, trace.question, trace.answer)
        for source, trace in recovered
        if source == "svamp"
    )

    used_training_hashes = {row.question_hash for row in train}
    validation = tuple(
        row
        for source in ("gsm8k", "math")
        for row in _validation_rows(official[source]["train"], used_training_hashes, source)
    )
    test = tuple(
        row
        for source in ("gsm8k", "svamp", "math")
        for row in official[source]["test"]
    )
    bundle = SplitBundle(train=train, validation=validation, test=test, exclusions=exclusions)
    assert_disjoint_splits(bundle)
    return bundle


def assert_disjoint_splits(bundle: SplitBundle) -> None:
    """Reject any question appearing in two of train, validation, and test."""
    seen: dict[str, tuple[str, DatasetExample]] = {}
    for split_name, rows in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        for row in rows:
            prior = seen.get(row.question_hash)
            if prior is not None and prior[0] != split_name:
                raise SplitOverlapError(
                    f"{row.source} question overlap between {prior[0]} and {split_name}: "
                    f"{row.question_hash}"
                )
            seen[row.question_hash] = (split_name, row)


def write_split_bundle(output_root: Path, config: ExperimentConfig, bundle: SplitBundle) -> None:
    """Atomically publish a complete immutable bundle, never a partial manifest."""
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen split manifests: {output_root}")
    assert_disjoint_splits(bundle)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root.parent, prefix=f".{output_root.name}-") as temporary:
        staging = Path(temporary)
        for split_name, rows in (
            ("train", bundle.train),
            ("validation", bundle.validation),
            ("test", bundle.test),
            ("exclusions", bundle.exclusions),
        ):
            write_jsonl_atomic(staging / f"{split_name}.jsonl", (_row(row) for row in rows))
        manifest = {
            "artifact_id": sha256_json(
                {
                    "config": asdict(config),
                    "train": [_row(row) for row in bundle.train],
                    "validation": [_row(row) for row in bundle.validation],
                    "test": [_row(row) for row in bundle.test],
                    "exclusions": [_row(row) for row in bundle.exclusions],
                }
            ),
            "config_hash": sha256_json(asdict(config)),
            "dataset_revisions": {
                "gsm8k": GSM8K_DATASET[2],
                "svamp": SVAMP_DATASET[2],
                "math": MATH_DATASET[2],
            },
            "overlap_count": 0,
            "source_counts": {
                name: dict(sorted(Counter(row.source for row in rows).items()))
                for name, rows in (
                    ("train", bundle.train),
                    ("validation", bundle.validation),
                    ("test", bundle.test),
                    ("exclusions", bundle.exclusions),
                )
            },
        }
        (staging / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        staging.replace(output_root)


def _load_official_rows(loader: DatasetLoader) -> dict[str, dict[str, tuple[DatasetExample, ...]]]:
    gsm_train = _examples_from_rows(
        "gsm8k", "train", loader.load(GSM8K_DATASET[0], GSM8K_DATASET[1], "train", GSM8K_DATASET[2])
    )
    gsm_test = _examples_from_rows(
        "gsm8k", "test", loader.load(GSM8K_DATASET[0], GSM8K_DATASET[1], "test", GSM8K_DATASET[2])
    )
    svamp_test = _examples_from_rows(
        "svamp", "test", loader.load(SVAMP_DATASET[0], SVAMP_DATASET[1], "train", SVAMP_DATASET[2])
    )
    if len(svamp_test) != SVAMP_TEST_SIZE:
        raise ValueError(f"SVAMP held-out dataset must contain exactly {SVAMP_TEST_SIZE} rows, got {len(svamp_test)}")
    math_train = _examples_from_rows(
        "math", "train", loader.load(MATH_DATASET[0], MATH_DATASET[1], "train", MATH_DATASET[2])
    )
    math_test = _examples_from_rows(
        "math", "test", loader.load(MATH_DATASET[0], MATH_DATASET[1], "test", MATH_DATASET[2])
    )
    return {
        "gsm8k": {"train": gsm_train, "test": gsm_test},
        "svamp": {"test": svamp_test},
        "math": {"train": math_train, "test": math_test},
    }


def _recover_training_traces(
    traces: Sequence[CanonicalTrace], unified_rows: Sequence[Mapping[str, object]]
) -> tuple[list[tuple[str, CanonicalTrace]], list[DatasetExample]]:
    sources_by_question: dict[str, set[str]] = defaultdict(set)
    for row in unified_rows:
        question = _required_text(row, "question")
        source = _required_text(row, "source").casefold()
        sources_by_question[normalize_question(question)].add(source)

    recovered: list[tuple[str, CanonicalTrace]] = []
    exclusions: list[DatasetExample] = []
    for trace in traces:
        sources = sources_by_question.get(normalize_question(trace.question), set())
        if len(sources) == 1:
            recovered.append((next(iter(sources)), trace))
        elif not sources:
            exclusions.append(
                _example("unknown", "excluded_unmatched", trace.trace_id, trace.question, trace.answer)
            )
        else:
            exclusions.append(
                _example("unknown", "excluded_ambiguous", trace.trace_id, trace.question, trace.answer)
            )
    return recovered, exclusions


def _validation_rows(
    candidates: Sequence[DatasetExample], used_training_hashes: set[str], source: str
) -> tuple[DatasetExample, ...]:
    unused = [row for row in candidates if row.question_hash not in used_training_hashes]
    selected = tuple(sorted(unused, key=lambda row: (row.question_hash, row.example_id))[:VALIDATION_PER_SOURCE])
    if len(selected) != VALIDATION_PER_SOURCE:
        raise ValueError(f"need {VALIDATION_PER_SOURCE} unused {source} train rows, got {len(selected)}")
    return selected


def _examples_from_rows(
    source: str, split: str, rows: Sequence[Mapping[str, object]]
) -> tuple[DatasetExample, ...]:
    return tuple(
        _example(
            source,
            split,
            _row_id(row, source, split, index),
            _question_from_row(source, row),
            _answer_from_row(source, row),
        )
        for index, row in enumerate(rows)
    )


def _question_from_row(source: str, row: Mapping[str, object]) -> str:
    if source == "svamp":
        body = _required_text(row, "Body")
        question = _required_text(row, "Question")
        return f"{body} {question}".strip()
    if source == "math":
        return _first_text(row, "problem", "question")
    return _required_text(row, "question")


def _answer_from_row(source: str, row: Mapping[str, object]) -> str:
    if source == "gsm8k":
        answer = _required_text(row, "answer")
        return answer.rsplit("####", 1)[-1].strip()
    if source == "svamp":
        value = row.get("Answer", row.get("answer"))
        if value is None:
            raise ValueError("SVAMP row missing Answer")
        return str(value).strip()
    return _boxed_answer(_first_text(row, "solution", "answer"))


def _boxed_answer(solution: str) -> str:
    marker = "\\boxed{"
    start = solution.rfind(marker)
    if start == -1:
        return solution.strip()
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[content_start:index].strip()
    return solution.strip()


def _row_id(row: Mapping[str, object], source: str, split: str, index: int) -> str:
    for key in ("id", "ID"):
        value = row.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return f"{source}-{split}-{index}"


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset row missing nonempty {key}")
    return value.strip()


def _first_text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"dataset row missing nonempty {' or '.join(keys)}")


def _example(source: str, split: str, example_id: str, question: str, answer: str) -> DatasetExample:
    return DatasetExample(
        source=source,
        split=split,
        example_id=example_id,
        question=question,
        answer=answer,
        question_hash=question_hash(source, question),
    )


def _row(row: DatasetExample) -> dict[str, str]:
    return {
        "source": row.source,
        "split": row.split,
        "example_id": row.example_id,
        "question": row.question,
        "answer": row.answer,
        "question_hash": row.question_hash,
    }


def _parse_traces(path: Path) -> tuple[CanonicalTrace, ...]:
    traces: list[CanonicalTrace] = []
    for row in read_jsonl(path):
        try:
            traces.append(parse_legacy_trace(row))
        except TraceParseError:
            continue
    return tuple(traces)


def main(argv: Sequence[str] | None = None, *, loader: DatasetLoader | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--traces", type=Path, required=True)
    build.add_argument("--unified", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "build":
        config = ExperimentConfig.load(arguments.config)
        try:
            bundle = build_split_bundle(
                config,
                loader or OfflineDatasetLoader(),
                _parse_traces(arguments.traces),
                tuple(read_jsonl(arguments.unified)),
            )
        except DatasetCacheMiss as error:
            raise SystemExit(str(error)) from error
        write_split_bundle(arguments.output_root, config, bundle)
        print(
            canonical_json(
                {
                    "overlap_count": 0,
                    "svamp_test_rows": sum(row.source == "svamp" for row in bundle.test),
                }
            )
        )


if __name__ == "__main__":
    main()
