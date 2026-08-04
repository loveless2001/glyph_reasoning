"""Frozen, contamination-free source, validation, and benchmark splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
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
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)


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
    *,
    dataset_specs: Sequence[Mapping[str, object]] | None = None,
) -> SplitBundle:
    """Build the full benchmark bundle before emitting any manifest files."""
    del config  # The frozen config is an explicit pipeline dependency and manifest parent.
    official = _load_official_rows(loader, dataset_specs or _dataset_specs())
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


def write_split_bundle(
    output_root: Path,
    config: ExperimentConfig,
    bundle: SplitBundle,
    *,
    dataset_specs: Sequence[Mapping[str, object]] | None = None,
    input_lineage: Mapping[str, Mapping[str, str]] | None = None,
    source_pool_accounting: Mapping[str, int] | None = None,
    parse_exclusion_provenance: Sequence[str] | None = None,
) -> None:
    """Atomically publish a complete immutable bundle, never a partial manifest."""
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite frozen split manifests: {output_root}")
    assert_disjoint_splits(bundle)
    frozen_specs = _validate_frozen_dataset_specs(dataset_specs or _dataset_specs())
    frozen_lineage = _validate_input_lineage(input_lineage)
    accounting = _validate_source_pool_accounting(source_pool_accounting)
    parse_provenance = _validate_parse_exclusion_provenance(
        bundle, accounting, parse_exclusion_provenance
    )
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
                    "datasets": frozen_specs,
                    "input_lineage": frozen_lineage,
                    "source_pool_accounting": accounting,
                    "parse_exclusion_provenance": parse_provenance,
                }
            ),
            "config_hash": sha256_json(asdict(config)),
            "datasets": frozen_specs,
            "input_lineage": frozen_lineage,
            "overlap_count": 0,
            "source_pool_accounting": accounting,
            "parse_exclusion_provenance": parse_provenance,
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


def _load_official_rows(
    loader: DatasetLoader, dataset_specs: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, tuple[DatasetExample, ...]]]:
    specs = {
        (_required_spec_text(spec, "source"), _required_spec_text(spec, "requested_split")): spec
        for spec in dataset_specs
    }
    gsm_train_spec = specs[("gsm8k", "train")]
    gsm_train = _examples_from_rows(
        "gsm8k", "train", _load_spec(loader, gsm_train_spec)
    )
    gsm_test_spec = specs[("gsm8k", "test")]
    gsm_test = _examples_from_rows(
        "gsm8k", "test", _load_spec(loader, gsm_test_spec)
    )
    svamp_spec = specs[("svamp", "train")]
    svamp_test = _examples_from_rows(
        "svamp", "test", _load_spec(loader, svamp_spec)
    )
    if len(svamp_test) != SVAMP_TEST_SIZE:
        raise ValueError(f"SVAMP held-out dataset must contain exactly {SVAMP_TEST_SIZE} rows, got {len(svamp_test)}")
    math_train_spec = specs[("math", "train")]
    math_train = _examples_from_rows(
        "math", "train", _load_spec(loader, math_train_spec)
    )
    math_test_spec = specs[("math", "test")]
    math_test = _examples_from_rows(
        "math", "test", _load_spec(loader, math_test_spec)
    )
    return {
        "gsm8k": {"train": gsm_train, "test": gsm_test},
        "svamp": {"test": svamp_test},
        "math": {"train": math_train, "test": math_test},
    }


def _recover_training_traces(
    traces: Sequence[CanonicalTrace], unified_rows: Sequence[Mapping[str, object]]
) -> tuple[list[tuple[str, CanonicalTrace]], list[DatasetExample]]:
    rows_by_question: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for index, row in enumerate(unified_rows):
        question = _required_text(row, "question")
        source = _required_text(row, "source").casefold()
        rows_by_question[normalize_question(question)].append((source, _unified_row_id(row, index)))

    recovered: list[tuple[str, CanonicalTrace]] = []
    exclusions: list[DatasetExample] = []
    for trace in traces:
        matches = rows_by_question.get(normalize_question(trace.question), [])
        if len(matches) == 1:
            recovered.append((matches[0][0], trace))
        elif not matches:
            exclusions.append(
                _example("unknown", "excluded_unmatched", trace.trace_id, trace.question, trace.answer)
            )
        else:
            candidates = ",".join(f"{source}:{row_id}" for source, row_id in matches)
            exclusions.append(
                _example(
                    "unknown",
                    "excluded_ambiguous",
                    f"{trace.trace_id}|candidates={candidates}",
                    trace.question,
                    trace.answer,
                )
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


def parse_trace_pool(
    path: Path,
) -> tuple[tuple[CanonicalTrace, ...], tuple[DatasetExample, ...], dict[str, int]]:
    traces: list[CanonicalTrace] = []
    exclusions: list[DatasetExample] = []
    input_rows = 0
    with path.open(encoding="utf-8") as handle:
        rows = enumerate(handle, start=1)
        for line_number, line in rows:
            if not line.strip():
                continue
            input_rows += 1
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise TraceParseError("invalid_row")
                traces.append(parse_legacy_trace(row))
            except json.JSONDecodeError:
                exclusions.append(_parse_exclusion(line_number, "invalid_json"))
            except TraceParseError as error:
                exclusions.append(_parse_exclusion(line_number, error.code))
    accounting = {
        "input_rows": input_rows,
        "parsed": len(traces),
        "parse_exclusions": len(exclusions),
    }
    return tuple(traces), tuple(exclusions), accounting


def _dataset_specs(
    *, gsm8k_revision: str = GSM8K_DATASET[2], svamp_revision: str = SVAMP_DATASET[2], math_revision: str = MATH_DATASET[2]
) -> tuple[dict[str, object], ...]:
    return (
        _dataset_spec("gsm8k", GSM8K_DATASET[0], GSM8K_DATASET[1], "train", gsm8k_revision),
        _dataset_spec("gsm8k", GSM8K_DATASET[0], GSM8K_DATASET[1], "test", gsm8k_revision),
        _dataset_spec("svamp", SVAMP_DATASET[0], SVAMP_DATASET[1], "train", svamp_revision),
        _dataset_spec("math", MATH_DATASET[0], MATH_DATASET[1], "train", math_revision),
        _dataset_spec("math", MATH_DATASET[0], MATH_DATASET[1], "test", math_revision),
    )


def _dataset_spec(
    source: str, dataset_id: str, config: str | None, requested_split: str, revision: str
) -> dict[str, object]:
    return {
        "source": source,
        "dataset_id": dataset_id,
        "config": config,
        "requested_split": requested_split,
        "revision": revision,
    }


def _load_spec(loader: DatasetLoader, spec: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    config = spec.get("config")
    if config is not None and not isinstance(config, str):
        raise ValueError("dataset config must be a string or null")
    return loader.load(
        _required_spec_text(spec, "dataset_id"),
        config,
        _required_spec_text(spec, "requested_split"),
        _required_spec_text(spec, "revision"),
    )


def _validate_frozen_dataset_specs(
    dataset_specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {
        ("gsm8k", "gsm8k", "main", "train"),
        ("gsm8k", "gsm8k", "main", "test"),
        ("svamp", "ChilleD/SVAMP", None, "train"),
        ("math", "EleutherAI/hendrycks_math", "all", "train"),
        ("math", "EleutherAI/hendrycks_math", "all", "test"),
    }
    normalized = [dict(spec) for spec in dataset_specs]
    actual = {
        (spec.get("source"), spec.get("dataset_id"), spec.get("config"), spec.get("requested_split"))
        for spec in normalized
    }
    if actual != expected or len(normalized) != len(expected):
        raise ValueError("frozen publication requires complete dataset specs")
    for spec in normalized:
        revision = _required_spec_text(spec, "revision")
        if IMMUTABLE_REVISION.fullmatch(revision) is None:
            raise ValueError(
                f"frozen publication requires immutable commit revision for {spec['dataset_id']}: {revision}"
            )
    return normalized


def _validate_input_lineage(
    input_lineage: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if input_lineage is None:
        raise ValueError("frozen publication requires trace and unified input lineage")
    normalized: dict[str, dict[str, str]] = {}
    for name in ("traces", "unified"):
        item = input_lineage.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"frozen publication requires {name} input lineage")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest, re.IGNORECASE):
            raise ValueError(f"invalid {name} input lineage")
        normalized[name] = {"path": path, "sha256": digest}
    return normalized


def _validate_source_pool_accounting(
    source_pool_accounting: Mapping[str, int] | None,
) -> dict[str, int]:
    if source_pool_accounting is None:
        raise ValueError("frozen publication requires source-pool accounting")
    expected_keys = {"input_rows", "parsed", "parse_exclusions"}
    if set(source_pool_accounting) != expected_keys or any(
        not isinstance(value, int) or value < 0 for value in source_pool_accounting.values()
    ):
        raise ValueError("invalid source-pool accounting")
    accounting = dict(source_pool_accounting)
    if accounting["input_rows"] != accounting["parsed"] + accounting["parse_exclusions"]:
        raise ValueError("source-pool accounting does not cover every input row")
    return accounting


def _validate_parse_exclusion_provenance(
    bundle: SplitBundle,
    accounting: Mapping[str, int],
    parse_exclusion_provenance: Sequence[str] | None,
) -> list[str]:
    if parse_exclusion_provenance is None:
        raise ValueError("frozen publication requires parse exclusion provenance")
    claimed = sorted(parse_exclusion_provenance)
    if len(claimed) != len(set(claimed)) or any(
        re.fullmatch(r"line-[1-9][0-9]*\|[a-z_]+", item) is None for item in claimed
    ):
        raise ValueError("invalid parse exclusion provenance")
    actual = sorted(
        _parse_exclusion_identity(row)
        for row in bundle.exclusions
        if row.split.startswith("excluded_parse_")
    )
    if (
        accounting["parse_exclusions"] != len(claimed)
        or actual != claimed
    ):
        raise ValueError("parse exclusion provenance does not match frozen exclusions")
    return claimed


def _required_spec_text(spec: Mapping[str, object], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"dataset spec missing {key}")
    return value


def _unified_row_id(row: Mapping[str, object], index: int) -> str:
    value = row.get("id")
    return str(value) if isinstance(value, (str, int)) else f"unified-line-{index + 1}"


def _parse_exclusion(line_number: int, reason: str) -> DatasetExample:
    return _example("legacy", f"excluded_parse_{reason}", f"line-{line_number}", "", "")


def _parse_exclusion_identity(row: DatasetExample) -> str:
    reason = row.split.removeprefix("excluded_parse_")
    return f"{row.example_id}|{reason}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_traces(path: Path) -> tuple[CanonicalTrace, ...]:
    """Backward-compatible parsed-trace accessor; callers needing accounting use parse_trace_pool."""
    return parse_trace_pool(path)[0]


def main(argv: Sequence[str] | None = None, *, loader: DatasetLoader | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--traces", type=Path, required=True)
    build.add_argument("--unified", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--gsm8k-revision", default=GSM8K_DATASET[2])
    build.add_argument("--svamp-revision", default=SVAMP_DATASET[2])
    build.add_argument("--math-revision", default=MATH_DATASET[2])
    arguments = parser.parse_args(argv)

    if arguments.command == "build":
        config = ExperimentConfig.load(arguments.config)
        traces, parse_exclusions, accounting = parse_trace_pool(arguments.traces)
        dataset_specs = _dataset_specs(
            gsm8k_revision=arguments.gsm8k_revision,
            svamp_revision=arguments.svamp_revision,
            math_revision=arguments.math_revision,
        )
        try:
            bundle = build_split_bundle(
                config,
                loader or OfflineDatasetLoader(),
                traces,
                tuple(read_jsonl(arguments.unified)),
                dataset_specs=dataset_specs,
            )
        except DatasetCacheMiss as error:
            raise SystemExit(str(error)) from error
        bundle = SplitBundle(
            train=bundle.train,
            validation=bundle.validation,
            test=bundle.test,
            exclusions=(*parse_exclusions, *bundle.exclusions),
        )
        input_lineage = {
            "traces": {"path": str(arguments.traces), "sha256": _sha256_file(arguments.traces)},
            "unified": {"path": str(arguments.unified), "sha256": _sha256_file(arguments.unified)},
        }
        write_split_bundle(
            arguments.output_root,
            config,
            bundle,
            dataset_specs=dataset_specs,
            input_lineage=input_lineage,
            source_pool_accounting=accounting,
            parse_exclusion_provenance=tuple(
                _parse_exclusion_identity(row) for row in parse_exclusions
            ),
        )
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
