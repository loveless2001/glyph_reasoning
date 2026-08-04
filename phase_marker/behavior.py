"""Behavioral evaluation cells, backends, and immutable raw generation records."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol

from phase_marker.config import ExperimentConfig
from phase_marker.io import sha256_json, write_jsonl_atomic
from phase_marker.prompts import MarkerSet, render_perturbation, render_prompt
from phase_marker.schema import ArtifactManifest, GenerationRecord
from phase_marker.scoring import score_generation
from phase_marker.splits import DatasetExample
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION


PRIMARY_ARMS = ("semantic", "glyph", "dot", "random")
PRIMARY_PROMPTS = ("neutral", "glyph", "dot", "headings")
SAMPLED_CONTRASTS = (
    ("semantic", "neutral"),
    ("glyph", "neutral"),
    ("glyph", "glyph"),
    ("glyph", "dot"),
)
PERTURBATIONS = ("delete", "cluster", "displace", "permute", "dot_replace", "unseen_replace")
FAKE_RUN_KIND = "fake_smoke"
PRODUCTION_RUN_KIND = "production"
FAKE_TOKENIZER_REVISION = "fake-byte-tokenizer"


@dataclass(frozen=True)
class EvaluationCell:
    kind: str
    training_arm: str
    prompt_condition: str
    perturbation: str | None
    decoding_name: str


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    decoding: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        object.__setattr__(self, "decoding", dict(self.decoding))
        if not self.generation_id or not self.prompt or self.max_new_tokens < 1:
            raise ValueError("generation requests require an ID, prompt, and positive token limit")


@dataclass(frozen=True)
class GenerationOutput:
    generation_id: str
    text: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if not self.generation_id:
            raise ValueError("generation outputs require a generation ID")


@dataclass(frozen=True)
class ProvenanceEnvelope:
    run_kind: str
    adapter_seed: int
    config_hash: str
    tokenizer_revision: str
    split_artifact_id: str
    split_parent_hashes: tuple[str, ...]
    checkpoint: str
    parent_hashes: tuple[str, ...]


class GenerationBackend(Protocol):
    """Backend contract: return one unique output in exactly request order."""

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]: ...


class FakeGenerationBackend:
    """Deterministic non-model backend used only for dry runs and tests."""

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]:
        _validate_requests(requests)
        outputs = tuple(
            GenerationOutput(
                generation_id=request.generation_id,
                text=f"Fake reasoning.\nFinal answer: {request.decoding.get('fake_answer', '0')}",
                token_ids=(0,),
                token_logprobs=(0.0,),
            )
            for request in requests
        )
        _validate_outputs(requests, outputs)
        return outputs


class VLLMGenerationBackend:
    """Lazy production adapter; importing this module never imports or loads vLLM."""

    def __init__(
        self, model: str, *, adapter_seed: int | None = None, **llm_kwargs: object
    ) -> None:
        if not _is_production_checkpoint(model):
            raise ValueError("vLLM backend requires a real checkpoint model identifier")
        if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
            raise ValueError("vLLM backend requires an explicit integer adapter seed")
        self._model = model
        self._adapter_seed = adapter_seed
        self._llm_kwargs = dict(llm_kwargs)
        self._llm: Any | None = None

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]:
        _validate_requests(requests)
        if not requests:
            return ()
        for request in requests:
            _validate_production_request(request)
            if request.decoding["checkpoint"] != self._model:
                raise ValueError("vLLM request checkpoint must exactly match backend model")
            if request.decoding["adapter_seed"] != self._adapter_seed:
                raise ValueError("vLLM request must match backend adapter seed")
        if self._llm is None:
            from vllm import LLM  # Imported only when the production backend is used.

            self._llm = LLM(model=self._model, **self._llm_kwargs)
        from vllm import SamplingParams

        outputs: list[GenerationOutput] = []
        for request in requests:
            results = self._llm.generate(
                prompt_token_ids=[list(request.prompt_token_ids)],
                sampling_params=SamplingParams(**_vllm_sampling_parameters((request,))),
                use_tqdm=False,
            )
            if len(results) != 1:
                raise ValueError("vLLM returned a missing output")
            result = results[0]
            candidate = result.outputs[0]
            outputs.append(
                GenerationOutput(
                    generation_id=request.generation_id,
                    text=candidate.text,
                    token_ids=tuple(candidate.token_ids),
                    token_logprobs=_vllm_logprobs(candidate),
                )
            )
        _validate_outputs(requests, outputs)
        return tuple(outputs)


def build_behavior_matrix(
    config: ExperimentConfig, split_manifest: ArtifactManifest
) -> tuple[EvaluationCell, ...]:
    """Build the frozen greedy and sampled behavior matrix without generating."""
    if split_manifest.kind != "phase_marker_splits":
        raise ValueError("behavior evaluation requires a phase_marker_splits manifest")
    missing = set(PRIMARY_ARMS).difference(config.arms)
    if missing:
        raise ValueError(f"config lacks required primary arms: {sorted(missing)}")
    primary = tuple(
        EvaluationCell("primary", arm, prompt, None, "greedy")
        for arm in PRIMARY_ARMS
        for prompt in PRIMARY_PROMPTS
    )
    sampled = tuple(
        EvaluationCell("sampled", arm, prompt, None, "sampled")
        for arm, prompt in SAMPLED_CONTRASTS
    )
    perturbations = tuple(
        EvaluationCell("perturbation", "glyph", "glyph", perturbation, "greedy")
        for perturbation in PERTURBATIONS
    )
    return primary + sampled + perturbations


def records_from_outputs(
    cell: EvaluationCell,
    examples: Sequence[DatasetExample],
    requests: Sequence[GenerationRequest],
    outputs: Sequence[GenerationOutput],
    parent_hashes: tuple[str, ...],
) -> list[GenerationRecord]:
    """Convert ordered raw backend output into independently rescorable records."""
    expanded_examples = _expand_examples(cell, examples, requests)
    _validate_requests(requests)
    _validate_outputs(requests, outputs)
    rows: list[GenerationRecord] = []
    for example, request, output in zip(expanded_examples, requests, outputs):
        sampling_seed = request.decoding.get("seed")
        adapter_seed = request.decoding.get("adapter_seed")
        checkpoint = request.decoding.get("checkpoint")
        if not isinstance(sampling_seed, int) or isinstance(sampling_seed, bool):
            raise ValueError("generation request decoding must include integer seed")
        if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
            raise ValueError("generation request decoding must include integer adapter seed")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("generation request decoding must include checkpoint")
        decoding = dict(request.decoding)
        decoding.update(
            {
                "max_new_tokens": request.max_new_tokens,
                "completion_token_logprobs": list(output.token_logprobs),
                "evaluation_kind": cell.kind,
                "perturbation": cell.perturbation,
            }
        )
        rows.append(
            GenerationRecord(
                generation_id=request.generation_id,
                source=example.source,
                question_hash=example.question_hash,
                gold_answer=example.answer,
                training_arm=cell.training_arm,
                seed=adapter_seed,
                checkpoint=checkpoint,
                prompt_condition=cell.prompt_condition,
                prompt_hash=_text_hash(request.prompt),
                raw_prompt=request.prompt,
                raw_completion=output.text,
                prompt_token_ids=request.prompt_token_ids,
                completion_token_ids=output.token_ids,
                decoding=decoding,
                parent_hashes=tuple(parent_hashes),
            )
        )
    return rows


def build_generation_requests(
    cell: EvaluationCell,
    examples: Sequence[DatasetExample],
    config: ExperimentConfig,
    *,
    checkpoint: str = "unconfigured://checkpoint",
    tokenize: Callable[[str], Sequence[int]] | None = None,
    tokenizer_revision: str | None = None,
    split_manifest: ArtifactManifest | None = None,
    adapter_seed: int | None = None,
    fake: bool = False,
) -> tuple[GenerationRequest, ...]:
    """Expand each sampled prompt into five independently persisted requests."""
    if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
        raise ValueError("request construction requires an explicit integer adapter seed")
    if adapter_seed not in (config.pilot_seed, *config.confirmatory_seeds):
        raise ValueError("adapter seed is not declared by the experiment config")
    if fake:
        if not checkpoint.startswith("fake://"):
            raise ValueError("fake request construction requires a fake:// checkpoint")
        if tokenize is not None or tokenizer_revision is not None or split_manifest is not None:
            raise ValueError("fake request construction uses only the fake byte tokenizer")
        encoder = _fake_tokenize
        run_kind = FAKE_RUN_KIND
        request_tokenizer_revision = FAKE_TOKENIZER_REVISION
        config_hash = sha256_json(asdict(config))
        split_artifact_id = "fake-smoke-split"
        split_parent_hashes: tuple[str, ...] = ()
    else:
        _validate_production_construction(
            config,
            checkpoint,
            tokenize,
            tokenizer_revision,
            split_manifest,
            adapter_seed,
        )
        assert tokenize is not None
        assert tokenizer_revision is not None
        assert split_manifest is not None
        encoder = tokenize
        run_kind = PRODUCTION_RUN_KIND
        request_tokenizer_revision = tokenizer_revision
        config_hash = sha256_json(asdict(config))
        split_artifact_id = split_manifest.artifact_id
        split_parent_hashes = tuple(split_manifest.parent_hashes)
    marker_set = MarkerSet(*config.phase_markers)
    completion_count = 5 if cell.kind == "sampled" else 1
    requests: list[GenerationRequest] = []
    for example in examples:
        prompt = (
            render_perturbation(example.question, cell.perturbation, marker_set)
            if cell.perturbation
            else render_prompt(example.question, cell.prompt_condition, marker_set)
        )
        for completion_index in range(completion_count):
            decoding: dict[str, object] = {
                "seed": adapter_seed + completion_index,
                "adapter_seed": adapter_seed,
                "checkpoint": checkpoint,
                "run_kind": run_kind,
                "config_hash": config_hash,
                "tokenizer_revision": request_tokenizer_revision,
                "split_artifact_id": split_artifact_id,
                "split_parent_hashes": split_parent_hashes,
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            }
            if fake:
                decoding["fake_answer"] = example.answer
            if cell.kind == "sampled":
                decoding.update(
                    {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "completion_index": completion_index,
                    }
                )
            completion_suffix = (
                f":completion-{completion_index}" if cell.kind == "sampled" else ""
            )
            requests.append(
                GenerationRequest(
                    generation_id=(
                        f"{cell.kind}:{cell.training_arm}:{cell.prompt_condition}:"
                        f"{cell.perturbation or 'base'}:{cell.decoding_name}:{example.example_id}"
                        f"{completion_suffix}"
                    ),
                    prompt=prompt,
                    prompt_token_ids=tuple(encoder(prompt)),
                    max_new_tokens=64,
                    decoding=decoding,
                )
            )
    _validate_requests(requests)
    return tuple(requests)


def serialize_generation_record(
    record: GenerationRecord, provenance: ProvenanceEnvelope
) -> dict[str, object]:
    """Add explicit token counts without changing the immutable raw record schema."""
    _validate_envelope_binding(record, provenance)
    row = asdict(record)
    row["prompt_token_count"] = len(record.prompt_token_ids)
    row["completion_token_count"] = len(record.completion_token_ids)
    row["provenance"] = asdict(provenance)
    return row


def _validate_envelope_binding(record: GenerationRecord, provenance: ProvenanceEnvelope) -> None:
    if provenance.checkpoint != record.checkpoint:
        raise ValueError("provenance checkpoint does not match generation record")
    if provenance.parent_hashes != tuple(record.parent_hashes):
        raise ValueError("provenance parent hashes do not match generation record")
    bindings = {
        "run_kind": provenance.run_kind,
        "adapter_seed": provenance.adapter_seed,
        "config_hash": provenance.config_hash,
        "tokenizer_revision": provenance.tokenizer_revision,
        "split_artifact_id": provenance.split_artifact_id,
        "split_parent_hashes": provenance.split_parent_hashes,
    }
    for name, expected in bindings.items():
        actual = record.decoding.get(name)
        if actual != expected:
            raise ValueError(f"provenance {name} does not match generation record")


def build_provenance_envelope(
    record: GenerationRecord, config: ExperimentConfig, split_manifest: ArtifactManifest
) -> ProvenanceEnvelope:
    """Validate additive run lineage before a raw record is persisted."""
    run_kind = record.decoding.get("run_kind")
    if run_kind not in {FAKE_RUN_KIND, PRODUCTION_RUN_KIND}:
        raise ValueError("record provenance requires an explicit run kind")
    config_hash = record.decoding.get("config_hash")
    tokenizer_revision = record.decoding.get("tokenizer_revision")
    split_artifact_id = record.decoding.get("split_artifact_id")
    split_parent_hashes = tuple(record.decoding.get("split_parent_hashes", ()))
    adapter_seed = record.decoding.get("adapter_seed")
    if not all(isinstance(value, str) and value for value in (config_hash, tokenizer_revision, split_artifact_id)):
        raise ValueError("record provenance requires config, tokenizer, and split identifiers")
    if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
        raise ValueError("record provenance requires an integer adapter seed")
    if record.seed != adapter_seed:
        raise ValueError("generation record adapter seed does not match request provenance")
    if run_kind == PRODUCTION_RUN_KIND:
        expected_config_hash = sha256_json(asdict(config))
        if config_hash != expected_config_hash or split_manifest.config_hash != expected_config_hash:
            raise ValueError("production provenance config hash mismatch")
        if tokenizer_revision != QWEN25_7B_TOKENIZER_REVISION:
            raise ValueError("production provenance tokenizer revision mismatch")
        if split_artifact_id != split_manifest.artifact_id:
            raise ValueError("production provenance split artifact mismatch")
        if not split_manifest.parent_hashes or split_parent_hashes != tuple(split_manifest.parent_hashes):
            raise ValueError("production provenance split parents mismatch")
        expected_parents = (split_manifest.artifact_id, *split_manifest.parent_hashes)
        if record.parent_hashes != expected_parents:
            raise ValueError("production provenance parent hashes mismatch")
        if not _is_production_checkpoint(record.checkpoint):
            raise ValueError("production provenance checkpoint is not real")
    else:
        if tokenizer_revision != FAKE_TOKENIZER_REVISION or not record.checkpoint.startswith("fake://"):
            raise ValueError("fake provenance must use explicit smoke lineage")
    return ProvenanceEnvelope(
        run_kind=str(run_kind),
        adapter_seed=adapter_seed,
        config_hash=str(config_hash),
        tokenizer_revision=str(tokenizer_revision),
        split_artifact_id=str(split_artifact_id),
        split_parent_hashes=split_parent_hashes,
        checkpoint=record.checkpoint,
        parent_hashes=tuple(record.parent_hashes),
    )


def _validate_requests(requests: Sequence[GenerationRequest]) -> None:
    ids = [request.generation_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("generation request IDs must be unique")


def _validate_outputs(
    requests: Sequence[GenerationRequest], outputs: Sequence[GenerationOutput]
) -> None:
    output_ids = [output.generation_id for output in outputs]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("generation output IDs must be unique")
    request_ids = [request.generation_id for request in requests]
    if output_ids != request_ids:
        raise ValueError("generation output IDs must be complete and in request order")


def _vllm_sampling_parameters(requests: Sequence[GenerationRequest]) -> dict[str, object]:
    """Extract one request's sampling settings for its independent vLLM stream."""
    _validate_requests(requests)
    if len(requests) != 1:
        raise ValueError("vLLM sampling parameters require one independent request")
    ignored = {
        "checkpoint",
        "adapter_seed",
        "fake_answer",
        "completion_index",
        "run_kind",
        "config_hash",
        "tokenizer_revision",
        "split_artifact_id",
        "split_parent_hashes",
    }
    parameters = {
        key: value for key, value in requests[0].decoding.items() if key not in ignored
    }
    return parameters


def _validate_production_construction(
    config: ExperimentConfig,
    checkpoint: str,
    tokenize: Callable[[str], Sequence[int]] | None,
    tokenizer_revision: str | None,
    split_manifest: ArtifactManifest | None,
    adapter_seed: int,
) -> None:
    if not _is_production_checkpoint(checkpoint):
        raise ValueError("production request construction requires a real checkpoint")
    if tokenize is None:
        raise ValueError("production request construction requires a pinned tokenizer encoder")
    if tokenizer_revision != QWEN25_7B_TOKENIZER_REVISION:
        raise ValueError("production request construction requires the pinned tokenizer revision")
    if split_manifest is None or not split_manifest.artifact_id or not split_manifest.parent_hashes:
        raise ValueError("production request construction requires split artifact and parents")
    expected_config_hash = sha256_json(asdict(config))
    if split_manifest.config_hash != expected_config_hash:
        raise ValueError("production request construction requires matching split config hash")
    if adapter_seed not in (config.pilot_seed, *config.confirmatory_seeds):
        raise ValueError("production request construction requires a declared adapter seed")


def _validate_production_request(request: GenerationRequest) -> None:
    if request.decoding.get("run_kind") != PRODUCTION_RUN_KIND:
        raise ValueError("vLLM accepts production requests only")
    if not _is_production_checkpoint(request.decoding.get("checkpoint")):
        raise ValueError("vLLM request checkpoint must be real")
    if request.decoding.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION:
        raise ValueError("vLLM request tokenizer revision is not pinned")
    adapter_seed = request.decoding.get("adapter_seed")
    if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
        raise ValueError("vLLM request lacks integer adapter seed")
    if not isinstance(request.decoding.get("config_hash"), str):
        raise ValueError("vLLM request lacks config hash")
    if not isinstance(request.decoding.get("split_artifact_id"), str):
        raise ValueError("vLLM request lacks split artifact ID")
    parents = request.decoding.get("split_parent_hashes")
    if not isinstance(parents, tuple) or not parents:
        raise ValueError("vLLM request lacks split parents")


def _is_production_checkpoint(checkpoint: object) -> bool:
    return isinstance(checkpoint, str) and bool(checkpoint) and not checkpoint.startswith(
        ("fake://", "unconfigured://")
    )


def _expand_examples(
    cell: EvaluationCell,
    examples: Sequence[DatasetExample],
    requests: Sequence[GenerationRequest],
) -> tuple[DatasetExample, ...]:
    if cell.kind != "sampled":
        if len(examples) != len(requests):
            raise ValueError("examples and requests must have the same length")
        return tuple(examples)
    if len(requests) != len(examples) * 5:
        raise ValueError("sampled cells require five requests per example")
    expanded: list[DatasetExample] = []
    groups = (requests[index : index + 5] for index in range(0, len(requests), 5))
    for example, group in zip(examples, groups):
        indexes = [request.decoding.get("completion_index") for request in group]
        if indexes != list(range(5)):
            raise ValueError("sampled request completion indexes must be ordered 0 through 4")
        expanded.extend((example,) * 5)
    return tuple(expanded)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_tokenize(text: str) -> tuple[int, ...]:
    """Deterministic dry-run token IDs; production callers inject the real tokenizer."""
    return tuple(text.encode("utf-8"))


def _vllm_logprobs(candidate: object) -> tuple[float, ...]:
    values = getattr(candidate, "logprobs", None)
    if values is None:
        return ()
    token_ids = getattr(candidate, "token_ids", ())
    selected: list[float] = []
    for token_id, options in zip(token_ids, values):
        option = options.get(token_id) if options else None
        value = getattr(option, "logprob", None)
        if value is not None:
            selected.append(float(value))
    return tuple(selected)


def _dry_run_examples(limit: int) -> tuple[DatasetExample, ...]:
    rows = (
        ("dry-1", "What is 1 + 1?", "2"),
        ("dry-2", "What is 2 + 2?", "4"),
        ("dry-3", "What is 3 + 3?", "6"),
    )
    if limit < 1:
        raise ValueError("--limit must be positive")
    return tuple(
        DatasetExample("gsm8k", "dry-run", row_id, question, answer, _text_hash(question))
        for row_id, question, answer in rows[:limit]
    )


def _dry_run(arguments: argparse.Namespace) -> int:
    if arguments.backend != "fake":
        raise SystemExit("dry-run supports only the non-model fake backend")
    config = ExperimentConfig.load(arguments.config)
    examples = _dry_run_examples(arguments.limit)
    split_manifest = ArtifactManifest(
        artifact_id=sha256_json({"kind": "dry-run-splits", "examples": [asdict(row) for row in examples]}),
        kind="phase_marker_splits",
        config_hash=sha256_json(asdict(config)),
        parent_hashes=(),
        row_count=len(examples),
        metadata={"dry_run": True},
    )
    backend = FakeGenerationBackend()
    rows: list[dict[str, object]] = []
    for cell in build_behavior_matrix(config, split_manifest):
        requests = build_generation_requests(
            cell,
            examples,
            config,
            checkpoint=f"fake://{cell.training_arm}",
            adapter_seed=config.pilot_seed,
            fake=True,
        )
        records = records_from_outputs(
            cell,
            examples,
            requests,
            backend.generate(requests),
            (split_manifest.artifact_id,),
        )
        for record in records:
            row = serialize_generation_record(
                record, build_provenance_envelope(record, config, split_manifest)
            )
            row["score"] = asdict(score_generation(record))
            rows.append(row)
    write_jsonl_atomic(arguments.output, rows)
    print(f"wrote {len(rows)} independently rescorable generation records")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--config", type=Path, required=True)
    dry_run.add_argument("--backend", choices=("fake",), required=True)
    dry_run.add_argument("--limit", type=int, required=True)
    dry_run.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "dry-run":
        return _dry_run(arguments)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
