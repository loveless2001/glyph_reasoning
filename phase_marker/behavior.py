"""Behavioral evaluation cells, backends, and immutable raw generation records."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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


PRIMARY_ARMS = ("semantic", "glyph", "dot", "random")
PRIMARY_PROMPTS = ("neutral", "glyph", "dot", "headings")
SAMPLED_CONTRASTS = (
    ("semantic", "neutral"),
    ("glyph", "neutral"),
    ("glyph", "glyph"),
    ("glyph", "dot"),
)
PERTURBATIONS = ("delete", "cluster", "displace", "permute", "dot_replace", "unseen_replace")


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

    def __init__(self, model: str, **llm_kwargs: object) -> None:
        self._model = model
        self._llm_kwargs = dict(llm_kwargs)
        self._llm: Any | None = None

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]:
        _validate_requests(requests)
        if not requests:
            return ()
        if self._llm is None:
            from vllm import LLM  # Imported only when the production backend is used.

            self._llm = LLM(model=self._model, **self._llm_kwargs)
        decoding = dict(requests[0].decoding)
        if any(dict(request.decoding) != decoding for request in requests):
            raise ValueError("a vLLM batch requires identical decoding settings")
        decoding.pop("checkpoint", None)
        decoding.pop("fake_answer", None)
        from vllm import SamplingParams

        results = self._llm.generate(
            prompt_token_ids=[list(request.prompt_token_ids) for request in requests],
            sampling_params=SamplingParams(**decoding),
            use_tqdm=False,
        )
        if len(results) != len(requests):
            raise ValueError("vLLM returned a missing output")
        outputs: list[GenerationOutput] = []
        for request, result in zip(requests, results):
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
    if len(examples) != len(requests):
        raise ValueError("examples and requests must have the same length")
    _validate_requests(requests)
    _validate_outputs(requests, outputs)
    rows: list[GenerationRecord] = []
    for example, request, output in zip(examples, requests, outputs):
        seed = request.decoding.get("seed")
        checkpoint = request.decoding.get("checkpoint")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("generation request decoding must include integer seed")
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
                seed=seed,
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


def serialize_generation_record(record: GenerationRecord) -> dict[str, object]:
    """Add explicit token counts without changing the immutable raw record schema."""
    row = asdict(record)
    row["prompt_token_count"] = len(record.prompt_token_ids)
    row["completion_token_count"] = len(record.completion_token_ids)
    return row


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


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _dry_run_requests(
    cell: EvaluationCell, examples: Sequence[DatasetExample], config: ExperimentConfig
) -> tuple[GenerationRequest, ...]:
    markers = MarkerSet(*config.phase_markers)
    requests: list[GenerationRequest] = []
    for example in examples:
        prompt = (
            render_perturbation(example.question, cell.perturbation, markers)
            if cell.perturbation
            else render_prompt(example.question, cell.prompt_condition, markers)
        )
        decoding: dict[str, object] = {
            "seed": config.pilot_seed,
            "checkpoint": f"fake://{cell.training_arm}",
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "fake_answer": example.answer,
        }
        if cell.decoding_name == "sampled":
            decoding.update({"temperature": 0.7, "top_p": 0.95, "n": 5})
        requests.append(
            GenerationRequest(
                generation_id=(
                    f"{cell.kind}:{cell.training_arm}:{cell.prompt_condition}:"
                    f"{cell.perturbation or 'base'}:{cell.decoding_name}:{example.example_id}"
                ),
                prompt=prompt,
                prompt_token_ids=tuple(prompt.encode("utf-8")),
                max_new_tokens=64,
                decoding=decoding,
            )
        )
    return tuple(requests)


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
        requests = _dry_run_requests(cell, examples, config)
        records = records_from_outputs(
            cell,
            examples,
            requests,
            backend.generate(requests),
            (split_manifest.artifact_id,),
        )
        for record in records:
            row = serialize_generation_record(record)
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
