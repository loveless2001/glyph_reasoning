"""Behavioral evaluation cells, backends, and immutable raw generation records."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Protocol

from phase_marker.config import ExperimentConfig, REQUIRED_MODEL_ID
from phase_marker.io import canonical_json, read_jsonl, sha256_json, write_jsonl_atomic
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
        self,
        model: str,
        *,
        model_revision: str = QWEN25_7B_TOKENIZER_REVISION,
        adapter_path: str | None = None,
        adapter_seed: int | None = None,
        **llm_kwargs: object,
    ) -> None:
        if model != REQUIRED_MODEL_ID or model_revision != QWEN25_7B_TOKENIZER_REVISION:
            raise ValueError("vLLM backend requires the frozen base model and revision")
        if not _is_production_checkpoint(adapter_path):
            raise ValueError("vLLM backend requires a real LoRA adapter path")
        if not isinstance(adapter_seed, int) or isinstance(adapter_seed, bool):
            raise ValueError("vLLM backend requires an explicit integer adapter seed")
        self._model = model
        self._model_revision = model_revision
        self._adapter_path = str(adapter_path)
        self._adapter_seed = adapter_seed
        self._llm_kwargs = dict(llm_kwargs)
        self._llm: Any | None = None

    def _ensure_llm(self) -> Any:
        if self._llm is None:
            from vllm import LLM  # Imported only when the production backend is used.

            self._llm = LLM(
                model=self._model,
                revision=self._model_revision,
                tokenizer_revision=self._model_revision,
                enable_lora=True,
                **self._llm_kwargs,
            )
        return self._llm

    def _lora_request(self) -> Any:
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            f"{Path(self._adapter_path).name}-seed-{self._adapter_seed}",
            self._adapter_seed,
            self._adapter_path,
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]:
        _validate_requests(requests)
        if not requests:
            return ()
        for request in requests:
            _validate_production_request(request)
            if request.decoding["checkpoint"] != self._adapter_path:
                raise ValueError("vLLM request checkpoint must exactly match backend adapter")
            if request.decoding["adapter_seed"] != self._adapter_seed:
                raise ValueError("vLLM request must match backend adapter seed")
        from vllm import SamplingParams
        llm = self._ensure_llm()
        lora_request = self._lora_request()

        outputs: list[GenerationOutput] = []
        for request in requests:
            results = llm.generate(
                prompt_token_ids=[list(request.prompt_token_ids)],
                sampling_params=SamplingParams(**_vllm_sampling_parameters((request,))),
                use_tqdm=False,
                lora_request=lora_request,
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

    def mean_gold_answer_logprob(
        self,
        requests: Sequence[GenerationRequest],
        gold_answers: Sequence[str],
        *,
        tokenize: Callable[[str], Sequence[int]],
        final_delimiter: str,
    ) -> float:
        """Teacher-force each frozen gold continuation and average its log probability."""
        values = self.gold_answer_logprobs(
            requests, gold_answers, tokenize=tokenize, final_delimiter=final_delimiter
        )
        return sum(values) / len(values)

    def gold_answer_logprobs(
        self,
        requests: Sequence[GenerationRequest],
        gold_answers: Sequence[str],
        *,
        tokenize: Callable[[str], Sequence[int]],
        final_delimiter: str,
    ) -> tuple[float, ...]:
        return tuple(
            sum(logprobs) for _, logprobs in self.gold_answer_token_evidence(
                requests, gold_answers, tokenize=tokenize,
                final_delimiter=final_delimiter,
            )
        )

    def gold_answer_token_evidence(
        self,
        requests: Sequence[GenerationRequest],
        gold_answers: Sequence[str],
        *,
        tokenize: Callable[[str], Sequence[int]],
        final_delimiter: str,
    ) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
        """Return exact ordered teacher-forced token IDs and their logprobs."""
        if len(requests) != len(gold_answers) or not requests:
            raise ValueError("gold-answer scoring requires one answer per request")
        from vllm import SamplingParams

        llm = self._ensure_llm()
        lora_request = self._lora_request()
        result_rows = []
        for request, answer in zip(requests, gold_answers, strict=True):
            suffix_ids = tuple(tokenize(f"\n{final_delimiter} {answer}"))
            prompt_ids = (*request.prompt_token_ids, *suffix_ids)
            results = llm.generate(
                prompt_token_ids=[list(prompt_ids)],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1),
                use_tqdm=False, lora_request=lora_request,
            )
            prompt_logprobs = getattr(results[0], "prompt_logprobs", None) if len(results) == 1 else None
            if not suffix_ids or not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(prompt_ids):
                raise ValueError("vLLM omitted teacher-forced gold-answer logprobs")
            values: list[float] = []
            for position, token_id in enumerate(suffix_ids, start=len(request.prompt_token_ids)):
                entry = prompt_logprobs[position]
                value = entry.get(token_id) if isinstance(entry, Mapping) else None
                logprob = getattr(value, "logprob", value)
                if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
                    raise ValueError("vLLM returned a malformed gold-answer logprob")
                values.append(float(logprob))
            result_rows.append((suffix_ids, tuple(values)))
        return tuple(result_rows)


def select_validation_checkpoint(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Apply the frozen validation-only checkpoint selection criterion."""
    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    normalized: list[Mapping[str, object]] = []
    for candidate in candidates:
        accuracy = candidate.get("strict_accuracy")
        logprob = candidate.get("mean_gold_answer_logprob")
        step = candidate.get("step")
        if (
            not isinstance(accuracy, (int, float))
            or isinstance(accuracy, bool)
            or not 0 <= float(accuracy) <= 1
            or not isinstance(logprob, (int, float))
            or isinstance(logprob, bool)
            or not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or not isinstance(candidate.get("path"), str)
        ):
            raise ValueError("checkpoint candidate metrics are malformed")
        normalized.append(candidate)
    return min(
        normalized,
        key=lambda candidate: (
            -float(candidate["strict_accuracy"]),
            -float(candidate["mean_gold_answer_logprob"]),
            int(candidate["step"]),
        ),
    )


def _selection_generation(
    example: DatasetExample,
    raw_completion: str,
    arm: str,
    seed: int,
    checkpoint: str,
    generation_id: str | None = None,
    prompt_condition: str = "validation",
) -> GenerationRecord:
    return GenerationRecord(
        generation_id=generation_id or f"selection:{checkpoint}:{example.example_id}",
        source=example.source, question_hash=example.question_hash,
        gold_answer=example.answer, training_arm=arm, seed=seed,
        checkpoint=checkpoint, prompt_condition=prompt_condition,
        prompt_hash=sha256_json(example.question), raw_prompt=example.question,
        raw_completion=raw_completion, prompt_token_ids=(), completion_token_ids=(),
        decoding={}, parent_hashes=(),
    )


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
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    run.add_argument("--seeds", type=int, nargs="+", required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument("--examples", type=Path, required=True)
    run.add_argument("--checkpoint-manifests", type=Path, nargs="+", required=True)
    run.add_argument("--backend", choices=("vllm", "tiny-fixture"), required=True)
    run.add_argument("--allow-test-backend", action="store_true")
    run.add_argument("--output-root", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--kind", choices=("pilot", "confirmatory"), required=True)
    select.add_argument("--seed", type=int, required=True)
    select.add_argument("--arm", required=True)
    select.add_argument("--split-manifest", type=Path, required=True)
    select.add_argument("--validation-examples", type=Path, required=True)
    select.add_argument("--training-manifest", type=Path, required=True)
    select.add_argument("--backend", choices=("vllm", "tiny-fixture"), required=True)
    select.add_argument("--allow-test-backend", action="store_true")
    select.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "dry-run":
        return _dry_run(arguments)
    if arguments.command == "run":
        return _run_behavior(arguments)
    if arguments.command == "select":
        return _run_selection(arguments)
    raise AssertionError("unreachable")


_CHECKPOINT_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "config_hash",
        "run_kind",
        "arm",
        "seed",
        "selected_on",
        "evidence_scope",
        "backend",
        "model_id",
        "model_revision",
        "criterion",
        "split_artifact_id",
        "split_manifest_hash",
        "validation_examples_file",
        "validation_examples_hash",
        "training_manifest_file",
        "training_manifest_hash",
        "materialization_artifact_id",
        "candidates",
        "evidence_file",
        "evidence_hash",
        "selected_path",
        "selected_checkpoint_hash",
        "selected_step",
        "parent_hashes",
        "completed",
        "artifact_id",
    }
)


def _run_selection(arguments: argparse.Namespace) -> int:
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint selection: {arguments.output}")
    if arguments.backend == "tiny-fixture" and not arguments.allow_test_backend:
        raise SystemExit("--allow-test-backend is required for tiny-fixture")
    config = ExperimentConfig.load(arguments.config)
    if arguments.backend == "vllm":
        _validate_canonical_split_examples(
            arguments.split_manifest, arguments.validation_examples, config,
            expected_split="validation",
        )
    expected_seeds = (config.pilot_seed,) if arguments.kind == "pilot" else tuple(config.confirmatory_seeds)
    if arguments.seed not in expected_seeds:
        raise ValueError(f"checkpoint selection seed must be one of {expected_seeds}")
    if arguments.arm not in config.arms:
        raise ValueError("checkpoint selection arm is not in the frozen protocol")
    split = _load_json_object(arguments.split_manifest, "split manifest")
    config_hash = sha256_json(asdict(config))
    if split.get("config_hash") != config_hash:
        raise ValueError("checkpoint selection split config hash mismatch")
    split_id = _required_hash(split, "artifact_id", "split manifest")
    examples = tuple(_example_from_row(row) for row in read_jsonl(arguments.validation_examples))
    if not examples or any(example.split != "validation" for example in examples):
        raise ValueError("checkpoint selection uses frozen validation examples only")
    training = _load_json_object(arguments.training_manifest, "training manifest")
    required_training = {
        "kind", "arm", "seed", "model_id", "model_revision", "tokenizer_revision",
        "config_hash", "dataset_path", "dataset_hash", "data_artifact_id", "parent_hashes",
        "data_parent_hashes", "arguments", "environment", "checkpoints", "saved_artifacts",
        "output_hash",
    }
    if set(training) != required_training:
        raise ValueError("training manifest fields do not match the producer schema")
    if (
        training.get("kind") != "phase_marker_training_run"
        or training.get("arm") != arguments.arm
        or training.get("seed") != arguments.seed
        or training.get("model_id") != config.model_id
        or training.get("model_revision") != QWEN25_7B_TOKENIZER_REVISION
        or training.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION
        or training.get("config_hash") != config_hash
        or training.get("data_parent_hashes") != [split_id]
    ):
        raise ValueError("training manifest identity or split lineage mismatch")
    data_path = Path(str(training.get("dataset_path")))
    if training.get("dataset_hash") != sha256_json(data_path.read_bytes().hex()):
        raise ValueError("training manifest dataset hash mismatch")
    materialization_path = data_path.with_suffix(".manifest.json")
    materialization = _load_json_object(materialization_path, "materialization manifest")
    materialization_id = _required_hash(materialization, "artifact_id", "materialization manifest")
    if (
        training.get("data_artifact_id") != materialization_id
        or training.get("parent_hashes") != [materialization_id]
    ):
        raise ValueError("training materialization lineage mismatch")
    declared = training.get("checkpoints")
    if not isinstance(declared, list) or not declared:
        raise ValueError("training manifest declares no checkpoints")
    candidate_inputs: list[tuple[Path, str, int]] = []
    for item in declared:
        if not isinstance(item, Mapping) or set(item) != {"path", "hash"}:
            raise ValueError("training checkpoint lineage is malformed")
        relative = item["path"]
        if not isinstance(relative, str) or not relative.startswith("checkpoint-"):
            raise ValueError("training checkpoint path is malformed")
        try:
            step = int(relative.removeprefix("checkpoint-"))
        except ValueError as error:
            raise ValueError("training checkpoint step is malformed") from error
        checkpoint = arguments.training_manifest.parent / relative
        if not checkpoint.is_dir() or item["hash"] != _behavior_directory_hash(checkpoint):
            raise ValueError("training checkpoint directory hash mismatch")
        _validate_adapter_compatibility(checkpoint, config)
        candidate_inputs.append((checkpoint, str(item["hash"]), step))

    tokenizer: object | None = None
    if arguments.backend == "vllm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=QWEN25_7B_TOKENIZER_REVISION, local_files_only=True
        )
    metrics: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    prompt_condition = arguments.arm if arguments.arm in {"glyph", "dot"} else "neutral"
    cell = EvaluationCell("primary", arguments.arm, prompt_condition, None, "greedy")
    for checkpoint, checkpoint_hash, step in candidate_inputs:
        if arguments.backend == "tiny-fixture":
            accuracy, mean_logprob = 0.0, 0.0
            raw_completions = tuple("Final answer: 0" for _ in examples)
            scored = [
                score_generation(_selection_generation(example, raw, arguments.arm, arguments.seed, str(checkpoint)))
                for example, raw in zip(examples, raw_completions, strict=True)
            ]
            token_rows = tuple(
                ((0,), (f"\n{config.final_delimiter} {example.answer}",), (0.0,))
                for example in examples
            )
        else:
            assert tokenizer is not None
            requests = build_generation_requests(
                cell, examples, config, checkpoint=str(checkpoint),
                tokenize=lambda value: tokenizer.encode(value, add_special_tokens=False),
                tokenizer_revision=QWEN25_7B_TOKENIZER_REVISION,
                split_manifest=ArtifactManifest(
                    split_id, "phase_marker_splits", config_hash, _split_parent_hashes(split),
                    len(examples), {},
                ),
                adapter_seed=arguments.seed,
            )
            backend = VLLMGenerationBackend(
                config.model_id, model_revision=QWEN25_7B_TOKENIZER_REVISION,
                adapter_path=str(checkpoint), adapter_seed=arguments.seed,
            )
            outputs = backend.generate(requests)
            records = records_from_outputs(
                cell, examples, requests, outputs,
                (split_id, materialization_id, _file_hash(arguments.training_manifest)),
            )
            scored = [score_generation(record) for record in records]
            accuracy = sum(row.correct for row in scored) / len(scored)
            raw_completions = tuple(output.text for output in outputs)
            teacher_forced = backend.gold_answer_token_evidence(
                requests,
                tuple(example.answer for example in examples),
                tokenize=lambda value: tokenizer.encode(
                    value, add_special_tokens=False
                ),
                final_delimiter=config.final_delimiter,
            )
            token_rows_list = []
            for (token_ids, token_logprobs), example in zip(teacher_forced, examples, strict=True):
                pieces: list[str] = []
                previous = ""
                for width in range(1, len(token_ids) + 1):
                    decoded = tokenizer.decode(
                        list(token_ids[:width]), skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    pieces.append(decoded[len(previous):])
                    previous = decoded
                continuation = f"\n{config.final_delimiter} {example.answer}"
                if previous != continuation:
                    raise ValueError("tokenizer does not exactly reproduce the gold continuation")
                token_rows_list.append((token_ids, tuple(pieces), token_logprobs))
            token_rows = tuple(token_rows_list)
            mean_logprob = sum(sum(row[2]) for row in token_rows) / len(token_rows)
        accuracy = sum(row.correct for row in scored) / len(scored)
        for example, raw_completion, score, token_row in zip(
            examples, raw_completions, scored, token_rows, strict=True
        ):
            token_ids, token_pieces, token_logprobs = token_row
            continuation = f"\n{config.final_delimiter} {example.answer}"
            evidence_rows.append({
                "dataset": example.source, "example_id": example.example_id,
                "question_hash": example.question_hash, "gold_answer": example.answer,
                "checkpoint_id": checkpoint_hash, "checkpoint_path": str(checkpoint),
                "raw_greedy_completion": raw_completion,
                "scorer_inputs": {"source": example.source, "gold_answer": example.answer},
                "scorer_outputs": asdict(score),
                "gold_continuation": continuation,
                "gold_token_ids": list(token_ids), "gold_token_pieces": list(token_pieces),
                "gold_token_logprobs": list(token_logprobs),
                "gold_answer_logprob_contribution": sum(token_logprobs),
                "tokenizer_revision": QWEN25_7B_TOKENIZER_REVISION,
                "tokenizer_snapshot_hash": QWEN25_7B_TOKENIZER_REVISION,
            })
        metrics.append({
            "path": str(checkpoint), "checkpoint_hash": checkpoint_hash, "step": step,
            "strict_accuracy": accuracy, "mean_gold_answer_logprob": mean_logprob,
            "row_count": len(examples),
        })
    selected = select_validation_checkpoint(metrics)
    manifest: dict[str, object] = {
        "schema_version": 1, "kind": "phase_marker_checkpoint_selection",
        "evidence_scope": "plumbing_only" if arguments.backend == "tiny-fixture" else "experiment",
        "backend": arguments.backend, "model_id": config.model_id,
        "model_revision": QWEN25_7B_TOKENIZER_REVISION,
        "config_hash": config_hash, "run_kind": arguments.kind,
        "arm": arguments.arm, "seed": arguments.seed, "selected_on": "validation",
        "criterion": {
            "primary": "maximize_strict_validation_exact_answer_accuracy",
            "tie_break_1": "higher_mean_gold_answer_logprob",
            "tie_break_2": "earliest_checkpoint_step",
        },
        "split_artifact_id": split_id, "split_manifest_hash": _file_hash(arguments.split_manifest),
        "validation_examples_file": str(arguments.validation_examples),
        "validation_examples_hash": _file_hash(arguments.validation_examples),
        "training_manifest_file": str(arguments.training_manifest),
        "training_manifest_hash": _file_hash(arguments.training_manifest),
        "materialization_artifact_id": materialization_id, "candidates": metrics,
        "evidence_file": str(arguments.output / "evidence.jsonl"),
        "evidence_hash": "",
        "selected_path": selected["path"],
        "selected_checkpoint_hash": selected["checkpoint_hash"],
        "selected_step": selected["step"],
        "parent_hashes": [split_id, materialization_id, _file_hash(arguments.training_manifest)],
        "completed": True,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=arguments.output.parent, prefix=f".{arguments.output.name}-staging-"
    ) as temporary:
        staging = Path(temporary) / "publish"
        staging.mkdir()
        evidence_path = staging / "evidence.jsonl"
        write_jsonl_atomic(evidence_path, evidence_rows)
        manifest["evidence_hash"] = _file_hash(evidence_path)
        manifest["artifact_id"] = sha256_json(manifest)
        (staging / "manifest.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        staging.replace(arguments.output)
    print(canonical_json(manifest))
    return 0


def _run_behavior(arguments: argparse.Namespace) -> int:
    if arguments.backend == "tiny-fixture" and not arguments.allow_test_backend:
        raise SystemExit("--allow-test-backend is required for tiny-fixture")
    config = ExperimentConfig.load(arguments.config)
    seeds = tuple(arguments.seeds)
    _require_frozen_run(config, arguments.kind, seeds)
    split_payload = _load_json_object(arguments.split_manifest, "split manifest")
    if arguments.backend == "vllm":
        _validate_canonical_split_examples(
            arguments.split_manifest, arguments.examples, config,
            expected_split="test",
        )
    config_hash = sha256_json(asdict(config))
    if split_payload.get("config_hash") != config_hash:
        raise ValueError("split manifest config hash mismatch")
    split_id = _required_hash(split_payload, "artifact_id", "split manifest")
    examples = tuple(_example_from_row(row) for row in read_jsonl(arguments.examples))
    if not examples:
        raise ValueError("behavior examples must be nonempty")
    selections = _load_checkpoint_selections(
        arguments.checkpoint_manifests,
        config,
        arguments.kind,
        seeds,
        allow_test=arguments.backend == "tiny-fixture",
    )
    if arguments.backend == "vllm":
        _validate_production_behavior_inputs(
            arguments.split_manifest, split_id, config, selections
        )
        for identity, selection in selections.items():
            checkpoint_root = Path(str(selection["selected_path"]))
            if not checkpoint_root.is_dir():
                raise FileNotFoundError(f"selected production checkpoint is missing for {identity}")
    if arguments.output_root.exists():
        raise FileExistsError(f"refusing to overwrite behavior output: {arguments.output_root}")

    split_parents = _split_parent_hashes(split_payload)
    split_manifest = ArtifactManifest(
        artifact_id=split_id,
        kind="phase_marker_splits",
        config_hash=config_hash,
        parent_hashes=split_parents,
        row_count=len(examples),
        metadata={"source_counts": split_payload.get("source_counts", {})},
    )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for cell in build_behavior_matrix(config, split_manifest):
            selection = selections[(seed, cell.training_arm)]
            checkpoint = str(selection["selected_path"])
            if arguments.backend == "tiny-fixture":
                requests = build_generation_requests(
                    cell,
                    examples,
                    config,
                    checkpoint=f"fake://{cell.training_arm}",
                    adapter_seed=seed,
                    fake=True,
                )
                backend: GenerationBackend = FakeGenerationBackend()
                record_parents = (split_id, str(selection["artifact_id"]))
            else:
                if not Path(checkpoint).is_dir():
                    raise FileNotFoundError(
                        f"selected production checkpoint is missing: {checkpoint}"
                    )
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_id,
                    revision=QWEN25_7B_TOKENIZER_REVISION,
                    local_files_only=True,
                )
                requests = build_generation_requests(
                    cell,
                    examples,
                    config,
                    checkpoint=checkpoint,
                    tokenize=lambda value: tokenizer.encode(value, add_special_tokens=False),
                    tokenizer_revision=QWEN25_7B_TOKENIZER_REVISION,
                    split_manifest=split_manifest,
                    adapter_seed=seed,
                )
                backend = VLLMGenerationBackend(
                    config.model_id,
                    model_revision=QWEN25_7B_TOKENIZER_REVISION,
                    adapter_path=checkpoint,
                    adapter_seed=seed,
                )
                record_parents = (split_id, *split_parents, str(selection["artifact_id"]))
            records = records_from_outputs(
                cell,
                examples,
                requests,
                backend.generate(requests),
                record_parents,
            )
            for record in records:
                envelope = build_provenance_envelope(record, config, split_manifest)
                row = serialize_generation_record(record, envelope)
                row["score"] = asdict(score_generation(record))
                row["checkpoint_selection_artifact_id"] = selection["artifact_id"]
                rows.append(row)

    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=arguments.output_root.parent, prefix=f".{arguments.output_root.name}-"
    ) as temporary:
        staging = Path(temporary)
        records_path = staging / "records.jsonl"
        write_jsonl_atomic(records_path, rows)
        selection_values = tuple(selections[key] for key in sorted(selections))
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "phase_marker_behavior_generations",
            "evidence_scope": (
                "plumbing_only"
                if arguments.backend == "tiny-fixture"
                else "experiment_candidate"
            ),
            "backend": arguments.backend,
            "config_hash": config_hash,
            "run_kind": arguments.kind,
            "seeds": list(seeds),
            "split_artifact_id": split_id,
            "split_manifest_hash": _file_hash(arguments.split_manifest),
            "materialization_artifact_ids": {
                f"{item['seed']}:{item['arm']}": item["materialization_artifact_id"]
                for item in selection_values
            },
            "checkpoint_artifact_ids": {
                f"{item['seed']}:{item['arm']}": item["artifact_id"]
                for item in selection_values
            },
            "checkpoint_manifest_hashes": {
                f"{item['seed']}:{item['arm']}": item["manifest_hash"]
                for item in selection_values
            },
            "checkpoint_manifests": {
                f"{item['seed']}:{item['arm']}": item["manifest_path"]
                for item in selection_values
            },
            "examples_file": str(arguments.examples),
            "examples_hash": _file_hash(arguments.examples),
            "records_file": records_path.name,
            "records_hash": _file_hash(records_path),
            "row_count": len(rows),
            "record_hashes": [sha256_json(row) for row in rows],
            "exclusions": [],
            "parent_hashes": [
                split_id,
                *(str(item["artifact_id"]) for item in selection_values),
            ],
            "completed": True,
        }
        manifest["artifact_id"] = sha256_json(manifest)
        (staging / "manifest.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        staging.replace(arguments.output_root)
    print(canonical_json(manifest))
    return 0


def _require_frozen_run(
    config: ExperimentConfig, kind: str, seeds: tuple[int, ...]
) -> None:
    if tuple(config.arms) != ("semantic", "glyph", "dot", "random", "direct", "filler"):
        raise ValueError("behavior run requires the frozen six-arm protocol")
    expected = (42,) if kind == "pilot" else (101, 202, 303)
    if seeds != expected:
        raise ValueError(f"{kind} behavior run requires exact seeds {expected}")


def _load_checkpoint_selections(
    paths: Sequence[Path],
    config: ExperimentConfig,
    kind: str,
    seeds: tuple[int, ...],
    *,
    allow_test: bool,
) -> dict[tuple[int, str], dict[str, object]]:
    expected = {(seed, arm) for seed in seeds for arm in config.arms}
    result: dict[tuple[int, str], dict[str, object]] = {}
    config_hash = sha256_json(asdict(config))
    for path in paths:
        payload = dict(_load_json_object(path, "checkpoint selection manifest"))
        if set(payload) != _CHECKPOINT_SELECTION_FIELDS:
            raise ValueError(f"checkpoint selection schema mismatch: {path}")
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "phase_marker_checkpoint_selection"
            or payload["config_hash"] != config_hash
            or payload["run_kind"] != kind
            or payload["selected_on"] != "validation"
            or payload["completed"] is not True
            or payload["model_id"] != config.model_id
            or payload["model_revision"] != QWEN25_7B_TOKENIZER_REVISION
        ):
            raise ValueError(f"checkpoint selection provenance mismatch: {path}")
        expected_scope = "plumbing_only" if allow_test else "experiment"
        expected_backend = "tiny-fixture" if allow_test else "vllm"
        if payload["evidence_scope"] != expected_scope or payload["backend"] != expected_backend:
            raise ValueError(f"checkpoint selection evidence scope mismatch: {path}")
        artifact_id = payload.pop("artifact_id")
        if artifact_id != sha256_json(payload):
            raise ValueError(f"checkpoint selection artifact hash mismatch: {path}")
        payload["artifact_id"] = artifact_id
        if payload["criterion"] != {
            "primary": "maximize_strict_validation_exact_answer_accuracy",
            "tie_break_1": "higher_mean_gold_answer_logprob",
            "tie_break_2": "earliest_checkpoint_step",
        }:
            raise ValueError(f"checkpoint selection criterion mismatch: {path}")
        for file_field, hash_field in (
            ("validation_examples_file", "validation_examples_hash"),
            ("training_manifest_file", "training_manifest_hash"),
            ("evidence_file", "evidence_hash"),
        ):
            bound_path = Path(str(payload[file_field]))
            if not bound_path.is_file() or payload[hash_field] != _file_hash(bound_path):
                raise ValueError(f"checkpoint selection bound file hash mismatch: {path}")
        selected_path = Path(str(payload["selected_path"]))
        if not allow_test and (
            not selected_path.is_dir()
            or payload["selected_checkpoint_hash"] != _behavior_directory_hash(selected_path)
        ):
            raise ValueError(f"checkpoint selection selected adapter hash mismatch: {path}")
        if not allow_test:
            _validate_adapter_compatibility(selected_path, config)
        candidates = payload["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"checkpoint selection candidates are missing: {path}")
        selected_candidates = [
            item for item in candidates
            if isinstance(item, Mapping)
            and item.get("path") == payload["selected_path"]
            and item.get("checkpoint_hash") == payload["selected_checkpoint_hash"]
            and item.get("step") == payload["selected_step"]
        ]
        if len(selected_candidates) != 1 or select_validation_checkpoint(candidates) != selected_candidates[0]:
            raise ValueError(f"checkpoint selection winner does not match frozen criterion: {path}")
        evidence = tuple(read_jsonl(Path(str(payload["evidence_file"]))))
        examples = tuple(read_jsonl(Path(str(payload["validation_examples_file"]))))
        expected_examples = {
            (row.get("example_id"), row.get("question_hash")): _example_from_row(row)
            for row in examples
        }
        evidence_keys = [
            (row.get("checkpoint_id"), row.get("example_id"), row.get("question_hash"))
            for row in evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError(f"checkpoint selection evidence contains duplicate rows: {path}")
        for candidate in candidates:
            assert isinstance(candidate, Mapping)
            candidate_rows = [
                row for row in evidence
                if row.get("checkpoint_id") == candidate.get("checkpoint_hash")
                and row.get("checkpoint_path") == candidate.get("path")
            ]
            if {
                (row.get("example_id"), row.get("question_hash")) for row in candidate_rows
            } != set(expected_examples) or len(candidate_rows) != len(expected_examples):
                raise ValueError(f"checkpoint selection per-example evidence mismatch: {path}")
            strict: list[bool] = []
            contributions: list[float] = []
            for row in candidate_rows:
                if set(row) != {
                    "dataset", "example_id", "question_hash", "gold_answer",
                    "checkpoint_id", "checkpoint_path", "raw_greedy_completion",
                    "scorer_inputs", "scorer_outputs", "gold_continuation",
                    "gold_token_ids", "gold_token_pieces", "gold_token_logprobs",
                    "gold_answer_logprob_contribution", "tokenizer_revision",
                    "tokenizer_snapshot_hash",
                }:
                    raise ValueError(f"checkpoint selection evidence schema mismatch: {path}")
                example = expected_examples[(row.get("example_id"), row.get("question_hash"))]
                if (
                    row.get("dataset") != example.source
                    or row.get("gold_answer") != example.answer
                    or row.get("tokenizer_revision") != QWEN25_7B_TOKENIZER_REVISION
                    or row.get("tokenizer_snapshot_hash") != QWEN25_7B_TOKENIZER_REVISION
                    or row.get("scorer_inputs") != {"source": example.source, "gold_answer": example.answer}
                    or not isinstance(row.get("raw_greedy_completion"), str)
                ):
                    raise ValueError(f"checkpoint selection canonical evidence mismatch: {path}")
                replayed = score_generation(_selection_generation(
                    example, str(row["raw_greedy_completion"]), str(payload["arm"]),
                    int(payload["seed"]), str(candidate["path"]),
                    generation_id=(
                        str(row["scorer_outputs"].get("generation_id"))
                        if isinstance(row.get("scorer_outputs"), Mapping) else None
                    ),
                    prompt_condition=(
                        str(row["scorer_outputs"].get("prompt_condition"))
                        if isinstance(row.get("scorer_outputs"), Mapping) else "validation"
                    ),
                ))
                if row.get("scorer_outputs") != asdict(replayed):
                    raise ValueError(f"checkpoint selection scorer replay mismatch: {path}")
                token_ids = row.get("gold_token_ids")
                pieces = row.get("gold_token_pieces")
                logprobs = row.get("gold_token_logprobs")
                if (
                    not isinstance(token_ids, list) or not token_ids
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in token_ids)
                    or not isinstance(pieces, list) or any(not isinstance(value, str) for value in pieces)
                    or not isinstance(logprobs, list)
                    or len(token_ids) != len(pieces) or len(token_ids) != len(logprobs)
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not math.isfinite(value) or value > 0 for value in logprobs)
                    or "".join(pieces) != row.get("gold_continuation")
                    or row.get("gold_continuation") != f"\n{config.final_delimiter} {example.answer}"
                    or row.get("gold_answer_logprob_contribution") != sum(logprobs)
                ):
                    raise ValueError(f"checkpoint selection token evidence mismatch: {path}")
                strict.append(replayed.correct)
                contributions.append(float(sum(logprobs)))
            if (
                candidate.get("row_count") != len(candidate_rows)
                or candidate.get("strict_accuracy") != sum(strict) / len(strict)
                or candidate.get("mean_gold_answer_logprob") != sum(contributions) / len(contributions)
            ):
                raise ValueError(f"checkpoint selection aggregate evidence mismatch: {path}")
        parents = payload["parent_hashes"]
        if parents != [
            payload["split_artifact_id"], payload["materialization_artifact_id"],
            payload["training_manifest_hash"],
        ]:
            raise ValueError(f"checkpoint selection parent lineage mismatch: {path}")
        payload["manifest_hash"] = _file_hash(path)
        payload["manifest_path"] = str(path)
        identity = (payload["seed"], payload["arm"])
        if identity in result:
            raise ValueError(f"duplicate checkpoint selection: {identity}")
        result[identity] = payload
    if set(result) != expected:
        raise ValueError(
            f"checkpoint selections must cover exact seed/arm matrix: missing={sorted(expected-set(result))}"
        )
    return result


def _load_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {label}: {path}") from None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_production_behavior_inputs(
    split_manifest: Path,
    split_id: str,
    config: ExperimentConfig,
    selections: Mapping[tuple[int, str], Mapping[str, object]],
) -> None:
    from phase_marker.pipeline import (
        GateFailure,
        _validate_materializations,
        _validate_split_manifest,
    )

    artifact_root = split_manifest.parent.parent
    try:
        split = _validate_split_manifest(artifact_root, config)
        materializations = _validate_materializations(artifact_root, config, split.artifact_id)
    except GateFailure as error:
        raise ValueError(str(error)) from error
    if split.artifact_id != split_id or split.path != split_manifest:
        raise ValueError("behavior split manifest does not match the canonical producer artifact")
    expected_materializations = dict(zip(config.arms, materializations, strict=True))
    required_training = {
        "kind", "arm", "seed", "model_id", "model_revision", "tokenizer_revision",
        "config_hash", "dataset_path", "dataset_hash", "data_artifact_id", "parent_hashes",
        "data_parent_hashes", "arguments", "environment", "checkpoints", "saved_artifacts",
        "output_hash",
    }
    for (seed, arm), selection in selections.items():
        path = Path(str(selection["training_manifest_file"]))
        training = _load_json_object(path, "training manifest")
        if set(training) != required_training:
            raise ValueError(f"training manifest fields mismatch: {path}")
        data_path = Path(str(training["dataset_path"]))
        if (
            training["kind"] != "phase_marker_training_run"
            or training["seed"] != seed
            or training["arm"] != arm
            or training["model_id"] != config.model_id
            or training["model_revision"] != QWEN25_7B_TOKENIZER_REVISION
            or training["tokenizer_revision"] != QWEN25_7B_TOKENIZER_REVISION
            or training["config_hash"] != sha256_json(asdict(config))
            or training["data_parent_hashes"] != [split_id]
            or training["data_artifact_id"] != expected_materializations[arm]
            or training["parent_hashes"] != [expected_materializations[arm]]
            or training["dataset_hash"] != sha256_json(data_path.read_bytes().hex())
            or selection["materialization_artifact_id"] != expected_materializations[arm]
        ):
            raise ValueError(f"training/materialization lineage mismatch: {path}")
        declared = training["checkpoints"]
        selected_path = Path(str(selection["selected_path"]))
        relative = str(selected_path.relative_to(path.parent))
        if not isinstance(declared, list) or {
            "path": relative, "hash": selection["selected_checkpoint_hash"]
        } not in declared:
            raise ValueError(f"selected checkpoint is not declared by training manifest: {path}")
        declared_candidates = {
            (str(path.parent / str(item["path"])), int(str(item["path"]).removeprefix("checkpoint-")), item["hash"])
            for item in declared
            if isinstance(item, Mapping) and set(item) == {"path", "hash"}
        }
        recorded_candidates = {
            (item.get("path"), item.get("step"), item.get("checkpoint_hash"))
            for item in selection["candidates"]  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        }
        if recorded_candidates != declared_candidates:
            raise ValueError(f"checkpoint selection candidates do not exactly match training manifest: {path}")


def _validate_canonical_split_examples(
    split_manifest: Path,
    examples_path: Path,
    config: ExperimentConfig,
    *,
    expected_split: str,
) -> None:
    from phase_marker.pipeline import GateFailure, _validate_split_manifest

    expected_path = split_manifest.parent / f"{expected_split}.jsonl"
    if split_manifest.name != "manifest.json" or examples_path.resolve() != expected_path.resolve():
        raise ValueError(f"production examples must be canonical sibling {expected_split}.jsonl")
    try:
        validated = _validate_split_manifest(split_manifest.parent.parent, config)
    except GateFailure as error:
        raise ValueError(str(error)) from error
    if validated.path.resolve() != split_manifest.resolve():
        raise ValueError("split manifest is not the canonical producer envelope")
    rows = tuple(read_jsonl(examples_path))
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"canonical {expected_split} examples are malformed")


def _example_from_row(row: Mapping[str, object]) -> DatasetExample:
    required = {"source", "split", "example_id", "question", "answer", "question_hash"}
    if set(row) != required or any(not isinstance(row[field], str) for field in required):
        raise ValueError("behavior example schema mismatch")
    return DatasetExample(**row)  # type: ignore[arg-type]


def _split_parent_hashes(payload: Mapping[str, object]) -> tuple[str, ...]:
    lineage = payload.get("input_lineage")
    if not isinstance(lineage, Mapping):
        return ()
    hashes = []
    for name in ("traces", "unified"):
        item = lineage.get(name)
        if isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
            hashes.append(str(item["sha256"]))
    return tuple(hashes)


def _required_hash(payload: Mapping[str, object], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} missing {field}")
    return value


def _behavior_directory_hash(path: Path) -> str:
    records = [
        {
            "path": str(candidate.relative_to(path)),
            "sha256": _file_hash(candidate),
        }
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.name != "run-manifest.json"
    ]
    return sha256_json(records)


def _validate_adapter_compatibility(checkpoint: Path, config: ExperimentConfig) -> None:
    adapter_config = _load_json_object(
        checkpoint / "adapter_config.json", "adapter configuration"
    )
    if (
        adapter_config.get("base_model_name_or_path") != config.model_id
        or adapter_config.get("revision") != QWEN25_7B_TOKENIZER_REVISION
        or not (checkpoint / "adapter_model.safetensors").is_file()
    ):
        raise ValueError("LoRA adapter is incompatible with the frozen base model/revision")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
