# Phase-Marker Mechanism Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a contamination-free behavioral and causal-mechanistic experiment that distinguishes semantic reasoning, generic computational workspace, boundary placement, and learned glyph control codes.

**Architecture:** Add a focused `phase_marker` package whose deterministic data and scoring layers feed configurable LoRA training and vLLM behavioral evaluation, then feed aligned Hugging Face activation capture and causal intervention tools. Every stage emits immutable manifests and raw per-example records; test-set outcomes, GPU launches, and mechanistic region selection are gated separately.

**Tech Stack:** Python 3.12, PyTorch 2.9, Transformers 4.57, PEFT 0.18, Datasets 4.4, vLLM, NumPy, SciPy, statsmodels, pytest, TOML configuration.

## Global Constraints

- Primary model revision: `Qwen/Qwen2.5-7B-Instruct`; record the resolved Hugging Face commit hash in every run manifest.
- Treat `🜞`, `🜆`, `🜂`, and `🜃` as reasoning-phase markers. Treat `🝞` only as the legacy final-answer delimiter and normalize every arm to the literal `Final answer:`.
- Exclude every SVAMP example from revised training; reserve the full 1,000-example SVAMP dataset for test.
- Use official GSM8K-test and MATH-test examples; require zero normalized-question hash overlap across training, validation, and test.
- Keep semantic phase content byte-identical across semantic-CoT, glyph-boundary, dot-boundary, and random-boundary arms after marker projection.
- Use one excluded pilot seed (`42`) and three fresh confirmatory seeds (`101`, `202`, `303`).
- Use LoRA `r=16`, `alpha=32`, dropout `0.05`, all attention/MLP projection targets, learning rate `2e-4`, cosine schedule, one epoch, effective batch size `16`, BF16, maximum sequence length `2048`, and checkpoints every `100` steps for all six arms.
- Train only on assistant tokens; user/chat-template tokens receive label `-100`.
- Greedy decoding is primary. Sampled robustness uses the frozen hash-selected 250-example subset per dataset and only the pre-registered primary contrasts.
- Sampled robustness uses temperature `0.7`, top-p `0.95`, five completions per prompt, and only `(semantic, neutral)`, `(glyph, neutral)`, `(glyph, glyph)`, and `(glyph, dot)` cells.
- Store checkpoints and activation tensors outside git. Store compact manifests, summaries, tests, and source code in git.
- Do not launch any paid or external GPU job without fresh approval for the exact command, model, arm set, seeds, and expected spend.

---

## File Structure

Create these bounded units:

- `phase_marker/config.py`: typed TOML configuration and fixed experiment constants.
- `phase_marker/schema.py`: immutable dataclasses for canonical traces, manifests, generations, scores, and interventions.
- `phase_marker/io.py`: canonical JSON hashing and atomic JSONL/TOML-adjacent artifact I/O.
- `phase_marker/scoring.py`: final-answer extraction, dataset normalization, equivalence, and audit sampling.
- `phase_marker/traces.py`: legacy-trace parsing, phase projection, and six deterministic training renderers.
- `phase_marker/splits.py`: source recovery, SVAMP exclusion, validation selection, official-test loading, and overlap gates.
- `phase_marker/token_audit.py`: tokenizer snapshots and matched-marker validation.
- `phase_marker/training.py`: assistant-only tokenization, LoRA construction, checkpointing, and run manifests.
- `phase_marker/prompts.py`: canonical inference templates and focused glyph perturbations.
- `phase_marker/behavior.py`: vLLM/PEFT generation matrix and immutable per-example records.
- `phase_marker/statistics.py`: paired contrasts, bootstrap intervals, hierarchical logistic analysis, and tables.
- `phase_marker/synthetic.py`: aligned four-state synthetic tasks and workspace renderings.
- `phase_marker/activations.py`: selected-position residual, attention, and logit-lens capture.
- `phase_marker/interventions.py`: residual patching, ablation, and KV-cache transplantation.
- `phase_marker/pipeline.py`: stage gates, dry runs, and manifest lineage.
- `configs/phase-marker-qwen25-7b.toml`: frozen experiment configuration.
- `tests/phase_marker/`: unit and integration tests mirroring the package modules.

Do not modify the legacy evaluators until the new pipeline reproduces their raw-generation capability. They remain historical references, not shared libraries.

---

### Task 1: Typed configuration, schemas, and artifact hashing

**Files:**
- Create: `phase_marker/__init__.py`
- Create: `phase_marker/config.py`
- Create: `phase_marker/schema.py`
- Create: `phase_marker/io.py`
- Create: `configs/phase-marker-qwen25-7b.toml`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Test: `tests/phase_marker/test_config_and_io.py`

**Interfaces:**
- Produces: `ExperimentConfig.load(path: Path) -> ExperimentConfig`
- Produces: `canonical_json(value: object) -> str`
- Produces: `sha256_json(value: object) -> str`
- Produces: `read_jsonl(path: Path) -> Iterator[dict[str, object]]`
- Produces: `write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None`
- Produces: frozen dataclasses `CanonicalTrace`, `ArtifactManifest`, `GenerationRecord`, `ScoreRecord`, and `InterventionRecord`

- [ ] **Step 1: Write the failing configuration and canonical-hash tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify the missing-package failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_config_and_io.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'phase_marker'`.

- [ ] **Step 3: Implement the minimal typed configuration and atomic I/O**

```python
# phase_marker/io.py
def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    temporary.replace(path)
```

Use `tomllib` in `ExperimentConfig.load`; reject unknown arms, duplicate seeds, a pilot seed present in confirmatory seeds, or a final delimiter containing a phase glyph. Add `scipy>=1.14` and `statsmodels>=0.14` to `requirements.txt`; add `pytest>=8.3` to `requirements-dev.txt`.

Define the shared records with these exact fields:

```python
@dataclass(frozen=True)
class PhaseSpan:
    name: Literal["guideline", "plan", "step", "takeaway"]
    body: str


@dataclass(frozen=True)
class CanonicalTrace:
    trace_id: str
    source: str
    question: str
    answer: str
    phases: tuple[PhaseSpan, PhaseSpan, PhaseSpan, PhaseSpan]


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    kind: str
    config_hash: str
    parent_hashes: tuple[str, ...]
    row_count: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    source: str
    question_hash: str
    gold_answer: str
    training_arm: str
    seed: int
    checkpoint: str
    prompt_condition: str
    prompt_hash: str
    raw_prompt: str
    raw_completion: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    decoding: Mapping[str, object]
    parent_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ScoreRecord:
    generation_id: str
    source: str
    question_hash: str
    training_arm: str
    seed: int
    prompt_condition: str
    gold_answer: str
    extracted_answer: str | None
    normalized_gold: str
    normalized_prediction: str | None
    correct: bool
    parse_error: str | None
    equivalence_reason: str


@dataclass(frozen=True)
class InterventionRecord:
    intervention_id: str
    recipient_id: str
    donor_id: str
    method: str
    layers: tuple[int, ...]
    positions: tuple[int, ...]
    baseline_target_logprob: float
    intervened_target_logprob: float
    baseline_target_rank: int
    intervened_target_rank: int
    baseline_correct: bool
    intervened_correct: bool
    parent_hashes: tuple[str, ...]
```

- [ ] **Step 4: Run the focused tests**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_config_and_io.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the foundation**

```bash
git add phase_marker/__init__.py phase_marker/config.py phase_marker/schema.py phase_marker/io.py configs/phase-marker-qwen25-7b.toml requirements.txt requirements-dev.txt tests/phase_marker/test_config_and_io.py
git commit -m "feat: add phase marker experiment foundation"
```

---

### Task 2: Strict final-answer scoring and manual-audit sampling

**Files:**
- Create: `phase_marker/scoring.py`
- Create: `tests/phase_marker/test_scoring.py`

**Interfaces:**
- Consumes: `ScoreRecord` from `phase_marker.schema`
- Produces: `extract_final_answer(text: str, delimiter: str = "Final answer:") -> str | None`
- Produces: `normalize_answer(source: str, answer: str) -> str`
- Produces: `answers_equivalent(source: str, predicted: str, gold: str) -> bool`
- Produces: `score_generation(record: GenerationRecord) -> ScoreRecord`
- Produces: `select_audit_sample(records: Sequence[ScoreRecord], per_source: int, seed: int) -> list[ScoreRecord]`

- [ ] **Step 1: Write scoring failures for the legacy last-number bug and required formats**

```python
import pytest

from phase_marker.scoring import answers_equivalent, extract_final_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 3 = 5\nFinal answer: 5", "5"),
        ("Final answer: -3/4", "-3/4"),
        ("Reasoning mentions 9.81 but has no delimiter", None),
        ("Final answer: 1,200", "1,200"),
    ],
)
def test_extract_final_answer_requires_delimiter(text, expected):
    assert extract_final_answer(text) == expected


def test_numeric_equivalence_handles_fraction_decimal_and_percent():
    assert answers_equivalent("gsm8k", "3/4", "0.75")
    assert answers_equivalent("svamp", "25%", "0.25")
    assert not answers_equivalent("gsm8k", "9.81", "5")
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_scoring.py -q`

Expected: FAIL because `phase_marker.scoring` does not exist.

- [ ] **Step 3: Implement delimiter extraction and dataset-aware normalization**

```python
FINAL_LINE = re.compile(r"(?im)^Final answer:\s*(.+?)\s*$")


def extract_final_answer(text: str, delimiter: str = "Final answer:") -> str | None:
    matches = list(FINAL_LINE.finditer(text))
    return matches[-1].group(1).strip() if matches else None


def _numeric_value(value: str) -> Fraction | Decimal | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        return Decimal(cleaned[:-1]) / Decimal(100)
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None
```

For MATH, implement deterministic normalization for whitespace, `\left`/`\right`, `\dfrac`/`\tfrac`, outer `\boxed{}`, simple fractions, and comma-separated finite sets. Never fall back to the last number. Store `parse_error`, normalized values, and equivalence reason in `ScoreRecord`.

- [ ] **Step 4: Add audit-sample stability and 1% gate tests**

```python
def test_audit_sample_is_stable_and_source_stratified(score_records):
    first = select_audit_sample(score_records, per_source=2, seed=20260804)
    second = select_audit_sample(list(reversed(score_records)), per_source=2, seed=20260804)
    assert [row.generation_id for row in first] == [row.generation_id for row in second]
    assert Counter(row.source for row in first) == {"gsm8k": 2, "svamp": 2, "math": 2}
```

- [ ] **Step 5: Run scoring tests**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_scoring.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit strict scoring**

```bash
git add phase_marker/scoring.py tests/phase_marker/test_scoring.py
git commit -m "feat: add strict phase marker scoring"
```

---

### Task 3: Canonical trace parsing and six deterministic renderers

**Files:**
- Create: `phase_marker/traces.py`
- Create: `tests/phase_marker/test_traces.py`

**Interfaces:**
- Consumes: legacy rows from `data/sft_final.jsonl`
- Produces: `parse_legacy_trace(row: Mapping[str, object]) -> CanonicalTrace`
- Produces: `render_training_example(trace: CanonicalTrace, arm: str, seed: int, max_filler_tokens: int) -> dict[str, object]`
- Produces: `semantic_projection(rendered_assistant: str) -> str`
- Produces: `recover_question(user_content: str) -> str`

- [ ] **Step 1: Write a five-glyph legacy fixture and marker-only identity tests**

```python
LEGACY = {
    "messages": [
        {"role": "user", "content": "Solve carefully.\n\nProblem:\nWhat is 2+3?\n"},
        {"role": "assistant", "content": (
            "🜞 Guideline:\nUse arithmetic.\n"
            "🜆 Plan:\nAdd the terms.\n"
            "🜂 Step:\n2+3=5.\n"
            "🜃 Takeaway:\nThe sum is five.\n"
            "🝞 Final answer: 5"
        )},
    ]
}


def test_parse_treats_final_glyph_as_delimiter_not_phase():
    trace = parse_legacy_trace(LEGACY)
    assert [phase.name for phase in trace.phases] == ["guideline", "plan", "step", "takeaway"]
    assert trace.answer == "5"


def test_marker_only_arms_have_identical_semantics():
    trace = parse_legacy_trace(LEGACY)
    outputs = [render_training_example(trace, arm, 101, 512)["messages"][1]["content"]
               for arm in ("semantic", "glyph", "dot", "random")]
    assert len({semantic_projection(output) for output in outputs}) == 1
    assert all("🝞" not in output for output in outputs)
    assert all(output.count("Final answer:") == 1 for output in outputs)
```

- [ ] **Step 2: Run the trace tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_traces.py -q`

Expected: FAIL because the parser is missing.

- [ ] **Step 3: Implement strict parsing with explicit phase boundaries**

```python
PHASE_SPECS = (
    ("guideline", "🜞", "Guideline:"),
    ("plan", "🜆", "Plan:"),
    ("step", "🜂", "Step:"),
    ("takeaway", "🜃", "Takeaway:"),
)
FINAL_GLYPH = "🝞"


def parse_legacy_trace(row: Mapping[str, object]) -> CanonicalTrace:
    assistant = str(row["messages"][1]["content"])
    positions = [assistant.index(glyph) for _, glyph, _ in PHASE_SPECS]
    final_position = assistant.index(FINAL_GLYPH)
    if positions != sorted(positions) or positions[-1] >= final_position:
        raise TraceParseError("phase markers are missing or out of order")
    # Slice exact content after each heading and before the next marker.
```

Use a stable SHA-256 of normalized source plus question as `trace_id`. Reject missing, repeated, or out-of-order markers and empty answers; write exclusions with reason codes rather than silently dropping them.

- [ ] **Step 4: Implement deterministic arms**

Rules:

- `semantic`: no headings or phase glyphs; preserve the four phase bodies in order.
- `glyph`: prepend the fixed glyph corresponding to each phase body.
- `dot`: prepend one tokenizer-selected neutral delimiter placeholder from configuration to every phase body.
- `random`: deterministically permute the four configured marker identities using `sha256(trace_id + seed)`.
- `direct`: emit only `Final answer: {answer}`.
- `filler`: deterministically assign one of `4`, `16`, `64`, or capped trace-matched filler lengths and emit dots followed by `Final answer: {answer}`.

- [ ] **Step 5: Run trace tests and a real-data parse audit**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_traces.py -q`

Run: `./.venv/bin/python -m phase_marker.traces audit --input data/sft_final.jsonl --output artifacts/phase-marker/trace-audit.jsonl`

Expected: tests PASS; audit accounts for all 3,850 rows as parsed or explicitly excluded and prints counts by reason.

- [ ] **Step 6: Commit the canonical renderer**

```bash
git add phase_marker/traces.py tests/phase_marker/test_traces.py
git commit -m "feat: render matched phase marker training arms"
```

---

### Task 4: Contamination-free splits and overlap gates

**Files:**
- Create: `phase_marker/splits.py`
- Create: `tests/phase_marker/test_splits.py`

**Interfaces:**
- Consumes: `CanonicalTrace`, `data/unified_dataset.jsonl`, Hugging Face dataset rows
- Produces: `normalize_question(text: str) -> str`
- Produces: `question_hash(source: str, question: str) -> str`
- Produces: `build_split_bundle(config: ExperimentConfig, loader: DatasetLoader, source_traces: Sequence[CanonicalTrace], unified_rows: Sequence[Mapping[str, object]]) -> SplitBundle`
- Produces: `assert_disjoint_splits(bundle: SplitBundle) -> None`

Define `DatasetExample(source, split, example_id, question, answer, question_hash)` as a frozen dataclass, `SplitBundle(train, validation, test, exclusions)` as tuples of `DatasetExample`, and `DatasetLoader` as a protocol with `load(dataset_id: str, config: str | None, split: str, revision: str) -> Sequence[Mapping[str, object]]`.

- [ ] **Step 1: Write overlap and wholesale-SVAMP exclusion tests**

```python
def test_overlap_gate_rejects_same_question_with_whitespace_changes():
    train = [example("gsm8k", "How many?\n", "5")]
    test = [example("gsm8k", "  How   many? ", "5")]
    with pytest.raises(SplitOverlapError, match="gsm8k"):
        assert_disjoint_splits(SplitBundle(train=train, validation=[], test=test))


def test_svamp_is_never_retained_for_training(source_traces, fake_loader):
    bundle = build_split_bundle(TEST_CONFIG, fake_loader, source_traces)
    assert not any(row.source == "svamp" for row in bundle.train)
    assert sum(row.source == "svamp" for row in bundle.test) == 1000
```

- [ ] **Step 2: Run split tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_splits.py -q`

Expected: FAIL because split functions are missing.

- [ ] **Step 3: Implement normalization, source recovery, and validation selection**

```python
def normalize_question(text: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(text.split())).strip().casefold()


def question_hash(source: str, question: str) -> str:
    payload = f"{source}\0{normalize_question(question)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Recover trace sources by normalized-question lookup against `unified_dataset.jsonl`; record ambiguous and unmatched rows. Exclude SVAMP traces. Select 300 unused GSM8K-train and 300 unused MATH-train examples by ascending stable hash for validation. Load full GSM8K test, full 1,000-row SVAMP, and full MATH test. Record dataset revisions and source counts.

- [ ] **Step 4: Add a CLI that writes frozen manifests and blocks overlap**

Run contract:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python -m phase_marker.splits build \
  --config configs/phase-marker-qwen25-7b.toml \
  --traces data/sft_final.jsonl \
  --unified data/unified_dataset.jsonl \
  --output-root artifacts/phase-marker/splits
```

The offline command must succeed when datasets are cached; if not cached, fail with the exact missing dataset/revision and do not emit a partial manifest.

- [ ] **Step 5: Run tests and the cached-data gate**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_splits.py -q`

Run the CLI above. Expected: tests PASS; CLI reports zero overlap and exactly 1,000 held-out SVAMP rows, or fails before writing with an explicit cache-miss error.

- [ ] **Step 6: Commit split integrity**

```bash
git add phase_marker/splits.py tests/phase_marker/test_splits.py
git commit -m "feat: enforce held-out phase marker splits"
```

---

### Task 5: Tokenizer audit and training-dataset materialization

**Files:**
- Create: `phase_marker/token_audit.py`
- Create: `tests/phase_marker/test_token_audit.py`
- Create: `tests/phase_marker/test_materialize.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `CanonicalTrace`, Hugging Face tokenizer
- Produces: `audit_marker_set(tokenizer, symbols: Sequence[str]) -> list[TokenAuditRow]`
- Produces: `select_neutral_delimiter(audit: Sequence[TokenAuditRow], target_width: int) -> str`
- Produces: `materialize_training_arms(config: ExperimentConfig, traces: Sequence[CanonicalTrace], tokenizer, output_root: Path) -> dict[str, ArtifactManifest]`

Define `TokenAuditRow(symbol, codepoints, utf8_hex, token_ids, token_strings, token_count, vocabulary_member, local_corpus_count)`. `local_corpus_count` is measured over the canonical training questions and traces and must be labeled as a local frequency proxy, never as pretraining frequency.

- [ ] **Step 1: Write fake-tokenizer matching and manifest tests**

```python
def test_neutral_delimiter_matches_glyph_token_width(fake_tokenizer):
    glyph_width = len(fake_tokenizer.encode("🜞", add_special_tokens=False))
    audit = audit_marker_set(fake_tokenizer, [".", "|", "§"])
    selected = select_neutral_delimiter(audit, target_width=glyph_width)
    assert len(fake_tokenizer.encode(selected, add_special_tokens=False)) == glyph_width


def test_materialized_marker_arms_share_semantic_hashes(materialized):
    hashes = {arm: manifest.metadata["semantic_dataset_hash"]
              for arm, manifest in materialized.items()
              if arm in {"semantic", "glyph", "dot", "random"}}
    assert len(set(hashes.values())) == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_token_audit.py tests/phase_marker/test_materialize.py -q`

Expected: FAIL because tokenizer audit/materialization is missing.

- [ ] **Step 3: Implement measured token selection and dataset manifests**

Snapshot code point, UTF-8 bytes, token IDs, token strings, token count, vocabulary membership, and local-corpus count for every glyph, dot candidate, emoji control, and random-symbol candidate. Reject a dot or random-marker comparison when widths do not match the configured target. Materialize all six JSONL arms under `artifacts/phase-marker/training-data/`; record row hashes, semantic hashes, exclusions, filler-length counts, tokenizer revision, and parent split hash.

- [ ] **Step 4: Run tests and materialize a 10-row dry run**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_token_audit.py tests/phase_marker/test_materialize.py -q`

Run: `./.venv/bin/python -m phase_marker.token_audit materialize --config configs/phase-marker-qwen25-7b.toml --limit 10 --output-root /tmp/phase-marker-materialize`

Expected: tests PASS; six 10-row datasets and six manifests are emitted; the four marker-only arms report one shared semantic hash.

- [ ] **Step 5: Commit token and materialization gates**

```bash
git add phase_marker/token_audit.py tests/phase_marker/test_token_audit.py tests/phase_marker/test_materialize.py
git commit -m "feat: audit and materialize phase marker arms"
```

---

### Task 6: Configurable assistant-only LoRA training

**Files:**
- Create: `phase_marker/training.py`
- Create: `tests/phase_marker/test_training.py`

**Interfaces:**
- Consumes: materialized arm JSONL and `ExperimentConfig`
- Produces: `tokenize_assistant_only(example: Mapping[str, object], tokenizer, max_length: int) -> dict[str, list[int]]`
- Produces: `build_lora_config(config: ExperimentConfig) -> LoraConfig`
- Produces: `build_training_arguments(config, arm, seed, output_dir) -> TrainingArguments`
- Produces: checkpoint directory plus run manifest

- [ ] **Step 1: Write assistant-mask and fixed-hyperparameter tests**

```python
def test_user_tokens_are_masked(fake_chat_tokenizer):
    encoded = tokenize_assistant_only(EXAMPLE, fake_chat_tokenizer, max_length=128)
    boundary = encoded["assistant_start"]
    assert set(encoded["labels"][:boundary]) == {-100}
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]


def test_lora_and_training_arguments_are_arm_invariant(config):
    left = build_training_arguments(config, "glyph", 101, Path("/tmp/glyph"))
    right = build_training_arguments(config, "semantic", 101, Path("/tmp/semantic"))
    assert left.learning_rate == right.learning_rate == 2e-4
    assert left.gradient_accumulation_steps * left.per_device_train_batch_size == 16
    assert build_lora_config(config).r == 16
```

- [ ] **Step 2: Run training tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_training.py -q`

Expected: FAIL because the training module is missing.

- [ ] **Step 3: Implement assistant-boundary tokenization**

Render the full conversation and the user-only prefix with `apply_chat_template`; tokenize both without truncation; set labels before the assistant boundary to `-100`; then truncate both inputs and labels together. Reject any example whose final-answer delimiter or assistant answer is truncated.

```python
labels = [-100] * assistant_start + input_ids[assistant_start:]
if len(input_ids) > max_length:
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
if final_delimiter_ids not in sliding_windows(input_ids, len(final_delimiter_ids)):
    raise TruncatedAnswerError(example_id)
```

- [ ] **Step 4: Implement LoRA training and immutable run manifests**

The CLI must require `--arm`, `--seed`, `--data`, `--output-dir`, and `--manifest`; reject confirmatory output directories that already contain a different config hash. Save tokenizer, adapter, trainer state, resolved model revision, CUDA/PyTorch versions, command arguments, dataset hash, and checkpoint list.

- [ ] **Step 5: Run tests and a CPU tokenization smoke test**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_training.py -q`

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python -m phase_marker.training tokenize-smoke --config configs/phase-marker-qwen25-7b.toml --data /tmp/phase-marker-materialize/glyph.jsonl --limit 10`

Expected: tests PASS; smoke reports 10/10 examples with user labels masked and final delimiters retained. It must not load model weights.

- [ ] **Step 6: Commit configurable training**

```bash
git add phase_marker/training.py tests/phase_marker/test_training.py
git commit -m "feat: train matched phase marker adapters"
```

---

### Task 7: Canonical prompts, perturbations, and raw behavioral evaluation

**Files:**
- Create: `phase_marker/prompts.py`
- Create: `phase_marker/behavior.py`
- Create: `tests/phase_marker/test_prompts.py`
- Create: `tests/phase_marker/test_behavior.py`

**Interfaces:**
- Produces: `render_prompt(question: str, condition: str, marker_set: MarkerSet) -> str`
- Produces: `render_perturbation(question: str, perturbation: str, marker_set: MarkerSet) -> str`
- Produces: `build_behavior_matrix(config: ExperimentConfig, split_manifest: ArtifactManifest) -> tuple[EvaluationCell, ...]`
- Produces: `records_from_outputs(cell: EvaluationCell, examples: Sequence[DatasetExample], requests: Sequence[GenerationRequest], outputs: Sequence[GenerationOutput], parent_hashes: tuple[str, ...]) -> list[GenerationRecord]`
- Defines: `GenerationBackend.generate(requests: Sequence[GenerationRequest]) -> Sequence[GenerationOutput]`

Define frozen `MarkerSet(guideline, plan, step, takeaway)`, `EvaluationCell(kind, training_arm, prompt_condition, perturbation, decoding_name)`, `GenerationRequest(generation_id, prompt, prompt_token_ids, max_new_tokens, decoding)`, and `GenerationOutput(generation_id, text, token_ids, token_logprobs)`. The `GenerationBackend` protocol returns outputs in request order and must reject missing or duplicate generation IDs.

- [ ] **Step 1: Write prompt-diff and matrix tests**

```python
def test_primary_prompts_differ_only_in_declared_format_span():
    rendered = {name: render_prompt("What is 2+3?", name, TEST_MARKERS)
                for name in ("neutral", "glyph", "dot", "headings")}
    projections = {strip_format_span(text) for text in rendered.values()}
    assert projections == {"What is 2+3?"}


def test_matrix_contains_preregistered_cells_only(config, split_manifest):
    cells = build_behavior_matrix(config, split_manifest)
    assert {(c.training_arm, c.prompt_condition) for c in cells if c.kind == "primary"} == {
        (arm, prompt)
        for arm in ("semantic", "glyph", "dot", "random")
        for prompt in ("neutral", "glyph", "dot", "headings")
    }
```

- [ ] **Step 2: Run tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_prompts.py tests/phase_marker/test_behavior.py -q`

Expected: FAIL because prompt/evaluation modules are missing.

- [ ] **Step 3: Implement one canonical template and focused perturbations**

```python
PROMPT_TEMPLATE = """Solve the problem carefully.{format_span}

Problem:
{question}

End with exactly one line of the form `Final answer: <answer>`.
"""
```

Implement `delete`, `cluster`, `displace`, `permute`, `dot_replace`, and `unseen_replace`. Each perturbation records the exact rendered prompt and token IDs. Do not reuse the existing natural/XML prompts because they change descriptions and nesting.

- [ ] **Step 4: Implement an injectable backend and vLLM adapter evaluation**

Use a fake backend in tests and a vLLM backend in production. Write one JSONL `GenerationRecord` per problem containing generation ID, source, question hash, arm, seed, checkpoint, prompt condition, prompt hash, raw prompt, raw completion, token IDs/counts, decoding settings, correctness fields, and all parent manifest hashes. Never write summary-only CSV as the source of truth.

The sampled robustness matrix is exactly the four cells declared in Global Constraints; all other cells remain greedy-only.

- [ ] **Step 5: Run tests and fake-backend end-to-end scoring**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_prompts.py tests/phase_marker/test_behavior.py -q`

Run: `./.venv/bin/python -m phase_marker.behavior dry-run --config configs/phase-marker-qwen25-7b.toml --backend fake --limit 3 --output /tmp/phase-marker-generations.jsonl`

Expected: tests PASS; dry run emits exactly the configured cells times three records, each independently rescorable.

- [ ] **Step 6: Commit behavioral evaluation**

```bash
git add phase_marker/prompts.py phase_marker/behavior.py tests/phase_marker/test_prompts.py tests/phase_marker/test_behavior.py
git commit -m "feat: evaluate phase marker behavior matrix"
```

---

### Task 8: Confirmatory statistics, audit gate, and result tables

**Files:**
- Create: `phase_marker/statistics.py`
- Create: `tests/phase_marker/test_statistics.py`

**Interfaces:**
- Consumes: immutable `GenerationRecord`/`ScoreRecord` JSONL
- Produces: `paired_bootstrap_delta(left, right, seed, draws=10_000) -> Interval`
- Produces: `fit_hierarchical_logit(records) -> ModelSummary`
- Produces: `apply_audit_gate(auto_scores, manual_scores, threshold=0.01) -> AuditResult`
- Produces: Markdown/LaTeX tables and a machine-readable summary

Define frozen `Interval(point, low, high, draws, seed)`, `ModelSummary(formula, coefficients, converged, diagnostics)`, and `AuditResult(passed, disagreements, total, rate, threshold)` records.

- [ ] **Step 1: Write paired-alignment, interval, and audit-gate tests**

```python
def test_paired_bootstrap_rejects_unaligned_question_sets():
    with pytest.raises(UnpairedComparisonError):
        paired_bootstrap_delta([score("a", True)], [score("b", True)], seed=7)


def test_audit_gate_blocks_more_than_one_percent_disagreement():
    auto = [audit_row(str(i), "1") for i in range(100)]
    manual = [audit_row(str(i), "2" if i < 2 else "1") for i in range(100)]
    assert not apply_audit_gate(auto, manual, threshold=0.01).passed
```

- [ ] **Step 2: Run statistical tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_statistics.py -q`

Expected: FAIL because statistical analysis is missing.

- [ ] **Step 3: Implement pre-registered contrasts**

Align by `(source, question_hash, seed)`. Use a local `np.random.default_rng(seed)` for 10,000 paired bootstrap draws. Fit `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` with fixed effects for training arm, prompt, their interaction, and dataset; random intercepts for question hash and seed. Emit coefficient, posterior SD, interval, and convergence diagnostics. Mark effects inconclusive when absolute delta is below `0.02` or the paired interval includes zero.

- [ ] **Step 4: Implement audit templates and table generation**

Generate a 300-row manual-audit TSV template (100 per source), ingest completed labels, and block confirmatory tables above 1% disagreement. Tables must distinguish evaluation-sample intervals from three-seed variation and apply Holm correction only to declared secondary contrasts.

- [ ] **Step 5: Run tests and synthetic-result smoke analysis**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_statistics.py -q`

Run: `./.venv/bin/python -m phase_marker.statistics smoke --output-root /tmp/phase-marker-analysis`

Expected: tests PASS; smoke produces a Markdown contrast table, model diagnostics, and an audit-gate status.

- [ ] **Step 6: Commit confirmatory analysis**

```bash
git add phase_marker/statistics.py tests/phase_marker/test_statistics.py
git commit -m "feat: analyze phase marker contrasts"
```

---

### Task 9: Aligned synthetic four-state mechanism suite

**Files:**
- Create: `phase_marker/synthetic.py`
- Create: `tests/phase_marker/test_synthetic.py`

**Interfaces:**
- Produces: `generate_synthetic_suite(seed: int, counts: SplitCounts) -> SyntheticBundle`
- Produces: `render_workspace(example: SyntheticExample, condition: str, total_tokens: int, tokenizer) -> WorkspacePrompt`
- Produces: exact intermediate values `(state_1, state_2, state_3, state_4)` per example

Define frozen `SplitCounts(train, validation, test)`, `SyntheticExample(example_id, family, parameters, parameter_hash, question, intermediates, answer)`, `SyntheticSplit(examples, parameter_hashes)`, `WorkspaceRegion(index, start, end, marker_position)`, `WorkspacePrompt(example_id, condition, total_tokens, text, token_ids, regions)`, and `SyntheticBundle(train: SyntheticSplit, validation: SyntheticSplit, test: SyntheticSplit, manifest)`.

- [ ] **Step 1: Write deterministic-state, split-disjointness, and slot-layout tests**

```python
def test_affine_chain_intermediates_are_exact():
    row = affine_example(x=3, operations=(("mul", 2), ("add", 5), ("mul", 3), ("sub", 4)))
    assert row.intermediates == (6, 11, 33, 29)
    assert row.answer == "29"


def test_workspace_has_four_aligned_regions(fake_tokenizer):
    prompt = render_workspace(SYNTHETIC_ROW, "glyph", 64, fake_tokenizer)
    assert len(prompt.regions) == 4
    assert [region.end - region.start for region in prompt.regions] == [16, 16, 16, 16]


def test_parameter_tuples_do_not_cross_splits():
    bundle = generate_synthetic_suite(101, SplitCounts(train=100, validation=20, test=20))
    assert bundle.train.parameter_hashes.isdisjoint(bundle.validation.parameter_hashes)
    assert bundle.train.parameter_hashes.isdisjoint(bundle.test.parameter_hashes)
```

- [ ] **Step 2: Run synthetic tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_synthetic.py -q`

Expected: FAIL because synthetic generation is missing.

- [ ] **Step 3: Implement four task families and aligned workspace regions**

Generate modular chains, affine chains, two-source numeric composition, and exact string transformation/composition. Each has four explicit intermediate values. For actual tokenizer workspace lengths 12, 16, and 64, create four equal regions of widths 3, 4, and 16. Each measured marker sequence begins its region (`glyph`, `dot`, `repeated_glyph`, `permuted_glyph`, `random_symbol`); reject a marker that exceeds the measured region width. Remaining token positions use the shared tokenizer-matched neutral filler. `no_slot` has no regions. Token IDs and region offsets always describe the complete rendered tokenizer output, never logical slots.

- [ ] **Step 4: Run tests and materialize a reproducible smoke suite**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_synthetic.py -q`

Run: `./.venv/bin/python -m phase_marker.synthetic build --seed 101 --train 100 --validation 20 --test 20 --output-root /tmp/phase-marker-synthetic`

Expected: tests PASS; manifest reports four families, exact scorer agreement, and zero parameter overlap.

- [ ] **Step 5: Commit the synthetic suite**

```bash
git add phase_marker/synthetic.py tests/phase_marker/test_synthetic.py
git commit -m "feat: add aligned phase marker mechanism tasks"
```

---

### Task 10: Selected-position activation, attention, and logit-lens capture

**Files:**
- Create: `phase_marker/activations.py`
- Create: `tests/phase_marker/test_activations.py`

**Interfaces:**
- Consumes: Hugging Face causal LM, tokenizer, `WorkspacePrompt`, `CaptureSpec`
- Produces: `capture_selected_states(model, batch, spec) -> ActivationBatch`
- Produces: `apply_logit_lens(model, activation_batch, candidate_token_ids=None) -> LogitLensBatch`
- Produces: `fit_phase_probe(train: ActivationBatch, validation: ActivationBatch, seed: int) -> PhaseProbe`
- Produces: `evaluate_phase_probe(probe: PhaseProbe, test: ActivationBatch) -> ProbeMetrics`
- Produces: compact tensor artifact plus metadata manifest

Define frozen `CaptureSpec(layers, positions, capture_residual, capture_attention)`, `ActivationBatch(example_ids, conditions, layers, positions, residual, attention, parent_hashes)`, `LogitLensBatch(token_ids, logprobs, ranks, parent_hashes)`, `PhaseProbe(weight, bias, source_condition, layer, seed)`, and `ProbeMetrics(accuracy, macro_f1, source_condition, target_condition, layer)`.

- [ ] **Step 1: Write hook-position and no-generation-change tests with a tiny local model**

```python
def test_capture_returns_only_requested_layers_and_positions(tiny_causal_lm):
    spec = CaptureSpec(layers=(0, 2), positions=(3, 7))
    captured = capture_selected_states(tiny_causal_lm, TINY_BATCH, spec)
    assert captured.residual.shape[:2] == (2, 2)
    assert captured.layers == (0, 2)
    assert captured.positions == (3, 7)


def test_capture_does_not_change_logits(tiny_causal_lm):
    baseline = tiny_causal_lm(**TINY_BATCH).logits
    with capture_context(tiny_causal_lm, CaptureSpec(layers=(1,), positions=(2,))):
        observed = tiny_causal_lm(**TINY_BATCH).logits
    torch.testing.assert_close(baseline, observed)
```

- [ ] **Step 2: Run activation tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_activations.py -q`

Expected: FAIL because activation capture is missing.

- [ ] **Step 3: Implement architecture adapters and bounded capture**

Resolve decoder layers through an explicit Qwen adapter (`model.model.layers`), register read-only forward hooks, gather only configured sequence positions, and remove hooks in `finally`. Request attentions only for the aligned validation subset because full attention tensors are quadratic. Save tensors with `torch.save` or safetensors plus a JSON manifest containing shape, dtype, selected IDs, layer/position map, and parent hashes.

- [ ] **Step 4: Implement logit-lens and intermediate decoding metrics**

Apply the model's final RMSNorm and unembedding to selected residual states. Record rank/log-probability of each known synthetic intermediate and the final answer. For natural math, only evaluate predeclared candidate answer/intermediate tokens; label teacher-forced and free-generation captures separately.

Fit a regularized linear phase probe in PyTorch on validation-selected layers. Train on glyph captures and test without refitting on dot and unseen-symbol captures, then reverse the source/target formats. Select regularization and layer on validation only; report held-out accuracy and macro-F1 as correlational evidence.

- [ ] **Step 5: Run tests and a tiny-model CLI smoke test**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_activations.py -q`

Run: `./.venv/bin/python -m phase_marker.activations smoke --output-root /tmp/phase-marker-activations`

Expected: tests PASS; smoke proves identical baseline/capture logits and writes bounded selected-position tensors.

- [ ] **Step 6: Commit activation capture**

```bash
git add phase_marker/activations.py tests/phase_marker/test_activations.py
git commit -m "feat: capture phase marker activations"
```

---

### Task 11: Residual patching, ablations, and KV-cache transplantation

**Files:**
- Create: `phase_marker/interventions.py`
- Create: `tests/phase_marker/test_interventions.py`

**Interfaces:**
- Consumes: aligned donor/recipient batches and `InterventionSpec`
- Produces: `patch_residual_positions(model, recipient_batch: Mapping[str, Tensor], donor_batch: Mapping[str, Tensor], spec: InterventionSpec) -> InterventionResult`
- Produces: `ablate_positions(model, batch: Mapping[str, Tensor], spec: InterventionSpec, validation_mean: Tensor | None = None) -> InterventionResult`
- Produces: `transplant_kv_positions(model, recipient_batch: Mapping[str, Tensor], donor_batch: Mapping[str, Tensor], spec: InterventionSpec) -> InterventionResult`
- Produces: one `InterventionRecord` per donor-recipient-layer-region combination

Define frozen `InterventionSpec(method, layers, positions, norm_match, target_token_ids, control_name)` and `InterventionResult(record, baseline_logits, intervened_logits)`.

- [ ] **Step 1: Write selectivity and negative-control tests**

```python
def test_residual_patch_changes_only_selected_positions():
    patched = replace_positions(RECIPIENT, DONOR, positions=(2, 5))
    torch.testing.assert_close(patched[:, [0, 1, 3, 4]], RECIPIENT[:, [0, 1, 3, 4]])
    torch.testing.assert_close(patched[:, [2, 5]], DONOR[:, [2, 5]])


def test_kv_transplant_rejects_unaligned_sequences(fake_cache):
    with pytest.raises(AlignmentError):
        transplant_cache_rows(fake_cache(length=8), fake_cache(length=9), positions=(3,))
```

- [ ] **Step 2: Run intervention tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_interventions.py -q`

Expected: FAIL because intervention functions are missing.

- [ ] **Step 3: Implement residual hooks and norm-matched replacements**

At a configured decoder layer, replace recipient residual rows with donor rows after optional mean-centering and recipient-norm matching. Implement zero, validation-mean, within-batch shuffle, matched non-marker position, and random-donor controls through the same code path. Always preserve unselected rows bit-for-bit.

- [ ] **Step 4: Implement Qwen DynamicCache transplantation**

Copy only selected sequence rows from donor key/value tensors into a cloned recipient cache for specified layers. Reject mismatched layer counts, batch size, head shape, dtype, device, or sequence alignment. Measure recipient answer log-probability/rank before and after; also measure the donor answer rank to detect causal answer transfer.

- [ ] **Step 5: Run tests and synthetic causal smoke test**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_interventions.py -q`

Run: `./.venv/bin/python -m phase_marker.interventions smoke --output-root /tmp/phase-marker-interventions`

Expected: tests PASS; selected-position patching changes the toy target logit, while random and non-marker controls remain within the test tolerance.

- [ ] **Step 6: Commit causal interventions**

```bash
git add phase_marker/interventions.py tests/phase_marker/test_interventions.py
git commit -m "feat: add causal phase marker interventions"
```

---

### Task 12: Pipeline gates, local verification, and operator documentation

**Files:**
- Create: `phase_marker/pipeline.py`
- Create: `tests/phase_marker/test_pipeline.py`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: every preceding stage and its manifests
- Produces: `run_gate(stage: str, config: ExperimentConfig, artifact_root: Path) -> GateResult`
- Produces: dry-run/pilot/confirmatory command manifests

Define frozen `GateResult(stage, passed, reason, checked_hashes, next_commands)`; `run_gate` never executes `next_commands`.

- [ ] **Step 1: Write pipeline dependency and stale-parent tests**

```python
def test_behavior_gate_requires_matching_split_and_training_hashes(tmp_path):
    write_manifest(tmp_path / "split.json", hash="split-a")
    write_manifest(tmp_path / "adapter.json", parent_split_hash="split-b")
    result = run_gate("behavior", TEST_CONFIG, tmp_path)
    assert not result.passed
    assert "parent split hash" in result.reason


def test_confirmatory_gate_rejects_pilot_seed(tmp_path):
    result = validate_run_request(kind="confirmatory", seeds=(42,), config=TEST_CONFIG)
    assert not result.passed
```

- [ ] **Step 2: Run pipeline tests and verify failure**

Run: `./.venv/bin/python -m pytest tests/phase_marker/test_pipeline.py -q`

Expected: FAIL because pipeline gates are missing.

- [ ] **Step 3: Implement stage gates and explicit launch manifests**

Stages are `splits`, `render`, `tokenize`, `train`, `behavior`, `audit`, `statistics`, `synthetic`, `capture`, and `intervene`. Every stage validates config hash, parent hashes, row counts, exclusions, and completion markers before producing the next command manifest. `--dry-run` prints commands and expected outputs without loading a model or mutating external state.

- [ ] **Step 4: Document local and GPU workflows**

Add README commands for:

```bash
./.venv/bin/python -m pytest tests/phase_marker -q
./.venv/bin/python -m phase_marker.pipeline dry-run --config configs/phase-marker-qwen25-7b.toml --artifact-root artifacts/phase-marker
./.venv/bin/python -m phase_marker.pipeline gate --stage train --kind pilot --seeds 42 --config configs/phase-marker-qwen25-7b.toml --artifact-root artifacts/phase-marker
```

Document that the final command only validates and prints the exact GPU command; it does not launch it. Update `.gitignore` to ignore `artifacts/phase-marker/checkpoints/`, `activations/`, `raw-generations/`, and `*.pt` while allowing compact Markdown/TOML summaries.

- [ ] **Step 5: Run the complete local verification suite**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python -m pytest tests/phase_marker -q`

Run: `./.venv/bin/python -m phase_marker.pipeline dry-run --config configs/phase-marker-qwen25-7b.toml --artifact-root /tmp/phase-marker-pipeline`

Run: `git diff --check`

Expected: all tests PASS; dry run lists all six arms and three confirmatory seeds without launching; diff check is clean.

- [ ] **Step 6: Commit the integrated local pipeline**

```bash
git add phase_marker/pipeline.py tests/phase_marker/test_pipeline.py README.md .gitignore
git commit -m "feat: gate phase marker experiment pipeline"
```

---

### Task 13: Approval-gated pilot, confirmatory runs, mechanism study, and manuscript revision

**Files:**
- Modify after results: `paper/main.tex`
- Modify after results: `paper/references.bib`
- Create after results: `paper/tables/phase-marker-confirmatory.tex`
- Create after results: `paper/tables/phase-marker-interventions.tex`
- Create after results: `paper/figures/phase-marker-workspace.pdf`
- Create: `docs/phase-marker-experiment-report.md`

**Interfaces:**
- Consumes: approved exact GPU commands and all pipeline manifests
- Produces: excluded pilot report, three-seed confirmatory artifacts, intervention artifacts, bounded manuscript claims

- [ ] **Step 1: Generate and review the exact pilot command without launching**

Run:

```bash
./.venv/bin/python -m phase_marker.pipeline commands \
  --kind pilot \
  --arms semantic glyph dot random direct filler \
  --seeds 42 \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker
```

Expected: one command per arm, estimated GPU-hours, output roots, model revision, and an explicit `approval_required=true` field.

- [ ] **Step 2: Obtain fresh approval and execute only the six-arm pilot**

Do not infer approval from design or plan approval. Ask for approval containing the exact six commands, hardware, maximum duration, and estimated spend. After approval, run only seed 42. Do not promote pilot results into confirmatory tables.

- [ ] **Step 3: Verify the pilot gates**

Run:

```bash
./.venv/bin/python -m phase_marker.pipeline gate \
  --stage behavior --kind pilot --seeds 42 \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker
```

Expected: all six adapters exist; no truncation or semantic-hash failures; raw generations rescore identically; parser audit is at or below 1%; activation capture reproduces baseline logits.

- [ ] **Step 4: Freeze the confirmatory command manifest and obtain fresh approval**

Generate commands for seeds `101 202 303`. Record hashes before launch. Request fresh approval for the exact 18 training jobs and bounded evaluation workload. No hyperparameter, prompt, layer, or split changes are allowed after approval without invalidating the confirmatory label.

- [ ] **Step 5: Execute and verify behavioral confirmatory runs**

After approval, train 18 adapters, select checkpoints on validation only, run the four-by-four primary grid, focused perturbations, direct/filler lengths, and the predeclared sampled robustness subset. Complete the 300-generation manual audit before producing confirmatory tables.

Run:

```bash
./.venv/bin/python -m phase_marker.statistics analyze \
  --config configs/phase-marker-qwen25-7b.toml \
  --generations artifacts/phase-marker/raw-generations \
  --manual-audit artifacts/phase-marker/audit/manual-labels.tsv \
  --output-root artifacts/phase-marker/analysis
```

Expected: all pre-registered contrasts, three individual seed rows, paired intervals, mixed-model diagnostics, and an interpretation-matrix verdict.

- [ ] **Step 6: Run validation-selected activation and intervention studies**

Select layers/regions using synthetic and natural validation data only, freeze the selection manifest, then run test captures, logit-lens decoding, attention relay analysis, residual patching, ablations, and KV transplantation. Verify random-donor and matched non-marker controls before interpreting glyph effects.

- [ ] **Step 7: Write the artifact-first experiment report**

`docs/phase-marker-experiment-report.md` must state exact split counts and hashes, exclusions, model/tokenizer revisions, run IDs, checkpoint choices, seed results, parser audit disagreement, behavioral contrasts, intervention effects, null results, and the one allowed conclusion selected from the design's interpretation matrix.

- [ ] **Step 8: Revise the manuscript without preserving superseded claims**

Replace the contaminated 1,000-example tables and the claims that glyphs are atomic, occupy empty representation space, or establish phase separation. Add the two filler-token papers, exact templates, tokenizer audit, held-out splits, seed uncertainty, causal controls, and only the mechanism language supported by interventions. Keep pilot outcomes explicitly excluded.

- [ ] **Step 9: Compile and verify the final paper and reproduction bundle**

Run:

```bash
tectonic paper/main.tex --outdir paper/build
./.venv/bin/python -m pytest tests/phase_marker -q
git diff --check
```

Expected: PDF compilation succeeds; all tests pass; tables reproduce from immutable raw records; report and manuscript cite exact artifact hashes.

- [ ] **Step 10: Commit the evidence-backed revision**

```bash
git add paper/main.tex paper/references.bib paper/tables/phase-marker-confirmatory.tex paper/tables/phase-marker-interventions.tex paper/figures/phase-marker-workspace.pdf docs/phase-marker-experiment-report.md
git commit -m "docs: revise phase marker mechanism evidence"
```
