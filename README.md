# Glyph Reasoning

Research code for testing whether symbolic phase markers improve mathematical
reasoning, and—if they do—what computation the markers support.

The existing glyph-tuned checkpoint is available at
[loveless2001/qwen2.5-7b-glyph-sft](https://huggingface.co/loveless2001/qwen2.5-7b-glyph-sft).
The revised experiment uses `Qwen/Qwen2.5-7B-Instruct` and separates behavioral
effects from mechanistic claims.

## Experiment

The central question is whether markers do more than decorate a chain of
thought. Six matched training arms isolate different explanations:

| Arm | Output format | What it tests |
| --- | --- | --- |
| `semantic` | Reasoning content without markers | Benefit from semantic reasoning alone |
| `glyph` | The same content with fixed glyphs at phase boundaries | Learned phase-specific codes |
| `dot` | The same content with neutral delimiters | Generic boundary or workspace effects |
| `random` | Marker identities remapped per example | Whether stable marker identity matters |
| `direct` | Final answer only | No-intermediate-token baseline |
| `filler` | Controlled dot sequences, then the answer | Whether extra token positions enable hidden computation |

The dot/filler comparison is motivated by work showing that apparently
meaningless intermediate tokens can provide computation without a readable
chain of thought. It is a competing mechanism, not evidence that glyphs have a
phase-specific role.

Training, validation checkpoint selection, held-out behavior evaluation, and
mechanism analysis are separate stages:

1. Build leakage-checked train, validation, and test manifests.
2. Materialize byte-matched renderings for all six arms.
3. Train one adapter per arm with matched data order and hyperparameters.
4. Select checkpoints using validation data only.
5. Score the frozen held-out test sets.
6. Run activation capture and causal interventions only after the behavioral
   result is known.

The excluded pilot uses seed `42`. Confirmatory seeds are `101`, `202`, and
`303`. The frozen configuration is
[`configs/phase-marker-qwen25-7b.toml`](configs/phase-marker-qwen25-7b.toml), and
the full study design is documented in
[`docs/superpowers/specs/2026-08-04-phase-marker-mechanism-design.md`](docs/superpowers/specs/2026-08-04-phase-marker-mechanism-design.md).

## Scoring

Strict final-answer accuracy is the primary behavioral outcome. Answers are
normalized by task-specific parsers, then compared against the canonical gold
answer. Format compliance, output length, answer log probability, dataset and
difficulty breakdowns, and wrong-to-correct transitions are secondary
diagnostics; they do not replace exact-answer accuracy.

Checkpoint selection uses validation accuracy only, with mean gold-answer log
probability and then earliest step as deterministic tie-breakers. Test results
must not influence checkpoint, prompt, layer, or intervention selection.

The main pre-registered contrasts are:

- glyph versus identical semantic reasoning;
- dots versus identical semantic reasoning;
- glyphs versus dots;
- aligned glyphs versus deleted, displaced, clustered, or permuted glyphs;
- filler versus direct answers at each filler length; and
- training-format by inference-format interaction.

A behavioral gain establishes that a condition helps under this protocol. It
does not establish the mechanism. Probe decodability is correlational; a
mechanistic claim requires a selective intervention—such as patching,
ablation, or KV-state transplantation—that changes downstream computation.

## Running the research pipeline

Install the research dependencies:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Inspect the experiment CLIs before constructing artifacts:

```bash
./.venv/bin/python -m phase_marker.splits --help
./.venv/bin/python -m phase_marker.training --help
./.venv/bin/python -m phase_marker.behavior --help
./.venv/bin/python -m phase_marker.statistics --help
```

Print the confirmatory command graph without launching work:

```bash
./.venv/bin/python -m phase_marker.pipeline dry-run \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker
```

The Modal pilot is defined by `modal_phase_marker.py`; its content-addressed
plan can be inspected locally with:

```bash
./.venv/bin/python -m phase_marker.modal_plan plan \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt
```

Modal execution uses the fully locked environment in
`requirements-modal-phase-marker.txt`. Remote runs write durable manifests and
receipts so completed arms can be reused after an interruption. A partial
canonical output is treated as failed; start a new content-bound run rather
than guessing which files are valid.

## Focused verification

The tests protect scientific comparisons, leakage controls, scoring, artifact
lineage, and the small Modal boundary. They are offline research checks—not
evidence that a GPU, scheduler, model download, or live Modal account works.

Run scientific tests with the research environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/python -m pytest tests/phase_marker --ignore=tests/phase_marker/test_modal.py -q
```

If Modal is installed in a separate operator environment, verify its compact
boundary there:

```bash
python -m pytest tests/phase_marker/test_modal.py -q
```

## Legacy scripts

Earlier one-off training and evaluation scripts remain in `train/`, `eval/`,
and `experiments/`. The maintained revised experiment lives in `phase_marker/`
with tests in `tests/phase_marker/`.

## Glyph dictionary

| Glyph | Name | Original role |
| --- | --- | --- |
| 🜞 | Crux | Guideline or core rule |
| 🜆 | Flux | Plan |
| 🜂 | Ignis | Execution step |
| 🜃 | Terra | Conclusion |

## License

Apache 2.0
