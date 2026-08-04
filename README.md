🔗 **[loveless2001/qwen2.5-7b-glyph-sft](https://huggingface.co/loveless2001/qwen2.5-7b-glyph-sft)**

## Project Structure

*   `train/`: Scripts for fine-tuning the models (SFT).
*   `eval/`: Evaluation suites comparing Glyph, XML, and Natural language prompting strategies.
*   `data/`: specific datasets used for training and evaluation.
*   `checkpoints/`: Local storage for model checkpoints.
*   `experiments/`: Miscellaneous experiment scripts.

## Installation

1.  Clone the repository.
2.  Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Inference

You can run a quick inference test using the provided script:

```bash
python infer_test.py
```

### Evaluation

To evaluate the model against different prompting strategies (Glyph vs. XML vs. Natural):

```bash
python eval/eval_structures.py --models loveless2001/qwen2.5-7b-glyph-sft --limit 20
```

### Training

Training scripts are located in the `train/` directory. For example, to run SFT:

```bash
python train/train_sft.py
```

## Phase-marker mechanism pipeline

The maintained local verification suite is designed to run without network
access, model weights, a GPU, or a scheduler:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python -m pytest tests/phase_marker -q
```

Inspect the complete confirmatory launch plan with a read-only dry run:

```bash
./.venv/bin/python -m phase_marker.pipeline dry-run \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker
```

The dry run loads only the TOML configuration. It does not load a tokenizer or
model, create the artifact root, execute a command, contact an external
service, or mutate scheduler/GPU state. Its JSON output lists all six arms for
confirmatory seeds `101`, `202`, and `303`, including each exact command and
expected output path.

Operator flow is deliberately gated:

1. Build and inspect immutable split and materialization manifests locally.
2. Run the relevant gate against those artifacts.
3. Review the printed command, model revision, arm, seed, and output paths.
4. Obtain fresh approval for the exact GPU launch, hardware, duration, and
   expected spend before running that command separately.

For example, the excluded pilot training gate is:

```bash
./.venv/bin/python -m phase_marker.pipeline gate \
  --stage train \
  --kind pilot \
  --seeds 42 \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --hardware '1x A100 80GB' \
  --max-duration-hours 8 \
  --training-gpu-hours 4 \
  --selection-gpu-hours 1 \
  --behavior-gpu-hours 1 \
  --spend-cap-usd 25 \
  --estimated-spend-usd 18 \
  --workload-schema-version 1 \
  --training-jobs 6 \
  --checkpoint-selection-jobs 6 \
  --behavior-evaluation-jobs 1 \
  --manual-audit-rows 300 \
  --statistics-jobs 1 \
  --mechanism-jobs-excluded
```

That command only validates manifest hashes, lineage, counts, exclusions, and
completion evidence, then prints the six exact GPU commands as data. It never
launches them. A train, behavior, capture, or intervention gate fails closed
unless every approval field above is present and the estimates remain within
the declared duration and spend caps. Counts must exactly match the requested
pilot (6 training and 6 selection jobs) or confirmatory (18 and 18) commands;
training, selection, and behavior GPU-hours are separate and summed. Capture
and intervention require a later approval and are explicitly excluded here.
The dry run remains planning-only and
never reports approval readiness.

Behavior runs additionally require the immutable split manifest, exact example
file, and one validation-selected checkpoint manifest for every requested
seed/arm pair. The approval manifest includes a separate `behavior select`
workload for each adapter: it scores every training-manifest-declared
checkpoint on the validation split only, ranks strict exact-answer accuracy,
then teacher-forced mean gold-answer log probability, then earliest step. The
candidate set must match the declared checkpoints one-for-one. Each selection
atomically publishes a bundle containing `manifest.json` and `evidence.jsonl`.
Every evidence row binds the canonical example/gold answer, raw greedy
completion, replayable scorer input/output, ordered gold-continuation token
IDs, incrementally decoded pieces, and each teacher-forced token logprob under
the pinned tokenizer snapshot. Consumers rerun scoring and token accounting
before recomputing candidate aggregates and the winner. Production selection and behavior accept
only the canonical sibling `validation.jsonl` and `test.jsonl` files from a
fully recomputed split envelope. The selected PEFT adapter must name the frozen
Qwen base model and revision;
production behavior loads that base with vLLM LoRA enabled and supplies the
selected adapter as a `LoRARequest`.

The audit command reads the operator-owned TSV at
`audit/manual-labels.tsv` and writes its schema-v1 evidence envelope below
`audit/<pilot|confirmatory>/`. It requires exactly 300 unique behavior-bound
generation IDs: 100 each from GSM8K, SVAMP, and MATH. Analysis accepts an
explicit `--audit-manifest`; when omitted it resolves exactly one matching
sibling audit envelope and rejects ambiguous or plumbing-only evidence.

Synthetic production builds require
`artifacts/phase-marker/synthetic-preregistration.json`, a schema-v1 envelope
that fixes the seed, split counts, family balance, workspace conditions and
lengths, and protocol hash; its seed must match the requested run before a
command is emitted. Capture and intervention commands consume explicit
schema-v1 selection, batch/pair, checkpoint, and parent manifests; the
`tiny-fixture` backends require `--allow-test-backend` and emit
`evidence_scope=plumbing_only`, which production gates reject.

Capture and intervention cannot reuse the experiment approval above. They
require a separate schema-v1 mechanism approval with exactly one capture and
one intervention job, separate GPU-hour estimates, two commands/four outputs,
hardware, duration/spend bounds, and the bound selection/activation parent
hashes. Supplying only the mechanism-excluded experiment approval emits no
mechanism command.

Compact Markdown, TOML, JSON, and JSONL manifests/summaries remain eligible for
version control. Large phase-marker checkpoints and activation/raw-generation
tensor files stay outside git.

## Glyph Dictionary

The project uses a set of alchemical glyphs to denote different stages of reasoning:

| Glyph | Name | Meaning/Usage |
| :--- | :--- | :--- |
| **🜞** | *Crux* | **Guideline**: Sets the intention or core rule for the problem. |
| **🜆** | *Flux* | **Plan**: Outlines the approach or strategy. |
| **🜂** | *Ignis* | **Step**: Execution of the reasoning steps or calculation. |
| **🜃** | *Terra* | **Takeaway**: The final answer or conclusion. |

## License

Apache 2.0
