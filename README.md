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
  --estimated-gpu-hours 6 \
  --spend-cap-usd 25 \
  --estimated-spend-usd 18 \
  --evaluation-workload 'six pilot adapters plus frozen behavior evaluation'
```

That command only validates manifest hashes, lineage, counts, exclusions, and
completion evidence, then prints the six exact GPU commands as data. It never
launches them. A train, behavior, capture, or intervention gate fails closed
unless every approval field above is present and the estimates remain within
the declared duration and spend caps. The dry run remains planning-only and
never reports approval readiness.

Behavior runs additionally require the immutable split manifest, exact example
file, and one validation-selected checkpoint manifest for every requested
seed/arm pair. Synthetic production builds require
`artifacts/phase-marker/synthetic-preregistration.json`, a schema-v1 envelope
that fixes the seed, split counts, family balance, workspace conditions and
lengths, and protocol hash. Capture and intervention commands consume explicit
schema-v1 selection, batch/pair, checkpoint, and parent manifests; the
`tiny-fixture` backends require `--allow-test-backend` and emit
`evidence_scope=plumbing_only`, which production gates reject.

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
