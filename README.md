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

### Modal seed-42 Stage A pilot (approval-gated)

The dedicated `modal_phase_marker.py` app prepares one content-addressed,
seed-42 pilot across the frozen arm order `semantic`, `glyph`, `dot`, `random`,
`direct`, and `filler`. It runs training and validation-only checkpoint
selection, then must stop before behavior evaluation. The workflow assumes one
operator/coordinator: do not run staging, resume, or Stage A concurrently from
multiple shells or people.

The following two commands are local planning operations only. They do not
import Modal, call a remote service, write a volume, load a model, or allocate a
GPU. Freeze and inspect their full content-bound identity before considering any
external action:

```bash
# Local only: derive the full content-bound identity without Modal or network.
PHASE_MARKER_RUN_ID="$(./.venv/bin/python -m phase_marker.modal_plan run-id \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt)"

# Local only: print the canonical plan and inert action manifest.
./.venv/bin/python -m phase_marker.modal_plan plan \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt
```

The plan's `action_manifest.external_actions` contains exactly three inert,
digest-bound command strings: `stage_inputs`, `cache_model`, and `smoke`. It
deliberately contains no Stage A command. Every `modal run` crosses an external
boundary and is not an offline dry run. Obtain fresh authorization for each
exact string, execute them one at a time, and inspect the result before moving
to the next boundary. Do not reconstruct shorter commands by dropping the full
plan or action digest flags.

The frozen Modal environment is `main`, and every generated command includes
`--env main`. Before executing an authorized command, independently confirm
that the active Modal profile and workspace are the reviewed account; this
repository does not claim or discover the live account configuration.

After cache population and CPU smoke, inspect the complete model-cache manifest
and the exact successful smoke receipt. Only then use their reviewed 64-character
artifact IDs to derive a prospective Stage A action locally:

```bash
# Local only: prints one inert, evidence-bound fresh Stage A action; does not
# import Modal, contact a service, write a volume, load a model, or allocate GPU.
./.venv/bin/python -m phase_marker.modal_plan stage-a-action \
  --repo-root . \
  --config configs/phase-marker-qwen25-7b.toml \
  --artifact-root artifacts/phase-marker \
  --dependency-lock requirements-modal-phase-marker.txt \
  --smoke-receipt-artifact-id '<REVIEWED_SMOKE_RECEIPT_ARTIFACT_ID>' \
  --model-cache-artifact-id '<REVIEWED_MODEL_CACHE_ARTIFACT_ID>' \
  --fresh
```

The output contains the exact H100 command as inert data. Review its run ID,
full plan and action digests, dependency IDs, fresh/resume mode, resources, and
spend envelope, then request a distinct authorization for that exact command.
For crash recovery, first inspect the durable status and quarantine plan, then
rerun the local planner with `--resume` instead of `--fresh`; the resulting
approval digest and command are intentionally different.

After an authorized action, inspection remains separate:

```bash

# Read-only remote inspection after an authorized run.
modal run --env main modal_phase_marker_inspect.py::status --run-id "$PHASE_MARKER_RUN_ID"

# Explicit local write of compact allowlisted evidence after successful status.
modal run --env main modal_phase_marker_inspect.py::download-evidence --run-id "$PHASE_MARKER_RUN_ID" --destination phase-marker-stage-a-evidence
```

Stage A requests one H100 per job, permits Modal's automatic H200-compatible
upgrade, allows at most two concurrent GPU containers, uses four-hour job
timeouts, and has zero application retries. Its worst-case allocation is 48
H100-hours; the approval estimate is USD 250 (approximately USD 189.56 at USD
3.9492/hour). The full pilot estimate is USD 600 with a USD 1,000
operator-acknowledged ceiling. That acknowledgement records the environment
budget; it is not an application-enforced account spending limit.

After the Stage A summary reports `stopped_before_behavior=true`, stop. Behavior
evaluation requires a separate approval after the downloaded receipts,
producer manifests, selection evidence, and summary have been inspected.
Activation capture or intervention is mechanism work and requires its own later,
stage-specific approval; neither is authorized by the Stage A commands above.
No live Modal, model-cache, H100, billing, or scheduler integration is claimed
by the offline implementation and tests in this repository.

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
completion evidence, then prints the six exact GPU commands as data. Selection
gates may load the pinned Qwen tokenizer from the local cache with
`local_files_only=True` solely to retokenize evidence; they never load model
weights, execute commands, use the network, or write artifacts. The command never
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
the pinned tokenizer snapshot. Consumers independently retokenize every exact
gold continuation, reproduce every incremental decoded piece, rerun scoring and
token accounting, and only then recompute candidate aggregates and the winner.
Every selection manifest carries
`origin_verification=execution_receipt_or_rerun_required`; its content hashes
certify internal replay and lineage, not that raw completions originated from a
model. Task 13 must bind production execution receipts/scheduler logs or a
deterministic rerun before confirmatory publication. Until then the pipeline
never labels selection output as origin-verified. Production selection and behavior accept
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
require separate, stage-specific schema-v1 mechanism approvals requested at
different times. Each approval names exactly one `capture` or `intervene`
stage, one job, one command, two exact output paths, one digest over all checked
stage inputs, and that stage's hardware, GPU-hour, duration, and spend bounds.
A capture approval binds the checked selection, synthetic, behavior,
checkpoint, and tokenized-input lineage and emits only the capture command. An
intervention approval may be requested only after the actual activation
artifact exists; it binds that activation plus the checked selection,
checkpoint, and aligned-pair lineage and emits only the intervention command.
The CLI fields are `--mechanism-stage`, `--mechanism-job-count 1`,
`--mechanism-command-count 1`, `--mechanism-expected-outputs <manifest>
<payload>`, `--mechanism-parent-hash`, `--mechanism-gpu-hours`, and the
hardware/duration/spend fields. Cross-stage approvals or mismatched parents or
outputs fail closed. Supplying only the mechanism-excluded experiment approval
emits no mechanism command.

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
