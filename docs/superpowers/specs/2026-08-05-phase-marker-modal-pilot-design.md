# Phase-Marker Modal Pilot Design

Date: 2026-08-05

## Purpose

Build a dedicated Modal launcher for the excluded seed-42 phase-marker pilot.
The launcher runs the six approved training arms and their six validation-only
checkpoint selections, verifies and retrieves compact evidence, and then
stops. It does not start behavioral evaluation, confirmatory seeds, synthetic
mechanism work, activation capture, or interventions.

The launcher is an execution adapter for the repository's existing immutable
pipeline. It does not define a second experiment protocol. The exact commands,
model revision, inputs, expected outputs, workload counts, and approval
metadata continue to come from `phase_marker.pipeline`.

## Scope

### Included

- Pilot seed `42` only.
- Six training arms in frozen order: `semantic`, `glyph`, `dot`, `random`,
  `direct`, and `filler`.
- One single-GPU training job per arm.
- One validation-only checkpoint-selection job per arm after every training
  job has completed and passed validation.
- Exact Qwen base revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Content-addressed input staging, model caching, output publication, receipts,
  status inspection, and compact evidence retrieval.
- Local dry runs and tests that perform no remote writes or compute.

### Excluded

- The behavior matrix and its approximately 307,000 completions per seed.
- Confirmatory seeds `101`, `202`, and `303`.
- Synthetic generation, activation capture, and causal interventions.
- Automatic publication to GitHub, Hugging Face, or another registry.
- Automatic retry of failed experiment commands.
- Modification or reuse of the legacy `modal_app.py` workflow.

Each excluded stage requires a later artifact review and fresh authorization.
Mechanism stages retain their separate approval type and cannot reuse the pilot
training approval.

## Chosen Approach

Add a dedicated thin Modal application, `modal_phase_marker.py`, backed by a
pure, locally testable planning module such as `phase_marker/modal_plan.py`.
Leave `modal_app.py` unchanged.

This is preferred over rewriting the legacy launcher because the old launcher
uses a mutable `/vol/work` copy that can silently retain stale code and data.
It is preferred over a monolithic remote job because per-arm jobs provide
clear lineage, bounded failure domains, and a mandatory inspection point before
the expensive behavior stage.

## Immutable Identities

The local planner derives a pilot identity from:

- run kind `pilot`;
- seed `42`;
- the experiment configuration hash;
- split artifact ID;
- all six materialization artifact IDs;
- the pinned Qwen revision;
- the included source-tree hash; and
- the locked Modal image dependency hash.

The human-readable run ID has the form:

```text
pilot-s42-cfg-<config8>-split-<split8>-src-<source12>
```

The full hashes remain in the bundle and run manifests. Truncated hashes are
display labels only and are never used as the sole integrity check.

Tracked source changes must be clean before staging or launch. Untracked files
are excluded from the image except for an explicit input allowlist containing
the approved configuration, split bundle, and six materialized datasets. The
launcher never copies local credentials, `.git`, checkpoints, raw generations,
or arbitrary files from `artifacts/` into the image.

## Components

### Pure local planner

`phase_marker/modal_plan.py` owns typed, deterministic values and validation:

- the run ID and source-tree hash;
- the exact six training and six selection commands;
- expected input and output paths;
- model and tokenizer revision;
- job counts, timeouts, concurrency, GPU-hour estimates, and spend metadata;
- content hashes for the staged bundle; and
- the rule that behavior and mechanism commands are absent.

It consumes the approval-ready output of `phase_marker.pipeline commands`. It
fails unless that output contains exactly six pilot jobs in frozen arm order,
all at seed `42`, with complete approval metadata and the pinned revision.

The planner has no Modal import and no network, filesystem-write, or subprocess
side effects. Its dry run prints canonical JSON only.

### Thin Modal adapter

`modal_phase_marker.py` defines one dedicated app, tagged for cost attribution,
and exposes explicit operator entrypoints:

- `plan`: local and read-only; print the exact workload and hashes.
- `stage-inputs`: upload only the approved input bundle.
- `cache-model`: download and verify the pinned Qwen snapshot on CPU.
- `run-stage-a`: run six training jobs, validate them, run six selections,
  validate them, publish a compact summary, and stop.
- `status`: inspect existing receipts and canonical outputs without mutation.
- `download-evidence`: retrieve compact manifests, receipts, logs, and selection
  evidence; leave large checkpoints on Modal unless explicitly requested.

Implementation may use Modal's Python SDK behind these entrypoints, but every
mutating or compute entrypoint must print its resolved run ID, exact operations,
hardware, timeouts, and spend envelope before acting.

### Container image

Use one pinned CUDA-capable image compatible with H100 and H200 for training
and vLLM checkpoint selection. The image contains:

- the exact included source tree at `/opt/glyph_reasoning`;
- a `.venv/bin/python` executable path so emitted repository commands run
  without translation;
- a fully locked dependency set for PyTorch, Transformers, PEFT, Datasets,
  Accelerate, vLLM, Modal, and their relevant transitive runtime packages; and
- no model weights, experiment inputs, outputs, or credentials.

Floating requirements are forbidden for launch. Dependency versions become
approved image inputs only after the image-build and CPU import smoke tests pass.
The lockfile hash is part of the run identity and every receipt.

### Persistent storage

Use three dedicated named volumes:

1. `phase-marker-pilot-inputs-v1`: immutable staged input bundles.
2. `phase-marker-pilot-model-cache-v1`: the pinned Qwen snapshot and its
   verification manifest.
3. `phase-marker-pilot-runs-v1`: attempt outputs, canonical outputs, receipts,
   logs, and compact run summaries.

GPU jobs mount input and model volumes read-only. They write only to the run
volume. No launcher path uses the legacy `glyph-reasoning-vol` or `/vol/work`.

Modal Volumes require explicit visibility coordination. Producers commit after
a validated publish; consumers reload before reading a producer's new output.
Concurrent jobs write disjoint arm-specific subtrees, and the design permits at
most two concurrent commits.

## Input Bundle

The upload allowlist is:

- `configs/phase-marker-qwen25-7b.toml`;
- `artifacts/phase-marker/splits/manifest.json`;
- `artifacts/phase-marker/splits/train.jsonl`;
- `artifacts/phase-marker/splits/validation.jsonl`;
- `artifacts/phase-marker/splits/test.jsonl`;
- `artifacts/phase-marker/splits/exclusions.jsonl`;
- all six `artifacts/phase-marker/training-data/<arm>.jsonl` files; and
- all six sibling materialization manifests.

The bundle manifest records the relative path, byte size, and SHA-256 of every
file plus all repository artifact IDs. Staging fails if a file is missing,
unexpected, changed during upload, or already exists under the same bundle ID
with different bytes. Re-staging byte-identical content is a no-op.

The input volume is the source of truth. For each experiment command, the job
copies the required read-only inputs into an isolated ephemeral workspace that
recreates the repository-relative `artifacts/phase-marker` layout. This lets
the repository execute its exact approved command without granting it write
access to the input bundle.

## Model Cache

The CPU `cache-model` entrypoint resolves Qwen at the exact revision and writes
to a revision-addressed temporary directory. Before promotion it verifies:

- the snapshot path resolves to the pinned revision;
- tokenizer assets pass the repository's strict Qwen JSON-BPE validator;
- the model index is valid JSON;
- every shard named by the index exists and is nonempty;
- configuration and generation metadata are present; and
- recorded hashes and sizes reproduce from the cached files.

It then writes a model-cache manifest, atomically promotes the temporary
snapshot into its canonical revision path, commits the volume, and returns the
manifest hash. GPU jobs refuse any cache without this exact manifest and mount
the cache read-only. Model download is therefore not billed as H100 time.

## Stage A Execution

### Preflight

Before remote GPU invocation, the local entrypoint repeats the repository's
approval-ready pilot command generation using:

- hardware: one Modal H100, allowing Modal's automatic H200 upgrade;
- maximum declared experiment envelope: 120 GPU-hours;
- training: 24 GPU-hours;
- selection: 24 GPU-hours;
- later behavior: 72 GPU-hours;
- estimated full-pilot spend: USD 600;
- spend ceiling: USD 1,000;
- six training jobs;
- six selection jobs;
- one later behavior job;
- 300 manual-audit rows;
- one statistics job; and
- mechanism jobs explicitly excluded.

Preflight also verifies the staged input bundle, model-cache manifest, source
and lock hashes, exact empty canonical output paths, and the environment-budget
operator acknowledgement. It performs no cleanup or overwrite.

### Training fan-out

Run one `gpu="H100"` function per arm with:

- exactly one visible GPU;
- at most two active training containers;
- a four-hour timeout per arm;
- no configured application retry; and
- the exact command emitted by `phase_marker.pipeline`.

Each invocation constructs a fresh ephemeral workspace, copies its approved
inputs, points `HF_HUB_CACHE` at the verified read-only model cache, and checks
that the GPU is BF16-capable. The command writes only to the ephemeral attempt.

After exit zero, the job executes the existing training artifact validators.
Only then does it copy the complete arm directory to a unique temporary attempt
path on the run volume, write an execution receipt, and atomically promote it
to the canonical arm path if that path is absent. It commits before returning.

The coordinator does not begin checkpoint selection unless all six canonical
training outputs exist and independently pass validation.

### Selection fan-out

After reloading the run volume, run one H100 selection job per arm, again with
at most two active containers, a four-hour timeout, and no application retry.
Each job copies its approved validation split, materialized data, and canonical
training output into a new ephemeral workspace and executes the exact emitted
vLLM selection command.

Successful selection must publish both `manifest.json` and `evidence.jsonl`
and pass the repository's replay, scoring, tokenizer, checkpoint, and lineage
validators before promotion and commit.

### Mandatory stop

After all six selections, a CPU coordinator reloads the run volume, runs the
behavior-stage prerequisite gate in read-only mode, and writes a compact Stage A
summary containing checked hashes, receipts, elapsed GPU time, and the later
behavior command as inert data. It does not invoke that command.

The process exits after printing where to download the compact evidence. There
is no automatic continuation into behavior, confirmatory, synthetic, capture,
or intervention stages.

## Receipts and Provenance

Every attempt receipt is canonical JSON and includes:

- schema version, run ID, bundle ID, stage, arm, seed, and attempt ID;
- exact command and its hash;
- source-tree and dependency-lock hashes;
- input artifact IDs and file hashes;
- model-cache manifest and pinned revision;
- Modal app/function version and invocation identifier when available;
- requested GPU, observed GPU model, CUDA/runtime package versions;
- start/end timestamps, elapsed seconds, exit status, and timeout;
- expected output paths and recomputed output hashes; and
- promotion status and failure reason.

A receipt certifies the observed execution and artifact bytes. It does not by
itself certify scientific correctness; repository gates remain authoritative.

## Failure and Resume Semantics

- Canonical outputs are immutable and never overwritten.
- Every invocation writes to a unique attempt path first.
- Nonzero exit, timeout, exception, failed validator, or incomplete receipt
  leaves the attempt quarantined and unpromoted.
- A rescheduled or manually repeated invocation that sees a partial attempt
  does not reuse it. It creates a new attempt ID.
- No experiment command receives automatic application retries. Modal may
  reschedule crashed containers according to platform behavior, but canonical
  promotion remains guarded by the same immutable validation.
- One arm's failure prevents the next stage from starting for every arm.
- Resume plans only missing canonical arms after revalidating every existing
  parent and require a new operator-visible launch command. Resume never deletes
  failed attempts automatically.
- Volume reloads occur only with no open files. Arm-specific writes prevent
  last-writer-wins collisions.

## Resource and Spend Controls

Modal currently lists H100 at USD 0.001097 per second, or USD 3.9492 per hour,
and may automatically satisfy `gpu="H100"` with an H200 at the H100 price. The
conservative Stage A envelope is:

- six training jobs × four hours = 24 H100-hours;
- six selection jobs × four hours = 24 H100-hours;
- Stage A maximum = 48 H100-hours, approximately USD 189.56 GPU-only; and
- declared Stage A estimate = USD 250 including CPU, memory, and storage
  overhead.

The full pilot retains the approved USD 600 estimate and USD 1,000 ceiling,
including the separately authorized 72-hour behavior envelope. A Modal
environment-level USD 1,000 budget must be configured and acknowledged before
the first remote write. The application records estimates and rejects workload
drift, but it does not claim that local metadata is an account-level hard stop.

The app is tagged with experiment, run-kind, seed, and run ID so billing reports
can attribute spend. Non-preemptible execution and region pinning are excluded
because they materially increase the listed base price.

References:

- <https://modal.com/pricing>
- <https://modal.com/docs/guide/gpu>
- <https://modal.com/docs/guide/billing>
- <https://modal.com/docs/guide/volumes>
- <https://modal.com/docs/guide/retries>

## Security and External State

- Local Modal credentials and `.modal.toml` are never included in the image or
  uploaded bundle.
- The public Qwen model requires no embedded token. If rate limits later require
  authentication, use a named Modal Secret and never record its value.
- Local dry runs and tests do not contact Modal or Hugging Face.
- `stage-inputs`, `cache-model`, the CPU remote smoke, and the H100 launch are
  distinct external state changes. The final handoff must list their exact
  commands and request fresh authorization before executing them.
- No Git push, PR, dataset publication, checkpoint upload, or behavior launch is
  implied by approving or implementing this launcher.

## Testing Strategy

### Pure unit tests

Test the Modal-independent planner for:

- exact six-arm order and seed `42`;
- exact commands and expected outputs from the repository gate;
- pinned model revision and full input allowlist;
- deterministic run and bundle IDs;
- rejection of dirty tracked source, path traversal, unexpected files, changed
  hashes, duplicated arms, workload drift, missing approval fields, or behavior
  and mechanism commands;
- 24/24/72 GPU-hour allocation, four-hour per-job timeouts, two-container
  concurrency, USD 600 estimate, and USD 1,000 ceiling; and
- canonical receipt hashing and promotion preconditions.

### Local filesystem integration tests

Using temporary directories and fake subprocesses, test:

- input bundle creation and byte-for-byte restaging;
- ephemeral workspace construction with repository-relative paths;
- exact command execution without argument translation;
- a failed attempt remaining quarantined;
- successful validation followed by one-time atomic promotion;
- refusal to overwrite canonical output;
- volume-style reload/commit sequencing through an adapter double;
- all-training-success as a prerequisite for any selection; and
- the mandatory stop with behavior represented only as inert command data.

### Modal adapter tests

Import the thin adapter with Modal boundaries replaced by test doubles and
verify function resource declarations, named volumes, read-only mounts where
applicable, timeouts, zero application retries, maximum concurrency, tags, and
entrypoint routing.

### Remote smoke and launch gates

After implementation and local review, but before GPU launch:

1. Obtain fresh authorization for the exact input upload and CPU model-cache
   commands.
2. Run a CPU-only Modal smoke that verifies image imports, source hash, staged
   inputs, and model-cache manifest without loading model weights on a GPU.
3. Download and review the smoke receipt.
4. Generate the exact Stage A GPU command, hardware, duration, and spend
   envelope again.
5. Obtain fresh authorization for that command.

No remote operation is part of ordinary unit or integration tests.

## Acceptance Criteria

The launcher implementation is ready for a GPU approval request only when:

1. the maintained offline repository suite passes;
2. all new planner, filesystem, and adapter tests pass;
3. local dry run reports exactly six training and six selection jobs for seed
   `42`, with no behavior or mechanism execution;
4. the source, dependency, split, and six materialization hashes match the
   reviewed local artifacts;
5. the input and model-cache paths are content addressed and canonical output
   paths are absent;
6. the CPU-only Modal smoke succeeds and its receipt reproduces locally;
7. the Modal environment USD 1,000 budget is acknowledged; and
8. the user receives and approves the exact H100 Stage A launch command.

Implementation completion does not authorize criteria 6 or 8 to execute. Those
remain separate external actions.
