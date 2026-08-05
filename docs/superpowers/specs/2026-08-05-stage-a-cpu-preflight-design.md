# Stage A CPU Preflight Design

Date: 2026-08-05

## Purpose

Replace the Stage A coordinator's local, full-byte model-cache validation with
an equivalent CPU-only Modal validation boundary. The change preserves the
rule that every staged dependency must be revalidated before any H100 or H200
task is requested, while preventing the local launcher from downloading and
materializing multi-gigabyte model shards.

This is a narrow repair to the approved Phase-Marker Modal Pilot design. It
does not change the experiment arms, commands, model revision, run identity
rules, concurrency, timeouts, spend envelope, receipt semantics, or the
mandatory stop before behavioral evaluation.

## Root Cause

The local `run-stage-a` entrypoint currently calls
`preflight_stage_a_dependencies`, which reads every model-cache file through
the Modal Volume batch API. `_read_volume_file_optional` joins the returned
chunks into one `bytes` value before hashing it. On the launch host, the Linux
kernel recorded an OOM kill of the `modal` process at approximately 6.14 GiB
resident memory while this path was running. The host has approximately
6.7 GiB total RAM.

The failure happened before a Modal task or GPU allocation. Replaying the same
coordinator cannot succeed reliably because the validation algorithm requires
one complete shard to fit in local memory.

The repository already has the required memory-safe implementation:
`validate_stage_a_remote_dependencies` validates the bundle, model-cache
manifest, every model-cache file, run provenance, and approved smoke receipt
from mounted paths. Its model-cache hashing reads files in 1 MiB chunks.

## Chosen Architecture

Add one CPU-only function to the existing Stage A Modal app. It mounts the
approved input and model volumes read-only and mounts the run volume for reads.
It receives only the canonical plan and the already validated Stage A approval
payload, calls `validate_stage_a_remote_dependencies`, and returns compact
dependency evidence.

The local coordinator performs the following sequence:

1. Rebuild and validate the immutable plan and explicit operator approval.
2. Validate the Stage A action digest and its named smoke and model-cache IDs.
3. Invoke the CPU preflight function.
4. Validate the returned evidence schema and require exact equality with the
   plan and approval identities.
5. Inspect the compact existing Stage A output and receipt namespaces.
6. Print and flush the complete execution plan.
7. Apply approved app tags.
8. Invoke recovery if explicitly resuming, then schedule only missing GPU jobs.

The CPU preflight call is the only new remote side effect. It does not mutate a
volume, create a lease, write a receipt, load model weights, or request a GPU.
Any validation error aborts before app tags or a GPU map call.

## Remote Function Contract

The new function uses the existing CPU image with no `gpu` declaration,
`retries=0`, a bounded timeout, and enough CPU memory for imports and small
metadata objects. Model files are never accumulated in memory; their hashes
are computed incrementally.

Its input contains exactly:

- `plan`: the canonical pilot plan payload;
- `approval`: the canonical `run-stage-a` approval payload.

Its output contains exactly:

- schema version;
- run ID;
- bundle ID;
- bundle-manifest artifact ID;
- model-cache artifact ID;
- smoke-receipt artifact ID; and
- the validated smoke receipt.

The local coordinator treats this result as untrusted external data. It
rejects missing or extra fields, invalid artifact identities, a mismatched run
or bundle, and any smoke receipt that does not reproduce the approved identity.

## Local Boundary

`run_stage_a_local` receives a CPU dependency-preflight callable instead of
input and model Volume batch clients. It retains the run Volume client because
existing Stage A receipts and summaries are compact and must still be checked
between fan-out stages.

The old local dependency validator is removed from the Stage A launch path.
No fallback may re-download model-cache files locally. A missing CPU callable
or malformed result is a hard failure.

## Failure Handling

- A CPU preflight exception propagates and schedules no GPU work.
- A malformed or identity-mismatched preflight result fails closed.
- A preflight timeout is not retried automatically; a new operator launch is
  required.
- Existing resume rules remain unchanged: canonical output is never
  overwritten, and only absent validated arms may be scheduled.
- The failed prior launches created only the approved smoke and input
  provenance objects, so the next launch remains an initial Stage A launch,
  not a resume launch.

## Testing

Tests must establish all of the following before implementation is accepted:

- the preflight function is CPU-only and uses read-only input/model mounts;
- the wrapper forwards the exact plan and approval to the existing mounted-path
  validator;
- the wrapper returns the compact, exact evidence schema;
- the local coordinator invokes CPU preflight before tags or GPU functions;
- the local coordinator does not read model files through a local Volume client;
- exceptions or malformed evidence abort before tags and GPU scheduling;
- tampered bundle, model-cache, smoke, or run identities are rejected;
- completed and resume flows retain their existing behavior; and
- the full offline test suite passes.

## Relaunch Protocol

This source change creates a new source hash, plan digest, run ID, and approval
digest. The existing smoke receipt therefore cannot authorize the changed
launcher. After tests and review:

1. derive and present the new identities;
2. obtain approval for the exact CPU smoke command;
3. execute and independently validate the new smoke receipt;
4. derive and present the exact Stage A command and action digest;
5. obtain fresh Stage A approval; and
6. run the exact command in a persistent session while monitoring the CPU
   preflight and subsequent GPU tasks.

The experiment still stops after the Stage A summary for artifact review.
