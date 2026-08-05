# Stage A Modal `listdir` Not-Found Compatibility Design

Date: 2026-08-06

## Purpose

Allow fresh Stage A output preflight to treat an absent canonical producer
directory as empty when the installed Modal client reports the absence with
`modal.exception.NotFoundError`.

This is a narrow client-contract compatibility repair. It does not change the
experiment arms, commands, model revision, dependency evidence, approval
semantics, GPU resources, resume behavior, spend limits, or the mandatory stop
before behavior.

## Observed Failure and Root Cause

The approved fresh Stage A launch completed the CPU dependency preflight and
then inspected canonical training and selection outputs before tags or GPU
work. Its first absent producer directory caused `Volume.listdir()` to raise:

```text
modal.exception.NotFoundError: No such file or directory
```

The local helper `_list_volume_files_optional` currently converts only Python
`FileNotFoundError` to an empty listing. The installed Modal 1.3.5 source shows
that `Volume.listdir()` propagates `modal.exception.NotFoundError`, whereas
`Volume.read_file()` explicitly converts the same backend error to Python
`FileNotFoundError`. The read helper is therefore already compatible; only the
list helper has the wrong exception boundary.

The launch stopped inside `_preflight_stage_a_outputs`, before the plan was
printed, application tags were applied, or any training/selection remote call
was made. The failed launch created no Stage A lease, producer, canonical
receipt, selection output, or terminal summary.

## Chosen Behavior

Change `_list_volume_files_optional` to return an empty tuple for exactly these
two exceptions:

- `FileNotFoundError`, used by local/test clients; and
- `modal.exception.NotFoundError`, used by real Modal `Volume.listdir()` for an
  absent path.

All other exceptions continue to propagate. In particular, permission,
authentication, conflict, data-loss, transport, and service errors must not be
reinterpreted as an empty namespace.

Leave `_read_volume_file_optional` unchanged. Broad exception normalization or
a shared volume-error abstraction is unnecessary for the observed contract and
would expand the fail-open surface.

## Data Flow and Failure Ordering

1. The approved CPU dependency function revalidates bundle, model-cache,
   approval, provenance, and smoke bytes.
2. The local coordinator validates the run namespace.
3. For each canonical training and selection arm, it reads the optional
   receipt, lists the optional producer directory, and reads the optional
   promotion lease.
4. An absent producer reported by either supported not-found exception becomes
   an empty listing.
5. Existing fresh/resume, orphan, receipt, lease, semantic, and canonical-path
   rules run unchanged.
6. Any non-not-found listing error aborts before tags or GPU calls.

This repair does not perform cleanup or mutate a Modal Volume.

## Testing

Add a test-only Stage A runs client that preserves normal stored-file listings
but raises `FakeModalNotFoundError` when a requested producer path has no
entries. Use it in a fresh Stage A integration regression with the existing
fake dependency, training, selection, and finalizer functions.

The regression must fail before the production change at the first absent
producer listing with `FakeModalNotFoundError`. After the change it must prove
that:

- CPU dependency preflight completes before tags and training calls;
- all six fake training and six fake selection arms complete;
- the fake finalizer returns the mandatory stopped-before-behavior summary; and
- the missing producer paths are treated as empty.

Add a separate focused helper regression whose listing client raises a
non-not-found error such as `PermissionError`. Assert that the exact exception
still propagates, protecting the fail-closed boundary from a future broad
catch.

Retain the existing namespace-directory, initial, resume, orphan-recovery,
canonical-output, dependency-evidence, and failure-ordering tests. Run the
focused regression, complete adapter suite, and full maintained offline suite
before review.

## Relaunch Boundary

Changing `modal_phase_marker.py` changes the source hash, plan digest, run ID,
smoke receipt, and Stage A action digest. The successful smoke receipt and
approved Stage A command for source `add344975af3...` become invalid for the
new source.

After implementation and independent review:

1. derive the new identities locally;
2. obtain exact approval for the new CPU smoke command;
3. execute and independently validate the downloaded smoke receipt;
4. derive a new fresh Stage A action using the reviewed model-cache artifact;
5. obtain separate approval for that exact GPU command; and
6. execute in fresh mode, not `--resume`.

No prior smoke or Stage A approval authorizes the changed source. Mechanism and
behavior execution remain outside Stage A approval.
