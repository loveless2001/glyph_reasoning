# Stage A Namespace Directory Compatibility Design

Date: 2026-08-05

## Purpose

Allow the Stage A run-namespace preflight to consume the directory entries
returned by Modal Volume's recursive listing without weakening its fail-closed
canonical-path policy.

This is a narrow compatibility repair. It does not change experiment arms,
commands, model revision, dependency evidence, approval semantics, GPU
resources, resume behavior, budget, or the mandatory stop before behavior.

## Root Cause

The CPU dependency preflight completed successfully, after which the local
coordinator inspected the run namespace. The real Modal recursive Volume
listing included the directory entry:

```text
/runs/<run-id>/receipts/smoke
```

The namespace validator permits descendants beginning with
`receipts/smoke/`, but not that root directory without the trailing slash. It
therefore rejected an approved namespace before applying tags or scheduling a
GPU.

The local Volume fake returns only file entries. Existing tests represented
the smoke receipt but omitted its parent directories, so they could not expose
the difference between the fake and Modal's listing contract.

## Chosen Behavior

Represent ignored mutable or evidence namespaces as exact roots rather than
trailing-slash prefixes. The namespace validator permits each exact root and
all of its descendants:

- `/runs/<run-id>/attempts`;
- `/runs/<run-id>/receipts/attempts`; and
- `/runs/<run-id>/receipts/smoke`.

Every other existing rule remains unchanged. The validator continues to
permit the run root, exact canonical producer/receipt/lock/provenance/summary
paths and their necessary ancestor directories. It rejects any unrecognized
file or directory, including siblings beneath `receipts` and unexpected
top-level run namespaces.

The implementation must compare normalized paths. It must not ignore entries
merely because Modal labels them as directories; doing so would hide
unapproved namespaces from the canonical-path gate.

## Data Flow and Failure Handling

1. The approved CPU dependency function validates bundle, cache, provenance,
   and smoke bytes.
2. The local coordinator recursively lists `/runs/<run-id>`.
3. Each listing path is normalized with one leading slash.
4. Exact approved ignored roots and their descendants are skipped.
5. Canonical Stage A paths and necessary ancestors are validated as before.
6. Any remaining entry raises `ValueError` before tags, recovery, or GPU work.

This change does not perform cleanup, mutate a volume, or reinterpret the
failed fresh launch as a resume. The failed launch created no Stage A lease,
training output, selection output, or summary.

## Testing

Add a real-style Volume-listing test that injects directory entries independently
of stored files. The test must prove that:

- each of the three exact approved roots is accepted;
- a descendant under each approved root is accepted;
- the existing validated smoke file remains accepted;
- an unexpected directory under `receipts` is rejected; and
- an unexpected top-level directory is rejected.

Retain the existing initial, resume, canonical-output, and evidence tests. Run
the complete adapter suite and full offline suite before review.

## Relaunch Boundary

Because `modal_phase_marker.py` changes, the source hash, plan digest, run ID,
smoke receipt, and Stage A action digest all change. After implementation and
review, derive a new CPU smoke command, obtain exact approval, execute and
independently validate its receipt, then derive and obtain separate approval
for the new Stage A command. No prior smoke or Stage A approval authorizes the
changed source.
