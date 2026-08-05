# Research Test and Harness Cleanup

## Goal

Keep tests that protect scientific validity and remove the product-style safety
harness that obscured whether the experiment could actually run. The resulting
repository should make the experiment, scoring contract, and evidence lineage
easy to understand without thousands of lines of mocked infrastructure tests.

## Isolation

Cleanup is performed on `cleanup/research-test-suite` in a separate worktree.
The live Stage A run remains bound to the unchanged `main` source tree so an
exact recovery remains possible until that run finishes.

## Retained coverage

Retain focused tests for configuration, prompts, answer scoring, dataset
materialization and splits, training semantics, checkpoint selection, behavior
evaluation, statistics, activation capture, and interventions. These tests
protect experimental comparisons, leakage controls, lineage, and reported
metrics.

Retain only a compact Modal boundary covering:

1. deterministic plan and run identity;
2. nested checkpoint-tree publication with ordinary files and directories;
3. one fresh Stage A orchestration path;
4. one failure or resume path that preserves durable completed arms.

## Removed coverage and code

Delete repetitive field-by-field validation tests, exhaustive fake-Modal graph
tests, tests for implementation-private helper branches, duplicated rejection
cases, and tests whose only purpose is to enforce prior approval ceremony.
Remove production helpers and validation layers left unreferenced by the compact
suite and actual CLI workflow. Prefer standard library filesystem operations
and one clear validation boundary over custom wrappers for every operation.

Remove superseded one-off repair specs and plans. Keep one short current
experiment description and update the README to describe the actual research
workflow rather than the historical development process.

## Verification

Do not recreate a large regression gauntlet. Verification consists of Python
compilation, the retained scientific test modules, the compact Modal boundary,
and a dry construction of the current experiment plan. No network, GPU, model
download, or new external experiment action is part of cleanup verification.

## Success criteria

- Scientific scoring and experiment semantics remain unchanged.
- The live run and its recovery source remain untouched on `main`.
- Modal tests and support code are materially smaller and readable end to end.
- No removed helper remains imported or documented.
- Transient repair documentation and stale worktree references are absent from
  the cleanup branch.
