# Stage A Pilot Complete — Selection Results & Recovery Log

Date: 2026-08-08 | Run: `pilot-s42-...-src-173973c847e6-plan-a0412ddaa...` | Status: SUCCESS

## Validation results (checkpoint selection, seed-42 excluded pilot)

Checkpoint-100 selected in ALL six arms (step 134 worse everywhere — late-training degradation).

| Arm | acc@100 | acc@134 | mean gold logprob@100 |
|---|---|---|---|
| semantic | **0.3800** | 0.3550 | -31.770 |
| glyph | 0.3500 | 0.2050 | -34.142 |
| random | 0.3350 | 0.2083 | -40.327 |
| dot | 0.3333 | 0.3017 | -30.527 |
| direct | 0.2133 | 0.1200 | -20.271 |
| filler | 0.1700 | 0.0833 | -24.705 |

## Preliminary read (VALIDATION ONLY — not the preregistered test contrasts)

- Reasoning >> no reasoning: all four reasoning arms beat direct by +12–17 pts.
- Filler ≤ direct: no evidence extra meaningless tokens enable hidden computation.
- Glyph does NOT beat semantic (35.0 vs 38.0): markers add nothing over plain reasoning here.
- Glyph ≈ dot ≈ random (33–35): no phase-specific or identity-stability signal.
- Preregistered contrasts must be decided on the frozen TEST set (behavior stage, separate approval; `next_command` in stage-a-summary.json) and confirmatory seeds 101/202/303.

## Cost note

Selection at full 1024-token budget: 14.5h GPU (52,369s) vs 45 min training. Backend generates one prompt at a time (protocol); batching would cut this ~10x if the protocol is ever revised for confirmatory seeds.

## Bugs found & fixed during recovery (all committed on main)

1. Merge `79eb9fa`: codex cleanup branch (-13,595 lines test/harness bloat), all 6 plan tasks done, suites 272+9 green.
2. `37d02ee` vllm 0.13 removed `prompt_token_ids` kwarg → TokensPrompt; also caused orphaned EngineCore wedging containers to 4h timeout.
3. `9c1a69b` finalizer read GateResult fields by wrong names (crash after gate passed).
4. `076afc6` one-ulp fsum mismatch: builder summed 12 durations at once, validator fsums two subtotals → summary identity check failed. Proven by local clause-by-clause simulation with real receipts.
5. `30e9496` evaluation decoding capped at 64 tokens → all reasoning arms structurally 0.0 (training targets ~300–600 tokens). Raised to 1024.
6. Infra: Modal streaming reads of 323MB files corrupt silently over this WAN link (~14 corrected reads/pass). Worked around with double-read-verified volume client in untracked operator scripts (`scratchpad/orchestrate-full-stage-a-pipeline-after-source-change.py`); tracked source untouched to preserve run identity. Root cause unfixed upstream.

## Garbage disposition

- Local: 4 worktrees + branches removed (3 merged stage-a fixes, 1 merged cleanup).
- Volume `phase-marker-pilot-runs-v1`: 12 stale run namespaces deleted (superseded source identities incl. 3 fully-trained runs invalidated by source-bound fixes); only `src-173973c847e6` retained.
- Kept: `artifacts/`, `model_cards/`, `paper/` untracked on main (input bundle + unrelated work).

## Unresolved questions

- Held-out behavior run (test set) not yet launched — needs explicit go-ahead: ~$100+ GPU at current per-prompt throughput.
- Confirmatory seeds 101/202/303 pending pilot review.
- Sequential single-prompt generation makes selection/behavior ~10x costlier than necessary; protocol change would alter run identity.
- Modal WAN read corruption root cause (WSL2? modal 1.3.5?) unreported upstream.
