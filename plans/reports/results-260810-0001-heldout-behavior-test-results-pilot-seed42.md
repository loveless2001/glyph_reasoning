# Held-Out Behavior Results — Pilot Seed 42 (PRELIMINARY, pre-audit)

Date: 2026-08-10 | Run: `...src-173973c847e6` | Source: `records.jsonl` (2.3GB, 26 cells, ~166K generations), hash-verified in-datacenter against behavior receipt `36dd0e7e...`. Aggregation: Modal CPU (scratchpad `aggregate-behavior-contrasts-on-modal-cpu.py`).

## Primary matrix (greedy, full test set n=7319/cell, strict accuracy)

| arm \ prompt | neutral | glyph | dot | headings |
|---|---|---|---|---|
| semantic | .3462 | .3462 | .1782 | .2170 |
| glyph | .3143 | **.2943** | .1810 | .1555 |
| dot | **.4078** | .2797 | .3177 | .2715 |
| random | .2603 | .2754 | .2476 | .2045 |

Matched-format diagonal: semantic .3462, glyph .2943, dot .3177, random(neutral) .2603.

## Preregistered contrasts (matched format)

- glyph − semantic = **−5.2 pts** (markers hurt vs identical reasoning)
- dot − semantic = −2.9 pts
- glyph − dot = **−2.3 pts** (glyphs lose to neutral delimiters)
- filler vs direct: validation-stage result (filler .170 < direct .213) — no hidden-computation benefit

## Glyph perturbations (glyph arm, glyph prompt baseline .2943)

| perturbation | acc | delta |
|---|---|---|
| delete | .1481 | −14.6 |
| dot_replace | .1764 | −11.8 |
| unseen_replace | .1911 | −10.3 |
| cluster | .3093 | +1.5 |
| displace | .3215 | +2.7 |
| permute | .3084 | +1.4 |

**Identity-sensitive, position-insensitive** → markers act as a global format cue / learned dependency, not phase-specific computation.

## Sampled decoding (temp .7, 5 completions, n=36595)

glyph/neutral .3475, semantic/neutral .3323, glyph/glyph .3201, glyph/dot .2456. Ordering consistent with greedy.

## Conclusion (pilot)

Phase markers do NOT improve mathematical reasoning under this protocol. Glyph training induces reliance on marker identity without accuracy benefit; permuting/clustering markers costs nothing, so no evidence of phase-specific roles. Anomaly worth investigating: dot-trained + neutral prompt = .4078 (best cell overall).

## Status of frozen pipeline

- Behavior outputs + receipt published in run namespace (only namespace on volume).
- Manual audit gate NOT run: needs 300 human labels (100/dataset, sample frozen by seed 20260804) → then `statistics audit` → `statistics analyze` for CIs/tests.
- Confirmatory seeds 101/202/303 not run.

## Unresolved questions

- dot/neutral .4078 anomaly — real effect or scoring artifact? Check before confirmatory spend.
- Sampled cells ran full test set ×5 (spec said 250/dataset subset; statistics stage may subset via `sampled_test_hashes`).
- Manual audit labeling: user time required (~300 items).
- 2.3GB records not locally mirrored (WAN corruption); all analysis done in-datacenter.

## Addendum: untrained base-model matrix (exploratory, 2026-08-10)

Frozen protocol, no adapter, greedy 1024 tokens, n=7319/condition:
neutral .0579 | glyph .3966 | dot .4335 | headings .3376
By source, neutral: gsm8k .0197, math .0144, svamp .3260 (neutral collapse is largely format/parse noncompliance).

Implications:
1. Structure prompts help the BASE model massively (+28..38 pts) but dots > glyphs — the effect is generic structure elicitation, not glyph-specific.
2. SFT added nothing: best trained cell (dot-trained/neutral .4078) < base+dot prompt (.4335). Glyph SFT destroyed value: .2943 vs base+glyph .3966.
3. Combined with identity-sensitive/position-insensitive perturbations: trained glyphs act as a brittle mode key; prompting elicits, training narrows and degrades.
4. Legacy +24pt zero-shot glyph effect = real elicitation + last-number-extractor artifact (legacy raw completions were never saved; extractor was `re.findall(r"-?\d+\.?\d*")[-1]`).

Base-model rows conflate reasoning with format compliance for the neutral condition; manual audit would quantify parser disagreement there too.

## Addendum 2: mechanism resolved (exploratory, 2026-08-10)

Training loss: glyph arm fit BEST (final 0.0866 vs semantic 0.0971) while generalizing worst → surface imitation.
Tokenization: each glyph = 3 byte tokens sharing prefix (9284,250), differing only in final byte → explains position-insensitivity (perturbations preserving the token bag were harmless).
Mode-key sufficiency (glyph adapter, n=7319/cond): matched .2966; uniform ONE trained glyph .2720; single lone glyph .2403; unseen same-prefix glyph .2151; delete .1481.
→ Graded rare-token mode key. Distinct 4-glyph ensemble worth ≤2.5 pts; phase-specific structure ruled out.

FINAL SYNTHESIS: (a) prompt-side structure elicitation is real but generic (base+dot .4335 > base+glyph .3966 >> base neutral .0579); (b) SFT converts the free elicitation benefit into graded rare-token dependency while degrading capability (glyph-SFT .2943 < base+glyph prompt .3966); (c) legacy +24pt glyph effect = elicitation + last-number-extractor artifact. Phase markers decorate; they do not compute.

## Addendum 3: mechinterp results (exploratory, 2026-08-10)

LoRA weight space (6 adapters, on-volume): norms ~equal (~10); glyph~random subspace overlap 0.395 vs 0.26-0.29 all other pairs (shared rare-byte machinery, identity-stability irrelevant); deltas concentrate late (peak layers 23-26/28).

Steering (n=300 sample; refs local to sample): adapter+deleted 26.7% -> +layer-20 trained direction 38.7% (matched probe ref 42.0%). The trained mode key IS a single steerable late-layer residual direction; glyph tokens dispensable. Cross-transfer FAILS: trained direction on base 8.3% (<= base neutral 9.7%); base's own glyph-direction only 13.3% vs prompt effect ~+30pts.

CAUSAL CHAIN CLOSED: prompting elicits structure via distributed in-context processing (not one direction); SFT lossy-compresses that behavior into a low-dimensional switch keyed to rare byte tokens (layer ~20), degrading general capability. Scripts: scratchpad analyze-lora-adapter-deltas-across-arms.py, steering-vector-mode-key-experiment.py; raw JSON in scratchpad/lora-delta-results.json and task b00cefc51 output.

## Addendum 4: layer sweep + KV access (exploratory, 2026-08-10, n=100 probe)

KV-mask after prompt: .41 vs matched .42 — glyph effect fully absorbed into residuals at prompt time; generation never re-reads glyph positions.
Layer sweep (steer deleted prompts): flat ≈ deleted (.25-.29) through L12; rises L14-18 (.36-.49); peaks L22-26 (.53-.56) — EXCEEDING matched (.42) by +14. Causal locus = late layers, coinciding with LoRA delta peak (L23-26).
Mechanistic account, final: glyphs deposit a late-layer residual direction during prompt reading; all downstream behavior runs off the direction, not the tokens; the direction can be injected directly and over-driven past natural glyph performance. Adapter-specific (no base-model transfer).
Scripts: layer-sweep-and-kv-access-mode-key-experiment.py (scratchpad); raw JSON in task b3cd32762 output.
