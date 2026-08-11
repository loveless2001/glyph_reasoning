# Research Thread Synthesis & Future Directions

Date: 2026-08-12 | Purpose: resumable record of post-experiment discussion (2026-08-10..12)

## Where things stand

- **Glyph paper** (`paper/phase-marker-findings.tex`, compiled PDF): controlled 6-arm study + full mechanism. Verdict: glyphs decorate, don't compute; SFT compresses prompt elicitation into a brittle late-layer (L22-26) rare-token mode key; steering the direction beats real glyphs by +14. Pilot seed only; manual audit + confirmatory seeds not run. 121 commits unpushed.
- **Nemotron paper** (`~/projects/nemotron_glyph_reasoning/plans/reports/technical-report-260404-sft-format-ablation.tex`): in MDPI round-2 review. Behavioral twin at 30B: supervision format shapes strategy; formula notation → cross-category contamination; compact neutral tags win.
- The two form a research line: Nemotron = behavior at scale, glyph = mechanism. Cross-citation paragraphs drafted-in-principle, not yet written (user deferred).

## Conceptual synthesis from discussion

1. **Tokens help via content, not count.** Filler < direct (trained, matched, preregistered) kills "extra tokens = thinking space." Structure prompts = behavior selection (same compute, different program), not extra scratch space.
2. **Marker identity irrelevant zero-shot; boring beats exotic.** Base model: dot .434 > glyph .397 > headings .338 >> neutral .058. Words-as-markers worst (semantic leakage). Best proposed marker: content-impossible sigil + index (`§1 §2 §3`) — collision-free + cheap tokens + free ordinal signal for recursion depth. `x1 x2 x3` = worst in math (collides with object language).
3. **Glyphs' one real virtue: zero collision with content** (never appear in problem text). Untested advantage in messy/recursive contexts. User's 2025 frontier-model anecdotes (ChatGPT/Claude/Grok solving recursive problems better with glyph prompts) = structure elicitation + possible collision advantage; lacked dot control; unblinded.
4. **Glyph-mysticism era of frontier models** explained: sci-fi trope prior + training-data feedback loop of viral AI-glyph content + attractor states in self-referential chats (documented "spiritual bliss" attractor) + compression illusion (glyphs are 3 tokens — LESS efficient).
5. **ICL = rank-1 finetune (Dherin et al. 2025)**: true as per-token input-dependent dual form, misleading as static claim. Our data: static direction captured +3.7 of ~+30 prompt effect on base (ICL is high-rank) but nearly all of SFT'd behavior (the finetune WAS the compressor). LoRA rank-16 bottleneck may have amplified compression — stated limitation.
6. **SFT format recipe**: prose reasoning + one conventional answer line; common tokens only; non-content-like per domain; matched train/inference templates with instruction kept in prompt; mask/monitor content-token loss (format tokens flatter loss — glyph arm best loss, worst generalization); sub-epoch early stop (ckpt-100 > 134 in all arms); measure base+prompt before training at all (base+dot .434 beat every trained arm).
7. **When binding is legitimate**: closed-world/competition specialization (Nemotron winner's code-speak); learned DSL = semantic compression ≠ decoration; RL-selected form ≠ SFT-imitated form; harness-guaranteed triggers (think-tags, tool tokens, chat templates) fine by design. Rule: **bind only to tokens you control at inference, knowingly** — loss curves won't warn you.
8. **Small agentic models (LiquidAI etc.)** = industrial format binding: synthetic formatted traces, heavy SFT, harness-owned triggers. Legitimate in-harness; symptoms predicted & observed: over-triggering, schema brittleness, benchmark scores overstating capability. Mitigations in better ones: schema randomization (dilution), execution-feedback RL, constrained decoding (de-binding via sampler).

## Benchmark survey (researcher report: `researcher-260812-0027-agentic-benchmarks-robustness-survey.md`)

29 public benchmarks vs 6 off-diagonal dimensions: multi-turn 12/29 (mature), error recovery 8/29, schema perturbation 5/29 (all May-Jun 2026 newcomers: RobustBench-TC, ToolMaze, AgentNoiseBench), unseen tools 4/29, abstention 3/29 (best model 59.5% — over-trigger prediction confirmed), **cross-harness transfer ~0 (no public standard — biggest gap)**. Mainstream boards (BFCL/WebArena/GAIA) diagonal-only and saturating (~76.7% frontier cluster). Small-vs-large gap 7-21 pts on-diagonal.

## Open research directions (ranked)

1. **"Third paper": format binding in agentic models.** Matched-vs-perturbed gap, stratified by model size, using RobustBench-TC + AgentEscapeBench + custom cross-harness suite (novel — nobody has published cross-harness transfer). Prediction: small models cliff, large degrade gracefully. Headline question: what fraction of leaderboard gains are format coverage vs reasoning?
2. **Frontier recursion test** (afternoon, ~$few API): glyph vs dot vs `x1x2x3` vs `§1§2§3` on recursive problems, blinded, ~50 problems × 4 conditions × 3 providers. Tests collision-freedom hypothesis. Prediction: §-indexed ≥ glyph; x1 fails on algebra-heavy only.
3. **Glyph paper hardening** (if publishing): 300-label manual audit → frozen statistics w/ CIs; confirmatory seeds 101/202/303 (arguably moot for dead hypothesis); full-FT or higher-rank control for the LoRA-bottleneck caveat; cross-cite Nemotron paper both ways.
4. **Deeper mechanism** (optional): head-level attribution in L22-26 band; single-glyph KV transplant; does constrained decoding reduce binding depth in agentic models (testable with open small agents).

## Key assets

- Results: `plans/reports/results-260810-0001-heldout-behavior-test-results-pilot-seed42.md` (+4 addenda)
- Recovery log: `plans/reports/results-260808-0130-...` | Cleanup audit: `cleanup-260806-1203-...`
- Papers: `paper/phase-marker-findings.tex/.pdf` (new), `paper/main.tex` (old claims, superseded, kept)
- Exploratory scripts (session scratchpad, copy out if needed long-term): base-model eval, mode-key sufficiency, steering, layer-sweep+KV, LoRA-delta, remote aggregation
- Modal volume `phase-marker-pilot-runs-v1`: single namespace `...src-173973c847e6...` holds everything receipt-chained
- Memory: `glyph-stage-a-pilot-status`, `glyph-modal-pipeline-gotchas`

## Unresolved questions

- Push 121 commits to origin? (suites green, ready)
- Scratchpad scripts are session-temporary — copy into repo `scripts/` for permanence?
- MDPI round-2: add glyph-mechanism citation before deadline?
