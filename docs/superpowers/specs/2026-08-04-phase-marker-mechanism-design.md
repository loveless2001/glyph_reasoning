# Phase-Marker Mechanism Experiment Design

Date: 2026-08-04

## Purpose

Revise the glyph-reasoning experiment to establish whether symbolic phase
markers causally improve mathematical reasoning and, if they do, distinguish
among four mechanisms:

1. additional computational positions;
2. generic boundary or workspace effects;
3. stable, learned phase-control codes; and
4. semantic reasoning scaffolding independent of the markers.

The study combines a held-out behavioral factorial with activation-level
diagnostics and causal interventions. It replaces the current 1,000-example
result as confirmatory evidence because 510 of those examples exactly overlap
the SFT set and the existing prompt conditions vary more than marker identity.

## Claims

### Primary claim

The strongest claim the study is designed to test is that phase markers
causally alter mathematical reasoning performance beyond matched semantic
structure and token budget. If supported, the effect may be consistent with
marker positions serving as computational routing or workspace sites, with
learned marker identities potentially adding phase-specific control. The final
claim must instead follow the interpretation matrix when the evidence is
narrower or null.

### Claim boundaries

The study will not claim that the model implements human-like cognitive phases,
that rare symbols occupy "empty" representation space, or that an activation
probe alone identifies a causal mechanism. Mechanistic language requires an
intervention on marker or filler states that changes downstream computation.

### Competing hypotheses

| Hypothesis | Critical prediction |
| --- | --- |
| Extra computation | Length-matched dots and glyphs perform similarly; benefit scales with filler count rather than placement or identity. |
| Boundary workspace | Arbitrary markers help at true phase boundaries but not when clustered or displaced. |
| Learned control code | Trained glyph identity and order outperform equally placed dots; glyph-role swaps cause selective degradation. |
| Semantic scaffolding | Unmarked plan, execution, and conclusion content retains most of the gain; filler-only conditions do not. |

The hypotheses are not assumed to be mutually exclusive. The analysis may find
a generic workspace effect plus an additional glyph-specific control effect.

## Scope and Approach

The primary experiment is a matched natural-mathematics factorial using
GSM8K, SVAMP, and MATH. A smaller aligned synthetic suite supplies tasks with
known intermediate values for activation decoding and causal patching.

This integrated approach is preferred over either a filler-only replication or
a synthetic-only mechanism study. It connects the mechanism directly to the
repo's phase-marker claim while retaining the alignment needed for interpretable
causal interventions.

## Data Integrity

### Splits

- Retain the existing 3,850 filtered training traces as the source training
  examples.
- Create a validation split used exclusively for checkpoint selection and
  implementation checks.
- Use frozen official test splits from GSM8K, SVAMP, and MATH for confirmatory
  evaluation.
- Normalize questions and require zero question-hash overlap across training,
  validation, and test data.
- Freeze test manifests before training. Test outcomes may not influence prompt,
  checkpoint, layer, or intervention selection.

### Canonical traces

Parse every retained teacher trace into a canonical representation containing:

- problem identifier and source;
- question and final answer;
- ordered phase spans;
- phase-boundary locations; and
- exact semantic content within each phase.

All marker-only training arms must be deterministic renderings of this canonical
representation. After removing or substituting marker spans, semantic content
must be byte-identical across arms.

## Training Matrix

All arms use identical examples, example order, optimizer configuration, LoRA
configuration, batch construction, stopping rule, and model revision. Run a
pipeline pilot with one excluded seed, then run three fresh confirmatory seeds.

### Phase-structure arms

| Arm | Target rendering | Purpose |
| --- | --- | --- |
| Semantic CoT | Original phase content with headings and markers removed | Isolate the benefit of reasoning content. |
| Glyph boundaries | Identical content with the four fixed glyphs at true phase boundaries | Measure glyph contribution. |
| Dot boundaries | Identical content with each glyph replaced by a tokenizer-matched neutral delimiter | Measure generic boundary effects. |
| Random boundaries | Identical content with marker identities randomly remapped per example | Test whether stable control codes are necessary. |

Marker sets must be audited under every target tokenizer. Comparisons should
match token count and, where practical, token fragmentation. Token frequency,
vocabulary membership, and tokenization are measured rather than inferred from
visual appearance.

### Hidden-computation pair

| Arm | Target rendering | Purpose |
| --- | --- | --- |
| Direct answer | Final answer only | No-intermediate-token baseline. |
| Filler answer | A controlled dot sequence followed by the same final answer | Direct test of filler-token computation. |

The filler arm is trained on a balanced, deterministically assigned mixture of
4, 16, and 64 filler tokens plus a trace-length-matched setting capped at the
model's declared output budget. Each test example is evaluated at every filler
length. The arm is analyzed separately from the phase-structure arms because
replacing semantic reasoning with dots is not a marker-only intervention.

## Behavioral Evaluation

### Confirmatory grid

Cross-evaluate the four phase-structure training arms under four inference
renderings:

1. a neutral prompt;
2. the fixed glyph format;
3. the dot-boundary format; and
4. natural-language phase headings.

Prompt templates must be generated from one canonical template. Only the
declared format span may vary. Greedy decoding is the primary protocol. A
hash-selected subset of 250 examples per dataset uses sampled decoding to test
robustness; its membership is frozen with the test manifest.

### Focused perturbations

Evaluate the fixed-glyph model under:

- correct glyph placement;
- glyph deletion;
- all glyphs clustered before reasoning;
- glyphs displaced into phase interiors;
- glyph-role permutation;
- dots at the same boundaries; and
- unseen tokenizer-matched symbols at the same boundaries.

These perturbations separate token presence, token count, boundary placement,
identity, and learned order.

### Primary outcome and contrasts

The primary outcome is strict final-answer accuracy on the frozen test sets.
Pre-registered contrasts are:

1. glyph-boundary CoT versus identical semantic CoT;
2. dot-boundary CoT versus identical semantic CoT;
3. glyph boundaries versus dot boundaries;
4. aligned glyphs versus clustered, displaced, and permuted glyphs;
5. filler answer versus direct answer at each filler length; and
6. the interaction between training rendering and inference rendering.

Secondary outcomes are output-token count, format compliance, dataset and
difficulty breakdowns, synthetic intermediate-state accuracy, answer
log-probability and rank, and correct-to-wrong and wrong-to-correct transitions.

## Synthetic Mechanism Suite

Construct aligned deterministic tasks with known intermediate values. Include:

- modular arithmetic chains;
- affine transformation chains;
- two-source numeric composition; and
- deterministic string transformation and composition with exact-string
  scoring.

Each task template has four intermediate states and four aligned slot regions.
At total workspace lengths of 4, 16, and 64 tokens, divide the workspace evenly
across the four regions. The first position in each region is respectively an
ordered glyph, a dot delimiter, one repeated glyph, a permuted glyph, or a
tokenizer-matched random symbol; remaining positions use a shared neutral
filler token. The no-slot condition omits all four regions. This preserves the
phase layout while varying total computational workspace.

Synthetic tasks must have separately generated training, validation, and test
instances with no parameter tuple reused across splits. Their purpose is
mechanistic identification, not evidence of broad mathematical generalization.

## Mechanistic Analysis

### Aligned synthetic protocol

For every layer and marker or filler position:

1. Apply the logit lens to determine when known intermediate values become
   decodable.
2. Measure attention routing from question tokens to marker or filler positions
   and from those positions to the answer.
3. Patch residual-stream states from a matched donor example into a recipient.
4. Transplant only marker or filler KV-cache rows between matched examples.
5. Zero, mean-replace, or shuffle marker states while preserving ordinary
   tokens.

A phase-specific causal signature requires selectivity. For example, replacing
an execution-phase state should preferentially move execution-dependent
intermediates and the final answer. Random donors and matched non-marker
positions must not produce the same effect.

### Natural-mathematics protocol

- Capture activations during free generation for behavioral association.
- Separately teacher-force canonical correct traces so boundary positions align.
- Test whether marker states predict the upcoming phase, next intermediate
  quantity, and final answer.
- Train representation probes on one marker rendering and test them on dots and
  unseen symbols.
- Patch glyph-boundary states into aligned dot and no-marker traces and measure
  downstream answer likelihood.

Probe decodability is supporting evidence only. The causal claim depends on
patching, ablation, or transplantation changing downstream computation.

### Intervention controls

- Random donor transplantation.
- Non-marker positions matched by layer and sequence index.
- Equal token counts across symbol conditions.
- Mean and norm matching before activation replacement.
- Layer and region-of-interest selection on validation data only.
- Separate reporting of correct-to-wrong and wrong-to-correct transitions.
- Complete reporting of null and adverse interventions.

## Scoring and Statistics

### Strict scoring

Replace last-number extraction with:

- a required final-answer delimiter;
- dataset-aware numeric and symbolic normalization;
- rescoring from immutable raw generations;
- parser regression cases for fractions, negatives, percentages, units, and
  multiple numbers; and
- a manually audited, hash-selected sample of 100 generations per dataset,
  stratified across training arms and inference renderings.

Automated-versus-manual disagreement above 1% in the 300-generation audit
blocks confirmatory analysis until resolved.

### Statistical analysis

- Report every seed independently and the across-seed mean.
- Use paired question-level differences and 95% paired bootstrap intervals.
- Fit a hierarchical logistic model across datasets for the pre-registered
  training-arm by inference-rendering interactions.
- Correct secondary comparisons for multiplicity.
- Treat effects below two percentage points or intervals spanning zero as
  inconclusive, not as mechanistic evidence.
- For interventions, report the average treatment effect on answer
  log-probability, answer rank, and strict correctness.

Evaluation-sample uncertainty and training-seed variation must be labeled
separately.

## Software and Artifact Architecture

Separate the experiment into five bounded components:

1. **Data:** split builder, overlap audit, canonical trace parser, and
   deterministic renderers.
2. **Behavior:** prompt matrix, generation, and strict rescoring.
3. **Mechanism:** activation capture, logit lens, attention summaries,
   activation patching, and KV-cache transplantation.
4. **Analysis:** pre-registered contrasts, uncertainty estimates, tables, and
   figures.
5. **Manifests:** machine-readable artifact lineage.

The artifact flow is:

```text
canonical examples and traces
          -> validated split manifest
          -> deterministic training-arm renderings
          -> training manifests and three-seed checkpoints
          -> immutable per-example behavioral generations
          -> strict rescoring and confirmatory statistics
          -> aligned activation captures and interventions
          -> tables, figures, and bounded manuscript claims
```

Every stage records hashes, parent-artifact identifiers, model and tokenizer
revisions, seed, prompt template, decoding settings, and command configuration.
Generated summaries must remain reproducible from raw generations and
activation artifacts.

## Validation and Failure Handling

Before any full training launch:

- validate zero split overlap;
- validate byte-identical semantics across marker-only arms;
- snapshot tokenization for every marker set and model;
- validate deterministic rendering and stable identifiers;
- run scorer regression tests;
- complete one excluded-seed end-to-end pilot; and
- verify that activation capture does not change generation outputs.

Stop or interpretation rules:

- Any split overlap blocks training and evaluation.
- Any semantic mismatch between marker-only arms blocks launch.
- A failed parser audit blocks analysis.
- If glyphs do not beat semantic CoT, do not claim a marker-specific effect.
- If glyphs beat semantic CoT but not dots, conclude a generic
  boundary/workspace benefit.
- If aligned glyphs beat dots and displaced glyphs, test the phase-specific
  control account with causal interventions.
- Decodability without a successful intervention remains correlational.
- Null mechanistic results remain part of the paper.

## Interpretation Matrix

| Behavioral result | Causal result | Allowed conclusion |
| --- | --- | --- |
| Dots equal glyphs and both beat no markers | Filler or boundary states causally affect answers | Generic computational workspace or routing effect. |
| Glyphs beat dots only when aligned and trained | Glyph-state patches selectively move phase-dependent values | Learned phase-control codes add to generic workspace. |
| Semantic CoT equals glyph CoT | No selective glyph intervention | Reasoning content, not markers, explains the gain. |
| Glyph advantage without causal marker-state effect | Marker probes decode phase but patches fail | Behavioral marker effect with mechanism unresolved. |
| No held-out glyph advantage | Any activation pattern | No confirmatory evidence that phase markers improve reasoning. |

## Related Work Anchors

- Pfau, Merrill, and Bowman, *Let's Think Dot by Dot: Hidden Computation in
  Transformer Language Models*, arXiv:2404.15758.
- Brauer, Mayrink Verdun, and Marks, *Reading Between the Dots: Decoding Hidden
  Computation across Filler Tokens*, arXiv:2607.03502.
- Wang et al., *Guiding Language Model Reasoning with Planning Tokens*,
  arXiv:2310.05707.
- Liu, Li, and Xu, *Think in Sentences: Explicit Sentence Boundaries Enhance
  Language Model's Capabilities*, ACL 2026.
- Lu et al., *Strings from the Library of Babel: Random Sampling as a Strong
  Baseline for Prompt Optimisation*, NAACL 2024.

## Deliverables

1. Validated split and rendering manifests.
2. Six training arms with an excluded pilot and three confirmatory seeds.
3. Immutable per-example behavioral generations and strict scores.
4. Synthetic probe corpus with known intermediate states.
5. Layer-by-position activation and attention artifacts.
6. Causal patching and KV-transplant results with negative controls.
7. Reproducible analysis tables and figures.
8. A revised manuscript whose claims follow the interpretation matrix above.
