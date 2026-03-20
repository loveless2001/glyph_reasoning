"""
Ablation prompt variants for glyph reasoning experiments.

Experiment 1: Shuffled glyph order — same glyphs, wrong order.
  Tests whether the specific ordering (🜞→🜆→🜂→🜃) matters,
  or if the glyphs work regardless of sequence.

Experiment 2: Emoji replacement — common emojis instead of alchemical glyphs.
  Tests whether the effect is specific to obscure/low-frequency symbols,
  or if any visual markers (including semantically-loaded emojis) work.
"""


# --- Experiment 1: Shuffled glyph order ---
# Original order: 🜞 🜆 🜂 🜃  (Crux → Flux → Ignis → Terra)
# Shuffled order: 🜂 🜃 🜞 🜆  (Ignis → Terra → Crux → Flux)
# This deliberately breaks the trained sequence while keeping the same symbols.

def glyph_shuffled_prompt(q):
    return f"""
Solve the problem using the glyph structure:

🜂 Solve the problem carefully.
🜃 Describe the approach.
🜞 Work through the calculation.
🜆 Give the final numeric answer.

Problem:
{q}
"""


# --- Experiment 2: Emoji replacement ---
# Replace alchemical glyphs with common, semantically-loaded emojis:
#   🜞 (Crux/Guideline) → 🧭 (compass — guidance)
#   🜆 (Flux/Plan)      → 📋 (clipboard — planning)
#   🜂 (Ignis/Step)     → 🔢 (numbers — calculation)
#   🜃 (Terra/Takeaway) → ✅ (checkmark — conclusion)

def emoji_prompt(q):
    return f"""
Solve the problem using the glyph structure:

🧭 Solve the problem carefully.
📋 Describe the approach.
🔢 Work through the calculation.
✅ Give the final numeric answer.

Problem:
{q}
"""
