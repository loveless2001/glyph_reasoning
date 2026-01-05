def glyph_prompt(question: str) -> str:
    return f"""Solve the following problem using structured reasoning.

You MUST use the following glyphs exactly once and in this order:

🜞 Guideline (overall approach)
🜆 Plan (high-level steps)
🜂 Step (detailed reasoning, may repeat)
🜃 Takeaway (summary)
🝞 Final answer (only the answer)

Problem:
{question}

Begin.
"""

def latent_prompt(question):
    return f"""
Solve the following problem carefully.
Do not mention any tags, symbols, or special formatting.

Problem:
{question}
"""

def natural_prompt(question):
    return f"""
Solve the following problem.

Problem:
{question}
"""
