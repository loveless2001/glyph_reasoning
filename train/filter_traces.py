import json
import re

def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

def valid_structure(text):
    return all(g in text for g in ["🜞", "🜆", "🜂", "🜃"])

inp = open("data/glyph_traces.jsonl")
out = open("data/glyph_traces_filtered.jsonl", "w")

for line in inp:
    r = json.loads(line)
    
    # Extract assistant output from messages
    # Assuming standard format: user then assistant
    try:
        glyph_output = r["messages"][1]["content"]
    except (KeyError, IndexError):
        continue

    pred = extract_answer(glyph_output)

    if pred != r["answer"]:
        continue
    if not valid_structure(glyph_output):
        continue
    if len(glyph_output) > 8000:
        continue

    out.write(json.dumps(r) + "\n")

inp.close()
out.close()
