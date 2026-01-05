import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import glyph_prompt

# =====================
# CONFIG
# =====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "data/unified_dataset.jsonl"
OUTPUT_FILE = "data/glyph_traces.jsonl"
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUIRED_GLYPHS = ["🜞", "🜆", "🜂", "🜃", "🝞"]

# =====================
# Helpers
# =====================
def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

def valid_glyph_structure(text):
    pos = [text.find(g) for g in REQUIRED_GLYPHS]
    return all(p != -1 for p in pos) and pos == sorted(pos)

# =====================
# Load model
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# =====================
# Load tasks
# =====================
with open(INPUT_FILE) as f:
    tasks = json.load(f)

kept = 0
dropped = 0

with open(OUTPUT_FILE, "w") as out:
    for task in tqdm(tasks):
        prompt = glyph_prompt(task["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # ===== Filters =====
        if not valid_glyph_structure(decoded):
            dropped += 1
            continue

        answer = extract_answer(decoded)
        if answer != task["answer"]:
            dropped += 1
            continue

        # ===== Keep =====
        record = {
            "id": task["id"],
            "source": task["source"],
            "question": task["question"],
            "answer": task["answer"],
            "messages": [
                {"role": "user", "content": task["question"]},
                {"role": "assistant", "content": decoded}
            ]
        }

        out.write(json.dumps(record) + "\n")
        kept += 1

print(f"\nDone.")
print(f"Kept: {kept}")
print(f"Dropped: {dropped}")
print(f"Output: {OUTPUT_FILE}")
