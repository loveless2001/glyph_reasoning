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
    tasks = [json.loads(line) for line in f]

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

BATCH_SIZE = 8

kept = 0
dropped = 0

def process_batch(batch_tasks):
    global kept, dropped
    prompts = [glyph_prompt(t["question"]) for t in batch_tasks]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    # Decode
    input_lens = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_lens:]
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    batch_records = []
    
    for task, decoded in zip(batch_tasks, decoded_texts):
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
        batch_records.append(json.dumps(record))
        kept += 1
        
    return batch_records

with open(OUTPUT_FILE, "w") as out:
    for i in tqdm(range(0, len(tasks), BATCH_SIZE)):
        batch = tasks[i : i + BATCH_SIZE]
        records = process_batch(batch)
        for r in records:
            out.write(r + "\n")

print(f"\nDone.")
print(f"Kept: {kept}")
print(f"Dropped: {dropped}")
print(f"Output: {OUTPUT_FILE}")
