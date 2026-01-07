import json
import torch
import random
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/qwen2.5-0.5b-glyph-sft"
DATA_FILE = "data/unified_dataset.jsonl"
REQUIRED_GLYPHS = ["🜞", "🜆", "🜂", "🜃"] # Guideline, Plan, Step, Takeaway
SAMPLE_SIZE = 50
MAX_NEW_TOKENS = 512

def difficulty_bucket(question):
    length = len(question.split())
    nums = len(re.findall(r"\d+", question))
    ops = sum(op in question.lower() for op in ["each", "per", "twice", "remaining", "total", "difference", "sum"])
    score = length + 5 * nums + 10 * ops
    if score < 60: return "easy"
    elif score < 120: return "medium"
    else: return "hard"

def main():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print(f"Loading data from {DATA_FILE}")
    data = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Shuffle and sample
    random.seed(42)
    sample_data = random.sample(data, min(SAMPLE_SIZE, len(data)))
    
    stats = {
        "easy": {"glyph": 0, "total": 0},
        "medium": {"glyph": 0, "total": 0},
        "hard": {"glyph": 0, "total": 0},
    }
    
    results = []

    print("Starting evaluation...")
    for item in tqdm(sample_data):
        question = item["question"]
        bucket = difficulty_bucket(question)
        
        # Use Chat Template logic
        messages = [
            {"role": "user", "content": f"Solve the following problem carefully.\nDo not mention any tags, symbols, or special formatting.\n\nProblem:\n{question}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False # Greedy decoding for reproducibility
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Check glyphs
        emerged = any(g in response for g in REQUIRED_GLYPHS)
        
        stats[bucket]["total"] += 1
        if emerged:
            stats[bucket]["glyph"] += 1
            
        results.append({
            "id": item.get("id"),
            "question": question,
            "response": response,
            "emerged": emerged,
            "bucket": bucket
        })

    # Print Report
    print("\n=== GLYPH EMERGENCE REPORT ===")
    print(f"{ 'Bucket':<10} | { 'Rate':<10} | { 'Count':<10}")
    print("-" * 36)
    total_emerged = 0
    for b in ["easy", "medium", "hard"]:
        t = stats[b]["total"]
        c = stats[b]["glyph"]
        total_emerged += c
        rate = (c / t * 100) if t > 0 else 0.0
        print(f"{b:<10} | {rate:<9.1f}% | {c}/{t}")
    
    print("-" * 36)
    print(f"Total Emergence: {total_emerged}/{len(sample_data)} ({(total_emerged/len(sample_data)*100):.1f}%)")
    
    # Save detailed results
    with open("eval/lora_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to eval/lora_eval_results.json")

if __name__ == "__main__":
    main()
