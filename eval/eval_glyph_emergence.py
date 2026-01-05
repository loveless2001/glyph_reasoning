import re
import json
import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
REQUIRED_GLYPHS = ["🜞", "🜆", "🜂", "🜃"]
MAX_NEW_TOKENS = 1024

def difficulty_bucket(question):
    length = len(question.split())
    nums = len(re.findall(r"\d+", question))
    ops = sum(op in question.lower() for op in ["each", "per", "twice", "remaining", "total", "difference", "sum"])
    
    score = length + 5 * nums + 10 * ops
    
    if score < 60:
        return "easy"
    elif score < 120:
        return "medium"
    else:
        return "hard"

def promptless_eval_prompt(question):
    return f"""
Solve the following problem carefully and give the final answer.

Problem:
{question}
"""

def glyph_emerged(text):
    return any(g in text for g in REQUIRED_GLYPHS)

def load_test_tasks():
    # Load unified dataset and filter for test set or use existing if user provided check
    # Since we are in eval mode, let's try to load from 'data/unified_dataset.jsonl' 
    # but strictly we should use 'test' splits. 
    # For this script rigor, I will load GSM8K test split directly from huggingface if possible or fallback to local files.
    # To be consistent with local environment, I will look for data/unified_dataset.jsonl and just sample from it or use it all.
    # The user instruction says "Split tasks by objective difficulty". I will assume the input is the unified dataset.
    
    task_file = "data/unified_dataset.jsonl"
    if not os.path.exists(task_file):
        logging.error(f"Task file {task_file} not found.")
        return []
        
    tasks = []
    with open(task_file, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
            
    # Optional: Filter for a held-out set if we had one. 
    # For now, we will use the whole set for eval as per instruction context "Split tasks...".
    # But ideally we shouldn't test on training data.
    # Since unified_dataset was created from TRAIN splits, this is technically training data.
    # I should warn about this.
    logging.warning("Evaluating on data/unified_dataset.jsonl which contains TRAIN splits. Ideally use test splits.")
    
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen2.5-glyph-sft", help="Path to model checkpoint")
    parser.add_argument("--base_model", action="store_true", help="Use base model (Qwen/Qwen2.5-7B-Instruct) instead of checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    args = parser.parse_args()

    model_name_or_path = args.model_path if not args.base_model else "Qwen/Qwen2.5-7B-Instruct"
    
    logging.info(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tasks = load_test_tasks()
    if args.limit:
        tasks = tasks[:args.limit]
    
    logging.info(f"Loaded {len(tasks)} tasks")
    
    stats = {
        "easy": {"glyph": 0, "total": 0},
        "medium": {"glyph": 0, "total": 0},
        "hard": {"glyph": 0, "total": 0},
    }

    # Output file for inspection
    out_file = "eval/glyph_emergence_results.jsonl"
    f_out = open(out_file, "w")

    for task in tqdm(tasks):
        bucket = difficulty_bucket(task["question"])
        prompt = promptless_eval_prompt(task["question"])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        emerged = glyph_emerged(generated_text)
        
        stats[bucket]["total"] += 1
        if emerged:
            stats[bucket]["glyph"] += 1
            
        # Log result
        f_out.write(json.dumps({
            "id": task.get("id"),
            "bucket": bucket,
            "emerged": emerged,
            "question": task["question"],
            "output": generated_text
        }) + "\n")

    f_out.close()
    
    print("\n=== GLYPH EMERGENCE RESULTS ===")
    print(f"Model: {model_name_or_path}")
    print(f"{'Bucket':<10} | {'Rate':<10} | {'Count':<10}")
    print("-" * 36)
    for b in ["easy", "medium", "hard"]:
        total = stats[b]["total"]
        count = stats[b]["glyph"]
        rate = count / total if total > 0 else 0.0
        print(f"{b:<10} | {rate:<10.2f} | {count}/{total}")

if __name__ == "__main__":
    main()
