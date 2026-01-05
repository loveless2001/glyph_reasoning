import json
import re
import torch
import os
import gc
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import xml_prompt, natural_prompt, glyph_prompt

# Default models to test
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512

PROMPTS = {
    "xml": xml_prompt,
    "natural": natural_prompt,
    "glyph": glyph_prompt,
}

def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

def structure_violation(text, mode):
    if mode == "xml":
        return not all(tag in text for tag in ["<guideline>", "<plan>", "<step>", "<takeaway>"])
    if mode == "natural":
        return not all(k in text for k in ["Guideline", "Plan", "Step", "Takeaway"])
    if mode == "glyph":
        return not all(g in text for g in ["🜞", "🜆", "🜂", "🜃"])
    return True

def evaluate_model(model_name, tasks):
    print(f"\nLoading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    results = {}

    for mode, prompt_fn in PROMPTS.items():
        correct = 0
        violations = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0

        # Define markers for each mode
        marker_map = {
            "glyph": "🜃",
            "xml": "<takeaway>",
            "natural": "Takeaway:"
        }
        marker = marker_map.get(mode)

        for task in tqdm(tasks, desc=f"[{model_name}] Mode: {mode}"):
            prompt = prompt_fn(task["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )
            
            gen_tokens = output[0][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Split by marker
            reasoning_part = gen_text
            answer_part = ""
            
            if marker and marker in gen_text:
                parts = gen_text.split(marker)
                answer_part = parts[-1] 
                reasoning_part = gen_text[:-(len(answer_part) + len(marker))]

            search_text = answer_part if (marker and marker in gen_text) else gen_text
            extracted = extract_answer(search_text)

            if extracted == task["answer"]:
                correct += 1

            if structure_violation(gen_text, mode):
                violations += 1
            
            r_tokens = len(tokenizer.encode(reasoning_part, add_special_tokens=False))
            a_tokens = len(tokenizer.encode(answer_part, add_special_tokens=False))
            
            total_reasoning_tokens += r_tokens
            total_answer_tokens += a_tokens

        results[mode] = {
            "accuracy": correct / len(tasks),
            "structure_violation_rate": violations / len(tasks),
            "avg_reasoning_tokens": total_reasoning_tokens / len(tasks),
            "avg_answer_tokens": total_answer_tokens / len(tasks),
            "avg_total_tokens": (total_reasoning_tokens + total_answer_tokens) / len(tasks),
        }
    
    # Cleanup to free VRAM
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple Qwen models on structure prompts")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to evaluate")
    parser.add_argument("--output", default="eval_results.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Load tasks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "tasks.json")) as f:
        tasks = json.load(f)

    all_results = {}

    for model_name in args.models:
        model_results = evaluate_model(model_name, tasks)
        if model_results:
            all_results[model_name] = model_results

    # Save to CSV
    if all_results:
        import csv
        print(f"\nSaving results to {args.output}...")
        try:
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Model", "Mode", "Accuracy", "Structure Violation Rate", 
                           "Avg Reasoning Tokens", "Avg Answer Tokens", "Avg Total Tokens"]
                writer.writerow(headers)

                for model_name, modes in all_results.items():
                    for mode, stats in modes.items():
                        writer.writerow([
                            model_name,
                            mode,
                            f"{stats['accuracy']:.4f}",
                            f"{stats['structure_violation_rate']:.4f}",
                            f"{stats['avg_reasoning_tokens']:.2f}",
                            f"{stats['avg_answer_tokens']:.2f}",
                            f"{stats['avg_total_tokens']:.2f}"
                        ])
            print("Save successful.")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    print("\n=== FINAL RESULTS ===")
    for model_name, modes in all_results.items():
        print(f"\nModel: {model_name}")
        for mode, stats in modes.items():
            print(f"  [{mode}]")
            for k, v in stats.items():
                print(f"    {k}: {v:.3f}")

if __name__ == "__main__":
    main()
