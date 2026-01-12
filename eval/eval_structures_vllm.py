import json
import re
import torch
import os
import gc
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompts import xml_prompt, natural_prompt, glyph_prompt

# Default models to test
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    #"Qwen/Qwen2.5-3B-Instruct",
    #"Qwen/Qwen2.5-7B-Instruct"
]

MAX_NEW_TOKENS = 1024

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

def evaluate_model_vllm(model_name, tasks, gpu_memory_utilization=0.9, max_model_len=8192):
    print(f"\n🚀 Loading model with vLLM: {model_name}...")
    try:
        # Load tokenizer for chat template application
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize vLLM
        # tensor_parallel_size=1 for single GPU.
        # gpu_memory_utilization can be adjusted if OOM.
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            stop_token_ids=[tokenizer.eos_token_id]
        )
        
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    results = {}

    for mode, prompt_fn in PROMPTS.items():
        print(f"Preparing prompts for mode: {mode}...")
        
        # Pre-calculate prompts
        prompts = []
        for task in tasks:
            raw_prompt = prompt_fn(task["question"])
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                full_prompt = raw_prompt
            prompts.append(full_prompt)

        # Generate in batch
        print(f"⚡ Generating {len(prompts)} responses...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Done in {duration:.2f}s ({len(prompts)/duration:.2f} it/s)")

        # Process results
        correct = 0
        violations = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0
        
        marker_map = {
            "glyph": "🜃",
            "xml": "<takeaway>",
            "natural": "Takeaway:"
        }
        marker = marker_map.get(mode)

        for i, output in enumerate(outputs):
            gen_text = output.outputs[0].text
            task = tasks[i]
            
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
            
            # vLLM provides token_ids in the output
            # We can map them back to reasoning/answer roughly by text usage, 
            # but since we already have text, let's re-encode purely for counting consistency with previous script
            # OR use logic on the text lengths.
            # Simpler: Re-use tokenizer to count tokens of parts.
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
    
    # Cleanup / Destroy LLM to free VRAM for next model
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple Qwen models with vLLM")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to evaluate")
    parser.add_argument("--output", default="eval/eval_results_vllm.csv", help="Output CSV file path")
    parser.add_argument("--data", default="data/tasks.json", help="Input data file (json or jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle tasks before limiting")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum context length for the model. Decrease to save VRAM.")
    args = parser.parse_args()

    # Load tasks
    print(f"Loading tasks from: {args.data}")
    tasks = []
    if args.data.endswith(".jsonl"):
        with open(args.data, "r") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
    else:
        with open(args.data, "r") as f:
            tasks = json.load(f)

    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(tasks)

    if args.limit:
        tasks = tasks[:args.limit]
    
    print(f"Evaluated on {len(tasks)} tasks.")

    all_results = {}

    def save_results(current_results):
        if not current_results:
            return
        import csv
        print(f"\nSaving results to {args.output}...")
        try:
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["Model", "Mode", "Accuracy", "Structure Violation Rate", 
                           "Avg Reasoning Tokens", "Avg Answer Tokens", "Avg Total Tokens"]
                writer.writerow(headers)

                for m_name, modes in current_results.items():
                    for mode, stats in modes.items():
                        writer.writerow([
                            m_name,
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

    for model_name in args.models:
        model_results = evaluate_model_vllm(
            model_name, 
            tasks, 
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )
        if model_results:
            all_results[model_name] = model_results
            save_results(all_results)
        
        # Explicit VRAM clearing pause
        time.sleep(2)

    print("\n=== FINAL RESULTS ===")
    for model_name, modes in all_results.items():
        print(f"\nModel: {model_name}")
        for mode, stats in modes.items():
            print(f"  [{mode}]")
            for k, v in stats.items():
                print(f"    {k}: {v:.3f}")

if __name__ == "__main__":
    main()
