"""
Ablation evaluation script for glyph reasoning experiments using vLLM.

Runs two ablation conditions against the fine-tuned model:
  1. glyph_shuffled — same glyphs, wrong order (tests if ordering matters)
  2. emoji — common emojis replacing alchemical glyphs (tests if symbol type matters)

Also runs the original glyph prompt as control for direct comparison.

Usage:
  python eval-ablation-shuffled-and-emoji-vllm.py \
    --models checkpoints/qwen2.5-7b-glyph-sft-hf \
    --data data/unified_dataset.jsonl \
    --limit 100 \
    --output eval/eval_results_ablation.csv

Streaming logs go to stdout + eval/logs/ablation-{timestamp}.log
"""

import json
import re
import torch
import os
import gc
import sys
import argparse
import time
import csv
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Import all prompt variants
from prompts import glyph_prompt
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ablation_prompts = import_module("prompts-ablation-shuffled-and-emoji")
glyph_shuffled_prompt = ablation_prompts.glyph_shuffled_prompt
emoji_prompt = ablation_prompts.emoji_prompt

MAX_NEW_TOKENS = 1024

# All conditions to evaluate
PROMPTS = {
    "glyph": glyph_prompt,                # control
    "glyph_shuffled": glyph_shuffled_prompt,  # ablation 1
    "emoji": emoji_prompt,                 # ablation 2
}

# Structure violation checks per mode
STRUCTURE_CHECKS = {
    "glyph": ["🜞", "🜆", "🜂", "🜃"],
    "glyph_shuffled": ["🜞", "🜆", "🜂", "🜃"],  # same glyphs, any order
    "emoji": ["🧭", "📋", "🔢", "✅"],
}

# Marker to split reasoning vs answer (last section marker)
MARKER_MAP = {
    "glyph": "🜃",
    "glyph_shuffled": "🜆",  # 🜆 is last in shuffled order
    "emoji": "✅",
}


class StreamLogger:
    """Tee stdout to both console and a log file for remote monitoring."""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def fileno(self):
        return self.terminal.fileno()

    def close(self):
        if not self.log.closed:
            self.log.close()


def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def structure_violation(text, mode):
    """Check if output contains all expected structure markers."""
    markers = STRUCTURE_CHECKS.get(mode, [])
    return not all(m in text for m in markers)


def evaluate_model_vllm(model_name, tasks, gpu_memory_utilization=0.9, max_model_len=8192):
    print(f"\n{'='*60}")
    print(f"Loading model with vLLM: {model_name}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        print(f"FAILED to load {model_name}: {e}")
        return None

    results = {}

    for mode, prompt_fn in PROMPTS.items():
        print(f"\n--- Mode: {mode} ---")

        # Build prompts
        prompts = []
        for task in tasks:
            raw_prompt = prompt_fn(task["question"])
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                full_prompt = raw_prompt
            prompts.append(full_prompt)

        # Batch generate
        print(f"Generating {len(prompts)} responses...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time
        print(f"Done in {duration:.2f}s ({len(prompts)/duration:.2f} samples/s)")

        # Score results
        correct = 0
        violations = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0
        marker = MARKER_MAP.get(mode)

        for i, output in enumerate(outputs):
            gen_text = output.outputs[0].text
            task = tasks[i]

            # Split reasoning vs answer at marker
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

        n = len(tasks)
        results[mode] = {
            "accuracy": correct / n,
            "structure_violation_rate": violations / n,
            "avg_reasoning_tokens": total_reasoning_tokens / n,
            "avg_answer_tokens": total_answer_tokens / n,
            "avg_total_tokens": (total_reasoning_tokens + total_answer_tokens) / n,
        }
        print(f"  accuracy={results[mode]['accuracy']:.4f}  "
              f"violations={results[mode]['structure_violation_rate']:.4f}  "
              f"avg_tokens={results[mode]['avg_total_tokens']:.1f}")

    # Cleanup VRAM
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results


def save_results(all_results, output_path):
    if not all_results:
        return
    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Mode", "Accuracy", "Structure Violation Rate",
            "Avg Reasoning Tokens", "Avg Answer Tokens", "Avg Total Tokens"
        ])
        for model_name, modes in all_results.items():
            for mode, stats in modes.items():
                writer.writerow([
                    model_name, mode,
                    f"{stats['accuracy']:.4f}",
                    f"{stats['structure_violation_rate']:.4f}",
                    f"{stats['avg_reasoning_tokens']:.2f}",
                    f"{stats['avg_answer_tokens']:.2f}",
                    f"{stats['avg_total_tokens']:.2f}"
                ])
    print("Saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation eval: shuffled glyphs + emoji replacement"
    )
    parser.add_argument("--models", nargs="+",
                        default=[
                            "Qwen/Qwen2.5-7B-Instruct",
                            "checkpoints/qwen2.5-7b-glyph-sft-hf",
                        ],
                        help="Models to evaluate (base + fine-tuned)")
    parser.add_argument("--output", default="eval/eval_results_ablation.csv")
    parser.add_argument("--data", default="data/unified_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    # Setup streaming log
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"eval/logs/ablation-{timestamp}.log"
    logger = StreamLogger(log_path)
    sys.stdout = logger
    print(f"Streaming log: {log_path}")
    print(f"Monitor with: tail -f {log_path}")

    # Load tasks
    print(f"\nLoading tasks from: {args.data}")
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

    print(f"Evaluating on {len(tasks)} tasks")
    print(f"Conditions: {list(PROMPTS.keys())}")
    print(f"Models: {args.models}")

    all_results = {}
    for model_name in args.models:
        model_results = evaluate_model_vllm(
            model_name, tasks,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )
        if model_results:
            all_results[model_name] = model_results
            save_results(all_results, args.output)
        time.sleep(2)

    # Final summary
    print(f"\n{'='*60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    for model_name, modes in all_results.items():
        print(f"\nModel: {model_name}")
        for mode, stats in modes.items():
            print(f"  [{mode}]")
            for k, v in stats.items():
                print(f"    {k}: {v:.4f}")

    logger.close()


if __name__ == "__main__":
    main()
