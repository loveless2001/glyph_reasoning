import argparse
import csv
import gc
import json
import os
import random
import re

import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import glyph_prompt, natural_prompt, xml_prompt

MAX_NEW_TOKENS = 512

PROMPTS = {
    "glyph": glyph_prompt,
    "xml": xml_prompt,
    "natural": natural_prompt,
}

MARKER_MAP = {
    "glyph": "🜃",
    "xml": "<takeaway>",
    "natural": "Takeaway:",
}


def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None


def structure_violation(text, mode):
    if mode == "xml":
        return not all(tag in text for tag in ["<guideline>", "<plan>", "<step>", "<takeaway>"])
    if mode == "natural":
        return not all(key in text for key in ["Guideline", "Plan", "Step", "Takeaway"])
    if mode == "glyph":
        return not all(glyph in text for glyph in ["🜞", "🜆", "🜂", "🜃"])
    return True


def load_tasks(data_path, limit=None, shuffle=False):
    tasks = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    tasks.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as handle:
            tasks = json.load(handle)

    if shuffle:
        random.seed(42)
        random.shuffle(tasks)
    if limit:
        tasks = tasks[:limit]
    return tasks


def load_model_and_tokenizer(model_name_or_path):
    is_adapter = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_adapter:
        config = PeftConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def evaluate_model(model_name_or_path, tasks):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name_or_path}")
    print(f"{'=' * 60}")

    try:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    except Exception as exc:
        print(f"FAILED to load {model_name_or_path}: {exc}")
        return None

    results = {}
    device = next(model.parameters()).device

    for mode, prompt_fn in PROMPTS.items():
        correct = 0
        violations = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0
        marker = MARKER_MAP[mode]

        for task in tqdm(tasks, desc=f"[{mode}]"):
            raw_prompt = prompt_fn(task["question"])
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                text_input = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_input = raw_prompt

            inputs = tokenizer([text_input], return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_tokens = output[0][inputs.input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            reasoning_part = gen_text
            answer_part = ""
            if marker in gen_text:
                parts = gen_text.split(marker)
                answer_part = parts[-1]
                reasoning_part = gen_text[:-(len(answer_part) + len(marker))]

            search_text = answer_part if marker in gen_text else gen_text
            extracted = extract_answer(search_text)
            if extracted == task["answer"]:
                correct += 1

            if structure_violation(gen_text, mode):
                violations += 1

            total_reasoning_tokens += len(tokenizer.encode(reasoning_part, add_special_tokens=False))
            total_answer_tokens += len(tokenizer.encode(answer_part, add_special_tokens=False))

        n = len(tasks)
        results[mode] = {
            "accuracy": correct / n,
            "structure_violation_rate": violations / n,
            "avg_reasoning_tokens": total_reasoning_tokens / n,
            "avg_answer_tokens": total_answer_tokens / n,
            "avg_total_tokens": (total_reasoning_tokens + total_answer_tokens) / n,
        }
        print(
            f"{mode}: acc={results[mode]['accuracy']:.4f} "
            f"viol={results[mode]['structure_violation_rate']:.4f} "
            f"avg_total={results[mode]['avg_total_tokens']:.1f}"
        )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def save_results(all_results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Model",
                "Mode",
                "Accuracy",
                "Structure Violation Rate",
                "Avg Reasoning Tokens",
                "Avg Answer Tokens",
                "Avg Total Tokens",
            ]
        )
        for model_name, modes in all_results.items():
            for mode, stats in modes.items():
                writer.writerow(
                    [
                        model_name,
                        mode,
                        f"{stats['accuracy']:.4f}",
                        f"{stats['structure_violation_rate']:.4f}",
                        f"{stats['avg_reasoning_tokens']:.2f}",
                        f"{stats['avg_answer_tokens']:.2f}",
                        f"{stats['avg_total_tokens']:.2f}",
                    ]
                )
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate base Llama vs LoRA adapter on structure prompts.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "/workspace/checkpoints/llama-3.1-8b-glyph-sft",
        ],
    )
    parser.add_argument("--data", default="data/unified_dataset.jsonl")
    parser.add_argument("--output", default="eval/eval_results_llama31_8b_sft.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.data, limit=args.limit, shuffle=args.shuffle)
    print(f"Evaluating on {len(tasks)} tasks")
    all_results = {}

    for model_name in args.models:
        result = evaluate_model(model_name, tasks)
        if result:
            all_results[model_name] = result

    if all_results:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
