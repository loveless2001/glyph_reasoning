import argparse
import csv
import gc
import json
import os
import random
import re

import torch
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from prompts import glyph_prompt, natural_prompt, xml_prompt

DEFAULT_MAX_NEW_TOKENS = 512

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


def resolve_models(model_args):
    if not model_args:
        return "meta-llama/Llama-3.1-8B-Instruct", None

    if len(model_args) == 1:
        only_path = model_args[0]
        if os.path.exists(os.path.join(only_path, "adapter_config.json")):
            config = PeftConfig.from_pretrained(only_path)
            return config.base_model_name_or_path, only_path
        return only_path, None

    base_model = model_args[0]
    adapter_path = None
    for candidate in model_args[1:]:
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            adapter_path = candidate
            break
    return base_model, adapter_path


def build_prompts(tasks, tokenizer, prompt_fn):
    prompts = []
    for task in tasks:
        raw_prompt = prompt_fn(task["question"])
        messages = [{"role": "user", "content": raw_prompt}]
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = raw_prompt
        prompts.append(full_prompt)
    return prompts


def score_outputs(outputs, tasks, tokenizer, mode):
    correct = 0
    violations = 0
    total_reasoning_tokens = 0
    total_answer_tokens = 0
    marker = MARKER_MAP[mode]

    for task, output in zip(tasks, outputs):
        gen_text = output.outputs[0].text
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
    return {
        "accuracy": correct / n,
        "structure_violation_rate": violations / n,
        "avg_reasoning_tokens": total_reasoning_tokens / n,
        "avg_answer_tokens": total_answer_tokens / n,
        "avg_total_tokens": (total_reasoning_tokens + total_answer_tokens) / n,
    }


def evaluate_label(label, llm, tokenizer, tasks, sampling_params, lora_request=None):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"{'=' * 60}")

    results = {}
    for mode, prompt_fn in PROMPTS.items():
        prompts = build_prompts(tasks, tokenizer, prompt_fn)
        print(f"Generating {len(prompts)} responses for {mode}...")
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        results[mode] = score_outputs(outputs, tasks, tokenizer, mode)
        print(
            f"{mode}: acc={results[mode]['accuracy']:.4f} "
            f"viol={results[mode]['structure_violation_rate']:.4f} "
            f"avg_total={results[mode]['avg_total_tokens']:.1f}"
        )
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
    parser = argparse.ArgumentParser(description="Evaluate base Llama vs LoRA adapter on structure prompts with vLLM.")
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    tasks = load_tasks(args.data, limit=args.limit, shuffle=args.shuffle)
    print(f"Evaluating on {len(tasks)} tasks")

    base_model, adapter_path = resolve_models(args.models)
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    llm = LLM(
        model=base_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=bool(adapter_path),
        hf_token=hf_token,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    all_results = {
        base_model: evaluate_label(base_model, llm, tokenizer, tasks, sampling_params),
    }

    if adapter_path:
        lora_request = LoRARequest("glyph_sft", 1, adapter_path, base_model_name=base_model)
        all_results[adapter_path] = evaluate_label(
            adapter_path,
            llm,
            tokenizer,
            tasks,
            sampling_params,
            lora_request=lora_request,
        )

    save_results(all_results, args.output)

    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
