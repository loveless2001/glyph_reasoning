import argparse
import os
import json
import torch
from lm_eval import evaluator, tasks, models
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Model using LM Eval Harness")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--tasks", type=str, default="gsm8k,arc_easy", help="Comma-separated list of tasks (e.g., gsm8k,mmlu)")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size (auto, or int)")
    parser.add_argument("--output_file", type=str, default="eval/benchmark_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    args = parser.parse_args()

    print(f"🚀 Loading model from: {args.model_path}")
    print(f"📊 Tasks: {args.tasks}")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 1. Load Model & Tokenizer
    # We use HFLM from lm_eval which wraps HuggingFace models
    # Note: 'trust_remote_code=True' might be needed for some Qwen versions
    model_args = f"pretrained={args.model_path},dtype=float16,trust_remote_code=True"
    
    # 2. Run Evaluation
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        device=args.device,
    )

    # 3. Save Results
    print("\n📈 Results:")
    for task, metrics in results["results"].items():
        print(f"  - {task}: {metrics.get('acc,none', metrics.get('acc'))}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
