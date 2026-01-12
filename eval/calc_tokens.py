
import json
import torch
import numpy as np
from transformers import AutoTokenizer

LOG_FILE = "glyph_reasoning/eval/debug_log_100.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading log: {LOG_FILE}")
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)

    stats = {}

    for entry in data:
        model = entry["model"]
        text = entry["response"]
        
        # Tokenize (just length)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        count = len(tokens)
        
        if model not in stats:
            stats[model] = []
        stats[model].append(count)

    print("\n=== TOKEN USAGE STATISTICS (Output) ===")
    for model, counts in stats.items():
        avg = np.mean(counts)
        std = np.std(counts)
        print(f"Model: {model}")
        print(f"  Avg Tokens: {avg:.2f}")
        print(f"  Std Dev:    {std:.2f}")
        print(f"  Min/Max:    {np.min(counts)} / {np.max(counts)}")
        print("-" * 30)

if __name__ == "__main__":
    main()
