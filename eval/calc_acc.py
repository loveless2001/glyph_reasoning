import json
import re

def extract_answer(text):
    # Use a simpler regex to avoid issues with backslashes in write_file
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # Fallback to last number
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else None

def main():
    log_path = "glyph_reasoning/eval/debug_log_100.json"
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    models = {}
    
    for entry in data:
        model = entry["model"]
        if model not in models:
            models[model] = {"correct": 0, "total": 0}
            
        gt = str(entry["answer_gt"])
        response = entry["response"]
        extracted = extract_answer(response)
        
        models[model]["total"] += 1
        if extracted == gt:
            models[model]["correct"] += 1
            
    print("=== ACCURACY RESULTS (100 Samples) ===")
    for model, stats in models.items():
        acc = (stats["correct"] / stats["total"]) * 100
        print(f"{model}: {acc:.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()
