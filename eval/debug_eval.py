
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "loveless2001/qwen2.5-7b-glyph-sft"
]

def load_data(path, limit=10):
    tasks = []
    # Try reading as jsonl
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
            if len(tasks) >= limit:
                break
    return tasks

def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
    
    # Slice off the prompt
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/unified_dataset.jsonl")
    parser.add_argument("--output", default="eval/debug_log.json")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        tasks = load_data(args.data, limit=args.limit)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    logs = []
    
    for model_name in MODELS:
        print(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/workspace/models")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                cache_dir="/workspace/models"
            )
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
            
        print(f"Generating responses for {model_name}...")
        for task in tqdm(tasks):
            prompt = task['question']
            response = generate(model, tokenizer, prompt)
            
            logs.append({
                "model": model_name,
                "id": task.get("id", "unknown"),
                "question": task['question'],
                "answer_gt": task['answer'],
                "response": response
            })
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    with open(args.output, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Saved logs to {args.output}")

if __name__ == "__main__":
    main()
