import os
import json
import logging
import datasets
import tqdm
import re
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "unified_dataset.jsonl")

def ensure_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created directory: {OUTPUT_DIR}")

def process_gsm8k():
    logging.info("Processing GSM8K...")
    ds = datasets.load_dataset("gsm8k", "main", split="train")
    
    # Calculate difficulty proxy: solution length
    # GSM8K solutions include the reasoning trace.
    # We sort by length and keep the top 50%.
    
    data_with_len = []
    for item in ds:
        sol_len = len(item['answer'])
        data_with_len.append({
            "item": item,
            "len": sol_len
        })
    
    # Sort descending by length
    data_with_len.sort(key=lambda x: x['len'], reverse=True)
    
    # Keep top 50%
    cutoff = len(data_with_len) // 2
    filtered_data = data_with_len[:cutoff]
    logging.info(f"GSM8K: Original size {len(ds)}, Filtered size {len(filtered_data)} (Hardest 50%)")
    
    processed = []
    for entry in filtered_data:
        item = entry['item']
        # Extract numerical answer (usually after #### in GSM8K)
        answer_split = item['answer'].split("####")
        final_answer = answer_split[-1].strip() if len(answer_split) > 1 else ""
        
        processed.append({
            "source": "gsm8k",
            "question": item['question'],
            "solution": item['answer'],
            "answer": final_answer,
            "difficulty_proxy": entry['len'] # keeping for debug
        })
    return processed

def process_svamp():
    logging.info("Processing SVAMP...")
    try:
        ds = datasets.load_dataset("ChilleD/SVAMP", split="train")
    except Exception as e:
        logging.warning("Failed to load ChilleD/SVAMP, trying 'ark-nlp/SVAMP' or manual loading if needed. Error: " + str(e))
        # Fallback to a generally available likely SVAMP source or assume ChilleD works. 
        # ChilleD/SVAMP is the common one.
        return []

    processed = []
    for item in ds:
        # SVAMP: Body + Question
        full_question = f"{item['Body']} {item['Question']}".strip()
        
        processed.append({
            "source": "svamp",
            "question": full_question,
            "solution": item['Equation'], # SVAMP provides equation as 'solution' mostly
            "answer": str(item['Answer']),
            "type": item.get('Type', 'unknown')
        })
    logging.info(f"SVAMP: Processed {len(processed)} items")
    return processed

def process_math():
    logging.info("Processing MATH (EleutherAI/hendrycks_math)...")
    configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    
    all_math_data = []
    
    for config in configs:
        try:
            logging.info(f"Loading MATH config: {config}")
            ds = datasets.load_dataset("EleutherAI/hendrycks_math", config, split="train", trust_remote_code=True)
            for item in ds:
                 # EleutherAI/hendrycks_math usually has 'problem' and 'solution'
                question = item.get('problem', item.get('question', ''))
                solution = item.get('solution', item.get('answer', ''))
                
                all_math_data.append({
                    "source": "math",
                    "question": question,
                    "solution": solution,
                    "answer": extract_boxed_answer(solution),
                    "level": item.get('level', ''),
                    "type": item.get('type', config) # Use config as type if type missing
                })
        except Exception as e:
            logging.error(f"Failed to load MATH config {config}: {e}")

    logging.info(f"MATH: Processed {len(all_math_data)} items")
    return all_math_data

def extract_boxed_answer(solution_text):
    # Heuristic to find \boxed{content}
    # This is a simple regex, might need more robustness for nested braces but sufficient for most MATH
    matches = re.findall(r'\\boxed\{(.*?)\}', solution_text)
    if matches:
        return matches[-1] # Usually the last boxed item is the answer
    return ""

def main():
    ensure_directory()
    
    all_data = []
    
    # GSM8K
    gsm8k_data = process_gsm8k()
    for i, item in enumerate(gsm8k_data):
        all_data.append({
            "id": f"gsm8k_{i}",
            "question": item["question"],
            "answer": item["answer"],
            "source": "gsm8k"
        })
    
    # SVAMP
    svamp_data = process_svamp()
    for i, item in enumerate(svamp_data):
        all_data.append({
            "id": f"svamp_{i}",
            "question": item["question"],
            "answer": item["answer"],
            "source": "svamp"
        })
    
    # MATH
    math_data = process_math()
    for i, item in enumerate(math_data):
        all_data.append({
            "id": f"math_{i}",
            "question": item["question"],
            "answer": item["answer"],
            "source": "math"
        })
    
    logging.info(f"Total Unified Records: {len(all_data)}")
    
    with open(OUTPUT_FILE, "w") as f:
        for record in all_data:
            f.write(json.dumps(record) + "\n")
            
    logging.info(f"Saved unified dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
