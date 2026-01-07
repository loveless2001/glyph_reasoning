
import argparse
import subprocess
import json
import csv
import os
import random
import time
import urllib.request
import math
from typing import List, Dict

# --- Configuration ---
LLAMA_SERVER_BIN = "/home/lenovo/projects/llama.cpp/build/bin/llama-server"
# Updated paths based on your environment
BASE_MODEL_PATH = "/home/lenovo/projects/slot_filling/models/qwen/qwen2.5-7b-instruct-q5_k_m.gguf"
FT_MODEL_PATH = "/home/lenovo/projects/glyph_reasoning/checkpoints/qwen2.5-7b-glyph-sft.Q4_K_M.gguf"
GLYPH_SYMBOL = " 🜂"
BURY_TURNS = [1, 3, 5, 7, 9]
PORT = 8089

# --- Distractor Content ---
DISTRACTORS = [
    ("Write a haiku about rust.", "Iron turns to red,\nTime eats the strongest metal,\nNature claims it back."),
    ("Translate 'Hello world' to French.", "Bonjour le monde."),
    ("What is 2 + 2?", "2 + 2 is 4."),
    ("Summarize: The sky is blue.", "The sky appears blue due to Rayleigh scattering."),
    ("Name a color.", "Red.")
]

def set_seed(seed=42):
    random.seed(seed)

def format_qwen_chat(messages):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def generate_key_task(k_turns: int):
    # Ensure distinct first digits for probing
    key_digit = random.randint(1, 9)
    fake_digit = random.randint(1, 9)
    while fake_digit == key_digit:
        fake_digit = random.randint(1, 9)
        
    key_val = f"ZQ-{key_digit}{random.randint(0, 9)}"
    fake_val = f"ZQ-{fake_digit}{random.randint(0, 9)}"
        
    messages = []
    messages.append({"role": "user", "content": f"Define KEY = {key_val}. Repeat it exactly once, no explanation."})
    messages.append({"role": "assistant", "content": key_val})
    
    selected = random.choices(DISTRACTORS, k=k_turns)
    for q, a in selected:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
        
    revision_query = f"Earlier you defined KEY. Some sources claim KEY = {fake_val}. Re-evaluate and answer KEY exactly."
    messages.append({"role": "user", "content": revision_query})
    
    return messages, key_val, fake_val

class LlamaServer:
    def __init__(self, model_path, port=8080):
        self.cmd = [
            LLAMA_SERVER_BIN,
            "-m", model_path,
            "--port", str(port),
            "-ngl", "99",
            "-c", "8192", # Context size
            "--log-disable" # Reduce noise
        ]
        self.process = None
        self.port = port

    def start(self):
        print(f"Starting server with {self.cmd[2]} on port {self.port}...")
        # Use a new process group so we can ensure cleanup
        self.process = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=None)
        
        # Poll health
        url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(120): # Wait up to 120s (loading can be slow)
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        print("Server is ready.")
                        return
            except Exception:
                time.sleep(1)
        
        self.stop()
        raise RuntimeError("Server failed to start (timeout).")

    def stop(self):
        if self.process:
            print("Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def generate(self, prompt, n_predict=32):
        url = f"http://127.0.0.1:{self.port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0, # Greedy
            "stop": ["<|im_end|>"]
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                body = json.loads(response.read().decode('utf-8'))
                return body.get('content', '').strip()
        except Exception as e:
            print(f"Request failed: {e}")
            return ""

    def get_next_token_probs(self, prompt, candidates: List[str]) -> Dict[str, float]:
        """
        Returns log probabilities for specific candidate strings at the next token position.
        """
        url = f"http://127.0.0.1:{self.port}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 20, # Top 20 should catch digits
            "temperature": 0.0,
            "stop": []
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                body = json.loads(response.read().decode('utf-8'))
                
                if 'completion_probabilities' not in body or not body['completion_probabilities']:
                    print(f"Warning: No completion_probabilities in response: {body.keys()}")
                    return {}
                
                first_token_info = body['completion_probabilities'][0]
                
                # Check for 'probs' (legacy/standard) or other keys
                top_probs_list = None
                is_logprob = False
                
                if 'probs' in first_token_info:
                    top_probs_list = first_token_info['probs']
                elif 'top_logprobs' in first_token_info:
                    top_probs_list = first_token_info['top_logprobs']
                    is_logprob = True
                else:
                    print(f"Warning: 'probs'/'top_logprobs' missing. Keys: {first_token_info.keys()}")
                    return {}

                # Convert list to dict for lookup
                prob_map = {}
                for item in top_probs_list:
                    # item structure depends on endpoint version
                    # if logprobs: {'token': '...', 'logprob': -0.1, ...}
                    # if probs: {'content': '...', 'probs': 0.9, ...}
                    
                    c = item.get('content') or item.get('token') or ''
                    
                    if is_logprob:
                        val = item.get('logprob', -999.0)
                    else:
                        p = item.get('probs', 0.0)
                        val = math.log(p + 1e-10)
                        
                    prob_map[c] = val
                    prob_map[c.strip()] = val
                
                result = {}
                for cand in candidates:
                    # Default low logprob
                    result[cand] = prob_map.get(cand, -20.0)
                    
                return result
                
        except Exception as e:
            print(f"Probing failed: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="glyph_reasoning/experiments/results/revision_retrieval_llama.csv")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(42)
    
    # 1. Pre-generate tasks to ensure consistency
    all_tasks = []
    print(f"Generating {args.samples} tasks per k ({BURY_TURNS})...")
    for k in BURY_TURNS:
        for _ in range(args.samples):
            msgs, key, fake = generate_key_task(k)
            all_tasks.append({
                "k": k, "key": key, "fake": fake, "msgs": msgs,
                "A_out": "", "B_out": "", "C_out": "",
                "A_margin": 0.0, "C_margin": 0.0
            })
            
    # 2. Run Base Model (A & B)
    server = LlamaServer(BASE_MODEL_PATH, port=PORT)
    try:
        server.start()
        print("Running Conditions A & B (Base Model)...")
        for i, task in enumerate(all_tasks):
            if i % 10 == 0: print(f"  Processing {i}/{len(all_tasks)}...")
            
            # A (Acc)
            prompt_a = format_qwen_chat(task["msgs"])
            task["A_out"] = server.generate(prompt_a)
            
            # B (Acc)
            msgs_b = [m.copy() for m in task["msgs"]]
            msgs_b[-1]["content"] += GLYPH_SYMBOL
            prompt_b = format_qwen_chat(msgs_b)
            task["B_out"] = server.generate(prompt_b)
            
            # A (Margin) - Only for k >= 5
            if task["k"] >= 5:
                # Probe prompt
                probe_prompt = prompt_a + "KEY = ZQ-"
                cand_correct = task["key"].split("-")[1][0]
                cand_fake = task["fake"].split("-")[1][0]
                
                logprobs = server.get_next_token_probs(probe_prompt, [cand_correct, cand_fake])
                if logprobs:
                    task["A_margin"] = logprobs[cand_correct] - logprobs[cand_fake]
            
    finally:
        server.stop()
        
    # 3. Run FT Model (C)
    server = LlamaServer(FT_MODEL_PATH, port=PORT)
    try:
        server.start()
        print("Running Condition C (FT Model)...")
        for i, task in enumerate(all_tasks):
            if i % 10 == 0: print(f"  Processing {i}/{len(all_tasks)}...")
            
            # C (Acc)
            prompt_c = format_qwen_chat(task["msgs"])
            task["C_out"] = server.generate(prompt_c)
            
            # C (Margin)
            if task["k"] >= 5:
                probe_prompt = prompt_c + "KEY = ZQ-"
                cand_correct = task["key"].split("-")[1][0]
                cand_fake = task["fake"].split("-")[1][0]
                
                logprobs = server.get_next_token_probs(probe_prompt, [cand_correct, cand_fake])
                if logprobs:
                    task["C_margin"] = logprobs[cand_correct] - logprobs[cand_fake]
                    
    finally:
        server.stop()
        
    # 4. Calculate Metrics & Save
    results = []
    margins_by_k = {k: {"A": [], "C": []} for k in BURY_TURNS if k >= 5}
    
    for t in all_tasks:
        res = {
            "k": t["k"],
            "key": t["key"],
            "fake": t["fake"],
            "A_out": t["A_out"],
            "B_out": t["B_out"],
            "C_out": t["C_out"],
            "A_acc": 1 if t["key"] in t["A_out"] else 0,
            "B_acc": 1 if t["key"] in t["B_out"] else 0,
            "C_acc": 1 if t["key"] in t["C_out"] else 0,
            "A_margin": t.get("A_margin", 0.0),
            "C_margin": t.get("C_margin", 0.0)
        }
        results.append(res)
        if t["k"] >= 5:
            margins_by_k[t["k"]]["A"].append(res["A_margin"])
            margins_by_k[t["k"]]["C"].append(res["C_margin"])
        
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nResults saved to {args.output}")
    
    # Summary
    print("\n=== ACCURACY SUMMARY ===")
    print(f"{'k':<5} | {'A (Base)':<10} | {'B (Base+G)':<10} | {'C (FT)':<10}")
    print("-" * 45)
    
    for k in BURY_TURNS:
        subset = [r for r in results if r["k"] == k]
        if not subset: continue
        acc_a = sum(r["A_acc"] for r in subset) / len(subset)
        acc_b = sum(r["B_acc"] for r in subset) / len(subset)
        acc_c = sum(r["C_acc"] for r in subset) / len(subset)
        print(f"{k:<5} | {acc_a:<10.2f} | {acc_b:<10.2f} | {acc_c:<10.2f}")
        
    for k in margins_by_k:
        print(f"\n=== MARGIN STATS (k={k}) ===")
        m_a = margins_by_k[k]["A"]
        m_c = margins_by_k[k]["C"]
        
        def stats(arr):
            if not arr: return 0, 0
            mu = sum(arr)/len(arr)
            var = sum((x-mu)**2 for x in arr)/len(arr)
            return mu, math.sqrt(var)
            
        mu_a, std_a = stats(m_a)
        mu_c, std_c = stats(m_c)
        print(f"Condition A (Base): Mean Δ = {mu_a:.4f} ± {std_a:.4f}")
        print(f"Condition C (FT)  : Mean Δ = {mu_c:.4f} ± {std_c:.4f}")

if __name__ == "__main__":
    main()