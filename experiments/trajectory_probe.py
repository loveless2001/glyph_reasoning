# mech_probe_revision_v4_reversible.py
# Implements "Reversibility Steering" Protocol (User Suggestion)
# pip install torch transformers accelerate sentencepiece

import argparse
import csv
import random
import re
import os
import sys
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Config & CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory Probe with Reversibility Steering")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    
    # Conditions: 
    # 'steer_reversible': The main protocol (push -> pull)
    # 'explicit_glyph': Reference C
    # 'none': Reference A (Baseline)
    parser.add_argument("--condition", type=str, default="steer_reversible",
                        choices=["none", "explicit_glyph", "steer_reversible", "glyph_legacy"],
                        help="Experiment condition.")
    
    parser.add_argument("--steer_source", type=str, default="explicit_glyph_delta",
                        choices=["explicit_glyph_delta", "internal_extraction"],
                        help="Source of the steering vector Delta.")
    
    parser.add_argument("--layer_index", type=int, default=-1, 
                        help="Layer to inject. If -1 or pointing to end, may target final norm if hook_point is set.")
    
    parser.add_argument("--hook_point", type=str, default="final_norm",
                        choices=["layer", "final_norm", "lm_head"],
                        help="Where to attach the hook. 'layer' uses layer_index. 'final_norm' targets ln_f. 'lm_head' targets logits.")
    
    parser.add_argument("--alpha", type=float, default=2.0, help="Injection strength (epsilon/alpha).")
    parser.add_argument("--decode_window", type=int, default=15, help="Window size T for push/pull schedule.")
    
    parser.add_argument("--N", type=int, default=20, help="Number of samples.")
    parser.add_argument("--base_seed", type=int, default=1000, help="Base seed.")
    parser.add_argument("--output_file", type=str, default="reversibility_results.csv")
    parser.add_argument("--do_sample", type=int, default=0, help="1 for sampling, 0 for deterministic (greedy).")
    
    return parser.parse_args()

@dataclass
class RunResult:
    sample_id: int
    condition: str
    alpha: float
    full_text: str
    phase_texts: Dict[str, str]
    label: str
    revision_depth: int
    
    # Reversibility Metrics
    logit_diff_max: float
    state_dist_trace: List[float] # Time series of distance vs baseline
    
    # Metadata
    steer_vec_norm: float
    decode_window: int

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Metrics & Labeling
# -------------------------

REVISE_MARKERS = [
    "i was wrong", "you're right", "correction", "i retract", "i need to correct",
    "i made a mistake", "let me revise", "update:", "i cannot verify", "i'm not sure",
    "uncertain", "cannot confirm", "may be wrong", "apologize", "misunderstanding",
    "might be some confusion", "appear to be associated", "does not correspond"
]

DOUBLEDOWN_MARKERS = [
    "as i said", "as stated", "definitely", "certainly", "the answer is",
    "stand by", "correctly stated", "remain confident", "remains correct",
    "i stand by", "maintain that"
]

def label_revision_behavior(commit_answer: str, revise_answer: str) -> str:
    ca = commit_answer.strip().lower()
    ra = revise_answer.strip().lower()

    if any(m in ra for m in REVISE_MARKERS):
        return "R"
    
    # Check for Double Down
    shared = 0
    commit_words = re.findall(r"\w+", ca)
    unique_long_words = set([w for w in commit_words if len(w) > 4])
    for w in unique_long_words:
        if w in ra:
            shared += 1
            
    if shared >= 3 and any(m in ra for m in DOUBLEDOWN_MARKERS):
        return "D"
    
    hedges = ["maybe", "might", "cannot", "uncertain", "not sure", "possible", "unclear"]
    if any(h in ra for h in hedges):
        return "W"
    
    return "W" # Default

def calculate_revision_depth(text: str) -> int:
    text_lower = text.lower()
    score = 0
    assumptions = ["assum", "presum", "premise", "suppos", "hypothes"]
    if any(x in text_lower for x in assumptions): score += 1
    alts = ["alternative", "possibility", "possibilities", "could also be", "another option", "scenario", "usually"]
    count_alts = sum(text_lower.count(a) for a in alts)
    if count_alts >= 2: score += 1
    retractions = ["incorrect", "mistaken", "wrongly", "error in", "not actually", "misidentified"]
    if any(r in text_lower for r in retractions): score += 1
    return score

# -------------------------
# Model & Injection Utils
# -------------------------

@torch.no_grad()
def get_hidden_states(model, input_ids: torch.Tensor, layer_idx: int, hook_point: str = "layer") -> torch.Tensor:
    """
    Runs forward pass and extracts hidden states at the specified point.
    Returns: [batch, seq, hidden_dim]
    """
    # This is tricky because we need to hook to get the value, 
    # unless using output_hidden_states for standard layers.
    # For final_norm, we definitely need a hook if it's not exposed.
    # But `output_hidden_states=True` usually gives all layer outputs.
    # `norm` output is NOT in hidden_states usually (it's often applied after).
    
    captured = {}
    def capture_hook(module, inp, out):
        if isinstance(out, tuple): captured['h'] = out[0].detach()
        else: captured['h'] = out.detach()
        
    handle = None
    target_module = None
    
    if hook_point == "layer":
        # Just use standard output_hidden_states
        pass # Handle below
    elif hook_point == "final_norm":
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            target_module = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            target_module = model.transformer.ln_f
    elif hook_point == "lm_head":
        # We can just get logits from output
        pass
        
    if target_module:
        handle = target_module.register_forward_hook(capture_hook)
        
    try:
        out = model(input_ids, output_hidden_states=True)
        
        if hook_point == "lm_head":
            return out.logits
            
        if hook_point == "final_norm":
            if 'h' in captured:
                return captured['h']
            else:
                # Fallback if hook failed?
                return out.hidden_states[-1] # Approximation
                
        # "layer" case
        # out.hidden_states is tuple of (embeddings, layer_1, ... layer_N)
        # So index 0 is embeddings. index i+1 is output of layer i.
        # Handle negative indexing carefully.
        # hidden_states length = num_layers + 1
        return out.hidden_states[layer_idx if layer_idx >= 0 else layer_idx]
        
    finally:
        if handle: handle.remove()


class WindowedInjector:
    def __init__(self, vec: torch.Tensor, alpha: float, decode_window: int, reversible: bool = False):
        self.vec = vec.view(1, 1, -1).clone().detach() # [1, 1, D]
        self.alpha = alpha
        self.decode_window = decode_window
        self.reversible = reversible
        self.step = 0
        self.handle = None
        self.active_delta = 0.0 # Tracking for debug
        
    def hook(self, module, input, output):
        # Determine if we are in decode phase
        # output: [Batch, Seq, Dim]
        if isinstance(output, tuple): o = output[0]
        else: o = output
        
        B, S, D = o.shape
        
        # We assume decode mode is when S == 1 (after prefill)
        # BUT be careful: if we just started generation without past_key_values, S could be large.
        # The user's protocol implies we are injecting during GENERATION.
        
        if S == 1:
            self.step += 1
            current_alpha = 0.0
            
            # T = decode_window
            # 1..T: +alpha
            if 1 <= self.step <= self.decode_window:
                current_alpha = self.alpha
                
            # T+1..2T: -alpha (if reversible)
            elif self.reversible and self.decode_window < self.step <= 2 * self.decode_window:
                current_alpha = -self.alpha
            
            if current_alpha != 0:
                # Inject
                # Ensure device match
                if self.vec.device != o.device:
                    self.vec = self.vec.to(o.device)
                if self.vec.dtype != o.dtype:
                    self.vec = self.vec.to(o.dtype)
                
                o = o + current_alpha * self.vec
                self.active_delta = current_alpha
            else:
                self.active_delta = 0.0
                
        if isinstance(output, tuple):
            return (o,) + output[1:]
        return o

    def reset(self):
        self.step = 0
        self.active_delta = 0.0

# -------------------------
# Generation & Dataset
# -------------------------

FAKE_ENTITIES = ["Orion protocol", "Hollow Flame framework", "Blue River Accord", "Silver Dawn Initiative"]
DOMAINS = ["embedded systems", "cryptography", "distributed computing", "int'l policy"]

def generate_dataset_entries(N=20):
    data = []
    for _ in range(N):
        e = random.choice(FAKE_ENTITIES)
        d = random.choice(DOMAINS)
        # We need a prompt that encourages commit, then we will force a revise prompt.
        commit_q = f"The {e} was introduced in the field of {d}. Who created it and in what year? Answer directly."
        # Revise prompt:
        revise_q = f"Earlier you answered about the {e}. Re-evaluate. What assumptions might be wrong?"
        data.append((commit_q, revise_q))
    return data

def generate_text_deterministic(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, # DETERMINISTIC
        # temperature=None,
        # top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    gen_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# -------------------------
# Main Logic
# -------------------------

def run_experiment(args):
    set_seed(args.base_seed)
    
    # Load Model
    print(f"Loading {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    samples = generate_dataset_entries(args.N)
    results = []

    # Identify injection module
    injection_module = None
    if args.hook_point == "final_norm":
        if hasattr(model, "model") and hasattr(model.model, "norm"): injection_module = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"): injection_module = model.transformer.ln_f
    elif args.hook_point == "lm_head":
        injection_module = model.lm_head # Usually available
    elif args.hook_point == "layer":
        # Typical HF naming
        if hasattr(model, "model") and hasattr(model.model, "layers"): injection_module = model.model.layers[args.layer_index]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"): injection_module = model.transformer.h[args.layer_index]

    if injection_module is None:
        print(f"CRITICAL: Could not find injection module for {args.hook_point}")
        return

    print(f"Injection target: {injection_module}")

    for i, (commit_q, revise_q) in enumerate(samples):
        # 1. Commit (just to establish context in memory if we were continuously chatting, 
        # but here we usually construct a full prompt history. 
        # Let's assume we just prompt the model to simulate the 'Revise' phase directly 
        # with a simulated history to ensure control.)
        
        # Construct Full Context
        # User: Commit Q -> Asst: [Fake Commit] -> User: Revise Q
        # We need a fake commit answer to ensure consistency across runs if we want pure control.
        # But let's generate it to be safe.
        
        # Commit Gen
        msgs_commit = [{"role": "user", "content": commit_q}]
        prompt_commit = tokenizer.apply_chat_template(msgs_commit, tokenize=False, add_generation_prompt=True)
        commit_ans = generate_text_deterministic(model, tokenizer, prompt_commit, max_new_tokens=100)
        
        # Revise Prompt Construction
        msgs_revise = [
            {"role": "user", "content": commit_q},
            {"role": "assistant", "content": commit_ans},
            {"role": "user", "content": revise_q}
        ]
        prompt_revise = tokenizer.apply_chat_template(msgs_revise, tokenize=False, add_generation_prompt=True)
        
        # ---------------------------
        # COMPUTE STEERING VECTOR (DELTA)
        # ---------------------------
        delta = None
        if args.steer_source == "explicit_glyph_delta":
            # Condition A: Natural Prompt (prompt_revise)
            # Condition C: Explicit Glyph (prompt_revise + GLYPH)
            # Note: We want the state difference caused by the glyph. 
            # We append the glyph to the prompt.
            glyph_char = " 🜂" # Explicit glyph
            
            prompt_A = prompt_revise
            prompt_C = prompt_revise + glyph_char
            
            # We need to compute the hidden state at the last token of the prompt
            input_A = tokenizer(prompt_A, return_tensors="pt").to(model.device)
            input_C = tokenizer(prompt_C, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                h_A = get_hidden_states(model, input_A.input_ids, args.layer_index, args.hook_point)
                h_C = get_hidden_states(model, input_C.input_ids, args.layer_index, args.hook_point)
                
            # Take last token state
            # Shape [1, seq, d]
            v_A = h_A[0, -1, :]
            v_C = h_C[0, -1, :]
            
            delta = v_C - v_A
            # Normalize? The user formula calculates mean difference. 
            # Usually steering vectors are normalized, but here 'amplitude' might matter.
            # "The state is not just noise".
            # Let's normalize Delta and control magnitude with Alpha.
            delta = delta / (delta.norm() + 1e-8)
            
        else:
            # Fallback or other source
            delta = torch.zeros(4096).to(model.device)

        steer_vec_norm = delta.norm().item()

        # ---------------------------
        # RUN BASELINE (No Injection)
        # ---------------------------
        with torch.no_grad():
            full_ids = tokenizer(prompt_revise, return_tensors="pt").to(model.device).input_ids
            # We want to trace the hidden states during generation to measure distance later.
            # Standard generate doesn't give us easy per-step hidden states unless we loop.
            # But the user wants to measure "State distance vs baseline".
            # So we need a baseline trace.
            
            # For simplicity, let's run the baseline generation first.
            base_out = model.generate(
                full_ids, 
                max_new_tokens=100, 
                do_sample=False, 
                output_hidden_states=True, 
                return_dict_in_generate=True,
                use_cache=True
            )
            base_ids = base_out.sequences[0]
            # Extract baseline states for the generated tokens
            # base_out.hidden_states is a tuple (one per step) of tuples (one per layer).
            # We need the specific layer/hook point.
            
            # Collect baseline trace (vectors at the hook point)
            base_trace = []
            if args.hook_point == "final_norm":
                # Only way to get final_norm for sure strictly from generate output is tough 
                # if it's not exposed. We might rely on the last hidden state of the last layer 
                # as a proxy if we can't hook during generate easily without custom streamer.
                # ACTUALLY, we can use the same Injector class but with 0 alpha to just capturing states!
                pass # Will do capture below
            
        
        # ---------------------------
        # RUN EXPERIMENT (With Injection + Capture)
        # ---------------------------
        
        # We need to capture states during generation.
        # Let's define a StateCapturer hook.
        
        class StateCapturer:
            def __init__(self):
                self.trace = []
            def hook(self, module, inp, out):
                # Only capture during decode (S=1)
                if isinstance(out, tuple): o = out[0]
                else: o = out
                if o.shape[1] == 1:
                    self.trace.append(o.detach().cpu().squeeze()) # [D]
                return out
        
        # Re-run Baseline with Capture (to be precise)
        capturer_base = StateCapturer()
        h_base = injection_module.register_forward_hook(capturer_base.hook)
        
        input_ids = tokenizer(prompt_revise, return_tensors="pt").to(model.device).input_ids
        model.generate(input_ids, max_new_tokens=60, do_sample=False, use_cache=True)
        h_base.remove()
        base_states = torch.stack(capturer_base.trace) if capturer_base.trace else torch.tensor([])
        
        # Run Steered (Reversible)
        injector = WindowedInjector(
            delta, 
            alpha=args.alpha if args.condition == "steer_reversible" else 0.0,
            decode_window=args.decode_window,
            reversible=(args.condition == "steer_reversible")
        )
        capturer_steer = StateCapturer()
        
        h_inj = injection_module.register_forward_hook(injector.hook)
        h_cap = injection_module.register_forward_hook(capturer_steer.hook)
        
        out_steer = model.generate(input_ids, max_new_tokens=60, do_sample=False, use_cache=True)
        
        h_inj.remove()
        h_cap.remove()
        
        steer_states = torch.stack(capturer_steer.trace) if capturer_steer.trace else torch.tensor([])
        steer_text = tokenizer.decode(out_steer[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # ---------------------------
        # METRICS
        # ---------------------------
        
        # 1. State Distance Trace
        # dist_t = 1 - cos(h_t, h_t_baseline)
        dists = []
        min_len = min(len(base_states), len(steer_states))
        for t in range(min_len):
            v1 = base_states[t]
            v2 = steer_states[t]
            cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            dists.append(1.0 - cos)
            
        # 2. Logit Diff (Approximate, requires running forward again to get logits precisely if we hooked norm)
        # But user asks for "max |Δlogits|". 
        # This is expensive to compute per step unless we hooked lm_head.
        # If hook_point == lm_head, our captured states ARE logits.
        logit_diff = 0.0
        if args.hook_point == "lm_head" and min_len > 0:
            diffs = (base_states[:min_len] - steer_states[:min_len]).abs()
            logit_diff = diffs.max().item()
            
        label = label_revision_behavior(commit_ans, steer_text)
        depth = calculate_revision_depth(steer_text)
        
        res = RunResult(
            sample_id=i,
            condition=args.condition,
            alpha=args.alpha,
            full_text=prompt_revise + steer_text,
            phase_texts={"commit": commit_ans, "revise": steer_text},
            label=label,
            revision_depth=depth,
            logit_diff_max=logit_diff,
            state_dist_trace=dists,
            steer_vec_norm=steer_vec_norm,
            decode_window=args.decode_window
        )
        results.append(res)
        
        # Print progress
        print(f"[Sample {i}] Cond: {args.condition} | Label: {label} | Depth: {depth}")
        if dists:
            # Print average dist during push, pull, and relax phases
            T = args.decode_window
            push_dist = sum(dists[:T])/T if len(dists) >= T else 0
            pull_dist = sum(dists[T:2*T])/T if len(dists) >= 2*T else 0
            relax_dist = sum(dists[2*T:])/len(dists[2*T:]) if len(dists) > 2*T else 0
            print(f"  Distances -> Push: {push_dist:.4f} | Pull: {pull_dist:.4f} | Relax: {relax_dist:.4f}")

    # Save CSV
    if args.output_file:
        with open(args.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "condition", "alpha", "label", "depth", "max_logit_diff", "push_dist_avg", "pull_dist_avg", "relax_dist_avg", "revise_text"])
            for r in results:
                dists = r.state_dist_trace
                T = r.decode_window
                push = sum(dists[:T])/T if len(dists) >= T else 0
                pull = sum(dists[T:2*T])/T if len(dists) >= 2*T else 0
                relax = sum(dists[2*T:])/len(dists[2*T:]) if len(dists) > 2*T else 0
                
                writer.writerow([
                    r.sample_id, r.condition, r.alpha, r.label, r.revision_depth, 
                    f"{r.logit_diff_max:.4f}", f"{push:.4f}", f"{pull:.4f}", f"{relax:.4f}",
                    r.phase_texts["revise"][:100].replace("\n", " ")
                ])
        print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    print("--- Starting Reversibility Probe ---")
    print(f"Condition: {args.condition}")
    print(f"Alpha: {args.alpha}, Window: {args.decode_window}")
    run_experiment(args)
