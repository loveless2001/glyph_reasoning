"""EXPLORATORY: localize the glyph mode key across all layers and test KV access.

(A) Steering recovery curve: direction per layer (0..27), injected on
    deleted-glyph prompts through the glyph adapter, n=100.
(B) KV-access: process the matched-glyph prompt normally, then mask glyph
    positions from generated tokens only. Survival => key absorbed into
    residuals at prompt time; collapse => key re-read from KV each step.
"""
import sys

sys.path.insert(0, "/home/lenovo/projects/glyph_reasoning")
import modal

import modal_phase_marker as mpm

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
app = modal.App("phase-marker-layer-sweep-kv-access")
GLYPH_BYTE_PREFIX = (9284, 250)


@app.function(image=mpm.gpu_image, gpu="H100", timeout=14_000, startup_timeout=1_200, volumes=mpm.GPU_VOLUMES)
def run_experiments() -> dict:
    import json
    import os
    from pathlib import Path

    import torch

    os.environ["HF_HUB_CACHE"] = "/model-cache/canonical"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(mpm.CODE_ROOT))
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from phase_marker.behavior import _load_pinned_local_tokenizer
    from phase_marker.config import ExperimentConfig
    from phase_marker.prompts import PROMPT_TEMPLATE, _phase_lines
    from phase_marker.schema import GenerationRecord
    from phase_marker.scoring import score_generation

    run_root = Path("/runs/runs") / RUN
    receipt = json.loads((run_root / "receipts/canonical/train/glyph.json").read_text())
    bundle = Path("/inputs/bundles") / str(receipt["bundle_id"])
    config = ExperimentConfig.load(bundle / "configs/phase-marker-qwen25-7b.toml")
    tokenizer = _load_pinned_local_tokenizer(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    adapter = str(run_root / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph/checkpoint-100")

    rows = [
        json.loads(line)
        for line in (bundle / "artifacts/phase-marker/splits/test.jsonl").read_text().splitlines()
        if line
    ]
    by_source: dict[str, list] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    probe = [r for source in sorted(by_source) for r in by_source[source][:34]][:100]

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=receipt["model_revision"],
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16)
    model.eval()
    n_layers = model.config.num_hidden_layers

    GLYPH = _phase_lines(("🜞", "🜆", "🜂", "🜃"))
    DELETED = _phase_lines(("", "", "", ""))

    def render(row, span):
        return PROMPT_TEMPLATE.format(question=row["question"], format_span=span)

    def batches(items, size):
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def is_correct(row, text):
        record = GenerationRecord(
            generation_id="x", source=row["source"], question_hash=row["question_hash"],
            gold_answer=row["answer"], training_arm="glyph", seed=42, checkpoint=adapter,
            prompt_condition="x", prompt_hash="0" * 64, raw_prompt="", raw_completion=text,
            prompt_token_ids=(), completion_token_ids=(),
            decoding={"seed": 42, "adapter_seed": 42}, parent_hashes=(),
        )
        return score_generation(record).correct

    # --- (A) all-layer directions from one hidden-state pass each ---
    @torch.no_grad()
    def mean_final_hidden(prompts):
        sums = [None] * (n_layers + 1)
        count = 0
        for chunk in batches(prompts, 16):
            enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model(**enc, output_hidden_states=True)
            for index, hidden in enumerate(out.hidden_states):
                final = hidden[:, -1, :].float().sum(dim=0)
                sums[index] = final if sums[index] is None else sums[index] + final
            count += len(chunk)
        return [total / count for total in sums]

    matched_means = mean_final_hidden([render(r, GLYPH) for r in probe])
    deleted_means = mean_final_hidden([render(r, DELETED) for r in probe])
    directions = [m - d for m, d in zip(matched_means, deleted_means)]

    hooks = []

    def add_steering(layer_index, direction):
        steered_layer = model.get_base_model().model.layers[layer_index]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + direction.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        hooks.append(steered_layer.register_forward_hook(hook))

    def clear_steering():
        while hooks:
            hooks.pop().remove()

    @torch.no_grad()
    def score_generate(items, span, max_new=768):
        correct = 0
        for chunk in batches(items, 16):
            enc = tokenizer([render(r, span) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            correct += sum(1 for row, text in zip(chunk, texts) if is_correct(row, text))
        return correct / len(items)

    results: dict[str, float] = {}
    results["ref_matched"] = score_generate(probe, GLYPH)
    results["ref_deleted"] = score_generate(probe, DELETED)
    sweep = {}
    for layer in range(1, n_layers + 1, 2):  # every other layer fits the timeout
        add_steering(layer - 1, directions[layer])
        sweep[str(layer - 1)] = score_generate(probe, DELETED)
        print(f"sweep layer {layer - 1}: {sweep[str(layer - 1)]:.3f}", flush=True)
        clear_steering()
    results_sweep = sweep

    # --- (B) KV-access: manual greedy loop masking glyph positions for new tokens ---
    @torch.no_grad()
    def score_kv_masked(items, max_new=768):
        correct = 0
        for chunk in batches(items, 16):
            enc = tokenizer([render(r, GLYPH) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            ids, mask = enc["input_ids"], enc["attention_mask"]
            glyph_positions = torch.zeros_like(ids, dtype=torch.bool)
            for b in range(ids.shape[0]):
                seq = ids[b].tolist()
                for i in range(len(seq) - 1):
                    if seq[i] == GLYPH_BYTE_PREFIX[0] and seq[i + 1] == GLYPH_BYTE_PREFIX[1]:
                        glyph_positions[b, i : i + 3] = True
            out = model(input_ids=ids, attention_mask=mask, use_cache=True)
            past = out.past_key_values
            next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_ids]
            gen_mask = mask.clone()
            gen_mask[glyph_positions] = 0  # invisible to all future queries
            finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
            for _ in range(max_new - 1):
                gen_mask = torch.cat([gen_mask, torch.ones_like(next_ids)], dim=1)
                out = model(input_ids=next_ids, attention_mask=gen_mask, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                finished |= next_ids.squeeze(1) == tokenizer.eos_token_id
                generated.append(next_ids)
                if bool(finished.all()):
                    break
            texts = tokenizer.batch_decode(torch.cat(generated, dim=1), skip_special_tokens=True)
            correct += sum(1 for row, text in zip(chunk, texts) if is_correct(row, text))
        return correct / len(items)

    results["kv_masked_after_prompt"] = score_kv_masked(probe)
    return {"results": results, "layer_sweep": results_sweep}


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run_experiments.remote(), sort_keys=True))
