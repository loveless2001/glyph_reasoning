"""EXPLORATORY: can a residual-stream direction substitute for the glyph mode key?

Directions are mean final-prompt-token residual differences. Tests:
(a) glyph-adapter + deleted-glyph prompt + trained-direction steering;
(b) untrained base + neutral prompt + base elicitation-direction steering;
(c) cross: base + neutral + trained-model direction.
"""
import sys

sys.path.insert(0, "/home/lenovo/projects/glyph_reasoning")
import modal

import modal_phase_marker as mpm

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
app = modal.App("phase-marker-steering-experiment")
PROBE_LAYERS = (6, 13, 20)
PROBE_ALPHAS = (1.0, 2.0)


@app.function(image=mpm.gpu_image, gpu="H100", timeout=14_000, startup_timeout=1_200, volumes=mpm.GPU_VOLUMES)
def steer() -> dict:
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
    final = [r for source in sorted(by_source) for r in by_source[source][:100]]

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, revision=receipt["model_revision"],
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16)
    model.eval()

    GLYPH = _phase_lines(("🜞", "🜆", "🜂", "🜃"))
    DELETED = _phase_lines(("", "", "", ""))

    def render(row, span):
        return PROMPT_TEMPLATE.format(question=row["question"], format_span=span)

    def batches(items, size):
        for start in range(0, len(items), size):
            yield items[start : start + size]

    @torch.no_grad()
    def mean_final_residuals(prompts, adapter_on):
        sums = {layer: None for layer in PROBE_LAYERS}
        count = 0
        for chunk in batches(prompts, 16):
            enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            if adapter_on:
                out = model(**enc, output_hidden_states=True)
            else:
                with model.disable_adapter():
                    out = model(**enc, output_hidden_states=True)
            for layer in PROBE_LAYERS:
                final_tokens = out.hidden_states[layer][:, -1, :].float().sum(dim=0)
                sums[layer] = final_tokens if sums[layer] is None else sums[layer] + final_tokens
            count += len(chunk)
        return {layer: sums[layer] / count for layer in PROBE_LAYERS}

    hooks = []

    def add_steering(layer_index, direction, alpha):
        steered_layer = model.get_base_model().model.layers[layer_index]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + alpha * direction.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        hooks.append(steered_layer.register_forward_hook(hook))

    def clear_steering():
        while hooks:
            hooks.pop().remove()

    @torch.no_grad()
    def score(items, span, adapter_on):
        correct = 0
        for chunk in batches(items, 16):
            enc = tokenizer(
                [render(r, span) for r in chunk],
                return_tensors="pt", padding=True, add_special_tokens=False,
            ).to("cuda")
            if adapter_on:
                out = model.generate(**enc, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            else:
                with model.disable_adapter():
                    out = model.generate(**enc, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, text in zip(chunk, texts):
                record = GenerationRecord(
                    generation_id="steer", source=row["source"], question_hash=row["question_hash"],
                    gold_answer=row["answer"], training_arm="glyph", seed=42, checkpoint=adapter,
                    prompt_condition="steer", prompt_hash="0" * 64, raw_prompt="", raw_completion=text,
                    prompt_token_ids=(), completion_token_ids=(),
                    decoding={"seed": 42, "adapter_seed": 42}, parent_hashes=(),
                )
                correct += 1 if score_generation(record).correct else 0
        return correct / len(items)

    results: dict[str, float] = {}
    trained_matched = mean_final_residuals([render(r, GLYPH) for r in probe], adapter_on=True)
    trained_deleted = mean_final_residuals([render(r, DELETED) for r in probe], adapter_on=True)
    trained_dir = {l: trained_matched[l] - trained_deleted[l] for l in PROBE_LAYERS}
    base_glyph = mean_final_residuals([render(r, GLYPH) for r in probe], adapter_on=False)
    base_neutral = mean_final_residuals([render(r, "") for r in probe], adapter_on=False)
    base_dir = {l: base_glyph[l] - base_neutral[l] for l in PROBE_LAYERS}

    results["probe_ref_adapter_matched"] = score(probe, GLYPH, True)
    results["probe_ref_adapter_deleted"] = score(probe, DELETED, True)
    results["probe_ref_base_neutral"] = score(probe, "", False)

    best = (PROBE_LAYERS[0], PROBE_ALPHAS[0], -1.0)
    for layer in PROBE_LAYERS:
        for alpha in PROBE_ALPHAS:
            add_steering(layer, trained_dir[layer], alpha)
            acc = score(probe, DELETED, True)
            clear_steering()
            results[f"probe_steer_adapter_deleted_L{layer}_a{alpha}"] = acc
            if acc > best[2]:
                best = (layer, alpha, acc)
    layer, alpha, _ = best

    add_steering(layer, trained_dir[layer], alpha)
    results[f"final_steer_adapter_deleted_L{layer}_a{alpha}"] = score(final, DELETED, True)
    clear_steering()
    add_steering(layer, base_dir[layer], alpha)
    results[f"final_steer_base_neutral_basedir_L{layer}_a{alpha}"] = score(final, "", False)
    clear_steering()
    add_steering(layer, trained_dir[layer], alpha)
    results[f"final_steer_base_neutral_traineddir_L{layer}_a{alpha}"] = score(final, "", False)
    clear_steering()
    results["final_ref_adapter_deleted"] = score(final, DELETED, True)
    results["final_ref_base_neutral"] = score(final, "", False)
    return results


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(steer.remote(), sort_keys=True))
