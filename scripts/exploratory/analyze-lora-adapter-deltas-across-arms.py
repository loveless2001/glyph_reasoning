"""Weight-space comparison of the six matched LoRA adapters (CPU, on-volume)."""
import sys
sys.path.insert(0, "/home/lenovo/projects/glyph_reasoning")
import modal
import modal_phase_marker as mpm

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
ARMS = ("semantic", "glyph", "dot", "random", "direct", "filler")
app = modal.App("phase-marker-lora-delta-analysis")


@app.function(image=mpm.gpu_image, volumes=mpm.GPU_VOLUMES, timeout=1800, cpu=8.0, memory=16384)
def analyze() -> dict:
    from collections import defaultdict
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    base = Path("/runs/runs") / RUN / "artifacts/phase-marker/checkpoints/pilot/seed-42"
    weights = {
        arm: load_file(str(base / arm / "checkpoint-100" / "adapter_model.safetensors"))
        for arm in ARMS
    }
    scaling = 2.0  # lora_alpha 32 / r 16

    # Frobenius norm of each module's delta via tr((B^T B)(A A^T)) — 16x16 only.
    def module_pairs(state):
        pairs = {}
        for key, tensor in state.items():
            if key.endswith("lora_A.weight"):
                pairs.setdefault(key[: -len(".lora_A.weight")], {})["A"] = tensor.float()
            elif key.endswith("lora_B.weight"):
                pairs.setdefault(key[: -len(".lora_B.weight")], {})["B"] = tensor.float()
        return pairs

    norms = {}
    layer_profiles = {}
    for arm in ARMS:
        pairs = module_pairs(weights[arm])
        per_layer = defaultdict(float)
        total = 0.0
        for name, ab in pairs.items():
            gram = (ab["B"].T @ ab["B"]) @ (ab["A"] @ ab["A"].T)
            sq = float(torch.trace(gram)) * scaling**2
            total += sq
            layer = next((part for part in name.split(".") if part.isdigit()), "?")
            per_layer[int(layer)] += sq
        norms[arm] = round(total**0.5, 3)
        profile = sorted(per_layer.items())
        layer_profiles[arm] = [round(value**0.5, 3) for _, value in profile]

    # Pairwise subspace overlap: mean cosine of principal angles between the
    # column spaces of B, averaged over all modules.
    def overlap(state_a, state_b):
        pa, pb = module_pairs(state_a), module_pairs(state_b)
        values = []
        for name in pa:
            qa, _ = torch.linalg.qr(pa[name]["B"])
            qb, _ = torch.linalg.qr(pb[name]["B"])
            values.append(float(torch.linalg.svdvals(qa.T @ qb).mean()))
        return round(sum(values) / len(values), 4)

    reasoning = ("semantic", "glyph", "dot", "random")
    overlaps = {
        f"{a}~{b}": overlap(weights[a], weights[b])
        for i, a in enumerate(reasoning)
        for b in reasoning[i + 1 :]
    }
    return {"frobenius_norms": norms, "layer_profiles": layer_profiles, "subspace_overlaps": overlaps}


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(analyze.remote()))
