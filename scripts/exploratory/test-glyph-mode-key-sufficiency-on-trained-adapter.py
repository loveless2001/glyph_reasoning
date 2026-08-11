"""EXPLORATORY: mode-key sufficiency tests on the glyph-trained adapter.

Conditions: matched glyph baseline; single trained glyph; all-positions
uniform trained glyph; uniform UNTRAINED alchemical glyph sharing the byte
prefix. Discriminates exact-byte mode key vs prefix key vs phase structure.
"""
import sys
sys.path.insert(0, "/home/lenovo/projects/glyph_reasoning")
import modal
import modal_phase_marker as mpm

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
app = modal.App("phase-marker-mode-key-sufficiency")

CONDITIONS = {
    "matched_glyph": ("🜞", "🜆", "🜂", "🜃"),
    "single_first": ("🜞", "", "", ""),
    "uniform_trained": ("🜞", "🜞", "🜞", "🜞"),
    "uniform_unseen_alchemical": ("🜁", "🜁", "🜁", "🜁"),
}


@app.function(image=mpm.gpu_image, gpu="H100", timeout=10_800, startup_timeout=1_200, volumes=mpm.GPU_VOLUMES)
def score_conditions() -> dict:
    import json
    import os
    from collections import defaultdict
    from pathlib import Path

    os.environ["HF_HUB_CACHE"] = "/model-cache/canonical"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(mpm.CODE_ROOT))
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
    adapter = str(run_root / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph/checkpoint-100")
    examples = [
        json.loads(line)
        for line in (bundle / "artifacts/phase-marker/splits/test.jsonl").read_text().splitlines()
        if line
    ]

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=config.model_id,
        revision=receipt["model_revision"],
        tokenizer_revision=receipt["model_revision"],
        enable_lora=True,
    )
    lora = LoRARequest("glyph-checkpoint-100-seed-42", 42, adapter)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, seed=42)

    results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, markers in CONDITIONS.items():
        prompts = [
            TokensPrompt(prompt_token_ids=list(tokenizer.encode(
                PROMPT_TEMPLATE.format(
                    question=row["question"],
                    format_span=_phase_lines(markers),
                ),
                add_special_tokens=False,
            )))
            for row in examples
        ]
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False, lora_request=lora)
        for row, output in zip(examples, outputs, strict=True):
            record = GenerationRecord(
                generation_id=f"modekey:{name}:{row['example_id']}",
                source=row["source"], question_hash=row["question_hash"],
                gold_answer=row["answer"], training_arm="glyph", seed=42,
                checkpoint=adapter, prompt_condition=name, prompt_hash="0" * 64,
                raw_prompt="", raw_completion=output.outputs[0].text,
                prompt_token_ids=(), completion_token_ids=tuple(output.outputs[0].token_ids),
                decoding={"seed": 42, "adapter_seed": 42}, parent_hashes=(),
            )
            results[name][0] += 1
            results[name][1] += 1 if score_generation(record).correct else 0
    return dict(results)


@app.local_entrypoint()
def main() -> None:
    import json
    print(json.dumps(score_conditions.remote(), sort_keys=True))
