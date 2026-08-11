"""EXPLORATORY (not preregistered): score the UNTRAINED Qwen2.5-7B-Instruct
on the frozen test set under all four prompt conditions.

Reuses the frozen prompt renderer, tokenizer snapshot, decoding settings
(greedy, 1024 tokens), and answer scorer, but loads no adapter. Answers
whether glyph prompts help the base model zero-shot.
"""
import sys

sys.path.insert(0, "/home/lenovo/projects/glyph_reasoning")

import modal

import modal_phase_marker as mpm

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
CONDITIONS = ("neutral", "glyph", "dot", "headings")

app = modal.App("phase-marker-base-model-exploratory")


@app.function(
    image=mpm.gpu_image,
    gpu="H100",
    timeout=10_800,
    startup_timeout=1_200,
    volumes=mpm.GPU_VOLUMES,
)
def score_base_model() -> dict:
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
    from phase_marker.prompts import MarkerSet, render_prompt
    from phase_marker.schema import GenerationRecord
    from phase_marker.scoring import score_generation

    run_root = Path("/runs/runs") / RUN
    receipt = json.loads(
        (run_root / "receipts" / "canonical" / "train" / "glyph.json").read_text()
    )
    bundle = Path("/inputs/bundles") / str(receipt["bundle_id"])
    config = ExperimentConfig.load(bundle / "configs/phase-marker-qwen25-7b.toml")
    marker_set = MarkerSet(*config.phase_markers)
    tokenizer = _load_pinned_local_tokenizer(config.model_id)

    examples = [
        json.loads(line)
        for line in (bundle / "artifacts/phase-marker/splits/test.jsonl").read_text().splitlines()
        if line
    ]

    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(
        model=config.model_id,
        revision=receipt["model_revision"],
        tokenizer_revision=receipt["model_revision"],
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, seed=42)

    results: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for condition in CONDITIONS:
        prompts = [
            TokensPrompt(
                prompt_token_ids=list(
                    tokenizer.encode(
                        render_prompt(row["question"], condition, marker_set),
                        add_special_tokens=False,
                    )
                )
            )
            for row in examples
        ]
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        for row, output in zip(examples, outputs, strict=True):
            record = GenerationRecord(
                generation_id=f"base:{condition}:{row['example_id']}",
                source=row["source"],
                question_hash=row["question_hash"],
                gold_answer=row["answer"],
                training_arm="base",
                seed=42,
                checkpoint="base://untrained",
                prompt_condition=condition,
                prompt_hash="0" * 64,
                raw_prompt="",
                raw_completion=output.outputs[0].text,
                prompt_token_ids=(),
                completion_token_ids=tuple(output.outputs[0].token_ids),
                decoding={"seed": 42, "adapter_seed": 42},
                parent_hashes=(),
            )
            score = score_generation(record)
            key = f"{condition}|{row['source']}"
            results[key][0] += 1
            results[key][1] += 1 if score.correct else 0
            results[condition][0] += 1
            results[condition][1] += 1 if score.correct else 0
    return dict(results)


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(score_base_model.remote(), sort_keys=True))
