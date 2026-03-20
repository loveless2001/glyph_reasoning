import os
import shutil
import subprocess
import time
import traceback

import modal

app = modal.App("glyph-reasoning-exp")
volume = modal.Volume.from_name("glyph-reasoning-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", "/root/glyph_reasoning")
)

WORKDIR = "/vol/work"

STAGES = {
    "unify": ["python", "train/unify_datasets.py"],
    "traces": ["python", "train/generate_traces.py"],
    "filter": ["python", "train/filter_traces.py"],
    "rewrite": ["python", "train/rewrite_latent.py"],
    "train": ["python", "train/train_sft.py"],
    "eval": [
        "python",
        "eval/eval_glyph_emergence.py",
        "--model_path",
        "checkpoints/qwen2.5-glyph-sft",
    ],
}

EXPECTED_OUTPUTS = {
    "unify": "data/unified_dataset.jsonl",
    "traces": "data/glyph_traces.jsonl",
    "filter": "data/glyph_traces_filtered.jsonl",
    "rewrite": "data/sft_final.jsonl",
    "train": "checkpoints/qwen2.5-glyph-sft",
    "eval": "eval/glyph_emergence_results.jsonl",
}


def _ensure_workspace() -> None:
    if not os.path.exists(WORKDIR):
        shutil.copytree("/root/glyph_reasoning", WORKDIR)
    os.makedirs(os.path.join(WORKDIR, "logs"), exist_ok=True)


def _validate_output(stage: str) -> None:
    relpath = EXPECTED_OUTPUTS[stage]
    target = os.path.join(WORKDIR, relpath)
    if not os.path.exists(target):
        raise RuntimeError(f"Stage '{stage}' completed but missing output: {relpath}")
    if os.path.isfile(target) and os.path.getsize(target) == 0:
        raise RuntimeError(f"Stage '{stage}' output is empty: {relpath}")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 8,
    startup_timeout=60 * 20,
    volumes={"/vol": volume},
    retries=2,
)
def run_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {sorted(STAGES)}")

    _ensure_workspace()
    os.chdir(WORKDIR)

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/vol/hf_cache")
    env.setdefault("HF_DATASETS_CACHE", "/vol/hf_cache/datasets")
    env.setdefault("OUTPUT_DIR", "/vol/checkpoints/qwen2.5-glyph-sft")

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(WORKDIR, "logs", f"{stage}-{ts}.log")

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"stage={stage}\n")
            log_file.write(f"cmd={' '.join(STAGES[stage])}\n\n")
            log_file.flush()
            subprocess.run(
                STAGES[stage],
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        _validate_output(stage)
        return f"Stage '{stage}' completed. Log: {log_path}"
    except Exception:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== ERROR ===\n")
            log_file.write(traceback.format_exc())
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    stages = (
        ["unify", "traces", "filter", "rewrite", "train", "eval"]
        if stage == "all"
        else [stage]
    )
    for stage_name in stages:
        print(run_stage.remote(stage_name))
