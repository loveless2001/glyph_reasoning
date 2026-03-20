#!/usr/bin/env bash
# Quick bootstrap script for cloud GPU containers.
# Run on a fresh container: bash cloud-gpu-setup-script.sh
# Reads config from cloud-gpu-container-setup.yaml (same directory).
set -euo pipefail

REPO="https://github.com/loveless2001/glyph_reasoning.git"
WORK_DIR="/root/glyph_reasoning"
HF_MODEL="loveless2001/qwen2.5-7b-glyph-sft"
CHECKPOINT_DIR="checkpoints/qwen2.5-7b-glyph-sft-hf"

echo "=== [1/5] System packages ==="
apt-get update -qq && apt-get install -y -qq git python3-pip python3-venv build-essential tmux

echo "=== [2/5] Clone/pull repo ==="
if [ -d "$WORK_DIR" ]; then
  cd "$WORK_DIR" && git pull
else
  git clone "$REPO" "$WORK_DIR"
  cd "$WORK_DIR"
fi

echo "=== [3/5] Python venv + deps ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install vllm -q

echo "=== [4/5] Download fine-tuned model ==="
if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
  huggingface-cli download "$HF_MODEL" --local-dir "$CHECKPOINT_DIR"
else
  echo "Checkpoint already exists, skipping download."
fi

echo "=== [5/5] Verify GPU ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "=== Setup complete ==="
echo "Activate: cd $WORK_DIR && source .venv/bin/activate"
echo "Run ablation: python eval/eval-ablation-shuffled-and-emoji-vllm.py --data data/unified_dataset.jsonl --limit 200"
