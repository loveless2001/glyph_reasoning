# Glyph Emergence Training Pipeline

This directory contains the scripts required to train a model to exhibit "Promptless Glyph Emergence" — the ability to use structured reasoning glyphs (`🜞`, `🜆`, `🜂`, `🜃`) selectively on hard problems without being explicitly prompted to do so.

## 📋 Prerequisites

Ensure you have the necessary dependencies installed:

```bash
pip install -r ../requirements.txt
```

**Recommended Hardware:** 1x NVIDIA A100 (80GB) or equivalent.
**Time Estimate:** ~4-6 hours total (Data Gen + Training).

## 🚀 Workflow

Run the following scripts in order from the project root.

### 1. Unify Datasets
Aggregates GSM8K, SVAMP, and MATH datasets into a single standard format.
- **Input**: HuggingFace Datasets
- **Output**: `data/unified_dataset.jsonl`

```bash
python train/unify_datasets.py
```

### 2. Generate Glyph Traces
Uses a strong teacher model (or the model itself in teacher-mode) to generated reasoning traces using the specific glyph structure.
- **Input**: `data/unified_dataset.jsonl`
- **Output**: `data/glyph_traces.jsonl`

```bash
python train/generate_traces.py
```

### 3. Filter Traces
Filters the generated traces for correctness (answer match) and structural validity (all glyphs present).
- **Input**: `data/glyph_traces.jsonl`
- **Output**: `data/glyph_traces_filtered.jsonl`

```bash
python train/filter_traces.py
```

### 4. Rewrite & Prepare Training Data
Prepares the final SFT dataset. This step performs two critical actions:
1.  **Latent Reformatting**: Converts the glyph trace into a "latent" training example where the model sees a *clean* prompt but is trained to output the glyphs.
2.  **Natural Baseline**: Adds standard Q&A pairs to maintain general ability.
- **Input**: `data/glyph_traces_filtered.jsonl`
- **Output**: `data/sft_final.jsonl`

```bash
python train/rewrite_latent.py
```

### 5. Train (Supervised Fine-Tuning)
Fine-tunes the base model (Qwen 2.5 7B) on the prepared dataset.
- **Input**: `data/sft_final.jsonl`
- **Output**: `checkpoints/qwen2.5-glyph-sft`

```bash
python train/train_sft.py
```

## 📊 Evaluation

After training, evaluate the model's emergence capabilities:

```bash
python eval/eval_glyph_emergence.py --model_path checkpoints/qwen2.5-glyph-sft
```

## 🛠 Directory Structure

- `unify_datasets.py`: Dataset aggregation.
- `prompts.py`: Centralized prompt definitions (Glyph, Latent, Natural).
- `generate_traces.py`: Generation loop.
- `filter_traces.py`: Quality control.
- `rewrite_latent.py`: SFT formatting.
- `train_sft.py`: HuggingFace Trainer script.
