#!/bin/bash

# Optimized run for 0.5B and 1.5B models on consumer GPUs (e.g. RTX 3060)
# Uses tighter memory controls to prevent OOM
echo "Running 0.5B/1.5B Evaluation on Local GPU..."
python eval/eval_structures_vllm.py \
    --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
    --data data/unified_dataset.jsonl \
    --limit 100 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096

# Optimized run for 7B+ models on High-End GPUs (e.g. A6000)
# Can use more memory and longer context
# Uncomment to run:
# echo "Running 7B Evaluation on A6000..."
# python eval/eval_structures_vllm.py \
#     --models Qwen/Qwen2.5-7B-Instruct \
#     --data data/unified_dataset.jsonl \
#     --limit 50 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 8192
