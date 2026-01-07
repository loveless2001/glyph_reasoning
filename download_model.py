from huggingface_hub import snapshot_download
import os

repo_id = "loveless2001/qwen2.5-7b-glyph-sft"
local_dir = "checkpoints/qwen2.5-7b-glyph-sft-hf"

# Only download essential files for inference
allow_patterns = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.jinja"
]

# Exclude large optimizer states and checkpoints
ignore_patterns = [
    "checkpoint-*",
    "*.pt",
    "*.pth",
    "*.bin"
]

print(f"Downloading model from {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
    local_dir_use_symlinks=False
)
print("Download complete.")
