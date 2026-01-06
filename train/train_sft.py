import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

# --------------------
# Environment & Paths
# --------------------
# Redirect cache to workspace to avoid "Disk quota exceeded" on root partition
os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/huggingface_cache/datasets"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "data/sft_final.jsonl"
OUTPUT_DIR = "/workspace/revision_learning/checkpoints/qwen2.5-glyph-sft"

# --------------------
# Load tokenizer & model
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, 
    # device_map="auto" # 🔑 Removed to let Trainer handle device placement
)

# --------------------
# Dataset
# --------------------
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def tokenize(example):
    # Check for empty or malformed messages
    if not example.get("messages") or len(example["messages"]) == 0:
        return {"input_ids": [], "labels": []}
        
    try:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False
        )
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=2048,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs
    except Exception as e:
        print(f"Skipping example due to error: {e}")
        return {"input_ids": [], "labels": []}

dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
# Filter out empty records
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

# --------------------
# Training args
# --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,          
    num_train_epochs=1,
    bf16=True,                   
    fp16=False,
    logging_steps=5,             
    logging_first_step=True,
    save_steps=100,
    save_total_limit=1,          
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=20,
    gradient_checkpointing=True,
    group_by_length=True,
    ddp_find_unused_parameters=False,
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
)

print("🚀 Starting training...")

# Check for existing checkpoint
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint:
    print(f"🔄 Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")
