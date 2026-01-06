import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "data/sft_final.jsonl"   # output of unify_datasets.py
OUTPUT_DIR = "checkpoints/qwen2.5-glyph-sft"

# --------------------
# Load tokenizer & model
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --------------------
# Dataset
# --------------------
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def tokenize(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=2048
    )
    inputs["labels"] = inputs["input_ids"]
    return inputs


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# --------------------
# Training args
# --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,          # 🔑 small
    num_train_epochs=1,          # 🔑 start with 1
    fp16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=50,
)

# --------------------
# Trainer
# --------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
