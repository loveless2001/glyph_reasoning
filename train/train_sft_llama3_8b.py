import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA SFT for Meta-Llama-3-8B-Instruct on glyph reasoning data."
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--data_file", default="data/sft_final.jsonl")
    parser.add_argument(
        "--output_dir",
        default="/workspace/checkpoints/llama3-8b-glyph-sft",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def ensure_env():
    if os.path.exists("/workspace"):
        os.environ.setdefault("HF_HOME", "/workspace/huggingface_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/huggingface_cache/datasets")

    has_token = os.environ.get("HF_TOKEN") or os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token")
    )
    if not has_token:
        raise RuntimeError(
            "Meta-Llama-3-8B-Instruct is gated. Set HF_TOKEN or run huggingface-cli login."
        )


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        print("flash-attn not installed, using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def load_and_tokenize_dataset(tokenizer, data_file, max_seq_length):
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Raw examples: {len(dataset)}")

    def tokenize(example):
        if not example.get("messages"):
            return {"input_ids": [], "labels": []}
        try:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            inputs = tokenizer(text, truncation=True, max_length=max_seq_length)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
        except Exception as exc:
            print(f"Skipping example due to error: {exc}")
            return {"input_ids": [], "labels": []}

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda row: len(row["input_ids"]) > 0)
    print(f"Tokenized examples: {len(dataset)}")

    lengths = [len(row["input_ids"]) for row in dataset]
    print(
        f"Seq lengths: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.0f}"
    )
    return dataset


def build_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=True,
        fp16=False,
        logging_steps=5,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=True,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def main():
    args = parse_args()
    ensure_env()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, args)
    dataset = load_and_tokenize_dataset(tokenizer, args.data_file, args.max_seq_length)
    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )

    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
