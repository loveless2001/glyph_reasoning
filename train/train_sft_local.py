import argparse
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
from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct") 
    parser.add_argument("--data_file", type=str, default="data/sft_final.jsonl") 
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen2.5-0.5b-glyph-sft")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--batch_size", type=int, default=1) # Safer default for local
    parser.add_argument("--grad_acc", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=2e-4) # Higher for LoRA
    args = parser.parse_args()

    # --------------------
    # Environment & Paths
    # --------------------
    # Redirect cache to workspace ONLY if on RunPod, otherwise use default
    if os.path.exists("/workspace"):
        os.environ["HF_HOME"] = "/workspace/huggingface_cache"
        os.environ["HF_DATASETS_CACHE"] = "/workspace/huggingface_cache/datasets"
    
    # Ensure output dir exists locally
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    print(f"Using LoRA: {args.use_lora}")

    # --------------------
    # Load tokenizer & model
    # --------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
    
    # For a 6GB GPU, we use float16 and potentially device_map="auto" 
    # if we need to offload, though 0.5B should fit in 6GB.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16, 
        device_map="auto" if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 8e9 else None
    )

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, # 🔑 Reduced from 16 for 6GB VRAM
            lora_alpha=16, 
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # --------------------
    # Dataset
    # --------------------
    dataset = load_dataset("json", data_files=args.data_file, split="train")

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
                max_length=1024, # 🔑 Reduced from 2048 for 6GB VRAM
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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=1,
        fp16=True, # 🔑 Use fp16 for RTX 3060 6GB
        bf16=False,
        logging_steps=5,
        logging_first_step=True,
        save_steps=200,
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=20,
        remove_unused_columns=False,
        gradient_checkpointing=True, # 🔑 Always use checkpointing on 6GB
        group_by_length=True,
    )

    # --------------------
    # Trainer
    # --------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
    )

    print("🚀 Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
