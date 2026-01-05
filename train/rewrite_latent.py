import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import latent_prompt, natural_prompt

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

inp = open("data/glyph_traces_filtered.jsonl").readlines()
out = open("data/sft_final.jsonl", "w")

BATCH_SIZE = 8
tokenizer.pad_token = tokenizer.eos_token

# Helper to process a batch
def process_batch(batch_lines):
    batch_data = [json.loads(line) for line in batch_lines]
    prompts = [latent_prompt(r["messages"][1]["content"]) if "messages" in r else latent_prompt(r["question"]) for r in batch_data]
    
    # Handle schema var: check if filtered traces have different structure.
    # Based on previous steps, filtered traces have "messages" and "question"/"answer" fields.
    # The latent_prompt expects a question.
    prompts = [latent_prompt(r["question"]) for r in batch_data]

    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        # Generate
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode batch
    # outputs includes input tokens, so we slice them off
    input_lens = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_lens:]
    decoded_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for r, lp, latent_text in zip(batch_data, prompts, decoded_texts):
        # Latent entry
        out.write(json.dumps({
            "messages": [
                {"role": "user", "content": lp},
                {"role": "assistant", "content": latent_text}
            ]
        }) + "\n")

        # Natural baseline entry
        np = natural_prompt(r["question"])
        out.write(json.dumps({
            "messages": [
                {"role": "user", "content": np},
                {"role": "assistant", "content": r["answer"]}
            ]
        }) + "\n")

# Main loop
for i in tqdm(range(0, len(inp), BATCH_SIZE)):
    batch = inp[i : i + BATCH_SIZE]
    process_batch(batch)

out.close()
