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

inp = open("data/glyph_traces_filtered.jsonl")
out = open("data/sft_final.jsonl", "w")

for line in tqdm(inp):
    r = json.loads(line)

    # latent glyph
    lp = latent_prompt(r["question"])
    inputs = tokenizer(lp, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        o = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    latent_text = tokenizer.decode(o[0], skip_special_tokens=True)

    out.write(json.dumps({
        "messages": [
            {"role": "user", "content": lp},
            {"role": "assistant", "content": latent_text}
        ]
    }) + "\n")

    # optional natural baseline
    np = natural_prompt(r["question"])
    out.write(json.dumps({
        "messages": [
            {"role": "user", "content": np},
            {"role": "assistant", "content": r["answer"]}
        ]
    }) + "\n")

inp.close()
out.close()
