import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "checkpoints/qwen2.5-7b-glyph-sft"

print(f"Loading base model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

messages = [
    {"role": "user", "content": "Solve the following problem carefully.\nDo not mention any tags, symbols, or special formatting.\n\nProblem:\nIf I have 3 apples and eat one, how many do I have?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("-" * 20)
print(response)
print("-" * 20)
