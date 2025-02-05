import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use torch.float16 (or torch.bfloat16 if needed) for optimal performance on your RTX 3060
dtype = torch.float16

# Load the tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
    torch_dtype=dtype
).cuda()

# Optional: Warm-up inference to reduce the initial latency.
warmup_messages = [{'role': 'user', 'content': "Hello"}]
warmup_inputs = tokenizer.apply_chat_template(
    warmup_messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
_ = model.generate(
    warmup_inputs,
    max_new_tokens=32,
    do_sample=False,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
print("Model warm-up complete. You can now start chatting. Type 'exit' or 'quit' to stop.\n")

# Interactive chat loop
while True:
    user_input = input("User:-> ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break

    # Create the prompt message
    messages = [{'role': 'user', 'content': user_input}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate the output from the model
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print("DeepSeekAI:-> ", response)

print()
