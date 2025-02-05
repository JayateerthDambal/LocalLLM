import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import typer
from rich.console import Console
from rich.prompt import Prompt

app = typer.Typer()
console = Console()

dtype = torch.float16

# Loading the tokenizer
console.print("[bold green]Loading deepSeek Model.... [/bold green]")
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
    torch_dtype=dtype
).cuda()

console.print("[bold green]Model Loaded Successfully.... [/bold green]")


# This is to reduce the inital latency at start-up
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


console.print(
    "[bold green]Model warm-up complete. Start chatting below![/bold green]")
console.print(
    "[dim]Type 'exit' or 'quit' at any time to end the session.[/dim]\n")

# Chat Session:


@app.command()
def chat():
    """Start an interactive chat session with the DeepSeekAI model."""
    while True:
        user_input = Prompt.ask("[bold blue]User[/bold blue]")
        if user_input.strip().lower() in {"exit", "quit"}:
            console.print(
                "[bold red]Exiting chat. Goodbye, See you soon mf...![/bold red]")
            break

        messages = [{'role': 'user', 'content': user_input}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        start_time = time.time()
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
        elapsed_time = start_time - time.time()
        response = tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True
        )
        console.print("[bold green]DeepSeekAI[/bold green]:", response)
        console.print(
            "[dim]Response generated in {:.2f} seconds[/dim]".format(elapsed_time))


if __name__ == "__main__":
    app()
