from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DeepSeekCoder:
    def __init__(self):
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True,
        )


    def generate(self, messages: list, max_new_tokens: int = 512):
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            return self.tokenizer.decode(
                outputs[0][len(inputs[0]):],
                skip_special_tokens=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")