import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelHandler:
    """
    A handler is used to load, warm-up and manage the selected model (For now it is DeepSeekModel)
    This class encapsulates the logic so that both the CLI and Fast API interfaces can use the 
    running LLM Model
    """

    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", dtype=torch.float16):
        self.model_name = model_name
        self.dtype = dtype
        self.tokenizer = None
        self.model = None
        self.initialized = False

    def initialize_model(self):
        """
        Loads the tokenizer and model, if thay are not loaded. Aliong with this it also
        runsa warm-up interference to get the model ready for use and returns a message
        """

        if self.initialized:
            return {
                "status": "already_initialized",
                "message": "Model is already initialized",
            }

        # Laoding the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Loading the model and move it to GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=self.dtype
        ).cuda()

        # Warm-UP Sequence: this runs a quicj dummy call to reduce the first-call latency

        warm_msgs = [{'role': "user", "content": "Hellio"}]
        warmup_inputs = self.tokenizer.apply_chat_template(
            warm_msgs,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.model.device)

        _ = self.model.generate(
            warmup_inputs,
            max_new_tokens=32,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.initialized = True

        return {
            "status": "initialized",
            "message": "Model is initialized and ready to use",
        }

    def get_model(self):
        """
        Returns the loaded model, initializing it if it is not
        """

        if not self.initialized:
            self.initialize_model()

        return self.model

    def get_tokenizer(self):
        """
        Returns the loaded tokenizer, initializing it first if necessary.
        """
        if not self.initialized:
            self.initialize_model()
        return self.tokenizer

    def deinitialize_running_model(self):
        """
        Unloads the model by deleting references
        """
        if self.initialized:
            del self.model
            del self.tokenizer
            # Clear CUDA cache data
            torch.cuda.empty_cache()
            self.initialized = False
            self.model = None
            self.tokenizer = None

            return {
                "status": "deinitialized",
                "message": "Model is deinitialized and unloaded",
            }

        else:
            return {
                "status": "not_initialized",
                "message": "Model is currently not loaded"
            }


# *Creating a Global intance for the model and tokenizer
handler = ModelHandler()
