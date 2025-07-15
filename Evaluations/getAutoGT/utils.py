# ./utils.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "kakaocorp/kanana-1.5-8b-instruct-2505" #"meta-llama/Llama-2-13b-chat-hf" #"skt/kogpt2-base-v2"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

grammar_book = ""
test_data = ""
result_dir = "./results"

MAX_PROMPT = 1024
MAX_CONTEXT = 512
top_k = 5

class KOGPT2:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model


class Llama2_7b:
    def __init__(self, quantize_8bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=quantize_8bit,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            resume_download=True,
            use_auth_token=True,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
