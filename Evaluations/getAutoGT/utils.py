# ./utils.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-chat-hf" #"skt/kogpt2-base-v2"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

grammar_book = "/home/nlplab/hdd1/soeon/Korean_QA_RAG_2025/data/GrammarBook_structured.json"
test_data = "/home/nlplab/hdd1/soeon/Korean_QA_RAG_2025/data/korean_language_rag_V1.0_dev.json"
few_shot_path = "/home/nlplab/hdd1/soeon/Korean_QA_RAG_2025/data/korean_language_rag_V1.0_train.json"
result_dir = "./results"

MAX_PROMPT = 1024
MAX_CONTEXT = 512
top_k = 10
few_shot_examples = 2

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

            # use_safetensors = False
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model