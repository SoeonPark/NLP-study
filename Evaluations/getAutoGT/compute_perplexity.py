# ./compute_perplexity.py
import torch
from utils import device, MAX_PROMPT, MAX_CONTEXT, top_k
from tqdm import tqdm 

"""
Baseline: Compute perplexity of 'Few-Shot Prompt(in dataset)-only' and Answer

Compare 'Baseline' perplexity with 'Context and Few-shot Prompt' perplexity
"""

class getPerplexity:
    def __init__(self, model):
        self.tokenizer = model.get_tokenizer()
        self.model = model.get_model()

    def compute_perplexity(self, prompt: str, answer: str) -> float:
        """
        Compute the perplexity of a given prompt and answer pair.
        """
        prompt_encoded = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT,
            add_special_tokens=False
        ).to(device)

        answer_encoded = self.tokenizer(
            answer, 
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT,
            add_special_tokens=False
        ).to(device)

        input_ids = torch.cat([prompt_encoded["input_ids"], answer_encoded["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_encoded["attention_mask"], answer_encoded["attention_mask"]], dim=1)

        labels = input_ids.clone()
        labels[:, :prompt_encoded["input_ids"].shape[1]] = -100

        with torch.no_grad():
            outputs = self.model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        
        loss = outputs.loss
        return torch.exp(loss).item()

    def build_few_shot_prompt(self, question: str, examples: list) -> str:
        few_shots = []

        for ex in examples:
            few_shots.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n")

        few_shots.append(f"Question: {question}\nAnswer: ")

        return "".join(few_shots)

    def build_context_prompt(self, context: str, question: str, examples: list) -> str:
        few_shots = []

        for ex in examples:
            few_shots.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n")

        few_shots.append(f"Context: {context}\nQuestion: {question}\nAnswer: ")

        return "".join(few_shots)

    def compute_baseline_ppl(self, question: str, answer:str, examples: list) -> float:
        """
        Compute the perplexity using the baseline method with few-shot examples.
        """
        prompt = self.build_few_shot_prompt(question, examples)
        return self.compute_perplexity(prompt, answer)

    def compute_context_ppl(self, context: str, question: str, answer: str, examples: list) -> float:
        """
        Compute the perplexity using the context-aware method with few-shot examples.
        """
        prompt = self.build_context_prompt(context, question, examples)
        return self.compute_perplexity(prompt, answer)
