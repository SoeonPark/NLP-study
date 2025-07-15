# ./compute_perplexity.py

import torch
from utils import device, MAX_PROMPT, MAX_CONTEXT, top_k
from tqdm import tqdm 
from prompts import PROMPT, DEFAULT_CORRECTION_PROMPT, DEFAULT_SELECTION_PROMPT

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

    def build_few_shot_prompt(self, question_type: str, question: str) -> str:
        parts = [PROMPT, "\n\n"]
        if question_type == "교정형":
            parts.append(DEFAULT_CORRECTION_PROMPT)
        else:
            parts.append(DEFAULT_SELECTION_PROMPT)
        parts.append("\n\n")

        # for ex in examples: 
        #     parts.append(f"질문: {ex['question']}\n답변: {ex['answer']}\n\n")
        
        parts.append(f"질문: {question}\n답변: ")
        return "".join(parts)

    def generate_answer(self, prompt: str, max_length: int = 256) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=MAX_PROMPT).to(device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                                     

    def build_context_prompt(self, question_type: str, context: str, question: str, examples: list[dict]) -> str:
        parts = [PROMPT, "\n\n"]
        if question_type == "교정형":
            parts.append(DEFAULT_CORRECTION_PROMPT)
        else:
            parts.append(DEFAULT_SELECTION_PROMPT)
        parts.append("\n\n")

        for ex in examples:
            parts.append(f"질문: {ex['question']}\n답변: {ex['answer']}\n\n")

        parts.append(f"참조 문서: {context}\n\n")

        parts.append(f"질문: {question}\n답변: ")
        return "".join(parts)

    # def compute_baseline_ppl(self, question_type: str, question: str, answer: str, examples: list[dict]) -> float:
    #     prompt = self.build_few_shot_prompt(question_type, question, examples)
    #     return self.compute_perplexity(prompt, answer)

    def compute_baseline_ppl(self, question_type: str, question: str, answer: str, examples: list[dict]) -> tuple[float, str]:
        prompt = self.build_few_shot_prompt(question_type, question, examples)
        ppl = self.compute_perplexity(prompt, answer)
        gen = self.generate_answer(prompt)
        return ppl, gen

    # def compute_context_ppl(self, question_type: str, context: str, question: str, answer: str, examples: list[dict]) -> float:
    #     prompt = self.build_context_prompt(question_type, context, question, examples)
    #     return self.compute_perplexity(prompt, answer)

    def compute_context_ppl(self, question_type: str, question: str, answer: str, examples: list[dict]) -> tuple[float, str]:
        prompt = self.build_context_prompt(question_type, context, question, examples)
        ppl = self.compute_perplexity(prompt, answer)
        gen = self.generate_answer(prompt)
        return ppl, gen
