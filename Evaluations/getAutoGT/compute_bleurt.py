import torch
from utils import device, MAX_PROMPT, MAX_CONTEXT, top_k
from tqdm import tqdm
import os
import evaluate

"""
BERTScore: Compute BERTScore of 'Few-Shot Prompt(in dataset)-only' and Answer

Compare 'Baseline' BERTScore with 'Context and Few-shot Prompt' BERTScore
"""

class getBLEURT:
    def __init__(self, model):
        self.tokenizer = model.get_tokenizer()
        self.model = model.get_model()
        
        # Initialize BERTScore scorer (using BERTScore instead of BLEURT)
        self.bertscore_scorer = evaluate.load("bertscore")

    def generate_answer(self, prompt: str, max_length: int = 512) -> str:
        prompt_encoded = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT,
            add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_encoded["input_ids"],
                attention_mask=prompt_encoded["attention_mask"],
                max_length=prompt_encoded["input_ids"].shape[1] + max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][prompt_encoded["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    def compute_bertscore(self, reference: str, candidate: str) -> float:
        results = self.bertscore_scorer.compute(references=[reference], predictions=[candidate], lang="ko")
        return results['f1'][0]

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

    def compute_baseline_bertscore(self, question: str, reference_answer: str, examples: list) -> float:
        prompt = self.build_few_shot_prompt(question, examples)
        generated_answer = self.generate_answer(prompt)
        return self.compute_bertscore(reference_answer, generated_answer)

    def compute_context_bertscore(self, context: str, question: str, reference_answer: str, examples: list) -> float:
        prompt = self.build_context_prompt(context, question, examples)
        generated_answer = self.generate_answer(prompt)
        return self.compute_bertscore(reference_answer, generated_answer)
