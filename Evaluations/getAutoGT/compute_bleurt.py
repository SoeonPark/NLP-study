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

# class ContextAnalyze:
#     def __init__(self, getBLEURT):
#         self.getBLEURT = getBLEURT

#     def find_best_context(self, question: str, reference_answer: str, rules: list, examples: list, top_k: int=top_k) -> tuple:
#         baseline_bertscore = self.getBLEURT.compute_baseline_bertscore(question, reference_answer, examples)

#         context_results = []

#         print("Computing context BERTScore...")
#         for r in tqdm(rules):
#             context_bertscore = self.getBLEURT.compute_context_bertscore(r["description"], question, reference_answer, examples)

#             delta_bertscore = context_bertscore - baseline_bertscore
#             if baseline_bertscore > 0:
#                 improvement_ratio = delta_bertscore / baseline_bertscore
#             else:
#                 improvement_ratio = 0
            
#             context_results.append({
#                 "rule_id": r["rule_id"],
#                 "category": r["category"],
#                 "title": r["title"],
#                 "description": r["description"],
#                 "context_bertscore": context_bertscore,
#                 "baseline_bertscore": baseline_bertscore,
#                 "delta_bertscore": delta_bertscore,
#                 "improvement_ratio": improvement_ratio
#             })

#         # Sort contexts by high BERTScore (higher is better)
#         context_results_by_bertscore = sorted(context_results, key = lambda x: x["context_bertscore"], reverse=True)
#         highest_bertscore_contexts = context_results_by_bertscore[:top_k]

#         # Sort contexts by high delta BERTScore (which means the context improved the most)
#         positive_delta_results = [r for r in context_results if r["delta_bertscore"] > 0]
#         context_results_by_delta = sorted(positive_delta_results, key = lambda x: x["delta_bertscore"], reverse=True)
#         highest_delta_contexts = context_results_by_delta[:top_k]

#         return highest_bertscore_contexts, highest_delta_contexts, baseline_bertscore