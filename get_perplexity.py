import json
import logging
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.nn import CrossEntropyLoss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_id = "skt/kogpt2-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

def compute_ppl(context: str, question: str, answer: str) -> float:
    text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

    return torch.exp(loss).item()

def compute_question_only_ppl(question: str, answer: str) -> float:
    text = f"Question: {question}\nAnswer: {answer}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return torch.exp(outputs.loss).item()

def compute_few_shot_ppl(question: str, answer: str, examples: list) -> float:
    prompt = ""

    for ex in examples:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer: {answer}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

def compute_context_few_shot_ppl(context: str, question: str, answer: str, examples: list) -> float:
    base_example = {
        "question": '다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"오늘은 퍼즐 마추기를 해 볼 거예요."',
        "answer": '"오늘은 퍼즐 맞추기를 해 볼 거예요."가 옳다. ...'
    }
    prompt = f"Question: {base_example['question']}\nAnswer: {base_example['answer']}\n\n"

    for ex in examples:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

grammar_book = "/home/nlplab/hdd1/soeon/Korean_QA_RAG_2025/data/GrammarBook_structured.json"
test_data = "/home/nlplab/hdd1/soeon/Korean_QA_RAG_2025/data/korean_language_rag_V1.0_dev.json"

with open(grammar_book, "r", encoding="utf-8") as f:
    rules = json.load(f)

with open(test_data, "r", encoding="utf-8") as f:
    qas = json.load(f)

print(f"# of QAs: {len(qas)} / # of Rules: {len(rules)}")

few_shot_examples = [
    {"question": qas[i]["input"]["question"], "answer": qas[i]["output"]["answer"]}
    for i in range(min(2, len(qas)))
]

base_ppl_results = []
few_shot_ppl_results = []
context_few_shot_results= []
ranking_results = []

top_k = 3

for qa in tqdm(qas, desc="QAs"):
    qid = qa["id"]
    question = qa["input"]["question"]
    answer = qa["output"]["answer"]

    base_ppl = compute_question_only_ppl(question, answer)
    base_ppl_results.append({
        "QID": qid,
        "Question": question,
        "Answer": answer,
        "Base PPL": np.round(base_ppl, 2)
    })

    fs_ppl = compute_few_shot_ppl(question, answer, few_shot_examples)
    few_shot_ppl_results.append({
        "QID": qid,
        "Question": question,
        "Answer": answer,
        "Few-shot Examples": few_shot_examples,
        "Few-shot PPL": np.round(fs_ppl, 2)
    })

    cf_ppl = compute_context_few_shot_ppl(rules[0]["description"], question, answer, few_shot_examples)
    context_few_shot_results.append({
        "QID": qid,
        "Question": question,
        "Answer": answer,
        "Context": rules[0]["description"],
        "Context Few-shot PPL": np.round(cf_ppl, 2)
    })

    perps = []
    for rule in rules:
        ppl = compute_ppl(rule["description"], question, answer)
        perps.append((
            rule["rule_id"], 
            rule["category"], 
            rule["title"], 
            ppl
        ))
    perps.sort(key=lambda x: x[3])

    ranking_results.append({
        "QID":  qid,
        "Top_k": [
            {
                "rank": i+1, 
                "rule_id": rid, 
                "category": cat, 
                "title": title, 
                "PPL": np.round(ppl,2)
            }
            for i, (rid, cat, title, ppl) in enumerate(perps[:top_k])
        ]
    })
å
with open("base_ppl.json", "w", encoding="utf-8") as f:
    json.dump(base_ppl_results, f, ensure_ascii=False, indent=4)

with open("few_shot_ppl.json", "w", encoding="utf-8") as f:
    json.dump(few_shot_ppl_results, f, ensure_ascii=False, indent=4)

with open("context_few_shot_ppl.json", "w", encoding="utf-8") as f:
    json.dump(context_few_shot_results, f, ensure_ascii=False, indent=4)

with open("ppl_ranking.json", "w", encoding="utf-8") as f:
    json.dump(ranking_results, f, ensure_ascii=False, indent=4)

print("Results saved successfully!")
