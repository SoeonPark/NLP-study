# ./data_loader.py
import json
from utils import grammar_book, test_data, few_shot_path, device, few_shot_examples

class DataLoader:
    def __init__(self):
        self.rules = []
        self.qas = []
        self.few_shot_examples = []

    def load_data(self):
        with open(grammar_book, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

        with open(test_data, "r", encoding="utf-8") as f:
            self.qas = json.load(f)

        with open(few_shot_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        for ex in examples[:few_shot_examples]:
            self.few_shot_examples.append({
                "question": ex["input"]["question"] if "input" in ex else ex["question"],
                "answer": ex["output"]["answer"] if "output" in ex else ex["answer"]
            })

    def get_rules(self):
        return self.rules
    
    def get_qas(self):
        return self.qas
    
    def get_few_shot_examples(self):
        return self.few_shot_examples