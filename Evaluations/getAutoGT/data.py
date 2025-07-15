# ./data.py

import json
from utils import grammar_book, test_data, device, few_shot_examples

class DataLoader:
    def __init__(self):
        self.rules = []
        self.qas = []

    def load_data(self):
        with open(grammar_book, "r", encoding="utf-8") as f:
            self.rules = json.load(f)

        with open(test_data, "r", encoding="utf-8") as f:
            self.qas = json.load(f)

    def get_rules(self):
        return self.rules
    
    def get_qas(self):
        return self.qas
