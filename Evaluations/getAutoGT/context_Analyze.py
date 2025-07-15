# ./context_Analyze.py

from abc import ABC, abstractmethod
from tqdm import tqdm

class MetricComputer(ABC):
    
    @abstractmethod
    def compute_baseline(self, question_type: str, question: str, answer: str):
        pass
    
    @abstractmethod
    def compute_context_score(self, context: str, question: str, answer: str, examples: list):
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        pass
    
    @abstractmethod
    def is_higher_better(self) -> bool:
        pass

class PerplexityComputer(MetricComputer):
    def __init__(self, getPerplexity):
        self.getPPL = getPerplexity

    def compute_baseline(self, question_type: str, question: str, answer: str, examples: list[dict]):
        return self.getPPL.compute_baseline_ppl(question_type, question, answer, examples)

    def compute_context_score(self, question_type: str, context: str,question: str, answer: str, examples: list[dict]):
        return self.getPPL.compute_context_ppl(question_type, context, question, answer, examples)

    def get_metric_name(self) -> str:
        return "ppl"

    def is_higher_better(self) -> bool:
        return False

class BERTScoreComputer(MetricComputer):
    def __init__(self, getBLEURT):
        self.getBLEURT = getBLEURT
    
    def compute_baseline(self, question: str, answer: str, examples: list):
        return self.getBLEURT.compute_baseline_bertscore(question, answer, examples)
    
    def compute_context_score(self, context: str, question: str, answer: str, examples: list):
        return self.getBLEURT.compute_context_bertscore(context, question, answer, examples)
    
    def get_metric_name(self) -> str:
        return "bertscore"
    
    def is_higher_better(self) -> bool:
        return True


class ROUGE1Computer(MetricComputer):
    def __init__(self, getROUGE1):
        self.getROUGE1 = getROUGE1
    
    def compute_baseline(self, question: str, answer: str, examples: list):
        return self.getROUGE1.compute_baseline_rouge1(question, answer, examples)
    
    def compute_context_score(self, context: str, question: str, answer: str, examples: list):
        return self.getROUGE1.compute_context_rouge1(context, question, answer, examples)
    
    def get_metric_name(self) -> str:
        return "rouge1"
    
    def is_higher_better(self) -> bool:
        return True


class ContextAnalyze:
    def __init__(self, metric_computer: MetricComputer):
        self.metric_computer = metric_computer
    
    def find_best_context(self, question_type: str, question: str, answer: str, rules: list, top_k: int = 5) -> tuple:

        context_ppl, context_gen = self.metric_computer.compute_context_score(question_type, r["description"], question, answer, examples)

        if isinstance(self.metric_computer, PerplexityComputer):
            baseline_ppl, baseline_gen = self.metric_computer.compute_baseline(question_type, question, answer, examples)
        else:
            baseline_ppl, _ = baseline_score
            baseline_gen = None

        metric_name = self.metric_computer.get_metric_name()
        is_higher_better = self.metric_computer.is_higher_better()
        
        context_results = []
        
        print(f"Computing context {metric_name.upper()} scores...")
        for r in tqdm(rules):
            context_score = self.metric_computer.compute_context_score(question_type, r["description"], question, answer)
            if isinstance(self.metric_computer, PerplexityComputer):
                context_ppl, context_gen = context_out
            else:
                context_ppl, _ = context_out
                context_gen = None
            
            if is_higher_better:
                delta_score = context_score - baseline_score
                improvement_ratio = delta_score / baseline_score if baseline_score > 0 else 0
            else:
                delta_score = baseline_score - context_score
                improvement_ratio = delta_score / baseline_score if baseline_score > 0 else 0
            
            context_results.append({
                "rule_id": r["rule_id"],
                "category": r["category"],
                "title": r["title"],
                "description": r["description"],
                f"context_{metric_name}": context_ppl,
                f"context_gen_{metric_name}": context_gen,
                f"baseline_{metric_name}": baseline_score,
                f"delta_{metric_name}": delta_score,
                "improvement_ratio": improvement_ratio
            })
        
        if is_higher_better:
            context_results_by_absolute = sorted(context_results, key=lambda x: x[f"context_{metric_name}"], reverse=True)
        else:
            context_results_by_absolute = sorted(context_results, key=lambda x: x[f"context_{metric_name}"])
        
        best_absolute_contexts = context_results_by_absolute[:top_k]
        
        positive_delta_results = [r for r in context_results if r[f"delta_{metric_name}"] > 0]
        context_results_by_delta = sorted(positive_delta_results, key=lambda x: x[f"delta_{metric_name}"], reverse=True)
        best_improvement_contexts = context_results_by_delta[:top_k]
        
        return best_absolute_contexts, best_improvement_contexts, baseline_score


def create_context_analyzer(metric_type: str, metric_computer_instance):
    if metric_type.lower() == "bertscore":
        computer = BERTScoreComputer(metric_computer_instance)
    elif metric_type.lower() == "perplexity":
        computer = PerplexityComputer(metric_computer_instance)
    elif metric_type.lower() == "rouge1":
        computer = ROUGE1Computer(metric_computer_instance)
    
    return ContextAnalyze(computer)
