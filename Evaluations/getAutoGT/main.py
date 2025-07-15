# ./main.py
from tqdm import tqdm

from utils import KOGPT2, Llama2_7b
import sys
from compute_perplexity import getPerplexity
from compute_bleurt import getBLEURT
# from compute_rouge1 import getROUGE1
from context_Analyze import ContextAnalyze, PerplexityComputer, BERTScoreComputer, ROUGE1Computer, create_context_analyzer
from data import DataLoader
from result_processor import ResultProcessor

def main():
    # Initialize models and computation classes
    model = Llama2_7b()
    print(">> Model loaded, now proceeding...", flush=True)

    compute_ppl = getPerplexity(model)
    # compute_bleurt = getBLEURT(model)
    # compute_rouge1 = getROUGE1(model)

    context_analyze = ContextAnalyze(PerplexityComputer(compute_ppl))
    # context_analyze_bleurt = ContextAnalyze(BERTScoreComputer(compute_bleurt))
    # context_analyze_rouge1 = ContextAnalyze(ROUGE1Computer(compute_rouge1))

    data_loader = DataLoader()
    data_loader.load_data()
    print(f"Loaded {len(data_loader.get_qas())} QAs", flush=True)

    # Load rules, QAs, and few-shot examples
    rules = data_loader.get_rules()
    qas = data_loader.get_qas()

    result_processor = ResultProcessor()

    print("Starting to process QAs...")
    for qa in tqdm(data_loader.get_qas(), desc="Processing QAs"):
        question_id = qa["id"]
        question = qa["input"]["question"]
        answer = qa["output"]["answer"]

        lowest_ppl_contexts, highest_delta_contexts, (baseline_ppl, _) = context_analyze.find_best_context(qa["input"]["question_type"], question, answer, rules, few_shot_examples, top_k=5)

        # lowest_bleurt_contexts, highest_delta_bleurt_contexts, baseline_bleurt = context_analyze_bleurt.find_best_context(
        #     question, answer, rules, few_shot_examples, top_k=10
        # )
        
        # lowest_rouge1_contexts, highest_delta_rouge1_contexts, baseline_rouge1 = context_analyze_rouge1.find_best_context(
        #     question, answer, rules, few_shot_examples, top_k=10
        # )

        result_processor.process_qa(
            question_id,
            question,
            answer,
            baseline_ppl,
            lowest_ppl_contexts,
            highest_delta_contexts,
            # baseline_bleurt,
            # lowest_bleurt_contexts,
            # highest_delta_bleurt_contexts,
            # baseline_rouge1,
            # lowest_rouge1_contexts,
            # highest_delta_rouge1_contexts
        )

    result_processor.save_results()
    print("Processing completed. Results saved.")

if __name__ == "__main__":
    main()
