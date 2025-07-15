# ./result_processor.py
import json
import os
import numpy as np
from utils import result_dir

class ResultProcessor:
    def __init__(self):
        self.lowest_ppl_results = []
        self.highest_delta_ppl_results = []
        self.lowest_bleurt_results = []
        self.highest_delta_bleurt_results = []
        self.lowest_rouge1_results = []
        self.highest_delta_rouge1_results = []

    def process_qa(
        self,
        question_id: str,
        question: str,
        answer: str,
        # PPL
        baseline_ppl: float,
        lowest_ppl_contexts: list,
        highest_delta_ppl_contexts: list,
        # BLEURT
        baseline_bleurt: float = None,
        lowest_bleurt_contexts: list = None,
        highest_delta_bleurt_contexts: list = None,
        # ROUGE-1
        baseline_rouge1: float = None,
        lowest_rouge1_contexts: list = None,
        highest_delta_rouge1_contexts: list = None
    ):
        # PERPLEXITY
        top3_low = lowest_ppl_contexts[:5]
        self.lowest_ppl_results.append({
            "Question_ID": question_id,
            "Question": question,
            "Answer": answer,
            "baseline_ppl": float(np.round(baseline_ppl, 2)),
            "Top-3_Lowest_context_PPL": [
                {
                    "rank": i+1,
                    "rule_id": context["rule_id"],
                    "category": context["category"],
                    "title": context["title"],
                    "description": context["description"],
                    "context_ppl": float(np.round(context["context_ppl"], 2)),
                    "delta_ppl": float(np.round(context["delta_ppl"], 2)),
                    "improvement_ratio": float(np.round(context["improvement_ratio"], 4)),
                    "context_ppl": float(np.round(context["context_ppl"], 2)),
                    "context_gen": context.get("context_gen", ""),
                } for i, context in enumerate(top3_low)
            ]

        })
        top3_high = highest_delta_ppl_contexts[:5]
        self.highest_delta_ppl_results.append({
            "Question_ID": question_id,
            "Question": question,
            "Answer": answer,
            "baseline_ppl": float(np.round(baseline_ppl, 2)),
            "Top-3_Highest_delta_PPL": [
                {
                    "rank": i+1,
                    "rule_id": context["rule_id"],
                    "category": context["category"],
                    "title": context["title"],
                    "description": context["description"],
                    "context_ppl": float(np.round(context["context_ppl"], 2)),
                    "delta_ppl": float(np.round(context["delta_ppl"], 2)),
                    "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
                } for i, context in enumerate(top3_high)
            ]
        })

        # # BLEURT
        # if baseline_bleurt is not None and lowest_bleurt_contexts is not None:
        #     top3_low_b = lowest_bleurt_contexts[:3]
        #     self.lowest_bleurt_results.append({
        #         "Question_ID": question_id,
        #         "Question": question,
        #         "Answer": answer,
        #         "baseline_bleurt": float(np.round(baseline_bleurt, 4)),
        #         "Top-3_Lowest_context_BLEURT": [
        #             {
        #                 "rank": i+1,
        #                 "rule_id": context["rule_id"],
        #                 "context_bleurt": float(np.round(context["context_bleurt"], 4)),
        #                 "delta_bleurt": float(np.round(context["delta_bleurt"], 4)),
        #                 "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
        #             } for i, context in enumerate(top3_low_b)
        #         ]
        #     })
        # if highest_delta_bleurt_contexts is not None:
        #     top3_high_b = highest_delta_bleurt_contexts[:3]
        #     self.highest_delta_bleurt_results.append({
        #         "Question_ID": question_id,
        #         "Question": question,
        #         "Answer": answer,
        #         "baseline_bleurt": float(np.round(baseline_bleurt, 4)),
        #         "Top-3_Highest_delta_BLEURT": [
        #             {
        #                 "rank": i+1,
        #                 "rule_id": context["rule_id"],
        #                 "context_bleurt": float(np.round(context["context_bleurt"], 4)),
        #                 "delta_bleurt": float(np.round(context["delta_bleurt"], 4)),
        #                 "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
        #             } for i, context in enumerate(top3_high_b)
        #         ]
        #     })

        # # ROUGE-1
        # if baseline_rouge1 is not None and lowest_rouge1_contexts is not None:
        #     top3_low_r = lowest_rouge1_contexts[:3]
        #     self.lowest_rouge1_results.append({
        #         "Question_ID": question_id,
        #         "Question": question,
        #         "Answer": answer,
        #         "baseline_rouge1": float(np.round(baseline_rouge1, 4)),
        #         "Top-3_Lowest_context_ROUGE1": [
        #             {
        #                 "rank": i+1,
        #                 "rule_id": context["rule_id"],
        #                 "context_rouge1": float(np.round(context["context_rouge1"], 4)),
        #                 "delta_rouge1": float(np.round(context["delta_rouge1"], 4)),
        #                 "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
        #             } for i, context in enumerate(top3_low_r)
        #         ]
        #     })
        # if highest_delta_rouge1_contexts is not None:
        #     top3_high_r = highest_delta_rouge1_contexts[:3]
        #     self.highest_delta_rouge1_results.append({
        #         "Question_ID": question_id,
        #         "Question": question,
        #         "Answer": answer,
        #         "baseline_rouge1": float(np.round(baseline_rouge1, 4)),
        #         "Top-3_Highest_delta_ROUGE1": [
        #             {
        #                 "rank": i+1,
        #                 "rule_id": context["rule_id"],
        #                 "context_rouge1": float(np.round(context["context_rouge1"], 4)),
        #                 "delta_rouge1": float(np.round(context["delta_rouge1"], 4)),
        #                 "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
        #             } for i, context in enumerate(top3_high_r)
        #         ]
        #     })

    def save_results(self):
        os.makedirs(result_dir, exist_ok=True)

        # PPL
        with open(os.path.join(result_dir, "(dev)lowest_ppl_results(kanana_prompt_top5).json"), "w", encoding="utf-8") as f:
            json.dump(self.lowest_ppl_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(result_dir, "(dev)highest_delta_ppl_results(kanana_prompt_top5).json"), "w", encoding="utf-8") as f:
            json.dump(self.highest_delta_ppl_results, f, ensure_ascii=False, indent=4)

        # with open(os.path.join(result_dir, "(train)lowest_ppl_results(kanana_prompt_top5).json"), "w", encoding="utf-8") as f:
        #     json.dump(self.lowest_ppl_results, f, ensure_ascii=False, indent=4)
        # with open(os.path.join(result_dir, "(train)highest_delta_ppl_results(kanana_prompt_top5).json"), "w", encoding="utf-8") as f:
        #     json.dump(self.highest_delta_ppl_results, f, ensure_ascii=False, indent=4)

        # # BLEURT
        # if self.lowest_bleurt_results:
        #     with open(os.path.join(result_dir, "lowest_bleurt_results.json"), "w", encoding="utf-8") as f:
        #         json.dump(self.lowest_bleurt_results, f, ensure_ascii=False, indent=4)
        # if self.highest_delta_bleurt_results:
        #     with open(os.path.join(result_dir, "highest_delta_bleurt_results.json"), "w", encoding="utf-8") as f:
        #         json.dump(self.highest_delta_bleurt_results, f, ensure_ascii=False, indent=4)

        # # ROUGE-1
        # if self.lowest_rouge1_results:
        #     with open(os.path.join(result_dir, "lowest_rouge1_results.json"), "w", encoding="utf-8") as f:
        #         json.dump(self.lowest_rouge1_results, f, ensure_ascii=False, indent=4)
        # if self.highest_delta_rouge1_results:
        #     with open(os.path.join(result_dir, "highest_delta_rouge1_results.json"), "w", encoding="utf-8") as f:
        #         json.dump(self.highest_delta_rouge1_results, f, ensure_ascii=False, indent=4)
