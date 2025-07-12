# import json
# import os
# import numpy as np
# from utils import result_dir

# class ResultProcessor:
#     def __init__(self):
#         self.lowest_ppl_results = []
#         self.highest_delta_results = []

#     def process_qa(self, question_id: str, question: str, answer: str, baseline_ppl: float, lowest_ppl_contexts: list, highest_delta_contexts: list):
#         top_3_lowest_ppl = lowest_ppl_contexts[:3]
#         lowest_ppl_result = {
#             "Question_ID": question_id,
#             "Question": question,
#             "Answer": answer,
#             "baseline_ppl": float(np.round(baseline_ppl, 2)),
#             "Top-3_Lowest_context_PPL": [
#                 {
#                     "rank": i + 1,
#                     "rule_id": context["rule_id"],
#                     "category": context["category"],
#                     "title": context["title"],
#                     "description": context["description"],
#                     "context_ppl": float(np.round(context["context_ppl"], 2)),
#                     "delta_ppl": float(np.round(context["delta_ppl"], 2)),
#                     "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
#                 }
#                 for i, context in enumerate(top_3_lowest_ppl)
#             ]
#         }
#         self.lowest_ppl_results.append(lowest_ppl_result)

#         top_3_highest_delta = highest_delta_contexts[:3]
#         highest_delta_result = {
#             "Question_ID": question_id,
#             "Question": question,
#             "Answer": answer,
#             "baseline_ppl": float(np.round(baseline_ppl, 2)),
#             "Top-3_Highest_delta_PPL": [
#                 {
#                     "rank": i + 1,
#                     "rule_id": context["rule_id"],
#                     "category": context["category"],
#                     "title": context["title"],
#                     "description": context["description"],
#                     "context_ppl": float(np.round(context["context_ppl"], 2)),
#                     "delta_ppl": float(np.round(context["delta_ppl"], 2)),
#                     "improvement_ratio": float(np.round(context["improvement_ratio"], 4))
#                 }
#                 for i, context in enumerate(top_3_highest_delta)
#             ]
#         }
#         self.highest_delta_results.append(highest_delta_result)

#     def save_results(self):
#         os.makedirs(result_dir, exist_ok=True)

#         with open(os.path.join(result_dir, "lowest_ppl_results.json"), "w", encoding="utf-8") as f:
#             json.dump(self.lowest_ppl_results, f, ensure_ascii=False, indent=4)
        
#         with open(os.path.join(result_dir, "highest_delta_results.json"), "w", encoding="utf-8") as f:
#             json.dump(self.highest_delta_results, f, ensure_ascii=False, indent=4)

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
        top3_low = lowest_ppl_contexts[:3]
        self.lowest_ppl_results.append({
            "Question_ID": question_id,
            "Question": question,
            "Answer": answer,
            "baseline_ppl": float(np.round(baseline_ppl, 2)),
            "Top-3_Lowest_context_PPL": [
                {
                    "rank": i+1,
                    "rule_id": ctx["rule_id"],
                    "category": ctx["category"],
                    "title": ctx["title"],
                    "description": ctx["description"],
                    "context_ppl": float(np.round(ctx["context_ppl"], 2)),
                    "delta_ppl": float(np.round(ctx["delta_ppl"], 2)),
                    "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                } for i, ctx in enumerate(top3_low)
            ]
        })
        top3_high = highest_delta_ppl_contexts[:3]
        self.highest_delta_ppl_results.append({
            "Question_ID": question_id,
            "Question": question,
            "Answer": answer,
            "baseline_ppl": float(np.round(baseline_ppl, 2)),
            "Top-3_Highest_delta_PPL": [
                {
                    "rank": i+1,
                    "rule_id": ctx["rule_id"],
                    "category": ctx["category"],
                    "title": ctx["title"],
                    "description": ctx["description"],
                    "context_ppl": float(np.round(ctx["context_ppl"], 2)),
                    "delta_ppl": float(np.round(ctx["delta_ppl"], 2)),
                    "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                } for i, ctx in enumerate(top3_high)
            ]
        })

        # BLEURT
        if baseline_bleurt is not None and lowest_bleurt_contexts is not None:
            top3_low_b = lowest_bleurt_contexts[:3]
            self.lowest_bleurt_results.append({
                "Question_ID": question_id,
                "Question": question,
                "Answer": answer,
                "baseline_bleurt": float(np.round(baseline_bleurt, 4)),
                "Top-3_Lowest_context_BLEURT": [
                    {
                        "rank": i+1,
                        "rule_id": ctx["rule_id"],
                        "context_bleurt": float(np.round(ctx["context_bleurt"], 4)),
                        "delta_bleurt": float(np.round(ctx["delta_bleurt"], 4)),
                        "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                    } for i, ctx in enumerate(top3_low_b)
                ]
            })
        if highest_delta_bleurt_contexts is not None:
            top3_high_b = highest_delta_bleurt_contexts[:3]
            self.highest_delta_bleurt_results.append({
                "Question_ID": question_id,
                "Question": question,
                "Answer": answer,
                "baseline_bleurt": float(np.round(baseline_bleurt, 4)),
                "Top-3_Highest_delta_BLEURT": [
                    {
                        "rank": i+1,
                        "rule_id": ctx["rule_id"],
                        "context_bleurt": float(np.round(ctx["context_bleurt"], 4)),
                        "delta_bleurt": float(np.round(ctx["delta_bleurt"], 4)),
                        "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                    } for i, ctx in enumerate(top3_high_b)
                ]
            })

        # ROUGE-1
        if baseline_rouge1 is not None and lowest_rouge1_contexts is not None:
            top3_low_r = lowest_rouge1_contexts[:3]
            self.lowest_rouge1_results.append({
                "Question_ID": question_id,
                "Question": question,
                "Answer": answer,
                "baseline_rouge1": float(np.round(baseline_rouge1, 4)),
                "Top-3_Lowest_context_ROUGE1": [
                    {
                        "rank": i+1,
                        "rule_id": ctx["rule_id"],
                        "context_rouge1": float(np.round(ctx["context_rouge1"], 4)),
                        "delta_rouge1": float(np.round(ctx["delta_rouge1"], 4)),
                        "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                    } for i, ctx in enumerate(top3_low_r)
                ]
            })
        if highest_delta_rouge1_contexts is not None:
            top3_high_r = highest_delta_rouge1_contexts[:3]
            self.highest_delta_rouge1_results.append({
                "Question_ID": question_id,
                "Question": question,
                "Answer": answer,
                "baseline_rouge1": float(np.round(baseline_rouge1, 4)),
                "Top-3_Highest_delta_ROUGE1": [
                    {
                        "rank": i+1,
                        "rule_id": ctx["rule_id"],
                        "context_rouge1": float(np.round(ctx["context_rouge1"], 4)),
                        "delta_rouge1": float(np.round(ctx["delta_rouge1"], 4)),
                        "improvement_ratio": float(np.round(ctx["improvement_ratio"], 4))
                    } for i, ctx in enumerate(top3_high_r)
                ]
            })

    def save_results(self):
        os.makedirs(result_dir, exist_ok=True)

        # PPL
        with open(os.path.join(result_dir, "lowest_ppl_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.lowest_ppl_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(result_dir, "highest_delta_ppl_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.highest_delta_ppl_results, f, ensure_ascii=False, indent=4)

        # BLEURT
        if self.lowest_bleurt_results:
            with open(os.path.join(result_dir, "lowest_bleurt_results.json"), "w", encoding="utf-8") as f:
                json.dump(self.lowest_bleurt_results, f, ensure_ascii=False, indent=4)
        if self.highest_delta_bleurt_results:
            with open(os.path.join(result_dir, "highest_delta_bleurt_results.json"), "w", encoding="utf-8") as f:
                json.dump(self.highest_delta_bleurt_results, f, ensure_ascii=False, indent=4)

        # ROUGE-1
        if self.lowest_rouge1_results:
            with open(os.path.join(result_dir, "lowest_rouge1_results.json"), "w", encoding="utf-8") as f:
                json.dump(self.lowest_rouge1_results, f, ensure_ascii=False, indent=4)
        if self.highest_delta_rouge1_results:
            with open(os.path.join(result_dir, "highest_delta_rouge1_results.json"), "w", encoding="utf-8") as f:
                json.dump(self.highest_delta_rouge1_results, f, ensure_ascii=False, indent=4)
