import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer
from datasets import load_dataset
import evaluate
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

"""
    >> GRPO (Group Relative Policy Optimization) for Text Summarization <<
    Goal: 
        1. Leverage the stable structure of Huggingface Trainer
        2. Override compute_loss function to implement custom loss (RL loss)
        3. Use Models better than T5-small (e.g., T5-base, BART, Pegasus, etc.)
        4. Train on SAMSUM dataset

    Key Concepts:
        - GRPO Samples multiple outputs per input (group)
        - Uses relative advantages within the group for policy gradient
        - More stable than vanilla policy gradient methods

    Steps:
        1. Basic Environment Setup
        2. GRPO Trainer Implementation (Do not Use GRPOTrainer)
        3. ROUGE-based Reward Function & Log probability Calculation
        4. Training Execution and Evaluation
"""

# 1. Basic Environment Setup
class SummarizationDataProcessor:
    """Data Processor for Summarization Tasks"""
    def __init__(self, tokenizer: AutoTokenizer, max_input_length: int = 512, max_target_length: int = 128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def preprocess_function(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """
            Returns:
                - input_ids: Token IDs of input dialogues
                - attention_mask: Attention masks for input dialogues
                - labels: Token IDs of target summaries (with -100 for padding)

            It helps to prepare minibatches for dynamic padding.
        """
        # Tokenize Input Dialogues
        inputs = self.tokenizer(
            examples["dialogue"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize Target Summaries
        labels = self.tokenizer(
            examples["summary"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        # Replace padding tokens in labels with -100 (Ignore in loss computation)
        labels_input_ids = labels["input_ids"]
        labels_input_ids = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_seq] for labels_seq in labels_input_ids
        ]

        inputs["labels"] = labels_input_ids
        return inputs
    
# 2. GRPO Trainer Implementation
class CustomGRPOTrainer(Trainer):
    """
        Custom GRPO Trainer to implement Group Relative Policy Optimization
        Within the Huggingface Trainer framework.
    """
    def __init__(self, *args, **kwargs):
        # Initialize ROUGE metric
        self.rouge_metric = evaluate.load("rouge")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs) -> torch.Tensor:
        """
            Compute the GRPO loss

            Need to Implement GRPO Logic
                1. Generate Summaries with current Policy (with Decoding Strategy)
                2. Calculate Lolg Probabilities of Generated Summaries
                3. Calculate ROUGE-based Rewards
                4. Compute Relative Advantages within the Group and GRPO Loss (-log_prob * advantage)

            Basically, use CrossEntropyLoss by hand -- the basic version
        """
        # Basic Forward Pass
        outputs = model(**inputs)
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        logits = outputs.logits

        loss = ce_loss_fct(logits.view(-1, logits.size(-1)), inputs["labels"].view(-1))
        loss = loss.view(inputs["labels"].size())

        # To return scalar loss, take mean over non-ignored tokens
        scalar_loss = loss.mean()

        return (scalar_loss, outputs) if return_outputs else scalar_loss

        

# Reward Function and Utilities (for GRPO Implementation)
class RewardCalculator:
    """ROUGE-based Reward Calculator"""
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load("rouge")

    def compute_rouge(self, predictions: List[str], references: List[str]) -> torch.Tensor:
        """
            Calculate ROUGE-L score as reward

            Args:
                - predictions: List of Generated Summaries
                - references: List of Reference Summaries

            Returns:
                - rewards: Tensor of ROUGE-L scores for each prediction(sample)
        """
        rouge_scores = self.rouge_metric.compute(
            predictions = predictions,
            references = references,
            use_stemmer = True
        )

        # Use ROUGE-L F1 Score as Reward
        reward = rouge_scores["rougeL"]
        return torch.tensor(reward, dtype=torch.float32)
    
def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
        Calculate Log Probabilities of Generated Summaries
        Args:
            - logits: Model output logits (batch_size, seq_len, vocab_size)
            - labels: Generated Token IDs (batch_size, seq_len)
            - pad_token_id: Padding Token ID to ignore in log prob calculation

        Returns:
            - log_probs: Sum of log probabilities for each sequence in the batch (batch_size, )
    """
    # Calculate log softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim = -1)

    # Extract log probabilities of actually selected tokens at each position
    # Gather log probabilities for the labels
    # gather: (batch_size, seq_len, 1) -> (batch_size, seq_len)
    """
        >> Use of gather function <<
            >>> log_probs = torch.tensor([\
                    [[0.1, 0.2, 0.7, 0.0, 0.0],
                     [0.3, 0.3, 0.2, 0.1, 0.1],
                     [0.5, 0.2, 0.1, 0.1, 0.1]],
                    [[0.0, 0.1, 0.9, 0.0, 0.0],
                     [0.2, 0.5, 0.1, 0.1, 0.1],
                     [0.1, 0.2, 0.3, 0.3, 0.1]]
                ])
            >>> labels = torch.tensor([
                [2, 0, 1],
                [1, 1, 3]
                ])
            >>> print("log_probs shape:", log_probs.shape)
            log_probs shape: torch.Size([2, 3, 5])
            >>> print("labels shape:", labels.shape)
            labels shape: torch.Size([2, 3])

            >>> manual = torch.zeros_like(labels, dtype=torch.float)
            >>> for b in range(log_probs.shape[0]):
            ...     for t in range(log_probs.shape[1]):
            ...         manual[b, t] = log_probs[b, t, labels[b, t]]
            -> If uses this manual way, need to inilitialize tensor first, and then fill it with multi for-loop

            >>> print("Extracted log_probs of each label pos(manual):\n", manual)
                Extracted log_probs of each label pos(manual):
                tensor([[0.7000, 0.3000, 0.2000],
                        [0.1000, 0.5000, 0.3000]])

            >>> token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            >>> print("torch.gather Result:\n", token_log_probs)
                torch.gather Result:
                tensor([[0.7000, 0.3000, 0.2000],
                        [0.1000, 0.5000, 0.3000]])
    """
    token_log_probs = torch.gather(log_probs, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding tokens
    mask = (labels != pad_token_id).float()

    # Sum log probabilities over the sequence length, ignoring padding
    sequence_log_probs = (token_log_probs * mask).sum(dim = -1)

    return sequence_log_probs

# def custom_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
#     """
#         Custom Collate Function to handle dynamic padding in DataLoader

#         Args:
#             - batch: List of samples from the dataset
#             - pad_token_id: Padding token ID for the tokenizer

#         Returns:
#             - collated_batch: Dictionary with padded tensors for input_ids, attention_mask, labels
#     """
#     input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
#     attention_mask = [torch.tensor(sample["attention_mask"]) for sample in batch]
#     labels = [torch.tensor(sample["labels"]) for sample in batch]

#     # Pad sequences to the maximum length in the batch
#     input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
#     attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
#     labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

#     collated_batch = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }
#     return collated_batch

# 기존 custom_collate_fn 함수를 대체합니다.

class DynamicDataCollator:
    """
        Callable Class for Dynamic Padding
        pad_token_id를 내부 변수로 저장하여 Trainer에 전달되는 batch만으로 동작합니다.
    """
    def __init__(self, pad_token_id: int):
        # 클래스 초기화 시 필요한 pad_token_id를 저장
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
            Trainer가 data_collator를 호출할 때 batch만 전달됩니다.
            (배치 크기 B는 여기서 결정됨)
        """
        input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
        attention_mask = [torch.tensor(sample["attention_mask"]) for sample in batch]
        # label의 패딩은 -100로 해야 손실 계산에서 무시되므로, padding_value를 다르게 설정
        labels = [torch.tensor(sample["labels"]) for sample in batch]

        # 1. input_ids, attention_mask는 self.pad_token_id (일반적으로 0)로 패딩
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # 2. labels는 -100로 패딩하여 손실 계산에서 무시되도록 함
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        collated_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Trainer의 Seq2Seq 모델은 decoder_input_ids를 내부적으로 생성하므로,
            # labels만 정확하게 전달하면 됩니다. (필요하다면 여기서 decoder_input_ids 생성 로직을 추가할 수 있습니다.)
            "labels": labels,
        }
        return collated_batch

# 4. Training Execution and Evaluation
def main():
    print("="*80)
    print("GRPO Text Summarization Training")
    print("="*80)

    # (1) Load Model and Tokenizer
    print("\n[Phase 1] Loading model and tokenizer...")
    model_name = "facebook/bart-base"  # Model better than T5-small

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"  >> Model: {model_name}")
    print(f"  >> Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # (2) Load and Preprocess Dataset
    print("\n[Phase 2] Loading and preprocessing dataset...")
    dataset = load_dataset("knkarthick/samsum")

    print(f"  >> Train samples: {len(dataset['train'])}")
    print(f"  >> Validation samples: {len(dataset['validation'])}")
    print(f"  >> Test samples: {len(dataset['test'])}")

    sample = dataset["train"][0] 
    print(f"\n    [Sample Data]")
    print(f"    Dialogue: {sample['dialogue'][:100]}...")
    print(f"    Summary: {sample['summary']}")

    # Preprocessing (e.g., batch processing, tokenization, etc.)
    preprocessor = SummarizationDataProcessor(tokenizer)
    train_dataset = dataset["train"].select(range(2000))  
    eval_dataset = dataset["validation"].select(range(200))

    train_dataset = train_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing train dataset"
    )

    eval_dataset = eval_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Preprocessing eval dataset"
    )
    print(f"  >> Preprocessed train samples: {len(train_dataset)}")
    print(f"  >> Preprocessed eval samples: {len(eval_dataset)}")

    # (3) Setup Data Collator
    pad_id = tokenizer.pad_token_id

    data_collator = DynamicDataCollator(pad_token_id = tokenizer.pad_token_id)

    # (4) Define Training Arguments
    print("\n[Phase 3] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./training_grpo_base",
        eval_strategy="steps",
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16 = torch.cuda.is_available(),
        report_to = "none"
    )  

    print(f"  >> Output directory: {training_args.output_dir}")
    print(f"  >> Total training epochs: {training_args.num_train_epochs}")
    print(f"  >> Train batch size per device: {training_args.per_device_train_batch_size}")
    print(f"  >> Learning rate: {training_args.learning_rate}")

    # (5) Define ROUGE evaluation metric
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # preds_list = preds.tolist() 
        preds_list = list(preds)

        if not preds_list or not isinstance(preds_list[0], (list, np.ndarray, tuple)):
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        max_pred_len = max(len(p) for p in preds_list)
        pad_id = tokenizer.pad_token_id
        
        padded_preds = []
        for p in preds_list:
            p_list = [int(token_id) for token_id in list(p)]
            padding_needed = max_pred_len - len(p_list)
            padded_preds.append(p_list + [pad_id] * padding_needed)

        preds = np.array(padded_preds, dtype=np.int32)

        # Replace -100 in labels as pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        result = rouge_metric.compute(
            predictions = decoded_preds,
            references = decoded_labels,
            use_stemmer = True
        )

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }

    # (6) Initialize Custom GRPO Trainer
    print("\n[Phase 4] Initializing Custom GRPO Trainer...")
    trainer = CustomGRPOTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    print("  >> Trainer initialized.")

    # (7) Start Training
    print("\n[Phase 5] Starting training...")
    trainer.train()

    print("\nTraining completed.")
    print("="*80)

    print("\n[Phase 6] Evaluating on test set...")
    eval_results = trainer.evaluate()

    print(f"  >> Evaluation results: {eval_results}")
    print(f"  >>Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"  >>ROUGE-1: {eval_results['eval_rouge1']:.4f}")
    print(f"  >>ROUGE-2: {eval_results['eval_rouge2']:.4f}")
    print(f"  >>ROUGE-L: {eval_results['eval_rougeL']:.4f}")
    print("\nAll done!")

    # (8) Save the final model
    print("\n[Phase 7] Sample Generation Test...")
    test_dialogue = dataset["test"][0]["dialogue"]
    test_summary = dataset["test"][0]["summary"]

    inputs = tokenizer(test_dialogue, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = trainer.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n  >> Test Dialogue: {test_dialogue}")
    print(f"  >> Reference Summary: {test_summary}")
    print(f"  >> Generated Summary: {generated_summary}")

    print("\nProcess completed.")

if __name__ == "__main__":
    main()
