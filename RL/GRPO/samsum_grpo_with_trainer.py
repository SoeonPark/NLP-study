import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training

from datasets import load_dataset
from dataclass import dataclass, field
from typing import Optional, Iterator, List, Dict, Any, Tuple, Union

from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
import evaluate
from tqdm import tqdm
import torch.nn.functional as F

# Callback for Saving and Logging
def save_model_and_tokenizer(model: nn.Module, tokenizer: AutoTokenizer, save_directory: str,
                             model_name: str):
    save_path = os.path.join(save_directory, model_name)
    os.makedirs(save_path, exist_ok=True)

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    # Check if the model is a PEFT model
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(save_path)
    else:
        torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        if hasattr(model_to_save, "config") and hasattr(model_to_save.config, "save_pretrained"):
            model_to_save.config.save_pretrained(save_path)

    tokenizer.save_pretrained(save_path)
    print(f" >> [READY] Model and tokenizer saved to {save_path}")

    return save_path

def load_model_and_tokenizer(load_directory: str, model_name: str,
                             use_peft: bool = False,
                             peft_config: Optional[LoraConfig] = None) -> Tuple[nn.Module, AutoTokenizer]:
    load_path = os.path.join(load_directory, model_name)
    if not load_path.exists():
        raise ValueError(f" >> [ERROR] Load path {load_path} does not exist.")
    
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto")

    print(f" >> [READY] Model and tokenizer loaded from {load_path}")

    return model, tokenizer

class CheckpointCallback

class GRPOConfig:
    # Params that Control the Data Preprocessing
    num_generations: int = field(default=4, metadata = {"help": "Number of generations per input during training"})
    max_prompt_length: int = field(default = 512, metadata = {"help": "Maximum length of the input prompt"})
    shuffle_dataset: bool = field(default = True, metadata = {"help": "Whether to shuffle the dataset during training"})

    # Params that Control Generation
    generation_batch_size: int = field(default = 2, metadata = {"help": "Batch size for generation"})
    temperature: float = field(default = 1.0, metadata = {"help": "Temperature for sampling"})
    top_k: int = field(default = 0, metadata = {"help": "Top-k sampling parameter"})
    top_p: float = field(default = 0.9, metadata = {"help": "Top-p sampling parameter"})

    # Params that Control the GRPO Training
    steps_per_generation: int = field(default = 4, metadata = {"help": "Number of optimization steps per generation"})
    beta: float = field(default = 0.5, metadata = {"help": "Beta parameter for the loss function"})
    larger_epsilon: float = field(default = 0.2, metadata = {"help": "Epsilon threshold for high entropy masking"})
    lesser_epsilon: float = field(default = 0.2, metadata = {"help": "Epsilon threshold for low entropy masking"})
    num_iterations: int = field(default = 1, metadata = {"help": "Number of GRPO iterations per batch"})

    # Params Overridden from TrainingArguments
    gradient_checkpointing: bool = field(default = True, metadata = {"help": "Whether to use gradient checkpointing"})
    learning_rate: float = field(default = 2e-5, metadata = {"help": "Learning rate for training"})
    logging_steps: int = field(default = 10, metadata = {"help": "Logging steps"})

class ROUGERewardCalculator:
    def __init__(self, use_rouge_1: bool = True, use_rouge_l: bool = True):
        self.rouge = evaluate.load('rouge')
        self.use_rouge_1 = use_rouge_1
        self.use_rouge_l = use_rouge_l

    def compute_rewards(self, predictions: List[str], references: List[str], group_size: int) -> torch.Tensor:
        batch_size = len(predictions)
        reference_list = []
        for ref in references:
            reference_list.extend([ref] * group_size)

        scores = []
        for pred, ref in zip(predictions, reference_list):
            result = self.rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
            if self.use_rouge_1:
                score = result['rouge1']
            if self.use_rouge_l:
                score = result['rougeL']
            elif self.use_rouge_1 and self.use_rouge_l:
                score = (result['rouge1'] + result['rougeL']) / 2.0
            else:
                raise ValueError("At least one of use_rouge_1 or use_rouge_l must be True.")
            scores.append(score)

        rewards = torch.tensor(scores, dtype=torch.float32).view(batch_size, group_size)
        return rewards
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
            Compute advantages by subtracting the baseline (mean reward) from each reward.

            Equation:
                A(s, a) = R(s, a) - b(s)
        """
        baseline = rewards.mean(dim=1, keepdim=True) # (batch_size, 1)
        advantages = rewards - baseline # (batch_size, group_size)
        return advantages

# Inherit from Trainer to create a custom GRPOTrainer
class GRPOTrainer(Trainer):
    def __init__(self, model: nn.Module, ref_model: nn.Module,
                 args: TrainingArguments, grpo_config: GRPOConfig,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 eval_dataset: Optional[torch.utils.data.Dataset] = None,
                 tokenizer: AutoTokenizer, data_collator = None):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         data_collator=data_collator)

        self.ref_model = ref_model
        self.ref_model.eval()

        self.grpo_config = grpo_config
        self.reward_calculator = ROUGERewardCalculator()

        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _get_train_dataloader(self):

    def _get_train_sampler(self) -> Optional[Sampler]:

    def _repeat_sampler(self, dataset: torch.utils.data.Dataset) -> Iterator[int]:

    def _get_eval_sampler(self, eval_dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[Sampler]:

    def _get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:

    def _get_per_token_logps_and_entropies(self, )

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

    def _calculate_rewards(self, ) -> torch.Tensor:
    
    def _generate_single_turn(self, )
        
    def _generate(self, ):

    def _generate_and_score_completions(self, ) -> Dict[str, Union[torch.Tensor, Any]]:

    def _compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False) -> torch.Tensor:
        """
            GRPO Loss Computation

        """
        # -- 1. Generate and Score Completions --
        gen_data = self._generate_and_score_completions(inputs)

        # -- 2. Current Policy Log Probabilities and Entropies --
        logits = model(
            gen_data["generated_ids"],
            attention_mask=gen_data["generated_attention_mask"],
        ).logits
        log_probs, entropies = self._get_per_token_logps_and_entropies(
            logits,
            gen_data["generated_ids"],
            gen_data["generated_attention_mask"],
        )

        # -- 3. GRPO Loss Computation --
        # Ratio = exp(log \pi(a|s) - log \pi_ref(a|s))
        ratios = torch.exp(log_probs - gen_data["ref_log_probs"]) # (batch_size, num_generations, seq_len)
        # Policy Loss Component
        policy_loss = - (ratios * gen_data["advantages"]).mean()
        # KL Divergence Loss Component
        kl_loss = (log_probs - gen_data["ref_entropies"]).mean()

        loss = policy_loss + self.grpo_config.beta * kl_loss

        return (loss, outputs) if return_outputs else loss

# Main Function using Trainer
def main():
    # Initialize TrainingArguments and GRPOConfig
    training_args = TrainingArguments(
        output_dir="./grpo_samsum_checkpoints",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=2,
        gradient_checkpointing=True,
    )

    grpo_config = GRPOConfig(
        use_rouge_1=True,
        use_rouge_l=True,
        num_generations=4,
        steps_per_generation=4,
        beta=0.5,
        larger_epsilon=0.2,
        lesser_epsilon=0.2,
        num_iterations=1,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = load_model_and_tokenizer(
        load_directory="./pretrained_models", model_name=model_name, use_peft=True, 
        peft_config=lora_config
    )

    # Prepare reference model
    ref_model, _ = load_model_and_tokenizer(
        load_directory="./pretrained_models", model_name=model_name, use_peft=True,
        peft_config=lora_config
    )

    # Prepare dataset
    print(" >> Loading SAMSum Dataset...")
    dataset = load_dataset("knkarthick/samsum")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        grpo_config=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()
