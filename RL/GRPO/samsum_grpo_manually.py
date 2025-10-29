import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
from pathlib import Path
import tqdm
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

import gc
# from transformers import Trainer, TrainingArguments # Not used Version

"""
    This script fine-tunes a pre-trained language model using the GRPO method on the SAMSum dataset.

    This implementation uses LoRA for parameter-efficient fine-tuning and evaluates the model using ROUGE metrics.
    It is designed based on TRL's GRPO implementation as follows:
        - Batched forward passes for efficiency.
        - Reward normalization for Advantage calculation.
        - Per-token loss computation with probability masking.
        - KL Divergence penalty to maintain alignment with the base model.
"""

# [Callback System for Saving and Logging]
def save_model_and_tokenizer(model: nn.Module, tokenizer: AutoTokenizer, 
                             save_dir: str, model_name: str):
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f" >> Model and tokenizer saved at {save_path}")

    return save_path

def load_model_and_tokenizer(model_dir: str, device: str = "auto") -> Tuple[nn.Module, AutoTokenizer]:
    load_path = Path(model_dir)
    if not load_path.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, load_in_8bit=True)

    print(f" >> Loaded model and tokenizer from {model_dir}")

    return model, tokenizer

class TrainingCallback:
    def on_step_end(self, step: int, logs: Dict[str, Any] = None, model: nn.Module = None) -> None:
        pass
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None, model: nn.Module = None) -> None:
        pass
    def on_train_end(self, model: nn.Module = None) -> None:
        pass

class CheckpointCallback(TrainingCallback):
    def __init__(self, save_dir: str, tokenizer: AutoTokenizer,
                 best_metric: str = "rougeL", mode: str = "max",
                 save_steps: int = 200, save_epoch: int = 1, save_best_only: bool = False, verbose: bool = True):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_metric = best_metric
        self.mode = mode
        self.save_steps = save_steps
        self.save_epoch = save_epoch
        self.save_best_only = save_best_only
        self.verbose = verbose

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_step = -1
        self.best_epoch = -1
        self.best_model_path = None
        self.history = []

    def _is_better(self, current: float, best: float) -> bool:
        return current > best if self.mode == "max" else current < best

    def on_step_end(self, step: int, metrics: Dict[str, float], model: nn.Module = None) -> None:
        current_metric = metrics.get(self.best_metric)
        if current_metric is None:
            raise ValueError(f"Metric {self.best_metric} not found in logs.")

        self.history.append({"step": step, **current_metric})
        is_best = self._is_better(current_metric, self.best_score)

        if is_best:
            self.best_score = current_metric
            self.best_step = step
            best_path = self.save_dir / f"best_model_step_{step}"
            save_model_and_tokenizer(model, self.tokenizer, self.save_dir, f"best_model_step_{step}")
            self.best_model_path = best_path
            if self.verbose:
                print(f" >> New best model at step {step} with {self.best_metric}: {self.best_score:.4f}")

        if not self.save_best_only:
            step_path = self.save_dir / f"checkpoint_step_{step}"
            save_model_and_tokenizer(model, self.tokenizer, self.save_dir, f"checkpoint_step_{step}")
            if self.verbose:
                print(f" >> Saved checkpoint at step {step}")
        
        return is_best

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float] = None, model: nn.Module = None) -> None:
        current_metric = metrics.get(self.best_metric)
        if current_metric is None:
            raise ValueError(f"Metric {self.best_metric} not found in logs.")

        self.history.append({"epoch": epoch, **current_metric})
        is_best = self._is_better(current_metric, self.best_score)

        if is_best:
            self.best_score = current_metric
            self.best_epoch = epoch
            best_path = self.save_dir / f"best_model_epoch_{epoch}"
            save_model_and_tokenizer(model, self.tokenizer, self.save_dir, f"best_model_epoch_{epoch}")
            self.best_model_path = best_path
            if self.verbose:
                print(f" >> New best model at epoch {epoch} with {self.best_metric}: {self.best_score:.4f}")

        if not self.save_best_only:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch}"
            save_model_and_tokenizer(model, self.tokenizer, self.save_dir, f"checkpoint_epoch_{epoch}")
            if self.verbose:
                print(f" >> Saved checkpoint at epoch {epoch}")
        
        return is_best

    def on_train_end(self, model: nn.Module = None) -> None:
        if self.best_model_path is not None:
            if self.verbose:
                print(f" >> Training complete. Best model at {self.best_model_path} with {self.best_metric}: {self.best_score:.4f}")
            loaded_model, _ = load_model_and_tokenizer(str(self.best_model_path))
            model.load_state_dict(loaded_model.state_dict())

        if self.verbose:
            print(f" >> Training History saved to {self.save_dir}.")

    def get_best_model_path(self) -> Optional[str]:
        return self.best_model_path

# [TRL-Style GRPO Implementation]
class GRPOSampler:
    """
        GRPO Sampler for generating summaries from dialogues.

        Key Concepts;
            - Batched Generation for Efficiency!!!
            2. Efficient Memory Usage
            3. Returns properly formatted outputs for reward calculation(loss computation).
    """
    def __init__(self, model: nn.Module, reference_model: nn.Module, tokenizer: AutoTokenizer,
                 group_size: int = 4, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate_group_samples(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               max_new_tokens: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Geneerate K samples per input in the batch.
            
            Returns:
                - generated_ids: Tensor of shape (batch_size, group_size, seq_len)
                - log_probs: Tensor of shape (batch_size, group_size, seq_len)
                - ref_log_probs: Tensor of shape (batch_size, group_size, seq_len)
        """
        self.model.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        all_generated_ids = [] 
        max_length = 0

        # Generate K Samples
        with torch.no_grad():
            for k in range(self.group_size):
                generated_outputs = model_to_use.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    temperature = self.temperature,
                    top_k = self.top_k,
                    top_p = self.top_p,
                    pad_token_id = self.tokenizer.eos_token_id,
                    eos_token_id = self.tokenizer.eos_token_id,
                    return_dict_in_generate = False,
                    use_cache = False
                )
                all_generated_ids.append(generated_outputs)
                max_length = max(max_length, generated_outputs.size(1))
        
        # Pad all sequences to max_length
        padded_ids = []
        for generate_idx in all_generated_ids:
            if generate_idx.size(1) < max_length:
                padding = torch.full(
                    (generate_idx.size(0), max_length - generate_idx.size(1)),
                    self.tokenizer.eos_token_id,
                    dtype=generate_idx.dtype,
                    device=device
                )
                generate_idx = torch.cat([generate_idx, padding], dim=1)
            padded_ids.append(generate_idx)

        generated_ids = torch.stack(padded_ids, dim=1)  # (batch_size, group_size, seq_len)
        self.model.train()

        # Compute Log Probabilities in batched manner (which TRL does)
        log_probs, ref_log_probs = self._compute_log_probs(
            generated_ids,
            input_length = input_ids.size(1)
        )

        return generated_ids, log_probs, ref_log_probs

    def _compute_log_probs(self, generated_ids: torch.Tensor, input_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Compute log probabilities for generated sequences from both policy and reference models.

            Key Concepts:
                - Single Forward pass for all group samples for entire batch.
                - Gradient Flow for Policy Model.
                - Reference Model in Eval Mode (No Gradient) -- Frozen

            Args:
                - generated_ids: Tensor of shape (batch_size, group_size, seq_len)
                - input_length: Length of the input prompt to exclude from log prob calculation.

            Returns:
                - log_probs: Tensor of shape (batch_size, group_size, seq_len)
                - ref_log_probs: Tensor of shape (batch_size, group_size, seq_len)
        """
        batch_size, group_size, total_seq_len = generated_ids.size()
        device = generated_ids.device

        # Reshape for batched processing
        flat_generated_ids = generated_ids.view(batch_size * group_size, total_seq_len)

        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        ref_model_to_use = self.reference_model.module if isinstance(self.reference_model, nn.DataParallel) else self.reference_model

        # Policy Model Forward Pass
        policy_outputs = model_to_use(input_ids = flat_generated_ids)
        policy_logits = policy_outputs.logits  # (batch_size * group_size, seq_len, vocab_size)

        # Reference Model Forward Pass
        with torch.no_grad():
            ref_outputs = ref_model_to_use(input_ids = flat_generated_ids)
            ref_logits = ref_outputs.logits  # (batch_size * group_size, seq_len, vocab_size)

        # Compute Log Probabilities
        policy_log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

        # Labels are the next tokens
        labels = flat_generated_ids[:, 1:]  # (batch_size * group_size, seq_len - 1)

        # Gather log probabilities corresponding to the generated tokens
        policy_token_log_probs = torch.gather(
            policy_log_probs, 
            dim = 2,
            index = labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size * group_size, seq_len - 1)

        ref_token_log_probs = torch.gather(
            ref_log_probs,
            dim = 2,
            index = labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size * group_size, seq_len - 1)

        # Reshape (current: (batch_size * group_size, seq_len - 1)) -> (batch_size, group_size, seq_len - 1)
        policy_token_log_probs = policy_token_log_probs.view(batch_size, group_size, -1)
        ref_token_log_probs = ref_token_log_probs.view(batch_size, group_size, -1)

        # Exclude input prompt tokens
        gen_start_idx = input_length - 1  # Because labels are shifted by 1
        policy_gen_log_probs = policy_token_log_probs[:, :, gen_start_idx:]  # (batch_size * group_size, gen_seq_len)
        ref_gen_log_probs = ref_token_log_probs[:, :, gen_start_idx:]        # (batch_size * group_size, gen_seq_len)

        return policy_gen_log_probs, ref_gen_log_probs

class ROUGERewardCalculator:
    """ Reward Calculator using ROUGE Metrics """
    def __init__(self, use_rouge_1: bool = True, use_rouge_l: bool = True):
        self.rouge = evaluate.load("rouge")
        self.use_rouge_1 = use_rouge_1
        self.use_rouge_l = use_rouge_l

    def compute_rewards(self, generated_summaries: List[str], 
                        reference_summaries: List[str], group_size: int = 4) -> torch.Tensor:
        batch_size = len(reference_summaries)
        expanded_references = []
        for ref in reference_summaries:
            expanded_references.extend([ref] * group_size)

        scores = []
        for pred, ref in zip(generated_summaries, expanded_references):
            result = self.rouge.compute(
                predictions = [pred],
                references = [ref],
                use_stemmer = True
            )
            if self.use_rouge_1 and self.use_rouge_l:
                score = (result["rouge1"] + result["rougeL"]) / 2.0
            elif self.use_rouge_1:
                score = result["rouge1"]
            elif self.use_rouge_l:
                score = result["rougeL"]
            else:
                raise ValueError("At least one of use_rouge_1 or use_rouge_l must be True.")
            scores.append(score)

        rewards = torch.tensor(scores, dtype=torch.float32).view(batch_size, group_size)
        return rewards
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
            Advantage = (reward - mean) / (std + eps)
        
            This provides better training stability.
        """
        # Group-wise normalization
        mean = rewards.mean(dim=1, keepdim=True) # (batch_size, 1)
        std = rewards.std(dim=1, keepdim=True)   # (batch_size, 1)
        advantages = (rewards - mean) / (std + 1e-8)

        return advantages
        
    
class GRPOLossCalculator:
    """
        GRPO Loss Calculator with KL Divergence Penalty.

        Key Concepts:
            - Per-token loss computation with masking.
            - KL Divergence penalty to maintain alignment with the reference model.
    """
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(self, log_probs: torch.Tensor, 
                     ref_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Compute GRPO Loss.

            Args:
                - log_probs: Tensor of shape (batch_size, group_size, seq_len)
                - ref_log_probs: Tensor of shape (batch_size, group_size, seq_len)
                - advantages: Tensor of shape (batch_size, group_size)
                - attention_mask: Tensor of shape (batch_size, seq_len)

            Returns:
                - total_loss: Scalar tensor representing the total loss.
                - metrics_dict: Dictionary containing average KL divergence and policy loss.
                    - avg_kl_divergence: Average KL divergence across the batch.
                    - avg_policy_loss: Average policy loss across the batch.    
        """
        log_probs = log_probs.float() 
        ref_log_probs = ref_log_probs.float()
        advantages = advantages.float()
        attention_mask = attention_mask.float() # Need to check for the shape

        if attention_mask.sum().item() == 0:
            return torch.tensor(0.0, device=log_probs.device), {'total_loss': 0.0, 'policy_loss': 0.0, 'kl_divergence': 0.0, 'mean_advantage': 0.0, 'std_advantage': 0.0}

        # 1. Policy Ratio
        log_ratio = log_probs - ref_log_probs  # (batch_size, group_size, seq_len)
        ratio = torch.exp(log_ratio)           # (batch_size, group_size, seq_len

        # 2. Policy Loss
        advantages = advantages.unsqueeze(-1)  # (batch_size, group_size, 1)
        policy_loss_per_token = -ratio * advantages * attention_mask  # (batch_size, group_size, seq_len)
        policy_loss = policy_loss_per_token.sum() / (attention_mask.sum() + 1e-8)  # Scalar

        # 3. KL Divergence
        kl_loss = (log_ratio * attention_mask).sum() / (attention_mask.sum() + 1e-8)  # Scalar

        # 4. Total Loss
        total_loss = policy_loss + self.beta * kl_loss

        # Metrics Dictionary
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_loss.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item()
        }

        return total_loss, metrics

# [Main Training Loop]
def train_grpo_llama(model: nn.Module, reference_model: nn.Module, 
                     tokenizer: AutoTokenizer, dataloader: DataLoader,
                     optimizer: torch.optim.Optimizer, device: torch.device,
                     num_epochs: int = 3, group_size: int = 4, 
                     temperature: float = 1.0, beta: float = 0.1) -> Dict[str, List[float]]:
    model.train()
    reference_model.eval()

    total_loss = 0.0
    total_reward = 0.0
    total_kl_divergence = 0.0
    total_policy_loss = 0.0

    sampler = GRPOSampler(
        model = model,
        reference_model = reference_model,
        tokenizer = tokenizer,
        group_size = group_size,
        temperature = temperature
    )
    reward_calculator = ROUGERewardCalculator()
    loss_calculator = GRPOLossCalculator(beta = beta)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        reference_summaries = batch["summaries"]

        # 1. Generate group samples
        generated_ids, log_probs, ref_log_probs = sampler.generate_group_samples(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32
        )

        batch_size, group_size, total_seq_len = generated_ids.size()
        gen_seq_len = log_probs.size(2)

        # 2. Decode and compute rewards
        generated_texts_2d = []
        for batch_idx in range(batch_size):
            batch_samples = []
            for k in range(group_size):
                generated_tokens = generated_ids[batch_idx, k, input_ids.size(1):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_samples.append(generated_text)
            generated_texts_2d.append(batch_samples)

        generated_texts = [text for batch_samples in generated_texts_2d for text in batch_samples]

        rewards = reward_calculator.compute_rewards(
            generated_summaries=generated_texts,
            reference_summaries=reference_summaries,
            group_size=group_size
        ).to(device)

        # 3. Compute advantages (TRL-style with whitening)
        advantages = reward_calculator.compute_advantages(rewards)

        # 4. Create attention mask for generated sequences
        attention_mask_gen = (
            generated_ids[:, :, -gen_seq_len:] != tokenizer.pad_token_id
        ).float()

        # 5. Compute GRPO loss (TRL-style)
        loss, metrics = loss_calculator.compute_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            attention_mask=attention_mask_gen
        )

        # 6. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.mean().item()
        total_policy_loss += metrics['policy_loss']
        total_kl_divergence += metrics['kl_divergence']
        
        if (step + 1) % 1 == 0:
            print(f" >> Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}, "
                  f"Avg Reward: {rewards.mean().item():.4f}, "
                  f"KL: {metrics['kl_divergence']:.4f}")
            
            # Print sample outputs
            display_batch_index = 0
            first_input_ids = input_ids[display_batch_index]
            first_attention_mask = attention_mask[display_batch_index]
            start_index = torch.where(first_attention_mask == 1)[0][0].item()
            non_padded_input_tokens = first_input_ids[start_index:]
            clean_input_text = tokenizer.decode(non_padded_input_tokens, skip_special_tokens=False)

            print(f"\n >> Sample Generation (Epoch {num_epochs}, Step {step+1}):")
            print(f"    Input: {clean_input_text[:200]}...")
            print(f"    Reference: {reference_summaries[display_batch_index]}")
            for k in range(group_size):
                print(f"    Gen-{k+1}: {generated_texts_2d[display_batch_index][k]} "
                      f"| Reward: {rewards[display_batch_index, k].item():.4f} "
                      f"| Adv: {advantages[display_batch_index, k].item():.4f}")
            print(" -------------------------------------------------------------------")

        # Flush for Memory Efficiency
        if (step + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(" >> Completed garbage collection after training.")

    avg_metrics = {
        "loss": total_loss / len(dataloader),
        "avg_reward": total_reward / len(dataloader),
        "avg_policy_loss": total_policy_loss / len(dataloader),
        "avg_kl_divergence": total_kl_divergence / len(dataloader)
    }

    return avg_metrics

def evaluate_grpo_llama(model: nn.Module, reference_model: nn.Module,
                        tokenizer: AutoTokenizer, dataloader: DataLoader,
                        group_size: int = 4, device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
    model.eval()

    sampler = GRPOSampler(model, reference_model, tokenizer, group_size, temperature=0.7)
    reward_calculator = ROUGERewardCalculator()
    rouge_metric = evaluate.load('rouge')

    all_rewards = []
    all_predictions = []
    all_references = []
    sample_outputs = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            reference_summaries = batch['summaries']

            generated_ids, _, _ = sampler.generate_group_samples(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32
            )
            batch_size, group_size, total_seq_len = generated_ids.size()

            generated_texts = []
            for batch_idx in range(batch_size):
                for k in range(group_size):
                    generated_tokens = generated_ids[batch_idx, k, input_ids.size(1):]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    generated_texts.append(generated_text)

            rewards = reward_calculator.compute_rewards(
                generated_summaries=generated_texts,
                reference_summaries=reference_summaries,
                group_size=group_size
            )

            all_rewards.append(rewards)

            for batch_idx in range(batch_size):
                best_k = rewards[batch_idx].argmax().item()
                best_sample_index = batch_idx * group_size + best_k
                all_predictions.append(generated_texts[best_sample_index])
                all_references.append(reference_summaries[batch_idx])

            if step == 0:
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                sample_outputs.append({
                    'input': input_text,
                    'generated': [generated_texts[k] for k in range(group_size)],
                    'reference': reference_summaries[0],
                    'rewards': rewards[0].tolist()
                })

    rouge_scores = rouge_metric.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True
    )

    all_rewards = torch.cat(all_rewards, dim=0)

    results = {
        'mean_reward': all_rewards.mean().item(),
        'max_reward': all_rewards.max().item(),
        'min_reward': all_rewards.min().item(),
        'std_reward': all_rewards.std().item(),
        'rouge1': rouge_scores['rouge1'],
        'rougeL': rouge_scores['rougeL'],
        'samples': sample_outputs
    }

    return results


def grpo_collate_fn(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:

    dialogues = [item['dialogue'] for item in batch]
    summaries = [item['summary'] for item in batch]

    message_list = []
    for dialogue in dialogues:
        messages = [
            {"role": "system", "content": "You are an expert dialogue summarization assistant. Summarize the following dialogue concisely, in English."},
            {"role": "user", "content": dialogue}
        ]
        message_list.append(messages)

    formatted_inputs = [
        tokenizer.apply_chat_template(
            conversation = messages,
            tokenize = False,
            add_generation_prompt = True
        ) for messages in message_list
    ]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        formatted_inputs,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        add_special_tokens = False
    )
    """
        >> Why add_special_tokens = False? <<

        >> The reason why not done tokenizing with apply_chat_format. <<
        
    """
    
    # Return to original padding side
    tokenizer.padding_side = original_padding_side

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "summaries": summaries
    }

def main():
    print(" >> TRL Style GRPO Fine-tuning on SAMSum Dataset << ")

    # Load dataset
    print(" >> Loading SAMSum Dataset...")
    dataset = load_dataset("knkarthick/samsum")
    
    # Initialize tokenizer and model
    print(" >> Initializing tokenizer and model...")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        """
            Even the model that we use is 'Llama' model, which does not have a pad_token by default.
            Also, we need to *predict* the next new token -> if <eos> token is predicted, it means the generation is finished.
            Therefore, we set the pad_token to eos_token with padding_side = "left".
        """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA Configuration
    print(" >> Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type = "CAUSAL_LM",
        bias="none"
    )

    # Load pre-trained model with 8-bit quantization (QLoRA)
    print(" >> Loading pre-trained model with 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True,
        bnb_8bit_quant_type = "nf4",
        bnb_8bit_use_double_quant = True,
        bnb_8bit_compute_type = torch.float16
    )

    # Set Policy model with LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto"
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Set Reference model with LoRA (frozen)
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = "auto"
    )
    # reference_model = get_peft_model(reference_model, lora_config)
    for param in reference_model.parameters():
        param.requires_grad = False # Freeze the reference model, which means no gradient computation.
    reference_model.eval()

    print(" >> Model and tokenizer are ready.")
    print(f" >> [PEFT Model] Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # GRPO Hyperparameters
    print(" >> Setting up GRPO hyperparameters...")
    grpo_config = {
        "num_epochs": 3,
        "batch_size": 2,
        "group_size": 4,
        "learning_rate": 5e-5,
        "temperature": 0.7,
        "beta": 0.1,
        "top_k": 50,
        "top_p": 0.9,
    }

    for key, value in grpo_config.items():
        print(f"    - {key}: {value}") 

    # Prepare DataLoaders
    print(" >> Preparing DataLoaders...")
    train_dataset = dataset["train"].shuffle(seed=42)
    val_dataset = dataset["validation"].shuffle(seed=42)

    train_loader = DataLoader(
        train_dataset,
        batch_size = grpo_config["batch_size"],
        shuffle = True,
        collate_fn = lambda batch: grpo_collate_fn(batch, tokenizer)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = grpo_config["batch_size"],
        shuffle = False,
        collate_fn = lambda batch: grpo_collate_fn(batch, tokenizer)
    )

    # Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=grpo_config["learning_rate"])

    # Create Checkpoint Directory
    checkpoint_callback = CheckpointCallback(
        save_dir = "./grpo_samsum_checkpoints",
        save_steps = 200,
        save_epoch = 1,
        tokenizer = tokenizer,
        best_metric = "rougeL",
        mode = "max",
        save_best_only = False,
        verbose = True
    )

    # Start GRPO Training
    print(" >> Starting GRPO Training...")
    
    for epoch in range(grpo_config["num_epochs"]):
        print(f" >> Epoch {epoch + 1}/{grpo_config['num_epochs']}")

        # Training
        train_metrics = train_grpo_llama(
            model = model,
            reference_model = reference_model,
            tokenizer = tokenizer,
            dataloader = train_loader,
            optimizer = optimizer,
            device = device,
            num_epochs = grpo_config["num_epochs"],
            group_size = grpo_config["group_size"],
            temperature = grpo_config["temperature"],
            beta = grpo_config["beta"]
        )

        print(f"\n >> Epoch {epoch + 1} Training Results:")
        print(f"    - Loss: {train_metrics['loss']:.4f}")
        print(f"    - Avg Reward: {train_metrics['avg_reward']:.4f}")
        print(f"    - KL Divergence: {train_metrics['avg_kl_divergence']:.4f}")
        print(f"   - Policy Loss: {train_metrics['avg_policy_loss']:.4f}")

        del train_metrics
        torch.cuda.empty_cache()
        gc.collect()
        print(" >> Completed garbage collection after training.")

        # Evaluation
        print(f" >> Evaluating after Epoch {epoch + 1} for Validation Set...")
        eval_results = evaluate_grpo_llama(
            model = model,
            reference_model = reference_model,
            tokenizer = tokenizer,
            dataloader = val_loader,
            group_size = grpo_config["group_size"],
            device = device
        )

        print(f"\n >> Epoch {epoch + 1} Evaluation Results:")
        print(f"    - ROUGE-1: {eval_results['rouge1']:.4f}")
        print(f"    - ROUGE-2: {eval_results['rouge2']:.4f}")
        print(f"    - ROUGE-L: {eval_results['rougeL']:.4f}")
        print(f"   - Avg Reward: {eval_results['avg_reward']:.4f}")
        print(f"   - Reward Std: {eval_results['reward_std']:.4f}")

        # Print Sample Summaries
        print("\n >> Sample Summaries:")
        if eval_results["samples"]:
            print(f"    - Dialogue: {eval_results['samples'][0]['dialogue']}")
            for i, sample in enumerate(eval_results["samples"]):
                print(f"  [Generated Sample {i + 1}]")
                print(f"    - Reference Summary: {sample['reference_summary']}")
                print(f"    - Generated Summary: {sample['generated_summary']}\n")
                # For every sample, print the reference and generated summary.
                # NEED TO FIXX

        # Save Checkpoint
        metrics_dict = {
            "loss": train_metrics["loss"],
            "avg_reward": train_metrics["avg_reward"],
            "kl_divergence": train_metrics["avg_kl_divergence"],
            "policy_loss": train_metrics["avg_policy_loss"],
            "rouge1": eval_results["rouge1"],
            "rouge2": eval_results["rouge2"],
            "rougeL": eval_results["rougeL"]
        }
        checkpoint_callback.on_epoch_end(epoch, metrics_dict, model)
        checkpoint_callback.on_step_end((epoch + 1)*len(train_loader), metrics_dict, model)

    # Final Save
    checkpoint_callback.on_train_end(model)
    print(" >> GRPO Training Completed and Model Saved.")

    # Save fincal Model
    final_save_path = save_model_and_tokenizer(
        model = model,
        tokenizer = tokenizer,
        save_dir = "./grpo_samsum_final_model",
        model_name = "grpo_samsum_llama3.2_3b"
    )
    print(f" >> Final model saved at {final_save_path}")

if __name__ == "__main__":
    main()
