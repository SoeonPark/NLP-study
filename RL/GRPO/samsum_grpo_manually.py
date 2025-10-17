import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
import math
import numpy as np

"""
    This script is for implementing the GRPO (Group Reward Policy Optimization) method
    on the SAMSum dataset using a modified T5 model for text summarization.

    Key Components:
        1. Group Sampling
        2. Reward Calculation using ROUGE
        2-1. Group-Relative Advantage Computation
        3. Modified Loss Function
        4. Policy Gradient

    Phase:
        1. Generate K samples using sampling decoding strategy.
        2. Compute ROUGE rewards for each sample.
        3. Normalize rewards within the group.
            -> Advantage = reward - baseline (mean reward of the group)
        4. Compute Policy Gradient Loss: -\Sum (advantage * log_prob)
        5. Update Model Parameters(Training Loop)
        6. Evaluation Loop
        7. Execution
"""

# 1. Group Sampling Module
class GRPOSampler:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer,
                 group_size: int = 5, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.95):
        self.model = model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate_group_samples(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               max_length: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Generate K size of Group Samples for each input in the batch.

            Args:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - max_length: maximum length of generated summaries
            
            Returns:
                - generated_ids: (batch_size, group_size, target_seq_len)
                - log_probs: (batch_size, group_size, target_seq_len) -- log probabilities of each generated token
        """
        self.model.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        all_generated_ids = []
        all_log_probs = []

        # Generate K size of samples
        for k in range(self.group_size):
            # Since it's just a sampling generation, don't need to compute for gradients
            with torch.no_grad():
                # Use model's generate method with sampling
                generated_ids = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    do_sample = True,
                    max_length = max_length,
                    temperature = self.temperature,
                    top_k = self.top_k,
                    top_p = self.top_p,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = self.tokenizer.eos_token_id,
                    return_dict_in_generate = False
                    # Note: return_dict_in_generate=False to get only generated_ids
                )

            all_generated_ids.append(generated_ids)

        # Stack generated ids: (batch_size, group_size, target_seq_len)
        generated_ids = torch.stack(all_generated_ids, dim=1) # dim=1 for group_size

        # Compute log probabilities for each generated sequence
        log_probs = self._compute_log_probs(input_ids, attention_mask, generated_ids)

        return generated_ids, log_probs
    
    def _compute_log_probs(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           generated_ids: torch.Tensor) -> torch.Tensor:
        """
            Compue Log Probabilities of generated sequences, Manually.

            Args:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - generated_ids: (batch_size, group_size, target_seq_len)

            Returns:
                - log_probs: (batch_size, group_size, target_seq_len)
        """
        batch_size, group_size, target_seq_len = generated_ids.size() # Extract sizes
        device = generated_ids.device

        # Prepare Decoder Inputs(shifted right) - (batch_size, group_size, target_seq_len); for teacher forcing
        decoder_input_ids = torch.full_like(generated_ids, self.tokenizer.pad_token_id)
        decoder_input_ids[:, :, 1:] = generated_ids[:, :, :-1] # for each the last token, shift right

        all_log_probs = [] 

        # Process each group sample
        for k in range(group_size):
            """
                curr_decoder_inputs: Indexed decoder inputs for k-th sample in group
                curr_labels: Indexed labels for k-th sample in group
            """
            curr_decoder_inputs = decoder_input_ids[:, k, :] # (batch_size, target_seq_len)
            curr_labels = generated_ids[:, k, :] # (batch_size, target_seq_len)

            with torch.no_grad():
                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    decoder_input_ids = curr_decoder_inputs,
                    labels = curr_labels
                )
            logits = outputs.logits # (batch_size, target_seq_len, vocab_size)

            # Compute log probabilities
            log_probs_dist = F.log_softmax(logits, dim=-1) # (batch_size, target_seq_len, vocab_size)

            # Gather Log Probabilities of generated tokens
            curr_log_probs = torch.gather(
                log_probs_dist,
                dim = 2, # vocab dimension
                index = curr_labels.unsqueeze(-1) # (batch_size, target_seq_len, 1)
            ).squeeze(-1) # (batch_size, target_seq_len)
            """
                >> Why unsqueeze and squeeze? <<
                    - unsqueeze: to match dimensions for gather
                    - squeeze: to remove the last dimension after gather

                    To extract elements from logits based on generated token ids, create an extra dimension with unsqueeze(index),
                    then remove it after gathering with squeeze().
            """
            all_log_probs.append(curr_log_probs)

        # Stack Log Probabilities: (batch_size, group_size, target_seq_len)
        log_probs = torch.stack(all_log_probs, dim=1) # dim=1 for group_size

        return log_probs

# 2. Reward Calculation Module
class ROUGERewardCalculator:
    """
        Calculate ROUGE-based Rewards for Generated Summaries
        Rewards are Normalized within each group.
    """
    def __init__(self, use_rouge_l: bool = True, use_rouge_1: bool = True): # Use ROUGE-1, L for reward
        self.rouge = evaluate.load('rouge')
        self.use_rouge_l = use_rouge_l
        self.use_rouge_1 = use_rouge_1

    # 2.1 ROUGE Reward Calculation
    def compute_rewards(self, generated_summaries: List[str],
                        reference_summaries: List[str], group_size: int = 5) -> torch.Tensor:
        """
            Compute ROUGE Rewards for each generated summary.

            Args:
                - generated_summaries: List of generated summaries (batch_size * group_size)
                - reference_summaries: List of reference summaries (batch_size)
                - group_size: number of samples per input in the batch

            Returns:
                - rewards: (batch_size, group_size) tensor of rewards
        """
        batch_size = len(reference_summaries)
        expanded_references = []
        for ref in reference_summaries:
            expanded_references.extend([ref] * group_size)

        # Compute ROUGE Scores
        scores = []
        for pred, ref in zip(generated_summaries, expanded_references):
            result = self.rouge.compute(
                predictions = [pred],
                references = [ref],
                use_stemmer = True
            )
            if self.use_rouge_l and self.use_rouge_1:
                score = (result["rouge1"] + result["rougeL"]) / 2.0
            elif self.use_rouge_l:
                score = result["rougeL"]
            elif self.use_rouge_1:
                score = result["rouge1"]
            else:
                raise ValueError("At least one of use_rouge_l or use_rouge_1 must be True.")
            
            scores.append(score)
        
        rewards = torch.tensor(scores, dtype=torch.float).view(batch_size, group_size)
        return rewards
    
    # 2.2 Group-Relative Advantage Computation
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
            Compute Group-Relative Advantages by normalizing rewards within each group.
            Normalize rewards first by subtracting the mean reward of the group.
            >> Equation: advantage = reward - baseline (mean reward of the group) <<

            Args:
                - rewards: (batch_size, group_size) tensor of rewards
            
            Returns:
                - advantages: (batch_size, group_size) tensor of advantages
        """
        # Compute mean reward for each group
        baseline = rewards.mean(dim=1, keepdim=True) # (batch_size, 1)

        # Compute advantages
        advantages = rewards - baseline # Broadcasting subtraction

        return advantages

        # rewards = []

        # # Expand References to match generated summaries
        # set_of_references = []
        # for i in reference_summaries:
        #     set_of_references.extend([i] * group_size) # Repeat each reference group_size times

        # # Compute ROUGE Scores
        # rouge_results = self.rouge.compute(
        #     predictions = generated_summaries,
        #     references = set_of_references,
        #     use_stemmer = True
        # )

        # # Extract relevant ROUGE scores
        # if self.use_rouge_l is True and self.use_rouge_1 is True:
        #     rouge_scores = [
        #         (rouge_results['rouge1'] + rouge_results['rougeL']) / 2.0
        #     ] * len(generated_summaries)
        # elif self.use_rouge_l is True and self.use_rouge_1 is False:
        #     rouge_scores = rouge_results['rougeL'] * len(generated_summaries)
        # elif self.use_rouge_l is False and self.use_rouge_1 is True:
        #     rouge_scores = rouge_results['rouge1'] * len(generated_summaries)
        # else:
        #     raise ValueError("At least one of use_rouge_l or use_rouge_1 must be True.")
        
        # # Reshape to (batch_size, group_size)
        # rewards = torch.tensor(rouge_scores).view(batch_size, group_size) # view() is for reshaping

        # return rewards

        # # 2.2 Group-Relative Advantage Computation
        # def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        #     """
        #         Compute Group-Relative Advantages by normalizing rewards within each group.
        #         Normalize rewards first by subtracting the mean reward of the group.
        #         >> Equation: advantage = reward - baseline (mean reward of the group) <<

        #         Args:
        #             - rewards: (batch_size, group_size) tensor of rewards
                
        #         Returns:
        #             - advantages: (batch_size, group_size) tensor of advantages
        #     """
        #     # Compute mean reward for each group
        #     baseline = rewards.mean(dim=1, keepdim=True) # (batch_size, 1)
    
        #     # Compute advantages
        #     advantages = rewards - baseline # Broadcasting subtraction

        #     return advantages   

# 3. Modified Model Class
class GRPOLossCalculator:
    """
        Compute GRPO Loss using Group-Relative Advantages.
        Loss = - Σ (advantage * log_prob)
    """
    def __init__(self, beta: float = 0.1):
        """
            Args: 
                - beta: scaling factor for advantages
        """
        self.beta = beta

    def compute_loss(self, log_probs: torch.Tensor, 
                     advantages: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
            Compute GRPO Policy Gradient Loss.

            Args:
                - log_probs: (batch_size, group_size, target_seq_len) -- log probabilities of generated tokens
                - advantages: (batch_size, group_size) -- group-relative advantages
                - attention_mask: (batch_size, group_size, target_seq_len) -- attention mask for padding tokens

            Returns:
                - loss: scalar tensor representing the GRPO loss
        """
        batch_size, group_size, target_seq_len = log_probs.size()

        # Sum log probabilities over sequence length to get sequence-level log probs
        if attention_mask is not None:
            """
                masked_log_probs: Mask log probabilities not to refers to next tokens
                sequence_log_probs: Sum of log probabilities over sequence length
            """
            masked_log_probs = log_probs * attention_mask # Mask padding tokens (batch_size, group_size, target_seq_len)
            sequence_log_probs = masked_log_probs.sum(dim=-1) # (batch_size, group_size) -- target_seq_len summed
        else:
            sequence_log_probs = log_probs.sum(dim=-1) # (batch_size, group_size)

        # Policy Gradient Loss Calculation
        # >> Equation: - \Sum [Advantage * log_prob] <<
        policy_gradient_loss = -(advantages * sequence_log_probs).mean() # Mean over batch and group

        return policy_gradient_loss
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
            Compute Group-Relative Advantages within each group.
            Advantage = reward - baseline (mean reward of the group)

            Args:
                - rewards: (batch_size, group_size) tensor of rewards

            Returns:
                - advantages: (batch_size, group_size) tensor of advantages
        """
        # Compute mean reward for each group
        baseline = rewards.mean(dim=1, keepdim=True) # (batch_size, 1)

        # Compute advantages
        advantages = rewards - baseline # Broadcasting subtraction

        return advantages

# 4. Main Training Script Structure (Training Loop)
def train_grpo(model: nn.Module, dataloader: DataLoader, 
               optimizer: torch.optim.Optimizer, device: torch.device,
               tokenizer: AutoTokenizer, epoch: int = 3,
               group_size: int = 5, temperature: float = 1.0,
               beta: float = 0.1) -> Tuple[float, float]:
    """
        Train the model using GRPO method.

        A Training Loop Description:
            1. Generate K samples using sampling decoding strategy.
            2. Compute ROUGE rewards for each sample.
            3. Compute for Advantages within the group.
            4. Compute Policy Gradient Loss: -\Sum (advantage * log_prob)
            5. Backpropagation and Model Update.
    """
    model.train()
    total_loss = 0.0
    total_reward = 0.0 

    sampler = GRPOSampler(model, tokenizer, group_size, temperature)
    reward_calculator = ROUGERewardCalculator()
    loss_calculator = GRPOLossCalculator(beta=beta)

    for step, batch in enumerate(dataloader):
        # Load on device
        breakpoint()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if epoch == 0 and step == 0:
            print(" [DBG] Input IDs Shape:", input_ids.shape)
            print(" [DBG] Attention Mask Shape:", attention_mask.shape)
            print(" [DBG] Labels Shape:", labels.shape)
            print(" [DBG] Group Size:", group_size)
            print("============================================")

        # 1. Generate group samples (K Samples per input)
        generate_ids, log_probs = sampler.generate_group_samples(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length = 128
        ) # generate_ids: (batch_size, group_size, target_seq_len); log_probs: (batch_size, group_size, target_seq_len)
        batch_size, group_size, target_seq_len = generate_ids.size()

        # 2. Compute ROUGE rewards for each generated sample
        generated_texts = []
        for batch in range(batch_size):
            for k in range(group_size):
                generated_text = tokenizer.decode(generate_ids[batch, k], skip_special_tokens=True)
                generated_texts.append(generated_text)

        reference_texts = []
        for batch in range(batch_size):
            reference_text = tokenizer.decode(labels[batch], skip_special_tokens=True)
            reference_texts.append(reference_text)

        # 3. Compute Rewards
        rewards = reward_calculator.compute_rewards(
            generated_summaries = generated_texts,
            reference_summaries = reference_texts,
            group_size = group_size
        ).to(device) # (batch_size, group_size)

        # 4. Compute Advantages
        advantages = reward_calculator.compute_advantages(rewards) # (batch_size, group_size

        # 5. Compute GRPO Loss
        # Firstly, Create attention mask for generated sequences
        attention_mask_for_generation = (generate_ids != tokenizer.pad_token_id).float() # (batch_size, group_size, target_seq_len)
        loss = loss_calculator.compute_loss(
            log_probs = log_probs,
            advantages = advantages,
            attention_mask = attention_mask_for_generation
        )

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.mean().item()

        if (step + 1) % 30 == 0:
            print(f" >> Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Avg Reward: {rewards.mean().item():.4f}")

            # Print Sample Generation
            if step == 0:
                print(" >> Sample Generation:")
                for k in range(group_size):
                    print(f"    - Generated Sample {k+1}: {generated_texts[k]} | Reward: {rewards[0, k].item():.4f}")
                    print("========================================================================================")

    avg_loss = total_loss / len(dataloader)
    avg_reward = total_reward / len(dataloader)

    return avg_loss, avg_reward

# 5. Evaluation Loop
def evaluate_grpo(model: nn.Module, dataloader: DataLoader,
                  device: torch.device, tokenizer: AutoTokenizer,
                  group_size: int = 5) -> Dict[str, Any]:
    model.eval()
    sampler = GRPOSampler(model, tokenizer, group_size, temperature=0.7)
    reward_calculator = ROUGERewardCalculator()

    all_rewards = []
    rouge1_scores = []
    rougeL_scores = []
    sample_outputs = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 1. Generate group samples
            generate_ids, _ = sampler.generate_group_samples(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length = 128
            ) # (batch_size, group_size, target_seq_len)
            batch_size, group_size, target_seq_len = generate_ids.size()

            # 2. Compute ROUGE rewards
            generated_texts = []
            for batch in range(batch_size):
                for k in range(group_size):
                    generated_text = tokenizer.decode(generate_ids[batch, k], skip_special_tokens=True)
                    generated_texts.append(generated_text)

            reference_texts = []
            for batch in range(batch_size):
                reference_text = tokenizer.decode(labels[batch], skip_special_tokens=True)
                reference_texts.append(reference_text)

            # 3. Compute Rewards
            rewards = reward_calculator.compute_rewards(
                generated_summaries = generated_texts,
                reference_summaries = reference_texts,
                group_size = group_size
            )
            
            # Save the First Batch Samples for Inspection
            if step == 0:
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                sample_outputs.append({
                    'input': input_text,
                    'generated': [generated_texts[k] for k in range(group_size)],
                    'reference': reference_texts[0],
                    'rewards': rewards[0].tolist()
                })
    
    # Compute statistics
    all_rewards = torch.cat(all_rewards, dim=0)  # [total_samples, group_size]
    
    results = {
        'mean_reward': all_rewards.mean().item(),
        'max_reward': all_rewards.max().item(),
        'min_reward': all_rewards.min().item(),
        'std_reward': all_rewards.std().item(),
        'samples': sample_outputs
    }
    
    return results

    
# 6. Data Collection and Preparation
def custom_collate_fn(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
        Custom Collate Function to prepare batch data for GRPO training/evaluation.

        Args:
            - batch: List of data samples from the dataset

    """
    dialogues = [item['dialogue'] for item in batch]
    summaries = [item['summary'] for item in batch]

    # Tokenize inputs (dialogues)
    inputs = tokenizer(
        dialogues,
        max_length = 512,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )

    # Tokenize targets (summaries)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            summaries,
            max_length = 128,
            padding = True,
            truncation = True,
            return_tensors = 'pt'
        )

    target_ids = targets["input_ids"]
    decoder_input_ids = torch.full_like(target_ids, tokenizer.pad_token_id)
    decoder_input_ids[:, 1:] = target_ids[:, :-1] # Shift right

    batch_data = {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': targets.input_ids,
        'decoder_attention_mask': targets.attention_mask,
        'decoder_input_ids': decoder_input_ids
    }

    return batch_data
    
# 7. Main Execution
def main():
    print(" >> GRPO Training and Evaluation Script <<")
    print("============================================")

    # Load Dataset
    print(" >> Loading SAMSum Dataset...")
    dataset = load_dataset("knkarthick/samsum")
    print(" [DBG] Train Size:", len(dataset['train']))
    print(" [DBG] Train Sample:", dataset['train'][0])
    print(" [DBG] Validation Size:", len(dataset['validation']))
    print(" [DBG] Validation Sample:", dataset['validation'][0])
    print(" [DBG] Test Size:", len(dataset['test']))
    print(" [DBG] Test Sample:", dataset['test'][0])

    # Initionalize Tokenizer and Model
    print(" >> Initializing Tokenizer and Model...")
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f" >> Using Device: {device}")
    print(f" >> Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # GRPO Hyperparameters
    group_size = 5
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    beta = 0.1
    learning_rate = 5e-5
    batch_size = 10
    num_epochs = 3

    print(" >> Setting up GRPO Components...")
    print("    - Group Size:", group_size)
    print("    - Temperature:", temperature)
    print("    - Top-K:", top_k)
    print("    - Top-P:", top_p)
    print("    - Beta:", beta)
    print("    - Learning Rate:", learning_rate)
    print("    - Batch Size:", batch_size)
    print("    - Number of Epochs:", num_epochs)

    # Prepare GRPO data
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)

    # Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    print(" >> Starting GRPO Training...")
    for epoch in range(num_epochs):
        print(f" >> Epoch {epoch+1}/{num_epochs}")
        
        train_loss, avg_reward = train_grpo(
            model = model,
            tokenizer=tokenizer,
            dataloader = train_loader,
            optimizer = optimizer,
            device = device,
            group_size = group_size,
            temperature = temperature,
            # top_k = top_k,
            # top_p = top_p,
            beta = beta
        )
        print(f" >> Epoch {epoch+1} Training Result:")
        print(f"    - Training Loss: {train_loss:.4f}, Average Reward: {avg_reward:.4f}")

        # Evaluation Loop
        print(" >> Starting GRPO Evaluation on Test Set...")
        eval_results = evaluate_grpo(
            model = model,
            dataloader = test_loader,
            device = device,
            group_size = group_size
        )
        print(f" >> Epoch {epoch+1} Evaluation Results:")
        print(f"    - ROUGE-1: {eval_results['rouge1']:.4f}, ROUGE-L: {eval_results['rougeL']:.4f}")
        print(f"    - Average Reward: {eval_results['avg_reward']:.4f}")
        print(f"    - Min/Max Reward: {eval_results['min_reward']:.4f}/{eval_results['max_reward']:.4f}")
        print(f"    - Std Reward: {eval_results['std_reward']:.4f}")

        # Log evaluation results
        if eval_results["samples"]:
            sample = eval_results["samples"][0]
            print("    - Sample Generation:")
            print("      * Input Dialogue:", sample["input_dialogue"])
            print("      * Reference Summary:", sample["reference_summary"])
            print("      * Generated Summary:", sample["generated_summary"])
            for i, (gen_sum, reward) in enumerate(sample["group_samples"]):
                print(f"        - Group Sample {i+1}: {gen_sum} | Reward: {reward:.4f}")

        print("============================================")

    print(" >> GRPO Training and Evaluation Completed.")

    # Save the final model
    model_save_path = "./grpo_samsum_model"
    torch.save(model.state_dict(), model_save_path)
    print(f" >> Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
