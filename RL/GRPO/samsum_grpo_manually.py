import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import List, Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
import math
import numpy as np
from tqdm import tqdm

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

# ========================================================================
# Supervised Fine-Tuning (SFT) Section for Policy Function Initialization
# ========================================================================

def sft_collate_fn(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
        Collate function for Supervised Fine-Tuning (SFT).
        Pads input and target sequences to the maximum length in the batch.

        Args:
            - batch: List of samples from the dataset
            - tokenizer: Tokenizer for padding

        Returns:
            - collated_batch: Dictionary containing padded input_ids, attention_mask, labels
    """
    dialogues = [item['dialogue'] for item in batch]
    summaries = [item['summary'] for item in batch]

    # Format with chat_template
    formatted_texts = []
    for dialogue, summary in zip(dialogues, summaries):
        messages = [
            {"role": "system", "content": "You are an expert summarizer. Summarize the following dialogue concisely."},
            {"role": "user", "content": dialogue},
            {"role": "assistant", "content": summary}
        ]
        text = tokenizer.apply_chat_template(
            conversation = messages,
            tokenize = False,
            add_generation_prompt = False
        )
        formatted_texts.append(text)

    # Tokenize inputs
    encodings = tokenizer(
        formatted_texts,
        max_length = 512,
        padding = True,
        truncation = True,
        return_tensors = 'pt'
    )

    # Labels are the same as input_ids for causal LM
    labels = encodings.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100 # Ignore padding tokens in loss computation

    return {
        'input_ids': encodings.input_ids,
        'attention_mask': encodings.attention_mask,
        'labels': labels
    }

def train_sft(model: nn.Module, dataloader: DataLoader,
              optimizer: torch.optim.Optimizer, scheduler: Any,
              device: torch.device, epoch: int, num_epochs: int = 3) -> float:

    """
        Supervised Fine-Tuning (SFT) Training Loop.
        This loop is for initializing the policy model before applying GRPO.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Step [{step+1}/{len(dataloader)}] | Average Loss: {avg_loss:.4f} | Loss: {loss.item():.4f}")

    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss

def evaluate_sft(model: nn.Module, dataloader: DataLoader,
                device: torch.device, tokenizer: AutoTokenizer) -> Dict[str, float]:
    model.eval()
    rouge = evaluate.load('rouge')

    all_predictions = []
    all_references = []
    sample_outputs = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="SFT Evaluation")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate summaries
            model_to_use = model.module if isinstance(model, nn.DataParallel) else model
            generated_ids = model_to_use.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 64,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )

            # Decode generated summaries
            generated_summaries = tokenizer.batch_decode(
                generated_ids[:, input_ids.size(1):],
                skip_special_tokens = True,
            )

            all_predictions.extend(generated_summaries)
            all_references.extend(batch["summary"])

            # Store sample outputs for analysis
            if step == 0:
                for i in range(min(3, len(generated_summaries))):
                    sample_outputs.append({
                        "generated": generated_summaries[i],
                        "reference": batch["summary"][i]
                    })

    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions = all_predictions,
        references= all_references,
        use_stemmer = True
    )

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "samples": sample_outputs
    }

# ========================================
# Group Reward Policy Optimization (GRPO)
# ========================================

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
                               max_new_tokens: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        # Going to use Llama model
        """
            Generate K size of Group Samples for each input in the batch.

            Args:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - max_length: maximum length of generated summaries
            
            Returns:
                - generated_ids: (batch_size, group_size, target_seq_len)
                - log_probs: (batch_size, group_size, target_seq_len) -- log probabilities of each generated token

            ====
            >> The reason why use max_new_tokens, not max_length <<
        """
        self.model.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        """
            Debug Remark:
                
                (Pdb) input_ids.shape
                torch.Size([10, 403])   
                (Pdb) batch_size
                10

        """

        all_generated_ids = []
        max_length = 0 # Initialize max_length for generated sequences -> we use dynamically

        # breakpoint()

        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Generate K size of samples
        for k in range(self.group_size):
            # Since it's just a sampling generation, don't need to compute for gradients
            with torch.no_grad():
                # Use model's generate method with sampling
                generated_outputs = model_to_use.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    do_sample = True,
                    max_new_tokens = max_new_tokens,
                    temperature = self.temperature,
                    top_k = self.top_k,
                    top_p = self.top_p,
                    pad_token_id = self.tokenizer.pad_token_id,
                    eos_token_id = self.tokenizer.eos_token_id,
                    return_dict_in_generate = False
                    # Note: return_dict_in_generate=False to get only generated_ids
                )
                # breakpoint()
                """
                    Debug Remark:
                        (Pdb) generated_outputs
                        tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                                [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                                [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                                ...,
                                [128000, 128000, 128006,  ...,    627,     12,   8529],
                                [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                                [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                            device='cuda:0')
                        (Pdb) generated_outputs.shape
                        torch.Size([10, 531])
                """
                #################################################################################
                """
                    return_dict_in_generate:
                        - If True, returns a 'GenerateOutput(dict-like Object)' containing more information, such as logits, scores, attentions, etc.
                        - If False, return Tensor or tuple, with only generated_ids(simple token ID sequences).

                    ```
                        >>> outputs = model.generate(
                        ...     input_ids,
                        ...     max_length=10,
                        ...     do_sample=False,
                        ...     return_dict_in_generate=False
                        ... )
                            The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
                            Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
                        >>> print(type(outputs))
                            <class 'torch.Tensor'>
                        >>> print(outputs)
                            tensor([[15496,    11,   616,  1438,   318,  1757,    13,   314,  1101,   257]])
                        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
                            Hello, my name is John. I'm a

                        >>> outputs = model.generate(
                        ...     input_ids,
                        ...     max_length=10,
                        ...     do_sample=False,
                        ...     return_dict_in_generate=True,
                        ...     output_scores=True
                        ... )
                            The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
                            Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
                        >>> print(type(outputs))
                            <class 'transformers.generation.utils.GenerateDecoderOnlyOutput'>
                        >>> print(outputs.keys())
                            dict_keys(['sequences', 'scores', 'past_key_values'])
                        >>> print(outputs.sequences)
                            tensor([[15496,    11,   616,  1438,   318,  1757,    13,   314,  1101,   257]])
                        >>> print(outputs.scores[0].shape)
                            torch.Size([1, 50257])
                    ```
                """
                # generated_outputs: (batch_size, generated_seq_lens)
            all_generated_ids.append(generated_outputs)
            max_length = max(max_length, generated_outputs.size(1))

        # breakpoint()
        """
            Debug Remark:
                (Pdb) generated_outputs
                tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,   1005,     13, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0')
                (Pdb) generated_outputs.shape
                torch.Size([10, 518])
                (Pdb) max_length
                531
                (Pdb) generated_outputs.size(1)
                518
                (Pdb) all_generated_ids
                [tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,    627,     12,   8529],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,  13263,   9499,     13],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ...,  16986,     13, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,     13, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,  38156,     13, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,   1005,     13, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0')]
                (Pdb) all_generated_ids.shape
                *** AttributeError: 'list' object has no attribute 'shape'
        """
        
        # Pad all sequences to max_length
        padded_ids = []
        for generate_idx in all_generated_ids:
            if generate_idx.size(1) < max_length:
                padding = torch.full(
                    (batch_size, max_length - generate_idx.size(1)),
                    self.tokenizer.pad_token_id,
                    dtype=generate_idx.dtype,
                    device=device
                )
                generate_idx = torch.cat([generate_idx, padding], dim=1) # Pad on the right
            padded_ids.append(generate_idx)

        # breakpoint()
        # 구뜨 ㅎㅎㅎ 잘 넘어간거 확인함
        """
            Debug Remark:
                (Pdb) padded_ids
                [tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,    627,     12,   8529],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ...,  13263,   9499,     13],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0'), tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0')]
                (Pdb) max_length
                531
                (Pdb) pad_token_id
                *** NameError: name 'pad_token_id' is not defined
                (Pdb) self.tokenizer.pad_token_id
                128009
                (Pdb) padding
                tensor([[128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009],
                        [128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
                        128009, 128009, 128009, 128009]], device='cuda:0')
                (Pdb) generate_idx
                tensor([[128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        ...,
                        [128000, 128000, 128006,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009],
                        [128009, 128009, 128009,  ..., 128009, 128009, 128009]],
                    device='cuda:0')
                (Pdb) len(generate_idx)
                10
        """

        # Stack generated ids: (batch_size, group_size, target_seq_len)
        generated_ids = torch.stack(padded_ids, dim=1) # dim=1 for group_size

        # Compute log probabilities for each generated sequence -- for generated part only
        log_probs = self._compute_log_probs(generated_ids, input_ids.size(1)) # (batch_size, group_size, target_seq_len)

        # breakpoint()
        """
            Debug Remark:
                (Pdb) log_probs.shape
                torch.Size([10, 5, 128])
                (Pdb) generated_ids.shape
                torch.Size([10, 5, 531])
                (Pdb) padded_ids.shape
                *** NameError: name 'padded_ids' is not defined

                >> The reason why 'padded_ids' was set earlier, but not defined in this scope. <<

        """

        return generated_ids, log_probs

    def _compute_log_probs(self, generated_ids: torch.Tensor,
                           input_length: int) -> torch.Tensor:
        """
            Compute Log Probabilities of generated sequences, Manually.

            Args:
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - generated_ids: (batch_size, group_size, total_seq_len)

            Returns:
                - log_probs: (batch_size, group_size, generated_seq_len)
        """
        batch_size, group_size, total_seq_len = generated_ids.size() # Extract sizes
        device = generated_ids.device

        # breakpoint()
        """
            Debug Remark:
                (Pdb) generated_ids.shape
                torch.Size([10, 5, 531])
                (Pdb) batch_size
                10
                (Pdb) group_size
                5
                (Pdb) total_seq_len
                531
        """
        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        all_log_probs = []

        # Process each sample in group
        for k in range(group_size):

            # breakpoint()
        
            curr_seq = generated_ids[:, k, :] # (batch_size, total_seq_len)

            outputs = model_to_use(
                    input_ids = curr_seq,
                    labels = curr_seq
                )
            logits = outputs.logits # (batch_size, target_seq_len, vocab_size)

            # Compute log probabilities
            log_probs_dist = F.log_softmax(logits, dim=-1) # (batch_size, target_seq_len, vocab_size)

            # Gather Log Probabilities of generated tokens
            token_log_probs = torch.gather(
                log_probs_dist,
                dim = 2, # vocab dimension
                index = curr_seq.unsqueeze(-1) # (batch_size, target_seq_len, 1)
            ).squeeze(-1) # (batch_size, target_seq_len)
            """
                >> Why unsqueeze and squeeze? <<
                    - unsqueeze: to match dimensions for gather
                    - squeeze: to remove the last dimension after gather

                    To extract elements from logits based on generated token ids, create an extra dimension with unsqueeze(index),
                    then remove it after gathering with squeeze().
            """

            # Extract log probabilities for generated tokens only
            # Shift by 1 because logits at position t corresponds to token at position t+1
            generated_log_probs = token_log_probs[:, input_length - 1: -1] # (batch_size, generated_seq_len)
            all_log_probs.append(generated_log_probs)

            # breakpoint()
            """
                Debug Remark:
                    (Pdb) logits.shape
                    torch.Size([10, 606, 128256])
                    (Pdb) curr_seq.shape
                    torch.Size([10, 606])    
                    (Pdb) generated_log_probs.shape
                    torch.Size([10, 128])
                    (Pdb) token_log_probs.shape
                    torch.Size([10, 606])
                    (Pdb) input_length
                    478     
            """
        
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

# 4. Main Training Script Structure (Training Loop)
def train_grpo_llama(model: nn.Module, dataloader: DataLoader, 
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
        # breakpoint()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        reference_summaries = batch['summaries']  # List of strings

        if epoch == 0 and step == 0:
            print(" [DBG] Input IDs Shape:", input_ids.shape)
            print(" [DBG] Attention Mask Shape:", attention_mask.shape)
            print(" [DBG] Reference Summaries Shape:", reference_summaries.shape)
            print(" [DBG] Group Size:", group_size)
            print("============================================")

        # 1. Generate group samples (K Samples per input)
        generated_ids, log_probs = sampler.generate_group_samples(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = 32
        ) 

        # breakpoint()
        
        # generated_ids: (batch_size, group_size, total_seq_len)
        # log_probs: (batch_size, group_size, generated_seq_len)
        batch_size, group_size, total_seq_len = generated_ids.size()
        generated_seq_len = total_seq_len - input_ids.size(1)

        # breakpoint()
        
        # 2. Compute ROUGE rewards for each generated sample
        generated_texts = []
        for batch in range(batch_size):
            for k in range(group_size):
                # Extract only the generated tokens (After input length)
                generated_tokens = generated_ids[batch, k, input_ids.size(1):] # (generated_seq_len)
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)

        # breakpoint()
        
        # 3. Compute Rewards
        rewards = reward_calculator.compute_rewards(
            generated_summaries = generated_texts,
            reference_summaries = reference_summaries,
            group_size = group_size
        ).to(device) # (batch_size, group_size)

        # breakpoint()
        
        # 4. Compute Advantages
        advantages = reward_calculator.compute_advantages(rewards) # (batch_size, group_size

        # 5. Compute GRPO Loss
        # Firstly, Create attention mask for generated sequences
        attention_mask_for_generation = (generated_ids[:, :, input_ids.size(1):] != tokenizer.pad_token_id).float()

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

        if (step + 1) % 5 == 0:
            print(f" >> Step [{step+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Avg Reward: {rewards.mean().item():.4f}")

            # # Print Sample Generation
            # if step == 0:
            #     print(" >> Sample Generation:")
            #     for k in range(group_size):
            #         print(f"    - Generated Sample {k+1}: {generated_texts[k]} | Reward: {rewards[0, k].item():.4f}")
            #         print("========================================================================================")

            first_input_ids = input_ids[0]
            first_attention_mask = attention_mask[0]

            full_padded_input_text = tokenizer.decode(first_input_ids, skip_special_tokens=False)

            # Extract the exact Index that 1 is started in attention mask (which represents for the padding end)
            start_index = torch.where(first_attention_mask == 1)[0][0].item()
            # Slicing the actual input tokens (without padding)
            non_padded_input_tokens = first_input_ids[start_index:]
            clean_input_text = tokenizer.decode(non_padded_input_tokens, skip_special_tokens=False)

            first_input_text = tokenizer.decode(non_padded_input_tokens, skip_special_tokens=True)

            print(" >> Sample Generation:")
            print(f"    - Group Samples (K={group_size}):")
            print(f"    - Input Tokens (Full, including PADDING - for verification):")
            print(full_padded_input_text) 
            print(f"\n    - Input Dialogue (Clean Prompt, Special Tokens Intact - for context):")
            print(clean_input_text)
            print(f"   - Reference Summary: {reference_summaries[0]}")
            for k in range(group_size):
                # generated_texts List is flattened, so calculate the correct index
                sample_index_in_flat_list = k 
                print(f"        -> Generated Sample {k+1}: {generated_texts[sample_index_in_flat_list]} | Reward: {rewards[0, k].item():.4f}")
            print(" -------------------------------------------------------------------")


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
            reference_summaries = batch['summaries']  # List of strings

            # 1. Generate group samples
            generated_ids, _ = sampler.generate_group_samples(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 32
            ) # (batch_size, group_size, total_seq_len)
            batch_size, group_size, total_seq_len = generated_ids.size()

            # 2. Decode Generated Parts only
            generated_texts = []
            for batch in range(batch_size):
                for k in range(group_size):
                    generated_tokens = generated_ids[batch, k, input_ids.size(1):] # (generated_seq_len)
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    generated_texts.append(generated_text)

            # 3. Compute Rewards
            rewards = reward_calculator.compute_rewards(
                generated_summaries = generated_texts,
                reference_summaries = reference_summaries,
                group_size = group_size
            )

            all_rewards.append(rewards)
            
            # Save the First Batch Samples for Inspection
            if step == 0:
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                sample_outputs.append({
                    'input': input_text,
                    'generated': [generated_texts[k] for k in range(group_size)],
                    'reference': reference_summaries[0],
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
def grpo_collate_fn(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
        Custom Collate Function to prepare batch data for GRPO training/evaluation with applying chat template for instruction.

        Args:
            - batch: List of data samples from the dataset

        -> This is a Collate Function for GRPO Training/Evaluation. Only Input Preparation is done here.

    """
    dialogues = [item['dialogue'] for item in batch] 
    summaries = [item['summary'] for item in batch] # Ground Truth Summaries (Targets)

    message_list = []
    for dialogue in dialogues:
        # Define the system instruction and user message
        messages = [
            {"role": "system", "content": "You are an expert summarizer. Summarize the following English dialogue concisely, in English."},
            {"role": "user", "content": dialogue}
            # Note: The expected *assistant* response (the summary) is added later if fine-tuning
            # For pure input preparation (as in the original Llama function):
        ]
        message_list.append(messages)

    # Format Inputs of model with 'apply_chat_template'
    formatted_inputs = [
        tokenizer.apply_chat_template(
            conversation = messages,
            tokenize = False, # Return string, not tokenized tensor
            add_generation_prompt = True # Add assistant prompt for generation
        ) for messages in message_list
    ]

    # Tokenize inputs (dialogues)
    inputs = tokenizer(
        formatted_inputs,
        max_length = 512,
        padding = True,
        truncation = True,
        return_tensors = 'pt',
        add_special_tokens = False
    )

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "summaries": summaries # Still keep as strings for evaluation, reward calculation
    }
    
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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float32,
        device_map = "auto"
    )

    # Set pad token and Ensure for other tokens if not exist
    print(" >> Setting up Tokenizer Special Tokens...")
    print(f"   - Special Tokens Map: {tokenizer.special_tokens_map}")

    if tokenizer.pad_token is None:
        print("    - Setting PAD token as EOS token...")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    print(f"   - PAD Token ID: {tokenizer.pad_token_id}")
    print(f"   - EOS Token ID: {tokenizer.eos_token_id}")
    print(f"   - BOS Token ID: {tokenizer.bos_token_id}")
    """
        Debug Remark:
            - PAD Token ID: 128009
            - EOS Token ID: 128009
            - BOS Token ID: 128000

        -> PAD token is set to EOS token as default.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    print(f" >> Using Device: {device}")
    print(f" >> Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # SFT Hyperparameters
    print(" >> Setting up SFT Components...")
    sft_learning_rate = 2e-5
    sft_batch_size = 4
    sft_num_epochs = 3

    print("    - Learning Rate:", sft_learning_rate)
    print("    - Batch Size:", sft_batch_size)
    print("    - Number of Epochs:", sft_num_epochs)

    # GRPO Hyperparameters
    print(" >> Setting up GRPO Components...")
    grop_group_size = 5
    grop_temperature = 0.7
    grop_top_k = 50
    grop_top_p = 0.95
    grop_beta = 0.1
    grop_learning_rate = 5e-5
    grop_batch_size = 2
    grop_num_epochs = 3

    print("    - Group Size:", grop_group_size)
    print("    - Temperature:", grop_temperature)
    print("    - Top-K:", grop_top_k)
    print("    - Top-P:", grop_top_p)
    print("    - Beta:", grop_beta)
    print("    - Learning Rate:", grop_learning_rate)
    print("    - Batch Size:", grop_batch_size)
    print("    - Number of Epochs:", grop_num_epochs)

    # Load Data for Training 
    train_data = dataset['train'].shuffle(seed=42)
    val_data = dataset['validation'].shuffle(seed=42)

    print("============================================")
    print(" >> Starting SFT Training... <<")
    print("============================================")

    sft_train_loader = DataLoader(train_data, batch_size=sft_batch_size, shuffle=True, collate_fn=lambda x: sft_collate_fn(x, tokenizer))
    sft_val_loader = DataLoader(val_data, batch_size=sft_batch_size, shuffle=False, collate_fn=lambda x: sft_collate_fn(x, tokenizer))

    # SFT Optimizer & Scheduler
    sft_optimizer = AdamW(model.parameters(), lr=sft_learning_rate)
    total_steps = len(sft_train_loader) * sft_num_epochs
    sft_scheduler = get_linear_schedule_with_warmup(
        sft_optimizer,
        num_warmup_steps = int(0.1 * total_steps),
        num_training_steps = total_steps
    )

    best_sft_rouge = 0.0
    for epoch in range(sft_num_epochs):
        print(f" >> SFT Epoch {epoch+1}/{sft_num_epochs}")
        
        sft_train_loss = train_sft(
            model = model,
            dataloader = sft_train_loader,
            optimizer = sft_optimizer,
            scheduler = sft_scheduler,
            device = device,
            epoch = epoch,
            num_epochs = sft_num_epochs
        )
        print(f" >> SFT Epoch {epoch+1} Training Loss: {sft_train_loss:.4f}")

        # SFT Evaluation
        print(" >> Starting SFT Evaluation on Validation Set...")
        eval_results = evaluate_sft(
            model = model,
            dataloader = sft_val_loader,
            device = device,
            tokenizer = tokenizer
        )

        print(f"   - ROGUE-1: {eval_results['rouge1']:.4f}")
        print(f"   - ROGUE-2: {eval_results['rouge2']:.4f}")
        print(f"   - ROGUE-L: {eval_results['rougeL']:.4f}")

        if eval_results["samples"]:
            print("   - Sample Generation:")
            for i, sample in enumerate(eval_results["samples"]):
                print(f"      * Sample {i+1}:")
                print("        - Input Dialogue:", sample["input_dialogue"])
                print("        - Reference Summary:", sample["reference_summary"])
                print("        - Generated Summary:", sample["generated_summary"])

        # Save best SFT model
        if eval_results['rouge1'] > best_sft_rouge:
            best_sft_rouge = eval_results["rougeL"]
            sft_model_save_path = "./sft_samsum_model"
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), sft_model_save_path)
            else:
                torch.save(model.state_dict(), sft_model_save_path)
            print(f" >> Best SFT Model saved to {sft_model_save_path}")
        print("============================================")

    train_loader = DataLoader(train_data, batch_size=grop_batch_size, shuffle=True, collate_fn=lambda x: grpo_collate_fn(x, tokenizer))
    val_loader = DataLoader(val_data, batch_size=grop_batch_size, shuffle=False, collate_fn=lambda x: grpo_collate_fn(x, tokenizer))

    # Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=grop_learning_rate)

    # Training Loop
    print("============================================")
    print(" >> Starting GRPO Training... <<")
    print("============================================")
    for epoch in range(grop_num_epochs):
        print(f" >> Epoch {epoch+1}/{grop_num_epochs}")
        
        train_loss, avg_reward = train_grpo_llama(
            model = model,
            tokenizer=tokenizer,
            dataloader = train_loader,
            optimizer = optimizer,
            device = device,
            group_size = grop_group_size,
            temperature = grop_temperature,
            # top_k = grop_top_k,
            # top_p = grop_top_p,
            beta = grop_beta
        )
        print(f" >> Epoch {epoch+1} Training Result:")
        print(f"    - Training Loss: {train_loss:.4f}, Average Reward: {avg_reward:.4f}")

        # Evaluation Loop
        print(" >> Starting GRPO Evaluation on Test Set...")
        eval_results = evaluate_grpo(
            model = model,
            dataloader = val_loader,
            device = device,
            group_size = grop_group_size
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
