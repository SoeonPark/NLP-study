import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any, Sized, Tuple, Union, Callable

from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
import evaluate
from tqdm import tqdm
import copy


# ============================================================================
# Utility Functions
# ============================================================================

def save_model_and_tokenizer(model: nn.Module, tokenizer: AutoTokenizer, save_directory: str,
                             model_name: str):
    """Save model and tokenizer to disk"""
    save_path = os.path.join(save_directory, model_name)
    os.makedirs(save_path, exist_ok=True)

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(save_path)
    else:
        torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        if hasattr(model_to_save, "config") and hasattr(model_to_save.config, "save_pretrained"):
            model_to_save.config.save_pretrained(save_path)

    tokenizer.save_pretrained(save_path)
    print(f" >> [READY] Model and tokenizer saved to {save_path}")

    return save_path


def load_model_and_tokenizer(model_name: str,
                             use_peft: bool = False,
                             peft_config: Optional[LoraConfig] = None,
                             use_8bit: bool = False) -> Tuple[nn.Module, AutoTokenizer]:
    """Load model and tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    quantization_config = None
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not use_8bit else None,
    )
    
    # Apply PEFT
    if use_peft and peft_config is not None:
        if use_8bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    print(f" >> [READY] Model and tokenizer loaded from {model_name}")
    
    return model, tokenizer


# ============================================================================
# Custom Sampler
# ============================================================================

class RepeatSampler(Sampler):
    """
    Sampler that repeats each data point 'num_generations' times.
    This is used in GRPO to generate multiple completions per prompt.
    """
    def __init__(self, data_source: Sized, mini_repeat_count: int, repeat_count: int = 1,
                 batch_size: int = 1, shuffle: bool = True, seed: int = 42):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count  # num_generations
        self.repeat_count = repeat_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.num_samples = len(data_source)

        if self.shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))
        
        # Split into batches: [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        
        # Remove incomplete batches: [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        
        # Repeat each index mini_repeat_count times for multiple generations
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        num_batches = self.num_samples // self.batch_size
        return num_batches * self.batch_size * self.mini_repeat_count * self.repeat_count


# ============================================================================
# GRPO Configuration
# ============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    
    # Data preprocessing parameters
    num_generations: int = field(default=4, metadata={"help": "Number of generations per input during training"})
    max_prompt_length: int = field(default=512, metadata={"help": "Maximum length of the input prompt"})
    max_completion_length: int = field(default=512, metadata={"help": "Maximum length of the completion"})
    shuffle_dataset: bool = field(default=True, metadata={"help": "Whether to shuffle the dataset during training"})

    # Generation parameters
    generation_batch_size: int = field(default=2, metadata={"help": "Batch size for generation"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling"})
    top_k: int = field(default=0, metadata={"help": "Top-k sampling parameter"})
    top_p: float = field(default=0.9, metadata={"help": "Top-p sampling parameter"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to use sampling"})

    # GRPO training parameters
    beta: float = field(default=0.05, metadata={"help": "KL penalty coefficient"})
    epsilon: float = field(default=0.2, metadata={"help": "Low entropy threshold (bottom percentile)"})
    epsilon_high: Optional[float] = field(default=None, metadata={"help": "High entropy threshold (top percentile)"})
    num_iterations: int = field(default=1, metadata={"help": "Number of GRPO iterations per batch"})
    
    # Optimization parameters
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing"})
    learning_rate: float = field(default=1e-5, metadata={"help": "Learning rate for training"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})


# ============================================================================
# Reward Calculator
# ============================================================================

class ROUGERewardCalculator:
    """Calculate ROUGE-based rewards for text generation"""
    
    def __init__(self, use_rouge_1: bool = True, use_rouge_l: bool = True):
        self.rouge = evaluate.load('rouge')
        self.use_rouge_1 = use_rouge_1
        self.use_rouge_l = use_rouge_l

    def compute_rewards(self, predictions: List[str], references: List[str]) -> torch.Tensor:
        """
        Compute ROUGE rewards for predictions against references.
        
        Args:
            predictions: List of generated texts (batch_size * num_generations)
            references: List of reference texts (batch_size)
        
        Returns:
            rewards: Tensor of shape (batch_size, num_generations)
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            result = self.rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)
            
            if self.use_rouge_1 and self.use_rouge_l:
                score = (result['rouge1'] + result['rougeL']) / 2.0
            elif self.use_rouge_1:
                score = result['rouge1']
            elif self.use_rouge_l:
                score = result['rougeL']
            else:
                raise ValueError("At least one of use_rouge_1 or use_rouge_l must be True.")
            
            scores.append(score)

        rewards = torch.tensor(scores, dtype=torch.float32)
        return rewards
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages by subtracting the baseline (mean reward) from each reward.
        
        Args:
            rewards: Tensor of shape (batch_size, num_generations)
        
        Returns:
            advantages: Tensor of shape (batch_size, num_generations)
        """
        baseline = rewards.mean(dim=1, keepdim=True)  # (batch_size, 1)
        advantages = rewards - baseline  # (batch_size, num_generations)
        return advantages


# ============================================================================
# GRPO Trainer
# ============================================================================

class GRPOTrainer(Trainer):
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    
    Based on the DeepSeekMath paper and TRL implementation.
    """
    
    _tag_names = ["trl", "grpo"]
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        args: TrainingArguments,
        grpo_config: GRPOConfig,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        data_collator: Optional[Callable] = None,
        reward_calculator: Optional[ROUGERewardCalculator] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )
        
        self.ref_model = ref_model
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.grpo_config = grpo_config
        
        # Initialize reward calculator
        if reward_calculator is None:
            self.reward_calculator = ROUGERewardCalculator()
        else:
            self.reward_calculator = reward_calculator
        
        # Multi-step parameters
        self.num_iterations = grpo_config.num_iterations
        self.epsilon_low = grpo_config.epsilon
        self.epsilon_high = grpo_config.epsilon_high if grpo_config.epsilon_high is not None else grpo_config.epsilon
        
        # Enable gradient checkpointing if specified
        if grpo_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _get_train_sampler(self, train_dataset=None) -> Optional[Sampler]:
        """Create custom sampler for GRPO training"""
        if train_dataset is None:
            train_dataset = self.train_dataset
            
        if train_dataset is None:
            return None
        
        return RepeatSampler(
            data_source=train_dataset,
            mini_repeat_count=self.grpo_config.num_generations,
            repeat_count=1,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=self.grpo_config.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[Sampler]:
        """Create sampler for evaluation"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            return None
        
        return torch.utils.data.SequentialSampler(eval_dataset)

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare inputs for model forward pass"""
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.args.device)
        return inputs

    def _get_per_token_logps_and_entropies(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities and entropies.
        
        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            labels: Token IDs of shape (batch_size, seq_len)
            mask: Attention mask of shape (batch_size, seq_len)
        
        Returns:
            log_probs: Per-token log probabilities of shape (batch_size, seq_len)
            entropies: Per-token entropies of shape (batch_size, seq_len)
        """
        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len - 1, vocab_size)
        shift_labels = labels[:, 1:].contiguous()      # (batch_size, seq_len - 1)
        shift_mask = mask[:, 1:].contiguous()          # (batch_size, seq_len - 1)
        
        # Compute log probabilities
        log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len - 1, vocab_size)
        
        # Gather log probs for the actual tokens
        log_probs = torch.gather(
            log_probs_all,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, seq_len - 1)
        
        # Compute entropies
        probs = F.softmax(shift_logits, dim=-1)  # (batch_size, seq_len - 1, vocab_size)
        entropies = -(probs * log_probs_all).sum(dim=-1)  # (batch_size, seq_len - 1)
        
        # Apply mask
        log_probs = log_probs * shift_mask
        entropies = entropies * shift_mask
        
        # Pad to match original sequence length
        batch_size, seq_len = labels.shape
        log_probs_padded = torch.zeros(batch_size, seq_len, device=log_probs.device, dtype=log_probs.dtype)
        entropies_padded = torch.zeros(batch_size, seq_len, device=entropies.device, dtype=entropies.dtype)
        
        log_probs_padded[:, 1:] = log_probs
        entropies_padded[:, 1:] = entropies
        
        return log_probs_padded, entropies_padded

    def _get_high_entropy_mask(
        self,
        entropies: torch.Tensor,
        mask: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Create mask for high-entropy tokens.
        
        Args:
            entropies: Per-token entropies of shape (batch_size, seq_len)
            mask: Attention mask of shape (batch_size, seq_len)
            threshold: Entropy percentile threshold (e.g., 0.8 for top 20%)
        
        Returns:
            high_entropy_mask: Binary mask of shape (batch_size, seq_len)
        """
        # Get valid entropies (where mask is True)
        valid_entropies = entropies[mask.bool()].float()
        
        if valid_entropies.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)
        
        # Compute threshold
        entropy_threshold = torch.quantile(valid_entropies, threshold)
        
        # Create mask for high entropy tokens
        masked_entropies = entropies * mask.float()
        entropy_mask = (masked_entropies >= entropy_threshold).bool()
        
        return entropy_mask

    def _generate_completions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate completions for given prompts.
        
        Args:
            input_ids: Prompt token IDs of shape (batch_size, prompt_len)
            attention_mask: Prompt attention mask of shape (batch_size, prompt_len)
        
        Returns:
            generated_ids: Generated sequences of shape (batch_size, total_len)
            generated_mask: Attention mask for generated sequences
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.grpo_config.max_completion_length,
                temperature=self.grpo_config.temperature,
                top_k=self.grpo_config.top_k,
                top_p=self.grpo_config.top_p,
                do_sample=self.grpo_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        self.model.train()
        
        # Create attention mask for generated sequences
        generated_mask = (outputs != self.tokenizer.pad_token_id).long()
        
        return outputs, generated_mask

    def _compute_rewards(
        self,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        references: List[str],
        prompt_lengths: List[int]
    ) -> torch.Tensor:
        """
        Compute rewards for generated completions.
        
        Args:
            generated_ids: Generated token IDs of shape (batch_size * num_generations, total_len)
            generated_mask: Attention mask of shape (batch_size * num_generations, total_len)
            references: List of reference texts (batch_size * num_generations) - already expanded
            prompt_lengths: List of prompt lengths (batch_size * num_generations)
        
        Returns:
            rewards: Tensor of shape (batch_size, num_generations)
        """
        # Decode only the completion part (excluding the prompt)
        completions = []
        for i, (ids, mask, prompt_len) in enumerate(zip(generated_ids, generated_mask, prompt_lengths)):
            # Extract completion tokens (after prompt)
            completion_ids = ids[prompt_len:]
            completion_mask = mask[prompt_len:]
            
            # Get valid completion tokens
            valid_length = completion_mask.sum().item()
            completion_ids = completion_ids[:valid_length]
            
            # Decode
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            completions.append(completion_text)
        
        # Compute rewards (references are already expanded by RepeatSampler)
        rewards = self.reward_calculator.compute_rewards(completions, references)
        
        # Reshape to (batch_size, num_generations)
        total_samples = len(completions)
        num_generations = self.grpo_config.num_generations
        batch_size = total_samples // num_generations
        
        rewards = rewards.view(batch_size, num_generations)
        
        return rewards

    def _generate_and_score_completions(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Generate completions and compute rewards.
        
        Args:
            inputs: Dictionary containing input_ids, attention_mask, and references
        
        Returns:
            Dictionary containing:
                - generated_ids: Generated sequences
                - generated_mask: Attention masks
                - rewards: Reward values
                - advantages: Computed advantages
                - ref_log_probs: Reference model log probabilities
                - prompt_lengths: List of prompt lengths
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        references = inputs.get("references", inputs.get("labels", None))
        
        # Store prompt lengths
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        
        # Generate completions
        generated_ids, generated_mask = self._generate_completions(input_ids, attention_mask)
        
        # Compute rewards
        batch_size = len(references)
        rewards = self._compute_rewards(
            generated_ids,
            generated_mask,
            references,
            prompt_lengths * self.grpo_config.num_generations
        )
        
        # Compute advantages
        advantages = self.reward_calculator.compute_advantages(rewards)  # (batch_size, num_generations)
        
        # Expand advantages to match generation shape
        advantages_expanded = advantages.view(-1).unsqueeze(-1)  # (batch_size * num_generations, 1)
        advantages_expanded = advantages_expanded.expand(-1, generated_ids.size(1))  # Match seq_len
        
        # Compute reference model log probabilities
        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=generated_ids,
                attention_mask=generated_mask
            ).logits
            
            ref_log_probs, _ = self._get_per_token_logps_and_entropies(
                ref_logits,
                generated_ids,
                generated_mask
            )
        
        return {
            "generated_ids": generated_ids,
            "generated_mask": generated_mask,
            "rewards": rewards,
            "advantages": advantages_expanded,
            "ref_log_probs": ref_log_probs,
            "prompt_lengths": prompt_lengths,
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute GRPO loss.
        
        Args:
            model: The model to train
            inputs: Dictionary containing input data
            return_outputs: Whether to return additional outputs
        
        Returns:
            loss: Computed loss value
            outputs (optional): Dictionary of additional outputs
        """
        # Step 1: Generate and score completions
        gen_data = self._generate_and_score_completions(inputs)
        
        # Step 2: Get current policy log probabilities and entropies
        logits = model(
            input_ids=gen_data["generated_ids"],
            attention_mask=gen_data["generated_mask"]
        ).logits
        
        log_probs, entropies = self._get_per_token_logps_and_entropies(
            logits,
            gen_data["generated_ids"],
            gen_data["generated_mask"]
        )
        
        # Step 3: Compute entropy masks
        high_entropy_mask = self._get_high_entropy_mask(
            entropies,
            gen_data["generated_mask"],
            self.epsilon_high
        )
        
        low_entropy_mask = self._get_high_entropy_mask(
            entropies,
            gen_data["generated_mask"],
            1.0 - self.epsilon_low  # Invert for low entropy
        )
        low_entropy_mask = ~low_entropy_mask  # Get the complement
        
        # Combine masks: train on high OR low entropy tokens
        training_mask = (high_entropy_mask | low_entropy_mask) & gen_data["generated_mask"].bool()
        training_mask = training_mask.float()
        
        # Step 4: Compute policy ratio
        # ratio = π_θ(a|s) / π_ref(a|s) = exp(log π_θ - log π_ref)
        log_ratio = log_probs - gen_data["ref_log_probs"]
        ratio = torch.exp(log_ratio)
        
        # Step 5: Compute policy loss
        # L_policy = -E[ratio * advantage]
        policy_loss = -(ratio * gen_data["advantages"] * training_mask).sum() / (training_mask.sum() + 1e-8)
        
        # Step 6: Compute KL divergence loss
        # KL(π_θ || π_ref) ≈ log π_θ - log π_ref
        kl_loss = (log_ratio * training_mask).sum() / (training_mask.sum() + 1e-8)
        
        # Step 7: Total loss
        loss = policy_loss + self.grpo_config.beta * kl_loss
        
        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            mean_reward = gen_data["rewards"].mean().item()
            mean_advantage = gen_data["advantages"].mean().item()
            
            self.log({
                "train/loss": loss.item(),
                "train/policy_loss": policy_loss.item(),
                "train/kl_loss": kl_loss.item(),
                "train/mean_reward": mean_reward,
                "train/mean_advantage": mean_advantage,
                "train/mean_entropy": entropies.mean().item(),
            })
        
        if return_outputs:
            outputs = {
                "loss": loss,
                "policy_loss": policy_loss,
                "kl_loss": kl_loss,
                "rewards": gen_data["rewards"],
                "advantages": gen_data["advantages"],
            }
            return loss, outputs
        
        return loss


# ============================================================================
# Data Collator
# ============================================================================

class GRPODataCollator:
    """Custom data collator for GRPO training"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of data samples.
        
        Args:
            features: List of data samples, each containing 'dialogue' and 'summary'
        
        Returns:
            Dictionary with batched tensors
        """
        # Extract dialogues and summaries
        dialogues = [f["dialogue"] for f in features]
        summaries = [f["summary"] for f in features]
        
        # Tokenize prompts (dialogues)
        tokenized = self.tokenizer(
            dialogues,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "references": summaries,
        }


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function"""
    
    # Configuration
    training_args = TrainingArguments(
        output_dir="./grpo_samsum_checkpoints",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        eval_strategy="no",
        save_total_limit=2,
        learning_rate=1e-5,
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    
    grpo_config = GRPOConfig(
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=128,
        temperature=1.0,
        top_p=0.9,
        beta=0.05,
        epsilon=0.2,
        epsilon_high=0.8,
        num_iterations=1,
        shuffle_dataset=True,
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f" >> Loading model: {model_name}")
    
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        use_peft=True,
        peft_config=lora_config,
        use_8bit=False
    )
    
    # Create reference model (frozen copy)
    print(" >> Creating reference model...")
    ref_model, _ = load_model_and_tokenizer(
        model_name=model_name,
        use_peft=False,
        use_8bit=False
    )
    
    # Load dataset
    print(" >> Loading SAMSum Dataset...")
    dataset = load_dataset("knkarthick/samsum")
    train_dataset = dataset["train"].select(range(100))  # Use subset for testing
    eval_dataset = dataset["validation"].select(range(20))
    
    # Data collator
    data_collator = GRPODataCollator(tokenizer, max_length=grpo_config.max_prompt_length)
    
    # Initialize trainer
    print(" >> Initializing GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        grpo_config=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print(" >> Starting GRPO Training...")
    trainer.train()
    
    # Save final model
    print(" >> Saving final model...")
    save_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        save_directory="./grpo_samsum_checkpoints",
        model_name="final_model"
    )
    
    print(" >> Training completed!")


if __name__ == "__main__":
    main()
