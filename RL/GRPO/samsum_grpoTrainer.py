import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #"0,1"

import warnings
warnings.filterwarnings("ignore")

import wandb
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
import evaluate
from typing import List, Dict, Any, Optional

# def collate_fn_dynamic_padding(batch: List[Dict[str, Any]], 
#                                tokenizer: AutoTokenizer, 
#                                max_length: int = 512):
#     """
#     Dynamic Padding Collate Function for DataLoader.
#     Pads input sequences in the batch to the length of the longest sequence.
#     """
#     input_ids = [item['input_ids'] for item in batch]
#     attention_masks = [item['attention_mask'] for item in batch]
    
#     # Pad sequences
#     padded_inputs = tokenizer.pad(
#         {'input_ids': input_ids, 'attention_mask': attention_masks},
#         padding=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )
    
#     return padded_inputs

def formatting_prompts_func(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[str]]:
    """    
    GRPOTrainer Input Formatting:
    {
        "query": Prompt Text,
        "reference": Reference Summary (for reward calculation)
    }

    Applies "apply_chat_template" function to each example in the dataset.
    """

    formatted_examples = {
        "prompt": [],
        "reference": []
    }

    for dialog, summary in zip(examples["dialogue"], examples["summary"]):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": dialog}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )

        formatted_examples["prompt"].append(prompt)
        formatted_examples["reference"].append(summary)

    return formatted_examples

def compute_rouge_reward(summaries: List[str], references: List[str]) -> List[float]:
    """
    Computes ROUGE-based rewards for generated summaries.
    """
    rouge = evaluate.load("rouge")
    rewards = []

    for summary, reference in zip(summaries, references):
        result = rouge.compute(
            predictions=[summary],
            references=[reference],
            use_stemmer=True
        )

        # Using ROUGE-L F1 score as reward
        reward = result["rougeL"]
        rewards.append(reward)

    return rewards

def compute_rouge_reward_batch(prompts: List[str], completions: List[str], 
                               *, reference: None, num_generations: int = 1,
                               **kwargs) -> List[float]:
    assert reference is not None
    rouge = evaluate.load("rouge")
    rewards = []

    for i, gen in enumerate(completions):
        ref = reference[i // num_generations] 
        score = rouge.compute(predictions=[gen], references=[ref], use_stemmer=True)["rougeL"]
        rewards.append(float(score))
    return rewards

def main():
    print(" >> Fine-tuning using GRPOTrainer << ")
    
    # 1. Load and prepare SAMSum dataset
    print("\n[1/6] Loading SAMSum Dataset...")
    dataset = load_dataset("knkarthick/samsum")

    # 2. Initialize Model and Tokenizer
    print("\n[2/6] Initializing Model and Tokenizer...")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for causal models
    
    # Formatting dataset
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000)).map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    eval_dataset = dataset["validation"].shuffle(seed=42).select(range(200)).map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Sample query: {train_dataset[0]['prompt'][:100]}...")
    
    # Load pre-trained model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # 8-bit quantization
    )

    # # Reference model (frozen)
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     load_in_8bit=True
    # )
    
    # for param in ref_model.parameters():
    #     param.requires_grad = False
    # ref_model.eval()
    
    print(f"   Model loaded: {model_name}")
    
    # 3. Setting up LoRA
    print("\n[3/6] Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 4. Configure GRPOTrainer
    print("\n[4/6] Configuring GRPOTrainer...")
    
    grpo_config = GRPOConfig(
        # vllm_device="cuda:1",

        # Basic training parameters
        output_dir="./grpo_samsum_output",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Params for generation (GRPO specific)
        num_generations=4,  # group_size
        temperature=0.7,
        max_completion_length=64,
        vllm_gpu_memory_utilization=0.5,

        # Optimization settings
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,
        loss_type="grpo",
        
        # KL divergence penalty
        beta=0.1,  # beta
        
        # Logging and evaluation
        logging_steps=1,
        eval_steps=100,
        save_steps=200,
        save_total_limit=None,
        
        # Extra settings
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb",

        use_vllm=True,
        # vllm_device="cuda:0",

    )
    
    print("   GRPO Config:")
    print(f"     - Epochs: {grpo_config.num_train_epochs}")
    print(f"     - Batch size: {grpo_config.per_device_train_batch_size}")
    print(f"     - Group size: {grpo_config.num_generations}")
    print(f"     - Learning rate: {grpo_config.learning_rate}")
    print(f"     - KL coefficient: {grpo_config.beta}")
    print(f"     - Use vLLM: {grpo_config.use_vllm}")

    # 5. Initialize WandB
    print("\n[5/6] Initializing WandB...")
    run_name = f"grpo_trl_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="GRPO-TRL",
        name=run_name,
        config={
            "model": model_name,
            "group_size": grpo_config.num_generations,
            "batch_size": grpo_config.per_device_train_batch_size,
            "learning_rate": grpo_config.learning_rate,
            "beta": grpo_config.beta,
            "num_epochs": grpo_config.num_train_epochs
        }
    )
    
    # 6. Initialize GRPOTrainer and Start Training
    print("\n[6/6] Creating GRPOTrainer and Starting Training...")
    
    trainer = GRPOTrainer(
        model=model,
        # ref_model=ref_model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=compute_rouge_reward_batch  # Custom reward function
    )
    
    print("\n" + "="*60)
    print(" >> Starting GRPO Training...")
    print("="*60 + "\n")

    # Start training
    trainer.train()
    
    # 7. Save final model and tokenizer
    print("\n >> Saving final model...")
    trainer.save_model("./grpo_samsum_final")
    tokenizer.save_pretrained("./grpo_samsum_final")
    
    # 8. Final Evaluation
    print("\n >> Final Evaluation...")
    eval_results = trainer.evaluate()
    
    print("\n >> Final Evaluation Results:")
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    wandb.finish()
    print("\n >> Training Completed.")


if __name__ == "__main__":
    main()
