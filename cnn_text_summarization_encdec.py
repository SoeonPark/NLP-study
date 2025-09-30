import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertConfig
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
import math
import numpy as np
import random
from collections import defaultdict

"""
    Goal: Implement Text Summarization with Encoder-Decoder Architecture from scratch
    
    Key Concepts:
        1. Encoder-Decoder: BERT encoder + Custom Transformer decoder
        2. Abstractive Summarization: Generate new text (not just extract)
        3. Attention Mechanism: Cross-attention between encoder and decoder
        4. Auto-regressive Generation: Generate tokens one by one
    
    Architecture:
        - BERT Encoder: Process input document
        - Custom Transformer Decoder: Generate summary tokens
        - Cross-Attention: Decoder attends to encoder outputs
        - Generation: Beam search or greedy decoding

    ===============================================================
    Basically, Decoder is an architecture with Multi TransformerDecoderLayer stacked.
        1. Causal Masked Self-Attention: Prevent attending to future tokens
        2. Cross-Attention: Attend to encoder outputs
            -> Uses encoder's last hidden states as key/value and
            determine which parts of the source to focus on
"""

class CustomLinear(torch.nn.Module):
    """Custom Linear Layer implementation"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features), requires_grad=True)
        else:
            self.bias = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class CustomMultiHeadAttention(torch.nn.Module):
    """
    Custom Multi-Head Attention implementation
    Supports both self-attention and cross-attention
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = CustomLinear(d_model, d_model)
        self.w_k = CustomLinear(d_model, d_model)
        self.w_v = CustomLinear(d_model, d_model)
        self.w_o = CustomLinear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None, is_cross_attention=False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] or [batch_size, 1, seq_len_k]
            is_cross_attention: Whether this is cross-attention (decoder->encoder)
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len_q, d_model]
        K = self.w_k(key)    # [batch_size, seq_len_k, d_model]  
        V = self.w_v(value)  # [batch_size, seq_len_v, d_model]
        
        # Reshape for multi-head: [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head
            if mask.dim() == 4:
                # It's already expanded
                mask_to_apply = mask
            elif mask.dim() == 3: # e.g., [batch_size, seq_len, seq_len]
                mask_to_apply = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            elif mask.dim() == 2: # e.g., [batch_size, seq_len]
                # 1. Causal mask for self-attention: [seq_len, seq_len]
                # 2. Padding mask for cross-attention: [batch_size, seq_len_k]
                if mask.size(0) == seq_len_q and mask.size(1) == seq_len_k:
                    # Causal Mask [seq_len_q, seq_len_k] -> need to expand to [batch_size, 1, seq_len_q, seq_len_k]
                    mask_to_apply = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, seq_len_q, seq_len_k)
                elif mask.size(0) == batch_size and mask.size(1) == seq_len_k:
                    # Padding Mask [batch_size, seq_len_k] -> need to expand to [batch_size, 1, 1, seq_len_k]
                    mask_to_apply = mask.unsqueeze(1).unsqueeze(2)

            scores = scores.masked_fill(mask_to_apply == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch_size, n_heads, seq_len_q, d_k]
        
        # Concatenate heads: [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attn_weights

class CustomFeedForward(torch.nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = CustomLinear(d_model, d_ff)
        self.linear2 = CustomLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerDecoderLayer(torch.nn.Module):
    """
    Custom Transformer Decoder Layer
    
    Components:
    1. Masked Self-Attention (decoder tokens attend to previous tokens)
    2. Cross-Attention (decoder tokens attend to encoder outputs)
    3. Feed-Forward Network
    4. Residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention (masked)
        self.self_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (to encoder)
        self.cross_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = CustomFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_outputs, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            x: decoder inputs [batch_size, tgt_len, d_model]
            encoder_outputs: encoder outputs [batch_size, src_len, d_model]
            self_attn_mask: causal mask for self-attention
            cross_attn_mask: mask for cross-attention (padding mask)
        """
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, mask=self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_outputs, encoder_outputs, mask=cross_attn_mask, is_cross_attention=True
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x, cross_attn_weights

class TransformerDecoder(torch.nn.Module):
    """
    Custom Transformer Decoder
    Stack of decoder layers with embedding and positional encoding
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, 
                 d_ff: int, max_length: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encodings
        self.pos_encoding = self._create_positional_encoding(max_length, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = CustomLinear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_length: int, d_model: int):
        """Create sinusoidal positional encodings"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_length, d_model]
    
    def _init_weights(self):
        """Initialize weights following transformer conventions"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create causal (triangular) mask for self-attention"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask  # [seq_len, seq_len]
    
    def forward(self, input_ids, encoder_outputs, encoder_attention_mask=None):
        """
        Args:
            input_ids: [batch_size, tgt_len] - decoder input tokens
            encoder_outputs: [batch_size, src_len, d_model] - encoder outputs
            encoder_attention_mask: [batch_size, src_len] - encoder padding mask
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings + positional encodings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encodings
        pos_encodings = self.pos_encoding[:, :seq_len, :].to(device)
        x = x + pos_encodings
        x = self.dropout(x)
        
        # Create causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Pass through decoder layers
        cross_attentions = []
        for layer in self.layers:
            x, cross_attn_weights = layer(
                x, 
                encoder_outputs, 
                self_attn_mask=causal_mask,
                cross_attn_mask=encoder_attention_mask
            )
            cross_attentions.append(cross_attn_weights)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits, cross_attentions

class BertForCustomSummarization(torch.nn.Module):
    """
    BERT-based Summarization Model
    
    Architecture:
        - BERT Encoder: Process input document
        - Custom Transformer Decoder: Generate summary
        - Cross-attention: Connect encoder and decoder
    """
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", 
                 decoder_layers: int = 6, decoder_heads: int = 8, 
                 decoder_d_ff: int = 2048, max_summary_length: int = 128):
        super().__init__()
        
        # BERT encoder
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.config = self.encoder.config
        
        # Get dimensions
        self.d_model = self.config.hidden_size  # 768 for bert-base
        self.vocab_size = self.config.vocab_size
        
        # Custom decoder
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            d_ff=decoder_d_ff,
            max_length=max_summary_length,
            dropout=self.config.hidden_dropout_prob
        )
        
        # Special tokens
        self.pad_token_id = None  # Will be set from tokenizer
        self.bos_token_id = None  # Beginning of sequence
        self.eos_token_id = None  # End of sequence
        
    def set_special_tokens(self, tokenizer):
        """Set special token IDs from tokenizer"""
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.cls_token_id  # Use [CLS] as BOS
        self.eos_token_id = tokenizer.sep_token_id  # Use [SEP] as EOS
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        """
        Forward pass for summarization
        
        Args:
            input_ids: [batch_size, src_len] - document tokens
            attention_mask: [batch_size, src_len] - document attention mask
            decoder_input_ids: [batch_size, tgt_len] - summary tokens (shifted right)
            decoder_attention_mask: [batch_size, tgt_len] - summary attention mask
            labels: [batch_size, tgt_len] - target summary tokens
        """
        # Encode document with BERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch_size, src_len, d_model]
        
        if decoder_input_ids is not None:
            # Training mode: use teacher forcing
            decoder_logits, cross_attentions = self.decoder(
                input_ids=decoder_input_ids,
                encoder_outputs=encoder_hidden_states,
                encoder_attention_mask=attention_mask
            )
            
            if labels is not None:
                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
                loss = loss_fct(decoder_logits.view(-1, self.vocab_size), labels.view(-1))
                return loss
            
            return decoder_logits
        
        # Inference mode: return encoder outputs for generation
        return encoder_hidden_states, attention_mask
    
    def generate(self, input_ids, attention_mask=None, max_length=64, num_beams=4, 
                 early_stopping=True, tokenizer=None):
        """
        Generate summary using beam search
        
        Args:
            input_ids: [batch_size, src_len] - document tokens
            attention_mask: [batch_size, src_len] - attention mask
            max_length: maximum summary length
            num_beams: beam search width
            early_stopping: stop when EOS token is generated
            tokenizer: tokenizer for special tokens
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode document
        encoder_hidden_states, encoder_attention_mask = self.forward(
            input_ids, attention_mask
        )
        
        # Initialize decoder input with BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.bos_token_id, 
            device=device, 
            dtype=torch.long
        )
        
        # Simple greedy generation (beam search would be more complex)
        generated_tokens = []
        
        for step in range(max_length):
            with torch.no_grad():
                # Get next token logits
                decoder_logits, _ = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_outputs=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
                
                # Get next token (greedy)
                next_token_logits = decoder_logits[:, -1, :]  # Last position
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to decoder input
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                generated_tokens.append(next_token)
                
                # Check for EOS token
                if early_stopping and (next_token == self.eos_token_id).all():
                    break
        
        # Remove BOS token from output
        generated_ids = decoder_input_ids[:, 1:]  # Skip BOS token
        
        return generated_ids

def prepare_summarization_data(examples, tokenizer, max_source_length=512, max_target_length=128):
    """Prepare data for summarization training"""
    
    # Tokenize source documents
    sources = examples["article"]
    targets = examples["highlights"]
    
    # Tokenize sources
    model_inputs = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    # Prepare decoder inputs (shift right with BOS token)
    decoder_input_ids = torch.zeros_like(labels["input_ids"])
    decoder_input_ids[:, 0] = tokenizer.cls_token_id  # BOS token
    decoder_input_ids[:, 1:] = labels["input_ids"][:, :-1]  # Shift right
    
    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def custom_collate_fn_summarization(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for summarization data"""
    articles = [item["article"] for item in batch]
    highlights = [item["highlights"] for item in batch]
    
    # Tokenize articles (source)
    source_encodings = tokenizer(
        articles,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Tokenize summaries (target)
    target_encodings = tokenizer(
        highlights,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Prepare decoder inputs (shifted right)
    decoder_input_ids = torch.zeros_like(target_encodings["input_ids"])
    decoder_input_ids[:, 0] = tokenizer.cls_token_id  # BOS
    decoder_input_ids[:, 1:] = target_encodings["input_ids"][:, :-1]
    
    return {
        "input_ids": source_encodings["input_ids"],
        "attention_mask": source_encodings["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": target_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }

def train_epoch_summarization(model, dataloader, optimizer, device):
    """Training function for summarization"""
    model.train()
    total_loss = 0.0
    
    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"]
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for seq2seq models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def evaluate_model_summarization(model, dataloader, device, tokenizer):
    """Evaluate summarization model"""
    model.eval()
    total_loss = 0.0
    
    # Load ROUGE metric for evaluation
    rouge_metric = evaluate.load("rouge")
    
    generated_summaries = []
    reference_summaries = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Compute loss
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"]
            )
            total_loss += loss.item()
            
            # Generate summaries for evaluation
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
                tokenizer=tokenizer
            )
            
            # Decode generated and reference summaries
            for i in range(generated_ids.shape[0]):
                # Generated summary
                gen_tokens = generated_ids[i]
                gen_summary = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated_summaries.append(gen_summary)
                
                # Reference summary
                ref_tokens = batch["labels"][i]
                ref_summary = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                reference_summaries.append(ref_summary)
    
    # Compute ROUGE scores
    rouge_scores = rouge_metric.compute(
        predictions=generated_summaries,
        references=reference_summaries,
        use_stemmer=True
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return rouge_scores, avg_loss, generated_summaries[:5], reference_summaries[:5]

# Load CNN/DailyMail dataset for summarization
print("Loading CNN/DailyMail dataset for text summarization...")
dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")

print("Dataset info:")
print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print(f"Test examples: {len(dataset['test'])}")

# Sample data inspection
sample = dataset["train"][0]
print(f"\nSample Article: {sample['article'][:300]}...")
print(f"Sample Summary: {sample['highlights']}")

# Initialize model and tokenizer
model = BertForCustomSummarization(
    decoder_layers=6,
    decoder_heads=8,
    max_summary_length=128
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Set special tokens
model.set_special_tokens(tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use subset for demonstration
train_data = dataset["train"].shuffle(seed=42).select(range(5000))
eval_data = dataset["validation"].shuffle(seed=42).select(range(500))

model.to(device)

# Print model information
print(f"\nModel Information:")
print(f"Total parameters: {sum(param.numel() for param in model.parameters()):,}")
print(f"Encoder (BERT) parameters: {sum(param.numel() for param in model.encoder.parameters()):,}")
print(f"Decoder parameters: {sum(param.numel() for param in model.decoder.parameters()):,}")
print(f"Trainable parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad):,}")

# Training setup
batch_size = 4  # Small batch size due to memory constraints
num_epochs = 3
learning_rate = 1e-4  # Lower learning rate for seq2seq

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    collate_fn=custom_collate_fn_summarization,
    shuffle=True
)
eval_loader = DataLoader(
    eval_data,
    batch_size=batch_size,
    collate_fn=custom_collate_fn_summarization
)

optimizer = AdamW(model.parameters(), lr=learning_rate)

print(f"\n >> Starting Text Summarization training...")
print(f" >> Dataset: CNN/DailyMail")
print(f" >> Task: Abstractive Text Summarization")
print(f" >> Architecture: BERT Encoder + Custom Transformer Decoder")
print(f" >> Batch size: {batch_size}")
print(f" >> Learning rate: {learning_rate}")
print(f" >> Epochs: {num_epochs}")
print("=" * 60)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Training
    train_loss = train_epoch_summarization(model, train_loader, optimizer, device)
    print(f"  Train Loss: {train_loss:.4f}")
    
    # Evaluation
    rouge_scores, eval_loss, gen_samples, ref_samples = evaluate_model_summarization(
        model, eval_loader, device, tokenizer
    )
    
    print(f" >> [RESULT] Eval Loss: {eval_loss:.4f}")
    print(f"    >> ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"    >> ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"    >> ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("-" * 60)

print("==== TEXT SUMMARIZATION TRAINING COMPLETED! ====")

# Example generations
print(" >> EXAMPLE GENERATED SUMMARIES:")

for i, (generated, reference) in enumerate(zip(gen_samples[:3], ref_samples[:3])):
    print(f"\n--- Example {i+1} ---")
    print(f"Generated: {generated}")
    print(f"Reference: {reference}")
    print("-" * 40)

def summarize_text(model, tokenizer, text, device, max_length=64):
    """Generate summary for a single text"""
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            tokenizer=tokenizer
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
example_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
"""

summary = summarize_text(model, tokenizer, example_text, device)
print(f"\nCustom Example:")
print(f"Original: {example_text}")
print(f"Generated Summary: {summary}")

"""
    SUMMARIZATION INSIGHTS:
    - Encoder-Decoder architecture enables abstractive summarization
    - Cross-attention allows decoder to focus on relevant source parts
    - ROUGE scores measure n-gram overlap with reference summaries
    - Generation requires careful handling of special tokens and decoding
    - This architecture is foundation for modern summarization models!!!!!
"""
