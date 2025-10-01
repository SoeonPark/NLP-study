import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Model, T5ForConditionalGeneration
from datasets import load_dataset
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
import math
import numpy as np

"""
    Goal: Implement a Text Summarization model, with writing code from scratch(only with PyTorch and Huggingface Transformers library).
        => I can use T5, BART, etc. Enc-Dec model.
    Dataset: SAMSum dataset

    Key Concepts:
        1. Causal Masking; Block future tokens in the Decoder(This is the Key point of the Decoder Model)
        2. past_key_values: Speed up the inference time -- with KV caching
        3. Cross-Attention: Pass the encoder's hidden states to the decoder for cross-attention

    Key Difference between Encoder-Decoder and Decoder-only model:
        - Encoder(BERT): Bidirectional attention, processes the entire input sequence at once.
            * Attention Masking: Allows each token to attend to all other tokens in the input sequence.

        - Decoder(GPT2): Unidirectional attention, generates output tokens one at a time.
            * Causal Masking: Prevents each token from attending to future tokens in the sequence.
                Lower triangular matrix
                    [[1,0,0,0],
                    [1,1,0,0],
                    [1,1,1,0],
                    [1,1,1,1]]
            * Uses past_key_values to cache previous key and value states for faster generation.

        - Encoder-Decoder(T5, BART): Combines both bidirectional and unidirectional attention mechanisms.
            * Encoder: Processes the entire input sequence with bidirectional attention.
            * Decoder: Generates output tokens one at a time with unidirectional attention and cross-attention to the encoder's outputs.

    ============================================================

"""
class CustomLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

class CustomMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # Dimension per head

        # Linear projections for Query, Key, Value
        self.w_q = CustomLinear(d_model, d_model)
        self.w_k = CustomLinear(d_model, d_model)
        self.w_v = CustomLinear(d_model, d_model)
        self.w_o = CustomLinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)

        # Linear projections
        Q = self.w_q(query) # (batch_size, query_len, d_model)
        K = self.w_k(key)   # (batch_size, key_len, d_model
        V = self.w_v(value) # (batch_size, value_len, d_model)

        # Reshape for multi-head attention: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        # If past_key_value is provided, concatenate with current K, V for faster decoding
        if past_key_value is not None:
            past_key, past_value = past_key_value
            K = torch.cat([past_key, K], dim=2)  # Concatenate on sequence length
            V = torch.cat([past_value, V], dim=2)
            seq_len_k = K.size(2)  # Update key length

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Score shape: (batch_size, n_heads, query_len, key_len)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            if mask.dim() == 4:
                # It's already expanded
                mask_to_apply = mask
            # Cases below all need to be expanded to 4D
            elif mask.dim() == 3: # e.g., (batch_size, seq_len_q, seq_len_k)
                mask_to_apply = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 2: # e.g., (batch_size, seq_len_k)
                # 1. Causal Mask for Self-Attention in Decoder: (seq_len_q, seq_len_k) => need to expand to (1, 1, seq_len_q, seq_len_k)
                if mask.size(0) == seq_len_q and mask.size(1) == seq_len_k:
                    mask_to_apply = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, seq_len_q, seq_len_k)  # (1, 1, seq_len_q, seq_len_k)
                # 2. Padding Mask for Encoder-Decoder Attention: (batch_size, seq_len_k) => need to expand to (batch_size, 1, 1, seq_len_k)
                elif mask.size(0) == batch_size and mask.size(1) == seq_len_k:
                    mask_to_apply = mask.unsqueeze(1).unsqueeze(2).expand(batch_size, self.n_heads, 1, seq_len_k)  # (batch_size, 1, 1, seq_len_k)
                elif seq_len_q == 1:
                    # If Q_len=1, create an all-ones mask allowing all past key tokens.
                    mask_to_apply = torch.ones(batch_size, self.n_heads, 1, seq_len_k, device=scores.device)
                else: # Exception case
                    raise ValueError(f"Unhandled 2D mask size: {mask.size()} for Q_len={seq_len_q}, K_len={seq_len_k}, Batch={batch_size}")

            scores = scores.masked_fill(mask_to_apply == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply Attention to Values
        context = torch.matmul(attn_weights, V)  # (batch_size, n_heads, query_len, d_k)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)  # (batch_size, query_len, d_model)

        # Final linear layer
        output = self.w_o(context)  # (batch_size, query_len, d_model)

        present_key_value = (K, V)  # For caching in decoder during inference

        return output, present_key_value

class CustomFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = CustomLinear(d_model, d_ff)
        self.linear2 = CustomLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))  # (batch_size, seq_len, d_model) 
        # Linear1 -> GELU -> Dropout -> Linear2

class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer (Bidirectional)"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = CustomFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerDecoderLayer(torch.nn.Module):
    """
        Custom Transformer Decoder Layer Implementation

        Components:
            1. Masked Self-Attention (Decoder Tokens attend to previous tokens)
            2. Cross-Attention (Decoder Tokens attend to Encoder Outputs)
            3. Position-wise Feed-Forward Network
            4. Layer Normalization and Residual Connections
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Self-Attention (Masked)
        self.self_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-Attention (to Encoder Outputs)
        self.cross_attn = CustomMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ffn = CustomFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, target_mask: Optional[torch.Tensor] = None, 
                source_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Args:
                x: (batch_size, tgt_seq_len, d_model) - Decoder input embeddings
                enc_output: (batch_size, src_seq_len, d_model) - Encoder output embeddings
                target_mask: Key Causal mask for self-attention in the decoder
                source_mask: Padding Mask for cross-attention with encoder outputs
                past_key_values: Cached key and value tensors for faster decoding (inference) -- (self_kv, cross_kv) 
        """
        # Unpack past if exists
        past_self_kv = past_key_values if past_key_values is not None else None

        # 1. Causal Self-Attention with Residual
        self_attn_out, present_self_kv = self.self_attn(
            query = x, key = x, value = x,
            mask = target_mask,
            past_key_value = past_self_kv
        )
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross-Attention with Encoder Outputs and Residual (Without past_key_values)
        cross_attn_out, _ = self.cross_attn(
            query = x, key = enc_output, value = enc_output,
            mask = source_mask,
            past_key_value = None  # No caching for cross-attention
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3. Feed-Forward Network with Residual
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x, present_self_kv

class CustomT5ForSummarization(nn.Module):
    """
        Custom T5 Model for Text Summarization
            - Encoder-Decoder Architecture
            - Uses Transformer Encoder and Decoder Layers
            - Implements Causal Masking in Decoder
            - Supports past_key_values for faster inference

        Architecture:
            - Encoder: Bidirectional Transformer Encoder (as BERT)
            - Decoder: Unidirectional Transformer Decoder (as GPT-2)
    """
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6, 
                 d_ff: int = 2048, max_length: int = 512, dropout: float = 0.1,
                 pad_token_id: int = None, bos_token_id: int = None, eos_token_id: int = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared Token Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_embedding = self._create_positional_encoding(max_length, d_model)
        pos_embedding = self._create_positional_encoding(max_length, d_model)
        self.register_buffer("pos_embedding", pos_embedding)

        # Encoder (Bidirectional)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Decoder (Unidirectional with Causal Masking)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Output Projection
        self.output_projection = CustomLinear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Set Special Tokens
        self.pad_token_id = pad_token_id
        self.bos_token_id = pad_token_id  # T5 uses pad_token_id as BOS
        self.eos_token_id = eos_token_id

        self._init_weights()

    def _create_positional_encoding(self, max_length: int, d_model: int) -> nn.Parameter:
        """Create Positional Encoding Matrix"""
        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0)  # (1, max_length, d_model)
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask for self-attention in the decoder"""
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))  # (1, 1, seq_len, seq_len) >> NOT SURE
        return mask  # 1s in allowed positions, 0s in masked positions
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode the input sequence"""
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Embedding + Positional Encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Scale embeddings
        x = x + self.pos_embedding[:, :seq_len, :].to(device)
        x = self.dropout(x)

        # Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask=attention_mask)

        return x  # (batch_size, seq_len, d_model)
    
    def decode(self, decoder_input_ids: torch.Tensor, enc_output: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None,
               decoder_attention_mask: Optional[torch.Tensor] = None,
               past_key_values = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        """Decode the target sequence with access to encoder outputs"""
        batch_size, target_seq_len = decoder_input_ids.size()
        device = decoder_input_ids.device

        # Embedding + Positional Encoding
        x = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)

        # Position Offset for Decoder when using past_key_values
        position_offset = 0 
        if past_key_values is not None and past_key_values[0] is not None:
            # past_key_values[0] corresponds to the self-attention layer's past keys
            position_offset = past_key_values[0][0].size(1)  # Length of cached keys

        x = x + self.pos_embedding[:, position_offset:position_offset + target_seq_len, :].to(device)
        x = self.dropout(x)

        # Create Causal Mask for Decoder Self-Attention (With considering KV Caching)
        # if past_key_values is not None and past_key_values[0] is not None:
        #     # If uses past_key_values, only need to attend to the last token, which is the current token being processed
        #     total_seq_len = position_offset + target_seq_len
        #     causal_mask = self.create_causal_mask(total_seq_len, device)
        #     causal_mask = causal_mask[-target_seq_len:, :] # (target_seq_len, total_seq_len)
        # else:
        #     causal_mask = self.create_causal_mask(target_seq_len, device)

        total_kv_len = position_offset + target_seq_len
        
        causal_mask = self.create_causal_mask(total_kv_len, device)
        causal_mask = causal_mask[position_offset:total_kv_len, :total_kv_len]

        # Decoder Layers
        present_key_values = []
        for i, layer in enumerate(self.decoder_layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(
                x, 
                enc_output, 
                target_mask=causal_mask, 
                source_mask=attention_mask,
                past_key_values=past_kv
            )

            present_key_values.append(present_kv)

        return x, present_key_values
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
                decoder_input_ids: torch.Tensor, decoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor]]] = None) -> Dict[str, Any]:
        """Forward pass through the model"""
        # Encode
        enc_output = self.encode(input_ids, attention_mask)
        
        if decoder_input_ids is not None:
            # Decode
            dec_output, present_key_values = self.decode(
                decoder_input_ids, 
                enc_output, 
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values
            )
            
            # Get logits
            logits = self.output_projection(dec_output)

            # Compute loss if labels provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
                loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
                return loss, logits
            
            return logits
        
        return enc_output
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                 max_length: int = 64, num_beams: int = 4, early_stopping: bool = True) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Encode the input sequence
        enc_output = self.encode(input_ids, attention_mask)

        # Start with BOS token
        decoder_input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device) 

        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            with torch.no_grad():
                if past_key_values is None:
                    decoder_input = decoder_input_ids
                else:
                    decoder_input = decoder_input_ids[:, -1:]  # Only the last token for efficiency

                # Decode step
                dec_output, present_key_values = self.decode(
                    decoder_input_ids[:, -1:], # Only the last token
                    enc_output,
                    attention_mask,
                    past_key_values=past_key_values
                )

                # Get logits for the last token
                logits = self.output_projection(dec_output[:, -1, :])  # (batch_size, vocab_size)
                next_token = torch.argmax(logits, dim=-1, keepdim=True) # (batch_size, 1)

                # Append predicted token
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

                # Update Cached Key-Values
                past_key_values = present_key_values

                # Check for EOS token to stop early
                finished |= next_token.squeeze(-1) == self.eos_token_id
                if early_stopping and finished.all():
                    break

        return decoder_input_ids[:, 1:]  # Remove the initial BOS token from the output

def prepare_summarization_data(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_input_length: int = 512, max_target_length: int = 128) -> Dict[str, Any]:
    """
        Prepare data for text summarization task.
        Tokenize and encode the input documents and target summaries.
    """
    # Tokenize Source Documents and Target Summaries
    source = examples["article"]
    targets = examples["summary"]

    # Tokenize Source
    model_inputs = tokenizer(
        source, max_length = max_input_length, padding = True, truncation = True, return_tensors = "pt"
    )

    # Tokenize Targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length = max_target_length, padding = True, truncation = True, return_tensors = "pt"
        )

    # Prepare Decoder Inputs (Shift Right with BOS Token)
    decoder_input_ids = torch.zeros_like(labels["input_ids"])
    decoder_input_ids[:, 0] = tokenizer.cls_token_id  # BOS token
    decoder_input_ids[:, 1:] = labels["input_ids"][:, :-1] # Shift right

    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
        Custom Collate Function with Dynamic Padding
    """
    dialogue = [item["dialogue"] for item in batch]
    summary = [item["summary"] for item in batch]

    # Tokenize Source Documents (Source)
    source_encodings = tokenizer(
        dialogue, max_length = 128, padding = True, truncation = True, return_tensors = "pt"
    )
    # Tokenize Target Summaries (Target)
    with tokenizer.as_target_tokenizer(): 
        target_encodings = tokenizer(
            summary, max_length = 64, padding = True, truncation = True, return_tensors = "pt"
        )   

    # Prepare Decoder Inputs (Shifted Right with BOS Token)
    labels = target_encodings["input_ids"]
    decoder_input_ids = torch.full_like(labels, tokenizer.pad_token_id)
    decoder_input_ids[:, 0] = tokenizer.pad_token_id  # BOS token
    # decoder_input_ids[:, 1:] = target_encodings["input_ids"][:, :-1] # Shift right
    if labels.size(1) > 1:
        decoder_input_ids[:, 1:] = labels[:, :-1]  # Shift right

    # T5의 BOS는 <pad> 토큰 ID (ID 0)입니다.
    # decoder_input_ids[:, 0]는 이미 pad_token_id로 채워져 있으므로 명시적인 할당이 필요 없습니다.


    return {
        "input_ids": source_encodings["input_ids"],
        "attention_mask": source_encodings["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": target_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


#####################################################
 # Train and Evaluation Functions
#####################################################
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, tokenizer: AutoTokenizer) -> float:
    """Train Function for One Epoch"""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        if epoch == 0 and step == 0:
            print(f"\n >> Sample Batch Shapes:")
            for key, value in batch.items():
                print(f"    >> {key}: {value.shape}")
            print("="*80)

            decoded_input = tokenizer.decode(batch["decoder_input_ids"][0], skip_special_tokens = False)
            input_mask = batch["decoder_attention_mask"][0].tolist()
            print(f"    >> Sample Decoder Input IDs: {batch['decoder_input_ids'][0].tolist()}")
            print(f"    >> Sample Decoded Source (Input) Text: {decoded_input}")
            print(f"    >> Sample Decoder Attention Mask: {input_mask[:20]}... (Total Length: {len(input_mask)})")

            decoded_label = tokenizer.decode(batch["labels"][0].tolist(), skip_special_tokens = False)
            dec_mask = batch["decoder_attention_mask"][0].tolist()
            print(f"    >> Sample Decoded Decoder Input (Shifted Right) Text: {decoded_label}")
            print(f"    >> Sample Decoder Attention Mask: {dec_mask[:20]}... (Total Length: {len(dec_mask)})")
            print("="*80)

        # Forward Pass
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"]
        )
        loss = output[0]  # Get loss

        # Backward
        loss.backward()

        # Gradient Clipping(Optional but Important for seq2seq models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Evaluation Function using ROUGE Metric"""
    model.eval()
    total_loss = 0.0

    rouge = evaluate.load("rouge")

    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute Loss
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"]
            )
            loss = output[0]  # Get loss
            total_loss += loss.item()

            # Generate Summaries for Evaluation
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
                early_stopping=True
            )

            # Decode Generated and Reference Summaries
            for i in range(generated_ids.size(0)):
                # Generated Summary
                gen_tokens = generated_ids[i]
                gen_summary = tokenizer.decode(gen_tokens, skip_special_tokens = True)
                generated_summaries.append(gen_summary)

                # Reference Summary
                ref_tokens = batch["labels"][i]
                ref_summary = tokenizer.decode(ref_tokens, skip_special_tokens = True)
                reference_summaries.append(ref_summary)

    # Compute ROUGE Scores
    rouge_scores = rouge.compute(predictions = generated_summaries, references = reference_summaries, use_stemmer = True)

    avg_loss = total_loss / len(dataloader)

    return rouge_scores, avg_loss, generated_summaries, reference_summaries

#####################################################
 # Main Execution for Text Summarization
#####################################################

# Load SamSum dataset
print("Loading SamSum dataset for Text Summarization...")
dataset = load_dataset("knkarthick/samsum")

print("Dataset Info: ")
print(f"Train Examples: {len(dataset['train'])}")
print(f"Validation Examples: {len(dataset['validation'])}")
print(f"Test Examples: {len(dataset['test'])}")

# Sample data inspection
sample = dataset["train"][0]
print(f"\nSample Article: {sample['dialogue']}...")
print(f"Sample Summary: {sample['summary']}")

# Initialize Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = CustomT5ForSummarization(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=2048,
    max_length=512,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

model.pad_token_id = tokenizer.pad_token_id
model.bos_token_id = tokenizer.pad_token_id  # T5 uses pad_token_id as BOS
model.eos_token_id = tokenizer.eos_token_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# print(f"\n >> Model Architecture:")
# print(f"    >> Device: {device}")
# print(f"    >> Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# print(f"    >> Encoder parameters: {sum(p.numel() for p in model.encoder_layers.parameters()):,}")
# print(f"    >> Decoder parameters: {sum(p.numel() for p in model.decoder_layers.parameters()):,}")

print(f"\n >> Model Architecture:")
print(f"Device: {device}")
print("="*80)

# 1. Shared Components
embedding_params = sum(p.numel() for p in model.embedding.parameters())
output_proj_params = sum(p.numel() for p in model.output_projection.parameters())

# 2. Encoder-only
encoder_params = sum(p.numel() for p in model.encoder_layers.parameters())

# 3. Decoder-only
decoder_params = sum(p.numel() for p in model.decoder_layers.parameters())

# 4. Total
total_params = sum(p.numel() for p in model.parameters())

print(f" >> Parameter Breakdown:")
print(f"     ├─ Shared Embedding:        {embedding_params:>12,} ({embedding_params/total_params*100:>5.2f}%)")
print(f"     ├─ Output Projection:       {output_proj_params:>12,} ({output_proj_params/total_params*100:>5.2f}%)")
print(f"     ├─ Encoder Layers:          {encoder_params:>12,} ({encoder_params/total_params*100:>5.2f}%)")
print(f"     ├─ Decoder Layers:          {decoder_params:>12,} ({decoder_params/total_params*100:>5.2f}%)")
print(f"     └─ Total:                   {total_params:>12,}")

print(f"\n >> Comparison with Standard Models:")
print(f"     ├─ Your Custom Model:       {total_params:>12,}")
print(f"     ├─ T5-Small (official):     {60_506_624:>12,}  (60M)")
print(f"     ├─ T5-Base:                 {222_903_552:>12,}  (223M)")
print(f"     └─ BART-Base:               {139_420_416:>12,}  (139M)")
print("="*80)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    >> Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

train_data = dataset['train'].shuffle(seed=42).select(range(2000))
eval_data = dataset['validation'].shuffle(seed=42).select(range(200))

train_loader = DataLoader(train_data, batch_size=8, collate_fn=custom_collate_fn, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=8, collate_fn=custom_collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-4)

print("\n >> Starting Training...")
print(f"    >> Batch size: 8")
print(f"    >> Learning rate: 5e-4")
print(f"    >> Epochs: 2")
print("="*80)

# Training Loop
num_epochs = 2
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 80)
    
    # Training
    train_loss = train_epoch(model, train_loader, optimizer, device, epoch, tokenizer)
    print(f" >> Epoch {epoch+1} - Avg Train Loss: {train_loss:.4f}")
    
    # Evaluation
    rouge_scores, eval_loss, gen_summaries, ref_summaries = evaluate_model(
        model, eval_loader, device, tokenizer
    )
    
    print(f"\n >> Evaluation Results:")
    print(f"    >> Eval Loss: {eval_loss:.4f}")
    print(f"    >> ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"    >> ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"    >> ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("="*80)

print("\n >> Training Completed! <<")

print("\n >> Sample Generated Summaries:")
for i in range(min(3, len(gen_summaries))):
    print(f"\n--- Example {i+1} ---")
    print(f"Generated: {gen_summaries[i]}")
    print(f"Reference: {ref_summaries[i]}")
    print("-"*60)
