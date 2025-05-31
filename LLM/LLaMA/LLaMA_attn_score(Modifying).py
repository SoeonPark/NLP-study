from transformers.models.llama.modeling_llama import LlamaAttention
import torch as nn
import numpy as np
import logging
import argparse

from sklearn import metrics
import math

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from argparser import ArguParser


# A Class involves functions to get each keys and importance matrix for sentences in each Dialogue
def compute_attn_bias(tokens, key_token_indices, scale = 1.0):
    seq_len = len(tokens)
    bias_matrix = torch.zeros((1, 1, seq_len, seq_len))

    for i in range(seq_len):
        for j in key_token_indices:
            if 0<=j<seq_len:
                bias_matrix[0, 0, i, j] += scale
        
    return bias_matrix

def compute_metrics(): # it requires acc only!
    sklearn.metrics(acc)


# Custom Model for Updating Bias for Attention Scores
class CustomLlamaAttention(LlamaAttention):
    def forward(self, hidden_states, attention_mask = None, position_ids, past_key_value = None, output_attentions = False, **kwargs):

        # Current Attention Score
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_porj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_score = torch.matmaul(query_sates, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if "attention_bias" in kwargs:
            attn_scores = attn_scores + kwargs["attention_bias"]

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = torch.nn.functional.softmax(attn_score, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_states)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else attn_output, None

if __name__ == "__main__":
    llama_parser = argparse.ArgumentParser()

    inputs = tokenizer(prompt, return_tensors = "pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    key_token_indices = [i for i, t in enumerate(tokens) if "<key>" in t]

    bias_matrix = compute_attention_bias(tokens, key_token_indices, scale = 2.0)
    compute = model(**inputs, attention_bias = bias_matrix.to(model.device))

main()
