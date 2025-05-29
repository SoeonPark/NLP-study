from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cahe, DynamicCache
from ...genertations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ALL_LAYERNORM_LAYERS
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__) # Create a logger for this module

@use_kernel_forward_from_hub("RMSNorm") # @use_kernel_forward_from_hub: A decorator that allows the function to use a kernel forward from the Hugging Face hub

class LlamaRMSNorm(nn.Module): # LLaMA uses RMSNorm instead of LayerNorm which is used in normal transformers
    def __init__(self, hidden_size, eps = 1e-6):
        """
          LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = n.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim = True) # pow(): square each element
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

ALL_LAYERNORM_LAYERS(LlamaRMSNorm)

class LlamaRotaryEmbedding(nn.Model):
    """
    LlamaRotaryEmbedding is used to apply Rotary Positional Embedding (ROPE) to the model.
    It initializes the ROPE based on the configuration and provides methods to apply it to the query and key tensors.
    """
    def __init__(self, config: LlamaConfig, device = None):
        super().__init__()
        # BC: "rope_type" was originally "type"

        # ROPE (Rotary Positional Embedding) is a technique to inject positional information into the model
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None: # rope_scaling: dictionary that contains parameters for scaling the ROPE
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type")) # get("type") for backward compatibility

        else:
            self.rope_type = "default" # Default type of ROPE, can be "default", "dynamic", or "static"
        self.max_seq_len_cached = config.max_position_embeddings # Maximum sequence length that can be cached
        self.original_max_seq_len = config.max_position_embeddings # Original maximum sequence length

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type] # Function to initialize the ROPE based on the type

        # inv_freq: Inverse(inv) frequencies for the ROPE, used to compute the sine and cosine values
        inv_freqs, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.original_inv_freq = self.inv_freq

def rotate_half(x):
    """ Rotates half the hidden dims of the input tensor x."""
    x1 = x[..., x.shape[-1] // 2] # Split the last dimension into two halves
    x2 = x[..., x.shape[-1] // 2 :] # Get the second half of the last dimension
    return torch.cat((-x2, x1), dim = -1) # Concatenate the negative of the second half with the first half, dim=-1 means the last dimension

def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    """
    - cos[position_ids].shape: [batch_size, seq_len, head_dim]
    - sin[position_ids].shape: [batch_size, seq_len, head_dim]
    Applies rotary positional embedding to the query and key tensors.
    """
    cos = cos.unsqeeze(unsqueeze_dim) 
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin) # Apply the cosine and sine to the query tensor
    k_embed = (k * cos) + (rotate_half(k) * sin) # Apply the cosine and sine to the key tensor
    # The reason why we use rotate_half is that the ROPE applies a rotation to the query and key tensors, which is equivalent to rotating half of the hidden dimensions.

    return q_embed, k_embed

class LlamaMLP(nn.Module):

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

def eager_attention_forward(

):

class LlamaAttention(nn.Module):
    """ Has Multi-headed Attention from "Attention is All You Need" paper."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        # Initialize the attention layer with the configuration and layer index.
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # Number of key-value groups in the attention layer
        # Since Llama uses a different number of key-value heads, we need to adjust the head_dim accordingly.
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        """
        - config: LlamaConfig object containing the configuration parameters for the model.

        - q_proj: Query projection layer
        - k_proj: Key projection layer
        - v_proj: Value projection layer
        - o_proj: Output projection layer

        - bias: Whether to use bias in the projection layers
        """
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = config.attention_bias)
    
    """
    def forward(): 
        This method implements the forward pass of the attention layer
        - hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)
        - position_embeddings: Tuple containing the cosine and sine embeddings for the rotary positional embedding
        - attention_mask: Optional tensor for masking attention scores
        - past_key_value: Optional cache for past key and value states
        - cache_position: Optional tensor for the position in the cache
        - **kwargs: Additional keyword arguments for flash attention
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_por_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation


class LlamaDecoderLayer(GradientCheckpointingLayer):

@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):

@auto_docstring
class LlamaModel(LlamaPreTrainedModel):

@auto_docstring
class LlamaForCausalLM(LlamaPretrainedModel, GenerationMixin):

@auto_docstring(
    custom_intro="""
    """
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):

@auto_docstring
class LlamaForQuestionAnswering(LlamaPreTrainedModel):

@auto_docstring
class LlamaForTokenClassification(LlamaPreTrainedModel):

__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
