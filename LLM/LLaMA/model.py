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

logger = logging.get_logger(__name__)

@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):

class LlamaRotaryEmbedding(nn.Model):

def rotate_half(x):

def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):

class LlamaMLP(nn.Module):

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

def eager_attention_forward(

):

class LlamaAttention(nn.Module):


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
