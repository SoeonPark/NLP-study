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
    """
    LlamaMLP is a Multi-Layer Perceptron (MLP) module used in the LLaMA model
    
    Attributes:
    - config (LlamaConfig): Configuration object containing the model parameters.
    - hidden_size (int): Size of the hidden layer.
    - intermediate_size (int): Size of the intermediate layer.
    - gate_proj (nn.Linear): Linear layer for the gate projection.
    - up_proj (nn.Linear): Linear layer for the up projection.
    - down_proj (nn.Linear): Linear layer for the down projection.
    - act_fn (Callable): Activation function to be applied in the MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = config.mlp_bias)
        self.act_fn =ACT2FN[config.hidden_act]

        def forward(self, x):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))
            return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Torch],
    scaling: flaot,
    dropout: float = 0.0,
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups) # Repeat the key states for each key-value group
    value_states = repeat_kv(value, module.num_key_value_groups) # Repeat the value states for each key-value group

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        
    attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query.dtype) # Apply softmax to the attention weights
    attn_weights = nn.functional.dropout(attn_weights, p = dropout, training = module.training) # Apply dropout to the attention weights
    attn_output = torch.matmul(attn_wights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous() #.view(query.shape[0], -1, module.config.hidden_size) # Reshape the attention output to (batch_size, seq_length, hidden_size)

    return attn_output, attn_weights

class LlamaAttention(nn.Module):
    """ 
    Multi-Headed Attention Module with support for Multi-Query Attention (MQA)
    Rotary Positional Embeddings (RoPE), and Multiple Attention backends(eager, flash, etc.)
    """

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
        # Projection Layers 
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = config.attention_bias)
    
    """
    def forward():
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings (Tuple[Tensor, Tensor]): Cosine and sine embeddings for RoPE
            attention_mask (Optional[Tensor]): Masking tensor for attention
            past_key_value (Optional[Cache]): Cached key/value tensors for KV cache
            cache_position (Optional[LongTensor]): Position in cache
            **kwargs: Additional arguments (e.g., FlashAttention-specific)
        Returns:
            Tuple:
                attn_output (Tensor): Output tensor of shape (batch, seq_len, hidden_size)
                attn_weights (Optional[Tensor]): Attention weights (if requested)
                past_key_value (Optional[Tuple[Tensor]]): Updated cache for inference
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

        # Linear Projections and Reshape([Bacth_Size, Seq_Length, Num_Heads])
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE(Rotary Positional Embedding)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_por_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward # Callable implementing the Actual Attention(e.g., eager, flash, etc.)
        if self.config._attn_implementation != "eager": # eager_attention_forward is the default attention implementation
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout = 0.0 if not self.training else self.attention_dropout,
            scaling = self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous() # Flatten Multi-Headed Outputs back to (batch_size, seq_length, hidden_size)
        attn_output = self.o_proj(attn_output) # Final Linear Projection back to hidden size
        return attn_output, attn_weights
        

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = LlamaAttention(config = config, layer_idx = layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pask_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    
            
        # Self Attention
        hidden_states, self_attn_weight = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            positio_ids = position_ids,
            past_key_value = past_key_value,
            output_attentions = output_attentions,
            use_cache = use_cache,
            cache_position = cache_position,
            position_embeddings = position_embeddings,
            **kwargs
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# @auto_docstring
# class LlamaPreTrainedModel(PreTrainedModel):
#     config_class = LlamaConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["LlamaDecoderLayer"]
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_flex_attn = True
#     _supports_cache_class = True
#     _supports_quantized_cache = True
#     _supports_static_cache = True
#     _supports_attention_backend = True

#     def _init_weights(self, module):
#         std = self.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, LlamaRMSNorm):
#             module.weight.data.fill_(1.0)


# @auto_docstring
# class LlamaModel(LlamaPreTrainedModel):
#     def __init__(self, config: LlamaConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = LlamaRotaryEmbedding(config=config)
#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
#     ) -> BaseModelOutputWithPast:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
#             )
#             use_cache = False

#         # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
#         if not isinstance(past_key_values, (type(None), Cache)):
#             raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if use_cache and past_key_values is None:
#             past_key_values = DynamicCache()

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )

#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         causal_mask = create_causal_mask(
#             config=self.config,
#             input_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             cache_position=cache_position,
#             past_key_values=past_key_values,
#         )

#         hidden_states = inputs_embeds

#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None

#         for decoder_layer in self.layers[: self.config.num_hidden_layers]:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=causal_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_values,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#                 **flash_attn_kwargs,
#             )

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=past_key_values if use_cache else None,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )


# class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# @auto_docstring
# class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
#     _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.model = decoder

#     def get_decoder(self):
#         return self.model

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[KwargsForCausalLM],
#     ) -> CausalLMOutputWithPast:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs: BaseModelOutputWithPast = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             cache_position=cache_position,
#             **kwargs,
#         )

#         hidden_states = outputs.last_hidden_state
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @auto_docstring(
#     custom_intro="""
#     The LLaMa Model transformer with a sequence classification head on top (linear layer).

#     [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-2) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """
# )
# class LlamaForSequenceClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = LlamaModel(config)
#         self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> SequenceClassifierOutputWithPast:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """

#         transformer_outputs: BaseModelOutputWithPast = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         hidden_states = transformer_outputs.last_hidden_state
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             last_non_pad_token = -1
#         elif input_ids is not None:
#             # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
#             non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
#             token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
#             last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
#         else:
#             last_non_pad_token = -1
#             logger.warning_once(
#                 f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                 "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#             )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @auto_docstring
# class LlamaForQuestionAnswering(LlamaPreTrainedModel):
#     base_model_prefix = "transformer"

#     # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = LlamaModel(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.transformer.embed_tokens

#     def set_input_embeddings(self, value):
#         self.transformer.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         **kwargs,
#     ) -> QuestionAnsweringModelOutput:
#         outputs: BaseModelOutputWithPast = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )

#         sequence_output = outputs.last_hidden_state

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         loss = None
#         if start_positions is not None and end_positions is not None:
#             loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

#         return QuestionAnsweringModelOutput(
#             loss=loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @auto_docstring
# class LlamaForTokenClassification(LlamaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = LlamaModel(config)
#         if getattr(config, "classifier_dropout", None) is not None:
#             classifier_dropout = config.classifier_dropout
#         elif getattr(config, "hidden_dropout", None) is not None:
#             classifier_dropout = config.hidden_dropout
#         else:
#             classifier_dropout = 0.1
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.score = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> TokenClassifierOutput:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """

#         outputs: BaseModelOutputWithPast = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.score(sequence_output)

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits, labels, self.config)

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
