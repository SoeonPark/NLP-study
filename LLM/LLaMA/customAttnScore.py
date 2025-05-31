import torch
import math
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------
# Attention Bias 생성 함수
# ------------------------
def compute_attn_bias(tokens, key_token_indices, scale=2.0):
    sequery_len = len(tokens)
    bias_matrix = torch.zeros((1, 1, sequery_len, sequery_len))  # [1, 1, sequery_len, sequery_len]

    for i in range(sequery_len):
        for j in key_token_indices:
            if 0 <= j < sequery_len:
                bias_matrix[0, 0, i, j] += scale

    return bias_matrix

# ------------------------
# Custom Attention Layer
# ------------------------
class CustomLlamaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        batch_size, query_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if "attention_bias" in kwargs:
            attn_scores = attn_scores + kwargs["attention_bias"]  # shape: [batch_size, num_heads, query_len, query_len]

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output, None

if __name__ == "__main__":
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    prompt = """[INST]
    문장1: <key>효겸</key>은 <key>화를 냈다</key>.
    문장3: <key>대책</key>을 마련해야 했다.

    다음 중 자연스럽게 이어지는 문장을 고르시오:
    ① 사람들은 그런 효겸의 행동을 싫어했다.
    ② 사람들은 그럼 효겸의 행동을 호감 있어 했다.

    정답:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    key_token_indices = [i for i, tok in enumerate(tokens) if "<key>" in tok]

    # 바이어스 생성
    bias_matrix = compute_attn_bias(tokens, key_token_indices, scale=2.0).to(model.device)

    # 추론
    with torch.no_grad():
        output = model(**inputs, attention_bias=bias_matrix)

    print("Bias 추가 후 추론 완료")
