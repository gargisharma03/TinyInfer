
import torch.nn as nn

from tinyinfer.normalization import TinyInferLayerNorm
from tinyinfer.attention import TinyInferAttention
from tinyinfer.mlp import TinyInferMLP


class TinyInferTransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        layer_norm_eps=1e-5,
        dropout=0.1
    ):
        super().__init__()

        self.ln_1 = TinyInferLayerNorm(
            hidden_size,
            eps=layer_norm_eps
        )

        self.attention = TinyInferAttention(
            hidden_size,
            num_heads
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.ln_2 = TinyInferLayerNorm(
            hidden_size,
            eps=layer_norm_eps
        )

        self.mlp = TinyInferMLP(hidden_size)

        self.mlp_dropout = nn.Dropout(dropout)

    
    def forward(self, x, past_key_value=None):

        # Attention + residual connection
        attention_output, present_key_value = self.attention(
            self.ln_1(x),
            past_key_value=past_key_value
        )

        x = x + self.attn_dropout(attention_output)

        # MLP + residual connection
        mlp_output = self.mlp(self.ln_2(x))

        x = x + self.mlp_dropout(mlp_output)

        return x, present_key_value

