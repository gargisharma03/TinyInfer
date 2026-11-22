
import torch
import torch.nn as nn

from tinyinfer.embeddings import TinyInferEmbeddings
from tinyinfer.normalization import TinyInferLayerNorm
from tinyinfer.transformer import TinyInferTransformerBlock


class TinyInferGPT2(nn.Module):

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        max_positions=1024,
        layer_norm_eps=1e-5,
        dropout=0.1
    ):
        super().__init__()

        self.embeddings = TinyInferEmbeddings(
            vocab_size,
            hidden_size,
            max_positions
        )

        self.blocks = nn.ModuleList([
            TinyInferTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = TinyInferLayerNorm(
            hidden_size,
            eps=layer_norm_eps
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

    
    
    def forward(self, input_ids, past_key_values=None):     

        past_length = (
            past_key_values[0][0].shape[-2]
            if past_key_values is not None
            else 0
        )

        x = self.embeddings(input_ids, past_length=past_length)

        present_key_values = []

        for i, block in enumerate(self.blocks):

            past_key_value = (
                past_key_values[i]
                if past_key_values is not None
                else None
            )

            x, present_key_value = block(
                x,
                past_key_value=past_key_value
            )

            present_key_values.append(present_key_value)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, tuple(present_key_values)