
import math

import torch
import torch.nn as nn


class TinyInferAttention(nn.Module):

    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Combined query, key and value projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

        # Projects concatenated attention heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    
    def forward(self, x, past_key_value=None):

        batch_size, seq_length, _ = x.shape

        # 1. Calculate query, key, and value
        qkv = self.qkv(x)
        query, key, value = qkv.split(self.hidden_size, dim=-1)

        # 2. Reshape into attention heads
        query = query.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        key = key.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        value = value.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 3. Append previously cached keys and values
        past_length = 0

        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.shape[-2]

            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        # Save the updated cache for the next generation step
        present_key_value = (key, value)

        # 4. Calculate attention scores
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 5. Apply causal mask, accounting for cached positions
        key_positions = torch.arange(
            past_length + seq_length, device=x.device
        )

        query_positions = past_length + torch.arange(
            seq_length, device=x.device
        )

        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        # 6. Calculate attention output
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, value)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.hidden_size)
        )

        output = self.out_proj(context)

        return output, present_key_value
