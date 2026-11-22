
import torch
import torch.nn as nn


class TinyInferLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

        self.bias = nn.Parameter(
            torch.zeros(hidden_size)
        )

        self.eps = eps

    def forward(self, x):

        # Calculate mean across hidden dimensions
        mean = x.mean(dim=-1, keepdim=True)

        # Calculate population variance
        variance = x.var(
            dim=-1,
            unbiased=False,
            keepdim=True
        )

        # Normalize
        normalized = (
            (x - mean) /
            torch.sqrt(variance + self.eps)
        )

        # Apply learnable scale and bias
        return self.weight * normalized + self.bias
