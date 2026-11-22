
import torch.nn as nn


class TinyInferMLP(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()

        intermediate_size = hidden_size * 4

        self.fc = nn.Linear(
            hidden_size,
            intermediate_size
        )

        # GPT-2 uses the approximate GELU activation
        self.activation = nn.GELU(approximate="tanh")

        self.proj = nn.Linear(
            intermediate_size,
            hidden_size
        )

    def forward(self, x):

        x = self.fc(x)
        x = self.activation(x)
        x = self.proj(x)

        return x
