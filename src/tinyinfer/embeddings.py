import torch
import torch.nn as nn


class TinyInferEmbeddings(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_positions):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            hidden_size
        )

        self.position_embeddings = nn.Embedding(
            max_positions,
            hidden_size
        )

    def forward(self, input_ids, past_length=0):

        batch_size, seq_length = input_ids.shape

        position_ids = torch.arange(
            past_length,
            past_length + seq_length,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0)

        token_embeds = self.token_embeddings(input_ids)

        position_embeds = self.position_embeddings(position_ids)

        return token_embeds + position_embeds