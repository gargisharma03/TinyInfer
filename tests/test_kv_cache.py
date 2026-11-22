
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights


def test_kv_cache_matches_full_forward():

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = TinyInferGPT2()
    load_gpt2_weights(model)
    model.eval()

    input_ids = tokenizer(
        "Artificial intelligence is changing the world",
        return_tensors="pt"
    )["input_ids"]

    # Method 1: Process the entire prompt at once
    with torch.no_grad():
        full_logits, _ = model(input_ids)

        # Method 2: Process all tokens except the last one
        _, past_key_values = model(input_ids[:, :-1])

        # Process the final token using the saved cache
        cached_logits, _ = model(
            input_ids[:, -1:],
            past_key_values=past_key_values
        )

    # Compare the final token's logits
    full_last_logits = full_logits[:, -1, :]
    cached_last_logits = cached_logits[:, -1, :]

    max_difference = (
        full_last_logits - cached_last_logits
    ).abs().max().item()

    print(f"\nMaximum logit difference: {max_difference}")

    torch.testing.assert_close(
        cached_last_logits,
        full_last_logits,
        atol=1e-4,
        rtol=1e-4
    )
