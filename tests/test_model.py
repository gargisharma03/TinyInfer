
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights


def test_full_model_matches_gpt2():

    # 1. Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. Create TinyInfer GPT-2 Small
    tiny_model = TinyInferGPT2(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        max_positions=1024
    )

    # 3. Load pretrained weights
    reference = load_gpt2_weights(tiny_model)

    tiny_model.eval()
    reference.eval()

    # 4. Prepare input
    text = "Artificial intelligence is changing the world"

    input_ids = tokenizer(
        text,
        return_tensors="pt"
    )["input_ids"]

    # 5. Compare model outputs
    with torch.no_grad():

        tiny_logits, _ = tiny_model(input_ids)

        reference_logits = reference(
            input_ids
        ).logits

    # 6. Measure numerical difference
    max_difference = (
        tiny_logits - reference_logits
    ).abs().max().item()

    mean_difference = (
        tiny_logits - reference_logits
    ).abs().mean().item()

    print("\nTinyInfer logits shape:", tiny_logits.shape)
    print("GPT-2 logits shape:", reference_logits.shape)
    print("Maximum difference:", max_difference)
    print("Mean difference:", mean_difference)

    assert tiny_logits.shape == reference_logits.shape

    assert torch.allclose(
        tiny_logits,
        reference_logits,
        atol=1e-4,
        rtol=1e-4
    )
