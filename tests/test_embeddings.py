
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinyinfer.embeddings import TinyInferEmbeddings


def test_embeddings_match_gpt2():

    # Load reference GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # Create TinyInfer embeddings
    tiny_embeddings = TinyInferEmbeddings(
        vocab_size=50257,
        hidden_size=768,
        max_positions=1024
    )

    # Copy pretrained weights
    with torch.no_grad():
        tiny_embeddings.token_embeddings.weight.copy_(
            model.transformer.wte.weight
        )

        tiny_embeddings.position_embeddings.weight.copy_(
            model.transformer.wpe.weight
        )

    # Prepare input
    text = "Artificial intelligence is"

    input_ids = tokenizer(
        text,
        return_tensors="pt"
    )["input_ids"]

    # Compare outputs
    with torch.no_grad():
        tiny_output = tiny_embeddings(input_ids)

        position_ids = torch.arange(
            input_ids.shape[1]
        ).unsqueeze(0)

        reference_output = (
            model.transformer.wte(input_ids)
            + model.transformer.wpe(position_ids)
        )

    # Verify shapes
    assert tiny_output.shape == reference_output.shape

    # Verify numerical correctness
    max_difference = (
        tiny_output - reference_output
    ).abs().max().item()

    print("\nMaximum difference:", max_difference)

    assert torch.allclose(
        tiny_output,
        reference_output,
        atol=1e-6,
        rtol=1e-5
    )
