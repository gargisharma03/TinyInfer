
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinyinfer.embeddings import TinyInferEmbeddings
from tinyinfer.normalization import TinyInferLayerNorm


def test_layernorm_matches_gpt2():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model.eval()

    # Create TinyInfer embeddings
    embeddings = TinyInferEmbeddings(
        vocab_size=50257,
        hidden_size=768,
        max_positions=1024
    )

    # Copy pretrained embedding weights
    with torch.no_grad():
        embeddings.token_embeddings.weight.copy_(
            model.transformer.wte.weight
        )

        embeddings.position_embeddings.weight.copy_(
            model.transformer.wpe.weight
        )

    # Create TinyInfer LayerNorm
    tiny_norm = TinyInferLayerNorm(
        hidden_size=768,
        eps=1e-5
    )

    # Copy GPT-2 LayerNorm weights
    reference_norm = model.transformer.h[0].ln_1

    with torch.no_grad():
        tiny_norm.weight.copy_(reference_norm.weight)
        tiny_norm.bias.copy_(reference_norm.bias)

    # Prepare input
    input_ids = tokenizer(
        "Artificial intelligence is",
        return_tensors="pt"
    )["input_ids"]

    # Compare outputs
    with torch.no_grad():

        x = embeddings(input_ids)

        tiny_output = tiny_norm(x)

        reference_output = reference_norm(x)

    max_difference = (
        tiny_output - reference_output
    ).abs().max().item()

    print("\nTinyInfer shape:", tiny_output.shape)
    print("Reference shape:", reference_output.shape)
    print("Maximum difference:", max_difference)

    assert torch.allclose(
        tiny_output,
        reference_output,
        atol=1e-6,
        rtol=1e-5
    )
