
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinyinfer.embeddings import TinyInferEmbeddings
from tinyinfer.transformer import TinyInferTransformerBlock


def test_transformer_block_matches_gpt2():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    reference = model.transformer.h[0]

    tiny_block = TinyInferTransformerBlock(
        hidden_size=768,
        num_heads=12,
        layer_norm_eps=1e-5,
        dropout=0.1
    )

    # Copy LayerNorm weights
    with torch.no_grad():

        tiny_block.ln_1.weight.copy_(reference.ln_1.weight)
        tiny_block.ln_1.bias.copy_(reference.ln_1.bias)

        tiny_block.ln_2.weight.copy_(reference.ln_2.weight)
        tiny_block.ln_2.bias.copy_(reference.ln_2.bias)

        # Copy attention weights
        tiny_block.attention.qkv.weight.copy_(
            reference.attn.c_attn.weight.T
        )
        tiny_block.attention.qkv.bias.copy_(
            reference.attn.c_attn.bias
        )

        tiny_block.attention.out_proj.weight.copy_(
            reference.attn.c_proj.weight.T
        )
        tiny_block.attention.out_proj.bias.copy_(
            reference.attn.c_proj.bias
        )

        # Copy MLP weights
        tiny_block.mlp.fc.weight.copy_(
            reference.mlp.c_fc.weight.T
        )
        tiny_block.mlp.fc.bias.copy_(
            reference.mlp.c_fc.bias
        )

        tiny_block.mlp.proj.weight.copy_(
            reference.mlp.c_proj.weight.T
        )
        tiny_block.mlp.proj.bias.copy_(
            reference.mlp.c_proj.bias
        )

    tiny_block.eval()

    input_ids = tokenizer(
        "Artificial intelligence is",
        return_tensors="pt"
    )["input_ids"]

    with torch.no_grad():

        embeddings = (
            model.transformer.wte(input_ids)
            + model.transformer.wpe(
                torch.arange(input_ids.shape[1]).unsqueeze(0)
            )
        )

        tiny_output, _ = tiny_block(embeddings)
        reference_result = reference(embeddings)

        reference_output = (
            reference_result[0]
            if isinstance(reference_result, tuple)
            else reference_result
        )

    max_difference = (
        tiny_output - reference_output
    ).abs().max().item()

    print("\nTinyInfer output:", tiny_output.shape)
    print("GPT-2 output:", reference_output.shape)
    print("Maximum difference:", max_difference)

    assert tiny_output.shape == reference_output.shape

    assert torch.allclose(
        tiny_output,
        reference_output,
        atol=1e-5,
        rtol=1e-4
    )
