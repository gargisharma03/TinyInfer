
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinyinfer.embeddings import TinyInferEmbeddings
from tinyinfer.attention import TinyInferAttention


def test_attention_matches_gpt2():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # Get the first GPT-2 transformer block's attention
    reference = model.transformer.h[0].attn

    tiny_attention = TinyInferAttention(
        hidden_size=768,
        num_heads=12
    )

    # Copy Hugging Face weights into TinyInfer
    with torch.no_grad():
        tiny_attention.qkv.weight.copy_(
            reference.c_attn.weight.T
        )
        tiny_attention.qkv.bias.copy_(
            reference.c_attn.bias
        )

        tiny_attention.out_proj.weight.copy_(
            reference.c_proj.weight.T
        )
        tiny_attention.out_proj.bias.copy_(
            reference.c_proj.bias
        )

    tiny_attention.eval()

    # Generate realistic input from GPT-2 embeddings
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

        hidden_states = model.transformer.h[0].ln_1(embeddings)

        tiny_output, _ = tiny_attention(hidden_states)

        reference_result = reference(
            hidden_states=hidden_states
        )

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

    assert torch.allclose(
        tiny_output,
        reference_output,
        atol=1e-5,
        rtol=1e-4
    )
