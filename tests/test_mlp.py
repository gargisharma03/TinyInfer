
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tinyinfer.mlp import TinyInferMLP


def test_mlp_matches_gpt2():

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    reference = model.transformer.h[0].mlp

    tiny_mlp = TinyInferMLP(hidden_size=768)

    # Copy pretrained GPT-2 weights
    with torch.no_grad():

        tiny_mlp.fc.weight.copy_(
            reference.c_fc.weight.T
        )
        tiny_mlp.fc.bias.copy_(
            reference.c_fc.bias
        )

        tiny_mlp.proj.weight.copy_(
            reference.c_proj.weight.T
        )
        tiny_mlp.proj.bias.copy_(
            reference.c_proj.bias
        )

    tiny_mlp.eval()

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

        hidden_states = model.transformer.h[0].ln_2(embeddings)

        tiny_output = tiny_mlp(hidden_states)
        reference_output = reference(hidden_states)

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
