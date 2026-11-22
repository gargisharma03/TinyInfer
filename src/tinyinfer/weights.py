
import torch
from transformers import GPT2LMHeadModel


def load_gpt2_weights(tiny_model, model_name="gpt2"):

    print(f"Loading pretrained weights from {model_name}...")

    reference = GPT2LMHeadModel.from_pretrained(model_name)
    reference.eval()

    source = reference.transformer

    with torch.no_grad():

        # 1. Token and positional embeddings
        tiny_model.embeddings.token_embeddings.weight.copy_(
            source.wte.weight
        )

        tiny_model.embeddings.position_embeddings.weight.copy_(
            source.wpe.weight
        )

        # 2. Transformer blocks
        for tiny_block, reference_block in zip(
            tiny_model.blocks,
            source.h
        ):

            # LayerNorm 1
            tiny_block.ln_1.weight.copy_(
                reference_block.ln_1.weight
            )
            tiny_block.ln_1.bias.copy_(
                reference_block.ln_1.bias
            )

            # Attention
            tiny_block.attention.qkv.weight.copy_(
                reference_block.attn.c_attn.weight.T
            )
            tiny_block.attention.qkv.bias.copy_(
                reference_block.attn.c_attn.bias
            )

            tiny_block.attention.out_proj.weight.copy_(
                reference_block.attn.c_proj.weight.T
            )
            tiny_block.attention.out_proj.bias.copy_(
                reference_block.attn.c_proj.bias
            )

            # LayerNorm 2
            tiny_block.ln_2.weight.copy_(
                reference_block.ln_2.weight
            )
            tiny_block.ln_2.bias.copy_(
                reference_block.ln_2.bias
            )

            # MLP
            tiny_block.mlp.fc.weight.copy_(
                reference_block.mlp.c_fc.weight.T
            )
            tiny_block.mlp.fc.bias.copy_(
                reference_block.mlp.c_fc.bias
            )

            tiny_block.mlp.proj.weight.copy_(
                reference_block.mlp.c_proj.weight.T
            )
            tiny_block.mlp.proj.bias.copy_(
                reference_block.mlp.c_proj.bias
            )

        # 3. Final LayerNorm
        tiny_model.final_norm.weight.copy_(
            source.ln_f.weight
        )
        tiny_model.final_norm.bias.copy_(
            source.ln_f.bias
        )

        # 4. Tie language-model head to token embeddings
        tiny_model.lm_head.weight = (
            tiny_model.embeddings.token_embeddings.weight
        )

    tiny_model.eval()

    print("GPT-2 weights loaded successfully!")

    return reference
