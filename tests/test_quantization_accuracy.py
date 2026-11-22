
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_model_dynamic
import torch.nn.functional as F


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Loading FP32 model...")
    fp32_model = TinyInferGPT2()
    load_gpt2_weights(fp32_model)
    fp32_model.eval()

    print("\nCreating INT8 model...")
    int8_model = quantize_model_dynamic(fp32_model)
    quantized_count = sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in int8_model.modules()
    )

    print("\nQUANTIZATION STRUCTURE")
    print("======================")
    print("Quantized Linear layers:", quantized_count)
    print("Output head type:", type(int8_model.lm_head).__name__)

    tied_weights = (
        int8_model.lm_head.weight.data_ptr()
        == int8_model.embeddings.token_embeddings.weight.data_ptr()
    )

    print("Embedding/output weights tied:", tied_weights)

    prompt = "Artificial intelligence is changing the world"

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    print("\nRunning FP32 inference...")
    with torch.inference_mode():
        fp32_logits, _ = fp32_model(input_ids)

    print("Running INT8 inference...")
    with torch.inference_mode():
        int8_logits, _ = int8_model(input_ids)

    # Compare output tensors.
    difference = (fp32_logits - int8_logits).abs()
    

    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    fp32_next_token = torch.argmax(fp32_logits[:, -1, :], dim=-1)
    int8_next_token = torch.argmax(int8_logits[:, -1, :], dim=-1)
        # Compare next-token probability distributions.
    fp32_last = fp32_logits[:, -1, :]
    int8_last = int8_logits[:, -1, :]

    fp32_probs = F.softmax(fp32_last, dim=-1)
    int8_probs = F.softmax(int8_last, dim=-1)

    # Top-5 predictions
    fp32_top_probs, fp32_top_ids = torch.topk(
        fp32_probs, k=5, dim=-1
    )

    int8_top_probs, int8_top_ids = torch.topk(
        int8_probs, k=5, dim=-1
    )

    print("\nTOP-5 NEXT TOKEN COMPARISON")
    print("===========================")

    print("\nFP32:")
    for token_id, probability in zip(
        fp32_top_ids[0], fp32_top_probs[0]
    ):
        token = tokenizer.decode([token_id.item()])
        print(f"{token!r}: {probability.item():.6f}")

    print("\nINT8:")
    for token_id, probability in zip(
        int8_top_ids[0], int8_top_probs[0]
    ):
        token = tokenizer.decode([token_id.item()])
        print(f"{token!r}: {probability.item():.6f}")

    overlap = len(
        set(fp32_top_ids[0].tolist())
        & set(int8_top_ids[0].tolist())
    )

    print(f"\nTop-5 token overlap: {overlap}/5")

    # Distribution similarity
    kl_divergence = F.kl_div(
        torch.log(int8_probs.clamp_min(1e-12)),
        fp32_probs,
        reduction="batchmean"
    )

    print(f"KL divergence (FP32 || INT8): {kl_divergence.item():.6f}")

    print("\nNUMERICAL COMPARISON")
    print("====================")
    print(f"FP32 logits shape: {tuple(fp32_logits.shape)}")
    print(f"INT8 logits shape: {tuple(int8_logits.shape)}")
    print(f"Maximum absolute difference: {max_difference:.6f}")
    print(f"Mean absolute difference: {mean_difference:.6f}")
    print("\nLOGIT SCALE")
    print("===========")
    print(
        "FP32 maximum absolute logit:",
        fp32_logits.abs().max().item()
    )
    print(
        "FP32 mean absolute logit:",
        fp32_logits.abs().mean().item()
    )
    print(
        "INT8 maximum absolute logit:",
        int8_logits.abs().max().item()
    )
    print(
        "INT8 mean absolute logit:",
        int8_logits.abs().mean().item()
    )

    relative_error = (
        difference.norm() / fp32_logits.norm()
    ).item()

    print("Relative L2 error:", relative_error)

    print("\nNEXT TOKEN COMPARISON")
    print("=====================")
    print("FP32:", tokenizer.decode(fp32_next_token))
    print("INT8:", tokenizer.decode(int8_next_token))
    print(
        "Same next token:",
        torch.equal(fp32_next_token, int8_next_token)
    )


if __name__ == "__main__":
    main()
