
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_selective


def calculate_kl(fp32_logits, quantized_logits):
    """
    Calculate KL divergence: FP32 || quantized.
    Compare probability distributions at the final position.
    """
    fp32_log_probs = F.log_softmax(
        fp32_logits[:, -1, :], dim=-1
    )

    quantized_log_probs = F.log_softmax(
        quantized_logits[:, -1, :], dim=-1
    )

    fp32_probs = fp32_log_probs.exp()

    return F.kl_div(
        quantized_log_probs,
        fp32_probs,
        reduction="batchmean"
    ).item()


def main():
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Loading FP32 model...")
    fp32_model = TinyInferGPT2()
    load_gpt2_weights(fp32_model)
    fp32_model.eval()

    prompt = "Artificial intelligence is changing the world"
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    )

    with torch.inference_mode():
        fp32_logits, _ = fp32_model(input_ids)

    experiments = {
        "All layers quantized": lambda name: name != "lm_head",

        "Exclude MLP output": lambda name: (
            name != "lm_head"
            and not name.endswith("mlp.proj")
        ),

        "Exclude attention output": lambda name: (
            name != "lm_head"
            and not name.endswith("attention.out_proj")
        ),

        "Exclude MLP expansion": lambda name: (
            name != "lm_head"
            and not name.endswith("mlp.fc")
        ),

        "Exclude attention QKV": lambda name: (
            name != "lm_head"
            and not name.endswith("attention.qkv")
        ),
    }

    print("\nSELECTIVE QUANTIZATION SWEEP")
    print("============================")

    results = []

    for experiment_name, selection in experiments.items():
        print(f"\nRunning: {experiment_name}")

        quantized_model = quantize_selective(
            fp32_model,
            should_quantize=selection
        )

        with torch.inference_mode():
            quantized_logits, _ = quantized_model(input_ids)

        kl = calculate_kl(
            fp32_logits,
            quantized_logits
        )

        quantized_count = sum(
            1
            for module in quantized_model.modules()
            if isinstance(
                module,
                torch.ao.nn.quantized.dynamic.Linear
            )
        )

        results.append(
            (experiment_name, kl, quantized_count)
        )

        print(f"KL divergence: {kl:.6f}")
        print(f"Quantized Linear layers: {quantized_count}")

    print("\nFINAL RESULTS")
    print("=============")

    for name, kl, count in results:
        print(
            f"{name:30} "
            f"KL: {kl:.6f} | "
            f"Quantized layers: {count}"
        )


if __name__ == "__main__":
    main()
