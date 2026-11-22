
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_model_dynamic


def main():
    print("Loading GPT-2...")

    model = TinyInferGPT2()
    load_gpt2_weights(model)
    model.eval()

    print("\nCreating dynamically quantized model...")
    quantized_model = quantize_model_dynamic(model)

    # Count quantized and remaining Linear layers.
    quantized_count = sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in quantized_model.modules()
    )

    original_linear_count = sum(
        isinstance(module, nn.Linear)
        for module in model.modules()
    )

    remaining_linear_count = sum(
        isinstance(module, nn.Linear)
        for module in quantized_model.modules()
    )

    print(f"\nOriginal Linear layers: {original_linear_count}")
    print(f"Quantized INT8 Linear layers: {quantized_count}")
    print(f"Remaining FP32 Linear layers: {remaining_linear_count}")

    assert quantized_count > 0, "No Linear layers were quantized!"

    print("\nINT8 quantization conversion verified!")


if __name__ == "__main__":
    main()
