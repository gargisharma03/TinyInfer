
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_model_dynamic
from tinyinfer.generation import generate


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Loading FP32 model...")
    fp32_model = TinyInferGPT2()
    load_gpt2_weights(fp32_model)
    fp32_model.eval()

    print("\nCreating selective INT8 model...")
    int8_model = quantize_model_dynamic(fp32_model)
    int8_model.eval()

    prompts = [
        "Artificial intelligence is",
        "The history of science",
        "Once upon a time",
    ]

    for prompt in prompts:
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt"
        )

        with torch.inference_mode():
            fp32_ids = generate(
                fp32_model,
                input_ids,
                max_new_tokens=30
            )

            int8_ids = generate(
                int8_model,
                input_ids,
                max_new_tokens=30
            )

        fp32_text = tokenizer.decode(
            fp32_ids[0],
            skip_special_tokens=True
        )

        int8_text = tokenizer.decode(
            int8_ids[0],
            skip_special_tokens=True
        )

        matching = (fp32_ids == int8_ids).sum().item()
        total = fp32_ids.numel()

        print("\n" + "=" * 60)
        print(f"PROMPT: {prompt}")
        print("=" * 60)

        print("\nFP32:")
        print(fp32_text)

        print("\nINT8:")
        print(int8_text)

        print("\nExact token sequence match:", torch.equal(
            fp32_ids, int8_ids
        ))
        print(f"Matching token positions: {matching}/{total}")


if __name__ == "__main__":
    main()
