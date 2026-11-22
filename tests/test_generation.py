
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.generation import generate


def test_text_generation():

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create TinyInfer model
    model = TinyInferGPT2()

    # Load pretrained GPT-2 weights
    reference = load_gpt2_weights(model)

    # Prepare prompt
    prompt = "Artificial intelligence"

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    # Generate text using TinyInfer
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=30
    )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print("\nPrompt:", prompt)
    print("Generated text:", generated_text)
    print("Generated token count:", output_ids.shape[1] - input_ids.shape[1])

    # Basic correctness checks
    assert output_ids.shape[1] == input_ids.shape[1] + 30
    assert generated_text.startswith(prompt)



