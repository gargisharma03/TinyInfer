
import time
import statistics
import csv
import torch
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights

LENGTHS = [10, 25, 50, 100, 200, 300, 400]
REPEATS = 3

PROMPT = (
    "Artificial intelligence is transforming many industries. "
    "Machine learning models can understand and generate text. "
    "Deep learning uses neural networks with multiple layers."
)


def generate(model, input_ids, count, cached):
    generated = input_ids.clone()

    with torch.inference_mode():
        logits, past = model(generated)

        for step in range(count):
            token = torch.argmax(
                logits[:, -1, :], dim=-1, keepdim=True
            )
            generated = torch.cat([generated, token], dim=1)

            if step < count - 1:
                if cached:
                    logits, past = model(
                        token, past_key_values=past
                    )
                else:
                    logits, _ = model(generated)

    return generated


def measure(model, input_ids, count, cached):
    start = time.perf_counter()

    generate(model, input_ids, count, cached)

    elapsed = time.perf_counter() - start
    return count / elapsed


def main():
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = TinyInferGPT2()
    load_gpt2_weights(model)
    model.eval()

    input_ids = tokenizer(
        PROMPT, return_tensors="pt"
    )["input_ids"]

    results = []

    print("\nTOKEN LENGTH SWEEP")
    print("==================")

    for length in LENGTHS:
        print(f"\nTesting {length} new tokens...")

        # Warm-up
        measure(model, input_ids, min(length, 5), True)
        measure(model, input_ids, min(length, 5), False)

        speeds = {"cached": [], "uncached": []}

        # Interleave execution order
        for run in range(REPEATS):
            order = (
                [("cached", True), ("uncached", False)]
                if run % 2 == 0
                else [("uncached", False), ("cached", True)]
            )

            for name, use_cache in order:
                speed = measure(
                    model, input_ids, length, use_cache
                )
                speeds[name].append(speed)

        cached_speed = statistics.median(speeds["cached"])
        uncached_speed = statistics.median(speeds["uncached"])

        print(f"Cached:   {cached_speed:.2f} tok/s")
        print(f"Uncached: {uncached_speed:.2f} tok/s")

        results.append({
            "tokens": length,
            "cached": cached_speed,
            "uncached": uncached_speed
        })

    with open(
        "benchmarks/token_length_results.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["tokens", "cached", "uncached"]
        )
        writer.writeheader()
        writer.writerows(results)

    x = [r["tokens"] for r in results]
    cached = [r["cached"] for r in results]
    uncached = [r["uncached"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(x, cached, marker="o", label="KV Cached")
    plt.plot(x, uncached, marker="s", label="Uncached")

    plt.xlabel("Number of generated tokens")
    plt.ylabel("Throughput (tokens/second)")
    plt.title("TinyInfer: KV Cache Performance vs Token Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "benchmarks/token_length_sweep.png",
        dpi=200
    )

    print("\nSaved:")
    print("benchmarks/token_length_results.csv")
    print("benchmarks/token_length_sweep.png")


if __name__ == "__main__":
    main()
