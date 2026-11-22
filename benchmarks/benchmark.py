
import time
import statistics
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights


REPEATS = 5

WORKLOADS = [
    {
        "name": "Short",
        "prompt": "Artificial intelligence is changing the world",
        "new_tokens": 30,
    },
    {
        "name": "Medium",
        "prompt": (
            "Artificial intelligence is transforming many industries. "
            "Machine learning models can understand and generate text."
        ),
        "new_tokens": 50,
    },
    {
        "name": "Long",
        "prompt": (
            "Artificial intelligence is transforming many industries. "
            "Machine learning models can understand and generate text. "
            "Deep learning uses neural networks with multiple layers. "
            "Transformer models use attention mechanisms to process "
            "sequences and capture relationships between tokens."
        ),
        "new_tokens": 100,
    },
]


def run_uncached(model, input_ids, max_new_tokens):

    generated_ids = input_ids.clone()

    total_start = time.perf_counter()

    with torch.inference_mode():

        prefill_start = time.perf_counter()
        logits, _ = model(generated_ids)
        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()

        for step in range(max_new_tokens):

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1
            )

            if step < max_new_tokens - 1:
                logits, _ = model(generated_ids)

        decode_time = time.perf_counter() - decode_start

    total_time = time.perf_counter() - total_start

    return generated_ids, prefill_time, decode_time, total_time


def run_cached(model, input_ids, max_new_tokens):

    generated_ids = input_ids.clone()

    total_start = time.perf_counter()

    with torch.inference_mode():

        prefill_start = time.perf_counter()

        logits, past_key_values = model(generated_ids)

        prefill_time = time.perf_counter() - prefill_start

        decode_start = time.perf_counter()

        for step in range(max_new_tokens):

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1
            )

            if step < max_new_tokens - 1:

                logits, past_key_values = model(
                    next_token,
                    past_key_values=past_key_values
                )

        decode_time = time.perf_counter() - decode_start

    total_time = time.perf_counter() - total_start

    return generated_ids, prefill_time, decode_time, total_time


def benchmark_method(name, function, model, input_ids, max_new_tokens):

    prefill_times = []
    decode_times = []
    total_times = []

    for run in range(REPEATS):

        _, prefill, decode, total = function(
            model,
            input_ids,
            max_new_tokens
        )

        prefill_times.append(prefill)
        decode_times.append(decode)
        total_times.append(total)

        print(
            f"  Run {run + 1}: "
            f"prefill={prefill:.4f}s, "
            f"decode={decode:.4f}s, "
            f"total={total:.4f}s"
        )

    average_prefill = statistics.mean(prefill_times)
    average_decode = statistics.mean(decode_times)
    average_total = statistics.mean(total_times)

    median_decode = statistics.median(decode_times)

    decode_std = (
        statistics.stdev(decode_times)
        if len(decode_times) > 1
        else 0.0
    )

    tokens_per_second = max_new_tokens / average_decode

    return {
        "name": name,
        "prefill": average_prefill,
        "decode": average_decode,
        "total": average_total,
        "median_decode": median_decode,
        "decode_std": decode_std,
        "tokens_per_second": tokens_per_second,
    }


def main():

    print("TinyInfer Multi-Workload CPU Benchmark")
    print("======================================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = TinyInferGPT2()
    load_gpt2_weights(model)
    model.eval()

    all_results = []

    for workload in WORKLOADS:

        prompt = workload["prompt"]
        max_new_tokens = workload["new_tokens"]

        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"]

        print("\n" + "=" * 55)
        print(f"Workload: {workload['name']}")
        print(f"Prompt tokens: {input_ids.shape[1]}")
        print(f"New tokens: {max_new_tokens}")

        # Warm up both methods
        print("Warming up...")
        run_uncached(model, input_ids, 3)
        run_cached(model, input_ids, 3)

        print("\nUncached:")
        uncached = benchmark_method(
            "Uncached",
            run_uncached,
            model,
            input_ids,
            max_new_tokens
        )

        print("\nKV Cached:")
        cached = benchmark_method(
            "KV Cached",
            run_cached,
            model,
            input_ids,
            max_new_tokens
        )

        # Verify identical generated sequences
        uncached_ids, *_ = run_uncached(
            model, input_ids, max_new_tokens
        )

        cached_ids, *_ = run_cached(
            model, input_ids, max_new_tokens
        )

        matches = torch.equal(uncached_ids, cached_ids)

        print(f"\nGenerated tokens match: {matches}")

        if not matches:
            raise RuntimeError(
                f"Generation mismatch in {workload['name']} workload"
            )

        speedup = uncached["decode"] / cached["decode"]

        print(f"Decode speedup: {speedup:.2f}x")

        print("\nUncached statistics:")
        print(f"  Median decode: {uncached['median_decode']:.4f} seconds")
        print(f"  Decode std dev: {uncached['decode_std']:.4f} seconds")

        print("\nKV Cached statistics:")
        print(f"  Median decode: {cached['median_decode']:.4f} seconds")
        print(f"  Decode std dev: {cached['decode_std']:.4f} seconds")

        all_results.append({
            "workload": workload["name"],
            "prompt_tokens": input_ids.shape[1],
            "new_tokens": max_new_tokens,
            "uncached": uncached,
            "cached": cached,
            "speedup": speedup,
        })

    # Final summary
    print("\n\nFINAL BENCHMARK SUMMARY")
    print("=======================")

    print(
        f"{'Workload':<12}"
        f"{'Prompt':>9}"
        f"{'Tokens':>9}"
        f"{'Uncached':>12}"
        f"{'Cached':>12}"
        f"{'Speedup':>10}"
    )

    for result in all_results:

        print(
            f"{result['workload']:<12}"
            f"{result['prompt_tokens']:>9}"
            f"{result['new_tokens']:>9}"
            f"{result['uncached']['tokens_per_second']:>10.2f}"
            f"{result['cached']['tokens_per_second']:>12.2f}"
            f"{result['speedup']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
