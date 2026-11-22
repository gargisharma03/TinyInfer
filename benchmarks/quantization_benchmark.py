
import time
import statistics
import io
import torch
from transformers import GPT2Tokenizer

from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_model_dynamic


REPEATS = 5
NEW_TOKENS = 100

PROMPT = (
    "Artificial intelligence is transforming many industries. "
    "Machine learning models can understand and generate text."
)


def get_model_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 ** 2)


def run_cached_decode(model, input_ids, max_new_tokens):
    with torch.inference_mode():
        logits, past_key_values = model(input_ids)

        start = time.perf_counter()

        for step in range(max_new_tokens):
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            if step < max_new_tokens - 1:
                logits, past_key_values = model(
                    next_token,
                    past_key_values=past_key_values
                )

        elapsed = time.perf_counter() - start

    return max_new_tokens / elapsed


def main():
    print("FP32 vs INT8 CPU Benchmark")
    print("==========================")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("\nLoading FP32 GPT-2...")
    fp32_model = TinyInferGPT2()
    load_gpt2_weights(fp32_model)
    fp32_model.eval()

    print("\nCreating INT8 model...")
    int8_model = quantize_model_dynamic(fp32_model)
    int8_model.eval()

    fp32_size = get_model_size_mb(fp32_model)
    int8_size = get_model_size_mb(int8_model)

    print("\nMODEL SIZE")
    print("==========")
    print(f"FP32: {fp32_size:.2f} MB")
    print(f"INT8: {int8_size:.2f} MB")
    print(
        f"Reduction: "
        f"{(1 - int8_size / fp32_size) * 100:.2f}%"
    )

    input_ids = tokenizer(
        PROMPT,
        return_tensors="pt"
    )["input_ids"]

    models = {
        "FP32": fp32_model,
        "INT8": int8_model
    }

    results = {
        "FP32": [],
        "INT8": []
    }

    print("\nWarming up both models...")
    for model in models.values():
        run_cached_decode(model, input_ids, 5)

    print("\nINTERLEAVED SPEED BENCHMARK")
    print("===========================")

    for run in range(REPEATS):
        order = (
            ["FP32", "INT8"]
            if run % 2 == 0
            else ["INT8", "FP32"]
        )

        for name in order:
            speed = run_cached_decode(
                models[name],
                input_ids,
                NEW_TOKENS
            )

            results[name].append(speed)

            print(
                f"Run {run + 1} | {name}: "
                f"{speed:.2f} tok/s"
            )

    fp32_avg = statistics.mean(results["FP32"])
    int8_avg = statistics.mean(results["INT8"])

    fp32_median = statistics.median(results["FP32"])
    int8_median = statistics.median(results["INT8"])

    print("\nFINAL SPEED RESULTS")
    print("===================")

    print(f"FP32 average:  {fp32_avg:.2f} tok/s")
    print(f"INT8 average:  {int8_avg:.2f} tok/s")

    print(f"FP32 median:   {fp32_median:.2f} tok/s")
    print(f"INT8 median:   {int8_median:.2f} tok/s")

    print(
        f"\nINT8 speed ratio: "
        f"{int8_avg / fp32_avg:.2f}x"
    )

    if int8_avg > fp32_avg:
        print("INT8 is faster in this experiment.")
    else:
        print("FP32 is faster in this experiment.")

    print("\nIndividual measurements:")
    print("FP32:", [round(x, 2) for x in results["FP32"]])
    print("INT8:", [round(x, 2) for x in results["INT8"]])


if __name__ == "__main__":
    main()
