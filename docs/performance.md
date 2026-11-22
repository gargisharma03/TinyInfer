# TinyInfer Performance Report

## Summary

TinyInfer is a from-scratch GPT-2 Small (124M) inference engine, verified against Hugging Face. On CPU, KV caching speeds up decoding by 2.1x at 10 new tokens and by 13.4x at 400 new tokens. Cached throughput stays between roughly 46 and 49 tok/s for 50-200 new tokens, while uncached throughput falls from 13.6 to 2.2 tok/s across the sweep. Cached and uncached runs produce identical token sequences. INT8 dynamic quantization reduced model size by 33.9% in the tested configuration, but ran about 16% slower than FP32 on this CPU at batch size 1.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.13.9 |
| PyTorch | 2.11.0+cpu |
| Model | GPT-2 Small (124M parameters) |
| Device | CPU (Intel Core i7-13620H) |
| Precision | FP32 (INT8 where stated) |
| Decoding | Greedy |

**Metric definition.** Throughput (tok/s) is generated tokens divided by decode time. Prefill is excluded.

## 1. KV Cache: Workload Benchmark

5 repetitions per method, warm-up run discarded.

| Workload | Prompt tokens | Generated tokens | Uncached (tok/s) | KV cached (tok/s) | Speedup |
|---|---:|---:|---:|---:|---:|
| Short | 7 | 30 | 22.94 | 53.47 | 2.33x |
| Medium | 17 | 50 | 16.59 | 51.59 | 3.11x |
| Long | 41 | 100 | 9.00 | 49.10 | 5.45x |

Generated sequences matched exactly in every workload.

## 2. KV Cache: Token Length Sweep

Fixed prompt, varying the number of new tokens. Raw data: `benchmarks/token_length_results.csv`. Plot: `benchmarks/token_length_sweep.png`.

| New tokens | Uncached (tok/s) | KV cached (tok/s) | Speedup |
|---:|---:|---:|---:|
| 10 | 13.56 | 28.64 | 2.11x |
| 25 | 11.82 | 32.63 | 2.76x |
| 50 | 11.82 | 46.31 | 3.92x |
| 100 | 10.24 | 48.62 | 4.75x |
| 200 | 6.25 | 47.07 | 7.53x |
| 300 | 2.93 | 30.03 | 10.25x |
| 400 | 2.20 | 29.51 | 13.41x |

### Observations

- Uncached throughput falls steadily as sequences grow (13.56 to 2.20 tok/s), because every new token recomputes the full sequence.
- Cached throughput is roughly flat (46-49 tok/s) from 50 to 200 new tokens, so the speedup grows with length.
- Cached throughput is lower at both ends of the sweep: 28-33 tok/s at 10-25 new tokens, and about 30 tok/s at 300-400 new tokens. **These two effects are not yet explained.** Candidate causes are fixed per-run overhead at small sizes, and thermal throttling or cache-growth cost at large sizes. None has been tested.
- Absolute throughput differs between benchmark sessions (for example, uncached at 25 tokens here versus uncached at 30 tokens in Section 1). Compare numbers only within a table.

## 3. KV Cache Memory

For GPT-2 Small the cache stores K and V for every layer and every token:

```
2 (K and V) x 12 layers x 768 dims x 4 bytes (FP32) = 73,728 bytes = 72 KiB per token
```

At the full 1024-token context this is 72 MiB, small compared with the roughly 475 MiB of FP32 weights.

> TODO: add a measured cache size to confirm the analytic figure.

## 4. INT8 Dynamic Quantization

Per-channel INT8 weights with dynamic activation quantization. The token embedding and tied output head stay in FP32.

### Size and speed (cached decoding, 5 interleaved runs after warm-up)

| Config | Size (MiB) | Size change | Median tok/s | Mean tok/s | Speed vs FP32 |
|---|---:|---:|---:|---:|---:|
| FP32 | 474.75 | n/a | 48.95 | 48.20 | 1.00x |
| INT8 (selective) | 313.92 | -33.9% | 41.45 | 40.54 | 0.84x |

> TODO: confirm which layers `quantization_benchmark.py` quantizes. The size is consistent with one MLP layer type left in FP32 (36 of 48 linears quantized), not all 48.

### Quality (KL divergence vs FP32, single prompt)

| Configuration | KL | Quantized layers |
|---|---:|---:|
| All layers | 0.0588 | 48 |
| Exclude MLP output (`mlp.proj`) | 0.0240 | 36 |
| Exclude MLP expansion (`mlp.fc`) | 0.0250 | 36 |
| Exclude attention output | 0.0374 | 36 |
| Exclude attention QKV | 0.0617 | 36 |

### Findings

- The MLP layers dominate quantization error. Keeping either MLP layer type in FP32 cuts KL by more than half.
- On this CPU at batch size 1, INT8 is slower than FP32, so quantization here trades speed for size. Dynamic quantization pays activation-quantization overhead on every layer call, which can outweigh the cheaper INT8 matmul at one token per step.
- The KL numbers come from one prompt, and differences of about 0.001 are noise. Multi-prompt KL and WikiText-2 perplexity are still needed.

## Methodology Notes

- Earlier 3-repetition runs measured 3.25 tok/s for the long uncached workload (a 15.10x apparent speedup). The 5-repetition rerun gave 9.00 tok/s (5.45x), with cached throughput unchanged. The 15.10x figure should not be cited.
- Timings from the Streamlit demo (about 13-16 tok/s FP32) are much lower than terminal benchmark timings (about 48-49 tok/s) and are not used in this report. The terminal baseline was reconfirmed in the quantization benchmark.

## Limitations

- One CPU, one software environment, one model.
- Results measure inference speed, not model quality (apart from the KL experiment).
- Timings vary with system load and thermal conditions.

## Remaining Work

- [x] Warm-up run discarded before timing.
- [x] Token-length sweep.
- [x] Interleaved FP32 vs INT8 timing.
- [ ] Explain the cached throughput dip at 300-400 new tokens and the low values at 10-25.
- [ ] Report medians and spread for the sweep.
- [ ] Fix the thread count with `torch.set_num_threads(...)`.
- [ ] Measure KV cache memory.
- [ ] Multi-prompt KL and WikiText-2 perplexity for each quantization config.
- [ ] Speed and size for each selective-quantization config.