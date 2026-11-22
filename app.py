# app.py  --  run with:  streamlit run app.py
import time

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer
from tinyinfer.model import TinyInferGPT2
from tinyinfer.weights import load_gpt2_weights
from tinyinfer.quantization import quantize_model_dynamic
from tinyinfer.generation import generate   # ADAPT 1: import your generate function

st.set_page_config(page_title="TinyInfer", layout="wide")


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@st.cache_resource
def load_model(quantized: bool):
    model = TinyInferGPT2()
    load_gpt2_weights(model)
    model.eval()
    return quantize_model_dynamic(model) if quantized else model


@torch.no_grad()
@torch.inference_mode()
def top_k_table(model, tok, ids, k):
    logits, _ = model(ids)
    probs = torch.softmax(logits[0, -1], dim=-1)
    p, i = probs.topk(k)
    return pd.DataFrame(
        {"token": [repr(tok.decode(int(x))) for x in i], "probability": p.tolist()}
    ).set_index("token")

@torch.inference_mode()
def run_generation(model, ids, max_new, use_cache, temperature=0.0, sample_k=40):
    generated = ids.clone()
    logits, past = model(generated)

    t0 = time.perf_counter()
    for step in range(max_new):
        next_token = pick_next(logits, temperature, sample_k)
        generated = torch.cat([generated, next_token], dim=1)
        if step < max_new - 1:
            if use_cache:
                logits, past = model(next_token, past_key_values=past)
            else:
                logits, _ = model(generated)
    dt = time.perf_counter() - t0
    return generated, max_new, dt

def pick_next(logits, temperature, sample_k):
    last = logits[:, -1, :]
    if temperature == 0.0:
        return torch.argmax(last, dim=-1, keepdim=True)
    vals, idx = torch.topk(last / temperature, sample_k, dim=-1)
    choice = torch.multinomial(torch.softmax(vals, dim=-1), 1)
    return idx.gather(-1, choice)

st.title("TinyInfer")
st.caption("A from-scratch GPT-2 inference engine with KV caching and INT8 quantization.")

with st.sidebar:
    st.header("Settings")
    max_new = st.slider("New tokens", 10, 200, 50, step=10)
    use_cache = st.toggle("KV cache", value=True)
    quantized = st.toggle("INT8 quantization", value=False)
    top_k = st.slider("Top-k predictions", 3, 10, 5)
    temperature = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, step=0.1)
    sample_k = st.slider("Sampling top-k", 5, 100, 40)  

prompt = st.text_area("Prompt", "Artificial intelligence is", height=100)
col_gen, col_cmp = st.columns(2)
do_gen = col_gen.button("Generate", type="primary", use_container_width=True)
do_cmp = col_cmp.button("Compare cached vs uncached", use_container_width=True)

tok = load_tokenizer()

if do_gen or do_cmp:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model(quantized)
    ids = torch.tensor([tok.encode(prompt)])

    st.subheader("Next-token probabilities")
    df = top_k_table(model, tok, ids, top_k)
    left, right = st.columns([1, 1])
    left.bar_chart(df)
    right.dataframe(df.style.format({"probability": "{:.4f}"}), use_container_width=True)

    if do_gen:
        with st.spinner("Generating..."):
            out, new, dt = run_generation(model, ids, max_new, use_cache, temperature, sample_k)
        st.subheader("Output")
        st.write(tok.decode(out[0]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens/sec", f"{new / dt:.1f}")
        c2.metric("Time", f"{dt:.2f} s")
        c3.metric("Mode", f"{'cache' if use_cache else 'no cache'} / {'INT8' if quantized else 'FP32'}")

    if do_cmp:
        with st.spinner("Running with KV cache..."):
            out_c, n_c, t_c = run_generation(model, ids, max_new, True)
        with st.spinner("Running without KV cache (slower)..."):
            out_u, n_u, t_u = run_generation(model, ids, max_new, False)

        st.subheader("Cached vs uncached")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cached", f"{n_c / t_c:.1f} tok/s")
        c2.metric("Uncached", f"{n_u / t_u:.1f} tok/s")
        c3.metric("Speedup", f"{(n_c / t_c) / (n_u / t_u):.2f}x")
        st.bar_chart(pd.DataFrame({"tok/s": [n_u / t_u, n_c / t_c]}, index=["Uncached", "KV cached"]))

        if torch.equal(out_c, out_u):
            st.success("Outputs are identical with and without the cache.")
        else:
            st.error("Outputs differ. Greedy decoding should match exactly, so check the cache path.")
        st.write(tok.decode(out_c[0]))
        st.caption(
            "Single-run timing for demonstration. See PERFORMANCE.md for the repeated-run benchmark."
        )