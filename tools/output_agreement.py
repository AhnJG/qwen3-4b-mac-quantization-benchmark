"""Do the four configs actually answer the same? Perplexity is an aggregate;
it can be identical while outputs diverge. Greedy-decode the same prompts on
all four and measure where they first differ."""
import sys, gc, json
sys.path.insert(0, ".")
import torch, benchmark_quantization as bq
from transformers import AutoTokenizer, BitsAndBytesConfig

PROMPTS = [
    "Explain the core concept of model quantization in 3 bullet points.",
    "What is the capital of France, and why did it become the capital?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarize the tradeoff between 4-bit and 16-bit inference in two sentences.",
]
N = 60
tok = AutoTokenizer.from_pretrained(bq.MODEL_ID)
out = {}

def hf(cfg):
    m = (bq.load_causal_lm(bq.MODEL_ID, dtype=torch.float16) if cfg == "FP16"
         else bq.load_causal_lm(bq.MODEL_ID, quantization_config=BitsAndBytesConfig(
             load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
             bnb_4bit_compute_dtype=torch.float16)))
    res = []
    for p in PROMPTS:
        ids = bq.build_chat_input_ids(tok, m, p)
        with torch.no_grad():
            o = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=N, do_sample=False)
        res.append(o[0][ids.shape[1]:].tolist())
    del m; gc.collect(); bq.empty_cache()
    return res

def gguf(cfg):
    from llama_cpp import Llama
    hint = bq.GGUF_F16_FILENAME_HINT if cfg == "GGUF_F16" else bq.GGUF_FILENAME_HINT
    llm = Llama(model_path=bq.download_gguf(hint), n_gpu_layers=-1, n_ctx=1024, verbose=False)
    res = []
    for p in PROMPTS:
        s = bq.build_chat_prompt_string(tok, p)
        llm.reset()
        o = llm(s, max_tokens=N, temperature=0.0, top_k=1, repeat_penalty=1.0)
        res.append(llm.tokenize(o["choices"][0]["text"].encode(), add_bos=False))
    del llm; gc.collect()
    return res

for cfg in ["FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M"]:
    out[cfg] = hf(cfg) if cfg in bq.HF_CONFIGS else gguf(cfg)

def divergence(a, b):
    """First index where token streams differ; None if identical over overlap."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None

base = "FP16"
print(f"\n{'비교':<28} " + "  ".join(f"P{i}" for i in range(len(PROMPTS))) + "   요약")
for cfg in ["GGUF_F16", "NF4_4bit", "GGUF_Q4_K_M"]:
    cells, ident = [], 0
    for i in range(len(PROMPTS)):
        d = divergence(out[base][i], out[cfg][i])
        cells.append("동일" if d is None else f"{d:>2}")
        ident += d is None
    print(f"  {base} vs {cfg:<16} " + "  ".join(f"{c:>2}" for c in cells) +
          f"   {ident}/{len(PROMPTS)} 완전일치")
print(f"\n(숫자 = {N}토큰 중 처음으로 갈라진 토큰 위치)")
print("\n--- P0 실제 출력 앞부분 ---")
for cfg in out:
    print(f"  {cfg:<12} {tok.decode(out[cfg][0])[:90]!r}")
