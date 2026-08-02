"""
benchmark_quantization.py

Efficient AI / On-Device Quantization Benchmark
-------------------------------------------------
Compares FP16 (baseline), 4-bit NF4 (bitsandbytes), and GGUF Q4_K_M (llama.cpp)
inference for Qwen3-4B-Instruct-2507 on three axes:
  1. Peak VRAM usage
  2. Generation speed (tokens/sec) and time-to-first-token (TTFT)
  3. Output quality (perplexity on the WikiText-2 test split)

It also sweeps prompt/context length (128 -> 4096 tokens) to show how VRAM
(KV-cache growth) and speed degrade as context grows -- this is the
differentiating angle beyond a standard "single short prompt" benchmark.

Designed to run top-to-bottom in Google Colab (T4/L4 GPU runtime).
See README.md for setup instructions.

Usage:
    python benchmark_quantization.py --stage all
    python benchmark_quantization.py --stage fp16
    python benchmark_quantization.py --stage nf4
    python benchmark_quantization.py --stage gguf
    python benchmark_quantization.py --stage context_sweep
    python benchmark_quantization.py --stage perplexity

Note: this script was authored and syntax-checked without access to a GPU or
to huggingface.co (sandboxed authoring environment). Expect to do a small
amount of first-run debugging on your actual GPU environment -- most likely
around the exact GGUF filename on the Hub and CUDA/driver versions.
"""

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GGUF_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILENAME_HINT = "Q4_K_M"  # script globs for a file containing this substring
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- device abstraction (CUDA or Apple Silicon MPS) --------------------------
if torch.cuda.is_available():
    DEVICE_KIND = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_KIND = "mps"
else:
    DEVICE_KIND = "cpu"

# On this stack (transformers 5.14 / accelerate 1.14 / torch 2.13), loading a
# plain fp16 model with device_map={"":"mps"} segfaults inside accelerate's
# meta-device weight dispatch. bitsandbytes NF4 loading with the same
# device_map is unaffected (different weight-loading path), so only the FP16
# baseline needs the cpu-load-then-.to("mps") workaround below.
MPS_FP16_DEVICE_MAP_SEGFAULTS = True


def sync_device():
    if DEVICE_KIND == "cuda":
        torch.cuda.synchronize()
    elif DEVICE_KIND == "mps":
        torch.mps.synchronize()


def reset_peak_memory():
    if DEVICE_KIND == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync_device()


def peak_memory_gb():
    if DEVICE_KIND == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    if DEVICE_KIND == "mps":
        # torch.mps exposes no reset-able peak counter; current_allocated_memory()
        # sampled right after generation is a close proxy since KV-cache/activation
        # memory grows ~monotonically during autoregressive decoding.
        return torch.mps.current_allocated_memory() / (1024 ** 3)
    return None


def empty_cache():
    if DEVICE_KIND == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE_KIND == "mps":
        torch.mps.empty_cache()


def load_causal_lm(model_id, dtype=None, quantization_config=None):
    """Load a causal LM, routing around the fp16+device_map=mps segfault."""
    from transformers import AutoModelForCausalLM

    if quantization_config is not None:
        # bitsandbytes' loading path handles device_map={"":"mps"} fine.
        device_map = "auto" if DEVICE_KIND == "cuda" else {"": DEVICE_KIND}
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map=device_map
        )

    if DEVICE_KIND == "mps" and MPS_FP16_DEVICE_MAP_SEGFAULTS:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        return model.to("mps")

    device_map = "auto" if DEVICE_KIND != "cpu" else None
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)

PROMPT = "Explain the core concept of model quantization in 3 bullet points."
MAX_NEW_TOKENS = 150
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096]  # used only in --stage context_sweep


@dataclasses.dataclass
class BenchmarkResult:
    config: str
    context_len: Optional[int]
    peak_vram_gb: Optional[float]
    ttft_s: float
    tokens_per_sec: float
    perplexity: Optional[float] = None

    def to_dict(self):
        return dataclasses.asdict(self)


def log_result(result: BenchmarkResult, path: Path = RESULTS_DIR / "results.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
    print(
        f"[{result.config}] ctx={result.context_len} "
        f"VRAM={result.peak_vram_gb} GB  "
        f"TTFT={result.ttft_s:.3f}s  "
        f"speed={result.tokens_per_sec:.2f} tok/s  "
        f"ppl={result.perplexity}"
    )


def make_long_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt of approximately `target_tokens` tokens by repeating a
    filler passage, so we can test behavior at growing context lengths."""
    filler = (
        "Large language models compress statistical patterns of language "
        "into billions of parameters, and quantization reduces the numerical "
        "precision of those parameters to save memory and increase speed. "
    )
    text = filler
    while len(tokenizer(text).input_ids) < target_tokens:
        text += filler
    ids = tokenizer(text).input_ids[:target_tokens]
    return tokenizer.decode(ids)


def build_chat_input_ids(tokenizer, model, prompt: str):
    """Use the model's chat template (this is an *Instruct* model, so raw
    continuation would be an unfair/unrealistic test). enable_thinking=False
    keeps output length comparable across configs; if your tokenizer's
    template doesn't recognize the kwarg it is silently ignored by Jinja."""
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    # transformers>=5 returns a BatchEncoding (dict-like) here instead of a
    # bare tensor; older versions returned the input_ids tensor directly.
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return input_ids.to(model.device)


def measure_hf_generate(model, tokenizer, prompt: str, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = build_chat_input_ids(tokenizer, model, prompt)
    attention_mask = torch.ones_like(input_ids)

    reset_peak_memory()

    start = time.time()
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=1, do_sample=False,
        )
    sync_device()
    ttft = time.time() - start

    start2 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
    sync_device()
    gen_time = time.time() - start2

    vram = peak_memory_gb()

    n_new = out.shape[1] - input_ids.shape[1]
    tps = n_new / gen_time if gen_time > 0 else float("nan")
    return ttft, tps, vram


def run_fp16(context_sweep=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_causal_lm(MODEL_ID, dtype=torch.float16)

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        prompt = make_long_prompt(tokenizer, ctx_len) if ctx_len else PROMPT
        ttft, tps, vram = measure_hf_generate(model, tokenizer, prompt)
        log_result(BenchmarkResult("FP16", ctx_len, vram, ttft, tps))

    del model
    gc.collect()
    empty_cache()


def run_nf4(context_sweep=False):
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if DEVICE_KIND == "cpu":
        print(
            "[NF4] No CUDA or MPS GPU detected -- bitsandbytes 4-bit needs a "
            "GPU backend. Skipping this stage."
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = load_causal_lm(MODEL_ID, quantization_config=bnb_config)

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        prompt = make_long_prompt(tokenizer, ctx_len) if ctx_len else PROMPT
        ttft, tps, vram = measure_hf_generate(model, tokenizer, prompt)
        log_result(BenchmarkResult("NF4_4bit", ctx_len, vram, ttft, tps))

    del model
    gc.collect()
    empty_cache()


def download_gguf() -> str:
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(GGUF_REPO)
    candidates = [
        f for f in files
        if GGUF_FILENAME_HINT.lower() in f.lower() and f.endswith(".gguf")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No file containing '{GGUF_FILENAME_HINT}' found in {GGUF_REPO}. "
            f"Check https://huggingface.co/{GGUF_REPO} and update GGUF_REPO / "
            f"GGUF_FILENAME_HINT at the top of this script to match an actual "
            f"filename in that repo."
        )
    local_path = hf_hub_download(GGUF_REPO, candidates[0])
    print(f"[GGUF] downloaded {candidates[0]}")
    return local_path


def make_long_prompt_gguf(llm, target_tokens: int) -> str:
    """Build a prompt of exactly `target_tokens` tokens using llama.cpp's own
    tokenizer, mirroring make_long_prompt's approach for the HF path. The
    original word-repetition approximation underestimated token count badly
    enough to exceed the allocated n_ctx and crash at ctx_len=4096."""
    filler = (
        "Large language models compress statistical patterns of language "
        "into billions of parameters, and quantization reduces the numerical "
        "precision of those parameters to save memory and increase speed. "
    )
    text = filler
    while len(llm.tokenize(text.encode("utf-8"))) < target_tokens:
        text += filler
    ids = llm.tokenize(text.encode("utf-8"))[:target_tokens]
    return llm.detokenize(ids).decode("utf-8", errors="ignore")


def run_gguf(context_sweep=False):
    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "[GGUF] llama-cpp-python not installed or failed to import with "
            "CUDA support. See README.md for the CUDA install command. Skipping."
        )
        return

    model_path = download_gguf()

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        n_ctx = max(4096, (ctx_len or 0) + MAX_NEW_TOKENS + 64)
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)

        if ctx_len is None:
            long_prompt = PROMPT
        else:
            long_prompt = make_long_prompt_gguf(llm, ctx_len)

        start = time.time()
        _ = llm(long_prompt, max_tokens=1)
        ttft = time.time() - start

        start2 = time.time()
        out = llm(long_prompt, max_tokens=MAX_NEW_TOKENS)
        gen_time = time.time() - start2
        n_new = out["usage"]["completion_tokens"]
        tps = n_new / gen_time if gen_time > 0 else float("nan")

        log_result(BenchmarkResult("GGUF_Q4_K_M", ctx_len, None, ttft, tps))
        del llm
        gc.collect()


def run_perplexity():
    """Sliding-window perplexity on WikiText-2 (test split), using the exact
    same dataset/stride/window for FP16 and NF4 so that comparison is
    apples-to-apples. GGUF perplexity is intentionally NOT computed the same
    way here -- llama.cpp has its own perplexity implementation, and mixing
    the two methods would produce numbers that look comparable but aren't.
    Report this as an explicit limitation (see report/technical_report_template.md).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # "wikitext" (no namespace) is a legacy script-based dataset that the
    # installed datasets/huggingface_hub versions fail to resolve; the
    # Salesforce/wikitext parquet mirror has identical content/config.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    encodings = tokenizer(text, return_tensors="pt")

    max_length = 1024
    stride = 512

    def compute_ppl(model):
        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end
            input_ids = encodings.input_ids[:, begin:end].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                nlls.append(outputs.loss * trg_len)
            prev_end = end
            if end == seq_len:
                break
        return torch.exp(torch.stack(nlls).sum() / prev_end).item()

    model_fp16 = load_causal_lm(MODEL_ID, dtype=torch.float16)
    ppl_fp16 = compute_ppl(model_fp16)
    log_result(BenchmarkResult("FP16", None, None, 0.0, 0.0, perplexity=ppl_fp16))
    del model_fp16
    gc.collect()
    empty_cache()

    if DEVICE_KIND != "cpu":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_nf4 = load_causal_lm(MODEL_ID, quantization_config=bnb_config)
        ppl_nf4 = compute_ppl(model_nf4)
        log_result(BenchmarkResult("NF4_4bit", None, None, 0.0, 0.0, perplexity=ppl_nf4))
        del model_nf4
        gc.collect()
        empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "fp16", "nf4", "gguf", "context_sweep", "perplexity"],
        default="all",
    )
    args = parser.parse_args()

    try:
        if args.stage in ("all", "fp16"):
            run_fp16()
        if args.stage in ("all", "nf4"):
            run_nf4()
        if args.stage in ("all", "gguf"):
            run_gguf()
        if args.stage == "context_sweep":
            run_fp16(context_sweep=True)
            run_nf4(context_sweep=True)
            run_gguf(context_sweep=True)
        if args.stage in ("all", "perplexity"):
            run_perplexity()
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA out of memory. Try: smaller MAX_NEW_TOKENS, a shorter max "
            "CONTEXT_LENGTHS list, or Runtime > Restart session in Colab "
            "between stages to fully release VRAM."
        )
        raise

    print(f"\nAll results appended to {RESULTS_DIR / 'results.jsonl'}")


if __name__ == "__main__":
    main()
