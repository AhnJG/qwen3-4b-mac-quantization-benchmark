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
import threading
import time
from pathlib import Path
from typing import Optional

import torch

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GGUF_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILENAME_HINT = "Q4_K_M"  # script globs for a file containing this substring
# F16 GGUF isolates the *engine* effect (llama.cpp vs HF transformers) from
# the *quantization* effect: GGUF-F16 vs HF-FP16 is same precision, different
# engine; GGUF-Q4_K_M vs GGUF-F16 is same engine, different precision. Without
# this, "GGUF is Nx faster" conflates both (an external review's main design
# critique -- see notes/).
GGUF_F16_FILENAME_HINT = "-F16"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- device abstraction (CUDA or Apple Silicon MPS) --------------------------
if torch.cuda.is_available():
    DEVICE_KIND = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_KIND = "mps"
else:
    DEVICE_KIND = "cpu"


def sync_device():
    if DEVICE_KIND == "cuda":
        torch.cuda.synchronize()
    elif DEVICE_KIND == "mps":
        torch.mps.synchronize()


class PeakMemoryTracker:
    """Tracks peak allocated device memory across a `with` block.

    On CUDA this just wraps torch.cuda's built-in reset/max_memory_allocated
    counters (exact). torch.mps exposes no equivalent reset-able peak counter
    -- current_allocated_memory() sampled *after* the block would only see
    whatever survives once the block's local tensors (e.g. generate()'s
    internal KV cache) have already been freed by Python refcounting, not the
    actual peak reached mid-generation. So on MPS a background thread polls
    current_allocated_memory() while the block runs and keeps the max.
    """

    def __init__(self, poll_interval_s=0.02):
        self.poll_interval_s = poll_interval_s
        self._peak_bytes = 0
        self._stop_event = None
        self._thread = None

    def __enter__(self):
        if DEVICE_KIND == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif DEVICE_KIND == "mps":
            self._peak_bytes = torch.mps.current_allocated_memory()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        sync_device()
        return self

    def _poll(self):
        while not self._stop_event.is_set():
            self._peak_bytes = max(self._peak_bytes, torch.mps.current_allocated_memory())
            self._stop_event.wait(self.poll_interval_s)

    def __exit__(self, *exc_info):
        sync_device()
        if DEVICE_KIND == "mps":
            self._stop_event.set()
            self._thread.join()

    def peak_gb(self):
        if DEVICE_KIND == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
        if DEVICE_KIND == "mps":
            return self._peak_bytes / (1024 ** 3)
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

    if DEVICE_KIND == "mps":
        # On this stack (transformers 5.14 / accelerate 1.14 / torch 2.13),
        # loading fp16 weights with device_map={"":"mps"} segfaults inside
        # accelerate's meta-device weight dispatch; load on CPU then move.
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
    device: Optional[str] = None
    prompt_tokens: Optional[int] = None
    n_new: Optional[int] = None

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


FILLER_TEXT = (
    "Large language models compress statistical patterns of language "
    "into billions of parameters, and quantization reduces the numerical "
    "precision of those parameters to save memory and increase speed. "
)


def build_long_prompt(target_tokens: int, tokenize, decode) -> str:
    """Build a prompt of exactly `target_tokens` tokens by repeating
    FILLER_TEXT, using the given tokenize/decode callables. Shared by the HF
    and GGUF paths so both context sweeps grow from the identical text --
    tweaking the filler in only one place would silently break the
    apples-to-apples comparison between backends."""
    text = FILLER_TEXT
    while len(tokenize(text)) < target_tokens:
        text += FILLER_TEXT
    return decode(tokenize(text)[:target_tokens])


def make_long_prompt(tokenizer, target_tokens: int) -> str:
    return build_long_prompt(
        target_tokens,
        tokenize=lambda t: tokenizer(t).input_ids,
        decode=tokenizer.decode,
    )


def build_chat_prompt_string(tokenizer, prompt: str) -> str:
    """Render `prompt` through the model's chat template as a plain string
    (this is an *Instruct* model, so raw continuation would be an unfair/
    unrealistic test -- and an unrealistic OOD input for the model, per the
    review that flagged the GGUF path skipping this). enable_thinking=False
    keeps output length comparable across configs; if your tokenizer's
    template doesn't recognize the kwarg it is silently ignored by Jinja.
    Both the HF and GGUF paths render through *this same* HF tokenizer call,
    so the two backends see byte-identical prompts -- not just "close enough"
    templates from two different Jinja engines."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def build_chat_input_ids(tokenizer, model, prompt: str):
    text = build_chat_prompt_string(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"].to(model.device)


def warmup_hf_generate(model, tokenizer):
    """One untimed generate() call to pay CUDA/MPS one-time costs (context
    init, cuBLAS handle creation, kernel autotune, lazy module init) before
    any timed measurement. Without this, whichever config/context-length
    happens to run first absorbs the whole cost, producing a non-monotonic
    TTFT that looks like a real effect but isn't (caught by external review:
    TTFT at ctx=128 was *longer* than at ctx=512)."""
    ids = tokenizer("Warm up.", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=4, do_sample=False)
    sync_device()


def measure_hf_generate(model, tokenizer, prompt: str, max_new_tokens=MAX_NEW_TOKENS):
    """Time a *single* generate() call, using a logits processor to stamp a
    timestamp on every decoding step (including the first). This replaces an
    earlier two-call design (one generate(max_new_tokens=1) for TTFT, then a
    second *independent* generate(max_new_tokens=150) for throughput) that an
    external review caught: the second call didn't reuse the first call's KV
    cache, so it silently re-ran the full O(context) prefill and folded that
    time into what was reported as decode throughput. That made tok/s crash
    with context length far faster than real decoding does, and made the
    (single-call, prefix-cache-reusing) GGUF path look artificially more
    robust to context growth by comparison -- not because it in fact was."""
    from transformers import LogitsProcessorList

    input_ids = build_chat_input_ids(tokenizer, model, prompt)
    attention_mask = torch.ones_like(input_ids)

    timestamps = []

    def _stamp(_input_ids, scores):
        timestamps.append(time.time())
        return scores

    with PeakMemoryTracker() as mem:
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                logits_processor=LogitsProcessorList([_stamp]),
            )
        sync_device()

    vram = mem.peak_gb()

    n_new = len(timestamps)
    ttft = timestamps[0] - start
    decode_time = timestamps[-1] - timestamps[0] if n_new > 1 else float("nan")
    tps = (n_new - 1) / decode_time if n_new > 1 and decode_time > 0 else float("nan")
    return ttft, tps, vram, input_ids.shape[1], n_new


def run_fp16(context_sweep=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_causal_lm(MODEL_ID, dtype=torch.float16)
    warmup_hf_generate(model, tokenizer)

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        prompt = make_long_prompt(tokenizer, ctx_len) if ctx_len else PROMPT
        ttft, tps, vram, prompt_tokens, n_new = measure_hf_generate(model, tokenizer, prompt)
        log_result(BenchmarkResult(
            "FP16", ctx_len, vram, ttft, tps,
            device=DEVICE_KIND, prompt_tokens=prompt_tokens, n_new=n_new,
        ))

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
    try:
        model = load_causal_lm(MODEL_ID, quantization_config=bnb_config)
    except Exception as e:
        # bitsandbytes' Apple Silicon/MPS support is version-dependent; on a
        # build without working MPS 4-bit kernels this fails at load time.
        print(
            f"[NF4] bitsandbytes 4-bit failed to load on {DEVICE_KIND} "
            f"({e!r}). Skipping this stage."
        )
        return

    warmup_hf_generate(model, tokenizer)

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        prompt = make_long_prompt(tokenizer, ctx_len) if ctx_len else PROMPT
        ttft, tps, vram, prompt_tokens, n_new = measure_hf_generate(model, tokenizer, prompt)
        log_result(BenchmarkResult(
            "NF4_4bit", ctx_len, vram, ttft, tps,
            device=DEVICE_KIND, prompt_tokens=prompt_tokens, n_new=n_new,
        ))

    del model
    gc.collect()
    empty_cache()


def download_gguf(filename_hint: str) -> str:
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(GGUF_REPO)
    candidates = [
        f for f in files
        if filename_hint.lower() in f.lower() and f.endswith(".gguf")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No file containing '{filename_hint}' found in {GGUF_REPO}. "
            f"Check https://huggingface.co/{GGUF_REPO} for an actual filename."
        )
    local_path = hf_hub_download(GGUF_REPO, candidates[0])
    print(f"[GGUF] downloaded {candidates[0]}")
    return local_path


def make_long_prompt_gguf(llm, target_tokens: int) -> str:
    """Build `target_tokens` worth of *filler content* using llama.cpp's own
    tokenizer to count length (this is the inner text later wrapped in a chat
    template -- see run_gguf). The original word-repetition approximation
    underestimated token count badly enough to exceed the allocated n_ctx and
    crash at ctx_len=4096."""
    return build_long_prompt(
        target_tokens,
        tokenize=lambda t: llm.tokenize(t.encode("utf-8")),
        decode=lambda ids: llm.detokenize(ids).decode("utf-8", errors="ignore"),
    )


def run_gguf(config_name: str, filename_hint: str, context_sweep=False):
    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "[GGUF] llama-cpp-python not installed or failed to import with "
            "CUDA support. See README.md for the CUDA install command. Skipping."
        )
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_path = download_gguf(filename_hint)

    for ctx_len in (CONTEXT_LENGTHS if context_sweep else [None]):
        n_ctx = (ctx_len or 0) + MAX_NEW_TOKENS + 64
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)

        filler = make_long_prompt_gguf(llm, ctx_len) if ctx_len else PROMPT
        # Same HF tokenizer, same chat template, as the HF path -- so both
        # backends see byte-identical formatted prompts (an external review
        # caught the earlier version feeding GGUF a raw, un-templated string,
        # which is an out-of-distribution input for an instruct model).
        prompt_str = build_chat_prompt_string(tokenizer, filler)
        # add_bos=False, special=True to match exactly what Llama.__call__
        # itself does internally when tokenizing a plain (no-FIM-suffix)
        # prompt -- otherwise this count wouldn't reflect what's actually fed
        # to the model (and literal "<|im_start|>" etc. would be counted as
        # ordinary text tokens instead of the single control token each is).
        prompt_tokens = len(llm.tokenize(prompt_str.encode("utf-8"), add_bos=False, special=True))

        # Untimed warmup on a throwaway prompt, then reset() so the timed
        # call below is forced to do a full, fresh prefill instead of
        # silently reusing the warmup's cached prefix (the mechanism behind
        # the earlier double-call bug: two calls on the *same* prompt let
        # llama.cpp's prefix-cache skip prefill on the second one entirely).
        _ = list(llm("Warm up.", max_tokens=4, temperature=0.0))
        llm.reset()

        timestamps = []
        n_new = 0
        start = time.time()
        for _chunk in llm(
            prompt_str, max_tokens=MAX_NEW_TOKENS, stream=True,
            temperature=0.0, top_k=1, repeat_penalty=1.0,
        ):
            timestamps.append(time.time())
            n_new += 1

        ttft = timestamps[0] - start
        decode_time = timestamps[-1] - timestamps[0] if n_new > 1 else float("nan")
        tps = (n_new - 1) / decode_time if n_new > 1 and decode_time > 0 else float("nan")

        log_result(BenchmarkResult(
            config_name, ctx_len, None, ttft, tps,
            device=DEVICE_KIND, prompt_tokens=prompt_tokens, n_new=n_new,
        ))
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
        try:
            model_nf4 = load_causal_lm(MODEL_ID, quantization_config=bnb_config)
        except Exception as e:
            model_nf4 = None
            print(
                f"[NF4] perplexity: bitsandbytes 4-bit failed to load on "
                f"{DEVICE_KIND} ({e!r}). Skipping NF4 perplexity."
            )
        if model_nf4 is not None:
            ppl_nf4 = compute_ppl(model_nf4)
            log_result(BenchmarkResult("NF4_4bit", None, None, 0.0, 0.0, perplexity=ppl_nf4))
            del model_nf4
            gc.collect()
            empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "fp16", "nf4", "gguf", "gguf_f16", "context_sweep", "perplexity"],
        default="all",
    )
    args = parser.parse_args()

    try:
        if args.stage in ("all", "fp16"):
            run_fp16()
        if args.stage in ("all", "nf4"):
            run_nf4()
        if args.stage in ("all", "gguf"):
            run_gguf("GGUF_Q4_K_M", GGUF_FILENAME_HINT)
        if args.stage in ("all", "gguf_f16"):
            run_gguf("GGUF_F16", GGUF_F16_FILENAME_HINT)
        if args.stage == "context_sweep":
            run_fp16(context_sweep=True)
            run_nf4(context_sweep=True)
            run_gguf("GGUF_Q4_K_M", GGUF_FILENAME_HINT, context_sweep=True)
            run_gguf("GGUF_F16", GGUF_F16_FILENAME_HINT, context_sweep=True)
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
