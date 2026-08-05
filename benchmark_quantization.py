"""
benchmark_quantization.py

Efficient AI / On-Device Quantization Benchmark
-------------------------------------------------
Compares four configs for Qwen3-4B-Instruct-2507:
  - HF FP16          (transformers, PyTorch)
  - HF NF4 4-bit     (transformers + bitsandbytes)
  - GGUF F16         (llama.cpp)          } same engine, so the pair isolates
  - GGUF Q4_K_M      (llama.cpp)          } the quantization effect from the engine

on three axes: peak VRAM (HF only), decode speed (tokens/sec) + TTFT, and
WikiText-2 perplexity (HF only). It also sweeps prompt length 128..4096.

Measurement protocol (v3 -- rewritten after external peer review found the v2
timing was still wrong):
  * HF timing uses a SINGLE generate() call with a logits-processor callback
    that timestamps every decode step. Crucially the callback calls
    sync_device() BEFORE stamping: MPS/CUDA execution is asynchronous, so a
    bare time.time() records when the CPU reached that line, not when the GPU
    finished the work -- which let the whole prefill leak into the reported
    "decode" time (empirically: TTFT read 0.03s for a 4096-token prefill that
    actually takes ~16s, and that ~16s reappeared inside decode_time). This is
    the v2 reincarnation of v1's double-prefill bug; the sync fixes it.
  * GGUF timing streams a single call and skips llama.cpp's trailing
    finish_reason chunk (which carries no token but was being counted, giving
    n_new = max_tokens+1 and a small systematic tok/s over-count).
  * Every measurement loads its backend fresh, warms up at its own prompt
    shape (discarded), then measures -- so HF is symmetric to GGUF (which
    already rebuilds per call), cross-measurement allocator fragmentation
    can't accumulate along the context axis, and MPS shape-specific kernel
    compilation is paid before the timed run.
  * The (config x context) task list is measured in randomized order and
    repeated N times; report the median and spread rather than a single run.

Usage:
    python benchmark_quantization.py --stage all [--repeats 3]
    python benchmark_quantization.py --stage context_sweep [--repeats 3]
    python benchmark_quantization.py --stage perplexity
"""

import argparse
import dataclasses
import gc
import json
import random
import threading
import time
from pathlib import Path
from typing import Optional

import torch

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GGUF_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
GGUF_FILENAME_HINT = "Q4_K_M"   # globs for a file containing this substring
GGUF_F16_FILENAME_HINT = "-F16"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PROMPT = "Explain the core concept of model quantization in 3 bullet points."
MAX_NEW_TOKENS = 150
CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096]   # used only in --stage context_sweep
HF_CONFIGS = ("FP16", "NF4_4bit")
GGUF_CONFIGS = ("GGUF_Q4_K_M", "GGUF_F16")
ALL_CONFIGS = HF_CONFIGS + GGUF_CONFIGS

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

    On CUDA this wraps torch.cuda's reset/max_memory_allocated (exact).
    torch.mps has no reset-able peak counter, so a background thread polls
    current_allocated_memory() every 20ms and keeps the max. Note this samples
    torch's *allocated* bytes, not the driver's high-water mark, and can miss
    spikes shorter than the poll interval -- reported as an approximation.
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
        device_map = "auto" if DEVICE_KIND == "cuda" else {"": DEVICE_KIND}
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map=device_map
        )

    if DEVICE_KIND == "mps":
        # transformers 5.14 / accelerate 1.14 / torch 2.13: fp16 weights with
        # device_map={"":"mps"} segfault in accelerate's meta dispatch; load on
        # CPU then move.
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        return model.to("mps")

    device_map = "auto" if DEVICE_KIND != "cpu" else None
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)


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
    repeat: Optional[int] = None
    max_new_tokens: Optional[int] = None

    def to_dict(self):
        return dataclasses.asdict(self)


def log_result(result: BenchmarkResult, path: Path = RESULTS_DIR / "results.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
    print(
        f"[{result.config}] ctx={result.context_len} rep={result.repeat} "
        f"VRAM={result.peak_vram_gb} GB  "
        f"TTFT={result.ttft_s:.3f}s  "
        f"decode={result.tokens_per_sec:.2f} tok/s  "
        f"ppl={result.perplexity}"
    )


FILLER_TEXT = (
    "Large language models compress statistical patterns of language "
    "into billions of parameters, and quantization reduces the numerical "
    "precision of those parameters to save memory and increase speed. "
)


def build_long_prompt(target_tokens: int, tokenize, decode) -> str:
    """Build `target_tokens` worth of filler by repeating FILLER_TEXT, counting
    length with the given tokenize/decode callables."""
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


def make_long_prompt_gguf(llm, target_tokens: int) -> str:
    """GGUF filler counter. add_bos=False so the count matches what actually
    gets fed later (Qwen3 GGUF has add_bos_token=false, but being explicit keeps
    this correct if the model is ever swapped for a BOS-adding one)."""
    return build_long_prompt(
        target_tokens,
        tokenize=lambda t: llm.tokenize(t.encode("utf-8"), add_bos=False),
        decode=lambda ids: llm.detokenize(ids).decode("utf-8", errors="ignore"),
    )


def build_chat_prompt_string(tokenizer, prompt: str) -> str:
    """Render `prompt` through the model's chat template as a plain string.
    Both HF and GGUF paths render the template wrapper through *this same* HF
    tokenizer call (the filler inside is length-matched by each backend's own
    tokenizer -- see prompt_tokens equality across the sweep)."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )


def build_chat_input_ids(tokenizer, model, prompt: str):
    text = build_chat_prompt_string(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"].to(model.device)


# --- HF timing ---------------------------------------------------------------

def measure_hf_generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Single generate(); a logits processor stamps a *synced* timestamp per
    decode step so TTFT and decode are separated on the real GPU timeline."""
    from transformers import LogitsProcessorList

    input_ids = build_chat_input_ids(tokenizer, model, prompt)
    attention_mask = torch.ones_like(input_ids)
    timestamps = []

    def _stamp(_input_ids, scores):
        sync_device()                       # ← C-1 fix: real wall-clock, not CPU-ahead
        timestamps.append(time.time())
        return scores

    with PeakMemoryTracker() as mem:
        start = time.time()
        with torch.no_grad():
            model.generate(
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


def warmup_hf(model, tokenizer, prompt):
    """Prefill this exact prompt + a few decode steps, untimed, so the timed
    run below doesn't pay MPS shape-specific kernel compilation."""
    ids = build_chat_input_ids(tokenizer, model, prompt)
    with torch.no_grad():
        model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=4, do_sample=False)
    sync_device()


# --- GGUF timing -------------------------------------------------------------

def measure_gguf_generate(llm, prompt_str, max_new_tokens=MAX_NEW_TOKENS):
    """Single streaming call. Skip the trailing empty finish_reason chunk so
    n_new counts real tokens (M-1 fix)."""
    timestamps = []
    n_new = 0
    start = time.time()
    for chunk in llm(
        prompt_str, max_tokens=max_new_tokens, stream=True,
        temperature=0.0, top_k=1, repeat_penalty=1.0,
    ):
        if chunk["choices"][0]["finish_reason"] is not None:
            break
        timestamps.append(time.time())
        n_new += 1
    ttft = timestamps[0] - start
    decode_time = timestamps[-1] - timestamps[0] if n_new > 1 else float("nan")
    tps = (n_new - 1) / decode_time if n_new > 1 and decode_time > 0 else float("nan")
    return ttft, tps, n_new


def download_gguf(filename_hint: str) -> str:
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(GGUF_REPO)
    candidates = [
        f for f in files
        if filename_hint.lower() in f.lower() and f.endswith(".gguf")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No file containing '{filename_hint}' found in {GGUF_REPO}."
        )
    local_path = hf_hub_download(GGUF_REPO, candidates[0])
    return local_path


# --- one self-contained measurement ------------------------------------------

def measure_one(config, ctx_len, max_new_tokens=MAX_NEW_TOKENS, use_mmap=True):
    """Load the backend for `config` fresh, build the ctx_len prompt, warm up at
    that exact shape (discarded), measure once, tear down. Returns a dict."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if config in HF_CONFIGS:
        if config == "FP16":
            model = load_causal_lm(MODEL_ID, dtype=torch.float16)
        else:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            )
            model = load_causal_lm(MODEL_ID, quantization_config=bnb)

        prompt = make_long_prompt(tokenizer, ctx_len) if ctx_len else PROMPT
        warmup_hf(model, tokenizer, prompt)
        ttft, tps, vram, ptok, nnew = measure_hf_generate(model, tokenizer, prompt, max_new_tokens)
        del model
        gc.collect()
        empty_cache()
        return dict(peak_vram_gb=vram, ttft_s=ttft, tokens_per_sec=tps,
                    prompt_tokens=ptok, n_new=nnew)

    # GGUF
    from llama_cpp import Llama
    hint = GGUF_F16_FILENAME_HINT if config == "GGUF_F16" else GGUF_FILENAME_HINT
    model_path = download_gguf(hint)
    n_ctx = (ctx_len or 0) + max_new_tokens + 64
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx,
                verbose=False, use_mmap=use_mmap)

    filler = make_long_prompt_gguf(llm, ctx_len) if ctx_len else PROMPT
    prompt_str = build_chat_prompt_string(tokenizer, filler)
    ptok = len(llm.tokenize(prompt_str.encode("utf-8"), add_bos=False, special=True))

    # per-shape warmup on the real prompt, then reset so the timed call prefills fresh
    _ = list(llm(prompt_str, max_tokens=4, temperature=0.0, top_k=1, repeat_penalty=1.0))
    llm.reset()

    ttft, tps, nnew = measure_gguf_generate(llm, prompt_str, max_new_tokens)
    del llm
    gc.collect()
    return dict(peak_vram_gb=None, ttft_s=ttft, tokens_per_sec=tps,
                prompt_tokens=ptok, n_new=nnew)


def run_sweep(configs, ctx_lens, repeats=3, max_new_tokens=MAX_NEW_TOKENS,
              results_path=RESULTS_DIR / "results.jsonl"):
    """Measure every (config, ctx) pair `repeats` times in randomized order."""
    tasks = [(c, x) for c in configs for x in ctx_lens]
    for r in range(repeats):
        order = list(tasks)
        random.Random(1000 + r).shuffle(order)   # deterministic, differs per repeat
        for config, ctx_len in order:
            if config in HF_CONFIGS and DEVICE_KIND == "cpu":
                print(f"[{config}] skipped (no GPU backend)")
                continue
            try:
                row = measure_one(config, ctx_len, max_new_tokens)
            except Exception as e:
                print(f"[{config}] ctx={ctx_len} rep={r} failed: {e!r}. Skipping.")
                continue
            log_result(BenchmarkResult(
                config, ctx_len, row["peak_vram_gb"], row["ttft_s"], row["tokens_per_sec"],
                device=DEVICE_KIND, prompt_tokens=row["prompt_tokens"], n_new=row["n_new"],
                repeat=r, max_new_tokens=max_new_tokens,
            ), path=results_path)


def run_perplexity(results_path=RESULTS_DIR / "results_perplexity.jsonl"):
    """Sliding-window perplexity on WikiText-2 (test) for FP16 and NF4 with the
    same dataset/stride/window. Written to its OWN file so the 0.0 speed fields
    don't pollute per-config speed aggregation of results.jsonl."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig

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
    log_result(BenchmarkResult("FP16", None, None, 0.0, 0.0, perplexity=ppl_fp16,
                               device=DEVICE_KIND), path=results_path)
    del model_fp16
    gc.collect()
    empty_cache()

    if DEVICE_KIND != "cpu":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            model_nf4 = load_causal_lm(MODEL_ID, quantization_config=bnb_config)
        except Exception as e:
            model_nf4 = None
            print(f"[NF4] perplexity failed to load on {DEVICE_KIND} ({e!r}). Skipping.")
        if model_nf4 is not None:
            ppl_nf4 = compute_ppl(model_nf4)
            log_result(BenchmarkResult("NF4_4bit", None, None, 0.0, 0.0, perplexity=ppl_nf4,
                                       device=DEVICE_KIND), path=results_path)
            del model_nf4
            gc.collect()
            empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "context_sweep", "perplexity"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    try:
        if args.stage == "all":
            run_sweep(ALL_CONFIGS, [None], repeats=args.repeats)
            run_perplexity()
        elif args.stage == "context_sweep":
            run_sweep(ALL_CONFIGS, CONTEXT_LENGTHS, repeats=args.repeats)
        elif args.stage == "perplexity":
            run_perplexity()
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA out of memory. Try smaller MAX_NEW_TOKENS or a shorter "
            "CONTEXT_LENGTHS list."
        )
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
