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
WikiText-2 perplexity (all four configs, same scorer). It also sweeps prompt length 128..4096.

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
  * The (config x context x repeat) task list is shuffled GLOBALLY and measured
    in that one order; report the median and spread rather than a single run.
    v5 ran repeats as sequential blocks, which turned out to bias every median:
    a multi-hour sweep steadily saturates this machine's memory, so throughput
    drifts down with elapsed time and that drift landed entirely on the last
    repeat (v3: rep2 -21% vs rep0; v5: -25%). Shuffling repeats together with
    everything else spreads the drift across all cells, where the median
    absorbs it instead of inheriting it. Each row now records started_at /
    ended_at / swap_used_gb so the drift is measurable after the fact.

Usage:
    # caffeinate -i prevents system sleep -- the v5 run slept ~6 of its 8 hours,
    # and wall-clock timing cannot distinguish that from a slow measurement.
    caffeinate -i python benchmark_quantization.py --stage context_sweep [--repeats 3]
    caffeinate -i python benchmark_quantization.py --stage all [--repeats 3]
    python benchmark_quantization.py --stage perplexity
"""

import argparse
import dataclasses
import datetime
import gc
import json
import random
import subprocess
import sys
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
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    swap_used_gb: Optional[float] = None
    ballast_gb: Optional[float] = None
    step_ms_p50: Optional[float] = None      # see step_stats(): p50 is the
    step_ms_p90: Optional[float] = None      # steady-state cost, tokens_per_sec
    step_ms_max: Optional[float] = None      # is the mean, and they disagree
    step_ms: Optional[list] = None           # full per-step array (v8+)
    schema: Optional[str] = None             # rows without it predate step_ms
    label: Optional[str] = None              # free-text run condition (v8.5+)
    mem_free_gb_start: Optional[float] = None    # vm_stat free+inactive+speculative
    mem_free_gb_end: Optional[float] = None
    mem_compressed_gb_end: Optional[float] = None
    memorystatus_level_end: Optional[int] = None  # sysctl kern.memorystatus_level (% free)

    def to_dict(self):
        return dataclasses.asdict(self)


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def swap_used_gb():
    """Current system swap in use, in GB (macOS `sysctl vm.swapusage`).

    Recorded per row so the *effective* memory pressure of a run is on record
    rather than assumed. (tools/pressure_probe.py has its own copy.)
    """
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        # e.g. "total = 6144.00M  used = 5286.38M  free = 857.62M  (encrypted)"
        used = [tok for tok in out.split() if tok.endswith("M")][1]
        return round(float(used[:-1]) / 1024, 2)
    except Exception:
        return None


def mem_snapshot():
    """Read-only memory snapshot (macOS `vm_stat` and `sysctl`). Returns a dict
    of GB values and the kernel's free-memory percentage; None where unavailable.
    Nothing here applies pressure -- it only reads counters."""
    snap = dict(free_gb=None, compressed_gb=None, level=None)
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = int(out.split("page size of")[1].split()[0])
        pages = {}
        for line in out.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                pages[k.strip()] = int(v.strip().rstrip("."))
        free = sum(pages.get(k, 0) for k in
                   ("Pages free", "Pages inactive", "Pages speculative"))
        snap["free_gb"] = round(free * page / 1024**3, 2)
        snap["compressed_gb"] = round(
            pages.get("Pages occupied by compressor", 0) * page / 1024**3, 2)
    except Exception:
        pass
    try:
        snap["level"] = int(subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_level"], text=True).strip())
    except Exception:
        pass
    return snap


def total_ram_gb():
    """Physical RAM in GB (macOS `sysctl hw.memsize`)."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return round(int(out.strip()) / 1024**3)
    except Exception:
        return None


def start_ballast(gb):
    """Hold `gb` GB of resident, incompressible memory for the whole run, to
    benchmark a *smaller machine class* than the host: this M3 Air has 24GB, and
    the same chip also ships as 16GB and 8GB with identical bandwidth and core
    counts, so subtracting capacity is a faithful simulation of those configs
    (8GB ballast ~= a 16GB Mac). Blocks until the fill is resident.
    (tools/pressure_probe.py has its own copy.)
    """
    if gb <= 0:
        return None
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve().parent / "tools" / "ballast.py"),
         "--gb", str(gb)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    while True:
        line = proc.stdout.readline()
        if not line or "filled" in line:
            break
    return proc


def stop_ballast(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


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

def step_stats(timestamps):
    """Per-decode-step latency quantiles, in ms.

    `tokens_per_sec` is n/total, i.e. a MEAN, and this path turns out to stall
    for whole seconds on a minority of steps -- so the mean reports the stalls,
    not the kernel. p50 is the steady-state cost (what a kernel comparison
    wants); the gap between 1000/p50 and tokens_per_sec is the stall tax (what
    a user actually waits through). Recording both is what makes the FP16
    @2048 "cliff" auditable instead of arguable: most of the cliff lives in
    the mean; p50 rises far less with context length.
    """
    gaps = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
    if not gaps:
        return dict(step_ms_p50=None, step_ms_p90=None, step_ms_max=None,
                    step_ms=None)
    ordered = sorted(gaps)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return dict(step_ms_p50=round(pick(0.50), 2),
                step_ms_p90=round(pick(0.90), 2),
                step_ms_max=round(ordered[-1], 2),
                # the whole array, in generation order. Three summary values
                # turned out not to be enough: with only p50/p90/max on record
                # there is no way to ask, after the fact, whether a cell's
                # spread is sampling noise or structure -- which is how v7 came
                # to lean on a min-over-repeats estimator that a later review
                # retracted as biased. ~1.5KB/row buys every other estimator
                # (trimmed mean, low percentiles, bootstrap CIs) forever.
                step_ms=[round(g, 2) for g in gaps])


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
    return ttft, tps, vram, input_ids.shape[1], n_new, step_stats(timestamps)


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
    return ttft, tps, n_new, step_stats(timestamps)


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
        ttft, tps, vram, ptok, nnew, steps = measure_hf_generate(
            model, tokenizer, prompt, max_new_tokens)
        del model
        gc.collect()
        empty_cache()
        return dict(peak_vram_gb=vram, ttft_s=ttft, tokens_per_sec=tps,
                    prompt_tokens=ptok, n_new=nnew, **steps)

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

    ttft, tps, nnew, steps = measure_gguf_generate(llm, prompt_str, max_new_tokens)
    del llm
    gc.collect()
    return dict(peak_vram_gb=None, ttft_s=ttft, tokens_per_sec=tps,
                prompt_tokens=ptok, n_new=nnew, **steps)


def run_sweep(configs, ctx_lens, repeats=3, max_new_tokens=MAX_NEW_TOKENS,
              results_path=RESULTS_DIR / "results.jsonl", ballast_gb=None,
              seed=1000, label=None):
    """Measure every (config, ctx) pair `repeats` times in one globally shuffled
    order, so that run-order drift (this machine saturates its memory over a
    multi-hour sweep) is spread across all cells rather than concentrated in the
    last repeat -- see the module docstring."""
    tasks = [(c, x, r) for r in range(repeats) for c in configs for x in ctx_lens]
    random.Random(seed).shuffle(tasks)   # deterministic (default seed 1000 = v7 order)
    for config, ctx_len, r in tasks:
        if config in HF_CONFIGS and DEVICE_KIND == "cpu":
            print(f"[{config}] skipped (no GPU backend)")
            continue
        started_at = now_iso()
        mem0 = mem_snapshot()
        try:
            row = measure_one(config, ctx_len, max_new_tokens)
        except Exception as e:
            print(f"[{config}] ctx={ctx_len} rep={r} failed: {e!r}. Skipping.")
            continue
        mem1 = mem_snapshot()
        log_result(BenchmarkResult(
            config, ctx_len, row["peak_vram_gb"], row["ttft_s"], row["tokens_per_sec"],
            device=DEVICE_KIND, prompt_tokens=row["prompt_tokens"], n_new=row["n_new"],
            repeat=r, max_new_tokens=max_new_tokens,
            started_at=started_at, ended_at=now_iso(), swap_used_gb=swap_used_gb(),
            ballast_gb=ballast_gb,
            step_ms_p50=row["step_ms_p50"], step_ms_p90=row["step_ms_p90"],
            step_ms_max=row["step_ms_max"], step_ms=row.get("step_ms"),
            schema="v8.5", label=label,
            mem_free_gb_start=mem0["free_gb"], mem_free_gb_end=mem1["free_gb"],
            mem_compressed_gb_end=mem1["compressed_gb"],
            memorystatus_level_end=mem1["level"],
        ), path=results_path)


def _nll_from_logprobs(logprobs, window, trg_len):
    """Sum NLL of the last `trg_len` targets of `window`, given per-position
    log-probabilities. Target w[t] is predicted from position t-1, so the very
    first token of the corpus is never scored. Returns (nll_sum, n_scored)."""
    first = max(1, len(window) - trg_len)
    nll = -sum(float(logprobs[t - 1][window[t]]) for t in range(first, len(window)))
    return nll, len(window) - first


def perplexity_windows(ids, max_length=1024, stride=512):
    """The sliding-window schedule, yielded as (window_tokens, trg_len) so every
    backend scores exactly the same targets."""
    prev_end = 0
    for begin in range(0, len(ids), stride):
        end = min(begin + max_length, len(ids))
        yield ids[begin:end], end - prev_end
        prev_end = end
        if end == len(ids):
            break


def compute_ppl_hf(model, ids, log_every=100):
    import torch.nn.functional as F

    total_nll, total_n = 0.0, 0
    for i, (window, trg_len) in enumerate(perplexity_windows(ids)):
        t = torch.tensor([window], device=model.device)
        with torch.no_grad():
            logits = model(t).logits[0].float()
        nll, n = _nll_from_logprobs(F.log_softmax(logits, dim=-1), window, trg_len)
        total_nll += nll
        total_n += n
        if log_every and i % log_every == 0:
            print(f"    window {i} · running ppl {torch.tensor(total_nll / total_n).exp():.3f}", flush=True)
    return float(torch.tensor(total_nll / total_n).exp())


def compute_ppl_gguf(llm, ids, log_every=100):
    """Same schedule, same targets, scored by llama.cpp. Requires the Llama to
    be built with logits_all=True so every position has logits."""
    import numpy as np

    total_nll, total_n = 0.0, 0
    for i, (window, trg_len) in enumerate(perplexity_windows(ids)):
        llm.reset()
        llm.eval(list(window))
        scores = np.array(llm._scores, dtype=np.float32)[-len(window):]
        logprobs = scores - scores.max(-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(-1, keepdims=True))
        nll, n = _nll_from_logprobs(logprobs, window, trg_len)
        total_nll += nll
        total_n += n
        if log_every and i % log_every == 0:
            print(f"    window {i} · running ppl {np.exp(total_nll / total_n):.3f}", flush=True)
    return float(np.exp(total_nll / total_n))


def run_perplexity(results_path=RESULTS_DIR / "results_perplexity_v7.jsonl",
                   max_tokens=None):
    """Sliding-window perplexity on WikiText-2 (test) for ALL FOUR configs.

    v1-v6 measured this for the HF arm only and declared GGUF "not directly
    comparable" -- which left the entire llama.cpp arm, where the 2.8x speedup
    lives, with no quality number at all. It is comparable if you make it so:
    every config here scores the SAME token id sequence (HF tokenizer), on the
    same window schedule, through the same _nll_from_logprobs(). What made the
    old numbers incomparable was using llama.cpp's own perplexity tool, not
    anything intrinsic.

    Written to its own file (speed fields are 0.0 here and would pollute
    per-config speed aggregation). The v1 HF numbers are kept in
    results_perplexity.jsonl -- they used HF's internal loss reduction rather
    than this scorer, so they are NOT directly comparable to these.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tokenizer(text).input_ids
    if max_tokens:
        ids = ids[:max_tokens]
    print(f"[ppl] WikiText-2 test: {len(ids)} tokens, "
          f"{sum(1 for _ in perplexity_windows(ids))} windows", flush=True)

    def record(config, ppl):
        print(f"[ppl] {config}: {ppl:.4f}", flush=True)
        log_result(BenchmarkResult(config, None, None, 0.0, 0.0, perplexity=ppl,
                                   device=DEVICE_KIND, prompt_tokens=len(ids)),
                   path=results_path)

    model = load_causal_lm(MODEL_ID, dtype=torch.float16)
    record("FP16", compute_ppl_hf(model, ids))
    del model
    gc.collect()
    empty_cache()

    if DEVICE_KIND != "cpu":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            model = load_causal_lm(MODEL_ID, quantization_config=bnb_config)
        except Exception as e:
            print(f"[NF4] perplexity failed to load on {DEVICE_KIND} ({e!r}). Skipping.")
        else:
            record("NF4_4bit", compute_ppl_hf(model, ids))
            del model
            gc.collect()
            empty_cache()

    from llama_cpp import Llama
    for config, hint in [("GGUF_F16", GGUF_F16_FILENAME_HINT),
                         ("GGUF_Q4_K_M", GGUF_FILENAME_HINT)]:
        llm = Llama(model_path=download_gguf(hint), n_gpu_layers=-1, n_ctx=1024,
                    logits_all=True, verbose=False)
        record(config, compute_ppl_gguf(llm, ids))
        del llm
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "context_sweep", "perplexity"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ppl-max-tokens", type=int, default=None,
                        help="Cap the perplexity corpus (for quick checks; "
                             "full WikiText-2 test takes hours across 4 configs)")
    parser.add_argument(
        "--out", default=str(RESULTS_DIR),
        help="Folder for the result files (default: results/, the audited data). "
             "Use a separate folder for test runs, e.g. --out results/smoke, "
             "so they are not appended to the audited results.jsonl.")
    parser.add_argument(
        "--ballast-gb", type=float, default=0.0,
        help="Hold this much resident memory for the whole run to benchmark a "
             "smaller machine class than the host (on this 24GB M3 Air, 8 "
             "simulates a 16GB Mac and 16 an 8GB one). Recorded per row.",
    )
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS),
                        help="Comma-separated subset of " + ",".join(ALL_CONFIGS))
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXT_LENGTHS)),
                        help="Comma-separated input lengths for context_sweep")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Shuffle seed for the task order (1000 = the v7 order)")
    parser.add_argument("--label", default=None,
                        help="Free-text run condition stored in every row, e.g. apps_closed")
    args = parser.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in configs if c not in ALL_CONFIGS]
    if bad:
        parser.error(f"unknown config(s): {bad}")
    ctx_lens = [int(x) for x in args.contexts.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "results.jsonl"
    ppl_path = out_dir / "results_perplexity_v7.jsonl"

    ballast = start_ballast(args.ballast_gb)
    if ballast is not None:
        print(f"[ballast] holding {args.ballast_gb}GB "
              f"(~{total_ram_gb() - args.ballast_gb:.0f}GB effective machine)",
              flush=True)
    try:
        if args.stage == "all":
            run_sweep(ALL_CONFIGS, [None], repeats=args.repeats,
                      results_path=sweep_path, ballast_gb=args.ballast_gb or None)
            run_perplexity(results_path=ppl_path, max_tokens=args.ppl_max_tokens)
        elif args.stage == "context_sweep":
            run_sweep(configs, ctx_lens, repeats=args.repeats,
                      results_path=sweep_path, ballast_gb=args.ballast_gb or None,
                      seed=args.seed, label=args.label)
        elif args.stage == "perplexity":
            run_perplexity(results_path=ppl_path, max_tokens=args.ppl_max_tokens)
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA out of memory. Try smaller MAX_NEW_TOKENS or a shorter "
            "CONTEXT_LENGTHS list."
        )
        raise
    finally:
        stop_ballast(ballast)   # release even on Ctrl-C or a crash

    print("\nDone.")


if __name__ == "__main__":
    main()
