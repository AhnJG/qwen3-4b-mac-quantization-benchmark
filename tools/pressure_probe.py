"""
tools/pressure_probe.py

Turns the v1 "Safari was open, throughput collapsed 14-17x" anecdote into
a controlled experiment: hold a fixed amount of system memory via a
ballast subprocess (tools/ballast.py), then measure FP16 / NF4 / GGUF
Q4_K_M (mmap on and off) throughput at a short and a long context length
under each pressure level, with `vm_stat` sampled concurrently so paging
activity -- not just a swap-usage snapshot -- is on the record.

`use_mmap=False` is a direct test of the v1 hypothesis that GGUF's mmap
loading is what makes it more robust under memory pressure: if that's the
real mechanism, disabling mmap should make GGUF collapse like the HF path
does; if it doesn't, the mechanism is something else (fragmentation,
allocator behavior, etc. -- see the external reviews in notes/).

Writes results/results_memory_pressure.jsonl and results/vm_stat_<gb>gb.log.

Usage:
    python tools/pressure_probe.py --ballast-gb 0,16 --context-lengths 128,4096
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import benchmark_quantization as bq  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results" / "results_memory_pressure.jsonl"
PROBE_MAX_NEW_TOKENS = 40  # shorter than the main benchmark's 150, to bound
                            # worst-case runtime under heavy pressure (v1 saw
                            # 0.1-0.3 tok/s collapses -- 150 tokens there would
                            # take 10-25 minutes *per measurement*)


def log(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


def start_ballast(gb):
    if gb <= 0:
        return None
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "tools" / "ballast.py"), "--gb", str(gb)],
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
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_vmstat_logging(tag):
    log_path = REPO_ROOT / "results" / f"vm_stat_{tag}.log"
    f = open(log_path, "w")
    proc = subprocess.Popen(["vm_stat", "1"], stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc, f


def stop_vmstat_logging(proc, f):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    f.close()


def measure_hf(config_name, ctx_lens, ballast_gb, quantization_config=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bq.MODEL_ID)
    if quantization_config is None:
        model = bq.load_causal_lm(bq.MODEL_ID, dtype=torch.float16)
    else:
        model = bq.load_causal_lm(bq.MODEL_ID, quantization_config=quantization_config)
    bq.warmup_hf_generate(model, tokenizer)

    for ctx_len in ctx_lens:
        prompt = bq.make_long_prompt(tokenizer, ctx_len) if ctx_len else bq.PROMPT
        ttft, tps, vram, prompt_tokens, n_new = bq.measure_hf_generate(
            model, tokenizer, prompt, max_new_tokens=PROBE_MAX_NEW_TOKENS,
        )
        log({
            "config": config_name, "ballast_gb": ballast_gb, "context_len": ctx_len,
            "peak_vram_gb": vram, "ttft_s": ttft, "tokens_per_sec": tps,
            "prompt_tokens": prompt_tokens, "n_new": n_new, "use_mmap": None,
        })

    del model
    gc.collect()
    bq.empty_cache()


def measure_gguf(config_name, ctx_lens, ballast_gb, use_mmap):
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(bq.MODEL_ID)
    model_path = bq.download_gguf(bq.GGUF_FILENAME_HINT)

    for ctx_len in ctx_lens:
        n_ctx = (ctx_len or 0) + PROBE_MAX_NEW_TOKENS + 64
        llm = Llama(
            model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False,
            use_mmap=use_mmap,
        )
        filler = bq.make_long_prompt_gguf(llm, ctx_len) if ctx_len else bq.PROMPT
        prompt_str = bq.build_chat_prompt_string(tokenizer, filler)
        prompt_tokens = len(llm.tokenize(prompt_str.encode("utf-8"), add_bos=False, special=True))

        _ = list(llm("Warm up.", max_tokens=4, temperature=0.0))
        llm.reset()

        timestamps, n_new = [], 0
        start = time.time()
        for _chunk in llm(
            prompt_str, max_tokens=PROBE_MAX_NEW_TOKENS, stream=True,
            temperature=0.0, top_k=1, repeat_penalty=1.0,
        ):
            timestamps.append(time.time())
            n_new += 1

        ttft = timestamps[0] - start
        decode_time = timestamps[-1] - timestamps[0] if n_new > 1 else float("nan")
        tps = (n_new - 1) / decode_time if n_new > 1 and decode_time > 0 else float("nan")

        log({
            "config": config_name, "ballast_gb": ballast_gb, "context_len": ctx_len,
            "peak_vram_gb": None, "ttft_s": ttft, "tokens_per_sec": tps,
            "prompt_tokens": prompt_tokens, "n_new": n_new, "use_mmap": use_mmap,
        })
        del llm
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballast-gb", default="0,16")
    parser.add_argument("--context-lengths", default="128,4096")
    args = parser.parse_args()

    ballast_levels = [float(x) for x in args.ballast_gb.split(",")]
    ctx_lens = [int(x) for x in args.context_lengths.split(",")]

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )

    for gb in ballast_levels:
        print(f"\n=== ballast={gb}GB ===", flush=True)
        ballast_proc = start_ballast(gb)
        time.sleep(3)
        vmstat_proc, vmstat_f = start_vmstat_logging(f"{gb}gb")
        try:
            measure_hf("FP16", ctx_lens, gb)
            measure_hf("NF4_4bit", ctx_lens, gb, quantization_config=bnb_config)
            measure_gguf("GGUF_Q4_K_M", ctx_lens, gb, use_mmap=True)
            measure_gguf("GGUF_Q4_K_M_nommap", ctx_lens, gb, use_mmap=False)
        finally:
            stop_vmstat_logging(vmstat_proc, vmstat_f)
            stop_ballast(ballast_proc)
        time.sleep(2)

    print(f"\nAll results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
