"""
tools/pressure_probe.py

Controlled memory-pressure experiment. Holds a fixed amount of resident,
incompressible memory via tools/ballast.py, then measures each config's decode
throughput at a short and a long context under each pressure level, in
randomized order, while `vm_stat` is logged concurrently. For each measurement
it also records the system swap-in-use at that moment, so the *effective*
pressure is on record rather than assumed.

Design points (addressing the v2 review):
  * Uses benchmark_quantization.measure_one -- the SAME fixed measurement path
    as the main benchmark (synced HF timing, real-token GGUF counting), so
    pressure numbers are comparable to the clean-run numbers.
  * GGUF is measured with use_mmap on AND off -- the direct test of the "mmap
    is why GGUF survives pressure" hypothesis.
  * GGUF F16 is included (8GB weights -- the config most likely to be forced
    into swap on a 24GB machine).
  * Graduated ballast levels (default 0,8,16) as a coarse positive-control
    sweep -- to find whether/where GGUF actually degrades, not just assume it.
  * Config order is shuffled per (level) so "GGUF measured last, after the
    system settled" can't be confounded with "GGUF is robust".

Caveat that the code cannot fix: a truly clean 0GB baseline needs a reboot
(this session accumulates swap over hours). The recorded per-measurement swap
figure makes that contamination visible instead of hidden.

Usage:
    python tools/pressure_probe.py --ballast-gb 0,8,16 --context-lengths 128,4096
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import benchmark_quantization as bq  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results" / "results_memory_pressure.jsonl"
PROBE_MAX_NEW_TOKENS = 40   # bound worst-case runtime when a config collapses

# (label, base config for measure_one, use_mmap)
PROBE_CONFIGS = [
    ("FP16", "FP16", None),
    ("NF4_4bit", "NF4_4bit", None),
    ("GGUF_Q4_K_M", "GGUF_Q4_K_M", True),
    ("GGUF_Q4_K_M_nommap", "GGUF_Q4_K_M", False),
    ("GGUF_F16", "GGUF_F16", True),
]


def log(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


def swap_used_gb():
    """Current system swap in use, in GB (macOS `sysctl vm.swapusage`)."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        # e.g. "total = 6144.00M  used = 5286.38M  free = 857.62M  (encrypted)"
        used = [tok for tok in out.split() if tok.endswith("M")][1]
        return round(float(used[:-1]) / 1024, 2)
    except Exception:
        return None


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
        proc.wait(timeout=20)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ballast-gb", default="0,8,16")
    parser.add_argument("--context-lengths", default="128,4096")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    ballast_levels = [float(x) for x in args.ballast_gb.split(",")]
    ctx_lens = [int(x) for x in args.context_lengths.split(",")]

    for gb in ballast_levels:
        print(f"\n=== ballast={gb}GB (swap now {swap_used_gb()}GB) ===", flush=True)
        ballast_proc = start_ballast(gb)
        time.sleep(5)   # let the fill settle before we start timing
        vmstat_proc, vmstat_f = start_vmstat_logging(f"{gb}gb")
        swap_at_level_start = swap_used_gb()
        try:
            tasks = [(label, base, mmap, ctx)
                     for (label, base, mmap) in PROBE_CONFIGS
                     for ctx in ctx_lens]
            for rep in range(args.repeats):
                order = list(tasks)
                random.Random(500 + rep).shuffle(order)
                for label, base, mmap, ctx in order:
                    use_mmap = True if mmap is None else mmap
                    swap_before = swap_used_gb()
                    try:
                        row = bq.measure_one(base, ctx, max_new_tokens=PROBE_MAX_NEW_TOKENS,
                                             use_mmap=use_mmap)
                    except Exception as e:
                        print(f"[{label}] ctx={ctx} gb={gb} failed: {e!r}")
                        continue
                    log({
                        "config": label, "ballast_gb": gb, "context_len": ctx,
                        "repeat": rep, "peak_vram_gb": row["peak_vram_gb"],
                        "ttft_s": row["ttft_s"], "tokens_per_sec": row["tokens_per_sec"],
                        "prompt_tokens": row["prompt_tokens"], "n_new": row["n_new"],
                        "use_mmap": mmap, "max_new_tokens": PROBE_MAX_NEW_TOKENS,
                        "swap_used_gb": swap_before,
                    })
        finally:
            swap_at_level_end = swap_used_gb()
            stop_vmstat_logging(vmstat_proc, vmstat_f)
            stop_ballast(ballast_proc)
            print(f"[level {gb}GB] swap start={swap_at_level_start}GB end={swap_at_level_end}GB", flush=True)
        time.sleep(3)

    print(f"\nAll results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
