"""
tools/recheck_summary.py

Summarise the 2048-token stall recheck (notes: recheck plan, 2026-10-16) and apply
the pass/fail rules that were fixed BEFORE the run. Reads <out>/phase1 and
<out>/phase2 (optional), compares with the v7 data in results/results.jsonl.

Usage:
    python tools/recheck_summary.py results/recheck-YYYYMMDD
"""

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import e2e_seconds  # noqa: E402

V7 = "results/results.jsonl"
CELLS = [("FP16", 1024), ("FP16", 2048), ("GGUF_Q4_K_M", 1024), ("GGUF_Q4_K_M", 2048)]
MEM = {0.0: "24GB", 8.0: "16GB"}

# rules fixed before the run (2026-10-16)
H1_REPRO, H1_GONE = 2.0, 1.2            # stall tax, 24GB FP16@2048, phase 1 median
H2_STRONG_MIN, V7_LO_MIN, V7_HI_MIN = 2.0, 4.4, 7.7   # minutes, 16GB FP16@2048 e2e
CTRL_TAX, CTRL_TOL = 1.05, 0.20         # control: Q4_K_M@2048


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def tax(r):
    return (1000 / r["step_ms_p50"]) / r["tokens_per_sec"]


def cell(rows, mem, cfg, ctx):
    return [r for r in rows if (r.get("ballast_gb") or 0.0) == mem
            and r["config"] == cfg and r["context_len"] == ctx]


def med(v):
    return st.median(v) if v else float("nan")


def table(name, rows):
    print(f"\n== {name} ({len(rows)} rows)")
    print(f"{'memory':6} {'cell':18} {'n':>2} {'stall tax per run':30} {'tax med':>7} {'e2e med':>9}")
    for mem in MEM:
        for cfg, ctx in CELLS:
            rs = cell(rows, mem, cfg, ctx)
            if not rs:
                continue
            t = [tax(r) for r in rs]
            e = [e2e_seconds(r) for r in rs]
            print(f"{MEM[mem]:6} {cfg + '@' + str(ctx):18} {len(rs):>2} "
                  f"{str([round(x, 2) for x in t]):30} {med(t):7.2f} {med(e):8.1f}s")


def main():
    out = sys.argv[1]
    v7 = load(V7)
    p1 = load(os.path.join(out, "phase1", "results.jsonl"))
    p2 = load(os.path.join(out, "phase2", "results.jsonl"))
    table("v7 (2026-08-09, apps open, conditions run one after the other)", v7)
    table("phase 1: apps closed", p1)
    if p2:
        table("phase 2: fixed apps open (Chrome 5 tabs + Claude app)", p2)

    print("\n== Verdicts (rules fixed before the run)")
    ctrl_ok = True
    for mem in MEM:
        rs = cell(p1, mem, "GGUF_Q4_K_M", 2048)
        base = med([e2e_seconds(r) for r in cell(v7, mem, "GGUF_Q4_K_M", 2048)])
        if not rs:
            print(f"control {MEM[mem]}: no data")
            ctrl_ok = False
            continue
        t, e = med([tax(r) for r in rs]), med([e2e_seconds(r) for r in rs])
        ok = t <= CTRL_TAX and abs(e / base - 1) <= CTRL_TOL
        ctrl_ok &= ok
        print(f"control {MEM[mem]} Q4_K_M@2048: tax {t:.2f}, e2e {e:.1f}s vs v7 {base:.1f}s "
              f"({e / base - 1:+.0%}) -> {'OK' if ok else 'FAILED'}")
    if not ctrl_ok:
        print("Control failed: do not compare with v7. Compare phase 1 and phase 2 of the same day only.")

    rs = cell(p1, 0.0, "FP16", 2048)
    if rs:
        t = med([tax(r) for r in rs])
        v = ("REPRODUCED (stalls remain with apps closed)" if t >= H1_REPRO else
             "GONE (stalls appeared only with apps open)" if t <= H1_GONE else "UNDECIDED")
        print(f"H1 24GB FP16@2048 stall tax median {t:.2f} -> {v}")
    rs = cell(p1, 8.0, "FP16", 2048)
    if rs:
        m = med([e2e_seconds(r) for r in rs]) / 60
        v = ("STRONG background effect (< 2 min)" if m < H2_STRONG_MIN else
             "UNCHANGED (inside v7 range 4.4-7.7 min)" if V7_LO_MIN <= m <= V7_HI_MIN else
             "PARTIAL (outside v7 range, not below 2 min)")
        print(f"H2 16GB FP16@2048 e2e median {m:.1f} min -> {v}")
    if not ctrl_ok:
        print("(H1/H2 verdicts compare against v7 and are on hold because the control failed.)")
    if p2:
        print("\n== Phase 2 vs phase 1 (observation only, no verdict)")
        for mem in MEM:
            a, b = cell(p1, mem, "FP16", 2048), cell(p2, mem, "FP16", 2048)
            if a and b:
                print(f"{MEM[mem]} FP16@2048: tax {med([tax(r) for r in a]):.2f} -> "
                      f"{med([tax(r) for r in b]):.2f}, e2e {med([e2e_seconds(r) for r in a]):.0f}s -> "
                      f"{med([e2e_seconds(r) for r in b]):.0f}s")


if __name__ == "__main__":
    main()
