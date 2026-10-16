"""
tools/recheck_summary.py

Summarise the 2048-token stall recheck and apply the rules that were fixed
BEFORE the run (2026-10-16; revised the same day after an independent review,
still before any measurement). Reads <out>/phase1 and <out>/phase2 (optional)
and compares with the v7 data in results/results.jsonl.

Rules
  Same-day soundness  Q4_K_M@2048 stall tax <= 1.05 at that memory size.
                      Needed before any verdict at that memory size.
  v7 comparability    Q4_K_M@2048 e2e within +-20% of v7 at that memory size.
                      Needed only for statements that compare with v7. Even then,
                      v7 differs in apps, session length and swap, so a v7
                      difference is never attributed to apps alone.
  H1 (24GB FP16@2048 stall tax, median of >= 3 runs per phase)
                      >= 2.0 remains, <= 1.2 gone, otherwise undecided.
      phase 1 remains                     -> stalls are not caused by the apps
      phase 1 gone and phase 2 remains    -> apps contribute
      phase 1 gone and phase 2 not remain -> session state (reboot, fresh swap);
                                             not attributable to apps
  H2 (16GB FP16@2048 e2e median, phase 1, vs v7 range 4.4-7.7 min)
                      < 2 min large drop, 4.4-7.7 inside, > 7.7 longer, else partial drop.
      app attribution only from the same day: phase2/phase1 e2e >= 1.5 apps
      contribute, <= 1.1 no app effect, otherwise unclear.
  Cells with fewer than 3 valid runs are reported as insufficient.

Usage:
    python tools/recheck_summary.py results/recheck-YYYYMMDD-HHMM
"""

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import e2e_seconds  # noqa: E402

V7 = "results/results.jsonl"
CELLS = [("FP16", 1024), ("FP16", 2048), ("GGUF_Q4_K_M", 1024), ("GGUF_Q4_K_M", 2048)]
MEM = {0.0: "24GB", 8.0: "16GB"}
N_MIN = 3
SOUND_TAX, V7_TOL = 1.05, 0.20
H1_REMAIN, H1_GONE = 2.0, 1.2
H2_LARGE, V7_LO, V7_HI = 2.0, 4.4, 7.7          # minutes
APP_YES, APP_NO = 1.5, 1.1                       # phase2 / phase1 e2e


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def valid(r):
    tps, p50 = r.get("tokens_per_sec"), r.get("step_ms_p50")
    return (tps is not None and p50 is not None and isinstance(tps, (int, float))
            and not math.isnan(tps) and tps > 0 and p50 > 0)


def tax(r):
    return (1000 / r["step_ms_p50"]) / r["tokens_per_sec"]


def cell(rows, mem, cfg, ctx):
    return [r for r in rows if (r.get("ballast_gb") or 0.0) == mem and r["config"] == cfg
            and r["context_len"] == ctx and valid(r)]


def med(v):
    return st.median(v) if v else float("nan")


def stat(rows, mem, cfg, ctx):
    rs = cell(rows, mem, cfg, ctx)
    return len(rs), med([tax(r) for r in rs]), med([e2e_seconds(r) for r in rs])


def table(name, rows):
    print(f"\n== {name} ({len(rows)} rows)")
    print(f"{'memory':6} {'cell':18} {'n':>2} {'stall tax per run':32} {'tax med':>7} {'e2e med':>9}")
    for mem in MEM:
        for cfg, ctx in CELLS:
            rs = cell(rows, mem, cfg, ctx)
            if rs:
                t = [tax(r) for r in rs]
                e = [e2e_seconds(r) for r in rs]
                print(f"{MEM[mem]:6} {cfg + '@' + str(ctx):18} {len(rs):>2} "
                      f"{str([round(x, 2) for x in t]):32} {med(t):7.2f} {med(e):8.1f}s")


def h1_state(n, t):
    if n < N_MIN:
        return "insufficient"
    return "remains" if t >= H1_REMAIN else "gone" if t <= H1_GONE else "undecided"


def main():
    out = sys.argv[1]
    v7 = load(V7)
    p1 = load(os.path.join(out, "phase1", "results.jsonl"))
    p2 = load(os.path.join(out, "phase2", "results.jsonl"))
    table("v7 (2026-08-09: apps open, 4-hour session, conditions run one after the other)", v7)
    table("phase 1: apps closed (Terminal only)", p1)
    if p2:
        table("phase 2: Chrome 5 tabs + Claude app open, same day, after phase 1", p2)

    print("\n== Controls")
    sound, comparable = {}, {}
    for mem in MEM:
        for name, rows in (("phase1", p1), ("phase2", p2)):
            if not rows:
                continue
            n, t, e = stat(rows, mem, "GGUF_Q4_K_M", 2048)
            ok = n >= N_MIN and t <= SOUND_TAX
            sound[(name, mem)] = ok
            print(f"same-day soundness {name} {MEM[mem]}: n={n} tax {t:.2f} -> {'OK' if ok else 'NOT OK'}")
        n, t, e = stat(p1, mem, "GGUF_Q4_K_M", 2048)
        _, _, base = stat(v7, mem, "GGUF_Q4_K_M", 2048)
        comparable[mem] = n >= N_MIN and abs(e / base - 1) <= V7_TOL
        print(f"v7 comparability {MEM[mem]}: e2e {e:.1f}s vs v7 {base:.1f}s ({e / base - 1:+.0%}) -> "
              f"{'OK' if comparable[mem] else 'NOT OK (do not compare this memory size with v7)'}")

    print("\n== H1: 24GB FP16@2048 stall")
    n1, t1, _ = stat(p1, 0.0, "FP16", 2048)
    s1 = h1_state(n1, t1) if sound.get(("phase1", 0.0)) else "on hold (control not sound)"
    print(f"phase 1: n={n1} stall tax {t1:.2f} -> {s1}")
    s2 = None
    if p2:
        n2, t2, _ = stat(p2, 0.0, "FP16", 2048)
        s2 = h1_state(n2, t2) if sound.get(("phase2", 0.0)) else "on hold (control not sound)"
        print(f"phase 2: n={n2} stall tax {t2:.2f} -> {s2}")
    if s1 == "remains":
        print("Verdict: stalls remain with apps closed; they are not caused by the background apps.")
    elif s1 == "gone" and s2 == "remains":
        print("Verdict: stalls appear only with the apps open; the apps contribute.")
    elif s1 == "gone" and s2 in ("gone", "undecided", "insufficient"):
        print("Verdict: stalls are gone in both phases; likely session state (fresh reboot, low swap), "
              "not attributable to the apps.")
    elif s1 == "gone":
        print("Verdict: stalls gone with apps closed; no phase 2 to attribute the cause.")
    else:
        print("Verdict: undecided.")

    print("\n== H2: 16GB FP16@2048 end-to-end time")
    n, _, e = stat(p1, 8.0, "FP16", 2048)
    m = e / 60
    if n < N_MIN or not sound.get(("phase1", 8.0)):
        print(f"phase 1: n={n} -> insufficient or control not sound")
    else:
        v = ("large drop (< 2 min)" if m < H2_LARGE else
             "inside the v7 range (4.4-7.7 min)" if V7_LO <= m <= V7_HI else
             "longer than the v7 range" if m > V7_HI else "partial drop (2-4.4 min)")
        note = "" if comparable[8.0] else " [v7 comparison not valid: control]"
        print(f"phase 1: n={n} e2e median {m:.1f} min -> {v}{note}")
        print("  (v7 differs in apps, session length and swap; do not attribute this difference to apps alone)")
    if p2:
        n2, _, e2 = stat(p2, 8.0, "FP16", 2048)
        if n >= N_MIN and n2 >= N_MIN and sound.get(("phase1", 8.0)) and sound.get(("phase2", 8.0)):
            r = e2 / e
            v = ("apps contribute" if r >= APP_YES else "no app effect" if r <= APP_NO else "unclear")
            print(f"same-day: phase 2 / phase 1 e2e = {e2:.0f}s / {e:.0f}s = {r:.2f} -> {v}")
        else:
            print("same-day comparison: insufficient data or control not sound")

    print("\n== Drift check (phase 1 vs phase 2, light cells)")
    for mem in MEM:
        for cfg, ctx in (("FP16", 1024), ("GGUF_Q4_K_M", 1024)):
            a, b = stat(p1, mem, cfg, ctx), stat(p2, mem, cfg, ctx) if p2 else (0, 0, float("nan"))
            print(f"{MEM[mem]} {cfg}@{ctx}: e2e {a[2]:.1f}s -> {b[2]:.1f}s")


if __name__ == "__main__":
    main()
