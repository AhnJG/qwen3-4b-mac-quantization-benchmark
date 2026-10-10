"""
tools/figures.py

Draw the two README figures from results/results.jsonl, using the same
estimators as tools/analyze.py (median over 5 repeats; steady state from
step_ms_p50). No new measurement -- this only re-plots the audited rows.

Usage:
    python tools/figures.py [results/results.jsonl] [--out figures]
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import CTX, load, med  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(CTX)))
    ax.set_xticklabels([str(c) for c in CTX])
    ax.set_xlabel("Prompt length (tokens)", color=INK2, fontsize=10)


def header(fig, title, sub):
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.885, sub, fontsize=9.5, color=INK2, va="top")


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))


def fig_opposite(d, out):
    """Same 4 bits, opposite directions (24GB control)."""
    p = lambda cfg, x: med(d, 0.0, cfg, x, "step_ms_p50")
    kq = [p("GGUF_F16", x) / p("GGUF_Q4_K_M", x) for x in CTX]
    nf = [p("FP16", x) / p("NF4_4bit", x) for x in CTX]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.70, top=0.78, bottom=0.14)
    style(ax)
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.set_ylim(0.25, 4)
    ax.axhline(1, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0, 1.06, "no change", color=INK2, fontsize=9, va="bottom")
    xs = range(len(CTX))
    for ys, c, lab in [(kq, BLUE, f"llama.cpp Q4_K_M\nvs GGUF F16\n{gm(kq):.2f}× faster"),
                       (nf, ORANGE, f"bitsandbytes NF4\nvs HF FP16\n{1 / gm(nf):.2f}× slower")]:
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(lab, (len(CTX) - 1, ys[-1]), xytext=(14, 0), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center", annotation_clip=False)
    ax.set_ylabel("Decode speed vs 16-bit baseline", color=INK2, fontsize=10)
    header(fig, "Same 4 bits, opposite directions",
           "Steady-state decode speed relative to the same engine's 16-bit weights "
           "(geometric mean in label)\nQwen3-4B-Instruct · Apple M3 24GB · per-step p50, "
           "median of 5 runs")
    fig.savefig(os.path.join(out, "fig1_same_4bit_opposite.png"), facecolor=SURFACE)
    plt.close(fig)
    return kq, nf


def fig_cliff(d, out):
    """The 2048-token 'cliff' was the metric (HF FP16, 24GB control)."""
    mean_ms = [1000 / med(d, 0.0, "FP16", x) for x in CTX]
    p50_ms = [med(d, 0.0, "FP16", x, "step_ms_p50") for x in CTX]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.70, top=0.78, bottom=0.14)
    style(ax)
    ax.set_ylim(0, 500)
    xs = range(len(CTX))
    i, j = CTX.index(1024), CTX.index(2048)
    for ys, c, lab, dy in [(mean_ms, ORANGE, "Mean-based\n1000 / tokens_per_sec", 16),
                           (p50_ms, BLUE, "Per-step median\nstep_ms_p50", -16)]:
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(lab, (len(CTX) - 1, ys[-1]), xytext=(14, dy), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center", annotation_clip=False)
    ax.annotate(f"×{mean_ms[j] / mean_ms[i]:.2f} from 1024", (j, mean_ms[j]),
                xytext=(10, 4), textcoords="offset points", color=INK, fontsize=9.5)
    ax.annotate(f"×{p50_ms[j] / p50_ms[i]:.2f}", (j, p50_ms[j]),
                xytext=(0, 12), textcoords="offset points", color=INK, fontsize=9.5, ha="center")
    ax.set_ylabel("ms per generated token", color=INK2, fontsize=10)
    header(fig, "The 2048-token \"cliff\" was the metric, not the kernel",
           "HF transformers FP16 · Apple M3 24GB · median of 5 runs\n"
           "A few multi-second stalls dominate the mean; the per-step median rises only modestly")
    fig.savefig(os.path.join(out, "fig2_cliff_was_metric.png"), facecolor=SURFACE)
    plt.close(fig)
    return mean_ms, p50_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/results.jsonl")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, _ = load(a.path)
    kq, nf = fig_opposite(d, a.out)
    mean_ms, p50_ms = fig_cliff(d, a.out)
    print("K-quant ratio", [round(v, 2) for v in kq], "gm", round(gm(kq), 2))
    print("NF4 ratio    ", [round(v, 3) for v in nf], "gm", round(gm(nf), 3))
    print("FP16 mean ms ", [round(v, 1) for v in mean_ms])
    print("FP16 p50 ms  ", [round(v, 1) for v in p50_ms])


if __name__ == "__main__":
    main()
