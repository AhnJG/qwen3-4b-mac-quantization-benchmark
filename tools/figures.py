"""
tools/figures.py

Draw the README figures (an overview and two result plots) from results/results.jsonl, using the same
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
    """Both called 4-bit, opposite directions (24GB control)."""
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
    header(fig, "Both called 4-bit, opposite directions",
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
    header(fig, "The 2048-token \"cliff\" came mostly from rare stalls",
           "HF transformers FP16 · Apple M3 24GB · median of 5 runs\n"
           "The mean includes a few multi-second stalls (total waiting time); the per-step median shows the typical token")
    fig.savefig(os.path.join(out, "fig2_cliff_was_metric.png"), facecolor=SURFACE)
    plt.close(fig)
    return mean_ms, p50_ms


def fig_overview(out):
    """One-picture summary: design, data pipeline, key findings. Every number is
    taken from tools/claims.py, so the picture cannot drift from the README."""
    from matplotlib.patches import FancyBboxPatch
    import claims
    c = {cid: text for cid, text, *_ in claims.collect()}
    n_anchor = sum(len(v) for v in claims.ANCHORS.values())

    fig = plt.figure(figsize=(12, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(-0.4, 5.6); ax.axis("off")

    def box(x, y, w, h, fc=SURFACE, ec=GRID, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=lw))

    def t(x, y, s, size=9.5, color=INK, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va)

    t(0.35, 5.25, "On-device LLM quantization benchmark: Qwen3-4B on a MacBook Air M3",
      size=14, weight="bold")
    for x, title in [(0.35, "1  What was measured"), (4.35, "2  How the numbers are produced"),
                     (8.15, "3  What came out (24GB unless noted)")]:
        t(x, 4.75, title, size=11, weight="bold")

    # 1. design: 2x2 grid plus the other axes
    gx, gy, cw, ch = 1.35, 3.05, 1.25, 0.62
    t(gx + cw / 2, gy + 2 * ch + 0.2, "transformers", size=9, color=INK2, ha="center")
    t(gx + 1.5 * cw, gy + 2 * ch + 0.2, "llama.cpp", size=9, color=INK2, ha="center")
    t(gx - 0.12, gy + 1.5 * ch, "16-bit", size=9, color=INK2, ha="right")
    t(gx - 0.12, gy + 0.5 * ch, "4-bit", size=9, color=INK2, ha="right")
    cells = [(0, 1, "FP16 (baseline)", INK), (1, 1, "F16", INK),
             (0, 0, "NF4", ORANGE), (1, 0, "Q4_K_M", BLUE)]
    for i, j, lab, col in cells:
        box(gx + i * cw + 0.03, gy + j * ch + 0.03, cw - 0.06, ch - 0.06)
        t(gx + i * cw + cw / 2, gy + j * ch + ch / 2, lab, size=9.5, color=col,
          weight="bold", ha="center")
    rows = ["\u00d7 2 memory sizes: 24GB, and 16GB imitated",
            "    by another program holding 8GB",
            "\u00d7 5 input lengths: 128 to 4096 tokens",
            "\u00d7 5 repeats, 150 generated tokens each"]
    for k, r in enumerate(rows):
        t(0.45, 2.55 - k * 0.3, r, size=9.5)
    t(0.45, 1.3, "= 200 measurements, run order shuffled", size=10, weight="bold")
    t(0.45, 0.95, "+ quality: WikiText-2 perplexity per setup", size=9.5, color=INK2)

    # 2. pipeline
    steps = [("benchmark_quantization.py", "times every token; appends one JSON line per run"),
             ("results/results.jsonl", "200 raw rows, kept as measured"),
             ("tools/audit.py", "stops on missing runs, sleep gaps, wrong token counts"),
             ("tools/analyze.py, figures.py", "median time per token; geometric mean of ratios"),
             ("tools/claims.py --check", f"{n_anchor} key README sentences checked against the data")]
    bx, bw, bh, top = 4.35, 3.6, 0.62, 4.2
    for k, (name, desc) in enumerate(steps):
        y = top - k * 0.82
        box(bx, y - bh, bw, bh, ec=BLUE if k == 2 else GRID, lw=1.6 if k == 2 else 1.2)
        t(bx + 0.12, y - 0.2, name, size=9.5, weight="bold", color=INK)
        t(bx + 0.12, y - 0.45, desc, size=8.5, color=INK2)
        if k < len(steps) - 1:
            ax.annotate("", xy=(bx + bw / 2, y - bh - 0.18), xytext=(bx + bw / 2, y - bh - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    # 3. findings
    fx = 8.15
    t(fx, 4.3, "Both called 4-bit, opposite directions", size=9.5, weight="bold")
    t(fx, 4.0, f"llama.cpp 4-bit: {c['kquant_gm_24GB']} faster ({c['kquant_range_24GB']})",
      size=9.5, color=BLUE)
    t(fx, 3.72, f"transformers 4-bit: {c['nf4_slower_24GB']} slower", size=9.5, color=ORANGE)
    t(fx, 3.44, f"Perplexity loss: {c['ppl_q4_vs_f16']} vs {c['ppl_nf4_vs_fp16']}", size=9.5)
    t(fx, 2.85, "2048-token slowdown: mostly rare stalls", size=9.5, weight="bold")
    t(fx, 2.55, f"transformers 16-bit, 1024 \u2192 2048 tokens:", size=9.5)
    t(fx, 2.27, f"mean {c['cliff_mean_v7']} slower, median only {c['cliff_p50_v7']}", size=9.5)
    t(fx, 1.68, "16GB: same 2048-token job", size=9.5, weight="bold")
    t(fx, 1.38, f"llama.cpp 4-bit: {c['q4_16GB_2048_e2e']}", size=9.5, color=BLUE)
    t(fx, 1.1, f"transformers 16-bit: {c['fp16_16GB_2048_e2e_range']} minutes", size=9.5,
      color=ORANGE)

    t(0.35, 0.0, "Speed ratios use the median time per token, as geometric means over the 5 input lengths.",
      size=8.5, color=INK2)
    t(0.35, -0.25, "Chrome and the Claude app were left open and not monitored; the two memory sizes "
      "ran one after the other. See Limitations.", size=8.5, color=INK2)
    fig.savefig(os.path.join(out, "fig0_overview.png"), facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/results.jsonl")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, _ = load(a.path)
    fig_overview(a.out)
    kq, nf = fig_opposite(d, a.out)
    mean_ms, p50_ms = fig_cliff(d, a.out)
    print("K-quant ratio", [round(v, 2) for v in kq], "gm", round(gm(kq), 2))
    print("NF4 ratio    ", [round(v, 3) for v in nf], "gm", round(gm(nf), 3))
    print("FP16 mean ms ", [round(v, 1) for v in mean_ms])
    print("FP16 p50 ms  ", [round(v, 1) for v in p50_ms])


if __name__ == "__main__":
    main()
