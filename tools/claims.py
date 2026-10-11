"""
tools/claims.py

Print every number the READMEs quote, each computed from the raw files and
tagged with its source (file, data version, metric, capacity). Documents --
README, docs, application materials -- should copy numbers from this output,
never from older prose: mixing numbers from two data versions in one sentence
is exactly the mistake this script exists to prevent.

    python tools/claims.py            # print all claims with sources
    python tools/claims.py --check    # key sentences checked one by one; other N× must be computed values

What --check does and does not do: key sentences and table rows listed in ANCHORS
are compared with the exact claim they state. Every other "N×" in the READMEs and
docs/details must equal *some* computed value, which catches typos but not a
correct number placed in the wrong sentence. Text outside these files
(docs/full-report.ko.md, measurement-history) is not checked.

Speed claims use the current 200-row sweep (results/results.jsonl, v7 data).
Steady state = 1000 / step_ms_p50, where p50 is the median over 5 repeats.
"""

import argparse
import json
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import CTX, e2e_median, e2e_seconds, load, med, ratio_across_ctx  # noqa: E402

SWEEP = "results/results.jsonl (v7, n=5)"
V6 = "results/archive/results_v6.jsonl (v6, n=3)"
PPL = "results/results_perplexity_v7.jsonl"
CHECK_SETS = {"en": ["README.md", "docs/details.md"], "ko": ["README.ko.md", "docs/details.ko.md"]}


# Sentence-level checks: each key sentence / table row is tied to the claim it states.
# A wrong number here fails even if it happens to equal some other computed value.
NUM = re.compile(r"\d+(?:\.\d+)?")
TABLE_ROW = r"\| {label} \([^|]*\|[^\n]*\*\*(\S+)\*\* \|"
ANCHORS = {
    "README.md": [
        (r"llama\.cpp 4-bit became (\S+) faster than its own", ["kquant_gm_24GB"]),
        (r"transformers 4-bit became (\S+) slower", ["nf4_slower_24GB"]),
        (r"lost less quality \(perplexity (\S+) vs (\S+)\)", ["ppl_q4_vs_f16", "ppl_nf4_vs_fp16"]),
        (r"to llama\.cpp 4-bit is (\S+) faster", ["total_gm_24GB"]),
        (r"switching the program gives (\S+) and", ["engine_gm_24GB"]),
        (r"switching to 4-bit (\S+)\.", ["kquant_gm_24GB"]),
        (r"the average speed dropped (\S+),", ["cliff_mean_v7"]),
        (r"\(the median\) only (\S+),", ["cliff_p50_v7"]),
        (r"took llama\.cpp 4-bit (\d+) seconds", ["q4_16GB_2048_e2e"]),
        (r"transformers 16-bit (\S+) minutes", ["fp16_16GB_2048_e2e_range"]),
    ],
    "README.ko.md": [
        (r"llama\.cpp 4비트는 자기 16비트보다 (\S+) 빨라졌고", ["kquant_gm_24GB"]),
        (r"transformers 4비트는 오히려 (\S+) 느려졌습니다", ["nf4_slower_24GB"]),
        (r"\(perplexity (\S+) 대 (\S+)\)", ["ppl_q4_vs_f16", "ppl_nf4_vs_fp16"]),
        (r"llama\.cpp 4비트로 바꾸면 (\S+) 빨라집니다", ["total_gm_24GB"]),
        (r"프로그램 교체가 (\S+),", ["engine_gm_24GB"]),
        (r"4비트 전환이 (\S+)입니다", ["kquant_gm_24GB"]),
        (r"평균 속도는 (\S+) 느려졌지만", ["cliff_mean_v7"]),
        (r"\(중앙값\)은 (\S+)만", ["cliff_p50_v7"]),
        (r"llama\.cpp 4비트는 (\d+)초", ["q4_16GB_2048_e2e"]),
        (r"transformers 16비트는 실행에 따라 (\S+)분", ["fp16_16GB_2048_e2e_range"]),
    ],
    "docs/details.md": [
        (TABLE_ROW.format(label="Program only"), ["engine_gm_24GB"]),
        (TABLE_ROW.format(label="Then 4-bit"), ["kquant_gm_24GB"]),
        (TABLE_ROW.format(label="4-bit in transformers"), ["nf4_gm_24GB"]),
        (TABLE_ROW.format(label="Both"), ["total_gm_24GB"]),
        (r"Q4_K_M\) makes things (\S+ \(\S+\))", ["engine_4bit_text"]),
        (r"an estimated (\S+) s of that", ["fp16_16GB_2048_stall_range"]),
    ],
    "docs/details.ko.md": [
        (TABLE_ROW.format(label="프로그램만 바꿈"), ["engine_gm_24GB"]),
        (TABLE_ROW.format(label="4비트로 바꿈"), ["kquant_gm_24GB"]),
        (TABLE_ROW.format(label="transformers에서 4비트로 바꿈"), ["nf4_gm_24GB"]),
        (TABLE_ROW.format(label="둘 다 바꿈"), ["total_gm_24GB"]),
        (r"Q4_K_M\)로 바꾸면 (\S+ \(\S+\))", ["engine_4bit_text"]),
        (r"네 번은 그중 약 (\S+)초가", ["fp16_16GB_2048_stall_range"]),
    ],
}


def p50(d, bal, cfg, x):
    return med(d, bal, cfg, x, "step_ms_p50")


def per_run(d, bal, cfg, x):
    return [r for r in d.get((bal, cfg, x), [])]


def collect():
    d, _ = load("results/results.jsonl")
    out = []   # (id, text, source, note, in_readme)

    def add(cid, text, source, note, readme=True):
        out.append((cid, text, source, note, readme))

    # --- finding 1 & 2: steady-state ratios, geometric mean + range over contexts
    for bal, cap in ((0.0, "24GB"), (8.0, "16GB")):
        for cid, num, den in [("engine", "GGUF_F16", "FP16"), ("kquant", "GGUF_Q4_K_M", "GGUF_F16"),
                              ("nf4", "NF4_4bit", "FP16"), ("total", "GGUF_Q4_K_M", "FP16"),
                              ("engine_4bit", "GGUF_Q4_K_M", "NF4_4bit"),
                              ("f16_over_nf4", "GGUF_F16", "NF4_4bit")]:
            rs, g = ratio_across_ctx(d, bal, num, den)
            src = f"{SWEEP}, step p50, {cap}"
            add(f"{cid}_gm_{cap}", f"{g:.2f}×", src, f"{num}/{den} geometric mean over 5 contexts",
                readme=(cap == "24GB" and cid in ("engine", "kquant", "total")))
            add(f"{cid}_range_{cap}", f"{min(rs):.2f}–{max(rs):.2f}×", src,
                f"{num}/{den} per context: " + ", ".join(f"{r:.2f}" for r in rs),
                readme=(cap == "24GB" and cid in ("engine", "kquant", "nf4")))
            if cid == "nf4":
                add(f"nf4_slower_{cap}", f"{1 / g:.2f}×", src, "how many times slower NF4 is (1 / geometric mean)")
                if cap == "16GB":
                    add("nf4_over_fp16_16GB_2048", f"{rs[CTX.index(2048)]:.2f}×", src,
                        "NF4/FP16 at ctx 2048 (audit FAIL cell FP16@2048)")
    # rounded forms the README prints
    _, g24 = ratio_across_ctx(d, 0.0, "GGUF_F16", "NF4_4bit")
    _, g16 = ratio_across_ctx(d, 8.0, "GGUF_F16", "NF4_4bit")
    add("f16_over_nf4_24GB_rounded", f"{g24:.1f}×", f"{SWEEP}, step p50, 24GB",
        "how much slower HF NF4 is than GGUF F16 (geometric mean)")
    add("f16_over_nf4_16GB_rounded", f"{g16:.1f}×", f"{SWEEP}, step p50, 16GB",
        "how much slower HF NF4 is than GGUF F16 (geometric mean)")
    lo = min(min(ratio_across_ctx(d, b, "GGUF_F16", "NF4_4bit")[0]) for b in (0.0, 8.0))
    hi = max(max(ratio_across_ctx(d, b, "GGUF_F16", "NF4_4bit")[0]) for b in (0.0, 8.0))
    add("f16_over_nf4_range_both", f"{lo:.1f}–{hi:.1f}×", f"{SWEEP}, step p50", "both capacities")
    kq = ratio_across_ctx(d, 0.0, "GGUF_Q4_K_M", "GGUF_F16")[0] + ratio_across_ctx(d, 8.0, "GGUF_Q4_K_M", "GGUF_F16")[0]
    add("kquant_range_both", f"{min(kq):.2f}–{max(kq):.2f}×", f"{SWEEP}, step p50", "both capacities")
    e4 = ratio_across_ctx(d, 0.0, "GGUF_Q4_K_M", "NF4_4bit")
    add("engine_4bit_text", f"{e4[1]:.1f}× ({min(e4[0]):.1f}–{max(e4[0]):.1f}×)", f"{SWEEP}, step p50, 24GB",
        "HF NF4 -> llama.cpp Q4_K_M")
    slower_all = all(p50(d, b, "NF4_4bit", x) > p50(d, b, "GGUF_Q4_K_M", x) for b in (0.0, 8.0) for x in CTX)
    add("nf4_slower_than_q4_all_cells", f"{'all' if slower_all else 'NOT all'} 10 contexts × capacities",
        f"{SWEEP}, step p50", "NF4 slower than Q4_K_M in every cell", readme=False)

    # --- perplexity
    ppl = {json.loads(l)["config"]: json.loads(l)["perplexity"] for l in open(PPL) if l.strip()}
    for k in ("FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M"):
        add(f"ppl_{k}", f"{ppl[k]:.4f}", PPL, "WikiText-2 test, 299,078 tokens")
    add("ppl_q4_vs_f16", f"+{(ppl['GGUF_Q4_K_M'] / ppl['GGUF_F16'] - 1) * 100:.2f}%", PPL, "Q4_K_M vs GGUF F16")
    add("ppl_nf4_vs_fp16", f"+{(ppl['NF4_4bit'] / ppl['FP16'] - 1) * 100:.2f}%", PPL, "NF4 vs HF FP16")
    add("ppl_f16_agreement", f"{abs(ppl['GGUF_F16'] / ppl['FP16'] - 1) * 100:.3f}%", PPL,
        "two independent F16 paths")

    # --- finding 3: the cliff, HF FP16 24GB, 1024 -> 2048
    m = med(d, 0.0, "FP16", 1024) / med(d, 0.0, "FP16", 2048)
    s = p50(d, 0.0, "FP16", 2048) / p50(d, 0.0, "FP16", 1024)
    add("cliff_mean_v7", f"{m:.2f}×", f"{SWEEP}, mean tok/s, 24GB", "FP16 1024->2048 slowdown, mean-based")
    add("cliff_p50_v7", f"{s:.2f}×", f"{SWEEP}, step p50, 24GB", "FP16 1024->2048 slowdown, per-step median")
    add("fp16_p50_2048_4096", f"{p50(d, 0.0, 'FP16', 2048):.1f} vs {p50(d, 0.0, 'FP16', 4096):.1f} ms",
        f"{SWEEP}, step p50, 24GB", "2048 still slower than 4096")
    try:
        d6, _ = load("results/archive/results_v6.jsonl")
        v6 = med(d6, 0.0, "FP16", 1024) / med(d6, 0.0, "FP16", 2048)
        add("cliff_mean_v6", f"{v6:.2f}×", f"{V6}, mean tok/s, 24GB",
            "the '4.3×' of older text; do NOT pair with the v7 p50", readme=False)
    except (OSError, KeyError, TypeError):
        pass

    # --- probe logs (single runs, quoted as saved)
    log = open("results/probe_cache.log").read()
    rows = {(int(a), b): (float(c), float(e), float(g), float(h)) for a, b, c, _, e, _, g, h in
            re.findall(r"^\s+(\d+)\s+(dynamic|static)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)",
                       log, re.M)}
    dy, sc = rows[(2048, "dynamic")], rows[(2048, "static")]
    add("static_max_2048", f"{dy[1]:,.0f} → {sc[1]:,.0f} ms", "results/probe_cache.log (2 runs/cell)",
        "max step, DynamicCache -> StaticCache")
    add("static_p50_2048", f"{dy[0]:.0f} to {sc[0]:.0f} ms", "results/probe_cache.log (2 runs/cell)",
        "per-step median also rises")
    add("static_tax_2048", f"{dy[2]:.2f}x → {sc[2]:.2f}x", "results/probe_cache.log", "stall tax", readme=False)
    add("static_p50_ratio", f"{sc[0] / dy[0]:.1f}×", "results/probe_cache.log", "StaticCache p50 / DynamicCache p50",
        readme=False)
    peak = re.search(r"생성 중 최고\s+([\d.]+)", open("results/probe_residue.log").read()).group(1)
    add("residue_peak", f"{peak} GiB", "results/probe_residue.log (HF FP16 only)", "peak driver memory")

    # --- finding 4: 16GB
    taxes = [1000 / p50(d, 8.0, k, x) / med(d, 8.0, k, x) for k in ("GGUF_F16", "GGUF_Q4_K_M") for x in CTX]
    add("gguf_tax_16GB", f"≤{max(taxes):.2f}× in all {len(taxes)}", f"{SWEEP}, 16GB", "llama.cpp stall tax")
    ch = [(p50(d, 0.0, k, x) / p50(d, 8.0, k, x) - 1) * 100 for k in ("GGUF_F16", "GGUF_Q4_K_M") for x in CTX]
    add("gguf_steady_change", f"{min(ch):+.0f}% to {max(ch):+.0f}%", f"{SWEEP}, step p50",
        "llama.cpp steady-state speed, 16GB vs 24GB")
    rs = per_run(d, 8.0, "FP16", 2048)
    e2e = sorted(e2e_seconds(r) for r in rs)
    add("fp16_16GB_2048_e2e_range", f"{e2e[0] / 60:.1f}–{e2e[-1] / 60:.1f}", f"{SWEEP}, 16GB, per run",
        "TTFT + (n_new-1)/tok/s for each of 5 runs, in minutes (audit FAIL cell)")
    add("fp16_16GB_2048_e2e_median", f"{e2e_median(d, 8.0, 'FP16', 2048):.0f} s", f"{SWEEP}, 16GB",
        "median over runs, as analyze.py", readme=False)
    # estimated stall time per run: decode time minus (tokens x that run's median step).
    # An estimate, not a measurement. It comes out negative when stalls are spread over so many steps
    # that the run's own p50 is inflated; such a run's stall time cannot be separated this way.
    stalls = sorted((r["n_new"] - 1) / r["tokens_per_sec"] - (r["n_new"] - 1) * r["step_ms_p50"] / 1000
                    for r in rs)
    pos = [s for s in stalls if s > 0]
    add("fp16_16GB_2048_stall_range", f"{min(pos):.0f}–{max(pos):.0f}", f"{SWEEP}, 16GB, per run",
        f"estimated stall seconds in {len(pos)} of {len(stalls)} runs: " + ", ".join(f"{s:.0f}" for s in stalls))
    q4 = e2e_median(d, 8.0, "GGUF_Q4_K_M", 2048)
    add("q4_16GB_2048_e2e", f"{q4:.0f} seconds", f"{SWEEP}, 16GB", "Q4_K_M, same task")

    # --- headline table ctx=128, 24GB
    for k in ("FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M"):
        add(f"tps128_{k}", f"{med(d, 0.0, k, 128):.2f}", f"{SWEEP}, mean tok/s, 24GB", "ctx 128 (stall tax 1.0)")
        add(f"ttft128_{k}", f"{med(d, 0.0, k, 128, 'ttft_s'):.2f}", f"{SWEEP}, 24GB", "TTFT ctx 128")
    for k in ("FP16", "NF4_4bit"):
        add(f"vram128_{k}", f"{med(d, 0.0, k, 128, 'peak_vram_gb'):.2f}", f"{SWEEP}, 24GB", "peak memory")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if README + docs/details no longer contain a claim's text")
    a = ap.parse_args()
    claims = collect()
    if not a.check:
        for cid, text, src, note, _ in claims:
            print(f"{cid:<30} {text:<36} {src}  — {note}")
        return
    bad = 0
    by_id = {c[0]: c[1] for c in claims}
    n_anchor = 0
    # 0) key sentences and table rows: the number written there must be the claim it states
    for f, rules in ANCHORS.items():
        body = open(f).read()
        for pattern, ids in rules:
            m = re.search(pattern, body)
            n_anchor += 1
            if not m:
                print(f"NOT FOUND {f}: /{pattern}/ (sentence changed? update ANCHORS)")
                bad += 1
                continue
            for got, cid in zip(m.groups(), ids):
                if NUM.findall(got) != NUM.findall(by_id[cid]):
                    print(f"WRONG    {f}: {got!r} should be {by_id[cid]!r} ({cid})")
                    bad += 1
    # 1) every "N.NN×" written in the READMEs / details must be a value some claim computes
    #    (catches a wrong number even when the right one still appears elsewhere)
    known = {n for c in claims for n in re.findall(r"\d+(?:\.\d+)?", c[1])} | {"1.0", "1"}
    for files in CHECK_SETS.values():
        for f in files:
            for n in re.findall(r"(\d+(?:\.\d+)?)×(?!\s?\d)", open(f).read()):   # skip "2×2"-style design labels
                if n not in known:
                    print(f"UNKNOWN  {f}: {n}× is not a computed value")
                    bad += 1
    # 2) every README claim must still appear in README + details of each language
    for lang, files in CHECK_SETS.items():
        body = "\n".join(open(f).read() for f in files)   # README + its details page
        for cid, text, *_, readme in claims:
            # the Korean pages word units differently; only unit-free texts are checked there
            if lang == "ko" and re.search(r"[A-Za-z]", text):
                continue
            if readme and text not in body:
                print(f"MISSING  {' + '.join(files)}: {cid} = {text!r}")
                bad += 1
    print(f"{'OK' if not bad else 'FAIL'}: {n_anchor} key sentences/rows checked against their claims, "
          f"every N× checked against computed values, {sum(c[4] for c in claims)} claims checked for presence; "
          f"{bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
