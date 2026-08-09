"""
tools/analyze.py

Regenerate every number quoted in README.md and report/ from a results file,
so the report is reproducible from the raw data by one command rather than by
trusting numbers someone typed once.

Reports, per capacity level (`ballast_gb`) present in the file:
  * headline table at ctx=128
  * engine / quantization factor decomposition
  * context-length scaling, median [min-max]
  * ms/token absolute increase 128->4096 (the "% is a trap" argument)
  * end-to-end seconds for a 150-token generation, prefill included
And, when two capacity levels are present, the degradation table between them.

Run tools/audit.py first — this script assumes the data has already been
checked for sleep gaps and completeness.

Usage:
    python tools/analyze.py [results/results.jsonl]
"""

import argparse
import json
import statistics as st
from collections import defaultdict

CONFIGS = [("FP16", "HF FP16"), ("NF4_4bit", "HF NF4"),
           ("GGUF_F16", "GGUF F16"), ("GGUF_Q4_K_M", "GGUF Q4_K_M")]
CTX = [128, 512, 1024, 2048, 4096]


def load(path):
    """-> {(ballast, config, ctx): [rows]}, plus the ordered capacity levels."""
    d = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("tokens_per_sec") in (None, 0.0) and r.get("perplexity") is not None:
            continue          # perplexity-only rows carry no speed
        bal = r.get("ballast_gb") or 0.0
        d[(bal, r["config"], r["context_len"])].append(r)
    levels = sorted({k[0] for k in d})
    return d, levels


def med(d, bal, cfg, ctx, key="tokens_per_sec"):
    rows = d.get((bal, cfg, ctx), [])
    vals = [r[key] for r in rows if r.get(key) is not None]
    return st.median(vals) if vals else None


def spread(d, bal, cfg, ctx):
    v = [r["tokens_per_sec"] for r in d.get((bal, cfg, ctx), [])]
    return (min(v), max(v), len(v)) if v else (None, None, 0)


def label_for(bal):
    return "24GB (대조군)" if bal == 0.0 else f"{24 - bal:.0f}GB 기계 (ballast {bal:.0f}GB)"


def report_level(d, bal):
    print(f"\n{'=' * 84}\n{label_for(bal)}\n{'=' * 84}")

    print("\n헤드라인 (ctx=128, 중앙값)")
    print(f"  {'Config':<14}{'VRAM(GB)':>10}{'tok/s':>9}{'TTFT(s)':>9}")
    for k, name in CONFIGS:
        v = med(d, bal, k, 128, "peak_vram_gb")
        print(f"  {name:<14}{('—' if v is None else f'{v:.2f}'):>10}"
              f"{med(d, bal, k, 128):>9.2f}{med(d, bal, k, 128, 'ttft_s'):>9.2f}")

    g = lambda k: med(d, bal, k, 128)
    print("\n배수 분해 (ctx=128)")
    print(f"  엔진      GGUF-F16 / HF-FP16    = {g('GGUF_F16'):.2f} / {g('FP16'):.2f} = "
          f"{g('GGUF_F16') / g('FP16'):.2f}배")
    print(f"  K-quant   Q4_K_M / GGUF-F16     = {g('GGUF_Q4_K_M'):.2f} / {g('GGUF_F16'):.2f} = "
          f"{g('GGUF_Q4_K_M') / g('GGUF_F16'):.2f}배")
    print(f"  NF4       HF-NF4 / HF-FP16      = {g('NF4_4bit'):.2f} / {g('FP16'):.2f} = "
          f"{g('NF4_4bit') / g('FP16'):.3f}배 ({g('FP16') / g('NF4_4bit'):.2f}배 느림)")
    print(f"  합계      Q4_K_M / HF-FP16      = {g('GGUF_Q4_K_M') / g('FP16'):.2f}배")

    print("\n컨텍스트 스케일링 — decode tok/s, 중앙값 [최소–최대]")
    print(f"  {'ctx':>5}  " + "".join(f"{n:<26}" for _, n in CONFIGS))
    for x in CTX:
        cells = []
        for k, _ in CONFIGS:
            m = med(d, bal, k, x)
            lo, hi, n = spread(d, bal, k, x)
            cells.append(f"{m:6.2f} [{lo:5.2f}-{hi:5.2f}]{'' if n >= 3 else f' n={n}':<5}")
        print(f"  {x:>5}  " + "".join(f"{c:<26}" for c in cells))

    print("\n토큰당 절대 지연 증가 128→4096  ('% 저하'가 함정인 이유)")
    for k, name in CONFIGS:
        a, b = 1000 / med(d, bal, k, 128), 1000 / med(d, bal, k, 4096)
        pct = (med(d, bal, k, 4096) - med(d, bal, k, 128)) / med(d, bal, k, 128) * 100
        print(f"  {name:<14}{a:8.1f} → {b:9.1f} ms   증가 +{b - a:8.1f} ms   (%로는 {pct:+.0f}%)")

    print("\n150토큰 생성 전체 소요 (프리필 포함, 초)")
    print(f"  {'Config':<14}" + "".join(f"{x:>9}" for x in CTX))
    for k, name in CONFIGS:
        row = "".join(f"{med(d, bal, k, x, 'ttft_s') + 150 / med(d, bal, k, x):>9.0f}" for x in CTX)
        print(f"  {name:<14}{row}")


def report_capacity_delta(d, lo_bal, hi_bal):
    print(f"\n{'=' * 84}\n용량 효과: {label_for(lo_bal)} → {label_for(hi_bal)}\n{'=' * 84}")
    print(f"  {'Config':<14}" + "".join(f"{x:>9}" for x in CTX))
    groups = {"mmap": [], "resident": []}
    for k, name in CONFIGS:
        cells = []
        for x in CTX:
            a, b = med(d, lo_bal, k, x), med(d, hi_bal, k, x)
            pct = (b - a) / a * 100
            cells.append(f"{pct:>8.0f}%")
            groups["mmap" if k.startswith("GGUF") else "resident"].append(pct)
        print(f"  {name:<14}" + "".join(cells))
    print(f"\n  최악 저하 — mmap 그룹(GGUF) {min(groups['mmap']):+.1f}%  vs  "
          f"상주 그룹(HF) {min(groups['resident']):+.1f}%")
    print("  → 연속 스펙트럼이 아니라 두 범주로 갈립니다. 1차 요인은 정밀도가 아니라 메모리 전략입니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/results.jsonl")
    args = ap.parse_args()

    d, levels = load(args.path)
    for bal in levels:
        report_level(d, bal)
    if len(levels) >= 2:
        report_capacity_delta(d, levels[0], levels[-1])


if __name__ == "__main__":
    main()
