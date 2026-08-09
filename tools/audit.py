"""
tools/audit.py

Check a results file for integrity BEFORE trusting any number in it.

This exists because v5 had to be thrown away: the machine slept for ~6 of the
run's 8 hours, and because rows carried no timestamps there was no way to prove
after the fact which measurements straddled a sleep. Wall-clock timing cannot
tell "slow measurement" apart from "suspended process", so an un-auditable run
is an unusable run.

Checks, per capacity level (rows are grouped by `ballast_gb`):
  1. Completeness   -- every (config, ctx) cell measured `--repeats` times
  2. Sleep/stall    -- gaps between one row's ended_at and the next's started_at
  3. Duration       -- total measured time vs wall-clock span; they must match,
                       and the slowest cells are listed (thrashing shows here)
  4. Pressure       -- swap range actually experienced, so it is on record
  5. Token sanity   -- n_new in range and prompt_tokens consistent per ctx

Exits non-zero if any hard check fails, so it can gate an analysis step.

Usage:
    python tools/audit.py [results/results.jsonl] [--repeats 3]
"""

import argparse
import datetime as dt
import json
import sys
from collections import Counter

CONFIGS = ("FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M")
GAP_ALERT_S = 120


def ts(s):
    return dt.datetime.fromisoformat(s)


def audit(rows, label, repeats):
    """Return (n_failures, n_warnings) after printing a report for one level."""
    fails = warns = 0
    rows = sorted(rows, key=lambda r: r["started_at"] or "")
    print(f"\n{'=' * 74}\n{label} — {len(rows)} rows\n{'=' * 74}")

    cells = Counter((r["config"], r["context_len"]) for r in rows)
    off = {k: v for k, v in sorted(cells.items(), key=str) if v != repeats}
    status = "OK" if not off else "PARTIAL"
    print(f"  [{status}] 완전성      {len(cells)} 셀, {repeats}회 아닌 셀: {off or '없음'}")
    if off:
        warns += 1

    if any(r.get("started_at") for r in rows):
        gaps = [
            (a["ended_at"], b["started_at"], (ts(b["started_at"]) - ts(a["ended_at"])).total_seconds())
            for a, b in zip(rows, rows[1:])
        ]
        big = [g for g in gaps if g[2] > GAP_ALERT_S]
        print(f"  [{'FAIL' if big else 'OK'}] 잠자기/중단  {GAP_ALERT_S}s 초과 공백 {len(big)}건")
        for a, b, g in big[:5]:
            print(f"           {a} → {b}  ({g / 60:.1f}분)")
        fails += bool(big)

        durs = sorted(
            (((ts(r["ended_at"]) - ts(r["started_at"])).total_seconds(), r) for r in rows),
            key=lambda x: x[0], reverse=True,   # equal durations must not compare dicts
        )
        measured = sum(d for d, _ in durs)
        span = (ts(rows[-1]["ended_at"]) - ts(rows[0]["started_at"])).total_seconds()
        # measured time should account for essentially all wall-clock time
        ratio = measured / span if span else 0
        print(f"  [{'OK' if ratio > 0.97 else 'WARN'}] 죽은 시간    "
              f"측정 {measured / 60:.0f}분 / 벽시계 {span / 60:.0f}분 ({ratio:.1%})")
        warns += ratio <= 0.97
        print("           최장: " + " | ".join(
            f"{r['config']}@{r['context_len']} {d / 60:.1f}분" for d, r in durs[:3]))
    else:
        print("  [WARN] 타임스탬프 없음 — 잠자기 오염을 검증할 수 없는 실행입니다 (v5 이전 형식)")
        warns += 1

    sw = [r["swap_used_gb"] for r in rows if r.get("swap_used_gb") is not None]
    if sw:
        print(f"  [INFO] 스왑        {min(sw):.2f} → {max(sw):.2f} GB "
              f"(중앙 {sorted(sw)[len(sw) // 2]:.2f}) — 압박은 제거되지 않고 기록될 뿐입니다")

    bad_n = [r for r in rows if r.get("n_new") is not None and not 140 <= r["n_new"] <= 150]
    ptok = {}
    for r in rows:
        ptok.setdefault(r["context_len"], set()).add(r.get("prompt_tokens"))
    incons = {k: v for k, v in ptok.items() if len(v) > 1}
    print(f"  [{'FAIL' if bad_n or incons else 'OK'}] 토큰 정합성  "
          f"n_new 이상 {len(bad_n)}건, prompt_tokens 불일치 {incons or '없음'}")
    fails += bool(bad_n or incons)
    return fails, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/results.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    levels = sorted({r.get("ballast_gb") for r in rows}, key=lambda x: (x is not None, x))

    total_f = total_w = 0
    for lv in levels:
        sub = [r for r in rows if r.get("ballast_gb") == lv]
        label = "24GB (ballast 없음)" if lv in (None, 0.0) else f"ballast {lv}GB (= {24 - lv:.0f}GB 기계)"
        f, w = audit(sub, label, args.repeats)
        total_f += f
        total_w += w

    print(f"\n{'=' * 74}")
    print(f"결과: {len(rows)}행 · 실패 {total_f} · 경고 {total_w}")
    print("실패가 0이 아니면 이 데이터로 결론을 내리지 마십시오." if total_f
          else "감사 통과 — 분석에 사용 가능합니다.")
    sys.exit(1 if total_f else 0)


if __name__ == "__main__":
    main()
