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
  6. Steady state   -- per-cell CV of step_ms_p50; a cell that swings across
                       repeats cannot support a single-summary conclusion

Exits non-zero if any integrity check fails (missing or short cells, sleep gaps,
token counts), so it
can gate an analysis step. The steady-state spread gate is reported separately:
a cell that fails it is real data, but no single summary of it should carry a
conclusion.

Usage:
    python tools/audit.py [results/results.jsonl] [--repeats 5]
"""

import argparse
import datetime as dt
import json
import sys
import statistics as st
from collections import Counter, defaultdict

CONFIGS = ("FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M")
GAP_ALERT_S = 120
# the experiment's fixed design: every level must contain every one of these cells
EXPECTED_CONFIGS = ("FP16", "NF4_4bit", "GGUF_F16", "GGUF_Q4_K_M")
EXPECTED_CTX = (128, 512, 1024, 2048, 4096)

CV_WARN = 0.15      # steady-state spread worth flagging
CV_FAIL = 0.50      # above this, no single summary of the cell is meaningful


def ts(s):
    return dt.datetime.fromisoformat(s)


def audit(rows, label, repeats, grid, check_gaps=True):
    """Return (n_failures, n_warnings) after printing a report for one level."""
    fails = warns = vfails = 0
    rows = sorted(rows, key=lambda r: r["started_at"] or "")
    print(f"\n{'=' * 74}\n{label} — {len(rows)} rows\n{'=' * 74}")

    cells = Counter((r["config"], r["context_len"]) for r in rows)
    missing = sorted(set(grid) - set(cells), key=str)
    off = {k: v for k, v in sorted(cells.items(), key=str) if v != repeats}
    status = "OK" if not (off or missing) else "FAIL"
    print(f"  [{status}] 완전성      {len(cells)}/{len(grid)} 셀, 빠진 셀: {missing or '없음'}, "
          f"{repeats}회 아닌 셀: {off or '없음'}")
    fails += bool(off or missing)

    if any(r.get("started_at") for r in rows):
        gaps = [
            (a["ended_at"], b["started_at"], (ts(b["started_at"]) - ts(a["ended_at"])).total_seconds())
            for a, b in zip(rows, rows[1:])
        ]
        big = [g for g in gaps if g[2] > GAP_ALERT_S]
        if check_gaps:
            print(f"  [{'FAIL' if big else 'OK'}] 잠자기/중단  {GAP_ALERT_S}s 초과 공백 {len(big)}건")
            for a, b, g in big[:5]:
                print(f"           {a} → {b}  ({g / 60:.1f}분)")
            fails += bool(big)
        else:
            print("  [--] 잠자기/중단  메모리 조건을 번갈아 돌린 파일이라 전체 행 기준으로 따로 검사")

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

    # Steady-state stability. A cell whose p50 swings wildly across repeats
    # cannot support a conclusion drawn from any single summary of it -- this
    # is the gate that would have caught v7's min-over-repeats estimator, whose
    # bias scales with exactly this spread.
    by_cell = defaultdict(list)
    for r in rows:
        if r.get("step_ms_p50") is not None:
            by_cell[(r["config"], r["context_len"])].append(r["step_ms_p50"])
    cvs = {k: (st.pstdev(v) / st.mean(v)) for k, v in by_cell.items() if len(v) > 1}
    if cvs:
        hot = sorted(((c, k) for k, c in cvs.items()), reverse=True)
        bad = [(c, k) for c, k in hot if c > CV_FAIL]
        warn = [(c, k) for c, k in hot if CV_WARN < c <= CV_FAIL]
        status = "FAIL" if bad else ("WARN" if warn else "OK")
        print(f"  [{status}] 정상상태 분산  셀별 CV(step_ms_p50): "
              f"FAIL(>{CV_FAIL:.0%}) {len(bad)}건, WARN(>{CV_WARN:.0%}) {len(warn)}건")
        for c, k in (bad + warn)[:5]:
            print(f"           {k[0]}@{k[1]}  CV {c:.0%}  "
                  f"p50 {[round(v, 1) for v in sorted(by_cell[k])]}")
        vfails += bool(bad)
        warns += bool(warn)
    else:
        print("  [WARN] step_ms_p50 없음 — 정상상태 분산을 검사할 수 없는 실행입니다 (v6 이전 형식)")
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
    return fails, warns, vfails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/results.jsonl")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--configs", default=None,
                    help="Comma-separated configs this run was designed to cover "
                         "(default: the fixed four)")
    ap.add_argument("--contexts", default=None,
                    help="Comma-separated input lengths this run was designed to cover "
                         "(default: the fixed five)")
    ap.add_argument("--interleaved", action="store_true",
                    help="Memory levels were alternated in one run: check sleep gaps "
                         "across all rows in time order instead of within each level")
    args = ap.parse_args()
    exp_cfg = tuple(args.configs.split(",")) if args.configs else EXPECTED_CONFIGS
    exp_ctx = tuple(int(x) for x in args.contexts.split(",")) if args.contexts else EXPECTED_CTX

    rows = [json.loads(l) for l in open(args.path) if l.strip()]
    levels = sorted({r.get("ballast_gb") for r in rows}, key=lambda x: (x is not None, x))

    # expected grid: the fixed design, plus anything else that appears in the file
    configs = set(exp_cfg) | {r["config"] for r in rows}
    ctxs = set(exp_ctx) | {r["context_len"] for r in rows}
    grid = sorted({(c, x) for c in configs for x in ctxs}, key=str)
    total_f = total_w = total_v = 0
    if args.interleaved:
        allr = sorted(rows, key=lambda r: r["started_at"] or "")
        big = [(a["ended_at"], b["started_at"]) for a, b in zip(allr, allr[1:])
               if (ts(b["started_at"]) - ts(a["ended_at"])).total_seconds() > GAP_ALERT_S]
        print(f"[{'FAIL' if big else 'OK'}] 잠자기/중단 (전체 행)  {GAP_ALERT_S}s 초과 공백 {len(big)}건")
        for a, b in big[:5]:
            print(f"           {a} → {b}")
        total_f += bool(big)
    for lv in levels:
        sub = [r for r in rows if r.get("ballast_gb") == lv]
        label = "24GB (ballast 없음)" if lv in (None, 0.0) else f"ballast {lv}GB (= {24 - lv:.0f}GB 기계)"
        f, w, v = audit(sub, label, args.repeats, grid, check_gaps=not args.interleaved)
        total_f += f
        total_w += w
        total_v += v

    print(f"\n{'=' * 74}")
    print(f"결과: {len(rows)}행 · 무결성 실패 {total_f} · 분산 실패 {total_v} · 경고 {total_w}")
    if total_f:
        print("무결성 실패가 있습니다 — 이 데이터로 결론을 내리지 마십시오.")
    elif total_v:
        print("무결성 감사 통과 — 분석에 사용 가능합니다.\n"
              "단, 분산 FAIL 셀은 반복 간 p50이 크게 흔들리므로 단일 요약값으로 결론을 내리지 마십시오.")
    else:
        print("감사 통과 — 분석에 사용 가능합니다.")
    sys.exit(1 if total_f else 0)


if __name__ == "__main__":
    main()
