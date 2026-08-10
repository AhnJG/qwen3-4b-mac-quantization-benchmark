"""
tools/cliff_probe.py

Investigate the unexplained FP16 @2048 non-monotonic cliff (README "미해결").
The README names two suspects to rule out first, and this script is one probe
per suspect, plus a third that decides between them:

  --probe prompt   Does make_long_prompt build a pathological input at 2048?
                   Rebuilds the exact tensor generate() is fed at every ctx and
                   checks length, re-tokenization stability, decode damage, and
                   periodicity. A harness defect shows up as a discontinuity at
                   2048 that the neighbouring lengths do not have.

  --probe sdpa     Does the SDPA kernel change behaviour with sequence length?
                   Times F.scaled_dot_product_attention alone, at Qwen3-4B's
                   real decode shapes, on a fine grid around 2048 -- and again
                   under a forced MATH backend. A kernel/shape cliff must be
                   visible here, with no model and no KV cache in the picture.

  --probe steps    Per-decode-step latency for the real model at 1024/2048/4096
                   (loads FP16 weights; slow). A cliff that is uniform across
                   all 150 steps is a per-step cost -- shape or kernel. A cliff
                   that starts at step k is state that accumulates -- allocator,
                   cache growth, paging.

  --probe cache    Are the stalls the KV cache reallocating? Compares the
                   default DynamicCache against StaticCache, which reserves the
                   whole cache up front the way llama.cpp does.

  --probe alloc    The direct causal test, on the real measurement path: polls
                   torch.mps.driver_allocated_memory() next to every decode
                   timestamp. If stalls are the allocator going to the OS, the
                   slow steps are exactly the steps where driver memory grows.
                   If slow steps show no driver growth, the stall is paging of
                   memory we already hold -- a different mechanism, and the
                   StaticCache result would then be a coincidence.

Nothing here re-runs the benchmark; these are diagnostics that isolate one
mechanism at a time.

Usage:
    python tools/cliff_probe.py --probe prompt
    python tools/cliff_probe.py --probe sdpa
    python tools/cliff_probe.py --probe steps        # loads the 8GB model
    python tools/cliff_probe.py --probe cache
    python tools/cliff_probe.py --probe alloc [--ctx 2048]
"""

import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmark_quantization import (
    FILLER_TEXT, MODEL_ID, build_chat_prompt_string, make_long_prompt, sync_device,
)

CTX = [128, 512, 1024, 2048, 4096]


# --- probe 1: is the prompt itself pathological at 2048? ---------------------

def probe_prompt():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    filler_ids = tok(FILLER_TEXT).input_ids
    period = len(filler_ids)
    print(f"FILLER_TEXT = {period} tokens; ctx targets are multiples of it only "
          f"by accident\n")

    print(f"  {'ctx':>5} {'filler':>7} {'fed':>6} {'retok':>6} {'uniq':>5} "
          f"{'U+FFFD':>7} {'period':>7} {'periodic':>12} {'maxid':>7}")
    rows = {}
    for ctx in CTX:
        filler = make_long_prompt(tok, ctx)
        n_filler = len(tok(filler).input_ids)

        # the exact tensor generate() sees
        text = build_chat_prompt_string(tok, filler)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        # re-tokenizing the decode of those ids must be a fixed point, or the
        # "prompt is N tokens" claim is only true of the tensor we happened to
        # build first
        retok = len(tok(tok.decode(ids), add_special_tokens=False).input_ids)

        bad = filler.count("�")

        # The filler body must be strictly periodic. Its in-context period is
        # not len(tok(FILLER_TEXT)) -- BPE merges across the repeat seam -- so
        # recover the period from the ids themselves, then count breaks.
        body = ids[10:len(ids) - 10].tolist()
        p = next((p for p in range(1, min(64, len(body) // 3))
                  if all(body[i] == body[i - p] for i in range(p, len(body)))),
                 None)
        nonperiodic = "-" if p else "BROKEN"

        rows[ctx] = dict(n_filler=n_filler, fed=len(ids), retok=retok,
                         nonperiodic=nonperiodic, bad=bad, period=p)
        print(f"  {ctx:>5} {n_filler:>7} {len(ids):>6} {retok:>6} "
              f"{len(set(ids.tolist())):>5} {bad:>7} {str(p):>7} "
              f"{nonperiodic:>12} {int(ids.max()):>7}")

    print()
    ok = True
    for ctx, r in rows.items():
        if r["n_filler"] != ctx:
            print(f"  FAIL ctx={ctx}: filler is {r['n_filler']} tokens, not {ctx}")
            ok = False
        if r["fed"] != r["retok"]:
            print(f"  FAIL ctx={ctx}: fed {r['fed']} tokens, re-tokenizes to "
                  f"{r['retok']} -- prompt length is not a fixed point")
            ok = False
        if r["bad"] or r["nonperiodic"] != "-":
            print(f"  FAIL ctx={ctx}: {r['bad']} replacement chars, "
                  f"periodicity {r['nonperiodic']}")
            ok = False
    overhead = {ctx: r["fed"] - ctx for ctx, r in rows.items()}
    print(f"  chat-template overhead per ctx: {overhead}")
    if len(set(overhead.values())) != 1:
        print("  FAIL: template overhead is not constant across ctx")
        ok = False
    print("\n  => 2048 is constructed exactly like its neighbours."
          if ok else "\n  => the harness IS building a different input somewhere.")


# --- probe 2: does SDPA change kernel with sequence length? ------------------

def sdpa_shapes():
    """Qwen3-4B-Instruct-2507 decode-step attention shapes, read from the real
    config so the probe cannot drift from the benchmarked model."""
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(MODEL_ID)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_attention_heads, c.num_key_value_heads, head_dim, c.num_hidden_layers


def time_sdpa(nh, nkv, hd, seq, dtype, iters=30, backend=None):
    import torch.nn.functional as F

    q = torch.randn(1, nh, 1, hd, device="mps", dtype=dtype)
    k = torch.randn(1, nkv, seq, hd, device="mps", dtype=dtype)
    v = torch.randn(1, nkv, seq, hd, device="mps", dtype=dtype)
    if nkv != nh:                       # GQA: transformers repeats KV heads
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    def run():
        F.scaled_dot_product_attention(q, k, v, is_causal=False)

    ctxm = None
    if backend is not None:
        from torch.nn.attention import sdpa_kernel
        ctxm = sdpa_kernel(backend)

    if ctxm is not None:
        ctxm.__enter__()
    try:
        for _ in range(5):
            run()
        sync_device()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            sync_device()
            ts.append(time.perf_counter() - t0)
    finally:
        if ctxm is not None:
            ctxm.__exit__(None, None, None)
    return st.median(ts) * 1e6          # microseconds


def probe_sdpa():
    from torch.nn.attention import SDPBackend

    nh, nkv, hd, nlayers = sdpa_shapes()
    print(f"Qwen3-4B decode attention: q[1,{nh},1,{hd}] kv[1,{nkv}->{nh},S,{hd}] "
          f"x {nlayers} layers, fp16 on mps\n")

    # dense grid straddling 2048 -- a dispatch threshold is a step, and a step
    # is only visible if the grid is finer than the effect
    seqs = [512, 1024, 1536, 1920, 2000, 2040, 2047, 2048, 2049, 2056, 2080,
            2176, 2560, 3072, 4096, 4104]

    print(f"  {'seq':>6} {'default(us)':>12} {'MATH(us)':>10} {'ratio':>7} "
          f"{'per-token x36 (ms)':>20}")
    for s in seqs:
        d = time_sdpa(nh, nkv, hd, s, torch.float16)
        try:
            m = time_sdpa(nh, nkv, hd, s, torch.float16, backend=SDPBackend.MATH)
        except (RuntimeError, ValueError):
            m = float("nan")
        print(f"  {s:>6} {d:>12.1f} {m:>10.1f} {m / d:>7.2f} "
              f"{d * nlayers / 1000:>20.2f}")

    print("\n  A backend switch at 2048 would show as a step in 'default' that "
          "\n  the neighbouring lengths do not have, or as ratio -> 1.00 there "
          "\n  (default having fallen back to MATH). The last column is the "
          "\n  most attention time a decode step could possibly be spending.")


# --- probe 3: where in the 150 steps does the cliff live? -------------------

def probe_steps(ctxs=(1024, 2048, 4096), n_new=150):
    import gc
    from transformers import AutoTokenizer, LogitsProcessorList
    from benchmark_quantization import (
        build_chat_input_ids, empty_cache, load_causal_lm, warmup_hf,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    for ctx in ctxs:
        # reload per ctx: the benchmark loads a fresh backend for every
        # measurement, and reusing one model here would test a protocol the
        # cliff was never observed under
        model = load_causal_lm(MODEL_ID, dtype=torch.float16)
        prompt = make_long_prompt(tok, ctx)
        warmup_hf(model, tok, prompt)
        input_ids = build_chat_input_ids(tok, model, prompt)
        stamps = []

        def _stamp(_ids, scores):
            sync_device()
            stamps.append(time.time())
            return scores

        sync_device()
        t0 = time.time()
        model.generate(input_ids=input_ids,
                       attention_mask=torch.ones_like(input_ids),
                       max_new_tokens=n_new, do_sample=False,
                       logits_processor=LogitsProcessorList([_stamp]))
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        q = lambda p: st.quantiles(gaps, n=100)[p - 1]
        deciles = [f"{st.median(gaps[i:i + len(gaps) // 10]) * 1000:.0f}"
                   for i in range(0, len(gaps) - 9, len(gaps) // 10)]
        print(f"ctx={ctx:<5} attn={model.config._attn_implementation}  "
              f"prefill={stamps[0] - t0:6.2f}s  "
              f"median={st.median(gaps) * 1000:7.1f}ms  p10={q(10) * 1000:7.1f}  "
              f"p90={q(90) * 1000:7.1f}  max={max(gaps) * 1000:7.1f}  "
              f"=> {1 / st.median(gaps):5.2f} tok/s")
        print(f"           per-decile median ms: {' '.join(deciles)}")
        del model
        gc.collect()
        empty_cache()

    print("\n  Flat deciles with a high median => per-step cost (shape/kernel)."
          "\n  Rising deciles or a fat max => state accumulating during decode.")


# --- probe 4: are the stalls the KV cache reallocating? ---------------------

def probe_cache(ctxs=(2048, 4096), n_new=150, repeats=2):
    """The mechanism test. HF's default DynamicCache grows the KV cache by
    torch.cat every decode step -- 150 steps x 36 layers x (K,V) is ~10k
    ever-larger allocations per generation, each a chance for the MPS allocator
    to go to the OS. llama.cpp instead reserves n_ctx up front, and GGUF is the
    engine that does not stall. StaticCache gives PyTorch the same up-front
    reservation, so if the stalls are reallocation, they should vanish.

    Runs dynamic/static interleaved with a fresh model per measurement (the
    benchmark's protocol), because a stall is a property of machine state and
    back-to-back runs would let the first warm the allocator for the second.
    """
    import gc
    import random
    from transformers import AutoTokenizer, LogitsProcessorList
    from benchmark_quantization import (
        PeakMemoryTracker, build_chat_input_ids, empty_cache, load_causal_lm,
        step_stats, warmup_hf,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tasks = [(c, k, r) for r in range(repeats) for c in ctxs
             for k in ("dynamic", "static")]
    random.Random(7).shuffle(tasks)

    out = {}
    for ctx, kind, rep in tasks:
        model = load_causal_lm(MODEL_ID, dtype=torch.float16)
        prompt = make_long_prompt(tok, ctx)
        warmup_hf(model, tok, prompt)
        input_ids = build_chat_input_ids(tok, model, prompt)
        stamps = []

        def _stamp(_ids, scores):
            sync_device()
            stamps.append(time.time())
            return scores

        kw = {"cache_implementation": "static"} if kind == "static" else {}
        with PeakMemoryTracker() as mem:
            sync_device()
            with torch.no_grad():
                model.generate(input_ids=input_ids,
                               attention_mask=torch.ones_like(input_ids),
                               max_new_tokens=n_new, do_sample=False,
                               logits_processor=LogitsProcessorList([_stamp]),
                               **kw)
            sync_device()

        s = step_stats(stamps)
        tps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        out.setdefault((ctx, kind), []).append((s, tps, mem.peak_gb()))
        print(f"  ran ctx={ctx} {kind:<8} rep{rep}  p50={s['step_ms_p50']:7.1f}  "
              f"max={s['step_ms_max']:8.1f}  tok/s={tps:5.2f}")
        del model
        gc.collect()
        empty_cache()

    print(f"\n  {'ctx':>5} {'cache':>9} {'p50(ms)':>9} {'p90(ms)':>9} "
          f"{'max(ms)':>9} {'tok/s':>7} {'스톨세금':>9} {'VRAM(GB)':>10}")
    for ctx in ctxs:
        for kind in ("dynamic", "static"):
            runs = out[(ctx, kind)]
            m = lambda f: st.median([f(r) for r in runs])
            p50, tps = m(lambda r: r[0]["step_ms_p50"]), m(lambda r: r[1])
            print(f"  {ctx:>5} {kind:>9} {p50:>9.1f} "
                  f"{m(lambda r: r[0]['step_ms_p90']):>9.1f} "
                  f"{m(lambda r: r[0]['step_ms_max']):>9.1f} {tps:>7.2f} "
                  f"{1000 / p50 / tps:>8.2f}x {m(lambda r: r[2]):>10.2f}")

    print("\n  static의 max가 p90 수준으로 내려오고 스톨 세금이 1.0x에 붙으면"
          "\n  스톨의 정체는 KV 캐시 재할당입니다. p50은 거의 그대로여야 합니다"
          "\n  (정상 상태 커널 비용은 캐시 구현과 무관) -- 그렇지 않다면 static이"
          "\n  다른 커널 경로를 타는 것이므로 비교 자체가 오염된 것입니다.")


# --- probe 5: do the stalls coincide with the driver taking new memory? -----

def probe_alloc(ctx=2048, n_new=150):
    """Poll the Metal driver's allocated bytes alongside every decode timestamp.

    torch's caching allocator serves most requests from its own pool without
    touching the driver; driver_allocated_memory() only grows when it has to go
    get more. So `slow step AND driver grew` is the signature of "the allocator
    went to the OS and the OS made it wait".
    """
    import gc
    from transformers import AutoTokenizer, LogitsProcessorList
    from benchmark_quantization import (
        build_chat_input_ids, empty_cache, load_causal_lm, warmup_hf,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_causal_lm(MODEL_ID, dtype=torch.float16)
    prompt = make_long_prompt(tok, ctx)
    warmup_hf(model, tok, prompt)
    input_ids = build_chat_input_ids(tok, model, prompt)

    stamps, drv = [], []

    def _stamp(_ids, scores):
        sync_device()
        stamps.append(time.time())
        drv.append(torch.mps.driver_allocated_memory())
        return scores

    sync_device()
    with torch.no_grad():
        model.generate(input_ids=input_ids,
                       attention_mask=torch.ones_like(input_ids),
                       max_new_tokens=n_new, do_sample=False,
                       logits_processor=LogitsProcessorList([_stamp]))
    sync_device()

    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    deltas = [(b - a) / 2**20 for a, b in zip(drv, drv[1:])]   # MiB
    p50 = st.median(gaps)
    slow = [i for i, g in enumerate(gaps) if g > 3 * p50]
    fast = [i for i, g in enumerate(gaps) if g <= 3 * p50]

    def summarize(idxs, name):
        if not idxs:
            print(f"  {name:<22} 0 steps")
            return
        grew = [i for i in idxs if deltas[i] > 0]
        print(f"  {name:<22} {len(idxs):>3} steps  "
              f"driver 증가한 스텝 {len(grew):>3} ({len(grew) / len(idxs):>4.0%})  "
              f"총 증가 {sum(deltas[i] for i in idxs):>8.0f} MiB  "
              f"시간 {sum(gaps[i] for i in idxs) / 1000:>6.1f}초")

    print(f"ctx={ctx}, {len(gaps)} 디코드 스텝, 스텝 p50={p50:.1f}ms "
          f"(스톨 기준: p50의 3배 = {3 * p50:.0f}ms)\n")
    summarize(slow, "느린 스텝(스톨)")
    summarize(fast, "정상 스텝")

    print("\n  가장 느린 10스텝:")
    print(f"    {'step':>5} {'지연(ms)':>10} {'driver 증가(MiB)':>18}")
    for i in sorted(range(len(gaps)), key=lambda i: -gaps[i])[:10]:
        print(f"    {i:>5} {gaps[i]:>10.0f} {deltas[i]:>18.1f}")

    print("\n  느린 스텝의 driver 증가 비율이 정상 스텝보다 뚜렷이 높으면"
          "\n  스톨 = 할당자가 OS로 나간 것. 두 비율이 비슷하면 스톨은 이미"
          "\n  들고 있는 메모리의 페이징이고, 원인은 할당이 아닙니다.")

    del model
    gc.collect()
    empty_cache()


# --- probe 6: what does the sweep leave behind between measurements? --------

def probe_residue(ctx=2048, n_new=150, rounds=4):
    """README's oldest Limitation says the sweep saturates its own machine and
    attributes it to "the file page cache filling up as the 8GB model is loaded
    and freed 100 times". That has been an *assumption* since v1. Now that a
    generation is known to grow driver memory by ~15GB, there is a competing
    explanation: the allocator does not give it all back.

    Replays the sweep's own load -> warm -> generate -> del/gc/empty_cache cycle
    and prints the driver's bytes at each boundary, so the residue per
    measurement is measured instead of assumed.
    """
    import gc
    from transformers import AutoTokenizer
    from benchmark_quantization import (
        empty_cache, load_causal_lm, measure_hf_generate, warmup_hf,
    )

    gb = lambda: torch.mps.driver_allocated_memory() / 2**30
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = gb()
    print(f"시작 driver 보유: {base:.2f} GB\n")
    print(f"  {'회차':>4} {'로드후':>9} {'생성후':>9} {'해제후':>9} "
          f"{'생성중 증가':>12} {'회수못한 잔여':>14}")

    prev = base
    for r in range(rounds):
        model = load_causal_lm(MODEL_ID, dtype=torch.float16)
        after_load = gb()
        prompt = make_long_prompt(tok, ctx)
        warmup_hf(model, tok, prompt)
        measure_hf_generate(model, tok, prompt, n_new)
        after_gen = gb()
        del model
        gc.collect()
        empty_cache()
        after_free = gb()
        print(f"  {r:>4} {after_load:>8.2f}G {after_gen:>8.2f}G {after_free:>8.2f}G "
              f"{after_gen - after_load:>11.2f}G {after_free - prev:>13.2f}G")
        prev = after_free

    print(f"\n  {rounds}회 누적 잔여: {prev - base:.2f} GB "
          f"(측정당 {(prev - base) / rounds * 1024:.0f} MB)")
    print("  측정당 잔여가 0에 가까우면 스윕 포화의 원인은 torch 할당자가 아니라"
          "\n  파일 페이지 캐시 쪽입니다(README Limitations의 기존 설명). 잔여가"
          "\n  쌓이면 기존 설명을 바꿔야 합니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe",
                    choices=["prompt", "sdpa", "steps", "cache", "alloc",
                             "residue"],
                    required=True)
    ap.add_argument("--ctx", type=int, default=2048,
                    help="--probe alloc / --probe residue only")
    args = ap.parse_args()
    if args.probe in ("alloc", "residue"):
        {"alloc": probe_alloc, "residue": probe_residue}[args.probe](ctx=args.ctx)
        return
    {"prompt": probe_prompt, "sdpa": probe_sdpa, "steps": probe_steps,
     "cache": probe_cache}[args.probe]()


if __name__ == "__main__":
    main()
