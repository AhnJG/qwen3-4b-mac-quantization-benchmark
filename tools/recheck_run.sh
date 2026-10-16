#!/bin/bash
# 2048-token stall recheck: apps closed (phase 1), then a fixed set of apps open (phase 2).
# Plan and pass/fail rules were fixed on 2026-10-16 before any measurement.
#
# Real run (plugged in, lid OPEN, only Terminal open):
#   caffeinate -dims bash tools/recheck_run.sh
# Smoke test (~5 min, apps may be open; separate folder; not used for any conclusion):
#   SMOKE=1 bash tools/recheck_run.sh
# Options (environment): WAIT_MIN=10  SKIP_PHASE2=1  PY=.venv/bin/python
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # use the cached models only
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then OUT=results/recheck-smoke-$(date +%Y%m%d-%H%M); else OUT=results/recheck-$(date +%Y%m%d-%H%M); fi
[ -e "$OUT" ] && { echo "$OUT already exists; refusing to append"; exit 1; }
mkdir -p "$OUT/logs"
LOG="$OUT/logs/run.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

snapshot() {   # read-only: apps, top memory users, thermal state, swap, free memory
  {
    date '+%F %T'
    echo "## GUI apps"; ps -axo comm | grep '\.app/Contents/MacOS/' | sed 's#.*/\([^/]*\.app\)/.*#\1#' | sort -u
    echo "## top 10 memory (RSS KB)"; ps -axo rss,comm | sort -nr | head -10
    echo "## thermal"; pmset -g therm
    echo "## swap"; sysctl vm.swapusage
    echo "## memorystatus_level (% free)"; sysctl -n kern.memorystatus_level
    echo "## vm_stat"; vm_stat
  } > "$OUT/logs/snapshot_$1.txt" 2>&1
}

cleanup_ballast() {   # a killed run cannot run its own `finally`; never let 8GB leak into the next block
  if pgrep -f tools/ballast.py >/dev/null; then
    log "leftover ballast found; killing"; pkill -f tools/ballast.py; sleep 3
  fi
}

block() {      # phase mem seed label ctxs : one shuffled pass over the cells
  local phase=$1 mem=$2 seed=$3 label=$4 ctxs=$5 ballast=0 rc
  [ "$mem" = 16 ] && ballast=8
  cleanup_ballast
  snapshot "${phase}_${seed}_${mem}GB"
  log "$phase ${mem}GB seed=$seed start"
  "$PY" benchmark_quantization.py --stage context_sweep --repeats 1 \
      --configs FP16,GGUF_Q4_K_M --contexts "$ctxs" \
      --seed "$seed" --label "$label" --ballast-gb "$ballast" \
      --out "$OUT/$phase" >> "$LOG" 2>&1
  rc=$?
  cleanup_ballast
  log "$phase ${mem}GB seed=$seed done (exit $rc, rows so far: $(wc -l < "$OUT/$phase/results.jsonl" 2>/dev/null || echo 0))"
}

# ---- record the environment (fills gaps the v7 run left unrecorded)
{
  echo "## date"; date '+%F %T %Z'
  echo "## macOS"; sw_vers
  echo "## commit"; git rev-parse HEAD; git status --porcelain
  echo "## disk"; df -h /
  echo "## RAM"; sysctl -n hw.memsize
  echo "## packages"; "$PY" -m pip freeze
  echo "## model files"; ls -lL ~/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/*/ 2>&1
  ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/ 2>&1
} > "$OUT/logs/environment.txt" 2>&1

# ---- preflight: environment and both models must load offline
"$PY" - >> "$LOG" 2>&1 <<'PYEOF' || { log "preflight failed; stopping (see $LOG)"; exit 1; }
import torch, llama_cpp, sys
sys.path.insert(0, ".")
import benchmark_quantization as b
from transformers import AutoTokenizer
assert torch.backends.mps.is_available(), "MPS not available"
print("q4", b.download_gguf(b.GGUF_FILENAME_HINT))
print("f16", b.download_gguf(b.GGUF_F16_FILENAME_HINT))
AutoTokenizer.from_pretrained(b.MODEL_ID)
print("preflight ok")
PYEOF
log "preflight ok; output $OUT"

if [ "$SMOKE" = 1 ]; then
  block smoke 16 101 smoke 1024
  "$PY" - "$OUT/smoke/results.jsonl" <<'PYEOF' | tee -a "$LOG"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    print(r["config"], r["context_len"], "ballast", r["ballast_gb"], "tps", round(r["tokens_per_sec"], 2),
          "free_start", r["mem_free_gb_start"], "free_end", r["mem_free_gb_end"],
          "level", r["memorystatus_level_end"], "label", r["label"])
print("SMOKE OK" if len(rows) == 2 and all(r["mem_free_gb_start"] is not None for r in rows) else "SMOKE PROBLEM")
PYEOF
  exit 0
fi

# ---- real run
if pgrep -x "Google Chrome" >/dev/null || pgrep -x Claude >/dev/null; then
  log "Chrome or the Claude app is running; phase 1 needs them closed. Stopping."; exit 1
fi
ORDER="24 16 16 24 24 16"      # A B B A A B: spreads time drift over both memory sizes
SEEDS="101 102 103 104 105 106" # same seeds in both phases, so the task order is paired
log "waiting ${WAIT_MIN:-10} min for the system to settle"
sleep $(( ${WAIT_MIN:-10} * 60 ))

set -- $SEEDS
for m in $ORDER; do block phase1 "$m" "$1" apps_closed 1024,2048; shift; done

if [ "${SKIP_PHASE2:-0}" != 1 ]; then
  open -a "Google Chrome" \
    "https://github.com/AhnJG/qwen3-4b-mac-quantization-benchmark" \
    "https://www.youtube.com" "https://mail.google.com" "https://www.naver.com" "https://claude.ai"
  open -a "Claude"
  log "phase 2 apps opened (Chrome 5 tabs + Claude app); waiting 5 min"
  sleep 300
  snapshot phase2_apps_opened
  set -- $SEEDS
  for m in $ORDER; do block phase2 "$m" "$1" apps_fixed 1024,2048; shift; done
fi

for p in phase1 phase2; do
  [ -f "$OUT/$p/results.jsonl" ] && "$PY" tools/audit.py "$OUT/$p/results.jsonl" --repeats 3 \
      --configs FP16,GGUF_Q4_K_M --contexts 1024,2048 --interleaved > "$OUT/audit_$p.txt" 2>&1
done
"$PY" tools/recheck_summary.py "$OUT" > "$OUT/summary.txt" 2>&1
log "all done; see $OUT/summary.txt"
