#!/bin/bash
# 2048-token stall recheck: apps closed (phase 1), then a fixed set of apps open (phase 2).
# Plan and pass/fail rules were fixed on 2026-10-16 before any measurement.
# Run from the repository root, plugged in, lid open:
#   caffeinate -dimsu bash tools/recheck_run.sh
# Options (environment): WAIT_MIN=10  SKIP_PHASE2=1  PY=.venv/bin/python
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=results/recheck-$(date +%Y%m%d)
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

block() {      # phase mem seed label : one shuffled pass over the 4 cells
  local phase=$1 mem=$2 seed=$3 label=$4 ballast=0
  [ "$mem" = 16 ] && ballast=8
  snapshot "${phase}_${seed}_${mem}GB"
  log "$phase ${mem}GB seed=$seed start"
  "$PY" benchmark_quantization.py --stage context_sweep --repeats 1 \
      --configs FP16,GGUF_Q4_K_M --contexts 1024,2048 \
      --seed "$seed" --label "$label" --ballast-gb "$ballast" \
      --out "$OUT/$phase" >> "$LOG" 2>&1
  log "$phase ${mem}GB seed=$seed done (exit $?)"
}

ORDER="24 16 16 24 24 16"      # A B B A A B: spreads time drift over both memory sizes
log "recheck start; output $OUT; waiting ${WAIT_MIN:-10} min for the system to settle"
"$PY" -c "import torch, llama_cpp; print('mps', torch.backends.mps.is_available())" >> "$LOG" 2>&1 \
  || { log "python environment check failed; stopping"; exit 1; }
sleep $(( ${WAIT_MIN:-10} * 60 ))

i=1; for m in $ORDER; do block phase1 "$m" $((100 + i)) apps_closed; i=$((i + 1)); done

if [ "${SKIP_PHASE2:-0}" != 1 ]; then
  open -a "Google Chrome" \
    "https://github.com/AhnJG/qwen3-4b-mac-quantization-benchmark" \
    "https://www.youtube.com" "https://mail.google.com" "https://www.naver.com" "https://claude.ai"
  open -a "Claude"
  log "phase 2 apps opened (Chrome 5 tabs + Claude app); waiting 5 min"
  sleep 300
  i=1; for m in $ORDER; do block phase2 "$m" $((200 + i)) apps_fixed; i=$((i + 1)); done
fi

for p in phase1 phase2; do
  [ -f "$OUT/$p/results.jsonl" ] && "$PY" tools/audit.py "$OUT/$p/results.jsonl" --repeats 3 \
      --configs FP16,GGUF_Q4_K_M --contexts 1024,2048 --interleaved > "$OUT/audit_$p.txt" 2>&1
done
"$PY" tools/recheck_summary.py "$OUT" > "$OUT/summary.txt" 2>&1
log "all done; see $OUT/summary.txt"
