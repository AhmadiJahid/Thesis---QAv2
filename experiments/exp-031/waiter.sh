#!/usr/bin/env bash
# Detached waiter + chain for exp-031 (issue #51, v2 pool-size sweep 2k-10k).
# Config: configs/pool_size_sweep_v2.json at 3bd3595. Code runs from the clean detached
# worktree $WT (HEAD 3bd3595); all data/run paths are ABSOLUTE into the main checkout's runs/.
#   wait: until runs/run.lock is absent AND GPU free memory >= MIN_FREE_MIB
#   lock: atomic create (noclobber) "exp-031 <ISO> experiment-runner"
#   per size 2000..10000: preflight (abort chain on refusal) -> decompose -> evaluate
#   compares: sizeN vs size2000 (8 pairs) + size2000 vs exp-020
#   trap: removes the lock on any exit, but only if this lane holds it.
# Test mode: DRY_RUN=1 SCRATCH=<dir> -> no lock, no GPU wait, size 2000 only,
#   preflight + decompose --dry-run --dry-run-limit 3 into SCRATCH.
set -uo pipefail

EXP=exp-031
MAIN=/cta/users/fyilmaz/Thesis---QAv2
WT=/cta/users/fyilmaz/worktrees/exp-031
EXPECTED_SHA=3bd3595e48ca8976e1748e566eaadfc03f3ae967
PY=$MAIN/.venv/bin/python
LOCK=$MAIN/runs/run.lock
RUNS=$MAIN/runs
OUT=$RUNS/$EXP
SIZES=(2000 3000 4000 5000 6000 7000 8000 9000 10000)
MIN_FREE_MIB=${MIN_FREE_MIB:-9000}
POLL_S=${POLL_S:-300}
DRY_RUN=${DRY_RUN:-0}
HOLD=0

if [ "$DRY_RUN" = 1 ]; then
  OUT=${SCRATCH:?SCRATCH required in dry-run mode}
  SIZES=(2000)
fi
mkdir -p "$OUT"
LOG=$OUT/run.log
STATUS=$OUT/chain_status.txt

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] $*" | tee -a "$LOG"; }
status() { echo "$*" >> "$STATUS"; }

release() {
  if [ "$HOLD" = 1 ] && [ -e "$LOCK" ] && head -c ${#EXP} "$LOCK" | grep -qx "$EXP"; then
    rm -f "$LOCK"; log "lock released"
  fi
}
on_exit() { rc=$?; log "exit rc=$rc"; release; }
trap on_exit EXIT
trap 'log "signal received"; exit 130' INT TERM HUP

cd "$WT" || { log "ERROR: worktree $WT missing"; exit 2; }
HEAD_SHA=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)
log "worktree HEAD=$HEAD_SHA dirty_lines=$DIRTY"
case "$HEAD_SHA" in $EXPECTED_SHA*) ;; *) log "ERROR: worktree not at $EXPECTED_SHA"; exit 2;; esac
[ "$DIRTY" = 0 ] || { log "ERROR: worktree dirty"; exit 2; }

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

if [ "$DRY_RUN" != 1 ]; then
  log "waiting for lock-free AND gpu free >= ${MIN_FREE_MIB} MiB (poll ${POLL_S}s)"
  status "waiting since $(date +%Y-%m-%dT%H:%M:%S%z)"
  while :; do
    if [ ! -e "$LOCK" ]; then
      F=$(gpu_free)
      if [ -n "$F" ] && [ "$F" -ge "$MIN_FREE_MIB" ] 2>/dev/null; then
        if ( set -o noclobber; echo "$EXP $(date +%Y-%m-%dT%H:%M:%S%z) experiment-runner" > "$LOCK" ) 2>/dev/null; then
          HOLD=1; log "acquired lock: $(cat "$LOCK") (gpu free ${F} MiB)"; break
        else
          log "lock race lost; keep waiting"
        fi
      else
        log "lock free but gpu free=${F:-?} MiB < ${MIN_FREE_MIB}; waiting"
      fi
    fi
    sleep "$POLL_S"
  done
  status "chain start $(date +%Y-%m-%dT%H:%M:%S%z) HEAD=$HEAD_SHA"
fi

CHAIN_START=$(date +%s)
for N in "${SIZES[@]}"; do
  D=$OUT/size$N
  mkdir -p "$D"
  log "=== size $N: preflight ==="
  "$PY" scripts/preflight_pool_sweep.py --config pool_size_sweep_v2.json --size "$N" --root "$MAIN" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "PREFLIGHT REFUSED size $N rc=$rc - aborting chain"; status "size$N PREFLIGHT_FAIL rc=$rc"; exit 3; fi

  log "=== size $N: decompose ==="
  T0=$(date +%s)
  EXTRA=()
  [ "$DRY_RUN" = 1 ] && EXTRA=(--dry-run --dry-run-limit 3)
  PYTHONUNBUFFERED=1 "$PY" components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit --ordering as_is \
    --retrieval-input "$RUNS/pool_size_sweep_v2/retrieval/size$N/pinned600_imbalanced_rerank_top5.jsonl" \
    --output-root "$D/decomposer" "${EXTRA[@]}" >> "$LOG" 2>&1 &
  DP=$!
  PEAK=0
  while kill -0 "$DP" 2>/dev/null; do
    U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$DP" '$1==p {print $2}')
    if [ -n "${U:-}" ] && [ "$U" -gt "$PEAK" ] 2>/dev/null; then PEAK=$U; fi
    sleep 10
  done
  wait "$DP"; rc=$?
  echo "$PEAK" > "$D/peak_vram_mib.txt"
  log "size $N decompose rc=$rc wall_s=$(( $(date +%s)-T0 )) peak_vram_mib=$PEAK"
  if [ $rc -ne 0 ]; then status "size$N DECOMPOSE_FAIL rc=$rc"; exit 4; fi
  RES="$(ls -td "$D"/decomposer/*/ 2>/dev/null | head -1)results.json"
  [ -f "$RES" ] || { log "ERROR: $RES missing"; status "size$N NO_RESULTS"; exit 4; }
  log "results: $RES"
  if [ "$DRY_RUN" = 1 ]; then status "size$N DRY_DECOMPOSE_OK"; continue; fi

  log "=== size $N: evaluate ==="
  "$PY" scripts/musique_decompositions_evaluator.py --predictions "$RES" --run-dir "$D/eval" --out-prefix eval --seed 42 >> "$LOG" 2>&1
  rc=$?
  log "size $N evaluate rc=$rc"
  if [ $rc -ne 0 ]; then status "size$N EVAL_FAIL rc=$rc"; exit 5; fi
  status "size$N OK $(date +%Y-%m-%dT%H:%M:%S%z)"
done

if [ "$DRY_RUN" = 1 ]; then log "dry run complete"; exit 0; fi

CFAIL=0
for N in 3000 4000 5000 6000 7000 8000 9000 10000; do
  log "=== compare size$N vs size2000 ==="
  "$PY" scripts/musique_decompositions_evaluator.py --compare "$OUT/size$N/eval/eval_per_item.json" "$OUT/size2000/eval/eval_per_item.json" \
    --run-dir "$OUT/compare/size${N}_vs_size2000" --out-prefix compare --seed 42 >> "$LOG" 2>&1
  rc=$?; log "compare size$N rc=$rc"; [ $rc -eq 0 ] || CFAIL=1
done
log "=== compare size2000 vs exp-020 (reproducibility) ==="
"$PY" scripts/musique_decompositions_evaluator.py --compare "$OUT/size2000/eval/eval_per_item.json" "$RUNS/exp-020/eval/eval_per_item.json" \
  --run-dir "$OUT/compare/size2000_vs_exp020" --out-prefix compare --seed 42 >> "$LOG" 2>&1
rc=$?; log "compare size2000 vs exp-020 rc=$rc"; [ $rc -eq 0 ] || CFAIL=1

log "=== chain done CFAIL=$CFAIL total_wall_s=$(( $(date +%s)-CHAIN_START )) ==="
status "chain done CFAIL=$CFAIL $(date +%Y-%m-%dT%H:%M:%S%z)"
[ $CFAIL -eq 0 ] || exit 6
exit 0
