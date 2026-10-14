#!/usr/bin/env bash
# Detached waiter + chain for exp-033 (issue #60, ADR 0039 item 5, H4 ordering noise floor).
# exp-021's committed arm (Qwen2.5-7B-Instruct 4-bit, unguided, seed 42, pinned 600, exp-014
# reranked top-5 retrieval) with ONLY --ordering shuffled --ordering-seed S, S in 1 2 3.
# Code runs from the clean detached worktree $WT (HEAD 5c4e442); all data/run paths are
# ABSOLUTE into the main checkout's runs/.
#   wait: until runs/run.lock is absent AND GPU free memory >= MIN_FREE_MIB
#   lock: atomic create (noclobber) "exp-033 <ISO> experiment-runner"
#   per seed 1..3: decompose -> evaluate (exp-021's evaluator invocation)
#   compares are NOT run here (a later agent runs the analysis plan in the log row).
#   trap: removes the lock on any exit, but only if this lane holds it.
# Test mode: DRY_RUN=1 SCRATCH=<dir> -> no lock, no GPU wait, seed 1 only,
#   decompose --dry-run --dry-run-limit 3 into SCRATCH.
set -uo pipefail

EXP=exp-033
MAIN=/cta/users/fyilmaz/Thesis---QAv2
WT=/cta/users/fyilmaz/worktrees/exp-032-033
EXPECTED_SHA=5c4e442
PY=$MAIN/.venv/bin/python
LOCK=$MAIN/runs/run.lock
RUNS=$MAIN/runs
OUT=$RUNS/$EXP
RETR=$RUNS/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl
SEEDS=(1 2 3)
MIN_FREE_MIB=${MIN_FREE_MIB:-9000}
POLL_S=${POLL_S:-300}
DRY_RUN=${DRY_RUN:-0}
HOLD=0

if [ "$DRY_RUN" = 1 ]; then
  OUT=${SCRATCH:?SCRATCH required in dry-run mode}
  SEEDS=(1)
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
on_exit() { rc=$?; log "exit rc=$rc"; status "exit rc=$rc $(date +%Y-%m-%dT%H:%M:%S%z)"; release; }
trap on_exit EXIT
trap 'log "signal received"; exit 130' INT TERM HUP

cd "$WT" || { log "ERROR: worktree $WT missing"; exit 2; }
HEAD_SHA=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)
log "worktree HEAD=$HEAD_SHA dirty_lines=$DIRTY"
case "$HEAD_SHA" in $EXPECTED_SHA*) ;; *) log "ERROR: worktree not at $EXPECTED_SHA"; exit 2;; esac
[ "$DIRTY" = 0 ] || { log "ERROR: worktree dirty"; exit 2; }
[ -f "$RETR" ] || { log "ERROR: retrieval input $RETR missing"; exit 2; }

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
for S in "${SEEDS[@]}"; do
  D=$OUT/seed$S
  mkdir -p "$D"
  log "=== seed $S: decompose ==="
  T0=$(date +%s)
  EXTRA=()
  [ "$DRY_RUN" = 1 ] && EXTRA=(--dry-run --dry-run-limit 3)
  PYTHONUNBUFFERED=1 "$PY" components/decomposer/run_decomposer.py --model qwen2_5_7b_instruct \
    --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
    --ordering shuffled --ordering-seed "$S" \
    --retrieval-input "$RETR" --output-root "$D/decomposer" "${EXTRA[@]}" >> "$LOG" 2>&1 &
  DP=$!
  PEAK=0
  while kill -0 "$DP" 2>/dev/null; do
    U=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$DP" '$1==p {print $2}')
    if [ -n "${U:-}" ] && [ "$U" -gt "$PEAK" ] 2>/dev/null; then PEAK=$U; fi
    sleep 10
  done
  wait "$DP"; rc=$?
  echo "$PEAK" > "$D/peak_vram_mib.txt"
  log "seed $S decompose rc=$rc wall_s=$(( $(date +%s)-T0 )) peak_vram_mib=$PEAK"
  if [ $rc -ne 0 ]; then status "seed$S DECOMPOSE_FAIL rc=$rc"; exit 4; fi
  RES="$(ls -td "$D"/decomposer/*/ 2>/dev/null | head -1)results.json"
  [ -f "$RES" ] || { log "ERROR: $RES missing"; status "seed$S NO_RESULTS"; exit 4; }
  log "results: $RES"
  if [ "$DRY_RUN" = 1 ]; then status "seed$S DRY_DECOMPOSE_OK"; continue; fi

  log "=== seed $S: evaluate ==="
  "$PY" scripts/musique_decompositions_evaluator.py --predictions "$RES" --run-dir "$D/eval" --out-prefix eval --seed 42 >> "$LOG" 2>&1
  rc=$?
  log "seed $S evaluate rc=$rc"
  if [ $rc -ne 0 ]; then status "seed$S EVAL_FAIL rc=$rc"; exit 5; fi
  status "seed$S OK $(date +%Y-%m-%dT%H:%M:%S%z)"
done

if [ "$DRY_RUN" = 1 ]; then log "dry run complete"; exit 0; fi
log "=== chain done total_wall_s=$(( $(date +%s)-CHAIN_START )) ==="
status "chain done $(date +%Y-%m-%dT%H:%M:%S%z)"
exit 0
