#!/usr/bin/env bash
# Detached waiter + chain TEMPLATE, adapted from experiments/exp-033/waiter.sh (the committed
# original; runs/exp-031/waiter.sh is the same pattern). Copy to experiments/<exp-id>/waiter.sh,
# fill the FILL-IN lines and the steps in run_chain(), commit it with the pre-run log row.
# Code runs from a clean detached worktree $WT at $EXPECTED_SHA; all data/run paths are ABSOLUTE
# into the main checkout's runs/.
#   wait: until runs/run.lock is absent AND GPU free memory >= MIN_FREE_MIB (configurable env var)
#   lock: atomic create (noclobber) "<exp-id> <ISO> $LAUNCHER"   (docs/compute.md; LAUNCHER
#         defaults to experiment-runner)
#   status: one line per step in $OUT/chain_status.txt; last line is always "exit rc=N <ISO>"
#   trap: removes the lock on any exit, but only if this waiter holds it.
# Test mode: DRY_RUN=1 SCRATCH=<dir> -> no lock, no GPU wait, steps get --dry-run flags.
# Refuses (exit 2, message on stderr, nothing created under runs/) when EXPECTED_SHA is empty,
# when a real run still has only the placeholder step, or when the worktree is missing, at the
# wrong SHA or dirty. The dry run is where such a refusal is seen; a detached tmux run hides stderr.
set -uo pipefail

EXP=exp-NNN                                          # FILL-IN
MAIN=/cta/users/fyilmaz/Thesis---QAv2
WT=/cta/users/fyilmaz/worktrees/$EXP                 # FILL-IN if different
EXPECTED_SHA=                                        # FILL-IN: merged SHA the worktree is at
PY=$MAIN/.venv/bin/python
LOCK=$MAIN/runs/run.lock
RUNS=$MAIN/runs
OUT=$RUNS/$EXP
MIN_FREE_MIB=${MIN_FREE_MIB:-9000}                   # exp-031/exp-033 value; size it to the model
POLL_S=${POLL_S:-300}
DRY_RUN=${DRY_RUN:-0}
LAUNCHER=${LAUNCHER:-experiment-runner}
HOLD=0

[ "$DRY_RUN" = 1 ] && OUT=${SCRATCH:?SCRATCH required in dry-run mode}
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

# --- validation, before anything is created under runs/ or the lock is touched ---
die() { echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] ERROR: $*" >&2; exit 2; }
[ -n "$EXPECTED_SHA" ] || die "EXPECTED_SHA is empty; fill it in"
SELF=$(readlink -f "${BASH_SOURCE[0]}")
if [ "$DRY_RUN" != 1 ] && grep -qE '^[[:space:]]*step placeholder ' "$SELF"; then
  die "run_chain() still has the placeholder step; fill in the real steps and delete it"
fi
cd "$WT" 2>/dev/null || die "worktree $WT missing"
HEAD_SHA=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)
case "$HEAD_SHA" in "$EXPECTED_SHA"*) ;; *) die "worktree HEAD=$HEAD_SHA not at $EXPECTED_SHA";; esac
[ "$DIRTY" = 0 ] || die "worktree dirty ($DIRTY lines)"

mkdir -p "$OUT"
trap on_exit EXIT
trap 'log "signal received"; exit 130' INT TERM HUP
log "worktree HEAD=$HEAD_SHA dirty_lines=$DIRTY"

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

if [ "$DRY_RUN" != 1 ]; then
  log "waiting for lock-free AND gpu free >= ${MIN_FREE_MIB} MiB (poll ${POLL_S}s)"
  status "waiting since $(date +%Y-%m-%dT%H:%M:%S%z)"
  while :; do
    if [ ! -e "$LOCK" ]; then
      F=$(gpu_free)
      if [ -n "$F" ] && [ "$F" -ge "$MIN_FREE_MIB" ] 2>/dev/null; then
        if ( set -o noclobber; echo "$EXP $(date +%Y-%m-%dT%H:%M:%S%z) $LAUNCHER" > "$LOCK" ) 2>/dev/null; then
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

# step NAME CMD... : runs one step, logs rc and wall time, writes "<NAME> OK|FAIL" to the status
# file, and stops the chain on failure (exit code 4).
step() {
  local name=$1; shift; local t0; t0=$(date +%s)
  log "=== $name ==="
  PYTHONUNBUFFERED=1 "$@" >> "$LOG" 2>&1; local rc=$?
  log "$name rc=$rc wall_s=$(( $(date +%s)-t0 ))"
  if [ $rc -ne 0 ]; then status "$name FAIL rc=$rc"; exit 4; fi
  status "$name OK $(date +%Y-%m-%dT%H:%M:%S%z)"
}

run_chain() {
  local extra=()
  [ "$DRY_RUN" = 1 ] && extra=(--dry-run --dry-run-limit 3)
  # FILL-IN: the committed commands from the log row, one `step` each, e.g.
  # step decompose "$PY" components/decomposer/run_decomposer.py --model ... --seed 42 \
  #   --output-root "$OUT/decomposer" "${extra[@]}"
  # [ "$DRY_RUN" = 1 ] && return 0
  # RES="$(ls -td "$OUT"/decomposer/*/ | head -1)results.json"
  # step evaluate "$PY" scripts/musique_decompositions_evaluator.py --predictions "$RES" \
  #   --run-dir "$OUT/eval" --out-prefix eval --seed 42
  step placeholder true "${extra[@]}"   # DELETE when the real steps are filled in
}

run_chain
[ "$DRY_RUN" = 1 ] && { log "dry run complete"; exit 0; }
status "chain done $(date +%Y-%m-%dT%H:%M:%S%z)"
exit 0
