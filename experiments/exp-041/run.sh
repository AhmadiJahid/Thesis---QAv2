#!/usr/bin/env bash
# exp-041: training-seed variance of exp-026's pool-only fine-tune (Mistral-7B QLoRA, 250 steps),
# seeds 42/43/44 on one GPU, each evaluated exactly as exp-026 was; plus exp-015's prompting
# command re-run on the same GPU as the comparator. Plan: experiments/exp-041/run_config.json.
# Run from the repo root of a CLEAN checkout with the data folder next to the clone and these
# stored runs restored under runs/: pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl,
# exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl, exp-032/exp-0{15,26}/{eval,reparse}.
#
#   bash experiments/exp-041/run.sh <step> [seed]
#
# steps, in this order (the notebook runs them so):
#   canary_prompted        20-row decode of exp-015's command + gate (environment check)
#   train <seed>           train_lora.py --seed <seed> + training gate (an interrupted run restarts at step 0)
#   eval <seed>            20-row canary with the adapter + gate, then the full 600 + gate + evaluator
#   prompted               exp-015's exact command, full 600 + gate + evaluator
#   compare                experiments/exp-041/compare.py -> experiments/exp-041/compare_metrics.json
#
# A finished step is never redone; an interrupted decode resumes (run_decomposer.py --resume).
# PY defaults to "python"; the notebook sets PY to its Python 3.13.9 venv.
set -euo pipefail
PY=${PY:-python}
R=runs/exp-041
RETR=runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl
RETR_SHA=3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0
POOL=runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl
POOL_SHA=cc3646625d759a1b58179b18c25f83d255db24f05acc08b666985af2e58d6646
DEC=(components/decomposer/run_decomposer.py --model mistral_7b_instruct --config decomposer_musique.json
     --condition unguided --seed 42 --retrieval-input "$RETR")

log() { echo "[exp-041 $(date -Iseconds)] $*"; }
sha() { $PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

preflight() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=no)
  if [ -n "$dirty" ]; then
    echo "refusing: tracked files differ from the commit (Gate 2 - runs come from committed code):" >&2
    echo "$dirty" >&2; exit 2
  fi
  COMMIT=$(git rev-parse HEAD); log "git $COMMIT (clean)"
  [ "$(sha "$RETR")" = "$RETR_SHA" ] || { echo "retrieval artifact sha256 mismatch / missing" >&2; exit 2; }
  [ "$(sha "$POOL")" = "$POOL_SHA" ] || { echo "pool sha256 mismatch / missing" >&2; exit 2; }
}

check_seed() { case "${1:-}" in 42|43|44) ;; *) echo "seed must be one of 42 43 44 (run_config.json training_seeds)" >&2; exit 2 ;; esac; }

completed_train() {  # $1 seed -> prints the one completed training dir (metrics.json + adapter), if any
  local d
  for d in "$R/seed$1/train"/*/; do
    [ -f "${d}metrics.json" ] && [ -f "${d}adapter/adapter_model.safetensors" ] && echo "${d%/}"
  done 2>/dev/null || true
}

train() {
  local s=$1 done_dirs d
  check_seed "$s"; preflight
  done_dirs=$(completed_train "$s")
  if [ -z "$done_dirs" ]; then
    mkdir -p "$R/seed$s/train" "$R/seed$s/train_incomplete"
    for d in "$R/seed$s/train"/*/; do [ -d "$d" ] && mv "$d" "$R/seed$s/train_incomplete/"; done
    log "train seed $s (exp-026's command, --seed $s)"
    $PY components/decomposer/train_lora.py --arm pool_2000_imbalanced --quantization 4bit --seed "$s" \
      --output-root "$R/seed$s/train"
    done_dirs=$(completed_train "$s")
  fi
  [ "$(printf '%s\n' "$done_dirs" | grep -c .)" -eq 1 ] || { echo "expected one completed training dir, got: $done_dirs" >&2; exit 2; }
  $PY experiments/exp-041/gates.py train --run-dir "$done_dirs" --seed "$s" --commit "$COMMIT" \
    --out "$R/seed$s/train_gate.json" || { echo "TRAIN GATE FAILED seed $s - do not evaluate" >&2; exit 3; }
  log "trained seed $s: $done_dirs"
}

decode_full() {  # $1 out dir root (…/decomposer), $2 role, rest: extra args
  local root=$1 role=$2; shift 2
  local dirs n
  mkdir -p "$root"
  dirs=$(find "$root" -mindepth 1 -maxdepth 1 -type d | sort)
  n=$(printf '%s' "$dirs" | grep -c . || true)
  if [ "$n" -gt 1 ]; then echo "$root holds $n run dirs; keep exactly one" >&2; exit 2; fi
  if [ "$n" -eq 1 ] && [ -f "$dirs/results.json" ]; then log "decode already complete: $dirs"
  elif [ "$n" -eq 1 ] && [ -f "$dirs/resume_key.json" ]; then log "resuming $dirs"; $PY "${DEC[@]}" "$@" --resume "$dirs"
  elif [ "$n" -eq 1 ]; then echo "$dirs is neither complete nor resumable" >&2; exit 2
  else $PY "${DEC[@]}" "$@" --output-root "$root"; fi
  dirs=$(find "$root" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  $PY experiments/exp-041/gates.py decode --run-dir "$dirs" --kind full --role "$role" --commit "$COMMIT" \
    --out "$(dirname "$root")/decode_gate.json" || { echo "DECODE GATE FAILED ($root) - not evaluated" >&2; exit 3; }
  if [ -f "$(dirname "$root")/eval/eval_metrics.json" ]; then log "eval already complete"
  else $PY scripts/musique_decompositions_evaluator.py --predictions "$dirs/results.json" \
         --run-dir "$(dirname "$root")/eval" --out-prefix eval --seed 42; fi
}

canary() {  # $1 name, $2 role, rest: extra args
  local name=$1 role=$2; shift 2
  local c="$R/canary/$name" d
  if [ -f "$c/gate.json" ] && grep -q '"ok": true' "$c/gate.json"; then log "canary $name already passed"; return; fi
  rm -rf "$c"; mkdir -p "$c"
  log "canary $name (20 rows; not an experiment arm)"
  $PY "${DEC[@]}" "$@" --sample-size 20 --allow-unpinned-eval-set --output-root "$c/decomposer"
  d=$(find "$c/decomposer" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  $PY experiments/exp-041/gates.py decode --run-dir "$d" --kind canary --role "$role" --commit "$COMMIT" \
    --out "$c/gate.json" || { echo "CANARY FAILED ($name)" >&2; exit 3; }
}

eval_seed() {
  local s=$1 tdir adapter
  check_seed "$s"; preflight
  [ -f "$R/seed$s/train_gate.json" ] && grep -q '"ok": true' "$R/seed$s/train_gate.json" \
    || { echo "seed $s has no passed training gate - run: run.sh train $s" >&2; exit 2; }
  tdir=$(completed_train "$s"); adapter="$tdir/adapter"
  if [ ! -f "$R/seed$s/eval/eval_metrics.json" ]; then
    canary "eval_seed$s" adapter --quantization none --adapter "$adapter" --no-few-shot
  fi
  decode_full "$R/seed$s/decomposer" adapter --quantization none --adapter "$adapter" --no-few-shot
  log "done eval seed $s"
}

case "${1:-}" in
  canary_prompted) preflight; canary prompted prompted ;;
  train) train "${2:-}" ;;
  eval) eval_seed "${2:-}" ;;
  prompted) preflight; decode_full "$R/prompted_exp015_rerun/decomposer" prompted ;;
  compare) preflight; $PY experiments/exp-041/compare.py ;;
  *) echo "usage: $0 {canary_prompted|train <seed>|eval <seed>|prompted|compare}" >&2; exit 2 ;;
esac
