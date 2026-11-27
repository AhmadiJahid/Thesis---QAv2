#!/usr/bin/env bash
# exp-037: count-first decomposition arm (memo C1.3) vs the unguided arm, same everything else.
# Run from the repo root of a CLEAN checkout of feature/37-count-first (or its merge), with the
# data folder next to the clone (docs/compute.md "Dataset location") and the stored runs listed
# in experiments/exp-037/README.md restored under runs/.
#
#   bash experiments/exp-037/run.sh <step>
#
# steps (run in this order; all four decode steps are required by the pre-declared design):
#   mistral_count_first      decompose + evaluate
#   mistral_unguided_rerun   decompose + evaluate (exp-020's exact command on this machine:
#                                                   the primary comparator AND the hardware control)
#   qwen_count_first         decompose + evaluate
#   qwen_unguided_rerun      decompose + evaluate (same, for exp-027)
#   compare                  every --compare whose two inputs exist
#   metrics                  experiments/exp-037/build_metrics.py -> experiments/exp-037/metrics.json
#
# A decode step is RESUMABLE: re-running it after an interruption continues the unfinished run
# directory with run_decomposer.py --resume (which refuses unless code, prompt, seed, retrieval
# and decoding are identical). A finished decode is never redone; a finished eval is not redone.
#
# PY defaults to "python"; set PY=.venv/bin/python on a box with a venv.
set -euo pipefail
PY=${PY:-python}
R=runs/exp-037
RETR=runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl
RETR_SHA=3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0
BASE_MISTRAL=runs/exp-032/exp-020/eval/eval_per_item.json   # exp-020 re-parsed (ADR 0039)
BASE_QWEN=runs/exp-032/exp-027/eval/eval_per_item.json      # exp-027 re-parsed (0 rows changed)

log() { echo "[exp-037 $(date -Iseconds)] $*"; }

preflight() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=no)
  if [ -n "$dirty" ]; then
    echo "refusing: tracked files differ from the commit (Gate 2 - runs come from committed code):" >&2
    echo "$dirty" >&2
    exit 2
  fi
  log "git $(git rev-parse HEAD) (clean)"
  [ -f "$RETR" ] || { echo "missing $RETR" >&2; exit 2; }
  got=$($PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$RETR")
  [ "$got" = "$RETR_SHA" ] || { echo "retrieval sha256 $got != $RETR_SHA" >&2; exit 2; }
  log "retrieval artifact sha256 OK"
}

decompose_eval() {  # $1 arm dir name, $2 model folder, $3 condition
  local arm=$1 model=$2 cond=$3 dirs n res
  preflight
  mkdir -p "$R/$arm/decomposer"
  dirs=$(find "$R/$arm/decomposer" -mindepth 1 -maxdepth 1 -type d | sort)
  n=$(printf '%s' "$dirs" | grep -c . || true)
  if [ "$n" -gt 1 ]; then
    echo "$R/$arm/decomposer holds $n run dirs; keep exactly one (move the others away)" >&2
    exit 2
  fi
  if [ "$n" -eq 1 ] && [ -f "$dirs/results.json" ]; then
    log "decode already complete: $dirs"
  elif [ "$n" -eq 1 ] && [ -f "$dirs/resume_key.json" ]; then
    log "resuming interrupted decode $dirs"
    $PY components/decomposer/run_decomposer.py --model "$model" --config decomposer_musique.json \
      --condition "$cond" --seed 42 --quantization 4bit --retrieval-input "$RETR" --resume "$dirs"
  elif [ "$n" -eq 1 ]; then
    echo "$dirs is neither complete nor resumable (no results.json, no resume_key.json)" >&2
    exit 2
  else
    log "decompose $arm ($model, $cond)"
    $PY components/decomposer/run_decomposer.py --model "$model" --config decomposer_musique.json \
      --condition "$cond" --seed 42 --quantization 4bit --retrieval-input "$RETR" \
      --output-root "$R/$arm/decomposer"
  fi
  res=$(ls "$R/$arm"/decomposer/*/results.json | head -n 1)
  if [ -f "$R/$arm/eval/eval_metrics.json" ]; then
    log "eval already complete: $R/$arm/eval"
  else
    log "evaluate $res"
    $PY scripts/musique_decompositions_evaluator.py --predictions "$res" \
      --run-dir "$R/$arm/eval" --out-prefix eval --seed 42
  fi
  log "done $arm"
}

compare_if() {  # $1 name, $2 system A per-item, $3 system B per-item
  if [ -f "$2" ] && [ -f "$3" ]; then
    log "compare $1"
    $PY scripts/musique_decompositions_evaluator.py --compare "$2" "$3" \
      --run-dir "$R/compare/$1" --out-prefix compare --seed 42
  else
    log "skip compare $1 (missing input)"
  fi
}

case "${1:-}" in
  mistral_count_first)    decompose_eval mistral_count_first    mistral_7b_instruct count_first ;;
  mistral_unguided_rerun) decompose_eval mistral_unguided_rerun mistral_7b_instruct unguided ;;
  qwen_count_first)       decompose_eval qwen_count_first       qwen3_5_9b          count_first ;;
  qwen_unguided_rerun)    decompose_eval qwen_unguided_rerun    qwen3_5_9b          unguided ;;
  compare)
    preflight
    # PRIMARY (Holm families F1/F2): count-first minus the same-machine unguided re-run.
    compare_if mistral_count_first_vs_rerun  "$R/mistral_count_first/eval/eval_per_item.json" "$R/mistral_unguided_rerun/eval/eval_per_item.json"
    compare_if qwen_count_first_vs_rerun     "$R/qwen_count_first/eval/eval_per_item.json"    "$R/qwen_unguided_rerun/eval/eval_per_item.json"
    # SECONDARY: count-first minus the stored baseline (3090 run).
    compare_if mistral_count_first_vs_exp020 "$R/mistral_count_first/eval/eval_per_item.json" "$BASE_MISTRAL"
    compare_if qwen_count_first_vs_exp027    "$R/qwen_count_first/eval/eval_per_item.json"    "$BASE_QWEN"
    # Hardware control: same-machine unguided re-run minus the stored baseline.
    compare_if mistral_rerun_vs_exp020       "$R/mistral_unguided_rerun/eval/eval_per_item.json" "$BASE_MISTRAL"
    compare_if qwen_rerun_vs_exp027          "$R/qwen_unguided_rerun/eval/eval_per_item.json"    "$BASE_QWEN"
    ;;
  metrics) preflight; $PY experiments/exp-037/build_metrics.py ;;
  *) echo "usage: $0 {mistral_count_first|mistral_unguided_rerun|qwen_count_first|qwen_unguided_rerun|compare|metrics}" >&2; exit 2 ;;
esac
