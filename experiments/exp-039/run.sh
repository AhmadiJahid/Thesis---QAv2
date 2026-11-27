#!/usr/bin/env bash
# exp-039: MetaQA end-to-end through the direct-KG compile-execute backend (NOT GRAG; ADR 0045).
# Run from the repo root of a CLEAN checkout, with the data folder next to the clone
# (docs/compute.md "Dataset location") holding metaqa/kb.txt.
#
#   bash experiments/exp-039/run.sh <step>
#
# steps:
#   eval_set          fetch the 3 vanilla qa_test.txt files (pinned mirror revision, sha256-checked),
#                     draw the 250/hop set, and refuse unless it equals the committed manifest
#   smoke             6 Qwen rows through decode + compile-execute (not an arm)
#   oracle            gold-like decompositions (CPU) -> compile-execute: the measured EM ceiling
#   qwen_prompted     Qwen3.5-9B, unguided, 4-bit, few-shot from the MetaQA pool -> compile-execute
#   mistral_prompted  Mistral-7B, same
#   mistral_ft        exp-026 MuSiQue pool adapter on Mistral-7B, 4-bit, --no-few-shot  (needs ADAPTER_MISTRAL)
#   qwen_ft           exp-036 MuSiQue pool adapter on Qwen3.5-9B, 4-bit, --no-few-shot   (needs ADAPTER_QWEN;
#                     CONDITIONAL: skipped, with a marker file, while the adapter does not exist)
#   report            scripts/metaqa_arm_report.py over every arm that finished
#
# A decode step is RESUMABLE (run_decomposer.py --resume). A finished decode or compile-execute
# is never redone. PY defaults to "python".
set -euo pipefail
PY=${PY:-python}
R=runs/exp-039
DATA=${DATA_ROOT:-../thesis-qav2-data}
MIRROR=https://huggingface.co/datasets/camazlucas/MetaQA/resolve/f83854092893610686ab19f2038c039afa9bc643

log() { echo "[exp-039 $(date -Iseconds)] $*"; }

preflight() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=no)
  if [ -n "$dirty" ]; then
    echo "refusing: tracked files differ from the commit (Gate 2):" >&2; echo "$dirty" >&2; exit 2
  fi
  log "git $(git rev-parse HEAD) (clean)"
}

need_eval_set() {
  for h in 1 2 3; do
    [ -f "$DATA/metaqa/eval250/refined_${h}hop.txt" ] || { echo "eval set missing: run '$0 eval_set' first" >&2; exit 2; }
  done
}

compile_execute() {  # $1 arm, $2 predictions json
  if [ -f "$R/$1/ce/metaqa_e2e_metrics.json" ]; then log "compile-execute already complete: $R/$1/ce"; return; fi
  log "compile-execute $1"
  $PY scripts/run_metaqa_compile_execute.py --config metaqa_compile_execute_eval250.json \
    --predictions "$2" --run-dir "$R/$1/ce" --seed 42
}

decompose() {  # $1 arm, $2 model folder, rest: extra run_decomposer args
  local arm=$1 model=$2 dirs n res; shift 2
  preflight; need_eval_set
  mkdir -p "$R/$arm/decomposer"
  dirs=$(find "$R/$arm/decomposer" -mindepth 1 -maxdepth 1 -type d | sort)
  n=$(printf '%s' "$dirs" | grep -c . || true)
  if [ "$n" -gt 1 ]; then echo "$R/$arm/decomposer holds $n run dirs; keep exactly one" >&2; exit 2; fi
  if [ "$n" -eq 1 ] && [ -f "$dirs/results.json" ]; then
    log "decode already complete: $dirs"
  elif [ "$n" -eq 1 ] && [ -f "$dirs/resume_key.json" ]; then
    log "resuming interrupted decode $dirs"
    $PY components/decomposer/run_decomposer.py --model "$model" --config decomposer_metaqa_eval250.json \
      --seed 42 --quantization 4bit "$@" --resume "$dirs"
  elif [ "$n" -eq 1 ]; then
    echo "$dirs is neither complete nor resumable" >&2; exit 2
  else
    log "decompose $arm ($model $*)"
    $PY components/decomposer/run_decomposer.py --model "$model" --config decomposer_metaqa_eval250.json \
      --seed 42 --quantization 4bit "$@" --output-root "$R/$arm/decomposer"
  fi
  res=$(ls "$R/$arm"/decomposer/*/results.json | head -n 1)
  compile_execute "$arm" "$res"
  log "done $arm"
}

adapter_arm() {  # $1 arm, $2 model folder, $3 adapter dir
  if [ -z "$3" ] || [ ! -f "$3/adapter_config.json" ]; then
    mkdir -p "$R/$1"
    echo "adapter not found: '${3}'" > "$R/$1/SKIPPED_no_adapter.txt"
    log "SKIP $1: adapter not found ('${3}'); recorded in $R/$1/SKIPPED_no_adapter.txt"
    return 0
  fi
  rm -f "$R/$1/SKIPPED_no_adapter.txt"
  decompose "$1" "$2" --adapter "$3" --no-few-shot
}

case "${1:-}" in
  eval_set)
    preflight
    for h in 1 2 3; do
      f="$DATA/metaqa/vanilla_mirror/${h}-hop/vanilla/qa_test.txt"
      if [ ! -f "$f" ]; then mkdir -p "$(dirname "$f")"; curl -sSfL -o "$f" "$MIRROR/${h}-hop/vanilla/qa_test.txt"; fi
    done
    mkdir -p "$R"
    $PY scripts/build_metaqa_eval_set.py --manifest "$R/eval_set_manifest.json"
    $PY - "$R/eval_set_manifest.json" experiments/exp-039/eval_set_manifest.json <<'PYEOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d in (a, b):
    d.pop("created_utc", None)
if a != b:
    sys.exit("drawn eval set differs from the committed experiments/exp-039/eval_set_manifest.json")
print("eval set identical to the committed manifest")
PYEOF
    ;;
  oracle)
    preflight; need_eval_set
    $PY scripts/metaqa_gold_path_check.py --config metaqa_gold_path_check_eval250.json --run-dir "$R/oracle/gold_path"
    compile_execute oracle "$R/oracle/gold_path/gold_like_decompositions_natural.json"
    ;;
  smoke)
    # The MetaQA prompting path (KB masking + e5-small few-shot retrieval + decode) has not been
    # run for real in v2. Decode 6 rows with Qwen and push them through compile-execute before
    # the full arms spend GPU time. Not an arm; its own directory, always redone.
    preflight; need_eval_set
    rm -rf "$R/smoke"
    $PY components/decomposer/run_decomposer.py --model qwen3_5_9b --config decomposer_metaqa_eval250.json \
      --seed 42 --quantization 4bit --sample-size 6 --output-root "$R/smoke/decomposer"
    $PY scripts/run_metaqa_compile_execute.py --config metaqa_compile_execute_eval250.json \
      --predictions "$(ls "$R"/smoke/decomposer/*/results.json | head -n 1)" --run-dir "$R/smoke/ce" \
      --seed 42 --allow-unpinned-eval-set
    ;;
  qwen_prompted)    decompose qwen_prompted qwen3_5_9b ;;
  mistral_prompted) decompose mistral_prompted mistral_7b_instruct ;;
  mistral_ft)       adapter_arm mistral_ft mistral_7b_instruct "${ADAPTER_MISTRAL:-}" ;;
  qwen_ft)          adapter_arm qwen_ft qwen3_5_9b "${ADAPTER_QWEN:-}" ;;
  report)
    preflight
    args=()
    for arm in oracle qwen_prompted mistral_prompted mistral_ft qwen_ft; do
      [ -f "$R/$arm/ce/metaqa_e2e_metrics.json" ] && args+=(--arm "$arm=$R/$arm/ce")
    done
    $PY scripts/metaqa_arm_report.py --config metaqa_arm_report.json "${args[@]}" --run-dir "$R/report"
    ;;
  *) echo "usage: $0 {eval_set|smoke|oracle|qwen_prompted|mistral_prompted|mistral_ft|qwen_ft|report}" >&2; exit 2 ;;
esac
