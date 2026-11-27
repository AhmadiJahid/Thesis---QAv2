#!/usr/bin/env bash
# exp-040: raw / uniform / typed query masking for few-shot retrieval, on the v2 pipeline
# (Mistral-7B 4-bit unguided, exp-020's command; only the retrieval mode moves). Plan and
# thresholds: experiments/exp-040/config.json. Run from the repo root of a CLEAN checkout, with
# the data folder next to the clone (docs/compute.md "Dataset location") and these stored runs
# restored under runs/: pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl,
# exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl, exp-032/exp-020/{eval,reparse}.
#
#   bash experiments/exp-040/run.sh <step>
#
# steps, in this order (the notebook runs them so):
#   retrieval              build the all-modes artifact on CPU, check it, pin its sha256
#   canary <arm>           20-row decode of <arm>'s command + output-health gate
#   typed_rerun            exp-020's exact command (exp-014 artifact, typed)  decode+health+eval
#   raw                    rebuilt artifact, --retrieval-mode raw              decode+health+eval
#   uniform                rebuilt artifact, --retrieval-mode uniform          decode+health+eval
#   typed_rebuilt          rebuilt artifact, --retrieval-mode typed (ONLY if check.json says the
#                          rebuild's typed exemplars differ from exp-014's for any query)
#   compare                experiments/exp-040/compare.py -> experiments/exp-040/compare_metrics.json
#
# A decode step is RESUMABLE (run_decomposer.py --resume refuses unless code, prompt, seed,
# retrieval file + mode and decoding are identical). A finished decode / eval is never redone.
# PY defaults to "python"; the notebook sets PY to its Python 3.13.9 venv.
set -euo pipefail
PY=${PY:-python}
R=runs/exp-040
EXP014=runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl
EXP014_SHA=3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0
POOL=runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl
POOL_SHA=cc3646625d759a1b58179b18c25f83d255db24f05acc08b666985af2e58d6646
ALLMODES=$R/retrieval/pinned600_allmodes_rerank_top5.jsonl
PIN=$R/retrieval/PINNED_SHA256

log() { echo "[exp-040 $(date -Iseconds)] $*"; }
sha() { $PY -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

preflight() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=no)
  if [ -n "$dirty" ]; then
    echo "refusing: tracked files differ from the commit (Gate 2 - runs come from committed code):" >&2
    echo "$dirty" >&2
    exit 2
  fi
  log "git $(git rev-parse HEAD) (clean)"
  [ -f "$EXP014" ] || { echo "missing $EXP014" >&2; exit 2; }
  [ "$(sha "$EXP014")" = "$EXP014_SHA" ] || { echo "exp-014 artifact sha256 mismatch" >&2; exit 2; }
}

# Free-VRAM gate (config.json gpu.free_vram_gate_gib): the GPU may be shared with a queued job, so
# refuse BEFORE loading a model when total - used < the measured peak + margin. Exit 4 = "GPU busy,
# nothing started": re-run the step when the other job has finished.
VRAM_DECODE=6
vram_gate() {  # $1 GiB needed, $2 label
  local q total used
  q=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1) \
    || { echo "nvidia-smi failed - no GPU?" >&2; exit 4; }
  total=${q%%,*}; used=${q##*,}
  if ! awk -v t="$total" -v u="$used" -v n="$1" 'BEGIN { exit !((t - u) / 1024 >= n) }'; then
    awk -v t="$total" -v u="$used" -v n="$1" -v l="$2" 'BEGIN { printf "refusing %s: %.1f GiB free on the GPU (%.1f total, %.1f in use by other processes), need >= %s GiB. Nothing was started; re-run this step when the GPU is free.\n", l, (t-u)/1024, t/1024, u/1024, n }' >&2
    exit 4
  fi
  log "free VRAM OK for $2 ($(( (total - used) / 1024 )) GiB free, need $1)"
}

# arm -> retrieval input and extra flags (the ONLY things that differ between arms)
arm_input() { case $1 in typed_rerun) echo "$EXP014" ;; raw|uniform|typed_rebuilt) echo "$ALLMODES" ;; *) echo "unknown arm $1" >&2; exit 2 ;; esac; }
arm_mode()  { case $1 in typed_rerun) echo "" ;; raw) echo "--retrieval-mode raw" ;; uniform) echo "--retrieval-mode uniform" ;; typed_rebuilt) echo "--retrieval-mode typed" ;; esac; }

check_input() {  # $1 arm
  if [ "$(arm_input "$1")" = "$ALLMODES" ]; then
    [ -f "$ALLMODES" ] || { echo "missing $ALLMODES - run the retrieval step first" >&2; exit 2; }
    $PY experiments/exp-040/check_retrieval.py --artifact "$ALLMODES" --exp014 "$EXP014" --pin "$PIN" --verify
  fi
}

decomposer() {  # $1 arm, rest: extra args
  local arm=$1; shift
  # shellcheck disable=SC2046
  $PY components/decomposer/run_decomposer.py --model mistral_7b_instruct --config decomposer_musique.json \
    --condition unguided --seed 42 --quantization 4bit --retrieval-input "$(arm_input "$arm")" $(arm_mode "$arm") "$@"
}

retrieval() {
  preflight
  [ -f "$POOL" ] || { echo "missing $POOL" >&2; exit 2; }
  [ "$(sha "$POOL")" = "$POOL_SHA" ] || { echo "pool sha256 mismatch" >&2; exit 2; }
  mkdir -p "$R/retrieval"
  if [ -f "$PIN" ] && [ -f "$ALLMODES" ]; then
    log "retrieval artifact already built; verifying the pin"
    $PY experiments/exp-040/check_retrieval.py --artifact "$ALLMODES" --exp014 "$EXP014" --pin "$PIN" --verify
    return
  fi
  [ -f "$PIN" ] && { echo "$PIN exists but $ALLMODES is missing - refusing to rebuild a pinned artifact" >&2; exit 2; }
  log "similarity (raw + typed + uniform, top-20, CPU)"
  $PY MusiQue/scripts/check_question_similarity.py --pool-file "$POOL" \
    --query-glob 'musique/dev_data/musique_ans_v1.0_dev_sample_*_hop_200.jsonl' --n 200 --mode all \
    --top-k 20 --device cpu --seed 42 --no-hop-match --no-cache \
    --out "$R/retrieval/pinned600_allmodes_top20.jsonl" --run-dir "$R/retrieval/similarity"
  log "rerank (cross-encoder, top-5, CPU)"
  $PY MusiQue/scripts/rerank_similarity_results.py --input "$R/retrieval/pinned600_allmodes_top20.jsonl" \
    --out "$ALLMODES" --rerank-k 5 --device cpu --seed 42 --run-dir "$R/retrieval/rerank"
  $PY experiments/exp-040/check_retrieval.py --artifact "$ALLMODES" --exp014 "$EXP014" --pin "$PIN" \
    --out "$R/retrieval/check.json"
}

canary() {  # $1 arm
  local arm=$1 d
  preflight; check_input "$arm"
  if [ -f "$R/canary/$arm/health.json" ] && grep -q '"ok": true' "$R/canary/$arm/health.json"; then
    log "canary $arm already passed"; return
  fi
  vram_gate "$VRAM_DECODE" "canary $arm"
  rm -rf "$R/canary/$arm"; mkdir -p "$R/canary/$arm"
  log "canary $arm (20 rows; not an experiment arm)"
  decomposer "$arm" --sample-size 20 --allow-unpinned-eval-set --output-root "$R/canary/$arm/decomposer"
  d=$(find "$R/canary/$arm/decomposer" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  $PY experiments/exp-040/check_health.py --run-dir "$d" --kind canary --out "$R/canary/$arm/health.json"
}

decompose_eval() {  # $1 arm
  local arm=$1 dirs n res
  preflight; check_input "$arm"
  mkdir -p "$R/$arm/decomposer"
  dirs=$(find "$R/$arm/decomposer" -mindepth 1 -maxdepth 1 -type d | sort)
  n=$(printf '%s' "$dirs" | grep -c . || true)
  if [ "$n" -gt 1 ]; then
    echo "$R/$arm/decomposer holds $n run dirs; keep exactly one (move the others away)" >&2; exit 2
  fi
  if [ "$n" -eq 1 ] && [ -f "$dirs/results.json" ]; then
    log "decode already complete: $dirs"
  elif [ "$n" -eq 1 ] && [ -f "$dirs/resume_key.json" ]; then
    vram_gate "$VRAM_DECODE" "decode $arm"
    log "resuming interrupted decode $dirs"
    decomposer "$arm" --resume "$dirs"
  elif [ "$n" -eq 1 ]; then
    echo "$dirs is neither complete nor resumable (no results.json, no resume_key.json)" >&2; exit 2
  else
    vram_gate "$VRAM_DECODE" "decode $arm"
    log "decompose $arm"
    decomposer "$arm" --output-root "$R/$arm/decomposer"
  fi
  res=$(ls "$R/$arm"/decomposer/*/results.json | head -n 1)
  if ! $PY experiments/exp-040/check_health.py --run-dir "$(dirname "$res")" --kind full --out "$R/$arm/health.json"; then
    cp "$R/$arm/health.json" "$R/$arm/HEALTH_FAILED.json"
    echo "output-health gate FAILED for $arm (see $R/$arm/HEALTH_FAILED.json); not evaluated" >&2; exit 3
  fi
  if [ -f "$R/$arm/eval/eval_metrics.json" ]; then
    log "eval already complete: $R/$arm/eval"
  else
    log "evaluate $res"
    $PY scripts/musique_decompositions_evaluator.py --predictions "$res" --run-dir "$R/$arm/eval" \
      --out-prefix eval --seed 42
  fi
  log "done $arm"
}

case "${1:-}" in
  retrieval) retrieval ;;
  canary) canary "${2:?arm}" ;;
  typed_rerun|raw|uniform|typed_rebuilt) decompose_eval "$1" ;;
  compare) preflight; $PY experiments/exp-040/compare.py ;;
  *) echo "usage: $0 {retrieval|canary <arm>|typed_rerun|raw|uniform|typed_rebuilt|compare}" >&2; exit 2 ;;
esac
