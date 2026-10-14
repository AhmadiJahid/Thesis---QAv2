#!/bin/bash
# exp-030 chain: five decomposer arms + evaluator, sequential. Commands copied verbatim from experiments/exp-030/README-commands.md.
cd /cta/users/fyilmaz/Thesis---QAv2
R=runs/exp-030
ST=$R/chain_status.txt
RET="--retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl --eval-set musique_dev_full"
declare -a NAMES=(qwen2_5_7b pool_ft qwen2_5_14b qwen3_5_9b mistral_7b)
declare -A QUANT=([qwen2_5_7b]=4bit [pool_ft]=none [qwen2_5_14b]=4bit [qwen3_5_9b]=4bit [mistral_7b]=4bit)
declare -A PARAMS=([qwen2_5_7b]=7,615,616,512 [pool_ft]=7,289,966,592 [qwen2_5_14b]=14,770,033,664 [qwen3_5_9b]=8,953,803,264 [mistral_7b]=7,248,023,552)
declare -A ARGS=(
 [qwen2_5_7b]="--model qwen2_5_7b_instruct --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit"
 [pool_ft]="--model mistral_7b_instruct --config decomposer_musique.json --condition unguided --seed 42 --quantization none --adapter runs/exp-026/20260920_024535/adapter --no-few-shot"
 [qwen2_5_14b]="--model qwen2_5_14b_instruct --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit"
 [qwen3_5_9b]="--model qwen3_5_9b --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit"
 [mistral_7b]="--model mistral_7b_instruct --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit"
)
echo "chain start $(date -Iseconds) HEAD=$(git rev-parse HEAD)" >> $ST
for a in "${NAMES[@]}"; do
  # wait for GPU to be free of other users' processes (never kill them)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && break
    echo "$(date -Iseconds) $a waiting: GPU used ${used} MiB" >> $ST; sleep 120
  done
  mkdir -p $R/$a
  LOG=$R/$a/decompose.log
  s=$(date +%s); echo "$a START $(date -Iseconds)" >> $ST
  PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py ${ARGS[$a]} $RET --output-root $R/$a/decomposer > $LOG 2>&1 &
  PID=$!
  ( while kill -0 $PID 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R/$a/vram_samples.txt; sleep 30; done ) &
  .venv/bin/python $R/tools/check_startup.py $LOG ${QUANT[$a]} ${PARAMS[$a]} $PID > $R/$a/startup_check.txt 2>&1
  crc=$?
  if [ $crc -eq 1 ]; then kill $PID; echo "$a STARTUP_CHECK_FAILED -> killed" >> $ST; fi
  wait $PID; rc=$?
  e=$(date +%s)
  echo "$a END $(date -Iseconds) rc=$rc check_rc=$crc wall_s=$((e-s))" >> $ST
  if [ $rc -eq 0 ] && [ $crc -eq 0 ]; then
    RD=$(ls -td $R/$a/decomposer/*/ | head -1)
    es=$(date +%s)
    .venv/bin/python scripts/musique_decompositions_evaluator.py --predictions ${RD}results.json \
      --run-dir $R/$a/eval --out-prefix eval --seed 42 --eval-set musique_dev_full > $R/$a/eval.log 2>&1
    erc=$?; echo "$a EVAL rc=$erc wall_s=$(( $(date +%s)-es )) run_dir=$RD" >> $ST
  fi
done
echo "chain done $(date -Iseconds)" >> $ST
