#!/usr/bin/env bash
# exp-034: CPU re-score of exp-031 onto the PR #61 parser (Issue #51, ADR 0039). Method of exp-032.
set -euo pipefail
WT=/cta/users/fyilmaz/worktrees/exp-034
M=/cta/users/fyilmaz/Thesis---QAv2
PY=$M/.venv/bin/python
R=$M/runs/exp-034
cd "$WT"
echo "git $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start $(date -Iseconds)"
for n in 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  res=$(ls $M/runs/exp-031/size$n/decomposer/*/results.json)
  echo "== size$n reparse $res"
  $PY scripts/reparse_decompositions.py --results "$res" --output-dir $R/size$n/reparse
  echo "== size$n eval"
  $PY scripts/musique_decompositions_evaluator.py --predictions $R/size$n/reparse/results.json --run-dir $R/size$n/eval --out-prefix eval --seed 42
done
for n in 3000 4000 5000 6000 7000 8000 9000 10000; do
  echo "== compare size$n vs size2000"
  $PY scripts/musique_decompositions_evaluator.py --compare $R/size$n/eval/eval_per_item.json $R/size2000/eval/eval_per_item.json --run-dir $R/compare/size${n}_vs_size2000 --out-prefix compare --seed 42
done
echo "== reproducibility compare size2000 vs exp-032 exp-020 (re-parsed)"
$PY scripts/musique_decompositions_evaluator.py --compare $R/size2000/eval/eval_per_item.json $M/runs/exp-032/exp-020/eval/eval_per_item.json --run-dir $R/compare/size2000_vs_exp020_reparsed --out-prefix compare --seed 42
echo "ALL DONE rc=0 $(date -Iseconds)"
