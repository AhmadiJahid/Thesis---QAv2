#!/usr/bin/env bash
# Print the next free experiment id and ADR number, scanning every place another
# session may already have used one: all local and remote-tracking branches
# (experiments/log.md rows, experiments/ dirs, docs/adr/ files) plus the working
# files of every worktree (uncommitted claims). Read-only; run `git fetch` first.
# Usage: next_id.sh [repo-dir]   (default: the repo containing the cwd)
set -uo pipefail
cd "${1:-.}" && git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repo" >&2; exit 2; }

max_exp=0; max_adr=0; exp_src=-; adr_src=-
bump() { # kind number source
  local n=$((10#$2))
  if [ "$1" = exp ] && [ "$n" -gt "$max_exp" ]; then max_exp=$n; exp_src=$3; fi
  if [ "$1" = adr ] && [ "$n" -gt "$max_adr" ]; then max_adr=$n; adr_src=$3; fi
}

for ref in $(git for-each-ref --format='%(refname:short)' refs/heads refs/remotes); do
  case $ref in */HEAD|origin) continue ;; esac
  for n in $(git show "$ref:experiments/log.md" 2>/dev/null | grep -oE '^\| exp-[0-9]{3}' | grep -oE '[0-9]{3}'); do bump exp "$n" "$ref:log"; done
  for n in $(git ls-tree --name-only "$ref" experiments/ 2>/dev/null | grep -oE 'exp-[0-9]{3}$' | grep -oE '[0-9]{3}'); do bump exp "$n" "$ref:dir"; done
  for n in $(git ls-tree --name-only "$ref" docs/adr/ 2>/dev/null | grep -oE '/[0-9]{4}-' | grep -oE '[0-9]{4}'); do bump adr "$n" "$ref"; done
done

while read -r wt; do
  [ -d "$wt" ] || continue
  for n in $(grep -oE '^\| exp-[0-9]{3}' "$wt/experiments/log.md" 2>/dev/null | grep -oE '[0-9]{3}'); do bump exp "$n" "worktree:$wt"; done
  for n in $(ls "$wt/experiments" 2>/dev/null | grep -oE '^exp-[0-9]{3}$' | grep -oE '[0-9]{3}'); do bump exp "$n" "worktree:$wt"; done
  for n in $(ls "$wt/docs/adr" 2>/dev/null | grep -oE '^[0-9]{4}-' | grep -oE '[0-9]{4}'); do bump adr "$n" "worktree:$wt"; done
done < <(git worktree list --porcelain | sed -n 's/^worktree //p')

printf 'highest exp: exp-%03d (seen on %s)\n' "$max_exp" "$exp_src"
printf 'highest ADR: %04d (seen on %s)\n' "$max_adr" "$adr_src"
printf 'next free:   exp-%03d   ADR %04d\n' $((max_exp + 1)) $((max_adr + 1))
