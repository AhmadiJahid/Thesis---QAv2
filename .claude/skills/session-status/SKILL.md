---
name: session-status
description: Session-start survey of the thesis repo, and the answer to "is anything running?" / "where do things stand?". Use at the start of every lead session, before briefing any lane, and whenever someone asks what is in flight, blocked, or finished but unrecorded.
---

# session-status

Read-only survey. Run from the main checkout (`/cta/users/fyilmaz/Thesis---QAv2`); never
checkout, stash or reset it — other sessions share it. Report one short digest at the end.

## Procedure

1. **Experiment log** (CLAUDE.md, Before any task). Rows are very long; print id + status only:
   `awk -F'|' '/^\| exp-/ {print $2 "|" substr($11,1,90)}' experiments/log.md | tail -20`
   Flag every row whose metrics column starts `PENDING` — the run is queued, running, or its
   results were never recorded. Read a flagged row in full before acting on it.
2. **Open work:** `gh issue list --state open` and `gh pr list --state open`. Open PRs are
   finished work that issues do not show.
3. **Local lanes:** `git fetch -q && git branch -vv` — list branches ahead of `main` (unmerged
   work hides here) and `git worktree list` — flag `prunable` entries and worktrees whose branch is
   already merged (`git branch --merged main`). Report them; do not remove another lane's worktree.
4. **GPU:** `nvidia-smi` and
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` then `ps -o user=,cmd= -p <pid>`
   for each pid. Name other users' processes (e.g. an ollama server holding ~21 GiB) and how much
   memory they hold. **Never kill another user's process.**
5. **Run lock** (docs/compute.md, Run lock convention): `cat runs/run.lock`. Absent = free.
   Present = holder exp id + ISO time; a lock older than 24 h is stale only if step 4 and `ps`
   show nothing live — report it, the age alone never justifies clearing it.
6. **Chains:** `ls -t runs/*/chain_status.txt | head -3` and `tail -3` each; `tmux ls`.
   The last line `exit rc=N` means the chain finished (N=0 success); `waiting since …` with no
   `chain start` means it is queued behind the lock or GPU. Cross-check against step 1: a finished
   chain whose log row is still PENDING is **done-but-unrecorded**.
7. **Push identity:** `gh auth status` — pushes from this box work only with account
   `AhmadiJahid` active; if another account is active, say so (`gh auth switch -u AhmadiJahid`).
8. **Uncommitted work:** `git status --short` in the main checkout and in each worktree from
   step 3. Report what is uncommitted and where; this survey is read-only, so it commits and
   deletes nothing.
9. **Optional — new supervision meeting:** search Fathom for meetings titled
   "Impromptu Google Meet Meeting" since the last ADR date; read the **transcript**, not the
   summary, before quoting anything from it. Any decision in it is Jahid's to confirm (CLAUDE.md,
   Gates 3).

## Output

One short digest, no transcript dumps:

- **In flight:** chains running / queued (exp id, step, since when, lock holder).
- **Blocked:** what and why (lock, GPU held by another user, review pending, Jahid's decision).
- **Done but unrecorded:** finished chains with PENDING rows; reviewed PRs not merged;
  branches ahead of main with no PR.
- **Next decisions:** only those that are Jahid's (Gates 3) or the lead's to make now.

Every line cites what it was read from (file, command). If something could not be checked
(e.g. Fathom unavailable), say it is unchecked.
