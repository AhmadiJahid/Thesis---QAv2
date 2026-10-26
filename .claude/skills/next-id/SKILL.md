---
name: next-id
description: Reserve the next experiment id (exp-NNN) or ADR number when other lead sessions or agents may be active. Use before writing any brief, branch, config, log row or ADR that names a new exp id or ADR number.
---

# next-id

**Why:** on 2026-10-14 two concurrent lead sessions picked the same numbers; one lane had to
renumber after review (`7fd87c5` "renumber to exp-030 / ADR 0037", then `3006169` "renumber the
sampling-variance ADR to 0038 — 0037 taken on main"). Numbers are gap-free and never reused
(docs/adr/README.md), so the fix is to claim a number on `main` before anyone cites it.

## Procedure

1. `git fetch -q` (pushes need `AhmadiJahid` active: `gh auth status`).
2. Find the highest number used anywhere — main, origin/main, every local branch, and every
   worktree's uncommitted files:
   `.claude/skills/next-id/next_id.sh /cta/users/fyilmaz/Thesis---QAv2`
   It prints the highest exp id / ADR number, where it was seen, and the next free pair.
   Manual equivalent if needed: `git branch -a`, `git worktree list`, then for each ref
   `git show <ref>:experiments/log.md | grep -oE '^\| exp-[0-9]{3}'` and
   `git ls-tree --name-only <ref> docs/adr/`; plus `ls experiments/ docs/adr/` in each worktree.
3. **Claim it immediately on main** — from a throwaway worktree at `origin/main`, never by
   moving the shared checkout's HEAD:
   - **exp id:** commit the pre-run row to `experiments/log.md` (schema in that file; metrics
     `PENDING — …`). This is an experiment-only commit, allowed directly on main (CLAUDE.md, Git).
   - **ADR:** commit `docs/adr/NNNN-<slug>.md` as a stub (title line + `- **Status:** reserved
     (<who>, <date>) for <one line>`) and its index row in `docs/adr/README.md` in the same
     commit (docs/adr/README.md requires the index row with every new record).
   Push: `git push origin HEAD:main`. If the push is rejected, fetch, re-run step 2 and take the
   next number — someone else claimed it first.
4. **Only then** put the number in briefs, branch names, configs or `Refs`.
5. A reservation that is later abandoned keeps its number; mark the row or stub's status as
   abandoned rather than deleting it (numbers are never reused).
