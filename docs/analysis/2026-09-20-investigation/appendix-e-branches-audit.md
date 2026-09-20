<!-- Lane ledger produced by a read-only investigator agent on 2026-09-20 for the lead's state-of-the-work investigation (`docs/analysis/2026-09-20-state-of-the-work-investigation.md`). Every claim cites a file, commit, issue or log row. Numbers are quoted from committed artifacts, never re-computed. -->

# Stranded-work investigation — branches, worktrees, untracked files, runs/

Read-only. `main` = f1bbdad (53 ahead of origin/main; nothing pushable from this box).
Snapshot 2026-09-20 15:45 +0300. git 2.34.1 (legacy `merge-tree` only; no `--write-tree`).

Conflict counts below = `git merge-tree $(git merge-base main <b>) main <b> | grep -c '<<<<<<<'`
(conflict *hunks*, not files). No merge was attempted.

---

## 1. feature/12-guided-vs-unguided-alt — tip c1b4aed (2026-08-18)

- merge-base 8038f5b. **1 commit ahead**, **121 behind** main.
- `main...branch`: 15 files, +1082/-14 — `src/step_cap.py` (123 L), `tests/test_step_cap.py`,
  `tests/test_musique_conditions.py` (380 L), a `configs/decomposer_musique.json`, step-cap
  wiring in `run_decomposer.py`, smoke-test hooks, three MuSiQue fixture files.
- **What it is / issue:** issue #12 — the three MuSiQue decomposition conditions
  (guided / unguided / unguided-capped) in one decomposer config, with a step-line cap module.
- **Status: fully superseded.** Issue #12 comment (line 40 of `gh issue view 12 --comments`)
  records it verbatim: *"#12 turned out to be implemented twice — PR #21 … and a second,
  independent implementation from 2026-08-18 parked on `feature/12-guided-vs-unguided-alt`
  (c1b4aed, no PR). **PR #21 is canonical**… The alt branch is held as a cherry-pick source for
  two pieces PR #21 lacks … The alt branch gets deleted once that's settled."*
  PR #21 (head `feature/12-guided-vs-unguided`, tip 768cfb9) squash-merged as **1ccea11**.
  Both cherry-pick candidates are now on main, in that same commit 1ccea11:
  - `src/step_lines.py` (218 L, added by 1ccea11) is the successor of alt's `src/step_cap.py`
    (123 L) and its docstring states the three-way agreement the alt test pinned;
  - the evaluator-agreement test lives at `tests/test_decomposer_conditions.py:610`
    ("the decomposer's step counting is the evaluator's, not a private copy", importing
    `musique_decompositions_evaluator`); the alt's version was `tests/test_step_cap.py:206`.
  - post-hoc truncation is in `components/decomposer/run_decomposer.py` (`truncate_at`,
    lines 1032–1107; `truncation_definitions` at 3079).
  `git diff c1b4aed 768cfb9` = 17 files, +985/-992 — two different implementations, not a delta.
- **Merge cleanly today:** no — **35 conflict hunks**.
- **Review pass:** none. Single commit, no PR, no "review pass" commit, no reviewer comment.
- **Verdict: superseded (abandon-candidate).** Its two reserved pieces are on main under
  different names; nothing unique remains.

---

## 2. feature/13-lora-finetuning — tip a6892bb (2026-08-19)

- merge-base b6154fe. **3 commits ahead**, **118 behind**.
  - a6892bb Merge origin/main into feature/13-lora-finetuning
  - eab9a3e fix: harden the fine-tuning arm guards after the PR #24 review (refs #13)
  - aa8b0b7 feat: LoRA fine-tuning arm for the decomposer, with cost-vs-quality reporting
- `main...branch`: 19 files, +3206/-17 — `components/decomposer/train_lora.py`,
  `src/finetune_data.py`, `configs/finetune_decomposer.json`,
  `scripts/compare_decomposer_arms.py`, `tests/test_finetune_lora.py` (809 L), ADR 0012.
- **Issue #13** (open) — the LoRA fine-tuning comparison arm.
- **Status: already merged (squash).** This branch *is* the head of **PR #24, MERGED
  2026-08-18T22:20:57Z**, merge commit **2f49c20** ("feat: LoRA fine-tuning arm for the
  decomposer, with cost-vs-quality reporting (#13)"). `gh pr view 24` lists exactly these three
  SHAs. Every file exists on main. The follow-on review work landed as **PR #25** (head
  `feature/13-adapter-eval-guards`, merge **eed40f0**). Main has since moved far past it —
  e.g. e1935ad (pool_2000_imbalanced arm), 15c0d89 (exp-026).
- **Merge cleanly today:** no — **76 conflict hunks**. Re-merging would *revert* 118 commits of
  later work on the same files.
- **Review pass:** yes — eab9a3e is the PR #24 review fix pass; PR #25 is the second pass.
- **Verdict: already merged — stale branch ref.** Not stranded work.

---

## 3. feature/40-literature-metrics — tip f96b3fe (2026-08-22)

- merge-base a940376. **5 commits ahead**, **82 behind**.
  - f96b3fe fix(eval): Gate-1 review pass — v2 numeric gate, GED cost record, note caveat
  - b770c42 / b89441f / 610431e / c832721 (feature + tests + METRICS.md + ADR renumber)
- `main...branch`: 7 files, +2030/-124 — `scripts/musique_decompositions_evaluator.py` (+984),
  `tests/test_decomposition_metrics.py` (+718), `docs/METRICS.md`, ADR 0026, requirements.
- **Issue #40** (open) — Break-faithful EM / SARI / GED + repaired chain validity.
- **Status: already merged (squash).** This branch is the head of **PR #44, MERGED
  2026-08-22T17:00:04Z**, merge commit **a99b573**; `gh pr view 44` lists exactly these five
  SHAs. Follow-ups already on main: **32bad4a** (PR #45, evaluator record fixes) and
  **e3725c7** (PR #48, decomposition-quality suite as reported primary), plus re-scores
  b33ac08 (exp-011) and 6a15158 (exp-016).
- **Merge cleanly today:** no — **59 conflict hunks**.
- **Review pass:** yes — f96b3fe is literally the Gate-1 review pass commit.
- **Verdict: already merged — stale branch ref.** Not stranded work.

---

## 4. feature/50-anchor-set — tip db16e82 (2026-09-14)

- merge-base 07667c1. **1 commit ahead**, **45 behind**. Worktree:
  `/cta/users/fyilmaz/worktrees/50-anchor-set` (clean, 5.2 MB).
- `main...branch`: 9 files, +1354 — `configs/anchor_set_ids.json` (the 45 ids),
  `configs/anchor_set_draw.json`, `configs/anchor_inspect.json`, `scripts/build_anchor_set.py`
  (321 L), `scripts/anchor_inspect.py` (338 L), `tests/test_anchor_set.py` (365 L), an ADR
  ("…item-anchor-set-the-implementers-conventions").
- **Issue #50** (open) — supervisor item A4, the frozen 45-item anchor set.
- **Status: re-implemented on main, with a design change.** Main commit **e95e550** (2026-09-20)
  says in its body: *"Squash of `feature/50-anchor-set-b2` @ 1601639. Gate 1 review passed
  2026-09-20 (no Critical; three Important findings fixed in the same branch). Design recorded
  in ADR 0033."* That b2 branch no longer exists locally. Differences main vs this branch:
  - ids file **renamed** `configs/anchor_set_ids.json` → `configs/anchor_set_45.json`;
  - `scripts/anchor_inspect.py` (338 L) **replaced** by `scripts/render_anchor_set.py` (587 L);
  - `tests/test_anchor_set.py` grew 365 → 729 L; `build_anchor_set.py` 321 → 351 L;
  - the ADR was rewritten as **0033 "the anchor set is a frozen id list inspected by
    post-filtering"** — i.e. the branch's approach was the one superseded;
  - main also has `src/anchor_set.py`, `configs/anchor_render.json`,
    `tests/test_anchor_prompt_logging.py` and the follow-on **3cce79c** (anchor prompts logged
    by id), none of which exist on the branch.
  The 45 ids themselves carry forward (seed 42 draw, same `anchor_set_draw.json` parameters).
- **Merge cleanly today:** no — **23 conflict hunks**; it would also resurrect the old filenames.
- **Review pass:** not on this branch (single commit, no review commit). The Gate-1 pass was on
  the successor b2 branch, recorded in e95e550's message.
- **Verdict: superseded.** Its output is on main under different filenames and a different ADR.

---

## 5. feature/50-qwen-decomposer-models — tip 9f0c2fd (2026-09-14)

- merge-base 07667c1. **2 commits ahead**, **45 behind**.
  - 9f0c2fd fix(decomposer): own ceiling key, explicit A1 pool, and a script behind the count
  - b03f1b1 feat(decomposer): register Qwen2.5-7B/14B-Instruct and raise the ceiling to 1.5e10
  Worktrees: `wt-50-qwen` @ 9f0c2fd, `wt-tip` @ 9f0c2fd (detached), `wt-base` @ b03f1b1
  (detached) — i.e. the base/tip pair was a review-diff setup. All three clean.
- `main...branch`: 17 files, +631/-43 — the two Qwen2.5 model folders, `configs/model_limits.json`
  (`decomposer_max_params: 1.5e10`, default back to 8e9), `scripts/print_model_params.py`,
  `src/model_size.py`, tests.
- **Issue #50** — the three-model comparison (A1 arms).
- **Status: re-implemented on main, explicitly citing this branch.** Main commit **feef949**
  (2026-09-20) body: *"Squash of `feature/50-qwen25-decomposer-folders` @ 0fc1ae9 (**reuses the
  2026-09-14 branch feature/50-qwen-decomposer-models b03f1b1+9f0c2fd, rebased**, plus notes and
  four review Minors). Gate 1 review passed 2026-09-20: no Critical; I-1 recorded in ADR
  0031/0032 amendments, I-2 in CLAUDE.md, I-3 as launch preconditions on issue #50. Both folders
  smoke-tested 5/5 rows at 4-bit; counts 7,615,616,512 and 14,770,033,664 asserted on the live
  loaded objects."*
  `git diff <branch> main` on the branch's own paths shows **main strictly ahead**: 5 files,
  +135/-7 — `print_model_params.py` +37 (sharded-index counting fix), `test_model_params.py` +96,
  plus ADR-0032 wording in `model_limits.json`, the 14B `config.json` (weights now cached,
  `runs/downloads/qwen2_5_14b_instruct.log` EXIT 0) and `src/model_size.py`. Nothing on the
  branch is absent from main.
- **Merge cleanly today:** no — **12 conflict hunks** (README, `decomposer_musique.json`,
  `model_limits.json`, `print_model_params.py`, `model_size.py`, two test files) — and merging
  would revert main's later fixes.
- **Review pass:** partly — 9f0c2fd is a review fix pass ("issue #50 PR review, I-1" appears in
  the `model_limits.json` note it wrote). The **full Gate-1 MERGE verdict was on the successor
  branch**, per feef949 and issue #50 comment ("Model folders reviewed MERGE, fix pass in flight").
- **Verdict: superseded (its content was rebased forward into feef949).**

---

## 6. worktree-agent-aa3be8e02852de0de / -ac35e0de8908c738f / -aebee2d46a9636399

- All three point at **0ab6abb** ("docs: ADR 0010 — router stays as a hop-count regressor,
  fine-tuning prioritized"), 2026-08-18.
- **0 commits ahead of main**, 120 behind. `main...<b>` diff is **empty**. Merge-base == tip.
- **Verdict: fully merged ancestors of main — dead refs.** No content, no review needed.
  No worktree is registered for any of them (leftovers from deleted agent worktrees).

## 7. worktree-agent-ace2f2912496463a0

- Points at **b6329ca** ("docs: amend ADR 0014 … and fix ADR 0013 heading"), 2026-08-19.
- **0 commits ahead**, 111 behind. Empty diff vs main.
- **Verdict: fully merged ancestor — dead ref.**

---

## Worktrees on /tmp (vanish on reboot; branch refs survive in the repo's .git)

`git worktree list`:

| path | branch/commit | on /tmp? | dirty? | size |
|---|---|---|---|---|
| `/cta/users/fyilmaz/Thesis---QAv2` | main f1bbdad | no | 3 untracked PDFs | — |
| `/cta/users/fyilmaz/worktrees/50-anchor-set` | feature/50-anchor-set db16e82 | **no** (survives) | clean | 5.2M |
| `/tmp/claude-1005/…/5704aeae…/scratchpad/wt-40` | feature/40-literature-metrics f96b3fe | **yes** | clean | 6.1M |
| `/tmp/claude-1005/…/eda3cd1d…/scratchpad/wt-50-qwen` | feature/50-qwen-decomposer-models 9f0c2fd | **yes** | clean | 6.8M |
| `/tmp/claude-1005/…/eda3cd1d…/scratchpad/wt-base` | detached b03f1b1 | **yes** | clean | 6.3M |
| `/tmp/claude-1005/…/eda3cd1d…/scratchpad/wt-tip` | detached 9f0c2fd | **yes** | clean | 6.3M |

Four of six worktrees live under `/tmp/claude-1005/…` and will disappear on reboot (~25 MB of
checkouts). **Nothing is lost by that**: all four are clean, and every commit they hold
(f96b3fe, 9f0c2fd, b03f1b1) is reachable from a branch ref in
`/cta/users/fyilmaz/Thesis---QAv2/.git` — except that the two **detached** worktrees (wt-base,
wt-tip) have no branch of their own; b03f1b1 and 9f0c2fd stay reachable only through
`feature/50-qwen-decomposer-models`. Registration metadata would become stale
(`git worktree prune` territory), which is a cleanup, not a loss.

---

## Untracked files in the root working tree

`git status --short` (3 entries, all in `Slides/`):

- `Slides/Artifact slides review.pdf` — 2.7 MB, mtime 2026-08-25
- `Slides/Final Slide before meeting.pdf` — 3.1 MB, mtime 2026-08-27
- `Slides/Thesis_QA_Slide_Deck.pdf` — 4.6 MB, mtime 2026-08-25

`Slides/` also holds tracked `deck_corrections.md` (2026-08-27, committed by b1f51fe "docs(slides):
track deck_corrections.md so the slide lane can read it (Refs #14, #40)"), and two **untracked but
not reported** files that `.gitignore` covers: `thesis-deck-v2.html` (64 KB) and
`thesis-deck-v2.tex` (51 KB), both 2026-09-15 — these are the LaTeX/Beamer sources behind the
compiled deck, newer than any of the three PDFs. The three PDFs are build outputs/meeting
snapshots; no source file is unbacked.

---

## `runs/` (gitignored) — 1.3 GB total

Top-level by size:

- **406M** `runs/exp-001`, **406M** `runs/finetune_decomposer`, **408M** `runs/exp-026`
  (LoRA checkpoints — 1.2 GB of the 1.3 GB)
- 39M `runs/pool_sweep`; 8.8M `runs/anchor`; 5.6M `runs/exp-007`; 3.1M `runs/exp-017`;
  2.7M `runs/exp-016`; 2.3M `runs/exp-014`; 1.9M `runs/smoke`; 1.8M each `runs/exp-004`,
  `runs/exp-005`, `runs/exp-011`; 1.3M `runs/exp-008`; 1.2M each `runs/exp-009`, `runs/exp-012`;
  896K `runs/exp-013`; 878K `runs/exp-027`; 811K `runs/exp-023`, `runs/exp-025`;
  809K `runs/exp-028`; 802K `runs/exp-024`; 721K `runs/exp-022`; 696K `runs/exp-020`;
  693K `runs/exp-021`; 669K `runs/exp-015`; 75K `runs/exp-010`; 41K `runs/decomposer_musique`;
  18K `runs/exp-006`; 12K `runs/cpu_integration_check`; 5K `runs/downloads`;
  3.5K each `runs/exp-002`, `runs/exp-003`; 512B `runs/musique_decomposition_eval`
- loose scripts/logs at the top level: `precheck_a1_arm.py`, `precheck_g2_train.py`,
  `precheck_g2_eval.py`, `precheck_g3_arm.py`, `check_regen_parity.py`,
  `check_regen_parity_chat.py`, `check_ordering_invariant.py`, `g2_eval.sh`, `g2_waiter.sh`,
  `g3_waiter.sh`, `run_all_2.sh`, `rearmed_wait_and_launch{,_day2}.sh`, and the
  `exp-004-005-*` wait/BLOCKED logs from the 2026-08-19/20 GPU-contention episode.

**`runs/run.lock` does not exist** — no run holds the lock.

**`nvidia-smi` (2026-09-20 15:45):** one RTX 3090, **10 MiB / 24576 MiB used, 0% util**, only
`/usr/lib/xorg/Xorg` (4 MiB, PID 3761) attached. No ollama, no contention — the GPU is free.

---

## Verdict table

| branch | ahead/behind | conflict hunks | reviewed | verdict |
|---|---|---|---|---|
| feature/12-guided-vs-unguided-alt (c1b4aed) | 1 / 121 | 35 | no | **superseded / abandon-candidate** — PR #21 canonical (1ccea11); both reserved pieces on main |
| feature/13-lora-finetuning (a6892bb) | 3 / 118 | 76 | yes (PR #24 + #25) | **already merged** — squash 2f49c20; stale ref |
| feature/40-literature-metrics (f96b3fe) | 5 / 82 | 59 | yes (Gate-1, f96b3fe) | **already merged** — squash a99b573; stale ref |
| feature/50-anchor-set (db16e82) | 1 / 45 | 23 | no (successor b2 was) | **superseded** — re-implemented as e95e550, ADR 0033, renamed files |
| feature/50-qwen-decomposer-models (9f0c2fd) | 2 / 45 | 12 | partial (I-1 fix pass) | **superseded** — rebased forward into feef949, which cites it by SHA |
| worktree-agent-aa3be8e02852de0de (0ab6abb) | 0 / 120 | 0 | n/a | **merged ancestor — dead ref** |
| worktree-agent-ac35e0de8908c738f (0ab6abb) | 0 / 120 | 0 | n/a | **merged ancestor — dead ref** |
| worktree-agent-ace2f2912496463a0 (b6329ca) | 0 / 111 | 0 | n/a | **merged ancestor — dead ref** |
| worktree-agent-aebee2d46a9636399 (0ab6abb) | 0 / 120 | 0 | n/a | **merged ancestor — dead ref** |

**No branch is a merge candidate.** Not one of the nine holds code that is absent from `main`.
