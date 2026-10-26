---
name: gpu-queue
description: Launch a GPU experiment safely on LittleGazor — queue it behind the run lock and other users' GPU memory with a detached tmux waiter, and watch it without sleep-polling. Use whenever a decomposer/fine-tune/inference run needs the GPU, or a run must wait for one already in flight.
---

# gpu-queue

Sources: CLAUDE.md (Two tracking surfaces, Gates 2, Git), docs/compute.md (Run lock convention,
Launching a run), `.claude/agents/experiment-runner.md`, and the committed launcher
`experiments/exp-033/waiter.sh` (exp-031's `runs/exp-031/waiter.sh` is the same pattern).
Template: `waiter.template.sh` in this directory.

## Procedure

1. **Reserve the id and write the pre-run row first.** Use the `next-id` skill. The log row
   (hypothesis + exact commands + commit + seed + model + eval set, metrics `PENDING — …`) is
   committed to `main` and pushed **before** launch. Experiment-only commits may go straight to
   main; nothing that changes shared pipeline code may (CLAUDE.md, Git).
2. **Code is merged and committed.** Every run is from committed code + committed config + a
   fixed seed (Gates 2). Note the merged SHA.
3. **Detached worktree at that SHA** — never run from the shared checkout, whose HEAD other
   sessions move:
   `git worktree add --detach /cta/users/fyilmaz/worktrees/<exp-id> <sha>`
   Data and outputs stay as absolute paths into the main checkout's `runs/` (gitignored).
4. **Write the waiter.** Copy `waiter.template.sh` to `experiments/<exp-id>/waiter.sh`, fill
   `EXP`, `WT`, `EXPECTED_SHA` and the `step` lines with the commands from the log row. It:
   - refuses to start unless the worktree is clean and at `EXPECTED_SHA`;
   - waits until `runs/run.lock` is absent **and** GPU free memory ≥ `MIN_FREE_MIB`
     (9000 MiB in exp-031/exp-033; configurable by env var — size it to the model);
   - acquires the lock atomically (`set -o noclobber`), contents `<exp-id> <ISO> experiment-runner`;
   - appends one line per step to `runs/<exp-id>/chain_status.txt`, ending `exit rc=N <ISO>`;
   - releases the lock in an EXIT trap, only if it holds it.
   Commit the waiter with the log row (step 1) or right after it.
5. **Dry run first.** `DRY_RUN=1 SCRATCH=/tmp/.../<exp-id>-dry bash experiments/<exp-id>/waiter.sh`
   — no lock, no GPU wait, `--dry-run --dry-run-limit 3`. It must exit 0 before the real launch.
6. **Launch detached** (docs/compute.md: tmux, log under `runs/<exp-id>/`):
   `tmux new-session -d -s <expNNN> "bash /cta/users/fyilmaz/worktrees/<exp-id>/experiments/<exp-id>/waiter.sh"`
   Then confirm `tail -2 runs/<exp-id>/chain_status.txt` shows `waiting since` or `chain start`.
7. **Watch without babysitting.** One background shell (Bash `run_in_background`), not a
   foreground sleep loop:
   `until grep -q '^exit rc=' runs/<exp-id>/chain_status.txt; do sleep 600; done; tail -5 runs/<exp-id>/chain_status.txt`
   It re-invokes you when the chain exits. Do not message in-flight agents with progress checks.
8. **On exit:** read `run.log` and the outputs; complete the log row (metrics or the error
   tail) and commit `experiments/<exp-id>/` (config snapshot, metrics JSON, run note). Confirm the
   lock is gone (`cat runs/run.lock`). Never re-launch with changed hyperparameters — that is a
   new experiment and the lead's call.

## Never

- Kill or pause another user's GPU process (e.g. ollama); wait for it instead.
- Clear a lock you do not hold unless docs/compute.md's stale-lock rule is met (24 h **and**
  nothing live in `nvidia-smi` / `ps`).
- Claim a metric before it is in the log row.
