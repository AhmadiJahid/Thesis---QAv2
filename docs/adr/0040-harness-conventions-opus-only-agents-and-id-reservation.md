# 0040 — Harness conventions: Opus-only agents are standing, and exp ids / ADR numbers are reserved on main before use

- **Status:** accepted (2026-10-26). Item 1 is Jahid's decision; item 2 is an implementer
  convention adopted under issue #62 and is his to reverse.
- **Amends** [0032](./0032-jahid-2026-09-20-go-ahead-for-the-50-block.md) item 4 (its scope only).
- **Context:** Issue #62 (harness refresh, PR #63). Two harness rules were in force without a
  record that stated them as standing: the model every agent runs on, and how a new experiment id
  or ADR number is claimed when several sessions are active.

## Settled by Jahid

1. **Agents run on Opus only, as a standing rule — not only for the issue #50 block.**
   - On 2026-09-20 Jahid said *"You can only use OPUS 5 agents"* when green-lighting the #50
     block. ADR 0032 item 4 recorded it scoped "for this block", with `experiment-runner`
     (then defined on Sonnet) spawned with an Opus override.
   - Every agent has been spawned on Opus since then.
   - On 2026-10-26, in session, Jahid approved the #62 recommendation ("go as recommended"). That
     recommendation explicitly proposed changing `experiment-runner` from Sonnet to Opus because
     the Sonnet definition contradicted his Opus-only rule. So the rule outlived the block, and
     `.claude/agents/experiment-runner.md` now says `model: opus`.
   - The lead session also runs on Opus: the project `.claude/settings.json` sets
     `"model": "opus"` (commit `84eb535`, Jahid's choice).
   - If Jahid names another model for any role, that supersedes this item.

## Implementer's convention (issue #62; Jahid may reverse)

2. **An exp id or ADR number is claimed on `main`, and pushed, before anything cites it.**
   - **Why:** on 2026-10-14 two concurrent lead sessions took the same numbers. One lane had to
     renumber after review: `7fd87c5` ("renumber to exp-030 / ADR 0037") and then `3006169`
     ("renumber the sampling-variance ADR to 0038 — 0037 taken on main"). Numbers are gap-free and
     never reused (docs/adr/README.md), so a number has to be owned before anyone cites it.
   - **exp id:** commit the pre-run row to `experiments/log.md` (the schema in that file, metrics
     `PENDING`). This is an experiment-only commit, allowed directly on main (CLAUDE.md, Git).
   - **ADR number:** commit **an index row only** in `docs/adr/README.md`, with status
     `Reserved (<who>, <date>) — <one line>` and no link. **No stub ADR file is written.**
     docs/adr/README.md says an ADR "records a decision that was made. It is never a proposal";
     a stub file would be one, so the reservation lives only in the index. The ADR file and the
     row's link and real status land together when the decision has been made.
   - **Abandoned reservation:** the row stays, with its status changed to
     `abandoned (number never reused)`.
   - The helper `.claude/skills/next-id/next_id.sh` counts every index row, `Reserved` and
     `abandoned` rows included, so a claimed number is never handed out again.
   - The push to `main` is the claim. If it is rejected, fetch, re-scan and take the next number.
   - The particular row format is arbitrary. What matters is that the claim exists on `main`.

## Consequences

- `CLAUDE.md` cites this ADR for the agents' model. ADR 0032 item 4 stays as the history.
- The ADR index can now contain a row with no file behind it, for as long as the reservation is
  open or once it is abandoned. Every other row still links its file.
- A session that is offline, or pushing under the wrong `gh` account, cannot claim a number. It
  has to wait, not guess.

## Alternatives considered

- **Stub ADR files as reservations** (the first draft of the `next-id` skill, PR #63). Dropped
  in the PR #63 fix pass: a stub file is a proposal sitting in `docs/adr/`, which the README forbids.
- **Renumber on collision after the fact** (what happened on 2026-10-14). This costs a re-review,
  and citations made in the meantime go stale.
