# 0041 — Memo audit: Tier 0 only, and how the Han & Gardent re-score is scored

- **Status:** accepted. Item 1 is Jahid's (2026-10-31, reserved by the lead that day; run
  decided by Jahid on 2026-11-27). Items 2–7 are implementer conventions written with exp-035
  (Claude subagent under the lead's brief), Jahid's to reverse.
- **Date:** 2026-11-27.

## Context

The external memo *"Jev, State of the Art, and a Plan to Improve Hop-Aware Question
Decomposition on MuSiQue"* (repo root) compares the thesis's EM / SARI / GED with Han & Gardent
(NAACL 2025, aclanthology 2025.naacl-long.55) and warns (Part B1, Caveats, C3 item 6) that the
comparison holds only if the thesis scores with the official `break-evaluator` under their
normalisation. Its table C4 row 0 ("Stats re-analysis (Holm, per-hop, official scorer)", 0 GPU)
is the item kept. The repo's evaluator ports the Break trio with three named deviations
(docs/METRICS.md §2.2, ADR 0026): no spaCy lemmas in GED, no `return`-strip / `;`-split, and a
node cap + time budget instead of dropping slow items.

Reading Han & Gardent's own code (github.com/hankelvin/complex_question_decomposition @
`9c9a6699ec440175192004a12fb2526c1d7a40c8`) showed three things the paper does not say:

- their fork's metric modules (`sari_hook.py`, `sequence_matcher.py`, `graph_matcher.py`) are
  upstream's; only the driver changed;
- the "lower-case and strip question marks" of footnote 11 is, in code, `re.sub(r'\?+$', '', …)`
  on the **joined** `@@SEP@@` string (only the last step's `?` goes) plus the lowercasing that
  upstream EM already does; EM and SARI see the strip, GED does not;
- they rewrote MuSiQue's `entity >> relation` gold sub-questions into natural language with
  their own templates (paper §4, footnote 10), so their gold is not MuSiQue's stored gold.

## Decision

1. **Tier 0 only, no new arms** (Jahid, 2026-10-31). From the memo, only re-analysis of stored
   outputs runs; the experiment ceiling holds. exp-035 is that item.
2. **The primary re-score is the fork's code path (`hg_fork`)**, implemented in
   `scripts/break_hg_rescore.py`: `clean_lingering_operators_cot` → `format_qdmr` (including
   its `return` removal and `;` split) → `to_string` → final-`?` strip; EM lowercased; official
   SARI; official GED with spaCy-lemma node cost, the optimizer run to exhaustion, and an item
   over 180 s dropped from the GED mean and counted.
3. **Footnote 11 read literally (`hg_paper_literal`) is reported beside it as a sensitivity**,
   never as the headline: every sub-question and the question lowercased and `?`-stripped first.
4. **spaCy 3.x replaces the fork's pinned 2.1.9** (which needs Python ≤ 3.8), with the versions
   recorded in the metrics. GED is therefore "official algorithm, spaCy-3 lemmas"; a reader is
   told so wherever the number appears. spaCy 3 keeps capitals in lemmas (checked: "Who is the
   Spouse of Caroline LeRoy" → `who be the Spouse of Caroline LeRoy`), so the node cost is
   case-sensitive; whether 2.1.9 lowercased lemmas is believed but not verified here.
   **The GED budget is checked only between the optimizer's successive values**: an item whose
   last value arrives before 180 s but whose search runs on past 180 s is still scored, and a
   single long step cannot be cut off; the official `@exit_after(180)` would drop it. Kept as is
   so the code at HEAD reproduces exp-035; the effect is bounded by the drop count (1 item).
5. **The re-score sits beside the repo's evaluator, never in place of it.**
   `musique_decompositions_evaluator.py`, its config and every recorded value are unchanged;
   the report card (ADR 0035/0037) keeps the repo port. The Han & Gardent-style numbers are for
   the related-work comparison only, quoted with item 6's caveat.
6. **Wherever a thesis number is placed next to Han & Gardent's table, the gold difference is
   stated** (stored MuSiQue gold vs their template-rewritten gold) — the scorer is matched, the
   gold is not, so the comparison is "same scorer, near-same gold", not identical footing.
7. **Holm (ADR 0036) applies to the one family exp-035 pre-registered** (pool-only FT vs each
   prompting arm × EM/SARI/GED × overall + hop 2/3/4 = 48 tests); per-hop means for every arm
   and the stored under/over-decomposition rates per hop are descriptive.

## Consequences

- The thesis can quote EM / SARI / GED computed by Han & Gardent's scorer path on the same
  2,417 rows exp-030 used, and state exactly what still differs (spaCy version, gold rewrite).
- `requirements.txt` is not changed: spaCy and `en_core_web_sm` are needed only by this
  re-score script, which imports `spacy` lazily, and are installed in a CPU-only venv; the versions
  that ran exp-035 are recorded under `environment` in experiments/exp-035/metrics.json (no pip
  freeze is committed).
- Mistral-7B's exp-030 predictions carry un-re-parsed scaffold lines (95/2,417); the re-score
  inherits them. Re-parsing them would be a new experiment (ADR 0039 item 4).
- `experiments/exp-035/parity_check.py` runs the fork's own functions against the re-score.
  It executes third-party code, so it is committed but run only on the owner's say-so; until
  then the port is checked by hand-computed unit tests (`tests/test_break_hg_rescore.py`).

## Alternatives considered

- **Vendor and run the fork as-is.** Needs Python ≤ 3.8, spaCy 2.1.9, pandas 0.25 and network
  downloads; also runs third-party code in the evaluation path. Rejected for the scorer; kept
  as the optional parity check above.
- **Change the repo evaluator to the fork's conventions.** Would move every recorded number
  and the report card for a related-work comparison. Rejected (item 5).
- **Rewrite MuSiQue's `>>` gold with Han & Gardent's templates.** Their predictions were
  trained on that NL gold; ours were trained/prompted on the stored gold, so rewriting only the
  gold would penalise every arm for a format it was never asked to produce. Not done; stated
  instead (item 6).
