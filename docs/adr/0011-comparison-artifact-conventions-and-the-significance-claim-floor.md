# 0011. Comparison-Artifact Conventions and the Significance-Claim Floor

- **Status**: Accepted (provisional items pending Jahid/supervisor, marked below)
- **Date**: 2026-08-18

Amends [0009](./0009-paired-bootstrap-and-mcnemar-as-the-significance-protocol.md) (the significance protocol of record) with three conventions established by PR #22 (issue #20 hardening). PR #22's Gate-1 review found the conventions sound but unrecorded; this ADR is that record. None of these came from the supervisor.

*(Amended 2026-08-20 by [0020](./0020-prior-work-re-analysis-convention.md): analysis notes reporting a statistics battery carry a machine-readable JSON companion beside the note.)*

## Context

PR #22 hardened `--compare` in `scripts/musique_decompositions_evaluator.py`. Three of its choices are protocol-shaped — they will appear in every run's config snapshot and every comparison output, and in November they would read as settled decisions unless their provenance is recorded.

## Decision

**1. Significance-claim floor (provisional — the number is the implementer's, not the supervisor's).** `paired_comparison.min_items_for_significance_claim = 30` in `configs/musique_eval.json`: below 30 aligned items, every comparison row is flagged `underpowered` and the output carries a warning. It is a *reporting guard* — no result is suppressed, and the boolean `significant` is still computed per ADR 0009. Changing the floor is a one-line config edit; Jahid and his supervisor confirm or change the number. Independently of the floor, each McNemar row records its minimum attainable p-value (derived, not chosen).

**2. Versioned per-item artifact.** The evaluator's per-item output is an object with header `schema: musique_decomposition_per_item/1`, carrying the composite weights and scale it was scored under, plus the items. This is a format break: pre-PR-#22 bare-list dumps under `runs/` are refused by `--compare` and must be regenerated. Nothing else in the repo reads that file.

> **Note added 2026-08-20 (PR #36, issues #30 and #6), touching conventions 2 and 3.** Three things changed around them, none of them a version bump.
>
> - **The bare-list format became readable on one explicit path.** `--compare --v1-per-item` reads v1-format bare-list per-item files as *prior-work* inputs under [ADR 0020](./0020-prior-work-re-analysis-convention.md). Convention 2's refusal still stands everywhere else: the flag is opt-in, a bare-list file without it is still refused (and the error now names the flag), and the comparison output records that its inputs are v1 and carry no commit SHA. For a v2 run, regenerating the per-item file is still the answer.
> - **The comparison metrics JSON gained additive keys**: `t_test` and `tests_reported` (the paired t-test of issue #30 / [ADR 0017](./0017-triage-of-the-2026-08-12-transcript-cross-check.md) item 4) and `v1_format_inputs` (`null` on the normal path).
> - **Convention 3's recorded key became tri-state.** `config_weights_match_per_item_files` is `true`/`false` as before, and `null` on the v1 path only — v1 files stamp no weights, so there is nothing for the config's weights to match or differ from, and a `false` there would read as a detected mismatch. The file-vs-file refusal of convention 3 is skipped on that path for the same reason (both headers are synthesized from one config, so it could only ever pass); what can be said is recorded in `v1_format_inputs.composite_score_weights_source` instead. Convention 3's warn-and-record rule for v2 inputs is untouched.
>
> **The comparison artifact itself remains unversioned** — it carries no `schema` key, so a consumer cannot detect these additions from a version. This ADR versions the *per-item* file and nothing else. Whether the comparison output should be versioned too is **open and Jahid's call**: PR #36 deliberately added no key, since introducing a version is a new convention, not an implementation detail. In-repo consumers tolerate the additions — `scripts/compare_decomposer_arms.py` reads the new block with `.get`, as it already did for `underpowered`. The shim's conventions are documented in `docs/METRICS.md` §5.

**3. Weight-mismatch rule (provisional house rule).** `--compare` **refuses** when the two per-item files were scored under different weights (file-vs-file — a comparison across scoring regimes is not a comparison, same stance as the id-mismatch refusal). When the files agree with each other but differ from the *current config*, it **warns and proceeds**, recording `config_weights_match_per_item_files: false` — deliberate, so older runs remain comparable to each other. The Gate-1 review noted this departs from issue #20's stated direction (refuse in both cases) and left it as Jahid's call; until he rules, warn-and-record is the house rule.

## Consequences

- Comparisons on small subsets (e.g. per-hop slices near n=30) will carry `underpowered` flags in thesis tables; that is intended.
- Any script that consumed the old bare-list per-item format must move to `/1`; future format changes bump the schema version.
- If Jahid hardens rule 3 to refusal, older-run comparability under changed config weights is lost by design; record the change here.

## Alternatives considered

For the floor: no floor (rejected by the #20 review — `significant: true` at n=3 is meaningless); a statistically derived floor (rejected as false precision — the honest object is the recorded minimum attainable p, which is emitted). For rule 3: refuse on config-vs-file mismatch (issue #20's literal direction) — deferred to Jahid, recorded above.
