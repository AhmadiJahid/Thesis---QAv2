# 0011. Comparison-Artifact Conventions and the Significance-Claim Floor

- **Status**: Accepted (provisional items pending Jahid/supervisor, marked below)
- **Date**: 2026-08-18

Amends [0009](./0009-paired-bootstrap-and-mcnemar-as-the-significance-protocol.md) (the significance protocol of record) with three conventions established by PR #22 (issue #20 hardening). PR #22's Gate-1 review found the conventions sound but unrecorded; this ADR is that record. None of these came from the supervisor.

## Context

PR #22 hardened `--compare` in `scripts/musique_decompositions_evaluator.py`. Three of its choices are protocol-shaped — they will appear in every run's config snapshot and every comparison output, and in November they would read as settled decisions unless their provenance is recorded.

## Decision

**1. Significance-claim floor (provisional — the number is the implementer's, not the supervisor's).** `paired_comparison.min_items_for_significance_claim = 30` in `configs/musique_eval.json`: below 30 aligned items, every comparison row is flagged `underpowered` and the output carries a warning. It is a *reporting guard* — no result is suppressed, and the boolean `significant` is still computed per ADR 0009. Changing the floor is a one-line config edit; Jahid and his supervisor confirm or change the number. Independently of the floor, each McNemar row records its minimum attainable p-value (derived, not chosen).

**2. Versioned per-item artifact.** The evaluator's per-item output is an object with header `schema: musique_decomposition_per_item/1`, carrying the composite weights and scale it was scored under, plus the items. This is a format break: pre-PR-#22 bare-list dumps under `runs/` are refused by `--compare` and must be regenerated. Nothing else in the repo reads that file.

**3. Weight-mismatch rule (provisional house rule).** `--compare` **refuses** when the two per-item files were scored under different weights (file-vs-file — a comparison across scoring regimes is not a comparison, same stance as the id-mismatch refusal). When the files agree with each other but differ from the *current config*, it **warns and proceeds**, recording `config_weights_match_per_item_files: false` — deliberate, so older runs remain comparable to each other. The Gate-1 review noted this departs from issue #20's stated direction (refuse in both cases) and left it as Jahid's call; until he rules, warn-and-record is the house rule.

## Consequences

- Comparisons on small subsets (e.g. per-hop slices near n=30) will carry `underpowered` flags in thesis tables; that is intended.
- Any script that consumed the old bare-list per-item format must move to `/1`; future format changes bump the schema version.
- If Jahid hardens rule 3 to refusal, older-run comparability under changed config weights is lost by design; record the change here.

## Alternatives considered

For the floor: no floor (rejected by the #20 review — `significant: true` at n=3 is meaningless); a statistically derived floor (rejected as false precision — the honest object is the recorded minimum attainable p, which is emitted). For rule 3: refuse on config-vs-file mismatch (issue #20's literal direction) — deferred to Jahid, recorded above.
