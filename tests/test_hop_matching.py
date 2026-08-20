#!/usr/bin/env python3
"""Checks for hop-matched few-shot retrieval (issue #15, ADR 0022).

CPU only, no model weights, no network. Everything here runs on hand-made pool/query rows
and hand-made score matrices; the CLI checks use ``--dry-run``, which loads no model.

What it covers:

1. **The filter itself** — all three hop buckets, both hop sources (``gold`` from the query
   id, ``predictions`` from an external JSONL), and that a prediction which *disagrees* with
   the gold hop moves the candidate set, which is the whole point of the router condition.
2. **Every failure mode is a hard error with counts** — an unparseable pool id, an
   unparseable query id, a query with no prediction, a predicted hop that is not a hop
   depth, a hop bucket too small for ``top_k``, a queried hop the pool has no bucket for at
   all (including at ``min_candidates: 0``, PR #39 finding 1), a malformed / duplicated /
   empty predictions file, and a ``predictions`` source with no file. None of them may fall
   back to the mixed condition, and none may surface as an uncaught exception.
3. **The regression guard for the mixed condition** — with the feature off,
   ``_top_k_from_scores`` returns exactly what the pre-change implementation returned
   (re-implemented verbatim in this file), on a seeded score matrix that contains ties; the
   config default is ``enabled: false``; and a disabled filter is ``None`` rather than
   "everything allowed", so mixed keeps the original code path.
4. **The filtered top-k is computed inside the bucket** — the neighbours are the bucket's
   own top-k, in the bucket's own order, and there are still ``top_k`` of them.
5. **The CLI wiring** — ``--dry-run`` exits 0 and writes the run trail for the gold and the
   predictions source, and exits non-zero with counts on an infeasible bucket.

Run::

    .venv/bin/python -m unittest tests.test_hop_matching -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_SCRIPT = REPO_ROOT / "MusiQue" / "scripts" / "check_question_similarity.py"
SIMILARITY_CONFIG = REPO_ROOT / "configs" / "similarity.json"

for _path in (REPO_ROOT / "src", SIMILARITY_SCRIPT.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hop_matching as HM  # noqa: E402


def _import_similarity_script() -> Any:
    """Import check_question_similarity.py the way the script itself is run."""
    name = "check_question_similarity"
    spec = importlib.util.spec_from_file_location(name, SIMILARITY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CQS = _import_similarity_script()


def _settings(**overrides: Any) -> HM.HopMatchSettings:
    params: dict[str, Any] = {
        "enabled": True,
        "hop_source": "gold",
        "predictions_file": None,
        "predictions_id_field": "query_id",
        "predictions_hop_field": "predicted_hop",
        "min_candidates": HM.MIN_CANDIDATES_TOP_K,
    }
    params.update(overrides)
    return HM.HopMatchSettings(**params)


def _pool(per_hop: dict[int, int]) -> list[dict[str, Any]]:
    """Pool rows with ids in this dataset's shape, interleaved across buckets.

    Interleaved on purpose: if the filter accidentally sliced a contiguous range it would
    still pass on a bucket-sorted pool.
    """
    rows: list[dict[str, Any]] = []
    counters = {hop: 0 for hop in per_hop}
    remaining = dict(per_hop)
    while any(remaining.values()):
        for hop in sorted(per_hop):
            if remaining[hop] <= 0:
                continue
            i = counters[hop]
            fine = "" if hop == 2 else str(1 + (i % 2))
            rows.append(
                {
                    "id": f"{hop}hop{fine}__pool_{hop}_{i}",
                    "question": f"pool question {hop}/{i}",
                }
            )
            counters[hop] += 1
            remaining[hop] -= 1
    return rows


def _queries(hops: list[int]) -> list[dict[str, Any]]:
    return [
        {"id": f"{hop}hop{'' if hop == 2 else '1'}__q_{i}", "question": f"query {i} ({hop} hop)"}
        for i, hop in enumerate(hops)
    ]


def _reference_top_k(
    scores: np.ndarray, pool_rows: list[dict[str, Any]], top_k: int
) -> list[list[dict[str, Any]]]:
    """The pre-change ``_top_k_from_scores`` body, verbatim.

    This is the baseline the mixed condition is held to: if the refactor that added the
    ``allowed_indices`` parameter changed anything about the unfiltered path — ordering,
    tie handling, rounding, which keys are emitted — this reference disagrees with it.
    """
    all_neighbours: list[list[dict[str, Any]]] = []
    for i in range(scores.shape[0]):
        row_scores = scores[i]
        top_idx = np.argsort(row_scores)[::-1][:top_k]
        neighbours: list[dict[str, Any]] = []
        for j in top_idx:
            prow = pool_rows[int(j)]
            entry: dict[str, Any] = {
                "pool_id": prow.get("id"),
                "pool_index": prow.get("index"),
                "pool_question": prow.get("question"),
                "pool_question_masked_typed": prow.get("question_masked_typed"),
                "pool_question_masked_uniform": prow.get("question_masked_uniform"),
                "pool_few_shot_decomposition_musique": prow.get("few_shot_decomposition_musique"),
                "score": round(float(row_scores[j]), 4),
            }
            neighbours.append({k: v for k, v in entry.items() if v is not None})
        all_neighbours.append(neighbours)
    return all_neighbours


def _scores(num_queries: int, num_pool: int, *, seed: int = 42) -> np.ndarray:
    """A seeded score matrix rounded to 2 decimals, so ties actually occur."""
    rng = np.random.default_rng(seed)
    return np.round(rng.random((num_queries, num_pool), dtype=np.float32), 2)


class TestParseHop(unittest.TestCase):
    def test_the_three_id_shapes(self) -> None:
        self.assertEqual(HM.parse_hop_from_id("2hop__482757_12019"), 2)
        self.assertEqual(HM.parse_hop_from_id("3hop1__a_b"), 3)
        self.assertEqual(HM.parse_hop_from_id("4hop2__a_b"), 4)

    def test_unparseable_is_none(self) -> None:
        for value in ("", "hop2__x", "musique_123", None, 3, ["2hop__x"]):
            self.assertIsNone(HM.parse_hop_from_id(value), msg=repr(value))


class TestPoolBuckets(unittest.TestCase):
    def test_buckets_hold_the_row_indices_in_order(self) -> None:
        rows = _pool({2: 3, 3: 2, 4: 1})
        buckets = HM.group_pool_by_hop(rows)
        self.assertEqual(sorted(buckets), [2, 3, 4])
        self.assertEqual([len(v) for _, v in sorted(buckets.items())], [3, 2, 1])
        for hop, idxs in buckets.items():
            self.assertEqual(idxs, sorted(idxs))
            for i in idxs:
                self.assertEqual(HM.parse_hop_from_id(rows[i]["id"]), hop)

    def test_unparseable_pool_id_is_refused_with_counts(self) -> None:
        rows = _pool({2: 2, 3: 1})
        rows.append({"id": "no_hop_here", "question": "?"})
        with self.assertRaises(SystemExit) as ctx:
            HM.group_pool_by_hop(rows)
        message = str(ctx.exception)
        self.assertIn("1 of 4 pool rows", message)
        self.assertIn("no_hop_here", message)


class TestSettings(unittest.TestCase):
    def _cfg(self, **hop_match: Any) -> dict[str, Any]:
        block = {
            "enabled": False,
            "hop_source": "gold",
            "predictions_file": None,
            "predictions_id_field": "query_id",
            "predictions_hop_field": "predicted_hop",
            "min_candidates": "top_k",
        }
        block.update(hop_match)
        return {"hop_match": block, "_config_path": "<test>"}

    def test_committed_config_defaults_to_the_mixed_condition(self) -> None:
        """The shipped default must be mixed: a stage that passes no flag is unchanged."""
        cfg = json.loads(SIMILARITY_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("hop_match", cfg)
        self.assertFalse(cfg["hop_match"]["enabled"])
        self.assertEqual(cfg["hop_match"]["hop_source"], "gold")
        self.assertEqual(cfg["hop_match"]["min_candidates"], "top_k")
        self.assertIsNone(cfg["hop_match"]["predictions_file"])

    def test_cli_overrides_the_config_both_ways(self) -> None:
        enabled = HM.settings_from_config(self._cfg(), enabled=True)
        self.assertTrue(enabled.enabled)
        disabled = HM.settings_from_config(self._cfg(enabled=True), enabled=False)
        self.assertFalse(disabled.enabled)
        self.assertFalse(HM.settings_from_config(self._cfg()).enabled)

    def test_unknown_hop_source_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            HM.settings_from_config(self._cfg(), enabled=True, hop_source="vibes")
        self.assertIn("hop_source", str(ctx.exception))

    def test_predictions_source_without_a_file_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            HM.settings_from_config(self._cfg(), enabled=True, hop_source="predictions")
        self.assertIn("--hop-predictions", str(ctx.exception))

    def test_predictions_source_takes_the_file_from_config_or_cli(self) -> None:
        from_cfg = HM.settings_from_config(
            self._cfg(hop_source="predictions", predictions_file="a/b.jsonl"), enabled=True
        )
        self.assertEqual(from_cfg.predictions_file, Path("a/b.jsonl"))
        from_cli = HM.settings_from_config(
            self._cfg(hop_source="predictions", predictions_file="a/b.jsonl"),
            enabled=True,
            predictions_file="c/d.jsonl",
        )
        self.assertEqual(from_cli.predictions_file, Path("c/d.jsonl"))

    def test_min_candidates_resolution_and_validation(self) -> None:
        self.assertEqual(_settings().resolve_min_candidates(20), 20)
        self.assertEqual(_settings(min_candidates=3).resolve_min_candidates(20), 3)
        for bad in (-1, "twenty", 2.5, True):
            with self.assertRaises(SystemExit, msg=repr(bad)):
                HM.settings_from_config(self._cfg(min_candidates=bad), enabled=True)

    def test_record_hides_nothing_and_says_disabled(self) -> None:
        self.assertEqual(_settings(enabled=False).as_record(20), {"enabled": False})
        record = _settings().as_record(20)
        self.assertTrue(record["enabled"])
        self.assertEqual(record["min_candidates_resolved"], 20)


class TestFilterGoldSource(unittest.TestCase):
    def test_disabled_yields_no_filter_at_all(self) -> None:
        got = HM.build_hop_filter(
            query_rows=_queries([2]),
            pool_buckets={2: [0]},
            settings=_settings(enabled=False),
            top_k=1,
        )
        self.assertIsNone(got)

    def test_every_bucket_restricts_to_its_own_pool_rows(self) -> None:
        pool = _pool({2: 6, 3: 5, 4: 4})
        buckets = HM.group_pool_by_hop(pool)
        queries = _queries([2, 3, 4, 4, 2])
        hop_filter = HM.build_hop_filter(
            query_rows=queries, pool_buckets=buckets, settings=_settings(), top_k=4
        )
        assert hop_filter is not None
        self.assertEqual(hop_filter.query_hops, [2, 3, 4, 4, 2])
        self.assertEqual(hop_filter.query_hop_counts, {2: 2, 3: 1, 4: 2})
        self.assertEqual(hop_filter.pool_bucket_counts, {2: 6, 3: 5, 4: 4})
        self.assertEqual(hop_filter.min_candidates, 4)
        for allowed, hop in zip(hop_filter.allowed, hop_filter.query_hops):
            self.assertEqual(allowed, buckets[hop])
            self.assertTrue(all(HM.parse_hop_from_id(pool[i]["id"]) == hop for i in allowed))

    def test_summary_reports_the_counts_a_run_must_record(self) -> None:
        pool = _pool({2: 6, 3: 5, 4: 4})
        hop_filter = HM.build_hop_filter(
            query_rows=_queries([2, 4]),
            pool_buckets=HM.group_pool_by_hop(pool),
            settings=_settings(),
            top_k=4,
        )
        assert hop_filter is not None
        summary = hop_filter.summary()
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["pool_bucket_counts"], {"2": 6, "3": 5, "4": 4})
        self.assertEqual(summary["query_hop_counts"], {"2": 1, "4": 1})
        self.assertEqual(summary["candidates_per_query_min"], 4)
        self.assertEqual(summary["candidates_per_query_max"], 6)

    def test_unparseable_query_id_is_refused_with_counts(self) -> None:
        pool = _pool({2: 6, 3: 6, 4: 6})
        queries = _queries([2, 3]) + [{"id": "not_an_id", "question": "?"}]
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=queries,
                pool_buckets=HM.group_pool_by_hop(pool),
                settings=_settings(),
                top_k=2,
            )
        message = str(ctx.exception)
        self.assertIn("1 of 3 queries", message)
        self.assertIn("not_an_id", message)

    def test_a_bucket_too_small_for_top_k_is_refused_with_counts(self) -> None:
        pool = _pool({2: 6, 3: 6, 4: 2})
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=_queries([2, 4, 4]),
                pool_buckets=HM.group_pool_by_hop(pool),
                settings=_settings(),
                top_k=5,
            )
        message = str(ctx.exception)
        self.assertIn("hop 4: 2 pool candidates for 2 queries", message)
        self.assertIn("required candidates per query: 5", message)

    def test_a_queried_hop_absent_from_the_pool_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=_queries([4]),
                pool_buckets=HM.group_pool_by_hop(_pool({2: 6, 3: 6})),
                settings=_settings(),
                top_k=1,
            )
        self.assertIn("hop 4: 0 pool candidates", str(ctx.exception))

    def test_a_missing_bucket_is_refused_even_at_min_candidates_zero(self) -> None:
        """PR #39 review finding 1: this used to pass the size test and raise KeyError.

        ``0 < 0`` is false, so a hop with no bucket at all slipped through the feasibility
        guard and hit a bare ``pool_buckets[hop]``. It must be the documented hard error
        with counts, at every ``min_candidates`` value.
        """
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=[{"id": "4hop1__a_b_c_d"}],
                pool_buckets={2: [0, 1, 2]},
                settings=_settings(min_candidates=0),
                top_k=5,
            )
        message = str(ctx.exception)
        self.assertIn("hop 4: 0 pool candidates for 1 query", message)
        self.assertIn("no such bucket in the pool", message)
        self.assertIn("min_candidates=0", message)

    def test_an_empty_bucket_is_refused_even_at_min_candidates_zero(self) -> None:
        """Same guard from the other side: a present-but-empty bucket serves nobody."""
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=[{"id": "4hop1__a_b_c_d"}],
                pool_buckets={2: [0, 1], 4: []},
                settings=_settings(min_candidates=0),
                top_k=5,
            )
        self.assertIn("hop 4: 0 pool candidates", str(ctx.exception))

    def test_min_candidates_never_resolves_below_one(self) -> None:
        self.assertEqual(_settings(min_candidates=0).resolve_min_candidates(20), 1)
        self.assertEqual(_settings(min_candidates=0).resolve_min_candidates(0), 1)
        self.assertEqual(_settings().resolve_min_candidates(0), 1)

    def test_min_candidates_as_an_int_relaxes_the_requirement(self) -> None:
        pool = _pool({2: 6, 4: 2})
        hop_filter = HM.build_hop_filter(
            query_rows=_queries([4]),
            pool_buckets=HM.group_pool_by_hop(pool),
            settings=_settings(min_candidates=2),
            top_k=5,
        )
        assert hop_filter is not None
        self.assertEqual(hop_filter.min_candidates, 2)
        self.assertEqual(len(hop_filter.allowed[0]), 2)


class TestFilterPredictionsSource(unittest.TestCase):
    def _predictions_file(self, dirpath: Path, rows: list[dict[str, Any]]) -> Path:
        path = dirpath / "hop_predictions.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        return path

    def test_the_prediction_wins_over_the_gold_hop(self) -> None:
        """A wrong prediction must move the candidate set — that is the router condition."""
        pool = _pool({2: 6, 3: 6, 4: 6})
        buckets = HM.group_pool_by_hop(pool)
        queries = _queries([2, 3, 4])
        predicted = {queries[0]["id"]: 4, queries[1]["id"]: 3, queries[2]["id"]: 2}
        hop_filter = HM.build_hop_filter(
            query_rows=queries,
            pool_buckets=buckets,
            settings=_settings(hop_source="predictions", predictions_file=Path("p.jsonl")),
            top_k=3,
            predicted_hops=predicted,
        )
        assert hop_filter is not None
        self.assertEqual(hop_filter.query_hops, [4, 3, 2])
        self.assertEqual(hop_filter.allowed[0], buckets[4])
        self.assertEqual(hop_filter.allowed[2], buckets[2])

    def test_a_missing_prediction_is_refused_with_counts(self) -> None:
        pool = _pool({2: 6, 3: 6, 4: 6})
        queries = _queries([2, 3, 4])
        with self.assertRaises(SystemExit) as ctx:
            HM.build_hop_filter(
                query_rows=queries,
                pool_buckets=HM.group_pool_by_hop(pool),
                settings=_settings(hop_source="predictions", predictions_file=Path("p.jsonl")),
                top_k=3,
                predicted_hops={queries[0]["id"]: 2},
            )
        message = str(ctx.exception)
        self.assertIn("2 of 3 queries have no prediction", message)
        self.assertIn(queries[1]["id"], message)

    def test_a_predicted_hop_that_is_not_a_depth_is_refused(self) -> None:
        """PR #39 review finding 2: 0 and -1 are not hop depths, and must say so.

        Caught here rather than surfacing later as a bucket-size complaint about a hop
        that could never exist.
        """
        pool = _pool({2: 6, 3: 6, 4: 6})
        queries = _queries([2, 3])
        for bad in (0, -1):
            with self.assertRaises(SystemExit, msg=repr(bad)) as ctx:
                HM.build_hop_filter(
                    query_rows=queries,
                    pool_buckets=HM.group_pool_by_hop(pool),
                    settings=_settings(hop_source="predictions", predictions_file=Path("p.jsonl")),
                    top_k=3,
                    predicted_hops={queries[0]["id"]: 2, queries[1]["id"]: bad},
                )
            message = str(ctx.exception)
            self.assertIn("not a hop depth", message)
            self.assertIn("1 of 2 queries", message)

    def test_is_hop_depth_domain(self) -> None:
        for good in (1, 2, 3, 4, 17):
            self.assertTrue(HM.is_hop_depth(good), msg=repr(good))
        for bad in (0, -1, True, False, 2.0, "2", None):
            self.assertFalse(HM.is_hop_depth(bad), msg=repr(bad))

    def test_predictions_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._predictions_file(
                Path(tmp),
                [
                    {"query_id": "2hop__a", "predicted_hop": 3},
                    {"query_id": "3hop1__b", "predicted_hop": 2, "extra": "ignored"},
                ],
            )
            hops = HM.load_predicted_hops(path, id_field="query_id", hop_field="predicted_hop")
            self.assertEqual(hops, {"2hop__a": 3, "3hop1__b": 2})

    def test_predictions_file_field_names_are_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._predictions_file(Path(tmp), [{"qid": "2hop__a", "hop": 4}])
            hops = HM.load_predicted_hops(path, id_field="qid", hop_field="hop")
            self.assertEqual(hops, {"2hop__a": 4})

    def test_bad_predictions_files_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cases = {
                "not an int": [{"query_id": "2hop__a", "predicted_hop": "three"}],
                "zero hops": [{"query_id": "2hop__a", "predicted_hop": 0}],
                "negative hops": [{"query_id": "2hop__a", "predicted_hop": -1}],
                "boolean": [{"query_id": "2hop__a", "predicted_hop": True}],
                "missing id": [{"predicted_hop": 3}],
                "duplicate": [
                    {"query_id": "2hop__a", "predicted_hop": 3},
                    {"query_id": "2hop__a", "predicted_hop": 2},
                ],
                "empty": [],
            }
            for label, rows in cases.items():
                path = self._predictions_file(tmpdir, rows)
                with self.assertRaises(SystemExit, msg=label):
                    HM.load_predicted_hops(
                        path, id_field="query_id", hop_field="predicted_hop"
                    )
            missing = tmpdir / "absent.jsonl"
            with self.assertRaises(SystemExit):
                HM.load_predicted_hops(
                    missing, id_field="query_id", hop_field="predicted_hop"
                )


class TestMixedConditionRegressionGuard(unittest.TestCase):
    """With the feature off, retrieval output must be what it was before the feature."""

    def test_unfiltered_path_matches_the_pre_change_implementation(self) -> None:
        pool = _pool({2: 9, 3: 8, 4: 6})
        scores = _scores(7, len(pool))
        for top_k in (1, 5, 20, len(pool)):
            expected = _reference_top_k(scores, pool, top_k)
            got = CQS._top_k_from_scores(scores, pool, top_k)
            self.assertEqual(got, expected, msg=f"top_k={top_k}")
            self.assertEqual(
                CQS._top_k_from_scores(scores, pool, top_k, allowed_indices=None),
                expected,
                msg=f"explicit None, top_k={top_k}",
            )

    def test_allowing_the_whole_pool_reproduces_the_unfiltered_ranking(self) -> None:
        pool = _pool({2: 5, 3: 5, 4: 5})
        scores = _scores(4, len(pool), seed=7)
        allowed = [list(range(len(pool)))] * scores.shape[0]
        self.assertEqual(
            CQS._top_k_from_scores(scores, pool, 5, allowed_indices=allowed),
            _reference_top_k(scores, pool, 5),
        )

    def test_a_disabled_filter_is_none_not_everything_allowed(self) -> None:
        self.assertIsNone(
            HM.build_hop_filter(
                query_rows=_queries([2, 3, 4]),
                pool_buckets={},
                settings=_settings(enabled=False),
                top_k=20,
            )
        )


class TestFilteredTopK(unittest.TestCase):
    def test_the_top_k_is_ranked_inside_the_bucket(self) -> None:
        pool = _pool({2: 9, 3: 8, 4: 7})
        buckets = HM.group_pool_by_hop(pool)
        queries = _queries([2, 3, 4])
        hop_filter = HM.build_hop_filter(
            query_rows=queries, pool_buckets=buckets, settings=_settings(), top_k=5
        )
        assert hop_filter is not None
        scores = _scores(len(queries), len(pool), seed=11)
        got = CQS._top_k_from_scores(scores, pool, 5, allowed_indices=hop_filter.allowed)

        for i, hop in enumerate(hop_filter.query_hops):
            self.assertEqual(len(got[i]), 5, msg=f"query {i} should still get top_k")
            in_bucket_ids = {pool[j]["id"] for j in buckets[hop]}
            self.assertTrue({nb["pool_id"] for nb in got[i]} <= in_bucket_ids)
            # The same ranking as scoring the bucket on its own.
            bucket_rows = [pool[j] for j in buckets[hop]]
            bucket_scores = scores[i : i + 1, buckets[hop]]
            self.assertEqual(got[i], _reference_top_k(bucket_scores, bucket_rows, 5)[0])

    def test_a_bucket_smaller_than_top_k_returns_the_whole_bucket(self) -> None:
        """Only reachable with an explicit min_candidates; the default refuses this case."""
        pool = _pool({2: 6, 4: 2})
        buckets = HM.group_pool_by_hop(pool)
        hop_filter = HM.build_hop_filter(
            query_rows=_queries([4]),
            pool_buckets=buckets,
            settings=_settings(min_candidates=1),
            top_k=5,
        )
        assert hop_filter is not None
        got = CQS._top_k_from_scores(
            _scores(1, len(pool), seed=3), pool, 5, allowed_indices=hop_filter.allowed
        )
        self.assertEqual(len(got[0]), 2)


class TestDryRunCLI(unittest.TestCase):
    """The CLI wiring, through --dry-run: no weights, no network, no output file."""

    def _write_inputs(self, tmpdir: Path, *, pool: dict[int, int], hops: list[int]) -> tuple[Path, Path]:
        pool_path = tmpdir / "pool.jsonl"
        pool_path.write_text(
            "".join(json.dumps(r) + "\n" for r in _pool(pool)), encoding="utf-8"
        )
        query_path = tmpdir / "queries.jsonl"
        query_path.write_text(
            "".join(json.dumps(r) + "\n" for r in _queries(hops)), encoding="utf-8"
        )
        return pool_path, query_path

    def _run(self, tmpdir: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["QAV2_PATHS_CONFIG"] = str(REPO_ROOT / "configs" / "smoke_paths.json")
        return subprocess.run(
            [sys.executable, str(SIMILARITY_SCRIPT), "--dry-run", "--no-cache",
             "--mode", "raw", "--run-dir", str(tmpdir / "run"), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )

    def test_gold_source_dry_run_writes_the_trail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pool_path, query_path = self._write_inputs(
                tmpdir, pool={2: 4, 3: 4, 4: 4}, hops=[2, 3, 4, 2]
            )
            out = tmpdir / "must_not_exist.jsonl"
            proc = self._run(
                tmpdir,
                "--pool-file", str(pool_path),
                "--query-file", str(query_path),
                "--top-k", "3",
                "--n", "10",
                "--hop-match",
                "--out", str(out),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertFalse(out.exists(), "a dry run must not write the output JSONL")
            metrics = json.loads((tmpdir / "run" / "similarity_metrics.json").read_text())
            self.assertTrue(metrics["dry_run"])
            self.assertTrue(metrics["hop_match"]["enabled"])
            self.assertEqual(metrics["hop_match"]["hop_source"], "gold")
            self.assertEqual(metrics["hop_match"]["min_candidates_resolved"], 3)
            self.assertEqual(metrics["total_queries"], 4)
            per_file = metrics["hop_match_per_query_file"][0]["hop_match"]
            self.assertEqual(per_file["query_hop_counts"], {"2": 2, "3": 1, "4": 1})
            self.assertEqual(per_file["pool_bucket_counts"], {"2": 4, "3": 4, "4": 4})
            for name in ("similarity_config.json", "similarity_notes.md"):
                self.assertTrue((tmpdir / "run" / name).exists(), msg=name)

    def test_predictions_source_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pool_path, query_path = self._write_inputs(
                tmpdir, pool={2: 4, 3: 4, 4: 4}, hops=[2, 3]
            )
            queries = _queries([2, 3])
            preds = tmpdir / "preds.jsonl"
            preds.write_text(
                "".join(
                    json.dumps({"query_id": q["id"], "predicted_hop": 4}) + "\n" for q in queries
                ),
                encoding="utf-8",
            )
            proc = self._run(
                tmpdir,
                "--pool-file", str(pool_path),
                "--query-file", str(query_path),
                "--top-k", "3",
                "--n", "10",
                "--hop-match",
                "--hop-source", "predictions",
                "--hop-predictions", str(preds),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            metrics = json.loads((tmpdir / "run" / "similarity_metrics.json").read_text())
            per_file = metrics["hop_match_per_query_file"][0]["hop_match"]
            self.assertEqual(per_file["query_hop_counts"], {"4": 2})
            self.assertEqual(metrics["hop_match"]["hop_source"], "predictions")

    def test_dry_run_refuses_an_infeasible_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pool_path, query_path = self._write_inputs(
                tmpdir, pool={2: 4, 3: 4, 4: 1}, hops=[4]
            )
            proc = self._run(
                tmpdir,
                "--pool-file", str(pool_path),
                "--query-file", str(query_path),
                "--top-k", "3",
                "--n", "10",
                "--hop-match",
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("hop 4: 1 pool candidates", proc.stdout + proc.stderr)

    def test_mixed_dry_run_needs_no_hop_information(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pool_path = tmpdir / "pool.jsonl"
            pool_path.write_text(
                json.dumps({"id": "no_hop_prefix", "question": "q"}) + "\n", encoding="utf-8"
            )
            query_path = tmpdir / "queries.jsonl"
            query_path.write_text(
                json.dumps({"id": "also_no_prefix", "question": "q"}) + "\n", encoding="utf-8"
            )
            proc = self._run(
                tmpdir,
                "--pool-file", str(pool_path),
                "--query-file", str(query_path),
                "--top-k", "1",
                "--n", "10",
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            metrics = json.loads((tmpdir / "run" / "similarity_metrics.json").read_text())
            self.assertEqual(metrics["hop_match"], {"enabled": False})
            self.assertEqual(metrics["hop_match_per_query_file"][0]["queries"], 1)

    def test_the_two_hop_flags_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pool_path, query_path = self._write_inputs(tmpdir, pool={2: 2}, hops=[2])
            proc = self._run(
                tmpdir,
                "--pool-file", str(pool_path),
                "--query-file", str(query_path),
                "--hop-match",
                "--no-hop-match",
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not allowed with argument", proc.stderr)


if __name__ == "__main__":
    unittest.main()
