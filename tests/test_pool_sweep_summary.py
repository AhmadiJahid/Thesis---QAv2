#!/usr/bin/env python3
"""Checks for the sweep summary table written by ``scripts/pool_sweep_orchestrator.py``.

``summary/all_runs.csv`` is the table the pool-size and under-decomposition questions get
asked from, so two things about it are pinned here: the directional step-count metrics are
columns of it (they used to stop at the per-run metrics JSON — issue #20, finding 3), and a
table written with a different column set is refused rather than appended to under
misaligned headers.

Run::

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = REPO_ROOT / "scripts" / "pool_sweep_orchestrator.py"

#: The five metrics finding 3 asked for. step_count_mae repeats step_count_abs_error_mae by
#: design, so it is listed with the family it belongs to.
DIRECTIONAL_FIELDS = (
    "step_count_mae",
    "mean_signed_step_count_error",
    "over_decomposition_rate",
    "under_decomposition_rate",
    "step_count_exact_rate",
)


def _import_orchestrator() -> Any:
    name = "pool_sweep_orchestrator"
    spec = importlib.util.spec_from_file_location(name, ORCHESTRATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORCH = _import_orchestrator()


class TestSummaryColumns(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="qav2_sweep_summary_test_")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _row(self) -> dict[str, Any]:
        """One fully populated summary row: 3 over-, 1 under-, 6 exact-step-count items."""
        row = {field: "" for field in ORCH.SUMMARY_FIELDS}
        row.update(
            {
                "run_key": "size1000_imbalanced_trial0__biencoder_only__uniform",
                "size": 1000,
                "balance": "imbalanced",
                "trial": 0,
                "pool_seed": 42,
                "variant": "biencoder_only",
                "mode": "uniform",
                "dev_seed": 42,
                "dev_per_hop": 10,
                "dev_sample_sha256": "deadbeef",
                "num_evaluated": 10,
                "step_count_mae": 0.5,
                "mean_signed_step_count_error": 0.2,
                "over_decomposition_rate": 0.3,
                "under_decomposition_rate": 0.1,
                "step_count_exact_rate": 0.6,
            }
        )
        return row

    def test_directional_metrics_are_columns(self) -> None:
        for field in DIRECTIONAL_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, ORCH.SUMMARY_FIELDS)
                # _METRIC_FIELDS is what gets copied out of the eval metrics JSON.
                self.assertIn(field, ORCH._METRIC_FIELDS)

    def test_directional_values_reach_the_csv(self) -> None:
        """The three rates must satisfy over + under + exact = 1.0 in the written row."""
        summary_csv = self.tmp / "all_runs.csv"
        ORCH._append_summary(summary_csv, self._row())

        with summary_csv.open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["over_decomposition_rate"], "0.3")
        self.assertEqual(rec["under_decomposition_rate"], "0.1")
        self.assertEqual(rec["step_count_exact_rate"], "0.6")
        self.assertEqual(rec["mean_signed_step_count_error"], "0.2")
        self.assertEqual(rec["step_count_mae"], "0.5")
        self.assertAlmostEqual(
            float(rec["over_decomposition_rate"])
            + float(rec["under_decomposition_rate"])
            + float(rec["step_count_exact_rate"]),
            1.0,
            places=9,
        )

    def test_stale_header_is_refused(self) -> None:
        """A table written before these columns existed must not be appended to."""
        summary_csv = self.tmp / "all_runs.csv"
        old_fields = [f for f in ORCH.SUMMARY_FIELDS if f not in DIRECTIONAL_FIELDS]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=old_fields).writeheader()

        with self.assertRaises(SystemExit) as ctx:
            ORCH._append_summary(summary_csv, self._row())
        message = str(ctx.exception)
        self.assertIn("REFUSING to append", message)
        self.assertIn("step_count_exact_rate", message)

    def test_mixed_dev_sample_is_still_refused(self) -> None:
        """The pre-existing dev-identity refusal survives the header check."""
        summary_csv = self.tmp / "all_runs.csv"
        ORCH._append_summary(summary_csv, self._row())
        other = self._row()
        other["run_key"] = "size1000_imbalanced_trial1__biencoder_only__uniform"
        other["dev_sample_sha256"] = "cafebabe"
        with self.assertRaises(SystemExit) as ctx:
            ORCH._append_summary(summary_csv, other)
        self.assertIn("different dev sample", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
