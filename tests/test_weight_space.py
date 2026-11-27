#!/usr/bin/env python3
"""Checks for exp-038's weight-space repair helpers (ADR 0042).

CPU only, numpy only, no model weights, no network. The torch side
(components/decomposer/weight_space_repair.py) cannot run here; every piece of arithmetic it
uses to decide a number lives in src/weight_space.py and is checked below. The projection
functions are written to work unchanged on torch tensors, so checking them on numpy arrays
checks the formula the GPU run applies.

Run::

    python -m pytest tests/test_weight_space.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import weight_space as ws  # noqa: E402


class TestLabels(unittest.TestCase):
    def test_four_cases(self) -> None:
        self.assertEqual(ws.label_boundary(1, 3, True), ws.PREMATURE_STOP)
        self.assertEqual(ws.label_boundary(1, 3, False), ws.CONTINUE_NEEDED)
        self.assertEqual(ws.label_boundary(3, 3, True), ws.CORRECT_STOP)
        self.assertEqual(ws.label_boundary(3, 3, False), ws.LATE_CONTINUE)
        self.assertEqual(ws.label_boundary(4, 3, True), ws.LATE_STOP)

    def test_bad_indices_refused(self) -> None:
        with self.assertRaises(ValueError):
            ws.label_boundary(0, 3, True)

    def test_records_of_an_under_decomposed_item(self) -> None:
        recs = ws.boundary_records(2, 4, True)
        self.assertEqual([r["label"] for r in recs], [ws.CONTINUE_NEEDED, ws.PREMATURE_STOP])
        self.assertEqual([r["j"] for r in recs], [1, 2])

    def test_records_of_an_over_decomposed_item(self) -> None:
        recs = ws.boundary_records(3, 2, True)
        self.assertEqual([r["label"] for r in recs], [ws.CONTINUE_NEEDED, ws.LATE_CONTINUE, ws.LATE_STOP])


class TestBoundaries(unittest.TestCase):
    def test_two_steps_closed_by_eos(self) -> None:
        toks = ["Who", "wrote", "X", "\n", "#", "1", "born"]
        self.assertEqual(ws.step_boundaries(toks, True), [2, 6])

    def test_truncated_last_step_has_no_boundary(self) -> None:
        toks = ["Who", "wrote", "X", "\n", "#", "1"]
        self.assertEqual(ws.step_boundaries(toks, False), [2])

    def test_blank_lines_and_stray_spaces(self) -> None:
        toks = ["a", " ", "\n", "\n", "b", "\n"]
        # 'a' ends step 1 (look past the space token); 'b' ends step 2; a trailing newline
        # before EOS adds no empty step.
        self.assertEqual(ws.step_boundaries(toks, True), [0, 4])

    def test_merged_text_and_newline_is_refused(self) -> None:
        with self.assertRaises(ws.MixedNewlineToken):
            ws.step_boundaries(["a", "b.\n", "c"], True)


class TestStepCountSummary(unittest.TestCase):
    def test_rates_overall_and_per_hop(self) -> None:
        s = ws.step_count_summary([2, 2, 3, 4, 5], [2, 3, 3, 4, 4])
        self.assertEqual(s["overall"]["n"], 5)
        self.assertAlmostEqual(s["overall"]["hop_count_exact_match_rate"], 3 / 5)
        self.assertAlmostEqual(s["overall"]["under_decomposition_rate"], 1 / 5)
        self.assertAlmostEqual(s["overall"]["over_decomposition_rate"], 1 / 5)
        self.assertAlmostEqual(s["hop4"]["over_decomposition_rate"], 1 / 2)
        self.assertAlmostEqual(s["hop3"]["mean_signed_error"], -0.5)

    def test_length_mismatch_refused(self) -> None:
        with self.assertRaises(ValueError):
            ws.step_count_summary([1], [1, 2])


class TestDirections(unittest.TestCase):
    def test_matching_removes_a_position_confound(self) -> None:
        # Activations carry a strong "boundary index" signal along e0 and a weak class
        # signal along e1. pos sits mostly at j=1, neg mostly at j=2, so an UNMATCHED
        # difference of means would point along e0; the matched one must point along e1.
        rng = np.random.default_rng(0)
        d = 6
        rows, labels, strata = [], [], []
        for j, n_pos, n_neg in [(1, 50, 5), (2, 5, 50)]:
            for lab, n in (("p", n_pos), ("n", n_neg)):
                for _ in range(n):
                    v = rng.normal(0, 0.01, d)
                    v[0] += 10.0 * j
                    v[1] += 1.0 if lab == "p" else 0.0
                    rows.append(v)
                    labels.append(lab)
                    strata.append(j)
        acts = np.array(rows)
        vec, rec = ws.matched_diff_of_means(acts, labels, strata, "p", "n")
        u = ws.unit(vec)
        self.assertGreater(abs(u[1]), 0.99)
        self.assertEqual(rec["weight_total"], 10.0)
        unmatched = acts[np.array(labels) == "p"].mean(0) - acts[np.array(labels) == "n"].mean(0)
        self.assertGreater(abs(ws.unit(unmatched)[0]), 0.99)

    def test_strata_with_one_class_are_skipped_and_all_missing_refused(self) -> None:
        acts = np.eye(3)
        vec, rec = ws.matched_diff_of_means(acts, ["p", "n", "p"], [1, 1, 2], "p", "n")
        self.assertEqual([c["used"] for c in rec["cells"]], [True, False])
        np.testing.assert_allclose(vec, acts[0] - acts[1])
        with self.assertRaises(ValueError):
            ws.matched_diff_of_means(acts, ["p", "p", "p"], [1, 1, 2], "p", "n")

    def test_min_per_cell(self) -> None:
        acts = np.eye(3)
        with self.assertRaises(ValueError):
            ws.matched_diff_of_means(acts, ["p", "n", "n"], [1, 1, 1], "p", "n", min_per_cell=2)

    def test_auroc_matches_brute_force_with_ties(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.integers(0, 5, 40).astype(float)
        b = rng.integers(0, 5, 30).astype(float)
        brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
        self.assertAlmostEqual(ws.auroc(a, b), float(brute))
        self.assertEqual(ws.auroc([2.0], [1.0]), 1.0)

    def test_random_orthogonal_is_unit_orthogonal_and_seeded(self) -> None:
        r = np.array([3.0, 4.0, 0.0, 0.0])
        v = ws.random_unit_orthogonal(r, 7)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0)
        self.assertAlmostEqual(float(np.dot(v, ws.unit(r))), 0.0)
        np.testing.assert_array_equal(v, ws.random_unit_orthogonal(r, 7))
        self.assertFalse(np.allclose(v, ws.random_unit_orthogonal(r, 8)))


class TestProjection(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.d, self.d_in = 8, 5
        self.W = rng.normal(size=(self.d, self.d_in))
        self.E = rng.normal(size=(11, self.d))
        self.r = ws.unit(rng.normal(size=self.d))
        self.x = rng.normal(size=self.d_in)

    def test_full_ablation_writes_nothing_along_r(self) -> None:
        W2 = ws.project_out_of_outputs(self.W, self.r, 1.0)
        np.testing.assert_allclose(self.r @ W2, np.zeros(self.d_in), atol=1e-12)

    def test_partial_ablation_scales_the_r_component(self) -> None:
        W2 = ws.project_out_of_outputs(self.W, self.r, 0.5)
        np.testing.assert_allclose(self.r @ W2, 0.5 * (self.r @ self.W), atol=1e-12)

    def test_orthogonal_complement_untouched(self) -> None:
        W2 = ws.project_out_of_outputs(self.W, self.r, 1.0)
        q = ws.random_unit_orthogonal(self.r, 3)
        np.testing.assert_allclose(q @ W2, q @ self.W, atol=1e-12)

    def test_weight_edit_equals_output_hook(self) -> None:
        # What the equivalence subcommand checks on GPU: editing W is the same as projecting
        # r out of W's output.
        y = self.W @ self.x
        hooked = y - 1.0 * np.dot(y, self.r) * self.r
        np.testing.assert_allclose(ws.project_out_of_outputs(self.W, self.r, 1.0) @ self.x, hooked, atol=1e-12)

    def test_embedding_rows(self) -> None:
        E2 = ws.project_out_of_rows(self.E, self.r, 1.0)
        np.testing.assert_allclose(E2 @ self.r, np.zeros(11), atol=1e-12)
        E3 = ws.project_out_of_rows(self.E, self.r, 0.25)
        np.testing.assert_allclose(E3 @ self.r, 0.75 * (self.E @ self.r), atol=1e-12)


class TestSplitsAndLeakage(unittest.TestCase):
    CANDS = {2: [f"2hop__{i}" for i in range(30)], 3: [f"3hop1__{i}" for i in range(12)]}
    SIZES = {"locate": {2: 10, 3: 4}, "val_select": {2: 5, 3: 2}, "val_confirm": {2: 5, 3: 2}}

    def test_sizes_disjoint_deterministic(self) -> None:
        a = ws.split_train_ids(self.CANDS, self.SIZES, 42)
        self.assertEqual({k: len(v) for k, v in a.items()}, {"locate": 14, "val_select": 7, "val_confirm": 7})
        allids = [i for v in a.values() for i in v]
        self.assertEqual(len(allids), len(set(allids)))
        self.assertEqual(a, ws.split_train_ids(self.CANDS, self.SIZES, 42))
        self.assertNotEqual(a, ws.split_train_ids(self.CANDS, self.SIZES, 43))

    def test_independent_of_input_order(self) -> None:
        rev = {h: list(reversed(v)) for h, v in self.CANDS.items()}
        self.assertEqual(ws.split_train_ids(self.CANDS, self.SIZES, 42), ws.split_train_ids(rev, self.SIZES, 42))

    def test_too_few_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ws.split_train_ids(self.CANDS, {"locate": {3: 13}}, 42)

    def test_disjointness_refuses_overlap(self) -> None:
        self.assertEqual(ws.assert_disjoint(["a"], ["b"], "x")["overlap"], 0)
        with self.assertRaises(SystemExit):
            ws.assert_disjoint(["a", "b"], ["b"], "pinned")

    def test_question_normalisation(self) -> None:
        self.assertEqual(ws.normalise_question("Who  wrote 'X'?"), ws.normalise_question("who wrote x"))

    def test_ids_sha_is_order_free(self) -> None:
        self.assertEqual(ws.ids_sha256(["b", "a"]), ws.ids_sha256(["a", "b"]))


class TestEditSpec(unittest.TestCase):
    def test_valid_specs(self) -> None:
        s = ws.validate_edit_spec(
            {"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 1, "layers": [3, 1, 3], "matrices": ["down_proj", "o_proj"]}
        )
        self.assertEqual(s["layers"], [1, 3])
        self.assertEqual(s["matrices"], ["down_proj", "o_proj"])
        self.assertEqual(ws.validate_edit_spec({"kind": "none"})["kind"], "none")
        self.assertEqual(ws.validate_edit_spec({"kind": "token_row_scale", "token_ids": "eos", "scale": 0.8})["token_ids"], "eos")
        self.assertEqual(ws.validate_edit_spec({"kind": "final_norm_scale", "scale": 0.6})["scale"], 0.6)

    def test_refusals(self) -> None:
        bad = [
            {"kind": "steer"},
            {"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 1.5, "layers": "all", "matrices": ["o_proj"]},
            {"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 1, "layers": [], "matrices": ["o_proj"]},
            {"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 1, "layers": "all", "matrices": ["q_proj"]},
            {"kind": "orthogonalize", "alpha": 1, "layers": "all", "matrices": ["o_proj"]},
            {"kind": "token_row_scale", "token_ids": "pad", "scale": 0.8},
            {"kind": "final_norm_scale", "scale": 0},
        ]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(SystemExit):
                ws.validate_edit_spec(spec)


class TestHolmAndSelection(unittest.TestCase):
    def test_holm_known_values(self) -> None:
        adj = ws.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])

    @staticmethod
    def _s(em, over, h2, steps=2.5):
        return {
            "overall": {"hop_count_exact_match_rate": em, "over_decomposition_rate": over, "mean_pred_steps": steps},
            "hop2": {"hop_count_exact_match_rate": h2},
        }

    def test_rule_rejects_over_decomposing_edits_and_flags_controls(self) -> None:
        sums = {
            "baseline": self._s(0.60, 0.03, 0.95, 2.4),
            "e_good": self._s(0.66, 0.06, 0.93, 2.7),
            "e_long": self._s(0.70, 0.20, 0.80, 3.3),
            "eos_a": self._s(0.64, 0.10, 0.90, 2.65),
            "eos_b": self._s(0.62, 0.05, 0.94, 2.5),
        }
        res = ws.select_candidate(sums, baseline="baseline", edit_names=["e_good", "e_long"],
                                  control_names=["eos_a", "eos_b"], max_over_increase=0.05, max_hop2_em_drop=0.05)
        self.assertEqual(res["chosen"], "e_good")
        self.assertIn("e_long", res["rejected"])
        self.assertTrue(res["beats_all_controls"])
        self.assertEqual(ws.closest_by_mean_steps(sums, "e_good", ["eos_a", "eos_b"]), "eos_a")

    def test_no_admissible_edit(self) -> None:
        sums = {"baseline": self._s(0.6, 0.0, 0.95), "e": self._s(0.9, 0.5, 0.95)}
        res = ws.select_candidate(sums, baseline="baseline", edit_names=["e"], control_names=[],
                                  max_over_increase=0.05, max_hop2_em_drop=0.05)
        self.assertIsNone(res["chosen"])


class TestCpuStagesOnSyntheticActivations(unittest.TestCase):
    """directions -> (fake sweep) -> choose -> make-specs, end to end on synthetic files.

    The GPU stages (locate, sweep, confirm) cannot run here; their output files are
    fabricated in the shapes those stages write, with a planted direction at one layer, and
    the CPU stages must find it and turn a choice into valid, hashed specs.
    """

    def test_pipeline(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))
        import weight_space_repair as wsr
        from run_config import load_config

        cfg = load_config("exp038_weight_space.json")
        rng = np.random.default_rng(0)
        n_layers, d, planted_layer = 32, 16, 10
        planted = ws.unit(rng.normal(size=d))

        def fake(name: str, n_items: int) -> None:
            acts, meta = [], []
            for i in range(n_items):
                gold_k = 2 + i % 3
                n = gold_k if i % 2 else max(1, gold_k - 1)  # half the items stop one step early
                for rec in ws.boundary_records(n, gold_k, True):
                    a = rng.normal(0, 1.0, size=(n_layers, d))
                    if rec["label"] == ws.PREMATURE_STOP:
                        a[planted_layer] += 3.0 * planted
                    acts.append(a)
                    meta.append({"id": f"{gold_k}hop__{i}", "gold_k": gold_k, **rec})
            np.save(out / f"acts_{name}.npy", np.array(acts, dtype=np.float16))
            (out / f"boundaries_{name}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fake("locate", 300)
            fake("val_select", 120)
            summary = wsr.cmd_directions(cfg, out)
            self.assertIn(planted_layer, summary["top_layers"]["stop_vs_continue"])
            with np.load(out / "directions.npz") as dirs:
                found = dirs["stop_vs_continue"][planted_layer]
            self.assertGreater(abs(ws.cosine(found, planted)), 0.9)
            cands = json.loads((out / "candidates.json").read_text())["candidates"]
            self.assertEqual(cands[0]["name"], "baseline")

            # Fabricate the sweep summaries the GPU stage would write.
            sdir = out / "sweep_val_select"
            sdir.mkdir()
            for c in cands:
                if c["kind"] == "actadd":
                    continue
                em = {"none": 0.60, "orthogonalize": 0.62, "token_row_scale": 0.61, "final_norm_scale": 0.60}[c["kind"]]
                if c["name"] == f"orth_stop_vs_continue_L{planted_layer}_a1.0":
                    em = 0.70
                s = {"n": 300, "hop_count_exact_match_rate": em, "under_decomposition_rate": 0.3,
                     "over_decomposition_rate": 0.04, "mean_pred_steps": 2.5 + (em - 0.6)}
                summ = {"overall": s, "hop2": {**s, "hop_count_exact_match_rate": 0.95}}
                (sdir / f"{c['name']}.summary.json").write_text(json.dumps({"candidate": c, "summary": summ}))
            res = wsr.cmd_choose(cfg, out)
            self.assertEqual(res["chosen"], f"orth_stop_vs_continue_L{planted_layer}_a1.0")
            self.assertTrue(res["beats_all_controls"])
            choice = out / "choice.json"
            choice.write_text(json.dumps({"edit_candidate": res["chosen"],
                                          "eos_control_candidate": res["length_matched_eos_control"],
                                          "chosen_by": "test", "choice_note": "synthetic"}))
            wsr.cmd_make_specs(cfg, out, choice)
            for name in ("baseline", "edit", "random", "eos_control", "norm_control"):
                spec, direction, prov = wsr.load_spec(out / "specs" / f"spec_{name}.json")
                if name in ("edit", "random"):
                    self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)
                    self.assertEqual(prov["direction_sha256"], spec["direction_sha256"])
            e = np.load(out / "specs" / "direction_edit.npy")
            r = np.load(out / "specs" / "direction_random.npy")
            self.assertAlmostEqual(float(np.dot(e, r)), 0.0, places=5)
            bad = out / "bad_choice.json"
            bad.write_text(json.dumps({"edit_candidate": "not_swept", "eos_control_candidate": "eos_row_x0.9",
                                       "chosen_by": "t", "choice_note": "t"}))
            with self.assertRaises(SystemExit):
                wsr.cmd_make_specs(cfg, out, bad)


class TestRunnerWeightEditFlag(unittest.TestCase):
    """run_decomposer.py --weight-edit on a --dry-run: validated before weights, recorded."""

    RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"

    def _run(self, out: Path, spec: Path) -> subprocess.CompletedProcess:
        env = dict(os.environ, QAV2_PATHS_CONFIG="configs/smoke_paths.json")
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
             "--model", "mistral_7b_instruct", "--config", "decomposer_musique.json",
             "--condition", "unguided", "--dry-run", "--dry-run-limit", "2",
             "--retrieval-input", str(self.RETRIEVAL), "--allow-unpinned-eval-set",
             "--output-root", str(out), "--weight-edit", str(spec)],
            env=env, capture_output=True, text=True,
        )

    def test_valid_spec_is_recorded_and_invalid_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "d.npy", np.ones(4, dtype=np.float32))
            good = tmp / "spec.json"
            good.write_text(json.dumps({"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 1.0,
                                        "layers": "all", "matrices": ["o_proj", "down_proj"]}))
            proc = self._run(tmp / "ok", good)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            (run_dir,) = sorted((tmp / "ok").iterdir())
            snap = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(snap["weight_edit_spec"], str(good))
            self.assertIsNone(metrics["weight_edit"])  # dry run: validated, not applied
            bad = tmp / "bad.json"
            bad.write_text(json.dumps({"kind": "orthogonalize", "direction_file": "d.npy", "alpha": 2.0,
                                       "layers": "all", "matrices": ["o_proj"]}))
            proc = self._run(tmp / "bad", bad)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("alpha", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
