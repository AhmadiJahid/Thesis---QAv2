#!/usr/bin/env python3
"""Construction checks for ``scripts/print_model_params.py``. No GPU, no weights, no network.

ADR 0027 point 5: a tool that scores no system takes no log entry, and is covered instead by
tests that pin its **construction**. The counts it prints are machine- and cache-dependent, so
what is asserted here is the arithmetic (a hand-computed safetensors header), the registry
wiring (the folder list comes from the committed configs, not a literal list in the script),
and the refusal to guess (an unresolvable model id is reported ``unmeasured``, never estimated).

Run::

    .venv/bin/python tests/test_model_params.py
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import print_model_params as pmp  # noqa: E402
from model_size import ceiling_for, load_limits  # noqa: E402


class TestHeaderArithmetic(unittest.TestCase):
    """The parameter count is a sum over tensor shapes, hand-checkable."""

    HEADER = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 8], "data_offsets": [0, 1]},
        "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [8, 4], "data_offsets": [1, 2]},
        "model.norm.weight": {"dtype": "BF16", "shape": [8], "data_offsets": [2, 3]},
        "model.scalar": {"dtype": "BF16", "shape": [], "data_offsets": [3, 4]},
    }
    #: 100*8 + 8*4 + 8 + 1 (the empty shape is a scalar), and __metadata__ holds no tensor.
    EXPECTED = 800 + 32 + 8 + 1

    def test_counts_every_tensor_and_skips_metadata(self) -> None:
        self.assertEqual(pmp.count_from_header(self.HEADER), self.EXPECTED)

    def test_dtype_is_irrelevant_to_the_count(self) -> None:
        """Counting shapes, not bytes: a byte-size/itemsize division would be wrong for a
        checkpoint with mixed dtypes, which is why this reads shapes directly."""
        mixed = json.loads(json.dumps(self.HEADER))
        mixed["model.norm.weight"]["dtype"] = "F32"
        self.assertEqual(pmp.count_from_header(mixed), self.EXPECTED)

    def test_reads_the_header_off_a_real_file_without_the_tensor_data(self) -> None:
        """The 8-byte little-endian length prefix is the format; the body after it is never
        read, so a shard with a truncated tensor section still counts."""
        blob = json.dumps(self.HEADER).encode("utf-8")
        tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        shard = path / "model-00001-of-00001.safetensors"
        shard.write_bytes(struct.pack("<Q", len(blob)) + blob)  # no tensor bytes at all
        self.assertEqual(pmp.count_from_header(pmp.safetensors_header(shard)), self.EXPECTED)


class TestTheIndexDecidesWhichFilesAreCounted(unittest.TestCase):
    """A snapshot holding shards AND a whole-model file must not be counted twice.

    Real caches do hold both: some repos ship ``consolidated.safetensors`` beside the
    shards, and a merge or an older single-file revision can leave one behind. Globbing
    ``*.safetensors`` would then report exactly double the true count, silently and with a
    plausible-looking number - which against a ceiling assertion is the worst failure shape
    there is (issue #50 PR review, M-4). ``model.safetensors.index.json`` is the repo's own
    statement of which files make up the model, so it decides when it exists.
    """

    #: 100*8 + 8*4 + 8 = 840 across two shards.
    SHARD_A = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 8], "data_offsets": [0, 1]},
    }
    SHARD_B = {
        "__metadata__": {"format": "pt"},
        "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [8, 4], "data_offsets": [0, 1]},
        "model.norm.weight": {"dtype": "BF16", "shape": [8], "data_offsets": [1, 2]},
    }
    EXPECTED = 800 + 32 + 8

    @staticmethod
    def _write(path: Path, header: dict) -> None:
        blob = json.dumps(header).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    def _snapshot(self, *, with_index: bool, with_extra: bool) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self._write(root / "model-00001-of-00002.safetensors", self.SHARD_A)
        self._write(root / "model-00002-of-00002.safetensors", self.SHARD_B)
        if with_extra:
            # Every tensor of the model again, in one file, exactly as a consolidated
            # export leaves it. Glob-counting this snapshot yields 2 x EXPECTED.
            self._write(
                root / "consolidated.safetensors", {**self.SHARD_A, **self.SHARD_B}
            )
        if with_index:
            weight_map = {
                name: "model-00001-of-00002.safetensors"
                for name in self.SHARD_A
                if name != "__metadata__"
            }
            weight_map.update(
                {
                    name: "model-00002-of-00002.safetensors"
                    for name in self.SHARD_B
                    if name != "__metadata__"
                }
            )
            (root / pmp.SAFETENSORS_INDEX).write_text(
                json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
                encoding="utf-8",
            )
        return root

    def _count(self, snapshot: Path) -> tuple[int | None, str]:
        with unittest.mock.patch.object(pmp, "cached_snapshot", lambda _model_id: snapshot):
            return pmp.count_from_cache("fake/model")

    def test_an_extra_whole_model_file_beside_the_shards_is_not_counted(self) -> None:
        snapshot = self._snapshot(with_index=True, with_extra=True)
        count, detail = self._count(snapshot)
        self.assertEqual(count, self.EXPECTED)
        self.assertNotEqual(count, 2 * self.EXPECTED, "the extra file was double-counted")
        self.assertIn("2 cached shard(s) of 2", detail)

    def test_the_index_also_governs_a_snapshot_with_nothing_extra(self) -> None:
        count, _ = self._count(self._snapshot(with_index=True, with_extra=False))
        self.assertEqual(count, self.EXPECTED)

    def test_without_an_index_it_still_falls_back_to_the_glob(self) -> None:
        """An unsharded repo ships no index; that case must keep working."""
        count, detail = self._count(self._snapshot(with_index=False, with_extra=False))
        self.assertEqual(count, self.EXPECTED)
        self.assertEqual(detail, "2 cached shard(s)")

    def test_a_shard_the_index_names_but_the_cache_lacks_is_reported_partial(self) -> None:
        """A half-finished download must say so rather than pass off a short count."""
        snapshot = self._snapshot(with_index=True, with_extra=False)
        (snapshot / "model-00002-of-00002.safetensors").unlink()
        count, detail = self._count(snapshot)
        self.assertEqual(count, 800)
        self.assertIn("MISSING", detail)
        self.assertIn("count is partial", detail)


class TestRegistryWiring(unittest.TestCase):
    """The folder list comes from the committed registry, not from a literal in the script."""

    def test_the_decomposer_registry_is_read_from_the_model_configs(self) -> None:
        models = dict(pmp.registry_models("decomposer"))
        self.assertEqual(models["qwen2_5_7b_instruct"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(models["qwen2_5_14b_instruct"], "Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(models["mistral_7b_instruct"], "mistralai/Mistral-7B-Instruct-v0.3")

    def test_every_registered_model_id_matches_its_committed_config(self) -> None:
        root = REPO_ROOT / "components" / "decomposer" / "models"
        for folder, model_id in pmp.registry_models("decomposer"):
            committed = json.loads((root / folder / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(model_id, committed["model_id"], folder)

    def test_the_router_registry_is_reachable_too(self) -> None:
        self.assertIn("qwen2_5_7b", dict(pmp.registry_models("router")))


class TestItRefusesToGuess(unittest.TestCase):
    def test_an_unresolvable_model_is_unmeasured_not_estimated(self) -> None:
        record = pmp.measure("not-a-real-org/not-a-real-model", offline=True)
        self.assertIsNone(record["count"])
        self.assertEqual(record["source"], "unmeasured")
        self.assertIn("not in the local HF cache", record["detail"])


class TestItComparesAgainstTheCommittedCeiling(unittest.TestCase):
    """The script must not carry its own copy of a ceiling (issue #50 PR review, I-1/I-4)."""

    def test_it_uses_the_component_keyed_ceiling(self) -> None:
        limits = load_limits()
        self.assertEqual(ceiling_for("decomposer", limits), 15_000_000_000)
        self.assertEqual(ceiling_for("router", limits), 8_000_000_000)

    def test_no_parameter_ceiling_is_hardcoded_in_the_script(self) -> None:
        source = (REPO_ROOT / "scripts" / "print_model_params.py").read_text(encoding="utf-8")
        for literal in ("8_000_000_000", "8000000000", "15_000_000_000", "15000000000"):
            self.assertNotIn(literal, source, f"{literal} is hardcoded in the script")


if __name__ == "__main__":
    unittest.main(verbosity=2)
