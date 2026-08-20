#!/usr/bin/env python3
"""Source-level guards for the clustered pool path (ADR 0016, issue #14).

Why this file exists. The clustered strategy has exactly two invariants that **no run
without model weights can reach**:

1. the bi-encoder's parameter count is asserted before it is used — the assertion lives
   after ``SentenceTransformer(...)`` returns, and the limits config is loaded before it,
   so neither line executes unless weights actually load;
2. the run-note formatter in ``main()`` reads keys off the clustering diagnostics dict,
   and that dict is only built on a real clustered run.

Both are the shape ADR [0016](../docs/adr/0016-real-run-only-invariants-get-source-level-guards.md)
exists for, and the exact shape that took down exp-002/exp-003 on 2026-08-19: a consumer
read ``gen["raw"]`` while the producer returned ``"text"``, and every dry run, smoke stage
and unit test passed over the broken line because a dry run never generates. The clustered
path had the same blind spot in a worse form — its only end-to-end coverage
(``TestClusteredCli`` in ``tests/test_pool_clustering.py``) **self-skips** when the encoder
is not in the local model cache, so on a fresh machine the suite goes green with the whole
strategy unexercised (PR #34 review, I-3).

So these checks parse the source instead. They load no weights, need no cache, and each one
ships a negative control that breaks the invariant in a copy of the source and asserts the
guard catches it — a guard nobody has watched fail is not evidence.

Run::

    .venv/bin/python tests/test_pool_clustering_guards.py
    .venv/bin/python -m unittest tests.test_pool_clustering_guards -v
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_POOL = REPO_ROOT / "MusiQue" / "scripts" / "sample_pool.py"
SIMILARITY = REPO_ROOT / "MusiQue" / "scripts" / "check_question_similarity.py"
POOL_EMBEDDINGS = REPO_ROOT / "src" / "pool_embeddings.py"

#: The encoder construction the ceiling assertion has to bracket.
ENCODER_CALL = "SentenceTransformer"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() was not found — the guard went blind")


def _call_linenos(node: ast.AST, func_name: str) -> list[int]:
    """Line numbers of every call to ``func_name`` (bare name or attribute) under ``node``."""
    out: list[int] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        target = child.func
        called = (
            target.id
            if isinstance(target, ast.Name)
            else target.attr if isinstance(target, ast.Attribute) else None
        )
        if called == func_name:
            out.append(child.lineno)
    return out


def ceiling_guard_violations(source: str, *, fn: str = "_embed_texts") -> list[str]:
    """Ways ``fn`` could load an encoder without the ceiling being asserted around it.

    The rule, in order: ``load_limits`` runs **before** the encoder is constructed (a
    limits config missing a key must fail before weights load — ``src/model_size.py``
    says so in as many words), the encoder is constructed, and
    ``assert_within_ceiling`` runs **after** it with a real component name. Any of those
    missing means a model could be embedded with its size unasserted.
    """
    node = _function(ast.parse(source), fn)
    limits = _call_linenos(node, "load_limits")
    encoder = _call_linenos(node, ENCODER_CALL)
    asserts = _call_linenos(node, "assert_within_ceiling")

    violations: list[str] = []
    if not encoder:
        return [f"{fn}() constructs no {ENCODER_CALL} — the guard went blind"]
    if not limits:
        violations.append(f"{fn}() never calls load_limits")
    if not asserts:
        violations.append(f"{fn}() never calls assert_within_ceiling")
    if limits and min(limits) > min(encoder):
        violations.append(
            f"load_limits (line {min(limits)}) does not precede "
            f"{ENCODER_CALL} (line {min(encoder)})"
        )
    if asserts and min(asserts) < min(encoder):
        violations.append(
            f"assert_within_ceiling (line {min(asserts)}) precedes the "
            f"{ENCODER_CALL} it must assert (line {min(encoder)})"
        )

    components = {
        kw.value.value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "assert_within_ceiling"
        for kw in call.keywords
        if kw.arg == "component"
        and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, str)
    }
    if asserts and not components:
        violations.append("assert_within_ceiling is called without a literal component=")
    return violations


def _dict_literal_keys(node: ast.AST, var: str) -> list[set[str]]:
    """Key sets of every dict literal assigned to ``var`` under ``node``."""
    out: list[set[str]] = []
    for stmt in ast.walk(node):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if not any(isinstance(t, ast.Name) and t.id == var for t in targets):
            continue
        if isinstance(stmt.value, ast.Dict):
            out.append(
                {
                    key.value
                    for key in stmt.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            )
    return out


def _subscript_assigned_keys(node: ast.AST, var: str) -> set[str]:
    """Keys written as ``var["key"] = ...`` under ``node``."""
    keys: set[str] = set()
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == var
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
    return keys


def _read_keys(tree: ast.Module, var: str) -> tuple[set[str], set[str]]:
    """Keys read as ``var["k"]`` and as ``var["embedding"]["k"]`` anywhere in the module."""
    top: set[str] = set()
    nested: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
            continue
        key = node.slice.value
        if not isinstance(key, str):
            continue
        inner = node.value
        if isinstance(inner, ast.Name) and inner.id == var:
            top.add(key)
        elif (
            isinstance(inner, ast.Subscript)
            and isinstance(inner.value, ast.Name)
            and inner.value.id == var
            and isinstance(inner.slice, ast.Constant)
            and inner.slice.value == "embedding"
        ):
            nested.add(key)
    return top, nested


def diagnostics_guard_missing(source: str) -> tuple[set[str], set[str]]:
    """(diagnostic keys, embedding keys) the run note reads but no producer guarantees.

    A key counts as produced only when **every** producer of that dict emits it, the same
    intersection rule ``tests/test_generation_contract.py`` uses: the embedding record has
    two shapes (a real embedding pass and the precomputed test seam), and a key present in
    only one of them is exactly the crash that waits for whichever path is not covered.
    """
    tree = ast.parse(source)
    select = _function(tree, "_kmeans_select")
    sample = _function(tree, "_sample_clustered")
    embed = _function(tree, "_embed_texts")

    literals = _dict_literal_keys(select, "diagnostics")
    if not literals:
        raise AssertionError("_kmeans_select builds no diagnostics dict literal")
    produced = set.intersection(*literals) | _subscript_assigned_keys(sample, "diagnostics")

    record_literals = _dict_literal_keys(embed, "record") + _dict_literal_keys(
        sample, "embed_record"
    )
    if len(record_literals) < 2:
        raise AssertionError(
            "expected two embedding-record producers (real pass + precomputed seam), "
            f"found {len(record_literals)}"
        )
    produced_embedding = set.intersection(*record_literals)

    read_top, read_nested = _read_keys(tree, "clustering_diagnostics")
    if not read_top:
        raise AssertionError("nothing reads clustering_diagnostics — the guard went blind")
    return (read_top - produced), (read_nested - produced_embedding)


class TestCeilingPrecedesEncoder(unittest.TestCase):
    """The bi-encoder's size is asserted on every real clustered run, or the run dies."""

    def setUp(self) -> None:
        self.source = SAMPLE_POOL.read_text(encoding="utf-8")

    def test_the_invariant_holds(self) -> None:
        self.assertEqual(ceiling_guard_violations(self.source), [])

    def test_limits_loaded_after_the_encoder_is_caught(self) -> None:
        """Negative control: a limits config missing a key must fail before weights load."""
        broken = self.source.replace(
            '    limits = load_limits("model_limits.json")\n'
            "    model = SentenceTransformer(model_id, device=device)\n",
            "    model = SentenceTransformer(model_id, device=device)\n"
            '    limits = load_limits("model_limits.json")\n',
        )
        self.assertNotEqual(broken, self.source, "the load order moved; update this control")
        self.assertTrue(
            any("does not precede" in v for v in ceiling_guard_violations(broken)),
            ceiling_guard_violations(broken),
        )

    def test_a_dropped_assertion_is_caught(self) -> None:
        """Negative control: embedding with the parameter count unasserted."""
        broken = self.source.replace(
            "    size_record = assert_within_ceiling(\n"
            '        model, component="retrieval", model_id=model_id, limits=limits\n'
            "    )\n",
            '    size_record = {"ceiling_asserted": False}\n',
        )
        self.assertNotEqual(broken, self.source, "the assertion moved; update this control")
        self.assertIn(
            "_embed_texts() never calls assert_within_ceiling",
            ceiling_guard_violations(broken),
        )

    def test_an_anonymous_assertion_is_caught(self) -> None:
        """Negative control: asserting against no named component asserts nothing useful."""
        broken = self.source.replace('component="retrieval", model_id=model_id', "model_id=model_id")
        self.assertNotEqual(broken, self.source, "the assertion moved; update this control")
        self.assertIn(
            "assert_within_ceiling is called without a literal component=",
            ceiling_guard_violations(broken),
        )


class TestDiagnosticsContract(unittest.TestCase):
    """The run note may only read diagnostic keys some producer guarantees."""

    def setUp(self) -> None:
        self.source = SAMPLE_POOL.read_text(encoding="utf-8")

    def test_every_key_the_note_reads_is_produced(self) -> None:
        missing_top, missing_embedding = diagnostics_guard_missing(self.source)
        self.assertEqual(missing_top, set())
        self.assertEqual(missing_embedding, set())

    def test_a_renamed_diagnostic_key_is_caught(self) -> None:
        """Negative control, the exp-002 shape: producer renamed, consumer not."""
        broken = self.source.replace(
            '"selected_from_topup": size - from_clusters,',
            '"topup_count": size - from_clusters,',
        )
        self.assertNotEqual(broken, self.source, "the diagnostics dict moved; update this control")
        self.assertEqual(diagnostics_guard_missing(broken)[0], {"selected_from_topup"})

    def test_a_renamed_late_assignment_is_caught(self) -> None:
        """Keys attached after _kmeans_select returns are covered too."""
        broken = self.source.replace(
            'diagnostics["text_field"] = field', 'diagnostics["field"] = field'
        )
        self.assertNotEqual(broken, self.source, "the assignment moved; update this control")
        self.assertEqual(diagnostics_guard_missing(broken)[0], {"text_field"})

    def test_a_key_missing_from_only_one_embedding_producer_is_caught(self) -> None:
        """The precomputed seam is a second producer; a key it lacks is still a crash."""
        broken = self.source.replace(
            '        embed_record = {"model_id": model_id, "precomputed": True, '
            '"num_texts": len(texts)}',
            '        embed_record = {"precomputed": True, "num_texts": len(texts)}',
        )
        self.assertNotEqual(broken, self.source, "the seam moved; update this control")
        self.assertEqual(diagnostics_guard_missing(broken)[1], {"model_id"})


class TestOneE5PrefixDefinition(unittest.TestCase):
    """"Same prefix as the retrieval stage" is one function, not three copies (N-2)."""

    def _defines(self, path: Path, name: str) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return any(
            isinstance(node, ast.FunctionDef) and node.name == name
            for node in ast.walk(tree)
        )

    def test_only_pool_embeddings_defines_it(self) -> None:
        self.assertTrue(self._defines(POOL_EMBEDDINGS, "needs_e5_prefix"))
        for path in (SAMPLE_POOL, SIMILARITY):
            with self.subTest(path=path.name):
                self.assertFalse(self._defines(path, "needs_e5_prefix"))
                self.assertFalse(self._defines(path, "_needs_e5_prefix"))
                source = path.read_text(encoding="utf-8")
                self.assertIn("from pool_embeddings import needs_e5_prefix", source)
                self.assertIn("needs_e5_prefix(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
