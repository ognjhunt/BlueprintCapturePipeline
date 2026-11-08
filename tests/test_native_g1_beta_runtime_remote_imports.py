"""The G1 beta runtime runs on a rented machine whose Python has no jsonschema.

The library job's first paid G1 navigation run (2026-11-07, attempt
screen-baf4d7ec…) ended before its first observation with
``ModuleNotFoundError: No module named 'jsonschema'``:
``native_g1_beta_runtime._inputs`` imports ``beta_policy_screening_g1``, which
imported ``beta_policy_screening_g1_catalog``, which imported the candidate
catalog module at load. The runtime and every package module it imports, at
load or inside its functions, must load without jsonschema.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline/native_g1_beta_runtime.py"

_PROBE = """
import importlib, importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "jsonschema":
            raise ModuleNotFoundError("No module named %r (not on the rented machine)" % name)
        return None

sys.meta_path.insert(0, Block())
for module in sys.argv[1:]:
    importlib.import_module(module)
print("ok")
"""


def _runtime_imports() -> list[str]:
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    return sorted({node.module for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module})


def test_the_g1_runtime_and_every_module_it_imports_load_without_jsonschema() -> None:
    modules = ["blueprint_pipeline.native_g1_beta_runtime",
               *("blueprint_pipeline." + name for name in _runtime_imports())]
    assert "blueprint_pipeline.beta_policy_screening_g1" in modules
    result = subprocess.run([sys.executable, "-c", _PROBE, *modules], capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-3000:]


def _module_facts(name: str, cache: dict) -> tuple[dict, bool, set] | None:
    if name not in cache:
        path = RUNTIME.parent / (name + ".py")
        if not path.exists():
            cache[name] = None
        else:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            functions = {node.name: node for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            uses_jsonschema = any(
                (isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "jsonschema")
                or (isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "jsonschema" for alias in node.names))
                for node in tree.body)
            relative = {(node.module or "").split(".")[0] for node in tree.body
                        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}
            cache[name] = (functions, uses_jsonschema, relative)
    return cache[name]


def _loads_jsonschema(name: str, cache: dict, seen: set) -> list[str]:
    """Package modules that load jsonschema when ``name`` is imported (its load-time imports, transitively)."""
    if name in seen:
        return []
    seen.add(name)
    facts = _module_facts(name, cache)
    if facts is None:
        return []
    found = [name] if facts[1] else []
    for module in facts[2]:
        found += _loads_jsonschema(module, cache, seen)
    return found


def test_no_function_the_runtime_reaches_imports_a_module_that_loads_jsonschema() -> None:
    """Imports inside functions count too: the 2026-11-08 retry crashed in ``validate_execution_spec`` →
    ``_record``, which imported the screening plan module. Walk from ``main`` through same-module calls and
    every ``from .x import f`` it executes, and check what each such import loads."""
    cache: dict = {}
    visited: set = set()
    offenders: list[str] = []

    def walk(module: str, function: str) -> None:
        if (module, function) in visited:
            return
        visited.add((module, function))
        facts = _module_facts(module, cache)
        if facts is None or function not in facts[0]:
            return
        for node in ast.walk(facts[0][function]):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = node.module.split(".")[0]
                loads = _loads_jsonschema(target, cache, set())
                if loads:
                    offenders.append(f"{module}.{function} imports {target}, which loads jsonschema via {loads}")
                for alias in node.names:
                    walk(target, alias.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in facts[0]:
                walk(module, node.func.id)

    walk("native_g1_beta_runtime", "main")
    assert ("beta_policy_screening_g1", "validate_execution_spec") in visited
    assert offenders == []


def test_the_light_reference_check_uses_the_catalog_digest_rule() -> None:
    from blueprint_pipeline.beta_policy_candidate_catalog import DIGEST_PATTERN
    from blueprint_pipeline.beta_policy_screening_artifact_reference import DIGEST

    assert DIGEST.pattern == DIGEST_PATTERN
