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
