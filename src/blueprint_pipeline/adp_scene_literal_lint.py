"""Prevent historical scene literals from leaking into reusable ADP modules.

The scan skips docstrings and only docstrings. A module that *names* the first
scene while explaining why anchoring on it was a mistake is documenting the
defect; a module that carries the same digits in code, or in any other string,
is committing it. A raw line scan cannot tell those apart, and answering that by
rewording the prose would trade an accurate explanation for a passing lint --
the explanation is the part that stops the next author from repeating it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


FORBIDDEN_HISTORICAL_LITERALS = re.compile(r"840313|ins160|canned_beverage")
# These modules are explicitly the retained first-rehearsal implementation or
# compatibility adapters. Adding a new path here requires conscious review;
# all other production modules must receive scene/task identity as data.
HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS = frozenset(
    {
        "adp009d_isaac_runtime.py",
        "adp009d_native_microcheck_bundle.py",
        "adp009d_sage_franka_placement.py",
        "adp009d_840313_runtime_bundle.py",
        "adp009d_live_readiness.py",
        "adp_content_agents_bundle_preflight.py",
        "adp_content_agents_vast.py",
        "adp_inpaint360_interiorgs_vast.py",
        "public_scene_hybrid_replacement_seal.py",
        "public_scene_inpaint360_adapter.py",
        "public_scene_simready_control.py",
        "public_scene_simready_isaac_bundle.py",
        "public_scene_simready_native.py",
        "public_scene_simready_replacement.py",
        "public_scene_suite_materializer.py",
        "vast_provider_adapter.py",
    }
)


def _docstring_line_numbers(source: str) -> frozenset[int]:
    """Every line occupied by a module, class, or function docstring.

    Parsed rather than pattern-matched: a docstring is a position in the syntax
    tree, and anything that merely looks like one from a line's text is a plain
    string literal, which this lint must still refuse.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # An unparseable module gets the strict line scan; a lint must not
        # relax because it could not read the file.
        return frozenset()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return frozenset(lines)


def scan_scene_literal_violations(source_root: str | Path) -> list[dict[str, Any]]:
    """Return forbidden literal locations outside explicit fixture modules."""

    root = Path(source_root).expanduser().resolve()
    violations = []
    for path in sorted(root.rglob("*.py")):
        if path.name in HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS or path.name == Path(
            __file__
        ).name:
            continue
        source = path.read_text(encoding="utf-8")
        documented = _docstring_line_numbers(source)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line_number in documented:
                continue
            matches = sorted(set(FORBIDDEN_HISTORICAL_LITERALS.findall(line)))
            if matches:
                violations.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "line_number": line_number,
                        "literals": matches,
                    }
                )
    return violations


__all__ = [
    "HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS",
    "scan_scene_literal_violations",
]
