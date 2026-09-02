"""Static import closure of the ``blueprint_pipeline`` package shipped to a provider.

A provider bundle ships a subset of this package.  Every ``from .x import``
and ``import blueprint_pipeline.x`` inside that subset must resolve to a
module that is also shipped, or the worker fails on the rented GPU after the
image pull, the checkpoint downloads, and both policy servers have already
been paid for.  Import-time closure is only half of it: the policy runtimes
import lazily inside functions, so a missing module surfaces at the first
real policy query, not at startup.

This module checks the closure statically, over module-level and
function-level imports alike, so the bundle builder can refuse to seal an
unshippable bundle and the fast lane can pin the shipped set.  A small,
explicit exemption table names function-level imports that only the control
plane ever executes; an exempted import that becomes module-level is a
blocker again, because it would then fail at import time on the provider.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PACKAGE_NAME = "blueprint_pipeline"

# Function-level imports inside shipped modules that resolve only on the
# control plane.  Each entry is (importing module, imported top-level module).
# Keep this table short and justified: every row is a module that a provider
# worker never executes, documented by the function that contains the import.
CONTROL_PLANE_ONLY_LAZY_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        # ``_validation_errors`` on the measurement routing contract classes;
        # provider workers only compute digests from this module.
        ("decision_evidence_contracts.py", "task_site_measurement_routing"),
        # ``validate_candidate_policy_rights_authorities`` runs at profile
        # materialization on the control plane, never on a provider.
        ("adp009d_policy_rights.py", "adp009d_scene_policy_readiness"),
        # Physics backend comparison admission runs on the control plane.
        ("adp009d_physics_backend_comparison.py", "spend_admission_lock"),
        # Control-search funnel packet reads run on the control plane.
        ("task_evaluation_control_search_funnel.py", "native_task_arena_packet"),
        # Remote cuRobo candidate generation allocates from the control plane.
        ("task_evaluation_curobo_candidate_generator.py", "gpu_render_providers"),
        (
            "task_evaluation_curobo_candidate_generator.py",
            "native_task_arena_warm_vast",
        ),
    }
)


def _import_targets(source: str) -> list[tuple[str, bool]]:
    """Return ``(top_level_module, is_function_level)`` for every package import."""

    tree = ast.parse(source)
    targets: list[tuple[str, bool]] = []
    function_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for child in ast.walk(node):
                function_nodes.add(id(child))
    for node in ast.walk(tree):
        lazy = id(node) in function_nodes
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module:
                    targets.append((node.module.split(".")[0], lazy))
                else:
                    targets.extend((alias.name, lazy) for alias in node.names)
            elif node.level == 0 and node.module:
                if node.module == PACKAGE_NAME:
                    targets.extend((alias.name, lazy) for alias in node.names)
                elif node.module.startswith(PACKAGE_NAME + "."):
                    targets.append((node.module.split(".")[1], lazy))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE_NAME + "."):
                    targets.append((alias.name.split(".")[1], lazy))
    return targets


def provider_runtime_import_closure_blockers(
    *,
    package_source_dir: str | Path,
    shipped_module_names: Iterable[str],
    exemptions: Mapping[tuple[str, str], Any] | frozenset[tuple[str, str]] = (
        CONTROL_PLANE_ONLY_LAZY_IMPORTS
    ),
) -> list[str]:
    """Return sorted blockers for every shipped import that the bundle cannot satisfy.

    ``shipped_module_names`` are file names (``x.py``) copied into the
    provider's ``blueprint_pipeline`` package directory.  Sub-packages are
    never shipped, so any import of a package directory is a blocker.
    """

    source_dir = Path(package_source_dir).expanduser().resolve()
    shipped = {str(name) for name in shipped_module_names}
    blockers: set[str] = set()
    for name in sorted(shipped):
        path = source_dir / name
        if not path.is_file() or path.is_symlink():
            blockers.add(f"provider_runtime_module_missing:{name}")
            continue
        try:
            targets = _import_targets(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            blockers.add(f"provider_runtime_module_unparseable:{name}:{type(exc).__name__}")
            continue
        for target, lazy in targets:
            target_file = f"{target}.py"
            if target_file in shipped:
                continue
            exempt = (name, target) in exemptions
            if exempt and lazy:
                continue
            if exempt and not lazy:
                blockers.add(
                    f"provider_runtime_import_exemption_not_lazy:{name}->{target}"
                )
                continue
            if (source_dir / target).is_dir():
                blockers.add(
                    f"provider_runtime_import_of_unshipped_subpackage:{name}->{target}"
                )
            else:
                blockers.add(
                    f"provider_runtime_import_unshipped:{name}->{target}"
                    + (":lazy" if lazy else "")
                )
    return sorted(blockers)


def assert_provider_runtime_import_closure(
    *, package_source_dir: str | Path, shipped_module_names: Sequence[str], code: str
) -> None:
    """Raise ``ValueError(code:...)`` when the shipped package is not import-closed."""

    blockers = provider_runtime_import_closure_blockers(
        package_source_dir=package_source_dir,
        shipped_module_names=shipped_module_names,
    )
    if blockers:
        raise ValueError(f"{code}:" + ",".join(blockers))


__all__ = [
    "CONTROL_PLANE_ONLY_LAZY_IMPORTS",
    "assert_provider_runtime_import_closure",
    "provider_runtime_import_closure_blockers",
]
