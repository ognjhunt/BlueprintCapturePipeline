"""The controls and policy execution modes must be able to launch at all.

Both lanes were unlaunchable. The shared Vast runtime contract required the
*construction* result filename in the entrypoint, and the adapter's required
entry set demanded two construction-only planner modules -- so a controls or
policy bundle failed the static preflight before offer search, after its
one-shot authority had already been consumed. Behind those, neither declared
module list was an import closure, so the worker died on ModuleNotFoundError
after Isaac had launched: the expensive end of a paid run.

These are contract tests over the declarations themselves, which is what the
existing per-bundle tests cannot be -- they assert a declared list is a subset
of the zip built from that same list, which is true by construction.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from blueprint_pipeline.native_task_arena_bundle import _entrypoint
from blueprint_pipeline.native_task_arena_controls_bundle import (
    CONTROLS_RUNTIME_MODULE_NAMES,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

POLICY_EXTRA_MODULE_NAMES = (
    "adp009d_policy_episode.py",
    "adp009d_droid_action_execution.py",
    "droid_policy_bridge.py",
    "openpi_droid_policy_runtime.py",
    "policy_ranking_thesis.py",
)


def _import_time_relative_imports(module: str) -> set[str]:
    """Every relative import evaluated at import time, however nested.

    Descends through `try`/`if`/`with` -- which run at import time -- but not
    into functions or classes, which do not. The distinction matters: these
    modules reach their siblings through a `try: absolute / except
    ModuleNotFoundError: relative` pair, so the import that actually resolves
    inside a bundle is never a direct child of the module body. A checker that
    only looked at top-level statements would pass while the bundle still died
    on startup.
    """

    found: set[str] = set()

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                found.add(node.module)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, (ast.If, ast.With)):
                walk(node.body)
                walk(getattr(node, "orelse", []))

    walk(ast.parse((_PACKAGE / f"{module}.py").read_text(encoding="utf-8")).body)
    return found


@pytest.mark.parametrize(
    "result_filename",
    [
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_result.v1.json",
    ],
)
def test_every_execution_mode_entrypoint_satisfies_the_runtime_contract(
    result_filename: str,
) -> None:
    entrypoint = _entrypoint(
        expected_output_filename=result_filename,
        runtime_source_packet_required=True,
    )
    # The runner is whichever worker the bundle copies into place; this test is
    # about the entrypoint half of the contract, so any real worker source will
    # do for the runner argument.
    runner = (_PACKAGE / "native_task_arena_construction_worker.py").read_text(
        encoding="utf-8"
    )

    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="native_task_arena",
        entrypoint_text=entrypoint,
        runner_text=runner,
    )

    assert "provider_entrypoint_missing_runtime_result_crash_fallback" not in blockers


def test_controls_declared_modules_are_an_import_closure() -> None:
    declared = {name[:-3] for name in CONTROLS_RUNTIME_MODULE_NAMES}
    unresolved = {
        f"{module} -> .{imported}"
        for module in sorted(declared)
        for imported in _import_time_relative_imports(module)
        if imported not in declared
    }

    assert unresolved == set()


def test_policy_declared_modules_are_an_import_closure() -> None:
    declared = {name[:-3] for name in CONTROLS_RUNTIME_MODULE_NAMES}
    declared |= {name[:-3] for name in POLICY_EXTRA_MODULE_NAMES}
    unresolved = {
        f"{module} -> .{imported}"
        for module in sorted(declared)
        for imported in _import_time_relative_imports(module)
        if imported not in declared
    }

    # One break is left, and it is pinned rather than hidden so that it cannot
    # grow silently. `openpi_droid_policy_runtime` imports `canonical_sha256`
    # from `policy_ranking_thesis`, which imports `write_json` from `common`,
    # which re-exports `core.common` -- neither `common` nor the `core`
    # subpackage is shipped in any arena bundle. Closing it means either
    # shipping that subpackage or giving the runtime a local digest helper, so
    # it is a separate change from making the two lanes launchable at all. It
    # only affects the `pi05_droid` candidate; `groot_n17_droid` imports clean.
    assert unresolved == {"policy_ranking_thesis -> .common"}
