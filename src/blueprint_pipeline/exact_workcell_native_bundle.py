"""Import-closed full-matrix sources for the existing native bundle builder.

This is a source staging port, not a new launcher or execution/spend authority.
Quick-10's shipped module set is unchanged. Callers pass the returned sources
and extras to build_native_task_arena_bundle under their existing admission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .provider_runtime_import_closure import _import_targets, assert_provider_runtime_import_closure


ROOT_MODULES = (
    "exact_workcell_evaluation_session.py",
    "exact_workcell_native_session.py",
    "exact_workcell_native_admission.py",
    "evaluation_run_execution.py",
    # This full-authority validation executes on the new provider path; the
    # canary's historical control-plane-only lazy exemption does not apply.
    "adp009d_scene_policy_readiness.py",
)


def full_matrix_native_runtime_sources(package_source_dir: Path | None = None) -> dict[str, Any]:
    """Resolve and verify source closure, including lazy per-episode imports.

    All returned files are inside the selected immutable Blueprint checkout;
    no provider or network is contacted. The existing bundle builder hashes
    and packages these exact bytes with the source commit and native packet.
    """
    root = (package_source_dir or Path(__file__).parent).resolve()
    modules: set[str] = set()
    packages: set[str] = set()
    pending = list(ROOT_MODULES)
    while pending:
        name = pending.pop()
        if name in modules or name in packages:
            continue
        source = root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError("full_matrix_runtime_source_missing:" + name)
        (packages if "/" in name else modules).add(name)
        for target, _lazy in _import_targets(source.read_text()):
            file = root / (target + ".py")
            directory = root / target
            if file.is_file():
                pending.append(file.name)
            elif directory.is_dir():
                pending.extend(
                    path.relative_to(root).as_posix() for path in directory.rglob("*.py")
                )
            else:
                raise ValueError("full_matrix_runtime_import_source_missing:" + target)
    assert_provider_runtime_import_closure(
        package_source_dir=root,
        shipped_module_names=sorted(modules),
        shipped_package_files=sorted(packages),
        code="full_matrix_native_bundle_import_closure_incomplete",
    )
    return {
        "runtime_module_sources": [root / name for name in sorted(modules)],
        "runtime_extra_files": {
            "blueprint_pipeline/" + name: root / name for name in sorted(packages)
        },
    }
