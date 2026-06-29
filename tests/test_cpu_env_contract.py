"""CPU environment contract (CRIT-02 / P0-20 / P0-23 / P1-42).

This meta-test pins the contract that the *canonical no-GPU interpreter* must
carry the full CPU validation stack. It HARD-FAILS (does not skip) if any of the
required modules is missing, so a future environment rebuild cannot silently drop
``pxr`` (usd-core) or ``mujoco`` and quietly re-skip the dry-render / placement /
POV-framing gates that those modules guard.

Ground truth (see docs/cpu-work-audit-2026-06-29.md, CRIT-02): the canonical
``.venv`` collects the whole suite with zero errors and carries PIL. The durable
gap was that ``pxr`` and ``mujoco`` were absent, so the no-GPU validation gates
skipped green instead of running. This test encodes the fix: the modules must be
*importable*, not merely *declared*.

Setup that satisfies this contract:

    uv sync --extra dev          # or: pip install -e '.[geometry,cloud]'

These are pure-CPU wheels. This test launches no GPU, no cloud, and no spend; it
only imports modules already present in the interpreter. The imports here are a
provenance/repro guard, not a policy-success or render-fidelity claim.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

# (module to import, friendly name, distribution/extra that provides it, why it matters)
REQUIRED_MODULES = [
    (
        "pxr",
        "usd-core (OpenUSD `pxr` bindings)",
        "geometry / dev extra (usd-core>=24.0)",
        "drives the no-GPU dry-render, scene-placement, and POV-framing gates",
    ),
    (
        "mujoco",
        "mujoco",
        "geometry / dev extra (mujoco>=3.1)",
        "preferred CPU physics-parity validation substrate",
    ),
    (
        "trimesh",
        "trimesh",
        "geometry / dev / runtime extra (trimesh>=4.4.0)",
        "mesh I/O for geometry and scenario-packet tests",
    ),
    (
        "collada",
        "pycollada",
        "geometry / dev / runtime extra (pycollada>=0.8)",
        "Collada/DAE mesh export used by MuJoCo scenario-packet materialization tests",
    ),
    (
        "PIL",
        "Pillow (PIL)",
        "core dependency (Pillow>=10.0.0)",
        "dry-render preview PNG rendering and seed-frame tests",
    ),
    (
        "numpy",
        "numpy",
        "core dependency (numpy>=1.24.0)",
        "numeric backbone for placement math and render checks",
    ),
    (
        "boto3",
        "boto3",
        "cloud / dev extra (boto3>=1.34.0)",
        "object-store staging subprocess that runs before any GPU pod launches",
    ),
    (
        "botocore",
        "botocore",
        "cloud / dev extra (botocore>=1.34.0)",
        "provider object-store signing, retry, and client exception support",
    ),
]


def _import_error(module_name: str) -> str | None:
    """Return None if the module imports, else a one-line failure reason."""
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - we want to report any import failure loudly
        return f"{type(exc).__name__}: {exc}"
    return None


@pytest.mark.parametrize(
    "module_name, friendly, provided_by, why",
    REQUIRED_MODULES,
    ids=[entry[0] for entry in REQUIRED_MODULES],
)
def test_required_cpu_module_imports(module_name, friendly, provided_by, why):
    """Each no-GPU stack module must import; absence is a hard failure, never a skip."""
    error = _import_error(module_name)
    assert error is None, (
        f"Canonical CPU env contract violated: `{friendly}` ({module_name!r}) is not "
        f"importable in this interpreter.\n"
        f"  Reason: {error}\n"
        f"  Provided by: {provided_by}\n"
        f"  Needed for: {why}.\n"
        f"  Fix: run `uv sync --extra dev` (or `pip install -e '.[geometry,cloud]'`) "
        f"into the canonical .venv (Python 3.12). See docs/DEV_SETUP.md.\n"
        f"  This contract exists so a rebuilt env cannot silently drop pxr/mujoco and "
        f"re-skip the dry-render / placement / POV gates."
    )


def test_full_cpu_stack_present_together():
    """All required modules must coexist in ONE interpreter (the split-brain guard).

    The historical failure mode was a hand-assembled venv with PIL but no pxr, or a
    system python with pxr but no PIL / no project package. This asserts the whole
    set imports in the single canonical interpreter that runs the suite.
    """
    missing = [
        f"{friendly} ({module_name})"
        for module_name, friendly, _provided_by, _why in REQUIRED_MODULES
        if _import_error(module_name) is not None
    ]
    assert not missing, (
        "Canonical CPU env is incomplete -- the following no-GPU stack modules are "
        "absent from this single interpreter: " + ", ".join(missing) + ".\n"
        "Run `uv sync --extra dev` (or `pip install -e '.[geometry,cloud]'`) so the "
        "full no-GPU validation stack lives in one interpreter. See docs/DEV_SETUP.md."
    )


def test_canonical_cpu_env_collects_without_errors():
    """The canonical interpreter must collect the test suite with zero errors.

    This is intentionally a hard-failing subprocess meta-test, not a skip. The
    child pytest run uses --collect-only, so it imports test modules and catches
    missing module-level dependencies without executing the full suite.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "Canonical CPU env failed pytest collection.\n"
        f"stdout:\n{result.stdout[-8000:]}\n"
        f"stderr:\n{result.stderr[-8000:]}"
    )
