"""The runtime guard must prove every control-plane entrypoint is importable.

On 2026-08-12 a rebuilt control-plane host passed every environment check and
served traffic while `paid_resource_allocator` -- the only provider-mutation
entrypoint -- could not be imported at all. The base install omits the
`runtime` extra, so a transitive `cv2` import failed. Nothing surfaced until a
stranded Vast record needed releasing, at which point the allocator exited 1
with no receipt and the release queue blocked.

An entrypoint that cannot be imported cannot allocate, release, reconcile, or
guard spend, so this belongs at startup rather than in the first operation that
needs it.
"""

import re
from pathlib import Path

from blueprint_pipeline.production_runtime_env_guard import (
    CONTROL_PLANE_ENTRYPOINTS,
    build_production_runtime_env_guard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"

READY_ENV = {
    "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
    "PRIVACY_PIPELINE_ENABLED": "true",
    "PRIVACY_FAIL_CLOSED": "true",
    "PIPELINE_SYNC_REQUIRED": "true",
    "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
}

BLOCKER_PREFIX = "control_plane_entrypoint_not_importable:"
ALLOCATOR = "blueprint_pipeline.paid_resource_allocator"


def test_covers_every_module_the_systemd_units_execute() -> None:
    """A new unit must not be able to ship an unchecked entrypoint."""
    referenced = set()
    for unit in SYSTEMD_DIR.glob("*.service"):
        referenced.update(
            re.findall(r"-m (blueprint_pipeline\.[a-z_]+)", unit.read_text(encoding="utf-8"))
        )
    missing = referenced - set(CONTROL_PLANE_ENTRYPOINTS)
    assert not missing, f"systemd units execute unchecked entrypoints: {sorted(missing)}"


def test_covers_the_canonical_allocator() -> None:
    """It is invoked as a subprocess, not an ExecStart, which is how it escaped."""
    assert ALLOCATOR in CONTROL_PLANE_ENTRYPOINTS


def test_reports_ready_when_every_entrypoint_imports() -> None:
    report = build_production_runtime_env_guard(READY_ENV, import_module=lambda name: object())
    assert report["status"] == "ready"
    assert report["control_plane_entrypoints"]["importable"] is True
    assert report["control_plane_entrypoints"]["failed"] == []


def test_blocks_and_names_each_unimportable_entrypoint() -> None:
    def selective(name: str):
        if name == ALLOCATOR:
            raise ModuleNotFoundError("No module named 'cv2'")
        return object()

    report = build_production_runtime_env_guard(READY_ENV, import_module=selective)
    assert report["status"] == "blocked"
    assert f"{BLOCKER_PREFIX}{ALLOCATOR}" in report["blockers"]

    failed = report["control_plane_entrypoints"]["failed"]
    assert [entry["module"] for entry in failed] == [ALLOCATOR]
    # An operator needs the missing dependency name to fix this in one step.
    assert "cv2" in str(failed[0]["error"])
    assert "runtime" in report["control_plane_entrypoints"]["remediation"]


def test_reports_every_failure_rather_than_stopping_at_the_first() -> None:
    report = build_production_runtime_env_guard(
        READY_ENV,
        import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    failed = {entry["module"] for entry in report["control_plane_entrypoints"]["failed"]}
    assert failed == set(CONTROL_PLANE_ENTRYPOINTS)


def test_import_check_is_independent_of_the_environment_flags() -> None:
    """A blocked environment must not mask, or be masked by, an import failure."""

    def explode(name: str):
        raise ModuleNotFoundError("No module named 'cv2'")

    report = build_production_runtime_env_guard({}, import_module=explode)
    assert any(b.startswith(BLOCKER_PREFIX) for b in report["blockers"])
    assert any(b.startswith("missing_or_false_") for b in report["blockers"])


def test_every_entrypoint_imports_in_this_environment() -> None:
    """Guards the repository's own declared dependency set, not just the fake."""
    report = build_production_runtime_env_guard(READY_ENV)
    failed = report["control_plane_entrypoints"]["failed"]
    assert failed == [], f"unimportable control-plane entrypoints: {failed}"
