from __future__ import annotations

import zipfile
from pathlib import Path

from blueprint_pipeline.provider_evidence_retention import (
    EXECUTE_ACK,
    apply_provider_evidence_retention,
    plan_provider_evidence_retention,
)


def _run(root: Path, name: str, *, drifted: bool = False) -> Path:
    run = root / name
    bundle = run / "bundle"
    unpacked = bundle / "provider_runtime"
    unpacked.mkdir(parents=True)
    (unpacked / "runner.py").write_text("print('x')\n")
    (unpacked / "manifest.json").write_text("{}")
    archive = bundle / "provider_bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("provider_runtime/runner.py", "print('x')\n")
        if not drifted:
            handle.writestr("provider_runtime/manifest.json", "{}")
    (bundle / "bundle_receipt.json").write_text("{}")
    attempts = run / "attempts/attempt_001/immutable_execution"
    attempts.mkdir(parents=True)
    (attempts / "result.json").write_text("{}")
    return run


def test_only_verified_duplicates_are_eligible(tmp_path: Path) -> None:
    """A tree that has drifted from its zip is not a duplicate of it."""

    _run(tmp_path, "run_a")
    _run(tmp_path, "run_b", drifted=True)

    plan = plan_provider_evidence_retention(evidence_root=tmp_path, keep_newest=0)

    paths = {str(row.path) for row in plan.candidates}
    assert str(tmp_path / "run_a/bundle/provider_runtime") in paths
    assert str(tmp_path / "run_b/bundle/provider_runtime") not in paths
    assert str(tmp_path / "run_b/bundle/provider_runtime") in plan.protected


def test_attempt_evidence_and_receipts_are_never_eligible(tmp_path: Path) -> None:
    _run(tmp_path, "run_a")

    plan = plan_provider_evidence_retention(
        evidence_root=tmp_path, keep_newest=0, include_superseded_zips=True
    )

    for row in plan.candidates:
        assert "attempts" not in row.path.parts
        assert not str(row.path).endswith("receipt.json")


def test_the_newest_runs_are_left_alone(tmp_path: Path) -> None:
    import os
    import time

    for index, name in enumerate(("old", "middle", "newest")):
        run = _run(tmp_path, name)
        os.utime(run, (time.time() + index, time.time() + index))

    plan = plan_provider_evidence_retention(evidence_root=tmp_path, keep_newest=1)

    touched = {row.run_directory.name for row in plan.candidates}
    assert "newest" not in touched
    assert touched == {"old", "middle"}


def test_zips_are_a_separate_and_later_tier(tmp_path: Path) -> None:
    _run(tmp_path, "run_a")

    without = plan_provider_evidence_retention(evidence_root=tmp_path, keep_newest=0)
    with_zips = plan_provider_evidence_retention(
        evidence_root=tmp_path, keep_newest=0, include_superseded_zips=True
    )

    assert {row.tier for row in without.candidates} == {"unpacked_duplicate"}
    assert {row.tier for row in with_zips.candidates} == {
        "unpacked_duplicate",
        "superseded_bundle_zip",
    }


def test_nothing_is_deleted_without_an_acknowledgement(tmp_path: Path) -> None:
    _run(tmp_path, "run_a")
    plan = plan_provider_evidence_retention(evidence_root=tmp_path, keep_newest=0)

    dry = apply_provider_evidence_retention(plan, apply=False, ack=None)
    assert dry["status"] == "dry_run"
    assert dry["deleted_count"] == 0
    assert dry["bytes_reclaimable"] > 0
    assert (tmp_path / "run_a/bundle/provider_runtime").is_dir()

    wrong = apply_provider_evidence_retention(plan, apply=True, ack="oops")
    assert wrong["applied"] is False
    assert (tmp_path / "run_a/bundle/provider_runtime").is_dir()


def test_applying_removes_duplicates_and_keeps_evidence(tmp_path: Path) -> None:
    _run(tmp_path, "run_a")
    plan = plan_provider_evidence_retention(evidence_root=tmp_path, keep_newest=0)

    result = apply_provider_evidence_retention(plan, apply=True, ack=EXECUTE_ACK)

    assert result["applied"] is True
    assert result["deleted_count"] == 1
    assert not (tmp_path / "run_a/bundle/provider_runtime").exists()
    assert (tmp_path / "run_a/bundle/provider_bundle.zip").is_file()
    assert (tmp_path / "run_a/attempts/attempt_001/immutable_execution/result.json").is_file()
