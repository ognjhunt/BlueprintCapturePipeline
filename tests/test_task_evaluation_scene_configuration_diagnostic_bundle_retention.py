from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_diagnostic_bundle_retention import (
    APPLY_ACKNOWLEDGEMENT,
    DiagnosticBundleRetentionError,
    apply_unlaunched_bundle_retention_plan,
    write_unlaunched_bundle_retention_plan,
)


SOURCE_COMMIT = "a" * 40


def _sealed_bundle(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "diagnostics/attempt-1/bundle-dry"
    root.mkdir(parents=True)
    bundle = root / "task_evaluation_scene_configuration_provider_bundle.zip"
    bundle.write_bytes(b"sealed diagnostic bundle")
    receipt = root / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    old = 1_700_000_000
    os.utime(bundle, (old, old))
    os.utime(receipt, (old, old))

    def validator(_path: Path, *, diagnostic_only: bool):
        assert diagnostic_only is True
        import hashlib

        digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
        return {
            "source_commit": SOURCE_COMMIT,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "bundle_path": str(bundle),
            "bundle_sha256": digest,
            "bundle_size_bytes": bundle.stat().st_size,
        }

    return root, validator


def test_plan_and_apply_reclaim_only_never_authorized_bundle(tmp_path: Path) -> None:
    root, validator = _sealed_bundle(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = write_unlaunched_bundle_retention_plan(
        bundle_root=root,
        diagnostics_root=tmp_path / "diagnostics",
        minimum_age_seconds=3600,
        now=1_700_010_000,
        destination=plan_path,
        bundle_validator=validator,
    )

    result = apply_unlaunched_bundle_retention_plan(
        plan_path=plan_path,
        receipt_out=tmp_path / "apply.json",
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        now=1_700_010_001,
        bundle_validator=validator,
    )

    assert result["removed_bytes"] == plan["predicted_removed_bytes"]
    assert not (root / "task_evaluation_scene_configuration_provider_bundle.zip").exists()
    assert not (root / f"{BUNDLE_SCHEMA_VERSION}.receipt.json").exists()
    assert json.loads((tmp_path / "apply.json").read_text())["status"] == (
        "unlaunched_diagnostic_bundle_reclaimed"
    )


def test_plan_refuses_any_paid_execution_marker(tmp_path: Path) -> None:
    root, validator = _sealed_bundle(tmp_path)
    (root.parent / "attempt-authority.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        DiagnosticBundleRetentionError,
        match="diagnostic_bundle_retention_execution_evidence_present",
    ):
        write_unlaunched_bundle_retention_plan(
            bundle_root=root,
            diagnostics_root=tmp_path / "diagnostics",
            minimum_age_seconds=3600,
            now=1_700_010_000,
            destination=tmp_path / "plan.json",
            bundle_validator=validator,
        )


def test_apply_refuses_bundle_changed_after_plan(tmp_path: Path) -> None:
    root, validator = _sealed_bundle(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_unlaunched_bundle_retention_plan(
        bundle_root=root,
        diagnostics_root=tmp_path / "diagnostics",
        minimum_age_seconds=3600,
        now=1_700_010_000,
        destination=plan_path,
        bundle_validator=validator,
    )
    bundle = root / "task_evaluation_scene_configuration_provider_bundle.zip"
    bundle.write_bytes(b"changed")
    old = 1_700_000_000
    os.utime(bundle, (old, old))

    with pytest.raises(
        DiagnosticBundleRetentionError,
        match="diagnostic_bundle_retention_plan_changed",
    ):
        apply_unlaunched_bundle_retention_plan(
            plan_path=plan_path,
            receipt_out=tmp_path / "apply.json",
            acknowledgement=APPLY_ACKNOWLEDGEMENT,
            now=1_700_010_001,
            bundle_validator=validator,
        )
