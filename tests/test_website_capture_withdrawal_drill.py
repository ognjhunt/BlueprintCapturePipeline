"""The offline consent-withdrawal drill (tracker G-02, exit criterion E5): $0, temporary roots and fakes only.

Beta journey step: site withdrawal. The WebApp's shared withdrawal vector becomes a
tombstone, a dry-run plan deletes nothing, a paid admission for the site is refused,
and an authorized apply on the synthetic site seals a receipt naming every
retention-policy data class, with nothing outside the drill's work root changed.
"""
# Covers: src/blueprint_pipeline/website_capture_withdrawal_drill.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal_deletion.py
# Covers: src/blueprint_pipeline/website_capture_withdrawal.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_capture_withdrawal_deletion as deletion
from blueprint_pipeline import website_capture_withdrawal_drill as drill

ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / "tests/fixtures/website-withdrawal-vector.json"
# The WebApp's strict acknowledgement schema (Blueprint-WebApp server/utils/websiteCaptureWithdrawal.ts,
# ``websiteWithdrawalReceipt``): an unknown key leaves the WebApp's withdrawal pending.
WEBAPP_RECEIPT_KEYS = {
    "schema_version", "request_id", "scene_id", "capture_id", "withdrawal_id", "command_digest", "tombstone_digest",
    "digest", "state", "pipeline_acknowledged", "serve_allowed", "future_processing_allowed",
    "local_cleanup_verified", "local_cleanup_receipt_digest", "deletion_confirmed", "provider_acknowledgement",
    "cloud_storage_cleanup_verified", "delivered_copy_cleanup_verified", "retained_audit_records", "legal_hold",
    "cloud_object_absence_verified", "cloud_object_cleanup_receipt_digest", "cloud_object_current_absence_status",
    "cloud_object_absence_observed_at_iso", "cleanup_scope", "external_cleanup_required"}


def test_withdrawal_drill_runs_offline_from_the_webapp_vector_to_a_sealed_receipt(tmp_path, capsys):
    result = drill.run_drill(tmp_path, vector_path=VECTOR)
    assert result["outside_work_root_unchanged"] is True
    # (a) the paid screening launch's current-source check refuses the withdrawn site.
    assert result["paid_admission_refusal"] == "team_scene_capture_withdrawn"
    # The acknowledgement still parses under the WebApp's strict schema.
    assert set(result["acknowledgement"]) <= WEBAPP_RECEIPT_KEYS
    receipt = result["receipt"]
    # (b) every data class the retention policy names is in the receipt, removed or remaining with a reason.
    policy = json.loads((ROOT / "docs/beta_data_retention_policy_2026-07-09.json").read_text())
    classes = {row["class"]: row for row in receipt["policy_copy_classes"]}
    assert set(classes) == set(policy["data_classes"]) == set(deletion.POLICY_COPY_CLASSES)
    for row in classes.values():
        assert row["removed"] or row["remaining"]
        assert all(item["reason"] for item in row["remaining"])
    holders = {row["holder"] for row in receipt["remaining"]}
    assert {"gpu_and_remote_cpu_providers", "model_and_reconstruction_providers",
            "object_store_provider_physical_erasure", "webapp_and_delivered_copies",
            "pipeline_root_not_covered", "retained_audit_and_lineage_metadata"} <= holders
    assert "capture_source_bucket" not in holders  # the drill named the (fake) source bucket
    uncovered = next(row for row in receipt["remaining"] if row["holder"] == "pipeline_root_not_covered")
    assert uncovered["path"] == str(tmp_path / "inputs/sam31-preparations")
    assert "reproducible SAM preparation artifacts" in uncovered["reason"]
    # The counts the record prints: every covered copy went from some to none.
    counts = {row["copy_class"]: (row["before"], row["after"]) for row in result["record"] if row["before"] is not None}
    assert counts == {"scene_files": (5, 0), "run_directories": (3, 0), "run_pointers": (1, 0),
                      "b2_object_versions": (5, 0), "capture_source_bucket_objects": (1, 0)}
    assert receipt["covered_copies_deleted"] is True and receipt["provider_copies_deleted"] is False
    # (c) the other site's capture, run, B2 object and bucket object are untouched (checked inside the drill too).
    assert (tmp_path / "capture-partition/scenes/site-drill-other/captures/walkthrough-drill-other/raw/video.mp4"
            ).read_bytes() == b"synthetic-video drill-other"
    assert (tmp_path / "control-plane/task-evaluation-launch-runs/launch-drill-other-construction-0003").is_dir()
    with capsys.disabled():
        sys.stdout.write("\n" + drill.format_record(result["record"]) + "\n")


def test_drill_entry_point_prints_the_record(tmp_path, capsys):
    assert drill.main(["--work-root", str(tmp_path / "drill"), "--vector", str(VECTOR)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("copy_class | before | after | remaining_reason")
    assert "paid admission after tombstone: refused (team_scene_capture_withdrawn)" in out


def test_a_run_root_the_planner_cannot_list_is_unresolved_not_skipped(tmp_path, monkeypatch):
    world = drill.build_world(tmp_path, "req1")
    withdrawal.acknowledge_withdrawal(capture_root=world["capture"],
                                      command=json.loads(VECTOR.read_text())["command"])
    unreadable = world["resolution"]["run_roots"][0]
    original = Path.iterdir

    def iterdir(path):
        if path == unreadable:
            raise PermissionError("synthetic")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    plan = deletion.plan_withdrawal_deletion(capture_root=world["capture"], **world["resolution"])
    assert {"path": str(unreadable), "reason": "website_withdrawal_deletion_root_unreadable"} in plan["unresolved"]


def test_the_drill_fails_when_anything_outside_its_work_root_changes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(drill.DrillFailure, match="drill_wrote_outside_work_root"), drill._confined(work):
        (tmp_path / "outside.txt").write_text("not the drill's")

