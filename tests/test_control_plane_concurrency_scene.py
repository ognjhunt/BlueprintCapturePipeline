"""ADP-009D: the load scene starts with real intake and publication joins."""

from __future__ import annotations

import json
import subprocess
from collections import namedtuple

import pytest

from scripts.control_plane_concurrency_scene import advance_fixture_intake, advance_fixture_preparation


@pytest.mark.slow
def test_intake_factory_publication_and_owned_queue_share_actual_identity(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    objects = tmp_path / "objects"
    objects.mkdir()
    result = advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects,
                                    source_commit=source)
    assert result["claim_ceiling"] == "development_only"
    assert result["source_commit"] == source
    assert result["progression"]["results"][0]["status"] == "running", result
    assert result["publication"]["status"] == "published_and_read_back"
    assert result["publication"]["raw_source_uploaded"] is False
    assert result["publication"]["provider_allocated"] is False
    request = json.loads(result["request_path"].read_text())
    assert request["expected_production_commit"] == source
    assert request["scene_intent_digest"] == result["intent"]["intent_digest"]
    pending = list((result["preparation_queue"] / "pending").glob("*.json"))
    assert len(pending) == 1
    envelope = json.loads(pending[0].read_text())
    assert envelope["request"] == request
    assert result["publication"]["manifest_sha256"] == result["factory"]["submission_manifest"]["sha256"]
    assert result["object_bytes_uploaded"] > 0


def test_intake_fixture_cannot_replace_the_actual_checkout_validator(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    with pytest.raises(ValueError, match="harness_checkout_source_mismatch"):
        advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects,
                               source_commit="0" * 40)
    assert not (tmp_path / "scene").exists()


@pytest.mark.slow
def test_preparation_reads_real_objects_and_seals_same_scene_construction(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    usage = namedtuple("Usage", "total used free")(200 * 1024**3, 50 * 1024**3, 150 * 1024**3)

    def measured_test_reservation(role, **kwargs):
        # This local contract test supplies disk capacity, retaining the real
        # ledger/reservation implementation. Live harness code has no seam.
        return reserve_control_plane_disk(role, **kwargs, disk_usage=lambda _: usage)

    monkeypatch.setattr(worker, "reserve_control_plane_disk", measured_test_reservation)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    objects = tmp_path / "objects"
    objects.mkdir()
    first = advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects, source_commit=source)
    result = advance_fixture_preparation(intake=first, object_root=objects,
                                        reservation_root=tmp_path / "reservations", pins_root=tmp_path / "pins")
    assert result["run"]["results"][0]["status"] == "queued_for_production_scene_configuration", result
    envelope = result["construction_envelope"]
    assert envelope["request"] == json.loads(first["request_path"].read_text())
    assert envelope["envelope_digest"] == result["run"]["results"][0]["construction_queue_envelope_digest"]
    assert all(row["full_byte_service_account_readback_passed"] for row in envelope["materialized_references"])
    assert envelope["render_inputs_result"]["fixture_provider"] is True
    assert not list((tmp_path / "reservations").glob("*.json"))
