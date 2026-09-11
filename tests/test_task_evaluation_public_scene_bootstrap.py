from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.public_scene_host_input_intake import _verified_checkout_head
from blueprint_pipeline.task_evaluation_public_scene_bootstrap import prepare_registered_public_scene
from blueprint_pipeline.task_evaluation_public_scene_catalog import (
    ROLE_REPOSITORIES, SCHEMA, SOURCE_SCHEMA, content_digest, load_catalog,
)
from tests.test_sage_collision_partition import combined_scene


@pytest.fixture(autouse=True)
def hermetic_capacity(monkeypatch):
    from blueprint_pipeline import control_plane_disk_budget as disk
    real = disk.reserve_control_plane_disk
    monkeypatch.setattr(disk, "reserve_control_plane_disk", lambda *a, **kw: real(*a, **kw,
        disk_usage=lambda _path: SimpleNamespace(total=100 * disk.GIB, used=10 * disk.GIB, free=90 * disk.GIB)))


def _seal(value, field):
    value[field] = cross_runtime_canonical_digest(value, digest_field=field)
    return value


def source_fixture(tmp_path, scene_id="123456"):
    fixture = tmp_path / "publisher"
    fixture.mkdir()
    collision, labels = combined_scene(fixture)
    values = {"appearance_3dgs": b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n",
              "semantic_metadata": labels.read_bytes(), "scene_structure": json.dumps({
                  "rooms": [{"profile": [[-1, -1], [3, -1], [3, 2], [-1, 2]]}], "walls": [], "holes": []}).encode(),
              "collision_usd": collision.read_bytes(), "publisher_scene_usdz": b"PK-source-usdz-fixture"}
    revision = "a" * 40
    paths = {"appearance_3dgs": f"0001_{scene_id}/3dgs_compressed.ply",
             "semantic_metadata": f"0001_{scene_id}/labels.json", "scene_structure": f"0001_{scene_id}/structure.json",
             "collision_usd": f"Collision_Mesh/{scene_id}/{scene_id}_collision.usd",
             "publisher_scene_usdz": f"InteriorGS_usdz/{scene_id}.usdz"}
    files = [{"role": role, "publisher_revision": revision,
              "publisher_url": f"https://huggingface.co/datasets/spatialverse/{ROLE_REPOSITORIES[role]}/resolve/{revision}/{paths[role]}",
              "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for role, data in values.items()]
    evidence = []
    for role, url in (("interiorgs_terms", "https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf"),
                      ("interiorgs_readme", f"https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/{revision}/README.md"),
                      ("sage_readme", f"https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh/resolve/{revision}/README.md")):
        data = ("fixture terms " + role).encode()
        values[role] = data
        evidence.append({"role": role, "publisher_url": url, "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                         "size_bytes": len(data)})
    rights = {"use_scope": "noncommercial_internal_research", "raw_redistribution_allowed": False,
              "provider_training_allowed": False, "evidence": evidence}
    task = {"task_id": "task-public-" + scene_id, "strategy": "pick_and_place",
            "subject": {"source_instance_id": "subject", "description": "small rigid object"},
            "support": {"source_instance_id": "support", "description": "table"},
            "destination": {"kind": "green_region", "relation": "on", "visible_label": "green spot",
                            "position_world_m": [.6, .2, .75], "orientation_xyzw": [0, 0, 0, 1], "radius_m": .07},
            "success": {"minimum_lift_m": .05}}
    choice = _seal({"schema_version": SOURCE_SCHEMA, "source_kind": "public_scene", "publisher_scene_id": scene_id,
        "binding_id": "public-" + scene_id, "label": "Fixture public scene", "claim_scope": "development_only",
        "files": files, "source_content_digest": content_digest(scene_id, files), "rights": rights,
        "rights_reference": cross_runtime_canonical_digest(rights), "task_proposal": task,
        "task_proposal_digest": cross_runtime_canonical_digest(task), "required_providers": ["vast", "openai"]}, "choice_digest")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_seal({"schema_version": SCHEMA, "sources": [choice],
                                            "provider_mutation_performed": False}, "catalog_digest")))
    intent = {"intent_id": "scene-test-" + scene_id, "intent_digest": "sha256:" + "1" * 64,
              "request": {"source": {"kind": "public_scene", "binding_id": choice["binding_id"],
                                     "content_digest": choice["source_content_digest"]}, "task": task,
                          "consent": {"rights_reference": choice["rights_reference"], "private_processing_authorized": True,
                                      "accepted_by": "owner", "accepted_at_epoch": time.time()}}}
    config = {"public_source_catalog_path": str(catalog_path), "factory_output_root": str(tmp_path / "controller"),
              "service_account": None, "preparation_worker": {"disk_reservation_root": str(tmp_path / "disk-reservations")}}
    return values, choice, intent, config


@pytest.mark.parametrize("scene_id", ["123456", "654321"])
def test_fresh_source_controller_installs_and_prepares_different_scenes_without_finished_inputs(tmp_path, scene_id):
    values, choice, intent, config = source_fixture(tmp_path, scene_id)
    calls = []
    def download(row, path):
        calls.append(row["role"])
        path.write_bytes(values[row["role"]])
    result = prepare_registered_public_scene(intent=intent, config=config,
        release={"source_commit": _verified_checkout_head()}, downloader=download)
    assert result.status == "awaiting_source"
    assert result.blockers == ("public_scene_configuration_binding_required",)
    retained = json.loads(Path(result.analysis_reference["path"]).read_text())
    assert retained["status"] == "source_prepared_pending_configuration_binding"
    assert retained["provider_allocation_performed"] is False
    prepared = json.loads(Path(retained["references"]["source_preparation_receipt"]["path"]).read_text())
    assert prepared["status"] == "source_context_prepared_pending_calibrated_views"
    assert prepared["collision_partition"]
    assert set(calls) == set(values)
    assert len(calls) == 8
    again = prepare_registered_public_scene(intent=intent, config=config,
        release={"source_commit": _verified_checkout_head()}, downloader=download)
    assert again.analysis_reference == result.analysis_reference
    assert len(calls) == 8
    assert not list((tmp_path / "controller").rglob("*destination*"))


def test_catalog_cannot_change_publisher_scene_under_a_valid_digest(tmp_path):
    _values, _choice, _intent, config = source_fixture(tmp_path)
    path = Path(config["public_source_catalog_path"])
    catalog = json.loads(path.read_text())
    row = catalog["sources"][0]
    row["files"][0]["publisher_url"] = row["files"][0]["publisher_url"].replace("123456", "123457")
    _seal(row, "choice_digest")
    path.write_text(json.dumps(_seal(catalog, "catalog_digest")))
    with pytest.raises(ValueError, match="publisher_scene_path_mismatch"):
        load_catalog(path)


def test_failed_publisher_read_resumes_only_missing_files_without_spend(tmp_path):
    values, _choice, intent, config = source_fixture(tmp_path)
    calls = []
    def fail_once(row, path):
        calls.append(row["role"])
        if row["role"] == "scene_structure":
            raise OSError("retained transport fixture failure")
        path.write_bytes(values[row["role"]])
    with pytest.raises(OSError, match="retained transport"):
        prepare_registered_public_scene(intent=intent, config=config,
            release={"source_commit": _verified_checkout_head()}, downloader=fail_once)
    resumed = []
    def succeed(row, path):
        resumed.append(row["role"])
        path.write_bytes(values[row["role"]])
    prepare_registered_public_scene(intent=intent, config=config,
        release={"source_commit": _verified_checkout_head()}, downloader=succeed)
    assert "appearance_3dgs" not in resumed and "semantic_metadata" not in resumed
    assert resumed[0] == "scene_structure"


def test_capacity_failure_precedes_all_publisher_reads(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_disk_budget as disk
    _values, _choice, intent, config = source_fixture(tmp_path)
    def refused(*_args, **_kwargs):
        raise disk.ControlPlaneDiskBudgetError("control_plane_disk_budget_exceeded")
    monkeypatch.setattr(disk, "reserve_control_plane_disk", refused)
    with pytest.raises(disk.ControlPlaneDiskBudgetError, match="budget_exceeded"):
        prepare_registered_public_scene(intent=intent, config=config,
            release={"source_commit": _verified_checkout_head()},
            downloader=lambda *_args: pytest.fail("read publisher before storage admission"))
    assert not Path(config["factory_output_root"]).exists()
