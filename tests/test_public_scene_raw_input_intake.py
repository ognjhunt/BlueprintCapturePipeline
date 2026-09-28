"""Raw scene intake does not demand future registration/qualification outputs."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import public_scene_host_input_intake as intake


def _request(root: Path, *, extra_registration: bool = False) -> Path:
    # These are tiny hermetic source bytes, never scene qualification evidence.
    source_specs = {
        "appearance_3dgs": ("scene.ply", b"ply\nformat ascii 1.0\nend_header\n"),
        "semantic_metadata": ("labels.json", b"[]\n"),
        "scene_structure": ("structure.json", b'{"rooms":[]}\n'),
        "collision_usd": ("collision.usda", b'#usda 1.0\ndef Xform "Scene" {}\n'),
        "publisher_scene_usdz": ("source.usdz", b"unqualified fixture bytes"),
    }
    if extra_registration:
        source_specs["shared_frame_registration"] = ("frame.json", b"{}\n")
    files = []
    for role, (name, content) in source_specs.items():
        path = root / name
        path.write_bytes(content)
        files.append({
            "role": role, "path": str(path),
            "sha256": intake._sha256_file(path),
            "rights_receipt_ids": ["fixture-human-authority"],
        })
    rights = root / "rights.json"
    rights.write_text(json.dumps({
        "schema_version": intake.RIGHTS_RECEIPT_SCHEMA,
        "reviewer_status": "approved_for_declared_use",
        "agent_accepted_terms": False,
        "authorized_source_sha256": [row["sha256"] for row in files],
    }))
    value = {
        "schema_version": intake.RAW_SCENE_REQUEST_SCHEMA,
        "scene_id": "fixture-new-scene",
        "packet_id": "fixture-raw-scene-v2",
        "source_commit_sha": intake._verified_checkout_head(),
        "adp_item": "ADP-009D",
        "rights_receipts": [{
            "receipt_id": "fixture-human-authority",
            "path": str(rights), "sha256": intake._sha256_file(rights),
        }],
        "files": files,
    }
    request = root / "request.json"
    request.write_text(json.dumps(value))
    return request


def test_raw_scene_is_installed_without_caller_registration_or_qualification(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    before = {row["path"]: Path(row["path"]).read_bytes()
              for row in json.loads(request.read_text())["files"]}
    stream = io.BytesIO()
    metadata = intake.build_packet_archive(request, stream)
    assert metadata["schema_version"] == intake.RAW_SCENE_PACKET_SCHEMA
    assert {row["role"] for row in metadata["files"]} == {
        "appearance_3dgs", "semantic_metadata", "scene_structure",
        "collision_usd", "publisher_scene_usdz",
    }
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        assert intake._validated_archive(archive)["packet_digest"] == metadata["packet_digest"]
    stream.seek(0)
    receipt = intake.install_packet_archive(
        stream, destination_root=tmp_path / "installed",
        allowed_roots=(tmp_path,), service_account=None,
    )
    assert receipt["service_readable"] is True
    assert receipt["packet_digest"] == metadata["packet_digest"]
    assert receipt["paid_resource_used"] is False
    assert receipt["provider_mutation_performed"] is False
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert metadata["claim_ceiling"] == "rights_bound_public_scene_source_bytes_only"


@pytest.mark.parametrize("missing", [
    "appearance_3dgs", "semantic_metadata", "scene_structure", "collision_usd",
])
def test_raw_scene_refuses_missing_source_components(tmp_path: Path, missing: str) -> None:
    path = _request(tmp_path)
    value = json.loads(path.read_text())
    value["files"] = [row for row in value["files"] if row["role"] != missing]
    path.write_text(json.dumps(value))
    with pytest.raises(intake.PublicSceneHostInputError, match="required_scene_source_files_invalid"):
        intake.build_packet_archive(path, io.BytesIO())


def test_raw_scene_cannot_masquerade_caller_registration_as_source_bytes(tmp_path: Path) -> None:
    with pytest.raises(intake.PublicSceneHostInputError, match="source_file_role_invalid"):
        intake.build_packet_archive(_request(tmp_path, extra_registration=True), io.BytesIO())


def test_raw_scene_keeps_digest_specific_human_rights_gate(tmp_path: Path) -> None:
    path = _request(tmp_path)
    value = json.loads(path.read_text())
    value["rights_receipts"] = []
    path.write_text(json.dumps(value))
    with pytest.raises(intake.PublicSceneHostInputError, match="rights_receipts_missing"):
        intake.build_packet_archive(path, io.BytesIO())
