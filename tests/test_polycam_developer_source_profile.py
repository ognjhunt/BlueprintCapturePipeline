from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import zipfile

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.polycam_developer_source_profile import (
    PolycamDeveloperSourceProfileError,
    build_polycam_developer_source_profile,
    compile_polycam_developer_source_declaration,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIGEST = "sha256:" + "a" * 64
SOURCE_COMMIT = "1" * 40


def _members() -> dict[str, bytes]:
    return {
        "metadata/capture.json": json.dumps(
            {
                "provider": "polycam",
                "capture_id": "polycam-capture-001",
                "device": "iPhone 15 Pro",
            }
        ).encode(),
        "keyframes/images/000001.jpg": b"full-resolution-rgb-fixture",
        "keyframes/cameras/000001.json": json.dumps(
            {
                "timestamp_ns": 1_000_000,
                "intrinsics": [100.0, 100.0, 50.0, 50.0],
                "world_from_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            }
        ).encode(),
        "keyframes/depth/000001.png": b"metric-depth-fixture",
        "keyframes/confidence/000001.png": b"confidence-fixture",
        "mesh/raw_mesh.glb": b"raw-mesh-fixture",
        "mesh/mesh_info.json": json.dumps(
            {"length_unit": "meter", "scale_to_meters": 1.0}
        ).encode(),
        "provider/unbound-note.txt": b"unbound-but-still-hashed",
    }


def _archive(path: Path, members: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (members or _members()).items():
            archive.writestr(name, payload)
    return path


def _declaration() -> dict:
    camera = "keyframes/cameras/000001.json"
    metadata = "metadata/capture.json"
    mesh_info = "mesh/mesh_info.json"
    return {
        "schema_version": "polycam_developer_source_declaration.v1",
        "source_profile": "polycam_developer_mode_lidar_raw_zip",
        "provider_identity": "polycam",
        "source_capture_identity": "site-capture-001",
        "provider_capture_identity": "polycam-capture-001",
        "provider_app_version": "4.2.0",
        "provider_export_timestamp": "2026-08-03T01:02:03-05:00",
        "layout_profile": "polycam-developer-raw-observed-v1",
        "capture_mode": "space_lidar",
        "developer_mode_enabled": True,
        "blueprint_remote_upload_performed": False,
        "device_identity": {
            "manufacturer": "apple",
            "model": "iPhone 15 Pro",
            "operating_system": "iOS 19.0",
            "lidar_capable": True,
        },
        "metric_units": {"length_unit": "meters", "scale_to_meters": 1},
        "semantic_bindings": {
            "source_rgb_frames": ["keyframes/images/000001.jpg"],
            "source_video": [],
            "frame_timestamps": [camera],
            "camera_intrinsics": [camera],
            "camera_extrinsics": [camera],
            "depth": ["keyframes/depth/000001.png"],
            "confidence": ["keyframes/confidence/000001.png"],
            "mesh_geometry": ["mesh/raw_mesh.glb"],
            "mesh_info": [mesh_info],
            "metric_units": [mesh_info],
            "capture_identity": [metadata],
            "device_identity": [metadata],
            "provider_identity": [metadata],
        },
    }


def _build(archive: Path, declaration: dict | None = None) -> dict:
    return build_polycam_developer_source_profile(
        archive_path=archive,
        declaration=declaration or _declaration(),
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
    )


def test_raw_zip_is_fully_hashed_bound_and_remains_provider_derived(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration = compile_polycam_developer_source_declaration(_declaration())
    profile = _build(archive_path, declaration)
    replay = _build(archive_path, declaration)

    assert replay == profile
    assert profile["status"] == "admitted_provider_derived_support"
    assert profile["blockers"] == []
    assert profile["smallest_missing_measurement"] is None
    assert profile["source_archive"]["original_archive_preserved"] is True
    assert profile["source_archive"]["archive_extracted_by_adapter"] is False
    assert profile["source_archive"]["member_count"] == len(_members())
    assert profile["source_archive"]["sha256"] == (
        "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    assert [row["member_path"] for row in profile["member_inventory"]] == sorted(
        _members()
    )
    with zipfile.ZipFile(archive_path) as archive:
        for row in profile["member_inventory"]:
            assert row["sha256"] == (
                "sha256:"
                + hashlib.sha256(archive.read(row["member_path"])).hexdigest()
            )
    assert profile["unbound_members"] == ["provider/unbound-note.txt"]
    assert profile["semantic_bindings"]["frame_timestamps"] == profile[
        "semantic_bindings"
    ]["camera_intrinsics"]
    assert profile["metric_units"] == {
        "authority": "provider_declared_unqualified",
        "length_unit": "meter",
        "scale_to_meters": 1.0,
    }
    assert profile["claim_boundary"]["provider_derived_support"] is True
    assert profile["claim_boundary"]["blueprint_raw_contract_truth"] is False
    assert profile["claim_boundary"]["encoder_attempt_evidence_present"] is False
    assert profile["claim_boundary"]["retained_frame_evidence_present"] is False
    assert profile["claim_boundary"]["metric_scale_independently_proven"] is False
    assert profile["claim_boundary"]["collision_geometry_qualified"] is False
    assert profile["claim_boundary"]["isaac_compatibility_proven"] is False
    assert profile["claim_boundary"]["task_success_proven"] is False
    assert profile["claim_boundary"]["physical_success_proven"] is False
    assert profile["source_profile_digest"] == canonical_digest(
        profile, digest_field="source_profile_digest"
    )

    for name, value in (
        ("polycam_developer_source_declaration.v1.schema.json", declaration),
        ("polycam_developer_source_profile.v1.schema.json", profile),
    ):
        schema = json.loads((ROOT / "docs" / "schemas" / name).read_text())
        jsonschema.validate(value, schema)


def test_missing_semantic_lane_abstains_with_smallest_measurement(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration = _declaration()
    declaration["semantic_bindings"]["confidence"] = []

    profile = _build(archive_path, declaration)

    assert profile["status"] == "abstained"
    assert profile["blockers"] == ["semantic_lane_missing:confidence"]
    assert profile["smallest_missing_measurement"] == {
        "code": "semantic_lane_missing:confidence",
        "instruction": "Export and bind the depth-confidence member or members.",
    }
    assert profile["proof_effect"] == "none"
    assert profile["claim_ceiling"] == "none"
    assert profile["legal_next_actions"] == [
        "preserve_original_archive",
        "supply_smallest_missing_measurement",
    ]


def test_declared_missing_member_abstains_without_extracting(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration = _declaration()
    declaration["semantic_bindings"]["depth"] = ["keyframes/depth/999999.png"]

    profile = _build(archive_path, declaration)

    assert profile["status"] == "abstained"
    assert profile["blockers"] == [
        "declared_member_missing:keyframes/depth/999999.png"
    ]
    assert not (tmp_path / "keyframes").exists()


def test_declaration_digest_and_metric_unit_drift_fail_closed(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration = compile_polycam_developer_source_declaration(_declaration())
    declaration["provider_capture_identity"] = "changed-after-digest"
    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="source_declaration_digest_mismatch",
    ):
        _build(archive_path, declaration)

    invalid_units = _declaration()
    invalid_units["metric_units"] = {"length_unit": "centimeter", "scale_to_meters": 0.01}
    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="metric_length_unit_invalid",
    ):
        _build(archive_path, invalid_units)

    unexpected = _declaration()
    unexpected["api_token"] = "must-not-enter-artifact"
    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="source_declaration_field_unsupported:api_token",
    ):
        _build(archive_path, unexpected)


@pytest.mark.parametrize("unsafe_name", ["../escape.json", "/absolute.json", "a\\b.json"])
def test_unsafe_archive_member_paths_are_rejected(
    tmp_path: Path, unsafe_name: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(unsafe_name, b"unsafe")
    with pytest.raises(
        PolycamDeveloperSourceProfileError, match="archive_member_path_unsafe"
    ):
        _build(archive_path)


def test_symlink_and_duplicate_archive_members_are_rejected(tmp_path: Path) -> None:
    symlink_path = tmp_path / "symlink.zip"
    symlink_info = zipfile.ZipInfo("metadata/link.json")
    symlink_info.create_system = 3
    symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_path, "w") as archive:
        archive.writestr(symlink_info, "../target")
    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="source_archive_symlink_member_forbidden",
    ):
        _build(symlink_path)

    duplicate_path = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_path, "w") as archive:
            archive.writestr("metadata/capture.json", b"first")
            archive.writestr("metadata/capture.json", b"second")
    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="source_archive_duplicate_member",
    ):
        _build(duplicate_path)


def test_source_archive_symlink_is_rejected(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    link = tmp_path / "polycam-link.zip"
    link.symlink_to(archive_path)
    with pytest.raises(
        PolycamDeveloperSourceProfileError, match="source_archive_symlink_forbidden"
    ):
        build_polycam_developer_source_profile(
            archive_path=link,
            declaration=_declaration(),
            source_commit_sha=SOURCE_COMMIT,
            implementation_digest=IMPLEMENTATION_DIGEST,
        )


def test_cli_writes_once_and_returns_nonzero_for_abstention(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(_declaration()))
    output_path = tmp_path / "profile.json"

    assert main(
        [
            "--archive",
            str(archive_path),
            "--declaration",
            str(declaration_path),
            "--output",
            str(output_path),
            "--source-commit-sha",
            SOURCE_COMMIT,
        ]
    ) == 0
    written = json.loads(output_path.read_text())
    assert written["status"] == "admitted_provider_derived_support"

    blocked = _declaration()
    blocked["semantic_bindings"]["source_rgb_frames"] = []
    declaration_path.write_text(json.dumps(blocked))
    blocked_output = tmp_path / "blocked.json"
    assert main(
        [
            "--archive",
            str(archive_path),
            "--declaration",
            str(declaration_path),
            "--output",
            str(blocked_output),
            "--source-commit-sha",
            SOURCE_COMMIT,
        ]
    ) == 2
    assert json.loads(blocked_output.read_text())["smallest_missing_measurement"][
        "code"
    ] == "source_appearance_missing"


def test_cli_rejects_symlink_output(tmp_path: Path) -> None:
    archive_path = _archive(tmp_path / "polycam-raw.zip")
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_text(json.dumps(_declaration()))
    target = tmp_path / "target.json"
    target.write_text("do not overwrite")
    output = tmp_path / "output.json"
    output.symlink_to(target)

    with pytest.raises(
        PolycamDeveloperSourceProfileError,
        match="source_profile_output_symlink_forbidden",
    ):
        main(
            [
                "--archive",
                str(archive_path),
                "--declaration",
                str(declaration_path),
                "--output",
                str(output),
                "--source-commit-sha",
                SOURCE_COMMIT,
            ]
        )
    assert target.read_text() == "do not overwrite"
