from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.run_public_indoor_proxy import (
    PublicIndoorProxyError,
    run_public_indoor_proxy,
)


SOURCE_COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "room.ply"
    bundle = tmp_path / "room.tar.gz"
    if not artifact.exists():
        artifact.write_bytes(
            b"ply\n"
            b"format ascii 1.0\n"
            b"element vertex 2\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"property uchar red\n"
            b"property uchar green\n"
            b"property uchar blue\n"
            b"end_header\n"
            b"0 0 0 255 0 0\n1 1 1 0 255 0\n"
        )
    if not bundle.exists():
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(artifact, arcname="room/room.ply")
    return bundle, artifact


def _run(tmp_path: Path, **overrides):
    bundle, artifact = _fixture(tmp_path)
    values = {
        "dataset_id": "public-room",
        "dataset_source_uri": "https://doi.example/public-room",
        "license_id": "CC-BY-4.0",
        "source_bundle": bundle,
        "source_bundle_sha256": _sha256(bundle),
        "source_artifact": artifact,
        "source_artifact_sha256": _sha256(artifact),
        "output_root": tmp_path / "output",
        "provider_identity": "public-room-provider",
        "consent_status": "accepted",
        "operator_identity": "fixture-operator",
        "source_commit": SOURCE_COMMIT,
        "expected_ply_vertices": 2,
        "acknowledge_test_double_malware_scan": True,
    }
    values.update(overrides)
    return run_public_indoor_proxy(**values)


def test_realistic_public_proxy_replay_is_partial_and_fail_closed(tmp_path: Path) -> None:
    summary = _run(tmp_path)

    assert summary["admission_status"] == "accepted"
    assert summary["qa_status"] == "accepted"
    assert summary["execution_state"] == "partial"
    assert summary["result_outputs"] == ["appearance_layer"]
    assert summary["source_artifact"]["vertex_count"] == 2
    assert summary["source_bundle"]["archive_inspection"]["member_count"] == 1
    assert summary["execution_cost_usd"] == 0.0
    assert summary["claim_ceiling"]["metric_geometry"] is False
    assert summary["claim_ceiling"]["physics"] is False
    assert summary["claim_ceiling"]["physical_task_success"] is False
    assert (
        summary["claim_ceiling"]["comparative_policy_ranking_verdict"]
        == "thesis_not_supported"
    )
    assert summary["proof_boundary"] == {
        "customer_upload_gate_passed": False,
        "deployment_or_safety_approved": False,
        "malware_scan_was_test_double": True,
        "physical_task_success_established": False,
        "production_security_gate_passed": False,
        "raw_capture_gate_passed": False,
        "transfer_was_local_test_double": True,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    persisted = json.loads(
        (tmp_path / "output" / "public_indoor_proxy_replay.json").read_text()
    )
    assert persisted == summary


def test_public_proxy_replay_is_idempotent(tmp_path: Path) -> None:
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert second == first


def test_public_proxy_requires_exact_digests_and_test_double_acknowledgment(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PublicIndoorProxyError,
        match="test_double_malware_scan_acknowledgment_required",
    ):
        _run(tmp_path, acknowledge_test_double_malware_scan=False)

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PublicIndoorProxyError, match="source_artifact_digest_mismatch"):
        _run(other, source_artifact_sha256="sha256:" + "0" * 64)

    uri_case = tmp_path / "uri-case"
    uri_case.mkdir()
    with pytest.raises(PublicIndoorProxyError, match="dataset_source_uri_invalid"):
        _run(
            uri_case,
            dataset_source_uri="https://user:secret@doi.example/public-room",
        )


def test_public_proxy_rejects_archive_traversal(tmp_path: Path) -> None:
    bundle, artifact = _fixture(tmp_path)
    unsafe = tmp_path / "unsafe.tar.gz"
    payload = b"do-not-extract"
    info = tarfile.TarInfo("../escape")
    info.size = len(payload)
    with tarfile.open(unsafe, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(PublicIndoorProxyError, match="source_bundle_archive_unsafe_member"):
        run_public_indoor_proxy(
            dataset_id="public-room",
            dataset_source_uri="https://doi.example/public-room",
            license_id="CC-BY-4.0",
            source_bundle=unsafe,
            source_bundle_sha256=_sha256(unsafe),
            source_artifact=artifact,
            source_artifact_sha256=_sha256(artifact),
            output_root=tmp_path / "output",
            provider_identity="public-room-provider",
            consent_status="accepted",
            operator_identity="fixture-operator",
            source_commit=SOURCE_COMMIT,
            expected_ply_vertices=2,
            acknowledge_test_double_malware_scan=True,
        )


def test_public_proxy_rejects_symlinked_source(tmp_path: Path) -> None:
    bundle, artifact = _fixture(tmp_path)
    link = tmp_path / "linked.ply"
    try:
        link.symlink_to(artifact)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(PublicIndoorProxyError, match="source_artifact_symlink_forbidden"):
        _run(tmp_path / "run", source_artifact=link, source_bundle=bundle)
