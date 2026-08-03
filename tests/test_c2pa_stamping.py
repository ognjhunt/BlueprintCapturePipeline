"""C2PA edge stamping: sidecar-only, fail-closed, ledger stays authoritative."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.c2pa_stamping import (
    C2PA_ENABLED_ENV,
    C2PA_SIGN_CERT_FILE_ENV,
    C2PA_SIGN_KEY_FILE_ENV,
    C2PA_TOOL_BIN_ENV,
    LEDGER_REF_ASSERTION_LABEL,
    SCHEMA_VERSION,
    C2paAssertionContentError,
    apply_edge_stamping,
    build_ledger_ref_assertion,
    stamp_3d_asset_sidecar,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_refs() -> dict:
    return {
        "scene_id": "scene-1",
        "capture_id": "cap-1",
        "consent_evidence_digest": "sha256:" + "a" * 64,
        "signed_chain_manifest_sha256": "b" * 64,
        "holdout_split_sha256": "c" * 64,
        "verification_uri": "https://blueprint.example/verify/scene-1/cap-1",
    }


def _package_with_media(tmp_path: Path) -> tuple[Path, list[str]]:
    package_dir = tmp_path / "package"
    objects = package_dir / "exports/video_bundle/objects/sha256/aa"
    objects.mkdir(parents=True)
    clip = objects / ("a" * 64 + ".mp4")
    clip.write_bytes(b"fake-mp4-bytes")
    return package_dir, [str(clip.relative_to(package_dir))]


def _tool_config(tmp_path: Path) -> dict[str, str]:
    tool = tmp_path / "c2patool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    cert = tmp_path / "sign.pem"
    cert.write_text("cert", encoding="utf-8")
    key = tmp_path / "sign.key"
    key.write_text("key", encoding="utf-8")
    return {
        C2PA_ENABLED_ENV: "1",
        C2PA_TOOL_BIN_ENV: str(tool),
        C2PA_SIGN_CERT_FILE_ENV: str(cert),
        C2PA_SIGN_KEY_FILE_ENV: str(key),
    }


class _FakeRunner:
    """Simulates c2patool: sign writes output asset + sidecar, verify prints report."""

    def __init__(
        self,
        *,
        mutate_output: bool = False,
        verify_mentions_label: bool = True,
        sign_returncode: int = 0,
    ) -> None:
        self.mutate_output = mutate_output
        self.verify_mentions_label = verify_mentions_label
        self.sign_returncode = sign_returncode
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str]) -> "subprocess_result":
        self.calls.append(list(command))
        if "-o" in command:
            source = Path(command[1])
            output = Path(command[command.index("-o") + 1])
            body = source.read_bytes()
            if self.mutate_output:
                body = body + b"-mutated"
            output.write_bytes(body)
            output.with_suffix(".c2pa").write_bytes(b"sidecar-manifest-store")
            return subprocess_result(self.sign_returncode, "", "")
        report = LEDGER_REF_ASSERTION_LABEL if self.verify_mentions_label else "no manifest"
        return subprocess_result(0, report, "")


class subprocess_result:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_disabled_by_default_reports_disabled_and_touches_nothing(tmp_path: Path) -> None:
    package_dir, media = _package_with_media(tmp_path)
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env={},
    )
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["status"] == "disabled"
    assert record["stamped_count"] == 0
    assert record["record_path"] is None
    assert not list(package_dir.rglob("*.c2pa"))


def test_enabled_without_tool_or_signer_is_unavailable_and_never_claims(
    tmp_path: Path,
) -> None:
    package_dir, media = _package_with_media(tmp_path)
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env={C2PA_ENABLED_ENV: "1"},
    )
    assert record["status"] == "unavailable"
    assert record["stamped_count"] == 0
    assert "c2pa_tool_bin_missing" in record["blockers"]
    assert "c2pa_sign_cert_file_missing" in record["blockers"]
    assert "c2pa_sign_key_file_missing" in record["blockers"]
    assert not list(package_dir.rglob("*.c2pa"))
    record_path = package_dir / record["record_path"]
    assert record_path.is_file()
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "unavailable"
    assert persisted["internal_ledger_authoritative"] is True


def test_assertion_is_digest_only_and_rejects_free_text_rights_content() -> None:
    assertion = build_ledger_ref_assertion(
        artifact_relative_path="exports/video_bundle/objects/sha256/aa/x.mp4",
        artifact_sha256="d" * 64,
        ledger_refs=_ledger_refs(),
    )
    assert assertion["schema_version"] == "blueprint.c2pa_ledger_ref.v1"
    assert assertion["internal_ledger_authoritative"] is True
    assert set(assertion) == {
        "schema_version",
        "artifact",
        "ledger",
        "scene_id",
        "capture_id",
        "verification_uri",
        "internal_ledger_authoritative",
    }
    refs = dict(_ledger_refs())
    refs["consent_scope"] = "filming permitted in the kitchen on weekdays"
    with pytest.raises(C2paAssertionContentError):
        build_ledger_ref_assertion(
            artifact_relative_path="a.mp4",
            artifact_sha256="d" * 64,
            ledger_refs=refs,
        )


def test_round_trip_stamps_sidecar_and_leaves_original_bytes_unchanged(
    tmp_path: Path,
) -> None:
    package_dir, media = _package_with_media(tmp_path)
    original = package_dir / media[0]
    before = _sha256(original)
    runner = _FakeRunner()
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env=_tool_config(tmp_path),
        runner=runner,
    )
    assert record["status"] == "stamped"
    assert record["stamped_count"] == 1
    assert _sha256(original) == before
    sidecar = original.with_suffix(".c2pa")
    assert sidecar.is_file()
    (file_record,) = record["files"]
    assert file_record["status"] == "stamped"
    assert file_record["sidecar_relative_path"] == str(sidecar.relative_to(package_dir))
    assert any("-o" in call for call in runner.calls)


def test_output_byte_mutation_is_fail_closed(tmp_path: Path) -> None:
    package_dir, media = _package_with_media(tmp_path)
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env=_tool_config(tmp_path),
        runner=_FakeRunner(mutate_output=True),
    )
    assert record["status"] == "failed"
    assert record["stamped_count"] == 0
    (file_record,) = record["files"]
    assert "c2pa_output_asset_bytes_changed" in file_record["blockers"]
    assert not list(package_dir.rglob("*.c2pa"))


def test_verification_failure_never_claims_stamped(tmp_path: Path) -> None:
    package_dir, media = _package_with_media(tmp_path)
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env=_tool_config(tmp_path),
        runner=_FakeRunner(verify_mentions_label=False),
    )
    assert record["status"] == "failed"
    (file_record,) = record["files"]
    assert "c2pa_verification_round_trip_failed" in file_record["blockers"]
    assert not list(package_dir.rglob("*.c2pa"))


def test_unsupported_container_is_recorded_not_stamped(tmp_path: Path) -> None:
    package_dir, media = _package_with_media(tmp_path)
    webm = package_dir / "exports/video_bundle/objects/sha256/aa" / ("b" * 64 + ".webm")
    webm.write_bytes(b"webm-bytes")
    media = media + [str(webm.relative_to(package_dir))]
    runner = _FakeRunner()
    record = apply_edge_stamping(
        package_dir=package_dir,
        media_relative_paths=media,
        ledger_refs=_ledger_refs(),
        env=_tool_config(tmp_path),
        runner=runner,
    )
    statuses = {row["relative_path"]: row["status"] for row in record["files"]}
    assert statuses[media[0]] == "stamped"
    assert statuses[media[1]] == "unsupported_format"
    assert record["stamped_count"] == 1
    assert not webm.with_suffix(".c2pa").exists()


def test_3d_sidecar_lane_is_fail_closed_pending_qualification(tmp_path: Path) -> None:
    asset = tmp_path / "scene.ply"
    asset.write_bytes(b"ply-bytes")
    result = stamp_3d_asset_sidecar(asset_path=asset, ledger_refs=_ledger_refs(), env={})
    assert result["status"] == "blocked_experimental_not_qualified"
    assert "c2pa_3d_format_binding_unqualified" in result["blockers"]
    forced = stamp_3d_asset_sidecar(
        asset_path=asset,
        ledger_refs=_ledger_refs(),
        env={"BLUEPRINT_PTDP_C2PA_3D_EXPERIMENTAL": "1"},
    )
    assert forced["status"] == "blocked_experimental_not_qualified"


def test_ptdp_hook_attaches_summary_and_record(tmp_path: Path) -> None:
    from blueprint_pipeline.post_training_data_package import _apply_c2pa_edge_stamping

    package_dir, media = _package_with_media(tmp_path)
    manifest = {"context": {"scene_id": "scene-1", "capture_id": "cap-1"}}

    _apply_c2pa_edge_stamping(package_dir, manifest, env={})
    assert manifest["c2pa_edge_stamping"]["status"] == "disabled"
    assert manifest["c2pa_edge_stamping"]["sidecar_only"] is True

    _apply_c2pa_edge_stamping(package_dir, manifest, env={C2PA_ENABLED_ENV: "1"})
    summary = manifest["c2pa_edge_stamping"]
    assert summary["status"] == "unavailable"
    assert summary["total_media_count"] == 1
    assert (package_dir / summary["record_path"]).is_file()


def test_buyer_readout_media_provenance_carries_c2pa_status() -> None:
    from blueprint_pipeline.buyer_package_readout import build_buyer_package_readout

    readout = build_buyer_package_readout(
        export_manifest={"c2pa_edge_stamping": {"status": "stamped", "stamped_count": 3}}
    )
    media = readout["sections"]["media_provenance"]
    assert media["c2pa_edge_stamping"]["status"] == "stamped"
    fallback = build_buyer_package_readout(export_manifest={})
    assert (
        fallback["sections"]["media_provenance"]["c2pa_edge_stamping"]["status"]
        == "not_attempted"
    )
