"""Actual source packet crosses the shared Vast admission boundary without IO."""
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import source_calibration_render_packet as packet
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
from tests.test_source_calibration_render_packet import valid_source_bundle_inputs


def _preflight(root, bundle):
    return _blueprint_bundle_preflight(
        job_dir=root, generated_at="2026-09-05T14:00:00Z",
        enable_blueprint_bundle=True, enable_isaac_smoke=False,
        provider_bundle_kind="adp_retained_scene_render", bundle_path=Path(bundle),
        provider_bundle_url="https://example.test/bundle",
        provider_output_put_url="https://example.test/output",
    )


def test_real_source_builder_crosses_shared_vast_preflight(tmp_path, monkeypatch):
    args, _ = valid_source_bundle_inputs(tmp_path, monkeypatch)
    bundle = packet.build_source_calibration_gpu_render_bundle(**args)
    result = _preflight(tmp_path / "preflight", bundle["bundle_path"])
    assert result["status"] == "passed", result
    assert result["missing_zip_entries"] == []
    assert result["zip_integrity_test_passed"] is True
    assert result["staging_url_verification_requested"] is False


@pytest.mark.parametrize("mutation", ["seal", "unknown_schema", "images", "target_support", "scene_without_target", "legacy"])
def test_real_source_preflight_rejects_broken_seal_and_missing_layers(tmp_path, monkeypatch, mutation):
    args, _ = valid_source_bundle_inputs(tmp_path, monkeypatch)
    bundle = packet.build_source_calibration_gpu_render_bundle(**args)
    manifest_name = "provider_runtime/adp_retained_scene_gpu_render_manifest.json"
    with zipfile.ZipFile(bundle["bundle_path"]) as original:
        members = {name: original.read(name) for name in original.namelist()}
    manifest = json.loads(members[manifest_name])
    if mutation == "seal":
        manifest["manifest_digest"] = "sha256:" + "0" * 64
    elif mutation == "unknown_schema":
        manifest["schema_version"] = "unrelated_bundle.v1"
        manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    elif mutation == "legacy":
        manifest["schema_version"] = "adp009d_retained_scene_gpu_render_bundle.v1"
        manifest["render_scope"] = "retained_scene"
        manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    else:
        del members[f"provider_runtime/input/{mutation}.ply"]
        # Removing a role and resealing must not weaken the fixed source contract.
        del manifest["layers"][mutation]
        manifest["inventory"] = [row for row in manifest["inventory"]
                                 if row["relative_path"] != f"input/{mutation}.ply"]
        manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    members[manifest_name] = canonical_json(manifest).encode()
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    result = _preflight(tmp_path / "changed-preflight", changed)
    assert result["status"] == "blocked", result
    if mutation == "legacy":
        assert "provider_runtime/input/shared_retained_scene.ply" in result["missing_zip_entries"]
        assert "provider_runtime/execution_authority.json" in result["missing_zip_entries"]
