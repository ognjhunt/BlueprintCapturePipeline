"""Preparing and returning exact source views preserves the original masks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import public_scene_inpainting_inputs as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_inpainting_preparation import validate_prepared_inputs
from tests.test_public_scene_inpainting_inputs import _write_v2_fixture, _fake_sealed_render


def _prepare(tmp_path):
    paths = _write_v2_fixture(tmp_path)
    request = json.loads((paths["repo"] / "request.json").read_text())
    original = request["camera_policy"]["views"]
    request["camera_policy"]["views"] = []
    for index in range(16):
        row = deepcopy(original[index % len(original)])
        row["camera_id"] = f"source-{index:02d}"
        row["position_offset_m"][0] += index * .001
        request["camera_policy"]["views"].append(row)
    request["rendering"]["graphics_backend"] = "egl"
    request.pop("request_digest")
    path = paths["data"] / "frozen-16-camera-request.json"
    path.write_text(json.dumps(module.build_public_scene_inpainting_input_request(request)))
    prepared = module.prepare_public_scene_inpainting_inputs(request_path=path,
        repo_root=paths["repo"], data_root=paths["data"], output_root=paths["output"])
    return paths, prepared


def _returned(prepared, root, monkeypatch):
    groups = {}
    for role, layer in prepared["layers"].items():
        directory = root / role
        manifest = _fake_sealed_render(output_dir=directory, cameras=prepared["cameras"])
        manifest["renders"] = [{"camera_id": row["camera_id"],
            "relative_path": f"frames/{row['camera_id']}.png",
            "digest": module._sha256(directory / "frames" / f"{row['camera_id']}.png")}
            for row in prepared["cameras"]]
        manifest["source_splat"] = {"digest": layer["sha256"],
                                    "retained_gaussian_count": layer["retained_gaussian_count"]}
        manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
            manifest, digest_field="sealed_camera_render_manifest_digest")
        path = directory / "sealed_camera_render_manifest.v1.json"
        path.write_text(json.dumps(manifest))
        groups[role] = {"root": directory, "manifest_path": path, "manifest": manifest}
    returned_path = root / "verified-test-return.json"
    returned_path.write_text(json.dumps({"test_fixture_only": True}))
    # The verifier itself is independently exercised by source-calibration
    # lifecycle tests; these tests protect the CPU continuation after it passes.
    monkeypatch.setitem(sys.modules, "blueprint_pipeline.source_calibration_render_return",
        SimpleNamespace(verify_source_calibration_return=lambda prepared_inputs, returned_group_path: groups))
    return returned_path, groups


def test_prepare_binds_three_actual_layers_and_16_clipped_cameras_without_render(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "render_splat_at_exact_cameras", lambda **kwargs: pytest.fail("prepare rendered"))
    paths, prepared = _prepare(tmp_path)
    assert prepared == validate_prepared_inputs(prepared["preparation_path"])
    assert prepared["rendered"] is False and prepared["candidate_policy_queried"] is False
    assert set(prepared["layers"]) == {"images", "target_support", "scene_without_target"}
    assert len(prepared["cameras"]) == 16
    assert all(row["intrinsics"]["near"] > 0 and row["intrinsics"]["far"] > row["intrinsics"]["near"]
               for row in prepared["cameras"])
    for layer in prepared["layers"].values():
        payload = Path(layer["path"]).read_bytes()
        assert layer["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert layer["size_bytes"] == len(payload)
    assert not list(paths["output"].rglob("*.png"))
    assert not (paths["output"] / "public_scene_interiorgs_edit_input_receipt.v2.json").exists()


def test_verified_return_produces_original_16_camera_mask_receipts(tmp_path, monkeypatch):
    paths, prepared = _prepare(tmp_path)
    returned, _groups = _returned(prepared, paths["data"] / "gpu-return", monkeypatch)
    finalized = module.finalize_public_scene_inpainting_inputs(
        preparation_path=prepared["preparation_path"], returned_group_path=returned)
    monkeypatch.setattr(module, "render_splat_at_exact_cameras", _fake_sealed_render)
    local = module.materialize_public_scene_inpainting_inputs(
        request_path=prepared["request_file"]["path"], repo_root=paths["repo"], data_root=paths["data"],
        output_root=paths["data"] / "local-continuation")
    assert finalized["scene"] == local["scene"]
    assert finalized["derived_artifacts"]["masks"] == local["derived_artifacts"]["masks"]
    assert finalized["derived_artifacts"]["images"] == local["derived_artifacts"]["images"]
    assert finalized["source_calibration_render"]["preparation_digest"] == prepared["preparation_digest"]
    assert len(list(paths["output"].glob("*/frames/*.png"))) == 48


@pytest.mark.parametrize("defect", ["missing_group", "changed_frame", "changed_manifest"])
def test_incomplete_or_changed_return_cannot_emit_receipt(tmp_path, monkeypatch, defect):
    paths, prepared = _prepare(tmp_path)
    returned, groups = _returned(prepared, paths["data"] / "gpu-return", monkeypatch)
    if defect == "missing_group":
        groups.pop("target_support")
    elif defect == "changed_frame":
        (groups["images"]["root"] / "frames" / "source-00.png").write_bytes(b"corrupt")
    else:
        groups["images"]["manifest_path"].write_text("{}")
    with pytest.raises(module.PublicSceneInpaintingInputError, match="edit_input_returned"):
        module.finalize_public_scene_inpainting_inputs(
            preparation_path=prepared["preparation_path"], returned_group_path=returned)
    assert not (paths["output"] / "public_scene_interiorgs_edit_input_receipt.v2.json").exists()


@pytest.mark.parametrize("defect", ["source_bytes", "count", "camera_binding", "render_options"])
def test_prepared_source_camera_count_and_render_binding_are_reopened(tmp_path, defect):
    _paths, prepared = _prepare(tmp_path)
    if defect == "source_bytes":
        path = Path(prepared["layers"]["images"]["path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif defect == "count":
        prepared["layers"]["images"]["retained_gaussian_count"] -= 1
    elif defect == "render_options":
        prepared["render_options"]["width"] = 256
    else:
        prepared["context"]["cameras"][0]["intrinsics"]["fx"] += 1
    if defect != "source_bytes":
        prepared["preparation_digest"] = canonical_digest(prepared, digest_field="preparation_digest")
        Path(prepared["preparation_path"]).write_text(json.dumps(prepared))
    with pytest.raises(module.PublicSceneInpaintingInputError, match="edit_input_preparation"):
        validate_prepared_inputs(prepared["preparation_path"])
