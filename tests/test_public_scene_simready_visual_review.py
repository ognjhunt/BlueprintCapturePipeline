from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_visual_review import (
    SimReadyVisualReviewError,
    materialize_native_visual_review,
    materialize_visual_review,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    from pxr import Gf, Usd, UsdGeom

    evidence = tmp_path / "evidence"
    frames = evidence / "frames"
    output = evidence / "review"
    frames.mkdir(parents=True)
    stage_path = evidence / "replacement.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/BlueprintReplacement")
    UsdGeom.Scope.Define(stage, "/World/BlueprintReplacement/visuals")
    mesh = UsdGeom.Mesh.Define(stage, "/World/BlueprintReplacement/visuals/body")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-0.2, -0.2, 1.0), Gf.Vec3f(0.2, -0.2, 1.0), Gf.Vec3f(0.0, 0.2, 1.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.8, 0.4)])
    stage.GetRootLayer().Save()

    frame = np.full((100, 120, 3), 230, dtype=np.uint8)
    frame_path = frames / "view.png"
    cv2.imwrite(str(frame_path), frame)
    replacement = {
        "status": "composed_static_candidate",
        "composition": {
            "relative_path": "replacement.usda",
            "sha256": _sha(stage_path),
            "size_bytes": stage_path.stat().st_size,
        },
    }
    replacement["receipt_digest"] = canonical_digest(
        replacement, digest_field="receipt_digest"
    )
    replacement_path = evidence / "replacement.json"
    _write(replacement_path, replacement)
    exact = {
        "sealed_camera_render_manifest_digest": "sha256:exact",
        "renders": [
            {
                "camera_id": "view",
                "relative_path": "view.png",
                "digest": _sha(frame_path),
            }
        ],
    }
    exact_path = evidence / "exact.json"
    _write(exact_path, exact)
    cameras = [
        {
            "camera_id": "view",
            "intrinsics": {
                "width": 120,
                "height": 100,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 60.0,
                "cy": 50.0,
            },
            "T_world_camera_opencv": np.eye(4).tolist(),
        }
    ]
    cameras_path = evidence / "cameras.json"
    _write(cameras_path, cameras)
    return {
        "evidence": evidence,
        "frames": frames,
        "frame": frame_path,
        "replacement": replacement_path,
        "exact": exact_path,
        "cameras": cameras_path,
        "output": output,
    }


def _run(paths: dict[str, Path]) -> dict[str, object]:
    return materialize_visual_review(
        replacement_receipt_path=paths["replacement"],
        exact_camera_manifest_path=paths["exact"],
        cameras_path=paths["cameras"],
        frame_root=paths["frames"],
        evidence_root=paths["evidence"],
        output_root=paths["output"],
    )


def test_visual_review_renders_digest_bound_mesh_into_exact_camera(tmp_path: Path) -> None:
    receipt = _run(_fixture(tmp_path))

    assert receipt["status"] == "rendered_visual_review_candidate"
    assert receipt["renderer_is_native_ovrtx"] is False
    assert receipt["dynamic_contact_proven"] is False
    assert receipt["artifacts"][0]["visible_pixel_count"] > 0
    assert (tmp_path / "evidence/review/view.before_after.png").is_file()


def test_visual_review_rejects_changed_sealed_frame(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["frame"].write_bytes(b"changed")

    with pytest.raises(SimReadyVisualReviewError, match="sealed_frame_digest_mismatch:view"):
        _run(paths)


def _native_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _fixture(tmp_path)
    native_root = paths["evidence"] / "native_run"
    provider_root = native_root / "native_ovrtx" / "view"
    physics_root = native_root / "native_ovphysx"
    provider_root.mkdir(parents=True)
    physics_root.mkdir(parents=True)
    rgb = np.zeros((100, 120, 4), dtype=np.uint8)
    rgb[35:75, 52:68, :3] = (120, 220, 190)
    rgb[:, :, 3] = 255
    depth = np.zeros((100, 120, 1), dtype=np.float32)
    depth[35:75, 52:68, 0] = 1.2
    np.save(provider_root / "rgb.npy", rgb, allow_pickle=False)
    np.save(provider_root / "depth.npy", depth, allow_pickle=False)
    report = {"checks": [{"name": "render", "status": "passed"}]}
    _write(provider_root / "ovrtx_result.json", report)
    physics = {"checks": [{"name": "settled", "status": "passed"}]}
    _write(physics_root / "ovphysx_result.json", physics)

    def record(path: Path) -> dict[str, object]:
        return {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
        }

    provider = {
        "status": "completed",
        "blockers": [],
        "native_ovrtx_exact_camera_executed": True,
        "native_ovphysx_drop_contact_settle_executed": True,
        "material_agent_executed": True,
        "texture_agent_executed": True,
        "physics_agent_executed": True,
        "validation_agent_executed": True,
        "native_probes": {
            "ovrtx": {
                "camera_count": 1,
                "expected_camera_count": 1,
                "renders": [
                    {
                        "camera_id": "view",
                        "executed": True,
                        "outputs": [
                            record(provider_root / "rgb.npy"),
                            record(provider_root / "depth.npy"),
                            record(provider_root / "ovrtx_result.json"),
                        ],
                        "report_sha256": _sha(provider_root / "ovrtx_result.json"),
                    }
                ],
            },
            "ovphysx": {
                "outputs": [record(physics_root / "ovphysx_result.json")],
                "report_sha256": _sha(physics_root / "ovphysx_result.json"),
            },
        },
    }
    provider_path = native_root / "adp_content_agents_vast_result.json"
    _write(provider_path, provider)
    exact = json.loads(paths["exact"].read_text(encoding="utf-8"))
    exact["renders"][0].update({"width": 120, "height": 100})
    _write(paths["exact"], exact)
    paths.update(
        {
            "native_root": native_root,
            "provider": provider_path,
            "rgb": provider_root / "rgb.npy",
            "native_output": paths["evidence"] / "native_review",
        }
    )
    return paths


def _run_native(paths: dict[str, Path]) -> dict[str, object]:
    return materialize_native_visual_review(
        provider_result_path=paths["provider"],
        exact_camera_manifest_path=paths["exact"],
        frame_root=paths["frames"],
        evidence_root=paths["evidence"],
        output_root=paths["native_output"],
    )


def test_native_visual_review_binds_ovrtx_and_ovphysx(tmp_path: Path) -> None:
    receipt = _run_native(_native_fixture(tmp_path))

    assert receipt["status"] == "rendered_native_visual_review_candidate"
    assert receipt["renderer_is_native_ovrtx"] is True
    assert receipt["background_rendered_by_native_ovrtx"] is False
    assert receipt["native_ovphysx_drop_contact_settle_proven"] is True
    assert receipt["artifacts"][0]["visible_pixel_count"] == 640
    assert (tmp_path / "evidence/native_review/view.before_after.png").is_file()


def test_native_visual_review_rejects_changed_ovrtx_array(tmp_path: Path) -> None:
    paths = _native_fixture(tmp_path)
    paths["rgb"].write_bytes(b"changed")

    with pytest.raises(
        SimReadyVisualReviewError,
        match="native_camera_output_digest_mismatch:view:rgb.npy",
    ):
        _run_native(paths)


def test_native_visual_review_rejects_caller_asserted_success(tmp_path: Path) -> None:
    paths = _native_fixture(tmp_path)
    provider = json.loads(paths["provider"].read_text(encoding="utf-8"))
    provider["native_probes"]["ovphysx"]["outputs"] = []
    _write(paths["provider"], provider)

    with pytest.raises(SimReadyVisualReviewError, match="native_ovphysx_report_missing"):
        _run_native(paths)
