from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
from pxr import Usd, UsdGeom

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.digest_bound_review_contact_sheet import (
    materialize_digest_bound_review_contact_sheets,
)
from blueprint_pipeline.public_scene_agent_cad_replacement_visual_review import (
    AgentCadReplacementVisualReviewError,
    materialize_agent_cad_replacement_visual_review,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict:
    value = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if relative_to is None:
        value["path"] = str(path.resolve())
    else:
        value["relative_path"] = path.relative_to(relative_to).as_posix()
    return value


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_png(path: Path, array: np.ndarray, mode: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)
    return path


def _fake_renderer(root: Path, *, alpha_kind: str = "valid") -> Path:
    renderer = root / f"fake_usdrecord_{alpha_kind}.py"
    renderer.write_text(
        """#!/usr/bin/env python3
import os
import sys

if '--version' in sys.argv:
    print('fixture usdrecord 1.0')
    raise SystemExit(0)
os.execv(%r, [%r, %r, %r, sys.argv[-1], sys.argv[sys.argv.index('--imageWidth') + 1], %r])
"""
        % (
            sys.executable,
            sys.executable,
            str(Path(__file__).resolve()),
            "--fake-render",
            alpha_kind,
        ),
        encoding="utf-8",
    )
    renderer.chmod(0o755)
    return renderer


def _run_fake_renderer() -> None:
    output = Path(__import__("sys").argv[2])
    width = int(__import__("sys").argv[3])
    alpha_kind = __import__("sys").argv[4]
    height = width * 3 // 4
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if alpha_kind == "valid":
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                image.putpixel((x, y), (180, 100, 40, 255))
    elif alpha_kind == "full":
        image = Image.new("RGBA", (width, height), (180, 100, 40, 255))
    image.save(output)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--fake-render":
    _run_fake_renderer()
    raise SystemExit(0)


def _task_fixture(
    root: Path, *, task_id: str, scene_id: str = "fixture_scene"
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True)
    scene = root / "scene"
    scene.mkdir()
    width, height = 8, 6
    cameras = ["front", "right"]
    trajectory_frames = []
    dual_frames = []
    raw_frames = []
    for index, camera_id in enumerate(cameras):
        background_array = np.full((height, width, 3), 20 + index * 10, np.uint8)
        background = _write_png(root / "backgrounds" / f"{index:05d}.png", background_array, "RGB")
        mask_array = np.zeros((height, width), np.uint8)
        mask_array[2:4, 3:5] = 255
        mask = _write_png(root / "masks" / f"{index:05d}.png", mask_array, "L")
        transform = np.eye(4, dtype=float)
        transform[0, 3] = float(index)
        trajectory_frames.append(
            {
                "physical_camera_index": index,
                "camera_id": camera_id,
                "camera_model": "OPENCV",
                "w": width,
                "h": height,
                "fl_x": 5.0,
                "fl_y": 5.0,
                "cx": width / 2,
                "cy": height / 2,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "transform_matrix": transform.tolist(),
            }
        )
        dual_frames.append(
            {
                "physical_camera_index": index,
                "camera_id": camera_id,
                "source_exact_repair_mask": _record(mask),
            }
        )
        raw_frames.append(
            {
                "frame_index": index,
                "camera_id": camera_id,
                **_record(background),
            }
        )
    trajectory = {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        "fl_x": 5.0,
        "fl_y": 5.0,
        "cx": width / 2,
        "cy": height / 2,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": trajectory_frames,
    }
    trajectory_path = _write_json(scene / "review_transforms.json", trajectory)
    dual = {
        "schema_version": "public_scene_artifixer3d_dual_target_inputs.v1",
        "status": "paired_target_inputs_prepared_no_model_no_execution",
        "publisher_scene_id": scene_id,
        "pipeline_mode": "dual_target_artifixer3d_only",
        "tasks": [
            {
                "task_id": task_id,
                "scene_directory": str(scene.resolve()),
                "physical_camera_count": len(cameras),
                "review_trajectory": _record(trajectory_path, relative_to=scene),
                "frames": dual_frames,
            }
        ],
        "receipt_digest": "",
    }
    dual["receipt_digest"] = canonical_digest(dual, digest_field="receipt_digest")
    dual_path = _write_json(root / "dual.json", dual)
    final = {
        "schema_version": "public_scene_artifixer3d_final_composite.v1",
        "status": "final_composite_materialized_pending_human_multiview_review",
        "publisher_scene_id": scene_id,
        "replacement_object_count": 1,
        "outside_support_changed_pixels_total": 0,
        "outside_support_invariance_proven": True,
        "appearance_repair_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "tasks": [
            {
                "task_id": task_id,
                "physical_camera_count": len(cameras),
                "outside_support_changed_pixels_total": 0,
                "outside_support_invariance_proven": True,
                "frames": [{**row, "outside_support_changed_pixels": 0} for row in raw_frames],
            }
        ],
        "receipt_digest": "",
    }
    final["receipt_digest"] = canonical_digest(final, digest_field="receipt_digest")
    final_path = _write_json(root / "final.json", final)

    candidate = root / "replacement.usda"
    stage = Usd.Stage.CreateNew(str(candidate))
    UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
    stage.GetRootLayer().Save()
    claim = {
        "agent_authored_step_visual_geometry": True,
        "appearance_materially_qualified": False,
        "collision_geometry_remains_graph_candidate": True,
        "deterministic_geometry_generator_used": False,
        "joint_physics_behavior_qualified": False,
        "native_simulator_import_qualified": False,
        "physical_equivalence_proven": False,
    }
    claim.update(
        {
            "agent_authored_display_colors_preserved": True,
            "generated_texture_maps_present": False,
        }
    )
    composition = {
        "schema_version": "simready_agent_cad_visual_composition.v2",
        "status": "agent_cad_visuals_composed",
        "scene_id": scene_id,
        "task_id": task_id,
        "asset_id": f"{task_id}_asset",
        "output_usd": _record(candidate),
        "visual_mesh_count": 1,
        "visual_meshes": [{"fixture": True}],
        "agent_authored_display_color_mesh_count": 1,
        "neutral_fallback_mesh_count": 0,
        "generated_texture_map_count": 0,
        "collision_visual_isolation_verified": True,
        "claim_boundary": claim,
        "receipt_digest": "",
    }
    composition["receipt_digest"] = canonical_digest(composition, digest_field="receipt_digest")
    composition_path = _write_json(root / "composition.json", composition)
    return dual_path, final_path, composition_path


def _admit_fixture_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_agent_cad_replacement_visual_review."
        "validate_agent_cad_visual_composition",
        lambda value, verify_files=True: dict(value),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_agent_cad_replacement_visual_review."
        "validate_registered_replacement_asset",
        lambda value, verify_files=True: dict(value),
    )


def _registered_fixture(composition_path: Path) -> Path:
    composition = json.loads(composition_path.read_text())
    source = Path(composition["output_usd"]["path"])
    registered_usd = composition_path.parent / "registered.usda"
    registered_usd.write_bytes(source.read_bytes())
    registration = composition_path.parent / "frame_registration.json"
    registration.write_text("{}\n", encoding="utf-8")
    value = {
        "schema_version": "registered_replacement_asset.v1",
        "status": "registered_replacement_materialized_pending_native_import",
        "scene_id": composition["scene_id"],
        "task_id": composition["task_id"],
        "asset_id": composition["asset_id"],
        "visual_composition_receipt": {
            **_record(composition_path),
            "receipt_digest": composition["receipt_digest"],
        },
        "frame_registration": {
            **_record(registration),
            "registration_digest": "sha256:" + "a" * 64,
        },
        "output_usd": _record(registered_usd),
        "T_observed_world_axes_from_asset_local_axes": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "source_root_translation_preserved": [0.0, 0.0, 0.0],
        "agent_authored_display_colors_preserved": True,
        "generated_texture_maps_present": False,
        "neutral_fallback_present": False,
        "native_import_qualified": False,
        "deterministic_pose_composition_only": True,
        "geometry_generated_or_modified": False,
        "physical_equivalence_proven": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return _write_json(composition_path.parent / "registered.json", value)


def test_materializes_lossless_two_task_review_and_contact_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_composition(monkeypatch)
    a = _task_fixture(tmp_path / "a", task_id="task_a")
    b = _task_fixture(tmp_path / "b", task_id="task_b")
    result = materialize_agent_cad_replacement_visual_review(
        dual_input_receipt_paths=[a[0], b[0]],
        final_composite_receipt_paths=[a[1], b[1]],
        visual_composition_receipt_paths=[a[2], b[2]],
        output_root=tmp_path / "review",
        renderer_executable=_fake_renderer(tmp_path),
        renderer_plugin="Fixture",
    )
    assert result["replacement_object_count"] == 2
    assert result["outside_replacement_alpha_changed_pixels_total"] == 0
    assert result["deterministic_geometry_generator_used"] is False
    assert result["appearance_repair_qualified"] is False
    assert result["native_simulator_import_qualified"] is False
    assert result["generated_output_is_capture_or_physical_evidence"] is False
    for task in result["tasks"]:
        for frame in task["frames"]:
            with Image.open(frame["background"]["path"]) as opened:
                background = np.asarray(opened.convert("RGB"))
            with Image.open(frame["path"]) as opened:
                combined = np.asarray(opened.convert("RGB"))
            layer_path = (
                tmp_path / "review" / task["task_id"] / frame["replacement_layer"]["relative_path"]
            )
            with Image.open(layer_path) as opened:
                alpha = np.asarray(opened.convert("RGBA"))[:, :, 3]
            assert np.array_equal(combined[alpha == 0], background[alpha == 0])
            wrapper_path = (
                tmp_path
                / "review"
                / task["task_id"]
                / frame["camera_reference_stage"]["relative_path"]
            )
            text = wrapper_path.read_text(encoding="utf-8")
            assert "references = @" in text
            assert 'def Camera "ReviewCamera"' in text
            assert "def Mesh" not in text

    sheets = materialize_digest_bound_review_contact_sheets(
        raw_result_path=result["receipt_path"], output_root=tmp_path / "sheets"
    )
    assert sheets["task_count"] == 2
    assert sheets["all_sheet_crops_pixel_identical"] is True


def test_review_renders_exact_registered_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_composition(monkeypatch)
    fixture = _task_fixture(tmp_path / "registered", task_id="task_registered")
    registered = _registered_fixture(fixture[2])
    result = materialize_agent_cad_replacement_visual_review(
        dual_input_receipt_paths=[fixture[0]],
        final_composite_receipt_paths=[fixture[1]],
        visual_composition_receipt_paths=[fixture[2]],
        registered_replacement_receipt_paths=[registered],
        output_root=tmp_path / "review-registered",
        renderer_executable=_fake_renderer(tmp_path),
        renderer_plugin="Fixture",
    )

    assert result["asset_frame_registration_applied_to_all_tasks"] is True
    assert result["tasks"][0]["asset_frame_registration_applied"] is True
    assert (
        result["inputs"][0]["registered_replacement_asset"]["registration_digest"]
        == "sha256:" + "a" * 64
    )


def test_accepts_reusable_five_task_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_composition(monkeypatch)
    fixtures = [
        _task_fixture(tmp_path / f"task_{index}", task_id=f"task_{index}") for index in range(5)
    ]
    result = materialize_agent_cad_replacement_visual_review(
        dual_input_receipt_paths=[row[0] for row in fixtures],
        final_composite_receipt_paths=[row[1] for row in fixtures],
        visual_composition_receipt_paths=[row[2] for row in fixtures],
        output_root=tmp_path / "review",
        renderer_executable=_fake_renderer(tmp_path),
        renderer_plugin="Fixture",
    )
    assert result["replacement_object_count"] == 5
    assert len(result["tasks"]) == 5


def test_selects_exact_task_from_shared_final_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_composition(monkeypatch)
    a = _task_fixture(tmp_path / "a", task_id="task_a")
    b = _task_fixture(tmp_path / "b", task_id="task_b")
    shared = json.loads(a[1].read_text(encoding="utf-8"))
    other = json.loads(b[1].read_text(encoding="utf-8"))
    shared["replacement_object_count"] = 2
    shared["tasks"].extend(other["tasks"])
    shared["receipt_digest"] = canonical_digest(shared, digest_field="receipt_digest")
    shared_path = _write_json(tmp_path / "shared_final.json", shared)

    result = materialize_agent_cad_replacement_visual_review(
        dual_input_receipt_paths=[a[0]],
        final_composite_receipt_paths=[shared_path],
        visual_composition_receipt_paths=[a[2]],
        output_root=tmp_path / "review",
        renderer_executable=_fake_renderer(tmp_path),
        renderer_plugin="Fixture",
    )

    assert [row["task_id"] for row in result["tasks"]] == ["task_a"]


@pytest.mark.parametrize("alpha_kind", ["empty", "full"])
def test_rejects_invalid_replacement_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alpha_kind: str
) -> None:
    _admit_fixture_composition(monkeypatch)
    fixture = _task_fixture(tmp_path / "task", task_id="task_a")
    with pytest.raises(
        AgentCadReplacementVisualReviewError,
        match="replacement_visual_layer_alpha_invalid",
    ):
        materialize_agent_cad_replacement_visual_review(
            dual_input_receipt_paths=[fixture[0]],
            final_composite_receipt_paths=[fixture[1]],
            visual_composition_receipt_paths=[fixture[2]],
            output_root=tmp_path / "review",
            renderer_executable=_fake_renderer(tmp_path, alpha_kind=alpha_kind),
            renderer_plugin="Fixture",
        )


def test_rejects_self_digest_valid_camera_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _admit_fixture_composition(monkeypatch)
    fixture = _task_fixture(tmp_path / "task", task_id="task_a")
    dual = json.loads(fixture[0].read_text(encoding="utf-8"))
    dual["tasks"][0]["frames"][0]["camera_id"] = "wrong"
    dual["receipt_digest"] = canonical_digest(dual, digest_field="receipt_digest")
    _write_json(fixture[0], dual)
    with pytest.raises(
        AgentCadReplacementVisualReviewError,
        match="replacement_visual_camera_binding_invalid",
    ):
        materialize_agent_cad_replacement_visual_review(
            dual_input_receipt_paths=[fixture[0]],
            final_composite_receipt_paths=[fixture[1]],
            visual_composition_receipt_paths=[fixture[2]],
            output_root=tmp_path / "review",
            renderer_executable=_fake_renderer(tmp_path),
            renderer_plugin="Fixture",
        )


def test_rejects_candidate_usd_byte_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _admit_fixture_composition(monkeypatch)
    fixture = _task_fixture(tmp_path / "task", task_id="task_a")
    composition = json.loads(fixture[2].read_text(encoding="utf-8"))
    Path(composition["output_usd"]["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(
        AgentCadReplacementVisualReviewError,
        match="replacement_visual_candidate_usd_invalid",
    ):
        materialize_agent_cad_replacement_visual_review(
            dual_input_receipt_paths=[fixture[0]],
            final_composite_receipt_paths=[fixture[1]],
            visual_composition_receipt_paths=[fixture[2]],
            output_root=tmp_path / "review",
            renderer_executable=_fake_renderer(tmp_path),
            renderer_plugin="Fixture",
        )
