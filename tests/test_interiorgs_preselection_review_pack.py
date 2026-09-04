from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.interiorgs_preselection_review_pack import (
    InteriorGSPreselectionReviewPackError,
    build_interiorgs_preselection_review_pack,
)


def _box(
    ins_id: str,
    label: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> dict[str, object]:
    corners = []
    for x in (center[0] - size[0] / 2, center[0] + size[0] / 2):
        for y in (center[1] - size[1] / 2, center[1] + size[1] / 2):
            for z in (center[2] - size[2] / 2, center[2] + size[2] / 2):
                corners.append({"x": x, "y": y, "z": z})
    return {"ins_id": ins_id, "label": label, "bounding_box": corners}


def _sources(root: Path) -> tuple[Path, Path, Path]:
    splat = root / "scene.ply"
    labels = root / "labels.json"
    structure = root / "structure.json"
    splat.write_bytes(b"ply\nnot decoded by injected renderer\n")
    labels.write_text(
        json.dumps(
            [
                _box("115", "book", (1.0, 1.0, 0.81), (0.30, 0.40, 0.02)),
                _box("85", "tv cabinet", (1.0, 1.0, 0.4), (1.8, 1.6, 0.8)),
                _box("9", "wall", (0.05, 1.0, 1.0), (0.1, 2.0, 2.0)),
            ]
        ),
        encoding="utf-8",
    )
    structure.write_text(
        json.dumps(
            {
                "rooms": [{"profile": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
                "walls": [],
                "holes": [],
            }
        ),
        encoding="utf-8",
    )
    return splat, labels, structure


def _completed_renderer(observed: dict[str, object]):
    def render(source: Path, output: Path, **kwargs: object) -> dict[str, object]:
        observed.update({"source": source, "output": output, **kwargs})
        frames_dir = output / "frames"
        frames_dir.mkdir(parents=True)
        cameras = kwargs["camera_specs"]
        rendered = []
        for index, camera in enumerate(cameras):
            path = frames_dir / f"{camera['id']}.png"
            Image.new("RGB", (80, 60), color=(20 + index, 40, 60)).save(path)
            rendered.append(
                {
                    "id": camera["id"],
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "nonblank": True,
                    "digest": "sha256:untrusted-renderer-value",
                }
            )
        return {
            "schema_version": "splat_scene_render.v1",
            "status": "completed",
            "camera_set_digest": canonical_digest({"cameras": cameras}),
            "rendered_by": "reference_spark_renderer",
            "cameras": rendered,
        }

    return render


def test_one_command_pack_binds_sources_views_labels_and_boundaries(
    tmp_path: Path,
) -> None:
    splat, labels, structure = _sources(tmp_path)
    observed: dict[str, object] = {}
    output = tmp_path / "review"

    result = build_interiorgs_preselection_review_pack(
        scene_id="841757",
        splat_path=splat,
        labels_path=labels,
        structure_path=structure,
        output_root=output,
        approved_roots=[tmp_path],
        target_ins_ids=["115"],
        width=1280,
        height=960,
        renderer=_completed_renderer(observed),
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert result["status"] == "completed"
    assert result["scene_id"] == "841757"
    assert result["target_closeups"][0]["target_ins_id"] == "115"
    assert result["source_files"]["splat"]["sha256"] == (
        "sha256:" + hashlib.sha256(splat.read_bytes()).hexdigest()
    )
    assert result["camera_file"]["sha256"].startswith("sha256:")
    assert result["overlay"]["file"]["sha256"].startswith("sha256:")
    assert result["overlay"]["applied_to_render"] is False
    assert result["frames"]
    assert result["frames"][0]["sha256"] != "sha256:untrusted-renderer-value"
    assert Path(result["manifest_path"]).is_file()
    assert (output / "index.html").is_file()
    assert (output / "contact_sheet.png").is_file()
    assert result["advisory_rigid_movable_candidates"][0]["ins_id"] == "115"
    assert result["advisory_support_candidates"][0]["ins_id"] == "85"
    assert result["claim_boundary"] == {
        "selection_reconnaissance_only": True,
        "method_input": False,
        "evaluation_authorized": False,
        "appearance_fidelity_qualified": False,
        "collision_identity_established": False,
        "robot_reachability_established": False,
        "camera_motion_recovers_uncaptured_regions": False,
    }
    sealed = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert sealed["manifest_digest"] == canonical_digest(
        sealed, digest_field="manifest_digest"
    )
    assert observed["decimate"] == 0
    assert observed["camera_specs"] == result["cameras"]
    assert observed["browser_executable"] is None
    assert observed["overlay_json"] is None


def test_blocked_renderer_retains_typed_non_authoritative_manifest(
    tmp_path: Path,
) -> None:
    splat, labels, structure = _sources(tmp_path)
    result = build_interiorgs_preselection_review_pack(
        scene_id="841757",
        splat_path=splat,
        labels_path=labels,
        structure_path=structure,
        output_root=tmp_path / "blocked-review",
        approved_roots=[tmp_path],
        target_ins_ids=["115"],
        renderer=lambda *_args, **_kwargs: {
            "schema_version": "splat_scene_render.v1",
            "status": "blocked",
            "blockers": ["renderer_unavailable"],
            "cameras": [],
        },
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["interiorgs_preselection_render_incomplete"]
    assert Path(result["manifest_path"]).is_file()
    assert result["html_review"] is None


def test_pack_refuses_unknown_target_and_existing_output(tmp_path: Path) -> None:
    splat, labels, structure = _sources(tmp_path)
    with pytest.raises(ValueError, match="target_instance_not_exactly_one"):
        build_interiorgs_preselection_review_pack(
            scene_id="841757",
            splat_path=splat,
            labels_path=labels,
            structure_path=structure,
            output_root=tmp_path / "unknown",
            approved_roots=[tmp_path],
            target_ins_ids=["missing"],
            renderer=lambda *_args, **_kwargs: {},
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        InteriorGSPreselectionReviewPackError,
        match="interiorgs_preselection_output_exists",
    ):
        build_interiorgs_preselection_review_pack(
            scene_id="841757",
            splat_path=splat,
            labels_path=labels,
            structure_path=structure,
            output_root=existing,
            approved_roots=[tmp_path],
            target_ins_ids=["115"],
            renderer=lambda *_args, **_kwargs: {},
        )
