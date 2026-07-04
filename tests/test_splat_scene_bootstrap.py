"""Hermetic tests for the PLY-only sidecar bootstrap (no Gemini, no chromium).

The end-to-end test builds a synthetic room splat (floor + two distinct object
blobs), injects a fake renderer (no-op) and a GEOMETRY-DRIVEN fake detector that
projects each blob into the camera with the same math the unprojector inverts —
so a passing test proves the whole chain (views -> detections -> splat depth ->
unproject -> fuse -> labels.json) round-trips to the true object positions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.scene_placement.perception_index import (
    camera_basis,
    resolve_extrinsics,
    resolve_intrinsics,
)
from blueprint_pipeline.splat_scene_bootstrap import (
    bootstrap_scene_sidecars,
    detections_from_gemini_boxes,
    estimate_floor_z_from_points,
    plan_interior_views,
    task_targets_payload_from_objects,
)


pytestmark = pytest.mark.slow


# ----------------------------- synthetic scene -----------------------------

# Two objects with known world AABBs inside a 6x6 room.
_OBJECTS = {
    "pot": ((2.0, 2.0, 0.7), (2.3, 2.3, 0.95)),
    "fridge": ((4.0, 0.5, 0.0), (4.7, 1.2, 1.8)),
}


def _synthetic_room_splat() -> SplatData:
    rng_pts = []
    # floor
    for x in np.arange(0.0, 6.0, 0.12):
        for y in np.arange(0.0, 6.0, 0.12):
            rng_pts.append((x, y, 0.0))
    # object blobs (dense surfaces)
    for bmin, bmax in _OBJECTS.values():
        xs = np.arange(bmin[0], bmax[0] + 1e-6, 0.05)
        ys = np.arange(bmin[1], bmax[1] + 1e-6, 0.05)
        zs = np.arange(bmin[2], bmax[2] + 1e-6, 0.05)
        for x in xs:
            for y in ys:
                for z in zs:
                    rng_pts.append((x, y, z))
    pts = np.array(rng_pts, dtype=np.float32)
    n = pts.shape[0]
    return SplatData(
        count=n,
        xyz=pts,
        opacity=np.full(n, 5.0, dtype=np.float32),
        f_dc=np.zeros((n, 3), dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        quats=np.zeros((n, 4), dtype=np.float32),
        properties=(),
    )


def _project_box_to_bbox_px(bmin, bmax, camera):
    """Project a world AABB's corners into the camera; None if behind/outside."""
    fx, fy, cx, cy = resolve_intrinsics(camera)
    eye, target, up = resolve_extrinsics(camera)
    right, up_cam, forward = camera_basis(eye, target, up)
    w, h = int(camera["width"]), int(camera["height"])
    xs, ys = [], []
    for x in (bmin[0], bmax[0]):
        for y in (bmin[1], bmax[1]):
            for z in (bmin[2], bmax[2]):
                rel = (x - eye[0], y - eye[1], z - eye[2])
                zc = sum(r * f for r, f in zip(rel, forward))
                if zc <= 0.1:
                    return None
                xc = sum(r * f for r, f in zip(rel, right))
                yc = sum(r * f for r, f in zip(rel, up_cam))
                xs.append(xc / zc * fx + cx)
                ys.append(cy - yc / zc * fy)
    x0, x1 = max(0.0, min(xs)), min(float(w), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(h), max(ys))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return (x0, y0, x1, y1)


def _geometry_detector(png_path: Path, camera) -> list[dict]:
    dets = []
    for label, (bmin, bmax) in _OBJECTS.items():
        bbox = _project_box_to_bbox_px(bmin, bmax, camera)
        if bbox is not None:
            dets.append({"label": label, "bbox_px": bbox, "confidence": 0.9})
    return dets


def _fake_renderer(splat_path, cameras, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rendered = {}
    for cam in cameras:
        p = Path(out_dir) / f"{cam['id']}.png"
        p.write_bytes(b"\x89PNG fake")
        rendered[str(cam["id"])] = str(p)
    return {"status": "completed", "rendered": rendered}


# ----------------------------- unit tests -----------------------------

class TestFloorEstimate:
    def test_mode_beats_underfloor_fuzz(self):
        z = np.concatenate([
            np.random.default_rng(7).uniform(-0.6, -0.3, 40),  # fuzz
            np.zeros(500),                                      # floor pile
            np.full(300, 0.9),                                  # furniture tops
        ])
        assert abs(estimate_floor_z_from_points(z)) <= 0.05


class TestPlanInteriorViews:
    def test_star_and_pullbacks(self):
        splat = _synthetic_room_splat()
        cams = plan_interior_views(splat, floor_z=0.0, n_azimuths=8, n_pullbacks=4)
        assert len(cams) == 12
        star = [c for c in cams if str(c["id"]).startswith("star_")]
        # Star cameras all share the interior eye point.
        eyes = {tuple(round(v, 3) for v in c["eye"]) for c in star}
        assert len(eyes) == 1
        ex, ey, ez = next(iter(eyes))
        assert 1.0 < ex < 5.0 and 1.0 < ey < 5.0  # inside the room
        assert ez == pytest.approx(1.4, abs=0.01)

    def test_empty_splat_raises(self):
        empty = SplatData(
            count=0, xyz=np.zeros((0, 3), np.float32), opacity=np.zeros(0, np.float32),
            f_dc=np.zeros((0, 3), np.float32), scales=np.zeros((0, 3), np.float32),
            quats=np.zeros((0, 4), np.float32), properties=(),
        )
        with pytest.raises(ValueError):
            plan_interior_views(empty, floor_z=0.0)


class TestGeminiBoxParsing:
    def test_box_2d_convention(self):
        dets = detections_from_gemini_boxes(
            [{"label": "Pot", "box_2d": [100, 200, 300, 400], "confidence": 0.7}],
            width=1000, height=500,
        )
        assert len(dets) == 1
        d = dets[0]
        assert d["label"] == "pot"
        # box_2d is [ymin, xmin, ymax, xmax] normalized 0-1000.
        assert d["bbox_px"] == (200.0, 50.0, 400.0, 150.0)

    def test_garbage_records_skipped(self):
        dets = detections_from_gemini_boxes(
            [
                {"label": "", "box_2d": [0, 0, 500, 500]},
                {"label": "x", "box_2d": [1, 2, 3]},
                {"label": "y", "box_2d": ["a", "b", "c", "d"]},
                "not a dict",
            ],
            width=100, height=100,
        )
        assert dets == []


class TestSidecarPayloads:
    def test_task_synthesis_pickable_vs_openable(self):
        from blueprint_pipeline.scene_placement import SceneObject

        pot = SceneObject(id="1", label="pot", bbox_min=(0, 0, 0.7), bbox_max=(0.2, 0.2, 0.9),
                          centroid=(0.1, 0.1, 0.8), source="perception")
        fridge = SceneObject(id="2", label="fridge", bbox_min=(2, 2, 0), bbox_max=(2.7, 2.7, 1.8),
                             centroid=(2.35, 2.35, 0.9), source="perception")
        chandelier = SceneObject(id="3", label="chandelier", bbox_min=(1, 1, 2.2),
                                 bbox_max=(1.4, 1.4, 2.6), centroid=(1.2, 1.2, 2.4),
                                 source="perception")
        payload = task_targets_payload_from_objects([pot, fridge, chandelier], floor_z=0.0)
        ids = [t["task_id"] for t in payload["tasks"]]
        assert "Pick up pot_1 and place it in the target zone" in ids
        assert "Open and close fridge_2" in ids
        # Ceiling fixture: neither pickable (too high) nor openable.
        assert not any("chandelier" in t for t in ids)
        assert payload["bootstrap_generated"] is True


# ----------------------------- end-to-end -----------------------------

class TestBootstrapEndToEnd:
    def test_round_trips_object_positions(self, tmp_path):
        splat = _synthetic_room_splat()
        ply = write_standard_3dgs_ply(splat, tmp_path / "scene.ply")
        report = bootstrap_scene_sidecars(
            ply, tmp_path / "boot",
            detector=_geometry_detector,
            renderer=_fake_renderer,
            n_azimuths=8, n_pullbacks=4,
            min_views=2,
        )
        assert report["status"] == "completed", report
        labels = json.loads(Path(report["labels_path"]).read_text())
        by_label = {}
        for entry in labels:
            by_label.setdefault(entry["label"], []).append(entry)
        assert set(by_label) >= {"pot", "fridge"}
        # Each generated box must land near the true object.
        for label, (bmin, bmax) in _OBJECTS.items():
            true_c = [0.5 * (bmin[i] + bmax[i]) for i in range(3)]
            best = None
            for entry in by_label[label]:
                xs = [c["x"] for c in entry["bounding_box"]]
                ys = [c["y"] for c in entry["bounding_box"]]
                zs = [c["z"] for c in entry["bounding_box"]]
                gen_c = (
                    0.5 * (min(xs) + max(xs)),
                    0.5 * (min(ys) + max(ys)),
                    0.5 * (min(zs) + max(zs)),
                )
                err = math.dist(gen_c[:2], true_c[:2])
                best = err if best is None else min(best, err)
            assert best is not None and best < 0.5, (label, best)
        # Task file exists and references generated instances.
        tasks = json.loads(Path(report["task_targets_path"]).read_text())
        assert tasks["bootstrap_generated"] is True
        assert any("fridge" in t["task_id"] for t in tasks["tasks"])

    def test_missing_splat_blocks(self, tmp_path):
        report = bootstrap_scene_sidecars(
            tmp_path / "nope.ply", tmp_path / "boot",
            detector=_geometry_detector, renderer=_fake_renderer,
        )
        assert report["status"] == "blocked"
        assert "splat_source_missing" in report["blockers"]

    def test_no_detections_blocks(self, tmp_path):
        splat = _synthetic_room_splat()
        ply = write_standard_3dgs_ply(splat, tmp_path / "scene.ply")
        report = bootstrap_scene_sidecars(
            ply, tmp_path / "boot",
            detector=lambda p, c: [],
            renderer=_fake_renderer,
        )
        assert report["status"] == "blocked"
        assert "no_fused_objects" in report["blockers"] or "no_views_with_detections" in report["blockers"]


class TestPreflightBootstrapWiring:
    def test_ply_only_scene_runs_full_gate_chain(self, tmp_path):
        from blueprint_pipeline.interiorgs_task_preflight import run_preflight

        scene = tmp_path / "scene"
        scene.mkdir()
        splat = _synthetic_room_splat()
        write_standard_3dgs_ply(splat, scene / "3dgs.ply")

        import blueprint_pipeline.splat_scene_bootstrap as boot_mod

        # Patch the spark renderer inside the bootstrap default path.
        orig = boot_mod.render_views_with_spark
        boot_mod.render_views_with_spark = lambda sp, cams, out, **kw: _fake_renderer(sp, cams, out)
        try:
            manifest = run_preflight(
                scene_dir=scene,
                out_dir=tmp_path / "out",
                bootstrap_missing_sidecars=True,
                bootstrap_detector=_geometry_detector,
                splat_refine=False,  # synthetic standard PLY; no compressed decode
            )
        finally:
            boot_mod.render_views_with_spark = orig
        assert manifest.get("sidecar_bootstrap", {}).get("status") == "completed"
        assert manifest["summary"]["tasks_evaluated"] >= 1
        # The pot pick-up task must resolve against the GENERATED labels.
        pot_tasks = [t for t in manifest["tasks"] if "pot" in t["task_id"]]
        assert pot_tasks and pot_tasks[0]["target"]["label"] == "pot"
