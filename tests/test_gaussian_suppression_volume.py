from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_suppression_volume import (
    GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION,
    GaussianSuppressionVolumeError,
    compose_suppression_volumes,
    derive_suppression_volume_from_twin,
    resolve_suppressed_indices,
)


BODY_MIN = (1.617248143733499, 1.1292180586542502, 0.0)
BODY_MAX = (2.3311802562665007, 1.82921814134575, 1.6318699975585937)
HINGE = (1.617248144, 1.829218141, 1.2859256235)
CLOSED_ENDPOINT = (2.331180256, 1.829218141, 1.2859256235)
UPPER_INTERVAL = (0.939981249, 1.631869998)


def _write_ply(path: Path, points: np.ndarray, scales_log: np.ndarray | None = None) -> Path:
    """Write a minimal standard 3DGS float PLY with the given centers."""

    count = int(points.shape[0])
    names = [
        "x",
        "y",
        "z",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    body = np.zeros((count, len(names)), dtype="<f4")
    body[:, 0:3] = points.astype("<f4")
    body[:, 3] = 2.0
    body[:, 7:10] = (
        np.full((count, 3), -6.0, dtype="<f4")
        if scales_log is None
        else scales_log.astype("<f4")
    )
    body[:, 10] = 1.0
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(body.tobytes())
    return path


def _fridge_like_scene(path: Path) -> tuple[Path, dict[str, list[int]]]:
    """Body splats, swept-wedge splats, and protected splats at known indices."""

    rows: list[tuple[float, float, float]] = []
    groups: dict[str, list[int]] = {"body": [], "swept": [], "protected": []}
    for zi in range(6):
        for xi in range(4):
            for yi in range(3):
                rows.append(
                    (
                        1.70 + 0.15 * xi,
                        1.20 + 0.2 * yi,
                        0.15 + 0.28 * zi,
                    )
                )
                groups["body"].append(len(rows) - 1)
    # inside the upper door's swing arc (in front of the closed fridge, z-band only)
    for step in range(4):
        angle = math.radians(20.0 + 15.0 * step)
        radius = 0.5
        rows.append(
            (
                HINGE[0] + radius * math.cos(angle),
                HINGE[1] + radius * math.sin(angle),
                1.20,
            )
        )
        groups["swept"].append(len(rows) - 1)
    # protected: floor, far wall, and the counter to the left
    for point in (
        (1.90, 2.60, 0.02),
        (0.80, 1.50, 0.90),
        (3.10, 1.40, 1.00),
        (1.95, 1.50, 1.90),
        (1.30, 1.60, 0.60),
    ):
        rows.append(point)
        groups["protected"].append(len(rows) - 1)
    _write_ply(path, np.array(rows, dtype=np.float64))
    return path, groups


def _members() -> list[dict]:
    return [
        {
            "member_id": "upper_door",
            "hinge_origin_world_m": list(HINGE),
            "closed_endpoint_world_m": list(CLOSED_ENDPOINT),
            "vertical_interval_m": list(UPPER_INTERVAL),
            "limit_degrees": [0.0, 90.0],
            "half_thickness_m": 0.05,
        }
    ]


def _derive(tmp_path: Path, ply: Path, **overrides):
    arguments = {
        "task_id": "refrigerator_upper_door_open",
        "canonical_ply_path": ply,
        "body_world_aabb_min_m": BODY_MIN,
        "body_world_aabb_max_m": BODY_MAX,
        "body_margin_m": 0.0,
        "articulated_members": _members(),
        "membership_mode": "center_in_volume",
        "destination": tmp_path / "receipt.json",
    }
    arguments.update(overrides)
    return derive_suppression_volume_from_twin(**arguments)


def test_receipt_binds_the_canonical_scan_without_modifying_it(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    before = ply.read_bytes()

    receipt = _derive(tmp_path, ply)

    assert receipt["schema_version"] == GAUSSIAN_SUPPRESSION_VOLUME_SCHEMA_VERSION
    assert receipt["status"] == "suppression_volume_bound"
    assert receipt["canonical_scan"]["sha256"].startswith("sha256:")
    assert receipt["canonical_scan"]["vertex_count"] == 81
    assert receipt["claim_boundary"]["canonical_scan_modified"] is False
    assert receipt["claim_boundary"]["reversible"] is True
    assert receipt["receipt_digest"].startswith("sha256:")
    assert ply.read_bytes() == before


def test_body_region_and_swept_regions_are_separately_accounted(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")

    receipt = _derive(tmp_path, ply)

    roles = [region["role"] for region in receipt["regions"]]
    assert roles == ["body", "swept_member"]
    body_region, swept_region = receipt["regions"]
    assert body_region["kind"] == "axis_aligned_box"
    assert swept_region["kind"] == "revolute_swept_prism"
    assert swept_region["member_id"] == "upper_door"
    assert swept_region["limit_degrees"] == [0.0, 90.0]
    assert receipt["capture"]["body_index_count"] == len(groups["body"])
    assert receipt["capture"]["swept_only_index_count"] == len(groups["swept"])
    assert receipt["capture"]["suppressed_index_count"] == len(groups["body"]) + len(
        groups["swept"]
    )


def test_resolver_suppresses_exactly_the_expected_indices(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _derive(tmp_path, ply)

    indices, digest = resolve_suppressed_indices(canonical_ply_path=ply, receipt=receipt)

    assert indices.tolist() == sorted(groups["body"] + groups["swept"])
    assert not set(indices.tolist()) & set(groups["protected"])
    assert digest == receipt["capture"]["suppressed_index_digest"]


def test_swept_capture_ceiling_fails_closed_on_real_scene_content(
    tmp_path: Path,
) -> None:
    """A door swinging through occupied space must not silently delete it."""

    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")

    with pytest.raises(GaussianSuppressionVolumeError) as excinfo:
        _derive(tmp_path, ply, swept_region_capture_ceiling=2)

    assert any(
        "swept_region_capture_exceeds_ceiling" in error for error in excinfo.value.errors
    )


def test_support_overlap_mode_captures_more_than_center_mode(tmp_path: Path) -> None:
    rows = np.array(
        [
            [BODY_MAX[0] + 0.03, 1.5, 1.0],  # centre just outside, wide support
            [BODY_MAX[0] + 0.03, 1.5, 1.2],
            [0.5, 0.5, 0.5],  # far away in every mode
        ]
    )
    scales = np.log(np.full((3, 3), 0.06, dtype=np.float64))
    scales[2, :] = math.log(0.001)
    ply = _write_ply(tmp_path / "halo.ply", rows, scales)

    centered = _derive(tmp_path, ply, articulated_members=[])
    overlapped = _derive(
        tmp_path,
        ply,
        articulated_members=[],
        membership_mode="support_overlap_k_sigma",
        support_sigma_multiplier=2.0,
        destination=tmp_path / "receipt_overlap.json",
    )

    assert centered["capture"]["suppressed_index_count"] == 0
    assert overlapped["capture"]["suppressed_index_count"] == 2
    assert overlapped["membership"]["mode"] == "support_overlap_k_sigma"
    assert overlapped["membership"]["support_sigma_multiplier"] == 2.0


def test_index_annex_unions_evidence_derived_indices(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    annex_indices = np.array(sorted(groups["protected"][:2]), dtype=np.int64)
    annex_path = tmp_path / "annex.npy"
    np.save(annex_path, annex_indices)

    receipt = _derive(
        tmp_path,
        ply,
        index_annex={
            "path": str(annex_path),
            "provenance": "flashsplat_direct_evidence_expansion",
            "justification": "photometric target-only contribution in >=2 calibration cameras",
        },
    )
    indices, _ = resolve_suppressed_indices(canonical_ply_path=ply, receipt=receipt)

    assert receipt["index_annex"]["count"] == 2
    assert receipt["index_annex"]["sha256"].startswith("sha256:")
    assert set(annex_indices.tolist()) <= set(indices.tolist())
    assert receipt["capture"]["annex_only_index_count"] == 2


def test_annex_out_of_range_fails_closed(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    annex_path = tmp_path / "annex.npy"
    np.save(annex_path, np.array([5, 10_000_000], dtype=np.int64))

    with pytest.raises(GaussianSuppressionVolumeError) as excinfo:
        _derive(tmp_path, ply, index_annex={"path": str(annex_path)})

    assert any("index_annex_out_of_range" in error for error in excinfo.value.errors)


def test_receipt_is_deterministic(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")

    first = _derive(tmp_path, ply, destination=tmp_path / "a.json")
    second = _derive(tmp_path, ply, destination=tmp_path / "b.json")

    assert first["receipt_digest"] == second["receipt_digest"]
    assert first["capture"] == second["capture"]
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_rigid_can_fixture_needs_no_articulated_member(tmp_path: Path) -> None:
    """The original 840313 rigid shape must work with a body region alone."""

    rows = np.array(
        [
            [0.0, 0.0, 0.55],
            [0.01, 0.0, 0.60],
            [0.4, 0.4, 0.55],
        ]
    )
    ply = _write_ply(tmp_path / "can.ply", rows)

    receipt = _derive(
        tmp_path,
        ply,
        task_id="canned_beverage_pick_place",
        body_world_aabb_min_m=(-0.04, -0.04, 0.52),
        body_world_aabb_max_m=(0.04, 0.04, 0.69),
        articulated_members=[],
    )
    indices, _ = resolve_suppressed_indices(canonical_ply_path=ply, receipt=receipt)

    assert [region["role"] for region in receipt["regions"]] == ["body"]
    assert indices.tolist() == [0, 1]


def test_composition_unions_many_twins_against_one_scan(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    fridge = _derive(tmp_path, ply, destination=tmp_path / "fridge.json")
    other = _derive(
        tmp_path,
        ply,
        task_id="counter_object",
        body_world_aabb_min_m=(1.25, 1.55, 0.55),
        body_world_aabb_max_m=(1.35, 1.65, 0.65),
        articulated_members=[],
        destination=tmp_path / "other.json",
    )

    composite = compose_suppression_volumes(
        canonical_ply_path=ply, receipts=[fridge, other]
    )

    assert composite["task_ids"] == ["counter_object", "refrigerator_upper_door_open"]
    assert composite["suppressed_index_count"] == len(groups["body"]) + len(
        groups["swept"]
    ) + 1
    assert composite["composite_digest"].startswith("sha256:")
    single = compose_suppression_volumes(canonical_ply_path=ply, receipts=[fridge])
    assert single["suppressed_index_count"] < composite["suppressed_index_count"]


def test_composition_rejects_receipts_from_a_different_scan(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    other_ply = _write_ply(tmp_path / "other.ply", np.array([[0.0, 0.0, 0.0]]))
    fridge = _derive(tmp_path, ply, destination=tmp_path / "fridge.json")
    foreign = _derive(
        tmp_path,
        other_ply,
        task_id="foreign",
        articulated_members=[],
        destination=tmp_path / "foreign.json",
    )

    with pytest.raises(GaussianSuppressionVolumeError) as excinfo:
        compose_suppression_volumes(
            canonical_ply_path=ply, receipts=[fridge, foreign]
        )

    assert any(
        "composition_canonical_scan_mismatch" in error for error in excinfo.value.errors
    )


def test_receipt_file_round_trips(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")

    receipt = _derive(tmp_path, ply)

    stored = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert stored == receipt


_LOCAL_EVIDENCE = Path(
    "/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804"
)
_LOCAL_SCAN = _LOCAL_EVIDENCE / "inpainting_inputs/840796_refrigerator_v2/scene_standard.ply"
_LOCAL_SEALED = (
    _LOCAL_EVIDENCE
    / "gaussian_excision/840796_v2/target_only_expansion_ladder_v1/rung_01_core95"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _LOCAL_SCAN.is_file(), reason="840796 canonical scan not present locally"
)
def test_840796_volume_reproduces_the_sealed_deletion_set(tmp_path: Path) -> None:
    """The suppression path must hide exactly what the sealed cutout deleted.

    Equality is the whole claim of the render-time method: if the resolved
    index set differs from the sealed deletion, the two paths cannot render
    identically and the earlier coverage evidence would not carry over. Adding
    the door's swept wedge is allowed to grow the set - that region is new -
    but only as a strict superset.
    """

    freeze = json.loads(
        (
            _LOCAL_EVIDENCE
            / "gaussian_excision/840796_v2/freeze"
            / "adp009b_gaussian_excision_audit_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    baseline = freeze["historical_baseline"]
    sealed = np.load(_LOCAL_SEALED / "union/deleted_source_indices.npy").astype(np.int64)
    annex_path = tmp_path / "direct_evidence.npy"
    np.save(
        annex_path,
        np.sort(np.load(_LOCAL_SEALED / "direct/deleted_source_indices.npy").astype(np.int64)),
    )
    common = {
        "task_id": "refrigerator_upper_door_open",
        "canonical_ply_path": _LOCAL_SCAN,
        "body_world_aabb_min_m": baseline["center_aabb_min_m"],
        "body_world_aabb_max_m": baseline["center_aabb_max_m"],
        "index_annex": {
            "path": str(annex_path),
            "provenance": "flashsplat_direct_evidence_expansion",
        },
    }

    body_only = derive_suppression_volume_from_twin(**common, articulated_members=[])
    resolved, _ = resolve_suppressed_indices(
        canonical_ply_path=_LOCAL_SCAN, receipt=body_only
    )
    assert body_only["capture"]["body_index_count"] == 3791
    assert np.array_equal(np.sort(resolved), np.sort(sealed))

    swept = derive_suppression_volume_from_twin(
        **common,
        articulated_members=[
            {
                "member_id": "upper_door",
                "hinge_origin_world_m": list(HINGE),
                "closed_endpoint_world_m": list(CLOSED_ENDPOINT),
                "vertical_interval_m": list(UPPER_INTERVAL),
                "limit_degrees": [0.0, 90.0],
                "half_thickness_m": 0.05,
            }
        ],
    )
    with_sweep, _ = resolve_suppressed_indices(
        canonical_ply_path=_LOCAL_SCAN, receipt=swept
    )
    assert set(sealed.tolist()) < set(with_sweep.tolist())
    assert swept["capture"]["swept_only_index_count"] == 2
