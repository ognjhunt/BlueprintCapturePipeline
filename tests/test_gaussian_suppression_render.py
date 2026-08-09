from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_suppression_render import (
    SUPPRESSED_PAYLOAD_SCHEMA_VERSION,
    GaussianSuppressionRenderError,
    build_suppressed_render_package,
    suppressed_render_payload,
    suppression_render_mask,
)
from blueprint_pipeline.gaussian_suppression_volume import (
    derive_suppression_volume_from_twin,
)
from tests.test_gaussian_suppression_volume import (
    BODY_MAX,
    BODY_MIN,
    _fridge_like_scene,
    _members,
)


def _receipt(tmp_path: Path, ply: Path, **overrides):
    arguments = {
        "task_id": "refrigerator_upper_door_open",
        "canonical_ply_path": ply,
        "body_world_aabb_min_m": BODY_MIN,
        "body_world_aabb_max_m": BODY_MAX,
        "articulated_members": _members(),
        "destination": tmp_path / "receipt.json",
    }
    arguments.update(overrides)
    return derive_suppression_volume_from_twin(**arguments)


def test_render_mask_marks_only_suppressed_rows(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)

    mask = suppression_render_mask(canonical_ply_path=ply, receipts=[receipt])

    assert mask.dtype == bool
    assert mask.sum() == len(groups["body"]) + len(groups["swept"])
    assert not mask[groups["protected"]].any()
    assert mask[groups["body"]].all()


def test_transient_payload_never_persists_and_leaves_the_scan_untouched(
    tmp_path: Path,
) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    before = ply.read_bytes()
    receipt = _receipt(tmp_path, ply)

    with suppressed_render_payload(canonical_ply_path=ply, receipts=[receipt]) as payload:
        assert payload.path.is_file()
        assert payload.record["retained_vertex_count"] == len(groups["protected"])
        assert payload.record["lifetime"] == "transient"
        assert payload.record["is_derived_cache_artifact"] is True
        held = payload.path

    assert not held.exists()
    assert ply.read_bytes() == before


def test_payload_rows_are_byte_identical_to_the_canonical_scan(tmp_path: Path) -> None:
    """Suppression must remove rows, never rewrite the ones it keeps."""

    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)

    with suppressed_render_payload(canonical_ply_path=ply, receipts=[receipt]) as payload:
        assert payload.record["retained_rows_byte_exact"] is True
        assert payload.record["retained_order_matches_source"] is True
        assert payload.record["retained_indices"] == sorted(groups["protected"])


def test_cached_payload_is_content_addressed_and_deterministic(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)
    cache = tmp_path / "cache"

    with suppressed_render_payload(
        canonical_ply_path=ply, receipts=[receipt], cache_dir=cache
    ) as first:
        first_bytes = first.path.read_bytes()
        first_name = first.path.name
        assert first.record["lifetime"] == "cached"
    assert (cache / first_name).is_file()

    with suppressed_render_payload(
        canonical_ply_path=ply, receipts=[receipt], cache_dir=cache
    ) as second:
        assert second.path.name == first_name
        assert second.path.read_bytes() == first_bytes
        assert second.record["cache_hit"] is True


def test_render_package_binds_the_whole_digest_chain(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)

    package = build_suppressed_render_package(
        canonical_ply_path=ply,
        receipts=[receipt],
        destination=tmp_path / "package",
    )

    assert package["schema_version"] == SUPPRESSED_PAYLOAD_SCHEMA_VERSION
    assert package["status"] == "suppressed_render_package_ready"
    assert package["canonical_scan"]["sha256"].startswith("sha256:")
    assert package["suppression"]["receipt_digests"] == [receipt["receipt_digest"]]
    assert package["suppression"]["suppressed_index_digest"].startswith("sha256:")
    assert package["payload"]["sha256"].startswith("sha256:")
    assert package["claim_boundary"]["canonical_scan_modified"] is False
    assert package["claim_boundary"]["payload_is_regenerable_cache"] is True
    assert Path(package["payload"]["path"]).is_file()


def test_render_package_is_byte_identical_across_builds(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)

    first = build_suppressed_render_package(
        canonical_ply_path=ply, receipts=[receipt], destination=tmp_path / "a"
    )
    second = build_suppressed_render_package(
        canonical_ply_path=ply, receipts=[receipt], destination=tmp_path / "b"
    )

    assert first["payload"]["sha256"] == second["payload"]["sha256"]
    assert (
        Path(first["payload"]["path"]).read_bytes()
        == Path(second["payload"]["path"]).read_bytes()
    )
    assert first["suppression"] == second["suppression"]


def test_two_twins_compose_against_one_canonical_scan(tmp_path: Path) -> None:
    ply, groups = _fridge_like_scene(tmp_path / "scene.ply")
    fridge = _receipt(tmp_path, ply, destination=tmp_path / "fridge.json")
    counter = _receipt(
        tmp_path,
        ply,
        task_id="counter_object",
        body_world_aabb_min_m=(1.25, 1.55, 0.55),
        body_world_aabb_max_m=(1.35, 1.65, 0.65),
        articulated_members=[],
        destination=tmp_path / "counter.json",
    )

    package = build_suppressed_render_package(
        canonical_ply_path=ply,
        receipts=[fridge, counter],
        destination=tmp_path / "both",
    )

    assert package["suppression"]["task_ids"] == [
        "counter_object",
        "refrigerator_upper_door_open",
    ]
    assert package["payload"]["retained_vertex_count"] == len(groups["protected"]) - 1


def test_removing_the_receipt_restores_the_original_scene(tmp_path: Path) -> None:
    """Reversibility: no receipts means the canonical scan renders unchanged."""

    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")

    with suppressed_render_payload(canonical_ply_path=ply, receipts=[]) as payload:
        assert payload.path == ply
        assert payload.record["lifetime"] == "canonical_passthrough"
        assert payload.record["suppressed_vertex_count"] == 0
        assert payload.path.read_bytes() == ply.read_bytes()


def test_payload_rejects_a_receipt_bound_to_another_scan(tmp_path: Path) -> None:
    from tests.test_gaussian_suppression_volume import _write_ply

    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    other = _write_ply(
        tmp_path / "other.ply", np.array([[1.7, 1.3, 0.5], [1.8, 1.4, 0.6]])
    )
    receipt = _receipt(tmp_path, other, destination=tmp_path / "other.json")

    with pytest.raises(GaussianSuppressionRenderError) as excinfo:
        with suppressed_render_payload(canonical_ply_path=ply, receipts=[receipt]):
            pass

    assert any("canonical_scan" in error for error in excinfo.value.errors)


def test_package_receipt_file_round_trips(tmp_path: Path) -> None:
    ply, _ = _fridge_like_scene(tmp_path / "scene.ply")
    receipt = _receipt(tmp_path, ply)

    package = build_suppressed_render_package(
        canonical_ply_path=ply, receipts=[receipt], destination=tmp_path / "package"
    )

    stored = json.loads(
        (tmp_path / "package" / "suppressed_render_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == package


_LOCAL_EVIDENCE = Path(
    "/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804"
)
_LOCAL_SCAN = _LOCAL_EVIDENCE / "inpainting_inputs/840796_refrigerator_v2/scene_standard.ply"
_LOCAL_SEALED_RETAINED = (
    _LOCAL_EVIDENCE
    / "gaussian_excision/840796_v2/target_only_expansion_ladder_v1/rung_01_core95"
    / "union/retained_scene_gaussians.ply"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _LOCAL_SEALED_RETAINED.is_file(),
    reason="840796 sealed retained scene not present locally",
)
def test_840796_payload_is_byte_identical_to_the_sealed_retained_scene(
    tmp_path: Path,
) -> None:
    """Byte identity is the strongest possible form of the pixel-identity gate.

    If the payload the suppression path hands a renderer is the same bytes the
    deletion path produced, every render through every renderer is identical by
    construction, and the sealed coverage evidence carries over unchanged.
    """

    import hashlib

    freeze = json.loads(
        (
            _LOCAL_EVIDENCE
            / "gaussian_excision/840796_v2/freeze"
            / "adp009b_gaussian_excision_audit_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    baseline = freeze["historical_baseline"]
    annex_path = tmp_path / "direct_evidence.npy"
    np.save(
        annex_path,
        np.sort(
            np.load(
                _LOCAL_EVIDENCE
                / "gaussian_excision/840796_v2/target_only_expansion_ladder_v1"
                / "rung_01_core95/direct/deleted_source_indices.npy"
            ).astype(np.int64)
        ),
    )
    receipt = derive_suppression_volume_from_twin(
        task_id="refrigerator_upper_door_open",
        canonical_ply_path=_LOCAL_SCAN,
        body_world_aabb_min_m=baseline["center_aabb_min_m"],
        body_world_aabb_max_m=baseline["center_aabb_max_m"],
        articulated_members=[],
        index_annex={
            "path": str(annex_path),
            "provenance": "flashsplat_direct_evidence_expansion",
        },
    )

    with suppressed_render_payload(
        canonical_ply_path=_LOCAL_SCAN, receipts=[receipt]
    ) as payload:
        observed = hashlib.sha256(payload.path.read_bytes()).hexdigest()
    expected = hashlib.sha256(_LOCAL_SEALED_RETAINED.read_bytes()).hexdigest()

    assert observed == expected
