"""SimReady scene preflight gate (the routed measurement execution lane was retired)."""

from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.simready_asset_lane import (
    build_simready_asset_request,
    generate_simready_asset_draft,
    preflight_simready_scene,
    validate_simready_asset_manifest,
)


mujoco = pytest.importorskip("mujoco")


SHA_A = "sha256:" + "a" * 64


_BOX_VERTICES = [
    [-0.05, -0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0],
    [-0.05, -0.05, 0.1], [0.05, -0.05, 0.1], [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1],
]
_BOX_FACES = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
]


def _draft_manifest() -> dict:
    return generate_simready_asset_draft(
        build_simready_asset_request(
            request_id="preflight-request",
            object_id="counter-mug",
            bundle_id="capture-1",
            bundle_hash=SHA_A,
            source_references={
                "segmentation_record_id": "record-object_segmentation",
                "mesh_record_id": "record-mesh",
                "provenance_record_id": "provenance-fixture",
            },
            density_class="ceramic_glass",
        ),
        vertices=_BOX_VERTICES,
        faces=_BOX_FACES,
        generated_on="2026-08-02",
    )


def test_scene_preflight_passes_a_sound_draft_without_granting_validity() -> None:
    report = preflight_simready_scene([_draft_manifest()], settle_steps=80)
    assert report["passed"] is True
    row = report["assets"][0]
    assert row["loaded"] is True
    assert row["structural_checks"]["mass_positive"] is True
    assert row["stability"]["state_finite"] is True
    assert row["stability"]["exploded"] is False
    assert report["preflight_pass_means_loadable_and_stable_only"] is True
    assert report["physical_validity_established"] is False
    assert report["qualification_created"] is False


def test_scene_preflight_blocks_a_corrupted_export_with_typed_codes() -> None:
    manifest = _draft_manifest()
    corrupted = copy.deepcopy(manifest)
    corrupted.pop("simready_asset_digest")
    exports = dict(corrupted["exports"])
    mjcf = dict(exports["mjcf"])
    mjcf["content"] = mjcf["content"][: len(mjcf["content"]) // 2]
    import hashlib as _hashlib

    mjcf["content_digest"] = (
        "sha256:" + _hashlib.sha256(mjcf["content"].encode("utf-8")).hexdigest()
    )
    exports["mjcf"] = mjcf
    corrupted["exports"] = exports
    corrupted = validate_simready_asset_manifest(corrupted)
    report = preflight_simready_scene([corrupted])
    assert report["passed"] is False
    row = report["assets"][0]
    assert row["loaded"] is False
    assert any(
        code.startswith("preflight_mjcf_load_failed:") for code in row["failure_codes"]
    )
