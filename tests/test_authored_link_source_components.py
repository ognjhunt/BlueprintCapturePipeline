"""Authored links must land where they actually are, in the asset frame.

The Joint Agent review anchors link membership on bounding boxes. Fed our
authored replacement, those boxes come from the links -- and the links author
their geometry in their own frames. Composing the rest pose is therefore not a
refinement: skip it and every child link collapses onto the origin, so Scene
840920's door would sit inside its own drum and membership would be assigned to
whichever box happened to overlap.

These tests pin the frame composition, the conservative bounding of rotated and
round primitives, and the refusals that keep an uncomputable bound from being
guessed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.authored_link_source_components import (
    AuthoredLinkSourceComponentsError,
    build_authored_link_source_components,
    main,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SPEC = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests"
    / "third_scene_840920_task_a_simready_graph_asset_spec.v1.json"
)


def _seal(payload: dict, field: str) -> dict:
    payload[field] = ""
    payload[field] = canonical_digest(payload, digest_field=field)
    return payload


def _spec(*, links: list[dict]) -> dict:
    return _seal(
        {
            "schema_version": "simready_graph_asset_spec.v1",
            "asset_id": "washer",
            "links": links,
            "spec_digest": "",
        },
        "spec_digest",
    )


def _receipt(spec: dict, *, link_ids: list[str]) -> dict:
    return _seal(
        {
            "schema_version": "simready_graph_asset_receipt.v1",
            "status": "simready_candidate_authored",
            "spec_digest": spec["spec_digest"],
            "link_paths": {name: f"/Asset/links/{name}" for name in link_ids},
            "receipt_digest": "",
        },
        "receipt_digest",
    )


def _link(
    link_id: str,
    *,
    size: list[float],
    translation: list[float],
    rest: list[float],
    orientation: list[float] | None = None,
) -> dict:
    return {
        "link_id": link_id,
        "rest_pose": {
            "translation_m": rest,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "geometry": [
            {
                "geometry_id": f"{link_id}_shell",
                "kind": "box",
                "size_m": size,
                "translation_m": translation,
                "orientation_xyzw": orientation or [0.0, 0.0, 0.0, 1.0],
            }
        ],
    }


def _pair(links: list[dict]) -> tuple[dict, dict]:
    spec = _spec(links=links)
    return spec, _receipt(spec, link_ids=[link["link_id"] for link in links])


def test_rest_pose_places_a_link_in_the_asset_frame() -> None:
    """Without this, every child link collapses onto the origin."""

    spec, receipt = _pair(
        [
            _link("body", size=[1.0, 1.0, 1.0], translation=[0, 0, 0], rest=[0, 0, 0]),
            _link(
                "door",
                size=[0.4, 0.1, 0.4],
                translation=[0, 0, 0],
                rest=[0.0, -0.3, 0.5],
            ),
        ]
    )
    report = build_authored_link_source_components(spec=spec, receipt=receipt)
    door = next(c for c in report["connected_components"] if c["link_id"] == "door")
    assert door["aabb_min_asset_m"] == pytest.approx([-0.2, -0.35, 0.3])
    assert door["aabb_max_asset_m"] == pytest.approx([0.2, -0.25, 0.7])


def test_component_index_follows_the_asset_not_the_document_order() -> None:
    """The same asset must always produce the same indices."""

    forward = _pair(
        [
            _link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0]),
            _link("door", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0]),
        ]
    )
    reversed_order = _pair(
        [
            _link("door", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0]),
            _link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0]),
        ]
    )
    names = [
        [c["link_id"] for c in build_authored_link_source_components(
            spec=spec, receipt=receipt
        )["connected_components"]]
        for spec, receipt in (forward, reversed_order)
    ]
    assert names[0] == names[1] == ["body", "door"]


def test_a_rotated_box_is_bounded_by_its_rotated_corners() -> None:
    """Not by its unrotated extents, which would under-cover it."""

    half_turn_z = [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]  # 45 deg
    spec, receipt = _pair(
        [
            _link(
                "panel",
                size=[2.0, 0.0001, 1.0],
                translation=[0, 0, 0],
                rest=[0, 0, 0],
                orientation=half_turn_z,
            )
        ]
    )
    panel = build_authored_link_source_components(spec=spec, receipt=receipt)[
        "connected_components"
    ][0]
    # A 2 m wide, near-zero-thickness plate turned 45 degrees about z: its
    # corners land at half-extent * cos(45), so x shrinks from 1.0 and y grows
    # from ~0 to the same value. Ignoring the rotation would report a plate
    # 2 m wide and paper-thin in y -- covering neither the space it occupies
    # nor the space it vacates.
    reach = math.cos(math.pi / 4.0)
    assert panel["aabb_max_asset_m"][0] == pytest.approx(reach, abs=1e-3)
    assert panel["aabb_max_asset_m"][1] == pytest.approx(reach, abs=1e-3)
    assert panel["aabb_max_asset_m"][1] > 0.5, (
        "an unrotated bound would be ~0 thick in y"
    )


def test_a_capsule_is_never_bounded_more_tightly_than_a_cylinder() -> None:
    """Its caps add a radius at each end; a cylinder bound would cut them off."""

    def _bounds(kind: str) -> float:
        spec, receipt = _pair(
            [
                {
                    "link_id": "rod",
                    "rest_pose": {
                        "translation_m": [0, 0, 0],
                        "orientation_xyzw": [0, 0, 0, 1],
                    },
                    "geometry": [
                        {
                            "geometry_id": "rod",
                            "kind": kind,
                            "radius_m": 0.1,
                            "height_m": 1.0,
                            "translation_m": [0, 0, 0],
                            "orientation_xyzw": [0, 0, 0, 1],
                        }
                    ],
                }
            ]
        )
        report = build_authored_link_source_components(spec=spec, receipt=receipt)
        return report["connected_components"][0]["aabb_max_asset_m"][2]

    assert _bounds("cylinder") == pytest.approx(0.5)
    assert _bounds("capsule") == pytest.approx(0.6)


def test_an_uncomputable_bound_is_refused_not_guessed() -> None:
    spec, receipt = _pair(
        [
            {
                "link_id": "mystery",
                "rest_pose": {
                    "translation_m": [0, 0, 0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "geometry": [{"geometry_id": "m", "kind": "torus"}],
            }
        ]
    )
    with pytest.raises(AuthoredLinkSourceComponentsError) as excinfo:
        build_authored_link_source_components(spec=spec, receipt=receipt)
    assert any(
        error.startswith("authored_link_components_geometry_kind_unsupported")
        for error in excinfo.value.errors
    )


def test_a_link_missing_a_rest_pose_is_refused() -> None:
    link = _link("door", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0])
    del link["rest_pose"]
    spec, receipt = _pair([link])
    with pytest.raises(AuthoredLinkSourceComponentsError) as excinfo:
        build_authored_link_source_components(spec=spec, receipt=receipt)
    assert "authored_link_components_rest_pose_invalid:door" in excinfo.value.errors


def test_a_spec_and_receipt_describing_different_assets_are_refused() -> None:
    """A link in only one of them means no synthesized bound would mean anything."""

    spec = _spec(
        links=[_link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0])]
    )
    receipt = _receipt(spec, link_ids=["body", "door"])
    with pytest.raises(AuthoredLinkSourceComponentsError) as excinfo:
        build_authored_link_source_components(spec=spec, receipt=receipt)
    assert "authored_link_components_link_set_mismatch" in excinfo.value.errors


def test_a_receipt_bound_to_another_spec_is_refused() -> None:
    spec, receipt = _pair(
        [_link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0])]
    )
    receipt["spec_digest"] = "sha256:" + "a" * 64
    receipt = _seal({**receipt, "receipt_digest": ""}, "receipt_digest")
    with pytest.raises(AuthoredLinkSourceComponentsError) as excinfo:
        build_authored_link_source_components(spec=spec, receipt=receipt)
    assert "authored_link_components_spec_receipt_mismatch" in excinfo.value.errors


def test_output_never_claims_independent_inference() -> None:
    spec, receipt = _pair(
        [_link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0])]
    )
    claim = build_authored_link_source_components(spec=spec, receipt=receipt)[
        "claim_boundary"
    ]
    assert claim["components_are_authored_links"] is True
    assert claim["independent_topology_inference"] is False
    assert claim["joint_topology_qualified"] is False
    assert claim["simready_qualified"] is False


def test_the_real_washer_spec_places_its_door_in_front_of_the_cabinet() -> None:
    """The check that would have caught a dropped rest pose on the real asset."""

    spec = json.loads(REAL_SPEC.read_text(encoding="utf-8"))
    receipt = _receipt(spec, link_ids=[link["link_id"] for link in spec["links"]])
    report = build_authored_link_source_components(spec=spec, receipt=receipt)
    by_id = {c["link_id"]: c for c in report["connected_components"]}
    assert set(by_id) == {"body", "door", "latch", "drum", "selector", "drawer"}

    # The cabinet's front face, and the door in front of it.
    cabinet_front_y = by_id["body"]["aabb_min_asset_m"][1]
    assert by_id["door"]["aabb_min_asset_m"][1] < cabinet_front_y

    # The drum lives inside the cabinet, not out in the room.
    drum = by_id["drum"]
    assert drum["aabb_min_asset_m"][1] > by_id["door"]["aabb_min_asset_m"][1]
    assert drum["aabb_max_asset_m"][2] < by_id["body"]["aabb_max_asset_m"][2]

    # Every link keeps a positive extent on all three axes.
    for component in report["connected_components"]:
        for low, high in zip(
            component["aabb_min_asset_m"], component["aabb_max_asset_m"]
        ):
            assert high > low


def test_cli_writes_components_and_refuses_a_mismatched_pair(tmp_path: Path) -> None:
    spec, receipt = _pair(
        [
            _link("body", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, 0, 0]),
            _link("door", size=[1, 1, 1], translation=[0, 0, 0], rest=[0, -1, 0]),
        ]
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "components.json"

    argv = [
        "--spec",
        str(spec_path),
        "--receipt",
        str(receipt_path),
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["connected_component_count"] == 2
    assert written["components_digest"].startswith("sha256:")

    receipt_path.write_text(
        json.dumps(_receipt(spec, link_ids=["body"])), encoding="utf-8"
    )
    assert main(argv) == 2
