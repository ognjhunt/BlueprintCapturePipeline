"""Industrial task-success proxy coverage for eval-ready task grounding (R006).

These tests mirror the sink-handle contract tests but exercise the industrial
material-handling / pick-place / transfer / delivery success proxies that run in
parallel to the articulated handle proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import eval_ready_task_grounding as grounding


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _object(object_id: str, label: str, *, cx: int, cy: int) -> dict:
    slug = object_id
    return {
        "object_id": object_id,
        "label": label,
        "source_prompt": label,
        "mean_confidence": 0.9,
        "reference_crop": f"object_index_artifacts/crops/{slug}.png",
        "all_crops": [f"object_index_artifacts/crops/{slug}.png"],
        "keypoints": {"center": [cx, cy]},
        "mean_box_px": {"x": cx - 20, "y": cy - 20, "width": 40, "height": 40},
    }


def _build(tmp_path: Path, *, objects: list, task_id: str, task_text: str, target_label: str) -> dict:
    capture_root = tmp_path / "capture"
    _write_json(capture_root / "raw" / "object_index.json", {"objects": objects})
    return grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id=task_id,
        task_text=task_text,
        target_label=target_label,
    )


# --------------------------------------------------------------------------- #
# Gating predicates
# --------------------------------------------------------------------------- #
def test_containment_gating_predicate() -> None:
    assert grounding._requires_containment_proxy(
        task_text="place the object into the bin", target_label="target bin"
    )
    # Inspection / transport of a container is not a containment task.
    assert not grounding._requires_containment_proxy(
        task_text="inspect the storage bin", target_label="storage bin"
    )
    assert not grounding._requires_containment_proxy(
        task_text="move the tote to staging", target_label="tote"
    )


def test_placement_at_target_gating_predicate() -> None:
    assert grounding._requires_placement_at_target_proxy(
        task_text="place the box on the pallet", target_label="pallet"
    )
    assert not grounding._requires_placement_at_target_proxy(
        task_text="inspect the pallet rack", target_label="pallet rack"
    )


def test_transfer_zone_arrival_gating_predicate() -> None:
    assert grounding._requires_transfer_zone_arrival_proxy(
        task_text="move the tote to the conveyor", target_label="conveyor"
    )
    assert grounding._requires_transfer_zone_arrival_proxy(
        task_text="deliver parts to the line side station", target_label="line side station"
    )
    assert not grounding._requires_transfer_zone_arrival_proxy(
        task_text="inspect the conveyor", target_label="conveyor"
    )


# --------------------------------------------------------------------------- #
# Containment proxy (place_object_into_bin family)
# --------------------------------------------------------------------------- #
def test_containment_proxy_present_for_pick_place_into_bin(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        objects=[
            _object("part_01", "machined part", cx=200, cy=200),
            _object("bin_01", "storage bin", cx=400, cy=300),
        ],
        task_id="place_object_into_bin",
        task_text="place the machined part into the storage bin",
        target_label="machined part",
    )

    proxies = manifest["industrial_state_proxies"]
    assert proxies["any_configured"] is True
    assert "containment_in_receptacle" in proxies["active_proxy_types"]

    containment = proxies["containment"]
    assert containment["available"] is True
    assert containment["proxy_type"] == "containment_in_receptacle"
    assert containment["success_rule"] == "moved_object_centroid_inside_receptacle_aabb"
    assert containment["target_object_id"] == "part_01"
    assert containment["receptacle_object_id"] == "bin_01"
    assert containment["state_success_proven"] is False
    assert containment["receptacle_center_px"] == [400.0, 300.0]
    assert containment["claim_boundary"]["proxy_is_not_real_containment_proof"] is True

    # Only the containment proxy should fire for this task.
    assert proxies["placement_at_target"]["available"] is False
    assert proxies["transfer_zone_arrival"]["available"] is False

    # Readiness surfaces the industrial grounding but never claims physical success.
    assert manifest["readiness"]["industrial_proxy_configured"] is True
    assert manifest["readiness"]["industrial_proxy_types"] == ["containment_in_receptacle"]
    assert manifest["readiness"]["exact_task_success_proven"] is False


# --------------------------------------------------------------------------- #
# Placement-at-target proxy
# --------------------------------------------------------------------------- #
def test_placement_at_target_proxy_present(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        objects=[
            _object("component_01", "steel component", cx=180, cy=220),
            _object("pallet_01", "pallet staging zone", cx=420, cy=260),
        ],
        task_id="stage_component",
        task_text="place the steel component on the pallet staging zone",
        target_label="steel component",
    )

    proxies = manifest["industrial_state_proxies"]
    assert proxies["any_configured"] is True
    assert "placement_at_target_pose" in proxies["active_proxy_types"]

    placement = proxies["placement_at_target"]
    assert placement["available"] is True
    assert placement["proxy_type"] == "placement_at_target_pose"
    assert placement["success_rule"] == "moved_object_within_tolerance_of_target_zone_or_pose"
    assert placement["target_object_id"] == "component_01"
    assert placement["target_zone_object_id"] == "pallet_01"
    assert placement["placement_tolerance_px"] == grounding.DEFAULT_PLACEMENT_TOLERANCE_PX
    assert placement["state_success_proven"] is False
    assert placement["claim_boundary"]["proxy_is_not_real_placement_proof"] is True

    assert proxies["containment"]["available"] is False
    assert proxies["transfer_zone_arrival"]["available"] is False


# --------------------------------------------------------------------------- #
# Transfer / line-side zone-arrival proxy (move_tote, cart_to_conveyor_transfer,
# line_side_delivery families)
# --------------------------------------------------------------------------- #
def test_transfer_zone_arrival_proxy_present_for_move_tote(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        objects=[
            _object("tote_01", "warehouse tote", cx=210, cy=200),
            _object("conveyor_01", "conveyor line", cx=450, cy=280),
        ],
        task_id="move_tote",
        task_text="move the warehouse tote from staging to the conveyor line",
        target_label="warehouse tote",
    )

    proxies = manifest["industrial_state_proxies"]
    assert proxies["any_configured"] is True
    assert "transfer_zone_arrival" in proxies["active_proxy_types"]

    transfer = proxies["transfer_zone_arrival"]
    assert transfer["available"] is True
    assert transfer["proxy_type"] == "transfer_zone_arrival"
    assert transfer["success_rule"] == "moved_object_or_end_effector_reaches_destination_zone"
    assert transfer["target_object_id"] == "tote_01"
    assert transfer["destination_zone_object_id"] == "conveyor_01"
    assert transfer["arrival_tolerance_px"] == grounding.DEFAULT_ZONE_ARRIVAL_TOLERANCE_PX
    assert transfer["state_success_proven"] is False
    assert transfer["claim_boundary"]["proxy_is_not_real_transfer_proof"] is True

    # move_tote transports the tote; it is not a "put something into the tote" task.
    assert proxies["containment"]["available"] is False
    assert proxies["placement_at_target"]["available"] is False


def test_transfer_zone_arrival_proxy_present_for_line_side_delivery(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        objects=[
            _object("tote_02", "parts tote", cx=200, cy=210),
            _object("station_07", "line side station", cx=440, cy=300),
        ],
        task_id="line_side_delivery",
        task_text="deliver the parts tote to the line side station",
        target_label="parts tote",
    )

    transfer = manifest["industrial_state_proxies"]["transfer_zone_arrival"]
    assert transfer["available"] is True
    assert transfer["proxy_type"] == "transfer_zone_arrival"
    assert transfer["target_object_id"] == "tote_02"
    assert transfer["destination_zone_object_id"] == "station_07"


# --------------------------------------------------------------------------- #
# Non-matching tasks are unaffected: handle proxy path and inspection tasks
# --------------------------------------------------------------------------- #
def test_industrial_proxies_absent_for_handle_task(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    _write_json(
        capture_root / "raw" / "object_index.json",
        {
            "objects": [
                _object("sink_parent", "sink", cx=300, cy=240),
                _object("right_sink_handle_01", "right sink handle", cx=322, cy=188),
            ]
        },
    )

    manifest = grounding.build_eval_ready_task_grounding(
        capture_root=capture_root,
        task_id="turn_on_sink_handle",
        task_text="turn on the sink right handle",
        target_label="right sink handle",
        articulated_handle_proxy=True,
    )

    proxies = manifest["industrial_state_proxies"]
    assert proxies["any_configured"] is False
    assert proxies["active_proxy_types"] == []
    assert proxies["containment"]["available"] is False
    assert proxies["placement_at_target"]["available"] is False
    assert proxies["transfer_zone_arrival"]["available"] is False
    assert proxies["containment"]["reason"] == "task_does_not_request_containment_proxy"

    # The existing articulated handle proxy is fully preserved.
    assert manifest["articulated_state_proxy"]["available"] is True
    assert manifest["articulated_state_proxy"]["proxy_type"] == "revolute_sink_handle"
    assert manifest["readiness"]["industrial_proxy_configured"] is False


def test_industrial_proxies_absent_for_inspection_task(tmp_path: Path) -> None:
    manifest = _build(
        tmp_path,
        objects=[_object("rack_01", "storage rack", cx=300, cy=200)],
        task_id="inspect_rack",
        task_text="inspect the storage rack",
        target_label="storage rack",
    )

    proxies = manifest["industrial_state_proxies"]
    assert proxies["any_configured"] is False
    assert proxies["active_proxy_types"] == []
    assert manifest["readiness"]["industrial_proxy_configured"] is False
