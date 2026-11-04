"""Movable chairs moved out of the Franka's way in the simulation, only on the site owner's approval.

capture-447ae4f4's Franka could not stand within reach of its dining table, most likely because of the
chairs around it (#3355). These rooms are synthetic: capture-447's oval table in one scanned mesh, as a
website collider is, with ten chairs around it. No ranked stance carries; moving one named chair opens
one that does. The half proposes that chair, the owner decides on the job page, and only an approval
rebuilds the half without it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import robot_access_furniture as access  # noqa: E402
from blueprint_pipeline import task_evaluation_scene_intake as intake  # noqa: E402
from blueprint_pipeline import team_scene_packet_worker as worker  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest  # noqa: E402
from tests import test_team_scene_packet_single_mesh_room as single  # noqa: E402
from tests.test_team_embodiment_stance_room_mesh import _box  # noqa: E402

SEAT_TOP, BACK_TOP = 0.45, 0.90
SUPPORT = {"bounds_min_xyz_m": [*single.SUPPORT_MIN[:2], 0.77], "bounds_max_xyz_m": [*single.SUPPORT_MAX[:2], 0.8189]}


def _chair(x0: float, y0: float, back: str, *, kind: str = "dining chair", target_class: str = "movable_object",
           role: str = "obstacle") -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """A 45 cm chair as the scan holds it, and the analysis target the preparation measured for it."""

    x1, y1 = x0 + 0.45, y0 + 0.45
    seat = _box((x0, y0, 0.0), (x1, y1, SEAT_TOP), bottom=False)
    bx0, by0, bx1, by1 = {"-x": (x0, y0, x0 + 0.05, y1), "+x": (x1 - 0.05, y0, x1, y1),
                          "-y": (x0, y0, x1, y0 + 0.05), "+y": (x0, y1 - 0.05, x1, y1)}[back]
    rest = _box((bx0, by0, SEAT_TOP), (bx1, by1, BACK_TOP), bottom=False)
    mesh = (np.vstack([seat[0], rest[0]]), np.vstack([seat[1], rest[1] + len(seat[0])]))
    target = {"target_id": f"chair_{back}_{x0:+.2f}_{y0:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_"),
              "semantic_label": kind, "segmentation_prompt": "wooden chair", "target_class": target_class,
              "target_role": role, "disposition": "keep", "task_effect": "static_obstacle",
              "estimated_visible_bounds": {"minimum": [x0 - 0.01, y0 - 0.01, -0.01],
                                           "maximum": [x1 + 0.01, y1 + 0.01, BACK_TOP + 0.01]}}
    return mesh, target


# Three chairs on each long side, two at each end, a gap of 5 cm between neighbours: no stance near the
# table is clear, and the one the carry needs stands where the middle chair on the +x side is.
CHAIRS = [_chair(-1.55, -0.95, "-x"), _chair(-1.55, -0.45, "-x"), _chair(-1.55, 0.05, "-x"),
          _chair(0.32, -1.17, "+x"), _chair(0.32, -0.67, "+x"), _chair(0.32, -0.17, "+x"),
          _chair(-0.85, -1.62, "-y"), _chair(-0.35, -1.62, "-y"), _chair(-0.85, 0.58, "+y"),
          _chair(-0.35, 0.58, "+y")]
BLOCKING = CHAIRS[4][1]["target_id"]


def _furniture(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """What the preparation records (sim frame = these bounds), as ``movable_furniture_for_capture`` returns it."""

    record = access.movable_furniture_record(targets, lambda bounds: (bounds["minimum"], bounds["maximum"]))
    return {"status": "measured", "record": record, "preparation_digest": "sha256:" + "5" * 64}


@pytest.fixture
def chairs_room(monkeypatch) -> None:
    """The capture-447 bundle's collider with the chairs in its one scanned mesh."""

    write = single.write_scanned_room
    monkeypatch.setattr(single, "write_scanned_room",
                        lambda path, **options: write(path, parts=[mesh for mesh, _ in CHAIRS], **options))


def _compile(tmp_path: Path, monkeypatch: Any, name: str, inputs: dict[str, Any], **extra: Any) -> dict[str, Any]:
    from blueprint_pipeline.team_scene_packet import materialize_team_scene

    compiled = materialize_team_scene(**inputs, output_root=tmp_path / name, **extra)
    return {**json.loads(Path(compiled["path"]).read_text()), "_compiled": compiled}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_proposal_names_what_the_analysis_does_not_call_movable_furniture(tmp_path, monkeypatch, chairs_room):
    # The same chairs, but the analysis kept them as fixed clutter: nothing is named, nothing proposed.
    inputs = single._inputs(tmp_path, monkeypatch)
    fixed = [{**target, "target_class": "fixed_clutter"} for _, target in CHAIRS]
    furniture = _furniture(fixed)
    assert furniture["record"]["items"] == []
    assert {row["reason"] for row in furniture["record"]["excluded"]} == {"analysis_class_not_movable_object"}
    record = _compile(tmp_path, monkeypatch, "fixed", inputs, movable_furniture=furniture,
                      robot_access_scope=inputs["owner_success_confirmation_scope"])
    assert record["robot_access_removal"]["status"] == "not_proposed"
    assert record["robot_access_removal"]["reason"] == "no_named_movable_furniture"
    assert "movable_obstacle_counterfactual" not in record["robot_station"]["search"]


def test_only_kept_chairs_and_stools_clear_of_the_support_are_ever_named() -> None:
    def target(**row: Any) -> dict[str, Any]:
        return {"target_id": "t", "segmentation_prompt": "", "target_class": "movable_object", "target_role": "obstacle",
                "disposition": "keep", "task_effect": "static_obstacle",
                "estimated_visible_bounds": {"minimum": [1.0, 1.0, 0.0], "maximum": [1.4, 1.4, 0.9]}, **row}

    rows = [target(target_id="wall", semantic_label="painted wall", target_class="fixed_clutter",
                   target_role="background", task_effect="unrelated"),
            target(target_id="counter", semantic_label="kitchen counter", target_class="fixed_clutter"),
            target(target_id="table", semantic_label="dining table", target_role="support", task_effect="static_contact"),
            target(target_id="tray", semantic_label="wooden tray", target_role="task_object", disposition="remove",
                   task_effect="manipulated"),
            target(target_id="person", semantic_label="person on a chair", target_class="person", target_role="person"),
            target(target_id="fixed_seat", semantic_label="built-in bench seat with chair cushions",
                   target_class="fixed_clutter"),
            target(target_id="chair_on_support", semantic_label="chair", target_role="support",
                   task_effect="static_contact"),
            target(target_id="vase", semantic_label="floor vase"),
            target(target_id="bar_stool", semantic_label="tall bar stool by the island", segmentation_prompt="stool"),
            target(target_id="tucked_chair", semantic_label="chair pushed under the table",
                   estimated_visible_bounds={"minimum": [0.2, -0.3, 0.0], "maximum": [0.6, 0.1, 0.9]}),
            target(target_id="chair_unmeasured", semantic_label="armchair", estimated_visible_bounds=None)]
    record = access.movable_furniture_record(rows, lambda bounds: (bounds["minimum"], bounds["maximum"]))
    assert [row["obstacle_id"] for row in record["items"]] == ["bar_stool", "tucked_chair"]
    assert record["items"][0]["kind"] == "stool"
    # The wall, the counter, the table, the tray, the person and the vase are never listed.
    assert {row["target_id"]: row["reason"] for row in record["excluded"]} == {
        "fixed_seat": "analysis_class_not_movable_object", "chair_on_support": "analysis_role_not_obstacle_or_background",
        "chair_unmeasured": "bounds_unmeasured"}
    named, refused = access.eligible_obstacles(record, supports=[SUPPORT])
    assert [row["obstacle_id"] for row in named] == ["bar_stool"]
    assert refused == [{"obstacle_id": "tucked_chair", "reason": "overlaps_task_support"}]
    assert access.eligible_obstacles(record, supports=[])[0] == []


def _queued(tmp_path: Path, monkeypatch: Any) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    """A completed request whose compiled half proposes moving a chair, bound to its own intent."""

    inputs = single._inputs(tmp_path, monkeypatch)
    scope = inputs["owner_success_confirmation_scope"]
    request = worker.validated_request(worker._seal({
        "schema_version": worker.REQUEST_SCHEMA, "intent_id": scope["intent"]["intent_id"],
        "intent_digest": scope["intent"]["intent_digest"], "capture_id": "walkthrough-1",
        "task_id": "website-" + "0" * 20, "source_launch_root": str(tmp_path / "launch"),
        "launch_receipt_digest": "sha256:" + "1" * 64, "scene_revision_digest": inputs["revision"]["revision_digest"],
        "robot_free_readiness_status": "refused"}, "request_digest"))
    progression = json.loads((scope["directory"] / "progression.json").read_text())
    progression["state"]["scene_preparation_completion"]["team_scene_request"]["request_digest"] = (
        request["request_digest"])
    (scope["directory"] / "progression.json").write_text(json.dumps(intake._seal(progression, "progression_digest")))
    record = _compile(tmp_path, monkeypatch, "proposed", inputs,
                      movable_furniture=_furniture([target for _, target in CHAIRS]), robot_access_scope=scope)
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    queue = worker._queue(tmp_path / "queue")
    key = worker.request_key(request)
    (queue / "processed" / f"{key}.json").write_text(json.dumps(request))
    (queue / "status" / f"{key}.json").write_text(json.dumps({
        "schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "completed", "blockers": [],
        "team_scene_inputs_path": record["_compiled"]["path"]}))
    return inputs, queue, key, record


class FakeWebApp:
    """The WebApp side: it keeps each proposal and answers with the owner's decision on it."""

    def __init__(self) -> None:
        self.proposals: list[dict[str, Any]] = []
        self.decision: dict[str, Any] | None = None

    def __call__(self, notice: dict[str, Any]) -> dict[str, Any]:
        notice = json.loads(json.dumps(notice))
        assert notice["proposal_digest"] == cross_runtime_canonical_digest(notice, digest_field="proposal_digest")
        self.proposals.append(notice)
        return {"schema_version": access.RECEIPT_SCHEMA, "status": self.decision["decision"] if self.decision else "pending",
                "intent_id": notice["intent_id"], "proposal_digest": notice["proposal_digest"], "decision": self.decision}

    def decide(self, kind: str) -> None:
        notice = self.proposals[-1]
        decision = {"schema_version": access.DECISION_SCHEMA, "intent_id": notice["intent_id"],
                    "intent_digest": notice["intent_digest"], "request_id": "req-1", "capture_id": "cap-1",
                    "proposal_digest": notice["proposal_digest"],
                    "configured_scene_revision_digest": notice["configured_scene_revision_digest"],
                    "removal_binding_digest": notice["removal_binding_digest"], "decision": kind,
                    "decided_via": "owner_job_link", "decided_at": "2026-11-04T09:00:00.000Z"}
        decision["decision_digest"] = cross_runtime_canonical_digest(decision)
        self.decision = decision


def _status(queue: Path, key: str) -> dict[str, Any]:
    return json.loads((queue / "status" / f"{key}.json").read_text())


def test_a_chair_in_the_way_is_proposed_held_by_name_on_a_decline_and_moved_only_on_approval(
        tmp_path, monkeypatch, chairs_room) -> None:
    inputs, queue, key, record = _queued(tmp_path, monkeypatch)
    scope = inputs["owner_success_confirmation_scope"]
    sealed = tmp_path / "bundle" / "collision.usda"
    sealed_digest = _sha(sealed)

    # No ranked stance carries: the half proposes the one chair that opens a stance within reach, and
    # changes nothing else. The station is the carry search's first, on the sealed collider, chair and all.
    search = record["robot_station_search"]
    assert search["cleared"] is False and search["chosen_rank"] == 0
    assert record["robot_station"]["stance_digest"] == search["results"][0]["stance_digest"]
    assert record["robot_station"]["scene_collision_sha256"] == sealed_digest
    assert not (tmp_path / "proposed" / access.DIRECTORY).exists()
    proposal = record["robot_access_removal"]
    assert proposal["status"] == "proposed" and proposal["model"] == access.MODEL and proposal["chosen"] is False
    assert [row["obstacle_id"] for row in proposal["items"]] == [BLOCKING]
    assert proposal["items"][0]["kind"] == "chair" and proposal["support_label"] == "wooden dining table"
    assert proposal["why"] == {"reason": "no_ranked_stance_carries", "carry_cleared": False,
                               "stances_tried": len(search["results"]),
                               "reach_shortfall": search["reach_shortfall"]["status"]}
    enabled = proposal["enabled_stance"]
    assert enabled["reach"]["status"] == "within_bound" and enabled["position_world_m"][:2] == [0.5, -0.42]
    assert proposal["removal_binding"]["scene_collision_sha256"] == sealed_digest
    assert proposal["appearance"]["status"] == "unchanged"

    clock = iter(1_793_800_000.0 + 2 * 86400.0 * step for step in range(10))
    monkeypatch.setattr(access, "_now", lambda: next(clock))
    webapp = FakeWebApp()
    noop = lambda notice: (_ for _ in ()).throw(AssertionError("no threshold exchange"))  # noqa: E731

    # The first pass sends the proposal in plain terms; no decision means no change.
    first = worker.requeue_transient_blocked(queue_root=queue, website_transport=noop, robot_access_transport=webapp)
    assert first["requeued"] == [] and first["website_robot_access"][key]["status"] == "pending"
    [notice] = webapp.proposals
    assert notice["items"] == [{"target_id": BLOCKING, "label": "dining chair", "kind": "chair"}]
    assert notice["support_label"] == "wooden dining table" and notice["appearance"] == "unchanged"
    assert "bounds_world_m" not in json.dumps(notice)
    assert _status(queue, key)[access.STATUS_FIELD]["status"] == "pending"

    # The owner keeps the chair where it is: nothing is written under the intent, and the job holds by name.
    webapp.decide("declined")
    declined = worker.requeue_transient_blocked(queue_root=queue, website_transport=noop, robot_access_transport=webapp)
    assert declined["requeued"] == []
    kept = _status(queue, key)[access.STATUS_FIELD]
    assert kept["status"] == "declined" and kept["hold"] == access.DECLINED_HOLD
    assert access.confirmations(scope["directory"], scope["intent"]) == []
    assert not (queue / "pending" / f"{key}.json").exists()

    # The owner changes their mind: the approval is sealed under the intent and the request compiles again.
    webapp.decide("approved")
    approved = worker.requeue_transient_blocked(queue_root=queue, website_transport=noop, robot_access_transport=webapp)
    [approval] = access.confirmations(scope["directory"], scope["intent"])
    assert approval["removal_binding_digest"] == proposal["removal_binding_digest"]
    assert approval["items"] == proposal["removal_binding"]["items"]
    assert approval["issuer_channel"] == "website" and approval["spend_authorized"] is False
    assert approved["requeued"] == [{"request_key": key, "reason": "robot_access_confirmation",
                                     "confirmation_digests": [approval["confirmation_digest"]]}]
    assert (queue / "pending" / f"{key}.json").exists()
    assert _status(queue, key)[access.REQUEUE_FIELD] == [approval["confirmation_digest"]]
    assert access.approved_proposal(record["_compiled"]["path"], json.loads(
        (queue / "pending" / f"{key}.json").read_text()))
    # A decision for another binding is refused and applies nothing.
    with pytest.raises(ValueError, match="robot_access_decision_invalid"):
        access.validated_receipt({"schema_version": access.RECEIPT_SCHEMA, "status": "approved",
                                  "intent_id": notice["intent_id"], "proposal_digest": notice["proposal_digest"],
                                  "decision": {**webapp.decision, "removal_binding_digest": "sha256:" + "9" * 64}},
                                 notice)

    # The half is rebuilt without that chair, and only that chair, then searched and carried on the CPU again.
    rebuilt = _compile(tmp_path, monkeypatch, "approved", inputs,
                       movable_furniture=_furniture([target for _, target in CHAIRS]), robot_access_scope=scope)
    applied = rebuilt["robot_access_removal"]
    assert applied["status"] == "applied" and applied["confirmation_digest"] == approval["confirmation_digest"]
    assert applied["proposal_digest"] == notice["proposal_digest"]
    assert applied["assumption"] == access.ASSUMPTION and "hold_reason" not in applied
    assert applied["cpu_checks"]["carry_cleared"] is True
    assert applied["cpu_checks"]["reach"]["status"] == "within_bound"
    assert applied["cpu_checks"]["rest_check"]["status"] == applied["cpu_checks"]["rest_check_before"]["status"]
    assert rebuilt["robot_station_search"]["cleared"] is True
    assert rebuilt["robot_station"]["position_world_m"][:2] == [0.5, -0.42]
    # A named, derived collider: the sealed file is untouched, the chair's own faces are gone, nothing else.
    derived = tmp_path / "approved" / applied["derived_collider"]["relative_path"]
    assert _sha(sealed) == sealed_digest == applied["derived_collider"]["derived_from_sha256"]
    assert _sha(derived) == applied["derived_collider"]["sha256"] == rebuilt["robot_station"]["scene_collision_sha256"]
    [moved] = applied["derived_collider"]["items"]
    assert moved["obstacle_id"] == BLOCKING and moved["faces_removed"] == 20  # its seat and back, five sides each
    assert applied["derived_collider"]["faces_removed"] == 20
    assert access.faces_left_inside(derived, items=applied["items"], floor_z=0.0) == 0
    plan = json.loads((Path(rebuilt["base_native_packet"]["path"]).parent
                       / "native_task_arena_scene_plan.v1.json").read_text())
    [collision] = [row for row in plan["objects"] if row["semantic_role"] == "scene_collision"]
    assert collision["sha256"] == applied["derived_collider"]["sha256"]
    # The rebuilt half holds no open proposal, so nothing more is asked.
    assert access.half_proposal(rebuilt["_compiled"]["path"]) is None


def test_the_preparation_measures_the_analysis_chairs_in_the_simulator_frame_and_the_half_reads_them(tmp_path):
    from blueprint_pipeline import website_task_preparation as preparation
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests import test_website_task_preparation as fixture

    def with_chair(geometry: dict[str, Any]) -> dict[str, Any]:
        masks = fixture._masks(geometry)
        masks["targets"].append({
            "target_id": "chair-1", "semantic_label": "dining chair by the window", "segmentation_prompt": "chair",
            "target_class": "movable_object", "target_role": "obstacle", "disposition": "keep",
            "task_effect": "static_obstacle", "track": masks["targets"][0]["track"],
            "estimated_visible_bounds": fixture._bounds([0.7, 0.0, 1.2], [0.9, 0.45, 1.4])})
        masks["digest"] = canonical_digest(masks, digest_field="digest")
        return {"task_masks": masks}

    value = preparation.compile_website_scene_preparation(**fixture._arguments(tmp_path, with_chair))
    assert value["status"] == "intake_ready", value["blockers"]
    [item] = value["movable_furniture"]["items"]
    matrix = np.asarray(value["registration"]["source_to_runtime"])
    to_sim = np.asarray(value["coordinate_frame"]["runtime_to_simulator"])
    low, high = preparation._runtime_bounds(fixture._bounds([0.7, 0.0, 1.2], [0.9, 0.45, 1.4]), matrix)
    assert item["bounds_world_m"] == dict(zip(("minimum", "maximum"), preparation._runtime_bounds(
        {"minimum": low, "maximum": high}, to_sim)))
    assert item["obstacle_id"] == "chair-1" and item["kind"] == "chair"
    assert value["movable_furniture"]["physical_measurement_proven"] is False
    # A preparation with no chair or stool records nothing, as before.
    (tmp_path / "plain").mkdir()
    assert "movable_furniture" not in fixture._compile(tmp_path / "plain")

    # The team half reads the record only from this task's own preparation.
    capture = tmp_path / "capture"
    (capture / "pipeline" / "website_scene_preparation").mkdir(parents=True)
    context = fixture._task_context()
    (capture / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    (capture / "pipeline" / "website_scene_preparation" / "preparation.json").write_text(json.dumps(value))
    task_id = "website-" + context["context_digest"][7:27]
    read = access.movable_furniture_for_capture(capture, task_id=task_id)
    assert read["status"] == "measured" and read["record"] == value["movable_furniture"]
    assert access.movable_furniture_for_capture(capture, task_id="website-other")["status"] == "preparation_not_this_task"
    assert access.movable_furniture_for_capture(tmp_path / "none", task_id=task_id)["status"] == "preparation_unreadable"
