"""Movable chairs measured when they can matter, then moved out of the Franka's way only on the owner's approval.

capture-447ae4f4's Franka could not stand within reach of its dining table, most likely because of the
chairs around it (#3355). Its preparation holds no chair: the Atlas scene review names them as one grouped
entry with 2-D boxes, and nothing has their 3-D bounds. So the half records their measurement as due, the
worker runs it as Blueprint's own preparation spend within the job's caps (the site is never asked), and
only then does the site owner decide whether to move the one in the way.

These rooms are synthetic: capture-447's oval table in one scanned mesh, as a website collider is, with ten
chairs around it. The intent is a renewal's, as capture-447's live intent is: its registration names the
renewed preparation, never the root one. The two paid calls are replaced by fakes, and nothing is spent.
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
from blueprint_pipeline import robot_access_measurement as measurement  # noqa: E402
from blueprint_pipeline import task_evaluation_scene_intake as intake  # noqa: E402
from blueprint_pipeline import team_scene_packet_worker as worker  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest  # noqa: E402
from tests import test_team_scene_packet_single_mesh_room as single  # noqa: E402
from tests.test_team_embodiment_stance_room_mesh import SUPPORT_TOP, _box, write_scanned_room  # noqa: E402

SEAT_TOP, BACK_TOP = 0.45, 0.90
SUPPORT = {"bounds_min_xyz_m": [*single.SUPPORT_MIN[:2], 0.77], "bounds_max_xyz_m": [*single.SUPPORT_MAX[:2], SUPPORT_TOP],
           "top_z_m": SUPPORT_TOP}
IDENTITY = np.eye(4).tolist()
RENEWAL = "ab" * 32


def _chair(x0: float, y0: float, back: str) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """A 45 cm chair as the scan holds it, and the analysis target naming it, with its lifted visible bounds."""

    x1, y1 = x0 + 0.45, y0 + 0.45
    seat = _box((x0, y0, 0.0), (x1, y1, SEAT_TOP), bottom=False)
    bx0, by0, bx1, by1 = {"-x": (x0, y0, x0 + 0.05, y1), "+x": (x1 - 0.05, y0, x1, y1),
                          "-y": (x0, y0, x1, y0 + 0.05), "+y": (x0, y1 - 0.05, x1, y1)}[back]
    rest = _box((bx0, by0, SEAT_TOP), (bx1, by1, BACK_TOP), bottom=False)
    mesh = (np.vstack([seat[0], rest[0]]), np.vstack([seat[1], rest[1] + len(seat[0])]))
    target = {"target_id": f"chair_{back}_{x0:+.2f}_{y0:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_"),
              "semantic_label": "dining chair", "segmentation_prompt": "wooden chair", "target_class": "movable_object",
              "target_role": "obstacle", "disposition": "keep", "task_effect": "static_obstacle",
              "decision_reason": "stands beside the table", "confidence": 0.9,
              "spatial_evidence": [{"timestamp_seconds": 1.0, "box_xywh_normalized": [0.1, 0.1, 0.2, 0.2]}],
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
# capture-447's review: the tray, the table, and the chairs grouped as one entry with boxes in two frames.
REVIEW = {"automation": {"objects": [
    {"target_id": "wooden_tray", "label": "wooden tray", "segmentation_prompt": "wooden tray", "disposition": "remove",
     "regions": [{"frame_id": "f1"}]},
    {"target_id": "dining_table", "label": "dining table", "segmentation_prompt": "wooden table",
     "disposition": "keep", "regions": [{"frame_id": "f1"}]},
    {"target_id": "dining_chairs", "label": "dining_chairs", "segmentation_prompt": "wooden chair",
     "disposition": "keep", "regions": [{"frame_id": "f1"}, {"frame_id": "f2"}]}]}}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def renewed_lineage(tmp_path: Path, scope: dict[str, Any], monkeypatch: Any) -> Path:
    """The intent's source registration naming a renewal's preparation, as capture-447's live intent has it."""

    monkeypatch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
    intent = scope["intent"]
    capture = tmp_path / "pubsub" / "scenes" / "site-1" / "captures" / "walkthrough-1"
    root = capture / "pipeline" / "website_scene_preparation"
    context = {"schema_version": "website_site_task_context.v1", "request_id": "req-1", "scene_id": "site-1",
               "capture_id": "walkthrough-1", "description": "Move the wooden tray to the other end of the table.",
               "confirmed": True}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    parent = {"schema_version": "website_scene_preparation.v1", "intake_request": {"submission_id": "the-parent"},
              "binding": {"task_context_digest": context["context_digest"]}}
    parent["digest"] = canonical_digest(parent, digest_field="digest")
    _write(root / "preparation.json", parent)
    preparation = {"schema_version": "website_scene_preparation.v1", "status": "intake_ready", "blockers": [],
                   "intake_request": intent["request"],
                   "binding": {"task_context_digest": context["context_digest"],
                               "source_geometry_digest": "sha256:" + "6" * 64},
                   "registration": {"source_to_runtime": IDENTITY},
                   "coordinate_frame": {"runtime_to_simulator": IDENTITY, "scale_authority": "registration_estimate"},
                   "website_preparation_renewal": {"parent_preparation_digest": parent["digest"],
                                                   "parent_intent_id": "the-parent-intent",
                                                   "authority_digest": "sha256:" + RENEWAL}}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    renewal = root / "renewals" / RENEWAL
    paths = {"preparation": _write(renewal / "preparation.json", preparation),
             "task_context": _write(renewal / "task_context.json", context),
             "runtime_inputs": _write(renewal / "native" / "runtime_inputs.json", {"schema_version": "fixture"})}
    registration = {"schema_version": "website_scene_source_registration.v1",
                    "request_digest": cross_runtime_canonical_digest(intent["request"]),
                    "references": {name: _record(path) for name, path in paths.items()},
                    "provider_mutation_performed": False, "execution_authority_granted": False,
                    "claim_ceiling": "development_only"}
    registration["registration_digest"] = canonical_digest(registration, digest_field="registration_digest")
    bindings = Path(scope["directory"]).parent.parent / "website-source-bindings"
    _write(bindings / (cross_runtime_canonical_digest(intent["request"])[7:] + ".json"), registration)
    _write(capture / "pipeline" / "atlas" / "atlas_scene_review.json", REVIEW)
    video = capture / "raw" / "walkthrough.mov"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fixture video bytes")
    # The video's metadata as ffprobe reads it: 30 s, 900 frames, analysed in one static pass.
    monkeypatch.setattr(measurement, "measurement_sources", lambda capture_root, current: {
        "video_path": str(video), "source_video_digest": _sha(video), "processing": "static",
        "duration_seconds": 30.0, "frames": 900})
    return capture


class PaidCalls:
    """Fakes of the two paid calls, recording whether anything would have been bought."""

    def __init__(self, targets: list[dict[str, Any]]) -> None:
        self.targets, self.calls = targets, []

    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("analysis", kwargs["maximum_cost_usd"]))
        return {"targets": [{key: value for key, value in row.items() if key != "estimated_visible_bounds"}
                            for row in self.targets], "blockers": []}

    def segment(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("segmentation", kwargs["maximum_cost_usd"], [row["text"] for row in kwargs["prompts"]]))
        return {"registry": [], "tracks": [], "reserved_cost_usd": 0.18 * len(kwargs["prompts"])}

    def lift(self, targets: list[dict[str, Any]], **_: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_id = {row["target_id"]: row for row in self.targets}
        return [{**row, "estimated_visible_bounds": by_id[row["target_id"]]["estimated_visible_bounds"]}
                for row in targets], []


@pytest.fixture
def chairs_room(monkeypatch) -> None:
    """The capture-447 bundle's collider with the chairs in its one scanned mesh."""

    write = single.write_scanned_room
    monkeypatch.setattr(single, "write_scanned_room",
                        lambda path, **options: write(path, parts=[mesh for mesh, _ in CHAIRS], **options))


def _compile(tmp_path: Path, name: str, inputs: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    from blueprint_pipeline.team_scene_packet import materialize_team_scene

    known = measurement.furniture_for_scope(scope, revision_digest=inputs["revision"]["revision_digest"])
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / name, movable_furniture=known,
                                      robot_access_scope=scope)
    return {**json.loads(Path(compiled["path"]).read_text()), "_compiled": compiled}


def _queued(tmp_path: Path, monkeypatch: Any, inputs: dict[str, Any], record: dict[str, Any]) -> tuple[Path, str, dict]:
    """A completed request whose compiled half is ``record``, bound to its own intent."""

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
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    queue = worker._queue(tmp_path / "queue")
    key = worker.request_key(request)
    (queue / "processed" / f"{key}.json").write_text(json.dumps(request))
    _complete(queue, key, record)
    return queue, key, request


def _complete(queue: Path, key: str, record: dict[str, Any]) -> None:
    """The request's status after a compile, as the worker writes it."""

    path = queue / "status" / f"{key}.json"
    prior = json.loads(path.read_text()) if path.is_file() else {}
    kept = {field: prior[field] for field in (access.REQUEUE_FIELD, access.STATUS_FIELD) if field in prior}
    path.write_text(json.dumps({"schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "completed",
                                "blockers": [], "team_scene_inputs_path": record["_compiled"]["path"], **kept}))
    # The worker claims a pending request and files it back under processed/ when it finishes.
    if (queue / "pending" / f"{key}.json").exists():
        (queue / "pending" / f"{key}.json").replace(queue / "processed" / f"{key}.json")


class FakeWebApp:
    """The WebApp side: it keeps each proposal and answers with the owner's decision on it."""

    def __init__(self) -> None:
        self.proposals: list[dict[str, Any]] = []
        self.decisions: dict[str, dict[str, Any]] = {}

    def __call__(self, notice: dict[str, Any]) -> dict[str, Any]:
        notice = json.loads(json.dumps(notice))
        assert notice["proposal_digest"] == cross_runtime_canonical_digest(notice, digest_field="proposal_digest")
        self.proposals.append(notice)
        decision = self.decisions.get(notice["proposal_digest"])
        return {"schema_version": access.RECEIPT_SCHEMA, "status": decision["decision"] if decision else "pending",
                "intent_id": notice["intent_id"], "proposal_digest": notice["proposal_digest"], "decision": decision}

    def decide(self, kind: str) -> dict[str, Any]:
        notice = self.proposals[-1]
        decision = {"schema_version": access.DECISION_SCHEMA, "intent_id": notice["intent_id"],
                    "intent_digest": notice["intent_digest"], "request_id": "req-1", "capture_id": "cap-1",
                    "proposal_digest": notice["proposal_digest"],
                    "configured_scene_revision_digest": notice["configured_scene_revision_digest"],
                    "binding_digest": notice["binding_digest"], "decision": kind,
                    "decided_via": "owner_job_link", "decided_at": "2026-11-04T09:00:00.000Z"}
        decision["decision_digest"] = cross_runtime_canonical_digest(decision)
        self.decisions[notice["proposal_digest"]] = decision
        return decision


def _status(queue: Path, key: str) -> dict[str, Any]:
    return json.loads((queue / "status" / f"{key}.json").read_text())


def test_chairs_are_measured_without_asking_the_site_then_moved_only_on_the_owners_approval(
        tmp_path, monkeypatch, chairs_room) -> None:
    import functools

    inputs = single._inputs(tmp_path, monkeypatch)
    scope = inputs["owner_success_confirmation_scope"]
    capture = renewed_lineage(tmp_path, scope, monkeypatch)
    paid = PaidCalls([target for _, target in CHAIRS])
    monkeypatch.setattr(measurement, "lift", paid.lift)
    clock = iter(1_793_800_000.0 + 2 * 86400.0 * step for step in range(20))
    monkeypatch.setattr(access, "_now", lambda: next(clock))
    webapp = FakeWebApp()
    noop = lambda notice: (_ for _ in ()).throw(AssertionError("no threshold exchange"))  # noqa: E731

    def passes() -> dict[str, Any]:
        return worker.requeue_transient_blocked(queue_root=queue, website_transport=noop, robot_access_transport=webapp)

    # 1. No stance carries, the nearer ones were refused on seats and backs, the review names the chairs, and
    #    nothing measured them: the half records the measurement as due, with the most it may reserve.
    first = _compile(tmp_path, "measurement-due", inputs, scope)
    due = first["robot_access_removal"]
    assert first["robot_station_search"]["cleared"] is False
    assert due["status"] == "measurement_due" and due["trigger"]["plausible"] is True
    assert due["items"] == [{"obstacle_id": "dining_chairs", "label": "dining_chairs", "kind": "chair"}]
    binding = due["binding"]
    # Bound to the intent's current preparation: the renewal's, never the root one.
    renewed = json.loads((capture / "pipeline/website_scene_preparation/renewals" / RENEWAL / "preparation.json")
                         .read_text())
    assert binding["lineage"]["preparation_digest"] == renewed["digest"]
    assert binding["lineage"]["renewal"]["parent_intent_id"] == "the-parent-intent"
    assert binding["lineage"]["intent_id"] == scope["intent"]["intent_id"]
    # At most: one analysis turn (USD 0.04 for 30 s) and two SAM concepts over 900 frames (USD 0.18 each).
    assert binding["analysis"]["maximum_cost_usd"] == 0.04 and binding["segmentation"]["maximum_cost_usd"] == 0.36
    assert due["maximum_cost_usd"] == binding["maximum_cost_usd"] == 0.4
    assert paid.calls == []
    queue, key, request = _queued(tmp_path, monkeypatch, inputs, first)

    # 2. The site is asked nothing; the request returns to pending once, for the worker to measure.
    first_pass = passes()
    assert webapp.proposals == [] and first_pass["website_robot_access"] == {}
    assert first_pass["requeued"] == [{"request_key": key, "reason": "robot_access_measurement_due",
                                       "binding_digest": due["binding_digest"]}]
    assert _status(queue, key)[worker.ROBOT_ACCESS_MEASUREMENT_REQUEUE_FIELD] == [due["binding_digest"]]
    assert measurement.measurement_due(first["_compiled"]["path"])["binding_digest"] == due["binding_digest"]

    # 3. The worker measures. A refused reservation (the job's caps or window) holds by name, spends nothing
    #    and changes nothing; a measurement of anything but the half's binding is refused before any call.
    def refused(**_: Any) -> Any:
        raise ValueError("website_preparation_spend_receipt_invalid")

    monkeypatch.setattr(measurement, "run_measurement", functools.partial(
        measurement.run_measurement, analyze=refused, segment=paid.segment, geometry=lambda *_: {"frames": []}))
    held = worker._measure_furniture(request, due["binding_digest"])
    assert held["status"] == "held"
    assert held["hold"] == "team_scene_robot_access_measurement_held:website_preparation_spend_receipt_invalid"
    assert held["binding_digest"] == due["binding_digest"]
    assert worker._measure_furniture(request, "sha256:" + "9" * 64)["hold"] == (
        "team_scene_robot_access_measurement_held:robot_access_measurement_binding_changed")
    assert paid.calls == [] and not (capture / "pipeline" / "robot_access" / renewed["digest"][7:23]
                                     / "movable_furniture.v1.json").exists()
    # Within the caps: one analysis and one SAM concept for the chairs, kept as a sidecar of the renewal's
    # preparation with what each call reserved.
    monkeypatch.setattr(measurement, "run_measurement", functools.partial(
        measurement.run_measurement.func, analyze=paid.analyze, segment=paid.segment,
        geometry=lambda *_: {"frames": []}))
    measured = worker._measure_furniture(request, due["binding_digest"])
    assert measured["status"] == "measured" and measured["reserved_cost_usd"] == 0.22
    assert paid.calls == [("analysis", 0.04), ("segmentation", 0.36, ["wooden chair"])]
    path = capture / "pipeline" / "robot_access" / renewed["digest"][7:23] / "movable_furniture.v1.json"
    sidecar = json.loads(path.read_text())
    assert sidecar["sidecar_digest"] == measured["sidecar_digest"] and sidecar["lineage"] == binding["lineage"]
    assert sidecar["measurement"]["binding_digest"] == due["binding_digest"]
    assert sidecar["analysis"]["reserved_cost_usd"] == 0.04 and sidecar["segmentation"]["reserved_cost_usd"] == 0.18
    assert len(sidecar["movable_furniture"]["items"]) == len(CHAIRS)
    # The sealed preparations are untouched; the sidecar is beside them, read through the registration.
    assert canonical_digest(renewed, digest_field="digest") == renewed["digest"] == json.loads(
        (capture / "pipeline/website_scene_preparation/renewals" / RENEWAL / "preparation.json").read_text())["digest"]
    assert "movable_furniture" not in renewed
    # Measuring again buys nothing.
    assert worker._measure_furniture(request, due["binding_digest"])["sidecar_digest"] == sidecar["sidecar_digest"]
    assert len(paid.calls) == 2

    # 4. With the chairs measured, the half proposes moving the one chair that opens a stance within reach.
    second = _compile(tmp_path, "proposes-a-move", inputs, scope)
    proposal = second["robot_access_removal"]
    assert proposal["status"] == "proposed" and proposal["sidecar_digest"] == sidecar["sidecar_digest"]
    assert [row["obstacle_id"] for row in proposal["items"]] == [BLOCKING]
    assert proposal["enabled_stance"]["reach"]["status"] == "within_bound"
    assert proposal["enabled_stance"]["position_world_m"][:2] == [0.5, -0.42]
    assert second["robot_station"]["position_world_m"] == first["robot_station"]["position_world_m"]
    _complete(queue, key, second)
    asked_to_move = passes()["website_robot_access"][key]
    assert asked_to_move["status"] == "pending", asked_to_move
    [move] = webapp.proposals
    assert "maximum_cost_usd" not in move and "question" not in move
    assert move["items"] == [{"target_id": BLOCKING, "label": "dining chair", "kind": "chair"}]
    assert "bounds_world_m" not in json.dumps(move)

    # 5. A decline holds by name; an approval rebuilds the half without that chair, checked on the CPU.
    webapp.decide("declined")
    passes()
    assert _status(queue, key)[access.STATUS_FIELD]["hold"] == access.DECLINED_HOLD
    webapp.decide("approved")
    moved_on = passes()
    assert [row["reason"] for row in moved_on["requeued"]] == ["robot_access_confirmation"], json.dumps(moved_on["website_robot_access"])
    [approval] = access.confirmations(scope["directory"], scope["intent"])
    assert approval["spend_authorized"] is False
    third = _compile(tmp_path, "moves-the-chair", inputs, scope)
    applied = third["robot_access_removal"]
    assert applied["status"] == "applied" and applied["confirmation_digest"] == approval["confirmation_digest"]
    assert applied["cpu_checks"]["carry_cleared"] is True and applied["cpu_checks"]["reach"]["status"] == "within_bound"
    assert applied["cpu_checks"]["rest_check"]["status"] == applied["cpu_checks"]["rest_check_before"]["status"]
    assert third["robot_station_search"]["cleared"] is True
    assert third["robot_station"]["position_world_m"][:2] == [0.5, -0.42]
    sealed = tmp_path / "bundle" / "collision.usda"
    derived = tmp_path / "moves-the-chair" / applied["derived_collider"]["relative_path"]
    assert _sha(sealed) == applied["derived_collider"]["derived_from_sha256"]
    assert _sha(derived) == third["robot_station"]["scene_collision_sha256"]
    [moved] = applied["derived_collider"]["items"]
    assert moved["obstacle_id"] == BLOCKING and moved["faces_removed"] == 20  # its seat and back, five sides each
    assert access.faces_left_inside(derived, items=applied["items"], floor_z=0.0) == 0
    assert access.half_proposal(third["_compiled"]["path"]) is None


def test_no_proposal_names_what_the_analysis_does_not_call_movable_furniture(tmp_path, monkeypatch, chairs_room):
    # The measuring analysis kept the chairs as fixed clutter: no SAM concept is bought, nothing is named, and
    # the half proposes nothing.
    inputs = single._inputs(tmp_path, monkeypatch)
    scope = inputs["owner_success_confirmation_scope"]
    renewed_lineage(tmp_path, scope, monkeypatch)
    offer = measurement.measurement_offer(scope, revision_digest=inputs["revision"]["revision_digest"])
    paid = PaidCalls([{**target, "target_class": "fixed_clutter"} for _, target in CHAIRS])
    sidecar = measurement.run_measurement(scope, binding_digest=offer["binding_digest"],
                                          revision_digest=inputs["revision"]["revision_digest"],
                                          analyze=paid.analyze, segment=paid.segment)
    assert paid.calls == [("analysis", 0.04)] and sidecar["movable_furniture"] is None
    record = _compile(tmp_path, "fixed", inputs, scope)
    assert record["robot_access_removal"]["status"] == "not_proposed"
    assert record["robot_access_removal"]["reason"] == "no_named_movable_furniture"
    assert "movable_obstacle_counterfactual" not in record["robot_station"]["search"]


def test_only_kept_chairs_and_stools_are_named_and_the_support_is_protected_by_height() -> None:
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
            target(target_id="pulled_up_chair", semantic_label="chair pulled up to the table",
                   estimated_visible_bounds={"minimum": [0.1, -0.3, 0.0], "maximum": [0.6, 0.1, 0.9]}),
            target(target_id="chair_unmeasured", semantic_label="armchair", estimated_visible_bounds=None)]
    record = access.movable_furniture_record(rows, lambda bounds: (bounds["minimum"], bounds["maximum"]))
    assert [row["obstacle_id"] for row in record["items"]] == ["bar_stool", "pulled_up_chair"]
    assert {row["target_id"]: row["reason"] for row in record["excluded"]} == {
        "fixed_seat": "analysis_class_not_movable_object", "chair_on_support": "analysis_role_not_obstacle_or_background",
        "chair_unmeasured": "bounds_unmeasured"}
    # A chair pulled up to the table meets it in plan, and is still named: the table is protected by height.
    named, refused = access.eligible_obstacles(record, supports=[SUPPORT])
    assert [row["obstacle_id"] for row in named] == ["bar_stool", "pulled_up_chair"] and refused == []
    assert named[1]["protected_world_m"] == [{"minimum": [*single.SUPPORT_MIN[:2], SUPPORT_TOP - 0.05],
                                              "maximum": [*single.SUPPORT_MAX[:2], SUPPORT_TOP + 3.0]}]
    assert access.eligible_obstacles(record, supports=[])[1][0]["reason"] == "support_unknown"


def test_a_chair_pulled_up_to_the_table_moves_without_any_of_the_table(tmp_path) -> None:
    # The chair's seat reaches under the table's edge and its measured box takes in the table's side. Its seat
    # and its back outside the table go; nothing of the table at its own height inside its footprint does, and
    # the station search's counterfactual holds exactly the same faces.
    from blueprint_pipeline.team_embodiment_stance import _inside, _world_triangles

    x0, y0, x1, y1 = 0.10, -0.30, 0.55, 0.15
    seat = _box((x0, y0, 0.0), (x1, y1, SEAT_TOP), bottom=False)
    back = _box((x1 - 0.05, y0, SEAT_TOP), (x1, y1, BACK_TOP), bottom=False)
    room = write_scanned_room(tmp_path / "pulled-up.usda", parts=[seat, back])
    item = {"obstacle_id": "pulled_up", "label": "chair", "kind": "chair",
            "bounds_world_m": {"minimum": [x0 - 0.01, y0 - 0.01, -0.01], "maximum": [x1 + 0.01, y1 + 0.01, BACK_TOP + 0.01]}}
    [named], _ = access.eligible_obstacles({"items": [item]}, supports=[SUPPORT])
    vertices, faces = _world_triangles(room, [0.0, 0.0, 0.0])
    corners = vertices[faces]
    low, high = corners.min(axis=1), corners.max(axis=1)
    inside = (low >= named["bounds_world_m"]["minimum"]).all(axis=1) & (high <= named["bounds_world_m"]["maximum"]).all(axis=1)
    table = inside & (high[:, 2] >= SUPPORT_TOP - 0.05) & (low[:, 0] <= single.SUPPORT_MAX[0])
    assert table.sum() > 0  # the table's side lies wholly inside the chair's measured box
    # The search measures what rises above the floor band; the floor under the chair is never an obstacle.
    rises = high[:, 2] >= access.FLOOR_BAND_M
    held = (_inside([named], low, high) >= 0) & rises
    assert not (held & table).any() and held.sum() == 20
    written = access.write_collider_without(room, tmp_path / "derived.usdc", items=[named], floor_z=0.0)
    assert written["faces_removed"] == 20
    derived_vertices, derived_faces = _world_triangles(tmp_path / "derived.usdc", [0.0, 0.0, 0.0])
    assert len(derived_faces) == len(faces) - 20
    kept = {tuple(np.round(row, 6).ravel()) for row in derived_vertices[derived_faces]}
    assert all(tuple(np.round(row, 6).ravel()) in kept for row in corners[table])
    assert access.faces_left_inside(tmp_path / "derived.usdc", items=[named], floor_z=0.0) == 0


def test_the_lift_takes_each_pieces_own_instance_and_never_guesses(tmp_path) -> None:
    from blueprint_pipeline.website_task_masks import estimate_target_bounds
    from tests.test_website_task_preparation import HEIGHT, WIDTH, _source_geometry

    geometry = _source_geometry(tmp_path)
    registry = [{"source_frame_id": frame["frame_id"], "decoded_pts_seconds": frame["timestamp_seconds"],
                 "width": WIDTH, "height": HEIGHT} for frame in geometry["frames"]]

    def track(track_id: str, rows: range, columns: range) -> dict[str, Any]:
        runs = [{"start": row * WIDTH + columns.start, "length": len(columns), "probability": 0.9} for row in rows]
        return {"track_id": track_id, "label": "chair-a", "observations": [
            {"source_frame_id": "frame-0", "height": HEIGHT, "width": WIDTH, "runs": runs}]}

    near, far = track("near", range(20, 25), range(21, 25)), track("far", range(2, 6), range(2, 6))
    prompts = [{"prompt_id": "chair-a", "text": "wooden chair", "output_label": "chair-a", "anchor_frame_index": 0}]

    def piece(target_id: str, box: list[float], prompt: str = "wooden chair") -> dict[str, Any]:
        return {"target_id": target_id, "segmentation_prompt": prompt, "semantic_label": "chair",
                "spatial_evidence": [{"timestamp_seconds": 0.0, "box_xywh_normalized": box}]}

    measured, left = measurement.lift(
        [piece("by_the_window", [21 / WIDTH, 20 / HEIGHT, 4 / WIDTH, 5 / HEIGHT]),
         piece("unclear", [0.0, 0.0, 1.0, 1.0]), piece("another_kind", [0.1, 0.1, 0.1, 0.1], prompt="black stool")],
        registry=registry, tracks=[near, far], prompts=prompts, geometry=geometry)
    [chair] = measured
    assert chair["target_id"] == "by_the_window" and chair["track_id"] == "near"
    assert chair["estimated_visible_bounds"] == estimate_target_bounds(
        {**near, "label": "by_the_window"}, geometry["frames"])
    assert left == [{"target_id": "unclear", "reason": "instance_ambiguous"},
                    {"target_id": "another_kind", "reason": "segmentation_concept_over_the_bound_count"}]
    # A geometry frame the tracks never saw at that time is refused, not lifted.
    with pytest.raises(ValueError, match="geometry_frame_mapping_invalid"):
        measurement.lift([], registry=registry[:1], tracks=[], prompts=prompts, geometry=geometry)


def test_measuring_is_asked_only_when_something_off_the_supports_height_blocks_the_nearer_stances() -> None:
    def stances(**refusals: Any) -> list[dict[str, Any]]:
        return [{"search": {"nearer_refusals": refusals}}]

    assert measurement.plausibly_blocked(stances(stances=40, blocker_tops={"below_mount": 3, "at_mount": 37,
                                                                           "above_mount": 0}))["plausible"]
    assert not measurement.plausibly_blocked(stances(stances=40, by_reason={"occupied": 40},
                                                     blocker_tops={"below_mount": 0, "at_mount": 40,
                                                                   "above_mount": 0}))["plausible"]
    assert measurement.plausibly_blocked(stances(stances=4, by_reason={"clips": 4}))["plausible"]
    assert not measurement.plausibly_blocked([])["plausible"]


def test_the_sidecar_is_the_current_preparations_and_a_stale_one_is_never_read(tmp_path, monkeypatch) -> None:
    from tests.test_task_evaluation_owner_success_confirmation import publish

    job = publish(tmp_path, monkeypatch)
    scope = {"intent": job.intent, "directory": job.directory}
    assert measurement.furniture_for_scope(scope, revision_digest="x")["reason"].startswith("preparation_unresolved")
    capture = renewed_lineage(tmp_path, scope, monkeypatch)
    current = measurement.current_preparation(scope)
    assert current["preparation_path"].endswith(f"renewals/{RENEWAL}/preparation.json")
    assert current["capture_root"] == str(capture)
    # A sidecar written for the root preparation (another digest) is not this intent's.
    root = json.loads((capture / "pipeline/website_scene_preparation/preparation.json").read_text())
    stale = capture / "pipeline" / "robot_access" / root["digest"][7:23] / "movable_furniture.v1.json"
    value = {"schema_version": measurement.SIDECAR_SCHEMA, "lineage": {**current["lineage"],
                                                                       "preparation_digest": root["digest"]},
             "movable_furniture": {"items": []}, "sidecar_digest": ""}
    value["sidecar_digest"] = canonical_digest(value, digest_field="sidecar_digest")
    _write(stale, value)
    assert measurement.furniture_for_scope(scope, revision_digest="x") == {
        "status": "not_measured", "reason": "no_sidecar_for_the_current_preparation"}
    # Nor is one for this preparation bound to another intent.
    other = capture / "pipeline" / "robot_access" / current["lineage"]["preparation_digest"][7:23] / "movable_furniture.v1.json"
    _write(other, {**value, "lineage": {**current["lineage"], "intent_id": "another-intent"}})
    assert measurement.furniture_for_scope(scope, revision_digest="x")["status"] == "not_measured"


def test_a_refused_measurement_is_a_typed_hold_and_never_compiles_the_half_again(monkeypatch) -> None:
    calls: list[Any] = []

    def failing(scope: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["binding_digest"])
        raise ValueError("website_preparation_spend_receipt_invalid")

    monkeypatch.setattr(measurement, "run_measurement", failing)
    monkeypatch.setattr(worker, "owner_success_confirmation_scope", lambda request: ({"intent": {}, "directory": "/x"}, None))
    outcome = worker._measure_furniture({"scene_revision_digest": "sha256:" + "1" * 64}, "sha256:" + "2" * 64)
    assert outcome["status"] == "held" and calls == ["sha256:" + "2" * 64]
    assert outcome["hold"] == "team_scene_robot_access_measurement_held:website_preparation_spend_receipt_invalid"
    # The same binding is tried once: a held measurement is not requeued again.
    status = {"status": "completed", "team_scene_inputs_path": "/nowhere",
              worker.ROBOT_ACCESS_MEASUREMENT_REQUEUE_FIELD: ["sha256:" + "2" * 64]}
    monkeypatch.setattr(measurement, "measurement_due", lambda path: {"binding_digest": "sha256:" + "2" * 64})
    assert worker._measurement_due_untried(status) is None
    assert worker._measurement_due_untried({**status, worker.ROBOT_ACCESS_MEASUREMENT_REQUEUE_FIELD: []}) == (
        "sha256:" + "2" * 64)
