"""A website job's scene package: never refused for the idle arm alone, never paying for a probe, and truthful.

Owner decision 2026-11-15: Blueprint builds the scene package deployment companies download and runs no test on it
(``package_only``). Real compile (``team_scene_packet.materialize_team_scene``) of the rigid fixture revision, real
stance search on scanned-room fixtures, real worker, scene-stage pass and download, with their own tests' fakes.
Hermetic: no network, nothing paid.
"""
# Covers: src/blueprint_pipeline/team_scene_robot_absent.py
# Covers: src/blueprint_pipeline/team_scene_packet.py
# Covers: src/blueprint_pipeline/team_scene_packet_remote.py
# Covers: src/blueprint_pipeline/team_embodiment_stance.py
# Covers: src/blueprint_pipeline/scene_stage_maintenance.py
# Covers: src/blueprint_pipeline/website_company_simulation.py
from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import team_scene_robot_absent as absent
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_embodiment import TeamEmbodimentError
from blueprint_pipeline.team_scene_packet import TeamScenePacketError, read_team_scene_inputs
from tests import test_team_scene_packet as packet_tests
from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, STATIC, _rewrite
from tests.test_website_company_simulation import Job, World, _package

pytest.importorskip("pxr")

WEBSITE = {"package_only": True}
NO_SIMULATOR = "Blueprint did not run this scene in Isaac Sim or any other simulator."
STATIC_PASSED = ("The object's file passed Blueprint's static USD physics check (one rigid body, collision shapes, "
                 "mass and friction within set ranges).")
ROOM_MESH = "team_embodiment_room_mesh_no_clear_stance"
NO_STANCE = "team_embodiment_task_motion_no_clear_stance"
FRAMING = "native_task_arena_camera_framing_infeasible:external"


# --------------------------------------------------------------------------------------------------- builders


def _said(readme: str, sentence: str) -> bool:
    return sentence in " ".join(readme.split())


def _not_run(monkeypatch: pytest.MonkeyPatch, *, static_passed: bool = False) -> None:
    """The fixture revision as the no-GPU chain publishes it (``not_run_receipt``), bound to its stage-4 record;
    with ``static_passed`` that record is the one a rigid object's static USD check writes."""
    from blueprint_pipeline.website_scene_without_simulator import not_run_receipt

    original = packet_tests.rigid_case

    def rigid_case(tmp_path: Path, **options: Any):
        configured, references, docs, spec = original(tmp_path, **options)
        if static_passed:
            # As a real stage-4 record: with the object's collision bounds, so the task is seated on the collider.
            observed = {**docs[STATIC]["observed_structure"], "collision_bounds_body_frame_m": {
                "minimum": [-0.0638749, -0.0658936, 0.0], "maximum": [0.0638749, 0.0658936, 0.127638]}}
            static = _sealed_doc({**docs[STATIC], "authored_structure_statically_qualified": True,
                                  "structural_findings": [], "observed_structure": observed})
            _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=STATIC,
                     document=static)
            docs = {**docs, STATIC: static}
        static_bytes = Path(references[STATIC]["materialized_path"]).read_bytes()
        document = not_run_receipt(replacement_identity=docs[NATIVE_IMPORT]["replacement_identity"],
                                   asset_digest=configured["replacement"]["asset"]["digest"],
                                   static_qualification_digest="sha256:" + hashlib.sha256(static_bytes).hexdigest())
        _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=NATIVE_IMPORT,
                 document=document)
        return configured, references, {**docs, NATIVE_IMPORT: document}, spec

    monkeypatch.setattr(packet_tests, "rigid_case", rigid_case)


def _sealed_doc(value: dict[str, Any], field: str = "result_digest") -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _stance_raises(monkeypatch, error: BaseException) -> None:
    from blueprint_pipeline import team_embodiment_stance as stance

    def raise_(**_):
        raise error
    monkeypatch.setattr(stance, "ground_stance", raise_)


def _packet_raises(monkeypatch, error: BaseException, *, cause: BaseException | None = None,
                   partial: bool = False) -> None:
    from blueprint_pipeline import native_task_arena_packet as arena

    def fail(*, output_dir, **_):
        if partial:
            Path(output_dir).mkdir(parents=True)
        raise error from cause
    monkeypatch.setattr(arena, "materialize_native_task_arena_packet", fail)


def _half(tmp_path: Path, **flags: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from blueprint_pipeline.team_scene_packet import materialize_team_scene

    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = packet_tests._scene(tmp_path)
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "team-scene", **flags)
    return compiled, json.loads(Path(compiled["path"]).read_text())


def _download(tmp_path: Path, record_path: str) -> tuple[dict[str, Any], dict[str, str]]:
    job = Job(tmp_path / "download")
    status = next((job.queue / "status").glob("*.json"))
    status.write_text(json.dumps({**json.loads(status.read_text()), "team_scene_inputs_path": record_path}))
    result = World(job).publish()
    if result["status"] != "published":
        return result, {}
    with zipfile.ZipFile(_package(job)) as archive:
        return result, {name: archive.read(name).decode("utf-8", "replace") for name in archive.namelist()
                        if name.endswith((".txt", ".json", ".usda"))}


def _publish_with_record(job: Job, **fields: Any) -> dict[str, str]:
    """The download of the test job whose half's sealed record carries ``fields``."""
    record = _sealed_doc({"task_success_contract": {"criteria": job.criteria}, **fields}, "inputs_digest")
    (job.scene / "team_scene_inputs.v1.json").write_text(json.dumps(record))
    result = World(job).publish()
    assert result["status"] == "published", result
    with zipfile.ZipFile(_package(job)) as archive:
        return {"README.txt": archive.read("README.txt").decode(), "revision": result["simulation_revision"]}


@pytest.fixture
def job(tmp_path):
    return Job(tmp_path)


# ------------------------------------------------------------------------------- which codes are the arm's


@pytest.mark.parametrize(("codes", "expected"), [
    ("team_embodiment_free_base_no_clear_stance,team_embodiment_task_motion_no_clear_stance", "same"),
    (FRAMING, "same"), ("team_scene_droid_profile_binding_invalid", "same"),
    ("team_scene_robot_access_unwritten:ValueError:bad mesh at /var/lib/blueprint/x.usdc",
     "team_scene_robot_access_unwritten:ValueError"),
    *[(code, None) for code in (
        "team_embodiment_scene_collision_unavailable", "team_embodiment_scene_collision_frame_unsupported",
        "team_embodiment_scene_collision_pose_unsupported", "team_embodiment_scene_floor_not_found",
        "team_embodiment_task_bounds_unavailable", "team_embodiment_stance_height_invalid", ROOM_MESH,
        "team_scene_station_inputs_invalid", "team_scene_carry_inputs_invalid", "team_scene_runtime_source_invalid",
        "team_scene_configured_bundle_binding_invalid", "team_scene_station_inputs_invalid_anything",
        "team_embodiment_anything_else", "native_task_arena_camera_framing_infeasible:",
        "native_task_arena_camera_framing_infeasible_anything:external", "xnative_task_arena_camera_framing_infeasible:x",
        "native_task_runtime_camera_missingx:wrist", "team_scene_robot_access_unwritten_too:OSError",
        "native_task_appearance_frame_excludes_scene_position:task_object", f"{NO_STANCE},team_scene_runtime_source_invalid",
        "")],
], ids=lambda value: str(value)[:60])
def test_only_the_arms_own_codes_are_let_through(codes, expected):
    for error in (TeamScenePacketError(codes), TeamEmbodimentError(codes.split(",")) if codes else ValueError("")):
        assert absent.robot_only_reason(error) == (codes if expected == "same" else expected)


def _room(tmp_path: Path, kind: str) -> TeamEmbodimentError:
    from tests.test_team_embodiment_stance_room_mesh import START, _box, _search, write_scanned_room

    if kind == "no_floor":  # no floor reconstructed within 2.4 m of the task
        room = write_scanned_room(tmp_path / "collision.usda",
                                  floor_keep=lambda xy: np.hypot(xy[:, 0] - START[0], xy[:, 1] - START[1]) > 2.4)
    else:  # floor everywhere, boxes all round the task
        room = write_scanned_room(tmp_path / "collision.usda", parts=[
            _box((START[0] + dx - 0.12, START[1] + dy - 0.12, 0.0), (START[0] + dx + 0.12, START[1] + dy + 0.12, 1.2))
            for dx in np.arange(-1.7, 1.71, 0.3) for dy in np.arange(-1.7, 1.71, 0.3)
            if not (abs(dx) < 0.25 and abs(dy) < 0.25)])
    with pytest.raises(TeamEmbodimentError) as refused:
        _search(room)
    return refused.value


@pytest.mark.parametrize(("kind", "arms"), [("no_floor", False), ("crowded", True)])
def test_the_stance_search_tells_a_room_with_no_floor_from_one_with_no_room_for_the_arm(tmp_path, kind, arms):
    error = _room(tmp_path, kind)
    assert error.blockers == [ROOM_MESH] and error.arm_refused_with_floor is arms  # the code is unchanged
    wrapped = TeamScenePacketError(ROOM_MESH)
    wrapped.__cause__ = error
    for item in (error, wrapped):
        assert absent.robot_only_reason(item) == (ROOM_MESH if arms else None)


# ------------------------------------------------------------------------------------------ the compile


@pytest.mark.parametrize(("case", "outcome"), [
    ("no_clear_stance", NO_STANCE), ("stance_height", r"\Ateam_embodiment_stance_height_invalid\Z"),
    ("room_no_floor", rf"\A{ROOM_MESH}\Z"), ("value_error", r"\Ateam_scene_station_inputs_invalid\Z"),
    ("key_error", r"\Ateam_scene_station_inputs_invalid\Z"), ("type_error", r"\Ateam_scene_station_inputs_invalid\Z"),
    *[(f"scene:{code}", rf"\A{code}\Z") for code in (
        "team_embodiment_scene_collision_unavailable", "team_embodiment_scene_collision_frame_unsupported",
        "team_embodiment_scene_collision_pose_unsupported", "team_embodiment_scene_floor_not_found")]])
def test_the_stance_search_writes_the_half_without_the_arm_only_for_the_arms_reasons(tmp_path, monkeypatch, case,
                                                                                      outcome):
    errors = {"no_clear_stance": TeamEmbodimentError([NO_STANCE]),
              "stance_height": TeamEmbodimentError(["team_embodiment_stance_height_invalid"]),
              "value_error": ValueError("collider mesh could not be read"), "key_error": KeyError("minimum"),
              "type_error": TypeError("not a number")}
    error = (_room(tmp_path / "room", "no_floor") if case == "room_no_floor"
             else TeamEmbodimentError([case.split(":", 1)[1]]) if case.startswith("scene:") else errors[case])
    _stance_raises(monkeypatch, error)
    if not outcome.startswith(r"\A"):
        compiled, record = _half(tmp_path / "half", **WEBSITE)
        assert record[absent.FIELD] == {"reason": outcome, "robot_included": False, "paid_step_started": False}
        return
    with pytest.raises(TeamScenePacketError, match=outcome):
        _half(tmp_path / "half", **WEBSITE)
    assert not list((tmp_path / "half").rglob("team_scene_inputs.v1.json"))


def test_a_half_without_the_arm_is_whole_and_published(tmp_path, monkeypatch):
    _stance_raises(monkeypatch, TeamEmbodimentError([NO_STANCE]))
    with pytest.raises(TeamScenePacketError, match=NO_STANCE):
        _half(tmp_path / "operator")  # an operator's compile still refuses by name
    compiled, record = _half(tmp_path / "website", **WEBSITE)
    scene = Path(compiled["path"]).parent
    assert compiled["robot_station"] is None and record["robot_station"] is None
    assert not (scene / "native-task-packet").exists()
    plan = json.loads((scene / absent.DIRECTORY / absent.PLAN_FILENAME).read_text())
    assert (plan["robot"], plan["cameras"], plan["robot_absent"]["reason"]) == (None, [], NO_STANCE)
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest") == record["plan_digest"]
    assert {row["semantic_role"]: row["visible"] for row in plan["objects"]}["scene_collision"] is False
    read_team_scene_inputs(compiled["path"])
    result, members = _download(tmp_path, compiled["path"])
    assert result["status"] == "published", result
    assert json.loads(members["success_rule.json"])["criteria"]
    assert "No robot is included: bring your own." in members["README.txt"]


def test_a_y_up_room_collider_is_refused_by_the_real_stance_search(tmp_path, monkeypatch):
    real_bundle = packet_tests._bundle

    def bundle_with_y_up_collision(tmp: Path) -> Path:
        path = real_bundle(tmp)
        with zipfile.ZipFile(path) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()
                        if name != "configured_scene_bundle_candidate.v1.json"}
        payloads["collision.usda"] = payloads["collision.usda"].decode().replace(
            'upAxis = "Z"', 'upAxis = "Y"').encode()
        manifest = _sealed_doc({
            "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
            "status": "assembled_pending_control_plane_publication", "robot_neutral": True,
            "robot_specific_base_registration_included": False, "assets": [
                {"role": name.split(".")[0], "relative_path": name, "digest": packet_tests._sha(data),
                 "size_bytes": len(data)} for name, data in payloads.items()]}, "manifest_digest")
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in payloads.items():
                archive.writestr(name, data)
            archive.writestr("configured_scene_bundle_candidate.v1.json", json.dumps(manifest))
        return path
    monkeypatch.setattr(packet_tests, "_bundle", bundle_with_y_up_collision)
    with pytest.raises(TeamScenePacketError, match="team_embodiment_scene_collision_frame_unsupported"):
        _half(tmp_path, **WEBSITE)


@pytest.mark.parametrize("case", ["framing", "quality", "mixed", "partial", "unwritten", "full_disk", "timeout"])
def test_the_arms_packet_failures(tmp_path, monkeypatch, case):
    from blueprint_pipeline.native_task_arena_packet import NativeTaskArenaPacketError as Error

    unwritten = "team_scene_robot_access_unwritten:{}:failed at '/var/lib/blueprint/halves/x/collision.usdc'"
    setup = {"framing": (Error([FRAMING]), None), "partial": (Error([FRAMING]), None),
             "quality": (Error([absent.APPEARANCE_QUALITY]), None),
             "mixed": (Error([FRAMING, "native_task_arena_packet_asset_copy_mismatch:task_object"]), None),
             "unwritten": (TeamScenePacketError(unwritten.format("ValueError")), ValueError("mesh")),
             "full_disk": (TeamScenePacketError(unwritten.format("OSError")), OSError(28, "No space left on device")),
             "timeout": (TeamScenePacketError(unwritten.format("TimeoutError")), TimeoutError("write timed out"))}
    error, cause = setup[case]
    _packet_raises(monkeypatch, error, cause=cause, partial=case == "partial")
    if case in ("mixed", "full_disk", "timeout"):
        # Any other code refuses; a full disk or a timeout is the host's and is retried, never a final half.
        with pytest.raises(Error if case == "mixed" else OSError):
            _half(tmp_path / "half", **WEBSITE)
        assert not list((tmp_path / "half").rglob("team_scene_inputs.v1.json"))
        return
    compiled, record = _half(tmp_path / "half", **WEBSITE)
    gone = record[absent.FIELD]
    assert not (Path(compiled["path"]).parent / "native-task-packet").exists()
    assert "/" not in json.dumps(gone)  # never a message that could name a host path
    assert gone["reason"] == {"unwritten": "team_scene_robot_access_unwritten:ValueError",
                              "quality": absent.APPEARANCE_QUALITY}.get(case, FRAMING)
    assert ("appearance_quality" in gone) is (case == "quality")
    plain = _publish_with_record(Job(tmp_path / "plain"))
    told = _publish_with_record(Job(tmp_path / "told"), robot_station_absent=gone)
    assert _said(told["README.txt"], sim.APPEARANCE_QUALITY_NOT_PASSED) is (case == "quality")
    assert (told["README.txt"] == plain["README.txt"]) is (case != "quality")


@pytest.mark.parametrize(("reasons", "sentence"), [
    (["rigid_support_goal_region_rest_outside_success_window"], "SUPPORT_REGION_NOT_HELD"),
    (["rigid_support_start_tips_off_footprint_points"], "SUPPORT_NOT_HELD"),
    (["rigid_support_goal_rest_outside_success_window:0.90000/[0.86668,0.87668]",
      "rigid_support_goal_region_rest_outside_success_window"], "SUPPORT_NOT_HELD"),
    ([], None)])
def test_a_held_support_ships_and_the_readme_says_so_from_its_own_reasons(tmp_path, reasons, sentence):
    """Owner, 2026-11-16: the support check is a flag. The package is built and says what the record says."""
    plain_job, told_job = Job(tmp_path / "plain"), Job(tmp_path / "told")
    # The test job's usdz carries its write time; the same bytes in both, so only the record differs.
    told_job.object.write_bytes(plain_job.object.read_bytes())
    told_job.seal()
    plain = _publish_with_record(plain_job)
    told = _publish_with_record(told_job, task_support_pad={
        "status": "held" if reasons else "consistent", "reasons": reasons})
    for name in ("SUPPORT_REGION_NOT_HELD", "SUPPORT_NOT_HELD"):
        assert _said(told["README.txt"], getattr(sim, name)) is (name == sentence)
    assert (told["README.txt"] == plain["README.txt"]) is (sentence is None)
    assert (told["revision"] == plain["revision"]) is (sentence is None)
    assert "simulat" not in getattr(sim, sentence or "SUPPORT_NOT_HELD").lower()


def test_the_robot_access_raise_site_keeps_the_operating_system_error_as_its_cause(tmp_path, monkeypatch):
    from blueprint_pipeline import robot_access_furniture as furniture
    from blueprint_pipeline.team_scene_packet import _robot_access_applied

    monkeypatch.setattr(furniture, "find_confirmation", lambda *_a, **_k: {"confirmation_digest": "x"})

    def full(*_a, **_k):
        raise OSError(28, "No space left on device", "/var/lib/blueprint/x/collision.usdc")
    monkeypatch.setattr(furniture, "write_collider_without", full)
    with pytest.raises(TeamScenePacketError, match=r"\Ateam_scene_robot_access_unwritten:OSError:") as raised:
        _robot_access_applied({"binding_digest": "d", "items": []}, scope={"directory": tmp_path, "intent": {}},
                              physics_collision=tmp_path / "c.usdc", root=tmp_path, place={}, floor_z=0.0,
                              station={}, task_spec={}, profile={}, support=None, adapter={}, start=[0, 0, 0],
                              target=[0, 0, 0], support_top=0.0)
    assert isinstance(raised.value.__cause__, OSError)


def test_a_broken_droid_profile_is_the_arms_except_for_a_destination_job(tmp_path):
    from blueprint_pipeline.team_scene_packet import materialize_team_scene
    from tests.test_rigid_destination_task_binding import _case

    (tmp_path / "plain").mkdir()
    scene = packet_tests._scene(tmp_path / "plain")
    kwargs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    compiled = materialize_team_scene(**{**kwargs, "droid_profile_digest": "sha256:" + "0" * 64},
                                      output_root=tmp_path / "plain" / "half", **WEBSITE)
    assert compiled[absent.FIELD]["reason"] == "team_scene_droid_profile_binding_invalid"
    # A destination job reads its execution binding from the profile: a scene input, so a typed refusal as before.
    (tmp_path / "destination").mkdir()
    scene, placement = _case(tmp_path / "destination")
    kwargs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    for name, flags in (("off", {}), ("on", WEBSITE)):
        with pytest.raises(TeamScenePacketError, match=r"\Ateam_scene_droid_profile_binding_invalid\Z"):
            materialize_team_scene(**{**kwargs, "droid_profile_digest": "sha256:" + "0" * 64},
                                   output_root=tmp_path / name, destination_placement_qualification_path=placement,
                                   **flags)
        assert not (tmp_path / name).exists()


# ----------------------------------------------------------------------- no paid probe: worker, callers, stage


@pytest.mark.parametrize("case", ["arm_placed", "not_run", "broken_profile", "full_disk"])
def test_the_worker_queues_no_probe_and_retries_a_full_disk(tmp_path, monkeypatch, case):
    from blueprint_pipeline import rigid_robot_free_probe_runner as probes
    from tests.test_team_scene_packet_worker import _registry, _world

    if case == "not_run":
        _not_run(monkeypatch)
    world = _world(tmp_path, monkeypatch)
    if case == "broken_profile":
        _registry(world["operator"], packet_inputs={**world["inputs"], "droid_profile_digest": "sha256:" + "0" * 64})
    if case == "full_disk":
        _packet_raises(monkeypatch, TeamScenePacketError("team_scene_robot_access_unwritten:OSError:full"),
                       cause=OSError(28, "No space left on device"))
    queued: list[Any] = []
    monkeypatch.setattr(probes, "enqueue_rigid_robot_free_probe", lambda root: queued.append(root) or {"status": "queued"})
    [status] = world["process"]()["results"]
    assert queued == [] and status["paid_execution_requested"] is False
    if case == "full_disk":
        assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_packet_failed:OSError"])
        assert "not_before" in status["automatic_retry"] and list(world["halves"].iterdir()) == []
        return
    assert (status["status"], status["outcome"], status["robot_free_probe"]) == ("completed", "compiled", "not_queued")
    assert ("robot_station_absent" in status) is (case == "broken_profile")
    assert json.loads(Path(status["team_scene_inputs_path"]).read_text())[absent.PROBE_FIELD] == absent.PROBE_NOT_QUEUED


@pytest.mark.parametrize(("caller", "queued"), [("operator_isaac", True), ("website", False)])
def test_who_queues_the_probe(tmp_path, monkeypatch, caller, queued):
    from blueprint_pipeline import rigid_robot_free_probe_runner as probes
    from blueprint_pipeline import team_scene_packet as tsp

    calls: list[Any] = []
    monkeypatch.setattr(probes, "enqueue_rigid_robot_free_probe", lambda root: calls.append(root) or {"status": "queued"})
    monkeypatch.setattr("blueprint_pipeline.team_candidate_quick10_registry.write_derived_profile",
                        lambda row: {"status": "written"})
    compiled, record = _half(tmp_path / "half", **(WEBSITE if caller == "website" else {}))
    assert (absent.PROBE_FIELD in record) is not queued
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": "site-x"}))
    result = tsp.record_team_scene(compiled, run_root=tmp_path, capture_root=capture, scenario_id="nominal",
                                   scene_revision_digest="sha256:" + "1" * 64)
    assert calls == ([capture] if queued else []) and (result["robot_free_probe"]["status"] == "queued") is queued


@pytest.mark.parametrize(("record", "ready"), [
    ({"robot_free_probe": "not_queued"}, True), ({}, False), ({"robot_free_probe": "queued"}, False),
    ("unsealed", False), ({"robot_free_probe": "not_queued", "offer_only_block": True}, True)],
    ids=["no_probe", "older_half", "other_value", "unsealed", "offer_only_probe_unavailable"])
def test_the_scene_stage_spares_only_a_half_that_records_no_probe(tmp_path, monkeypatch, record, ready):
    """A half compiled before (capture-447's, for one) keeps its probe and its hold exactly as before."""
    from blueprint_pipeline import rigid_robot_free_probe_bundle as bundle
    from blueprint_pipeline import rigid_robot_free_probe_runner as runner
    from blueprint_pipeline import scene_stage_maintenance as maintenance
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_scene_packet_worker as packets
    from tests.scene_stage_support import prepared_source_queue

    world = prepared_source_queue(tmp_path, monkeypatch)
    if isinstance(record, dict) and record.pop("offer_only_block", False):
        # A no-GPU revision's request (``probe_unavailable``) whose robot-team offer post failed, retries spent,
        # under the running release (else a new release's recovery returns it to pending first).
        [processed] = (world["packet_queue"] / "processed").glob("*.json")
        request = {**json.loads(processed.read_text()), "robot_free_readiness_status": "probe_unavailable"}
        processed.write_text(json.dumps(packets._seal(request, "request_digest")))
        status_path = world["packet_queue"] / "status" / processed.name
        status_path.write_text(json.dumps({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"],
                                           "automatic_retry": {"attempts": 3, "exhausted": True},
                                           packets.RELEASE_COMMIT_FIELD: packets.current_release_commit()}))
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"robot_free_probe": "not_queued", "inputs_digest": "sha256:" + "0" * 64}
                               if record == "unsealed" else _sealed_doc(
                                   {"schema_version": "blueprint.team_scene_inputs.v1", **record}, "inputs_digest")))
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    profile = {**registry.profile_for_capture(capture), "team_scene_inputs_path": str(path)}
    monkeypatch.setattr(registry, "profile_for_capture", lambda target: profile if target == capture else None)
    queued: list[Any] = []
    monkeypatch.setattr(runner, "enqueue_rigid_robot_free_probe", lambda root: queued.append(root) or {"status": "queued"})
    monkeypatch.setattr(runner, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(bundle, "current_probe_state", lambda _root: {"status": "receipt_decided", "receipt_status": "failed"})

    def gate(_profile):
        raise ValueError("team_evaluation_robot_free_probe_not_passed")
    monkeypatch.setattr(maintenance, "probe_gate", gate)
    [outcome] = maintenance.scene_stage_pass(queue_root=world["screening_queue"],
                                             packet_queue_root=world["packet_queue"])["sources"]
    if ready:
        assert (outcome["status"], outcome["scene_stage"]) == ("scene_stage_ready", maintenance.PROBE_NOT_QUEUED)
        assert queued == [] and (outcome.get("evaluation_hold") or {}).get("reason_code") is None
    else:
        assert queued == [capture] and outcome["evaluation_hold"]["reason_code"] == "robot_free_probe_failed"


def test_the_remote_result_carries_the_arms_absence(tmp_path, monkeypatch):
    import types

    from blueprint_pipeline import team_scene_packet as packet
    from blueprint_pipeline import team_scene_packet_remote as remote

    fs = tmp_path / "fs"
    (fs / "jobs").mkdir(parents=True)
    envelope = _sealed_doc({"schema_version": remote.ENVELOPE_SCHEMA_VERSION, "source_commit": "c" * 40,
                            "request_id": "r", "arguments": {}}, "envelope_digest")
    (fs / "jobs" / "env.json").write_text(json.dumps(envelope))
    roots = types.SimpleNamespace(filesystem_root=str(fs), local=lambda p: fs / str(p).lstrip("/"))
    monkeypatch.setattr(remote, "_local_arguments", lambda arguments, roots: {})
    gone = {"reason": NO_STANCE, "robot_included": False, "paid_step_started": False}
    monkeypatch.setattr(packet, "materialize_team_scene", lambda **_: {
        "path": str(fs / "out" / "team_scene_inputs.v1.json"), "inputs_digest": "sha256:" + "1" * 64,
        "scene_id": "s", "task_id": "t", absent.FIELD: gone})
    result = remote.run_team_scene_packet_in_worker(
        {"inputs": [{"role": "queue_envelope", "materialize_at": "/jobs/env.json"}], "code": {"source_commit": "c" * 40}},
        roots)
    assert result["status"] == remote.COMPLETED and result["compiled"][absent.FIELD] == gone


# --------------------------------------------------------------------------------- what the package says


def test_only_the_package_compile_reads_a_stage_5_that_ran_no_simulator(tmp_path, monkeypatch):
    _not_run(monkeypatch)
    with pytest.raises(ValueError, match="rigid_relocation_native_adapter_source_contract_invalid"):
        _half(tmp_path / "operator")
    compiled, record = _half(tmp_path / "website", **WEBSITE)
    assert record["robot_station"] is not None and record[absent.PROBE_FIELD] == absent.PROBE_NOT_QUEUED


@pytest.mark.parametrize(("stage_5", "static_passed", "arm"), [
    ("isaac", False, True), ("not_run", False, True), ("not_run", True, True), ("not_run", True, False)])
def test_the_readme_says_what_stage_5_and_stage_4_show(tmp_path, monkeypatch, stage_5, static_passed, arm):
    if stage_5 == "not_run":
        _not_run(monkeypatch, static_passed=static_passed)
    if not arm:
        _stance_raises(monkeypatch, TeamEmbodimentError([NO_STANCE]))
    compiled, _record = _half(tmp_path / "half", **WEBSITE)
    source = sim._half(Path(compiled["path"]).parent, {"team_scene_inputs_path": compiled["path"]})
    if stage_5 == "isaac":
        # The fixture's Isaac record has no ``native_isaac_executed``: no wording either way; the revision as before.
        assert source["physics_check"] in ("isaac_sim", None)
        assert sim.simulation_revision({**source, "physics_check": "isaac_sim"}) == sim.simulation_revision(
            {**source, "physics_check": None})
    else:
        assert source["physics_check"] == ("not_run_static_check_passed" if static_passed else "not_run")
        assert sim.simulation_revision(source) != sim.simulation_revision({**source, "physics_check": None})
    # The furniture and look-quality notes change the README, so each names a new package revision.
    assert len({sim.simulation_revision({**source, **note}) for note in (
        {}, {"removed_items": 2}, {"appearance_quality": "not_passed"})}) == 3
    result, members = _download(tmp_path, compiled["path"])
    assert result["status"] == "published", result
    readme = members["README.txt"]
    assert _said(readme, NO_SIMULATOR) is (stage_5 == "not_run") and _said(readme, STATIC_PASSED) is static_passed
    assert "estimated from the video and checked" not in readme
    assert all("qualif" not in text.lower() for text in members.values())
    assert sim.PHYSICS_CHECK_WORDS == {"isaac_sim": (), "not_run": (NO_SIMULATOR,),
                                       "not_run_static_check_passed": (STATIC_PASSED, NO_SIMULATOR),
                                       "not_run_joint_as_authored": (NO_SIMULATOR, sim.JOINT_AS_AUTHORED),
                                       "not_run_joint_as_authored_no_rule": (NO_SIMULATOR,
                                                                             sim.JOINT_AS_AUTHORED_NO_RULE)}


def test_physics_check_reads_stage_5_by_its_own_reader(tmp_path):
    from blueprint_pipeline.website_scene_without_simulator import not_run_receipt

    record = not_run_receipt(replacement_identity={"id": "o", "version": "v1"}, asset_digest="sha256:" + "a" * 64,
                             static_qualification_digest="sha256:" + "b" * 64)
    qualified = _sealed_doc({"schema_version": "task_evaluation_replacement_native_import_result.v1",
                             "status": "qualified", "native_isaac_executed": True,
                             "native_simulator_import_qualified": True, "blockers": []})
    cases = [(record, "not_run"), (qualified, "isaac_sim"), ({**record, "status": "qualified"}, None)]  # stale seal
    cases += [(_sealed_doc(forged), None) for forged in (
        {**record, "native_isaac_executed": "false"}, {**record, "native_simulator_import_qualified": None},
        {key: value for key, value in record.items() if key != "gpu_execution_performed"},
        {**record, "status": "qualified"}, {**qualified, "native_simulator_import_qualified": False})]
    for stage, expected in cases:
        assert sim.physics_check(tmp_path, {"stage_5": stage}) == expected, stage
    assert sim.physics_check(tmp_path, {}) is None


@pytest.mark.parametrize(("changes", "passed"), [
    ({}, True), ({"structural_findings": ["collision_approximation_missing"]}, False),
    ({"structural_findings": None}, False), ("stale_seal", False), ("changed_after_seal", False),
    ({"schema_version": "task_evaluation_articulated_replacement_static_qualification.v1"}, False)])
def test_the_static_check_sentence_needs_no_findings_and_a_sealed_stage_4_record(changes, passed):
    record = _sealed_doc({"schema_version": sim.RIGID_STATIC_SCHEMA, "status": "authored_structure_statically_qualified",
                          "authored_structure_statically_qualified": True, "structural_findings": [],
                          **(changes if isinstance(changes, dict) else {})})
    if changes == "stale_seal":
        record["result_digest"] = "sha256:" + "0" * 64
    if changes == "changed_after_seal":
        record["observed_structure"] = {"rigid_body_paths": ["/A", "/B"]}
    assert sim.static_check_passed(record) is passed


@pytest.mark.parametrize("change", [
    {"native_isaac_executed": "false"}, {"native_simulator_import_qualified": None}, {"status": "qualified"},
    {"gpu_execution_performed": True}, {"physics_authority": "isaac"}, {"blockers": ["x"]},
    {"native_isaac_executed": True}, "missing_gpu_field", "qualified_without_flag", "another_asset", "no_flag"],
    ids=str)
def test_the_adapter_accepts_only_the_exact_not_run_record_and_only_for_the_package(tmp_path, change):
    """B v2's adapter: the true record is read only with ``accept_simulator_not_run`` (``package_only`` passes it),
    and only when it names the revision's own object."""
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
        TaskEvaluationRigidRelocationNativeAdapterError, adapt_rigid_relocation_task_template,
    )
    from blueprint_pipeline.website_scene_without_simulator import not_run_receipt
    from tests.test_task_evaluation_rigid_relocation_native_adapter import _case

    launch, configured, references, docs = _case(tmp_path)
    record = not_run_receipt(replacement_identity=docs[NATIVE_IMPORT]["replacement_identity"],
                             asset_digest=configured["replacement"]["asset"]["digest"],
                             static_qualification_digest="sha256:" + "b" * 64)
    flag = {"accept_simulator_not_run": change != "no_flag"}
    forged = record if change == "no_flag" else _sealed_doc({**record, "asset_digest": "sha256:" + "c" * 64}) if (
        change == "another_asset") else (_sealed_doc({key: v for key, v in record.items() if key != "gpu_execution_performed"})
              if change == "missing_gpu_field"
              else _sealed_doc({**docs[NATIVE_IMPORT], "native_simulator_import_qualified": False})
              if change == "qualified_without_flag" else _sealed_doc({**record, **change}))
    for document, accepted in ((record, True), (forged, False)):
        _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=NATIVE_IMPORT,
                 document=document)
        launch["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
        if accepted:
            adapt_rigid_relocation_task_template(request=launch, configured_revision=configured,
                                                 materialized_references=references, accept_simulator_not_run=True)
            continue
        with pytest.raises(TaskEvaluationRigidRelocationNativeAdapterError):
            adapt_rigid_relocation_task_template(request=launch, configured_revision=configured,
                                                 materialized_references=references, **flag)


@pytest.mark.parametrize(("removal", "line"), [
    (None, False), ({"status": "held", "items": [{"label": "x"}]}, False),
    ({"status": "applied", "station_after": {"stance_digest": "y"},
      "items": [{"label": "Blue office chair"}, {"label": "Stool by the till"}]}, True)])
def test_furniture_moved_for_the_arm_is_told_by_count_only(tmp_path, removal, line):
    plain_job, told_job = Job(tmp_path / "plain"), Job(tmp_path / "told")
    # The test job's usdz carries its write time; the same bytes in both, so only the record differs.
    told_job.object.write_bytes(plain_job.object.read_bytes())
    told_job.seal()
    plain = _publish_with_record(plain_job)
    told = _publish_with_record(told_job, **({"robot_access_removal": removal} if removal else {}))
    assert _said(told["README.txt"], sim.FURNITURE_REMOVED.format(count=2)) is line
    assert "chair" not in told["README.txt"].lower() and "stool" not in told["README.txt"].lower()
    assert (told["README.txt"] == plain["README.txt"] and told["revision"] == plain["revision"]) is not line


def test_a_default_contract_half_ships_its_sealed_rule_and_only_from_a_sealed_record(tmp_path):
    compiled, record = _half(tmp_path / "half", **WEBSITE)
    scene = Path(compiled["path"]).parent
    plan = json.loads((scene / "native-task-packet" / sim.PLAN_FILE).read_text())
    plan["task_spec"].pop("task_success_contract", None)
    copy = tmp_path / "copy"
    for name in ("native-task-packet", "native-task-packet-three-view"):
        if (scene / name / sim.PLAN_FILE).is_file():
            (copy / name).mkdir(parents=True)
            (copy / name / sim.PLAN_FILE).write_bytes((scene / name / sim.PLAN_FILE).read_bytes())
    status = {"team_scene_inputs_path": str(copy / "team_scene_inputs.v1.json")}
    for written, ships in ((record, True), ({**record, "inputs_digest": "sha256:" + "0" * 64}, False),
                           ({**record, "plan_digest": "sha256:" + "0" * 64}, False)):
        (copy / "team_scene_inputs.v1.json").write_text(json.dumps(
            _sealed_doc(written, "inputs_digest") if written.get("plan_digest") != record["plan_digest"] else written))
        rule = sim._with_sealed_rule(plan, copy, status)["task_spec"].get("task_success_contract")
        assert (rule == record["task_success_contract"]) is ships
    # A record sealed on the sibling plan supplies its rule only when that plan holds the same task.
    (copy / "team_scene_inputs.v1.json").write_text(json.dumps(record))
    sibling = {**plan, "plan_digest": "sha256:" + "1" * 64}
    for task, ships in ((plan["task_spec"], True), ({**plan["task_spec"], "task_id": "another-task"}, False)):
        for written in (copy / name / sim.PLAN_FILE for name in sim.PACKET_DIRECTORIES if (copy / name).is_dir()):
            written.write_text(json.dumps({**json.loads((scene / written.parent.name / sim.PLAN_FILE).read_text()),
                                           "task_spec": task}))
        rule = sim._with_sealed_rule(sibling, copy, status)["task_spec"].get("task_success_contract")
        assert (rule == record["task_success_contract"]) is ships


def test_a_half_blocked_only_for_its_robot_team_offer_is_still_the_packages_source(job):
    status_path = next((job.queue / "status").glob("*.json"))
    status = json.loads(status_path.read_text())
    for blockers, found in ((["team_scene_offer_publication_failed"], True), (["team_scene_packet_interrupted"], False)):
        status_path.write_text(json.dumps({**status, "status": "blocked", "blockers": blockers}))
        assert (sim.simulation_source(job.capture, queue_root=job.queue) is not None) is found


def test_a_package_that_is_not_out_is_named_by_capture_in_the_journal(job, caplog):
    plan_path = job.packet / sim.PLAN_FILE
    plan = json.loads(plan_path.read_text())
    plan["objects"][2]["sha256"] = "not-a-digest"
    plan_path.write_text(json.dumps(plan))
    world = World(job)
    with caplog.at_level(logging.WARNING, logger=sim.logger.name):
        counts = sim.tick(job.tmp / "storage", now=lambda: 1_794_500_000, check_rights=world.check_rights,
                          archive=world.archive, retract=world.retract, deliver=world.deliver, queue_root=job.queue)
    assert counts == {"refused": 1}
    assert any("website_company_simulation_outcome" in row.message and job.capture.name in row.message
               and "status=refused" in row.message for row in caplog.records)


def test_a_half_compiled_with_its_rule_withheld_seals_no_rule_and_ships_its_object(tmp_path):
    """Objects step 1: the real compile of a job no task profile models, then its real package: the object, no
    rule in the plan, the record or the package, and the README's sentence. Only the website's package-only
    compile takes the reason."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline.team_scene_packet_worker import WEBSITE_COMPILE

    withheld = {"code": "website_native_task_profile_preparation_required"}
    compiled, record = _half(tmp_path / "withheld", **WEBSITE_COMPILE, rule_withheld=withheld)
    assert record["rule_withheld"] == {"reason": withheld["code"]}
    assert not {"task_success_contract", "task_success_contract_digest", "owner_success_assumption"} & set(record)
    for name in sim.PACKET_DIRECTORIES:
        plan = Path(compiled["path"]).parent / name / sim.PLAN_FILE
        assert not plan.exists() or "task_success_contract" not in json.loads(plan.read_text())["task_spec"]
    result, files = _download(tmp_path / "withheld", compiled["path"])
    assert result["status"] == "published", result
    assert "success_rule.json" not in files and "object.usda" in files
    assert " ".join(files["README.txt"].split()).count(sim.NO_RULE) == 1 and "success_rule" not in files["README.txt"]


def test_only_the_website_package_compile_withholds_the_rule(tmp_path):
    """A compile that is not the website worker's package-only one, handed a reason, seals the rule as before."""
    withheld = {"code": "website_native_task_profile_preparation_required"}
    for name, flags in (("operator", {}), ("assumed", {"assume_unconfirmed_standard_thresholds": True})):
        _, record = _half(tmp_path / name, **flags, rule_withheld=withheld)
        assert "rule_withheld" not in record and record["task_success_contract"]["contract_digest"]


def test_a_drawer_that_reaches_the_compile_with_its_rule_withheld_still_gets_its_package(tmp_path):
    """Not expected (the handoff sends only a rigid object), but never a dead end once the object is built: the
    half ships with no rule, the case flagged in its sealed record."""
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline.team_scene_packet import materialize_team_scene
    from blueprint_pipeline.team_scene_packet_worker import WEBSITE_COMPILE

    # Its owner criteria unconfirmed: no owner contract is read for a withheld rule, so none can refuse it.
    scene = packet_tests._articulated_scene(tmp_path, not_run=True, owner_confirmed=False)
    inputs = {key: value for key, value in scene.items() if key not in ("profile", "store", "revision_reference")}
    compiled = materialize_team_scene(**inputs, output_root=tmp_path / "team-scene", **WEBSITE_COMPILE,
                                      rule_withheld={"code": "website_native_task_profile_preparation_required"})
    record = json.loads(Path(compiled["path"]).read_text())
    assert record["rule_withheld"] == {"reason": "website_native_task_profile_preparation_required",
                                       "articulated": True}
    assert not {"task_success_contract", "owner_success_assumption"} & set(record)
    result, files = _download(tmp_path, compiled["path"])
    assert result["status"] == "published", result
    readme = " ".join(files["README.txt"].split())
    assert "success_rule.json" not in files
    assert readme.count(sim.NO_RULE) == 1
    assert "success interval" not in readme and readme.count(sim.JOINT_AS_AUTHORED_NO_RULE) == 1
