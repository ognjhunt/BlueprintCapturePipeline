# Covers: src/blueprint_pipeline/scene_packet.py
# Covers: src/blueprint_pipeline/scene_packet_queue.py
"""The new scene packet stage gives the kept readers what the old one gave them, on the same inputs.

Case A replays the committed synthetic record ``rigid-pick-place-1080p-short`` through the OLD stage
(``task_evaluation_downstream_replay`` until ``team_packet``) and, from a second copy stopped right
after the request is queued, through the NEW queue. Case B compiles the real ``proposal_only`` job of
the owner-confirmation tests with both stages, assumed thresholds. Case C compiles the articulated
drawer fixture; the real download publishes the new half with its open/close success rule. Each new half is handed to the real readers: the scripted check's reader
(``rigid_robot_free_probe_bundle.current_probe_state`` / ``probe_inputs``) and the company download
(``website_company_simulation`` with the fakes of its own tests). Hermetic: no network, nothing paid.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import replay_set
from blueprint_pipeline import scene_packet as new
from blueprint_pipeline import scene_packet_queue as queue
from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline.team_scene_packet import materialize_team_scene, read_team_scene_inputs
from tests.test_team_scene_packet import _articulated_scene
from tests.test_team_scene_packet_owner_confirmation import _job
from tests.test_website_company_simulation import Job, World, _package

SET = Path(__file__).resolve().parent / "fixtures" / "replay_set_v1"
RECORD = "rigid-pick-place-1080p-short"
PLAN = "native_task_arena_scene_plan.v1.json"


# ------------------------------------------------------------------------------------------------ comparison


def _json(path: Any) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plan(record_path: Any) -> dict[str, Any]:
    return _json(Path(record_path).parent / "native-task-packet" / PLAN)


def _plan_fields(plan: dict[str, Any]) -> dict[str, Any]:
    spec = plan["task_spec"]
    return {"objects": [{key: row.get(key) for key in ("semantic_role", "pose_world", "scale", "visible", "sha256",
                                                       "size_bytes")} for row in plan["objects"]],
            "lift_unit_world": spec["interaction_affordance"].get("lift_unit_world"),
            "cadence": plan.get("cadence"),
            "task_object_extent_m": (plan.get("task_object_observability") or {}).get("task_object_extent_m"),
            "robot_id": plan["robot"]["robot_id"], "robot_base_pose_world": plan["robot"]["base_pose_world"]}


def _contract(value: dict[str, Any]) -> dict[str, Any]:
    """The contract minus provenance ids (author and confirmer), which name who stands behind it."""

    provenance = {key: item for key, item in value["provenance"].items() if key not in {"author_id"}}
    return {**{key: item for key, item in value.items() if key not in {"provenance", "contract_digest"}},
            "provenance": provenance}


def _record_fields(path: Any) -> dict[str, Any]:
    value = _json(path)
    pad = value.get("task_support_pad") or {}
    return {"task_success_contract": _contract(value["task_success_contract"]),
            "pad": {key: pad.get(key) for key in ("status", "rest_check")},
            "seating": value.get("task_support_seating")}


def _download(tmp_path: Path, record_path: Any) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    """The company download's pass over the half: its result, ``success_rule.json`` and the zip's member names."""

    job = Job(tmp_path)
    status = next((job.queue / "status").glob("*.json"))
    status.write_text(json.dumps({**_json(status), "team_scene_inputs_path": str(record_path)}))
    result = World(job).publish()
    if result["status"] != "published":
        return result, None, []
    with zipfile.ZipFile(_package(job)) as archive:
        names = sorted(archive.namelist())
        return result, json.loads(archive.read("success_rule.json")), names


def _fetcher(host: Path):
    """Published objects from the record's mirror, else a byte-identical file of the record (as the replay does)."""

    files: dict[tuple[int, str], Path] = {}
    for path in host.rglob("*"):
        if path.is_file() and not path.is_symlink():
            files.setdefault((path.stat().st_size, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()), path)

    def fetch(reference: dict[str, Any]) -> bytes:
        return files[(reference["size_bytes"], reference["digest"])].read_bytes()

    return fetch


# ------------------------------------------------------------------------------------------------- case A


def test_case_a_the_replay_record_gives_the_kept_readers_the_same_scene(tmp_path, monkeypatch):
    from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state, probe_inputs
    from blueprint_pipeline.task_evaluation_downstream_replay import replay

    manifest = _json(SET / replay_set.MANIFEST_FILENAME)
    entry = next(row for row in manifest["records"] if row["record_id"] == RECORD)
    tool = entry["tools"][0]
    root = Path(manifest["materialize_root"])
    with replay_set._locked(root):  # records stand at fixed paths; one replay at a time
        host = root / RECORD / "host"
        replay_set.materialize(SET / "records" / RECORD, host, empty_dirs=entry["empty_dirs"])
        try:
            with replay_set._replay_clock(manifest[replay_set.CLOCK_FIELD]):
                inputs = {name: host / relative for name, relative in tool["inputs"].items()}
                old = replay(**inputs, until="team_packet", replay_root=tmp_path / "old")
                assert old["first_stop"]["code"] == "until_team_packet"
                staged = replay(**inputs, until="readiness", replay_root=tmp_path / "new")
                assert staged["first_stop"]["code"] == "until_readiness"
                for name, value in staged["environment"].items():
                    monkeypatch.setenv(name, value)
                run = Path(staged["run_root"])
                packets = run / "state" / "team-scene-packets"
                assert len(list((packets / "pending").glob("*.json"))) == 1
                result = queue.process_queue(queue_root=packets, fetcher=_fetcher(host))
                # The scripted check's real reader path accepts the new half while the record's clock stands.
                state = current_probe_state(run / "inputs" / "partition" / "scenes" / "site-req-1" / "captures"
                                            / "walkthrough-req-1")
                probe = probe_inputs(state=state) if state["status"] == "probe_required" else None
                _compare_case_a(tmp_path, old, result, state, probe)
        finally:
            shutil_rmtree(root / RECORD)


def _compare_case_a(tmp_path: Path, old: dict[str, Any], result: dict[str, Any], state: dict[str, Any],
                    probe: dict[str, Any] | None) -> None:

    [status] = result["results"]
    old_run = Path(old["run_root"])
    old_status = _json(next((old_run / "state" / "team-scene-packets" / "status").glob("*.json")))
    # O1: the same outcome; the status keeps exactly what the download, exchange and deletion read.
    for key in ("status", "outcome", "blockers", "capture_id", "request_key", "scene_revision_digest"):
        assert status[key] == old_status[key], key
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    # The record reads through the old reader and the new one alike.
    read_team_scene_inputs(status["team_scene_inputs_path"])
    new.read_half(status["team_scene_inputs_path"])
    # Plan: every placed object byte for byte, the lift axis, the cadence, the extent and the robot.
    old_plan, new_plan = _plan(old_status["team_scene_inputs_path"]), _plan(status["team_scene_inputs_path"])
    assert _plan_fields(new_plan) == _plan_fields(old_plan)
    # O3: the same contract, support pad and seating.
    assert _record_fields(status["team_scene_inputs_path"]) == _record_fields(old_status["team_scene_inputs_path"])
    # Intended difference 1: the old plan carries no success rule; the new one does. (The download now reads the old
    # half's rule from its sealed record, ``website_company_simulation._with_sealed_rule``.)
    assert "task_success_contract" not in old_plan["task_spec"]
    contract = _json(status["team_scene_inputs_path"])["task_success_contract"]
    assert new_plan["task_spec"]["task_success_contract"] == contract
    # O4: the same derived row but for the half's path.
    old_row, new_row = _json(old_status["derived_profile"]["path"]), _json(status["derived_profile"]["path"])
    ignore = {"capture_root", "robot_free_probe_root", "team_scene_inputs_path", "profile_digest"}
    assert {k: v for k, v in new_row.items() if k not in ignore} == {k: v for k, v in old_row.items() if k not in ignore}
    # The scripted check: its reader admits the half and stages the nominal plan from it.
    assert state["status"] in {"probe_required", "receipt_decided"}, state.get("reason")
    if probe is not None:
        assert probe["scene_plan"]["robot"]["robot_id"] == "franka_panda"
        assert probe["task_success_contract_digest"] == contract["contract_digest"]
    # The download: published for both halves, with the same criteria in their rule (the old one's from its record).
    old_published, old_rule, _ = _download(tmp_path / "dl-old", old_status["team_scene_inputs_path"])
    assert old_published["status"] == "published", old_published
    published, rule, names = _download(tmp_path / "dl-new", status["team_scene_inputs_path"])
    assert published["status"] == "published", published
    assert rule["criteria"] and "success_rule.json" in names
    assert old_rule["criteria"] == rule["criteria"]



def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- case B


def test_case_b_an_assumed_proposal_only_job_compiles_the_same_contract_and_success_rule(tmp_path, monkeypatch):
    job, inputs = _job(tmp_path, monkeypatch)
    scope = {"intent": job.intent, "directory": job.directory}

    def documents(name: str) -> dict[str, Path]:
        return rigid.fetch_revision_documents(revision=job.revision, output_root=tmp_path / name / "work",
                                              fetcher=lambda reference: job.store[reference["uri"]])

    old = materialize_team_scene(revision=job.revision, documents=documents("old"),
                                 output_root=tmp_path / "old" / "scene", owner_success_confirmation_scope=scope,
                                 assume_unconfirmed_standard_thresholds=True, **inputs)
    compiled = new.compile_half(revision=job.revision, documents=documents("new"),
                                output_root=tmp_path / "new" / "scene", owner_scope=scope, **inputs)
    assert compiled["owner_success_assumption"]["status"] == old["owner_success_assumption"]["status"]
    assert compiled["owner_success_assumption"] == old["owner_success_assumption"]
    assert _record_fields(compiled["path"]) == _record_fields(old["path"])
    assert _json(compiled["path"])["task_success_contract"] == _json(old["path"])["task_success_contract"]
    old_plan, new_plan = _plan(old["path"]), _plan(compiled["path"])
    assert _plan_fields(new_plan) == _plan_fields(old_plan)
    assert new_plan["task_spec"]["task_success_contract"] == old_plan["task_spec"]["task_success_contract"]
    # The assumption marker is read by the site-confirmation exchange's reader.
    from blueprint_pipeline.website_owner_success_assumption import half_assumption

    assert half_assumption(compiled["path"]) == compiled["owner_success_assumption"]
    # The download's success rule is byte-for-byte the old one.
    first, old_rule, old_names = _download(tmp_path / "dl-old", old["path"])
    second, new_rule, new_names = _download(tmp_path / "dl-new", compiled["path"])
    assert first["status"] == second["status"] == "published", (first, second)
    assert json.dumps(new_rule, sort_keys=True) == json.dumps(old_rule, sort_keys=True)
    assert new_names == old_names


# ------------------------------------------------------------------------------------------------- case C


def _articulated_halves(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = _articulated_scene(tmp_path)
    inputs = {key: scene[key] for key in ("revision", "documents", "configured_scene_bundle_path",
                                          "droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path")}
    return (materialize_team_scene(**inputs, output_root=tmp_path / "old" / "scene"),
            new.compile_half(**inputs, output_root=tmp_path / "new" / "scene"))


def _object_row(plan: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in plan["objects"] if row["semantic_role"] == "task_object")


def test_case_c_an_articulated_job_carries_its_success_rule_in_the_plan_and_is_published(tmp_path):
    from pxr import Sdf

    old, compiled = _articulated_halves(tmp_path)
    old_plan, new_plan = _plan(old["path"]), _plan(compiled["path"])
    contract = _json(compiled["path"])["task_success_contract"]
    assert contract == _json(old["path"])["task_success_contract"]
    assert contract["provenance"]["author_source"] == "task_owner"
    # The old plan left the articulated contract out, so the download refused the job; the new plan carries it.
    assert "task_success_contract" not in old_plan["task_spec"]
    assert new_plan["task_spec"]["task_success_contract"] == contract
    # The object file differs only by the line flattening wrote into its documentation, naming the host file.
    packets = [Path(path).parent / "native-task-packet" for path in (old["path"], compiled["path"])]
    layers = [Sdf.Layer.FindOrOpen(str(packet / _object_row(plan)["usd_path"]))
              for packet, plan in ((packets[0], old_plan), (packets[1], new_plan))]
    assert "Generated from Composed Stage of root layer " in layers[0].documentation
    assert layers[1].documentation == layers[0].documentation.split("\n\nGenerated from")[0]
    layers[1].documentation = layers[0].documentation
    assert layers[1].ExportToString() == layers[0].ExportToString()
    unchanged = [{**row, "sha256": None, "size_bytes": None} if row["semantic_role"] == "task_object" else row
                 for row in _plan_fields(old_plan)["objects"]]
    assert [{**row, "sha256": None, "size_bytes": None} if row["semantic_role"] == "task_object" else row
            for row in _plan_fields(new_plan)["objects"]] == unchanged
    assert {**_plan_fields(new_plan), "objects": None} == {**_plan_fields(old_plan), "objects": None}
    # The plan states no lift direction; the room's files declare Z, so the real download publishes the new half
    # with the open/close goal in its success rule. The old half's object file still names the host file it was
    # flattened from; the download drops that line (before 2026-11-16 its scan refused the file for it).
    assert "lift_unit_world" not in new_plan["task_spec"]["interaction_affordance"]
    kept, _, _ = _download(tmp_path / "dl-old", old["path"])
    assert kept["status"] == "published", kept
    result, rule, names = _download(tmp_path / "dl-new", compiled["path"])
    assert result["status"] == "published", result
    assert rule["up_axis"] == "Z" and rule["units"] == "metres, radians, seconds"
    assert rule["criteria"]["opening"] == {"mode": "required", "success_interval": [0.216, 0.36],
                                           "joint_hard_limits": [0.0, 0.36], "reset_position": 0.0}
    assert rule["criteria"]["hold"] == {"mode": "required", "window_samples": 15,
                                        "maximum_settled_target_speed": 0.02}
    assert rule["criteria"]["motion"] == {"movement_epsilon": 0.0036}
    assert rule["criteria"]["safety"] == {"mode": "required"}
    assert "object_start" in rule and rule["timing"]
    assert "success_rule.json" in names and "scene.usda" in names
    text = json.dumps(rule)
    assert "drawer" not in text and "joint_id" not in text


def test_an_articulated_plan_whose_room_states_no_up_axis_is_still_refused(tmp_path):
    from pxr import Sdf

    _, compiled = _articulated_halves(tmp_path)
    packet = Path(compiled["path"]).parent / "native-task-packet"
    for row in _plan(compiled["path"])["objects"]:
        if row["semantic_role"] in {"scene_collision", "scene_appearance"}:
            path = packet / row["usd_path"]
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            layer.pseudoRoot.ClearInfo("upAxis")
            text = layer.ExportToString()
            path.unlink()  # the packet's files are read-only links; this half gets its own
            path.write_text(text)
    refused, _, _ = _download(tmp_path / "dl", compiled["path"])
    assert refused == {"status": "refused", "reason": "website_company_simulation_up_axis_not_stated",
                       "detail": ""}


def test_the_up_axis_is_never_assumed(tmp_path):
    from blueprint_pipeline.scene_package_stage import StageRefused, up_axis

    def room(name: str, axis: str | None) -> dict[str, Any]:
        header = f'#usda 1.0\n(\n    upAxis = "{axis}"\n)\n' if axis else "#usda 1.0\n"
        (tmp_path / name).write_text(header + 'def Xform "World"\n{\n}\n')
        return {"semantic_role": "scene_collision" if name.startswith("c") else "scene_appearance", "usd_path": name}

    def plan(*rows: dict[str, Any], lift: Any = None) -> dict[str, Any]:
        affordance = {} if lift is None else {"lift_unit_world": lift}
        return {"task_spec": {"interaction_affordance": affordance}, "objects": list(rows)}

    assert up_axis(plan(room("c.usda", "Z"), room("a.usda", None)), packet=tmp_path) == "Z"
    assert up_axis(plan(room("c.usda", "Y")), packet=tmp_path) == "Y"
    for refused in (plan(room("c.usda", None), room("a.usda", None)),       # nothing stated
                    plan(room("c.usda", "Z"), room("a.usda", "Y")),         # the room disagrees with itself
                    plan(room("c.usda", "Z"), lift=[1.0, 0.0, 0.0]),        # the plan's own lift says X
                    plan()):                                                 # no room
        with pytest.raises(StageRefused, match="scene_stage_up_axis_not_stated"):
            up_axis(refused, packet=tmp_path)
    with pytest.raises(StageRefused, match="scene_stage_up_axis_not_stated"):
        up_axis(plan(room("c.usda", "Z")))                                  # no files to read


# ------------------------------------------------------------------------------------------------ hard stops


def _rigid_inputs(tmp_path: Path) -> dict[str, Any]:
    from tests.test_team_scene_packet import _scene

    scene = _scene(tmp_path)
    return {key: scene[key] for key in ("revision", "documents", "configured_scene_bundle_path",
                                        "droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path")}


def test_a_runtime_source_that_may_not_be_redistributed_is_refused_before_anything_is_written(tmp_path):
    inputs = _rigid_inputs(tmp_path)
    receipt = Path(inputs["runtime_source_receipt_path"])
    receipt.write_text(json.dumps({**_json(receipt), "redistribution_permitted": False}))
    with pytest.raises(new.ScenePacketError, match="rights"):
        new.compile_half(**inputs, output_root=tmp_path / "scene")
    assert not (tmp_path / "scene").exists()
    # A rights stop: no timed retry re-drives it.
    blocked = {"status": "blocked", "blockers": ["team_scene_runtime_source_redistribution_rights_not_granted"]}
    assert queue.refusal_retry_cause(blocked) is None and queue.transient_cause(blocked) is None


def test_the_package_strips_the_same_flatten_note_scene_packet_does():
    from blueprint_pipeline import scene_packet, website_company_simulation
    assert website_company_simulation.FLATTEN_PROVENANCE == scene_packet.FLATTEN_PROVENANCE
