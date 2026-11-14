# Covers: src/blueprint_pipeline/scene_packet.py
# Covers: src/blueprint_pipeline/scene_packet_queue.py
"""The new scene packet stage gives the kept readers what the old one gave them, on the same inputs.

Case A replays the committed synthetic record ``rigid-pick-place-1080p-short`` through the OLD stage
(``task_evaluation_downstream_replay`` until ``team_packet``) and, from a second copy stopped right
after the request is queued, through the NEW queue. Case B compiles the real ``proposal_only`` job of
the owner-confirmation tests with both stages, assumed thresholds. Case C compiles the articulated
drawer fixture. Each new half is handed to the real readers: the scripted check's reader
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
    # Intended difference 1: the old plan carries no success rule, so the download refused it; the new one does.
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
    # The download: refused for the old half, published for the new one, with the criteria in its rule.
    refused, _, _ = _download(tmp_path / "dl-old", old_status["team_scene_inputs_path"])
    assert refused["status"] != "published"
    published, rule, names = _download(tmp_path / "dl-new", status["team_scene_inputs_path"])
    assert published["status"] == "published", published
    assert rule["criteria"] and "success_rule.json" in names



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


def test_case_c_an_articulated_job_carries_its_success_rule_in_the_plan(tmp_path):
    scene = _articulated_scene(tmp_path)
    inputs = {key: scene[key] for key in ("revision", "documents", "configured_scene_bundle_path",
                                          "droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path")}
    old = materialize_team_scene(**inputs, output_root=tmp_path / "old" / "scene")
    compiled = new.compile_half(**inputs, output_root=tmp_path / "new" / "scene")
    old_plan, new_plan = _plan(old["path"]), _plan(compiled["path"])
    contract = _json(compiled["path"])["task_success_contract"]
    assert contract == _json(old["path"])["task_success_contract"]
    assert contract["provenance"]["author_source"] == "task_owner"
    # The old plan left the articulated contract out, so the download refused the job; the new plan carries it.
    assert "task_success_contract" not in old_plan["task_spec"]
    assert new_plan["task_spec"]["task_success_contract"] == contract
    assert _plan_fields(new_plan) == _plan_fields(old_plan)
    # Not yet shipped: an articulated plan states no lift axis, so the download refuses both halves before the
    # rule (``scene_package_stage.up_axis``). That is the download's to fix; this stage invents no axis.
    for name, half in (("dl-old", old["path"]), ("dl-new", compiled["path"])):
        refused, _, _ = _download(tmp_path / name, half)
        assert refused == {"status": "refused", "reason": "website_company_simulation_up_axis_not_stated",
                           "detail": ""}


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
