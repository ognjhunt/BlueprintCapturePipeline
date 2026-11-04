"""A completed team half no stance carried from is compiled again once per stance-search version (beta L5).

capture-447ae4f4's half is ``completed`` and scorable: the Franka's carry search tried eight stances, each
``robot_rigid_pick_place_unreachable:approach``, and its station's reach proof put the pick and the place
beyond the arm. A completed half is reused by every later pass, and no release requeue moves a completed
request, so a better stance search would never have run for it. The worker now compiles such a half again,
once, on the CPU, when the running stance search is newer than the one that chose its station
(``team_embodiment_stance.STANCE_SEARCH_VERSION``), and a release that leaves the search alone compiles
nothing. The scene is the worker's real rigid fixture (tests/test_team_scene_packet_worker.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from blueprint_pipeline import team_embodiment_stance
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_scene_packet import read_team_scene_inputs
from tests.test_team_scene_packet_worker import RELEASE_1, RELEASE_2, _at, _release, _status, _world

GRASPS = ["task", "top_down", *(f"tilt{tilt}_yaw{yaw}" for tilt in (45, 90) for yaw in (0, 90, 180, 270))]
STALE = f"station_search_not_cleared:stance_search_v1<v{team_embodiment_stance.STANCE_SEARCH_VERSION}"


def _as_capture447_before_the_pedestal_search(half: str, *, cleared: bool = False) -> None:
    """Rewrite a compiled half's record as capture-447's was sealed, by a stance search before its version.

    Its station records no ``search`` (version 1). Its carry search tried eight stances with every grasp,
    each ``unreachable:approach``, and its reach proof put the pick and place beyond the Panda's bound.
    """

    path = Path(half)
    value = json.loads(path.read_text())
    station = {key: item for key, item in value["robot_station"].items() if key != "search"}
    station["stance_digest"] = canonical_digest(station, digest_field="stance_digest")
    value["robot_station"] = station
    digests = [station["stance_digest"], *("sha256:" + f"{index:x}" * 64 for index in range(1, 8))]
    blocked = ["robot_rigid_pick_place_unreachable:approach"]
    value["robot_station_search"] = {
        "rule": "first_ranked_stance_that_carries", "cleared": cleared, "chosen_rank": 0, "grasps_tried": 10,
        "results": [{"stance_digest": digest, "status": "clear" if cleared else "blocked",
                     "blockers": [] if cleared else list(blocked), "grasp": grasp}
                    for grasp in GRASPS for digest in digests],
        **({} if cleared else {
            "best_attempt": {"stance_digest": digests[0], "grasp": "tilt45_yaw0", "blockers": list(blocked),
                             "furthest_sample": 2, "path_samples": 42},
            "reach_shortfall": {"rule": "grasp_point_within_arm_chain_reach_bound_of_shoulder",
                                "robot_id": "franka_panda", "reach_bound_m": 0.9903, "status": "out_of_reach",
                                "beyond": ["place", "pick"], "grasp_independent": False,
                                "claim_ceiling": "cpu_bounded_estimate", "simulator_executed": False}})}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path.chmod(0o644)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    read_team_scene_inputs(half)  # still a sealed half every reader accepts


def test_a_completed_half_no_stance_carried_from_is_recompiled_once_per_stance_search_version(
        tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    first = status["team_scene_inputs_path"]
    # Every stance a search writes now names the search's version.
    station = json.loads(Path(first).read_text())["robot_station"]
    assert station["search"]["stance_search_version"] == team_embodiment_stance.STANCE_SEARCH_VERSION
    assert worker.stale_unreached_station(first) is None
    # As capture-447ae4f4's half was sealed: completed, eight stances all unreachable at approach.
    _as_capture447_before_the_pedestal_search(first)
    assert worker.stale_unreached_station(first) == STALE
    # Without a known running release nothing moves.
    _release(monkeypatch, None)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # The release with the newer stance search recompiles it, once, on the CPU, recorded before the move.
    _release(monkeypatch, RELEASE_2)
    version = team_embodiment_stance.STANCE_SEARCH_VERSION
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
        {"request_key": key, "reason": "stance_search_version", "detail": STALE, "stance_search_version": version,
         "release_commit": RELEASE_2}]
    assert (queue / "pending" / f"{key}.json").is_file()
    assert _status(world)[worker.STANCE_SEARCH_REQUEUE_FIELD] == [version]
    run = world["process"]()
    assert run["automatic_requeue"] == []
    [status] = run["results"]
    # Compiled again, never reused, and every reader now names the new half.
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    second = status["team_scene_inputs_path"]
    assert second != first and worker.stale_unreached_station(second) is None
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == second
    assert status[worker.STANCE_SEARCH_REQUEUE_FIELD] == [version]
    # Were the new search to find no carrying stance either, no later release recompiles it under this version.
    _as_capture447_before_the_pedestal_search(second)
    for release in (RELEASE_2, "c" * 40, "d" * 40):
        _release(monkeypatch, release)
        assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
        assert world["process"]()["automatic_requeue"] == []
    # Only a release that bumps the stance search's version tries it once more.
    monkeypatch.setattr(team_embodiment_stance, "STANCE_SEARCH_VERSION", version + 1)
    _release(monkeypatch, "e" * 40)
    [requeued] = worker.requeue_transient_blocked(queue_root=queue)["requeued"]
    assert (requeued["reason"], requeued["stance_search_version"]) == ("stance_search_version", version + 1)
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    assert status[worker.STANCE_SEARCH_REQUEUE_FIELD] == [version, version + 1]
    assert json.loads(Path(status["team_scene_inputs_path"]).read_text())["robot_station"]["search"][
        "stance_search_version"] == version + 1


def test_a_half_that_carries_or_was_searched_by_this_version_is_never_recompiled(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    queue = world["queue"]
    _release(monkeypatch, RELEASE_1)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    half = status["team_scene_inputs_path"]
    # A carry search that cleared, under the older search: the half is reused by every release.
    _as_capture447_before_the_pedestal_search(half, cleared=True)
    assert worker.stale_unreached_station(half) is None
    _release(monkeypatch, RELEASE_2)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # A half with no carry search (this fixture's push) is never one, nor an unreadable path.
    assert worker.stale_unreached_station(None) is None
    assert worker.stale_unreached_station(str(tmp_path / "missing.json")) is None
    # A changed record is not a sealed half: never read as one.
    value = json.loads(Path(half).read_text())
    value["robot_station_search"]["cleared"] = False
    Path(half).write_text(json.dumps(value))
    assert worker.stale_unreached_station(half) is None


def test_the_versions_tried_survive_an_interrupted_recompile(tmp_path, monkeypatch) -> None:
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    _as_capture447_before_the_pedestal_search(status["team_scene_inputs_path"])
    _release(monkeypatch, RELEASE_2)
    [requeued] = worker.requeue_transient_blocked(queue_root=queue)["requeued"]
    assert requeued["reason"] == "stance_search_version"
    # The recompile is killed mid-run (capture-447's compiles were, for host memory): claimed, never finished.
    os.replace(queue / "pending" / f"{key}.json", queue / "processing" / f"{key}.json")
    result: dict[str, Any] = worker.requeue_transient_blocked(queue_root=queue)
    assert result["recovered"] == [key]
    recovered = _status(world)
    assert recovered["blockers"] == ["team_scene_packet_interrupted"]
    assert recovered[worker.STANCE_SEARCH_REQUEUE_FIELD] == [team_embodiment_stance.STANCE_SEARCH_VERSION]
