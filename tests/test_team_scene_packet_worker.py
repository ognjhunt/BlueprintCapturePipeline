"""A completed website scene's team half is compiled, recorded and offered without a person (beta L5).

The scene, its published revision and its configured bundle are the real rigid
fixture ``team_scene_packet`` compiles (tests/test_team_scene_packet.py). The
revision and its documents are fetched by reference from an in-memory store and
the WebApp transport is faked, so nothing reaches a provider, a GPU or the network.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as source
from blueprint_pipeline import team_scene_packet
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_candidate_quick10_registry import (
    REGISTRY_ENV,
    REGISTRY_SCHEMA,
    profile_for_capture,
    scene_facts,
)
from blueprint_pipeline.team_scene_packet import read_team_scene_inputs
from tests.test_team_scene_packet import _scene

REQUEST_ID = "req-1"
SCENE_ID, CAPTURE_ID = f"site-{REQUEST_ID}", f"walkthrough-{REQUEST_ID}"
RECEIPT_DIGEST = "sha256:" + "e" * 64


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _registry(operator: Path, *, packet_inputs: dict[str, Any] | None, profiles: list[dict] | None = None,
              defaults: dict[str, Any] | None = None) -> None:
    # The operator decides who may fund a run, at what rate, the salt and the fixed compile inputs; never the scene.
    derived: dict[str, Any] = {
        "root": str(operator / "derived"), "scored_matrix_salt_key_path": str(operator / "salt.key"),
        "defaults": {"allowed_founder_approvers": ["founder@blueprint"], "max_hourly_rate_usd": 0.8,
                     "open_loop_horizon": 8, **(defaults or {})}}
    if packet_inputs is not None:
        derived["packet_inputs"] = packet_inputs
    path = operator / "registry.json"
    path.write_text(json.dumps(_sealed({"schema_version": REGISTRY_SCHEMA, "profiles": profiles or [],
                                        "derived_profiles": derived}, "registry_digest")))
    path.chmod(0o644)


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, packet_inputs: bool = True) -> dict[str, Any]:
    scene = _scene(tmp_path)
    revision = scene["revision"]
    # The completed source launch, with its published revision and the configured bundle beside it.
    run = tmp_path / "launch-runs" / "scene-configuration-1"
    publication = run / "configured_scene_publication"
    publication.mkdir(parents=True)
    (publication / "configured_scene_revision.v1.json").write_text(json.dumps(revision, sort_keys=True))
    shutil.copyfile(scene["configured_scene_bundle_path"], publication / worker.CONFIGURED_BUNDLE_FILENAME)
    result = {"configured_scene_revision_reference": scene["revision_reference"],
              "configured_scene_revision_digest": revision["revision_digest"],
              "configured_scene_revision_path": str(publication / "configured_scene_revision.v1.json")}
    monkeypatch.setattr(source, "_validate_source",
                        lambda root: (dict(result), {"receipt_digest": RECEIPT_DIGEST}, {}))
    # The website capture inside the executor's partition, with the task context its scene's task is named for.
    partition = tmp_path / "partition"
    capture = partition / "scenes" / SCENE_ID / "captures" / CAPTURE_ID
    (capture / "pipeline").mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({
        "scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "site_submission_id": REQUEST_ID,
        "metadata": {"capture_entry_source": "browser_self_capture", "site_submission_id": REQUEST_ID}}))
    context = _sealed({"schema_version": "website_site_task_context.v1", "request_id": REQUEST_ID,
                       "scene_id": SCENE_ID, "capture_id": CAPTURE_ID, "description": "Move the box"},
                      "context_digest")
    (capture / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    monkeypatch.setenv(worker.CAPTURE_PARTITION_ENV, str(partition))
    monkeypatch.setenv("BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE", str(tmp_path / "rigid-robot-free-probes"))
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "salt.key").write_text("k" * 48)
    (operator / "salt.key").chmod(0o600)
    inputs = {"droid_profile_path": str(scene["droid_profile_path"]),
              "droid_profile_digest": scene["droid_profile_digest"],
              "runtime_source_receipt_path": str(scene["runtime_source_receipt_path"]),
              "output_root": str(tmp_path / "team-scene-halves")}
    _registry(operator, packet_inputs=inputs if packet_inputs else None)
    monkeypatch.setenv(REGISTRY_ENV, str(operator / "registry.json"))
    queue = tmp_path / "team-scene-packets"
    intent = {"intent_id": "scene-" + "1" * 64, "intent_digest": "sha256:" + "2" * 64,
              "request": {"submission_id": CAPTURE_ID,
                          "task": {"task_id": worker.WEBSITE_TASK_PREFIX + context["context_digest"][7:27]}}}
    queued = worker.enqueue_team_scene_request(
        intent=intent, run_root=run, receipt={"receipt_digest": RECEIPT_DIGEST}, result=result,
        readiness={"status": "probe_unavailable", "retained": True}, queue_root=queue)
    assert queued["status"] == "enqueued"
    calls: list[dict[str, Any]] = []

    def transport(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "accepted"}

    def process() -> dict[str, Any]:
        return worker.process_team_scene_queue(
            queue_root=queue, fetcher=lambda reference: scene["store"][reference["uri"]], offer_transport=transport)

    return {"scene": scene, "run": run, "capture": capture.resolve(), "context": context, "operator": operator,
            "queue": queue, "inputs": inputs, "calls": calls, "key": queued["request_key"], "process": process,
            "halves": Path(inputs["output_root"]) / worker.capture_key(capture.resolve())}


def _passed_rigid_probe(profile: dict[str, Any]) -> dict[str, Any]:
    """The receipt the rigid probe lane retains once it scores the compiled scored specification."""

    facts = scene_facts(profile)
    task_spec = facts["scene"]["task_spec"]
    return _sealed({"schema_version": gate.RIGID_RECEIPT_SCHEMA, "status": "passed",
                    "configured_scene_revision_digest": facts["scene"]["scene_revision_digest"],
                    "task_spec_digest": canonical_digest(dict(task_spec)),
                    "task_success_contract_digest": facts["inputs"]["task_success_contract"]["contract_digest"],
                    "learned_policy_evaluation_admitted": True,
                    "checks": dict.fromkeys(gate.REQUIRED_CHECKS["rigid_pick_place"], True),
                    "proof_boundary": {"robot_embodiment_used": False}}, "receipt_digest")


def test_a_completed_website_scene_is_compiled_recorded_and_offered_without_a_person(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    first = world["process"]()
    assert first["status"] == "processed" and first["provider_mutation_performed"] is False
    [status] = first["results"]
    assert (status["status"], status["outcome"], status["blockers"]) == ("completed", "compiled", [])
    # The derived row the team registry reads: this capture, the nominal scenario, the launch's probe root.
    profile = profile_for_capture(world["capture"])
    assert profile["scenario_id"] == worker.SCENARIO_ID == "nominal"
    assert profile["team_scene_inputs_path"] == status["team_scene_inputs_path"]
    assert profile["robot_free_probe_root"] == str(world["run"].resolve())
    assert profile["scene_revision_digest"] == world["scene"]["revision"]["revision_digest"]
    assert Path(profile["team_scene_inputs_path"]).is_relative_to(Path(status["attempt_root"]))
    assert Path(status["attempt_root"]).parent == world["halves"]
    assert read_team_scene_inputs(profile["team_scene_inputs_path"])["scene_revision_digest"] == (
        profile["scene_revision_digest"])
    # No team body is registered: the half is the Franka-only one, with no body stations and nothing named.
    assert "embodiment_stations" not in json.loads(Path(profile["team_scene_inputs_path"]).read_text())
    assert "embodiment_manifests_unreadable" not in status
    # The offer was republished; a rigid scene stays unlisted, by name, until its probe passes.
    assert status["offer"]["team_evaluation_listed"] is False
    assert status["offer"]["team_candidate_offer_refused"] == "team_evaluation_robot_free_probe_receipt_missing"
    assert world["calls"] == []
    assert json.loads((queue / "status" / f"{key}.json").read_text()) == status
    assert not list((queue / "pending").iterdir()) and (queue / "processed" / f"{key}.json").is_file()

    # The probe lane passes on the compiled specification; delivering the same request again reuses the
    # recorded half (no second compile, no rewritten row) and republishes the job's offer, now listed.
    (world["run"] / gate.RECEIPT_FILENAME).write_text(json.dumps(_passed_rigid_probe(profile)))
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **_: pytest.fail("compiled twice"))
    shutil.copyfile(queue / "processed" / f"{key}.json", queue / "pending" / f"{key}.json")
    [again] = world["process"]()["results"]
    assert (again["status"], again["outcome"]) == ("completed", "already_compiled")
    assert again["team_scene_inputs_path"] == profile["team_scene_inputs_path"]
    assert again["offer"]["status"] == "published" and again["offer"]["team_evaluation_listed"] is True
    [call] = world["calls"]
    assert (call["capture_id"], call["operation"]) == (CAPTURE_ID, "agent-execution-offer")
    offer = call["payload"]["offer"]
    assert (offer["scenario_id"], offer["capture_root"]) == ("nominal", str(world["capture"]))
    assert offer["team_evaluation"]["contract"]["task_kind"] == "rigid_pick_place"
    assert profile_for_capture(world["capture"]) == profile
    assert [path.name for path in world["halves"].iterdir()] == [Path(status["attempt_root"]).name]


def test_every_registered_team_body_is_compiled_with_the_scene_at_its_own_station(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from tests.test_team_scene_packet import bodies

    world = _world(tmp_path, monkeypatch)
    body = bodies(tmp_path)
    records = {name: launch._record(launch._write(tmp_path / f"{name}.manifest.json", value))
               for name, value in body.items()}
    changed = {**records["short"], "sha256": "sha256:" + "0" * 64}
    # The long arm's own model, as registered; the rover is registered with a model that is not its own.
    arm = launch._record(tmp_path / "robot.usda")
    _registry(world["operator"], packet_inputs=world["inputs"], defaults={"screening_candidates": {
        "team_long": {"robot_policy": {"manifest": records["long"], "model": arm}},
        "team_long_again": {"robot_policy": {"manifest": records["long"]}},
        "team_rover": {"robot_policy": {"manifest": records["wheeled"], "model": arm}},
        "team_changed": {"robot_policy": {"manifest": changed}},
        "team_g1": {"native_g1": {"registered": "with its own scene packet"}}}})
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    # A manifest whose bytes changed is named and left out; nothing is guessed for it.
    assert status["embodiment_manifests_unreadable"] == ["team_changed"]
    record = json.loads(Path(status["team_scene_inputs_path"]).read_text())
    assert set(record["embodiment_stations"]) == {body["long"]["embodiment_digest"],
                                                  body["wheeled"]["embodiment_digest"]}
    assert read_team_scene_inputs(status["team_scene_inputs_path"])["scene_revision_digest"] == (
        world["scene"]["revision"]["revision_digest"])
    # Only the model that is the body's own is handed on: the long arm's station is also solved with its arm.
    assert worker.registered_embodiment_models(json.loads((world["operator"] / "registry.json").read_text())) == {
        body["long"]["embodiment_digest"]: tmp_path / "robot.usda"}
    long = record["embodiment_stations"][body["long"]["embodiment_digest"]]
    assert long["reach"]["own_arm"]["rule"] == "closest_clear_stance_own_arm_reaches_every_phase"
    assert "reach" not in record["embodiment_stations"][body["wheeled"]["embodiment_digest"]] or (
        record["embodiment_stations"][body["wheeled"]["embodiment_digest"]]["reach"] is None)


@pytest.mark.parametrize("owner", ["operator_row", "frozen_pair"])
def test_an_operator_row_for_the_capture_wins_and_nothing_is_compiled(tmp_path, monkeypatch, owner):
    from blueprint_pipeline.team_candidate_quick10_registry import write_derived_profile

    world = _world(tmp_path, monkeypatch)
    row = {"capture_root": str(world["capture"]), "scene_id": SCENE_ID, "task_id": "pick_place",
           "scenario_id": "operator-chosen", "robot_free_probe_root": str(world["run"]),
           "scene_revision_digest": world["scene"]["revision"]["revision_digest"]}
    if owner == "operator_row":
        _registry(world["operator"], packet_inputs=world["inputs"], profiles=[row])
    else:  # the operator's frozen internal pair, recorded as this capture's derived row
        pair = {"pair_runtime_inputs_path": str(tmp_path / "pair-inputs.json"),
                "pair_activation_path": str(tmp_path / "pair-activation.json")}
        assert write_derived_profile({**row, **pair})["status"] == "written"
    before = profile_for_capture(world["capture"])
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **_: pytest.fail("compiled beside a row"))
    [status] = world["process"]()["results"]
    assert (status["status"], status["reason"], status["blockers"]) == (
        "skipped", {"operator_row": "team_candidate_operator_row_present",
                    "frozen_pair": "team_candidate_frozen_pair_present"}[owner], [])
    assert profile_for_capture(world["capture"]) == before
    assert not world["halves"].exists() and world["calls"] == []


def test_a_registry_without_the_packet_inputs_block_skips_by_name_and_an_invalid_one_refuses(
        tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, packet_inputs=False)
    [status] = world["process"]()["results"]
    assert (status["status"], status["reason"]) == ("skipped", "team_scene_packet_inputs_not_configured")
    # An older registry still reads as before: no derived row, nothing compiled, nothing posted.
    assert profile_for_capture(world["capture"]) is None and not world["halves"].exists() and not world["calls"]
    # Once the operator adds the block, the skipped request goes back to its queue for one more run.
    assert worker.requeue_unfinished(queue_root=world["queue"])["requeued"] == [world["key"]]
    assert (world["queue"] / "pending" / f"{world['key']}.json").is_file()
    for broken in ({**world["inputs"], "droid_profile_digest": "not-a-digest"},
                   {**world["inputs"], "output_root": "relative/root"},
                   {key: value for key, value in world["inputs"].items() if key != "output_root"}):
        _registry(world["operator"], packet_inputs=broken)
        [status] = world["process"]()["results"]
        assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_packet_inputs_invalid"])
        worker.requeue_unfinished(queue_root=world["queue"])
    Path(world["inputs"]["droid_profile_path"]).chmod(0o666)
    _registry(world["operator"], packet_inputs=world["inputs"])
    [status] = world["process"]()["results"]
    assert status["blockers"] == ["team_scene_packet_inputs_not_operator_owned"]


def test_a_refused_compile_leaves_no_partial_half_and_its_retry_compiles_cleanly(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    # The revision's documents are fetched into the attempt before the DROID profile's binding refuses.
    _registry(world["operator"], packet_inputs={**world["inputs"], "droid_profile_digest": "sha256:" + "0" * 64})
    [status] = world["process"]()["results"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_droid_profile_binding_invalid"])
    assert "error" not in status  # A typed refusal is its own cause.
    assert list(world["halves"].iterdir()) == []
    assert profile_for_capture(world["capture"]) is None and world["calls"] == []
    _registry(world["operator"], packet_inputs=world["inputs"])
    assert worker.requeue_unfinished(queue_root=world["queue"])["requeued"] == [world["key"]]
    [retried] = world["process"]()["results"]
    assert (retried["status"], retried["outcome"]) == ("completed", "compiled")
    assert [path.name for path in world["halves"].iterdir()] == [Path(retried["attempt_root"]).name]
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == retried["team_scene_inputs_path"]
    # A completed request is never requeued.
    assert worker.requeue_unfinished(queue_root=world["queue"])["requeued"] == []


def test_an_untyped_compile_failure_keeps_its_bounded_cause_beside_the_type_code(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device", "/var/lib/halves/token=abc")
    monkeypatch.setattr(worker, "_compile", full)
    [status] = world["process"]()["results"]
    # The code stays the type name; the record keeps the redacted, trimmed message.
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_packet_failed:OSError"])
    assert status["error"] == {"type": "builtins.OSError",
                               "message": "[Errno 28] No space left on device: '/var/lib/halves/token=<redacted>"}
    assert json.loads((world["queue"] / "status" / f"{world['key']}.json").read_text())["error"] == status["error"]


def test_an_interrupted_attempt_is_blocked_by_name_and_its_unnamed_directory_removed(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    # A run killed mid-compile: its request is still claimed and its attempt half-written.
    os.replace(queue / "pending" / f"{key}.json", queue / "processing" / f"{key}.json")
    attempt = world["halves"] / ("0" * 16 + "-" + "0" * 8)
    (attempt / "documents").mkdir(parents=True)
    (queue / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "processing", "attempt_root": str(attempt), "capture_root": str(world["capture"])}))
    run = world["process"]()
    assert run["results"] == [] and [row["blockers"] for row in run["recovered"]] == [
        ["team_scene_packet_interrupted"]]
    assert not attempt.exists() and (queue / "processed" / f"{key}.json").is_file()
    assert worker.requeue_unfinished(queue_root=queue)["requeued"] == [key]


@pytest.mark.parametrize("change", ["task_context", "withdrawn", "partition", "bundle"])
def test_a_capture_or_launch_that_no_longer_binds_the_scene_is_refused_by_name(tmp_path, monkeypatch, change):
    world = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **_: pytest.fail("compiled unbound"))
    if change == "task_context":  # the site's brief changed: this scene was prepared for another task
        context = _sealed({**world["context"], "description": "Another job"}, "context_digest")
        (world["capture"] / "pipeline" / "website_task_context.json").write_text(json.dumps(context))
    elif change == "withdrawn":
        tombstone = _sealed({"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": SCENE_ID,
                             "request_id": REQUEST_ID}, "digest")
        marker = world["capture"].parent.parent / "website_withdrawal" / "tombstone.json"
        marker.parent.mkdir()
        marker.write_text(json.dumps(tombstone))
    elif change == "partition":
        monkeypatch.delenv(worker.CAPTURE_PARTITION_ENV)
    else:
        (world["run"] / "configured_scene_publication" / worker.CONFIGURED_BUNDLE_FILENAME).unlink()
    [status] = world["process"]()["results"]
    assert status["status"] == "blocked" and status["blockers"] == [{
        "task_context": "team_scene_website_task_context_mismatch", "withdrawn": "team_scene_capture_withdrawn",
        "partition": "team_scene_capture_partition_not_configured",
        "bundle": "team_scene_configured_bundle_unavailable"}[change]]
    assert profile_for_capture(world["capture"]) is None and world["calls"] == []


def test_the_worker_command_needs_a_queue_and_reports_an_idle_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(worker.QUEUE_ROOT_ENV, raising=False)
    assert worker.main([]) == 2
    assert json.loads(capsys.readouterr().out)["blockers"] == ["team_scene_packet_queue_root_not_configured"]
    assert worker.main(["--queue-root", str(tmp_path / "queue")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "idle"
    with pytest.raises(worker.TeamScenePacketRequestRefused, match="queue_root_unsafe"):
        worker.process_team_scene_queue(queue_root="relative/queue")


T0 = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)


def _at(monkeypatch: pytest.MonkeyPatch, seconds: float) -> datetime:
    moment = T0 + timedelta(seconds=seconds)
    monkeypatch.setattr(worker, "_utc_now", lambda: moment)
    return moment


def _status(world: dict[str, Any]) -> dict[str, Any]:
    return json.loads((world["queue"] / "status" / f"{world['key']}.json").read_text())


def test_a_failed_offer_post_is_retried_automatically_after_each_wait_and_then_stays_blocked(
        tmp_path, monkeypatch):
    from blueprint_pipeline import agent_execution_offer

    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    posts: list[Path] = []
    # One WebApp outage after another: the half is recorded once, only the offer post keeps failing.
    monkeypatch.setattr(agent_execution_offer, "publish_agent_execution_offer",
                        lambda capture, **_: posts.append(capture) or {"status": "failed", "reason": "http_503"})
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    assert (status["status"], status["blockers"]) == ("blocked", ["team_scene_offer_publication_failed"])
    assert status["automatic_retry"] == {"attempts": 0, "limit": 3, "reason": "team_scene_offer_publication_failed",
                                         "not_before": "2026-10-31T12:05:00+00:00"}
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **_: pytest.fail("compiled twice"))
    elapsed = 0
    for attempt, wait in enumerate(worker.AUTOMATIC_RETRY_BACKOFF_SECONDS, start=1):
        # Not before its wait: the timer's pass leaves it where it is.
        _at(monkeypatch, elapsed + wait - 1)
        assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
        assert (queue / "processed" / f"{key}.json").is_file()
        moment = _at(monkeypatch, elapsed + wait)
        assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
            {"request_key": key, "attempt": attempt, "reason": "team_scene_offer_publication_failed"}]
        assert (queue / "pending" / f"{key}.json").is_file()
        assert _status(world)["automatic_retry"]["requeued_at"] == moment.isoformat(timespec="seconds")
        [status] = world["process"]()["results"]
        assert (status["outcome"], status["blockers"]) == ("already_compiled", ["team_scene_offer_publication_failed"])
        assert status["automatic_retry"]["attempts"] == attempt
        elapsed += wait
    # Every automatic retry is spent: it stays blocked, by name, until someone fixes the cause.
    assert status["automatic_retry"] == {"attempts": 3, "limit": 3, "reason": "team_scene_offer_publication_failed",
                                         "requeued_at": moment.isoformat(timespec="seconds"), "exhausted": True}
    _at(monkeypatch, 10 ** 7)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    assert world["process"]()["automatic_requeue"] == []
    assert len(posts) == 4 and _status(world) == status
    assert worker.requeue_unfinished(queue_root=queue)["requeued"] == [key]


def test_a_run_killed_by_its_timeout_is_recovered_by_the_timer_pass_and_compiled_on_its_retry(
        tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    # The unit's timeout killed a compile: the request is still claimed and its attempt half-written.
    os.replace(queue / "pending" / f"{key}.json", queue / "processing" / f"{key}.json")
    attempt = world["halves"] / ("0" * 16 + "-" + "0" * 8)
    (attempt / "documents").mkdir(parents=True)
    (queue / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "processing", "attempt_root": str(attempt), "capture_root": str(world["capture"])}))
    _at(monkeypatch, 0)
    # Nothing is pending, so the path unit never starts the worker; a run holding the queue is left alone.
    with worker._locked(queue):
        assert worker.requeue_transient_blocked(queue_root=queue)["status"] == "busy"
    assert (queue / "processing" / f"{key}.json").is_file()
    swept = worker.requeue_transient_blocked(queue_root=queue)
    assert (swept["recovered"], swept["requeued"]) == ([key], [])
    assert not attempt.exists() and (queue / "processed" / f"{key}.json").is_file()
    assert _status(world)["blockers"] == ["team_scene_packet_interrupted"]
    assert _status(world)["automatic_retry"]["not_before"] == "2026-10-31T12:05:00+00:00"
    _at(monkeypatch, 300)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
        {"request_key": key, "attempt": 1, "reason": "team_scene_packet_interrupted"}]
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    # The completed status keeps how it got there.
    assert status["automatic_retry"] == {"attempts": 1, "limit": 3, "reason": "team_scene_packet_interrupted",
                                         "requeued_at": "2026-10-31T12:05:00+00:00"}
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == status["team_scene_inputs_path"]


def test_the_worker_pass_requeues_a_due_block_recorded_before_automatic_retries(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    os.replace(queue / "pending" / f"{key}.json", queue / "processed" / f"{key}.json")
    (queue / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "blocked", "blockers": ["team_scene_packet_failed:OSError"]}))
    # The transient path alone: with a known release, a status naming none takes the new-release retry first.
    monkeypatch.setattr(worker, "current_release_commit", lambda: None)
    _at(monkeypatch, 0)
    run = world["process"]()
    assert run["automatic_requeue"] == [{"request_key": key, "attempt": 1, "reason": "team_scene_packet_failed:OSError"}]
    [status] = run["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")


RELEASE_1, RELEASE_2 = "a" * 40, "b" * 40


def _release(monkeypatch: pytest.MonkeyPatch, commit: str | None) -> None:
    monkeypatch.setattr(worker, "current_release_commit", lambda: commit)


def test_an_unlisted_refusal_is_retried_on_the_shared_schedule_six_times_then_stays_under_its_release(
        tmp_path, monkeypatch):
    from blueprint_pipeline.job_retry_policy import REFUSAL_RETRY_DELAYS_SECONDS

    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    _registry(world["operator"], packet_inputs={**world["inputs"], "droid_profile_digest": "sha256:" + "0" * 64})
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    assert status["blockers"] == ["team_scene_droid_profile_binding_invalid"] and "automatic_retry" not in status
    assert status[worker.RELEASE_COMMIT_FIELD] == RELEASE_1
    assert status[worker.REFUSAL_RETRY_FIELD] == {"attempts": 0, "limit": 6,
                                                  "reason": "team_scene_droid_profile_binding_invalid",
                                                  "not_before": "2026-10-31T12:15:00+00:00"}
    assert not worker.final_block(status, release=RELEASE_1)
    elapsed = 0
    # No deploy: the clock alone retries it, alternately from the timer's pass and the worker's own.
    for attempt, wait in enumerate(REFUSAL_RETRY_DELAYS_SECONDS, start=1):
        _at(monkeypatch, elapsed + wait - 1)
        assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
        assert world["process"]()["automatic_requeue"] == []
        _at(monkeypatch, elapsed + wait)
        expected = [{"request_key": key, "reason": "refusal_retry", "attempt": attempt,
                     "blocker": "team_scene_droid_profile_binding_invalid"}]
        if attempt % 2:
            assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == expected
            [status] = world["process"]()["results"]
        else:
            run = world["process"]()
            assert run["automatic_requeue"] == expected
            [status] = run["results"]
        assert status["blockers"] == ["team_scene_droid_profile_binding_invalid"]
        assert status[worker.REFUSAL_RETRY_FIELD]["attempts"] == attempt
        elapsed += wait
    # Spent: it stays blocked under this release, final, from either pass, however long it waits.
    assert status[worker.REFUSAL_RETRY_FIELD]["exhausted"] is True
    assert "not_before" not in status[worker.REFUSAL_RETRY_FIELD]
    assert worker.final_block(status, release=RELEASE_1)
    _at(monkeypatch, 10 ** 7)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    assert world["process"]()["automatic_requeue"] == []
    assert (queue / "processed" / f"{key}.json").is_file()
    # A new release still gets its own one try, and does not reset the count: refused again, still spent.
    _release(monkeypatch, RELEASE_2)
    assert [row["reason"] for row in worker.requeue_transient_blocked(queue_root=queue)["requeued"]] == [
        "new_release"]
    [status] = world["process"]()["results"]
    assert status[worker.REFUSAL_RETRY_FIELD]["attempts"] == 6 and status[worker.REFUSAL_RETRY_FIELD]["exhausted"]
    _at(monkeypatch, 2 * 10 ** 7)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # The cause fixed and a timed retry left: the clock re-drives it and it compiles, keeping the count.
    _registry(world["operator"], packet_inputs=world["inputs"])
    (queue / "status" / f"{key}.json").write_text(json.dumps(
        {**status, worker.REFUSAL_RETRY_FIELD: {**status[worker.REFUSAL_RETRY_FIELD], "attempts": 5,
                                                "not_before": "2026-10-31T12:00:00+00:00"}}))
    assert [row["attempt"] for row in worker.requeue_transient_blocked(queue_root=queue)["requeued"]] == [6]
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    assert status[worker.REFUSAL_RETRY_FIELD]["attempts"] == 6


def test_a_deploy_that_fixes_a_deterministic_refusal_re_drives_it_once_per_release(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    broken = {**world["inputs"], "droid_profile_digest": "sha256:" + "0" * 64}
    _registry(world["operator"], packet_inputs=broken)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    assert (status["blockers"], status[worker.RELEASE_COMMIT_FIELD]) == (
        ["team_scene_droid_profile_binding_invalid"], RELEASE_1)
    # A new release is deployed: the timer's pass returns the refusal to pending once, recorded before the move.
    _release(monkeypatch, RELEASE_2)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
        {"request_key": key, "reason": "new_release", "release_commit": RELEASE_2,
         "blocked_release_commit": RELEASE_1}]
    assert (queue / "pending" / f"{key}.json").is_file()
    assert _status(world)[worker.RELEASE_REQUEUE_FIELD] == [RELEASE_2]
    # This release did not fix it: refused again, and never retried again under this release, from either pass.
    [status] = world["process"]()["results"]
    assert (status["blockers"], status[worker.RELEASE_COMMIT_FIELD], status[worker.RELEASE_REQUEUE_FIELD]) == (
        ["team_scene_droid_profile_binding_invalid"], RELEASE_2, [RELEASE_2])
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    assert world["process"]()["automatic_requeue"] == []
    # Even a status naming an older release is not retried for a release already tried.
    (queue / "status" / f"{key}.json").write_text(json.dumps({**status, worker.RELEASE_COMMIT_FIELD: RELEASE_1}))
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # The next release fixes the cause: the worker's own pass re-drives and compiles it.
    _registry(world["operator"], packet_inputs=world["inputs"])
    _release(monkeypatch, "c" * 40)
    run = world["process"]()
    assert run["automatic_requeue"] == [{"request_key": key, "reason": "new_release", "release_commit": "c" * 40,
                                         "blocked_release_commit": RELEASE_1}]
    [status] = run["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    assert status[worker.RELEASE_REQUEUE_FIELD] == [RELEASE_2, "c" * 40]
    # A completed request is never requeued, by any release.
    _release(monkeypatch, "d" * 40)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []


def _spawn_off_start(half: str, offset_m: float) -> None:
    """Rewrite a compiled half's plan as a release before the seating fix compiled capture-447's (14.7 mm)."""

    plan_path = Path(read_team_scene_inputs(half)["base_native_packet"]["path"]).parent / (
        "native_task_arena_scene_plan.v1.json")
    plan = json.loads(plan_path.read_text())
    (row,) = [row for row in plan["objects"] if row.get("task_subject") is True]
    row["reset_state"]["root_pose_world"]["position_world_m"][2] += offset_m
    plan_path.chmod(0o644)
    plan_path.write_text(json.dumps(plan))


def test_a_half_no_episode_can_be_scored_from_is_recompiled_once_per_new_release(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    first = status["team_scene_inputs_path"]
    assert worker.unscorable_half(first) is None
    # As capture-447ae4f4's half was compiled: its tray spawned 14.7 mm from the start the scorer reads.
    _spawn_off_start(first, 0.0147)
    assert worker.unscorable_half(first) == "rigid_task_start_pose_not_spawn_pose:14.7mm>2.0mm"
    # The release that compiled it never recompiles it ...
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # ... a newly deployed one does, once, on the CPU, recorded before the move.
    _release(monkeypatch, RELEASE_2)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
        {"request_key": key, "reason": "unscorable_half",
         "detail": "rigid_task_start_pose_not_spawn_pose:14.7mm>2.0mm", "release_commit": RELEASE_2}]
    assert _status(world)[worker.UNSCORABLE_HALF_REQUEUE_FIELD] == [RELEASE_2]
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    assert status["team_scene_inputs_path"] != first and worker.unscorable_half(status["team_scene_inputs_path"]) is None
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == status["team_scene_inputs_path"]
    assert status[worker.UNSCORABLE_HALF_REQUEUE_FIELD] == [RELEASE_2]
    # Even were it still unscorable, this release never tries it again: no loop.
    _spawn_off_start(status["team_scene_inputs_path"], 0.0147)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    assert world["process"]()["automatic_requeue"] == []
    # A half within the scorer's reset tolerance is reused by every release.
    _spawn_off_start(status["team_scene_inputs_path"], -0.0147 + 0.0015)
    _release(monkeypatch, "c" * 40)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []


def _as_compiled_before_the_support_pad(half: str) -> None:
    """Rewrite a compiled half's record as a release before the support pad wrote capture-447's: no support record."""

    path = Path(half)
    value = json.loads(path.read_text())
    value.pop("task_support_pad")
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path.chmod(0o644)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_a_rigid_half_compiled_before_its_support_was_checked_is_recompiled_once_per_new_release(
        tmp_path, monkeypatch):
    from blueprint_pipeline.rigid_support_pad import UNMODELLED_HALF

    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    _at(monkeypatch, 0)
    [status] = world["process"]()["results"]
    first = status["team_scene_inputs_path"]
    # This fixture's replacement has no qualified collision bounds: its support is recorded unchecked, by name.
    assert json.loads(Path(first).read_text())["task_support_pad"]["reason"] == (
        "rigid_support_qualified_collision_bounds_missing")
    assert worker.unscorable_half(first) is None
    # As capture-447ae4f4's half was compiled (release 9f47f0466): its tray rested on the scan, unchecked.
    _as_compiled_before_the_support_pad(first)
    assert worker.unscorable_half(first) == UNMODELLED_HALF
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    # The release that models the support recompiles it, once, on the CPU; the new half is reused after.
    _release(monkeypatch, RELEASE_2)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == [
        {"request_key": key, "reason": "unscorable_half", "detail": UNMODELLED_HALF, "release_commit": RELEASE_2}]
    [status] = world["process"]()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled")
    second = status["team_scene_inputs_path"]
    assert second != first and "task_support_pad" in json.loads(Path(second).read_text())
    assert worker.unscorable_half(second) is None
    assert profile_for_capture(world["capture"])["team_scene_inputs_path"] == second
    _release(monkeypatch, "c" * 40)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []


def test_a_refusal_recorded_without_a_release_requeues_once_under_a_known_release(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    os.replace(queue / "pending" / f"{key}.json", queue / "processed" / f"{key}.json")
    # As capture-447's request was left: a deterministic refusal written before statuses named their release.
    legacy = {"schema_version": worker.STATUS_SCHEMA, "request_key": key, "status": "blocked",
              "blockers": ["native_task_appearance_layer_transform_spurious"],
              worker.OWNER_CONFIRMATION_REQUEUE_FIELD: ["sha256:" + "4" * 64]}
    (queue / "status" / f"{key}.json").write_text(json.dumps(legacy))
    _at(monkeypatch, 0)
    # Without a known running release nothing is requeued on that basis, from either pass.
    _release(monkeypatch, None)
    assert worker.requeue_transient_blocked(queue_root=queue)["requeued"] == []
    assert world["process"]()["automatic_requeue"] == []
    # The shared schedule's clock starts at the first pass that sees it: it is not due at once.
    started = _status(world)
    assert started.pop(worker.REFUSAL_RETRY_FIELD)["not_before"] == "2026-10-31T12:15:00+00:00"
    assert started == legacy
    _release(monkeypatch, RELEASE_1)
    run = world["process"]()
    assert run["automatic_requeue"] == [{"request_key": key, "reason": "new_release", "release_commit": RELEASE_1,
                                         "blocked_release_commit": None}]
    [status] = run["results"]
    assert (status["status"], status[worker.RELEASE_COMMIT_FIELD]) == ("completed", RELEASE_1)
    assert status[worker.RELEASE_REQUEUE_FIELD] == [RELEASE_1]
    assert status[worker.OWNER_CONFIRMATION_REQUEUE_FIELD] == ["sha256:" + "4" * 64]


@pytest.mark.parametrize(("status", "release", "reason"), [
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}, RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"], "release_commit": RELEASE_1},
     RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"], "release_commit": RELEASE_2},
     RELEASE_2, None),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"], "release_requeued": [RELEASE_2]},
     RELEASE_2, None),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}, None, None),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}, "not-a-commit", None),
    ({"status": "completed", "blockers": [], "release_commit": RELEASE_1}, RELEASE_2, None),
    ({"status": "completed", "blockers": []}, RELEASE_2, None),
    ({"status": "skipped", "reason": "team_candidate_operator_row_present", "blockers": [],
      "release_commit": RELEASE_1}, RELEASE_2, None),
    ({}, RELEASE_2, None),
    # An owner-contract refusal is one too: nothing waits for the owner's confirmation any more, so a request
    # an earlier release left waiting for it is compiled by the next one (on assumed thresholds).
    ({"status": "blocked", "blockers": [worker.OWNER_CONTRACT_BLOCKER_PREFIX + "authority_missing"],
      "release_commit": RELEASE_1}, RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": [worker.OWNER_CONTRACT_BLOCKER_PREFIX + "authority_missing"],
      "release_commit": RELEASE_2}, RELEASE_2, None),
    # A new release gets one try at a transient block at once: inside its backoff, after its retries are
    # spent, or due (capture-447ae4f4, 2026-11-03: killed three times for host memory, then two hours from
    # its last retry when the memory fix shipped).
    ({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"], "release_commit": RELEASE_1,
      "automatic_retry": {"attempts": 0, "not_before": "2026-10-31T12:05:00+00:00"}}, RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"], "release_commit": RELEASE_1,
      "automatic_retry": {"attempts": 3, "exhausted": True}}, RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted"], "release_commit": RELEASE_1,
      "automatic_retry": {"attempts": 2, "limit": 3, "not_before": "2026-10-31T14:00:00+00:00",
                          "reason": "team_scene_packet_interrupted"}}, RELEASE_2, "new_release"),
    ({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"], "release_commit": RELEASE_1},
     RELEASE_2, "new_release"),
    # Under the same release a transient block keeps its own bounded backoff: not before its wait, never
    # past its limit, and once due, as a transient retry.
    ({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"], "release_commit": RELEASE_2,
      "automatic_retry": {"attempts": 0, "not_before": "2026-10-31T12:05:00+00:00"}}, RELEASE_2, None),
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted"], "release_commit": RELEASE_2,
      "automatic_retry": {"attempts": 3, "limit": 3}}, RELEASE_2, None),
    ({"status": "blocked", "blockers": ["team_scene_offer_publication_failed"], "release_commit": RELEASE_2},
     RELEASE_2, "team_scene_offer_publication_failed"),
])
def test_only_a_deterministic_refusal_is_requeued_by_a_new_release(tmp_path, status, release, reason):
    queue = worker._queue(tmp_path / "queue")
    (queue / "processed" / "k.json").write_text("{}")
    (queue / "status" / "k.json").write_text(json.dumps(status))
    requeued = worker._requeue_due(queue, T0, release)
    assert [row["reason"] for row in requeued] == ([reason] if reason else [])
    assert (queue / ("pending" if reason else "processed") / "k.json").is_file()
    if reason == "new_release":
        assert json.loads((queue / "status" / "k.json").read_text())[worker.RELEASE_REQUEUE_FIELD] == [release]
        # Back in processed with the same status: never twice for one release (a transient block may
        # still take its own bounded retry under it).
        os.replace(queue / "pending" / "k.json", queue / "processed" / "k.json")
        assert "new_release" not in [row["reason"] for row in worker._requeue_due(queue, T0, release)]


def test_the_running_release_is_the_release_checkouts_commit_or_unknown(monkeypatch):
    from blueprint_pipeline import task_evaluation_release_identity as identity

    monkeypatch.setattr(identity, "running_release_commit", lambda _path=None: RELEASE_1)
    assert worker.current_release_commit() == RELEASE_1
    # A checkout whose HEAD cannot be read falls back to the release directory named by its commit.
    monkeypatch.setattr(identity, "running_release_commit", lambda _path=None: "")
    monkeypatch.setattr(worker, "__file__",
                        f"/opt/blueprint/task-evaluation-control-plane-releases/{RELEASE_2}/src/blueprint_pipeline/"
                        "team_scene_packet_worker.py")
    assert worker.current_release_commit() == RELEASE_2
    monkeypatch.setattr(worker, "__file__", "/srv/checkout/src/blueprint_pipeline/x.py")
    assert worker.current_release_commit() is None

    def unreadable(_path=None):
        raise OSError("no checkout")

    monkeypatch.setattr(identity, "running_release_commit", unreadable)
    assert worker.current_release_commit() is None


@pytest.mark.parametrize(("status", "cause"), [
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted"]}, "team_scene_packet_interrupted"),
    ({"status": "blocked", "blockers": ["configured_controls_deferred_fetch_failed"]},
     "configured_controls_deferred_fetch_failed"),
    ({"status": "blocked", "blockers": ["team_scene_packet_failed:TimeoutError"]},
     "team_scene_packet_failed:TimeoutError"),
    ({"status": "blocked", "blockers": ["team_scene_packet_failed:KeyError"]}, None),
    ({"status": "blocked", "blockers": ["team_scene_packet_failed:__class__"]}, None),
    ({"status": "blocked", "blockers": ["team_scene_capture_withdrawn"]}, None),
    ({"status": "blocked", "blockers": ["team_scene_packet_interrupted", "team_scene_capture_withdrawn"]}, None),
    ({"status": "skipped", "blockers": ["team_scene_packet_interrupted"]}, None),
])
def test_only_a_cause_a_later_run_can_clear_is_transient(status, cause):
    assert worker.transient_cause(status) == cause


BLOCKED_AT = {"status": "blocked", "release_commit": RELEASE_1,
              "refusal_retry": {"attempts": 0, "not_before": "2026-10-31T12:15:00+00:00"}}


@pytest.mark.parametrize("status", [
    # Privacy, rights and money keep their own rules: the clock never returns them.
    {**BLOCKED_AT, "blockers": ["team_scene_capture_withdrawn"]},
    {**BLOCKED_AT, "blockers": ["website_site_consent_missing"]},
    {**BLOCKED_AT, "blockers": ["dataset_license_unverified"]},
    {**BLOCKED_AT, "blockers": ["upstream_spend_limit_reached"]},
    {**BLOCKED_AT, "blockers": ["team_scene_bundle_invalid", "preparation_budget_exhausted"]},
    # The owner's confirmation returns an owner-contract refusal, never the clock.
    {**BLOCKED_AT, "blockers": [worker.OWNER_CONTRACT_BLOCKER_PREFIX + "proposal_only"]},
    # A held furniture measurement (caps, window, any cause) keeps its own once-per-release retry: a second
    # try could buy the same measurement again.
    {**BLOCKED_AT, "blockers": [worker.ROBOT_ACCESS_MEASUREMENT_HOLD + ":analysis_reservation_refused"]},
    {**BLOCKED_AT, "blockers": ["team_scene_derived_profile_conflict"],
     "robot_access_measurement": {"status": "held", "hold": worker.ROBOT_ACCESS_MEASUREMENT_HOLD + ":window",
                                  "cause": "refusal"}},
    {**BLOCKED_AT, "blockers": ["team_scene_derived_profile_conflict"], "robot_access_measurement": "garbled"},
    # A block from an attempt that reached the remote CPU lane: a paid compile ran or was given up on.
    {**BLOCKED_AT, "blockers": ["native_task_appearance_layer_transform_spurious"],
     "remote_compile": {"mode": "remote", "queue_row": "row.json"}},
    {**BLOCKED_AT, "blockers": ["team_scene_derived_profile_conflict"],
     "remote_compile": {"mode": "fallback_host", "reason": "remote_cpu_attempts_exhausted"}},
    # A transient block keeps its own three backoffs, spent ones included.
    {**BLOCKED_AT, "blockers": ["team_scene_packet_interrupted"], "automatic_retry": {"attempts": 3}},
    {**BLOCKED_AT, "status": "skipped", "blockers": []},
    {**BLOCKED_AT, "status": "completed", "blockers": []},
])
def test_the_clock_never_returns_a_block_another_rule_owns(tmp_path, status):
    assert worker.refusal_retry_cause(status) is None
    queue = worker._queue(tmp_path / "queue")
    (queue / "processed" / "k.json").write_text("{}")
    (queue / "status" / "k.json").write_text(json.dumps(status))
    late = T0 + timedelta(days=30)
    assert [row for row in worker._requeue_due(queue, late, RELEASE_1) if row["reason"] == "refusal_retry"] == []
    assert json.loads((queue / "status" / "k.json").read_text()) == status
    # Nor does any run's end put such a block on the schedule.
    assert worker.REFUSAL_RETRY_FIELD not in worker._with_refusal_retry(
        {key: value for key, value in status.items() if key != worker.REFUSAL_RETRY_FIELD}, now=T0)


@pytest.mark.parametrize("record", [
    "garbled", ["attempts", 0], {}, {"attempts": "1"}, {"attempts": -1}, {"attempts": True}, {"attempts": 1.0},
    {"attempts": None}, {"attempts": 6}, {"attempts": 7},
])
def test_a_spent_or_unreadable_timed_retry_count_never_retries(tmp_path, record):
    status = {"status": "blocked", "blockers": ["team_scene_bundle_invalid"], "release_commit": RELEASE_1,
              worker.REFUSAL_RETRY_FIELD: record}
    queue = worker._queue(tmp_path / "queue")
    (queue / "processed" / "k.json").write_text("{}")
    (queue / "status" / "k.json").write_text(json.dumps(status))
    assert worker._requeue_due(queue, T0 + timedelta(days=365), RELEASE_1) == []
    assert (queue / "processed" / "k.json").is_file()
    assert worker.final_block(status, release=RELEASE_1)
    # A run's end reads it as spent, and says so; it never starts the count again.
    assert worker._with_refusal_retry(status, now=T0)[worker.REFUSAL_RETRY_FIELD] == {
        **(record if isinstance(record, dict) else {}), "attempts": 7 if record == {"attempts": 7} else 6, "limit": 6,
        "reason": "team_scene_bundle_invalid", "exhausted": True}


def test_the_timed_count_survives_an_interrupted_run_and_a_remote_deferral(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    queue, key = world["queue"], world["key"]
    _release(monkeypatch, RELEASE_1)
    record = {"attempts": 4, "limit": 6, "reason": "team_scene_bundle_invalid",
              "requeued_at": "2026-10-31T11:00:00+00:00"}
    os.replace(queue / "pending" / f"{key}.json", queue / "processing" / f"{key}.json")
    (queue / "status" / f"{key}.json").write_text(json.dumps(
        {"status": "processing", "capture_root": str(world["capture"]), worker.REFUSAL_RETRY_FIELD: record}))
    _at(monkeypatch, 0)
    swept = worker.requeue_transient_blocked(queue_root=queue)
    assert swept["recovered"] == [key]
    assert _status(world)[worker.REFUSAL_RETRY_FIELD] == record
    deferred = worker._deferred(queue, queue / "processed" / f"{key}.json", _status(world), {"mode": "fallback_host"})
    assert deferred[worker.REFUSAL_RETRY_FIELD] == record
