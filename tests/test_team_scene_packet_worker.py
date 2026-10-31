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
    _registry(world["operator"], packet_inputs=world["inputs"], defaults={"screening_candidates": {
        "team_long": {"robot_policy": {"manifest": records["long"]}},
        "team_long_again": {"robot_policy": {"manifest": records["long"]}},
        "team_rover": {"robot_policy": {"manifest": records["wheeled"]}},
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
