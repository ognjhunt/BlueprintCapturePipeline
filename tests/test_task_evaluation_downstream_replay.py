"""A website job's post-completion chain replays offline to its first typed stop (B-01 phase 1).

The pulled files are a real completed website launch: the rigid pick-and-place
fixture revision ``team_scene_packet`` compiles (tests/test_team_scene_packet.py),
published with its configured bundle, its sealed launch receipt, WebApp readback
and provider-zero closure, the scene intent and attempt that own it, the website
capture with its task context, the operator registry and the WebApp's answers.
The replay runs this tree's real join, readiness scorer, team compile, probe lane
and screening planner on scratch copies; nothing reaches a provider, a GPU, the
object store, the WebApp or the network, and the pulled files are only read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from blueprint_pipeline import task_evaluation_downstream_replay as downstream
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_probe_plan import physics_plan, probe_context
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.robot_free_readiness_retention import TRACE_FILENAME
from blueprint_pipeline.task_evaluation_launch_terminal_evidence import _scene_configuration_terminal_projection
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION_CONTRACT_PATH,
    NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
)
from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
from blueprint_pipeline.task_evaluation_scene_rigid_robot_free_readiness import rigid_task_for_revision
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_SCHEMA
from blueprint_pipeline.team_scene_packet_worker import CONFIGURED_BUNDLE_FILENAME
from tests.test_beta_policy_screening_job_ceiling import ceiling_summary
from tests.test_rigid_robot_free_probe import FlatSupport, binding
from tests.test_team_scene_packet import _scene
from tests.test_website_scene_preparation_scope import preparation_request

REQUEST_ID = "req-1"
SITE, CAPTURE = f"site-{REQUEST_ID}", f"walkthrough-{REQUEST_ID}"
LAUNCH, CONFIGURATION_RUN = "scene-website-qualifying", "scene-website-configuration"
COMMIT = "c" * 40
SOURCE_VIDEO = "sha256:" + "5" * 64


def _sealed(value: dict[str, Any], field: str) -> dict[str, Any]:
    value = {**value, field: ""}
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _save(path: Path, value: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, tuple[str, int, int]]:
    """Every pulled file's bytes, mode and modification time."""
    return {path.relative_to(root).as_posix(): (_sha(path), path.stat().st_mode, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _scene_for(task_id: str, root: Path) -> dict[str, Any]:
    """The fixture revision, its task named for the job and grasped from above (it clears the table)."""
    root.mkdir(parents=True)
    scene = _scene(root, strategy="pick_and_place")
    revision, documents = scene["revision"], scene["documents"]
    identity = {"id": task_id, "version": "v1"}
    definition = json.loads(Path(documents[DEFINITION_CONTRACT_PATH]).read_text())
    definition["task_identity"] = identity
    definition["interaction_affordance"].update(contact_point_scoring_frame_m=[0.0, 0.0, 0.06],
                                                approach_unit_scoring_frame=[0.0, 0.0, 1.0])
    path = root / "job-definition.json"
    path.write_text(json.dumps(definition) + "\n")
    documents[DEFINITION_CONTRACT_PATH] = path
    revision["task_template"]["identity"] = identity
    revision["task_template"]["definition"].update(digest=_sha(path), size_bytes=path.stat().st_size)
    scene["store"][revision["task_template"]["definition"]["uri"]] = path.read_bytes()
    revision["revision_digest"] = canonical_digest(revision, digest_field="revision_digest")
    revision_bytes = json.dumps(revision, sort_keys=True).encode()
    scene["store"][scene["revision_reference"]["uri"]] = revision_bytes
    scene["revision_reference"].update(digest="sha256:" + hashlib.sha256(revision_bytes).hexdigest(),
                                       size_bytes=len(revision_bytes))
    scene["revision_bytes"] = revision_bytes
    return scene


def _recorded_trace(scene: dict[str, Any]) -> dict[str, Any]:
    """What the construction probe recorder returns for this revision: the object rests at its goal."""
    revision, documents = scene["revision"], scene["documents"]
    spec = rigid_task_for_revision(revision=revision, documents=documents)["task_spec"]
    asset = json.loads(Path(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH]).read_text()).get("asset_digest")
    return record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]),
                                         plan=physics_plan(probe_context(spec)),
                                         binding=binding(spec, revision["revision_digest"], asset))


def website_job(tmp_path: Path, *, trace: bool = True, runtime_source: Path | None = None) -> dict[str, Any]:
    """The files an operator pulls for a completed website scene job, at their own paths under ``host``."""
    host = tmp_path / "host"
    context = _sealed({"schema_version": "website_site_task_context.v1", "request_id": REQUEST_ID,
                       "scene_id": SITE, "capture_id": CAPTURE, "purpose": "scene_preparation", "confirmed": False,
                       "confirmed_at": None, "description": "Move the box along the table",
                       "capture_rights": {"consent_status": "granted", "consent_revoked": False,
                                          "derived_scene_generation_allowed": True,
                                          "consent_scope": ["robot_evaluation"]}}, "context_digest")
    task_id = "website-" + context["context_digest"][7:27]
    scene = _scene_for(task_id, tmp_path / "fixture")
    revision = scene["revision"]

    # The scene intent the website submitted, and the attempt that launched its preparation.
    request = preparation_request()
    request["submission_id"], request["task"]["task_id"] = CAPTURE, task_id
    intents = host / "intents"
    staged = intake.stage_scene_intent(value=request, queue_root=intents, authenticated_client="webapp",
                                       trusted_clients={"webapp"}, now=100)
    intent = intake._read(intents / staged["intent_id"] / "intent.json", "intent_digest")
    attempt = intake.reserve_scene_attempt(
        queue_root=intents, intent_id=intent["intent_id"], attempt_id="prepare", source_commit=COMMIT,
        runtime_digest="sha256:" + "a" * 64, input_digest="sha256:" + "b" * 64, provider="vast",
        maximum_spend_usd=2, now=101)

    # The completed launch: its published revision and configured bundle, terminal result and closure.
    run = host / "launch-runs" / LAUNCH
    profile = _save(run / "launch_profile.json", _sealed({
        **bind_scene_attempt(attempt), "source_commit": COMMIT,
        "task_evaluation_run": {"run_mode": "scene_configuration", "configuration_run_id": CONFIGURATION_RUN,
                                "task_id": task_id}}, "profile_digest"))
    publication = run / "configured_scene_publication"
    publication.mkdir(parents=True)
    (publication / "configured_scene_revision.v1.json").write_bytes(scene["revision_bytes"])
    shutil.copyfile(scene["configured_scene_bundle_path"], publication / CONFIGURED_BUNDLE_FILENAME)
    if trace:
        _save(run / TRACE_FILENAME, _recorded_trace(scene))
    reference, bundle = scene["revision_reference"], revision["configured_scene_bundle"]
    thumbnail = {"uri": "s3://scenes/thumbnail.png", "digest": "sha256:" + "d" * 64, "size_bytes": 100}
    offering = _sealed({"schema_version": "task_evaluation_configured_scene_offering.v1",
                        "status": "configured_controls_pending", "configuration_run_id": CONFIGURATION_RUN,
                        "catalog_visibility": "team_only",
                        "presentation": {"task_thumbnail": thumbnail, "selection_receipt": thumbnail},
                        "evaluation_preparation_binding": {
                            "configured_scene_revision": reference, "configured_scene_bundle": bundle,
                            "configured_scene_revision_digest": revision["revision_digest"]}}, "offering_digest")
    finalization = _sealed({"schema_version": "task_evaluation_scene_construction_finalization.v1",
                            "status": "completed", "queue_state": "completed", "finalization_performed": True,
                            "run_id": CONFIGURATION_RUN, "source_commit": COMMIT}, "result_digest")
    result_path = run / "allocator" / "result.json"
    result = _save(result_path, _sealed({
        "schema_version": "task_evaluation_scene_configuration_vast_result.v1", "run_id": CONFIGURATION_RUN,
        "status": "completed", "source_commit": COMMIT, "configuration_completed": True,
        "configured_scene_published": True, "full_byte_service_account_readback_passed": True,
        "configured_scene_revision_reference": reference, "configured_scene_bundle_reference": bundle,
        "configured_scene_revision_digest": revision["revision_digest"],
        "publication_result_digest": reference["digest"], "task_thumbnail_reference": thumbnail,
        "task_thumbnail_selection_receipt_reference": thumbnail, "configured_scene_offering": offering,
        "scene_construction_queue_finalization": finalization,
        "configured_scene_revision_path": str(publication / "configured_scene_revision.v1.json")},
        "result_digest"))
    receipt = _save(run / "launch_receipt.json", _sealed({
        "schema_version": "task_evaluation_launch_receipt.v1", "status": "completed", "launch_id": LAUNCH,
        "source_commit": COMMIT, "run_id": "scene-website", "request_digest": "sha256:" + "1" * 64,
        "launch_profile_digest": profile["profile_digest"],
        "terminal_evidence": {"status": "passed",
                              "result": {"path": str(result_path), "exists": True, "digest": _sha(result_path)},
                              "scene_configuration": _scene_configuration_terminal_projection(result)[0]}},
        "receipt_digest"))
    bound = {key: receipt[key] for key in ("launch_id", "run_id", "request_digest", "receipt_digest")}
    offered = {"configured_scene_offering_digest": offering["offering_digest"],
               "configured_scene_offering_status": offering["status"]}
    _save(run / "webapp_sync_succeeded.json", _sealed({
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1", "status": "succeeded", **bound,
        "attempt_number": 1, "attempted_at": "2026-11-01T20:00:00+00:00", "provider_mutation_performed": False,
        **offered, "response": {"schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                                "status": "completed", "already_exists": False, **bound, **offered}},
        "sync_result_digest"))
    _save(run / "post_teardown_provider_zero_receipt.json", _sealed({
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1", "status": "provider_zero_confirmed",
        **bound, "launch_profile_digest": profile["profile_digest"], "provider_zero_verified": True,
        "continuing_spend_from_this_run": False, "allocator_invoked": False, "provider_mutation_performed": False,
        "automatic_retry_performed": False, "blockers": []}, "provider_zero_receipt_digest"))

    # The website capture in the executor's partition, with its task context and retained Atlas configuration.
    capture = host / "partition" / "scenes" / SITE / "captures" / CAPTURE
    _save(capture / "capture_descriptor.json", {
        "scene_id": SITE, "capture_id": CAPTURE, "site_submission_id": REQUEST_ID,
        "metadata": {"capture_entry_source": "browser_self_capture", "site_submission_id": REQUEST_ID}})
    _save(capture / "pipeline" / "website_task_context.json", context)
    _save(capture / "pipeline" / "atlas" / "configuration.json", {
        "schema_version": "website_atlas_configuration.v1", "source_video_digest": SOURCE_VIDEO,
        "task_context_digest": context["context_digest"], "dispatch_authorized": False})
    # The grant as the WebApp's non-creating read returns it; the replay answers that read from this copy.
    from tests.test_rigid_robot_free_probe_lane import website_sponsorship

    _save(capture / "pipeline" / "website_scene_sponsorship.json", website_sponsorship(context))

    # The operator registry: who funds, the rate, and the fixed compile inputs; never the scene.
    operator = host / "operator"
    operator.mkdir(parents=True)
    droid = operator / "droid_profile.json"
    shutil.copyfile(scene["droid_profile_path"], droid)
    runtime = runtime_source or operator / "runtime_source.json"
    if runtime_source is None:
        shutil.copyfile(scene["runtime_source_receipt_path"], runtime)
    salt = operator / "scored-matrix-salt.key"
    salt.write_text("k" * 48)
    salt.chmod(0o600)
    registry = operator / "team-evaluation-registry.json"
    _save(registry, _sealed({"schema_version": REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
        "root": str(operator / "derived"), "scored_matrix_salt_key_path": str(salt),
        "defaults": {"allowed_founder_approvers": ["founder@blueprint"], "max_hourly_rate_usd": 0.5,
                     "open_loop_horizon": 8, "screening_candidate_cap_usd": 3},
        "packet_inputs": {"droid_profile_path": str(droid), "droid_profile_digest": scene["droid_profile_digest"],
                          "runtime_source_receipt_path": str(runtime),
                          "output_root": str(host / "team-scene-halves")}}}, "registry_digest"))
    registry.chmod(0o644)

    # The published documents in a local mirror of the object store; the revision itself is answered by
    # the byte-identical copy the launch retained beside its configured bundle.
    mirror = host / "mirror"
    for uri, payload in scene["store"].items():
        if uri != reference["uri"]:
            parsed = urlsplit(uri)
            target = mirror / parsed.netloc / parsed.path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    # The WebApp's current answers for this capture.
    webapp = host / "webapp"
    _save(webapp / "task-context.json", context)
    _save(webapp / "atlas-configuration.json", {
        "schema_version": "website_atlas_configuration.v1", "dispatch_authorized": False,
        "grant_creation_allowed": False, "task_context_digest": context["context_digest"],
        "source_video_digest": SOURCE_VIDEO, "job_ceiling": ceiling_summary(40)})
    return {"host": host, "run": run, "intent_dir": intents / intent["intent_id"], "capture": capture,
            "registry": registry, "mirror": mirror, "webapp": webapp, "scene": scene, "task_id": task_id,
            "result_path": result_path}


def _replay(job: dict[str, Any], tmp_path: Path, until: str = "screening_plan", **changes: Any) -> dict[str, Any]:
    inputs = {"launch_run": job["run"], "intent_dir": job["intent_dir"], "capture_dir": job["capture"],
              "registry": job["registry"], "mirror": job["mirror"], "webapp_snapshots": job["webapp"], **changes}
    return downstream.replay(**inputs, until=until, replay_root=tmp_path / "replays")


def test_a_completed_website_scene_replays_to_its_screening_plan_and_names_it(tmp_path):
    job = website_job(tmp_path)
    before = _tree(job["host"])
    report = _replay(job, tmp_path)

    stop = report["first_stop"]
    assert (stop["kind"], stop["stage"]) == ("completed", "screening_plan"), stop
    assert [(row["stage"], row["status"]) for row in report["stages"]] == [
        (stage, "passed") for stage in downstream.STAGES]
    stages = {row["stage"]: row for row in report["stages"]}
    # The join closed the launch and queued its team half; readiness passed on the retained trace.
    assert stages["completion"]["outcome"]["team_scene_request"]["status"] == "enqueued"
    assert stages["readiness"]["outcome"]["status"] == "passed"
    # The team half was compiled in the replay root, and the offer was recorded, never sent.
    assert stages["team_packet"]["outcome"]["outcome"] == "compiled"
    assert Path(stages["team_packet"]["outcome"]["team_scene_inputs_path"]).is_relative_to(Path(report["run_root"]))
    [offer] = report["recorded_effects"]
    assert (offer["boundary"], offer["operation"], offer["capture_id"]) == (
        "webapp", "agent-execution-offer", CAPTURE)
    # A decided receipt wants no probe run, so nothing was queued for the paid lane.
    assert stages["probe_enqueue"]["outcome"]["probe_state"] == "receipt_decided"
    # The plan the screening worker froze, within the job's dollar ceiling.
    plan = stop["plan"]
    assert plan["budget_usd"] == "15.00" and plan["budget_basis"] == "website_job_ceiling"
    assert {row["candidate_id"] for row in plan["candidates"] if row["status"] == "selected"}
    assert Path(plan["path"]).is_file() and Path(plan["path"]).is_relative_to(Path(report["run_root"]))
    # The WebApp reads were answered from the pulled snapshots; the documents from the mirror, and the
    # revision from the launch's own byte-identical copy.
    answered = {(row["operation"], Path(row["answered_from"]).name) for row in report["webapp_answers"]}
    assert answered == {("task-context", "task-context.json"),
                        ("atlas-configuration", "atlas-configuration.json")}
    sources = {row["uri"]: row["answered_from"] for row in report["object_reads"]}
    assert sources[job["scene"]["revision_reference"]["uri"]].endswith("configured_scene_revision.v1.json")
    assert all(row["answered_from"] for row in report["object_reads"])
    # Nothing was bought, sent or written outside the replay; the pulled files are byte-identical.
    assert report["would_buy"] == [] and report["network_attempts"] == [] and report["new_paid_calls"] == 0
    assert report["evidence"]["unchanged"] is True
    assert all(row["unchanged"] for row in report["inputs"].values() if row)
    assert _tree(job["host"]) == before
    # The registry's write roots point at scratch, and the salt key is a placeholder: no secret is copied.
    rewrites = {row["field"]: row for row in report["configuration_rewrites"]}
    assert set(rewrites) == {"derived_profiles.root", "derived_profiles.packet_inputs.output_root",
                             "derived_profiles.scored_matrix_salt_key_path"}
    assert all(Path(row["replay"]).is_relative_to(Path(report["run_root"])) for row in rewrites.values())
    assert b"k" * 48 not in Path(rewrites["derived_profiles.scored_matrix_salt_key_path"]["replay"]).read_bytes()
    path = Path(report["report_path"])
    assert path.name == downstream.REPORT_FILENAME and json.loads(path.read_text()) == report


@pytest.mark.parametrize("until", ["completion", "readiness", "team_packet", "probe_enqueue"])
def test_a_replay_stops_once_it_reaches_until(tmp_path, until):
    job = website_job(tmp_path)
    # A mirror or snapshot directory that was never made is only empty: completion needs neither.
    absent = {"mirror": tmp_path / "no-mirror", "webapp_snapshots": tmp_path / "no-snapshots"}
    report = _replay(job, tmp_path, until=until, **(absent if until == "completion" else {}))
    assert (report["first_stop"]["kind"], report["first_stop"]["stage"]) == ("completed", until)
    reached = downstream.STAGES.index(until)
    assert [row["status"] for row in report["stages"]] == (
        ["passed"] * (reached + 1) + ["not_reached"] * (len(downstream.STAGES) - reached - 1))
    if until == "completion":
        # Stopped before readiness retention ran: nothing was fetched or scored.
        assert report["object_reads"] == []


def test_the_cli_writes_the_report_and_exits_zero_on_a_completed_replay(tmp_path, capsys):
    job = website_job(tmp_path)
    code = downstream.main(["--launch-run", str(job["run"]), "--intent-dir", str(job["intent_dir"]),
                            "--capture-dir", str(job["capture"]), "--registry", str(job["registry"]),
                            "--mirror", str(job["mirror"]), "--webapp-snapshots", str(job["webapp"]),
                            "--until", "readiness", "--replay-root", str(tmp_path / "replays")])
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and printed["first_stop"]["kind"] == "completed"
    assert json.loads(Path(printed["report_path"]).read_text())["first_stop"] == printed["first_stop"]


def test_a_probe_the_scene_still_needs_stops_where_the_lane_would_buy_its_run(tmp_path):
    from tests.test_native_task_arena_bundle import _runtime_source_packet

    job = website_job(tmp_path, trace=False, runtime_source=_runtime_source_packet(tmp_path / "runtime"))
    before = _tree(job["host"])
    report = _replay(job, tmp_path)
    stop = report["first_stop"]
    assert (stop["kind"], stop["stage"]) == ("would_buy", "probe_enqueue"), stop
    # The canonical allocator invocation, named: the capped GPU probe, never started.
    assert stop["code"] == "paid_resource_allocator:gpu-canary:rigid-robot-free-probe"
    assert stop["max_spend_usd"] == "1.0" and "--execute" in stop["argv"]
    [bought] = report["would_buy"]
    assert bought["boundary"] == "rigid_robot_free_probe_runner.run_allocator"
    stages = {row["stage"]: row for row in report["stages"]}
    assert stages["readiness"]["outcome"]["status"] == "probe_unavailable"
    assert stages["probe_enqueue"]["outcome"]["enqueue"][0]["status"] == "queued"
    assert stages["screening_plan"]["status"] == "not_reached"
    # The host-only launch gates are named as not evaluated rather than passed.
    assert {"driver_ceiling_blockers", "launch_environment_blockers"} <= set(report["host_only_gates_not_evaluated"])
    assert report["evidence"]["unchanged"] is True and _tree(job["host"]) == before


@pytest.mark.parametrize("case", ["launch_closure", "published_document", "result_moved"])
def test_a_missing_input_is_named_with_the_path_to_pull(tmp_path, case):
    job = website_job(tmp_path)
    if case == "launch_closure":
        missing = job["run"] / "post_teardown_provider_zero_receipt.json"
        missing.unlink()
    elif case == "published_document":
        uri = job["scene"]["revision"]["task_template"]["definition"]["uri"]
        parsed = urlsplit(uri)
        (job["mirror"] / parsed.netloc / parsed.path.lstrip("/")).unlink()
    else:
        # The receipt names the result by its absolute host path; a pull elsewhere leaves it unresolved.
        moved = tmp_path / "elsewhere"
        shutil.move(str(job["result_path"].parent), str(moved))
    report = _replay(job, tmp_path)
    stop = report["first_stop"]
    assert stop["kind"] == "missing_input", stop
    if case == "launch_closure":
        assert stop["stage"] == "completion"
        assert stop["code"] == str(missing)
        assert stop["pull"] == "launch-run:post_teardown_provider_zero_receipt.json"
        assert stop["underlying_refusal"] == "configured_controls_worker_post_teardown_provider_zero_missing"
    elif case == "published_document":
        assert stop["stage"] == "readiness"
        assert stop["code"] == uri
        assert stop["mirror_path"] == str(job["mirror"] / parsed.netloc / parsed.path.lstrip("/"))
    else:
        assert stop["stage"] == "completion"
        assert stop["code"] == str(job["result_path"])
        assert stop["underlying_refusal"] == "configured_controls_worker_terminal_artifact_invalid"


def test_a_typed_refusal_names_the_code_and_the_predicates_that_fired(tmp_path):
    job = website_job(tmp_path)
    profile = json.loads((job["run"] / "launch_profile.json").read_text())
    profile["task_evaluation_run"]["configuration_run_id"] = "another-configuration"
    _save(job["run"] / "launch_profile.json", _sealed(profile, "profile_digest"))
    report = _replay(job, tmp_path)
    stop = report["first_stop"]
    assert (stop["kind"], stop["stage"], stop["code"]) == (
        "refusal", "completion", "scene_intake_preparation_completion_publication_mismatch")
    assert "result['run_id'] == scope['configuration_run_id']" in stop["fired_predicates"]


def _harness(tmp_path: Path) -> downstream._Harness:
    run = tmp_path / "run"
    run.mkdir()
    return downstream._Harness(run=run.resolve())


def _webapp_post() -> Any:
    from blueprint_pipeline.website_task_context import website_webapp_request
    return website_webapp_request(capture_id=CAPTURE, operation="prepared-scene", payload={"request": {}})


@pytest.mark.parametrize("attempt, kind, code", [
    (lambda tmp: socket.create_connection(("api.vast.ai", 443), timeout=1), "external_effect", "network:api.vast.ai"),
    (lambda tmp: urllib.request.urlopen("https://s3.us-west-004.backblazeb2.com/bucket/key", timeout=1),
     "external_effect", "network:s3.us-west-004.backblazeb2.com"),
    (lambda tmp: _webapp_post(), "external_effect", "webapp:prepared-scene"),
    (lambda tmp: subprocess.run([sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", "gpu-canary",
                                 "--execute"], check=False), "would_buy", "paid_resource_allocator:gpu-canary"),
    (lambda tmp: subprocess.run(["vastai", "show", "instances"], check=False), "external_effect", "subprocess:vastai"),
    (lambda tmp: (tmp / "outside.json").write_text("{}"), "external_effect", "write_outside_replay_root"),
])
def test_every_network_allocator_or_webapp_post_attempt_is_a_named_stop(tmp_path, attempt, kind, code):
    lifted = (socket.create_connection, socket.getaddrinfo, subprocess.Popen.__init__, os.open, open)
    harness = _harness(tmp_path)
    with harness, pytest.raises(downstream.ReplayStop) as raised:
        attempt(tmp_path)
    report = raised.value.report()
    assert report["kind"] == kind and report["code"].startswith(code), report
    assert not (tmp_path / "outside.json").exists()
    # The boundaries are lifted once the replay ends.
    assert (socket.create_connection, socket.getaddrinfo, subprocess.Popen.__init__, os.open, open) == lifted


def test_the_registry_copy_points_its_write_roots_and_this_jobs_rows_at_the_replay(tmp_path):
    capture = tmp_path / "pulled" / "scenes" / SITE / "captures" / CAPTURE
    _save(capture / "capture_descriptor.json", {"scene_id": SITE, "capture_id": CAPTURE})
    launch = tmp_path / "pulled" / "launch-runs" / LAUNCH
    other = {"capture_root": "/var/lib/blueprint/agent-runs/scenes/site-x/captures/walkthrough-x",
             "robot_free_probe_root": "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/x"}
    mine = {"capture_root": f"/var/lib/blueprint/agent-runs/scenes/{SITE}/captures/{CAPTURE}",
            "robot_free_probe_root": f"/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs/{LAUNCH}"}
    source = tmp_path / "team-evaluation-registry.json"
    _save(source, _sealed({"schema_version": REGISTRY_SCHEMA, "profiles": [dict(other), dict(mine)],
                           "derived_profiles": {"root": "/var/lib/blueprint/team-registry", "defaults": {}}},
                          "registry_digest"))
    run = tmp_path / "run"
    (run / "inputs" / "registry").mkdir(parents=True)
    target = run / "inputs" / "registry" / source.name
    copies = {"capture_copy": run / "inputs" / "partition" / "scenes" / SITE / "captures" / CAPTURE,
              "launch_copy": run / "inputs" / "launch-runs" / LAUNCH}
    rewrites = downstream._rewrite_registry(source, target, run=run, capture=capture, launch=launch, **copies)
    value = json.loads(target.read_text())
    # Re-sealed, so the registry reader admits the copy; other captures' rows are untouched.
    assert value["registry_digest"] == canonical_digest(value, digest_field="registry_digest")
    assert value["profiles"] == [other, {"capture_root": str(copies["capture_copy"]),
                                         "robot_free_probe_root": str(copies["launch_copy"])}]
    assert [(row["field"], row["host"]) for row in rewrites] == [
        ("derived_profiles.root", "/var/lib/blueprint/team-registry"),
        ("profiles[1].capture_root", mine["capture_root"]),
        ("profiles[1].robot_free_probe_root", mine["robot_free_probe_root"])]
    # A registry whose seal is already broken stays broken: the replay refuses it as production does.
    broken = {**json.loads(source.read_text()), "registry_digest": "sha256:" + "0" * 64}
    _save(source, broken)
    downstream._rewrite_registry(source, target, run=run, capture=capture, launch=launch, **copies)
    assert json.loads(target.read_text())["registry_digest"] == "sha256:" + "0" * 64
