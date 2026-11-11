"""The evaluation service's door operations: test policies on a job, restore a cold job, register a served policy.

Covers (for impacted-test selection):
  deploy/operator-door/door-test-policy-on-job.sh
  deploy/operator-door/door-restore-launch-run.sh
  deploy/operator-door/door-register-session-served-policy.sh
  deploy/operator-door/operator_door/session_served_registration.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door import session_served_registration as served  # noqa: E402
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.requests import RequestRefused, enqueue, required_scope, validate_request, validate_request_id  # noqa: E402
from operator_door.spool_runner import process_spool  # noqa: E402
from tests.test_operator_door_runner import FakeRunner  # noqa: E402

DOOR = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
ENTRY = {"schema_version": "blueprint.session_served_policy.v1", "server": "openpi_droid_contract_server.v1",
         "native_candidate_id": "pi05_droid", "display_name": "pi0.5 DROID (standard connection)"}
JOB_TEST = {"kind": "test-policy-on-job", "site_id": "site-job1", "candidate_ids": ["pi05_droid"],
            "funding": "standing_test", "authorization_reference": "owner approval fixture"}


def registry(**screening):
    value = {"schema_version": served.REGISTRY_SCHEMA, "profiles": [],
             "derived_profiles": {"defaults": {"screening_candidates": dict(screening)}}}
    value["registry_digest"] = served.digest(value, "registry_digest")
    return value


def registration(action="add", candidate_id="pi05_droid_served", entry=ENTRY, *, before=None):
    return served.seal_request({"schema_version": served.REQUEST_SCHEMA, "action": action,
                                "candidate_id": candidate_id, "entry": entry if action == "add" else None,
                                "expected_registry_digest": (before or registry())["registry_digest"],
                                "authorization_reference": "owner approval fixture"})


# Requests: shape, scope and ids.

def test_each_operation_is_an_operate_request_with_a_fixed_shape():
    for kind in ("test-policy-on-job", "restore-launch-run", "register-session-served-policy"):
        assert required_scope(kind) == "operate"
        assert validate_request_id(f"20261111T190000Z-{kind}-0123abcd")
    assert validate_request(JOB_TEST) == JOB_TEST
    assert validate_request({"kind": "restore-launch-run", "site_id": "site-job1"}) == {
        "kind": "restore-launch-run", "site_id": "site-job1"}
    body = {"kind": "register-session-served-policy", "request": registration()}
    assert validate_request(body) == body


@pytest.mark.parametrize("change", [
    {"site_id": "../etc"}, {"candidate_ids": []}, {"candidate_ids": ["a"] * 2}, {"candidate_ids": ["../x"]},
    {"candidate_ids": [f"p{n}" for n in range(9)]}, {"funding": "owner_card"}, {"authorization_reference": ""},
    {"extra": 1}])
def test_a_malformed_job_test_is_refused_before_the_spool(change):
    with pytest.raises(RequestRefused):
        validate_request({**JOB_TEST, **change})


@pytest.mark.parametrize("request_body", [
    registration(candidate_id="pi05_droid"),  # a packaged id is never shadowed
    registration(entry={**ENTRY, "native_candidate_id": "groot_n17_droid"}),
    registration(entry={**ENTRY, "port": 8000}),
    {**registration(), "authorization_reference": "changed after sealing"},
    "x", [1], 5])
def test_a_malformed_registration_is_refused_before_the_spool(request_body):
    with pytest.raises(RequestRefused):
        validate_request({"kind": "register-session-served-policy", "request": request_body})


# The root runner: one bounded unit per request, running the installed script.

@pytest.fixture()
def config(tmp_path, monkeypatch):
    for state in ("pending", "processing", "completed", "results"):
        (tmp_path / "requests" / state).mkdir(parents=True)
    value = DoorConfig(state_root=str(tmp_path), install_root="/opt/blueprint/operator-door")
    monkeypatch.setattr(FakeRunner, "guard_root", Path(value.spool_root) / "holds")
    return value


@pytest.mark.parametrize(("body", "script", "env"), [
    (JOB_TEST, "door-test-policy-on-job.sh", "DOOR_JOB_TEST_REQUEST"),
    ({"kind": "restore-launch-run", "site_id": "site-job1"}, "door-restore-launch-run.sh", "DOOR_SITE_ID=site-job1"),
    ({"kind": "register-session-served-policy", "request": registration()}, "door-register-session-served-policy.sh",
     "DOOR_SESSION_SERVED_REQUEST"),
])
def test_the_runner_launches_the_installed_script(config, body, script, env):
    request_id = enqueue(config, validate_request(body), requested_by="cloud")
    runner = FakeRunner()
    process_spool(config, runner=runner)
    [call] = [row for row in runner.calls if row[0] == "systemd-run"]
    assert call[-1] == f"/opt/blueprint/operator-door/{script}"
    assert any(arg.startswith("--setenv=" + env) for arg in call)
    assert "--property=ProtectSystem=strict" in call
    assert "site-job1" not in call[1]  # caller text never names a unit
    result = json.loads((Path(config.spool_root) / "results" / f"{request_id}.json").read_text())
    assert result["status"] == "launched"
    assert (DOOR / script).is_file()


# The scripts, run for real against a stub interpreter.

def _script_env(tmp_path, stub_output, rc=0):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "setpriv").write_text('#!/bin/bash\nwhile [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n')
    venv = tmp_path / "venv-python"
    venv.write_text(f"#!/bin/bash\necho \"$@\" >> {tmp_path}/calls.log\n"
                    f"for a in \"$@\"; do if [ \"$prev\" = --result-out ]; then echo '{stub_output}' > \"$a\"; fi;"
                    f" prev=\"$a\"; done\necho '{stub_output}'\nexit {rc}\n")
    for path in (stubs / "setpriv", venv):
        path.chmod(0o755)
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "release").mkdir()
    return {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "DOOR_RESULTS_DIR": str(results),
            "DOOR_VENV_PYTHON": str(venv), "DOOR_CONTROL_PLANE_REPO": str(tmp_path / "release"),
            "DOOR_SERVICE_USER": "blueprint"}


def _run(script, env, request_id):
    done = subprocess.run(["/bin/bash", str(DOOR / script)], env={**env, "DOOR_REQUEST_ID": request_id},
                          capture_output=True, text=True, timeout=60, check=False)
    outcome_path = Path(env["DOOR_RESULTS_DIR"]) / f"{request_id}.outcome.json"
    return done.returncode, json.loads(outcome_path.read_text()) if outcome_path.exists() else {}


@pytest.mark.parametrize(("stub", "rc", "status", "code"), [
    ({"door_outcome": "enqueued", "status": "enqueued", "code": None}, 0, "enqueued", None),
    ({"door_outcome": "refused", "status": "refused", "code": "standing_test_monthly_total_unset"}, 3, "refused",
     "standing_test_monthly_total_unset"),
    ({}, 1, "failed", "job_test_failed")])
def test_the_job_test_script_reports_enqueued_refused_or_failed(tmp_path, stub, rc, status, code):
    env = _script_env(tmp_path, json.dumps(stub), rc)
    request_id = "20261111T190000Z-test-policy-on-job-0123abcd"
    returned, outcome = _run("door-test-policy-on-job.sh", {**env, "DOOR_JOB_TEST_REQUEST": json.dumps(JOB_TEST)},
                             request_id)
    assert outcome["status"] == status and outcome["code"] == code and (returned == 0) == (status == "enqueued")
    calls = (tmp_path / "calls.log").read_text()
    assert "-m blueprint_pipeline.beta_policy_screening_on_job" in calls and f"--door-request-id {request_id}" in calls


@pytest.mark.parametrize(("stub", "rc", "status"), [
    ({"status": "restored"}, 0, "restored"), ({"status": "already_on_disk"}, 0, "already_on_disk"),
    ({"status": "refused", "reason": "launch_run_restore_disk_headroom_insufficient"}, 3, "refused")])
def test_the_restore_script_reports_the_module_s_receipt(tmp_path, stub, rc, status):
    env = _script_env(tmp_path, json.dumps(stub), rc)
    request_id = "20261111T190000Z-restore-launch-run-0123abcd"
    returned, outcome = _run("door-restore-launch-run.sh", {**env, "DOOR_SITE_ID": "site-job1"}, request_id)
    assert outcome["status"] == status and (returned == 0) == (status != "refused")
    if status == "refused":
        assert outcome["code"] == "launch_run_restore_disk_headroom_insufficient"
    assert "--site-id site-job1" in (tmp_path / "calls.log").read_text()


def test_the_restore_script_refuses_a_malformed_site_before_running_anything(tmp_path):
    env = _script_env(tmp_path, "{}")
    returned, outcome = _run("door-restore-launch-run.sh", {**env, "DOOR_SITE_ID": "../etc"},
                             "20261111T190000Z-restore-launch-run-0123abcd")
    assert returned != 0 and outcome["status"] == "refused" and not (tmp_path / "calls.log").exists()


def test_every_new_script_parses_and_is_installed():
    install = (DOOR / "install.sh").read_text()
    for script in ("door-test-policy-on-job.sh", "door-restore-launch-run.sh", "door-register-session-served-policy.sh"):
        assert subprocess.run(["/bin/bash", "-n", str(DOOR / script)], check=False).returncode == 0
        assert f'"$source_dir"/{script}' in install
    assert "session-served-registrations" in install


# The registration installer: a CAS on the registry with a kept receipt.

def _install(tmp_path, value, request, **kwargs):
    path = tmp_path / "etc" / "team-evaluation-registry.json"
    path.parent.mkdir(exist_ok=True, mode=0o755)
    if not path.exists():
        path.write_text(json.dumps(value))
        path.chmod(0o644)
    return served.apply(request, registry_path=path, audit_root=tmp_path / "audit", require_owner=False,
                        **kwargs), path


def test_adding_and_removing_an_entry_keeps_receipts_and_touches_nothing_else(tmp_path):
    before = registry(other_policy={"robot_policy": {"kept": True}})
    receipt, path = _install(tmp_path, before, registration(before=before))
    after = json.loads(path.read_text())
    assert receipt["status"] == "added" and receipt["after_registry_digest"] == after["registry_digest"]
    assert after["derived_profiles"]["defaults"]["screening_candidates"] == {
        "other_policy": {"robot_policy": {"kept": True}}, "pi05_droid_served": {"session_served_policy": ENTRY}}
    assert receipt["authorization_reference"] == "owner approval fixture"
    assert receipt["provider_allocations_performed"] == 0
    # A replay returns the kept receipt.
    assert _install(tmp_path, before, registration(before=before))[0] == receipt
    removed, _ = _install(tmp_path, before, registration("remove", before=after))
    assert removed["status"] == "removed"
    assert json.loads(path.read_text())["derived_profiles"]["defaults"]["screening_candidates"] == {
        "other_policy": {"robot_policy": {"kept": True}}}
    # The entry is read exactly as the pipeline validates it.
    from blueprint_pipeline.session_served_policy import validate_entry

    assert validate_entry(after["derived_profiles"]["defaults"]["screening_candidates"]["pi05_droid_served"]
                          ["session_served_policy"], candidate_id="pi05_droid_served")


@pytest.mark.parametrize(("setup", "code"), [
    ("stale_digest", "registry_compare_failed"),
    ("taken", "candidate_id_taken"),
    ("absent", "entry_absent"),
    ("packaged", "candidate_id_is_packaged"),
])
def test_registration_refusals_change_nothing(tmp_path, setup, code):
    before = registry(**({"pi05_droid_served": {"robot_policy": {}}} if setup == "taken" else {}))
    request = registration(before=before)
    kwargs = {}
    if setup == "stale_digest":
        request = registration(before=registry(other={"robot_policy": {}}))
    elif setup == "absent":
        request = registration("remove", before=before)
    elif setup == "packaged":
        catalog = tmp_path / "catalog.json"
        catalog.write_text(json.dumps({"candidates": [{"candidate_id": "pi05_droid_served"}]}))
        kwargs["catalog_path"] = catalog
    with pytest.raises(ValueError, match=code):
        _install(tmp_path, before, request, **kwargs)
    path = tmp_path / "etc" / "team-evaluation-registry.json"
    assert json.loads(path.read_text()) == before


def test_the_installer_schema_matches_the_pipeline_s():
    from blueprint_pipeline import session_served_policy as pipeline

    assert {key: tuple(value["native_candidates"]) for key, value in pipeline.SERVERS.items()} == served.SERVERS
    assert pipeline._ENTRY_KEYS == served.ENTRY_KEYS and pipeline.ENTRY_SCHEMA == served.ENTRY_SCHEMA


# The cold-job restore: disk first, the existing restore, a receipt.

@pytest.fixture()
def cold_job(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_on_job as on_job
    from blueprint_pipeline.control_plane_evidence_offload import POINTER_SUFFIX, POINTER_SCHEMA_VERSION
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    launch = tmp_path / "launch-runs" / "run-447"
    launch.parent.mkdir()
    member = b'{"receipt_digest": "sha256:aa"}'
    archive = tmp_path / "evidence.tar"
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "launch_receipt.json").write_bytes(member)
    with tarfile.open(archive, "w") as stream:
        stream.add(staged / "launch_receipt.json", arcname="launch_receipt.json")
    sha = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    pointer = {"schema_version": POINTER_SCHEMA_VERSION, "status": "offloaded", "directory": launch.name,
               "uri": "s3://bucket/evidence.tar", "digest": sha, "size_bytes": archive.stat().st_size,
               "members": [{"relative_path": "launch_receipt.json", "size_bytes": len(member),
                            "sha256": "sha256:" + hashlib.sha256(member).hexdigest()}], "pointer_digest": ""}
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    (launch.parent / f"{launch.name}{POINTER_SUFFIX}").write_text(json.dumps(pointer))
    monkeypatch.setattr(on_job, "job_source", lambda root, site: ({"source_launch_root": str(launch)}, tmp_path))
    consent = []  # The privacy read every restore makes first; this job's site has not withdrawn.
    monkeypatch.setattr(on_job, "current_consent", lambda site, request, capture: consent.append(site) or ({}, {}))

    def materialize(*, reference, destination, maximum_size_bytes):
        destination.write_bytes(archive.read_bytes())
        return {}
    return {"launch": launch, "materializer": materialize, "queue": tmp_path / "queue", "packets": tmp_path,
            "consent": consent, "archive": archive}


def test_a_cold_launch_run_is_restored_after_its_disk_peak_is_reserved(cold_job):
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    reserved = []

    class Reservation:
        def release(self):
            reserved.append("released")

    def reserve(role, *, target_root, expected_bytes, reservation_root):
        reserved.append((role, expected_bytes))
        return Reservation()
    receipt = restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=cold_job["packets"],
                                 queue_root=cold_job["queue"], materializer=cold_job["materializer"], reserve=reserve)
    assert receipt["status"] == "restored" and receipt["member_count"] == 1
    assert (cold_job["launch"] / "launch_receipt.json").is_file()
    assert reserved[0][0] == "evidence_offload" and reserved[0][1] > 0 and reserved[-1] == "released"
    kept = json.loads((cold_job["queue"] / "launch_run_restores" / "req-1.json").read_text())
    assert kept == receipt and kept["provider_allocations_performed"] == 0
    again = restore_launch_run(site_id="site-job1", request_id="req-2", packet_queue_root=cold_job["packets"],
                               materializer=cold_job["materializer"], reserve=reserve)
    assert again["status"] == "already_on_disk"


def test_no_disk_room_refuses_and_restores_nothing(cold_job):
    from blueprint_pipeline.control_plane_disk_ledger import ControlPlaneDiskBudgetError
    from blueprint_pipeline.screening_launch_run_restore import NO_ROOM, restore_launch_run

    def full(*args, **kwargs):
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_insufficient")
    receipt = restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=cold_job["packets"],
                                 materializer=cold_job["materializer"], reserve=full)
    assert receipt["status"] == "refused" and receipt["reason"] == NO_ROOM
    assert receipt["disk_refusal"] == "control_plane_disk_budget_insufficient"
    assert not cold_job["launch"].exists()


def _reserve_ok(reserved):
    class Reservation:
        def release(self):
            reserved.append("released")

    def reserve(role, **kwargs):
        reserved.append(role)
        return Reservation()
    return reserve


@pytest.mark.parametrize(("state", "reason"), [("withdrawn", "job_test_site_withdrawn"),
                                               ("unknown", "job_test_site_withdrawal_unknown")])
def test_a_withdrawn_site_s_launch_run_is_never_restored(cold_job, monkeypatch, state, reason):
    """Review finding 2 (privacy): refused as a job test is, before any byte is fetched, with its receipt."""
    from blueprint_pipeline import beta_policy_screening_on_job as on_job
    from blueprint_pipeline import screening_site_withdrawal
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    monkeypatch.undo()  # the real consent read, so the real withdrawal check decides
    launch = cold_job["launch"]
    monkeypatch.setattr(on_job, "job_source", lambda root, site: ({"source_launch_root": str(launch)}, launch.parent))
    monkeypatch.setattr(screening_site_withdrawal, "site_withdrawal_state", lambda site_id: state)
    fetched, reserved = [], []

    def materialize(**kwargs):
        fetched.append(kwargs)
    receipt = restore_launch_run(site_id="site-job1", request_id="req-w", packet_queue_root=cold_job["packets"],
                                 queue_root=cold_job["queue"], materializer=materialize,
                                 reserve=_reserve_ok(reserved))
    assert receipt["status"] == "refused" and receipt["reason"] == reason
    assert fetched == [] and reserved == [] and not launch.exists()
    assert json.loads((cold_job["queue"] / "launch_run_restores" / "req-w.json").read_text()) == receipt


def test_a_revoked_consent_refuses_the_restore(cold_job, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_on_job as on_job
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    def revoked(site, request, capture):
        raise ValueError("beta_screening_current_capture_consent_required")
    monkeypatch.setattr(on_job, "current_consent", revoked)
    receipt = restore_launch_run(site_id="site-job1", request_id="req-c", packet_queue_root=cold_job["packets"],
                                 queue_root=cold_job["queue"], materializer=cold_job["materializer"],
                                 reserve=_reserve_ok([]))
    assert receipt["status"] == "refused" and receipt["reason"] == "beta_screening_current_capture_consent_required"
    assert not cold_job["launch"].exists()


@pytest.mark.parametrize("fault", ["pointer_not_json", "pointer_members", "download", "digest", "unsealed"])
def test_every_restore_failure_has_a_typed_reason_and_a_receipt(cold_job, fault):
    """Review finding 4: a bad pointer, a download error or a digest mismatch each name their reason;
    the reservation is released and nothing is left half written."""
    from blueprint_pipeline.control_plane_evidence_offload import POINTER_SUFFIX
    from blueprint_pipeline.screening_launch_run_restore import (
        DOWNLOAD_FAILED, POINTER_INVALID, restore_launch_run,
    )

    launch = cold_job["launch"]
    pointer_path = launch.parent / f"{launch.name}{POINTER_SUFFIX}"
    materializer = cold_job["materializer"]
    if fault == "pointer_not_json":
        pointer_path.write_text("{not json")
    elif fault == "pointer_members":
        pointer = json.loads(pointer_path.read_text())
        pointer_path.write_text(json.dumps({**pointer, "members": [{"relative_path": "x", "size_bytes": "big"}]}))
    elif fault == "unsealed":
        pointer = json.loads(pointer_path.read_text())
        pointer_path.write_text(json.dumps({**pointer, "uri": "s3://bucket/other.tar"}))
    elif fault == "download":
        def materializer(**kwargs):
            raise ConnectionResetError("object store dropped the read")
    else:
        def materializer(*, reference, destination, maximum_size_bytes):
            destination.write_bytes(cold_job["archive"].read_bytes()[:-1] + b"\x01")
    reserved = []
    receipt = restore_launch_run(site_id="site-job1", request_id="req-f", packet_queue_root=cold_job["packets"],
                                 queue_root=cold_job["queue"], materializer=materializer,
                                 reserve=_reserve_ok(reserved))
    expected = {"pointer_not_json": POINTER_INVALID, "pointer_members": POINTER_INVALID,
                "download": DOWNLOAD_FAILED, "digest": "control_plane_evidence_restore_digest_mismatch",
                "unsealed": "control_plane_evidence_restore_pointer_invalid"}[fault]
    assert receipt["status"] == "refused" and receipt["reason"] == expected
    assert json.loads((cold_job["queue"] / "launch_run_restores" / "req-f.json").read_text()) == receipt
    assert not launch.exists() and not list(launch.parent.glob(".restore-*"))
    assert reserved == ([] if fault.startswith("pointer_") else ["evidence_offload", "released"])


def test_a_restored_run_is_a_live_tree_again_and_goes_cold_again_after_its_window(cold_job):
    """Review finding 3: a restored run is counted where every live run is and, once its hot window has
    passed since the restore, the offloader removes the local copy again without a second upload."""
    import time

    from blueprint_pipeline.control_plane_evidence_offload import (
        EXECUTE_ACK, POINTER_SUFFIX, apply_evidence_offload, build_evidence_offload_manifest,
    )
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    launch = cold_job["launch"]
    receipt = restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=cold_job["packets"],
                                 materializer=cold_job["materializer"], reserve=_reserve_ok([]))
    assert receipt["status"] == "restored"
    restored_at = time.time()
    window = 2 * 86400

    def manifest(now):
        return build_evidence_offload_manifest(evidence_roots=[launch.parent], hot_window_seconds=window,
                                               now=lambda: now, classifier=lambda *a, **k: None)
    # Inside the window from the restore: a hot live tree, its bytes counted with the others.
    hot = manifest(restored_at + 60)
    assert hot["candidate_count"] == 0 and hot["retained_by_reason"]["hot"]["count"] == 1
    assert "already_offloaded" not in hot["retained_by_reason"]
    # After it: a candidate again, removed with no upload, the pointer kept.
    cold = manifest(restored_at + window + 60)
    assert [row["name"] for row in cold["candidates"]] == [launch.name]

    def never(**kwargs):
        raise AssertionError("a restored run is never uploaded again")
    heads = []
    applied = apply_evidence_offload(cold, ack=EXECUTE_ACK, stream_publisher=never, publisher=never,
                                     now=lambda: restored_at + window + 60,
                                     remote_verifier=lambda reference: heads.append(reference) or {})
    assert len(heads) == 1 and heads[0]["digest"] == applied["offloaded"][0]["digest"]
    assert applied["offloaded_count"] == 1 and applied["offloaded"][0]["transfer_mode"] == "already_archived"
    assert not launch.exists() and (launch.parent / f"{launch.name}{POINTER_SUFFIX}").is_file()
    # And it can be restored again from the same pointer.
    again = restore_launch_run(site_id="site-job1", request_id="req-2", packet_queue_root=cold_job["packets"],
                               materializer=cold_job["materializer"], reserve=_reserve_ok([]))
    assert again["status"] == "restored"


def test_a_restored_run_that_changed_is_kept(cold_job):
    import time

    from blueprint_pipeline.control_plane_evidence_offload import (
        EXECUTE_ACK, apply_evidence_offload, build_evidence_offload_manifest,
    )
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    launch = cold_job["launch"]
    restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=cold_job["packets"],
                       materializer=cold_job["materializer"], reserve=_reserve_ok([]))
    later = time.time() + 3 * 86400
    manifest = build_evidence_offload_manifest(evidence_roots=[launch.parent], hot_window_seconds=86400,
                                               now=lambda: later, classifier=lambda *a, **k: None)
    assert manifest["candidate_count"] == 1
    receipt = launch / "launch_receipt.json"
    receipt.write_bytes(receipt.read_bytes().replace(b"aa", b"bb"))  # same size, other bytes
    applied = apply_evidence_offload(manifest, ack=EXECUTE_ACK, now=lambda: later)
    assert applied["offloaded_count"] == 0 and applied["skipped"] == [
        {"name": launch.name, "reason": "restored_tree_changed"}]
    assert receipt.is_file()


def test_a_job_whose_scene_workspace_is_retired_names_that_restore_first(tmp_path, monkeypatch):
    from blueprint_pipeline import beta_policy_screening_on_job as on_job
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run

    def retired(root, site):
        raise on_job.JobTestRefused(on_job.SCENE_WORKSPACE_RETIRED, restore_operation="restore-scene-workspace")
    monkeypatch.setattr(on_job, "job_source", retired)
    receipt = restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=tmp_path)
    assert receipt["status"] == "refused" and receipt["restore_operation"] == "restore-scene-workspace"



@pytest.mark.parametrize(("remote", "reason"), [
    ("missing", "restored_archive_missing"), ("differs", "restored_archive_differs"),
    ("unreadable", "restored_archive_check_failed")])
def test_a_restored_copy_is_removed_only_while_the_store_still_holds_its_archive(cold_job, remote, reason):
    """Follow-up 2: one metadata read before the local copy goes. Missing: kept, flagged, and offloaded as a
    new run on the next pass. Other bytes: kept and flagged. The read failing: kept and asked again."""
    import time

    from blueprint_pipeline.control_plane_evidence_offload import (
        EXECUTE_ACK, MISSING_ARCHIVE_POINTER_NAME, POINTER_SUFFIX, apply_evidence_offload,
        build_evidence_offload_manifest,
    )
    from blueprint_pipeline.screening_launch_run_restore import restore_launch_run
    from blueprint_pipeline.task_evaluation_configured_scene_object_store import (
        TaskEvaluationConfiguredSceneObjectStoreError as StoreError,
    )

    launch = cold_job["launch"]
    pointer = launch.parent / f"{launch.name}{POINTER_SUFFIX}"
    sealed = pointer.read_text()
    restore_launch_run(site_id="site-job1", request_id="req-1", packet_queue_root=cold_job["packets"],
                       materializer=cold_job["materializer"], reserve=_reserve_ok([]))
    later = time.time() + 3 * 86400

    def manifest():
        return build_evidence_offload_manifest(evidence_roots=[launch.parent], hot_window_seconds=86400,
                                               now=lambda: later, classifier=lambda *a, **k: None)

    def verifier(reference):
        raise StoreError({"missing": "configured_scene_artifact_missing",
                          "differs": "configured_scene_artifact_existing_identity_mismatch",
                          "unreadable": "configured_scene_artifact_head_failed"}[remote])

    def never(**kwargs):
        raise AssertionError("nothing is uploaded in this pass")
    first = manifest()
    assert first["candidates"][0].get("archived_pointer_digest")
    applied = apply_evidence_offload(first, ack=EXECUTE_ACK, stream_publisher=never, publisher=never,
                                     now=lambda: later, remote_verifier=verifier)
    assert applied["offloaded_count"] == 0 and applied["skipped"] == [{"name": launch.name, "reason": reason}]
    assert (launch / "launch_receipt.json").is_file()
    following = manifest()
    if remote == "missing":
        # The pointer moves into the run, never deleted and never left beside it; the next pass offloads
        # the run as a new one (the pointer's record goes with it into the new archive).
        assert (launch / MISSING_ARCHIVE_POINTER_NAME).read_text() == sealed and not pointer.exists()
        assert sorted(path.name for path in launch.parent.iterdir()) == [launch.name]
        assert [row["name"] for row in following["candidates"]] == [launch.name]
        assert "archived_pointer_digest" not in following["candidates"][0]
    else:
        assert pointer.read_text() == sealed
        assert following["candidates"][0].get("archived_pointer_digest")  # asked again next pass
