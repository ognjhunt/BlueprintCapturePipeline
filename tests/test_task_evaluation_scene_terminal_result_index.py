"""R8/R9/R10: the owner terminal-result index files REAL producer outputs.

Three idempotent, validate-everything-first, atomically-published stages carry a
paid policy-canary run's sealed evidence into the owner-scoped directory the
scene-progression terminal reconciler joins:

* stage A -- the launch bridge (``launch_request.json``/``launch_profile.json``
  retained by the launch dispatcher; the profile's ``scene_policy_binding`` names
  the owner intent);
* stage B -- the canary terminal set (the dispatcher's persisted projection and
  authenticated Website sync, its Vast post-teardown provider-zero receipt and the
  sealed ``dispatch_receipt.json`` that binds all three by file digest);
* stage C -- the durable result publication, derived from the evidence-offload
  pointer the production GC retention step leaves when it archives the canary
  root to the artifact store (bound to THIS run through the archive members'
  digests), never from a caller-supplied URI.

The canary root here is produced by the REAL dispatcher resume path and the REAL
Website-sync producer (transport stubbed); the pointer by the REAL offload.
"""
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_scene_terminal_result_index as index
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as reconciler
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient
from tests.test_task_evaluation_policy_canary_dispatcher import materialize_canary_root
from tests.test_task_evaluation_scene_terminal_reconciler import _env, _launch_request, _profile, _write

BUCKET = "blueprint-production-inputs"
RUN_ID = "scene-839873-policy-canary-owner-run"


def _owner_launch_run(state_root: Path, env, *, run_id: str = RUN_ID, launch_id: str = "launch-owner-1") -> Path:
    """The launch dispatcher's retained run root for an owner-bound policy-canary
    launch: ``launch_request.json`` (task_evaluation_launch_request.v1),
    ``launch_profile.json`` (task_evaluation_launch_profile.v1, owner-bound through
    ``internal_policy_canary_execution_plan.scene_policy_binding``) and the canary
    launch receipt (``execute_requested`` False -- the paid mutation happens in the
    canary dispatcher, so the launch reconciler never closes THIS run)."""
    root = state_root / launch_id
    profile = _profile(env["binding"])
    request = {**_launch_request(profile["profile_digest"]), "run_id": run_id, "launch_id": launch_id}
    _write(root / "launch_profile.json", profile)
    _write(root / "launch_request.json", request)
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1", "status": "queued_for_no_spend_preparation",
        "launch_id": launch_id, "run_id": run_id, "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"], "execute_requested": False,
        "provider_mutation_attempted": False, "allocator_invoked": False, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(root / "launch_receipt.json", receipt)
    # The launch-level Website sync already succeeded (retained by the launch
    # reconciler in its exact committed shape), so the tick has nothing to re-sync.
    identity = {key: receipt[key] for key in ("launch_id", "run_id", "request_digest", "receipt_digest")}
    attempt = {"schema_version": "task_evaluation_launch_webapp_sync_result.v1", "status": "succeeded",
               "provider_mutation_performed": False, "attempt_number": 1,
               "attempted_at": "2026-09-06T00:00:00+00:00", **identity,
               "response": {"schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                            "status": receipt["status"], "already_exists": False, **identity},
               "sync_result_digest": ""}
    attempt["sync_result_digest"] = canonical_digest(attempt, digest_field="sync_result_digest")
    _write(root / "webapp_sync_succeeded.json", attempt)
    return root


def _terminal_dir(env) -> Path:
    return Path(env["config"]["terminal_result_root"]) / env["intent_id"]


def _files(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()} if directory.is_dir() else set()


def _bridge(env, launch_root: Path):
    return index.index_launch_bridge(launch_run_root=launch_root, scene_intent_root=env["config"]["intent_root"],
                                     terminal_result_root=env["config"]["terminal_result_root"])


def _canary(env, canary_root: Path):
    return index.index_policy_canary_terminal(canary_run_root=canary_root,
                                              terminal_result_root=env["config"]["terminal_result_root"])


def _reconcile(env):
    return reconciler.reconcile_terminal_owner_result(intent=env["intent"], config=env["config"],
                                                      release=env["release"], now=env["now"], output=env["output"])


def _offload(canary_root: Path) -> Path:
    """Run the REAL evidence offload (the production GC retention step) over the
    canary root: archive to the artifact store, seal the pointer, delete the dir."""
    evidence_root = canary_root.parent
    manifest = offload.build_evidence_offload_manifest(
        evidence_roots=[evidence_root], hot_window_seconds=0, now=lambda: time.time() + 60,
        classifier=lambda *_args, **_kwargs: None)
    result = offload.apply_evidence_offload(
        manifest, ack=offload.EXECUTE_ACK,
        publisher=functools.partial(store.publish_configured_scene_artifact, client=_ContentAddressedClient(),
                                    bucket=BUCKET))
    assert result["offloaded"] and result["offloaded"][0]["name"] == canary_root.name, result
    return evidence_root / (canary_root.name + offload.POINTER_SUFFIX)


def _reseal(value: dict, field: str) -> dict:
    value = {k: v for k, v in value.items() if k != field}
    value[field] = canonical_digest(value, digest_field=field)
    return value


# ------------------------------------------------------------------ stage A: launch bridge


def test_launch_bridge_from_the_real_public_canary_launch_is_not_owner_bound(tmp_path, monkeypatch):
    """A public (website-direct) canary launch has no owner binding: the REAL
    launch dispatcher's run root is recognised and skipped, nothing is filed."""
    from blueprint_pipeline.task_evaluation_launch_dispatcher import dispatch_launch_request
    from tests.test_task_evaluation_policy_canary_preparation_dispatch import _profile_and_request

    env = _env(tmp_path)
    (tmp_path / "public").mkdir()
    profile, request = _profile_and_request(tmp_path / "public")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / f"{profile['profile_id']}.json").write_text(json.dumps(profile), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT", str(tmp_path / "preparations"))
    receipt = dispatch_launch_request(
        request_path=request_path, profile_dir=profiles, state_root=tmp_path / "runs", execute=True,
        allocator_runner=lambda _argv: (_ for _ in ()).throw(AssertionError("allocator must not run")))
    assert receipt["status"] == "queued_for_no_spend_preparation"
    launch_root = tmp_path / "runs" / request["launch_id"]
    assert (launch_root / "launch_profile.json").is_file() and (launch_root / "launch_request.json").is_file()
    result = _bridge(env, launch_root)
    assert result["status"] == "not_owner_bound"
    assert result["provider_mutation_performed"] is False
    assert not Path(env["config"]["terminal_result_root"]).exists()


def test_launch_bridge_files_the_owner_bound_launch_into_the_intent_directory(tmp_path):
    env = _env(tmp_path)
    launch_root = _owner_launch_run(tmp_path / "state", env)
    result = _bridge(env, launch_root)
    assert result["status"] == "launch_bridge_indexed"
    assert result["intent_id"] == env["intent_id"] and result["run_id"] == RUN_ID
    assert result["provider_mutation_performed"] is False
    directory = _terminal_dir(env)
    assert _files(directory) == {"launch_request.json", "launch_profile.json"}
    # Byte-exact copies of the producer's files.
    assert (directory / "launch_profile.json").read_bytes() == (launch_root / "launch_profile.json").read_bytes()
    assert (directory / "launch_request.json").read_bytes() == (launch_root / "launch_request.json").read_bytes()
    assert _bridge(env, launch_root) == result  # idempotent, byte-identical
    # The reconciler sees the bridge but there is no terminal result yet: nothing to join.
    assert _reconcile(env) is None


def test_launch_bridge_refuses_a_profile_whose_seal_does_not_match_and_writes_nothing(tmp_path):
    env = _env(tmp_path)
    launch_root = _owner_launch_run(tmp_path / "state", env)
    profile = json.loads((launch_root / "launch_profile.json").read_text())
    profile["source_commit"] = "e" * 40  # bytes changed after sealing
    (launch_root / "launch_profile.json").write_text(json.dumps(profile))
    with pytest.raises(index.TerminalResultIndexError, match="launch_profile_invalid"):
        _bridge(env, launch_root)
    assert not Path(env["config"]["terminal_result_root"]).exists()


def test_launch_bridge_without_a_matching_owner_intent_is_typed_and_writes_nothing(tmp_path):
    env = _env(tmp_path)
    launch_root = _owner_launch_run(tmp_path / "state", env)
    other_store = tmp_path / "other-intents"
    other_store.mkdir()
    result = index.index_launch_bridge(launch_run_root=launch_root, scene_intent_root=other_store,
                                       terminal_result_root=env["config"]["terminal_result_root"])
    assert result["status"] == "owner_intent_unresolved"
    assert not Path(env["config"]["terminal_result_root"]).exists()


# ------------------------------------------------------------- stage B: canary terminal set


def test_policy_canary_terminal_from_the_real_dispatcher_output_reconciles_the_blocked_owner_status(
        tmp_path, monkeypatch):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    result = _canary(env, run["root"])
    assert result["status"] == "policy_canary_terminal_indexed", result
    assert result["intent_id"] == env["intent_id"] and result["run_id"] == RUN_ID
    assert result["provider_mutation_performed"] is False
    directory = _terminal_dir(env)
    assert _files(directory) == {
        "launch_request.json", "launch_profile.json", "policy_canary_result_projection.json",
        "policy_canary_webapp_sync.json", "provider_zero_closure.json", "policy_canary_dispatch_receipt.json",
        "terminal_index_state.json"}
    # Byte-exact copies of the producer's sealed files -- the receipt binds them by digest.
    root = run["root"]
    assert (directory / "provider_zero_closure.json").read_bytes() == (
        root / "post_teardown_global_provider_zero.json").read_bytes()
    assert (directory / "policy_canary_dispatch_receipt.json").read_bytes() == (
        root / "dispatch_receipt.json").read_bytes()
    assert (directory / "policy_canary_result_projection.json").read_bytes() == (
        root / "artifacts/result_delivery/policy_canary_result_projection.json").read_bytes()
    state = json.loads((directory / "terminal_index_state.json").read_text())
    assert state["canary_run_root"] == str(root) and state["run_id"] == RUN_ID
    # The REAL reconciler joins them into a truthful BLOCKED terminal owner status.
    terminal = _reconcile(env)
    assert terminal["terminal"] is True and terminal["status"] == "blocked", terminal
    assert "policy_canary_episode_runner_failed" in terminal["blockers"]
    assert len(terminal["state"]["terminal_failed_children"]) == 20
    assert terminal["result_reference"] is None
    assert _canary(env, run["root"]) == result  # idempotent


def test_policy_canary_terminal_waits_for_the_launch_bridge_and_writes_nothing(tmp_path, monkeypatch):
    env = _env(tmp_path)
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    result = _canary(env, run["root"])
    assert result["status"] == "launch_bridge_pending" and result["run_id"] == RUN_ID
    assert not Path(env["config"]["terminal_result_root"]).exists()


def test_policy_canary_terminal_before_the_dispatch_receipt_is_typed_pending(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    (run["root"] / "dispatch_receipt.json").unlink()  # a run still in flight
    result = _canary(env, run["root"])
    assert result["status"] == "dispatch_receipt_pending"
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}


def test_policy_canary_terminal_refuses_a_projection_that_no_longer_matches_the_sealed_receipt(
        tmp_path, monkeypatch):
    """R10: the whole cross-bound set is validated BEFORE anything is written."""
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    projection_path = run["root"] / "artifacts/result_delivery/policy_canary_result_projection.json"
    projection_path.write_text(projection_path.read_text() + "\n")  # bytes drift from the receipt record
    with pytest.raises(index.TerminalResultIndexError, match="dispatch_receipt_binding_invalid"):
        _canary(env, run["root"])
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}


def test_policy_canary_terminal_refuses_a_receipt_without_persisted_records(tmp_path, monkeypatch):
    """A receipt sealed by a release that did not persist the projection/sync (the
    retained production run abe19c87 is one) cannot be indexed: nothing binds the
    files, and the index never manufactures them."""
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    receipt_path = run["root"] / "dispatch_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("policy_canary_result_projection")
    receipt.pop("policy_canary_webapp_sync")
    receipt_path.write_text(json.dumps(_reseal(receipt, "receipt_digest"), sort_keys=True,
                                       separators=(",", ":")) + "\n")
    with pytest.raises(index.TerminalResultIndexError, match="dispatch_receipt_records_missing"):
        _canary(env, run["root"])
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}


def test_policy_canary_terminal_refuses_an_unconfirmed_provider_zero(tmp_path, monkeypatch):
    """Resource closure is part of the cross-bound set: a receipt that binds a
    provider-zero receipt which does not confirm zero is never indexed."""
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID)
    zero_path = run["root"] / "post_teardown_global_provider_zero.json"
    zero = json.loads(zero_path.read_text())
    zero.update(status="provider_zero_pending", live_instance_count=1, provider_zero_verified=False)
    zero_path.write_text(json.dumps(_reseal(zero, "receipt_digest"), sort_keys=True) + "\n")
    receipt_path = run["root"] / "dispatch_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["provider_zero"] = {**receipt["provider_zero"], **index.file_record(zero_path)}
    receipt_path.write_text(json.dumps(_reseal(receipt, "receipt_digest"), sort_keys=True,
                                       separators=(",", ":")) + "\n")
    with pytest.raises(index.TerminalResultIndexError, match="provider_zero_not_confirmed"):
        _canary(env, run["root"])
    assert _files(_terminal_dir(env)) == {"launch_request.json", "launch_profile.json"}


# --------------------------------------------------------- stage C: durable publication


def test_publication_derives_from_the_real_offload_pointer_and_completes_the_owner_status(
        tmp_path, monkeypatch):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID, completed=True)
    assert run["receipt"]["status"] == "completed_unqualified"
    assert _canary(env, run["root"])["status"] == "policy_canary_terminal_indexed"
    # Before retention archives the run there is no durable publication: explicit,
    # never completed, and no publication is manufactured.
    pending = _reconcile(env)
    assert pending["terminal"] is False and pending["blockers"] == ["terminal_result_publication_pending"]
    assert not (_terminal_dir(env) / "terminal_result_publication.json").exists()
    # The REAL retention step archives the canary root and leaves the sealed pointer.
    pointer_path = _offload(run["root"])
    assert pointer_path.is_file() and not run["root"].exists()
    terminal = _reconcile(env)
    assert terminal["terminal"] is True and terminal["status"] == "completed", terminal
    pointer = json.loads(pointer_path.read_text())
    assert terminal["result_reference"] == {
        "uri": pointer["uri"], "digest": run["receipt"]["policy_canary_projection_digest"],
        "size_bytes": pointer["size_bytes"]}
    assert terminal["result_reference"]["uri"].startswith("s3://")
    publication = json.loads((_terminal_dir(env) / "terminal_result_publication.json").read_text())
    assert publication["schema_version"] == reconciler.PUBLICATION_SCHEMA
    assert publication["archive_digest"] == pointer["digest"] and publication["provider_allocated"] is False
    assert publication["publication_digest"] == canonical_digest(publication, digest_field="publication_digest")
    assert _reconcile(env) == terminal  # duplicate tick is byte-identical


def test_publication_refuses_a_pointer_that_does_not_bind_this_run(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID, completed=True)
    _canary(env, run["root"])
    pointer_path = _offload(run["root"])
    pointer = json.loads(pointer_path.read_text())
    for member in pointer["members"]:
        if member["relative_path"].endswith("policy_canary_result_projection.json"):
            member["sha256"] = "sha256:" + "0" * 64  # the archive holds a different projection
    pointer_path.chmod(0o640)
    pointer_path.write_text(json.dumps(_reseal(pointer, "pointer_digest"), indent=2, sort_keys=True) + "\n")
    result = index.index_result_publication(terminal_directory=_terminal_dir(env))
    assert result["status"] == "publication_pointer_unbound"
    terminal = _reconcile(env)
    assert terminal["status"] != "completed"
    assert "terminal_result_publication_pointer_unbound" in terminal["blockers"]
    assert not (_terminal_dir(env) / "terminal_result_publication.json").exists()


@pytest.mark.skipif(os.getuid() == 0, reason="root bypasses discretionary mode bits")
def test_unreadable_pointer_is_typed_not_a_crash(tmp_path, monkeypatch):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path / "state", env))
    run = materialize_canary_root(tmp_path / "canaries", monkeypatch, run_id=RUN_ID, completed=True)
    _canary(env, run["root"])
    pointer_path = _offload(run["root"])
    pointer_path.chmod(0)
    try:
        result = index.index_result_publication(terminal_directory=_terminal_dir(env))
        assert result["status"] == "publication_pointer_unreadable"
        terminal = _reconcile(env)
        assert terminal["status"] != "completed"
        assert "terminal_result_publication_pointer_unreadable" in terminal["blockers"]
    finally:
        pointer_path.chmod(0o440)
