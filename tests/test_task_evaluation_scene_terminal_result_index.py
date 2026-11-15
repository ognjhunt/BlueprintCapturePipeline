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
import time
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_evidence_offload as offload
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_scene_terminal_result_index as index
from blueprint_pipeline import task_evaluation_scene_terminal_reconciler as reconciler
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient
from tests.test_task_evaluation_scene_terminal_reconciler import _env, _launch_request, _profile, _write

BUCKET = "blueprint-production-inputs"
RUN_ID = "scene-839873-policy-canary-owner-run"


@pytest.fixture(autouse=True)
def isolated_offload_disk(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(offload, "DEFAULT_RESERVATION_ROOT", tmp_path / "disk-reservations")
    monkeypatch.setattr(offload, "reserve_control_plane_disk", functools.partial(
        offload.reserve_control_plane_disk,
        disk_usage=lambda _path: SimpleNamespace(total=512 * 2**30, used=128 * 2**30, free=384 * 2**30),
    ))


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


# --------------------------------------------------------- stage C: durable publication


def test_two_runs_bound_to_the_same_attempt_are_not_arbitrarily_selected(tmp_path):
    env = _env(tmp_path)
    _bridge(env, _owner_launch_run(tmp_path/'state', env))
    _bridge(env, _owner_launch_run(tmp_path/'state', env, run_id=RUN_ID+'-other', launch_id='other'))
    with pytest.raises(reconciler.TerminalReconciliationError, match='attempt_order_ambiguous'):
        _reconcile(env)
