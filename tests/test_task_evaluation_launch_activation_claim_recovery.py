# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_launch_activation_claim_recovery.py
#   src/blueprint_pipeline/task_evaluation_launch_activation_worker.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
"""A launch-activation claim whose worker died is retired, so it stops holding disk.

Production shape (2026-10-31, capture-447): the oneshot activation unit hit its
then one-hour TimeoutStartSec while the worker was still building a provider
bundle and was killed with SIGTERM. Its envelope stayed in ``processing/``
with no lock and no result. Later runs claim only from ``pending/``, so nothing
ever moved it, and the storage GC, which reads pending and processing rows as
live references, kept the activation's 14.8 GB directory as ``queue_referenced``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_activation_claim_recovery as recovery
from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, build_derived_directory_manifest, run_storage_gc
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    launch_activation_status,
    stage_launch_activation_request,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
from tests.test_task_evaluation_launch_activation_contract import request as activation_request

ROOT = Path(__file__).resolve().parents[1]
CLAIM_COMMIT = "a" * 40  # activation_request()'s expected_production_commit
RECOVERING = "b" * 40
LANE = "task_evaluation_scene_configuration"
SERVICE_ACCOUNT = __import__("pwd").getpwuid(os.geteuid()).pw_name


def _killed_claim(tmp_path: Path, *, lane: str = LANE) -> tuple[dict, Path, str]:
    """Stage an activation and leave it as a killed worker does: claimed, unlocked, with no result."""

    value = activation_request(lane=lane)
    queue = tmp_path / "state" / "task-evaluation-launch-activations"
    stage_launch_activation_request(value=value, queue_root=queue, submitted_by="fixture-automation")
    name = next((queue / "pending").glob("*.json")).name
    os.replace(queue / "pending" / name, queue / "processing" / name)
    return value, queue, name


def _claim_time(queue: Path, name: str) -> float:
    info = (queue / "processing" / name).stat()
    return max(info.st_mtime, info.st_ctime)


def _abandoned(queue: Path, name: str) -> float:
    return _claim_time(queue, name) + recovery.ABANDONED_AFTER_SECONDS + 60


def _retire(queue: Path, now: float, **overrides) -> list[dict]:
    return recovery.retire_abandoned_activation_claims(
        queue, **{"recovering_source_commit": RECOVERING, "now": now, **overrides})


def _states(queue: Path, name: str) -> list[str]:
    return [state for state in ("pending", "processing", "prepared", "blocked") if (queue / state / name).exists()]


def _sealed_result(queue: Path, name: str, value: dict, **fields) -> dict:
    result = {"schema_version": recovery.RESULT_SCHEMA_VERSION, "activation_id": value["activation_id"],
              "provider_mutation_performed": False, "paid_execution_requested": False, "blockers": [],
              **fields, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_launch_preparation_record_exclusive(queue / "results" / name, result)
    return result


def test_the_abandonment_bound_is_the_activation_units_start_timeout_plus_a_stop_margin(
    tmp_path, monkeypatch
) -> None:
    unit = (ROOT / "deploy/systemd/blueprint-task-evaluation-launch-activation.service").read_text()
    timeouts = [line.split("=", 1)[1] for line in unit.splitlines() if line.startswith("TimeoutStartSec=")]
    assert timeouts == ["255min"]
    assert recovery.ACTIVATION_UNIT_TIMEOUT_SECONDS == 255 * 60
    seen = {}
    receipt = tmp_path / "receipt.json"

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        receipt.write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    worker.default_activation_preparer(
        lane=LANE, context_path=tmp_path / "context.json", receipt_path=receipt, repository_root=tmp_path)
    # The worker's own preparation command ends first, then the unit, then any claim it made.
    assert seen["timeout"] < recovery.ACTIVATION_UNIT_TIMEOUT_SECONDS < recovery.ABANDONED_AFTER_SECONDS
    assert recovery.ABANDONED_AFTER_SECONDS - recovery.ACTIVATION_UNIT_TIMEOUT_SECONDS == recovery.STOP_MARGIN_SECONDS
    assert recovery.STOP_MARGIN_SECONDS > 90  # systemd's default TimeoutStopSec before SIGKILL


def test_the_result_is_the_workers_own_document() -> None:
    from blueprint_pipeline.control_plane_pin_proofs import PREPARED_ACTIVATION_STATUSES

    assert recovery.RESULT_SCHEMA_VERSION == worker.RESULT_SCHEMA_VERSION
    assert recovery.PREPARED_STATUSES == PREPARED_ACTIVATION_STATUSES


def test_a_killed_workers_claim_is_sealed_blocked_and_moved_byte_for_byte(tmp_path) -> None:
    value, queue, name = _killed_claim(tmp_path)
    original = (queue / "processing" / name).read_bytes()
    claimed_at = _claim_time(queue, name)
    now = _abandoned(queue, name)

    rows = _retire(queue, now, recovered_by="control_plane_storage_gc")

    assert [(row["name"], row["action"]) for row in rows] == [(name, "retired")]
    assert _states(queue, name) == ["blocked"]
    assert (queue / "blocked" / name).read_bytes() == original
    sealed = json.loads((queue / "results" / name).read_text())
    assert rows[0]["result_digest"] == sealed["result_digest"]
    assert sealed["result_digest"] == canonical_digest(sealed, digest_field="result_digest")
    assert sealed["schema_version"] == worker.RESULT_SCHEMA_VERSION
    assert sealed["status"] == "blocked" and sealed["activation_id"] == value["activation_id"]
    assert sealed["blockers"] == ["launch_activation_claim_interrupted:worker_process_terminated"]
    assert sealed["source_commit"] == CLAIM_COMMIT
    assert sealed["provider_mutation_performed"] is False and sealed["paid_execution_requested"] is False
    assert sealed["catalog_mutation_state"] == "unknown_if_preparation_started"
    claim = sealed["claim_recovery"]
    assert claim["schema_version"] == "task_evaluation_launch_activation_claim_recovery.v1"
    assert claim["envelope_name"] == name
    assert claim["envelope_sha256"] == "sha256:" + hashlib.sha256(original).hexdigest()
    assert claim["envelope_size_bytes"] == len(original)
    assert claim["claim_source_commit"] == CLAIM_COMMIT
    assert claim["recovered_by_source_commit"] == RECOVERING
    assert claim["recovered_by"] == "control_plane_storage_gc"
    assert claim["claimed_at_epoch"] == claimed_at and claim["recovered_at_epoch"] == now
    assert claim["abandoned_after_seconds"] == recovery.ABANDONED_AFTER_SECONDS
    assert claim["worker_lease_held"] is False
    assert claim["request_re_executed"] is False and claim["envelope_bytes_modified"] is False
    # The status the WebApp reads is terminal and says why.
    status = launch_activation_status(activation_id=value["activation_id"], queue_root=queue)
    assert status["status"] == "blocked" and status["worker_status"] == "blocked"
    assert status["blockers"] == [recovery.INTERRUPTED_CLAIM_BLOCKER]
    # A second pass finds nothing to do and writes nothing again.
    sealed_bytes = (queue / "results" / name).read_bytes()
    assert _retire(queue, now + 3600) == []
    assert (queue / "results" / name).read_bytes() == sealed_bytes


def test_an_unknown_recovering_release_is_recorded_as_unknown(tmp_path) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    _retire(queue, _abandoned(queue, name), recovering_source_commit="")
    assert json.loads((queue / "results" / name).read_text())["claim_recovery"]["recovered_by_source_commit"] is None


def test_a_young_claim_is_never_touched(tmp_path) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    original = (queue / "processing" / name).read_bytes()
    rows = _retire(queue, _claim_time(queue, name) + recovery.ABANDONED_AFTER_SECONDS - 60)
    assert rows == [{"name": name, "action": "kept", "reason": "young"}]
    assert _states(queue, name) == ["processing"] and not (queue / "results" / name).exists()
    assert (queue / "processing" / name).read_bytes() == original


def test_a_locked_claim_is_never_touched_however_old(tmp_path) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    now = _abandoned(queue, name) + 30 * 86400
    with (queue / "processing" / name).open("rb") as lease:  # the worker's lease, held by a live process
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert _retire(queue, now) == [{"name": name, "action": "kept", "reason": "lease_held"}]
        assert _states(queue, name) == ["processing"] and not (queue / "results" / name).exists()
    assert [row["action"] for row in _retire(queue, now)] == ["retired"]


@pytest.mark.parametrize("shape", ["linked", "directory", "oversized", "tampered", "renamed", "empty"])
def test_a_row_that_is_not_a_readable_claim_is_kept(tmp_path, monkeypatch, shape) -> None:
    value, queue, name = _killed_claim(tmp_path)
    claimed = queue / "processing" / name
    if shape == "linked":
        target = tmp_path / "elsewhere.json"
        os.replace(claimed, target)
        claimed.symlink_to(target)
    elif shape == "directory":
        claimed.unlink()
        claimed.mkdir()
    elif shape == "oversized":
        monkeypatch.setattr(recovery, "MAX_QUEUE_MESSAGE_BYTES", claimed.stat().st_size - 1)
    elif shape == "tampered":
        envelope = json.loads(claimed.read_text())
        envelope["submitted_by"] = "someone-else"
        claimed.unlink()
        claimed.write_text(json.dumps(envelope))
    elif shape == "renamed":
        os.replace(claimed, queue / "processing" / ("other-" + name))
        name = "other-" + name
    else:
        claimed.unlink()
        claimed.write_bytes(b"")  # a claim placeholder whose pending row is gone
    reason = {"linked": "linked", "directory": "not_regular", "oversized": "oversized"}.get(shape, "envelope_invalid")
    now = (queue / "processing").stat().st_mtime + recovery.ABANDONED_AFTER_SECONDS + 3600
    assert _retire(queue, now) == [{"name": name, "action": "kept", "reason": reason}]
    assert not list((queue / "results").glob("*.json")) and not list((queue / "blocked").iterdir())


def test_a_claim_also_held_in_another_state_is_kept(tmp_path) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    shutil.copyfile(queue / "processing" / name, queue / "prepared" / name)
    assert _retire(queue, _abandoned(queue, name)) == [{"name": name, "action": "kept", "reason": "identity_ambiguous"}]
    assert not (queue / "results" / name).exists()


@pytest.mark.parametrize("status,state", [
    ("profile_authority_materialized_no_execution", "prepared"), ("blocked", "blocked")])
def test_a_result_sealed_before_the_worker_died_only_finishes_the_move(tmp_path, status, state) -> None:
    value, queue, name = _killed_claim(tmp_path)
    original = (queue / "processing" / name).read_bytes()
    _sealed_result(queue, name, value, status=status)
    sealed_bytes = (queue / "results" / name).read_bytes()
    rows = _retire(queue, _abandoned(queue, name))
    assert rows == [{"name": name, "action": "finished", "terminal_state": state,
                     "activation_id": value["activation_id"]}]
    assert _states(queue, name) == [state] and (queue / state / name).read_bytes() == original
    assert (queue / "results" / name).read_bytes() == sealed_bytes


@pytest.mark.parametrize("problem", ["foreign", "unsealed", "unknown_status", "policy_canary"])
def test_a_result_recovery_cannot_finish_keeps_the_claim(tmp_path, problem) -> None:
    value, queue, name = _killed_claim(tmp_path)
    if problem == "foreign":
        _sealed_result(queue, name, dict(value, activation_id="another-activation"), status="blocked")
    elif problem == "unsealed":
        (queue / "results" / name).write_text(json.dumps({"status": "blocked"}))
    elif problem == "unknown_status":
        _sealed_result(queue, name, value, status="something_new")
    else:
        # The worker writes this campaign's dispatch envelope between its result and its move.
        _sealed_result(queue, name, value, status="policy_campaign_queue_materialized_no_execution",
                       run_kind="internal_policy_canary")
    reason = {"foreign": "result_invalid", "unsealed": "result_invalid", "unknown_status": "result_status_unknown",
              "policy_canary": "policy_canary_dispatch_unverified"}[problem]
    assert _retire(queue, _abandoned(queue, name)) == [{"name": name, "action": "kept", "reason": reason}]
    assert _states(queue, name) == ["processing"]


def test_a_recovery_that_dies_between_its_result_and_its_move_is_finished_by_the_next(
    tmp_path, monkeypatch
) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    now = _abandoned(queue, name)
    real_replace = os.replace

    def killed(source, destination):
        raise OSError("killed before the move")

    monkeypatch.setattr(recovery.os, "replace", killed)
    assert _retire(queue, now) == [{"name": name, "action": "failed", "error": "OSError"}]
    assert _states(queue, name) == ["processing"]
    sealed_bytes = (queue / "results" / name).read_bytes()
    monkeypatch.setattr(recovery.os, "replace", real_replace)
    rows = _retire(queue, now + 3600)
    assert [(row["action"], row["terminal_state"]) for row in rows] == [("finished", "blocked")]
    assert _states(queue, name) == ["blocked"]
    assert (queue / "results" / name).read_bytes() == sealed_bytes
    assert not [path for path in (queue / "results").iterdir() if path.name.startswith(".")]


def test_a_placeholder_left_beside_its_pending_row_is_removed(tmp_path) -> None:
    value, queue, name = _killed_claim(tmp_path)
    os.replace(queue / "processing" / name, queue / "pending" / name)
    (queue / "processing" / name).write_bytes(b"")  # killed between the claim's reservation and its replace
    now = _abandoned(queue, name)
    assert _retire(queue, now - 3600 - recovery.ABANDONED_AFTER_SECONDS) == [
        {"name": name, "action": "kept", "reason": "young"}]
    assert _retire(queue, now) == [{"name": name, "action": "placeholder_removed"}]
    assert _states(queue, name) == ["pending"] and not (queue / "results" / name).exists()
    status = launch_activation_status(activation_id=value["activation_id"], queue_root=queue)
    assert status["status"] == "pending"


def test_a_dry_run_reports_without_mutating(tmp_path) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    rows = _retire(queue, _abandoned(queue, name), apply=False)
    assert [row["action"] for row in rows] == ["would_retire"]
    assert _states(queue, name) == ["processing"] and not list((queue / "results").iterdir())


def test_a_root_caller_seals_the_result_owned_like_the_results_directory(tmp_path, monkeypatch) -> None:
    _value, queue, name = _killed_claim(tmp_path)
    results = queue / "results"
    owner = results.stat()
    calls = []
    real_chown = os.chown

    def chown(path, uid, gid, *, follow_symlinks=True):
        calls.append((Path(path).parent, Path(path).name.startswith("."), uid, gid, follow_symlinks,
                      (results / name).exists()))
        real_chown(path, uid, gid, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(recovery.os, "geteuid", lambda: 0)
    monkeypatch.setattr(recovery.os, "chown", chown)
    assert [row["action"] for row in _retire(queue, _abandoned(queue, name))] == ["retired"]
    # The staged copy gets the owner before the result name exists, never after.
    assert calls == [(results, True, owner.st_uid, owner.st_gid, False, False)]
    sealed = (results / name).stat()
    assert (sealed.st_uid, sealed.st_gid) == (owner.st_uid, owner.st_gid)
    assert oct(sealed.st_mode & 0o777) == oct(0o440)
    assert [path.name for path in results.iterdir()] == [name]


def test_the_worker_retires_an_abandoned_claim_before_it_claims(tmp_path, monkeypatch) -> None:
    value, queue, name = _killed_claim(tmp_path)
    moment = _abandoned(queue, name)
    monkeypatch.setattr(recovery, "_clock", lambda: moment)
    for directory in ("preparations", "inputs"):
        (tmp_path / directory).mkdir()

    run = worker.process_launch_activation_queue(
        queue_root=queue, preparation_queue_root=tmp_path / "preparations", preparation_input_root=tmp_path / "inputs",
        activation_root=tmp_path / "activations", allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT, service_group=SERVICE_ACCOUNT, repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix="s3://blueprint-production-inputs/coordinator-release-windows/",
        profile_dir=tmp_path / "profiles", webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "standing", source_commit=RECOVERING,
        preparer=lambda **_kwargs: pytest.fail("an abandoned claim is never re-executed"))

    assert run["status"] == "idle" and run["processed_count"] == 0
    assert [(row["name"], row["action"]) for row in run["recovered_claims"]] == [(name, "retired")]
    assert _states(queue, name) == ["blocked"]
    claim = json.loads((queue / "results" / name).read_text())["claim_recovery"]
    assert claim["recovered_by"] == "task_evaluation_launch_activation_worker"
    assert claim["recovered_by_source_commit"] == RECOVERING
    # A worker run with nothing stranded reports nothing extra.
    again = worker.process_launch_activation_queue(
        queue_root=queue, preparation_queue_root=tmp_path / "preparations", preparation_input_root=tmp_path / "inputs",
        activation_root=tmp_path / "activations", allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT, service_group=SERVICE_ACCOUNT, repository_root=tmp_path,
        destination_prefix="s3://blueprint-production-inputs/activated",
        release_window_prefix="s3://blueprint-production-inputs/coordinator-release-windows/",
        profile_dir=tmp_path / "profiles", webapp_catalog=tmp_path / "catalog.json",
        standing_authorization_dir=tmp_path / "standing", source_commit=RECOVERING)
    assert again["status"] == "idle" and "recovered_claims" not in again


def _activation_directory(root: Path, activation_id: str, *, idle_seconds: float, now: float) -> Path:
    directory = root / activation_id
    (directory / "launch-set").mkdir(parents=True)
    (directory / "launch-set" / "bundle.zip").write_bytes(b"z" * 64)
    for path in (directory / "launch-set" / "bundle.zip", directory / "launch-set", directory):
        os.utime(path, (now - idle_seconds, now - idle_seconds))
    return directory


@pytest.mark.parametrize("retention", ["none", "pinned", "young"])
def test_after_recovery_the_activation_directory_is_no_longer_queue_referenced(tmp_path, retention) -> None:
    value, queue, name = _killed_claim(tmp_path)
    now = _abandoned(queue, name)
    derived = tmp_path / "launch-activations"
    directory = _activation_directory(derived, value["activation_id"], now=now,
                                      idle_seconds=60 if retention == "young" else 2 * 86400)
    pins = tmp_path / "pins"
    if retention == "pinned":
        write_storage_pin(pins_root=pins, kind="activation", owner_id=value["activation_id"], paths=[directory],
                          now=lambda: now)

    def manifest() -> dict:
        return build_derived_directory_manifest(derived_roots=[derived], pins_root=pins, queue_roots=[queue],
                                                minimum_age_seconds=3600, now=lambda: now,
                                                classifier=lambda *_a, **_k: None)

    before = manifest()
    reason = {"none": "queue_referenced", "pinned": "pinned", "young": "queue_referenced"}[retention]
    assert before["candidate_count"] == 0 and before["retained_counts"][reason] == 1
    assert [row["action"] for row in _retire(queue, now)] == ["retired"]
    after = manifest()
    assert after["retained_counts"]["queue_referenced"] == 0
    if retention == "none":
        assert [row["name"] for row in after["candidates"]] == [value["activation_id"]]
    else:  # pins and age still apply
        assert after["candidate_count"] == 0 and after["retained_counts"][retention] == 1


def test_the_storage_gc_retires_the_claim_and_reclaims_its_directory_in_one_tick(tmp_path, monkeypatch) -> None:
    import blueprint_pipeline.control_plane_storage_gc as gc_module

    monkeypatch.setattr(gc_module, "reconcile_terminal_cache_pins", lambda **_kwargs: {"status": "applied"})
    value, queue, name = _killed_claim(tmp_path)
    now = _abandoned(queue, name)
    derived = tmp_path / "launch-activations"
    directory = _activation_directory(derived, value["activation_id"], idle_seconds=2 * 86400, now=now)
    common = dict(content_store_roots=[], derived_roots=[derived], queue_roots=[queue], pins_root=tmp_path / "pins",
                  derived_minimum_age_seconds=3600, now=lambda: now, classifier=lambda *_a, **_k: None)

    planned = run_storage_gc(**common)
    assert planned["abandoned_activation_claims"]["status"] == "dry_run"
    assert [row["action"] for row in planned["abandoned_activation_claims"]["claims"]] == ["would_retire"]
    assert planned["derived_directories"]["retained_counts"]["queue_referenced"] == 1
    assert _states(queue, name) == ["processing"] and directory.exists()

    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK)
    assert not applied.get("phase_errors")
    phase = applied["abandoned_activation_claims"]
    assert phase["status"] == "applied" and phase["action_counts"] == {"retired": 1}
    assert applied["derived_directories"]["removed"] == [{"name": value["activation_id"], "size_bytes": 64}]
    assert _states(queue, name) == ["blocked"] and not directory.exists()
    claim = json.loads((queue / "results" / name).read_text())["claim_recovery"]
    assert claim["recovered_by"] == "control_plane_storage_gc"
    assert claim["recovered_by_source_commit"] is None  # the GC's own release is unknown on the host


def test_the_storage_gc_skips_the_phase_without_an_activation_queue(tmp_path) -> None:
    queue = tmp_path / "queue"
    (queue / "processing").mkdir(parents=True)
    report = run_storage_gc(content_store_roots=[], derived_roots=[], queue_roots=[queue], pins_root=tmp_path / "pins",
                            now=lambda: 1_000.0, classifier=lambda *_a, **_k: None)
    assert "abandoned_activation_claims" not in report and not report.get("phase_errors")


def test_a_retired_claim_never_re_activates_its_stale_preparation(tmp_path) -> None:
    """The automation that staged it reads the retirement as terminal, never as a reason to stage again."""

    from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as automation
    from tests.test_task_evaluation_scene_configuration_activation_automation import (
        NOW, SUPERSEDED_RUNNING_COMMIT, _advance, _intent, _preparation,
    )

    intent_root, intent = _intent(tmp_path)
    result_path = _preparation(tmp_path)
    observed, _lineage, _window = _advance(tmp_path, result_path, intent_root)
    queue = tmp_path / "activations"
    name = next((queue / "pending").glob("*.json")).name
    request = json.loads((queue / "pending" / name).read_text())["request"]
    os.replace(queue / "pending" / name, queue / "processing" / name)
    assert [row["action"] for row in _retire(queue, _abandoned(queue, name))] == ["retired"]

    again, lineage, window = _advance(tmp_path, result_path, intent_root, now=NOW + timedelta(minutes=5))
    assert again == observed and lineage.published == {} and window.published == {}
    superseded, _l, _w = _advance(tmp_path, result_path, intent_root, running_commit=SUPERSEDED_RUNNING_COMMIT)
    assert superseded["status"] == "preparation_bound_to_superseded_release"
    restaged = stage_launch_activation_request(value=request, queue_root=queue, submitted_by="fixture-automation")
    assert restaged["already_exists"] is True
    assert not list((queue / "pending").glob("*.json")) and _states(queue, name) == ["blocked"]
    launch = automation.advance_scene_configuration_launch(
        progression=observed, activation_queue_root=queue, profile_dir=tmp_path / "profiles",
        standing_authorization_dir=tmp_path / "standing", progression_root=tmp_path / "progression",
        intent=intent, submitter=lambda _request: pytest.fail("a retired activation never launches"),
        now=NOW + timedelta(minutes=6))
    assert launch["status"] == "awaiting_scene_configuration_authority"


def test_other_consumers_read_a_retired_claim_as_terminal_and_never_launched(tmp_path) -> None:
    from blueprint_pipeline.control_plane_pin_proofs import _expired_unlaunched
    from blueprint_pipeline.task_evaluation_blocked_activation_retention import (
        BlockedActivationRetentionError, build_blocked_activation_retention_plan,
    )
    from blueprint_pipeline.task_evaluation_configured_controls_progression_worker import _activation_authority

    value, queue, name = _killed_claim(tmp_path)
    now = _abandoned(queue, name)
    base = tmp_path / "launch-activations"
    _activation_directory(base, value["activation_id"], idle_seconds=2 * 86400, now=now)
    _retire(queue, now)
    owner = value["activation_id"]
    # The storage pin proof never counts it as a prepared activation that lapsed unlaunched.
    assert _expired_unlaunched(owner, queue, now=now + 30 * 86400) == (None, "activation_result_not_prepared")
    # No progression finds an authority to launch with.
    assert _activation_authority(activation_queue_root=queue, profile_dir=tmp_path / "profiles",
                                 activation_id=owner) is None
    # The bundle reclaimer accepts it as a terminal blocked claim and stops only at the missing receipt.
    with pytest.raises(BlockedActivationRetentionError, match="^blocked_activation_retention_preparation_receipt_invalid$"):
        build_blocked_activation_retention_plan(activation_root=base / owner, activation_base_root=base,
                                                state_root=queue.parent, activation_queue_root=queue)


def test_recovery_stays_off_the_hot_lane_and_out_of_the_worker() -> None:
    from tests.test_live_pipeline_import_isolation import HOT_LANE_MODULES, _transitive_local_modules

    reachable = _transitive_local_modules(("task_evaluation_launch_activation_claim_recovery",))
    assert "task_evaluation_launch_activation_queue" in reachable
    assert not reachable & HOT_LANE_MODULES
    assert not reachable & {"task_evaluation_launch_activation_worker", "control_plane_storage_gc",
                            "control_plane_pin_proofs"}
