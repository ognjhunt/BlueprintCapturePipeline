# Covers (for impacted-test selection):
#   src/blueprint_pipeline/policy_canary_staged_object_absence.py
#   src/blueprint_pipeline/policy_canary_official_billing.py
#   src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py
#   src/blueprint_pipeline/provider_output_promotion.py
#   src/blueprint_pipeline/provider_output_promotion_records.py
"""A canary whose promotion failed closes once resume proves its staged objects absent (review C2).

The lane seals ``all_staged_objects_absent`` once. Before this, a deferred
cleanup kept the run in ``awaiting_official_billing`` forever at the queue
head, because resume never rewrites the sealed lane result. Closeout and
billing now accept resume's digest-bound absence proof instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import vast_official_billing_extractor as billing
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import provider_output_promotion_records as records
from blueprint_pipeline.policy_canary_staged_object_absence import (
    NOT_DURABLE,
    NOT_FINAL,
    billing_staged_objects_absent,
)
from blueprint_pipeline.task_evaluation_configured_scene_object_store import (
    TaskEvaluationConfiguredSceneObjectStoreError,
)
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _join_session_closeout,
    dispatch_policy_canary_activation,
)
from tests.provider_output_fixtures import quick10_shaped_archive, write_staged_absence_proof
from tests.test_provider_output_promotion import SMALL, World
from tests.test_task_evaluation_policy_canary_dispatcher import COMMIT, _inputs

INSTANCE = 49_247_792
DISPATCHER = "blueprint_pipeline.task_evaluation_policy_canary_dispatcher"
ZERO = {"schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1", "provider_zero_verified": True,
        "live_instance_count": 0, "blockers": []}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _completed_inner() -> dict:
    return {"episodes": [{"status": "completed"} for _ in range(20)], "blockers": []}


def _lane(attempt: Path, *, sealed_absent: bool, native: str | None = None) -> dict:
    return {"continuing_spend_from_this_run": False, "vast_instance_ids": [INSTANCE], "attempt_root": str(attempt),
            "native_control_result_path": native,
            "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                  "all_staged_objects_absent": sealed_absent}}


@pytest.mark.parametrize(("status", "native", "blockers"), [
    ("promoted", "immutable_execution/result.json", []),
    ("absent_confirmed", None, []),
    ("absent_confirmed", "immutable_execution/result.json", [NOT_DURABLE]),
    ("failed", None, [NOT_DURABLE]),
])
def test_closeout_accepts_an_absence_proof_and_names_an_output_that_was_not_kept(tmp_path, status, native, blockers):
    attempt = tmp_path / "attempt_001"
    lane = _lane(attempt, sealed_absent=False, native=native)
    deferred = _join_session_closeout(inner=_completed_inner(), adapter=lane, provider_zero=ZERO)
    assert deferred["session_closeout"]["teardown_completed"] is False
    assert "policy_canary_teardown_incomplete" in deferred["blockers"]

    write_staged_absence_proof(attempt, promotion_status=status)
    joined = _join_session_closeout(inner=_completed_inner(), adapter=lane, provider_zero=ZERO)

    assert joined["session_closeout"]["teardown_completed"] is True
    assert "policy_canary_teardown_incomplete" not in joined["blockers"]
    assert [blocker for blocker in joined["blockers"] if blocker == NOT_DURABLE] == blockers
    assert joined["status"] == ("completed_unqualified" if not blockers else "blocked")


def test_closeout_reads_no_proof_when_the_lane_proved_absence_and_refuses_a_foreign_one(tmp_path):
    attempt = tmp_path / "attempt_001"
    proof = write_staged_absence_proof(attempt, promotion_status="failed")
    # Download mode: the sealed flag answers; a proof on disk is not consulted.
    sealed = _join_session_closeout(inner=_completed_inner(), adapter=_lane(attempt, sealed_absent=True),
                                    provider_zero=ZERO)
    assert sealed["status"] == "completed_unqualified" and NOT_DURABLE not in sealed["blockers"]
    # A proof for a manifest that never required promotion is not a streamed attempt's.
    write_staged_absence_proof(tmp_path / "ungated", gated=False)
    ungated = _join_session_closeout(inner=_completed_inner(), adapter=_lane(tmp_path / "ungated", sealed_absent=False),
                                     provider_zero=ZERO)
    assert ungated["session_closeout"]["teardown_completed"] is False
    # A proof altered after it was sealed proves nothing, and says so.
    value = json.loads(proof.read_text())
    value["promotion_status"] = "promoted"
    proof.write_text(json.dumps(value), encoding="utf-8")
    tampered = _join_session_closeout(inner=_completed_inner(), adapter=_lane(attempt, sealed_absent=False),
                                      provider_zero=ZERO)
    assert tampered["session_closeout"]["teardown_completed"] is False
    assert "policy_canary_staged_object_absence_proof_invalid:staged_object_absence_proof_digest_mismatch" in (
        tampered["blockers"])


def _receipt(staging: Path, *, witness: str) -> dict:
    """A promotion receipt bound to the staging manifest, its witness in ``witness`` state."""
    manifest = json.loads((staging / records.STAGING_MANIFEST_FILENAME).read_text())
    key = manifest["output_key"]
    return records.write_promotion_receipt(staging, {
        "schema_version": records.RECEIPT_SCHEMA, "status": "promoted",
        "staging_manifest_sha256": records.staging_manifest_sha256(staging),
        "staged_objects": {
            "output": {"key_sha256": records.key_sha256(key), "state": "promoted",
                       "versions": [{"size_bytes": 10, "etag": '"spaces-1"'}]},
            "paired_witness": {"key_sha256": records.key_sha256(key + ".witness"), "state": witness,
                               "versions": []}},
        "witness": {"disposition": witness, "reference": None, "redundancy": None},
        "blockers": [], "private_url_recorded": False, "raw_secret_values_recorded": False})


def test_a_promotion_checkpoint_with_its_witness_pending_is_not_final(tmp_path):
    """A proof stands for closeout and billing only while no promotion is mid-run: a receipt
    checkpointed before its witness step (witness pending) means one is running or was cut short."""
    attempt = tmp_path / "attempt_001"
    proof = write_staged_absence_proof(attempt, promotion_status="promoted")
    lane = _lane(attempt, sealed_absent=False)
    assert billing_staged_objects_absent(lane) == (True, proof)

    _receipt(proof.parent, witness="pending")
    waiting = _join_session_closeout(inner=_completed_inner(), adapter=lane, provider_zero=ZERO)
    assert waiting["session_closeout"]["teardown_completed"] is False
    assert NOT_FINAL in waiting["blockers"]
    assert billing_staged_objects_absent(lane) == (False, None)

    _receipt(proof.parent, witness="absent_confirmed")  # the promotion ran to the end
    closed = _join_session_closeout(inner=_completed_inner(), adapter=lane, provider_zero=ZERO)
    assert closed["session_closeout"]["teardown_completed"] is True and NOT_FINAL not in closed["blockers"]
    assert billing_staged_objects_absent(lane) == (True, proof)


def _failing_publisher(**_kwargs):
    raise TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_publication_failed")


@pytest.fixture
def stranded(tmp_path, monkeypatch):
    """A streamed canary whose lane promotion failed: the output stays staged, the lane result is sealed."""
    activation_result, setup_path, _ = _inputs(tmp_path)
    output = tmp_path / "dispatch"
    world = World(output / "allocator" / "attempts", monkeypatch, witness=False)
    archive = quick10_shaped_archive(**SMALL).archive
    world.stage("output", archive)
    observation = {"size_bytes": archive.size, "etag": '"spaces-1"'}
    _write(world.run / "vast_provider_command_result.json", {"provider_output_remote_observation": observation})
    receipt, cleanup = world.promote(observation=observation, publisher=_failing_publisher)
    assert receipt["status"] == "failed" and cleanup["all_objects_absent"] is False

    def fake_allocator(argv):
        adapter_path = Path(argv[argv.index("--adapter-output") + 1])
        provider = _write(world.run / "vast_provider_adapter_result.json",
                          {"vast_instance_ids": [INSTANCE], "continuing_spend_from_this_run": False})
        teardown = _write(world.run / "vast_teardown_manifest.json",
                          {"status": "completed", "vast_instance_ids": [INSTANCE],
                           "continuing_spend_from_this_run": False, "runner_gpu_teardown_completed": True})
        manifest = _write(world.attempt / "artifact_manifest.json", {"status": "blocked"})
        _write(adapter_path, {
            "schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
            "retry_cap": 0, "vast_instance_ids": [INSTANCE], "continuing_spend_from_this_run": False,
            "independent_watchdog": {"status": "provider_terminal", "instance_ids": [INSTANCE],
                                     "provider_absence_confirmed": True},
            "attempt_root": str(world.attempt), "adapter_result_path": str(provider),
            "teardown_manifest_path": str(teardown), "artifact_manifest_path": str(manifest),
            "object_store_cleanup_path": str(world.staging / "wam_provider_object_store_cleanup.json"),
            "native_control_result_path": None, "all_staged_objects_absent": False,
            "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                  "all_staged_objects_absent": False},
            "blockers": ["provider_output_promotion_failed"]})
        return 0

    def post_billing(**kwargs):
        # The real terminal-evidence gate decides whether this run's charge can post.
        try:
            evidence = billing._terminal_evidence(instance_id=INSTANCE,
                                                  terminal_result_path=kwargs["adapter_result_path"])
        except billing.VastOfficialBillingExtractionError:
            return False
        _write(Path(kwargs["output_path"]), {"status": "reconciled_official_posted_charges",
                                             "terminal_execution_evidence": evidence})
        return True

    zero = {**ZERO, "status": "provider_zero_confirmed", "api_confirmed": True, "receipt_digest": ""}
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    monkeypatch.setattr(f"{DISPATCHER}._materialize_official_billing_if_posted", post_billing)
    monkeypatch.setattr(f"{DISPATCHER}.validate_vast_official_same_goal_reconciliation", lambda _path: {})
    monkeypatch.setattr(f"{DISPATCHER}.build_policy_canary_session_bundle", lambda **kwargs: _write(
        Path(kwargs["job_dir"]) / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
        {"bundle_sha256": "sha256:" + "b" * 64}) and {"bundle_sha256": "sha256:" + "b" * 64})
    monkeypatch.setattr(f"{DISPATCHER}.validate_provider_bundle", lambda value, **_kwargs: value)
    monkeypatch.setattr(f"{DISPATCHER}.materialize_policy_canary_result_delivery", lambda **kwargs: {
        "run_id": kwargs["run_id"], "result_status": kwargs["result_status"],
        "delivery_digest": "sha256:" + "d" * 64, "report": {}, "closure": {}})
    monkeypatch.setattr(f"{DISPATCHER}._projection", lambda **_kwargs: {"projection_digest": "sha256:" + "e" * 64})
    monkeypatch.setattr(f"{DISPATCHER}.materialize_policy_canary_website_delivery",
                        lambda *, run_root, delivery: dict(delivery))
    monkeypatch.setattr(
        "blueprint_pipeline.policy_canary_episode_interpretation_closeout.materialize_policy_canary_episode_interpretations",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interpreter unavailable")))

    def dispatch(allocator):
        return dispatch_policy_canary_activation(
            activation_result_path=activation_result, execution_setup_path=setup_path, output_root=output,
            implementation_commit=COMMIT, execute=True, allocator_runner=allocator,
            provider_zero_collector=lambda: zero,
            progress_sync_runner=lambda **_kwargs: {"status": "succeeded", "response": {"status": "recorded"}},
            sync_runner=lambda **_kwargs: {"status": "succeeded", "notification_delivery": {
                "status": "failed", "run_result_digest": "sha256:" + "e" * 64}})

    first = dispatch(fake_allocator)
    assert first["status"] == "awaiting_official_billing" and first["allocator_invoked"] is True
    # Nothing moves while the staged output is still deferred.
    again = dispatch(lambda _argv: pytest.fail("allocator invoked twice"))
    assert again["status"] == "awaiting_official_billing"
    return world, output, dispatch


def test_a_run_whose_promotion_failed_and_was_resumed_closes(stranded):
    world, output, dispatch = stranded

    resumed = world.resume()
    assert resumed["status"] == "completed" and resumed["promotion"]["status"] == "promoted"

    closed = dispatch(lambda _argv: pytest.fail("allocator invoked on resume"))

    assert closed["allocator_invoked"] is False and closed["status"] != "awaiting_official_billing"
    assert (output / "dispatch_receipt.json").is_file()
    terminal = json.loads((output / "policy_canary_terminal_result.json").read_text())
    assert terminal["session_closeout"]["teardown_completed"] is True
    assert "policy_canary_teardown_incomplete" not in terminal["blockers"]
    assert NOT_DURABLE not in terminal["blockers"]  # promoted: the output is durable in B2
    posted = json.loads((output / "official_billing_reconciliation.json").read_text())
    assert posted["terminal_execution_evidence"]["staged_object_absence_proof"]["path"] == str(
        world.staging / "staged_object_absence_proof.v1.json")
    # The sealed lane result was never rewritten.
    assert json.loads((output / "allocator_result.json").read_text())["all_staged_objects_absent"] is False


def test_a_resumed_run_whose_staged_output_was_lost_seals_not_durable(stranded):
    world, output, dispatch = stranded
    world.spaces.stores.pop(world.keys["output"])  # expired before anyone could promote it

    resumed = world.resume()
    assert resumed["promotion"]["status"] == "failed" and resumed["absence_proof"] is not None

    closed = dispatch(lambda _argv: pytest.fail("allocator invoked on resume"))

    assert closed["status"] == "blocked" and (output / "dispatch_receipt.json").is_file()
    terminal = json.loads((output / "policy_canary_terminal_result.json").read_text())
    assert terminal["session_closeout"]["teardown_completed"] is True
    assert NOT_DURABLE in terminal["blockers"]
