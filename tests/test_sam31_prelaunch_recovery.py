"""A real prelaunch producer can enter existing bounded scene recovery."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import sam31_paid_resource_allocator_lane as lane
from blueprint_pipeline.sam31_vast_source_track_canary import (
    PRELAUNCH_INVENTORY_RECEIPT_NAME, Sam31VastCanaryError, run_sam31_vast_source_track_canary,
)
from blueprint_pipeline.task_evaluation_scene_progression_recovery import failure_kind, retain_failure
from blueprint_pipeline.task_evaluation_scene_intake import reserve_scene_attempt
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from tests.test_sam31_paid_resource_allocator_lane import _args, _ReadOnlyProvider, _write_private
from tests.test_sam31_vast_source_track_canary import _bound_request
from tests.test_task_evaluation_scene_recovery import setup, recover, write


def _produce(root, monkeypatch, fault=None):
    root.mkdir()
    args = _args(root, execute=True)
    for path in (args.provider_launch_request, args.sam31_input_bundle_receipt):
        write_json(Path(path), {"fixture": True})
    write_json(Path(args.sam31_attempt_authority), {"request_authority_id": "fixture-authority"})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    _write_private(Path(args.sam31_hf_token_file), "fixture-secret")
    consumption = {"status": "consumed", "authorization_digest": "sha256:" + "a" * 64}
    monkeypatch.setattr(lane, "validate_sam31_paid_attempt_authority", lambda *a, **kw: {})
    def consume(*a, **kw):
        assert not (root / "consumption.json").exists()
        write_json(root / "consumption.json", consumption)
        return consumption
    monkeypatch.setattr(lane, "consume_sam31_paid_attempt_authority_once", consume)

    class Provider(_ReadOnlyProvider):
        name = "vast"
        canary = False
        reads = 0
        def billable_inventory(self, *, name_prefix):
            if not self.canary:
                return super().billable_inventory(name_prefix=name_prefix)
            self.reads += 1
            if self.reads == 2:
                return {"api_confirmed": False, "live_resource_count": None, "resources": [], "http": 429}
            if fault == "nonzero_before" and self.reads == 1 or fault == "nonzero_after" and self.reads > 2:
                return {"api_confirmed": True, "live_resource_count": 1, "resources": [{"id": 42}]}
            if fault == "unconfirmed_after" and self.reads > 2:
                return {"api_confirmed": False, "live_resource_count": None, "resources": []}
            return super().billable_inventory(name_prefix=name_prefix)
        def build_request(self, *a, **kw):
            pytest.fail("prelaunch failure must not invoke the provider adapter")
        def launch(self, *a, **kw):
            pytest.fail("prelaunch failure must not create a provider instance")
    provider = Provider()

    def execute(**kwargs):
        provider.canary = True
        try:
            return run_sam31_vast_source_track_canary(**kwargs, clock=lambda: 1000., watchdog_validator=lambda *a: True)
        except Sam31VastCanaryError:
            path = Path(kwargs["job_dir"]) / PRELAUNCH_INVENTORY_RECEIPT_NAME
            if not path.is_file():
                raise
            value = json.loads(path.read_text())
            if fault == "legacy":
                value.pop("provider_launch_invoked")
            if fault in {"wrong_binding", "tamper"}:
                value["bound_request_digest"] = "sha256:" + "f" * 64
            if fault != "tamper":
                value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
            write_json(path, value)
            if fault == "launch_marker":
                write_json(Path(kwargs["job_dir"]) / "pending_teardowns/ambiguous.json", {"status": "open"})
            raise

    def prepare(**kwargs):
        bound = _bound_request()
        bound["bound_preflight_digest"] = canonical_digest(json.loads(Path(kwargs["preflight_path"]).read_text()))
        bound["bound_request_digest"] = canonical_digest(bound, digest_field="bound_request_digest")
        write_json(Path(kwargs["bound_request_out"]), bound)
        return {"status": "execute_ready", "blockers": []}
    def stage(**kwargs):
        output = Path(kwargs["job_dir"])
        output.mkdir()
        for name in ("provider_bundle_url.txt", "provider_output_put_url.txt", "provider_output_get_url.txt"):
            _write_private(output / name, "https://objects.example/fixture")
        return {"status": "completed", "blockers": []}
    def cleanup(output):
        result = {"all_objects_absent": fault != "cleanup"}
        write_json(output / "wam_provider_object_store_cleanup.json", result)
        return result
    def close(**kwargs):
        result = {"status": "armed" if fault == "watchdog" else "cancelled_no_allocation"}
        write_json(root / "independent_vast_watchdog/groot_oscar_runpod_canary_watchdog.json", result)
        return result
    result = lane.run_sam31_paid_resource_allocator_lane(args, checkout_commit="c" * 40,
        prepare=prepare, provider_factory=lambda _: provider, execute_canary=execute,
        stage_bundle=stage, cleanup_bundle=cleanup, close_watchdog=close,
        arm_watchdog=lambda **kw: ({"watchdog_pid": 123, "watchdog_started_epoch": 1000,
            "watchdog_deadline_epoch": 9999999999, "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-"},
            SimpleNamespace(started_instance_id_path=root / "started-id")))
    assert result["authorization_consumption"] == consumption
    assert json.loads((root / "consumption.json").read_text()) == consumption
    assert provider.reads == 4, result.get("blockers")
    return result, Path(args.adapter_output)


@pytest.mark.parametrize("fault", ["nonzero_before", "nonzero_after", "unconfirmed_after", "cleanup", "watchdog",
                                   "launch_marker", "legacy", "wrong_binding", "tamper"])
def test_uncertain_or_historical_outcome_does_not_gain_recovery_evidence(tmp_path, monkeypatch, fault):
    result, _ = _produce(tmp_path / "producer", monkeypatch, fault)
    assert "allocation_created" not in result
    assert failure_kind(result) is None


@pytest.mark.parametrize("field,changed", [("allocation_outcome_ambiguous", True),
    ("provider_mutation_outcome_ambiguous", True), ("provider_mutations_performed", 1), ("instance_id", "42")])
def test_contradictory_launch_evidence_refuses_even_with_retained_zero(tmp_path, monkeypatch, field, changed):
    from blueprint_pipeline.sam31_prelaunch_recovery import proven_prelaunch_inventory_throttle
    root = tmp_path / "producer"
    result, _ = _produce(root, monkeypatch)
    result[field] = changed
    assert not proven_prelaunch_inventory_throttle(
        receipt=json.loads((root / "sam31_vast_source_track_canary" / PRELAUNCH_INVENTORY_RECEIPT_NAME).read_text()),
        bound_request=json.loads((root / "bound.json").read_text()), result=result,
        cleanup={"all_objects_absent": True}, watchdog={"status": "cancelled_no_allocation"}, launch_evidence_present=False)


@pytest.mark.parametrize("limit", [None, "retry", "spend"])
def test_proven_429_recovery_uses_existing_attempt_caps_and_preserves_failure(tmp_path, monkeypatch, limit):
    intent, first, evidence = setup(tmp_path, retries=0 if limit == "retry" else 1)
    result, producer_path = _produce(tmp_path / "producer", monkeypatch)
    assert result["allocation_created"] is False
    assert result["provider_mutation_outcome_ambiguous"] is False
    assert result["allocation_failure_phase"] == "prelaunch_inventory_read"
    assert failure_kind(result) == "create_refused"
    parent = "sha256:" + "b" * 64
    key = {"parent_request_digest": parent, "plan_digest": "sha256:" + "c" * 64,
           "phase": "sam31_tracking", "inputs_digest": "sha256:" + "d" * 64}
    child = "sam31-" + canonical_digest(key)[7:]
    queue = tmp_path / "child-queue"
    (queue / "failed").mkdir(parents=True)
    (queue / "results").mkdir()
    job = {**key, "expected_source_commit": first["source_commit"], "parent_preparation_id": "fixture-parent", "child_id": child}
    write(queue / "failed" / (child + ".json"), job, "job_digest")
    write(queue / "results" / (child + ".json"), {"job_digest": job["job_digest"], "child_id": child,
        "status": "failed", "artifacts": {"sam31_allocator_result": record(producer_path)}}, "result_digest")
    failure = retain_failure(attempt=first, link={"request_digest": parent, "preparation_id": "fixture-parent"},
        child_queue_root=queue, output_root=tmp_path / "retained-failure", now=102)
    assert failure is not None
    evidence["failure"] = record(failure)
    old_files = [producer_path, failure, tmp_path / intent["intent_id"] / "attempts/a1.json",
                 tmp_path / "producer/consumption.json", queue / "failed" / (child + ".json")]
    before = {p: p.read_bytes() for p in old_files}
    if limit == "spend":
        with pytest.raises(ValueError, match="spend_cap_exhausted"):
            reserve_scene_attempt(queue_root=tmp_path, intent_id=intent["intent_id"], attempt_id="a2",
                source_commit="c" * 40, runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                provider="vast", maximum_spend_usd=20, now=104, recovery_from_attempt_id="a1", recovery_evidence=evidence)
    elif limit == "retry":
        with pytest.raises(ValueError, match="retry_cap_exhausted"):
            recover(tmp_path, intent, evidence)
    else:
        successor = recover(tmp_path, intent, evidence)
        assert successor["recovery"]["prior_attempt_digest"] == first["attempt_digest"]
        assert successor["maximum_spend_usd"] == 2
        assert recover(tmp_path, intent, evidence) == successor
    assert all(p.read_bytes() == raw for p, raw in before.items())
