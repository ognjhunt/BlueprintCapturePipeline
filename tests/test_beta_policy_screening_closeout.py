"""Fresh own-allocation absence permits other sites to keep running."""
from __future__ import annotations

from copy import deepcopy
import json
import time

import pytest

from blueprint_pipeline import beta_policy_screening_closeout as closeout
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PROVIDER_BUNDLE_SCHEMA_VERSION, PROVIDER_RESULT_FILENAME,
    consume_session_authority_once, validate_session_authority,
)
from tests.test_beta_policy_screening_coordinator import native_result, plan, seal
from tests.test_native_task_arena_policy_canary_session import _authority, _record


@pytest.fixture
def attempt(tmp_path):
    value = plan()
    inner = native_result(value)
    authority, _ = _authority(tmp_path)
    authority.update(run_kind=BETA_RUN_KIND, candidate_ids=inner["candidate_ids"], learned_policy_rollout_count=10,
                     beta_screening_admission=inner["beta_screening_admission"],
                     **{key: inner[key] for key in ("plan_digest", "configuration_digest", "scene_revision_digest")})
    seal(authority, "authority_digest")
    validate_session_authority(authority)
    def record(name, content):
        path = tmp_path / name
        path.write_text(json.dumps(content))
        return _record(path)
    bundle = {"schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "status": "ready",
              "execution_mode": "blueprint_beta_screening_session", "run_kind": BETA_RUN_KIND,
              "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": inner["candidate_ids"],
              "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "maximum_provider_allocations": 1,
              "retry_cap": 0, "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
              **{key: authority[key] for key in ("runtime_inputs_digest", "authority_digest", "task_success_contract_digest")},
              "bundle_path": str(tmp_path / "bundle.zip"), "bundle_size_bytes": 1, "bundle_sha256": "sha256:" + "f" * 64}
    prepared = {"authority": record("authority.json", authority), "bundle_receipt": record("bundle.json", bundle),
                "candidate_id": inner["candidate_ids"][0], "run_id": authority["run_id"],
                "plan_digest": inner["plan_digest"], "configuration_digest": inner["configuration_digest"]}
    seal(prepared, "prepared_digest")
    adapter = {"vast_instance_ids": [7001], "bundle_sha256": bundle["bundle_sha256"],
               "authorization_consumption": consume_session_authority_once(authority, consumption_path=tmp_path / "consume.json"),
               "continuing_spend_from_this_run": False, "native_control_result_digest": inner["result_digest"],
               "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                     "all_staged_objects_absent": True,
                                     "adapter_result": record("raw_adapter.json", {"vast_instance_ids": [7001]}),
                                     "teardown_manifest": record("teardown.json", {"vast_instance_ids": [7001]})}}
    return {"adapter": adapter, "prepared": prepared, "adapter_path": tmp_path / "allocator_result.json"}, inner, authority


class Provider:
    def __init__(self, resources=None, *, change=None):
        self.resources = resources if resources is not None else [{"instance_id": "8002", "name": "unrelated-site"}]
        self.change = change
        self.reads = []

    def inspect(self, instance_id):
        self.reads.append(("inspect", instance_id))
        return {"status": "observed" if self.change == "inspect_live" else "absent", "provider": "vast",
                "instance_id": instance_id, "api_confirmed": self.change != "inspect_unconfirmed",
                "provider_absence_confirmed": self.change != "inspect_live"}

    def billable_inventory(self, *, name_prefix):
        self.reads.append(("inventory", name_prefix))
        value = {"status": "observed", "provider": "vast", "name_prefix": name_prefix,
                 "api_confirmed": True, "observed_at_epoch": time.time(),
                 "resources": self.resources, "live_resource_count": len(self.resources)}
        if self.change == "stale":
            value["observed_at_epoch"] -= 200
        elif self.change == "incomplete":
            value["live_resource_count"] += 1
        elif self.change == "scoped_inventory":
            value["name_prefix"] = "other-site"
        elif self.change == "unconfirmed":
            value["api_confirmed"] = False
        elif self.change == "missing_inventory":
            return None
        return value


def test_unrelated_live_site_does_not_block_exact_attempt_closure(attempt):
    args, inner, _ = attempt
    provider = Provider()
    receipt = closeout.collect_scoped_closeout(**args, provider=provider)
    assert closeout.scoped_closeout_verified(receipt, **args)
    assert receipt["unrelated_live_resources"] == provider.resources
    assert receipt["global_provider_zero_claimed"] is False
    assert provider.reads == [("inspect", "7001"), ("inventory", "")]
    result = closeout.join_scoped_session_closeout(inner=inner, receipt=receipt, **args)
    assert result["status"] == "completed_unqualified"
    assert result["session_closeout"]["provider_zero_scope"] == closeout.SCOPE
    assert result["session_closeout"]["global_provider_zero_claimed"] is False
    assert result["episodes"] == inner["episodes"]


@pytest.mark.parametrize("identity", ["instance_id", "resource_name"])
def test_owned_id_or_label_still_live_blocks(attempt, identity):
    args, _inner, authority = attempt
    row = {"instance_id": "8002", "name": "unrelated-site"}
    row["instance_id" if identity == "instance_id" else "name"] = (
        "7001" if identity == "instance_id" else authority["resource_name"])
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider([row]))
    assert not closeout.scoped_closeout_verified(receipt, **args)
    assert "beta_screening_owned_resource_still_live" in receipt["blockers"]


@pytest.mark.parametrize("change", ["stale", "incomplete", "scoped_inventory", "unconfirmed",
                                    "missing_inventory", "inspect_live", "inspect_unconfirmed"])
def test_unknown_stale_or_narrow_inventory_cannot_close(attempt, change):
    args, _inner, _authority = attempt
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider(change=change))
    assert receipt["status"] == "blocked"
    assert not closeout.scoped_closeout_verified(receipt, **args)


@pytest.mark.parametrize("change", ["allocation", "consumption", "bundle", "authority"])
def test_other_attempt_binding_refuses_before_provider_reads(attempt, change):
    args, _inner, _authority = attempt
    provider = Provider()
    if change == "allocation":
        args["adapter"]["vast_instance_ids"] = [8002]
    elif change == "consumption":
        args["adapter"]["authorization_consumption"]["authority_digest"] = "sha256:" + "0" * 64
        seal(args["adapter"]["authorization_consumption"], "consumption_digest")
    elif change == "bundle":
        args["adapter"]["bundle_sha256"] = "sha256:" + "0" * 64
    else:
        args["prepared"]["candidate_id"] = "other"
        seal(args["prepared"], "prepared_digest")
    with pytest.raises(ValueError):
        closeout.collect_scoped_closeout(**args, provider=provider)
    assert provider.reads == []


@pytest.mark.parametrize("change", ["teardown", "staged", "retained", "spend"])
def test_fresh_absence_does_not_replace_own_teardown_proofs(attempt, change):
    args, _inner, _authority = attempt
    adapter = args["adapter"]
    if change == "teardown":
        adapter["provider_closeout"]["provider_zero_confirmed"] = False
    elif change == "staged":
        adapter["provider_closeout"]["all_staged_objects_absent"] = False
    elif change == "retained":
        adapter["provider_closeout"]["warm_session_retained"] = True
    else:
        adapter["continuing_spend_from_this_run"] = True
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider())
    assert not closeout.scoped_closeout_verified(receipt, **args)


def test_receipt_expiry_scope_mutation_and_wrong_native_result_refuse(attempt, monkeypatch):
    args, inner, _authority = attempt
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider())
    wrong = deepcopy(inner)
    wrong["candidate_ids"] = ["another-policy"]
    seal(wrong, "result_digest")
    with pytest.raises(ValueError, match="native_result_binding"):
        closeout.join_scoped_session_closeout(inner=wrong, receipt=receipt, **args)
    altered = {**receipt, "scope": "global"}
    seal(altered, "receipt_digest")
    assert not closeout.scoped_closeout_verified(altered, **args)
    monkeypatch.setattr(closeout.time, "time", lambda: receipt["observed_at_epoch"] + 121)
    assert not closeout.scoped_closeout_verified(receipt, **args)


def test_absent_staging_with_lost_output_closes_billing_but_cannot_publish_score(attempt, monkeypatch):
    args, inner, _authority = attempt
    monkeypatch.setattr(closeout, "closeout_staged_objects", lambda _: {
        "absent": True, "blockers": ["policy_canary_provider_output_not_durable"],
    })
    receipt = closeout.collect_scoped_closeout(**args, provider=Provider())
    assert closeout.scoped_closeout_verified(receipt, **args)
    with pytest.raises(ValueError, match="scientific_output_not_durable"):
        closeout.join_scoped_session_closeout(inner=inner, receipt=receipt, **args)
