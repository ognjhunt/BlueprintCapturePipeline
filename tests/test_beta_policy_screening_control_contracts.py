"""Exact WebApp current-source control contract with fake signed transport."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from blueprint_pipeline import beta_policy_screening_control_contracts as contract
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def binding():
    return {"campaign_id": "campaign-1", "site_id": "site-one", "plan_digest": "sha256:" + "a" * 64,
        "candidate_id": "pi05_droid", "configuration_digest": "sha256:" + "b" * 64,
        "attempt_id": "screen-one", "invocation_id": "sha256:" + "c" * 64,
        "prepared_digest": "sha256:" + "d" * 64, "authority_digest": "sha256:" + "e" * 64,
        "resource_name": "blueprint-exact-one", "source": {"scene_intent_id": "intent-one",
            "scene_intent_digest": "sha256:" + "f" * 64, "capture_id": "capture-one", "task_id": "task-one",
            "task_context_digest": "sha256:" + "1" * 64, "scene_revision_digest": "sha256:" + "2" * 64}}


def response(*, scope=None, now=1000, stopped=False, deadline=2000, revision=1):
    scope = scope or binding()
    value = {"schema_version": "blueprint.beta_policy_screening_execution_control.v1", "binding": scope,
        "binding_digest": cross_runtime_canonical_digest(scope), "revision": revision,
        "status": "stopped" if stopped else "current", "cancellation_requested": stopped,
        "stop_reason": "screening_run_cancelled" if stopped else None, "execution_deadline_at_ms": deadline * 1000,
        "observed_at_iso": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"),
        "next_check_at_ms": (now + 60) * 1000, "new_execution_authorized": False,
        "read_only_closeout_allowed": True, "applicability": "historical_only" if stopped else "current"}
    return {**value, "control_digest": cross_runtime_canonical_digest(value)}


@pytest.mark.parametrize("stopped", [False, True])
def test_exact_control_echoes_scope_and_carries_no_new_authority(stopped):
    value = response(stopped=stopped)
    assert contract.validate_control(value, binding=binding(), now=1001) == value
    if stopped:
        assert contract.validate_control(value, binding=binding(), now=100_000) == value


@pytest.mark.parametrize("field", ["binding", "binding_digest", "control_digest", "status", "revision",
    "cancellation_requested", "stop_reason", "execution_deadline_at_ms", "observed_at_iso", "next_check_at_ms",
    "new_execution_authorized", "read_only_closeout_allowed", "applicability"])
def test_malformed_resealed_response_never_extends_current_processing(field):
    value = deepcopy(response())
    value[field] = {"other": "site"} if field == "binding" else True if field.endswith("authorized") else "unexpected" if field == "stop_reason" else None
    if field != "control_digest":
        value["control_digest"] = cross_runtime_canonical_digest(value, digest_field="control_digest")
    with pytest.raises(ValueError, match="execution_control_invalid"):
        contract.validate_control(value, binding=binding(), now=1001)


def test_stale_current_control_cannot_mint_freshness():
    with pytest.raises(ValueError, match="execution_control_invalid"):
        contract.validate_control(response(), binding=binding(), now=1121)
