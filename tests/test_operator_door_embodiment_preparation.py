"""Explicit money authority survives the API, privileged spool and fixed enqueue seam."""

from __future__ import annotations

import copy
import configparser
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))

from operator_door.embodiment_preparation import PUBLIC_EMBODIMENT_IDS  # noqa: E402
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult  # noqa: E402
from operator_door.fsview import FileView, FsRefused  # noqa: E402
from operator_door.requests import RequestRefused, enqueue, required_scope, validate_request  # noqa: E402
from operator_door.spool_runner import process_spool  # noqa: E402
from blueprint_pipeline.decision_evidence_contracts import canonical_digest  # noqa: E402
from blueprint_pipeline.deployment_embodiment_assets import asset_recipes  # noqa: E402
from blueprint_pipeline.deployment_embodiment_preparation_contract import validate_request as worker_validate  # noqa: E402
from blueprint_pipeline.deployment_embodiment_runtime_configuration import runtime_profiles  # noqa: E402


def grant(**changes):
    value = {"schema_version": "embodiment_asset_preparation_request.v1", "authorization_id": "development-only-fixture",
             "release_commit": "a" * 40, "embodiment_ids": ["universal_robots_ur5e", "fanuc_crx_10ia"],
             "max_spend_usd": 4.0, "max_hourly_rate_usd": .80, "ttl_seconds": 14_400,
             "expires_at_epoch": int(time.time()) + 86400, "authorization_reference": "development_only fixture"}
    value.update(changes)
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def body(value=None):
    return {"kind": "public-embodiment-preparation", "request": grant() if value is None else value}


def config(tmp_path):
    result = DoorConfig(state_root=str(tmp_path))
    for name in ("pending", "processing", "completed", "results"):
        (Path(result.spool_root) / name).mkdir(parents=True)
    return result


def test_door_allowlist_and_sealed_bytes_match_actual_worker_contract():
    allowed = ({row["embodiment_id"] for row in runtime_profiles()["profiles"]}
               & {row["embodiment_id"] for row in asset_recipes()["recipes"] if row["status"] == "source_available"})
    assert PUBLIC_EMBODIMENT_IDS == allowed
    value = grant(embodiment_ids=list(reversed(sorted(allowed))), authorization_reference="development_only café fixture")
    assert validate_request(body(value)) == body(worker_validate(value))
    assert required_scope("public-embodiment-preparation") == "operate"


@pytest.mark.parametrize("changes", [
    {"authorization_id": "../../grant"}, {"authorization_id": 123}, {"release_commit": "a" * 39},
    {"release_commit": "A" * 40}, {"embodiment_ids": ["boston_dynamics_atlas"]},
    {"embodiment_ids": ["universal_robots_ur5e", "universal_robots_ur5e"]}, {"embodiment_ids": []},
    {"embodiment_ids": [{"path": "/tmp/model.usd"}]}, {"max_spend_usd": 4.01}, {"max_spend_usd": True},
    {"max_spend_usd": 1}, {"max_hourly_rate_usd": .81}, {"max_hourly_rate_usd": 0},
    {"max_hourly_rate_usd": float("nan")}, {"max_spend_usd": float("inf")}, {"max_spend_usd": 10**400},
    {"ttl_seconds": 14_401}, {"ttl_seconds": 1799}, {"ttl_seconds": 1800.0}, {"ttl_seconds": True},
    {"expires_at_epoch": 1}, {"expires_at_epoch": time.time() + 8 * 86400},
    {"authorization_reference": "fixture\nnew command"}, {"authorization_reference": ""},
    {"command": "run-gpu"}, {"image": "arbitrary"}, {"url": "https://example.com/asset"},
    {"state_root": "/tmp"}, {"runtime_path": "/tmp/runtime"},
])
def test_rejects_unbounded_authority_and_caller_controlled_execution(changes):
    value = grant(**changes)
    with pytest.raises(RequestRefused):
        validate_request(body(value))
    with pytest.raises(ValueError):
        worker_validate(value)


def test_digest_binds_order_and_all_fields_and_outer_envelope_has_no_overrides():
    value = grant()
    value["embodiment_ids"].reverse()
    with pytest.raises(RequestRefused, match="request_digest_invalid"):
        validate_request(body(value))
    with pytest.raises(RequestRefused, match="request_key_unknown"):
        validate_request({**body(), "command": "run-queue"})
    with pytest.raises(RequestRefused, match="request_schema_invalid"):
        validate_request(body([]))


def test_idempotent_retries_keep_authority_and_tampering_never_launches(tmp_path):
    settings = config(tmp_path)
    value = body()
    key = "public-embodiment-preparation-" + value["request"]["authorization_id"]
    first = enqueue(settings, value, requested_by="development-only", operation_key=key)
    assert enqueue(settings, copy.deepcopy(value), requested_by="development-only", operation_key=key) == first
    changed = body(grant(embodiment_ids=["fanuc_crx_10ia"]))
    with pytest.raises(RequestRefused, match="operation_key_conflict"):
        enqueue(settings, changed, requested_by="development-only", operation_key=key)
    pending = Path(settings.spool_root) / "pending" / (first + ".json")
    record = json.loads(pending.read_text())
    record["request"]["request"]["max_spend_usd"] = 400
    pending.write_text(json.dumps(record))
    calls = []
    runner = SimpleNamespace(run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", ""))
    process_spool(settings, runner=runner)
    assert not calls
    result = json.loads((Path(settings.spool_root) / "results" / (first + ".json")).read_text())
    assert result["status"] == "refused"


def test_privileged_runner_launches_networkless_fixed_enqueue_only(tmp_path):
    settings = config(tmp_path)
    value = body(grant(authorization_reference="development_only $(touch /tmp/no) `false` %n fixture"))
    enqueue(settings, value, requested_by="development-only")
    calls = []
    runner = SimpleNamespace(run=lambda argv, timeout: calls.append(argv) or CommandResult(0, "", ""))
    process_spool(settings, runner=runner)
    launch, = calls
    assert launch[0] == "systemd-run"
    assert launch[-2:] == ["/bin/bash", settings.install_root + "/door-public-embodiment-preparation.sh"]
    assert "--property=PrivateNetwork=yes" in launch
    assert "--property=RuntimeMaxSec=2min" in launch
    assert "--property=ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/embodiment-preparation " + str(
        Path(settings.spool_root) / "results") in launch
    env = dict(item.removeprefix("--setenv=").split("=", 1) for item in launch if item.startswith("--setenv="))
    assert env["DOOR_SERVICE_USER"] == "blueprint"
    assert json.loads(env["DOOR_EMBODIMENT_PREPARATION_REQUEST"]) == value["request"]
    assert not any("KEY" in key or "TOKEN" in key for key in env)


def test_existing_door_policy_reads_preparation_receipts_and_model_store_without_new_roots():
    from blueprint_pipeline.deployment_embodiment_preparation_contract import MODEL_ROOT, STATE_ROOT
    view = FileView(DoorConfig())
    paths = [STATE_ROOT / "authorizations/batch-20261029.json", STATE_ROOT / "pending/batch-20261029.json",
             STATE_ROOT / "latest.json", STATE_ROOT / "jobs/batch-20261029/allocator_result.json",
             STATE_ROOT / "jobs/batch-20261029/publication.json",
             STATE_ROOT / "jobs/batch-20261029/bundle/embodiment_asset_preparation_bundle.v1.json",
             MODEL_ROOT / "batches/batch-20261029.json",
             MODEL_ROOT / "fanuc_crx_10ia" / ("a" * 64) / "binding-inputs.json",
             MODEL_ROOT / "fanuc_crx_10ia" / ("a" * 64) / "body-result.json"]
    for path in paths:
        assert view._check_readable(str(path)) == str(path)
    for path in ("/etc/blueprint/provider-secrets/vast_api_key", "/var/lib/blueprint/spend-authority/grants.json",
                 str(STATE_ROOT / "jobs/batch-20261029/provider-token.json")):
        with pytest.raises(FsRefused):
            view._check_readable(path)
    # Both processes have the owner identity that can read private grant/output
    # files. No group/world readability or secret-root exception is necessary.
    for unit in ("blueprint-operator-door.service", "blueprint-embodiment-preparation.service"):
        parser = configparser.RawConfigParser(strict=False, interpolation=None)
        parser.read(ROOT / "deploy/systemd" / unit)
        assert parser["Service"]["User"] == "blueprint"
        assert parser["Service"]["UMask"] == "0077"


def test_native_private_grant_and_retained_receipts_are_readable_with_secret_guards(tmp_path):
    from blueprint_pipeline.deployment_embodiment_preparation_controller import enqueue as queue_grant
    state = tmp_path / "pipeline-control-plane/embodiment-preparation"
    request = grant()
    queued = queue_grant(request, root=state, commit=request["release_commit"])
    assert queued["provider_allocations_performed"] == 0
    view = FileView(DoorConfig(read_roots=(str(tmp_path),)))
    actual = state / "authorizations" / (request["authorization_id"] + ".json")
    actual.chmod(0o600)
    raw, _ = view.read_range(str(actual))
    assert json.loads(raw) == request
    summary = {"status": "completed", "claim_ceiling": "development_only", "results": [
        {"embodiment_id": "fanuc_crx_10ia", "status": "retained", "model_sha256": "sha256:" + "a" * 64}],
        "request_digest": request["request_digest"], "candidate_policy_queried": False, "task_executed": False}
    for path in (state / "latest.json", state / "jobs/batch-20261029/publication.json",
                 tmp_path / "robot-models/batches/batch-20261029.json"):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(summary))
        path.chmod(0o600)
        raw, _ = view.read_range(str(path))
        assert json.loads(raw) == summary
    contaminated = state / "jobs/batch-20261029/allocator_result.json"
    contaminated.write_text(json.dumps({"api_key": "development_only_credential_value"}))
    with pytest.raises(FsRefused, match="secret_content_refused"):
        view.read_range(str(contaminated))
