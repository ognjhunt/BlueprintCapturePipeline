"""Terminal finalization is hermetic; only transport/readback edges are faked."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import operator_policy_canary_terminal_delivery as terminal
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import build_session_authority
from blueprint_pipeline.task_evaluation_run_webapp_sync import (
    sync_task_evaluation_policy_canary_to_webapp,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def test_intent_preserves_remote_paths_without_resolving_author_host_symlinks(monkeypatch):
    def forbidden_resolution(*args, **kwargs):
        raise AssertionError('A remote control-plane path cannot resolve on the author host')

    monkeypatch.setattr(Path, 'resolve', forbidden_resolution)
    paths = {
        name: '/var/lib/blueprint/canary/' + name
        for name in ('run_root', 'allocator_result_path', 'official_billing_path',
                     'provider_zero_path', 'billing_audit_root', 'native_result_path')
    }
    intent = terminal.operator_terminal_delivery_intent(run_id='cross-host', records={}, **paths)
    assert {name: intent[name] for name in paths} == paths
    assert intent['intent_digest'] == canonical_digest(intent, digest_field='intent_digest')


@pytest.mark.parametrize('invalid', ['relative/run', '/var/../tmp/run', '/var/./run',
                                    '//server/run', '/var/run\x00', 'C:\\run'])
def test_intent_rejects_nonabsolute_or_traversing_control_plane_paths(invalid):
    paths = {
        name: '/var/lib/blueprint/canary/' + name
        for name in ('run_root', 'allocator_result_path', 'official_billing_path',
                     'provider_zero_path', 'billing_audit_root', 'native_result_path')
    }
    for name in paths:
        with pytest.raises(terminal.OperatorTerminalDeliveryError, match='control_plane_path_invalid'):
            terminal.operator_terminal_delivery_intent(
                run_id='cross-host', records={}, **{**paths, name: invalid})


def _fixture(tmp_path, monkeypatch, *, partial=False):
    from tests.test_task_evaluation_policy_canary_dispatcher import _inputs

    source = tmp_path / "source"
    _, setup_path, activation_path = _inputs(source)
    runtime_path = activation_path.parent / "task_evaluation_policy_canary_runtime_inputs.v1.json"
    setup, runtime, activation = [
        json.loads(p.read_text()) for p in (setup_path, runtime_path, activation_path)
    ]
    setup.update(run_kind="internal_policy_canary", claim_ceiling="diagnostic_policy_execution")
    setup["runtime_inputs"] = {
        "nested": {
            "scene_plan": terminal.file_record(
                _write(source / "nested-scene-plan.json", {"fixture": True})
            )
        }
    }
    for role, record in setup["records"].items():
        if role.endswith("execution_spec"):
            path = Path(record["path"])
            value = json.loads(path.read_text())
            value.update(
                checkpoint_digest="sha256:" + "a" * 64, runtime_identity_digest="sha256:" + "b" * 64
            )
            _write(path, value)
            setup["records"][role] = terminal.file_record(path)
    setup["setup_digest"] = canonical_digest(setup, digest_field="setup_digest")
    _write(setup_path, setup)
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=terminal.file_record(activation_path),
        runtime_inputs=runtime,
        runtime_input_record=terminal.file_record(runtime_path),
        resource_name=runtime["resource_authority"]["resource_name"],
        hard_cap_usd=4.0,
        hard_ttl_seconds=14400,
    )
    auth = {
        "schema_version": "task_evaluation_operator_policy_authorization.v1",
        "run_id": runtime["run_id"],
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
    }
    auth["authorization_digest"] = canonical_digest(auth, digest_field="authorization_digest")
    registration = {
        "schema_version": "task_evaluation_operator_policy_canary_registration.v1",
        "run_id": runtime["run_id"],
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "team_namespace": "fixture-team",
        **{
            k: setup[k]
            for k in ("capture_session_id", "intake_id", "request_digest", "setup_digest")
        },
        **{
            k: runtime[k]
            for k in (
                "runtime_inputs_digest",
                "configuration_digest",
                "plan_digest",
                "task_success_contract",
            )
        },
        "operator_authorization_digest": auth["authorization_digest"],
    }
    registration["registration_digest"] = cross_runtime_canonical_digest(
        registration, digest_field="registration_digest"
    )
    ack = {
        "schema_version": "task_evaluation_operator_policy_canary_registration_receipt.v1",
        "status": "registered",
        "run_id": runtime["run_id"],
        "registration_digest": registration["registration_digest"],
    }
    bundle = {
        "schema_version": "native_task_arena_policy_canary_provider_bundle.v1",
        "status": "ready",
        "execution_mode": "internal_policy_canary_paired_session",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": authority["candidate_ids"],
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_policy_canary_session_result.v1.json",
        "runtime_inputs_digest": runtime["runtime_inputs_digest"],
        "authority_digest": authority["authority_digest"],
        "task_success_contract_digest": runtime["task_success_contract_digest"],
        "execution_release": authority.get("execution_release"),
        "bundle_path": str(source / "bundle.tar"),
        "bundle_size_bytes": 1,
        "bundle_sha256": "sha256:" + "c" * 64,
    }
    native = source / "native" / "native_task_arena_policy_canary_session_result.v1.json"
    gap = _write(native.parent / "gap.json", {"type": "before_first_observation"})
    episodes = [
        {
            "candidate_id": candidate,
            "cell_id": cell["cell_id"],
            "seed": cell["seed"],
            "status": "blocked",
            "candidate_policy_queried": False,
            "actions_reached_robot": False,
            "arm_moved": False,
            "policy_outcome_interpretable": False,
            "typed_harness_failure": "before_first_observation",
            "checkpoint_digest": "sha256:" + "a" * 64,
            "runtime_identity_digest": "sha256:" + "b" * 64,
            "reset_state_digest": canonical_digest({"cell": cell["cell_id"], "seed": cell["seed"]}),
            "visual_evidence": {
                "media_gap": {"type": "before_first_observation", "reason": "fixture"}
            },
            "evidence_artifacts": {},
        }
        for candidate in authority["candidate_ids"]
        for cell in runtime["cells"]
    ]
    raw = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked" if partial else "runtime_completed_unqualified_pending_closeout",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": authority["candidate_ids"],
        "episodes": [] if partial else episodes,
        "artifact_inventory": [
            {
                "role": "typed_media_gap",
                "relative_path": "gap.json",
                **{k: terminal.file_record(gap)[k] for k in ("sha256", "size_bytes")},
            }
        ],
        "task_success_contract": runtime["task_success_contract"],
        "task_success_contract_digest": runtime["task_success_contract_digest"],
        "blockers": ["fixture_harness_failure"],
    }
    raw["result_digest"] = canonical_digest(raw, digest_field="result_digest")
    _write(native, raw)
    if partial:
        child_path = native.parent / "cell_runs/00" / native.name
        child = dict(
            raw, status="blocked", selected_cell_index=0, episodes=[], artifact_inventory=[]
        )
        child_file = _write(child_path.parent / "failure.json", {"type": "retained_native_failure"})
        child["artifact_inventory"] = [
            {
                "role": "runtime_supporting_evidence",
                "relative_path": child_file.name,
                **{k: terminal.file_record(child_file)[k] for k in ("sha256", "size_bytes")},
            }
        ]
        child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        _write(child_path, child)
    teardown = _write(
        source / "teardown.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "runner_gpu_teardown_completed": True,
            "continuing_spend_from_this_run": False,
            "vast_instance_ids": [1234],
        },
    )
    cleanup = _write(
        source / "cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "all_ephemeral_objects_absent": True,
            "blockers": [],
            "exact_object_count": 1,
            "objects": [{"absence": {"absence_confirmed": True}}],
        },
    )
    adapter = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "blocked",
        "vast_instance_ids": [1234],
        "bundle_sha256": bundle["bundle_sha256"],
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "native_control_result_path": str(native),
        "native_control_result_digest": raw["result_digest"],
        "object_store_cleanup_path": str(cleanup),
        "provider_closeout": {
            "teardown_manifest": terminal.file_record(teardown),
            "provider_zero_confirmed": True,
            "warm_session_retained": False,
            "all_staged_objects_absent": True,
        },
    }
    adapter_path = _write(source / "allocator_result.json", adapter)
    zero = {
        "schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "api_confirmed": True,
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "blockers": [],
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    zero_path = _write(source / "provider-zero.json", zero)
    billing = {
        "official_total_usd": 0.5,
        "entries": [
            {
                "provider_instance_id": 1234,
                "launch_label": authority["resource_name"],
                "official_charge_usd": 0.5,
                "terminal_execution_evidence": {
                    "terminal_result": terminal.file_record(adapter_path)
                },
            }
        ],
    }
    billing_path = _write(source / "billing.json", billing)
    # Synthetic tests inject only the official-source read edge. The optional
    # retained V26 integration below exercises the actual official validator.
    monkeypatch.setattr(
        terminal,
        "validate_vast_official_same_goal_reconciliation",
        lambda p: json.loads(Path(p).read_text()),
    )
    paths = {
        "registration": _write(source / "registration.json", registration),
        "registration_ack": _write(source / "ack.json", ack),
        "operator_authorization": _write(source / "authorization.json", auth),
        "setup": setup_path,
        "runtime_inputs": runtime_path,
        "session_authority": _write(source / "authority.json", authority),
        "bundle": _write(source / "bundle.json", bundle),
    }
    intent = terminal.operator_terminal_delivery_intent(
        run_id=runtime["run_id"],
        run_root=tmp_path / "finalized",
        records={k: terminal.file_record(p) for k, p in paths.items()},
        allocator_result_path=adapter_path,
        official_billing_path=billing_path,
        provider_zero_path=zero_path,
        billing_audit_root=source / "audit",
    )
    return intent, source


def _adapters(monkeypatch, *, readback_mode="verified"):
    from blueprint_pipeline import task_evaluation_run_webapp_sync as sync

    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self.value).encode()

    def post(request, timeout):
        payload = json.loads(request.data)
        calls.append(("publish", payload))
        projection = payload["policy_canary_result"]["projection_digest"]
        response = {
            "schema_version": "capture_task_evaluation_policy_canary_publication_receipt.v1",
            "status": payload["result_status"],
            "already_exists": False,
            **{
                k: payload[k]
                for k in (
                    "capture_session_id",
                    "intake_id",
                    "run_id",
                    "request_digest",
                    "configuration_digest",
                    "plan_digest",
                    "operator_registration_digest",
                )
            },
            "result_delivery_digest": payload["result_delivery"]["delivery_digest"],
            "policy_canary_projection_digest": projection,
            "notification_delivery": {
                "terminal_state": "blocked",
                "status": "failed",
                "attempts": 1,
                "run_result_digest": projection,
                "reason": "email_transport_unavailable",
            },
        }
        return Response(response)

    monkeypatch.setattr(sync.urllib_request, "urlopen", post)

    def publish(**kwargs):
        return sync_task_evaluation_policy_canary_to_webapp(
            **kwargs,
            endpoint_url="https://fixture.example/operator",
            token="fixture-only-token",
            max_attempts=1,
        )

    def readback(**kwargs):
        calls.append(("readback", None))
        delivery, projection, reg = (
            kwargs["result_delivery"],
            kwargs["policy_canary_result"],
            kwargs["registration"],
        )
        registry = json.loads(
            (kwargs["run_root"] / "artifacts/result_delivery/artifact_registry.json").read_text()
        )
        registered = {item["artifact_id"]: item for item in registry["artifacts"]}
        actual_artifacts = []
        for item in delivery["artifacts"]:
            record = registered[item["artifact_id"]]
            body = (Path(record["evidence_root"]) / record["relative_path"]).read_bytes()
            actual_artifacts.append(
                {
                    "artifact_id": item["artifact_id"],
                    "sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                    "size_bytes": len(body),
                    "verified": True,
                    "http_status": 200,
                }
            )
        value = {
            "status": "verified",
            "run_id": reg["run_id"],
            "operator_registration_digest": reg["registration_digest"],
            "result_delivery_digest": delivery["delivery_digest"],
            "policy_canary_projection_digest": projection["projection_digest"],
            "artifacts": actual_artifacts,
            "inbox": {
                "status": "verified",
                "run_id": reg["run_id"],
                "projection_digest": projection["projection_digest"],
                "team_namespace": reg["team_namespace"],
                "source": "website_owner_run_index_readback",
            },
        }
        if readback_mode == "missing_download":
            value["artifacts"] = value["artifacts"][:-1]
        if readback_mode == "prepared_inbox":
            value["inbox"]["status"] = "prepared"
        value["readback_digest"] = canonical_digest(value, digest_field="readback_digest")
        return value

    return terminal.OperatorTerminalDeliveryAdapters(
        sync_runner=publish, download_readback=readback
    ), calls


def test_verified_publication_downloads_and_inbox_complete_despite_waived_email_failure(
    tmp_path, monkeypatch
):
    intent, source = _fixture(tmp_path, monkeypatch)
    adapters, calls = _adapters(monkeypatch)
    original = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source / "native").rglob("*")
        if p.is_file()
    }
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "completed" and result["all_required_phases_done"] is True
    assert (
        result["execution_status"] == "blocked"
        and result["phases"]["qualification"]["status"] == "unqualified"
    )
    assert result["phases"]["notification"]["status"] == "failed"
    assert result["email_delivery_required_for_completion"] is False
    assert result["allocator_invoked"] is False and result["policy_execution_repeated"] is False
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original} == original
    repeated = terminal.finalize_operator_policy_canary(
        intent, adapters=terminal.OperatorTerminalDeliveryAdapters()
    )
    assert repeated == result and [c[0] for c in calls] == ["publish", "readback"]
    payload = calls[0][1]
    assert payload["operator_registration_digest"] == result["operator_registration_digest"]
    assert payload["plan_digest"] == result["plan_digest"]


@pytest.mark.parametrize("mode", ["missing_download", "prepared_inbox"])
def test_prepared_or_incomplete_readback_stays_pending_and_resumes_without_republishing(
    tmp_path, monkeypatch, mode
):
    intent, _ = _fixture(tmp_path, monkeypatch)
    adapters, calls = _adapters(monkeypatch, readback_mode=mode)
    pending = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert pending["status"] == "pending" and pending["all_required_phases_done"] is False
    assert pending["phases"]["website_publication"]["status"] == "complete"
    good, good_calls = _adapters(monkeypatch)
    completed = terminal.finalize_operator_policy_canary(intent, adapters=good)
    assert completed["status"] == "completed"
    assert [c[0] for c in calls] == ["publish", "readback"]
    assert [c[0] for c in good_calls] == ["readback"]


@pytest.mark.parametrize(
    "missing", ["allocator", "teardown", "cleanup", "provider_zero", "billing"]
)
def test_missing_closeout_never_publishes_or_reallocates(tmp_path, monkeypatch, missing):
    intent, source = _fixture(tmp_path, monkeypatch)
    paths = {
        "allocator": source / "allocator_result.json",
        "teardown": source / "teardown.json",
        "cleanup": source / "cleanup.json",
        "provider_zero": source / "provider-zero.json",
        "billing": source / "billing.json",
    }
    paths[missing].unlink()
    adapters, calls = _adapters(monkeypatch)
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "pending" and not calls
    assert result["allocator_invoked"] is False


def test_partial_recovery_never_writes_native_producer_history(tmp_path, monkeypatch):
    intent, source = _fixture(tmp_path, monkeypatch, partial=True)
    before = {
        p.relative_to(source): p.read_bytes() for p in (source / "native").rglob("*") if p.is_file()
    }
    adapters, _ = _adapters(monkeypatch)
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "completed"
    after = {
        p.relative_to(source): p.read_bytes() for p in (source / "native").rglob("*") if p.is_file()
    }
    assert after == before
    assert not (source / "native" / "partial_terminal_evidence").exists()


def test_registration_ack_mismatch_is_refused_before_transport(tmp_path, monkeypatch):
    intent, source = _fixture(tmp_path, monkeypatch)
    ack = json.loads((source / "ack.json").read_text())
    ack["run_id"] = "different-run"
    _write(source / "ack.json", ack)
    intent["records"]["registration_ack"] = terminal.file_record(source / "ack.json")
    intent["intent_digest"] = canonical_digest(intent, digest_field="intent_digest")
    with pytest.raises(terminal.OperatorTerminalDeliveryError, match="registration_not_authorized"):
        terminal.finalize_operator_policy_canary(intent)


@pytest.mark.external_data
@pytest.mark.skipif(
    not os.getenv("BLUEPRINT_OPERATOR_TERMINAL_REPLAY_ROOT"),
    reason="explicit retained-job replay only",
)
@pytest.mark.parametrize("existing_package", [False, True])
def test_retained_v26_saved_job_replays_with_real_validators_and_fake_publication(
    tmp_path, monkeypatch, existing_package
):
    source = Path(os.environ["BLUEPRINT_OPERATOR_TERMINAL_REPLAY_ROOT"])
    names = {
        "registration": "website-operator-registration.json",
        "registration_ack": "website-operator-registration-response.json",
        "operator_authorization": "operator-authorization.json",
        "setup": "setup/task_evaluation_policy_canary_execution_setup.v1.json",
        "runtime_inputs": "policy_canary_runtime_inputs.json",
        "session_authority": "policy_canary_session_authority.json",
        "bundle": "bundle/native_task_arena_policy_canary_session_bundle_receipt.v1.json",
    }
    run = json.loads((source / names["runtime_inputs"]).read_text())["run_id"]
    intent = terminal.operator_terminal_delivery_intent(
        run_id=run,
        run_root=tmp_path / "finalized",
        records={k: terminal.file_record(source / v) for k, v in names.items()},
        allocator_result_path=source / "allocator_result.json",
        official_billing_path=source / "official_billing_reconciliation.json",
        provider_zero_path=source / "post_teardown_global_provider_zero.json",
        billing_audit_root=source / "audit-not-needed",
    )
    if existing_package:
        shutil.copytree(source / "artifacts", Path(intent["run_root"]) / "artifacts")
    original = terminal.file_record(
        json.loads((source / "allocator_result.json").read_text())["native_control_result_path"]
    )
    adapters, calls = _adapters(monkeypatch)
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "completed" and result["all_required_phases_done"] is True
    assert result["execution_status"] == "blocked"
    assert result["phases"]["official_billing"]["official_total_usd"] == pytest.approx(0.366)
    assert terminal.file_record(original["path"]) == original
    payload = calls[0][1]
    omission = payload["policy_canary_result"]["control_omission"]
    registration = json.loads((source / names["registration"]).read_text())
    assert omission["authority_digest"] == registration["control_omission_authority_digest"]
    assert omission["qualified_comparison_permitted"] is False
    if existing_package:
        assert payload["policy_canary_result"] == json.loads(
            (source / "website-policy-canary-projection.json").read_text()
        )


def test_publication_transport_failure_resumes_sealed_package_without_recovery_or_reformat(
    tmp_path, monkeypatch
):
    intent, _ = _fixture(tmp_path, monkeypatch)

    def unavailable(**kwargs):
        raise TimeoutError("transport unavailable")

    pending = terminal.finalize_operator_policy_canary(
        intent, adapters=terminal.OperatorTerminalDeliveryAdapters(sync_runner=unavailable)
    )
    assert pending["status"] == "pending"
    assert pending["phases"]["artifact_package"]["status"] == "complete"

    def forbidden(*args, **kwargs):
        pytest.fail("sealed native/package stages must not be replayed")

    monkeypatch.setattr(terminal, "_native_result", forbidden)
    monkeypatch.setattr(terminal, "materialize_policy_canary_result_delivery", forbidden)
    adapters, _ = _adapters(monkeypatch)
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "completed"


def test_pending_billing_does_not_hide_observed_terminal_execution(tmp_path, monkeypatch):
    intent, source = _fixture(tmp_path, monkeypatch)
    (source / "billing.json").unlink()
    result = terminal.finalize_operator_policy_canary(intent)
    assert result["status"] == "pending"
    assert result["phases"]["execution"]["status"] == "complete"
    assert result["phases"]["official_billing"]["status"] == "pending"


def test_unverified_provider_readback_is_not_sealed_and_later_verified_readback_can_resume(
    tmp_path, monkeypatch
):
    intent, source = _fixture(tmp_path, monkeypatch)
    zero = json.loads((source / "provider-zero.json").read_text())
    (source / "provider-zero.json").unlink()
    pending = terminal.finalize_operator_policy_canary(
        intent,
        adapters=terminal.OperatorTerminalDeliveryAdapters(
            provider_zero_reader=lambda **kwargs: {"provider_zero_verified": True}
        ),
    )
    assert pending["status"] == "pending"
    assert not (Path(intent["run_root"]) / "operator_terminal_delivery/provider_zero.json").exists()
    adapters, _ = _adapters(monkeypatch)
    adapters = terminal.OperatorTerminalDeliveryAdapters(
        sync_runner=adapters.sync_runner,
        download_readback=adapters.download_readback,
        provider_zero_reader=lambda **kwargs: zero,
    )
    assert (
        terminal.finalize_operator_policy_canary(intent, adapters=adapters)["status"] == "completed"
    )


def test_crash_after_canonical_delivery_before_stage_marker_adopts_existing_sealed_bytes(
    tmp_path, monkeypatch
):
    intent, _ = _fixture(tmp_path, monkeypatch)
    pending = terminal.finalize_operator_policy_canary(intent)
    assert pending["status"] == "pending"
    root = Path(intent["run_root"])
    (root / "operator_terminal_delivery/packaged.json").unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("already sealed canonical delivery must be adopted")

    monkeypatch.setattr(terminal, "_native_result", forbidden)
    monkeypatch.setattr(terminal, "materialize_policy_canary_result_delivery", forbidden)
    adapters, _ = _adapters(monkeypatch)
    assert (
        terminal.finalize_operator_policy_canary(intent, adapters=adapters)["status"] == "completed"
    )


def test_cross_host_fixed_and_nested_records_validate_after_all_original_paths_are_removed(
    tmp_path, monkeypatch
):
    intent, source = _fixture(tmp_path, monkeypatch)
    originals = [*intent["records"].values()]
    for record in intent["records"].values():
        originals.extend(
            terminal._declared_file_records(json.loads(Path(record["path"]).read_text()))
        )
    locations = {}
    staged = tmp_path / "control-plane"
    staged.mkdir()
    for record in originals:
        original = Path(record["path"])
        copied = staged / (record["sha256"].removeprefix("sha256:") + ".json")
        if not copied.exists():
            shutil.copyfile(original, copied)
        locations[str(original)] = terminal.file_record(copied)
        assert copied.read_bytes() == original.read_bytes()
    intent["artifact_locations"] = locations
    intent["intent_digest"] = canonical_digest(intent, digest_field="intent_digest")
    shutil.rmtree(source)  # Test-owned fixture only; simulates the original Mac being unavailable.
    validated = terminal.validate_operator_terminal_delivery_inputs(intent, require_relocated=True)
    assert validated["artifact_relocation"]["unmapped_record_paths"] == []
    assert validated["artifact_relocation"]["declared_file_record_count"] > 10
    missing = next(path for path in locations if path.endswith("nested-scene-plan.json"))
    del intent["artifact_locations"][missing]
    intent["intent_digest"] = canonical_digest(intent, digest_field="intent_digest")
    with pytest.raises(terminal.OperatorTerminalDeliveryError):
        terminal.validate_operator_terminal_delivery_inputs(intent, require_relocated=True)


def test_ephemeral_download_urls_are_used_in_memory_but_never_persisted(tmp_path, monkeypatch):
    intent, _ = _fixture(tmp_path, monkeypatch)
    base, _ = _adapters(monkeypatch)
    secret = "https://private.example/artifact?signature=never-persist-this"

    def publish(**kwargs):
        result = dict(base.sync_runner(**kwargs))
        result["response"]["ephemeral_downloads"] = [{"url": secret}]
        result["notification_delivery"]["debug_url"] = secret
        return result

    def readback(**kwargs):
        assert kwargs["publication"]["response"]["ephemeral_downloads"][0]["url"] == secret
        result = dict(base.download_readback(**kwargs))
        result["ephemeral_downloads"] = [{"url": secret}]
        result["readback_digest"] = canonical_digest(result, digest_field="readback_digest")
        return result

    adapters = terminal.OperatorTerminalDeliveryAdapters(
        sync_runner=publish, download_readback=readback
    )
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result["status"] == "completed"
    for path in (Path(intent["run_root"]) / "operator_terminal_delivery").rglob("*.json"):
        assert "never-persist-this" not in path.read_text()
    proof = json.loads(
        Path(result["sealed_records"]["download_inbox_readback"]["path"]).read_text()
    )
    assert proof["ephemeral_download_urls_retained"] is False
    assert proof["readback_digest"] == canonical_digest(proof, digest_field="readback_digest")
    assert terminal.finalize_operator_policy_canary(intent)["status"] == "completed"


def test_finalizer_retains_exact_relocated_registration_for_live_artifact_routing_and_completed_resume(tmp_path, monkeypatch):
    from blueprint_pipeline.live_pipeline_result_artifact_resolution import resolve_live_pipeline_result_artifact
    # Mirror CP custody: provider evidence is physically inside the run root.
    intent, source = _fixture(tmp_path / 'scene-839873-canary-1', monkeypatch)
    registration_path = Path(intent['records']['registration']['path'])
    raw = b'\n\n' + registration_path.read_bytes() + b'\n'
    relocated = tmp_path / 'immutable-inputs/registration.raw.json'
    relocated.parent.mkdir()
    relocated.write_bytes(raw)
    expected = terminal.file_record(relocated)
    intent['records']['registration'] = {**expected, 'path': '/mac-only/registration.raw.json'}
    intent['artifact_locations'] = {'/mac-only/registration.raw.json': expected}
    intent['run_root'] = str(tmp_path / intent['run_id'])
    intent['intent_digest'] = canonical_digest(intent, digest_field='intent_digest')
    adapters, _ = _adapters(monkeypatch)
    result = terminal.finalize_operator_policy_canary(intent, adapters=adapters)
    assert result['status'] == 'completed', result
    root = Path(intent['run_root'])
    alias = root / 'website-operator-registration.json'
    assert alias.read_bytes() == raw
    assert terminal.file_record(alias)['sha256'] == expected['sha256']
    registry_path = root / 'artifacts/result_delivery/artifact_registry.json'
    original_registry = registry_path.read_bytes()
    registry = json.loads(original_registry)
    requested = registry['artifacts'][0]
    path, row = resolve_live_pipeline_result_artifact(legacy_state_root=tmp_path/'legacy',
        policy_canary_result_root=root.parent, run_id=intent['run_id'], artifact_id=requested['artifact_id'])
    assert path.is_file() and row['sha256'] == requested['sha256']
    # Recover the historical missing-alias state even after terminal receipts
    # exist; registry and producer input bytes remain unchanged.
    alias.unlink()
    assert terminal.finalize_operator_policy_canary(intent, adapters=adapters) == result
    assert alias.read_bytes() == raw and registry_path.read_bytes() == original_registry
    assert relocated.read_bytes() == raw


@pytest.mark.parametrize('conflict', ['different_bytes', 'symlink'])
def test_finalizer_never_replaces_conflicting_registration_alias(tmp_path, monkeypatch, conflict):
    intent, _ = _fixture(tmp_path, monkeypatch)
    root = Path(intent['run_root'])
    root.mkdir()
    alias = root / 'website-operator-registration.json'
    if conflict == 'symlink':
        alias.symlink_to(intent['records']['registration']['path'])
    else:
        alias.write_bytes(b'{"unrelated_owner":true}\n')
    original = alias.read_bytes()
    with pytest.raises(terminal.OperatorTerminalDeliveryError):
        terminal.finalize_operator_policy_canary(intent)
    assert alias.read_bytes() == original
    assert alias.is_symlink() == (conflict == 'symlink')
