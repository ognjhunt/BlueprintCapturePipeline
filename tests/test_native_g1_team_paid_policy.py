"""The selected path reaches only canonical, single-use paid transport."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_team_paid_policy as lane
from tests.test_native_g1_team_dispatch_preflight import _prepared
from tests.test_native_g1_team_provider_bundle import COMMIT
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _setup(tmp_path, monkeypatch):
    inputs, bundle, _, _ = _prepared(tmp_path, monkeypatch)
    authority = inputs["authority_arguments"]
    args = SimpleNamespace(
        provider="vast", execute=True,
        g1_team_bundle_receipt=str(inputs["bundle_receipt_path"]),
        g1_team_intent=str(authority["intent_path"]),
        g1_team_registry=str(authority["registry_path"]),
        g1_team_approval=str(authority["approval_path"]),
        g1_team_trusted_client=list(authority["trusted_clients"]),
        g1_team_credential_registry=str(inputs["credential_registry_path"]),
        adp_job_dir=str(tmp_path / "paid"), admission_out=str(tmp_path / "admission.json"),
        adapter_output=str(tmp_path / "adapter.json"),
        adp_max_hourly_rate_usd=inputs["max_hourly_rate_usd"],
        adp_max_spend_usd=inputs["hard_cap_usd"],
        adp_hard_ttl_seconds=inputs["hard_ttl_seconds"],
        adp_allowed_active_vast_instance_id=[], adp_machine_avoidlist=None,
    )
    monkeypatch.setattr(lane, "_controller_release_authority", lambda commit, identity: {"status": "exact_main_checkout", "source_commit": commit})
    monkeypatch.setattr(lane, "_early_spend_lock_blockers", lambda: [])
    monkeypatch.setattr(lane, "_early_provider_credit_blockers", lambda cap: [])
    class Gate:
        def release(self):
            pass
    monkeypatch.setattr(lane, "_hold_pre_stage_launch_gate", lambda: (Gate(), None))
    monkeypatch.setattr(lane, "disk_headroom", lambda **kwargs: {"available_bytes": 64 * 1024**3})
    class Reservation:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def receipt(self):
            return {"expected_bytes": lane.COLLECTION_FORECAST_BYTES}
    class Health:
        def check(self):
            pass
    from contextlib import nullcontext
    monkeypatch.setattr(lane, "reserve_control_plane_disk", lambda *args, **kwargs: Reservation())
    monkeypatch.setattr(lane, "keep_reservation_live", lambda *args, **kwargs: nullcontext(Health()))
    return args, bundle, {"orchestrator_source_commit": COMMIT}


def test_real_selected_admission_consumes_once_and_uses_private_watchdog_transport(tmp_path, monkeypatch):
    args, bundle, identity = _setup(tmp_path, monkeypatch)
    mutations = []
    def adapter(**kwargs):
        assert kwargs["paid_resource_admission_grant"] is not None
        assert kwargs["require_independent_watchdog"] is True
        assert kwargs["provider_bundle_kind"] == "native_g1_team_policy"
        assert kwargs["prepared_bundle"]["bundle_sha256"] == bundle["bundle_sha256"]
        assert kwargs["prepared_bundle"]["status"] == "ready"
        assert set(kwargs["runtime_secret_file_paths"]) == {"BLUEPRINT_G1_TEAM_CREDENTIAL_FILE"}
        first = kwargs["pre_provider_mutation_hook"]()
        assert first["status"] == "consumed"
        assert kwargs["pre_provider_mutation_hook"]()["status"] == "blocked"
        mutations.append(first)
        return {"status": "blocked", "continuing_spend_from_this_run": False, "blockers": ["hermetic_worker_failed"]}
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "blocked"
    assert len(mutations) == 1
    # A stopped caller with the same attempt cannot enter transport again.
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("duplicate allocator transport"))
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert "g1_team_paid_attempt_already_consumed" in result["blockers"]


def test_authority_revocation_during_slow_admission_blocks_create(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    def adapter(**kwargs):
        approval = Path(args.g1_team_approval)
        value = json.loads(approval.read_text())
        value["site_observation_exchange_authorized"] = False
        approval.write_text(json.dumps(value))
        check = kwargs["pre_provider_mutation_hook"]()
        assert check["status"] == "blocked"
        assert check["blockers"] == ["g1_team_paid_inputs_changed_before_create"]
        return {"status": "blocked", "provider_mutations_performed": 0, "blockers": check["blockers"]}
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["provider_mutations_performed"] == 0
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_controller_change_before_create_refuses_exact_attempt(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    def adapter(**kwargs):
        check = kwargs["pre_provider_mutation_hook"]()
        assert check["status"] == "blocked"
        return {"status": "blocked", "provider_mutations_performed": 0, "blockers": check["blockers"]}
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(
        args, control_identity=identity, control_blockers=[],
        control_recheck=lambda: ([], {"orchestrator_source_commit": "b" * 40}),
    )
    assert result["blockers"] == ["g1_team_paid_controller_changed_before_create"]


def test_spend_or_identity_block_does_not_resolve_credential_or_stage_bundle(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(lane, "verify_g1_team_dispatch_inputs", lambda **kwargs: pytest.fail("blocked admission resolved private inputs"))
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("blocked admission reached transport"))
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=["hermetic_control_blocker"])
    assert result["provider_mutations_performed"] == 0
    assert not Path(args.adp_job_dir).exists()


def test_dry_run_never_consumes_or_supplies_mutation_hook(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    args.execute = False
    def adapter(**kwargs):
        assert kwargs["execute"] is False
        assert kwargs["paid_resource_admission_grant"] is None
        assert kwargs["pre_provider_mutation_hook"] is None
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "dry_run_ready"
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_real_canonical_dry_transport_checks_actual_call_arguments(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    args.execute = False
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "dry_run_ready"
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_canonical_allocator_cli_reaches_real_selected_dry_transport(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import paid_resource_allocator as allocator
    args, _, identity = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers", lambda: ([], identity))
    argv = ["gpu-canary", "--probe-kind", lane.PROBE_KIND, "--provider", "vast"]
    for name in (
        "g1_team_bundle_receipt", "g1_team_intent", "g1_team_registry", "g1_team_approval",
        "g1_team_credential_registry", "adp_job_dir", "admission_out", "adapter_output",
        "adp_max_hourly_rate_usd", "adp_max_spend_usd", "adp_hard_ttl_seconds",
    ):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for client in args.g1_team_trusted_client:
        argv.extend(["--g1-team-trusted-client", client])
    assert allocator.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    result = json.loads(Path(args.adapter_output).read_text())
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_unknown_transport_exception_does_not_invent_provider_zero(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    def adapter(**kwargs):
        raise ValueError("private provider response must not be printed")
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "blocked"
    assert "provider_mutations_performed" not in result
    assert result["provider_mutation_status"] == "unproven_reconcile_exact_attempt"
    assert result["provider_teardown_verified"] is False
    assert "private provider response" not in json.dumps(result)


@pytest.mark.parametrize("execute", [False, True])
def test_collection_shortage_refuses_before_provider_transport(tmp_path, monkeypatch, execute):
    args, _, identity = _setup(tmp_path, monkeypatch)
    args.execute = execute
    monkeypatch.setattr(lane, "disk_headroom", lambda **kwargs: {"available_bytes": 11_000_000_000})
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("disk shortage reached provider"))
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "blocked"
    assert "g1_team_paid_collection_capacity_insufficient" in result["blockers"]
    assert result["provider_mutations_performed"] == 0
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_collection_reservation_is_held_through_transport_and_released_on_failure(tmp_path, monkeypatch):
    args, _, identity = _setup(tmp_path, monkeypatch)
    events = []
    class Reservation:
        def __enter__(self):
            events.append("reserved")
            return self
        def __exit__(self, *args):
            events.append("released")
        def receipt(self):
            return {"expected_bytes": lane.COLLECTION_FORECAST_BYTES}
    def reservation(*positional, **kwargs):
        assert positional == ("policy_canary_dispatch",)
        assert kwargs["expected_bytes"] == lane.COLLECTION_FORECAST_BYTES
        assert kwargs["ttl_seconds"] > args.adp_hard_ttl_seconds
        return Reservation()
    monkeypatch.setattr(lane, "reserve_control_plane_disk", reservation)
    def adapter(**kwargs):
        assert events == ["reserved"]
        raise ValueError("hermetic transport failure")
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["status"] == "blocked"
    assert events == ["reserved", "released"]


def test_lost_collection_reservation_refuses_before_single_use_record(tmp_path, monkeypatch):
    from contextlib import nullcontext
    args, _, identity = _setup(tmp_path, monkeypatch)
    class Lost:
        def check(self):
            raise lane.ControlPlaneDiskBudgetError("hermetic expired reservation")
    monkeypatch.setattr(lane, "keep_reservation_live", lambda *args, **kwargs: nullcontext(Lost()))
    def adapter(**kwargs):
        check = kwargs["pre_provider_mutation_hook"]()
        assert check["blockers"] == ["g1_team_paid_collection_reservation_lost"]
        return {"status": "blocked", "blockers": check["blockers"], "provider_mutations_performed": 0}
    monkeypatch.setattr(lane, "run_arena_native_control_vast", adapter)
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["provider_mutations_performed"] == 0
    assert not (Path(args.adp_job_dir) / lane.CONSUMPTION_FILENAME).exists()


def test_postrun_verifier_reopens_real_retained_score_frames_and_video_bytes(tmp_path, monkeypatch):
    from tests.test_native_g1_team_paid_output import _evidence
    from blueprint_pipeline.native_g1_team_paid_output import verify_g1_team_paid_output
    evidence, _ = _evidence(tmp_path, monkeypatch)
    verified = verify_g1_team_paid_output(**evidence)
    archive = tmp_path / "sealed-input.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(lane.PACKET_RELATIVE_PATH, json.dumps(evidence["execution_packet"]))
    bundle = {
        "bundle_path": str(archive), "bundle_sha256": lane._sha256(archive),
        "scene_plan_digest": evidence["scene_plan_digest"],
        "scene_packet_receipt_digest": evidence["scene_packet_receipt_digest"],
    }
    attempt = tmp_path / "attempts/attempt_001"
    root = attempt / "immutable_execution"
    (root / "selected-worker").mkdir(parents=True)
    shutil.copytree(evidence["output_dir"], root / "selected-worker/worker")
    native = {
        "schema_version": "native_g1_team_provider_result.v1", "status": "completed_development_only",
        "execution_packet_digest": evidence["execution_packet"]["packet_digest"],
        "worker_output_relative_path": "selected-worker/worker",
        "candidate_policy_queried": True, "public_redistribution_authorized": False,
        "verified_output": verified,
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    (root / lane.RESULT_FILENAME).write_text(json.dumps(native))
    result = {"attempt_root": str(attempt)}
    assert lane._verify_output(result, bundle, job=tmp_path) == verified
    with pytest.raises(ValueError, match="output_path_invalid"):
        lane._verify_output(result, bundle, job=tmp_path / "foreign-job")
    worker = root / "selected-worker/worker"
    (worker / verified["media"]["review_videos"]["head"]["relative_path"]).write_bytes(b"altered")
    with pytest.raises(ValueError):
        lane._verify_output(result, bundle, job=tmp_path)


def test_renewal_failure_during_closeout_preserves_returned_teardown(tmp_path, monkeypatch):
    from contextlib import contextmanager
    args, _, identity = _setup(tmp_path, monkeypatch)
    @contextmanager
    def failing_heartbeat(*args, **kwargs):
        class Health:
            def check(self):
                pass
        yield Health()
        raise lane.ControlPlaneDiskBudgetError("hermetic renewal failed at exit")
    monkeypatch.setattr(lane, "keep_reservation_live", failing_heartbeat)
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **kwargs: {
        "status": "blocked", "blockers": ["hermetic worker failure"],
        "provider_teardown_verified": True, "continuing_spend_from_this_run": False,
    })
    result = lane.dispatch_g1_team_paid_policy(args, control_identity=identity, control_blockers=[])
    assert result["provider_teardown_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert "g1_team_paid_collection_reservation_lost" in result["blockers"]
    assert json.loads(Path(args.adapter_output).read_text()) == result
