"""Sealed G1 beta authority reaches canonical transport exactly once."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_beta_paid as paid
from blueprint_pipeline import native_g1_beta_bundle as bundle_builder
from blueprint_pipeline import control_plane_disk_budget as disk
from blueprint_pipeline import control_plane_disk_reservation_heartbeat as heartbeat
from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import vast_independent_watchdog_control as watchdog
from tests.test_native_g1_beta_bundle import fixture


def setup(tmp_path, monkeypatch):
    # These fixtures model a standalone offline caller, not the CI runner's
    # systemd service or live spend configuration. Separate cases below opt
    # into an unproven systemd caller and retain the real production guard.
    for name in ("INVOCATION_ID", watchdog.CALLER_EXIT_SURVIVAL_ENV,
                 arena.SPEND_ADMISSION_LOCK_PATH_ENV, arena.REQUIRE_SPEND_ADMISSION_LOCK_ENV,
                 "BLUEPRINT_LAUNCH_PROOF_MODE"):
        monkeypatch.delenv(name, raising=False)
    authority, spec = fixture(tmp_path, monkeypatch)
    bundle = bundle_builder.build_bundle(job_dir=tmp_path / "sealed", authority=authority, spec=spec, implementation_commit="a" * 40)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    args = SimpleNamespace(provider="vast", execute=True, g1_beta_session_authority=str(authority_path),
        g1_beta_session_bundle_receipt=str(tmp_path / "sealed/native_g1_beta_provider_bundle.v1.json"),
        adp_job_dir=str(tmp_path / "allocator"), admission_out=str(tmp_path / "admission.json"),
        adapter_output=str(tmp_path / "adapter.json"), adp_max_hourly_rate_usd=float(authority["max_hourly_rate_usd"]),
        adp_max_spend_usd=authority["hard_cap_usd"], adp_hard_ttl_seconds=authority["hard_ttl_seconds"],
        adp_machine_avoidlist=None, adp_allowed_active_vast_instance_id=[])
    identity = {"orchestrator_source_commit": "a" * 40}
    monkeypatch.setattr(paid, "_controller_release_authority", lambda commit, actual:
        {"source_commit": commit} if actual == identity else None)
    monkeypatch.setattr(paid, "_early_spend_lock_blockers", lambda: [])
    monkeypatch.setattr(paid, "_early_provider_credit_blockers", lambda cap: [])
    gate = SimpleNamespace(release=lambda: None)
    monkeypatch.setattr(paid, "_hold_pre_stage_launch_gate", lambda: (gate, None))
    monkeypatch.setattr(disk, "disk_headroom", lambda **kwargs: {"available_bytes": 64 * 1024**3})
    reservations = []
    class Reservation:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def receipt(self): return {"expected_bytes": reservations[-1]["expected_bytes"]}
    def reserve(*args, **kwargs):
        reservations.append(kwargs)
        return Reservation()
    monkeypatch.setattr(disk, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(heartbeat, "keep_reservation_live", lambda reservation: nullcontext(SimpleNamespace(check=lambda: None)))
    sources = []
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
                        lambda source, *, site_id: sources.append((source, site_id)))
    return args, authority, bundle, identity, reservations, sources


def dispatch(args, identity, **kwargs):
    return paid.dispatch_session(args, control_identity=identity, control_blockers=[],
        control_recheck=kwargs.pop("control_recheck", lambda: ([], identity)), **kwargs)


def test_exact_session_admission_consumes_once_and_rechecks_current_source(tmp_path, monkeypatch):
    args, authority, bundle, identity, reservations, sources = setup(tmp_path, monkeypatch)
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        assert kwargs["paid_resource_admission_grant"] is not None
        assert kwargs["instance_label_exact"] == authority["resource_name"]
        assert kwargs["require_independent_watchdog"] is True and kwargs["stale_offer_create_retry_limit"] == 0
        assert kwargs["expected_provider_download_bytes"] == bundle["transfer_quote"]["expected_provider_download_bytes"]
        assert kwargs["expected_provider_upload_bytes"] == paid.MAX_OUTPUT_BYTES
        consumed = json.loads((Path(args.adp_job_dir) / "authorization_consumption.json").read_text())
        assert kwargs["hard_deadline_epoch"] == consumed["hard_deadline_epoch"]
        assert kwargs["pre_provider_mutation_hook"]() == consumed
        assert kwargs["pre_provider_mutation_hook"]()["status"] == "blocked"
        return {"status": "blocked", "blockers": ["hermetic_native_failure"], "provider_mutations_performed": 1}
    monkeypatch.setattr(paid, "run_arena_native_control_vast", transport)
    result = dispatch(args, identity)
    assert result["provider_mutations_performed"] == 1 and len(calls) == 1
    assert sources == [(authority["beta_screening_admission"]["source"], authority["beta_screening_admission"]["site_id"])]
    assert reservations[0]["expected_bytes"] == 4 * paid.MAX_OUTPUT_BYTES + bundle["bundle_size_bytes"]
    monkeypatch.setattr(paid, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("replayed allocation"))
    assert "g1_beta_paid_authority_already_consumed" in dispatch(args, identity)["blockers"]


@pytest.mark.parametrize("changed", ["source", "release", "authority"])
def test_precreate_scope_change_prevents_provider_mutation(tmp_path, monkeypatch, changed):
    args, _, _, identity, _, _ = setup(tmp_path, monkeypatch)
    def transport(**kwargs):
        if changed == "source":
            def revoked(*args, **kwargs): raise ValueError("capture revoked")
            monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", revoked)
        elif changed == "authority":
            Path(args.g1_beta_session_authority).write_text("{}")
        observed = kwargs["pre_provider_mutation_hook"]()
        assert observed["status"] == "blocked"
        return {"status": "blocked", "blockers": observed["blockers"], "provider_mutations_performed": 0}
    monkeypatch.setattr(paid, "run_arena_native_control_vast", transport)
    result = dispatch(args, identity, control_recheck=lambda: ([], {} if changed == "release" else identity))
    assert result["provider_mutations_performed"] == 0
    # A consumed launch remains consumed even after a definite pre-create block.
    assert (Path(args.adp_job_dir) / "authorization_consumption.json").is_file()


def test_uncertain_transport_never_claims_zero_or_retries(tmp_path, monkeypatch):
    args, _, _, identity, _, _ = setup(tmp_path, monkeypatch)
    def failed(**kwargs):
        assert kwargs["pre_provider_mutation_hook"]()["status"] == "consumed"
        raise ValueError("untrusted provider response text")
    monkeypatch.setattr(paid, "run_arena_native_control_vast", failed)
    result = dispatch(args, identity)
    assert "provider_mutations_performed" not in result
    assert result["provider_mutation_status"] == "unproven_reconcile_exact_attempt"
    assert "untrusted provider" not in json.dumps(result)
    monkeypatch.setattr(paid, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("retried unknown allocation"))
    assert "g1_beta_paid_authority_already_consumed" in dispatch(args, identity)["blockers"]


@pytest.mark.parametrize("inherited_host_context", [False, True])
def test_dry_run_reaches_real_canonical_transport_without_consumption(tmp_path, monkeypatch, inherited_host_context):
    if inherited_host_context:
        monkeypatch.setenv("INVOCATION_ID", "hermetic-systemd-fixture")
        monkeypatch.setenv(watchdog.CALLER_EXIT_SURVIVAL_ENV, "unproven")
        monkeypatch.setenv(arena.SPEND_ADMISSION_LOCK_PATH_ENV, str(tmp_path / "absent-live-lock"))
        monkeypatch.setenv(arena.REQUIRE_SPEND_ADMISSION_LOCK_ENV, "true")
        monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    args, _, _, identity, reservations, _ = setup(tmp_path, monkeypatch)
    args.execute = False
    result = dispatch(args, identity)
    assert result["status"] == "dry_run_ready", result
    assert result["live_spend_preflight_checked"] is False
    assert result["independent_watchdog_preflight"] == {
        "caller_exit_survival_contract": "detached_posix_session",
        "caller_exit_survival_supported": True, "watchdog_armed": False}
    assert reservations == []
    assert not (Path(args.adp_job_dir) / "authorization_consumption.json").exists()


def test_real_dry_transport_still_refuses_unproven_systemd_survival(tmp_path, monkeypatch):
    args, _, _, identity, reservations, _ = setup(tmp_path, monkeypatch)
    args.execute = False
    monkeypatch.setenv("INVOCATION_ID", "hermetic-systemd-fixture")
    monkeypatch.setattr(arena, "run_vast_provider_adapter", lambda **kwargs: pytest.fail("provider boundary entered"))
    result = dispatch(args, identity)
    assert result["status"] == "blocked", result
    assert result["blockers"] == ["native_g1_beta_independent_watchdog_caller_exit_survival_unproven"]
    assert result["provider_mutations_performed"] == 0
    assert result["independent_watchdog_preflight"]["caller_exit_survival_supported"] is False
    assert reservations == []
    assert not (Path(args.adp_job_dir) / "authorization_consumption.json").exists()


@pytest.mark.parametrize("case", ["budget", "disk", "spend"])
def test_budget_and_capacity_blocks_never_enter_transport(tmp_path, monkeypatch, case):
    args, _, _, identity, _, _ = setup(tmp_path, monkeypatch)
    if case == "budget":
        args.adp_max_spend_usd += 1
    if case == "disk":
        monkeypatch.setattr(disk, "disk_headroom", lambda **kwargs: {"available_bytes": 0})
    if case == "spend":
        monkeypatch.setattr(paid, "_early_spend_lock_blockers", lambda: ["hermetic_spend_lock"])
    monkeypatch.setattr(paid, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("blocked admission entered transport"))
    result = dispatch(args, identity)
    assert result["provider_mutations_performed"] == 0
    assert not (Path(args.adp_job_dir) / "authorization_consumption.json").exists()
