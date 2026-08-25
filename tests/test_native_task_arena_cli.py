from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_authority_cli_supplies_complete_single_attempt_contract(monkeypatch, tmp_path) -> None:
    module = _load("issue_native_task_arena_paid_attempt_authority")
    observed = {}

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {"authorization_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(
        module, "materialize_native_task_arena_paid_attempt_authority", fake_materialize
    )
    output = tmp_path / "authority.json"
    result = module.main(
        [
            "--bundle-receipt",
            "bundle.json",
            "--prior-authority",
            "prior-authority.json",
            "--prior-result",
            "prior-result.json",
            "--prior-provider-zero",
            "prior-zero.json",
            "--prior-spend-reconciliation",
            "prior-spend.json",
            "--authority-reference",
            "explicit-user-goal",
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-13",
            "--blueprint-commit",
            "a" * 40,
            "--max-hourly-rate-usd",
            "0.75",
            "--hard-cap-usd",
            "1.25",
            "--hard-ttl-seconds",
            "3600",
            "--allow-active-instance",
            "41",
            "--allow-active-instance",
            "42",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed == {
        "bundle_receipt_path": "bundle.json",
        "prior_authority_path": "prior-authority.json",
        "prior_result_path": "prior-result.json",
        "prior_provider_zero_path": "prior-zero.json",
        "prior_spend_reconciliation_path": "prior-spend.json",
        "supplemental_prior_result_paths": [],
        "authorization_reference": "explicit-user-goal",
        "authorized_by": "user",
        "authorized_on": "2026-08-13",
        "blueprint_commit": "a" * 40,
        "max_hourly_rate_usd": 0.75,
        "hard_cap_usd": 1.25,
        "hard_ttl_seconds": 3600,
        "output_path": str(output),
        "allowed_active_instance_ids": (41, 42),
        "retain_warm_session": False,
    }


def test_authority_cli_supplies_new_lane_genesis_contract(monkeypatch, tmp_path) -> None:
    module = _load("issue_native_task_arena_paid_attempt_authority")
    observed = {}

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {"authorization_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(
        module, "materialize_native_task_arena_paid_attempt_authority", fake_materialize
    )
    output = tmp_path / "authority.json"
    result = module.main(
        [
            "--bundle-receipt",
            "bundle.json",
            "--project-spend-reconciliation",
            "project-spend.json",
            "--initial-provider-zero",
            "provider-zero.json",
            "--authority-reference",
            "explicit-new-lane",
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-25T14:30:00+00:00",
            "--blueprint-commit",
            "a" * 40,
            "--max-hourly-rate-usd",
            "0.8",
            "--hard-cap-usd",
            "0.75",
            "--hard-ttl-seconds",
            "3300",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed["project_spend_reconciliation_path"] == "project-spend.json"
    assert observed["initial_provider_zero_path"] == "provider-zero.json"
    assert observed["prior_authority_path"] is None
    assert observed["prior_result_path"] is None
    assert observed["prior_provider_zero_path"] is None
    assert observed["prior_spend_reconciliation_path"] is None
    assert observed["allowed_active_instance_ids"] == ()


def test_warm_authority_cli_supplies_zero_allocation_contract(
    monkeypatch, tmp_path
) -> None:
    module = _load("issue_native_task_arena_warm_attempt_authority")
    observed = {}
    prepared = {
        "bundle_sha256": "sha256:" + "b" * 64,
        "input_digest": "sha256:" + "c" * 64,
    }

    monkeypatch.setattr(
        module,
        "load_verified_native_task_arena_controls_bundle",
        lambda *_args, **_kwargs: prepared,
    )

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {"authorization_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(
        module, "materialize_native_task_arena_warm_attempt_authority", fake_materialize
    )
    output = tmp_path / "warm-authority.json"
    result = module.main(
        [
            "--warm-session",
            "warm-session.json",
            "--bundle-receipt",
            "bundle.json",
            "--blueprint-commit",
            "a" * 40,
            "--packet-receipt-digest",
            "sha256:" + "d" * 64,
            "--runtime-source-packet-digest",
            "sha256:" + "e" * 64,
            "--authority-reference",
            "explicit-user-goal",
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-21",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed == {
        "warm_session_path": "warm-session.json",
        "bundle_receipt_path": "bundle.json",
        "prepared_bundle": prepared,
        "authorization_reference": "explicit-user-goal",
        "authorized_by": "user",
        "authorized_on": "2026-08-21",
        "output_path": str(output),
    }


def test_provider_zero_cli_supplies_retained_closeout_contract(monkeypatch, tmp_path) -> None:
    module = _load("seal_native_task_arena_provider_zero")
    observed = {}

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {"receipt_digest": "sha256:" + "b" * 64}

    monkeypatch.setattr(module, "materialize_native_task_arena_provider_zero", fake_materialize)
    output = tmp_path / "provider-zero.json"
    result = module.main(
        [
            "--authority",
            "authority.json",
            "--result",
            "result.json",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert observed == {
        "authority_path": "authority.json",
        "result_path": "result.json",
        "output_path": str(output),
    }


def test_preallocation_closeout_cli_supplies_exact_zero_cost_evidence(
    monkeypatch, tmp_path
) -> None:
    module = _load("seal_native_task_arena_preallocation_closeout")
    observed = {}

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {
            "provider_zero_receipt_digest": "sha256:" + "a" * 64,
            "provider_mutation_performed": False,
        }

    monkeypatch.setattr(
        module, "materialize_native_task_arena_preallocation_closeout", fake_materialize
    )
    result = module.main(
        [
            "--authority",
            "authority.json",
            "--allocator-result",
            "result.json",
            "--watchdog-handoff",
            "watchdog.json",
            "--object-store-cleanup",
            "cleanup.json",
            "--api-provider-zero",
            "zero.json",
            "--sibling-preallocation-closeout",
            "sibling.json",
            "--output-dir",
            str(tmp_path / "closeout"),
        ]
    )

    assert result == 0
    assert observed == {
        "authority_path": "authority.json",
        "allocator_result_path": "result.json",
        "watchdog_handoff_path": "watchdog.json",
        "object_store_cleanup_path": "cleanup.json",
        "api_provider_zero_path": "zero.json",
        "sibling_preallocation_closeout_paths": ["sibling.json"],
        "output_dir": str(tmp_path / "closeout"),
    }


def test_pre_spend_closeout_cli_binds_consumption_and_api_zero(
    monkeypatch, tmp_path
) -> None:
    module = _load("seal_native_task_arena_pre_spend_closeout")
    observed = {}

    def fake_materialize(**kwargs):
        observed.update(kwargs)
        return {
            "provider_zero_receipt_digest": "sha256:" + "a" * 64,
            "provider_mutation_performed": False,
        }

    monkeypatch.setattr(
        module, "materialize_native_task_arena_pre_spend_closeout", fake_materialize
    )
    result = module.main(
        [
            "--authority",
            "authority.json",
            "--allocator-result",
            "result.json",
            "--authority-consumption",
            "consumption.json",
            "--api-provider-zero",
            "zero.json",
            "--output-dir",
            str(tmp_path / "closeout"),
        ]
    )

    assert result == 0
    assert observed == {
        "authority_path": "authority.json",
        "allocator_result_path": "result.json",
        "authority_consumption_path": "consumption.json",
        "api_provider_zero_path": "zero.json",
        "output_dir": str(tmp_path / "closeout"),
    }


def test_authority_cli_fails_closed_without_provider_mutation(monkeypatch, capsys) -> None:
    module = _load("issue_native_task_arena_paid_attempt_authority")

    def fail(**_kwargs):
        raise ValueError("retained_predecessor_invalid")

    monkeypatch.setattr(module, "materialize_native_task_arena_paid_attempt_authority", fail)
    result = module.main(
        [
            "--bundle-receipt",
            "bundle.json",
            "--prior-authority",
            "prior-authority.json",
            "--prior-result",
            "prior-result.json",
            "--prior-provider-zero",
            "prior-zero.json",
            "--prior-spend-reconciliation",
            "prior-spend.json",
            "--authority-reference",
            "explicit-user-goal",
            "--authorized-by",
            "user",
            "--authorized-on",
            "2026-08-13",
            "--blueprint-commit",
            "a" * 40,
            "--max-hourly-rate-usd",
            "0.75",
            "--hard-cap-usd",
            "1.25",
            "--hard-ttl-seconds",
            "3600",
            "--output",
            "authority.json",
        ]
    )

    assert result == 2
    assert '"provider_mutation_performed": false' in capsys.readouterr().out
