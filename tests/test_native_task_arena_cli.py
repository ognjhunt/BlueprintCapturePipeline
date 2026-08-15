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
        "authorization_reference": "explicit-user-goal",
        "authorized_by": "user",
        "authorized_on": "2026-08-13",
        "blueprint_commit": "a" * 40,
        "max_hourly_rate_usd": 0.75,
        "hard_cap_usd": 1.25,
        "hard_ttl_seconds": 3600,
        "output_path": str(output),
        "allowed_active_instance_ids": (41, 42),
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
