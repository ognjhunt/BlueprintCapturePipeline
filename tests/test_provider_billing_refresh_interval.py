"""The billing refresh keeps its ten-minute cadence however often the spend guard pulls it.

``blueprint-gpu-spend-guard.service`` wants and orders after the reconciler, so
every guard start (its own one-minute timer and every unit that wants the
guard) ran a full refresh first: 31 runs and 1,279 s of CPU an hour on
2026-11-02, up to 920 MB each. The reconciler now exits at once while its own
last export is younger than the interval and covers every provider the guard's
own last report requires, so the dependency still guarantees a refresh attempt
before the guard reads older or incomplete evidence, and the guard's 24-hour
freshness rule is never approached.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import cloud_vm_render_providers
from blueprint_pipeline import provider_billing_reconciler as billing
from blueprint_pipeline.spend_admission_lock import MAX_BILLING_AGE_SECONDS
from scripts import gpu_spend_guard as guard

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "deploy" / "systemd"


def _write_guard_report(
    path: Path, *, age: timedelta = timedelta(minutes=1), live: tuple[str, ...] = ("vast",),
    judged: str = "sha256:" + "0" * 64, blockers: tuple[str, ...] = (), **overrides,
) -> None:
    """The spend guard's own ``--json-report``: only the guard sees live inventory."""
    report = {
        "schema_version": guard.SCHEMA_VERSION,
        "generated_at": (datetime.now(timezone.utc) - age).isoformat(),
        "billing_reconciliation": {
            "status": "blocked" if blockers else "reconciled",
            "billing_export_sha256": judged,
            "live_providers": sorted(live),
            "blockers": list(blockers),
        },
    }
    report.update(overrides)
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _export_digest(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_export(path: Path, *, age: timedelta, **overrides) -> None:
    payload = {
        "schema_version": billing.BILLING_EXPORT_SCHEMA_VERSION,
        "generated_at": (datetime.now(timezone.utc) - age).isoformat(),
        "currency": "USD",
        "scope": billing.BILLING_SCOPE,
        "provider_totals_usd": {"digitalocean": 3.0, "vast": 12.5},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


class _Refresh:
    """Stands in for the provider requests; records whether a refresh ran."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[dict] = []

        def refresh(**kwargs):
            self.calls.append(kwargs)
            return {"schema_version": "blueprint.provider_billing_reconciliation_run.v1",
                    "status": "reconciled"}

        monkeypatch.setattr(billing, "reconcile_provider_billing", refresh)

        class _NoInventory:
            def inventory_projects(self) -> list[str]:
                return []

        monkeypatch.setattr(cloud_vm_render_providers, "GCPRenderProvider", _NoInventory)


@pytest.fixture
def refresh(monkeypatch: pytest.MonkeyPatch) -> _Refresh:
    return _Refresh(monkeypatch)


def _main(tmp_path: Path, export: Path, *extra: str, report: Path | None = None) -> int:
    """One unit start; a current guard report vouching for ``vast`` unless one is given."""
    if report is None:
        report = tmp_path / "gpu_spend_guard_report.json"
        if not report.exists() and not report.is_symlink():
            _write_guard_report(report)
    return billing.main([
        "--secrets-dir", str(tmp_path / "secrets"),
        "--billing-export", str(export),
        "--audit-root", str(tmp_path / "billing-audit"),
        "--cohort-start-at", "2026-01-01T00:00:00Z",
        "--spend-guard-report", str(report),
        *extra,
    ])


def test_a_fresh_export_is_retained_without_any_provider_request(
    tmp_path: Path, refresh: _Refresh, capsys: pytest.CaptureFixture[str]
) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=4))
    before = export.read_bytes()

    assert _main(tmp_path, export, "--min-refresh-interval-seconds", "600") == 0

    assert refresh.calls == []
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "fresh_export_retained"
    assert result["provider_requests_added"] == 0
    assert result["provider_mutation_performed"] is False
    assert 200 < result["billing_export_age_seconds"] < 300
    assert result["min_refresh_interval_seconds"] == 600
    assert result["spend_guard_live_providers"] == ["vast"]
    assert export.read_bytes() == before
    assert not (tmp_path / "billing-audit").exists()


def _tampered(tmp_path: Path, export: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    fresh = timedelta(minutes=2)
    if case == "older_than_the_interval":
        _write_export(export, age=timedelta(minutes=11))
    elif case == "generated_in_the_future":
        _write_export(export, age=-timedelta(minutes=2))
    elif case == "wrong_schema":
        _write_export(export, age=fresh, schema_version="blueprint.provider_billing_export.v0")
    elif case == "wrong_scope":
        _write_export(export, age=fresh, scope="another_cohort")
    elif case == "wrong_currency":
        _write_export(export, age=fresh, currency="EUR")
    elif case == "no_totals":
        _write_export(export, age=fresh, provider_totals_usd={})
    elif case == "negative_total":
        _write_export(export, age=fresh, provider_totals_usd={"vast": -1.0})
    elif case == "boolean_total":
        _write_export(export, age=fresh, provider_totals_usd={"vast": True})
    elif case == "unknown_provider":
        _write_export(export, age=fresh, provider_totals_usd={"vast": 1.0, "lambda": 2.0})
    elif case == "naive_timestamp":
        _write_export(export, age=fresh, generated_at="2026-11-02T16:00:00")
    elif case == "malformed":
        export.write_text("{not json", encoding="utf-8")
        export.chmod(0o600)
    elif case == "not_an_object":
        export.write_text("[]", encoding="utf-8")
        export.chmod(0o600)
    elif case == "symlinked":
        real = tmp_path / "elsewhere.json"
        _write_export(real, age=fresh)
        export.symlink_to(real)
    elif case == "group_writable":
        _write_export(export, age=fresh)
        export.chmod(0o620)
    elif case == "foreign_owned":
        _write_export(export, age=fresh)
        owner = export.stat().st_uid
        monkeypatch.setattr(billing, "_euid", lambda: owner + 1)
    elif case == "too_large":
        _write_export(export, age=fresh, padding="x" * (1024 * 1024))
    elif case == "missing":
        pass
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "older_than_the_interval", "generated_in_the_future", "wrong_schema", "wrong_scope",
        "wrong_currency", "no_totals", "negative_total", "boolean_total", "unknown_provider",
        "naive_timestamp", "malformed", "not_an_object", "symlinked", "group_writable",
        "foreign_owned", "too_large", "missing",
    ],
)
def test_anything_but_a_fresh_trustworthy_export_refreshes(
    tmp_path: Path, refresh: _Refresh, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    export = tmp_path / "provider_billing_export.json"
    _tampered(tmp_path, export, monkeypatch, case)

    assert _main(tmp_path, export, "--min-refresh-interval-seconds", "600") == 0

    assert len(refresh.calls) == 1


def test_a_required_provider_missing_from_the_export_refreshes(tmp_path: Path, refresh: _Refresh) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=2))

    _main(tmp_path, export, "--min-refresh-interval-seconds", "600", "--require-provider", "runpod")

    assert len(refresh.calls) == 1


def test_a_live_provider_missing_from_the_export_refreshes_at_once(
    tmp_path: Path, refresh: _Refresh
) -> None:
    """Review of #3278: the guard blocks paid admission while a live provider is uncovered.

    One provider's API failing during a refresh leaves it out of the export. A
    skip keyed on age alone kept that export for ten minutes; the guard's own
    report names the provider, so the next start refreshes.
    """
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))  # this refresh lost runpod
    report = tmp_path / "guard-report.json"
    # The guard last judged the previous export, which covered runpod; it has
    # not run since, so nothing in the report blocks yet -- only the live list.
    _write_guard_report(report, live=("runpod", "vast"), judged="sha256:" + "1" * 64)

    _main(tmp_path, export, "--min-refresh-interval-seconds", "600", report=report)

    assert len(refresh.calls) == 1


def test_a_guard_verdict_against_this_export_is_honoured(tmp_path: Path, refresh: _Refresh) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))
    report = tmp_path / "guard-report.json"
    _write_guard_report(report, live=(), judged=_export_digest(export),
                        blockers=("provider_billing_export_owner_untrusted",))

    _main(tmp_path, export, "--min-refresh-interval-seconds", "600", report=report)

    assert len(refresh.calls) == 1


def test_a_report_on_an_earlier_export_vouches_only_for_live_coverage(
    tmp_path: Path, refresh: _Refresh, capsys: pytest.CaptureFixture[str]
) -> None:
    """The previous export lacked vast; the current one covers it, so nothing is missing now."""
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))
    report = tmp_path / "guard-report.json"
    _write_guard_report(report, live=("vast",), judged="sha256:" + "1" * 64,
                        blockers=("provider_billing_export_missing:vast",))

    _main(tmp_path, export, "--min-refresh-interval-seconds", "600", report=report)

    assert refresh.calls == []
    assert json.loads(capsys.readouterr().out)["status"] == "fresh_export_retained"


def _guard_report_case(tmp_path: Path, case: str) -> Path:
    report = tmp_path / "guard-report.json"
    if case == "missing":
        return report
    if case == "older_than_the_interval":
        _write_guard_report(report, age=timedelta(minutes=11))
    elif case == "generated_in_the_future":
        _write_guard_report(report, age=-timedelta(minutes=10))
    elif case == "naive_timestamp":
        _write_guard_report(report, generated_at="2026-11-02T16:00:00")
    elif case == "no_billing_section":
        _write_guard_report(report, billing_reconciliation=None)
    elif case == "live_providers_not_a_list":
        _write_guard_report(report)
        value = json.loads(report.read_text(encoding="utf-8"))
        value["billing_reconciliation"]["live_providers"] = "vast"
        report.write_text(json.dumps(value), encoding="utf-8")
    elif case == "malformed":
        report.write_text("{not json", encoding="utf-8")
        report.chmod(0o600)
    elif case == "symlinked":
        real = tmp_path / "elsewhere-report.json"
        _write_guard_report(real)
        report.symlink_to(real)
    elif case == "group_writable":
        _write_guard_report(report)
        report.chmod(0o620)
    else:
        raise AssertionError(case)
    return report


@pytest.mark.parametrize(
    "case",
    [
        "missing", "older_than_the_interval", "generated_in_the_future", "naive_timestamp",
        "no_billing_section", "live_providers_not_a_list", "malformed", "symlinked", "group_writable",
    ],
)
def test_without_a_current_trustworthy_guard_report_every_run_refreshes(
    tmp_path: Path, refresh: _Refresh, case: str
) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))

    _main(tmp_path, export, "--min-refresh-interval-seconds", "600",
          report=_guard_report_case(tmp_path, case))

    assert len(refresh.calls) == 1


def test_without_a_named_guard_report_every_run_refreshes(
    tmp_path: Path, refresh: _Refresh, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_GPU_SPEND_GUARD_REPORT", raising=False)
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))

    billing.main([
        "--secrets-dir", str(tmp_path / "secrets"), "--billing-export", str(export),
        "--audit-root", str(tmp_path / "billing-audit"), "--cohort-start-at", "2026-01-01T00:00:00Z",
        "--min-refresh-interval-seconds", "600",
    ])

    assert len(refresh.calls) == 1


def test_without_an_interval_every_run_refreshes(tmp_path: Path, refresh: _Refresh) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(seconds=5))

    _main(tmp_path, export)
    _main(tmp_path, export, "--min-refresh-interval-seconds", "0")
    _main(tmp_path, export, "--min-refresh-interval-seconds", "nan")

    assert len(refresh.calls) == 3


def test_an_interval_can_never_stretch_past_an_hour(tmp_path: Path, refresh: _Refresh) -> None:
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=61))

    _main(tmp_path, export, "--min-refresh-interval-seconds", "100000")

    assert len(refresh.calls) == 1
    assert billing.MAX_MIN_REFRESH_INTERVAL_SECONDS == 3600


def _unit(name: str) -> str:
    return (UNITS / name).read_text(encoding="utf-8")


def test_the_deployed_interval_stays_well_inside_the_guards_freshness_rule() -> None:
    service = _unit("blueprint-provider-billing-reconciler.service")
    declared = re.search(r"^Environment=BLUEPRINT_PROVIDER_BILLING_MIN_REFRESH_SECONDS=(\d+)$", service, re.M)
    assert declared is not None
    interval = int(declared[1])
    assert interval == 600, "the reconciler's intended ten-minute cadence"
    assert '--min-refresh-interval-seconds "$${BLUEPRINT_PROVIDER_BILLING_MIN_REFRESH_SECONDS:-600}"' in service
    # Both consumers fail closed on billing evidence older than a day.
    guard_bound = guard.reconcile_billing_export.__kwdefaults__["max_age_seconds"]
    assert guard_bound == MAX_BILLING_AGE_SECONDS == 24 * 60 * 60
    assert interval <= billing.MAX_MIN_REFRESH_INTERVAL_SECONDS <= guard_bound / 24
    # The skip reads the report the guard itself writes, at the same path.
    guard_report = re.search(
        r"^Environment=BLUEPRINT_GPU_SPEND_GUARD_REPORT=(\S+)$", _unit("blueprint-gpu-spend-guard.service"), re.M
    )
    assert guard_report is not None
    assert f"Environment=BLUEPRINT_GPU_SPEND_GUARD_REPORT={guard_report[1]}" in service
    assert '--spend-guard-report "$${BLUEPRINT_GPU_SPEND_GUARD_REPORT}"' in service


def test_the_spend_guard_still_pulls_a_refresh_before_every_run() -> None:
    """The dependency is what makes a skipped refresh safe: a stale export is refreshed first."""
    spend_guard = _unit("blueprint-gpu-spend-guard.service")
    unit = dict(
        line.split("=", 1) for line in spend_guard.split("[Service]", 1)[0].splitlines() if "=" in line
    )
    assert "blueprint-provider-billing-reconciler.service" in unit["Wants"].split()
    assert "blueprint-provider-billing-reconciler.service" in unit["After"].split()
    assert "--require-billing-reconciliation" in spend_guard


def test_the_retained_path_is_reached_before_any_provider_module_loads(tmp_path: Path) -> None:
    """A skipped refresh must cost an interpreter start, not the provider inventory imports."""
    export = tmp_path / "provider_billing_export.json"
    _write_export(export, age=timedelta(minutes=1))
    report = tmp_path / "guard-report.json"
    _write_guard_report(report)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from blueprint_pipeline import provider_billing_reconciler as billing;"
        "rc = billing.main(['--secrets-dir', sys.argv[2], '--billing-export', sys.argv[3],"
        " '--audit-root', sys.argv[2], '--cohort-start-at', '2026-01-01T00:00:00Z',"
        " '--min-refresh-interval-seconds', '600', '--spend-guard-report', sys.argv[4]]);"
        "assert rc == 0;"
        "assert 'blueprint_pipeline.cloud_vm_render_providers' not in sys.modules"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT / "src"), str(tmp_path / "unused"), str(export),
         str(report)],
        capture_output=True, text=True, timeout=120, check=False,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["status"] == "fresh_export_retained"
