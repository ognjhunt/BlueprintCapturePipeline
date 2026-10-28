"""AUT-17/18: route evidence, safe publication, and independent door readback."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight
from blueprint_pipeline.production_preflight_summary import (
    PreflightSummaryError, build_safe_summary, write_safe_summary,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.status import _preflight, _unit_health  # noqa: E402


def _report(*, observed: float | None = None) -> dict:
    when = time.time() if observed is None else observed
    return {"schema_version": preflight.SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when)),
            "active_sha": "a" * 40, "lane": "beta_rigid", "host_findings": [],
            "units": {"blueprint-beta-policy-screening.service": {
                "configuration_sha256": "b" * 64,
                "coverage": {"status": "selected_route_contract"},
                "effective_environment": {"PRIVATE_VALUE": "not-for-publication"},
                "findings": [{"severity": "blocker", "code": "required_path_access_refused",
                              "path": "/private/customer-site", "error": "not-for-publication"}],
            }}}


def _summary(report: dict) -> dict:
    return build_safe_summary(report, raw_report_bytes=json.dumps(report).encode())


def test_summary_projects_only_fixed_safe_fields_and_groups_causes() -> None:
    report = _report()
    report["units"]["blueprint-beta-policy-screening.service"]["findings"].append(
        {"severity": "blocker", "code": "required_path_access_refused", "url": "https://private.example"})
    summary = _summary(report)
    encoded = json.dumps(summary)
    assert "not-for-publication" not in encoded and "customer-site" not in encoded and "private.example" not in encoded
    assert summary["counts"]["blocker"] == 2
    assert len(summary["findings"]) == 1 and summary["findings"][0]["count"] == 2
    assert summary["findings"][0]["next_action"] == "inspect_selected_route_access_contract"
    assert summary["proof_boundary"] == "selected_route_plumbing_only"


@pytest.mark.parametrize("field,value", [("active_sha", "unknown"), ("active_sha", ""), ("active_sha", None),
                                        ("generated_at", "tomorrow"),
                                        ("units", []), ("host_findings", {})])
def test_malformed_report_is_not_a_safe_summary(field: str, value: object) -> None:
    report = _report()
    report[field] = value
    with pytest.raises(PreflightSummaryError):
        _summary(report)


def test_raw_private_modes_and_exact_current_safe_door_readback(tmp_path: Path) -> None:
    raw_parent = tmp_path / "preflight"
    raw_parent.mkdir(mode=0o700)
    raw = raw_parent / "latest.json"
    report = _report()
    raw.write_text(json.dumps(report))
    raw.chmod(0o600)
    target = raw_parent / "public" / "summary.json"
    write_safe_summary(target, _summary(report), private_parent=raw_parent,
                       reader_ids=(os.getuid(), os.getgid()))
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    assert stat.S_IMODE(raw_parent.stat().st_mode) == 0o711
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    releases = tmp_path / "releases" / ("a" * 40)
    releases.mkdir(parents=True)
    active = tmp_path / "active"
    active.symlink_to(releases)
    result = _preflight(dataclasses.replace(DoorConfig(), preflight_summary=str(target), active_release_link=str(active)))
    assert result["status"] == "observed"
    assert result["report_sha256"] == _summary(report)["report_sha256"]
    # The public reader never needs to open the raw private file.
    raw.chmod(0)
    try:
        assert _preflight(dataclasses.replace(DoorConfig(), preflight_summary=str(target), active_release_link=str(active)))["status"] == "observed"
    finally:
        raw.chmod(0o600)


def test_summary_rejects_symlink_and_untraversable_ancestor(tmp_path: Path) -> None:
    public = tmp_path / "private" / "public"
    public.mkdir(parents=True)
    target = public / "summary.json"
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(PreflightSummaryError, match="symlink"):
        write_safe_summary(target, _summary(_report()), reader_ids=(os.getuid(), os.getgid()))
    target.unlink()
    public.parent.chmod(0o700)
    with pytest.raises(PreflightSummaryError, match="ancestor_not_traversable"):
        write_safe_summary(target, _summary(_report()), reader_ids=(os.getuid() + 10000, os.getgid() + 10000))


def test_interrupted_summary_replace_preserves_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "public" / "summary.json"
    target.parent.mkdir()
    target.write_bytes(b"old summary")
    from blueprint_pipeline import production_preflight_summary as summaries
    monkeypatch.setattr(summaries.os, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(PreflightSummaryError):
        write_safe_summary(target, _summary(_report()), reader_ids=(os.getuid(), os.getgid()))
    assert target.read_bytes() == b"old summary"
    assert not list(target.parent.glob(".preflight-summary-*"))


@pytest.mark.parametrize("case,reason", [("stale", "stale_or_future"), ("release", "release_mismatch"),
                                         ("changed", "digest_mismatch"), ("secret", "secret_shaped")])
def test_door_refuses_stale_changed_or_secret_shaped_summary(tmp_path: Path, case: str, reason: str) -> None:
    report = _report(observed=time.time() - 4000 if case == "stale" else None)
    if case == "release":
        report["active_sha"] = "c" * 40
    summary = _summary(report)
    if case == "changed":
        summary["counts"]["blocker"] += 1
    elif case == "secret":
        summary["unexpected"] = "ghp_" + "a" * 36
    target = tmp_path / "summary.json"
    target.write_text(json.dumps(summary))
    releases = tmp_path / ("a" * 40)
    releases.mkdir()
    active = tmp_path / "active"
    active.symlink_to(releases)
    result = _preflight(dataclasses.replace(DoorConfig(), preflight_summary=str(target), active_release_link=str(active)))
    assert result["status"] == "refused" and reason in result["reason"]


def _probe_args(tmp_path: Path) -> object:
    package = tmp_path / "release" / "src" / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "beta_policy_screening_worker.py").write_text(
        'def unused():\n    from .legacy import OLD\n'
        'TEMPLATE = "/var/lib/blueprint/{root}"\n')
    (package / "legacy.py").write_text('OLD = "/var/lib/blueprint/unselected-legacy"\n')
    return preflight.argparse.Namespace(unit="blueprint-beta-policy-screening.service",
        module="blueprint_pipeline.beta_policy_screening_worker", release=str(tmp_path / "release"),
        active_sha="", read_write_paths="", read_only_paths="", lane="beta_rigid")


def test_scoped_probe_keeps_unused_templates_visible_without_write_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _probe_args(tmp_path)
    queue, inputs = tmp_path / "queue", tmp_path / "inputs"
    queue.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", str(queue))
    monkeypatch.setenv("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", str(inputs))
    writes = []
    monkeypatch.setattr(preflight, "write_probe", lambda path: writes.append(path) or {"status": "writable"})
    report = preflight.run_probe(args)
    assert writes == [queue]
    assert report["paths"][str(inputs)]["status"] == "readable"
    assert report["paths"]["/var/lib/blueprint/{root}"]["code"] == "path_template_unresolved"
    assert report["paths"]["/var/lib/blueprint/unselected-legacy"]["requirement"] == "static_candidate"
    assert not any(finding["severity"] == "blocker" for finding in report["findings"])


def test_required_output_denied_is_still_a_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _probe_args(tmp_path)
    queue, inputs = tmp_path / "queue", tmp_path / "inputs"
    queue.mkdir()
    inputs.mkdir()
    monkeypatch.setenv("BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", str(queue))
    monkeypatch.setenv("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", str(inputs))
    monkeypatch.setattr(preflight, "write_probe", lambda _path: {"status": "read_only", "errno": 30})
    report = preflight.run_probe(args)
    blockers = [row for row in report["findings"] if row["severity"] == "blocker"]
    assert len(blockers) == 1 and blockers[0]["code"] == "required_path_access_refused"


def test_required_queue_template_refuses_before_any_path_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _probe_args(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    template = "/var/lib/blueprint/{root}"
    monkeypatch.setenv("BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", template)
    monkeypatch.setenv("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", str(inputs))
    writes = []
    monkeypatch.setattr(preflight, "write_probe", lambda path: writes.append(path) or {"status": "read_only"})
    report = preflight.run_probe(args)
    assert not writes
    assert report["coverage"]["status"] == "unknown"
    assert report["coverage"]["reason"] == "required_path_binding_unresolved"
    row = report["paths"][template]
    assert row["requirement"] == "required_unresolved" and row["severity"] == "blocker"
    assert any(finding["code"] == "required_path_template_unresolved" and finding["severity"] == "blocker" for finding in report["findings"])


def test_route_health_separates_held_unused_and_broken_required_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    def properties(_trigger: str) -> dict:
        return {"ActiveState": ["inactive"]}
    monkeypatch.setattr(preflight, "unit_properties", properties)
    props = {"ActiveState": ["inactive"], "Result": ["success"], "LoadState": ["loaded"], "TriggeredBy": ["blueprint-x.timer"]}
    unit = "blueprint-beta-policy-screening.service"
    assert preflight.unit_health_checks({unit: {"properties": props}}, lane="beta_rigid")[0]["code"] == "unit_trigger_inactive"
    held = {**props, "Result": ["exec-condition"],
            "ExecCondition": ["{ path=/usr/bin/false ; argv[]=/usr/bin/false ; ignore_errors=no ; code=exited ; status=1 }"]}
    findings = preflight.unit_health_checks({unit: {"properties": held}}, lane="beta_rigid")
    assert all(row["severity"] == "info" and row["code"] == "unit_intentionally_held" for row in findings)
    rows = _unit_health([{"Id": unit, "ActiveState": "inactive", "Type": "oneshot", "Result": "success"}], ["blueprint-old-diagnostic.service"])
    assert rows["current"][0]["state"] == "successful_inactive_oneshot"
    assert rows["historical_diagnostics"] == ["blueprint-old-diagnostic.service"]
