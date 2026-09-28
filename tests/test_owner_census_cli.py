"""Issuer/report CLI parsing stays bounded and never silently runs a census."""

# Covers (for impacted-test selection):
#   scripts/lane_scratch_census.py
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_lane_owner_consents as consent
from tests.test_owner_census_issuance import issuer as _issuer_fixture

issuer = _issuer_fixture


def script():
    path = Path(__file__).parents[1] / "scripts/lane_scratch_census.py"
    spec = importlib.util.spec_from_file_location("owner_census_script_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "argv",
    [
        ["--issue-owner-consent", "--unexpected=" + ("PRIVATE" * 400)],
        ["--issue-owner-consent", "--expected-census-size-bytes", "PRIVATE"],
        ["--issue-owner-consent", "--selected-path"],
        ["--iss", "--process-root", "/proc"],
    ],
)
def test_issue_parser_fixed_json_no_argv_no_scan(argv, capsys, monkeypatch):
    module = script()
    monkeypatch.setattr(module, "build_census", lambda **k: pytest.fail("issuance scanned"))
    assert module.main(argv) == 1
    captured = capsys.readouterr()
    assert not captured.err and len(captured.out) < 4096 and "PRIVATE" not in captured.out
    assert json.loads(captured.out)["blockers"] == ["owner_consent_options_invalid"]


def test_existing_cli_connects_to_issuer_and_never_uses_old_roots_as_authority(monkeypatch, capsys):
    module = script()
    seen = []
    monkeypatch.setattr(module, "build_census", lambda **k: pytest.fail("issuance scanned"))
    monkeypatch.setattr(
        consent,
        "_run_issue",
        lambda *a, **k: seen.append((a, k)) or b'{"status":"owner_consent_issued","mutations":0}\n',
    )
    args = [
        "--issue-owner-consent",
        "--census",
        "/retained/census.json",
        "--annotations",
        "/retained/annotations.json",
        "--expected-census-sha256",
        "sha256:" + "a" * 64,
        "--expected-census-size-bytes",
        "10",
        "--expected-annotations-sha256",
        "sha256:" + "b" * 64,
        "--expected-annotations-size-bytes",
        "20",
        "--principal",
        "operator",
        "--selected-path",
        "/work/one",
        "--consent-expires-at-epoch",
        "2000",
    ]
    assert module.main(args) == 0
    assert seen[0][1]["selected_paths"] == ["/work/one"] and "allowed_roots" not in seen[0][1]
    assert json.loads(capsys.readouterr().out)["mutations"] == 0
    assert module.main(args + ["--work-root", "/forged"]) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["report", "--unexpected=" + ("PRIVATE" * 400)],
        ["report", "--consent-id"],
        [
            "report",
            "--consent-id",
            "a" * 32,
            "--expected-sha256",
            "sha256:" + "b" * 64,
            "--expected-size-bytes",
            "PRIVATE",
        ],
    ],
)
def test_report_parser_bounded_typed(argv, capsys):
    assert consent.main(argv) == 1
    captured = capsys.readouterr()
    assert not captured.err and "PRIVATE" not in captured.out and len(captured.out) < 4096
    assert json.loads(captured.out)["blockers"] == ["owner_consent_options_invalid"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--principal", "operator"],
        ["--selected-path", "/work/a"],
        ["--expected-census-size-bytes", "10"],
        ["--door-config", "/forged.json"],
    ],
)
def test_issuance_only_options_without_mode_refuse_before_ordinary_scan(argv, monkeypatch, capsys):
    module = script()
    monkeypatch.setattr(
        module, "build_census", lambda **k: pytest.fail("incomplete issuance scanned")
    )
    assert module.main(argv) == 1
    assert json.loads(capsys.readouterr().out)["blockers"] == ["owner_consent_options_invalid"]


def test_issue_cli_encodes_fixed_success_under_same_open_budget(issuer, monkeypatch, capsys):
    module = script()
    paths, _, args, _ = issuer
    previous = consent._installed_config
    budgets = []
    dump = json.dumps

    def acquire(files, path):
        budgets.append(files.budget)
        return previous(files, path)

    def encode(*a, **kw):
        if budgets:
            assert not budgets[-1].closed, "encoded issuance after invocation closed"
        return dump(*a, **kw)

    monkeypatch.setattr(consent, "_installed_config", acquire)
    monkeypatch.setattr(json, "dumps", encode)
    monkeypatch.setattr(module.time, "time", lambda: args["now"])
    monkeypatch.setattr(module.time, "monotonic", lambda: 0)
    argv = [
        "--issue-owner-consent",
        "--census",
        str(paths[0]),
        "--annotations",
        str(paths[1]),
        "--expected-census-sha256",
        args["census_sha256"],
        "--expected-census-size-bytes",
        str(args["census_size_bytes"]),
        "--expected-annotations-sha256",
        args["annotations_sha256"],
        "--expected-annotations-size-bytes",
        str(args["annotations_size_bytes"]),
        "--principal",
        args["principal"],
        "--selected-path",
        args["selected_paths"][0],
        "--consent-expires-at-epoch",
        str(args["expires_at_epoch"]),
        "--door-config",
        str(args["installed_config_path"]),
    ]
    assert module.main(argv) == 0
    assert budgets[-1].closed
    assert json.loads(capsys.readouterr().out)["status"] == "owner_consent_issued"
