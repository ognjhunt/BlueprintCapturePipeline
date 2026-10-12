# Covers (for impacted-test selection):
#   scripts/replay_website_assessment_resume.py
"""ADP-010/day7: the host replay cannot become a mutation or paid launcher."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline import website_assessment_resume as resume
from blueprint_pipeline import website_task_context as context_reader
from tests.test_website_assessment_resume import _claim_guest

pytest_plugins = ("tests.test_website_assessment_resume",)


@pytest.fixture
def replay(waiting, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/replay_website_assessment_resume.py"
    spec = importlib.util.spec_from_file_location("replay_website_assessment_resume", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=os.geteuid()))
    monkeypatch.setattr(driver.os, "statvfs", lambda _: SimpleNamespace(f_flag=os.ST_RDONLY))
    monkeypatch.setattr(sys, "argv", [str(path), "--candidate-src", str(path.parents[1] / "src"),
        "--candidate-commit", "a" * 40, "--capture-root", str(waiting[0])])
    for name in list(os.environ):
        if any(word in name.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")) and name != "PIPELINE_SYNC_TOKEN":
            monkeypatch.delenv(name)
    return driver


@pytest.mark.parametrize("prepared_source", [None, "guest"], indirect=True)
def test_read_only_replay_passes_stored_context_and_never_enters_mutators(replay, waiting, monkeypatch, capsys):
    if waiting[1].get("purpose") == "scene_preparation":
        _claim_guest(waiting, monkeypatch)
    root = waiting[0]
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(context_reader, "load_website_scene_sponsorship", lambda **_: pytest.fail("sponsorship called"))
    monkeypatch.setattr(listener, "process_handoff_payload", lambda *_, **__: pytest.fail("producer called"))
    monkeypatch.setattr(resume, "reconcile_waiting_assessments", lambda *_, **__: pytest.fail("mutating reconciliation called"))
    assert replay.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed" and result["steps"][-1] == "stored_context"
    assert result["ledger_bytes_unchanged"] is True
    assert result["owner_check"] == ("passed_verified_guest_claim" if waiting[1].get("purpose") else "passed_unchanged_birth_owner")
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_replay_refuses_provider_credentials_without_printing_values(replay, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-value")
    assert replay.main() == 1
    output = capsys.readouterr().out
    assert "replay_provider_credentials_not_withheld" in output
    assert "synthetic-provider-value" not in output


def test_replay_refuses_root_and_writable_capture_mount(replay, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(replay.os, "geteuid", lambda: 0)
        with pytest.raises(SystemExit, match="replay_requires_blueprint_service_user"):
            replay.main()
    monkeypatch.setattr(replay.os, "statvfs", lambda _: SimpleNamespace(f_flag=0))
    with pytest.raises(SystemExit, match="replay_requires_read_only_capture_mount"):
        replay.main()
