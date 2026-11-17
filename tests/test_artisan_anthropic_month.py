"""Anthropic's list-price spend always reaches the account's month, as its own labelled provider row.

Each test runs the real month path: the paid-spend journal on disk, ``refresh_report`` reading it
with the host's provider readings, the screening ledger and the policy (``tests.test_account_monthly_spend``).
"""
from __future__ import annotations

import json
import zipfile

import pytest

from blueprint_pipeline import account_monthly_spend_report as report_module
from blueprint_pipeline import artisan_anthropic_month as month
from blueprint_pipeline import artisan_provider_choice as choice
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests import test_account_monthly_spend as account_tests

NOW = account_tests.NOW
host = account_tests.host  # the month's real inputs on disk

TERMS = "sha256:" + "a" * 64
RECEIPT = {"run_id": "website-run-1", "replacement_authoring_text_provider": "anthropic"}
AUTHORITY = {"external_service_spend_caps": {"openai": {"stage_max_cost_usd": {"content_agents": 12.0}}}}
BASIS = "metered_list_price_not_officially_read"


@pytest.fixture(autouse=True)
def _no_host_folders(tmp_path, monkeypatch):
    """Nothing here may write to the host's own month journal or grant records."""
    from blueprint_pipeline import paid_spend_meter
    monkeypatch.setattr(paid_spend_meter, "DEFAULT_METER_ROOT", tmp_path / "unexpected" / "meter")
    monkeypatch.setattr(choice, "DEFAULT_SUCCESSOR_ROOT", tmp_path / "unexpected" / "choices")
    yield
    assert not (tmp_path / "unexpected").exists()


def _report():
    return report_module.refresh_report(now=NOW)


def _others(report):
    return {name: row for name, row in report["providers"].items() if name != "anthropic"}


def _grant(**changes):
    value = {"schema_version": "website_scene_sponsorship.v1", "request_id": "req-9",
             "authoring_providers": ["openai", "anthropic"], "anthropic_provider_terms_reference": TERMS,
             "privacy_policy_version": "2026-11-13.1", "site_consents": [], **changes}
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


@pytest.fixture
def chooser(tmp_path, monkeypatch):
    key = tmp_path / "anthropic.key"
    key.write_text("issued")
    key.chmod(0o600)
    monkeypatch.setenv(choice.SUCCESSOR_ROOT_ENV, str(tmp_path / "choices"))
    monkeypatch.setenv(choice.PREFER_ANTHROPIC_ENV, "1")  # the host as chosen, read again at each attempt
    monkeypatch.setenv(choice.KEY_FILE_ENV, str(key))
    environ = {choice.PREFER_ANTHROPIC_ENV: "1", choice.KEY_FILE_ENV: str(key),
               choice.SUCCESSOR_ROOT_ENV: str(tmp_path / "choices")}
    calls = []

    def read(status=choice.MODEL_READ_PASSED):
        return lambda: calls.append(1) or {"status": status, "model_metadata_digest": "sha256:" + "b" * 64}

    def choose(root, grant, **read_args):
        root.mkdir(parents=True, exist_ok=True)
        return choice.retained_or_new_choice(root=root, grant=grant, environ=environ,
                                             read=read(**read_args))
    return choose, calls


def _archive(path, calls, *, configuration=None, run_id="website-run-1"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(month.USAGE_MEMBER, json.dumps({"schema_version": "artisan_upstream_usage.v1", "calls": calls}))
        archive.writestr(month.STAGE_INPUT_MEMBER, json.dumps({"run_id": run_id, "configuration": configuration or {}}))
    return path


def _row(kind, cost, *, status="priced", reserved=None, origin="upstream_priced_usage"):
    return {"id": f"{kind}-{cost}", "kind": kind, "status": status, "cost_usd": cost if status == "priced" else None,
            "reserved_usd": reserved if reserved is not None else cost, "cost_origin": origin}


def test_an_attempt_is_held_at_its_allowance_then_settles_to_its_anthropic_subtotal(host, tmp_path):
    before = _report()
    assert month.hold_attempt(RECEIPT, AUTHORITY, ttl_seconds=7200, now=NOW - 100)
    held = _report()
    assert held["held_usd"] == pytest.approx(before["held_usd"] + 12.0) and _others(held) == _others(before)
    # Text $2 priced, one call never priced counted at its $0.5 bound; the OpenAI board image ($0.3,
    # in OpenAI's own reading) is left out.
    archive = _archive(tmp_path / "out.zip", [_row("messages", 2.0),
                                              _row("images", 0.3), _row("messages", None, status="in_flight", reserved=0.5)])
    assert month.settle_attempt(RECEIPT, archive, now=NOW - 50)
    settled = _report()
    assert settled["held_usd"] == pytest.approx(before["held_usd"])
    row = settled["providers"]["anthropic"]
    assert (row["status"], row["cost_basis"], row["counted_usd"]) == ("metered_only", BASIS, 2.5)
    assert _others(settled) == _others(before)
    assert settled["counted_usd"] == pytest.approx(before["counted_usd"] + 2.5)


def test_a_lost_run_keeps_its_hold_counted(host, tmp_path):
    before = _report()
    month.hold_attempt(RECEIPT, AUTHORITY, ttl_seconds=7200, now=NOW - 100)
    assert month.settle_attempt(RECEIPT, None) is False
    assert month.settle_attempt(RECEIPT, tmp_path / "missing.zip") is False
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    assert month.settle_attempt(RECEIPT, broken) is False
    assert _report()["held_usd"] == pytest.approx(before["held_usd"] + 12.0)


def test_a_job_with_no_anthropic_rows_writes_nothing_and_the_month_is_unchanged(host, tmp_path):
    from blueprint_pipeline.paid_spend_meter import read_journal
    import os
    before_events, _ = read_journal(__import__("pathlib").Path(os.environ["BLUEPRINT_PAID_SPEND_METER_ROOT"]))
    before = _report()
    openai = {"run_id": "website-run-2"}
    assert month.hold_attempt(openai, AUTHORITY, ttl_seconds=7200) is False
    assert month.settle_attempt(openai, _archive(tmp_path / "o.zip", [_row("images", 0.3)])) is False
    after_events, _ = read_journal(__import__("pathlib").Path(os.environ["BLUEPRINT_PAID_SPEND_METER_ROOT"]))
    assert after_events == before_events
    after = _report()
    assert {k: v for k, v in after.items() if k not in {"generated_at", "forecast"}} == {
        k: v for k, v in before.items() if k not in {"generated_at", "forecast"}}


def test_the_largest_of_reading_or_settled_cannot_hide_anthropic(host, tmp_path):
    # Anthropic has no reading (it is not an account provider), so what counts is its meter. A large
    # OpenAI reading leaves Anthropic's row as it was.
    month.hold_attempt(RECEIPT, AUTHORITY, ttl_seconds=7200, now=NOW - 100)
    month.settle_attempt(RECEIPT, _archive(tmp_path / "a.zip", [_row("messages", 0.25)]), now=NOW - 50)
    first = _report()["providers"]["anthropic"]["counted_usd"]
    account_tests._observation(host["root"], openai=290.0)
    second = _report()
    assert first == second["providers"]["anthropic"]["counted_usd"] == 0.25
    assert second["spend_usd"] == pytest.approx(sum(row["counted_usd"] for row in second["providers"].values()))


def test_a_no_funds_end_makes_the_grant_openai_for_its_later_attempts(host, chooser, tmp_path):
    choose, calls = chooser
    grant = _grant()
    first = choose(tmp_path / "prep", grant)
    assert first["provider"] == "anthropic" and len(calls) == 1
    configuration = {"authoring_models": dict(choice.ARTISAN_CLAUDE_MODELS), "artisan_provider_choice": first}
    refused = _row("messages", 0.0, origin=month.NO_FUNDS_ORIGIN)
    archive = _archive(tmp_path / "out.zip", [_row("messages", 1.0), refused], configuration=configuration)
    month.hold_attempt(RECEIPT, AUTHORITY, ttl_seconds=7200, now=NOW - 100)
    assert month.settle_attempt(RECEIPT, archive, now=NOW - 50)
    successor = choice.retained_successor(grant["authority_digest"])
    assert successor["provider"] == "openai" and successor["reason"] == choice.NO_FUNDS_REASON
    assert successor["ended_attempt_run_id"] == "website-run-1"
    assert successor["supersedes"] == {"digest": first["digest"], "provider": "anthropic"}
    assert successor["model_read"] is None and choice.validate_choice(successor) == successor
    # The next attempt under the grant: its handoff reads OpenAI, choosing nothing ...
    assert choose(tmp_path / "prep", grant) == successor and len(calls) == 1
    # ... a restart midway reads the same answer ...
    assert choose(tmp_path / "prep-restart", grant) == successor and len(calls) == 1
    # ... and a stage 3 configuration frozen on Anthropic is rebound to OpenAI.
    rebound = choice.with_provider_successor(configuration)
    from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS
    assert rebound["authoring_models"] == ARTISAN_MODELS and rebound["artisan_provider_choice"] == successor
    assert choice.artisan_text_provider(rebound) == "openai"
    # A new grant chooses afresh.
    renewed = choose(tmp_path / "prep-2", _grant(max_paid_attempts=2))
    assert renewed["provider"] == "anthropic" and len(calls) == 2
    assert choice.with_provider_successor(
        {"authoring_models": dict(choice.ARTISAN_CLAUDE_MODELS), "artisan_provider_choice": renewed})[
            "authoring_models"] == choice.ARTISAN_CLAUDE_MODELS


def test_only_a_typed_no_funds_end_of_an_anthropic_choice_writes_a_successor(host, chooser, tmp_path):
    choose, _ = chooser
    grant = _grant()
    first = choose(tmp_path / "prep", grant)
    configuration = {"artisan_provider_choice": first}
    endings = ([_row("messages", 0.0, origin="upstream_rejected_request")],  # a $0 rejection, not no-funds
               [_row("messages", 0.0, origin=month.NO_FUNDS_ORIGIN), _row("messages", 1.0)],  # not the last call
               [_row("images", 0.0, origin=month.NO_FUNDS_ORIGIN)])  # the board's refusal is OpenAI's
    for index, calls in enumerate(endings):
        month.settle_attempt(RECEIPT, _archive(tmp_path / f"e{index}.zip", calls, configuration=configuration))
    other_run = _archive(tmp_path / "other.zip", [_row("messages", 0.0, origin=month.NO_FUNDS_ORIGIN)],
                         configuration=configuration, run_id="website-run-other")
    month.settle_attempt(RECEIPT, other_run)
    assert choice.retained_successor(grant["authority_digest"]) is None
    openai_choice = choose(tmp_path / "prep-o", _grant(max_paid_attempts=3), status="claude_model_metadata_unavailable")
    month.settle_attempt(RECEIPT, _archive(tmp_path / "o.zip", [_row("messages", 0.0, origin=month.NO_FUNDS_ORIGIN)],
                                           configuration={"artisan_provider_choice": openai_choice}))
    assert choice.retained_successor(openai_choice["grant_authority_digest"]) is None


def test_a_hold_that_cannot_be_written_on_the_host_raises(host, tmp_path, monkeypatch):
    from blueprint_pipeline.paid_spend_meter import METER_ROOT_ENV
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    monkeypatch.setenv(METER_ROOT_ENV, str(blocked))
    with pytest.raises(OSError):
        month.hold_attempt(RECEIPT, AUTHORITY, ttl_seconds=7200, now=NOW - 100)
