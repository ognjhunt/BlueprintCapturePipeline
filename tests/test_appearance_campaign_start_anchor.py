"""The campaign's spend anchor, when it has no paid predecessor.

The retired compatibility path expected a completed paid predecessor that the
active campaign does not have. The replacement measures the exact live ledgers
rather than inheriting a historical backend dependency.

The replacement must not be "skip the anchor". Its job is to carry the
campaign's running spend into the `prior_spend + hard_cap_usd > aggregate_cap`
check, so dropping it would uncap a paid campaign. These pin that the number
survives and only its source changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_repair_spend_chain import (
    AGGREGATE_GOAL_SPEND_CAP_USD,
    CAMPAIGN_START_SCHEMA_VERSION,
    validate_campaign_start_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "seal_appearance_campaign_start"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sealer = _load()


@pytest.fixture()
def ledgers(tmp_path: Path) -> dict:
    consumed = tmp_path / "consumed"
    consumed.mkdir()
    ledger = tmp_path / "spend_ledger.json"
    ledger.write_text(json.dumps({"instances": [], "daily_spend_usd": 0.0}), encoding="utf-8")
    return {"consumed": consumed, "ledger": ledger, "out": tmp_path / "campaign_start.json"}


def _seal(ledgers, marker: str = "artifixer"):
    return sealer.seal(
        consumed_root=ledgers["consumed"],
        spend_ledger=ledgers["ledger"],
        campaign_marker=marker,
        output_path=ledgers["out"],
    )


def test_an_empty_campaign_measures_zero_and_validates(ledgers) -> None:
    receipt = _seal(ledgers)

    validated, spend = validate_campaign_start_receipt(ledgers["out"])
    assert spend == 0.0
    assert validated["receipt_digest"] == receipt["receipt_digest"]
    assert validated["aggregate_goal_spend_cap_usd"] == AGGREGATE_GOAL_SPEND_CAP_USD


def test_the_ledger_is_bound_even_when_it_names_nothing(ledgers) -> None:
    """"We looked and it was empty" is the claim, so the looking is evidence."""

    receipt = _seal(ledgers)

    assert receipt["measured_from"], "a zero with no evidence is an assertion"
    bound = receipt["measured_from"][0]
    assert bound["path"] == str(ledgers["ledger"].resolve())
    assert bound["sha256"] == "sha256:" + hashlib.sha256(ledgers["ledger"].read_bytes()).hexdigest()


def test_a_consumed_attempt_is_measured_into_the_anchor(ledgers) -> None:
    """A real prior attempt has to raise prior spend, or the cap is fiction."""

    (ledgers["consumed"] / "artifixer3d-attempt-1.json").write_text(
        json.dumps({"authorization_digest": "sha256:" + "a" * 64, "terminal_cost_usd": 2.5}),
        encoding="utf-8",
    )

    receipt = _seal(ledgers)

    assert receipt["prior_goal_spend_usd"] == 2.5
    assert len(receipt["measured_paid_attempts"]) == 1
    _, spend = validate_campaign_start_receipt(ledgers["out"])
    assert spend == 2.5


def test_an_attempt_from_another_campaign_is_not_counted(ledgers) -> None:
    (ledgers["consumed"] / "joint-agent-attempt.json").write_text(
        json.dumps({"authorization_digest": "sha256:" + "b" * 64, "terminal_cost_usd": 9.0}),
        encoding="utf-8",
    )

    assert _seal(ledgers)["prior_goal_spend_usd"] == 0.0


def test_a_receipt_that_names_attempts_but_claims_zero_is_refused(ledgers) -> None:
    """The shape a fabricated anchor takes; believing it would uncap the cap."""

    receipt = _seal(ledgers)
    receipt["measured_paid_attempts"] = [{"source": "consumed_authority", "cost_usd": 5.0}]
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    ledgers["out"].write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_campaign_start_receipt(ledgers["out"])

    assert "spend_disagrees_with_evidence" in str(excinfo.value)


def test_an_edited_spend_is_refused_by_its_own_digest(ledgers) -> None:
    receipt = _seal(ledgers)
    receipt["prior_goal_spend_usd"] = 0.0
    receipt["measured_paid_attempts"] = []
    # Digest left stale on purpose: this is the cheapest possible forgery.
    ledgers["out"].write_text(json.dumps(receipt), encoding="utf-8")
    original = json.loads(ledgers["out"].read_text(encoding="utf-8"))
    original["prior_goal_spend_usd"] = 11.0
    ledgers["out"].write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_campaign_start_receipt(ledgers["out"])


def test_evidence_that_moved_since_sealing_is_refused(ledgers) -> None:
    """A bound record whose file changed no longer supports the measurement."""

    _seal(ledgers)
    ledgers["ledger"].write_text(json.dumps({"instances": [{"x": 1}]}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_campaign_start_receipt(ledgers["out"])

    assert "evidence_unbound" in str(excinfo.value)


def test_measuring_a_missing_ledger_refuses_rather_than_reporting_zero(tmp_path: Path) -> None:
    """Absence of a ledger is not evidence of absence of spend."""

    consumed = tmp_path / "consumed"
    consumed.mkdir()

    with pytest.raises(ValueError) as excinfo:
        sealer.seal(
            consumed_root=consumed,
            spend_ledger=tmp_path / "absent.json",
            campaign_marker="artifixer",
            output_path=tmp_path / "out.json",
        )

    assert "spend_ledger_missing" in str(excinfo.value)


def test_the_schema_version_is_the_one_the_validator_expects() -> None:
    assert CAMPAIGN_START_SCHEMA_VERSION == "appearance_campaign_spend_start.v1"


def test_live_campaign_start_help_contains_no_retired_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        sealer.main(["--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "Aura" not in help_text
    assert "Measure" in help_text
