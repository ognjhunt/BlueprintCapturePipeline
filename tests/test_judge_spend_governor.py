"""Tests for judge inference spend governance."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import judge_spend_governor as gov
from blueprint_pipeline import roboworld_progress_judge as judge


def _policy(**overrides):
    kwargs = {
        "campaign_id": "cohort-1",
        "usd_per_1k_input_tokens": 0.10,
        "usd_per_1k_output_tokens": 0.40,
        "estimated_tokens_per_frame": 250,
        "target_spend_usd": 1.00,
        "hard_cap_usd": 2.00,
        "max_requests": 100,
        "max_frames": 10_000,
        "ttl_seconds": 3600,
    }
    kwargs.update(overrides)
    return gov.build_judge_spend_policy(**kwargs)


def test_policy_requires_operator_supplied_rates() -> None:
    """Spend you cannot price is spend you cannot govern."""

    unpriced = gov.build_judge_spend_policy(campaign_id="c", usd_per_1k_input_tokens=None)

    assert unpriced["status"] == "blocked"
    assert "judge_spend_rates_not_operator_supplied" in unpriced["blockers"]
    assert unpriced["priceable"] is False
    assert (
        unpriced["claim_boundary"]["rates_are_operator_supplied_not_blueprint_measurements"]
        is True
    )


def test_policy_rejects_a_target_above_its_hard_cap() -> None:
    policy = _policy(target_spend_usd=5.0, hard_cap_usd=2.0)
    assert "judge_spend_target_above_hard_cap" in policy["blockers"]


def test_frames_are_priced_as_tokens_when_no_per_image_rate_exists() -> None:
    policy = _policy()
    cost = gov.estimate_request_cost_usd(policy=policy, frame_count=60)
    # 60 frames * 250 tokens = 15000 input tokens at $0.10/1k.
    assert cost == pytest.approx(1.5)


def test_per_image_rate_takes_precedence_over_token_estimation() -> None:
    policy = _policy(usd_per_image=0.002, estimated_tokens_per_frame=250)
    cost = gov.estimate_request_cost_usd(policy=policy, frame_count=60)
    assert cost == pytest.approx(0.12)


def test_unpriceable_request_is_denied_not_waved_through() -> None:
    policy = dict(_policy())
    policy["rates"] = {
        "usd_per_1k_input_tokens": None,
        "usd_per_1k_output_tokens": None,
        "usd_per_image": None,
    }
    governor = gov.JudgeSpendGovernor(policy=policy)

    decision = governor.authorize(frame_count=10)

    assert decision["authorized"] is False
    assert "judge_spend_request_not_priceable" in decision["blockers"]


def test_hard_cap_bounds_total_spend_rather_than_being_noticed_late() -> None:
    governor = gov.JudgeSpendGovernor(policy=_policy(hard_cap_usd=2.0))

    first = governor.authorize(frame_count=60)  # $1.50 projected
    assert first["authorized"] is True
    governor.settle(frame_count=60, estimated_cost_usd=first["estimated_cost_usd"])

    # The next request would project past the cap, so it is refused before the
    # provider is contacted.
    second = governor.authorize(frame_count=60)
    assert second["authorized"] is False
    assert "judge_spend_hard_cap_reached" in second["blockers"]
    assert governor.spent_usd == pytest.approx(1.5)


def test_cohort_stops_hard_once_the_cap_is_reached() -> None:
    governor = gov.JudgeSpendGovernor(policy=_policy(hard_cap_usd=1.0))
    decision = governor.authorize(frame_count=20)
    governor.settle(frame_count=20, actual_cost_usd=1.0)

    assert governor.stopped is True
    # Every later request is denied, including cheap ones.
    later = governor.authorize(frame_count=1)
    assert later["authorized"] is False
    assert "judge_spend_hard_cap_reached" in later["blockers"]
    assert decision["authorized"] is True


def test_request_and_frame_ceilings_stop_the_cohort() -> None:
    requests_bound = gov.JudgeSpendGovernor(policy=_policy(max_requests=1))
    requests_bound.settle(frame_count=1, actual_cost_usd=0.01)
    assert requests_bound.authorize(frame_count=1)["blockers"] == [
        "judge_spend_request_ceiling_reached"
    ]

    frames_bound = gov.JudgeSpendGovernor(policy=_policy(max_frames=50))
    assert frames_bound.authorize(frame_count=60)["blockers"] == [
        "judge_spend_frame_ceiling_reached"
    ]


def test_ttl_expiry_stops_the_cohort() -> None:
    clock = {"value": 0.0}
    governor = gov.JudgeSpendGovernor(
        policy=_policy(ttl_seconds=10), monotonic=lambda: clock["value"]
    )
    assert governor.authorize(frame_count=1)["authorized"] is True
    clock["value"] = 11.0
    decision = governor.authorize(frame_count=1)
    assert decision["authorized"] is False
    assert "judge_spend_campaign_ttl_expired" in decision["blockers"]


def test_failed_requests_are_still_settled() -> None:
    """A governor that only counts successes under-reports real spend."""

    governor = gov.JudgeSpendGovernor(policy=_policy())
    governor.settle(frame_count=60, estimated_cost_usd=1.5)
    ledger = governor.ledger()

    assert ledger["request_count"] == 1
    assert ledger["spent_usd"] == pytest.approx(1.5)
    assert ledger["entries"][-1]["cost_is_actual"] is False


def test_ledger_is_written_to_disk_and_reports_remaining_budget(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    governor = gov.JudgeSpendGovernor(policy=_policy(hard_cap_usd=4.0), ledger_path=path)
    decision = governor.authorize(frame_count=60)
    governor.settle(frame_count=60, actual_cost_usd=decision["estimated_cost_usd"])

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ledger = governor.ledger()

    assert len(rows) == 2
    assert ledger["remaining_usd"] == pytest.approx(2.5)
    assert ledger["status"] == "open"
    assert ledger["claim_boundary"]["estimated_costs_are_not_provider_invoices"] is True


def test_over_target_is_flagged_without_blocking() -> None:
    governor = gov.JudgeSpendGovernor(policy=_policy(target_spend_usd=0.5, hard_cap_usd=10.0))
    decision = governor.authorize(frame_count=60)

    assert decision["authorized"] is True
    assert decision["over_target_spend"] is True


def test_governor_from_env_requires_a_configured_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(gov.POLICY_ENV, raising=False)
    assert gov.governor_from_env(campaign_id="c") is None

    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(_policy()), encoding="utf-8")
    monkeypatch.setenv(gov.POLICY_ENV, str(policy_file))
    monkeypatch.setenv(gov.LEDGER_ENV, str(tmp_path / "ledger.jsonl"))

    governor = gov.governor_from_env(campaign_id="c")
    assert governor is not None
    assert governor.authorize(frame_count=10)["authorized"] is True


def test_progress_judge_refuses_to_run_ungoverned(monkeypatch, tmp_path) -> None:
    """The most token-hungry lane treats an absent spend policy as a refusal."""

    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(judge.JUDGE_COMMAND_ENV, "true")
    monkeypatch.delenv(gov.POLICY_ENV, raising=False)

    request = judge.build_judge_request(
        rollout_id="rollout-1",
        criterion_id="registered_task_success",
        task_instruction="place the box",
        frame_uris=[f"frame://{index}" for index in range(60)],
        view_roles={"fixed_external_left": ["task_progress"]},
        duration_seconds=25.0,
        segment_count=3,
        source_frame_count=300,
    )
    result = judge.run_progress_judge_command(request, output_dir=tmp_path)

    assert result["status"] == "blocked"
    assert "progress_judge_spend_policy_not_configured" in result["blockers"]
    assert not list(tmp_path.iterdir())


def test_progress_judge_blocked_by_an_exhausted_cohort(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(judge.JUDGE_COMMAND_ENV, "true")

    governor = gov.JudgeSpendGovernor(policy=_policy(hard_cap_usd=0.5))
    request = judge.build_judge_request(
        rollout_id="rollout-1",
        criterion_id="registered_task_success",
        task_instruction="place the box",
        frame_uris=[f"frame://{index}" for index in range(60)],
        view_roles={"fixed_external_left": ["task_progress"]},
        duration_seconds=25.0,
        segment_count=3,
        source_frame_count=300,
    )
    result = judge.run_progress_judge_command(
        request, output_dir=tmp_path, governor=governor
    )

    assert result["status"] == "blocked"
    assert "judge_spend_hard_cap_reached" in result["blockers"]
    assert not list(tmp_path.iterdir())
