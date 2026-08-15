"""A standing authorization must not be weaker than the handshake it replaces.

Admitting a paid launch required ``--execute-launch-id`` equal to that launch's
id, minted per launch by the website. Every run therefore required a human to
copy an id into the host's environment file and restart the unit -- a
hand-patched env var per run, untestable, and gone the moment the host is
rebuilt.

Moving the decision to the profile is only an improvement if every property the
per-run handshake provided survives: bound to exact bytes, expiring, bounded in
count and in spend, and counted across restarts. These pin each one.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
    SCHEMA_VERSION,
    StandingAuthorizationError,
    standing_authorization_admits,
    validate_standing_authorization,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
DIGEST = "sha256:" + "a" * 64
URI = "https://raw.githubusercontent.com/example/repo/" + "0" * 40 + "/request.json"


def _profile(spend: float = 5.0, digest: str = DIGEST) -> dict:
    return {
        "profile_id": "adp-retained-scene-render-live",
        "profile_digest": digest,
        "allocator": {"max_spend_usd": spend},
    }


def _authorization(**overrides) -> dict:
    value = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": "adp-retained-scene-render-live",
        "profile_digest": DIGEST,
        "max_launches": 5,
        "max_total_spend_usd": 50.0,
        "expires_at": (NOW + timedelta(days=7)).isoformat(),
    }
    value.update(overrides)
    return value


def _validate(authorization, *, profile=None, launches=0, spend=0.0):
    return validate_standing_authorization(
        authorization,
        profile=profile or _profile(),
        launches_consumed=launches,
        spend_consumed_usd=spend,
        now=NOW,
    )


def test_a_valid_authorization_admits_a_launch() -> None:
    assert _validate(_authorization()) == []


def test_a_republished_profile_does_not_inherit_the_approval() -> None:
    """Same id, different bytes: a different artifact, so a different decision."""
    blockers = _validate(_authorization(), profile=_profile(digest="sha256:" + "b" * 64))

    assert "standing_authorization_profile_digest_mismatch" in blockers


def test_an_authorization_for_another_profile_is_refused() -> None:
    blockers = _validate(_authorization(profile_id="some-other-profile"))

    assert "standing_authorization_profile_mismatch" in blockers


def test_an_expired_authorization_stops_admitting() -> None:
    """An approval left behind must not keep funding launches forever."""
    blockers = _validate(
        _authorization(expires_at=(NOW - timedelta(seconds=1)).isoformat())
    )

    assert "standing_authorization_expired" in blockers


def test_an_unparseable_expiry_is_refused_not_ignored() -> None:
    blockers = _validate(_authorization(expires_at="whenever"))

    assert "standing_authorization_expires_at_invalid" in blockers


def test_the_launch_count_is_bounded() -> None:
    assert _validate(_authorization(max_launches=3), launches=2) == []
    assert "standing_authorization_launches_exhausted" in _validate(
        _authorization(max_launches=3), launches=3
    )


def test_total_spend_is_bounded_across_launches() -> None:
    """One approval must not fund unbounded spend through repeated launches."""
    assert _validate(_authorization(max_total_spend_usd=12.0), spend=6.0) == []
    assert "standing_authorization_spend_ceiling_reached" in _validate(
        _authorization(max_total_spend_usd=12.0), spend=8.0
    )


def test_the_next_launch_must_fit_before_it_runs() -> None:
    """Checking after the fact would authorize the overspend it must prevent."""
    blockers = _validate(
        _authorization(max_total_spend_usd=10.0), profile=_profile(spend=4.0), spend=7.0
    )

    assert "standing_authorization_spend_ceiling_reached" in blockers


def test_a_zero_or_negative_bound_is_refused() -> None:
    assert "standing_authorization_max_launches_invalid" in _validate(
        _authorization(max_launches=0)
    )
    assert "standing_authorization_max_total_spend_usd_invalid" in _validate(
        _authorization(max_total_spend_usd=0)
    )


def test_a_boolean_is_not_a_launch_count() -> None:
    """`True == 1` in Python, so a bool would otherwise authorize one launch."""
    assert "standing_authorization_max_launches_invalid" in _validate(
        _authorization(max_launches=True)
    )


def test_every_reason_is_reported_not_just_the_first() -> None:
    """An operator should not rediscover the next fault on the next paid run."""
    blockers = _validate(
        _authorization(
            profile_id="wrong",
            expires_at=(NOW - timedelta(days=1)).isoformat(),
            max_launches=1,
        ),
        launches=5,
    )

    assert {
        "standing_authorization_profile_mismatch",
        "standing_authorization_expired",
        "standing_authorization_launches_exhausted",
    } <= set(blockers)


def test_a_missing_field_is_named(tmp_path: Path) -> None:
    authorization = _authorization()
    del authorization["expires_at"]

    assert "standing_authorization_missing_expires_at" in _validate(authorization)


def test_an_absent_authorization_is_not_an_error(tmp_path: Path) -> None:
    """A host with none simply has none; the per-launch handshake still works."""
    result = standing_authorization_admits(
        profile=_profile(),
        directory=tmp_path,
        launches_consumed=0,
        spend_consumed_usd=0.0,
        now=NOW,
    )

    assert result["admitted"] is False
    assert result["reason"] == "standing_authorization_absent"
    assert result["blockers"] == []


def test_an_unconfigured_directory_admits_nothing(tmp_path: Path) -> None:
    result = standing_authorization_admits(
        profile=_profile(),
        directory=None,
        launches_consumed=0,
        spend_consumed_usd=0.0,
        now=NOW,
    )

    assert result["admitted"] is False
    assert result["reason"] == "standing_authorization_not_configured"


def test_a_file_on_disk_admits_the_matching_profile(tmp_path: Path) -> None:
    (tmp_path / "adp-retained-scene-render-live.json").write_text(
        json.dumps(_authorization()), encoding="utf-8"
    )

    result = standing_authorization_admits(
        profile=_profile(),
        directory=tmp_path,
        launches_consumed=1,
        spend_consumed_usd=5.0,
        now=NOW,
    )

    assert result["admitted"] is True
    assert result["blockers"] == []


def test_a_corrupt_authorization_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "adp-retained-scene-render-live.json").write_text(
        "{not json", encoding="utf-8"
    )

    result = standing_authorization_admits(
        profile=_profile(),
        directory=tmp_path,
        launches_consumed=0,
        spend_consumed_usd=0.0,
        now=NOW,
    )

    assert result["admitted"] is False
    assert result["blockers"] == ["standing_authorization_unreadable:adp-retained-scene-render-live.json"]


def test_a_symlinked_authorization_fails_closed(tmp_path: Path) -> None:
    """A symlink lets the approved bytes change without the approval changing."""
    real = tmp_path / "real.json"
    real.write_text(json.dumps(_authorization()), encoding="utf-8")
    (tmp_path / "adp-retained-scene-render-live.json").symlink_to(real)

    result = standing_authorization_admits(
        profile=_profile(),
        directory=tmp_path,
        launches_consumed=0,
        spend_consumed_usd=0.0,
        now=NOW,
    )

    assert result["admitted"] is False
    assert any("source_invalid" in blocker for blocker in result["blockers"])


def test_a_naive_expiry_is_read_as_utc() -> None:
    """A timestamp without an offset must not be read in the host's timezone."""
    assert _validate(_authorization(expires_at="2026-08-19T12:00:00")) == []
    assert "standing_authorization_expired" in _validate(
        _authorization(expires_at="2026-08-05T12:00:00")
    )


def test_a_wrong_schema_version_is_refused() -> None:
    blockers = _validate(_authorization(schema_version="something.v9"))

    assert "standing_authorization_schema_version_mismatch" in blockers


def test_pytest_raises_on_a_directory_that_is_not_readable(tmp_path: Path) -> None:
    path = tmp_path / "adp-retained-scene-render-live.json"
    path.write_text(json.dumps(_authorization()), encoding="utf-8")
    path.chmod(0o000)
    try:
        with pytest.raises(StandingAuthorizationError, match="unreadable"):
            from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
                load_standing_authorization,
            )

            load_standing_authorization(
                profile_id="adp-retained-scene-render-live", directory=tmp_path
            )
    finally:
        path.chmod(0o600)


def test_consumption_survives_a_restart(tmp_path: Path) -> None:
    """Bounds read from disk, because an in-process counter resets on restart."""
    from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
        consumption_totals,
        record_launch,
    )

    for index, spend in enumerate([5.0, 3.0]):
        record_launch(
            directory=tmp_path,
            profile_id="p",
            launch_id=f"launch-{index}",
            max_spend_usd=spend,
        )

    assert consumption_totals(directory=tmp_path, profile_id="p") == (2, 8.0)


def test_a_replayed_launch_id_cannot_re_spend(tmp_path: Path) -> None:
    from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
        record_launch,
    )

    record_launch(directory=tmp_path, profile_id="p", launch_id="l", max_spend_usd=5.0)

    with pytest.raises(StandingAuthorizationError, match="already_recorded"):
        record_launch(
            directory=tmp_path, profile_id="p", launch_id="l", max_spend_usd=5.0
        )


def test_unaccountable_consumption_fails_closed(tmp_path: Path) -> None:
    """Spend we cannot read must not be counted as zero."""
    from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
        consumption_totals,
    )

    root = tmp_path / "consumed" / "p"
    root.mkdir(parents=True)
    (root / "broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StandingAuthorizationError, match="consumption_unreadable"):
        consumption_totals(directory=tmp_path, profile_id="p")


def test_dispatcher_admits_a_launch_without_a_copied_launch_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the whole change: no per-run env edit on the host."""
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher

    (tmp_path / "adp-retained-scene-render-live.json").write_text(
        json.dumps(_authorization()), encoding="utf-8"
    )
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", str(tmp_path)
    )

    decision = dispatcher._standing_authorization_decision(
        _profile(), True, tmp_path / "launch-runs"
    )

    assert decision["admitted"] is True


def test_an_unconfigured_host_derives_the_directory_and_still_refuses_an_empty_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployed control plane never set the variable, so the capability
    could not admit anything there. Deriving the location makes it usable on
    a rebuilt host; finding nothing there still refuses, with no blocker."""
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher

    monkeypatch.delenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", raising=False
    )
    state_root = tmp_path / "control-plane" / "launch-runs"
    state_root.mkdir(parents=True)

    assert dispatcher.standing_authorization_directory(state_root) == str(
        tmp_path / "control-plane" / "standing-authorizations"
    )
    decision = dispatcher._standing_authorization_decision(_profile(), True, state_root)

    assert decision["admitted"] is False
    assert decision["blockers"] == []


def test_dispatcher_never_consults_on_a_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher

    (tmp_path / "adp-retained-scene-render-live.json").write_text(
        json.dumps(_authorization()), encoding="utf-8"
    )
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR", str(tmp_path)
    )

    assert (
        dispatcher._standing_authorization_decision(
            _profile(), False, tmp_path / "launch-runs"
        )["admitted"]
        is False
    )
