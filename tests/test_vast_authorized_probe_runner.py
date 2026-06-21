from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import vast_authorized_probe_runner as runner


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_minimal_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/runner.py", "print('ok')\n")


def test_authorized_probe_runner_internal_json_reader_edges(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"ready": true}', encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text('["not", "a", "mapping"]', encoding="utf-8")

    assert runner._read_json(mapping_path) == {"ready": True}
    assert runner._read_json(list_path) == {}
    assert runner._number(True) is None
    assert runner._number("bad") is None
    assert runner._number("1.25") == 1.25
    assert runner._attempt_estimated_cost({"estimated_cost_usd_using_observed_rate": -1}) == 0.0
    assert runner._attempt_estimated_cost({"estimated_cost_usd": "0.2"}) == 0.2
    assert runner._attempt_estimated_cost({"estimated_cost_usd": True}) == 0.0

    missing_budget = tmp_path / "missing-budget.json"
    assert runner._session_estimated_cost(missing_budget) == (0.0, None)
    invalid_budget = tmp_path / "invalid-budget.json"
    invalid_budget.write_text("{", encoding="utf-8")
    assert runner._session_estimated_cost(invalid_budget) == (
        0.0,
        "session_budget_ledger_parse_failed:JSONDecodeError",
    )
    attempts_budget = tmp_path / "attempts-budget.json"
    attempts_budget.write_text(
        json.dumps(
            {
                "attempts": [
                    {"estimated_cost_usd": "0.10"},
                    "ignored",
                    {"estimated_cost_usd_using_observed_rate": 0.25},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert runner._session_estimated_cost(attempts_budget) == (0.35, None)
    empty_budget = tmp_path / "empty-budget.json"
    empty_budget.write_text(json.dumps({"status": "empty"}), encoding="utf-8")
    assert runner._session_estimated_cost(empty_budget) == (0.0, None)
    exhausted = runner._target_spend_guard(
        budget_path=attempts_budget,
        target_spend_usd=0.35,
        max_hourly_rate=0.0,
        max_live_minutes=1,
        allow_target_spend_overrun=False,
    )
    assert exhausted["blockers"] == ["session_estimated_spend_target_exhausted"]
    parse_blocked = runner._target_spend_guard(
        budget_path=invalid_budget,
        target_spend_usd=1.0,
        max_hourly_rate=0.0,
        max_live_minutes=1,
        allow_target_spend_overrun=False,
    )
    assert parse_blocked["blockers"] == ["session_budget_ledger_parse_failed"]


def test_authorized_probe_runner_blocks_without_public_url_or_paid_flag(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    token_file = tmp_path / "token"

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        token_file=token_file,
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert result["local_staging_self_test_status"] == "passed"
    assert "public_base_url_missing" in result["blockers"]
    assert "public_base_url_missing_for_paid_vast_launch" in result["blockers"]
    assert "paid_vast_launch_not_authorized_by_runner_flag" in result["blockers"]
    persisted = (tmp_path / "vast_authorized_probe_runner_manifest.json").read_text(
        encoding="utf-8"
    )
    assert token_file.read_text(encoding="utf-8").strip() not in persisted


def test_authorized_probe_runner_defaults_to_shared_session_budget_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    shared_budget = tmp_path / "shared-session-cost.json"
    shared_budget.write_text(json.dumps({"estimated_cost_usd": 0.0}), encoding="utf-8")

    monkeypatch.setattr(runner, "_vast_session_budget_ledger_path", lambda: shared_budget)

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["session_budget_ledger"] == str(shared_budget)
    manifest = json.loads(
        (tmp_path / "vast_authorized_probe_runner_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["session_budget_ledger"] == str(shared_budget)


def test_authorized_probe_runner_records_local_self_test_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    monkeypatch.setattr(
        runner,
        "run_local_staging_self_test",
        lambda **_: {"status": "failed"},
    )

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert "local_staging_self_test_failed" in result["blockers"]
    assert "paid_vast_launch_not_authorized_by_runner_flag" in result["blockers"]


def test_authorized_probe_runner_requires_urls_for_paid_launch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "paid_vast_launch_requires_public_staging_urls" in result["blockers"]
    assert "public_base_url_missing_for_paid_vast_launch" in result["blockers"]


def test_authorized_probe_runner_prepares_public_urls_without_paid_launch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    token_file = tmp_path / "token"
    secret_env = tmp_path / "urls.env"

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=token_file,
        secret_env_file=secret_env,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["staging_manifest_status"] == "ready"
    assert result["paid_launch_attempted"] is False
    assert result["blockers"] == ["paid_vast_launch_not_authorized_by_runner_flag"]
    raw_token = token_file.read_text(encoding="utf-8").strip()
    assert raw_token in secret_env.read_text(encoding="utf-8")
    assert raw_token not in (tmp_path / "vast_authorized_probe_runner_manifest.json").read_text(
        encoding="utf-8"
    )
    staging_manifest = _read_json(tmp_path / "vast_bundle_staging_manifest.json")
    assert staging_manifest["provider_fetchable_bundle_uri_ready"] is True


def test_authorized_probe_runner_paid_path_delegates_to_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "status": "blocked",
            "reason": "vast_session_budget_guard_blocked",
            "blockers": ["session_live_runtime_limit_exhausted"],
        }

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        runner,
        "verify_public_staging_urls",
        lambda **_: {"status": "passed", "blockers": [], "raw_secret_values_recorded": False},
    )

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=tmp_path / "budget.json",
        allow_paid_vast_launch=True,
        max_live_minutes=1,
        startup_timeout_seconds=900,
        ngc_image_login_mode="always",
        isaac_image="ghcr.io/blueprint/isaac-smoke:cached",
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is True
    assert result["adapter_result_status"] == "blocked"
    assert result["public_staging_verification_status"] == "passed"
    assert "session_live_runtime_limit_exhausted" in result["blockers"]
    assert captured["mode"] == "live-startup-probe"
    assert captured["enable_isaac_smoke"] is True
    assert captured["enable_blueprint_bundle"] is True
    assert captured["startup_timeout_seconds"] == 900
    assert captured["require_known_supported_isaac_driver"] is True
    assert captured["ngc_image_login_mode"] == "always"
    assert captured["isaac_image"] == "ghcr.io/blueprint/isaac-smoke:cached"
    assert captured["allow_cold_isaac_image_pull"] is False
    assert captured["use_vast_template_image"] is False
    assert str(captured["provider_bundle"]) == str(bundle.resolve())
    assert "token=" in str(captured["provider_bundle_url"])
    assert "token=" in str(captured["provider_output_put_url"])
    persisted = (tmp_path / "vast_authorized_probe_runner_manifest.json").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "token").read_text(encoding="utf-8").strip() not in persisted
    assert "token=<redacted-token>" in persisted
    assert '"require_known_supported_isaac_driver": true' in persisted
    assert '"ngc_image_login_mode": "always"' in persisted
    assert '"isaac_image": "ghcr.io/blueprint/isaac-smoke:cached"' in persisted
    assert '"startup_timeout_seconds": 900' in persisted
    assert '"allow_cold_isaac_image_pull": false' in persisted


def test_authorized_probe_runner_blocks_paid_launch_when_public_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    def fail_adapter(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected paid adapter call: {kwargs}")

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fail_adapter)
    monkeypatch.setattr(
        runner,
        "verify_public_staging_urls",
        lambda **_: {
            "status": "blocked",
            "blockers": ["provider_bundle_fetch_url_unreachable"],
            "raw_secret_values_recorded": False,
        },
    )

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        allow_target_spend_overrun=True,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert result["public_staging_verification_status"] == "blocked"
    assert "public_staging_url_verification_failed" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]


def test_authorized_probe_runner_blocks_paid_launch_when_staging_urls_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    def fail_adapter(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected paid adapter call: {kwargs}")

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fail_adapter)

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        verify_staging_urls=False,
        allow_target_spend_overrun=True,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "public_staging_urls_not_verified_for_paid_launch" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]
    guard = result["staging_verification_guard"]
    assert guard["status"] == "blocked"
    assert guard["verify_staging_urls"] is False
    assert guard["allow_unverified_public_staging_for_paid_launch"] is False


def test_authorized_probe_runner_allows_explicit_unverified_staging_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        runner,
        "verify_public_staging_urls",
        lambda **_: {"status": "passed", "blockers": [], "raw_secret_values_recorded": False},
    )

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        verify_staging_urls=False,
        allow_unverified_public_staging_for_paid_launch=True,
        allow_target_spend_overrun=True,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    assert result["paid_launch_attempted"] is True
    assert result["staging_verification_guard"]["status"] == "passed"
    assert result["staging_verification_guard"]["verify_staging_urls"] is False
    assert result["allow_unverified_public_staging_for_paid_launch"] is True
    assert captured["verify_staging_urls"] is False


def test_authorized_probe_runner_blocks_paid_launch_when_target_spend_would_be_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    budget = tmp_path / "vast_session_cost_summary.json"
    budget.write_text(
        json.dumps(
            {
                "schema_version": "vast_session_cost_summary.v1",
                "estimated_cost_usd": 0.346388,
            }
        ),
        encoding="utf-8",
    )

    def fail_adapter(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected paid adapter call: {kwargs}")

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fail_adapter)

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=budget,
        allow_paid_vast_launch=True,
        target_spend_usd=0.35,
        max_hourly_rate=0.1311111111111111,
        max_live_minutes=2,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "requested_max_spend_would_exceed_target" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]
    guard = result["target_spend_guard"]
    assert guard["status"] == "blocked"
    assert guard["prior_estimated_cost_usd"] == 0.346388
    assert guard["allow_target_spend_overrun"] is False


def test_authorized_probe_runner_allows_explicit_target_spend_overrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    budget = tmp_path / "vast_session_cost_summary.json"
    budget.write_text(json.dumps({"estimated_cost_usd": 0.35}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        runner,
        "verify_public_staging_urls",
        lambda **_: {"status": "passed", "blockers": [], "raw_secret_values_recorded": False},
    )

    result = runner.run_vast_authorized_probe_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=budget,
        allow_paid_vast_launch=True,
        allow_target_spend_overrun=True,
        target_spend_usd=0.35,
        max_hourly_rate=0.1311111111111111,
        max_live_minutes=2,
        generated_at="2026-06-20T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    assert result["paid_launch_attempted"] is True
    assert result["target_spend_guard"]["status"] == "passed"
    assert result["target_spend_guard"]["allow_target_spend_overrun"] is True
    assert captured["mode"] == "live-startup-probe"


def test_authorized_probe_runner_main_forwards_options_and_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(runner, "run_vast_authorized_probe_runner", fake_run)

    exit_code = runner.main(
        [
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--public-base-url",
            "https://example.trycloudflare.com",
            "--token-file",
            str(tmp_path / "token"),
            "--secret-env-file",
            str(tmp_path / "urls.env"),
            "--output-path",
            str(tmp_path / "output.zip"),
            "--session-budget-ledger",
            str(tmp_path / "budget.json"),
            "--allow-paid-vast-launch",
            "--max-hourly-rate",
            "0.42",
            "--target-spend-usd",
            "0.25",
            "--hard-cap-usd",
            "0.75",
            "--allow-target-spend-overrun",
            "--max-live-minutes",
            "3",
            "--session-max-live-minutes",
            "4",
            "--startup-timeout-seconds",
            "123",
            "--no-verify-staging-urls",
            "--public-staging-max-wait-seconds",
            "11",
            "--public-staging-retry-interval-seconds",
            "0.5",
            "--public-staging-timeout-seconds",
            "2.5",
            "--allow-unverified-public-staging-for-paid-launch",
            "--allow-known-unsupported-isaac-driver",
            "--ngc-image-login-mode",
            "never",
            "--isaac-image",
            "ghcr.io/blueprint/isaac-smoke:cached",
            "--vast-template-hash-id",
            "template-123",
            "--use-vast-template-image",
            "--allow-cold-isaac-image-pull",
            "--min-cold-isaac-pull-live-minutes",
            "17",
        ]
    )

    assert exit_code == 0
    assert captured["job_dir"] == str(tmp_path)
    assert captured["bundle_path"] == str(tmp_path / "bundle.zip")
    assert captured["public_base_url"] == "https://example.trycloudflare.com"
    assert captured["allow_paid_vast_launch"] is True
    assert captured["max_hourly_rate"] == 0.42
    assert captured["target_spend_usd"] == 0.25
    assert captured["hard_cap_usd"] == 0.75
    assert captured["allow_target_spend_overrun"] is True
    assert captured["max_live_minutes"] == 3
    assert captured["session_max_live_minutes"] == 4
    assert captured["startup_timeout_seconds"] == 123
    assert captured["verify_staging_urls"] is False
    assert captured["public_staging_max_wait_seconds"] == 11
    assert captured["public_staging_retry_interval_seconds"] == 0.5
    assert captured["public_staging_timeout_seconds"] == 2.5
    assert captured["allow_unverified_public_staging_for_paid_launch"] is True
    assert captured["require_known_supported_isaac_driver"] is False
    assert captured["ngc_image_login_mode"] == "never"
    assert captured["isaac_image"] == "ghcr.io/blueprint/isaac-smoke:cached"
    assert captured["vast_template_hash_id"] == "template-123"
    assert captured["use_vast_template_image"] is True
    assert captured["allow_cold_isaac_image_pull"] is True
    assert captured["min_cold_isaac_pull_live_minutes"] == 17
    output = capsys.readouterr().out
    assert "[vast-authorized-probe-runner] status=completed" in output


def test_authorized_probe_runner_main_prints_blockers_and_returns_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "run_vast_authorized_probe_runner",
        lambda **_: {"status": "blocked", "blockers": ["needs_public_url", "budget_guard"]},
    )

    exit_code = runner.main(
        [
            "--job-dir",
            str(tmp_path),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "[vast-authorized-probe-runner] status=blocked" in output
    assert "[vast-authorized-probe-runner] blockers=needs_public_url,budget_guard" in output
