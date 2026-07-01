from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import vast_wam_authorized_runner as runner
from blueprint_pipeline.oscar_official_release import OFFICIAL_OSCAR_WAM_IMAGE_REF


def test_vast_wam_authorized_runner_defaults_to_official_oscar_image() -> None:
    assert runner.DEFAULT_WAM_PUBLIC_IMAGE == OFFICIAL_OSCAR_WAM_IMAGE_REF


def _write_minimal_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/run_wam_provider_runtime.sh", "echo ok\n")


def _passed_public_staging(**_kwargs: object) -> dict[str, object]:
    return {
        "status": "passed",
        "blockers": [],
        "required_consecutive_successes": 3,
        "successful_attempt_count": 3,
    }


def test_vast_wam_authorized_runner_blocks_without_paid_flag(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert result["provider_bundle_kind"] == "wam"
    assert result["staging_manifest_status"] == "ready"
    assert result["blockers"] == ["paid_vast_launch_not_authorized_by_runner_flag"]
    token = (tmp_path / "token").read_text(encoding="utf-8").strip()
    assert token not in (tmp_path / "vast_wam_authorized_runner_manifest.json").read_text(
        encoding="utf-8"
    )


def test_vast_wam_authorized_runner_defaults_to_shared_session_budget_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    shared_budget = tmp_path / "shared-session-cost.json"
    shared_budget.write_text(json.dumps({"estimated_cost_usd": 0.0}), encoding="utf-8")

    monkeypatch.setattr(runner, "_vast_session_budget_ledger_path", lambda: shared_budget)

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["session_budget_ledger"] == str(shared_budget)
    manifest = json.loads(
        (tmp_path / "vast_wam_authorized_runner_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["session_budget_ledger"] == str(shared_budget)


def test_vast_wam_authorized_runner_paid_path_delegates_wam_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    budget = tmp_path / "vast_session_cost_summary.json"
    budget.write_text(json.dumps({"estimated_cost_usd": 0.0}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_adapter(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "status": "blocked",
            "reason": "simulated_wam_block",
            "blockers": ["simulated_wam_block"],
        }

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(runner, "verify_public_staging_urls", _passed_public_staging)

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=budget,
        allow_paid_vast_launch=True,
        max_live_minutes=1,
        public_image="pytorch/pytorch:test",
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is True
    assert result["adapter_result_status"] == "blocked"
    assert result["public_staging_verification_status"] == "passed"
    assert captured["provider_bundle_kind"] == "wam"
    assert captured["enable_blueprint_bundle"] is True
    assert captured["enable_isaac_smoke"] is False
    assert captured["ngc_image_login_mode"] == "never"
    assert captured["require_known_supported_isaac_driver"] is False
    assert captured["vast_launch_mode"] == "auto"
    assert captured["public_image"] == "pytorch/pytorch:test"
    assert "token=" in str(captured["provider_bundle_url"])
    assert "token=" in str(captured["provider_output_put_url"])


def test_vast_wam_authorized_runner_helper_and_blocker_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert runner._read_json(array_json) == {}
    assert runner._string(42) == ""
    assert runner._redacted_path("/bundle.zip") == "/bundle.zip?token=<redacted-token>"

    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    output = tmp_path / "output.zip"
    output.write_bytes(b"zip")

    monkeypatch.setattr(
        runner,
        "prepare_vast_bundle_staging",
        lambda **kwargs: {"status": "blocked", "blockers": ["staging_blocked"]},
    )
    monkeypatch.setattr(
        runner,
        "run_local_staging_self_test",
        lambda **kwargs: {"status": "failed"},
    )
    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path / "blocked",
        bundle_path=bundle,
        output_path=output,
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert {
        "staging_blocked",
        "local_staging_self_test_failed",
        "public_base_url_missing_for_paid_vast_launch",
        "paid_vast_launch_requires_public_staging_urls",
    }.issubset(set(result["blockers"]))
    assert result["output_inspection"]["output_zip_present"] is True

    real_path = runner.Path
    stat_failing_output = tmp_path / "stat-failing-output.zip"

    class StatFailingOutputPath:
        def __init__(self, path: Path) -> None:
            self.path = path

        def expanduser(self) -> "StatFailingOutputPath":
            return self

        def resolve(self) -> "StatFailingOutputPath":
            return self

        def is_file(self) -> bool:
            return True

        def stat(self) -> Any:
            raise OSError("stat failed")

        def __str__(self) -> str:
            return str(self.path)

    def path_factory(value: Any) -> Any:
        path = real_path(value)
        if path == stat_failing_output:
            return StatFailingOutputPath(path)
        return path

    monkeypatch.setattr(runner, "Path", path_factory)
    fallback = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path / "stat-fallback",
        bundle_path=bundle,
        output_path=stat_failing_output,
        token_file=tmp_path / "token-2",
        secret_env_file=tmp_path / "urls-2.env",
        generated_at="2026-06-21T00:00:00+00:00",
    )
    assert fallback["output_inspection"] == {"output_zip_present": True}


def test_vast_wam_authorized_runner_completed_path_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    monkeypatch.setattr(
        runner,
        "run_vast_provider_adapter",
        lambda **kwargs: {"status": "completed", "reason": "ok", "blockers": []},
    )
    monkeypatch.setattr(runner, "verify_public_staging_urls", _passed_public_staging)
    completed = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path / "completed",
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=tmp_path / "budget.json",
        output_path=tmp_path / "completed-output.zip",
        allow_paid_vast_launch=True,
        max_hourly_rate=0.5,
        target_spend_usd=1.5,
        hard_cap_usd=2.5,
        max_live_minutes=12,
        session_max_live_minutes=None,
        startup_timeout_seconds=34,
        verify_staging_urls=False,
        allow_unverified_public_staging_for_paid_launch=True,
        public_image="pytorch:test",
        vast_launch_mode="template",
        vast_template_hash_id="template-123",
        use_vast_template_image=True,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert completed["status"] == "completed"
    assert completed["adapter_result_status"] == "completed"
    assert completed["session_max_live_minutes"] is None
    assert completed["vast_launch_mode"] == "template"
    assert completed["allow_unverified_public_staging_for_paid_launch"] is True

    captured: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "status": "blocked",
            "blockers": ["cli_blocker"],
        }

    monkeypatch.setattr(runner, "run_vast_wam_authorized_runner", fake_runner)
    code = runner.main(
        [
            "--job-dir",
            str(tmp_path / "cli-job"),
            "--bundle-path",
            str(bundle),
            "--public-base-url",
            "https://example.trycloudflare.com",
            "--token-file",
            str(tmp_path / "cli-token"),
            "--secret-env-file",
            str(tmp_path / "cli.env"),
            "--output-path",
            str(tmp_path / "cli-output.zip"),
            "--session-budget-ledger",
            str(tmp_path / "cli-budget.json"),
            "--allow-paid-vast-launch",
            "--max-hourly-rate",
            "0.6",
            "--target-spend-usd",
            "1.6",
            "--hard-cap-usd",
            "2.6",
            "--max-live-minutes",
            "13",
            "--session-max-live-minutes",
            "14",
            "--startup-timeout-seconds",
            "35",
            "--no-verify-staging-urls",
            "--allow-unverified-public-staging-for-paid-launch",
            "--allow-target-spend-overrun",
            "--public-image",
            "image:test",
            "--vast-launch-mode",
            "ssh_proxy",
            "--vast-template-hash-id",
            "template-456",
            "--use-vast-template-image",
        ]
    )
    output = capsys.readouterr().out
    assert code == 1
    assert "[vast-wam-authorized-runner] blockers=cli_blocker" in output
    assert captured["allow_paid_vast_launch"] is True
    assert captured["verify_staging_urls"] is False
    assert captured["allow_unverified_public_staging_for_paid_launch"] is True
    assert captured["allow_target_spend_overrun"] is True
    assert captured["vast_template_hash_id"] == "template-456"


def test_vast_wam_authorized_runner_blocks_paid_launch_when_target_spend_would_be_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)
    budget = tmp_path / "vast_session_cost_summary.json"
    budget.write_text(json.dumps({"estimated_cost_usd": 0.0}), encoding="utf-8")

    def fail_adapter(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected paid adapter call: {kwargs}")

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fail_adapter)
    monkeypatch.setattr(runner, "verify_public_staging_urls", _passed_public_staging)

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=budget,
        allow_paid_vast_launch=True,
        max_hourly_rate=0.60,
        target_spend_usd=0.35,
        max_live_minutes=45,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "requested_max_spend_would_exceed_target" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]
    assert result["target_spend_guard"]["status"] == "blocked"


def test_vast_wam_authorized_runner_blocks_paid_launch_when_staging_urls_unverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle.zip"
    _write_minimal_bundle(bundle)

    def fail_adapter(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(f"unexpected paid adapter call: {kwargs}")

    monkeypatch.setattr(runner, "run_vast_provider_adapter", fail_adapter)

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        verify_staging_urls=False,
        allow_target_spend_overrun=True,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "public_staging_urls_not_verified_for_paid_launch" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]
    assert result["staging_verification_guard"]["status"] == "blocked"


def test_vast_wam_authorized_runner_blocks_paid_launch_when_public_verification_fails(
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
        lambda **kwargs: {"status": "blocked", "blockers": ["bundle_url_unstable"]},
    )

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        allow_target_spend_overrun=True,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "blocked"
    assert result["paid_launch_attempted"] is False
    assert "bundle_url_unstable" in result["blockers"]
    assert "paid_vast_launch_preflight_blocked" in result["blockers"]


def test_vast_wam_authorized_runner_allows_explicit_preflight_overrides(
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

    result = runner.run_vast_wam_authorized_runner(
        job_dir=tmp_path,
        bundle_path=bundle,
        public_base_url="https://example.trycloudflare.com",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        allow_paid_vast_launch=True,
        verify_staging_urls=False,
        allow_unverified_public_staging_for_paid_launch=True,
        allow_target_spend_overrun=True,
        max_hourly_rate=0.60,
        max_live_minutes=45,
        generated_at="2026-06-21T00:00:00+00:00",
    )

    assert result["status"] == "completed"
    assert result["paid_launch_attempted"] is True
    assert result["staging_verification_guard"]["status"] == "passed"
    assert result["target_spend_guard"]["status"] == "passed"
    assert result["allow_target_spend_overrun"] is True
    assert captured["verify_staging_urls"] is False
