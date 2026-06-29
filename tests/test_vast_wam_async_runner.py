from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any, Callable

import pytest

from blueprint_pipeline import vast_wam_async_runner as runner


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bundle(path: Path) -> Path:
    path.write_text("bundle", encoding="utf-8")
    return path


def _write_state(job_dir: Path, **overrides: Any) -> dict[str, Any]:
    state = {
        "instance_id": 123,
        "token_file": str(job_dir / "token"),
        "public_base_url": "https://public.example",
        "output_path": str(job_dir / "output.zip"),
        "created_at_epoch": 10.0,
        "max_live_deadline_epoch": 1_000.0,
        "target_spend_usd": 1.0,
        "hard_cap_usd": 2.0,
        "max_hourly_rate_usd": 0.2,
        "max_live_minutes": 5,
        "selected_offer": {"ask_contract_id": 707, "hourly_rate_usd": 0.2},
        "session_budget_ledger": str(job_dir / "session-budget.json"),
    }
    state.update(overrides)
    _write_json(runner._state_path(job_dir), state)
    return state


def _install_common_create_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    staging_status: str = "ready",
    staging_blockers: list[str] | None = None,
    self_test_status: str = "passed",
    public_status: str = "passed",
    public_blockers: list[str] | None = None,
    api_key: str = "secret-vast-key",
    api_gate_blockers: list[str] | None = None,
    bundle_blockers: list[str] | None = None,
    image_blockers: list[str] | None = None,
    session_blockers: list[str] | None = None,
    session_warnings: list[str] | None = None,
    lock_handle: object | None = object(),
    lock_manifest: dict[str, Any] | None = None,
    inventory_blockers: list[str] | None = None,
    offers: list[dict[str, Any]] | None = None,
    selected_offer: dict[str, Any] | None | object = Ellipsis,
    create_response: dict[str, Any] | None = None,
    create_exception: Exception | None = None,
) -> None:
    offer = {
        "ask_contract_id": 707,
        "hourly_rate_usd": 0.2,
        "gpu_name": "RTX 4090",
    }
    resolved_offers = [offer] if offers is None else offers
    resolved_selected_offer = offer if selected_offer is Ellipsis else selected_offer
    monkeypatch.setattr(
        runner,
        "prepare_vast_bundle_staging",
        lambda **_kwargs: {"status": staging_status, "blockers": staging_blockers or []},
    )
    monkeypatch.setattr(
        runner,
        "run_local_staging_self_test",
        lambda **_kwargs: {"status": self_test_status},
    )
    monkeypatch.setattr(
        runner,
        "verify_public_staging_urls",
        lambda **_kwargs: {"status": public_status, "blockers": public_blockers or []},
    )
    monkeypatch.setattr(
        runner,
        "_read_secret_file",
        lambda *_args, **_kwargs: (
            api_key,
            {"present": bool(api_key), "mode_is_0600": bool(api_key)},
        ),
    )
    monkeypatch.setattr(runner, "_api_gate_blockers", lambda **_kwargs: api_gate_blockers or [])
    monkeypatch.setattr(runner, "_runtime_discovery", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_fill_missing_phase_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_provider_plan", lambda **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "_blueprint_bundle_preflight",
        lambda **_kwargs: {"blockers": bundle_blockers or []},
    )
    monkeypatch.setattr(
        runner,
        "_isaac_image_startup_preflight",
        lambda **_kwargs: {"blockers": image_blockers or []},
    )
    monkeypatch.setattr(
        runner,
        "_session_budget_guard",
        lambda **_kwargs: {
            "blockers": session_blockers or [],
            "warnings": session_warnings or [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_budget_ledger",
        lambda **_kwargs: {"estimated_cost_usd": 0.01},
    )
    monkeypatch.setattr(runner, "_vast_launch_lock_path", lambda: Path("/tmp/fake-vast.lock"))
    monkeypatch.setattr(
        runner,
        "_try_acquire_vast_launch_lock",
        lambda **_kwargs: (lock_handle, lock_manifest or {}),
    )
    monkeypatch.setattr(runner, "_release_vast_launch_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_prelaunch_inventory_guard",
        lambda **_kwargs: {"blockers": inventory_blockers or []},
    )
    monkeypatch.setattr(runner, "_search_payload", lambda **kwargs: {"search": kwargs})
    monkeypatch.setattr(runner, "_offers_from_response", lambda _response: resolved_offers)
    monkeypatch.setattr(runner, "_select_offer", lambda *_args, **_kwargs: resolved_selected_offer)
    monkeypatch.setattr(runner, "_offer_summary", lambda offer: dict(offer))
    monkeypatch.setattr(runner, "_resolve_image_login", lambda **_kwargs: (None, {"mode": "never"}))
    monkeypatch.setattr(runner, "_probe_shell_script", lambda *_args, **_kwargs: "probe")
    monkeypatch.setattr(runner, "_probe_env", lambda **_kwargs: {"ENV": "value"})
    monkeypatch.setattr(runner, "_create_payload", lambda **kwargs: {"payload": kwargs})
    monkeypatch.setattr(runner, "_forwarded_secret_values", lambda: [])
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner, "_create_request_summary", lambda *_args, **_kwargs: {"summary": True})
    monkeypatch.setattr(
        runner,
        "_instance_id_from_create_response",
        lambda response: response.get("new_contract") or response.get("instance_id"),
    )
    monkeypatch.setattr(
        runner,
        "_poll_instance",
        lambda **_kwargs: ("running", [{"status": "running"}], {"actual_status": "running"}),
    )

    def fake_api_json(**kwargs: Any) -> tuple[int, dict[str, Any]]:
        if kwargs["method"] == "POST" and kwargs["path"] == "/bundles/":
            return 200, {"offers": resolved_offers}
        if kwargs["method"] == "PUT" and kwargs["path"].startswith("/asks/"):
            if create_exception is not None:
                raise create_exception
            return 200, create_response if create_response is not None else {"new_contract": 909}
        raise AssertionError(kwargs)

    monkeypatch.setattr(runner, "_api_json", fake_api_json)


def test_async_provider_urls_and_blocked_result(tmp_path: Path) -> None:
    bundle_url, output_url, token_status = runner._provider_urls(
        "https://public.example",
        tmp_path / "token",
    )

    assert "/bundle.zip" in bundle_url
    assert "/output.zip" in output_url
    assert token_status["path"] == str(tmp_path / "token")
    assert token_status["present"] is True

    result = runner._write_blocked_result(
        tmp_path,
        generated_at="now",
        reason="preflight_blocked",
        blockers=["missing_secret"],
    )

    assert result["status"] == "blocked"
    assert json.loads((tmp_path / "vast_final_validation.json").read_text())["blockers"] == [
        "missing_secret"
    ]


def test_async_state_recovery_from_malformed_state(tmp_path: Path) -> None:
    job_dir = tmp_path / "recover"
    job_dir.mkdir()
    runner._state_path(job_dir).write_text(
        '{"instance_id": 321, "bundle_path": "/tmp/bundle.zip", '
        '"output_path": "/tmp/out.zip", "public_base_url": "https://public.example", '
        '"token_file": "/tmp/token", "secret_env_file": "/tmp/urls.env", '
        '"session_budget_ledger": "/tmp/budget.json", "created_at_epoch": 12.5, '
        '"max_live_minutes": 4, "max_live_deadline_epoch": 42.0, '
        '"selected_hourly_rate_usd": 0.2, "target_spend_usd": 1.0, '
        '"hard_cap_usd": 2.0, "max_hourly_rate_usd": 0.3',
        encoding="utf-8",
    )

    assert runner._regex_field("{}", "missing") == ""
    assert runner._regex_number("{}", "missing") is None
    recovered = runner._read_async_state(job_dir)

    assert recovered["status"] == "recovered_from_malformed_state"
    assert recovered["instance_id"] == 321
    assert recovered["output_path"] == "/tmp/out.zip"
    recovery_manifest = json.loads(
        (job_dir / "vast_wam_async_state_recovery_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert recovery_manifest["recovered_instance_id"] == 321


def test_create_blocks_on_preflight_before_paid_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common_create_harness(
        monkeypatch,
        staging_status="blocked",
        staging_blockers=["staging_not_ready"],
        self_test_status="failed",
        public_status="blocked",
        api_key="",
        api_gate_blockers=["api_gate_blocked"],
        bundle_blockers=["bundle_blocked"],
        image_blockers=["image_blocked"],
        session_blockers=["session_blocked"],
        session_warnings=["requested_max_spend_would_exceed_target"],
    )

    manifest = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "preflight",
        bundle_path=_write_bundle(tmp_path / "bundle.zip"),
        public_base_url="https://public.example",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        session_budget_ledger=tmp_path / "budget.json",
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["allow_paid_vast_launch"] is False
    assert "paid_vast_launch_not_authorized_by_runner_flag" in manifest["blockers"]
    assert "public_staging_url_stability_not_proven" in manifest["blockers"]
    result = json.loads(
        (tmp_path / "preflight" / "vast_provider_adapter_result.json").read_text()
    )
    assert result["reason"] == "vast_wam_async_create_preflight_blocked"


def test_create_accepts_direct_provider_url_files_without_public_tunnel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle.zip")
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_url_file = tmp_path / "provider_output_put_url.txt"
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    bundle_url = "https://object.example/bundle.zip?X-Amz-Signature=bundle-secret"
    output_url = "https://object.example/output.zip?X-Amz-Signature=output-secret"
    output_get_url = "https://object.example/output.zip?X-Amz-Signature=output-get-secret"
    bundle_url_file.write_text(bundle_url + "\n", encoding="utf-8")
    output_url_file.write_text(output_url + "\n", encoding="utf-8")
    output_get_url_file.write_text(output_get_url + "\n", encoding="utf-8")
    bundle_url_file.chmod(0o600)
    output_url_file.chmod(0o600)
    output_get_url_file.chmod(0o600)
    _install_common_create_harness(monkeypatch, public_status="passed")

    def fail_local_staging(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"local staging should be skipped for direct urls: {kwargs}")

    captured_verify: dict[str, Any] = {}

    def fake_verify_public_staging_urls(**kwargs: Any) -> dict[str, Any]:
        captured_verify.update(kwargs)
        return {"status": "passed", "blockers": []}

    monkeypatch.setattr(runner, "prepare_vast_bundle_staging", fail_local_staging)
    monkeypatch.setattr(runner, "run_local_staging_self_test", fail_local_staging)
    monkeypatch.setattr(runner, "verify_public_staging_urls", fake_verify_public_staging_urls)
    monkeypatch.setattr(
        runner,
        "_inline_provider_bundle_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(f"inline bundle should be skipped for direct urls: {kwargs}")
        ),
    )

    manifest = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "direct",
        bundle_path=bundle,
        public_base_url="",
        provider_bundle_url_file=bundle_url_file,
        provider_output_put_url_file=output_url_file,
        provider_output_get_url_file=output_get_url_file,
        token_file=tmp_path / "unused-token",
        secret_env_file=tmp_path / "unused-urls.env",
        session_budget_ledger=tmp_path / "budget.json",
        allow_paid_vast_launch=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["explicit_provider_urls_used"] is True
    assert manifest["provider_bundle_inline_transport_used"] is False
    assert "paid_vast_launch_not_authorized_by_runner_flag" in manifest["blockers"]
    assert captured_verify["provider_bundle_url"] == bundle_url
    assert captured_verify["provider_output_put_url"] == output_url
    assert captured_verify["required_consecutive_successes"] == 1
    assert captured_verify["allow_output_put_probe"] is False
    assert captured_verify["cleanup_output_probe"] is False
    assert captured_verify["require_bundle_fetch_probe"] is False
    direct_manifest = json.loads(
        (tmp_path / "direct" / "vast_wam_direct_provider_urls_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert direct_manifest["explicit_provider_urls_used"] is True
    assert direct_manifest["provider_output_get_url_file"]["mode_is_0600"] is True
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "direct").glob("*.json")
    )
    assert "bundle-secret" not in persisted
    assert "output-secret" not in persisted
    assert "output-get-secret" not in persisted


def test_create_forwards_allowed_machine_ids_to_offer_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle.zip")
    _install_common_create_harness(monkeypatch)
    captured_select: dict[str, Any] = {}

    def fake_select_offer(*_args: Any, **kwargs: Any) -> None:
        captured_select.update(kwargs)
        return None

    monkeypatch.setattr(runner, "_select_offer", fake_select_offer)

    manifest = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "allowed",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-allowed",
        secret_env_file=tmp_path / "urls-allowed.env",
        session_budget_ledger=tmp_path / "budget-allowed.json",
        allow_paid_vast_launch=True,
        excluded_machine_ids=[49407],
        allowed_machine_ids=[16571],
        generated_at="now",
    )

    assert manifest["reason"] == "no_vast_offer_selected"
    assert captured_select["excluded_machine_ids"] == [49407]
    assert captured_select["allowed_machine_ids"] == [16571]
    offer_manifest = json.loads(
        (tmp_path / "allowed" / "vast_offer_selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert offer_manifest["excluded_machine_ids"] == [49407]
    assert offer_manifest["allowed_machine_ids"] == [16571]


def test_create_lock_inventory_offer_hard_cap_and_instance_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle.zip")

    _install_common_create_harness(monkeypatch, lock_handle=None, lock_manifest={})
    lock_blocked = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "lock",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-lock",
        secret_env_file=tmp_path / "urls-lock.env",
        session_budget_ledger=tmp_path / "budget-lock.json",
        allow_paid_vast_launch=True,
        generated_at="now",
    )
    assert lock_blocked["blockers"] == ["vast_paid_launch_lock_busy"]

    _install_common_create_harness(monkeypatch, inventory_blockers=["active_instance"])
    inventory_blocked = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "inventory",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-inventory",
        secret_env_file=tmp_path / "urls-inventory.env",
        session_budget_ledger=tmp_path / "budget-inventory.json",
        allow_paid_vast_launch=True,
        generated_at="now",
    )
    assert inventory_blocked["reason"] == "vast_prelaunch_inventory_guard_blocked"

    _install_common_create_harness(monkeypatch, offers=[], selected_offer=None)
    no_offer = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "no-offer",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-no-offer",
        secret_env_file=tmp_path / "urls-no-offer.env",
        session_budget_ledger=tmp_path / "budget-no-offer.json",
        allow_paid_vast_launch=True,
        generated_at="now",
    )
    assert no_offer["reason"] == "no_vast_offer_selected"

    expensive = {"ask_contract_id": 808, "hourly_rate_usd": 30.0}
    _install_common_create_harness(monkeypatch, offers=[expensive], selected_offer=expensive)
    hard_cap = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "hard-cap",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-hard-cap",
        secret_env_file=tmp_path / "urls-hard-cap.env",
        session_budget_ledger=tmp_path / "budget-hard-cap.json",
        allow_paid_vast_launch=True,
        hard_cap_usd=0.01,
        generated_at="now",
    )
    assert hard_cap["reason"] == "selected_offer_projected_max_runtime_exceeds_hard_cap"

    _install_common_create_harness(monkeypatch, create_response={})
    missing_instance = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "missing-instance",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-missing-instance",
        secret_env_file=tmp_path / "urls-missing-instance.env",
        session_budget_ledger=tmp_path / "budget-missing-instance.json",
        allow_paid_vast_launch=True,
        generated_at="now",
    )
    assert missing_instance["reason"] == "vast_create_response_missing_instance_id"

    _install_common_create_harness(
        monkeypatch,
        create_exception=urllib.error.HTTPError(
            url="https://vast.ai",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=None,
        ),
    )
    http_error = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "create-http-error",
        bundle_path=bundle,
        public_base_url="https://public.example",
        token_file=tmp_path / "token-create-http-error",
        secret_env_file=tmp_path / "urls-create-http-error.env",
        session_budget_ledger=tmp_path / "budget-create-http-error.json",
        allow_paid_vast_launch=True,
        generated_at="now",
    )
    assert http_error["status"] == "blocked"
    assert http_error["blockers"] == ["vast_create_instance_http_error:400"]
    persisted_error = json.loads(
        (tmp_path / "create-http-error" / "vast_wam_async_create_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_error["create_http_status_code"] == 400


def test_create_success_writes_async_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_offer = {
        "ask_contract_id": 707,
        "hourly_rate_usd": 0.2,
        "gpu_name": "RTX 6000 Ada",
        "gpu_ram_mb": 49152,
    }
    _install_common_create_harness(
        monkeypatch,
        offers=[selected_offer],
        selected_offer=selected_offer,
        create_response={"new_contract": 5150},
    )
    selected_offer_kwargs: dict[str, Any] = {}

    def fake_select_offer(_offers: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        selected_offer_kwargs.update(kwargs)
        return selected_offer

    monkeypatch.setattr(runner, "_select_offer", fake_select_offer)
    manifest = runner.create_async_vast_wam_run(
        job_dir=tmp_path / "created",
        bundle_path=_write_bundle(tmp_path / "bundle.zip"),
        public_base_url="https://public.example",
        token_file=tmp_path / "token",
        secret_env_file=tmp_path / "urls.env",
        output_path=tmp_path / "created-output.zip",
        session_budget_ledger=tmp_path / "budget.json",
        allow_paid_vast_launch=True,
        max_live_minutes=3,
        min_gpu_ram_mb=48000,
        generated_at="now",
    )

    assert manifest["status"] == "instance_created"
    assert manifest["instance_id"] == 5150
    assert manifest["min_gpu_ram_mb"] == 48000
    assert selected_offer_kwargs["min_gpu_ram_mb"] == 48000
    state = json.loads(Path(manifest["state_path"]).read_text())
    assert state["status"] == "instance_created"
    assert state["min_gpu_ram_mb"] == 48000
    assert state["create_request_summary"] == {"summary": True}
    offer_manifest = json.loads(
        (tmp_path / "created" / "vast_offer_selection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert offer_manifest["min_gpu_ram_mb"] == 48000


def test_poll_blocks_for_missing_state_and_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_state_job = tmp_path / "missing-state"
    _write_json(runner._state_path(missing_state_job), {})

    missing_state = runner.poll_async_vast_wam_run(
        job_dir=missing_state_job,
        generated_at="now",
    )
    assert missing_state["blockers"] == ["vast_wam_async_state_missing_instance_id"]

    missing_secret_job = tmp_path / "missing-secret"
    _write_state(missing_secret_job)
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("", {}))

    missing_secret = runner.poll_async_vast_wam_run(
        job_dir=missing_secret_job,
        generated_at="now",
    )
    assert missing_secret["blockers"] == ["missing_file_based_secret_VAST_API_KEY_FILE"]


def test_write_poll_phase_artifacts_for_completed_and_blocked_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    state = {"instance_id": 123, "output_path": str(tmp_path / "out.zip")}

    no_zip = runner._write_poll_phase_artifacts(
        job_dir=tmp_path / "no-zip",
        generated_at="now",
        state=state,
        heartbeat_text="BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:remote_failure nvidia-smi: command not found",
        onstart_logs={"output_log_path": str(tmp_path / "no-zip.log")},
        output_download_manifest={"status": "not_requested"},
        output_zip_inspection={
            "zip_present": False,
            "runtime_result_present": False,
            "mp4_validation": {},
            "video_smoke_proven": False,
        },
    )
    assert "provider_runtime_output_zip_not_received_locally" in no_zip["blockers"]
    assert "provider_remote_blocker:remote_failure" in no_zip["blockers"]

    runtime_missing = runner._write_poll_phase_artifacts(
        job_dir=tmp_path / "runtime-missing",
        generated_at="now",
        state=state,
        heartbeat_text=(
            "BLUEPRINT_VAST_HEARTBEAT_OK BLUEPRINT_VAST_GPU_SANITY_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED "
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED "
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED"
        ),
        onstart_logs={"output_log_path": str(tmp_path / "runtime-missing.log")},
        output_download_manifest={"status": "completed"},
        output_zip_inspection={
            "zip_present": True,
            "runtime_result_present": False,
            "runtime_result": {},
            "mp4_validation": {"blockers": ["bad_mp4"]},
            "video_smoke_proven": False,
            "mp4_count": 0,
            "mp4_members": [],
        },
    )
    assert "provider_runtime_result_missing_from_output_zip" in runtime_missing["blockers"]

    completed = runner._write_poll_phase_artifacts(
        job_dir=tmp_path / "completed",
        generated_at="now",
        state=state,
        heartbeat_text=(
            "BLUEPRINT_VAST_HEARTBEAT_OK BLUEPRINT_VAST_GPU_SANITY_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED "
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED "
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED"
        ),
        onstart_logs={"output_log_path": str(tmp_path / "completed.log")},
        output_download_manifest={"status": "completed"},
        output_zip_inspection={
            "zip_present": True,
            "runtime_result_present": True,
            "runtime_result": {"status": "completed", "blockers": []},
            "mp4_validation": {},
            "video_smoke_proven": True,
            "mp4_count": 4,
            "mp4_members": ["a.mp4"],
        },
    )
    assert completed["status"] == "completed"
    assert completed["blueprint_provider_bundle_execution_proven"] is True


def _install_poll_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    heartbeat_text: str,
    output_zip_inspection: dict[str, Any],
    delete_effect: Callable[[], tuple[int, dict[str, Any]]] | Exception | None = None,
    status_probe_error: bool = False,
) -> None:
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("secret-vast-key", {}))
    monkeypatch.setattr(runner, "_forwarded_secret_values", lambda: [])
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_fill_missing_phase_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_instance_status", lambda _payload: "running")
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner, "_budget_ledger", lambda **_kwargs: {"estimated_cost_usd": 0.02})
    monkeypatch.setattr(runner, "_append_session_budget_attempt", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_final_validation", lambda **kwargs: {"status": "completed" if not kwargs["continuing_spend"] else "blocked"})
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(runner.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: output_zip_inspection,
    )

    def fake_logs(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_log_path"]).write_text(heartbeat_text, encoding="utf-8")
        return {"output_log_path": str(kwargs["output_log_path"]), "status": "completed"}

    monkeypatch.setattr(runner, "_request_logs_and_fetch", fake_logs)

    def fake_api_json(**kwargs: Any) -> tuple[int, dict[str, Any]]:
        if kwargs["method"] == "GET":
            if status_probe_error:
                raise RuntimeError("status failed")
            return 200, {"instances": {"actual_status": "running"}}
        if kwargs["method"] == "DELETE":
            if isinstance(delete_effect, Exception):
                raise delete_effect
            if delete_effect is not None:
                return delete_effect()
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(runner, "_api_json", fake_api_json)


def test_poll_deferred_running_and_teardown_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running_job = tmp_path / "running"
    _write_state(running_job, max_live_deadline_epoch=1_000.0)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text="BLUEPRINT_VAST_HEARTBEAT_OK BLUEPRINT_VAST_GPU_SANITY_OK",
        output_zip_inspection={
            "zip_present": False,
            "runtime_result_present": False,
            "runtime_result": {},
            "mp4_validation": {},
            "video_smoke_proven": False,
        },
    )

    running = runner.poll_async_vast_wam_run(job_dir=running_job, generated_at="now")

    assert running["status"] == "running"
    teardown = json.loads((running_job / "vast_teardown_manifest.json").read_text())
    assert teardown["status"] == "deferred_async_run_still_active"

    completed_job = tmp_path / "completed"
    _write_state(completed_job, max_live_deadline_epoch=1_000.0)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text=(
            "BLUEPRINT_VAST_HEARTBEAT_OK BLUEPRINT_VAST_GPU_SANITY_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED "
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED "
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED"
        ),
        output_zip_inspection={
            "zip_present": True,
            "runtime_result_present": True,
            "runtime_result": {"status": "completed", "blockers": []},
            "mp4_validation": {},
            "video_smoke_proven": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 4,
        },
    )

    completed = runner.poll_async_vast_wam_run(job_dir=completed_job, generated_at="now")

    assert completed["status"] == "completed"
    assert completed["teardown_performed"] is True
    state = json.loads(runner._state_path(completed_job).read_text())
    assert state["status"] == "teardown_completed"


def test_poll_extends_missing_container_retry_count_with_wait_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "missing-container-retry-window"
    _write_state(job, max_live_deadline_epoch=1_000.0)
    observed: dict[str, Any] = {}
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("secret-vast-key", {}))
    monkeypatch.setattr(runner, "_forwarded_secret_values", lambda: [])
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_fill_missing_phase_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_instance_status", lambda _payload: "running")
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner, "_budget_ledger", lambda **_kwargs: {"estimated_cost_usd": 0.01})
    monkeypatch.setattr(runner, "_append_session_budget_attempt", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_final_validation", lambda **_kwargs: {"status": "completed"})
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": False,
            "runtime_result_present": False,
            "runtime_result": {},
            "mp4_validation": {},
            "video_smoke_proven": False,
        },
    )

    def fake_logs(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        Path(kwargs["output_log_path"]).write_text(
            "Error response from daemon: No such container: C.123",
            encoding="utf-8",
        )
        return {"output_log_path": str(kwargs["output_log_path"]), "status": "blocked"}

    monkeypatch.setattr(runner, "_request_logs_and_fetch", fake_logs)
    monkeypatch.setattr(
        runner,
        "_api_json",
        lambda **kwargs: (200, {"success": True})
        if kwargs["method"] == "DELETE"
        else (200, {"instances": {"actual_status": "running"}}),
    )

    result = runner.poll_async_vast_wam_run(
        job_dir=job,
        max_wait_seconds=900,
        retry_interval_seconds=15,
        teardown=True,
        generated_at="now",
    )

    assert result["status"] == "blocked"
    # The missing-container tolerance is capped at the bounded boot/pull window (default 720s)
    # rather than the full live window, so a dud offer is torn down quickly: 720s / 15s = 48
    # (not 900s / 15s = 60). This is what stops a never-booting container idling the deadline.
    assert observed["container_missing_retry_attempts"] == 48


def test_poll_missing_container_cap_is_env_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "missing-container-retry-env"
    _write_state(job, max_live_deadline_epoch=1_000.0)
    observed: dict[str, Any] = {}
    monkeypatch.setenv(runner.VAST_WAM_CONTAINER_MISSING_MAX_SECONDS_ENV, "300")
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("secret-vast-key", {}))
    monkeypatch.setattr(runner, "_forwarded_secret_values", lambda: [])
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_fill_missing_phase_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_instance_status", lambda _payload: "running")
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner, "_budget_ledger", lambda **_kwargs: {"estimated_cost_usd": 0.01})
    monkeypatch.setattr(runner, "_append_session_budget_attempt", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_final_validation", lambda **_kwargs: {"status": "completed"})
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": False,
            "runtime_result_present": False,
            "runtime_result": {},
            "mp4_validation": {},
            "video_smoke_proven": False,
        },
    )

    def fake_logs(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        Path(kwargs["output_log_path"]).write_text("No such container: C.123", encoding="utf-8")
        return {"output_log_path": str(kwargs["output_log_path"]), "status": "blocked"}

    monkeypatch.setattr(runner, "_request_logs_and_fetch", fake_logs)
    monkeypatch.setattr(
        runner,
        "_api_json",
        lambda **kwargs: (200, {"success": True})
        if kwargs["method"] == "DELETE"
        else (200, {"instances": {"actual_status": "running"}}),
    )

    runner.poll_async_vast_wam_run(
        job_dir=job,
        max_wait_seconds=900,
        retry_interval_seconds=15,
        teardown=True,
        generated_at="now",
    )

    # 300s override / 15s interval = 20 (well under the full-window 60)
    assert observed["container_missing_retry_attempts"] == 20


def test_poll_caps_log_wait_to_remaining_live_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "deadline-capped-log-wait"
    _write_state(job, max_live_deadline_epoch=130.0)
    observed: dict[str, Any] = {}
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("secret-vast-key", {}))
    monkeypatch.setattr(runner, "_forwarded_secret_values", lambda: [])
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_fill_missing_phase_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_instance_status", lambda _payload: "running")
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner, "_budget_ledger", lambda **_kwargs: {"estimated_cost_usd": 0.01})
    monkeypatch.setattr(runner, "_append_session_budget_attempt", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_final_validation", lambda **_kwargs: {"status": "completed"})
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner,
        "_inspect_provider_runtime_output_zip",
        lambda *_args, **_kwargs: {
            "zip_present": False,
            "runtime_result_present": False,
            "runtime_result": {},
            "mp4_validation": {},
            "video_smoke_proven": False,
        },
    )

    def fake_logs(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        Path(kwargs["output_log_path"]).write_text(
            "BLUEPRINT_VAST_HEARTBEAT_OK",
            encoding="utf-8",
        )
        return {"output_log_path": str(kwargs["output_log_path"]), "status": "completed"}

    monkeypatch.setattr(runner, "_request_logs_and_fetch", fake_logs)
    monkeypatch.setattr(
        runner,
        "_api_json",
        lambda **kwargs: (200, {"success": True})
        if kwargs["method"] == "DELETE"
        else (200, {"instances": {"actual_status": "running"}}),
    )

    result = runner.poll_async_vast_wam_run(
        job_dir=job,
        max_wait_seconds=900,
        retry_interval_seconds=15,
        teardown=True,
        generated_at="now",
    )

    assert result["status"] == "blocked"
    assert observed["max_wait_seconds"] == 30
    assert observed["container_missing_retry_attempts"] == 2
    assert result["effective_log_fetch_max_wait_seconds"] == 30
    assert result["log_wait_deadline_cap_applied"] is True


def test_poll_downloads_direct_provider_output_get_url_without_leaking_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "direct-poll"
    bundle_url_file = tmp_path / "provider_bundle_url.txt"
    output_put_url_file = tmp_path / "provider_output_put_url.txt"
    output_get_url_file = tmp_path / "provider_output_get_url.txt"
    bundle_url_file.write_text(
        "https://object.example/bundle.zip?X-Amz-Signature=bundle-secret\n",
        encoding="utf-8",
    )
    output_put_url_file.write_text(
        "https://object.example/output.zip?X-Amz-Signature=put-secret\n",
        encoding="utf-8",
    )
    output_get_url_file.write_text(
        "https://object.example/output.zip?X-Amz-Signature=get-secret\n",
        encoding="utf-8",
    )
    for path in (bundle_url_file, output_put_url_file, output_get_url_file):
        path.chmod(0o600)
    _write_state(
        job,
        explicit_provider_urls_used=True,
        output_path=str(job / "vast_provider_runtime_output.zip"),
        provider_bundle_url_file={"path": str(bundle_url_file)},
        provider_output_put_url_file={"path": str(output_put_url_file)},
        provider_output_get_url_file={"path": str(output_get_url_file)},
    )
    _install_poll_harness(
        monkeypatch,
        heartbeat_text=(
            "BLUEPRINT_VAST_HEARTBEAT_OK BLUEPRINT_VAST_GPU_SANITY_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_DOWNLOADED "
            "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_STARTED "
            "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK "
            "BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED"
        ),
        output_zip_inspection={
            "zip_present": True,
            "runtime_result_present": True,
            "runtime_result": {"status": "completed", "blockers": []},
            "mp4_validation": {},
            "video_smoke_proven": True,
            "runtime_result_status": "completed",
            "runtime_result_blockers": [],
            "mp4_count": 4,
        },
    )

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"zip-bytes"

    requested_urls: list[str] = []

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        del timeout
        requested_urls.append(getattr(request, "full_url", str(request)))
        return FakeResponse()

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    result = runner.poll_async_vast_wam_run(job_dir=job, generated_at="now")

    assert result["status"] == "completed"
    assert requested_urls == ["https://object.example/output.zip?X-Amz-Signature=get-secret"]
    assert (job / "vast_provider_runtime_output.zip").read_bytes() == b"zip-bytes"
    download_manifest = json.loads(
        (job / "vast_provider_output_download_manifest.json").read_text(encoding="utf-8")
    )
    assert download_manifest["status"] == "completed"
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in job.glob("*.json"))
    assert "bundle-secret" not in persisted
    assert "put-secret" not in persisted
    assert "get-secret" not in persisted


def test_poll_teardown_http_and_generic_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_zip = {
        "zip_present": False,
        "runtime_result_present": False,
        "runtime_result": {},
        "mp4_validation": {},
        "video_smoke_proven": False,
    }

    already_absent_job = tmp_path / "already-absent"
    _write_state(already_absent_job)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text="BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
        output_zip_inspection=output_zip,
        delete_effect=urllib.error.HTTPError(
            url="https://vast.ai",
            code=404,
            msg="missing",
            hdrs=None,
            fp=None,
        ),
    )
    already_absent = runner.poll_async_vast_wam_run(
        job_dir=already_absent_job,
        teardown=True,
        generated_at="now",
    )
    assert already_absent["continuing_spend_from_this_run"] is False

    failed_delete_job = tmp_path / "failed-delete"
    _write_state(failed_delete_job)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text="BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
        output_zip_inspection=output_zip,
        delete_effect=urllib.error.HTTPError(
            url="https://vast.ai",
            code=500,
            msg="failed",
            hdrs=None,
            fp=None,
        ),
    )
    failed_delete = runner.poll_async_vast_wam_run(
        job_dir=failed_delete_job,
        teardown=True,
        generated_at="now",
    )
    assert failed_delete["status"] == "running"

    generic_failure_job = tmp_path / "generic-failure"
    _write_state(generic_failure_job)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text="BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
        output_zip_inspection=output_zip,
        delete_effect=RuntimeError("delete failed"),
        status_probe_error=True,
    )
    generic_failure = runner.poll_async_vast_wam_run(
        job_dir=generic_failure_job,
        teardown=True,
        generated_at="now",
    )
    assert generic_failure["instance_status"] == "status_probe_failed:RuntimeError"
    teardown = json.loads((generic_failure_job / "vast_teardown_manifest.json").read_text())
    assert teardown["teardown_actions_performed"][0]["error_type"] == "RuntimeError"


def test_poll_teardown_retries_transient_destroy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destroy that fails transiently on the first attempt (e.g. instance still ``loading``)
    must be retried so the instance does not keep billing — the exact leak observed on a dud
    offer whose first DELETE was rejected.
    """
    output_zip = {
        "zip_present": False,
        "runtime_result_present": False,
        "runtime_result": {},
        "mp4_validation": {},
        "video_smoke_proven": False,
    }
    attempts = {"n": 0}

    def flaky_delete() -> tuple[int, dict[str, Any]]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise urllib.error.HTTPError(
                url="https://vast.ai", code=500, msg="still loading", hdrs=None, fp=None
            )
        return (200, {"success": True})

    job = tmp_path / "retry-destroy"
    _write_state(job)
    _install_poll_harness(
        monkeypatch,
        heartbeat_text="BLUEPRINT_VAST_PROVIDER_BUNDLE_COMPLETED_OR_BLOCKED",
        output_zip_inspection=output_zip,
        delete_effect=flaky_delete,
    )

    result = runner.poll_async_vast_wam_run(job_dir=job, teardown=True, generated_at="now")

    assert result["continuing_spend_from_this_run"] is False  # retry succeeded -> no spend leak
    assert attempts["n"] == 2  # one failure then one success
    teardown = json.loads((job / "vast_teardown_manifest.json").read_text())
    actions = teardown["teardown_actions_performed"]
    assert actions[0]["status"] == "failed" and actions[0]["attempt"] == 1
    assert actions[-1]["status"] == "completed" and actions[-1]["attempt"] == 2


def test_direct_destroy_async_vast_wam_run_skips_log_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "destroy-direct"
    _write_state(job)
    calls: list[tuple[str, str]] = []
    monkeypatch.setenv(runner.VAST_API_GATE_ENV, "true")
    monkeypatch.setenv(runner.VAST_INSTANCE_LAUNCH_GATE_ENV, "true")
    monkeypatch.setattr(runner, "_read_secret_file", lambda *_args, **_kwargs: ("secret-vast-key", {}))
    monkeypatch.setattr(runner, "_append_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_redact_runtime_value", lambda value, _secrets: value)
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)

    def fake_api_json(**kwargs: Any) -> tuple[int, dict[str, Any]]:
        calls.append((kwargs["method"], kwargs["path"]))
        if kwargs["method"] == "DELETE":
            return 200, {"success": True}
        raise AssertionError(kwargs)

    monkeypatch.setattr(runner, "_api_json", fake_api_json)

    manifest = runner.destroy_async_vast_wam_run(job_dir=job, generated_at="now")

    assert calls == [("DELETE", "/instances/123/")]
    assert manifest["status"] == "completed"
    assert manifest["continuing_spend_from_this_run"] is False
    teardown = json.loads((job / "vast_teardown_manifest.json").read_text())
    assert teardown["runner_gpu_teardown_completed"] is True
    state = json.loads(runner._state_path(job).read_text())
    assert state["status"] == "teardown_completed"


def test_async_runner_main_dispatches_create_and_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_create: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> dict[str, Any]:
        captured_create.update(kwargs)
        return {"status": "instance_created"}

    monkeypatch.setattr(runner, "create_async_vast_wam_run", fake_create)

    create_code = runner.main(
        [
            "create",
            "--job-dir",
            str(tmp_path / "job"),
            "--bundle-path",
            str(tmp_path / "bundle.zip"),
            "--public-base-url",
            "https://public.example",
            "--provider-output-get-url-file",
            str(tmp_path / "provider-output-get-url.txt"),
            "--token-file",
            str(tmp_path / "token"),
            "--secret-env-file",
            str(tmp_path / "urls.env"),
            "--output-path",
            str(tmp_path / "out.zip"),
            "--session-budget-ledger",
            str(tmp_path / "budget.json"),
            "--allow-paid-vast-launch",
            "--max-hourly-rate",
            "0.3",
            "--target-spend-usd",
            "1.5",
            "--hard-cap-usd",
            "2.5",
            "--allow-target-spend-overrun",
            "--max-live-minutes",
            "7",
            "--session-max-live-minutes",
            "9",
            "--startup-poll-seconds",
            "11",
            "--public-staging-verify-max-wait-seconds",
            "13",
            "--public-staging-verify-retry-interval-seconds",
            "0.5",
            "--public-staging-verify-timeout-seconds",
            "1.5",
            "--public-staging-required-consecutive-successes",
            "4",
            "--public-image",
            "image:test",
            "--vast-launch-mode",
            "ssh",
            "--disk-gb",
            "88",
            "--heartbeat-url",
            "https://heartbeat.example",
        ]
    )

    assert create_code == 0
    assert captured_create["allow_paid_vast_launch"] is True
    assert captured_create["provider_output_get_url_file"] == str(
        tmp_path / "provider-output-get-url.txt"
    )
    assert captured_create["disk_gb"] == 88
    assert json.loads(capsys.readouterr().out)["status"] == "instance_created"

    monkeypatch.setattr(
        runner,
        "poll_async_vast_wam_run",
        lambda **kwargs: {"status": "blocked", "teardown": kwargs["teardown"]},
    )
    poll_code = runner.main(
        [
            "poll",
            "--job-dir",
            str(tmp_path / "job"),
            "--max-wait-seconds",
            "3",
            "--retry-interval-seconds",
            "1",
            "--teardown",
        ]
    )

    assert poll_code == 1
    assert json.loads(capsys.readouterr().out)["teardown"] is True

    monkeypatch.setattr(
        runner,
        "destroy_async_vast_wam_run",
        lambda **kwargs: {"status": "completed", "job_dir": str(kwargs["job_dir"])},
    )
    destroy_code = runner.main(["destroy", "--job-dir", str(tmp_path / "job")])

    assert destroy_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
