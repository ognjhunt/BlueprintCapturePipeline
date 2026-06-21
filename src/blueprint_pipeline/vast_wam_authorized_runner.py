"""Coordinate a gated Vast run for OSCAR/Cosmos-style WAM bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .vast_bundle_staging import (
    BUNDLE_ROUTE,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_SECRET_ENV_FILE,
    DEFAULT_TOKEN_FILE,
    OUTPUT_ROUTE,
    _read_or_create_token,
    _url_with_token,
    prepare_vast_bundle_staging,
    run_local_staging_self_test,
    verify_public_staging_urls,
)
from .vast_provider_adapter import (
    DEFAULT_HARD_CAP_USD,
    DEFAULT_MAX_HOURLY_RATE,
    DEFAULT_TARGET_SPEND_USD,
    VAST_API_GATE_ENV,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    run_vast_provider_adapter,
    _vast_session_budget_ledger_path,
)
from .vast_authorized_probe_runner import (
    _staging_verification_guard,
    _target_spend_guard,
)


VAST_WAM_AUTHORIZED_RUNNER_SCHEMA_VERSION = "vast_wam_authorized_runner.v1"
DEFAULT_WAM_PUBLIC_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
DEFAULT_WAM_VAST_LAUNCH_MODE = "args"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _redacted_path(route: str) -> str:
    return f"/{route.strip('/')}?token=<redacted-token>"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _truth_boundaries() -> dict[str, Any]:
    return {
        "wam_vla_runtime_proven": False,
        "action_conditioned_video_rollout_generated": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "official_policy_execution_proven": False,
        "controller_grade_execution_proven": False,
    }


def run_vast_wam_authorized_runner(
    *,
    job_dir: str | Path,
    bundle_path: str | Path,
    public_base_url: str | None = None,
    token_file: str | Path | None = None,
    secret_env_file: str | Path | None = None,
    output_path: str | Path | None = None,
    session_budget_ledger: str | Path | None = None,
    allow_paid_vast_launch: bool = False,
    max_hourly_rate: float = DEFAULT_MAX_HOURLY_RATE,
    target_spend_usd: float = DEFAULT_TARGET_SPEND_USD,
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD,
    allow_target_spend_overrun: bool = False,
    max_live_minutes: int = 45,
    session_max_live_minutes: int | None = 45,
    startup_timeout_seconds: int = 1800,
    verify_staging_urls: bool = True,
    allow_unverified_public_staging_for_paid_launch: bool = False,
    public_staging_verify_max_wait_seconds: int = 180,
    public_staging_verify_retry_interval_seconds: float = 5.0,
    public_staging_verify_timeout_seconds: float = 20.0,
    public_staging_required_consecutive_successes: int = 3,
    allow_staging_output_put_probe: bool = True,
    public_image: str = DEFAULT_WAM_PUBLIC_IMAGE,
    vast_launch_mode: str = DEFAULT_WAM_VAST_LAUNCH_MODE,
    vast_template_hash_id: str | None = None,
    use_vast_template_image: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_job_dir = Path(job_dir).expanduser().resolve()
    resolved_bundle = Path(bundle_path).expanduser().resolve()
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else resolved_job_dir / DEFAULT_OUTPUT_FILENAME
    )
    resolved_token_file = (
        Path(token_file).expanduser().resolve()
        if token_file
        else Path(DEFAULT_TOKEN_FILE).expanduser().resolve()
    )
    resolved_secret_env_file = (
        Path(secret_env_file).expanduser().resolve()
        if secret_env_file
        else Path(DEFAULT_SECRET_ENV_FILE).expanduser().resolve()
    )
    resolved_session_budget_ledger = (
        Path(session_budget_ledger).expanduser().resolve()
        if session_budget_ledger
        else _vast_session_budget_ledger_path()
    )
    ensure_dir(resolved_job_dir)

    token, token_status = _read_or_create_token(resolved_token_file)
    staging_manifest = prepare_vast_bundle_staging(
        job_dir=resolved_job_dir,
        bundle_path=resolved_bundle,
        public_base_url=public_base_url,
        token_file=resolved_token_file,
        secret_env_file=resolved_secret_env_file,
        output_path=resolved_output,
        generated_at=generated,
    )
    self_test = run_local_staging_self_test(
        job_dir=resolved_job_dir,
        bundle_path=resolved_bundle,
        output_path=resolved_job_dir / "vast_wam_staging_self_test_output.zip",
        token_file=resolved_token_file,
        generated_at=generated,
    )

    blockers: list[str] = []
    if staging_manifest.get("status") != "ready":
        blockers.extend(str(item) for item in staging_manifest.get("blockers") or [])
    if self_test.get("status") != "passed":
        blockers.append("local_staging_self_test_failed")
    provider_bundle_url = ""
    provider_output_put_url = ""
    if _string(public_base_url):
        provider_bundle_url = _url_with_token(_string(public_base_url), BUNDLE_ROUTE, token)
        provider_output_put_url = _url_with_token(_string(public_base_url), OUTPUT_ROUTE, token)
    else:
        blockers.append("public_base_url_missing_for_paid_vast_launch")
    public_staging_verification: dict[str, Any] = {"status": "not_requested"}
    if (
        allow_paid_vast_launch
        and verify_staging_urls
        and not allow_unverified_public_staging_for_paid_launch
        and provider_bundle_url
        and provider_output_put_url
    ):
        public_staging_verification = verify_public_staging_urls(
            job_dir=resolved_job_dir,
            provider_bundle_url=provider_bundle_url,
            provider_output_put_url=provider_output_put_url,
            bundle_path=resolved_bundle,
            output_path=resolved_output,
            max_wait_seconds=public_staging_verify_max_wait_seconds,
            retry_interval_seconds=public_staging_verify_retry_interval_seconds,
            timeout_seconds=public_staging_verify_timeout_seconds,
            required_consecutive_successes=public_staging_required_consecutive_successes,
            allow_output_put_probe=allow_staging_output_put_probe,
            cleanup_output_probe=True,
            generated_at=generated,
        )
        if public_staging_verification.get("status") != "passed":
            blockers.extend(
                str(item)
                for item in public_staging_verification.get("blockers")
                or ["public_staging_url_stability_not_proven"]
            )
    staging_verification_guard = _staging_verification_guard(
        verify_staging_urls=verify_staging_urls,
        allow_unverified_public_staging_for_paid_launch=(
            allow_unverified_public_staging_for_paid_launch
        ),
        staging_manifest=staging_manifest,
        public_staging_verification=public_staging_verification,
    )
    if allow_paid_vast_launch:
        blockers.extend(str(item) for item in staging_verification_guard.get("blockers") or [])
    target_spend_guard = _target_spend_guard(
        budget_path=resolved_session_budget_ledger,
        target_spend_usd=target_spend_usd,
        max_hourly_rate=max_hourly_rate,
        max_live_minutes=max_live_minutes,
        allow_target_spend_overrun=allow_target_spend_overrun,
    )
    if allow_paid_vast_launch:
        blockers.extend(str(item) for item in target_spend_guard.get("blockers") or [])

    adapter_result: dict[str, Any] | None = None
    paid_launch_attempted = False
    if allow_paid_vast_launch:
        if not provider_bundle_url or not provider_output_put_url:
            blockers.append("paid_vast_launch_requires_public_staging_urls")
        elif blockers:
            blockers.append("paid_vast_launch_preflight_blocked")
        else:
            paid_launch_attempted = True
            adapter_result = run_vast_provider_adapter(
                job_dir=resolved_job_dir,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate,
                target_spend_usd=target_spend_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=max_live_minutes,
                public_image=public_image,
                provider_bundle=resolved_bundle,
                provider_bundle_url=provider_bundle_url,
                provider_output_put_url=provider_output_put_url,
                provider_runtime_output_zip=resolved_output,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind="wam",
                vast_launch_mode=vast_launch_mode,
                startup_timeout_seconds=startup_timeout_seconds,
                session_budget_ledger_path=resolved_session_budget_ledger,
                session_max_live_minutes=session_max_live_minutes,
                verify_staging_urls=verify_staging_urls,
                ngc_image_login_mode="never",
                vast_template_hash_id=vast_template_hash_id,
                use_vast_template_image=use_vast_template_image,
                require_known_supported_isaac_driver=False,
                disk_gb=80,
            )
            if adapter_result.get("status") != "completed":
                blockers.extend(str(item) for item in adapter_result.get("blockers") or [])
    else:
        blockers.append("paid_vast_launch_not_authorized_by_runner_flag")

    status = (
        "completed"
        if paid_launch_attempted and adapter_result and adapter_result.get("status") == "completed"
        else "blocked"
    )
    output_inspection: dict[str, Any] = {}
    if resolved_output.is_file():
        try:
            output_inspection = {
                "output_zip_present": True,
                "output_zip_size_bytes": resolved_output.stat().st_size,
            }
        except OSError:
            output_inspection = {"output_zip_present": True}
    manifest = {
        "schema_version": VAST_WAM_AUTHORIZED_RUNNER_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "bundle_present": resolved_bundle.is_file(),
        "bundle_size_bytes": resolved_bundle.stat().st_size if resolved_bundle.is_file() else 0,
        "output_path": str(resolved_output),
        "output_inspection": output_inspection,
        "token_file": token_status,
        "secret_env_file": str(resolved_secret_env_file),
        "public_base_url_present": bool(_string(public_base_url)),
        "bundle_url_path": _redacted_path(BUNDLE_ROUTE),
        "output_put_url_path": _redacted_path(OUTPUT_ROUTE),
        "local_staging_self_test_status": self_test.get("status"),
        "staging_manifest_status": staging_manifest.get("status"),
        "provider_bundle_kind": "wam",
        "public_image": public_image,
        "vast_launch_mode": vast_launch_mode,
        "allow_paid_vast_launch": allow_paid_vast_launch,
        "paid_launch_attempted": paid_launch_attempted,
        "adapter_result_status": adapter_result.get("status") if adapter_result else None,
        "adapter_result_reason": adapter_result.get("reason") if adapter_result else None,
        "adapter_result_path": str(resolved_job_dir / "vast_provider_adapter_result.json")
        if adapter_result
        else None,
        "session_budget_ledger": str(resolved_session_budget_ledger),
        "max_live_minutes": max_live_minutes,
        "session_max_live_minutes": session_max_live_minutes,
        "startup_timeout_seconds": startup_timeout_seconds,
        "staging_verification_guard": staging_verification_guard,
        "public_staging_verification_status": public_staging_verification.get("status"),
        "public_staging_verification_path": str(
            resolved_job_dir / "vast_public_staging_verification.json"
        )
        if public_staging_verification.get("status") != "not_requested"
        else None,
        "public_staging_required_consecutive_successes": (
            public_staging_required_consecutive_successes
        ),
        "allow_staging_output_put_probe": allow_staging_output_put_probe,
        "target_spend_guard": target_spend_guard,
        "allow_target_spend_overrun": allow_target_spend_overrun,
        "allow_unverified_public_staging_for_paid_launch": (
            allow_unverified_public_staging_for_paid_launch
        ),
        "adapter_env_gates_required": [VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV],
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        **_truth_boundaries(),
    }
    write_json(resolved_job_dir / "vast_wam_authorized_runner_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fail-closed staging and optional paid Vast WAM bundle execution."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--public-base-url")
    parser.add_argument("--token-file")
    parser.add_argument("--secret-env-file")
    parser.add_argument("--output-path")
    parser.add_argument("--session-budget-ledger")
    parser.add_argument("--allow-paid-vast-launch", action="store_true")
    parser.add_argument("--max-hourly-rate", type=float, default=DEFAULT_MAX_HOURLY_RATE)
    parser.add_argument("--target-spend-usd", type=float, default=DEFAULT_TARGET_SPEND_USD)
    parser.add_argument("--hard-cap-usd", type=float, default=DEFAULT_HARD_CAP_USD)
    parser.add_argument(
        "--allow-target-spend-overrun",
        action="store_true",
        help="Allow paid launch even if the projected request exceeds target spend; hard cap is still enforced by the adapter.",
    )
    parser.add_argument("--max-live-minutes", type=int, default=45)
    parser.add_argument("--session-max-live-minutes", type=int, default=45)
    parser.add_argument("--startup-timeout-seconds", type=int, default=1800)
    parser.add_argument("--no-verify-staging-urls", action="store_true")
    parser.add_argument(
        "--allow-unverified-public-staging-for-paid-launch",
        action="store_true",
        help="Allow a paid launch even when public staging URLs are not probed first. Use only with an independently verified tunnel.",
    )
    parser.add_argument("--public-staging-verify-max-wait-seconds", type=int, default=180)
    parser.add_argument("--public-staging-verify-retry-interval-seconds", type=float, default=5.0)
    parser.add_argument("--public-staging-verify-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--public-staging-required-consecutive-successes", type=int, default=3)
    parser.add_argument("--no-staging-output-put-probe", action="store_true")
    parser.add_argument("--public-image", default=DEFAULT_WAM_PUBLIC_IMAGE)
    parser.add_argument("--vast-launch-mode", default=DEFAULT_WAM_VAST_LAUNCH_MODE)
    parser.add_argument("--vast-template-hash-id")
    parser.add_argument("--use-vast-template-image", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_vast_wam_authorized_runner(
        job_dir=args.job_dir,
        bundle_path=args.bundle_path,
        public_base_url=args.public_base_url,
        token_file=args.token_file,
        secret_env_file=args.secret_env_file,
        output_path=args.output_path,
        session_budget_ledger=args.session_budget_ledger,
        allow_paid_vast_launch=args.allow_paid_vast_launch,
        max_hourly_rate=args.max_hourly_rate,
        target_spend_usd=args.target_spend_usd,
        hard_cap_usd=args.hard_cap_usd,
        allow_target_spend_overrun=args.allow_target_spend_overrun,
        max_live_minutes=args.max_live_minutes,
        session_max_live_minutes=args.session_max_live_minutes,
        startup_timeout_seconds=args.startup_timeout_seconds,
        verify_staging_urls=not args.no_verify_staging_urls,
        allow_unverified_public_staging_for_paid_launch=(
            args.allow_unverified_public_staging_for_paid_launch
        ),
        public_staging_verify_max_wait_seconds=args.public_staging_verify_max_wait_seconds,
        public_staging_verify_retry_interval_seconds=args.public_staging_verify_retry_interval_seconds,
        public_staging_verify_timeout_seconds=args.public_staging_verify_timeout_seconds,
        public_staging_required_consecutive_successes=(
            args.public_staging_required_consecutive_successes
        ),
        allow_staging_output_put_probe=not args.no_staging_output_put_probe,
        public_image=args.public_image,
        vast_launch_mode=args.vast_launch_mode,
        vast_template_hash_id=args.vast_template_hash_id,
        use_vast_template_image=args.use_vast_template_image,
    )
    print(
        "[vast-wam-authorized-runner] manifest="
        + str(Path(args.job_dir).resolve() / "vast_wam_authorized_runner_manifest.json")
    )
    print(f"[vast-wam-authorized-runner] status={manifest.get('status')}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[vast-wam-authorized-runner] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
