"""Coordinate the next authorized Vast Blueprint bundle probe.

This runner is deliberately conservative. It can assemble the staging contract
and run no-spend readiness checks, but it refuses to call the paid Vast adapter
unless an explicit paid-launch flag is supplied in addition to the adapter's
normal environment and CLI gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .vast_bundle_staging import (
    BUNDLE_ROUTE,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS,
    DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS,
    DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS,
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
    DEFAULT_ISAAC_IMAGE,
    DEFAULT_NGC_IMAGE_LOGIN_MODE,
    DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
    DEFAULT_HARD_CAP_USD,
    DEFAULT_MAX_HOURLY_RATE,
    DEFAULT_TARGET_SPEND_USD,
    NGC_IMAGE_LOGIN_MODES,
    VAST_INSTANCE_LAUNCH_GATE_ENV,
    VAST_API_GATE_ENV,
    run_vast_provider_adapter,
    _vast_session_budget_ledger_path,
)


VAST_AUTHORIZED_PROBE_RUNNER_SCHEMA_VERSION = "vast_authorized_probe_runner.v1"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _redacted_path(route: str) -> str:
    return f"/{route.strip('/')}?token=<redacted-token>"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _attempt_estimated_cost(attempt: Mapping[str, Any]) -> float:
    for key in ("estimated_cost_usd_using_observed_rate", "estimated_cost_usd"):
        value = _number(attempt.get(key))
        if value is not None:
            return max(0.0, value)
    return 0.0


def _session_estimated_cost(path: Path) -> tuple[float, str | None]:
    if not path.is_file():
        return 0.0, None
    try:
        payload = _read_json(path)
    except Exception as exc:
        return 0.0, f"session_budget_ledger_parse_failed:{type(exc).__name__}"
    for key in ("total_observed_estimated_cost_usd", "estimated_cost_usd"):
        value = _number(payload.get(key))
        if value is not None:
            return max(0.0, value), None
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        return sum(_attempt_estimated_cost(item) for item in attempts if isinstance(item, Mapping)), None
    return 0.0, None


def _target_spend_guard(
    *,
    budget_path: Path,
    target_spend_usd: float,
    max_hourly_rate: float,
    max_live_minutes: int,
    allow_target_spend_overrun: bool,
) -> dict[str, Any]:
    prior_cost, parse_error = _session_estimated_cost(budget_path)
    projected_incremental = max(0.0, max_hourly_rate) * max(0, max_live_minutes) / 60.0
    blockers: list[str] = []
    if parse_error:
        blockers.append("session_budget_ledger_parse_failed")
    elif not allow_target_spend_overrun:
        if prior_cost >= target_spend_usd:
            blockers.append("session_estimated_spend_target_exhausted")
        elif prior_cost + projected_incremental > target_spend_usd:
            blockers.append("requested_max_spend_would_exceed_target")
    return {
        "schema_version": "vast_authorized_probe_target_spend_guard.v1",
        "status": "blocked" if blockers else "passed",
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_path.is_file(),
        "budget_parse_error": parse_error,
        "target_spend_usd": target_spend_usd,
        "prior_estimated_cost_usd": round(prior_cost, 6),
        "projected_max_incremental_cost_usd": round(projected_incremental, 6),
        "projected_total_estimated_cost_usd": round(prior_cost + projected_incremental, 6),
        "remaining_to_target_before_request_usd": round(target_spend_usd - prior_cost, 6),
        "allow_target_spend_overrun": allow_target_spend_overrun,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def _staging_verification_guard(
    *,
    verify_staging_urls: bool,
    allow_unverified_public_staging_for_paid_launch: bool,
    staging_manifest: Mapping[str, Any],
    public_staging_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    bundle_url_ready = staging_manifest.get("provider_fetchable_bundle_uri_ready") is True
    output_url_ready = staging_manifest.get("provider_output_callback_ready") is True
    public_status = (
        public_staging_verification.get("status")
        if isinstance(public_staging_verification, Mapping)
        else "not_requested"
    )
    if not bundle_url_ready:
        blockers.append("provider_bundle_fetch_url_not_ready")
    if not output_url_ready:
        blockers.append("provider_output_put_url_not_ready")
    if (
        verify_staging_urls
        and not allow_unverified_public_staging_for_paid_launch
        and public_status != "passed"
    ):
        blockers.append("public_staging_url_verification_failed")
    if not verify_staging_urls and not allow_unverified_public_staging_for_paid_launch:
        blockers.append("public_staging_urls_not_verified_for_paid_launch")
    return {
        "schema_version": "vast_authorized_probe_staging_verification_guard.v1",
        "status": "blocked" if blockers else "passed",
        "verify_staging_urls": verify_staging_urls,
        "allow_unverified_public_staging_for_paid_launch": (
            allow_unverified_public_staging_for_paid_launch
        ),
        "provider_fetchable_bundle_uri_ready": bundle_url_ready,
        "provider_output_callback_ready": output_url_ready,
        "public_staging_verification_status": public_status,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }


def _truth_boundaries() -> dict[str, Any]:
    return {
        "blueprint_provider_bundle_execution_proven": False,
        "video_smoke_proven": False,
        "official_policy_execution_proven": False,
        "controller_grade_execution_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "dexterous_hand_policy_proven": False,
        "wam_vla_runtime_proven": False,
    }


def run_vast_authorized_probe_runner(
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
    max_live_minutes: int = 1,
    session_max_live_minutes: int | None = 45,
    startup_timeout_seconds: int = 420,
    verify_staging_urls: bool = True,
    allow_unverified_public_staging_for_paid_launch: bool = False,
    public_staging_max_wait_seconds: int = DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS,
    public_staging_retry_interval_seconds: float = (
        DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS
    ),
    public_staging_timeout_seconds: float = DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS,
    require_known_supported_isaac_driver: bool = True,
    ngc_image_login_mode: str = DEFAULT_NGC_IMAGE_LOGIN_MODE,
    isaac_image: str | None = None,
    vast_template_hash_id: str | None = None,
    use_vast_template_image: bool = False,
    allow_cold_isaac_image_pull: bool = False,
    min_cold_isaac_pull_live_minutes: int = DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
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
        output_path=resolved_job_dir / "vast_staging_self_test_output.zip",
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
    target_spend_guard = _target_spend_guard(
        budget_path=resolved_session_budget_ledger,
        target_spend_usd=target_spend_usd,
        max_hourly_rate=max_hourly_rate,
        max_live_minutes=max_live_minutes,
        allow_target_spend_overrun=allow_target_spend_overrun,
    )
    if allow_paid_vast_launch:
        blockers.extend(str(item) for item in target_spend_guard.get("blockers") or [])
    public_staging_verification: dict[str, Any] = {
        "status": "not_requested",
        "reason": "paid_launch_not_authorized",
        "raw_secret_values_recorded": False,
    }
    if allow_paid_vast_launch:
        if not verify_staging_urls:
            public_staging_verification = {
                "status": "skipped",
                "reason": "verify_staging_urls_false",
                "raw_secret_values_recorded": False,
            }
        elif not provider_bundle_url or not provider_output_put_url:
            public_staging_verification = {
                "status": "skipped",
                "reason": "provider_staging_urls_missing",
                "raw_secret_values_recorded": False,
            }
        elif blockers:
            public_staging_verification = {
                "status": "skipped",
                "reason": "pre_public_staging_verification_blockers_present",
                "blockers": blockers.copy(),
                "raw_secret_values_recorded": False,
            }
        else:
            public_staging_verification = verify_public_staging_urls(
                job_dir=resolved_job_dir,
                provider_bundle_url=provider_bundle_url,
                provider_output_put_url=provider_output_put_url,
                bundle_path=resolved_bundle,
                output_path=resolved_output,
                max_wait_seconds=public_staging_max_wait_seconds,
                retry_interval_seconds=public_staging_retry_interval_seconds,
                timeout_seconds=public_staging_timeout_seconds,
                allow_output_put_probe=True,
                cleanup_output_probe=True,
                generated_at=generated,
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
                isaac_image=isaac_image or DEFAULT_ISAAC_IMAGE,
                provider_bundle=resolved_bundle,
                provider_bundle_url=provider_bundle_url,
                provider_output_put_url=provider_output_put_url,
                provider_runtime_output_zip=resolved_output,
                enable_isaac_smoke=True,
                enable_blueprint_bundle=True,
                startup_timeout_seconds=startup_timeout_seconds,
                session_budget_ledger_path=resolved_session_budget_ledger,
                session_max_live_minutes=session_max_live_minutes,
                verify_staging_urls=verify_staging_urls,
                require_known_supported_isaac_driver=require_known_supported_isaac_driver,
                ngc_image_login_mode=ngc_image_login_mode,
                vast_template_hash_id=vast_template_hash_id,
                use_vast_template_image=use_vast_template_image,
                allow_cold_isaac_image_pull=allow_cold_isaac_image_pull,
                min_cold_isaac_pull_live_minutes=min_cold_isaac_pull_live_minutes,
            )
            if adapter_result.get("status") != "completed":
                blockers.extend(str(item) for item in adapter_result.get("blockers") or [])
    else:
        blockers.append("paid_vast_launch_not_authorized_by_runner_flag")

    status = "completed" if paid_launch_attempted and adapter_result and adapter_result.get("status") == "completed" else "blocked"
    manifest = {
        "schema_version": VAST_AUTHORIZED_PROBE_RUNNER_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "job_dir": str(resolved_job_dir),
        "bundle_path": str(resolved_bundle),
        "bundle_present": resolved_bundle.is_file(),
        "bundle_size_bytes": resolved_bundle.stat().st_size if resolved_bundle.is_file() else 0,
        "output_path": str(resolved_output),
        "token_file": token_status,
        "secret_env_file": str(resolved_secret_env_file),
        "public_base_url_present": bool(_string(public_base_url)),
        "bundle_url_path": _redacted_path(BUNDLE_ROUTE),
        "output_put_url_path": _redacted_path(OUTPUT_ROUTE),
        "local_staging_self_test_status": self_test.get("status"),
        "staging_manifest_status": staging_manifest.get("status"),
        "public_staging_verification_status": public_staging_verification.get("status"),
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
        "public_staging_verification": public_staging_verification,
        "target_spend_guard": target_spend_guard,
        "allow_target_spend_overrun": allow_target_spend_overrun,
        "allow_unverified_public_staging_for_paid_launch": (
            allow_unverified_public_staging_for_paid_launch
        ),
        "require_known_supported_isaac_driver": require_known_supported_isaac_driver,
        "ngc_image_login_mode": ngc_image_login_mode,
        "isaac_image": isaac_image or DEFAULT_ISAAC_IMAGE,
        "vast_template_hash_present": bool(_string(vast_template_hash_id)),
        "use_vast_template_image": use_vast_template_image,
        "allow_cold_isaac_image_pull": allow_cold_isaac_image_pull,
        "min_cold_isaac_pull_live_minutes": min_cold_isaac_pull_live_minutes,
        "adapter_env_gates_required": [VAST_API_GATE_ENV, VAST_INSTANCE_LAUNCH_GATE_ENV],
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
        **_truth_boundaries(),
    }
    write_json(resolved_job_dir / "vast_authorized_probe_runner_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed staging and optional paid Vast bundle probe."
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
    parser.add_argument("--max-live-minutes", type=int, default=1)
    parser.add_argument("--session-max-live-minutes", type=int, default=45)
    parser.add_argument("--startup-timeout-seconds", type=int, default=420)
    parser.add_argument("--no-verify-staging-urls", action="store_true")
    parser.add_argument(
        "--public-staging-max-wait-seconds",
        type=int,
        default=DEFAULT_PUBLIC_VERIFY_MAX_WAIT_SECONDS,
    )
    parser.add_argument(
        "--public-staging-retry-interval-seconds",
        type=float,
        default=DEFAULT_PUBLIC_VERIFY_RETRY_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--public-staging-timeout-seconds",
        type=float,
        default=DEFAULT_PUBLIC_VERIFY_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--allow-unverified-public-staging-for-paid-launch",
        action="store_true",
        help="Allow a paid launch even when public staging URLs are not probed first. Use only with an independently verified tunnel.",
    )
    parser.add_argument(
        "--ngc-image-login-mode",
        choices=NGC_IMAGE_LOGIN_MODES,
        default=DEFAULT_NGC_IMAGE_LOGIN_MODE,
        help="Force, disable, or auto-select NGC registry login for the Isaac image.",
    )
    parser.add_argument(
        "--isaac-image",
        default=DEFAULT_ISAAC_IMAGE,
        help="Isaac-capable image to use for the provider bundle run.",
    )
    parser.add_argument("--vast-template-hash-id")
    parser.add_argument("--use-vast-template-image", action="store_true")
    parser.add_argument(
        "--allow-cold-isaac-image-pull",
        action="store_true",
        help="Explicitly allow a paid direct cold pull of the official Isaac image.",
    )
    parser.add_argument(
        "--min-cold-isaac-pull-live-minutes",
        type=int,
        default=DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
    )
    parser.add_argument(
        "--allow-known-unsupported-isaac-driver",
        action="store_true",
        help="Allow Vast offers in the known unsupported Omniverse RTX driver range. Default is to require a known-supported driver for bundle/video proof.",
    )
    args = parser.parse_args(argv)
    manifest = run_vast_authorized_probe_runner(
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
        public_staging_max_wait_seconds=args.public_staging_max_wait_seconds,
        public_staging_retry_interval_seconds=(
            args.public_staging_retry_interval_seconds
        ),
        public_staging_timeout_seconds=args.public_staging_timeout_seconds,
        require_known_supported_isaac_driver=not args.allow_known_unsupported_isaac_driver,
        ngc_image_login_mode=args.ngc_image_login_mode,
        isaac_image=args.isaac_image,
        vast_template_hash_id=args.vast_template_hash_id,
        use_vast_template_image=args.use_vast_template_image,
        allow_cold_isaac_image_pull=args.allow_cold_isaac_image_pull,
        min_cold_isaac_pull_live_minutes=args.min_cold_isaac_pull_live_minutes,
    )
    print(
        "[vast-authorized-probe-runner] manifest="
        + str(Path(args.job_dir).resolve() / "vast_authorized_probe_runner_manifest.json")
    )
    print(f"[vast-authorized-probe-runner] status={manifest.get('status')}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[vast-authorized-probe-runner] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
