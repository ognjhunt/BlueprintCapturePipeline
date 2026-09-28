"""Argument parsing for the hard-disabled direct Vast adapter CLI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .gpu_selection_policy import GPU_SELECTION_POLICIES


def main(argv: Sequence[str] | None = None, *, adapter_module: Any) -> int:
    adapter = adapter_module
    parser = argparse.ArgumentParser(
        description="Run a gated Vast.ai startup probe for Blueprint robot-eval GPU lanes."
    )
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["dry-run", "template-discovery", "live-startup-probe"],
        default="dry-run",
    )
    parser.add_argument(
        "--gpu-selection-policy",
        choices=sorted(GPU_SELECTION_POLICIES),
        default=None,
        help="workload GPU policy; see gpu_selection_policy.GPU_SELECTION_POLICIES",
    )
    parser.add_argument(
        "--max-hourly-rate", type=float, default=adapter.DEFAULT_MAX_HOURLY_RATE
    )
    parser.add_argument(
        "--target-spend-usd", type=float, default=adapter.DEFAULT_TARGET_SPEND_USD
    )
    parser.add_argument(
        "--hard-cap-usd", type=float, default=adapter.DEFAULT_HARD_CAP_USD
    )
    parser.add_argument(
        "--max-live-minutes", type=int, default=adapter.DEFAULT_MAX_LIVE_MINUTES
    )
    parser.add_argument("--public-image", default=adapter.DEFAULT_PUBLIC_CUDA_IMAGE)
    parser.add_argument("--isaac-image", default=adapter.DEFAULT_ISAAC_IMAGE)
    parser.add_argument("--heartbeat-url", default=adapter.DEFAULT_HEARTBEAT_URL)
    parser.add_argument("--previous-job-dir")
    parser.add_argument("--provider-bundle")
    parser.add_argument(
        "--provider-bundle-url",
        help="Provider-fetchable URL for isaac_provider_runtime_bundle.zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-output-put-url",
        help="Provider-writable PUT URL for vast_provider_runtime_output.zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-output-get-url",
        help="Provider-readable GET URL for downloading the uploaded runtime output zip; token values are redacted from artifacts.",
    )
    parser.add_argument(
        "--provider-runtime-output-zip",
        help="Local path expected to contain the uploaded provider runtime output zip.",
    )
    parser.add_argument("--enable-isaac-smoke", action="store_true")
    parser.add_argument("--enable-blueprint-bundle", action="store_true")
    parser.add_argument(
        "--provider-bundle-kind",
        choices=adapter.VAST_PROVIDER_BUNDLE_KINDS,
        default="isaac",
        help="Provider bundle runtime contract to execute. Defaults to the existing Isaac path.",
    )
    parser.add_argument(
        "--vast-launch-mode",
        choices=adapter.VAST_LAUNCH_MODES,
        default=adapter.DEFAULT_VAST_LAUNCH_MODE,
        help="Use auto to select args for Isaac smoke and ssh_direct otherwise.",
    )
    parser.add_argument(
        "--ngc-image-login-mode",
        choices=adapter.NGC_IMAGE_LOGIN_MODES,
        default=os.getenv(
            adapter.VAST_IMAGE_LOGIN_MODE_ENV, adapter.DEFAULT_NGC_IMAGE_LOGIN_MODE
        ),
        help="Use auto to avoid login for the official public Isaac image; always forces NGC credentials.",
    )
    parser.add_argument(
        "--vast-template-hash-id",
        help=(
            "Optional Vast template hash for launch configuration reuse. "
            "A template hash alone is not image-cache or prewarm proof."
        ),
    )
    parser.add_argument(
        "--use-vast-template-image",
        action="store_true",
        help="Omit the direct image override and use the image configured on --vast-template-hash-id.",
    )
    parser.add_argument(
        "--allow-cold-isaac-image-pull",
        action="store_true",
        dest="allow_cold_isaac_image_pull",
        help="Allow direct cold pulls of the official Isaac image. The authorized wrapper disables this by default.",
    )
    parser.add_argument(
        "--block-cold-isaac-image-pull",
        action="store_false",
        dest="allow_cold_isaac_image_pull",
        help="Block paid live probes that would directly cold-pull the official Isaac image.",
    )
    parser.set_defaults(allow_cold_isaac_image_pull=True)
    parser.add_argument(
        "--min-cold-isaac-pull-live-minutes",
        type=int,
        default=adapter.DEFAULT_MIN_COLD_ISAAC_PULL_LIVE_MINUTES,
        help="Minimum live window required when allowing a direct cold pull of the official Isaac image.",
    )
    parser.add_argument(
        "--disk-gb",
        type=int,
        help=(
            f"Override Vast disk GB. Defaults to {adapter.DEFAULT_ISAAC_DISK_GB} "
            f"for Isaac smoke and {adapter.DEFAULT_PUBLIC_DISK_GB} otherwise."
        ),
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument("--startup-timeout-seconds", type=int, default=420)
    parser.add_argument(
        "--heartbeat-no-progress-seconds",
        type=int,
        default=None,
        help=(
            "Maximum seconds to wait with no onstart/request_logs progress before "
            f"blocking startup. Defaults to {adapter.VAST_HEARTBEAT_NO_PROGRESS_SECONDS_ENV} "
            f"or {adapter.DEFAULT_HEARTBEAT_NO_PROGRESS_SECONDS}."
        ),
    )
    parser.add_argument(
        "--machine-avoidlist",
        help="Optional JSON avoidlist of Vast machine IDs to exclude from offer selection. Defaults to <job-dir>/vast_machine_avoidlist.json.",
    )
    parser.add_argument(
        "--allowed-machine-id",
        action="append",
        default=[],
        help=(
            "Restrict offer selection to this Vast machine ID. Can be repeated; "
            "use after a host-specific canary has passed."
        ),
    )
    parser.add_argument(
        "--session-budget-ledger",
        help=(
            "Optional session cost summary JSON used to block paid launches before Vast API calls. "
            f"Defaults to {adapter.DEFAULT_VAST_SESSION_BUDGET_FILENAME} beside "
            f"{adapter.VAST_API_KEY_FILE_ENV}; "
            f"{adapter.VAST_SESSION_BUDGET_LEDGER_FILE_ENV} can override the default."
        ),
    )
    parser.add_argument(
        "--vast-launch-lock-file",
        help=(
            "Optional single-flight lock file for paid Vast launches. Defaults to "
            "vast_paid_launch.lock beside VAST_API_KEY_FILE."
        ),
    )
    parser.add_argument(
        "--session-max-live-minutes",
        type=int,
        default=adapter.DEFAULT_SESSION_MAX_LIVE_MINUTES,
        help=(
            "Maximum cumulative live Vast runtime allowed for this session. "
            f"Defaults to {adapter.DEFAULT_SESSION_MAX_LIVE_MINUTES}."
        ),
    )
    parser.add_argument(
        "--verify-staging-urls",
        action="store_true",
        help="Verify provider bundle URL reachability before Vast offer search.",
    )
    parser.add_argument(
        "--allow-staging-output-put-probe",
        action="store_true",
        help="Allow a small pre-allocation PUT probe to the provider output URL. This is intentionally opt-in because some signed URLs are one-shot or overwrite targets.",
    )
    parser.add_argument(
        "--require-known-supported-isaac-driver",
        action="store_true",
        help="Exclude offers in the known unsupported Omniverse RTX driver range; recommended for Blueprint bundle/video proof.",
    )
    parser.add_argument(
        "--allow-vast-api-call",
        action="store_true",
        help=f"Required with {adapter.VAST_API_GATE_ENV}=true for live Vast API calls.",
    )
    parser.add_argument(
        "--allow-vast-instance-launch",
        action="store_true",
        help=(
            f"Required with {adapter.VAST_INSTANCE_LAUNCH_GATE_ENV}=true for paid "
            "Vast instance launch."
        ),
    )
    args = parser.parse_args(argv)
    if args.mode == "live-startup-probe":
        print("legacy_vast_provider_mutation_cli_disabled", file=sys.stderr)
        return 2
    adapter_kwargs = vars(args).copy()
    for cli_name, adapter_name in (
        ("allow_vast_instance_launch", "allow_instance_launch"),
        ("machine_avoidlist", "machine_avoidlist_path"),
        ("allowed_machine_id", "allowed_machine_ids"),
        ("session_budget_ledger", "session_budget_ledger_path"),
    ):
        adapter_kwargs[adapter_name] = adapter_kwargs.pop(cli_name)
    result = adapter.run_vast_provider_adapter(**adapter_kwargs)
    print(
        "[vast-provider-adapter] result="
        f"{Path(args.job_dir).resolve() / 'vast_provider_adapter_result.json'}"
    )
    print(f"[vast-provider-adapter] status={result.get('status')}")
    print(
        "[vast-provider-adapter] instance_ids="
        + ",".join(str(item) for item in result.get("vast_instance_ids", []))
    )
    blockers = adapter._string_list(result.get("blockers"))
    if blockers:
        print("[vast-provider-adapter] blockers=" + ",".join(blockers))
    return 0 if result.get("status") in {"completed", "dry_run_ready"} else 1


__all__ = ["main"]
