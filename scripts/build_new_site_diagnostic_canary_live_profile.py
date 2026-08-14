#!/usr/bin/env python3
"""Build a live launch profile for the new-site diagnostic canary.

The allocator has been able to execute this probe kind for a long time and no
live profile builder emitted it, so it could not be triggered from the product
path at all: `tests/test_website_reachable_probe_kinds.py` carried it as
`awaiting_builder`. This is what removes that row.

Two things about this lane decide almost everything below.

**Its probe kind is a label, not a route.** The allocator branch for
`new-site-diagnostic-canary` is shared with the frozen `openpi-policy-ranking`
campaign, and `--probe-kind` is never forwarded to the transport. Which lane
actually runs is read off the *input bundle receipt*:
`build_openpi_policy_ranking_gpu_admission` derives `execution_mode` from its
schema, and the worker is handed that mode in an environment variable. A
profile labelled as the canary that pinned a full-campaign receipt would launch
the frozen campaign under the canary's name, and every downstream artifact
would agree with the label. So the receipt's identity is checked here, against
the same freeze the admission applies.

**Its spend and TTL are separate arguments from the profile's.** The campaign
takes `--openpi-hard-ttl-seconds` (default 14,400) and `--openpi-max-spend-usd`
(default $3.00) rather than the allocator-wide ones, so a profile that omits
them declares one ceiling to the standing authorization and runs under another.
They are derived from the same numbers the profile publishes.

The rest is the shared skeleton in `task_evaluation_live_profile`: residency,
spend binding, the control surface, and the terminal contract are not per-lane
decisions.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.openpi_policy_ranking_gpu_admission import (
    CANARY_INPUT_RECEIPT_SCHEMA_VERSION,
    CANARY_INPUT_SCHEMA_VERSION,
    MAX_TTL_SECONDS,
    NEW_SITE_CANARY_PROBE_KIND,
    VAST_DEFAULT_MAX_HOURLY_RATE_USD,
)
from blueprint_pipeline.openpi_policy_ranking_runpod import CANARY_NAME_PREFIX
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256
from blueprint_pipeline.production_gpu_campaign_budget import (
    AUTHORIZED_GPU_WALL_CAP_SECONDS,
    AUTHORIZED_SPEND_CAP_USD,
    SCHEMA_VERSION as BUDGET_LEDGER_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

#: The worker fetches its signed input under a 180 s timeout and uploads its
#: output under a 300 s one, and the provider request derives
#: `hard_timeout_seconds = ttl - 120`. A rental whose hard timeout cannot
#: contain the transport it must perform is spent before it starts.
MIN_TTL_SECONDS = 180 + 300 + 120

#: The admission refuses anything above this, so it is imported rather than
#: restated. `MAX_TTL_SECONDS` x the frozen Vast ceiling is exactly the $3.00
#: the allocator's own `--openpi-max-spend-usd` default declares.
MAX_LANE_TTL_SECONDS = MAX_TTL_SECONDS

CANARY_PURPOSE = "private_internal_noncommercial_new_site_diagnostic_canary"
SUPPORTED_PROVIDERS = ("vast", "runpod")

#: The url files hold presigned, secret-bearing URLs. Their *paths* travel in
#: argv; their bytes are never read here and never digest-bound, because a
#: rotated URL would otherwise turn into an immutable-input mismatch.
SECRET_URL_FILES = (
    "input_secret_url_file",
    "output_secret_put_url_file",
    "output_secret_get_url_file",
)


@dataclass(frozen=True)
class Param:
    """One builder keyword and the flag that supplies it.

    The parser and the call are both generated from this table, so a keyword
    cannot quietly acquire a second, different default on the command line --
    which for this lane would mean a paid ceiling fixed at a value nobody chose.
    """

    flag: str
    help: str = ""
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None
    choices: Sequence[str] | None = None


PARAMS: dict[str, Param] = {
    "bundle_receipt_path": Param(
        "--bundle-receipt",
        "Canary input bundle receipt; its schema is what picks the lane.",
        required=True,
    ),
    "release_evidence_path": Param(
        "--release-evidence", "Digest-pinned OpenPI GPU release evidence.", required=True
    ),
    "provider_preflight_path": Param(
        "--provider-preflight", "Mutation-free provider capacity snapshot.", required=True
    ),
    "input_secret_url_file": Param(
        "--input-secret-url-file", "0600 file holding the signed input URL.", required=True
    ),
    "output_secret_put_url_file": Param(
        "--output-secret-put-url-file", "0600 file holding the signed PUT URL.", required=True
    ),
    "output_secret_get_url_file": Param(
        "--output-secret-get-url-file",
        "0600 file holding the signed GET URL; required only under --execute.",
        required=True,
    ),
    "campaign_budget_ledger_path": Param(
        "--campaign-budget-ledger", "Dual-cap reservation ledger.", required=True
    ),
    "campaign_initial_spent_usd": Param(
        "--campaign-initial-spent-usd",
        "Campaign spend already accounted for; part of the ledger's identity.",
        required=True,
        type=float,
    ),
    "campaign_initial_used_gpu_seconds": Param(
        "--campaign-initial-used-gpu-seconds",
        "Campaign GPU wall time already accounted for; part of the identity.",
        required=True,
        type=int,
    ),
    "source_commit": Param("--source-commit", required=True),
    "raw_manifest_uri": Param("--raw-manifest-uri", required=True),
    "revision": Param(
        "--revision", "Distinguish a rebuilt profile whose inputs changed at the same commit."
    ),
    "provider": Param(
        "--provider",
        "Frozen default is Vast.",
        default="vast",
        choices=SUPPORTED_PROVIDERS,
    ),
    "campaign_total_spend_cap_usd": Param(
        "--campaign-total-spend-cap-usd", type=float, default=AUTHORIZED_SPEND_CAP_USD
    ),
    "campaign_wall_cap_seconds": Param(
        "--campaign-wall-cap-seconds", type=int, default=AUTHORIZED_GPU_WALL_CAP_SECONDS
    ),
    "max_hourly_rate_usd": Param(
        "--max-hourly-rate-usd", type=float, default=VAST_DEFAULT_MAX_HOURLY_RATE_USD
    ),
    "hard_ttl_seconds": Param(
        "--hard-ttl-seconds", type=int, default=MAX_LANE_TTL_SECONDS
    ),
}


@dataclass(frozen=True)
class CampaignSettings:
    """The lane's non-path decisions, which argv has to carry explicitly."""

    provider: str = "vast"
    initial_spent_usd: float = 0.0
    initial_used_gpu_seconds: int = 0
    total_spend_cap_usd: float = AUTHORIZED_SPEND_CAP_USD
    wall_cap_seconds: int = AUTHORIZED_GPU_WALL_CAP_SECONDS


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def _receipt_blockers(context: LaneLiveProfileContext) -> list[str]:
    """Refuse any receipt that is not this lane's, by the admission's own rules.

    Every check here has a named counterpart in
    `build_openpi_policy_ranking_gpu_admission`, which applies it after the
    launch has started.
    """

    found: list[str] = []
    receipt = context.receipt
    if receipt.get("schema_version") != CANARY_INPUT_RECEIPT_SCHEMA_VERSION:
        found.append(
            "new_site_canary_input_receipt_schema_invalid:"
            f"{receipt.get('schema_version')}"
        )
    manifest_value = receipt.get("manifest")
    manifest = dict(manifest_value) if isinstance(manifest_value, Mapping) else {}
    if manifest.get("schema_version") != CANARY_INPUT_SCHEMA_VERSION:
        found.append("new_site_canary_input_manifest_schema_invalid")
    if (
        manifest.get("arm_id") != "skeleton_only"
        or manifest.get("variant") != "center"
        or manifest.get("label_free") is not True
        or manifest.get("raw_3dgs_included") is not False
        or manifest.get("redistribution_authorized") is not False
        or manifest.get("purpose") != CANARY_PURPOSE
        or not str(manifest.get("policy_id") or "")
        or not str(manifest.get("scene_id") or "")
        or not str(manifest.get("task_instruction") or "")
    ):
        found.append("new_site_canary_input_freeze_invalid")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(payload):
        found.append("new_site_canary_input_manifest_sha256_invalid")
    return found


def _campaign_blockers(
    context: LaneLiveProfileContext, settings: CampaignSettings
) -> list[str]:
    """Refuse a campaign identity `ProductionGpuCampaignBudget` would reject.

    Its constructor raises rather than returning a blocked result, and the
    campaign does not catch it, so these arrive as an allocator traceback part
    way through a launch instead of as a refusal.
    """

    found: list[str] = []
    if (
        settings.total_spend_cap_usd > AUTHORIZED_SPEND_CAP_USD
        or settings.wall_cap_seconds > AUTHORIZED_GPU_WALL_CAP_SECONDS
    ):
        found.append("new_site_canary_campaign_cap_exceeds_authorization")
    if not 0 <= settings.initial_spent_usd <= settings.total_spend_cap_usd:
        found.append("new_site_canary_campaign_initial_spend_exceeds_cap")
    if not 0 <= settings.initial_used_gpu_seconds <= settings.wall_cap_seconds:
        found.append("new_site_canary_campaign_initial_wall_time_exceeds_cap")

    ledger = context.extra_paths.get("campaign_budget_ledger")
    # A ledger that does not exist yet is created by the run; only one already
    # on disk carries an identity this profile has to agree with.
    if ledger is None or not ledger.is_file():
        return found
    state = _read(ledger)
    if state.get("schema_version") != BUDGET_LEDGER_SCHEMA_VERSION:
        found.append("new_site_canary_campaign_budget_ledger_schema_invalid")
    expected = {
        "total_spend_cap_usd": float(settings.total_spend_cap_usd),
        "combined_gpu_wall_cap_seconds": int(settings.wall_cap_seconds),
        "initial_spent_usd": float(settings.initial_spent_usd),
        "initial_used_gpu_seconds": int(settings.initial_used_gpu_seconds),
    }
    found.extend(
        f"new_site_canary_campaign_budget_ledger_identity_mismatch:{field}"
        for field, value in expected.items()
        if state.get(field) != value
    )
    return found


def _lane_blockers(settings: CampaignSettings):
    def blockers(context: LaneLiveProfileContext) -> list[str]:
        found = _receipt_blockers(context)
        found.extend(_campaign_blockers(context, settings))

        if settings.provider not in SUPPORTED_PROVIDERS:
            found.append(f"new_site_canary_provider_unsupported:{settings.provider}")
        if settings.provider == "vast" and (
            context.max_hourly_rate_usd < VAST_DEFAULT_MAX_HOURLY_RATE_USD
        ):
            # The execute path re-collects the Vast preflight with no rate
            # argument, so the admission compares this profile's declared spend
            # against the lane's frozen ceiling however low a rate it names.
            found.append(
                "new_site_canary_hourly_rate_below_vast_lane_ceiling:"
                f"{context.max_hourly_rate_usd}"
            )

        for name in SECRET_URL_FILES:
            path = context.extra_paths.get(name)
            if path is None:
                continue
            if path.is_symlink() or not path.is_file():
                found.append(f"new_site_canary_secret_url_file_missing:{name}")
            elif stat.S_IMODE(path.stat().st_mode) != 0o600:
                # `_read_private_https_url` demands exactly this and raises
                # mid-launch otherwise.
                found.append(f"new_site_canary_secret_url_file_not_private:{name}")

        for name in ("release_evidence", "provider_preflight"):
            path = context.extra_paths.get(name)
            if path is None or path.is_symlink() or not path.is_file():
                found.append(f"new_site_canary_{name}_missing")
        if not Path(context.bundle_path or "").is_file():
            found.append("new_site_canary_input_bundle_missing")

        release_path = context.extra_paths.get("release_evidence")
        if release_path is not None and release_path.is_file():
            release = _read(release_path)
            if release.get("status") != "passed":
                found.append(f"new_site_canary_release_not_passed:{release.get('status')}")
            if str(release.get("source_commit") or "").strip().lower() != (
                context.source_commit.strip().lower()
            ):
                found.append("new_site_canary_release_source_commit_mismatch")
        return found

    return blockers


def _lane_argv(settings: CampaignSettings):
    def argv(context: LaneLiveProfileContext) -> list[str]:
        return [
            "--openpi-input-bundle-receipt", str(context.receipt_path),
            "--release-evidence", str(context.extra_paths["release_evidence"]),
            "--preflight-bundle", str(context.extra_paths["provider_preflight"]),
            "--openpi-input-secret-url-file",
            str(context.extra_paths["input_secret_url_file"]),
            "--openpi-output-secret-put-url-file",
            str(context.extra_paths["output_secret_put_url_file"]),
            # Demanded only under --execute, which is exactly why it is easy to
            # leave out and impossible to discover without a live launch.
            "--openpi-output-secret-get-url-file",
            str(context.extra_paths["output_secret_get_url_file"]),
            "--openpi-provider", settings.provider,
            # Not the allocator-wide TTL and spend: these are what the campaign
            # reserves, bills, and arms its watchdog against.
            "--openpi-hard-ttl-seconds", str(context.hard_ttl_seconds),
            "--openpi-max-spend-usd", str(context.max_spend_usd),
            "--campaign-budget-ledger", str(context.extra_paths["campaign_budget_ledger"]),
            "--campaign-initial-spent-usd", str(settings.initial_spent_usd),
            "--campaign-initial-used-gpu-seconds", str(settings.initial_used_gpu_seconds),
            "--campaign-total-spend-cap-usd", str(settings.total_spend_cap_usd),
            "--campaign-wall-cap-seconds", str(settings.wall_cap_seconds),
        ]

    return argv


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    release = context.extra_paths["release_evidence"]
    preflight = context.extra_paths["provider_preflight"]
    return [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            # What will actually execute: the digest-pinned runtime image and
            # the frozen OpenPI and Menagerie revisions it was built from.
            "name": "evaluation_run_spec",
            "path": str(release),
            "digest": file_digest(release),
        },
        {
            "name": "new_site_canary_input_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
        {
            "name": "openpi_provider_preflight",
            "path": str(preflight),
            "digest": file_digest(preflight),
        },
    ]


def _spec(settings: CampaignSettings = CampaignSettings()) -> LaneLiveProfileSpec:
    return LaneLiveProfileSpec(
        # The campaign refuses any pod name outside the teardown watchdog's
        # scope, and the watchdog only reaps instances under this prefix.
        profile_id_prefix=f"{CANARY_NAME_PREFIX}new-site",
        probe_kind=NEW_SITE_CANARY_PROBE_KIND,
        min_ttl_seconds=MIN_TTL_SECONDS,
        max_ttl_seconds=MAX_LANE_TTL_SECONDS,
        source_bundle_id=lambda context: (
            f"new-site-diagnostic-canary-{context.source_commit[:12]}"
        ),
        # The dispatcher admits three source kinds and this is not free text.
        # The canary's frozen scene is InteriorGS 0787, the same public
        # substrate the Arena family declares.
        source_kind="interiorgs_sage",
        lane_argv=_lane_argv(settings),
        immutable_inputs=_immutable_inputs,
        lane_blockers=_lane_blockers(settings),
        required_providers=(settings.provider,),
        provider=settings.provider,
        extra_path_names=(
            "release_evidence",
            "provider_preflight",
            "campaign_budget_ledger",
            *SECRET_URL_FILES,
        ),
    )


#: The lane's declaration at its defaults, for the contract and reachability
#: gates that rediscover builders from `scripts/`.
SPEC = _spec()


def build_new_site_diagnostic_canary_live_profile(
    *,
    bundle_receipt_path: str | Path,
    release_evidence_path: str | Path,
    provider_preflight_path: str | Path,
    input_secret_url_file: str | Path,
    output_secret_put_url_file: str | Path,
    output_secret_get_url_file: str | Path,
    campaign_budget_ledger_path: str | Path,
    campaign_initial_spent_usd: float,
    campaign_initial_used_gpu_seconds: int,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    provider: str = "vast",
    campaign_total_spend_cap_usd: float = AUTHORIZED_SPEND_CAP_USD,
    campaign_wall_cap_seconds: int = AUTHORIZED_GPU_WALL_CAP_SECONDS,
    max_hourly_rate_usd: float = VAST_DEFAULT_MAX_HOURLY_RATE_USD,
    hard_ttl_seconds: int = MAX_LANE_TTL_SECONDS,
) -> dict[str, Any]:
    """Derive the canary's live profile from the receipts it will run."""

    settings = CampaignSettings(
        provider=str(provider),
        initial_spent_usd=float(campaign_initial_spent_usd),
        initial_used_gpu_seconds=int(campaign_initial_used_gpu_seconds),
        total_spend_cap_usd=float(campaign_total_spend_cap_usd),
        wall_cap_seconds=int(campaign_wall_cap_seconds),
    )
    return build_lane_live_profile(
        _spec(settings),
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        # Left to the skeleton: rate x TTL is the worst case this profile can
        # reach, and it is also what `--openpi-max-spend-usd` carries, so the
        # two cannot be given different numbers.
        max_spend_usd=None,
        extra_paths={
            "release_evidence": release_evidence_path,
            "provider_preflight": provider_preflight_path,
            "campaign_budget_ledger": campaign_budget_ledger_path,
            "input_secret_url_file": input_secret_url_file,
            "output_secret_put_url_file": output_secret_put_url_file,
            "output_secret_get_url_file": output_secret_get_url_file,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for keyword, param in PARAMS.items():
        options: dict[str, Any] = {"dest": keyword, "help": param.help or None}
        if param.type is not None:
            options["type"] = param.type
        if param.choices is not None:
            options["choices"] = list(param.choices)
        if param.required:
            options["required"] = True
        else:
            options["default"] = param.default
        parser.add_argument(param.flag, **options)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        profile = build_new_site_diagnostic_canary_live_profile(
            **{keyword: getattr(args, keyword) for keyword in PARAMS}
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_launch_profile.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "built",
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "probe_kind": NEW_SITE_CANARY_PROBE_KIND,
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "hard_ttl_seconds": profile["allocator"]["hard_ttl_seconds"],
                "output": str(output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
