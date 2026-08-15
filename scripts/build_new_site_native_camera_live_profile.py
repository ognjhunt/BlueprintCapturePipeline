#!/usr/bin/env python3
"""Build a live launch profile for the fresh-site native camera probe.

`new-site-native-camera` is the first fresh-site lane to get one. It already had
a hash-bound input bundle, an exact-source release contract, a Vast transport,
and an allocator branch; what it did not have was a launch profile, which is the
only thing that carries a lane across the website boundary.
`tests/test_website_reachable_probe_kinds.py` recorded it as `awaiting_builder`,
and this is what removes it from that row.

The skeleton -- residency, spend binding, terminal contract, validation -- is
shared with every other paid lane in `task_evaluation_live_profile`. What is
here is only what makes this lane different, and three of those differences are
each worth a paid round trip:

* **The pod name is load-bearing.** `run_native_camera_gpu_lane` refuses any
  name outside its watchdog's scope, so the profile id is derived from
  `CANARY_NAME_PREFIX` rather than chosen. A profile whose id merely looks
  right is refused after the allocator has already admitted it.
* **Six of its arguments only matter under `--execute`.** The three presigned
  URL files and the three campaign-budget values are absent from a dry run and
  decisive in a live one, so omitting one does not fail -- it silently fixes a
  paid decision at a default nobody chose. The flag table below is the call
  signature precisely so a parameter cannot be added without a way to set it.
* **The preflight is pinned but refreshed.** The lane re-collects provider
  capacity at launch and reads only `container_disk_bytes` off the pinned
  document, so the pinned copy binds a paid sizing decision while the freshness
  the admission demands comes from the launch itself.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_admission import (
    CANARY_NAME_PREFIX,
    MIN_CONTAINER_DISK_BYTES,
    MIN_GPU_MEMORY_BYTES,
    PROBE_KIND,
    RELEASE_SCHEMA_VERSION,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle import (
    BUNDLE_SCHEMA_VERSION,
    RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256
from blueprint_pipeline.production_gpu_campaign_budget import (
    AUTHORIZED_GPU_WALL_CAP_SECONDS,
    AUTHORIZED_SPEND_CAP_USD,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

#: `build_native_camera_gpu_admission` refuses anything outside this band, after
#: the allocator has admitted the run.
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 3_600
#: The preflight schemas the lane's admission accepts.
PREFLIGHT_SCHEMAS = frozenset(
    {
        "openpi_policy_ranking_runpod_preflight.v1",
        "openpi_policy_ranking_provider_preflight.v2",
    }
)
#: The lane refuses any pod name outside its watchdog's scope, so the profile id
#: is derived from that prefix rather than picked to look like it.
PROFILE_ID_PREFIX = CANARY_NAME_PREFIX.rstrip("-")
#: The presigned URL files, which `_read_private_https_url` opens O_NOFOLLOW and
#: insists are exactly 0600 -- after a GPU has been rented.
SECRET_URL_INPUTS = (
    "provider_bundle_url_file",
    "provider_output_put_url_file",
    "provider_output_get_url_file",
)
EXTRA_PATH_NAMES = (
    "release_evidence",
    "preflight_bundle",
    "campaign_budget_ledger",
    *SECRET_URL_INPUTS,
)
_DIGEST_REF = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class CampaignLimits:
    """The campaign-budget numbers this launch reserves against."""

    initial_spent_usd: float
    initial_used_gpu_seconds: int
    total_spend_cap_usd: float
    wall_cap_seconds: int


@dataclass(frozen=True)
class Flag:
    """One command-line flag, named for the parameter it fills.

    The parser and the call are built from this single table. A flag table kept
    beside a call signature drifts from it, and the failure is silent: the
    parameter keeps its default and the run is decided by a value nobody chose.
    """

    parameter: str
    help: str
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None


FLAGS: dict[str, Flag] = {
    "--bundle-receipt": Flag(
        "bundle_receipt_path", "Hash-bound native camera GPU input bundle receipt.", True
    ),
    "--release-evidence": Flag(
        "release_evidence_path", "Exact-source camera release evidence.", True
    ),
    "--preflight-bundle": Flag(
        "preflight_bundle_path",
        "Provider preflight; only container_disk_bytes survives the launch refresh.",
        True,
    ),
    "--provider-bundle-url-file": Flag(
        "provider_bundle_url_file", "0600 file holding the bundle's presigned GET URL.", True
    ),
    "--provider-output-put-url-file": Flag(
        "provider_output_put_url_file", "0600 file holding the output PUT URL.", True
    ),
    "--provider-output-get-url-file": Flag(
        "provider_output_get_url_file", "0600 file holding the output GET URL.", True
    ),
    "--campaign-budget-ledger": Flag(
        "campaign_budget_ledger", "Durable dual-cap campaign reservation ledger.", True
    ),
    "--campaign-initial-spent-usd": Flag(
        "campaign_initial_spent_usd",
        "Campaign spend before this attempt.",
        True,
        type=float,
    ),
    "--campaign-initial-used-gpu-seconds": Flag(
        "campaign_initial_used_gpu_seconds",
        "Campaign GPU wall time before this attempt.",
        True,
        type=int,
    ),
    "--source-commit": Flag("source_commit", "The commit the control plane runs.", True),
    "--raw-manifest-uri": Flag(
        "raw_manifest_uri",
        "Local digest-bound GCS publication receipt for this run spec.",
        True,
    ),
    "--campaign-total-spend-cap-usd": Flag(
        "campaign_total_spend_cap_usd",
        "Campaign spend ceiling.",
        type=float,
        default=AUTHORIZED_SPEND_CAP_USD,
    ),
    "--campaign-wall-cap-seconds": Flag(
        "campaign_wall_cap_seconds",
        "Campaign combined GPU wall ceiling.",
        type=int,
        default=AUTHORIZED_GPU_WALL_CAP_SECONDS,
    ),
    "--max-hourly-rate-usd": Flag(
        "max_hourly_rate_usd", "Worst-case hourly rate this profile admits.", type=float, default=1.0
    ),
    "--max-spend-usd": Flag(
        "max_spend_usd", "Declared spend ceiling for this attempt.", type=float, default=1.0
    ),
    "--hard-ttl-seconds": Flag(
        "hard_ttl_seconds",
        f"Single-resource TTL, {MIN_TTL_SECONDS}..{MAX_TTL_SECONDS}.",
        type=int,
        default=MAX_TTL_SECONDS,
    ),
    "--revision": Flag(
        "revision", "Distinguish a rebuilt profile whose inputs changed at one commit."
    ),
}


def _read(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _identity_matches(document: Mapping[str, Any], field: str) -> bool:
    payload = dict(document)
    declared = payload.pop(field, None)
    return bool(declared) and declared == canonical_sha256(payload)


def _number(value: Any) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return None
    return float(value)


def _receipt_blockers(context: LaneLiveProfileContext) -> list[str]:
    """The bundle checks the allocator makes, made before an attempt is spent.

    Read from the receipt's own bytes rather than from the resolved view: the
    view exists to make a lane-native receipt portable, and an identity check
    run against a translation of a document is not an identity check.
    """

    found: list[str] = []
    receipt = _read(context.receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        found.append("native_camera_input_receipt_schema_invalid")
    if receipt.get("status") != "completed":
        found.append(f"native_camera_input_receipt_not_completed:{receipt.get('status')}")
    if not _identity_matches(receipt, "receipt_sha256"):
        found.append("native_camera_input_receipt_identity_invalid")
    manifest = receipt.get("manifest")
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    if manifest.get("source_commit") != context.source_commit:
        found.append(f"bundle_commit_not_source_commit:{manifest.get('source_commit')}")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("label_free") is not True
        or manifest.get("rankings_or_policy_outcomes_accessed") is not False
        or manifest.get("purpose")
        != "private_internal_nvidia_warehouse_native_camera_canary"
    ):
        found.append("native_camera_input_freeze_invalid")
    if not _identity_matches(manifest, "manifest_sha256"):
        found.append("native_camera_input_manifest_identity_invalid")
    return found


def _release_blockers(context: LaneLiveProfileContext) -> list[str]:
    found: list[str] = []
    release = _read(context.extra_paths.get("release_evidence"))
    if release.get("schema_version") != RELEASE_SCHEMA_VERSION:
        found.append("native_camera_release_schema_invalid")
    if release.get("status") != "passed":
        found.append(f"native_camera_release_status_not_passed:{release.get('status')}")
    if release.get("source_commit") != context.source_commit:
        found.append(
            f"native_camera_release_source_commit_mismatch:{release.get('source_commit')}"
        )
    if not _DIGEST_REF.fullmatch(str(release.get("resolved_digest_ref") or "")):
        found.append("native_camera_release_image_not_digest_pinned")
    if release.get("runnable_platform") != "linux/amd64":
        found.append("native_camera_release_platform_invalid")
    if release.get("isaac_sim_major_version") != 6:
        found.append("native_camera_release_isaac_major_invalid")
    if (
        release.get("source_dirty_patch_sha256")
        != CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    ):
        found.append("native_camera_release_dirty_overlay_forbidden")
    return found


def _preflight_blockers(context: LaneLiveProfileContext) -> list[str]:
    """Provider shape, plus the one sizing number that survives the refresh."""

    found: list[str] = []
    preflight = _read(context.extra_paths.get("preflight_bundle"))
    if preflight.get("schema_version") not in PREFLIGHT_SCHEMAS:
        found.append("native_camera_preflight_schema_invalid")
    if preflight.get("status") != "verified":
        found.append("native_camera_preflight_not_verified")
    if preflight.get("provider") != "vast" or preflight.get("provider_api_verified") is not True:
        found.append("native_camera_preflight_provider_not_verified")
    if preflight.get("provider_inventory_verified_zero") is not True:
        found.append("native_camera_preflight_inventory_not_zero")
    if preflight.get("single_gpu_available") is not True:
        found.append("native_camera_preflight_single_gpu_unavailable")
    memory = preflight.get("gpu_memory_bytes")
    if type(memory) is not int or memory < MIN_GPU_MEMORY_BYTES:
        found.append("native_camera_preflight_gpu_memory_below_floor")
    disk = preflight.get("container_disk_bytes")
    if type(disk) is not int or disk < MIN_CONTAINER_DISK_BYTES:
        found.append("native_camera_preflight_container_disk_below_floor")
    price = _number(preflight.get("on_demand_price_usd_per_hour"))
    if price is None or price <= 0:
        found.append("native_camera_preflight_hourly_price_invalid")
    elif price * context.hard_ttl_seconds / 3600.0 > context.max_spend_usd:
        # The lane's admission compares exactly this, having already been handed
        # a provider.
        found.append(
            "native_camera_preflight_price_exceeds_declared_spend:"
            f"{price * context.hard_ttl_seconds / 3600.0}>{context.max_spend_usd}"
        )
    return found


def _secret_url_blockers(context: LaneLiveProfileContext) -> list[str]:
    """A presigned URL file is opened at the paid boundary, or not at all."""

    found: list[str] = []
    for name in SECRET_URL_INPUTS:
        path = context.extra_paths.get(name)
        if path is None:
            continue
        if path.is_symlink() or not path.is_file():
            found.append(f"secret_url_file_missing_or_unsafe:{name}")
            continue
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            found.append(f"secret_url_file_not_private_0600:{name}")
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            found.append(f"secret_url_file_unreadable:{name}")
            continue
        if not value.startswith("https://") or any(char.isspace() for char in value):
            found.append(f"secret_url_file_not_https_url:{name}")
    return found


def _campaign_blockers(
    context: LaneLiveProfileContext, limits: CampaignLimits
) -> list[str]:
    """`ProductionGpuCampaignBudget` raises on each of these, once armed."""

    found: list[str] = []
    ledger = context.extra_paths.get("campaign_budget_ledger")
    if ledger is None:
        return found
    if ledger.exists() and (ledger.is_symlink() or not ledger.is_file()):
        found.append("campaign_budget_ledger_not_a_regular_file")
    if not ledger.parent.is_dir():
        found.append("campaign_budget_ledger_directory_missing")
    if limits.initial_spent_usd < 0 or limits.initial_used_gpu_seconds < 0:
        found.append("campaign_initial_usage_negative")
    if limits.total_spend_cap_usd > AUTHORIZED_SPEND_CAP_USD:
        found.append("campaign_spend_cap_exceeds_authorization")
    if limits.wall_cap_seconds > AUTHORIZED_GPU_WALL_CAP_SECONDS:
        found.append("campaign_wall_cap_exceeds_authorization")
    if limits.initial_spent_usd > limits.total_spend_cap_usd:
        found.append("campaign_initial_spend_exceeds_cap")
    if limits.initial_used_gpu_seconds > limits.wall_cap_seconds:
        found.append("campaign_initial_wall_time_exceeds_cap")
    if limits.initial_spent_usd + context.max_spend_usd > limits.total_spend_cap_usd:
        found.append("campaign_attempt_would_exceed_cap")
    return found


def _lane_blockers(limits: CampaignLimits) -> Callable[[LaneLiveProfileContext], list[str]]:
    def blockers(context: LaneLiveProfileContext) -> list[str]:
        found = [
            *_receipt_blockers(context),
            *_release_blockers(context),
            *_preflight_blockers(context),
            *_secret_url_blockers(context),
            *_campaign_blockers(context, limits),
        ]
        if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
            found.append("native_camera_budget_invalid")
        return found

    return blockers


def _lane_argv(limits: CampaignLimits) -> Callable[[LaneLiveProfileContext], list[str]]:
    def argv(context: LaneLiveProfileContext) -> list[str]:
        extras = context.extra_paths
        return [
            "--release-evidence", str(extras["release_evidence"]),
            "--native-camera-input-bundle-receipt", str(context.receipt_path),
            "--preflight-bundle", str(extras["preflight_bundle"]),
            "--openpi-hard-ttl-seconds", str(context.hard_ttl_seconds),
            "--openpi-max-spend-usd", str(context.max_spend_usd),
            # Six arguments the allocator only demands under `--execute`. A dry
            # run is green without them and a live run is decided by them.
            "--provider-bundle-url-file", str(extras["provider_bundle_url_file"]),
            "--provider-output-put-url-file", str(extras["provider_output_put_url_file"]),
            "--provider-output-get-url-file", str(extras["provider_output_get_url_file"]),
            "--campaign-budget-ledger", str(extras["campaign_budget_ledger"]),
            "--campaign-initial-spent-usd", str(limits.initial_spent_usd),
            "--campaign-initial-used-gpu-seconds", str(limits.initial_used_gpu_seconds),
            "--campaign-total-spend-cap-usd", str(limits.total_spend_cap_usd),
            "--campaign-wall-cap-seconds", str(limits.wall_cap_seconds),
        ]

    return argv


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    """Everything the run reads that is not a rotating secret.

    The three presigned URL files are deliberately absent. Their bytes change
    without the run changing, so pinning them would either forbid a re-issued
    URL or record a secret's digest in a published document.
    """

    release = context.extra_paths["release_evidence"]
    preflight = context.extra_paths["preflight_bundle"]
    return [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "native_camera_input_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
        {
            "name": "native_camera_release_evidence",
            "path": str(release),
            "digest": file_digest(release),
        },
        {
            "name": "native_camera_provider_preflight",
            "path": str(preflight),
            "digest": file_digest(preflight),
        },
    ]


DEFAULT_CAMPAIGN = CampaignLimits(
    initial_spent_usd=0.0,
    initial_used_gpu_seconds=0,
    total_spend_cap_usd=AUTHORIZED_SPEND_CAP_USD,
    wall_cap_seconds=AUTHORIZED_GPU_WALL_CAP_SECONDS,
)

SPEC = LaneLiveProfileSpec(
    profile_id_prefix=PROFILE_ID_PREFIX,
    profile_builder="build_new_site_native_camera_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"new-site-native-camera-{context.source_commit[:12]}",
    # The scene is the public NVIDIA PhysicalAI SimReady Warehouse dataset at a
    # pinned revision. It is not a capture and not the InteriorGS/SAGE pair, and
    # `source_kind` is a provenance claim rather than a label to borrow.
    source_kind="nvidia_simready_warehouse",
    lane_argv=_lane_argv(DEFAULT_CAMPAIGN),
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers(DEFAULT_CAMPAIGN),
    extra_path_names=EXTRA_PATH_NAMES,
)


def build_new_site_native_camera_live_profile(
    *,
    bundle_receipt_path: str | Path,
    release_evidence_path: str | Path,
    preflight_bundle_path: str | Path,
    provider_bundle_url_file: str | Path,
    provider_output_put_url_file: str | Path,
    provider_output_get_url_file: str | Path,
    campaign_budget_ledger: str | Path,
    campaign_initial_spent_usd: float,
    campaign_initial_used_gpu_seconds: int,
    source_commit: str,
    raw_manifest_uri: str,
    campaign_total_spend_cap_usd: float = AUTHORIZED_SPEND_CAP_USD,
    campaign_wall_cap_seconds: int = AUTHORIZED_GPU_WALL_CAP_SECONDS,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 1.0,
    hard_ttl_seconds: int = MAX_TTL_SECONDS,
    revision: str | None = None,
) -> dict[str, Any]:
    """Derive this lane's live profile, or refuse with every reason at once."""

    limits = CampaignLimits(
        initial_spent_usd=float(campaign_initial_spent_usd),
        initial_used_gpu_seconds=int(campaign_initial_used_gpu_seconds),
        total_spend_cap_usd=float(campaign_total_spend_cap_usd),
        wall_cap_seconds=int(campaign_wall_cap_seconds),
    )
    return build_lane_live_profile(
        replace(
            SPEC,
            lane_argv=_lane_argv(limits),
            lane_blockers=_lane_blockers(limits),
        ),
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        max_spend_usd=max_spend_usd,
        extra_paths={
            "release_evidence": release_evidence_path,
            "preflight_bundle": preflight_bundle_path,
            "campaign_budget_ledger": campaign_budget_ledger,
            "provider_bundle_url_file": provider_bundle_url_file,
            "provider_output_put_url_file": provider_output_put_url_file,
            "provider_output_get_url_file": provider_output_get_url_file,
        },
    )


def _destination(flag: str) -> str:
    return flag.removeprefix("--").replace("-", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, spec in FLAGS.items():
        options: dict[str, Any] = {"help": spec.help}
        if spec.type is not None:
            options["type"] = spec.type
        if spec.required:
            options["required"] = True
        else:
            options["default"] = spec.default
        parser.add_argument(flag, **options)
    parser.add_argument("--output", required=True, help="Where to write the profile.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = build_new_site_native_camera_live_profile(
            **{
                spec.parameter: getattr(args, _destination(flag))
                for flag, spec in FLAGS.items()
            }
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
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "output": str(output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
