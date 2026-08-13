"""One derivation of a live launch profile, shared by every paid lane.

Fourteen paid lanes can be launched by the canonical allocator. Three could be
launched from the website, because a launch profile is what carries a lane
across that boundary and only three lanes had a builder for one. The obvious
next step was eleven more builders, and the two that already existed were 319
and 281 lines sharing 184 identical lines between them.

That copy is the hazard, not the line count. `stage_paid_lane_bundle.py` says it
about staging and it is just as true here: a per-lane copy is a per-lane
opportunity to omit a check. The three things a live profile exists to
guarantee -- that its inputs are host-resident, that its declared spend is the
one its attempt authority was issued against, and that its terminal contract
asks for the evidence the lane actually emits -- are exactly the three a
hurried copy drops, and each one is only discovered at the paid boundary.

So the skeleton lives here once and a lane brings a `LaneLiveProfileSpec`: its
probe kind, its TTL band, the allocator arguments only it understands, and the
receipts it binds. Roughly forty lines of declaration instead of three hundred
of near-duplicate procedure.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .host_resident_launch_inputs import (
    HostResidentInputError,
    launch_profile_residency_blockers,
    resolve_host_resident_bundle_receipt,
)
from .task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    TaskEvaluationLaunchError,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)

#: Expanded by the dispatcher to this launch's run directory. A profile that
#: names a real directory names one machine's directory.
RUN_ROOT = "{launch_run_root}"
#: The secret profile every ADP lane runs under.
SECRET_PROFILE_ID = "canonical-vast-adp"
PROFILE_SCHEMA_VERSION = "task_evaluation_launch_profile.v1"
PROGRAM_ID = "arm-decision-proof-v1"


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def shared_control_surface(*, required_providers: Sequence[str] = ("vast",)) -> dict[str, Any]:
    """The part of a launch profile that is the same for every paid lane.

    These four blocks are what make a run provable rather than merely finished:
    which allocator may run it, that a watchdog and a teardown are required,
    that provider zero must be shown, that no retry is permitted, and which
    artifacts the terminal contract will insist on.

    They are also the part a hurried copy gets subtly wrong, and a profile is
    published once and read on every later launch -- so a lane that quietly
    dropped `provider_zero_required` would keep passing until the day it
    mattered. One definition, used by every builder, including the ones whose
    inputs are too different to share the whole skeleton below.
    """

    return {
        "reconciliation": {
            "max_guard_age_seconds": 300,
            "required_providers": list(required_providers),
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": SECRET_PROFILE_ID,
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "terminal_contract": {
            "result_path": f"{RUN_ROOT}/allocator/result.json",
            "success_statuses": ["completed"],
            "required_values": {"continuing_spend_from_this_run": False, "retry_cap": 0},
            # Both are sealed by `seal_lane_terminal_artifacts`. Asking for them
            # here is what makes a lane that stops emitting them fail.
            "required_path_fields": ["teardown_manifest_path", "artifact_manifest_path"],
        },
        "webapp_sync": {"max_attempts": 20},
    }


@dataclass(frozen=True)
class LaneLiveProfileContext:
    """Everything a lane hook needs, already resolved against *this* host."""

    receipt_path: Path
    receipt: Mapping[str, Any]
    #: Where the receipt's references actually live here, not where they were
    #: authored. Lane argv and immutable inputs must use these.
    resolutions: Mapping[str, Any]
    profile_id: str
    source_commit: str
    max_hourly_rate_usd: float
    hard_ttl_seconds: int
    max_spend_usd: float
    raw_manifest_uri: str
    #: Lane-specific paths the caller passed through, already resolved.
    extra_paths: Mapping[str, Path]

    @property
    def bundle_sha256(self) -> str:
        return str(self.receipt.get("bundle_sha256") or "")

    @property
    def bundle_path(self) -> str:
        return str((self.resolutions.get("bundle") or {}).get("path") or "")

    def job_dir(self, dirname: str) -> str:
        return f"{RUN_ROOT}/allocator/{dirname}"


@dataclass(frozen=True)
class LaneLiveProfileSpec:
    """What actually differs between one paid lane's profile and another's."""

    #: Prefixes the profile id. The commit (and any revision) is appended.
    profile_id_prefix: str
    probe_kind: str
    #: The allocator refuses a TTL outside this band for this probe, so a
    #: profile outside it is refused after a provider has been handed over.
    min_ttl_seconds: int
    max_ttl_seconds: int
    #: `bundle_id` and `source_kind` for the profile's source-bundle record.
    source_bundle_id: Callable[[LaneLiveProfileContext], str]
    source_kind: str
    #: The arguments only this lane understands, appended after the shared ones.
    lane_argv: Callable[[LaneLiveProfileContext], list[str]]
    #: The digest-bound inputs this lane pins, beyond the bundle itself.
    immutable_inputs: Callable[[LaneLiveProfileContext], list[dict[str, Any]]]
    #: Refusals only this lane can state. Returning any blocks the build.
    lane_blockers: Callable[[LaneLiveProfileContext], list[str]] = (
        lambda context: []
    )
    #: Overrides the declared spend. Defaults to rate x TTL -- the worst case
    #: this profile can actually reach, not a lane's lifetime ceiling.
    declared_spend: Callable[[LaneLiveProfileContext], float] | None = None
    #: Descriptor for `evaluation_run_spec` / `execution_admission`. Defaults to
    #: the raw manifest URI at the bundle digest.
    run_spec_digest: Callable[[LaneLiveProfileContext], str] | None = None
    required_providers: Sequence[str] = ("vast",)
    claim_ceiling: str = "development_only"
    subcommand: str = "gpu-canary"
    provider: str = "vast"
    extra_path_names: Sequence[str] = field(default_factory=tuple)


def build_lane_live_profile(
    spec: LaneLiveProfileSpec,
    *,
    bundle_receipt_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    max_hourly_rate_usd: float,
    hard_ttl_seconds: int,
    revision: str | None = None,
    max_spend_usd: float | None = None,
    extra_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Derive one lane's live profile, or refuse with every reason at once."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()

    # The receipt records the absolute paths of the machine that built it.
    # Resolve it against this host first: a profile built from an unresolvable
    # receipt publishes those paths into argv, and argv is what the allocator
    # opens.
    blockers: list[str] = []
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_path)
    except HostResidentInputError as exc:
        raise TaskEvaluationLaunchError(str(exc)) from exc
    blockers.extend(resolution["blockers"])
    receipt = resolution["receipt"]

    if receipt.get("status") != "ready":
        blockers.append(f"bundle_receipt_not_ready:{receipt.get('status')}")
    if not spec.min_ttl_seconds <= hard_ttl_seconds <= spec.max_ttl_seconds:
        blockers.append(f"hard_ttl_out_of_band:{hard_ttl_seconds}")

    resolved_extras: dict[str, Path] = {}
    for name in spec.extra_path_names:
        raw = (extra_paths or {}).get(name)
        if raw is None:
            blockers.append(f"lane_input_missing:{name}")
            continue
        resolved_extras[name] = Path(raw).expanduser().resolve()

    profile_id = f"{spec.profile_id_prefix}-{source_commit}"
    if revision:
        # Published profiles are immutable, so a profile whose inputs changed
        # needs its own id rather than a conflicting rewrite of an existing
        # one. Inputs change at a fixed commit more often than they look like
        # they would -- regenerating a config preflight is enough -- and the
        # collision surfaces as an immutable-input digest mismatch on the next
        # launch rather than at publish time.
        profile_id = f"{profile_id}-{revision}"

    worst_case = max_hourly_rate_usd * hard_ttl_seconds / 3600.0
    context = LaneLiveProfileContext(
        receipt_path=receipt_path,
        receipt=receipt,
        resolutions=resolution["resolutions"],
        profile_id=profile_id,
        source_commit=source_commit,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_spend_usd=(
            max_spend_usd
            if max_spend_usd is not None
            else round(worst_case, 6)
        ),
        raw_manifest_uri=raw_manifest_uri,
        extra_paths=resolved_extras,
    )

    blockers.extend(spec.lane_blockers(context))
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    declared_spend = (
        spec.declared_spend(context)
        if spec.declared_spend is not None
        else context.max_spend_usd
    )
    run_spec_digest = (
        spec.run_spec_digest(context)
        if spec.run_spec_digest is not None
        else context.bundle_sha256
    )

    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "program_id": PROGRAM_ID,
        "claim_ceiling": spec.claim_ceiling,
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": spec.subcommand,
            "argv": [
                "--admission-out", f"{RUN_ROOT}/allocator/admission.json",
                "--bound-request-out", f"{RUN_ROOT}/allocator/bound-request.json",
                "--adapter-output", f"{RUN_ROOT}/allocator/result.json",
                "--pod-name", profile_id,
                "--expected-source-commit", source_commit,
                "--provider", spec.provider,
                "--probe-kind", spec.probe_kind,
                *spec.lane_argv(context),
            ],
            "max_spend_usd": float(declared_spend),
            "hard_ttl_seconds": hard_ttl_seconds,
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": True,
            "blockers": [],
            "readiness_receipt": {"uri": raw_manifest_uri, "digest": run_spec_digest},
        },
        "evaluation_run_spec": {"uri": raw_manifest_uri, "digest": run_spec_digest},
        "source_bundle": {
            "bundle_id": spec.source_bundle_id(context),
            "source_kind": spec.source_kind,
            "uri": raw_manifest_uri,
            "digest": run_spec_digest,
        },
        "immutable_inputs": spec.immutable_inputs(context),
        "runtime_environment": {},
        **shared_control_surface(required_providers=spec.required_providers),
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")

    validation = [
        *validate_launch_profile(profile),
        *verify_profile_immutable_inputs(profile),
        # A profile is published once and read on every later launch, so this is
        # the last chance to catch an authoring path.
        *launch_profile_residency_blockers(profile),
    ]
    if validation:
        raise TaskEvaluationLaunchError(",".join(sorted(set(validation))))
    return profile


def allowlist_arguments(allowlist: Sequence[int]) -> list[str]:
    """Repeat `--adp-allowed-active-vast-instance-id` once per admitted id."""

    return [
        argument
        for instance_id in allowlist
        for argument in ("--adp-allowed-active-vast-instance-id", str(instance_id))
    ]
