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
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .host_resident_launch_inputs import (
    HostResidentInputError,
    launch_profile_residency_blockers,
    resolve_host_resident_bundle_receipt,
)
from .paid_attempt_authority import validate_same_goal_spend_reconciliation
from .robot_eval_provider_input_setup import LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS
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


def expand_prior_spend_immutable_inputs(
    immutable_inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose every local prior-spend dependency to profile publication.

    A paid-attempt authority binds a lane-local reconciliation, and that
    reconciliation in turn binds the terminal result, teardown, provider-zero,
    and official-billing receipts it proves. The allocator reopens all of
    those files as the service account. Binding only the authority itself in a
    live profile lets a root-authored, service-unreadable nested receipt survive
    publication and fail at the paid boundary.

    Keep this expansion in the shared profile skeleton so every paid issuer gets
    the same consumer-readability preflight. The canonical reconciliation
    validator also reopens and digest-checks each source before any paths are
    added to the profile.
    """

    expanded = [dict(item) for item in immutable_inputs]
    observed_paths = {
        str(Path(str(item.get("path") or "")).expanduser().resolve())
        for item in expanded
        if str(item.get("path") or "").strip()
    }
    for item in tuple(expanded):
        name = str(item.get("name") or "")
        if "authority" not in name:
            continue
        authority_path = Path(str(item.get("path") or "")).expanduser().resolve()
        if authority_path.suffix.lower() != ".json":
            continue
        try:
            authority = json.loads(authority_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationLaunchError(
                f"live_profile_authority_input_invalid:{name}"
            ) from exc
        if not isinstance(authority, Mapping):
            raise TaskEvaluationLaunchError(
                f"live_profile_authority_input_invalid:{name}"
            )
        reconciliation_record = authority.get("prior_spend_reconciliation")
        if reconciliation_record is None:
            continue
        if not isinstance(reconciliation_record, Mapping):
            raise TaskEvaluationLaunchError(
                f"live_profile_prior_spend_dependency_invalid:{name}"
            )
        reconciliation_path = Path(
            str(reconciliation_record.get("path") or "")
        ).expanduser().resolve()
        uses_shared_lane_reconciliation = (
            "prior_terminal_attempts" in authority
            and "prior_actual_provider_spend_usd" in authority
        )
        try:
            if uses_shared_lane_reconciliation:
                reconciliation, observed_record = (
                    validate_same_goal_spend_reconciliation(
                        reconciliation_path,
                        expected_total_cost_usd=authority.get(
                            "prior_actual_provider_spend_usd"
                        ),
                    )
                )
                if observed_record != dict(reconciliation_record):
                    raise ValueError("prior_spend_reconciliation_record_mismatch")
            else:
                # The semantic-teacher issuer predates the shared five-lane
                # reconciliation contract. Its lane validator has already
                # reopened and validated that schema above; retain support for
                # it while still exposing any nested receipt paths to profile
                # publication and its service-account readability check.
                reconciliation = json.loads(
                    reconciliation_path.read_text(encoding="utf-8")
                )
                if (
                    reconciliation_path.is_symlink()
                    or not isinstance(reconciliation, Mapping)
                    or reconciliation_path.stat().st_size
                    != reconciliation_record.get("size_bytes")
                    or file_digest(reconciliation_path)
                    != reconciliation_record.get("sha256")
                ):
                    raise ValueError("prior_spend_reconciliation_record_mismatch")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise TaskEvaluationLaunchError(
                f"live_profile_prior_spend_dependency_invalid:{name}"
            ) from exc

        dependencies: list[tuple[str, Mapping[str, Any]]] = [
            ("reconciliation", reconciliation_record)
        ]
        for entry_index, entry in enumerate(reconciliation.get("entries") or []):
            if not isinstance(entry, Mapping):
                raise TaskEvaluationLaunchError(
                    f"live_profile_prior_spend_dependency_invalid:{name}"
                )
            for source in entry.get("source_receipts") or []:
                if not isinstance(source, Mapping) or not isinstance(
                    source.get("record"), Mapping
                ):
                    raise TaskEvaluationLaunchError(
                        f"live_profile_prior_spend_dependency_invalid:{name}"
                    )
                role = str(source.get("role") or "source")
                dependencies.append(
                    (f"entry_{entry_index}_{role}", source["record"])
                )

        for suffix, record in dependencies:
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            resolved = str(path)
            if resolved in observed_paths:
                continue
            expanded.append(
                {
                    "name": f"{name}_prior_spend_{suffix}",
                    "path": resolved,
                    "digest": str(record.get("sha256") or ""),
                }
            )
            observed_paths.add(resolved)
    return expanded


def shared_control_surface(
    *,
    required_providers: Sequence[str] = ("vast",),
    additional_required_path_fields: Sequence[str] = (),
) -> dict[str, Any]:
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

    A lane whose result carries one more terminal artifact names it in
    `additional_required_path_fields` rather than restating the whole contract.
    The SAM 3.1 lane restated it to add one field, and the copy it needed to
    make was the four blocks this function exists to keep single.
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
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
                *additional_required_path_fields,
            ],
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
    #: Small non-secret scalar/list controls embedded directly into the final
    #: profile argv and therefore covered by the profile digest.
    extra_values: Mapping[str, Any] = field(default_factory=dict)

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
    #: Exact operator-facing builder filename bound by a generated-manifest
    #: publication receipt.
    profile_builder: str
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
    #: A narrow lane-specific field covered by the launch-profile digest. This
    #: is deliberately data, never argv: providers receive only the bytes they
    #: need, while the dispatcher can reopen the governed evidence before a
    #: standing authorization is consumed.
    profile_fields: Callable[[LaneLiveProfileContext], Mapping[str, Any]] = (
        lambda context: {}
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
    #: Require the website standing authorization to be a single-use launch
    #: capability.  The dispatcher consumes it atomically before invoking the
    #: allocator, so an explicit launch-id scope cannot bypass this lane's
    #: exactly-once paid boundary.
    one_use_standing_authority_required: bool = False


def bind_live_profile_manifest_publication(
    *,
    reference: str,
    source_commit: str,
    run_spec_digest: str,
    profile_builder: str,
    immutable_inputs: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve a pinned URI or fail-closed publication receipt into a URI.

    Exact-commit raw GitHub inputs are immutable by construction. Every other
    reference must be a local, self-digesting publication receipt; accepting a
    bare generated URI would let an operator bypass provider readback.
    """

    normalized = str(reference or "").strip()
    publication_seam = LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS.get(profile_builder)
    if publication_seam is None:
        raise TaskEvaluationLaunchError("manifest_publication_profile_builder_unregistered")
    pinned = re.fullmatch(
        rf"https://raw\.githubusercontent\.com/[^/]+/[^/]+/{re.escape(source_commit)}/.+",
        normalized,
    )
    inputs = [dict(item) for item in immutable_inputs]
    if publication_seam == "exact_commit_raw_github" and pinned:
        return normalized, None, inputs
    if publication_seam == "exact_commit_raw_github":
        raise TaskEvaluationLaunchError("exact_commit_raw_github_manifest_required")
    receipt_path = Path(normalized).expanduser().resolve()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise TaskEvaluationLaunchError("manifest_publication_receipt_required")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchError("manifest_publication_receipt_invalid") from exc
    source = receipt.get("source") if isinstance(receipt, Mapping) else None
    published_uri = str(receipt.get("published_uri") or "") if isinstance(receipt, Mapping) else ""
    digest_identity = run_spec_digest.removeprefix("sha256:")
    expected_uri_suffix = (
        f"/sha256/{digest_identity[:2]}/{digest_identity}.json"
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version")
        != "task_evaluation_immutable_manifest_publication.v1"
        or receipt.get("status") != "published"
        or receipt.get("profile_builder") != profile_builder
        or receipt.get("publication_seam")
        not in {
            "content_addressed_full_readback",
            # Preserve already-published GCS receipts from before this seam
            # became provider-neutral.
            "gcs_content_addressed_full_readback",
        }
        or urlparse(published_uri).scheme not in {"gs", "s3", "r2"}
        or not published_uri.endswith(expected_uri_suffix)
        or receipt.get("storage_scheme") != urlparse(published_uri).scheme
        or (
            receipt.get("publication_seam") == "gcs_content_addressed_full_readback"
            and receipt.get("storage_scheme") != "gs"
        )
        or not isinstance(source, Mapping)
        or source.get("sha256") != run_spec_digest
        or receipt.get("remote_sha256") != run_spec_digest
        or source.get("size_bytes") != receipt.get("remote_size_bytes")
        or receipt.get("provider_full_byte_readback_verified") is not True
        or receipt.get("content_addressed_key") is not True
        or receipt.get("exclusive_create") is not True
        or receipt.get("blockers") != []
        or receipt.get("provider_compute_mutation_performed") is not False
        or receipt.get("paid_resource_allocated") is not False
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(receipt.get("upload_receipt_digest") or "")
        )
        or receipt.get("raw_secret_values_recorded") is not False
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise TaskEvaluationLaunchError("manifest_publication_receipt_invalid")
    source_path = Path(str(source.get("path") or "")).expanduser().resolve()
    source_inputs = [
        item for item in inputs if item.get("name") == "source_bundle_manifest"
    ]
    if (
        len(source_inputs) != 1
        or source_path.is_symlink()
        or not source_path.is_file()
        or source.get("size_bytes") != source_path.stat().st_size
        or file_digest(source_path) != run_spec_digest
        or Path(str(source_inputs[0].get("path") or "")).expanduser().resolve()
        != source_path
        or source_inputs[0].get("digest") != run_spec_digest
    ):
        raise TaskEvaluationLaunchError("manifest_publication_source_not_immutable_input")
    inputs.append(
        {
            "name": "manifest_publication_receipt",
            "path": str(receipt_path),
            "digest": file_digest(receipt_path),
        }
    )
    return (
        published_uri,
        {
            "receipt_path": str(receipt_path),
            "receipt_digest": receipt["receipt_digest"],
            "published_uri": published_uri,
            "source_sha256": run_spec_digest,
            "remote_sha256": run_spec_digest,
            "profile_builder": profile_builder,
            "provider_full_byte_readback_verified": True,
        },
        inputs,
    )


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
    extra_values: Mapping[str, Any] | None = None,
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
        extra_values=dict(extra_values or {}),
    )

    blockers.extend(spec.lane_blockers(context))
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    declared_spend = (
        spec.declared_spend(context)
        if spec.declared_spend is not None
        else context.max_spend_usd
    )
    immutable_inputs = expand_prior_spend_immutable_inputs(
        spec.immutable_inputs(context)
    )
    source_manifests = [
        item
        for item in immutable_inputs
        if item.get("name") == "source_bundle_manifest"
    ]
    if spec.run_spec_digest is None and len(source_manifests) != 1:
        raise TaskEvaluationLaunchError("source_bundle_manifest_input_not_unique")
    run_spec_digest = (
        spec.run_spec_digest(context)
        if spec.run_spec_digest is not None
        else str(source_manifests[0].get("digest") or "")
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", run_spec_digest):
        raise TaskEvaluationLaunchError("live_profile_run_spec_digest_invalid")
    manifest_uri, manifest_publication, immutable_inputs = (
        bind_live_profile_manifest_publication(
            reference=raw_manifest_uri,
            source_commit=source_commit,
            run_spec_digest=run_spec_digest,
            profile_builder=spec.profile_builder,
            immutable_inputs=immutable_inputs,
        )
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
            "readiness_receipt": {"uri": manifest_uri, "digest": run_spec_digest},
        },
        "evaluation_run_spec": {"uri": manifest_uri, "digest": run_spec_digest},
        "source_bundle": {
            "bundle_id": spec.source_bundle_id(context),
            "source_kind": spec.source_kind,
            "uri": manifest_uri,
            "digest": run_spec_digest,
        },
        "immutable_inputs": immutable_inputs,
        "runtime_environment": {},
        **shared_control_surface(required_providers=spec.required_providers),
    }
    lane_profile_fields = spec.profile_fields(context)
    if not isinstance(lane_profile_fields, Mapping) or set(profile).intersection(
        lane_profile_fields
    ):
        raise TaskEvaluationLaunchError("lane_live_profile_fields_invalid")
    profile.update(dict(lane_profile_fields))
    if spec.one_use_standing_authority_required:
        profile["standing_launch_authorization"] = {
            "schema_version": (
                "task_evaluation_standing_launch_authorization_requirement.v1"
            ),
            "required_for_live_execution": True,
            "maximum_launches": 1,
            "consumption_must_precede_allocator": True,
        }
    if manifest_publication is not None:
        profile["manifest_publication"] = manifest_publication
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
