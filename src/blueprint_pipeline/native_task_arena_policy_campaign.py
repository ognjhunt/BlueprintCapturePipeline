"""Two-member spend and resource-name authority for native policy launches.

The native Task Arena candidates are intentionally launched as a pair.  A
single-attempt authority can prove one member is bounded, but it cannot prove
that two authorities issued from the same predecessor fit under the aggregate
goal ceiling.  It also cannot name a sibling instance before Vast assigns an
instance id.

This manifest closes both gaps without weakening the ordinary authority path:
it freezes exactly the pi0.5 and GR00T bundles, their launch ids and provider
resource-name prefixes, both members' complete spend bounds, the controls
instance ids that may already be active, and the official prior-spend chain.
Each ordinary retry-0 authority then binds one member of this same manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_paid_authority import (
    AGGREGATE_GOAL_SPEND_CAP_USD,
    native_task_arena_attempt_budget_blockers,
    validate_terminal_spend_chain,
)
from .native_task_arena_policy_bundle import (
    load_verified_native_task_arena_policy_bundle,
)
from .native_task_arena_policy_diagnostic_bundle import (
    load_verified_native_task_arena_policy_diagnostic_bundle,
)
from .paid_attempt_authority import bind_lane_prior_spend
from .task_evaluation_immutable_input_resolver import (
    ImmutableInputResolutionError,
    resolve_immutable_input,
)


SCHEMA_VERSION = "native_task_arena_policy_campaign.v1"
MEMBER_IDS = ("pi05_droid", "groot_n17_droid")
POLICY_EXECUTION_MODES = frozenset({"policy", "policy_diagnostic"})
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,192}$")
# Campaign resource names are exact provider labels, not prefixes.  They must
# remain inside the Native Task Arena policy family accepted by the independent
# watchdog; accepting an arbitrary ``blueprint-*`` label lets authority issue
# successfully and then guarantees a before-allocation watchdog failure.  The
# final 128 bits are operator-generated entropy sealed before either member
# starts, keeping the sibling identity collision-resistant without depending
# on a provider-assigned instance id that does not exist at authority issuance.
_RESOURCE_NAME = re.compile(
    r"^blueprint-native-task-policy-[a-z0-9-]{1,60}-[0-9a-f]{32}$"
)
_POLICY_SPEC_ARCHIVE_PATH = (
    "provider_runtime/runtime_inputs/"
    "native_task_arena_policy_execution_spec.v1.json"
)
_SCENE_PLAN_ARCHIVE_PATH = (
    "provider_runtime/native_task_packet/native_task_arena_scene_plan.v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if path.is_symlink() or not isinstance(value, dict):
        raise ValueError(code)
    return value


def _bound_record(value: Any, *, code: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(code)
    try:
        path = resolve_immutable_input(
            str(value.get("path") or ""),
            expected_digest=str(value.get("sha256") or ""),
            expected_size_bytes=value.get("size_bytes"),
        )
    except ImmutableInputResolutionError as exc:
        raise ValueError(code) from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value.get("size_bytes")
        or _sha256(path) != value.get("sha256")
    ):
        raise ValueError(code)
    return path


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _verified_policy_bundle(path: Path, *, blueprint_commit: str) -> dict[str, Any]:
    raw = _read(path, code="native_task_arena_policy_campaign_bundle_invalid")
    mode = str(raw.get("execution_mode") or "")
    loader = {
        "policy": load_verified_native_task_arena_policy_bundle,
        "policy_diagnostic": load_verified_native_task_arena_policy_diagnostic_bundle,
    }.get(mode)
    if loader is None:
        raise ValueError("native_task_arena_policy_campaign_bundle_mode_invalid")
    return loader(
        path,
        expected_implementation_commit=blueprint_commit,
        expected_packet_receipt_digest=raw.get("packet_receipt_digest"),
        expected_runtime_source_packet_digest=(raw.get("runtime_source_packet") or {}).get(
            "receipt_digest"
        ),
    )


def _archive_json(receipt: Mapping[str, Any], member: str, *, code: str) -> dict[str, Any]:
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    try:
        with zipfile.ZipFile(bundle) as archive:
            info = archive.getinfo(member)
            if info.file_size > 4 * 1024 * 1024:
                raise ValueError(code)
            value = json.loads(archive.read(info).decode("utf-8"))
    except (OSError, KeyError, UnicodeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _runtime_input_record(receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in receipt.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping) and Path(str(row.get("relative_path") or "")).name == name
    ]
    if len(matches) != 1:
        raise ValueError("native_task_arena_policy_campaign_shared_input_missing")
    row = matches[0]
    if (
        not _DIGEST.fullmatch(str(row.get("sha256") or ""))
        or isinstance(row.get("size_bytes"), bool)
        or not isinstance(row.get("size_bytes"), int)
        or row["size_bytes"] <= 0
    ):
        raise ValueError("native_task_arena_policy_campaign_shared_input_invalid")
    return {
        "relative_path": str(row["relative_path"]),
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
    }


def _shared_scientific_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    spec = _archive_json(
        receipt,
        _POLICY_SPEC_ARCHIVE_PATH,
        code="native_task_arena_policy_campaign_execution_spec_invalid",
    )
    scene = _archive_json(
        receipt,
        _SCENE_PLAN_ARCHIVE_PATH,
        code="native_task_arena_policy_campaign_scene_plan_invalid",
    )
    cadence = scene.get("control_cadence") or scene.get("cadence")
    if not isinstance(cadence, Mapping):
        raise ValueError("native_task_arena_policy_campaign_episode_limits_missing")
    runtime_source = receipt.get("runtime_source_packet")
    runtime_source = dict(runtime_source) if isinstance(runtime_source, Mapping) else {}
    projection: dict[str, Any] = {
        "scene_id": receipt.get("scene_id"),
        "task_id": receipt.get("task_id"),
        "packet": {
            "request_digest": receipt.get("request_digest"),
            "receipt_digest": receipt.get("packet_receipt_digest"),
            "scene_plan_digest": receipt.get("arena_scene_plan_digest"),
            "runtime_contract_digest": receipt.get("runtime_contract_digest"),
            "scenario_instance_digest": receipt.get("scenario_instance_digest"),
        },
        "runtime_source": {
            key: runtime_source.get(key)
            for key in ("receipt_digest", "packet_sha256", "packet_size_bytes")
        },
        "construction_result": {
            "record": _runtime_input_record(
                receipt, "native_task_arena_construction_result.v1.json"
            ),
            "result_digest": spec.get("construction_result_digest"),
        },
        "control_result": {
            "record": _runtime_input_record(
                receipt, "native_task_arena_control_result.v1.json"
            ),
            "result_digest": spec.get("control_result_digest"),
            "control_pair_digest": spec.get("control_pair_digest"),
        },
        "task_cell": {
            "task_id": spec.get("task_id"),
            "cell_id": spec.get("cell_id"),
            "prompt": spec.get("prompt"),
            "task_spec_digest": canonical_digest(scene.get("task_spec") or {}),
        },
        "execution_limits": {
            "max_policy_queries": spec.get("max_policy_queries"),
            "open_loop_horizon": spec.get("open_loop_horizon"),
            "control_cadence": dict(cadence),
            "control_cadence_digest": canonical_digest(cadence),
        },
        "predecessor_identity": {
            key: spec.get(key)
            for key in (
                "execution_authority",
                "claim_ceiling",
                "initial_state",
                "controls_qualified",
                "zero_action_negative_bound_separately",
            )
        },
    }
    required_digests = (
        projection["packet"]["request_digest"],
        projection["packet"]["receipt_digest"],
        projection["packet"]["scene_plan_digest"],
        projection["packet"]["runtime_contract_digest"],
        projection["packet"]["scenario_instance_digest"],
        projection["runtime_source"]["receipt_digest"],
        projection["runtime_source"]["packet_sha256"],
        projection["construction_result"]["result_digest"],
        projection["control_result"]["result_digest"],
        projection["control_result"]["control_pair_digest"],
    )
    limits = projection["execution_limits"]
    if (
        not str(projection["scene_id"] or "")
        or not str(projection["task_id"] or "")
        or projection["task_id"] != projection["task_cell"]["task_id"]
        or not str(projection["task_cell"]["cell_id"] or "")
        or not str(projection["task_cell"]["prompt"] or "")
        or any(not _DIGEST.fullmatch(str(value or "")) for value in required_digests)
        or isinstance(limits["max_policy_queries"], bool)
        or not isinstance(limits["max_policy_queries"], int)
        or limits["max_policy_queries"] <= 0
        or isinstance(limits["open_loop_horizon"], bool)
        or not isinstance(limits["open_loop_horizon"], int)
        or limits["open_loop_horizon"] <= 0
    ):
        raise ValueError("native_task_arena_policy_campaign_shared_projection_invalid")
    projection["projection_digest"] = canonical_digest(
        projection, digest_field="projection_digest"
    )
    return projection


def validate_native_task_arena_policy_campaign(
    value: Mapping[str, Any],
    *,
    expected_blueprint_commit: str | None = None,
) -> dict[str, Any]:
    """Validate the self-contained campaign projection.

    Source files are reopened when the campaign is created.  Later consumers
    reopen the immutable campaign file itself and compare its digest from each
    member authority; the member's own bundle is independently reopened by the
    ordinary paid-authority validator.
    """

    payload = json.loads(json.dumps(value, allow_nan=False))
    errors: list[str] = []
    commit = str(payload.get("blueprint_commit") or "")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("program_id") != "arm-decision-proof-v1"
        or not _IDENTIFIER.fullmatch(str(payload.get("campaign_id") or ""))
        or not _COMMIT.fullmatch(commit)
        or (expected_blueprint_commit is not None and commit != expected_blueprint_commit)
        or payload.get("aggregate_goal_spend_cap_usd") != AGGREGATE_GOAL_SPEND_CAP_USD
        or payload.get("provider_wide_launch_lock_required") is not True
        or payload.get("independent_watchdog_required_per_member") is not True
        or payload.get("automatic_retry_authorized") is not False
    ):
        errors.append("native_task_arena_policy_campaign_identity_invalid")

    controls_ids = payload.get("controls_allowed_active_instance_ids")
    if (
        not isinstance(controls_ids, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in controls_ids
        )
        or controls_ids != sorted(set(controls_ids))
    ):
        errors.append("native_task_arena_policy_campaign_controls_allowlist_invalid")

    prior = payload.get("prior_official_spend")
    prior = dict(prior) if isinstance(prior, Mapping) else {}
    prior_spend = prior.get("aggregate_goal_spend_before_campaign_usd")
    if (
        not _finite_nonnegative(prior_spend)
        or not _finite_nonnegative(prior.get("reconciled_actual_total_usd"))
        or not isinstance(prior.get("prior_terminal_attempt"), Mapping)
        or not isinstance(prior.get("reconciliation"), Mapping)
    ):
        errors.append("native_task_arena_policy_campaign_prior_spend_invalid")

    members = payload.get("members")
    observed_candidates: list[str] = []
    observed_launches: list[str] = []
    observed_resource_names: list[str] = []
    caps: list[float] = []
    modes: list[str] = []
    if not isinstance(members, list) or len(members) != 2:
        errors.append("native_task_arena_policy_campaign_members_invalid")
        members = []
    for member in members:
        row = dict(member) if isinstance(member, Mapping) else {}
        candidate = str(row.get("candidate_id") or "")
        launch_id = str(row.get("launch_id") or "")
        resource_name = str(row.get("resource_name") or "")
        rate = row.get("maximum_hourly_rate_usd")
        cap = row.get("hard_attempt_spend_cap_usd")
        ttl = row.get("maximum_single_resource_ttl_seconds")
        mode = str(row.get("execution_mode") or "")
        if (
            row.get("member_id") != candidate
            or candidate not in MEMBER_IDS
            or not _IDENTIFIER.fullmatch(launch_id)
            or not _RESOURCE_NAME.fullmatch(resource_name)
            or not _DIGEST.fullmatch(str(row.get("bundle_sha256") or ""))
            or not _DIGEST.fullmatch(str(row.get("bundle_input_digest") or ""))
            or not isinstance(row.get("bundle_receipt"), Mapping)
            or row.get("blueprint_commit") != commit
            or mode not in POLICY_EXECUTION_MODES
            or native_task_arena_attempt_budget_blockers(
                max_hourly_rate_usd=rate,
                hard_cap_usd=cap,
                hard_ttl_seconds=ttl,
            )
            or row.get("maximum_automatic_retries") != 0
            or row.get("maximum_provider_allocations") != 1
        ):
            errors.append("native_task_arena_policy_campaign_member_invalid")
            continue
        observed_candidates.append(candidate)
        observed_launches.append(launch_id)
        observed_resource_names.append(resource_name)
        caps.append(float(cap))
        modes.append(mode)
    if (
        tuple(observed_candidates) != MEMBER_IDS
        or len(set(observed_launches)) != 2
        or len(set(observed_resource_names)) != 2
        or len(set(modes)) != 1
        or (modes and payload.get("execution_mode") != modes[0])
    ):
        errors.append("native_task_arena_policy_campaign_member_set_invalid")

    shared = payload.get("shared_scientific_projection")
    shared = dict(shared) if isinstance(shared, Mapping) else {}
    shared_digest = shared.get("projection_digest")
    if (
        not _DIGEST.fullmatch(str(shared_digest or ""))
        or shared_digest != canonical_digest(shared, digest_field="projection_digest")
        or not isinstance(shared.get("packet"), Mapping)
        or not isinstance(shared.get("runtime_source"), Mapping)
        or not isinstance(shared.get("construction_result"), Mapping)
        or not isinstance(shared.get("control_result"), Mapping)
        or not isinstance(shared.get("task_cell"), Mapping)
        or not isinstance(shared.get("execution_limits"), Mapping)
        or not isinstance(shared.get("predecessor_identity"), Mapping)
    ):
        errors.append("native_task_arena_policy_campaign_shared_projection_invalid")
    packet = shared.get("packet") or {}
    runtime_source = shared.get("runtime_source") or {}
    construction = shared.get("construction_result") or {}
    control = shared.get("control_result") or {}
    task_cell = shared.get("task_cell") or {}
    records = (
        construction.get("record") if isinstance(construction, Mapping) else None,
        control.get("record") if isinstance(control, Mapping) else None,
    )
    required_shared_digests = [
        packet.get(key) if isinstance(packet, Mapping) else None
        for key in (
            "request_digest",
            "receipt_digest",
            "scene_plan_digest",
            "runtime_contract_digest",
            "scenario_instance_digest",
        )
    ]
    required_shared_digests.extend(
        runtime_source.get(key) if isinstance(runtime_source, Mapping) else None
        for key in ("receipt_digest", "packet_sha256")
    )
    required_shared_digests.extend(
        (
            construction.get("result_digest") if isinstance(construction, Mapping) else None,
            control.get("result_digest") if isinstance(control, Mapping) else None,
            control.get("control_pair_digest") if isinstance(control, Mapping) else None,
            task_cell.get("task_spec_digest") if isinstance(task_cell, Mapping) else None,
        )
    )
    if (
        not str(shared.get("scene_id") or "")
        or not str(shared.get("task_id") or "")
        or not isinstance(task_cell, Mapping)
        or shared.get("task_id") != task_cell.get("task_id")
        or not str(task_cell.get("cell_id") or "")
        or not str(task_cell.get("prompt") or "")
        or any(not _DIGEST.fullmatch(str(value or "")) for value in required_shared_digests)
        or any(
            not isinstance(record, Mapping)
            or not _DIGEST.fullmatch(str(record.get("sha256") or ""))
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or int(record.get("size_bytes") or 0) <= 0
            for record in records
        )
    ):
        errors.append("native_task_arena_policy_campaign_shared_projection_invalid")
    limits = shared.get("execution_limits") or {}
    if (
        not isinstance(limits, Mapping)
        or isinstance(limits.get("max_policy_queries"), bool)
        or not isinstance(limits.get("max_policy_queries"), int)
        or int(limits.get("max_policy_queries") or 0) <= 0
        or isinstance(limits.get("open_loop_horizon"), bool)
        or not isinstance(limits.get("open_loop_horizon"), int)
        or int(limits.get("open_loop_horizon") or 0) <= 0
        or not isinstance(limits.get("control_cadence"), Mapping)
        or limits.get("control_cadence_digest")
        != canonical_digest(limits.get("control_cadence") or {})
    ):
        errors.append("native_task_arena_policy_campaign_execution_limits_invalid")

    if members and any(
        row.get("maximum_hourly_rate_usd") != members[0].get("maximum_hourly_rate_usd")
        or row.get("hard_attempt_spend_cap_usd")
        != members[0].get("hard_attempt_spend_cap_usd")
        or row.get("maximum_single_resource_ttl_seconds")
        != members[0].get("maximum_single_resource_ttl_seconds")
        for row in members[1:]
        if isinstance(row, Mapping)
    ):
        errors.append("native_task_arena_policy_campaign_member_limits_asymmetric")

    member_cap = round(sum(caps), 6)
    projected = (
        round(float(prior_spend) + member_cap, 6) if _finite_nonnegative(prior_spend) else None
    )
    if (
        payload.get("maximum_campaign_spend_usd") != member_cap
        or payload.get("projected_aggregate_goal_spend_usd") != projected
        or projected is None
        or projected > AGGREGATE_GOAL_SPEND_CAP_USD
    ):
        errors.append("native_task_arena_policy_campaign_aggregate_spend_invalid")
    if payload.get("campaign_digest") != canonical_digest(payload, digest_field="campaign_digest"):
        errors.append("native_task_arena_policy_campaign_digest_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return payload


def materialize_native_task_arena_policy_campaign(
    *,
    campaign_id: str,
    blueprint_commit: str,
    pi05_bundle_receipt_path: str | Path,
    groot_bundle_receipt_path: str | Path,
    prior_authority_path: str | Path,
    prior_result_path: str | Path,
    prior_provider_zero_path: str | Path,
    prior_spend_reconciliation_path: str | Path,
    controls_allowed_active_instance_ids: Sequence[int],
    pi05_launch_id: str,
    pi05_resource_name: str,
    pi05_max_hourly_rate_usd: float,
    pi05_hard_cap_usd: float,
    pi05_hard_ttl_seconds: int,
    groot_launch_id: str,
    groot_resource_name: str,
    groot_max_hourly_rate_usd: float,
    groot_hard_cap_usd: float,
    groot_hard_ttl_seconds: int,
    output_path: str | Path,
    supplemental_prior_result_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Seal one exact two-member campaign before either authority is valid."""

    if not _COMMIT.fullmatch(str(blueprint_commit or "")):
        raise ValueError("native_task_arena_policy_campaign_commit_invalid")
    bundle_paths = {
        "pi05_droid": Path(pi05_bundle_receipt_path).expanduser().resolve(),
        "groot_n17_droid": Path(groot_bundle_receipt_path).expanduser().resolve(),
    }
    bundles = {
        candidate: _verified_policy_bundle(path, blueprint_commit=blueprint_commit)
        for candidate, path in bundle_paths.items()
    }
    if (
        any(bundle.get("policy_candidate_id") != candidate for candidate, bundle in bundles.items())
        or len({bundle.get("execution_mode") for bundle in bundles.values()}) != 1
    ):
        raise ValueError("native_task_arena_policy_campaign_bundle_pair_invalid")
    projections = {
        candidate: _shared_scientific_projection(bundle)
        for candidate, bundle in bundles.items()
    }
    if projections["pi05_droid"] != projections["groot_n17_droid"]:
        raise ValueError("native_task_arena_policy_campaign_shared_science_mismatch")

    prior = validate_terminal_spend_chain(
        authority_path=prior_authority_path,
        result_path=prior_result_path,
        provider_zero_path=prior_provider_zero_path,
    )
    prior_result_paths = (
        prior["records"]["terminal_result"]["path"],
        *(str(Path(item).expanduser().resolve()) for item in supplemental_prior_result_paths),
    )
    if len(prior_result_paths) != len(set(prior_result_paths)):
        raise ValueError("native_task_arena_policy_campaign_prior_result_duplicate")
    reconciled = bind_lane_prior_spend(
        prior_result_paths=prior_result_paths,
        reconciliation_path=prior_spend_reconciliation_path,
        lane="native_task_arena",
    )
    prior_spend = round(
        prior["aggregate_goal_spend_before_attempt_usd"] + reconciled["actual_total_usd"],
        6,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in controls_allowed_active_instance_ids
    ):
        raise ValueError("native_task_arena_policy_campaign_controls_allowlist_invalid")
    controls_ids = sorted(set(controls_allowed_active_instance_ids))

    configs = {
        "pi05_droid": {
            "launch_id": pi05_launch_id,
            "resource_name": pi05_resource_name,
            "maximum_hourly_rate_usd": pi05_max_hourly_rate_usd,
            "hard_attempt_spend_cap_usd": pi05_hard_cap_usd,
            "maximum_single_resource_ttl_seconds": pi05_hard_ttl_seconds,
        },
        "groot_n17_droid": {
            "launch_id": groot_launch_id,
            "resource_name": groot_resource_name,
            "maximum_hourly_rate_usd": groot_max_hourly_rate_usd,
            "hard_attempt_spend_cap_usd": groot_hard_cap_usd,
            "maximum_single_resource_ttl_seconds": groot_hard_ttl_seconds,
        },
    }
    members: list[dict[str, Any]] = []
    for candidate in MEMBER_IDS:
        bundle = bundles[candidate]
        members.append(
            {
                "member_id": candidate,
                "candidate_id": candidate,
                "blueprint_commit": blueprint_commit,
                "execution_mode": bundle["execution_mode"],
                "bundle_receipt": _record(bundle_paths[candidate]),
                "bundle_sha256": bundle["bundle_sha256"],
                "bundle_input_digest": bundle["input_digest"],
                **configs[candidate],
                "maximum_automatic_retries": 0,
                "maximum_provider_allocations": 1,
            }
        )
    maximum_campaign_spend = round(
        sum(float(row["hard_attempt_spend_cap_usd"]) for row in members), 6
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "campaign_id": campaign_id,
        "blueprint_commit": blueprint_commit,
        "execution_mode": members[0]["execution_mode"],
        "controls_allowed_active_instance_ids": controls_ids,
        "aggregate_goal_spend_cap_usd": AGGREGATE_GOAL_SPEND_CAP_USD,
        "prior_official_spend": {
            "aggregate_goal_spend_before_campaign_usd": prior_spend,
            "reconciled_actual_total_usd": reconciled["actual_total_usd"],
            "prior_terminal_attempt": {
                **prior["records"],
                "authority_digest": prior["authority_digest"],
            },
            "prior_terminal_attempts": reconciled["prior_terminal_attempts"],
            "reconciliation": reconciled["reconciliation"],
        },
        "shared_scientific_projection": projections["pi05_droid"],
        "members": members,
        "maximum_campaign_spend_usd": maximum_campaign_spend,
        "projected_aggregate_goal_spend_usd": round(prior_spend + maximum_campaign_spend, 6),
        "provider_wide_launch_lock_required": True,
        "independent_watchdog_required_per_member": True,
        "automatic_retry_authorized": False,
        "campaign_digest": "",
    }
    payload["campaign_digest"] = canonical_digest(payload, digest_field="campaign_digest")
    validated = validate_native_task_arena_policy_campaign(
        payload, expected_blueprint_commit=blueprint_commit
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("native_task_arena_policy_campaign_output_exists")
    ensure_dir(destination.parent)
    write_json(destination, validated)
    return validated


def load_verified_native_task_arena_policy_campaign(
    path: str | Path,
    *,
    expected_blueprint_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    campaign = validate_native_task_arena_policy_campaign(
        _read(source, code="native_task_arena_policy_campaign_invalid"),
        expected_blueprint_commit=expected_blueprint_commit,
    )
    return campaign, _record(source)


def verify_native_task_arena_policy_campaign_bundles(
    campaign: Mapping[str, Any], *, expected_blueprint_commit: str
) -> None:
    """Reopen both exact member receipts and bundle bytes."""

    for member in campaign["members"]:
        receipt_path = _bound_record(
            member.get("bundle_receipt"),
            code="native_task_arena_policy_campaign_bundle_unbound",
        )
        bundle = _verified_policy_bundle(
            receipt_path, blueprint_commit=expected_blueprint_commit
        )
        if (
            bundle.get("bundle_sha256") != member.get("bundle_sha256")
            or bundle.get("input_digest") != member.get("bundle_input_digest")
            or bundle.get("policy_candidate_id") != member.get("candidate_id")
            or bundle.get("execution_mode") != member.get("execution_mode")
        ):
            raise ValueError(
                "native_task_arena_policy_campaign_bundle_binding_invalid"
            )


def campaign_member(
    campaign: Mapping[str, Any], *, member_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_native_task_arena_policy_campaign(campaign)
    members = {
        str(row["member_id"]): dict(row) for row in validated["members"] if isinstance(row, Mapping)
    }
    member = members.get(member_id)
    sibling = next(
        (row for name, row in members.items() if name != member_id),
        None,
    )
    if member is None or sibling is None:
        raise ValueError("native_task_arena_policy_campaign_member_missing")
    return member, sibling


__all__ = [
    "MEMBER_IDS",
    "POLICY_EXECUTION_MODES",
    "SCHEMA_VERSION",
    "campaign_member",
    "load_verified_native_task_arena_policy_campaign",
    "materialize_native_task_arena_policy_campaign",
    "verify_native_task_arena_policy_campaign_bundles",
    "validate_native_task_arena_policy_campaign",
]
