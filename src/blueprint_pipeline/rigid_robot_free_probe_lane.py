"""Isolated canonical-allocator branch: one capped GPU session records one rigid scene's robot-free probe.

Beta journey step: L5 team policy evaluation. Owner approval 2026-10-26
(verbatim): "Yes: approve A. Under each capture's website grant, build the
scene and run the no-robot check automatically; make the first real run a
capped test."

The lane is switched off unless ``BLUEPRINT_RIGID_ROBOT_FREE_PROBE_ENABLED`` is
set, and every check below runs before any provider mutation:

- the provider is Vast; the run binds one sealed bundle
  (``rigid_robot_free_probe_bundle``) at the control plane's exact commit and
  one single-use authority staged by ``rigid_robot_free_probe_runner``;
- the bound request is still the one the capture's current team profile
  plans, against the task specification teams are scored with, and the
  scene's retained receipt is not decided (decided receipts are never
  replaced, so a run could not change anything);
- the rate is at most $0.80/h, the cap at most $1.00 and the hard TTL at most
  3600 s, with the rate over the TTL inside the cap; the Isaac 580-branch
  driver ceiling is in force; the launch gate, this lane's own watchdog name
  scope and the caller-exit survival contract pass the shared pre-spend
  preflight;
- spend authority is the capture's own website preparation grant: one
  ``gpu_render``/``vast`` reservation of the cap through the WebApp
  (``website_task_context.reserve_website_preparation_spend``) bound to this
  allocation's binding digest. A missing, expired or exhausted grant is a
  typed refusal and nothing is allocated; the run's watchdog deadline never
  outlives the grant.

The transport is the shared native Arena Vast path with the existing
``native_task_arena`` bundle kind: zero retry and one allocation, US-only
offers, an RT-capable GPU of at least 24 GB, the 580-branch driver floor and
ceiling, an independent hard-TTL watchdog armed before create on this lane's
instance-label prefix, then teardown and provider-zero readback. The worker
queries no policy (``candidate_policy_query_expected=False``).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_budget,
)
from .rigid_robot_free_probe_bundle import (
    PROBE_KIND,
    RigidRobotFreeProbeBundleError,
    current_probe_state,
    load_verified_rigid_robot_free_probe_bundle,
)
from .rigid_robot_free_probe_worker import INPUTS_FILENAME, RESULT_FILENAME, validate_probe_inputs

ENABLE_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_ENABLED"
PROGRAM_ID = "blueprint-free-beta-evaluated-pilot"
INSTANCE_LABEL_PREFIX = "blueprint-rigid-robot-free-probe-"
LANE = "rigid_robot_free_probe"
MAX_HOURLY_RATE_USD = 0.80
HARD_CAP_USD = 1.00
HARD_TTL_SECONDS = 3_600
MIN_HARD_TTL_SECONDS = 1_800
MIN_GPU_RAM_MB = 24_000
RESULT_SCHEMA_VERSION = "rigid_robot_free_probe_vast_run.v1"
AUTHORITY_SCHEMA = "blueprint.rigid_robot_free_probe_paid_authority.v1"
CONSUMPTION_SCHEMA = "blueprint.rigid_robot_free_probe_authority_consumption.v1"
CONSUMPTION_FILENAME = "rigid_robot_free_probe_authority_consumption.json"
FUNDING_FILENAME = "website_preparation_spend_admission.json"
# A staged authority must reach the allocator within this window.
AUTHORITY_WINDOW_SECONDS = 6 * 3600
# The result archive carries one trace and its wrapper.
RESULT_UPLOAD_BYTES = 200_000_000
WEBSITE_GRANT_REFUSALS = {
    "consent_expired": "rigid_robot_free_probe_website_grant_expired",
    "website_scene_preparation_budget_exhausted": "rigid_robot_free_probe_website_grant_exhausted",
    "website_scene_sponsorship_missing": "rigid_robot_free_probe_website_grant_missing",
}
_RESOURCE_NAME = re.compile(r"^blueprint-rigid-robot-free-probe-[0-9a-f]{32}$")
_WEBSITE_REFUSAL = re.compile(r"^website_control_preparation-spend_http_[0-9]{3}:([a-z][a-z0-9_]{0,99})$")


class RigidRobotFreeProbeLaneError(ValueError):
    """The probe's paid inputs do not bind one capped run."""


def lane_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """The explicit enable setting; anything but a true value keeps the lane switched off."""

    value = (os.environ if environ is None else environ).get(ENABLE_ENV, "")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_record_invalid")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_name_for(attempt_id: str) -> str:
    import hashlib

    return INSTANCE_LABEL_PREFIX + hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]


def build_paid_authority(*, attempt_id: str, inputs_path: Path, bundle_receipt_path: Path,
                         bundle: Mapping[str, Any], source_commit: str, now: float | None = None) -> dict[str, Any]:
    """The single-use authority for one attempt; it binds, the website grant pays."""

    inputs = validate_probe_inputs(_load(inputs_path))
    issued = float(time.time() if now is None else now)
    value = {
        "schema_version": AUTHORITY_SCHEMA, "program_id": PROGRAM_ID, "probe_kind": PROBE_KIND,
        "attempt_id": attempt_id, "capture_root": inputs["capture_root"],
        "robot_free_probe_root": inputs["robot_free_probe_root"],
        "scene_revision_digest": inputs["scene_revision_digest"],
        "probe_request_digest": inputs["probe_request"]["request_digest"],
        "task_spec_digest": inputs["probe_request"]["binding"]["task_spec_digest"],
        "inputs_path": str(Path(inputs_path).resolve()), "inputs_sha256": _sha256(inputs_path),
        "inputs_digest": inputs["inputs_digest"], "bundle_receipt_path": str(Path(bundle_receipt_path).resolve()),
        "bundle_sha256": bundle["bundle_sha256"], "implementation_commit": source_commit,
        "resource_name": resource_name_for(attempt_id), "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
        "hard_cap_usd": HARD_CAP_USD, "hard_ttl_seconds": HARD_TTL_SECONDS, "maximum_provider_allocations": 1,
        "retry_cap": 0, "spend_authority": "website_preparation_grant", "candidate_policy_queried": False,
        "issued_at_epoch": issued, "expires_at_epoch": issued + AUTHORITY_WINDOW_SECONDS, "authority_digest": "",
    }
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


def validate_paid_authority(value: Mapping[str, Any], *, bundle: Mapping[str, Any],
                            now: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The authority and the probe input it binds, or raise with a typed reason."""

    current = float(time.time() if now is None else now)
    if (value.get("schema_version") != AUTHORITY_SCHEMA or value.get("program_id") != PROGRAM_ID
            or value.get("probe_kind") != PROBE_KIND
            or value.get("authority_digest") != canonical_digest(dict(value), digest_field="authority_digest")):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_invalid")
    if not isinstance(value.get("resource_name"), str) or not _RESOURCE_NAME.fullmatch(value["resource_name"]) \
            or value["resource_name"] != resource_name_for(str(value.get("attempt_id") or "")):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_resource_name_invalid")
    if (value.get("retry_cap") != 0 or value.get("maximum_provider_allocations") != 1
            or value.get("max_hourly_rate_usd") != MAX_HOURLY_RATE_USD or value.get("hard_cap_usd") != HARD_CAP_USD
            or value.get("hard_ttl_seconds") != HARD_TTL_SECONDS):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_bounds_invalid")
    expires = value.get("expires_at_epoch")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not current < expires:
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_expired")
    if value.get("bundle_sha256") != bundle.get("bundle_sha256") or value.get("implementation_commit") != bundle.get(
            "implementation_commit"):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_bundle_mismatch")
    inputs_path = Path(str(value.get("inputs_path") or ""))
    rows = [row for row in bundle.get("bound_runtime_inputs") or [] if isinstance(row, Mapping)
            and row.get("relative_path") == f"runtime_inputs/{INPUTS_FILENAME}"]
    if (inputs_path.is_symlink() or not inputs_path.is_file() or _sha256(inputs_path) != value.get("inputs_sha256")
            or len(rows) != 1 or rows[0].get("sha256") != value.get("inputs_sha256")):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_inputs_mismatch")
    inputs = validate_probe_inputs(_load(inputs_path))
    if any(inputs.get(key) != value.get(key) for key in (
            "capture_root", "robot_free_probe_root", "scene_revision_digest", "inputs_digest")) \
            or inputs["probe_request"]["request_digest"] != value.get("probe_request_digest"):
        raise RigidRobotFreeProbeLaneError("rigid_robot_free_probe_authority_inputs_mismatch")
    return dict(value), inputs


def budget_blockers(*, max_hourly_rate_usd: Any, hard_cap_usd: Any, hard_ttl_seconds: Any) -> list[str]:
    """The lane's fixed ceilings: $0.80/h, $1.00 and 3600 s, with the rate over the TTL inside the cap."""

    numbers = (max_hourly_rate_usd, hard_cap_usd)
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
           for item in numbers):
        return ["rigid_robot_free_probe_budget_invalid"]
    blockers = []
    if float(max_hourly_rate_usd) > MAX_HOURLY_RATE_USD:
        blockers.append("rigid_robot_free_probe_hourly_rate_above_ceiling")
    if float(hard_cap_usd) > HARD_CAP_USD:
        blockers.append("rigid_robot_free_probe_spend_cap_above_ceiling")
    if (isinstance(hard_ttl_seconds, bool) or not isinstance(hard_ttl_seconds, int)
            or not MIN_HARD_TTL_SECONDS <= hard_ttl_seconds <= HARD_TTL_SECONDS):
        blockers.append("rigid_robot_free_probe_hard_ttl_invalid")
    elif float(max_hourly_rate_usd) * hard_ttl_seconds / 3600 > float(hard_cap_usd):
        blockers.append("rigid_robot_free_probe_runtime_cost_exceeds_cap")
    return blockers


def driver_ceiling_blockers() -> list[str]:
    """The Isaac RTX driver ceiling (580 branch) must be in force: no override may raise or disable it."""

    from .isaac_driver_support import ISAAC_MAX_SUPPORTED_DRIVER_EXCLUSIVE, resolve_max_isaac_driver

    ceiling = resolve_max_isaac_driver()
    if ceiling is None or tuple(ceiling) > tuple(ISAAC_MAX_SUPPORTED_DRIVER_EXCLUSIVE):
        return ["rigid_robot_free_probe_isaac_driver_ceiling_not_enforced"]
    return []


def retained_website_grant(capture_root: str | Path, *, now: float | None = None
                           ) -> tuple[dict[str, Any] | None, list[str]]:
    """The capture's retained website task context, or the typed reason its grant cannot fund a run.

    The local read is a precheck; the WebApp's reservation remains the spend authority.
    """

    from .website_task_context_contracts import validate_website_task_context

    current = float(time.time() if now is None else now)
    pipeline = Path(capture_root) / "pipeline"
    paths = (pipeline / "website_task_context.json", pipeline / "website_scene_sponsorship.json")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return None, ["rigid_robot_free_probe_website_grant_missing"]
    try:
        context, sponsorship = _load(paths[0]), _load(paths[1])
        context = validate_website_task_context(
            context, request_id=str(context.get("request_id") or ""), scene_id=str(context.get("scene_id") or ""),
            capture_id=str(context.get("capture_id") or ""), purpose="scene_preparation")
    except (OSError, ValueError) as exc:
        return None, [f"rigid_robot_free_probe_website_grant_invalid:{str(exc)[:120]}"]
    expires = sponsorship.get("expires_at_epoch")
    if Path(capture_root).name != context.get("capture_id"):
        return None, ["rigid_robot_free_probe_website_grant_invalid:capture_binding"]
    if (sponsorship.get("schema_version") != "website_scene_sponsorship.v1"
            or sponsorship.get("authority_digest") != canonical_digest(sponsorship, digest_field="authority_digest")
            or any(sponsorship.get(key) != context.get(key) for key in ("request_id", "scene_id", "capture_id"))
            or sponsorship.get("task_context_digest") != context.get("context_digest")
            or isinstance(expires, bool) or not isinstance(expires, (int, float))):
        return None, ["rigid_robot_free_probe_website_grant_invalid:sponsorship_binding"]
    if expires - current < MIN_HARD_TTL_SECONDS:
        # Expired, or ending before the shortest run could finish.
        return None, ["rigid_robot_free_probe_website_grant_expired"]
    return context, []


def _grant_refusal(message: str) -> str:
    match = _WEBSITE_REFUSAL.fullmatch(message)
    if match is None:
        return "rigid_robot_free_probe_website_grant_unavailable:" + message[:120]
    return WEBSITE_GRANT_REFUSALS.get(match.group(1), "rigid_robot_free_probe_website_grant_refused:" + match.group(1))


def reserve_probe_funding(*, task_context: Mapping[str, Any], binding_digest: str, hard_cap_usd: float,
                          job_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Reserve the cap once from the capture's website preparation grant, or say why not."""

    from .website_task_context import reserve_website_preparation_spend

    path = Path(job_dir) / FUNDING_FILENAME
    if path.exists() or path.is_symlink():
        return None, ["rigid_robot_free_probe_website_grant_already_reserved"]

    def retain(value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")

    try:
        admission, _grant = reserve_website_preparation_spend(
            task_context=task_context, binding_digest=binding_digest, maximum_cost_usd=float(hard_cap_usd),
            request_count=1, resource_class="gpu_render", provider="vast", retain_admission=retain)
    except PaidResourceAdmissionBlocked as exc:
        if "paid_resource_admission_not_admitted" in exc.blockers:
            # The WebApp already reserved this exact binding: one binding is never paid twice.
            return None, ["rigid_robot_free_probe_website_grant_already_reserved"]
        return None, ["rigid_robot_free_probe_paid_budget_blocked", *exc.blockers]
    except (OSError, RuntimeError, ValueError) as exc:
        return None, [_grant_refusal(str(exc))]
    if not path.is_file():
        return None, ["rigid_robot_free_probe_website_grant_admission_not_retained"]
    return dict(admission), []


def consume_authority_once(authority: Mapping[str, Any], *, job_dir: Path) -> dict[str, Any]:
    """Record the authority's single use; a second use is refused (zero retry)."""

    path = Path(job_dir) / CONSUMPTION_FILENAME
    value = {"schema_version": CONSUMPTION_SCHEMA, "status": "consumed",
             "authority_digest": authority["authority_digest"], "resource_name": authority["resource_name"],
             "consumed_at": utc_now_iso(), "consumption_digest": ""}
    value["consumption_digest"] = canonical_digest(value, digest_field="consumption_digest")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        return {"status": "blocked", "blockers": ["rigid_robot_free_probe_authority_already_consumed"]}
    return value


def launch_environment_blockers(*, job_dir: Path, resource_name: str, container_image: str,
                                max_hourly_rate_usd: float, hard_cap_usd: float, hard_ttl_seconds: int) -> list[str]:
    """The deterministic live-launch gates, the same in dry and execute modes (and the queue's precheck)."""

    from .adp_isaac_lab_arena_vast import (
        _bounded_spend_gate_open,
        _remaining_session_live_minutes,
        _vast_credential_file_present,
    )
    from .paid_lane_guard import (
        SPEND_ADMISSION_LOCK_PATH_ENV,
        PreSpendPreflightBlocked,
        image_contract_from_ref,
        require_pre_spend_preflight,
    )
    from .vast_independent_watchdog_control import (
        _caller_exit_survival_contract,
        _caller_exit_survival_proven,
        validate_independent_vast_watchdog_names,
    )
    from .vast_launch_slots import inspect_vast_launch_admission

    blockers = list(inspect_vast_launch_admission()["blockers"])
    try:
        validate_independent_vast_watchdog_names(pod_name_prefix=INSTANCE_LABEL_PREFIX,
                                                 resource_name_exact=resource_name)
    except ValueError as exc:
        blockers.append(str(exc))
    if not _caller_exit_survival_proven(_caller_exit_survival_contract()):
        blockers.append("independent_vast_watchdog_caller_exit_survival_unproven")
    minutes = _remaining_session_live_minutes(job=Path(job_dir), hard_cap_usd=hard_cap_usd,
                                              hard_ttl_seconds=hard_ttl_seconds,
                                              max_hourly_rate_usd=max_hourly_rate_usd)
    if minutes < 30:
        blockers.append("rigid_robot_free_probe_budget_below_minimum_live_window")
    credential_present = _vast_credential_file_present()
    try:
        require_pre_spend_preflight(
            lane=LANE, provider="vast", credential_present=credential_present,
            capacity_evidence={"available": credential_present,
                               "detail": "credential_bound; provider adapter rechecks inventory and offers"},
            image_contract=image_contract_from_ref(container_image),
            runtime_contract={"startup_marker": "vast_instance_started_or_blocked",
                              "progress_marker": "vast_provider_bundle_progress",
                              "startup_timeout_seconds": minutes * 60, "no_progress_timeout_seconds": 1800},
            spend_gate_open=_bounded_spend_gate_open(max_hourly_rate_usd=max_hourly_rate_usd,
                                                     hard_cap_usd=hard_cap_usd, remaining_live_minutes=minutes),
            record_dir=Path(job_dir) / "launch_preflight",
            spend_admission_lock=None if str(os.getenv(SPEND_ADMISSION_LOCK_PATH_ENV) or "").strip() else {},
        )
    except PreSpendPreflightBlocked as exc:
        blockers.extend(str(value) for value in exc.preflight.get("blockers", []))
    return sorted(set(blockers))


def scene_blockers(authority: Mapping[str, Any]) -> list[str]:
    """The authority's request is still the current one and the scene's receipt is still undecided."""

    state = current_probe_state(authority["capture_root"])
    if state["status"] == "receipt_decided":
        return ["rigid_robot_free_probe_receipt_already_decided"]
    if state["status"] != "probe_required":
        return ["rigid_robot_free_probe_scene_" + str(state["status"])]
    blockers = []
    if state["request"]["request_digest"] != authority["probe_request_digest"]:
        blockers.append("rigid_robot_free_probe_request_stale")
    if state["robot_free_probe_root"] != authority["robot_free_probe_root"]:
        blockers.append("rigid_robot_free_probe_receipt_root_mismatch")
    return blockers


def _blocked(blockers: list[str], **fields: Any) -> dict[str, Any]:
    return {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked", "provider_mutations_performed": 0,
            "retry_cap": 0, "candidate_policy_queried": False, "blockers": sorted(set(blockers)), **fields}


def run_rigid_robot_free_probe_allocator_lane(args: Any, control_context: tuple[list[str], Mapping[str, Any]]
                                              ) -> int:
    """Admit, fund and run one capped probe session, or refuse before any provider mutation."""

    from .native_task_arena_vast import (
        DEFAULT_KEY_PREFIX,
        MINIMUM_DRIVER_VERSION,
        NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES,
        NO_POLICY_PREFERRED_GPU_KEYWORDS,
    )

    control_blockers, control_identity = control_context
    if not getattr(args, "admission_out", None) or not getattr(args, "adapter_output", None):
        # Without its two receipts the lane cannot record a refusal, so it refuses without one.
        print(json.dumps({"success": False, "blockers": ["rigid_robot_free_probe_receipt_paths_missing"]},
                         sort_keys=True))
        return 2
    blockers = list(control_blockers)
    if not lane_enabled():
        blockers.append("rigid_robot_free_probe_lane_disabled")
    if args.provider != "vast":
        blockers.append("rigid_robot_free_probe_provider_must_be_vast")
    for name in ("adp_job_dir", "native_task_arena_bundle_receipt", "native_task_arena_attempt_authority"):
        if not getattr(args, name, None):
            blockers.append(f"rigid_robot_free_probe_{name}_missing")
    if any(int(value) > 0 for value in getattr(args, "adp_allowed_active_vast_instance_id", None) or ()):
        blockers.append("rigid_robot_free_probe_active_instance_allowlist_unexpected")
    blockers.extend(budget_blockers(max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
                                    hard_cap_usd=args.adp_max_spend_usd, hard_ttl_seconds=args.adp_hard_ttl_seconds))
    blockers.extend(driver_ceiling_blockers())
    bundle: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    if args.native_task_arena_bundle_receipt and args.native_task_arena_attempt_authority:
        try:
            bundle = load_verified_rigid_robot_free_probe_bundle(
                args.native_task_arena_bundle_receipt,
                expected_implementation_commit=str(control_identity.get("orchestrator_source_commit") or ""))
            authority, _inputs = validate_paid_authority(_load(args.native_task_arena_attempt_authority),
                                                         bundle=bundle)
        except (OSError, RigidRobotFreeProbeBundleError, RigidRobotFreeProbeLaneError, ValueError) as exc:
            blockers.extend([part for part in str(exc).split(";") if part]
                            or ["rigid_robot_free_probe_contract_invalid"])
            bundle = authority = None
    if authority is not None:
        if (float(authority["hard_cap_usd"]) != float(args.adp_max_spend_usd)
                or int(authority["hard_ttl_seconds"]) != int(args.adp_hard_ttl_seconds)
                or float(authority["max_hourly_rate_usd"]) != float(args.adp_max_hourly_rate_usd)):
            blockers.append("rigid_robot_free_probe_resource_bounds_mismatch")
        if getattr(args, "pod_name", None) not in (None, "", authority["resource_name"]):
            blockers.append("rigid_robot_free_probe_resource_name_mismatch")
        if args.adp_job_dir and (Path(args.adp_job_dir) / CONSUMPTION_FILENAME).exists():
            # Checked before any grant is reserved; the exclusive write below closes the race.
            blockers.append("rigid_robot_free_probe_authority_already_consumed")
    task_context = None
    if not blockers and authority is not None and bundle is not None:
        blockers.extend(scene_blockers(authority))
        task_context, grant_blockers = retained_website_grant(authority["capture_root"])
        blockers.extend(grant_blockers)
        blockers.extend(launch_environment_blockers(
            job_dir=Path(args.adp_job_dir), resource_name=authority["resource_name"],
            container_image=str(bundle["container_image"]), max_hourly_rate_usd=args.adp_max_hourly_rate_usd,
            hard_cap_usd=args.adp_max_spend_usd, hard_ttl_seconds=args.adp_hard_ttl_seconds))
    binding = {
        "program_id": PROGRAM_ID, "probe_kind": PROBE_KIND,
        "orchestrator_source_commit": control_identity.get("orchestrator_source_commit"),
        "run_kind": "rigid_robot_free_probe", "claim_ceiling": "scene_robot_free_readiness_evidence",
        "bundle_sha256": bundle.get("bundle_sha256") if bundle else None,
        "authority_digest": authority.get("authority_digest") if authority else None,
        "probe_request_digest": authority.get("probe_request_digest") if authority else None,
        "resource_name": authority.get("resource_name") if authority else None,
        "capture_root": authority.get("capture_root") if authority else None,
        "candidate_ids": [], "learned_policy_rollout_count": 0, "maximum_provider_allocations": 1, "retry_cap": 0,
        "max_hourly_rate_usd": args.adp_max_hourly_rate_usd, "hard_cap_usd": args.adp_max_spend_usd,
        "hard_ttl_seconds": args.adp_hard_ttl_seconds,
    }
    binding_digest = canonical_digest(binding)
    funding: dict[str, Any] | None = None
    if args.execute and not blockers and task_context is not None:
        try:
            # A refusal the local budget monitor already knows never leaves a remote hold behind.
            require_paid_resource_budget(resource_class="vast_provider_adapter")
        except PaidResourceAdmissionBlocked as exc:
            blockers.extend(exc.blockers)
        else:
            funding, funding_blockers = reserve_probe_funding(
                task_context=task_context, binding_digest=binding_digest, hard_cap_usd=args.adp_max_spend_usd,
                job_dir=Path(args.adp_job_dir))
            blockers.extend(funding_blockers)
    admission = build_paid_lane_admission(resource_class="vast_provider_adapter", blockers=blockers)
    admission.update({
        "program_id": PROGRAM_ID, "probe_kind": PROBE_KIND, "control_plane_identity": dict(control_identity),
        "max_hourly_rate_usd": args.adp_max_hourly_rate_usd, "hard_cap_usd": args.adp_max_spend_usd,
        "hard_ttl_seconds": args.adp_hard_ttl_seconds, "retry_cap": 0, "maximum_provider_allocations": 1,
        "candidate_policy_queried": False, "spend_authority": "website_preparation_grant",
        "website_preparation_admission_digest": canonical_digest(funding) if funding else None,
        "allocation_binding": binding, "allocation_binding_digest": binding_digest,
    })
    write_json(Path(args.admission_out), admission)
    grant = None
    if args.execute:
        try:
            grant = require_paid_resource_admission(admission, resource_class="vast_provider_adapter",
                                                    expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
        except PaidResourceAdmissionBlocked as exc:
            write_json(Path(args.adapter_output), _blocked([*blockers, *exc.blockers]))
            print(json.dumps({"success": False}, sort_keys=True))
            return 2
    if blockers or bundle is None or authority is None:
        result = _blocked(blockers)
    else:
        consumption = consume_authority_once(authority, job_dir=Path(args.adp_job_dir)) if args.execute else None
        if consumption is not None and consumption.get("status") != "consumed":
            result = _blocked(list(consumption.get("blockers") or []), authorization_consumption=consumption)
        else:
            from .adp_isaac_lab_arena_vast import run_arena_native_control_vast

            source = bundle.get("runtime_source_packet") or {}
            result = run_arena_native_control_vast(
                approval_path=".", job_dir=args.adp_job_dir, paid_resource_admission_grant=grant,
                execute=args.execute, prepared_bundle=bundle, machine_avoidlist_path=args.adp_machine_avoidlist,
                max_hourly_rate_usd=args.adp_max_hourly_rate_usd, hard_cap_usd=args.adp_max_spend_usd,
                hard_ttl_seconds=args.adp_hard_ttl_seconds, expected_output_filename=RESULT_FILENAME,
                container_image=str(bundle["container_image"]), provider_bundle_kind="native_task_arena",
                result_schema_version=RESULT_SCHEMA_VERSION,
                object_store_key_prefix=f"{DEFAULT_KEY_PREFIX}/rigid-robot-free-probe",
                instance_label_prefix=INSTANCE_LABEL_PREFIX, instance_label_exact=authority["resource_name"],
                blocker_prefix=LANE, min_gpu_ram_mb=MIN_GPU_RAM_MB, allowed_active_instance_ids=(),
                vast_launch_lock_file=None, candidate_policy_query_expected=False,
                preferred_gpu_keywords=NO_POLICY_PREFERRED_GPU_KEYWORDS, minimum_driver_version=MINIMUM_DRIVER_VERSION,
                require_known_supported_isaac_driver=True, require_independent_watchdog=True,
                hard_deadline_epoch=float(funding["expires_at_epoch"]) if funding else None,
                authorization_consumption=consumption, stale_offer_create_retry_limit=0,
                expected_provider_download_bytes=int(bundle["bundle_size_bytes"])
                + int(source.get("packet_size_bytes") or 0),
                expected_provider_upload_bytes=RESULT_UPLOAD_BYTES,
                allowed_geolocation_country_codes=NATIVE_TASK_ARENA_ALLOWED_GEOLOCATION_COUNTRY_CODES,
            )
    write_json(Path(args.adapter_output), result)
    success = result.get("status") in {"dry_run_ready", "completed"}
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


__all__ = [
    "AUTHORITY_SCHEMA",
    "ENABLE_ENV",
    "FUNDING_FILENAME",
    "HARD_CAP_USD",
    "HARD_TTL_SECONDS",
    "INSTANCE_LABEL_PREFIX",
    "MAX_HOURLY_RATE_USD",
    "PROBE_KIND",
    "RigidRobotFreeProbeLaneError",
    "budget_blockers",
    "build_paid_authority",
    "consume_authority_once",
    "driver_ceiling_blockers",
    "lane_enabled",
    "launch_environment_blockers",
    "reserve_probe_funding",
    "retained_website_grant",
    "run_rigid_robot_free_probe_allocator_lane",
    "scene_blockers",
    "validate_paid_authority",
]
