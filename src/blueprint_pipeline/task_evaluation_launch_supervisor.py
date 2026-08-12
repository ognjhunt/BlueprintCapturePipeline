"""Optional read-only Agents SDK supervisor for production launch operations.

The deterministic state machine remains authoritative.  This module exposes no
tools and cannot invoke the allocator, mutate provider state, retry a launch,
grant rights, approve spend, or close teardown.  It may only explain the
observed state, recommend one deterministically admissible Pipeline profile,
or formulate one human decision request.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .task_evaluation_launch_dispatcher import (
    SECRET_PROFILE_ID_ENV,
    TaskEvaluationLaunchError,
    canonical_digest,
    load_public_launch_profile_catalog,
    validate_launch_profile,
)
from .task_evaluation_supervisor.agents_sdk import (
    DEFAULT_SUPERVISOR_AGENT_MODEL,
    AgentsSDKAgentSpec,
    AgentsSDKInvocationBlocked,
    AgentsSDKInvoker,
    OpenAIAgentsSDKConfig,
    OpenAIAgentsSDKInvoker,
)
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


SUPERVISOR_SCHEMA_VERSION = "task_evaluation_launch_supervision.v1"
SUPERVISOR_ENABLED_ENV = "BLUEPRINT_TASK_EVALUATION_AGENT_SUPERVISOR_ENABLED"
SUPERVISOR_MODEL_ENV = "BLUEPRINT_TASK_EVALUATION_AGENT_SUPERVISOR_MODEL"
SUPERVISOR_BUDGET_ENV = "BLUEPRINT_TASK_EVALUATION_AGENT_SUPERVISOR_BUDGET_USD"

# The production supervisor reserves its entire worst-case response before an
# SDK call.  Keep its immutable observation bounded so a growing receipt
# history cannot silently turn a configured inference ceiling into a permanent
# advisory outage.  At the current 2,000-token / USD 0.10 deployment envelope,
# 24 KiB leaves a conservative reserve for the output response.
DEFAULT_SUPERVISOR_SNAPSHOT_MAX_BYTES = 24_000
SUPERVISOR_SNAPSHOT_INPUT_CEILING_BLOCKER = "launch_supervisor_snapshot_input_ceiling_exceeded"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _snapshot_with_digest(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied snapshot with the stable advisory-cache digest."""

    value = json.loads(json.dumps(snapshot))
    digest_basis = json.loads(json.dumps(value))
    guard = digest_basis.get("guard")
    if isinstance(guard, Mapping):
        guard.pop("generated_at", None)
    value["snapshot_digest"] = canonical_digest(
        digest_basis,
        digest_field="snapshot_digest",
    )
    return value


def _snapshot_size_bytes(snapshot: Mapping[str, Any]) -> int:
    """Match the exact JSON encoding passed to the production SDK invoker."""

    return len(json.dumps(snapshot, sort_keys=True).encode("utf-8"))


class LaunchSupervisorRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal[
        "recommend_profile", "explain_blocker", "request_human_decision", "no_action"
    ]
    summary: str = Field(min_length=1, max_length=4000)
    recommended_profile_id: str | None = Field(default=None, max_length=192)
    cited_blockers: list[str] = Field(default_factory=list, max_length=40)
    human_decision_required: bool = False
    human_decision_prompt: str | None = Field(default=None, max_length=2000)
    smallest_missing_input: str | None = Field(default=None, max_length=1000)
    provider_mutation_requested: Literal[False] = False
    automatic_retry_requested: Literal[False] = False
    authority_granted: Literal[False] = False

    @model_validator(mode="after")
    def validate_disposition(self) -> "LaunchSupervisorRecommendation":
        if self.disposition == "recommend_profile" and not self.recommended_profile_id:
            raise ValueError("recommended_profile_id_required")
        if self.disposition != "recommend_profile" and self.recommended_profile_id is not None:
            raise ValueError("recommended_profile_id_forbidden")
        if self.disposition == "request_human_decision":
            if not self.human_decision_required or not self.human_decision_prompt:
                raise ValueError("human_decision_prompt_required")
        elif self.human_decision_required:
            raise ValueError("human_decision_disposition_required")
        return self


def build_supervisor_snapshot(
    *,
    profile_dir: str | Path,
    queue_root: str | Path,
    state_root: str | Path,
    guard_report_path: str | Path,
    public_catalog_path: str | Path | None = None,
    max_snapshot_bytes: int = DEFAULT_SUPERVISOR_SNAPSHOT_MAX_BYTES,
) -> dict[str, Any]:
    if not isinstance(max_snapshot_bytes, int) or max_snapshot_bytes <= 0:
        raise ValueError("launch_supervisor_snapshot_byte_ceiling_invalid")
    profiles: list[dict[str, Any]] = []
    guard_path = Path(guard_report_path).expanduser().resolve()
    guard = _read(guard_path) if guard_path.is_file() else {}
    raw_guard_blockers = guard.get("blockers")
    if not guard:
        guard_blockers = ["gpu_spend_guard_report_unavailable"]
    elif not isinstance(raw_guard_blockers, list) or any(
        not isinstance(item, str) or not item for item in raw_guard_blockers
    ):
        guard_blockers = ["gpu_spend_guard_report_blockers_invalid"]
    else:
        guard_blockers = list(raw_guard_blockers)
    inventories = {
        str(row.get("provider")): row
        for row in guard.get("inventory_results") or []
        if isinstance(row, Mapping)
    }
    spend_admission = guard.get("spend_admission_lock")
    spend_admission = spend_admission if isinstance(spend_admission, Mapping) else {}
    published_profile_keys: set[tuple[str, str]] | None = None
    published_descriptors: dict[tuple[str, str], dict[str, Any]] = {}
    profile_catalog: dict[str, Any] = {
        "status": "not_configured",
        "published_profile_count": None,
        "blockers": [],
    }
    if public_catalog_path is not None:
        try:
            catalog = load_public_launch_profile_catalog(public_catalog_path)
        except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError):
            published_profile_keys = set()
            profile_catalog = {
                "status": "blocked",
                "published_profile_count": 0,
                "blockers": ["launch_profile_public_catalog_invalid"],
            }
        else:
            published_descriptors = {
                (str(row["profile_id"]), str(row["profile_digest"])): dict(row)
                for row in catalog["profiles"]
            }
            published_profile_keys = set(published_descriptors)
            profile_catalog = {
                "status": "verified",
                "published_profile_count": len(published_descriptors),
                "blockers": [],
            }

    materialized_profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(Path(profile_dir).expanduser().resolve().glob("*.json")):
        try:
            profile = _read(path)
        except (OSError, json.JSONDecodeError):
            continue
        key = (str(profile.get("profile_id") or ""), str(profile.get("profile_digest") or ""))
        if published_profile_keys is not None and key not in published_profile_keys:
            continue
        materialized_profiles.setdefault(key, profile)

    for profile in materialized_profiles.values():
        blockers = validate_launch_profile(profile)
        execution_admission = profile.get("execution_admission")
        execution_admission = (
            execution_admission if isinstance(execution_admission, Mapping) else {}
        )
        if execution_admission.get("live_enabled") is not True:
            blockers.append("launch_profile_live_execution_disabled")
        controls = profile.get("required_controls")
        controls = controls if isinstance(controls, Mapping) else {}
        if str(os.getenv(SECRET_PROFILE_ID_ENV) or "").strip() != str(
            controls.get("secret_profile_id") or ""
        ):
            blockers.append("canonical_secret_profile_mismatch")
        reconciliation = profile.get("reconciliation")
        reconciliation = reconciliation if isinstance(reconciliation, Mapping) else {}
        for provider in reconciliation.get("required_providers") or []:
            inventory = inventories.get(str(provider))
            if not inventory or inventory.get("status") != "succeeded":
                blockers.append(f"gpu_inventory_not_confirmed:{provider}")
        if guard.get("live_instance_count") not in (0, None):
            blockers.append("gpu_fleet_not_zero_before_launch")
        if guard.get("provider_zero_verified") is False:
            blockers.append("gpu_provider_zero_not_verified")
        if spend_admission.get("admission_allowed") is not True:
            blockers.append("paid_spend_admission_not_open")
        blockers.extend(guard_blockers)
        allocator = profile.get("allocator")
        allocator = allocator if isinstance(allocator, Mapping) else {}
        profiles.append(
            {
                "profile_id": profile.get("profile_id"),
                "profile_digest": profile.get("profile_digest"),
                "source_kind": (profile.get("source_bundle") or {}).get("source_kind")
                if isinstance(profile.get("source_bundle"), Mapping)
                else None,
                "claim_ceiling": profile.get("claim_ceiling"),
                "max_spend_usd": allocator.get("max_spend_usd"),
                "hard_ttl_seconds": allocator.get("hard_ttl_seconds"),
                "required_providers": reconciliation.get("required_providers") or [],
                "live_enabled": execution_admission.get("live_enabled") is True,
                "readiness_blockers": execution_admission.get("blockers") or [],
                "admissible": not blockers,
                "blockers": sorted(set(blockers)),
            }
        )

    # The website can select only entries in the publisher-generated catalog.
    # A descriptor that is published but not materialized locally must remain a
    # visible, typed blocker rather than disappearing into a wider historical
    # profile directory or being recommended by the advisory agent.
    for key, descriptor in published_descriptors.items():
        if key in materialized_profiles:
            continue
        execution_admission = descriptor.get("execution_admission")
        execution_admission = (
            execution_admission if isinstance(execution_admission, Mapping) else {}
        )
        authorization = descriptor.get("required_authorization")
        authorization = authorization if isinstance(authorization, Mapping) else {}
        source_bundle = descriptor.get("source_bundle")
        source_bundle = source_bundle if isinstance(source_bundle, Mapping) else {}
        profiles.append(
            {
                "profile_id": descriptor.get("profile_id"),
                "profile_digest": descriptor.get("profile_digest"),
                "source_kind": source_bundle.get("source_kind"),
                "claim_ceiling": descriptor.get("claim_ceiling"),
                "max_spend_usd": authorization.get("max_spend_usd"),
                "hard_ttl_seconds": authorization.get("hard_ttl_seconds"),
                "required_providers": [],
                "live_enabled": execution_admission.get("live_enabled") is True,
                "readiness_blockers": execution_admission.get("blockers") or [],
                "admissible": False,
                "blockers": ["published_profile_not_materialized"],
            }
        )
    profiles.sort(key=lambda row: (str(row["profile_id"]), str(row["profile_digest"])))

    queue = Path(queue_root).expanduser().resolve()
    queue_counts = {
        name: len(list((queue / name).glob("*.json")))
        for name in ("pending", "processing", "completed", "blocked")
    }
    terminal_rows: list[dict[str, Any]] = []
    for path in sorted(Path(state_root).expanduser().resolve().glob("*/launch_receipt.json")):
        try:
            receipt = _read(path)
        except (OSError, json.JSONDecodeError):
            continue
        terminal_row = {
            "launch_id": receipt.get("launch_id"),
            "status": receipt.get("status"),
            "request_digest": receipt.get("request_digest"),
            "blockers": receipt.get("blockers") or [],
            "provider_mutation_attempted": receipt.get("provider_mutation_attempted") is True,
            "terminal_evidence_status": (receipt.get("terminal_evidence") or {}).get("status")
            if isinstance(receipt.get("terminal_evidence"), Mapping)
            else None,
        }
        terminal_unmatched_path = path.with_name("webapp_sync_terminal_unmatched.json")
        if terminal_unmatched_path.is_file():
            try:
                terminal_unmatched = _read(terminal_unmatched_path)
            except (OSError, json.JSONDecodeError):
                terminal_row["webapp_sync_status"] = "terminal_unmatched_invalid"
                terminal_row["webapp_sync_blockers"] = [
                    "webapp_sync_terminal_unmatched_invalid"
                ]
            else:
                terminal_row["webapp_sync_status"] = terminal_unmatched.get("status")
                terminal_row["webapp_record_bound"] = (
                    terminal_unmatched.get("webapp_record_bound") is True
                )
                terminal_row["website_trigger_proven"] = (
                    terminal_unmatched.get("website_trigger_proven") is True
                )
                terminal_row["webapp_sync_blockers"] = terminal_unmatched.get("blockers") or []
        terminal_rows.append(terminal_row)
    included_terminal_rows = list(terminal_rows)
    omitted_terminal_rows: list[dict[str, Any]] = []

    def snapshot_for_window(*, input_ceiling_exceeded: bool = False) -> dict[str, Any]:
        terminal_history: dict[str, Any] = {
            "selection": "lexicographically_latest_launch_receipts",
            "total_count": len(terminal_rows),
            "included_count": len(included_terminal_rows),
            "omitted_count": len(omitted_terminal_rows),
            "omitted_terminal_rows_digest": (
                canonical_digest(
                    {"terminal_launches": omitted_terminal_rows},
                    digest_field="omitted_terminal_rows_digest",
                )
                if omitted_terminal_rows
                else None
            ),
            "input_byte_ceiling": max_snapshot_bytes,
            "status": "input_ceiling_exceeded" if input_ceiling_exceeded else "bounded",
        }
        return {
            "schema_version": "task_evaluation_launch_supervisor_snapshot.v1",
            "profiles": profiles,
            "profile_catalog": profile_catalog,
            "admissible_profile_ids": sorted(
                str(row["profile_id"]) for row in profiles if row["admissible"]
            ),
            "queue_counts": queue_counts,
            "terminal_launches": included_terminal_rows,
            "terminal_history": terminal_history,
            "guard": {
                "status": guard.get("status"),
                "generated_at": guard.get("generated_at"),
                "live_instance_count": guard.get("live_instance_count"),
                "total_burn_per_hour_usd": guard.get("total_burn_per_hour_usd"),
                "provider_zero_verified": guard.get("provider_zero_verified"),
                "provider_zero_blockers": (guard.get("provider_zero") or {}).get("blockers")
                if isinstance(guard.get("provider_zero"), Mapping)
                else [],
                "spend_admission_allowed": spend_admission.get("admission_allowed") is True,
                "blockers": guard_blockers,
            },
            "authority_boundary": {
                "agent_may_mutate_provider": False,
                "agent_may_invoke_allocator": False,
                "agent_may_retry": False,
                "agent_may_approve_rights_or_spend": False,
                "agent_may_close_teardown": False,
            },
        }

    while True:
        snapshot = _snapshot_with_digest(snapshot_for_window())
        if _snapshot_size_bytes(snapshot) <= max_snapshot_bytes:
            return snapshot
        if not included_terminal_rows:
            return _snapshot_with_digest(snapshot_for_window(input_ceiling_exceeded=True))
        omitted_terminal_rows.append(included_terminal_rows.pop(0))


def run_launch_supervisor(
    *,
    snapshot: Mapping[str, Any],
    output_dir: str | Path,
    invoker: AgentsSDKInvoker | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    snapshot_value = dict(snapshot)
    snapshot_digest = str(snapshot_value.get("snapshot_digest") or "")
    output_root = Path(output_dir).expanduser().resolve()
    cached_path = output_root / f"{snapshot_digest.removeprefix('sha256:')}.json"
    if cached_path.is_file():
        return _read(cached_path)
    active = _truthy(os.getenv(SUPERVISOR_ENABLED_ENV)) if enabled is None else enabled
    if not active:
        return {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "status": "disabled",
            "snapshot_digest": snapshot_digest,
            "agent_invoked": False,
            "provider_mutation_performed": False,
            "automatic_retry_performed": False,
        }

    terminal_history = snapshot_value.get("terminal_history")
    if (
        isinstance(terminal_history, Mapping)
        and terminal_history.get("status") == "input_ceiling_exceeded"
    ):
        result = {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "status": "blocked",
            "snapshot_digest": snapshot_digest,
            "agent_invoked": False,
            "blockers": [SUPERVISOR_SNAPSHOT_INPUT_CEILING_BLOCKER],
            "tool_count": 0,
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "authority_granted": False,
        }
        result["supervision_digest"] = canonical_digest(result, digest_field="supervision_digest")
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    model = str(os.getenv(SUPERVISOR_MODEL_ENV) or DEFAULT_SUPERVISOR_AGENT_MODEL).strip()
    try:
        budget = float(os.getenv(SUPERVISOR_BUDGET_ENV) or "0")
    except ValueError:
        budget = 0.0
    if invoker is None:
        sdk_invoker = OpenAIAgentsSDKInvoker(
            OpenAIAgentsSDKConfig(
                model=model,
                max_turns=1,
                max_output_tokens=2000,
                allow_live_invocation=True,
                max_inference_cost_usd=budget,
            )
        )
        run_id = f"launch-supervision-{snapshot_digest.removeprefix('sha256:')[:24]}"
        audit = InferenceReservationAudit(
            run_root=output_root / "inference-audit" / run_id,
            run_id=run_id,
        )
        manifest = audit.manifest()
        sdk_invoker.configure_reservation_audit(
            record_reservation=audit.record_reservation,
            record_completion=audit.record_completion,
            restored_reserved_cost_usd=float(manifest["reserved_max_cost_usd"]),
        )
        invoker = sdk_invoker
    else:
        audit = None
    spec = AgentsSDKAgentSpec(
        run_id=f"launch-supervision-{snapshot_digest.removeprefix('sha256:')[:24]}",
        capability="task_evaluation_launch_supervision",
        name="Blueprint Task Evaluation Launch Supervisor",
        model=model,
        max_turns=1,
        max_output_tokens=2000,
        output_type=LaunchSupervisorRecommendation,
        tool_bindings=(),
        instructions=(
            "Observe the supplied immutable production launch snapshot. You have no tools and "
            "no execution authority. Explain blockers, recommend at most one profile only from "
            "admissible_profile_ids, or request one precise human decision. Never request a "
            "provider mutation, allocator call, retry, rights approval, spend approval, teardown "
            "closeout, or proof upgrade. A simulator result is not physical evidence."
        ),
    )
    try:
        invocation = invoker.invoke(spec, json.dumps(snapshot_value, sort_keys=True))
        recommendation = LaunchSupervisorRecommendation.model_validate(invocation.output)
        if (
            recommendation.recommended_profile_id is not None
            and recommendation.recommended_profile_id
            not in snapshot_value.get("admissible_profile_ids", [])
        ):
            raise ValueError("agent_recommended_non_admissible_profile")
        result = {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "status": "completed",
            "snapshot_digest": snapshot_digest,
            "agent_invoked": True,
            "provider": invocation.provider,
            "model": invocation.model,
            "agents_sdk_version": invocation.sdk_version,
            "trace_id": invocation.trace_id,
            "recommendation": recommendation.model_dump(mode="json"),
            "tool_count": 0,
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "authority_granted": False,
        }
    except (AgentsSDKInvocationBlocked, ValueError) as exc:
        result = {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "status": "blocked",
            "snapshot_digest": snapshot_digest,
            "agent_invoked": False,
            "blockers": [str(exc)],
            "provider_mutation_performed": False,
            "allocator_invoked": False,
            "automatic_retry_performed": False,
            "authority_granted": False,
        }
    if audit is not None:
        result["inference_reservation_manifest"] = audit.write_manifest()
    result["supervision_digest"] = canonical_digest(result, digest_field="supervision_digest")
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--guard-report", required=True)
    parser.add_argument("--public-catalog")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latest-out", required=True)
    args = parser.parse_args(argv)
    snapshot = build_supervisor_snapshot(
        profile_dir=args.profile_dir,
        queue_root=args.queue_root,
        state_root=args.state_root,
        guard_report_path=args.guard_report,
        public_catalog_path=args.public_catalog,
    )
    result = run_launch_supervisor(snapshot=snapshot, output_dir=args.output_dir)
    latest = Path(args.latest_out).expanduser().resolve()
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] in {"completed", "blocked"}:
        from .task_evaluation_launch_webapp_sync import (
            sync_launch_supervision_to_webapp,
        )

        sync_result = sync_launch_supervision_to_webapp(supervision=result)
        sync_path = latest.with_name(f"{latest.stem}.webapp-sync{latest.suffix}")
        sync_path.write_text(
            json.dumps(sync_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"completed", "disabled"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
