"""Admit ONE robot team's own policy to a practice run or a scored evaluation of a site job.

Beta journey step: L5 team policy evaluation. The website scene runtime ran
only Blueprint's two frozen public policies. This control-plane admission (no
network, no provider, no spend) decides whether one team candidate may run on
the scene, for ``practice`` (the ten published Quick-10 cells, feedback only)
or ``scored`` (200 hidden cells, the only evidence a recommendation may rest
on; ``team_evaluation_plan``), and seals what the runtime may run:

- the candidate is the team's own policy on the DROID Franka (Robotiq 2F-85)
  interface, served by the team over authenticated HTTPS
  (``openpi_droid_https.v1``); a private-checkpoint delivery is refused by
  name because Blueprint has no runtime that serves a team's uploaded
  pi0.5/GR00T weights on the GPU yet;
- its rights record grants a licence to run and forbids self-grading,
  ranking, secret retention and Arm Decision Proof evidence use;
- a single-use KMS bearer-credential lease for exactly that origin is
  available (the secret itself never enters this record);
- the run is free, single-attempt, has exactly its purpose's episode count, and
  stays inside the free-beta provider cap;
- the founder admitted exactly this team, candidate and scene revision, and
  the admission has not expired;
- for a scored run, the scene's robot-free success probe passed at this
  revision for the exact task specification (``robot_free_probe_gate``); a
  practice run may go ahead while the probe is still pending (it records the
  rigid probe itself), never after it failed; and
- the scene's success contract and embodiment are the ones the run scores.

Every refusal is collected and raised together with typed reasons. A sealed
admission is evidence of permission only; it is never a result, and its
claim ceiling stays ``diagnostic_policy_execution``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from .adp_articulated_task_success_contract import task_kind_of_contract, validate_task_success_contract
from .adp_task_scoring import TaskNeutralScoringError
from .decision_evidence_contracts import canonical_digest
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    CLAIM_CEILING,
    TEAM_ADMISSION_SCHEMA_VERSION,
    TEAM_RUN_KIND,
)
from .policy_interface_binding import TEAM_HTTPS_INTERFACES
from .robot_free_probe_gate import RobotFreeProbeGateError, require_robot_free_probe, task_success_contract_binding
from .team_evaluation_plan import EPISODES, PRACTICE, PURPOSES, SCORED, SHARD_CELLS, shard_ttl_seconds

CANDIDATE_SCHEMA = "blueprint.team_policy_candidate.v1"
RIGHTS_SCHEMA = "blueprint.team_policy_candidate_rights.v1"
LEASE_SCHEMA = "blueprint.team_policy_credential_lease_availability.v1"
FOUNDER_SCHEMA = "blueprint.team_candidate_founder_admission.v1"
EXECUTION_SPEC_SCHEMA = "native_task_arena_policy_canary_execution_spec.v1"
RUNTIME = "team_evaluation"
FREE_BETA_PROVIDER_CAP_USD = 20.0
# Per GPU session (one shard): the longest shard is SHARD_CELLS cells.
MAX_TTL_SECONDS = shard_ttl_seconds(SHARD_CELLS)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]{2,58}_FILE$")


class TeamCandidateAdmissionError(ValueError):
    """The team candidate may not run; ``blockers`` names every reason."""

    def __init__(self, blockers: list[str]):
        self.blockers = sorted(set(blockers))
        super().__init__(";".join(self.blockers))


def _m(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _time(value: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _self_digest_ok(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == canonical_digest(dict(value), digest_field=field)


def _candidate_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers = []
    delivery = _m(candidate.get("delivery"))
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        blockers.append("team_candidate_record_invalid")
    if not all(_ID.fullmatch(str(candidate.get(key) or "")) for key in ("team_id", "checkpoint_id", "candidate_id")):
        blockers.append("team_candidate_identity_invalid")
    if candidate.get("candidate_id") in CANDIDATE_IDS:
        blockers.append("team_candidate_cannot_impersonate_frozen_candidate")
    if candidate.get("embodiment_preset_id") != DROID_POLICY_CANARY_PRESET_ID:
        blockers.append("team_candidate_interface_mismatch:embodiment")
    if delivery.get("mode") == "private_checkpoint":
        blockers.append("team_candidate_private_checkpoint_runtime_unavailable")
    elif delivery.get("mode") != "https_policy" or candidate.get("policy_interface_id") not in TEAM_HTTPS_INTERFACES:
        blockers.append("team_candidate_interface_mismatch:delivery")
    else:
        urls = [str(delivery.get(key) or "") for key in ("endpoint_url", "metadata_url")]
        origins = {f"{urlsplit(url).scheme}://{urlsplit(url).netloc}" for url in urls}
        if (any(urlsplit(url).scheme != "https" or not urlsplit(url).hostname or urlsplit(url).username
                or urlsplit(url).password or urlsplit(url).fragment for url in urls) or len(origins) != 1):
            blockers.append("team_candidate_endpoint_invalid")
        if (delivery.get("action_chunk_rows") not in (10, 15) or type(delivery.get("open_loop_horizon")) is not int
                or not 1 <= delivery["open_loop_horizon"] <= int(delivery.get("action_chunk_rows") or 0)
                or not _DIGEST.fullmatch(str(delivery.get("model_identity_digest") or ""))
                or not _ID.fullmatch(str(delivery.get("team_policy_id") or ""))):
            blockers.append("team_candidate_interface_mismatch:action_chunk")
    return blockers


def _rights_blockers(rights: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    if (rights.get("schema_version") != RIGHTS_SCHEMA or rights.get("status") != "admitted"
            or not _self_digest_ok(rights, "rights_digest")
            or any(rights.get(key) != candidate.get(key) for key in ("team_id", "candidate_id", "checkpoint_id"))):
        return ["team_candidate_rights_record_missing_or_mismatched"]
    if (rights.get("license_to_run_granted") is not True
            or any(rights.get(key) is not False for key in (
                "policy_self_grading_permitted", "official_ranking_permitted",
                "secret_material_recorded", "adp_qualified_evidence_permitted"))):
        return ["team_candidate_rights_scope_insufficient"]
    return []


def _lease_blockers(lease: Mapping[str, Any], candidate: Mapping[str, Any], now: datetime) -> list[str]:
    delivery = _m(candidate.get("delivery"))
    endpoint = str(delivery.get("endpoint_url") or "")
    expires = _time(lease.get("expires_at_iso"))
    if (lease.get("schema_version") != LEASE_SCHEMA or lease.get("kind") != "kms_bearer_lease"
            or lease.get("status") != "available" or lease.get("single_use") is not True
            or not _ID.fullmatch(str(lease.get("lease_id") or ""))
            or not _DIGEST.fullmatch(str(lease.get("credential_ref_digest") or ""))
            or not _ENV.fullmatch(str(lease.get("credential_env") or ""))
            or lease.get("origin") != f"{urlsplit(endpoint).scheme}://{urlsplit(endpoint).netloc}"
            or lease.get("candidate_id") != candidate.get("candidate_id")
            or lease.get("secret_material_recorded") is not False
            or expires is None or expires <= now):
        return ["team_candidate_credential_lease_unavailable"]
    return []


def _budget_blockers(budget: Mapping[str, Any], purpose: str) -> list[str]:
    cap = budget.get("cap_usd")
    ttl = budget.get("hard_ttl_seconds")
    if (budget.get("payer") != "blueprint" or budget.get("customer_price_usd") != 0
            or budget.get("max_attempts") != 1 or budget.get("episodes") != EPISODES.get(purpose)
            or type(cap) not in (int, float) or not 0 < float(cap) <= FREE_BETA_PROVIDER_CAP_USD
            or type(ttl) is not int or not 0 < ttl <= MAX_TTL_SECONDS):
        return ["team_candidate_budget_cap_invalid"]
    return []


def _founder_blockers(founder: Mapping[str, Any], candidate: Mapping[str, Any], revision: Any,
                      now: datetime) -> list[str]:
    scope = _m(founder.get("scope"))
    expires = _time(founder.get("expires_at_iso"))
    if (founder.get("schema_version") != FOUNDER_SCHEMA or not str(founder.get("approved_by") or "").strip()
            or not str(founder.get("approval_reference") or "").strip()
            or not _self_digest_ok(founder, "approval_digest") or expires is None or expires <= now
            or scope != {"team_id": candidate.get("team_id"), "candidate_id": candidate.get("candidate_id"),
                         "scene_revision_digest": revision, "run_kind": TEAM_RUN_KIND}):
        return ["team_candidate_founder_admission_missing"]
    return []


def admit_team_candidate(*, candidate: Mapping[str, Any], rights: Mapping[str, Any],
                         credential_lease: Mapping[str, Any], budget: Mapping[str, Any],
                         founder_admission: Mapping[str, Any], scene: Mapping[str, Any],
                         probe_receipt: Mapping[str, Any] | None, purpose: str = SCORED,
                         now: datetime | None = None) -> dict[str, Any]:
    """Seal one team candidate's practice or scored admission, or raise with every typed reason."""

    moment = now or datetime.now(timezone.utc)
    candidate, scene = _m(candidate), _m(scene)
    revision = scene.get("scene_revision_digest")
    task_spec = _m(scene.get("task_spec"))
    blockers = [] if purpose in PURPOSES else ["team_candidate_evaluation_purpose_invalid"]
    blockers += [*_candidate_blockers(candidate), *_rights_blockers(_m(rights), candidate),
                 *_lease_blockers(_m(credential_lease), candidate, moment), *_budget_blockers(_m(budget), purpose),
                 *_founder_blockers(_m(founder_admission), candidate, revision, moment)]
    contract: dict[str, Any] = {}
    try:
        raw_contract = _m(scene.get("task_success_contract"))
        contract = dict(validate_task_success_contract(raw_contract, task_kind=task_kind_of_contract(raw_contract)))
    except (TaskNeutralScoringError, ValueError, KeyError, TypeError):
        blockers.append("team_candidate_scene_success_contract_invalid")
    if (contract.get("criteria") or {}).get("controls", {}).get("mode") == "required_per_cell":
        blockers.append("team_candidate_strict_controls_runtime_unavailable")
    if scene.get("embodiment_preset_id") != candidate.get("embodiment_preset_id"):
        blockers.append("team_candidate_interface_mismatch:scene_embodiment")
    probe: dict[str, Any] = {}
    try:
        probe = require_robot_free_probe(probe_receipt, scene_revision_digest=revision,
                                         task_spec_digest=canonical_digest(task_spec),
                                         task_kind=task_spec.get("task_kind"),
                                         task_success_contract_digest=task_success_contract_binding(
                                             task_spec.get("task_kind"), scene.get("task_success_contract")))
    except RobotFreeProbeGateError as exc:
        # Practice is feedback, never evidence: it may run (and record the
        # rigid probe) while the probe is pending, but never on a scene whose
        # probe was decided and failed.
        if purpose != PRACTICE or _m(probe_receipt).get("status") not in (None, "probe_unavailable"):
            blockers.append(str(exc))
    if blockers:
        raise TeamCandidateAdmissionError(blockers)
    delivery = candidate["delivery"]
    admission = {
        "schema_version": TEAM_ADMISSION_SCHEMA_VERSION, "status": "admitted", "run_kind": TEAM_RUN_KIND,
        "runtime": RUNTIME, "evaluation_purpose": purpose, "episodes": EPISODES[purpose],
        "team_id": candidate["team_id"], "checkpoint_id": candidate["checkpoint_id"],
        "candidate_id": candidate["candidate_id"], "policy_interface_id": candidate["policy_interface_id"],
        "embodiment_preset_id": candidate["embodiment_preset_id"],
        "delivery": {key: delivery[key] for key in (
            "mode", "endpoint_url", "metadata_url", "action_chunk_rows", "open_loop_horizon", "model_identity_digest",
            "team_policy_id")},
        "credential": {key: credential_lease[key] for key in ("lease_id", "credential_ref_digest", "credential_env",
                                                              "origin")},
        "scene_revision_digest": revision, "task_spec_digest": canonical_digest(task_spec),
        "task_success_contract_digest": contract["contract_digest"],
        "robot_free_probe_receipt_digest": probe.get("receipt_digest"),
        "rights_digest": rights["rights_digest"], "founder_approval_digest": founder_admission["approval_digest"],
        "budget": {key: budget[key] for key in ("cap_usd", "hard_ttl_seconds", "max_attempts", "episodes")},
        "claim_ceiling": CLAIM_CEILING, "evidence_scope": "development_only",
        "adp_qualified_evidence_eligible": False, "physical_performance_claimed": False,
        "admission_digest": "",
    }
    admission["admission_digest"] = canonical_digest(admission, digest_field="admission_digest")
    return admission


def validate_admission(admission: Mapping[str, Any]) -> dict[str, Any]:
    value = _m(admission)
    if (value.get("schema_version") != TEAM_ADMISSION_SCHEMA_VERSION or value.get("status") != "admitted"
            or value.get("run_kind") != TEAM_RUN_KIND or value.get("runtime") != RUNTIME
            or not _self_digest_ok(value, "admission_digest")):
        raise TeamCandidateAdmissionError(["team_candidate_admission_invalid"])
    return value


def runtime_binding(admission: Mapping[str, Any]) -> dict[str, Any]:
    """The ``team_candidate_admission`` block the Quick-10 runtime inputs carry."""

    value = validate_admission(admission)
    return {"schema_version": TEAM_ADMISSION_SCHEMA_VERSION, "admission_digest": value["admission_digest"],
            "candidate_id": value["candidate_id"], "evaluation_purpose": value["evaluation_purpose"],
            "robot_free_probe_receipt_digest": value["robot_free_probe_receipt_digest"],
            "scene_revision_digest": value["scene_revision_digest"],
            "task_success_contract_digest": value["task_success_contract_digest"]}


def execution_spec(admission: Mapping[str, Any], *, task_success_contract: Mapping[str, Any], prompt: str,
                   max_policy_queries: int) -> dict[str, Any]:
    """The sealed per-candidate execution spec the canary worker loads (no secret material)."""

    value = validate_admission(admission)
    if task_success_contract.get("contract_digest") != value["task_success_contract_digest"]:
        raise TeamCandidateAdmissionError(["team_candidate_execution_contract_mismatch"])
    delivery, credential = value["delivery"], value["credential"]
    # The team's own declared id; Blueprint's candidate id stays the evidence identity.
    policy_spec = {"policy_id": delivery["team_policy_id"], "endpoint_url": delivery["endpoint_url"],
                   "metadata_url": delivery["metadata_url"], "allowed_origin": credential["origin"],
                   "model_identity_digest": delivery["model_identity_digest"],
                   "action_chunk_rows": delivery["action_chunk_rows"],
                   "open_loop_horizon": delivery["open_loop_horizon"],
                   "policy_interface_id": value["policy_interface_id"]}
    spec = {
        "schema_version": EXECUTION_SPEC_SCHEMA, "candidate_id": value["candidate_id"],
        "policy_interface_id": value["policy_interface_id"],
        "execution_authority": "team_candidate_evaluation_unqualified", "claim_ceiling": CLAIM_CEILING,
        "ranking_permitted": False, "qualification_permitted": False, "scene_promotion_permitted": False,
        "policy_endpoint": {"credential_env": credential["credential_env"]}, "policy_spec": policy_spec,
        "candidate_rights_binding": {"status": "admitted", "rights_digest": value["rights_digest"]},
        "team_candidate_admission_digest": value["admission_digest"],
        "checkpoint_digest": delivery["model_identity_digest"],
        "runtime_identity_digest": canonical_digest(policy_spec),
        "worker_identity_requirement": "none", "require_observed_eef_support": False,
        "task_success_contract": dict(task_success_contract),
        "task_success_contract_digest": value["task_success_contract_digest"],
        "prompt": prompt, "max_policy_queries": int(max_policy_queries),
        "open_loop_horizon": delivery["open_loop_horizon"], "execution_spec_digest": "",
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


__all__ = [
    "RUNTIME",
    "TeamCandidateAdmissionError",
    "admit_team_candidate",
    "execution_spec",
    "runtime_binding",
    "validate_admission",
]
