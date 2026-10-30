"""Validate a team candidate's sealed run rights and hosted-disclosure authority.

Rights are a hard stop: a registration runs only under an admitted, self-sealed
licence to run that forbids self-grading, ranking, secret retention and Arm
Decision Proof evidence use. This is the single implementation that team
admission (``team_candidate_quick10_admission``), screening registration and
the effective catalog apply. It has no network, provider, scene or spend
dependency.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .dreamlabs_policy_client import INTERFACE_ID as DREAM_INTERFACE

RIGHTS_SCHEMA = "blueprint.team_policy_candidate_rights.v1"
PROVIDER_DISCLOSURE_SCHEMA = "blueprint.dreamlabs_provider_disclosure.v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def validate_provider_disclosure(rights: Mapping[str, Any], *, team_id: str, checkpoint_id: str,
                                 candidate_id: str, model_id: str, scene_revision_digest: str,
                                 now: datetime | None = None) -> dict[str, Any]:
    """An operator's explicit hosted-disclosure authority, not inferred from a licence.

    The receipt accepts the referenced provider retention/training terms. It
    does not assert that the inference server disables either practice.
    """
    raw = rights.get("provider_disclosure")
    if not isinstance(raw, Mapping):
        raise TeamCandidateAdmissionError(["team_candidate_provider_disclosure_required"])
    receipt = dict(raw)
    fields = {"schema_version", "team_id", "checkpoint_id", "candidate_id", "model_id", "scene_revision_digest",
              "inference_origin", "metadata_origin", "approved_by", "approval_reference", "accepted_at_iso",
              "disclosure", "provider_terms", "sdk_extra_recording", "hosted_billing", "disclosure_digest"}
    accepted = _time(receipt.get("accepted_at_iso"))
    terms = _m(receipt.get("provider_terms"))

    def reference(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip()) and len(value) <= 2048 and not any(ord(c) < 32 for c in value)

    billing = _m(receipt.get("hosted_billing"))
    if (set(billing) != {"payer", "authorized_by", "approval_reference", "policy_inference_usage_authorized",
                         "blueprint_billing_permitted"}
            or billing.get("payer") != "team_account" or not reference(billing.get("authorized_by"))
            or not reference(billing.get("approval_reference")) or billing.get("policy_inference_usage_authorized") is not True
            or billing.get("blueprint_billing_permitted") is not False):
        raise TeamCandidateAdmissionError(["team_candidate_hosted_policy_billing_authority_required"])
    if (set(receipt) != fields or receipt.get("schema_version") != PROVIDER_DISCLOSURE_SCHEMA
            or not _self_digest_ok(receipt, "disclosure_digest")
            or any(receipt.get(key) != expected for key, expected in {
                "team_id": team_id, "checkpoint_id": checkpoint_id, "candidate_id": candidate_id,
                "model_id": model_id, "scene_revision_digest": scene_revision_digest,
                "inference_origin": "https://api.dreamlabs.ai", "metadata_origin": "https://platform.dreamlabs.ai"}.items())
            or not _DIGEST.fullmatch(str(scene_revision_digest)) or not reference(receipt.get("approved_by"))
            or not reference(receipt.get("approval_reference")) or accepted is None
            or accepted > (now or datetime.now(timezone.utc))
            or receipt.get("disclosure") != {"rgb": True, "robot_state": True, "task_instruction": True}
            or set(terms) != {"retention_terms_reference", "training_terms_reference", "accepted"}
            or terms.get("accepted") is not True
            or not all(reference(terms.get(key)) for key in ("retention_terms_reference", "training_terms_reference"))
            or receipt.get("sdk_extra_recording") is not False):
        raise TeamCandidateAdmissionError(["team_candidate_provider_disclosure_terms_invalid"])
    # Avoid Python's True==1 equality admitting numeric disclosure claims.
    if any(value is not True for value in receipt["disclosure"].values()):
        raise TeamCandidateAdmissionError(["team_candidate_provider_disclosure_terms_invalid"])
    return receipt


def validate_candidate_rights(rights: Mapping[str, Any], *, team_id: str,
                              checkpoint_id: str, candidate_id: str, policy_interface_id: str | None = None,
                              model_id: str | None = None, scene_revision_digest: str | None = None,
                              now: datetime | None = None) -> None:
    """Validate static run rights before consuming a team's bearer lease."""

    blockers = _rights_blockers(_m(rights), {"team_id": team_id, "checkpoint_id": checkpoint_id,
                                            "candidate_id": candidate_id})
    if blockers:
        raise TeamCandidateAdmissionError(blockers)
    if policy_interface_id == DREAM_INTERFACE:
        validate_provider_disclosure(rights, team_id=team_id, checkpoint_id=checkpoint_id, candidate_id=candidate_id,
                                      model_id=model_id, scene_revision_digest=scene_revision_digest, now=now)


__all__ = [
    "PROVIDER_DISCLOSURE_SCHEMA",
    "RIGHTS_SCHEMA",
    "TeamCandidateAdmissionError",
    "validate_candidate_rights",
    "validate_provider_disclosure",
]
