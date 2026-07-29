"""Rights- and proof-gated evidence-use determination inside one evaluation run."""

from __future__ import annotations

from typing import Any, Mapping

from .decision_evidence_contracts import DecisionEnvelope, canonical_digest


def legacy_evidence_export_requested(request: Mapping[str, Any]) -> bool:
    evidence_use = request.get("evidence_use")
    evidence_use = dict(evidence_use) if isinstance(evidence_use, Mapping) else {}
    legacy = request.get("post_training_data_package")
    legacy = dict(legacy) if isinstance(legacy, Mapping) else {}
    return evidence_use.get("export_requested") is True or legacy.get("requested") is True


def evidence_export_not_requested() -> dict[str, Any]:
    return {
        "schema_version": "task_evaluation_evidence_use.v1",
        "status": "not_requested",
        "standalone_product_created": False,
        "legacy_export_generated": False,
        "training_occurred": False,
        "policy_improved": False,
    }


def build_legacy_evidence_export_if_requested(
    request: Mapping[str, Any], *, capture_root: Any, job_dir: Any, output_dir: Any = None
) -> tuple[bool, dict[str, Any]]:
    if not legacy_evidence_export_requested(request):
        return False, evidence_export_not_requested()
    from .post_training_data_package import build_post_training_data_package_export

    return True, build_post_training_data_package_export(
        capture_root=capture_root, job_dir=job_dir, output_dir=output_dir
    )


def determine_evidence_use(
    decision_value: Mapping[str, Any],
    *,
    rights: Mapping[str, Any],
    provenance: Mapping[str, Any],
    robot_action_alignment: Mapping[str, Any],
    quality: Mapping[str, Any],
    leakage: Mapping[str, Any],
) -> dict[str, Any]:
    """Determine evaluation/post-training eligibility without creating a SKU."""

    decision = DecisionEnvelope.from_mapping(decision_value).to_mapping()
    rights_ok = all(
        rights.get(key) is True
        for key in ("evaluation_use_allowed", "consent_current", "revocation_clear")
    )
    proof_ok = decision["overall_outcome"] in {"decision", "partial_decision"}
    evaluation_allowed = rights_ok and proof_ok
    post_training_checks = {
        "rights_allow_post_training": rights.get("post_training_use_allowed") is True,
        "provenance_complete": provenance.get("complete") is True,
        "robot_action_aligned": robot_action_alignment.get("aligned") is True,
        "quality_gate_passed": quality.get("gate_passed") is True,
        "heldout_leakage_absent": leakage.get("heldout_leakage_absent") is True,
    }
    post_training_allowed = evaluation_allowed and all(post_training_checks.values())
    blockers = []
    if not rights_ok:
        blockers.append("evaluation_rights_or_consent_gate_failed")
    if not proof_ok:
        blockers.append("decision_proof_gate_failed")
    blockers.extend(
        f"post_training_gate_failed:{key}"
        for key, passed in sorted(post_training_checks.items())
        if not passed
    )
    artifact = {
        "schema_version": "task_evaluation_evidence_use.v1",
        "decision_envelope_digest": decision["decision_envelope_digest"],
        "evaluation_use": {"allowed": evaluation_allowed},
        "post_training_use": {
            "allowed": post_training_allowed,
            "gates": post_training_checks,
        },
        "blockers": blockers,
        "standalone_product_created": False,
        "training_occurred": False,
        "policy_improved": False,
        "raw_capture_authority_preserved": True,
    }
    artifact["evidence_use_digest"] = canonical_digest(
        artifact, digest_field="evidence_use_digest"
    )
    return artifact


__all__ = [
    "determine_evidence_use",
    "build_legacy_evidence_export_if_requested",
    "evidence_export_not_requested",
    "legacy_evidence_export_requested",
]
