"""R011 — site-aware ``policy_only`` consent-evidence gate.

A bare ``policy_only`` consent claim previously self-cleared the consent-evidence
gate with zero operator permission document and no site-type gate. That is only
defensible for public / publicly-accessible sites: industrial sites
(warehouses/factories/cold storage/stockrooms) and explicitly private/restricted
property require an actual operator authorization — a permission document OR an
explicit lawful-basis attestation — before ``policy_only`` may clear.

These tests pin the site-aware behavior in
``proof_contracts.build_rights_provenance_review``:

- ``policy_only`` + industrial site + no operator authorization does NOT clear and
  surfaces ``policy_only_insufficient_for_private_or_industrial_site``.
- ``policy_only`` + public/unknown site still self-clears (backward compatible).
- ``policy_only`` + industrial site + a permission document OR lawful-basis
  attestation clears.
- The gate holds whether industrial-ness comes from the ``site_type`` text or the
  privacy manifest's ``is_industrial_site`` flag, and it also applies to an
  explicit private/restricted (non-industrial) site.
"""

from __future__ import annotations

from blueprint_pipeline.proof_contracts import build_rights_provenance_review

_POLICY_ONLY_BLOCKER = "policy_only_insufficient_for_private_or_industrial_site"


def _policy_only_rights(**overrides: object) -> dict:
    summary: dict = {
        "consent_status": "policy_only",
        "derived_scene_generation_allowed": True,
    }
    summary.update(overrides)
    return summary


def _grounded_provenance() -> dict:
    return {"status": "grounded", "record": {"canonical_truth": True}}


def _industrial_privacy_cleared() -> dict:
    # An industrial site must also satisfy R010's privacy-coverage gate to clear;
    # that is orthogonal to R011's consent gate but required for an overall clear.
    return {
        "status": "person_removed",
        "is_industrial_site": True,
        "industrial_sensitive_classes_handled": True,
    }


def _review(
    rights_summary: dict,
    privacy_processing: dict | None = None,
    **kwargs: object,
) -> dict:
    return build_rights_provenance_review(
        rights_summary=rights_summary,
        privacy_processing=privacy_processing or {"status": "person_removed"},
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        **kwargs,
    )


# ----------------------------------------------------------------------------------
# Industrial site: policy_only alone is not a free pass
# ----------------------------------------------------------------------------------


def test_policy_only_industrial_site_without_permission_doc_does_not_clear() -> None:
    review = _review(_policy_only_rights(), site_type="warehouse distribution center")

    # Gate NOT cleared and the specific blocker is surfaced.
    assert review["rights"]["consent_evidence_complete"] is False
    assert review["rights"]["status"] != "cleared"
    assert review["status"] != "cleared"
    assert _POLICY_ONLY_BLOCKER in review["blockers"]
    assert review["rights"]["site_requires_operator_authorization"] is True
    assert review["rights"]["policy_only_insufficient_for_site"] is True


def test_policy_only_industrial_flag_from_privacy_manifest_without_site_type() -> None:
    """Industrial-ness carried on the privacy manifest gates even without site_type."""

    review = build_rights_provenance_review(
        rights_summary=_policy_only_rights(),
        privacy_processing={"status": "person_removed", "is_industrial_site": True},
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
    )

    assert review["rights"]["consent_evidence_complete"] is False
    assert _POLICY_ONLY_BLOCKER in review["blockers"]
    assert review["status"] != "cleared"


# ----------------------------------------------------------------------------------
# Industrial site: operator authorization clears the gate
# ----------------------------------------------------------------------------------


def test_policy_only_industrial_site_with_permission_document_clears() -> None:
    review = _review(
        _policy_only_rights(
            permission_document_uri="gs://bucket/rights/operator-permission.pdf"
        ),
        privacy_processing=_industrial_privacy_cleared(),
        site_type="factory assembly line",
    )

    assert review["rights"]["consent_evidence_complete"] is True
    assert review["rights"]["status"] == "cleared"
    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]
    assert review["rights"]["operator_authorization_present"] is True


def test_policy_only_industrial_site_with_lawful_basis_attestation_clears() -> None:
    review = _review(
        _policy_only_rights(lawful_basis_attestation="operator_authorized_capture"),
        privacy_processing=_industrial_privacy_cleared(),
        site_type="warehouse fulfillment center",
    )

    assert review["rights"]["consent_evidence_complete"] is True
    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]


# ----------------------------------------------------------------------------------
# Public / unknown sites: policy_only still self-clears (backward compatible)
# ----------------------------------------------------------------------------------


def test_policy_only_public_site_still_clears() -> None:
    review = _review(_policy_only_rights(), site_type="retail sales floor")

    assert review["rights"]["consent_evidence_complete"] is True
    assert review["rights"]["site_requires_operator_authorization"] is False
    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]


def test_policy_only_unknown_site_type_still_clears() -> None:
    # No site_type threaded and no industrial/private signal — unchanged behavior.
    review = _review(_policy_only_rights())

    assert review["rights"]["consent_evidence_complete"] is True
    assert review["rights"]["site_requires_operator_authorization"] is False
    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]


# ----------------------------------------------------------------------------------
# Explicit private/restricted (non-industrial) sites also require authorization
# ----------------------------------------------------------------------------------


def test_policy_only_explicit_private_property_flag_requires_authorization() -> None:
    review = _review(
        _policy_only_rights(private_property=True), site_type="office workspace"
    )

    assert review["rights"]["site_requires_operator_authorization"] is True
    assert review["rights"]["consent_evidence_complete"] is False
    assert _POLICY_ONLY_BLOCKER in review["blockers"]
    assert review["status"] != "cleared"


def test_policy_only_restricted_site_access_requires_authorization() -> None:
    review = _review(_policy_only_rights(site_access="restricted"))

    assert review["rights"]["site_requires_operator_authorization"] is True
    assert _POLICY_ONLY_BLOCKER in review["blockers"]


def test_policy_only_publicly_accessible_false_requires_authorization() -> None:
    review = _review(_policy_only_rights(publicly_accessible=False))

    assert review["rights"]["site_requires_operator_authorization"] is True
    assert _POLICY_ONLY_BLOCKER in review["blockers"]


def test_policy_only_publicly_accessible_true_still_clears() -> None:
    review = _review(_policy_only_rights(publicly_accessible=True))

    assert review["rights"]["site_requires_operator_authorization"] is False
    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]


# ----------------------------------------------------------------------------------
# The documented-consent path is untouched by R011
# ----------------------------------------------------------------------------------


def test_documented_consent_with_doc_clears_on_industrial_site() -> None:
    review = _review(
        {
            "consent_status": "documented",
            "derived_scene_generation_allowed": True,
            "permission_document_uri": "gs://bucket/rights/consent-packet.pdf",
        },
        privacy_processing=_industrial_privacy_cleared(),
        site_type="warehouse loading dock",
    )

    assert review["status"] == "cleared"
    assert _POLICY_ONLY_BLOCKER not in review["blockers"]
