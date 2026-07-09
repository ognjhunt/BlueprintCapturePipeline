import json
from pathlib import Path

from blueprint_pipeline.alpha_readiness import validate_operator_launch_evidence


def test_identity_kyc_background_decision_evidence_validates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    evidence_path = (
        repo_root
        / "docs"
        / "examples"
        / "operator_launch_evidence.identity_kyc_background_decisions.json"
    )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    result = validate_operator_launch_evidence(
        evidence,
        [
            "identity_kyc_provider_decision",
            "background_check_provider_decision",
        ],
    )

    assert result["status"] == "verified"
    assert result["remaining_ids"] == []
    assert result["verified_ids"] == [
        "identity_kyc_provider_decision",
        "background_check_provider_decision",
    ]


def test_identity_kyc_background_decision_record_preserves_claim_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    evidence_path = (
        repo_root
        / "docs"
        / "examples"
        / "operator_launch_evidence.identity_kyc_background_decisions.json"
    )

    checks = json.loads(evidence_path.read_text(encoding="utf-8"))["checks"]
    identity = checks["identity_kyc_provider_decision"]["metadata"]
    background = checks["background_check_provider_decision"]["metadata"]

    assert identity["kyc_account_requirements_path"] == "stripe_connect_onboarding"
    assert identity["separate_identity_provider_integrated"] is False
    assert "not_live_connected_account" in identity["claim_boundary"]
    assert background["checkr_integrated"] is False
    assert background["background_check_claims_allowed"] is False
    assert background["physical_site_access_screening_claims_allowed"] is False
    assert "not_background_check_readiness" in background["claim_boundary"]
