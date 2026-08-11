from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import blueprint_pipeline.semantic_review_attestation as attestation_module
from blueprint_pipeline.semantic_review_attestation import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SemanticReviewAttestationError,
    canonical_semantic_authority_selection_bytes,
    canonical_semantic_review_attestation_bytes,
    materialize_semantic_authority_selection,
    materialize_semantic_review_attestation,
    materialize_semantic_review_payload,
    semantic_frame_evidence_digest,
    semantic_review_signature_message,
    verify_semantic_review_attestation,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _signed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    signer: Ed25519PrivateKey | None = None,
) -> dict[str, object]:
    signer = signer or Ed25519PrivateKey.from_private_bytes(b"\x31" * 32)
    frames_digest = semantic_frame_evidence_digest(
        [
            {
                "target_id": "basket_87",
                "camera_id": "basket_e00",
                "sha256": _digest("png-bytes"),
                "size_bytes": 1234,
                "decoded_rgb_sha256": _digest("decoded-rgb"),
            }
        ]
    )
    payload = materialize_semantic_review_payload(
        attestation_id="fixture-semantic-review-001",
        selection_id="fixture-authority-freeze-001",
        authority_id="blueprint-semantic-review-fixture",
        authority_key_id="fixture-semantic-key-2026-08",
        scene_id="840873",
        target_id="basket_87",
        source_instance_id="87",
        semantic_role="destination_receptacle",
        visual_review_digest=_digest("review"),
        render_manifest_digest=_digest("render"),
        collision_topology_receipt_digest=_digest("topology"),
        cited_frames_digest=frames_digest,
        learned_policy_outcomes_inspected=False,
        semantic_assertions={
            "rigid_exterior_observed": True,
            "open_rim_observed": True,
            "source_destination_admitted": False,
            "selection_role": "engineered_twin_design_basis",
        },
    )
    public_key = _public_bytes(signer)
    attestation = materialize_semantic_review_attestation(
        payload=payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            signer.sign(semantic_review_signature_message(payload))
        ).decode("ascii"),
    )
    selection = materialize_semantic_authority_selection(attestation=attestation)
    attestation_path = tmp_path / "semantic-attestation.json"
    selection_path = tmp_path / "semantic-selection.json"
    attestation_path.write_bytes(canonical_semantic_review_attestation_bytes(attestation))
    selection_path.write_bytes(canonical_semantic_authority_selection_bytes(selection))
    fingerprint = "sha256:" + hashlib.sha256(public_key).hexdigest()
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, fingerprint)
    return {
        "attestation": attestation,
        "selection": selection,
        "attestation_path": attestation_path,
        "selection_path": selection_path,
        "fingerprint": fingerprint,
    }


def _verify(fixture: dict[str, object]) -> dict:
    return verify_semantic_review_attestation(
        attestation_path=fixture["attestation_path"],
        selection_contract_path=fixture["selection_path"],
    )


def test_verifies_signed_path_only_semantic_authority_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)

    result = _verify(fixture)

    assert result["status"] == "verified"
    assert result["semantic_authority_verified"] is True
    assert result["signature_cryptographically_valid"] is True
    assert result["configured_authority_key_matched"] is True
    assert result["authority"] == {
        "authority_id": "blueprint-semantic-review-fixture",
        "key_id": "fixture-semantic-key-2026-08",
        "public_key_sha256": fixture["fingerprint"],
    }
    assert result["scene_id"] == "840873"
    assert result["source_target"] == {
        "target_id": "basket_87",
        "source_instance_id": "87",
        "semantic_role": "destination_receptacle",
    }
    assert result["learned_policy_outcomes_inspected"] is False
    assert result["claim_boundary"]["native_simulator_qualified"] is False
    assert "physical_material_equivalence" in result["does_not_establish"]


def test_verifier_exposes_no_caller_trust_root_or_expected_binding_override() -> None:
    parameters = inspect.signature(verify_semantic_review_attestation).parameters

    assert set(parameters) == {"attestation_path", "selection_contract_path"}


def test_unconfigured_or_wrong_authority_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    monkeypatch.delenv(TRUSTED_PUBLIC_KEY_SHA256_ENV)

    with pytest.raises(SemanticReviewAttestationError) as unconfigured:
        _verify(fixture)

    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, _digest("wrong-key"))
    with pytest.raises(SemanticReviewAttestationError) as wrong:
        _verify(fixture)

    assert "semantic_review_attestation_trust_root_not_configured" in unconfigured.value.errors
    assert "semantic_review_attestation_public_key_not_authorized" in wrong.value.errors


def test_valid_untrusted_signer_cannot_authorize_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = Ed25519PrivateKey.from_private_bytes(b"\x41" * 32)
    attacker = Ed25519PrivateKey.from_private_bytes(b"\x51" * 32)
    fixture = _signed_files(tmp_path, monkeypatch, signer=attacker)
    trusted_fingerprint = "sha256:" + hashlib.sha256(_public_bytes(trusted)).hexdigest()
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, trusted_fingerprint)

    with pytest.raises(SemanticReviewAttestationError) as error:
        _verify(fixture)

    assert error.value.errors == ("semantic_review_attestation_public_key_not_authorized",)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("evidence", "semantic_authority_selection_evidence_mismatch"),
        ("selection_id", "semantic_authority_selection_selection_id_mismatch"),
    ],
)
def test_self_rehashed_selection_cannot_swap_any_signed_freeze_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    selection = copy.deepcopy(fixture["selection"])
    if mutation == "evidence":
        selection["evidence"]["visual_review_digest"] = _digest("attacker-review")
    else:
        selection["selection_id"] = "attacker-selected-after-outcomes"
    selection["selection_digest"] = ""
    unsigned = {key: value for key, value in selection.items() if key != "selection_digest"}
    selection["selection_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    fixture["selection_path"].write_bytes(canonical_semantic_authority_selection_bytes(selection))

    with pytest.raises(SemanticReviewAttestationError) as error:
        _verify(fixture)

    assert error.value.errors == (expected_error,)


def test_noncanonical_and_duplicate_json_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    fixture["attestation_path"].write_text(
        json.dumps(fixture["attestation"], indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(SemanticReviewAttestationError) as noncanonical:
        _verify(fixture)

    fixture["attestation_path"].write_text(
        '{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8"
    )
    with pytest.raises(SemanticReviewAttestationError) as duplicate:
        _verify(fixture)

    assert "semantic_review_attestation_encoding_not_canonical" in noncanonical.value.errors
    assert duplicate.value.errors == ("semantic_review_attestation_json_invalid",)


def test_malformed_payload_is_typed_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    malformed = copy.deepcopy(fixture["attestation"])
    malformed["payload"] = 7
    fixture["attestation_path"].write_text(
        json.dumps(malformed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SemanticReviewAttestationError) as error:
        _verify(fixture)

    assert "semantic_review_attestation_payload_invalid" in error.value.errors


def test_symlink_and_mapping_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    link = tmp_path / "linked-attestation.json"
    link.symlink_to(fixture["attestation_path"])

    with pytest.raises(SemanticReviewAttestationError) as symlink:
        verify_semantic_review_attestation(
            attestation_path=link,
            selection_contract_path=fixture["selection_path"],
        )
    with pytest.raises(SemanticReviewAttestationError) as mapping:
        verify_semantic_review_attestation(
            attestation_path=fixture["attestation"],
            selection_contract_path=fixture["selection_path"],
        )

    assert symlink.value.errors == ("semantic_review_attestation_file_open_failed",)
    assert mapping.value.errors == ("semantic_review_attestation_file_path_invalid",)


def test_no_follow_unavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _signed_files(tmp_path, monkeypatch)
    monkeypatch.delattr(attestation_module.os, "O_NOFOLLOW")

    with pytest.raises(SemanticReviewAttestationError) as error:
        _verify(fixture)

    assert error.value.errors == ("semantic_review_attestation_file_no_follow_unavailable",)


@pytest.mark.parametrize("bad_size", [True, "1234", 1234.0])
def test_frame_evidence_rejects_numeric_and_boolean_coercion(bad_size: object) -> None:
    with pytest.raises(SemanticReviewAttestationError) as error:
        semantic_frame_evidence_digest(
            [
                {
                    "target_id": "basket_87",
                    "camera_id": "basket_e00",
                    "sha256": _digest("png"),
                    "size_bytes": bad_size,
                    "decoded_rgb_sha256": _digest("rgb"),
                }
            ]
        )

    assert error.value.errors == ("semantic_review_frame_evidence_invalid",)


def test_payload_rejects_numeric_identity_coercion() -> None:
    with pytest.raises(SemanticReviewAttestationError) as error:
        materialize_semantic_review_payload(
            attestation_id="fixture-semantic-review-001",
            selection_id="fixture-authority-freeze-001",
            authority_id=7,  # type: ignore[arg-type]
            authority_key_id="fixture-semantic-key-2026-08",
            scene_id="840873",
            target_id="basket_87",
            source_instance_id="87",
            semantic_role="destination_receptacle",
            visual_review_digest=_digest("review"),
            render_manifest_digest=_digest("render"),
            collision_topology_receipt_digest=_digest("topology"),
            cited_frames_digest=_digest("frames"),
            learned_policy_outcomes_inspected=False,
            semantic_assertions={"rigid_exterior_observed": True},
        )

    assert "semantic_review_attestation_authority_id_invalid" in error.value.errors
