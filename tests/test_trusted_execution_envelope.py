from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import trusted_execution_envelope as envelope_module
from blueprint_pipeline.trusted_execution_envelope import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    TrustedExecutionEnvelopeError,
    canonical_trusted_execution_envelope_bytes,
    materialize_trusted_execution_envelope,
    materialize_trusted_execution_payload,
    trusted_execution_signature_message,
    verify_trusted_execution_envelope,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _sign(payload: dict, private_key: Ed25519PrivateKey) -> dict:
    public_key = _public_bytes(private_key)
    return materialize_trusted_execution_envelope(
        payload=payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            private_key.sign(trusted_execution_signature_message(payload))
        ).decode("ascii"),
    )


@dataclass(frozen=True)
class SignedFixture:
    envelope_path: Path
    return_zip_path: Path
    envelope: dict
    payload: dict
    fingerprint: str
    expected: dict


@pytest.fixture
def signed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SignedFixture:
    returned = tmp_path / "native-return.zip"
    with zipfile.ZipFile(returned, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("receipt.json", b'{"status":"fixture"}\n')
    returned_bytes = returned.read_bytes()
    lifecycle = {
        "admission_receipt": _digest("admission"),
        "allocation_receipt": _digest("allocation"),
        "teardown_receipt": _digest("teardown"),
        "watchdog_receipt": _digest("watchdog"),
    }
    payload = materialize_trusted_execution_payload(
        nonce="runner-nonce-20260810-0001",
        run_digest=_digest("run"),
        package_digest=_digest("package"),
        execution_request_digest=_digest("execution-request"),
        worker_entrypoint="python -m blueprint_pipeline.native_worker",
        worker_source_tree_digest=_digest("source-tree"),
        worker_container_digest=_digest("container"),
        instance_id="vast-instance-1234",
        return_zip_sha256="sha256:" + hashlib.sha256(returned_bytes).hexdigest(),
        return_zip_size_bytes=len(returned_bytes),
        started_at="2026-08-10T10:00:00Z",
        ended_at="2026-08-10T10:02:30.125000Z",
        allocator_lifecycle_artifact_digests=lifecycle,
    )
    private_key = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)
    envelope = _sign(payload, private_key)
    envelope_path = tmp_path / "trusted-execution-envelope.json"
    envelope_path.write_bytes(canonical_trusted_execution_envelope_bytes(envelope))
    fingerprint = "sha256:" + hashlib.sha256(_public_bytes(private_key)).hexdigest()
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, fingerprint)
    expected = {
        "return_zip_path": returned,
        "expected_nonce": payload["nonce"],
        "expected_run_digest": payload["run_digest"],
        "expected_package_digest": payload["package_digest"],
        "expected_execution_request_digest": payload["execution_request_digest"],
        "expected_worker_entrypoint": payload["worker"]["entrypoint"],
        "expected_worker_source_tree_digest": payload["worker"]["source_tree_digest"],
        "expected_worker_container_digest": payload["worker"]["container_digest"],
        "expected_instance_id": payload["instance_id"],
        "expected_allocator_lifecycle_artifact_digests": lifecycle,
    }
    return SignedFixture(
        envelope_path=envelope_path,
        return_zip_path=returned,
        envelope=envelope,
        payload=payload,
        fingerprint=fingerprint,
        expected=expected,
    )


def _verify(fixture: SignedFixture, **overrides: object) -> dict:
    arguments = dict(fixture.expected)
    arguments.update(overrides)
    return verify_trusted_execution_envelope(fixture.envelope_path, **arguments)


def _write_unchecked(path: Path, envelope: dict) -> None:
    path.write_bytes(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def test_verifies_exact_runner_signed_structure_and_return_bytes(
    signed_fixture: SignedFixture,
) -> None:
    result = _verify(signed_fixture)

    assert result["status"] == "verified"
    assert result["structural_trust_verified"] is True
    assert result["signature_cryptographically_valid"] is True
    assert result["configured_runner_key_matched"] is True
    assert result["blockers"] == []
    assert result["claim_scope"] == "signed_runner_execution_structure_only"
    assert result["does_not_establish"] == [
        "allocator_lifecycle_semantics",
        "provider_zero",
        "native_simulator_gate_outcomes",
        "task_or_policy_success",
        "physical_truth",
    ]
    assert result["envelope_artifact"]["opened_once_no_follow"] is True
    assert result["return_zip_artifact"] == {
        "path": str(signed_fixture.return_zip_path),
        "sha256": signed_fixture.payload["return_zip"]["sha256"],
        "size_bytes": signed_fixture.payload["return_zip"]["size_bytes"],
        "opened_once_no_follow": True,
    }


def test_verifier_opens_each_final_file_once_with_no_follow(
    signed_fixture: SignedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open
    calls: list[tuple[Path, int]] = []

    def recording_open(path: str | bytes | os.PathLike[str], flags: int, *args: int) -> int:
        calls.append((Path(path), flags))
        return real_open(path, flags, *args)

    monkeypatch.setattr(envelope_module.os, "open", recording_open)
    result = _verify(signed_fixture)

    assert result["status"] == "verified"
    assert [path for path, _ in calls].count(signed_fixture.envelope_path) == 1
    assert [path for path, _ in calls].count(signed_fixture.return_zip_path) == 1
    assert all(flags & os.O_NOFOLLOW for _, flags in calls)


def test_verifier_has_no_caller_supplied_trust_root_override() -> None:
    parameters = inspect.signature(verify_trusted_execution_envelope).parameters

    assert "trusted_public_key_sha256" not in parameters


def test_no_follow_unavailable_fails_closed(
    signed_fixture: SignedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(envelope_module.os, "O_NOFOLLOW")

    result = _verify(signed_fixture)

    assert "trusted_execution_envelope_file_no_follow_unavailable" in result["blockers"]
    assert "trusted_execution_envelope_return_zip_no_follow_unavailable" in result["blockers"]


def test_file_identity_change_during_single_descriptor_read_fails_closed(
    signed_fixture: SignedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fstat = os.fstat
    observations: dict[tuple[int, int], int] = {}

    def drifting_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        observed = real_fstat(descriptor)
        identity = (observed.st_dev, observed.st_ino)
        observations[identity] = observations.get(identity, 0) + 1
        if observations[identity] == 1:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_mtime_ns=observed.st_mtime_ns,
            st_size=observed.st_size + 1,
        )

    monkeypatch.setattr(envelope_module.os, "fstat", drifting_fstat)

    result = _verify(signed_fixture)

    assert "trusted_execution_envelope_file_changed_while_reading" in result["blockers"]
    assert "trusted_execution_envelope_return_zip_changed_while_reading" in result["blockers"]


def test_unconfigured_and_wrong_trusted_keys_fail_closed(
    signed_fixture: SignedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TRUSTED_PUBLIC_KEY_SHA256_ENV)
    unconfigured = _verify(signed_fixture)
    monkeypatch.setenv(TRUSTED_PUBLIC_KEY_SHA256_ENV, _digest("different-trusted-key"))
    wrong = _verify(signed_fixture)

    assert unconfigured["status"] == "blocked"
    assert (
        "trusted_execution_envelope_trusted_public_key_not_configured" in unconfigured["blockers"]
    )
    assert unconfigured["signature_cryptographically_valid"] is True
    assert "trusted_execution_envelope_public_key_not_authorized" in wrong["blockers"]
    assert wrong["configured_runner_key_matched"] is False


def test_untrusted_but_valid_signer_cannot_replace_configured_runner(
    signed_fixture: SignedFixture,
) -> None:
    attacker = Ed25519PrivateKey.from_private_bytes(b"\x29" * 32)
    attacked = _sign(signed_fixture.payload, attacker)
    _write_unchecked(signed_fixture.envelope_path, attacked)

    result = _verify(signed_fixture)

    assert result["signature_cryptographically_valid"] is True
    assert result["structural_trust_verified"] is False
    assert "trusted_execution_envelope_public_key_not_authorized" in result["blockers"]


def test_modified_signature_fails_cryptographic_verification(
    signed_fixture: SignedFixture,
) -> None:
    attacked = copy.deepcopy(signed_fixture.envelope)
    attacked["signature"]["signature_base64"] = base64.b64encode(b"\x00" * 64).decode()
    _write_unchecked(signed_fixture.envelope_path, attacked)

    result = _verify(signed_fixture)

    assert result["status"] == "blocked"
    assert result["signature_cryptographically_valid"] is False
    assert "trusted_execution_envelope_signature_verification_failed" in result["blockers"]


@pytest.mark.parametrize(
    ("field", "wrong_value", "blocker"),
    [
        (
            "expected_nonce",
            "different-runner-nonce-0001",
            "trusted_execution_envelope_nonce_mismatch",
        ),
        (
            "expected_run_digest",
            _digest("wrong-run"),
            "trusted_execution_envelope_run_digest_mismatch",
        ),
        (
            "expected_package_digest",
            _digest("wrong-package"),
            "trusted_execution_envelope_package_digest_mismatch",
        ),
        (
            "expected_execution_request_digest",
            _digest("wrong-request"),
            "trusted_execution_envelope_execution_request_digest_mismatch",
        ),
        (
            "expected_worker_entrypoint",
            "python -m attacker.worker",
            "trusted_execution_envelope_worker_entrypoint_mismatch",
        ),
        (
            "expected_worker_source_tree_digest",
            _digest("wrong-source-tree"),
            "trusted_execution_envelope_worker_source_tree_digest_mismatch",
        ),
        (
            "expected_worker_container_digest",
            _digest("wrong-container"),
            "trusted_execution_envelope_worker_container_digest_mismatch",
        ),
        (
            "expected_instance_id",
            "vast-instance-9999",
            "trusted_execution_envelope_instance_id_mismatch",
        ),
    ],
)
def test_expected_identity_binding_mismatches_fail_closed(
    signed_fixture: SignedFixture,
    field: str,
    wrong_value: str,
    blocker: str,
) -> None:
    result = _verify(signed_fixture, **{field: wrong_value})

    assert result["status"] == "blocked"
    assert blocker in result["blockers"]
    assert result["signature_cryptographically_valid"] is True


def test_return_zip_digest_and_size_are_derived_from_exact_opened_bytes(
    signed_fixture: SignedFixture,
) -> None:
    signed_fixture.return_zip_path.write_bytes(
        signed_fixture.return_zip_path.read_bytes() + b"tampered"
    )

    result = _verify(signed_fixture)

    assert "trusted_execution_envelope_return_zip_sha256_mismatch" in result["blockers"]
    assert "trusted_execution_envelope_return_zip_size_mismatch" in result["blockers"]
    assert result["signature_cryptographically_valid"] is True


def test_allocator_lifecycle_digest_inventory_is_exact(
    signed_fixture: SignedFixture,
) -> None:
    expected = dict(signed_fixture.expected["expected_allocator_lifecycle_artifact_digests"])
    expected["watchdog_receipt"] = _digest("wrong-watchdog")

    result = _verify(
        signed_fixture,
        expected_allocator_lifecycle_artifact_digests=expected,
    )

    assert "trusted_execution_envelope_allocator_lifecycle_digests_mismatch" in result["blockers"]


def test_invalid_expected_lifecycle_configuration_fails_closed(
    signed_fixture: SignedFixture,
) -> None:
    result = _verify(
        signed_fixture,
        expected_allocator_lifecycle_artifact_digests=None,
    )

    assert result["status"] == "blocked"
    assert "trusted_execution_envelope_expected_lifecycle_digests_invalid" in result["blockers"]


@pytest.mark.parametrize(
    "lifecycle",
    [
        {},
        {f"receipt_{index}": _digest(f"receipt-{index}") for index in range(33)},
    ],
)
def test_lifecycle_digest_cardinality_is_bounded_and_nonempty(
    signed_fixture: SignedFixture,
    lifecycle: dict[str, str],
) -> None:
    payload = copy.deepcopy(signed_fixture.payload)
    payload["allocator_lifecycle_artifact_digests"] = lifecycle

    with pytest.raises(TrustedExecutionEnvelopeError) as caught:
        trusted_execution_signature_message(payload)

    assert "trusted_execution_envelope_allocator_lifecycle_digests_invalid" in caught.value.errors


def test_recomputed_unkeyed_payload_digest_cannot_replace_signature(
    signed_fixture: SignedFixture,
) -> None:
    attacked = copy.deepcopy(signed_fixture.envelope)
    attacked["payload"]["package_digest"] = _digest("attacker-package")
    attacked["signature"]["signed_payload_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(attacked["payload"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    _write_unchecked(signed_fixture.envelope_path, attacked)

    result = _verify(
        signed_fixture,
        expected_package_digest=attacked["payload"]["package_digest"],
    )

    assert result["status"] == "blocked"
    assert "trusted_execution_envelope_signature_verification_failed" in result["blockers"]
    assert "trusted_execution_envelope_package_digest_mismatch" not in result["blockers"]


def test_symlink_and_noncanonical_envelope_bytes_are_rejected(
    signed_fixture: SignedFixture,
    tmp_path: Path,
) -> None:
    symlink = tmp_path / "envelope-link.json"
    symlink.symlink_to(signed_fixture.envelope_path)
    symlinked = verify_trusted_execution_envelope(symlink, **signed_fixture.expected)
    signed_fixture.envelope_path.write_text(
        json.dumps(signed_fixture.envelope, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    noncanonical = _verify(signed_fixture)

    assert "trusted_execution_envelope_file_unreadable" in symlinked["blockers"]
    assert "trusted_execution_envelope_encoding_not_canonical" in noncanonical["blockers"]


def test_duplicate_json_keys_are_rejected_before_signature_use(
    signed_fixture: SignedFixture,
) -> None:
    canonical = json.dumps(signed_fixture.envelope, sort_keys=True, separators=(",", ":"))
    duplicate = '{"schema_version":"trusted_execution_envelope.v1",' + canonical[1:] + "\n"
    signed_fixture.envelope_path.write_text(duplicate, encoding="utf-8")

    result = _verify(signed_fixture)

    assert "trusted_execution_envelope_json_invalid" in result["blockers"]
    assert result["signature_cryptographically_valid"] is False


def test_payload_materializer_rejects_bad_time_order_and_extra_fields(
    signed_fixture: SignedFixture,
) -> None:
    payload = copy.deepcopy(signed_fixture.payload)
    payload["ended_at"] = "2026-08-10T09:00:00Z"
    payload["unsigned_success"] = True

    with pytest.raises(TrustedExecutionEnvelopeError) as caught:
        trusted_execution_signature_message(payload)

    assert caught.value.errors == (
        "trusted_execution_envelope_payload_fields_invalid",
        "trusted_execution_envelope_time_order_invalid",
    )
