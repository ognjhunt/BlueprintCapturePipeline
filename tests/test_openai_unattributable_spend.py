"""Contract tests for conservative OpenAI spend closure.

An OpenAI run that was executed without a pre-run zero-baseline reservation can
never be given an official per-run cost: ``OpenAIProjectCandidateCostAuthority``
refuses to reserve once the attribution window is non-zero, so the attribution
is structurally unavailable rather than merely late.  This module pins the only
honest closure -- reserve the full authority cap, name the reason, and never
present the result as a final cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.openai_unattributable_spend import (
    ATTRIBUTION_UNAVAILABLE_REASONS,
    RESERVATION_SCHEMA_VERSION,
    OpenAIUnattributableSpendError,
    materialize_openai_unattributable_spend,
)


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _manifest(
    tmp_path: Path,
    *,
    reserved: float = 1.0,
    in_flight: int = 0,
    count: int = 1,
    name: str = "manifest",
) -> Path:
    # Distinct filenames matter: `_call` evaluates its own default
    # `_manifest(tmp_path)` after the caller's override has already been
    # constructed, so a shared path would silently clobber the variant the
    # test just built.
    manifest: dict[str, Any] = {
        "schema_version": "inference_reservation_manifest.v1",
        "run_id": "sam31-ai-visual-review-abc123",
        "reservations": [
            {
                "reservation_id": f"sha256:{'a' * 64}",
                "reservation_digest": f"sha256:{'b' * 64}",
                "reservation_path": "inference_reservations/reserved/a.json",
                "projected_max_cost_usd": reserved,
                "status": "in_flight_unknown" if in_flight else "completed",
                "completion_digest": None if in_flight else f"sha256:{'c' * 64}",
                "completion_path": None
                if in_flight
                else "inference_reservations/completed/a.json",
            }
        ]
        * count,
        "reservation_count": count,
        "in_flight_unknown_count": in_flight,
        "reserved_max_cost_usd": reserved,
        "proof_effect": "none",
    }
    manifest["inference_reservation_manifest_digest"] = canonical_digest(
        manifest, digest_field="inference_reservation_manifest_digest"
    )
    return _write(tmp_path / f"{name}.json", manifest)


def _call(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "family_id": "openai_sam31_ai_visual_review",
        "run_id": "sam31-ai-visual-review-abc123",
        "reservation_manifest_path": _manifest(tmp_path),
        "authority_cap_usd": 1.0,
        "model_id": "gpt-5.6-terra",
        "attribution_unavailable_reason": "no_pre_run_zero_baseline",
        "output_path": tmp_path / "out" / "reservation.json",
    }
    kwargs.update(overrides)
    return materialize_openai_unattributable_spend(**kwargs)


def test_reserves_the_full_cap_and_never_marks_cost_final(tmp_path: Path) -> None:
    receipt = _call(tmp_path)
    assert receipt["schema_version"] == RESERVATION_SCHEMA_VERSION
    assert receipt["reserved_spend_usd"] == 1.0
    assert receipt["cost_is_final"] is False
    assert receipt["official_per_run_cost_available"] is False
    assert receipt["candidate_reported_cost_accepted"] is False
    assert receipt["attribution_unavailable_reason"] == "no_pre_run_zero_baseline"
    assert receipt["proof_effect"] == "none"


def test_receipt_is_digest_sealed(tmp_path: Path) -> None:
    receipt = _call(tmp_path)
    assert receipt["reservation_digest"] == canonical_digest(
        receipt, digest_field="reservation_digest"
    )


def test_receipt_is_written_to_disk(tmp_path: Path) -> None:
    receipt = _call(tmp_path)
    written = json.loads((tmp_path / "out" / "reservation.json").read_text("utf-8"))
    assert written == receipt


def test_refuses_an_estimate_below_the_cap(tmp_path: Path) -> None:
    """The whole point is to refuse to guess a smaller number."""

    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(tmp_path, reserved_spend_usd=0.42)
    assert "openai_unattributable_spend_estimate_forbidden" in str(exc.value)


def test_refuses_when_a_reservation_is_still_in_flight(tmp_path: Path) -> None:
    """Unknown outstanding spend cannot be closed at any number."""

    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(
            tmp_path,
            reservation_manifest_path=_manifest(tmp_path, in_flight=1, name="inflight"),
        )
    assert "openai_unattributable_spend_in_flight_reservation" in str(exc.value)


def test_refuses_an_unlisted_reason(tmp_path: Path) -> None:
    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(tmp_path, attribution_unavailable_reason="because_i_said_so")
    assert "openai_unattributable_spend_reason_invalid" in str(exc.value)


def test_admitted_reasons_are_the_two_structural_causes(tmp_path: Path) -> None:
    assert ATTRIBUTION_UNAVAILABLE_REASONS == frozenset(
        {"no_pre_run_zero_baseline", "shared_api_key_scope"}
    )


def test_refuses_a_tampered_manifest_digest(tmp_path: Path) -> None:
    path = _manifest(tmp_path, name="tampered")
    value = json.loads(path.read_text("utf-8"))
    value["reserved_max_cost_usd"] = 0.01
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(tmp_path, reservation_manifest_path=path)
    assert "openai_unattributable_spend_manifest_digest_invalid" in str(exc.value)


def test_refuses_when_the_manifest_exceeds_the_authority_cap(tmp_path: Path) -> None:
    """A run that reserved more than its authority permitted is not closeable."""

    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(
            tmp_path,
            reservation_manifest_path=_manifest(tmp_path, reserved=5.0, name="oversized"),
            authority_cap_usd=1.0,
        )
    assert "openai_unattributable_spend_exceeds_authority" in str(exc.value)


def test_refuses_a_run_id_that_does_not_match_the_manifest(tmp_path: Path) -> None:
    with pytest.raises(OpenAIUnattributableSpendError) as exc:
        _call(tmp_path, run_id="some-other-run")
    assert "openai_unattributable_spend_run_mismatch" in str(exc.value)


def test_records_the_model_that_incurred_the_spend(tmp_path: Path) -> None:
    receipt = _call(tmp_path)
    assert receipt["model_id"] == "gpt-5.6-terra"


def test_receipt_states_the_structural_cause_in_words(tmp_path: Path) -> None:
    """A reader must not have to infer why no official cost exists."""

    receipt = _call(tmp_path)
    assert "zero-baseline" in receipt["explanation"]
    assert receipt["remediation"]
