"""Reopen one selected G1 result for private post-run delivery.

This boundary proves retained input/output bytes only. Official provider charge,
fresh inventory and launch authority are distinct settlement/admission gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from .native_g1_team_private_review import project_g1_team_private_review
from .native_g1_team_provider_bundle import read_retained_g1_team_provider_bundle


@dataclass(frozen=True, repr=False)
class G1TeamReviewEvidence:
    review: Mapping[str, Any] = field(repr=False)
    verification: Mapping[str, Any] = field(repr=False)
    bundle: Mapping[str, Any] = field(repr=False)
    execution_packet: Mapping[str, Any] = field(repr=False)
    source_root: Path = field(repr=False)

    def __repr__(self) -> str:
        return "G1TeamReviewEvidence(spend_admitted=False, publication_authorized=False)"


def _json(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute() or not path.is_file()
        or any(part.is_symlink() for part in (path, *path.parents))
        or path.resolve() != path
    ):
        raise ValueError("g1_team_review_evidence_path_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_team_review_evidence_receipt_invalid")
    return value


def verify_retained_g1_team_review(
    *, adapter_result_path: Path, bundle_receipt_path: Path,
) -> G1TeamReviewEvidence:
    """Never trust a completion flag in place of independent byte verification."""

    from .native_g1_team_paid_policy import RESULT_SCHEMA, _verify_output

    adapter_path, bundle_path = Path(adapter_result_path), Path(bundle_receipt_path)
    adapter, receipt = _json(adapter_path), _json(bundle_path)
    if (
        adapter.get("schema_version") != RESULT_SCHEMA
        or adapter.get("status") != "completed"
        or adapter.get("continuing_spend_from_this_run") is not False
        or not isinstance(adapter.get("g1_team_output_verification"), dict)
    ):
        raise ValueError("g1_team_review_evidence_controller_incomplete")
    retained = read_retained_g1_team_provider_bundle(
        bundle_path, expected_implementation_commit=receipt.get("implementation_commit"),
    )
    verification = _verify_output(adapter, dict(retained.bundle), job=adapter_path.parent)
    if adapter["g1_team_output_verification"] != verification:
        raise ValueError("g1_team_review_evidence_reported_verification_changed")
    review = project_g1_team_private_review(
        verification=verification, bundle=retained.bundle,
        execution_packet=retained.execution_packet,
    )
    return G1TeamReviewEvidence(
        review=review, verification=verification, bundle=retained.bundle,
        execution_packet=retained.execution_packet,
        source_root=Path(adapter["attempt_root"]) / "immutable_execution",
    )
