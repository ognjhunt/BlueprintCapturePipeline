"""Seal a byte-exact replay of an Aura exact-residual visual abstention."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_aura_exact_residual_visual_abstention import (
    RECEIPT_SCHEMA as ABSTENTION_SCHEMA,
    materialize_aura_exact_residual_visual_abstention,
)


REPLAY_SCHEMA = "public_scene_aura_exact_residual_visual_abstention_replay.v1"


class AuraExactResidualVisualAbstentionReplayError(ValueError):
    """Stable failure for an incomplete or non-reproducible abstention."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file(value: str | Path, *, role: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise AuraExactResidualVisualAbstentionReplayError(
            f"aura_exact_residual_visual_replay_symlink:{role}"
        )
    path = unresolved.resolve()
    if not path.is_file():
        raise AuraExactResidualVisualAbstentionReplayError(
            f"aura_exact_residual_visual_replay_file_missing:{role}"
        )
    return path


def _read(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExactResidualVisualAbstentionReplayError(
            f"aura_exact_residual_visual_replay_json_invalid:{role}"
        ) from exc
    if not isinstance(value, dict):
        raise AuraExactResidualVisualAbstentionReplayError(
            f"aura_exact_residual_visual_replay_json_invalid:{role}"
        )
    return value


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def materialize_aura_exact_residual_visual_abstention_replay(
    *,
    request_path: str | Path,
    composite_receipt_path: str | Path,
    sealed_abstention_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Rebuild the reject-only receipt and seal an exact equality proof."""

    request = _file(request_path, role="request")
    composite = _file(composite_receipt_path, role="composite")
    sealed_path = _file(sealed_abstention_path, role="sealed_abstention")
    sealed = _read(sealed_path, role="sealed_abstention")
    if (
        sealed.get("schema_version") != ABSTENTION_SCHEMA
        or sealed.get("receipt_digest")
        != canonical_digest(sealed, digest_field="receipt_digest")
        or (sealed.get("claim_boundary") or {}).get("visual_rejection_only")
        is not True
        or (sealed.get("claim_boundary") or {}).get(
            "native_simulator_qualified"
        )
        is not False
    ):
        raise AuraExactResidualVisualAbstentionReplayError(
            "aura_exact_residual_visual_replay_sealed_receipt_invalid"
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="adp-aura-visual-abstention-replay-"
        ) as directory:
            replayed_path = Path(directory) / "replayed.json"
            replayed = materialize_aura_exact_residual_visual_abstention(
                request_path=request,
                composite_receipt_path=composite,
                output_path=replayed_path,
            )
            replayed_bytes = replayed_path.read_bytes()
    except Exception as exc:
        raise AuraExactResidualVisualAbstentionReplayError(
            "aura_exact_residual_visual_replay_execution_failed"
        ) from exc

    sealed_bytes = sealed_path.read_bytes()
    if replayed != sealed or replayed_bytes != sealed_bytes:
        raise AuraExactResidualVisualAbstentionReplayError(
            "aura_exact_residual_visual_replay_not_byte_exact"
        )

    receipt: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "status": "byte_exact_visual_abstention_replay_verified",
        "bindings": {
            "review_request": _record(request),
            "exact_composite_receipt": {
                **_record(composite),
                "composite_digest": _read(composite, role="composite").get(
                    "composite_digest"
                ),
            },
            "sealed_visual_abstention": {
                **_record(sealed_path),
                "receipt_digest": sealed["receipt_digest"],
            },
        },
        "replay_result": {
            "byte_exact_match": True,
            "replayed_sha256": _sha256(sealed_path),
            "replayed_receipt_digest": replayed["receipt_digest"],
        },
        "automatic_paid_retry_executed": False,
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "claim_boundary": {
            "replay_verification_only": True,
            "visual_candidate_admitted": False,
            "native_simulator_qualified": False,
            "policy_or_physical_claim": False,
        },
        "replay_digest": "",
    }
    receipt["replay_digest"] = canonical_digest(
        receipt, digest_field="replay_digest"
    )
    output = Path(output_path).expanduser()
    if output.is_symlink() or output.exists():
        raise AuraExactResidualVisualAbstentionReplayError(
            "aura_exact_residual_visual_replay_output_exists"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--sealed-abstention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_exact_residual_visual_abstention_replay(
        request_path=args.request,
        composite_receipt_path=args.composite_receipt,
        sealed_abstention_path=args.sealed_abstention,
        output_path=args.output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuraExactResidualVisualAbstentionReplayError",
    "REPLAY_SCHEMA",
    "main",
    "materialize_aura_exact_residual_visual_abstention_replay",
]
