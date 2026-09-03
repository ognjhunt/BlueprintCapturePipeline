"""Build an immutable Website sidecar for a historical policy-canary interpretation.

The original publication and deterministic scores remain untouched.  This
module binds a newly projected set of learned interpretation receipts to the
exact already-published run, delivery, optional score-correction sidecar, and
twenty episode identities.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest


SCHEMA_VERSION = "task_evaluation_policy_canary_episode_interpretation_sidecar.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class PolicyCanaryEpisodeInterpretationBackfillError(ValueError):
    """Stable failure for a sidecar that cannot be bound exactly."""


def _mapping(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyCanaryEpisodeInterpretationBackfillError(code)
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _publication(value: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    root = _mapping(value, code="interpretation_backfill_source_invalid")
    publication = _mapping(
        root.get("publication", root), code="interpretation_backfill_publication_invalid"
    )
    correction = root.get("score_correction")
    correction_digest = None
    if correction is not None:
        correction = _mapping(
            correction, code="interpretation_backfill_score_correction_invalid"
        )
        correction_digest = str(correction.get("sidecar_digest") or "")
        if not _DIGEST.fullmatch(correction_digest):
            raise PolicyCanaryEpisodeInterpretationBackfillError(
                "interpretation_backfill_score_correction_invalid"
            )
    return publication, correction_digest


def _projection(value: Any, *, code: str) -> dict[str, Any]:
    projection = _mapping(value, code=code)
    if (
        projection.get("schema_version")
        != "task_evaluation_policy_canary_result_projection.v1"
        or projection.get("run_kind") != "internal_policy_canary"
        or projection.get("claim_ceiling") != "diagnostic_policy_execution"
        or not _DIGEST.fullmatch(str(projection.get("projection_digest") or ""))
        or projection.get("projection_digest")
        != cross_runtime_canonical_digest(
            projection, digest_field="projection_digest"
        )
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(code)
    return projection


def _episode_key(value: Mapping[str, Any]) -> tuple[str, str, str, int]:
    episode_id = str(value.get("episode_id") or "")
    candidate_id = str(value.get("candidate_id") or "")
    cell_id = str(value.get("cell_id") or "")
    seed = value.get("seed")
    if (
        not _IDENTIFIER.fullmatch(episode_id)
        or candidate_id not in {"pi05_droid", "groot_n17_droid"}
        or not _IDENTIFIER.fullmatch(cell_id)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2_147_483_647
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_episode_identity_invalid"
        )
    return episode_id, candidate_id, cell_id, seed


def _interpretation(value: Any) -> dict[str, Any]:
    result = _mapping(value, code="interpretation_backfill_receipt_invalid")
    receipt = _mapping(
        result.get("receipt"), code="interpretation_backfill_receipt_invalid"
    )
    if (
        result.get("status") not in {"completed", "abstained"}
        or result.get("learned_interpretation_only") is not True
        or result.get("authoritative_task_success_unchanged") is not True
        or result.get("ranking_or_promotion_effect") != "none"
        or result.get("deterministic_agreement")
        not in {"agrees", "disagrees", "abstains"}
        or not _DIGEST.fullmatch(str(receipt.get("digest") or ""))
        or not str(receipt.get("artifact_id") or "")
        or not isinstance(receipt.get("size_bytes"), int)
        or receipt["size_bytes"] <= 0
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_receipt_invalid"
        )
    return result


def build_policy_canary_episode_interpretation_sidecar(
    *,
    source_site_record: Mapping[str, Any],
    backfill_projection: Mapping[str, Any],
    record_id: str,
    verified_at_iso: str,
) -> dict[str, Any]:
    """Bind learned explanations to one historical run without rewriting it."""

    if not _IDENTIFIER.fullmatch(record_id):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_record_id_invalid"
        )
    try:
        verified_at = datetime.fromisoformat(verified_at_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_verified_at_invalid"
        ) from exc
    if verified_at.tzinfo is None:
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_verified_at_invalid"
        )
    publication, score_correction_digest = _publication(source_site_record)
    if publication.get("schema_version") != "task_evaluation_run_publication.v4":
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_publication_invalid"
        )
    source = _projection(
        publication.get("policy_canary_result"),
        code="interpretation_backfill_source_projection_invalid",
    )
    delivery = _mapping(
        publication.get("result_delivery"),
        code="interpretation_backfill_source_delivery_invalid",
    )
    delivery_digest = str(delivery.get("delivery_digest") or "")
    if (
        not _DIGEST.fullmatch(delivery_digest)
        or source.get("result_delivery_digest") != delivery_digest
        or publication.get("run_id") != source.get("run_id")
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_source_delivery_invalid"
        )
    backfill = _projection(
        backfill_projection,
        code="interpretation_backfill_projection_invalid",
    )
    for field in (
        "run_id",
        "request_digest",
        "configuration_digest",
        "matrix_digest",
        "candidate_ids",
    ):
        if backfill.get(field) != source.get(field):
            raise PolicyCanaryEpisodeInterpretationBackfillError(
                "interpretation_backfill_scientific_identity_changed"
            )
    source_episodes = source.get("episodes")
    backfill_episodes = backfill.get("episodes")
    if (
        not isinstance(source_episodes, list)
        or not isinstance(backfill_episodes, list)
        or len(source_episodes) != 20
        or len(backfill_episodes) != 20
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_episode_inventory_invalid"
        )
    source_keys = {_episode_key(row) for row in source_episodes if isinstance(row, Mapping)}
    projected_by_key = {
        _episode_key(row): row for row in backfill_episodes if isinstance(row, Mapping)
    }
    if len(source_keys) != 20 or set(projected_by_key) != source_keys:
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_episode_inventory_invalid"
        )
    summary = _mapping(
        backfill.get("episode_interpretation"),
        code="interpretation_backfill_summary_invalid",
    )
    if (
        summary.get("schema_version")
        != "policy_canary_episode_interpretation_closeout.v1"
        or summary.get("episode_count") != 20
        or summary.get("receipt_count") != 20
        or summary.get("completed_count", 0) + summary.get("abstained_count", 0)
        != 20
        or summary.get("authoritative_deterministic_result_unchanged") is not True
        or summary.get("score_overwrite_performed") is not False
        or summary.get("ranking_or_promotion_effect") != "none"
        or not _DIGEST.fullmatch(str(summary.get("summary_digest") or ""))
    ):
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_summary_invalid"
        )
    episodes = []
    for key in sorted(source_keys):
        episode_id, candidate_id, cell_id, seed = key
        episodes.append(
            {
                "episode_id": episode_id,
                "candidate_id": candidate_id,
                "cell_id": cell_id,
                "seed": seed,
                "interpretation": _interpretation(
                    projected_by_key[key].get("interpretation")
                ),
            }
        )
    sidecar: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_binding": {
            "record_id": record_id,
            "source_run_id": source["run_id"],
            "source_projection_digest": source["projection_digest"],
            "source_delivery_digest": delivery_digest,
            "source_score_correction_sidecar_digest": score_correction_digest,
        },
        "summary": summary,
        "episodes": episodes,
        "audit": {
            "original_publication_preserved": True,
            "deterministic_scores_unchanged": True,
            "learned_interpretation_only": True,
            "ranking_or_promotion_effect": "none",
            "verified_at_iso": verified_at_iso,
        },
        "sidecar_digest": "",
    }
    sidecar["sidecar_digest"] = cross_runtime_canonical_digest(
        sidecar, digest_field="sidecar_digest"
    )
    return sidecar


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_input_invalid"
        )
    return _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        code="interpretation_backfill_input_invalid",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-site-record", type=Path, required=True)
    parser.add_argument("--backfill-projection", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--verified-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sidecar = build_policy_canary_episode_interpretation_sidecar(
        source_site_record=_read(args.source_site_record.expanduser().resolve()),
        backfill_projection=_read(args.backfill_projection.expanduser().resolve()),
        record_id=args.record_id,
        verified_at_iso=args.verified_at,
    )
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(sidecar, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise PolicyCanaryEpisodeInterpretationBackfillError(
            "interpretation_backfill_output_exists"
        ) from exc
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PolicyCanaryEpisodeInterpretationBackfillError",
    "SCHEMA_VERSION",
    "build_policy_canary_episode_interpretation_sidecar",
]
