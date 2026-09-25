"""Project verified G1 episodes into a private, embodiment-aware review record.

The record is an ingest handoff, not a publication or a claim upgrade. Media
paths stay relative to the controller's immutable execution output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_g1_development_pair import PAIR_ORDER
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID


SCHEMA = "native_g1_private_review.v1"


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _artifact(row: Mapping[str, Any], pair_name: str) -> dict[str, Any]:
    relative = row.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or not _digest(row.get("sha256"))
        or type(row.get("size_bytes")) is not int
        or row["size_bytes"] <= 0
    ):
        raise ValueError("g1_private_review_artifact_invalid")
    return {
        "relative_path": pair_name + "_pair/" + relative,
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
    }


def project_g1_private_review(
    *, verification: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a digest-bound four-episode review only from verified evidence."""

    episodes = verification.get("episodes")
    if (
        verification.get("status") != "verified_development_only"
        or verification.get("public_redistribution_authorized") is not False
        or verification.get("physical_outcome_claimed") is not False
        or verification.get("campaign_plan_digest") != bundle.get("campaign_plan_digest")
        or not _digest(verification.get("campaign_plan_digest"))
        or not _digest(verification.get("terminal_result_digest"))
        or not _digest(bundle.get("bundle_sha256"))
        or any(
            not isinstance(bundle.get(key), str) or not bundle[key]
            for key in ("scene_id", "task_id", "container_image")
        )
        or not isinstance(episodes, list)
        or len(episodes) != len(PAIR_ORDER)
        or [row.get("candidate_id") for row in episodes if isinstance(row, Mapping)]
        != list(PAIR_ORDER)
        or bundle.get("candidate_ids") != list(PAIR_ORDER)
    ):
        raise ValueError("g1_private_review_unverified_input")
    projected = []
    for index, row in enumerate(episodes):
        pair_name, objective = (
            ("manipulation", "task_success") if index < 2 else ("movement", "g1_navigation_goal")
        )
        score = row.get("score")
        videos = row.get("review_videos")
        if (
            row.get("objective_id") != objective
            or type(row.get("policy_query_count")) is not int
            or row["policy_query_count"] < 1
            or not isinstance(score, Mapping)
            or not _digest(score.get("score_digest"))
            or not _digest(score.get("episode_result_digest"))
            or not isinstance(score.get("outcome"), str)
            or not score["outcome"]
            or not isinstance(row.get("frame_manifest"), Mapping)
            or not isinstance(videos, Mapping)
            or set(videos) != {"head", "overview"}
        ):
            raise ValueError("g1_private_review_episode_invalid")
        projected.append(
            {
                "candidate_id": row["candidate_id"],
                "objective_id": objective,
                "policy_query_count": row["policy_query_count"],
                "score": dict(score),
                "frame_manifest": _artifact(row["frame_manifest"], pair_name),
                "review_videos": {
                    camera: _artifact(videos[camera], pair_name) for camera in ("head", "overview")
                },
            }
        )
    review = {
        "schema_version": SCHEMA,
        "status": "verified_private_development_review",
        "claim_ceiling": "development_only",
        "scene_id": bundle["scene_id"],
        "task_id": bundle["task_id"],
        "embodiment_id": G1_EMBODIMENT_ID,
        "runtime_delivery_mode": "container",
        "container_image": bundle["container_image"],
        "campaign_plan_digest": bundle["campaign_plan_digest"],
        "provider_bundle_sha256": bundle["bundle_sha256"],
        "terminal_result_digest": verification["terminal_result_digest"],
        "episodes": projected,
        "public_redistribution_authorized": False,
        "physical_outcome_claimed": False,
        "simulator_result_is_physical_proof": False,
    }
    # The Website verifies this digest after JSON.parse, whose number encoding
    # follows ECMAScript rather than Python's json.dumps float formatting.
    review["review_digest"] = cross_runtime_canonical_digest(
        review, digest_field="review_digest"
    )
    return review


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_private_review_input_path_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_private_review_input_invalid")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Backfill an older exact run after provider zero using its sealed bytes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-result", type=Path, required=True)
    parser.add_argument("--bundle-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .native_g1_paid_campaign import verify_g1_paid_output
    from .native_g1_provider_bundle import load_verified_g1_provider_bundle

    adapter = _read(args.adapter_result)
    receipt = _read(args.bundle_receipt)
    bundle = load_verified_g1_provider_bundle(
        args.bundle_receipt,
        expected_implementation_commit=receipt["implementation_commit"],
    )
    verified = verify_g1_paid_output(adapter, bundle)
    review = project_g1_private_review(verification=verified, bundle=bundle)
    if not args.output.is_absolute() or args.output.is_symlink():
        raise ValueError("g1_private_review_output_path_invalid")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(review, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": review["status"], "review_digest": review["review_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
