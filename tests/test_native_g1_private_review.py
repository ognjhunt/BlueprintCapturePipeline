"""Private G1 review handoff preserves verified media and claim limits."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.native_g1_development_pair import PAIR_ORDER
from blueprint_pipeline import native_g1_private_review as review_module
from blueprint_pipeline.native_g1_private_review import project_g1_private_review


def _inputs() -> tuple[dict, dict]:
    digest = "sha256:" + "a" * 64
    artifact = {
        "relative_path": "candidate/episode/media/review.mp4",
        "sha256": digest,
        "size_bytes": 42,
    }
    episodes = []
    for index, candidate in enumerate(PAIR_ORDER):
        episodes.append(
            {
                "candidate_id": candidate,
                "objective_id": "task_success" if index < 2 else "g1_navigation_goal",
                "policy_query_count": index + 1,
                "score": {
                    "score_digest": digest,
                    "episode_result_digest": digest,
                    "outcome": "task_succeeded",
                },
                "frame_manifest": {
                    **artifact,
                    "relative_path": "candidate/episode/media/frames.json",
                },
                "review_videos": {"head": artifact, "overview": artifact},
            }
        )
    verification = {
        "status": "verified_development_only",
        "campaign_plan_digest": digest,
        "terminal_result_digest": digest,
        "episodes": episodes,
        "public_redistribution_authorized": False,
        "physical_outcome_claimed": False,
    }
    bundle = {
        "scene_id": "interiorgs-841757",
        "task_id": "scene-841757-book-to-marked-area",
        "candidate_ids": list(PAIR_ORDER),
        "campaign_plan_digest": digest,
        "container_image": "pinned-isaac-image",
        "bundle_sha256": digest,
    }
    return verification, bundle


def test_private_review_retains_four_scores_and_relative_media() -> None:
    verification, bundle = _inputs()
    review = project_g1_private_review(verification=verification, bundle=bundle)
    assert review["status"] == "verified_private_development_review"
    assert review["claim_ceiling"] == "development_only"
    assert review["public_redistribution_authorized"] is False
    assert review["physical_outcome_claimed"] is False
    assert [row["candidate_id"] for row in review["episodes"]] == list(PAIR_ORDER)
    assert [row["policy_query_count"] for row in review["episodes"]] == [1, 2, 3, 4]
    assert review["episodes"][0]["review_videos"]["head"]["relative_path"].startswith(
        "manipulation_pair/"
    )
    assert review["episodes"][3]["frame_manifest"]["relative_path"].startswith("movement_pair/")
    assert review["review_digest"] == cross_runtime_canonical_digest(
        review, digest_field="review_digest"
    )


def test_private_review_digest_uses_website_number_encoding() -> None:
    verification, bundle = _inputs()
    verification["episodes"][0]["score"]["progress_score"] = 1.0
    review = project_g1_private_review(verification=verification, bundle=bundle)
    assert review["review_digest"] == cross_runtime_canonical_digest(
        review, digest_field="review_digest"
    )


@pytest.mark.parametrize(
    "change", ["missing_video", "wrong_order", "public", "physical", "path_traversal"]
)
def test_private_review_rejects_unverified_or_unsafe_input(change: str) -> None:
    verification, bundle = _inputs()
    verification = deepcopy(verification)
    if change == "missing_video":
        del verification["episodes"][0]["review_videos"]["head"]
    elif change == "wrong_order":
        verification["episodes"].reverse()
    elif change == "public":
        verification["public_redistribution_authorized"] = True
    elif change == "physical":
        verification["physical_outcome_claimed"] = True
    else:
        verification["episodes"][0]["frame_manifest"]["relative_path"] = "../escape.json"
    with pytest.raises(ValueError, match="g1_private_review"):
        project_g1_private_review(verification=verification, bundle=bundle)


def test_private_review_backfills_older_adapter_result_from_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verification, bundle = _inputs()
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        json.dumps(
            {
                "status": "completed",
                "g1_output_verification": {"status": "verified_development_only", "episodes": []},
            }
        )
    )
    receipt = tmp_path / "bundle.json"
    receipt.write_text(json.dumps({"implementation_commit": "a" * 40}))
    output = tmp_path / "review.json"
    from blueprint_pipeline import native_g1_paid_campaign as lane
    from blueprint_pipeline import native_g1_provider_bundle as bundle_module

    monkeypatch.setattr(bundle_module, "load_verified_g1_provider_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(lane, "verify_g1_paid_output", lambda *_a, **_k: verification)
    assert (
        review_module.main(
            [
                "--adapter-result",
                str(adapter),
                "--bundle-receipt",
                str(receipt),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["review_digest"] == cross_runtime_canonical_digest(
        json.loads(output.read_text()), digest_field="review_digest"
    )
