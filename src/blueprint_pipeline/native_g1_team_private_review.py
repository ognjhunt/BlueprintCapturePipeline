"""One selected G1 episode on the existing private review surface.

Projection follows independent retained-output verification. It does not prove
provider settlement, grant execution authority or authorize publication.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_shared_scene_episode import team_policy_candidate_id
from .native_g1_team_policy_execution_packet import SCHEMA as PACKET_SCHEMA
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID


SCHEMA = "native_g1_team_private_review.v1"
PREFIX = "selected-worker/worker/"
MODES = frozenset({"authenticated_endpoint", "container", "noncontainer_artifact"})
FIELDS = frozenset({
    "schema_version", "status", "claim_ceiling", "intent_id", "run_id", "owner_user_id",
    "organization_id", "scene_id", "task_id", "embodiment_id", "runtime_delivery_mode",
    "container_image", "policy_delivery_mode", "execution_packet_digest",
    "policy_profile_digest", "provider_bundle_sha256", "worker_result_digest",
    "episodes", "public_redistribution_authorized", "physical_outcome_claimed",
    "simulator_result_is_physical_proof", "review_digest",
})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INTENT = re.compile(r"g1-team-policy-[0-9a-f]{64}\Z")


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _artifact(value: Any, *, prefix: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("g1_team_private_review_artifact_invalid")
    path = value.get("relative_path")
    if (
        not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or not _matches(_DIGEST, value.get("sha256"))
        or type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0
        or value["size_bytes"] > 2**53 - 1
    ):
        raise ValueError("g1_team_private_review_artifact_invalid")
    return {"relative_path": (PREFIX if prefix else "") + path,
            "sha256": value["sha256"], "size_bytes": value["size_bytes"]}


def validate_g1_team_private_review(value: Any) -> dict[str, Any]:
    """Validate the selected scope before registry writes or private ingestion."""

    if (
        not isinstance(value, Mapping) or set(value) != FIELDS
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "verified_private_development_review"
        or value.get("claim_ceiling") != "development_only"
        or value.get("embodiment_id") != G1_EMBODIMENT_ID
        or value.get("runtime_delivery_mode") != "container"
        or not isinstance(value.get("policy_delivery_mode"), str)
        or value["policy_delivery_mode"] not in MODES
        or not _matches(_INTENT, value.get("intent_id"))
        or any(not _matches(_IDENTIFIER, value.get(key))
               for key in ("run_id", "owner_user_id", "organization_id"))
        or any(not isinstance(value.get(key), str) or not value[key].strip()
               or len(value[key]) > 256 for key in ("scene_id", "task_id", "container_image"))
        or any(not _matches(_DIGEST, value.get(key)) for key in (
            "execution_packet_digest", "policy_profile_digest", "provider_bundle_sha256",
            "worker_result_digest", "review_digest",
        ))
        or any(value.get(key) is not False for key in (
            "public_redistribution_authorized", "physical_outcome_claimed",
            "simulator_result_is_physical_proof",
        ))
        or not isinstance(value.get("episodes"), list) or len(value["episodes"]) != 1
        or value["review_digest"] != digest(value, digest_field="review_digest")
    ):
        raise ValueError("g1_team_private_review_invalid")
    expected_intent = "g1-team-policy-" + digest({
        "owner": {"user_id": value["owner_user_id"], "organization_id": value["organization_id"]},
        "run_id": value["run_id"],
    }).removeprefix("sha256:")
    if value["intent_id"] != expected_intent:
        raise ValueError("g1_team_private_review_intent_invalid")
    episode = value["episodes"][0]
    if (
        not isinstance(episode, Mapping) or set(episode) != {
            "candidate_id", "objective_id", "policy_query_count", "score",
            "frame_manifest", "review_videos",
        }
        or episode.get("candidate_id") != team_policy_candidate_id(value["policy_profile_digest"])
        or not isinstance(episode.get("objective_id"), str)
        or episode["objective_id"] not in {"task_success", "g1_navigation_goal"}
        or type(episode.get("policy_query_count")) is not int
        or not 0 < episode["policy_query_count"] <= 2**53 - 1
        or not isinstance(episode.get("score"), Mapping)
        or episode["score"].get("status") != "scored"
        or not isinstance(episode["score"].get("outcome"), str)
        or not episode["score"]["outcome"]
        or not _matches(_DIGEST, episode["score"].get("episode_result_digest"))
        or episode["score"].get("score_digest")
        != digest(episode["score"], digest_field="score_digest")
        or not isinstance(episode.get("review_videos"), Mapping)
        or set(episode["review_videos"]) != {"head", "overview"}
    ):
        raise ValueError("g1_team_private_review_episode_invalid")
    artifacts = [episode.get("frame_manifest"), *episode["review_videos"].values()]
    paths = []
    for row in artifacts:
        parsed = _artifact(row)
        if parsed != row or not parsed["relative_path"].startswith(PREFIX):
            raise ValueError("g1_team_private_review_artifact_invalid")
        paths.append(parsed["relative_path"])
    if len(set(paths)) != 3:
        raise ValueError("g1_team_private_review_duplicate_artifact")
    return dict(value)


def project_g1_team_private_review(
    *, verification: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one independently verified worker; never relabel it as a campaign."""

    request = execution_packet.get("request")
    setup = execution_packet.get("trusted_setup")
    if (
        execution_packet.get("schema_version") != PACKET_SCHEMA
        or execution_packet.get("packet_digest")
        != digest(execution_packet, digest_field="packet_digest")
        or not isinstance(request, Mapping) or not isinstance(setup, Mapping)
        or not isinstance(request.get("policy_profile"), Mapping)
        or not isinstance(request.get("owner"), Mapping)
    ):
        raise ValueError("g1_team_private_review_packet_invalid")
    profile = request["policy_profile"]
    delivery = profile.get("delivery")
    profile_digest = profile.get("profile_digest")
    if (
        verification.get("schema_version") != "native_g1_team_paid_output_verification.v1"
        or verification.get("status") != "verified_development_only"
        or verification.get("claim_ceiling") != "development_only"
        or verification.get("public_redistribution_authorized") is not False
        or verification.get("execution_packet_digest") != execution_packet["packet_digest"]
        or verification.get("profile_digest") != profile_digest
        or not _matches(_DIGEST, profile_digest)
        or verification.get("candidate_id") != team_policy_candidate_id(profile_digest)
        or verification.get("objective_id") != request.get("objective_id")
        or not isinstance(delivery, Mapping)
        or any(bundle.get(key) != expected for key, expected in {
            "intent_id": execution_packet.get("intent_id"),
            "execution_packet_digest": execution_packet["packet_digest"],
            "policy_profile_digest": profile_digest, "delivery_mode": delivery.get("mode"),
            "objective_id": request.get("objective_id"),
            "scene_id": setup.get("scene_id"), "task_id": setup.get("task_id"),
        }.items())
        or not isinstance(verification.get("score"), Mapping)
        or not isinstance(verification.get("media"), Mapping)
    ):
        raise ValueError("g1_team_private_review_unbound_evidence")
    media = verification["media"]
    videos = media.get("review_videos")
    if not isinstance(videos, Mapping) or set(videos) != {"head", "overview"}:
        raise ValueError("g1_team_private_review_media_invalid")
    score = {**verification["score"], "episode_result_digest": verification.get("episode_result_digest")}
    score["score_digest"] = digest(score, digest_field="score_digest")
    review = {
        "schema_version": SCHEMA, "status": "verified_private_development_review",
        "claim_ceiling": "development_only", "intent_id": execution_packet.get("intent_id"),
        "run_id": request.get("run_id"),
        "owner_user_id": request["owner"].get("user_id"),
        "organization_id": request["owner"].get("organization_id"),
        "scene_id": bundle.get("scene_id"), "task_id": bundle.get("task_id"),
        "embodiment_id": G1_EMBODIMENT_ID, "runtime_delivery_mode": "container",
        "container_image": bundle.get("container_image"), "policy_delivery_mode": delivery.get("mode"),
        "execution_packet_digest": execution_packet["packet_digest"],
        "policy_profile_digest": profile_digest, "provider_bundle_sha256": bundle.get("bundle_sha256"),
        "worker_result_digest": verification.get("worker_result_digest"),
        "episodes": [{
            "candidate_id": verification["candidate_id"], "objective_id": verification["objective_id"],
            "policy_query_count": verification.get("policy_query_count"), "score": score,
            "frame_manifest": _artifact(media.get("frame_manifest"), prefix=True),
            "review_videos": {camera: _artifact(videos[camera], prefix=True)
                              for camera in ("head", "overview")},
        }],
        "public_redistribution_authorized": False, "physical_outcome_claimed": False,
        "simulator_result_is_physical_proof": False,
    }
    review["review_digest"] = digest(review, digest_field="review_digest")
    return validate_g1_team_private_review(review)
