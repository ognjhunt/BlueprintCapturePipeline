"""Verify one team-owned G1 episode before private review delivery.

This checks retained worker, runtime, score, and lossless media bytes. Provider
teardown, official billing, and global provider-zero remain separate gates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .episode_visual_evidence import validate_multicamera_frame_manifest
from .native_g1_development_pair import _verified_review_media
from .native_g1_shared_scene_episode import team_policy_candidate_id
from .native_g1_team_policy_conformance import SCHEMA as CONFORMANCE_SCHEMA
from .native_g1_team_policy_worker import (
    FILENAME as WORKER_FILENAME,
    PRECLOSE_FILENAME,
    PRECLOSE_SCHEMA,
    SCHEMA as WORKER_SCHEMA,
)
from .native_g1_team_policy_execution_packet import SCHEMA as PACKET_SCHEMA
from .native_g1_team_runtime_session import CONFORMANCE_FILENAME, SESSION_SCHEMA
from .native_g1_team_scored_scene_episode import RESULT_FILENAME as EPISODE_FILENAME
from .native_g1_team_supervised_episode import FILENAME as SUPERVISED_FILENAME


SCHEMA = "native_g1_team_paid_output_verification.v1"


def _receipt(path: Path, *, digest_field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_team_paid_output_receipt_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get(digest_field) != canonical_digest(value, digest_field=digest_field)
    ):
        raise ValueError("g1_team_paid_output_receipt_digest_invalid")
    return value


def verify_g1_team_paid_output(
    *, output_dir: Path, execution_packet: Mapping[str, Any],
    scene_plan_digest: str, scene_packet_receipt_digest: str,
) -> dict[str, Any]:
    """Bind the approved profile to one scored episode and complete media."""

    root = Path(output_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_team_paid_output_root_invalid")
    request = execution_packet.get("request")
    approval = execution_packet.get("operator_approval")
    setup = execution_packet.get("trusted_setup")
    if (
        not isinstance(request, Mapping) or not isinstance(approval, Mapping)
        or not isinstance(setup, Mapping)
        or not isinstance(request.get("policy_profile"), Mapping)
        or execution_packet.get("schema_version") != PACKET_SCHEMA
        or execution_packet.get("packet_digest")
        != cross_runtime_canonical_digest(execution_packet, digest_field="packet_digest")
    ):
        raise ValueError("g1_team_paid_output_packet_invalid")
    profile = request["policy_profile"]
    digest = profile.get("profile_digest")
    objective = request.get("objective_id")
    candidate = team_policy_candidate_id(digest)
    worker = _receipt(root / WORKER_FILENAME, digest_field="result_digest")
    preclose = _receipt(root / PRECLOSE_FILENAME, digest_field="preclose_digest")
    if (
        worker.get("schema_version") != WORKER_SCHEMA
        or worker.get("status") != "completed_development_only"
        or worker.get("claim_ceiling") != "development_only"
        or worker.get("execution_packet_digest") != execution_packet["packet_digest"]
        or worker.get("operator_approval_digest") != approval.get("approval_digest")
        or worker.get("scene_packet_receipt_digest") != scene_packet_receipt_digest
        or worker.get("profile_digest") != digest
        or worker.get("objective_id") != objective
        or worker.get("delivery_mode") != profile.get("delivery", {}).get("mode")
        or worker.get("blocker_type") is not None
        or worker.get("ranking_eligible") is not False
        or worker.get("provider_teardown_verified") is not False
        or worker.get("official_billing_reconciled") is not False
        or worker.get("public_redistribution_authorized") is not False
        or worker.get("teardown") != {"environment": "closed", "simulator": "closed"}
        or preclose.get("schema_version") != PRECLOSE_SCHEMA
        or preclose.get("status") != "awaiting_simulator_close"
        or preclose.get("execution_packet_digest") != execution_packet["packet_digest"]
        or preclose.get("profile_digest") != digest
        or preclose.get("teardown") != {"environment": "closed", "simulator": "close_requested"}
    ):
        raise ValueError("g1_team_paid_output_worker_invalid")
    supervised_root = root / "episode"
    supervised = _receipt(supervised_root / SUPERVISED_FILENAME, digest_field="result_digest")
    runtime_root = supervised_root / "runtime"
    conformance = _receipt(runtime_root / CONFORMANCE_FILENAME, digest_field="receipt_digest")
    session = _receipt(runtime_root / (SESSION_SCHEMA + ".json"), digest_field="receipt_digest")
    scored_root = supervised_root / "episode"
    episode_path = scored_root / EPISODE_FILENAME
    episode = _receipt(episode_path, digest_field="result_digest")
    queries = episode.get("policy_query_count")
    score = episode.get("score")
    if (
        supervised.get("status") != "completed_development_only"
        or supervised.get("profile_digest") != digest
        or supervised.get("operator_approval_digest") != approval.get("approval_digest")
        or supervised.get("scored_episode_result_digest") != episode["result_digest"]
        or supervised.get("synthetic_conformance_digest") != conformance["receipt_digest"]
        or supervised.get("runtime_teardown_digest") != session["receipt_digest"]
        or supervised.get("policy_query_count") != queries
        or worker.get("supervised_episode_result_digest") != supervised["result_digest"]
        or worker.get("policy_query_count") != queries
        or conformance.get("schema_version") != CONFORMANCE_SCHEMA
        or conformance.get("status") != "synthetic_wire_compatible"
        or conformance.get("profile_digest") != digest
        or session.get("schema_version") != SESSION_SCHEMA
        or session.get("status") != "closed"
        or session.get("profile_digest") != digest
        or session.get("linked_scored_episode_result_digest") != episode["result_digest"]
        or episode.get("status") != "development_only_scored_episode"
        or episode.get("claim_ceiling") != "development_only"
        or episode.get("profile_digest") != digest
        or episode.get("owner") != request.get("owner")
        or episode.get("source_setup_digest") != setup.get("setup_digest")
        or episode.get("source_packet_receipt_digest") != setup.get("source_packet_receipt_digest")
        or episode.get("scene_plan_digest") != scene_plan_digest
        or episode.get("candidate_id") != candidate
        or episode.get("objective_id") != objective
        or episode.get("ranking_eligible") is not False
        or episode.get("physical_outcome_claimed") is not False
        or episode.get("public_redistribution_authorized") is not False
        or type(queries) is not int or queries < 1
        or not isinstance(score, Mapping) or score.get("status") != "scored"
    ):
        raise ValueError("g1_team_paid_output_episode_invalid")
    media = _verified_review_media(episode_path, episode=episode, pair_root=root)
    manifest_path = root / media["frame_manifest"]["relative_path"]
    validate_multicamera_frame_manifest(
        _receipt(manifest_path, digest_field="frame_manifest_digest"),
        output_dir=scored_root, verify_files=True,
    )
    return {
        "schema_version": SCHEMA,
        "status": "verified_development_only",
        "claim_ceiling": "development_only",
        "execution_packet_digest": execution_packet["packet_digest"],
        "profile_digest": digest,
        "candidate_id": candidate,
        "objective_id": objective,
        "worker_result_digest": worker["result_digest"],
        "supervised_result_digest": supervised["result_digest"],
        "episode_result_digest": episode["result_digest"],
        "policy_query_count": queries,
        "score": dict(score),
        "media": media,
        "provider_teardown_verified": False,
        "official_billing_reconciled": False,
        "public_redistribution_authorized": False,
    }
