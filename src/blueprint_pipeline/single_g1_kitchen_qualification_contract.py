"""Stable evidence and claim contracts for the retained G1 qualification fixture."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


RELEASE_BINDING_SCHEMA_VERSION = "single_g1_kitchen_qualification_release_binding.v1"
EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_DIGEST_REF_RE = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})")


def valid_source_commit(value: object) -> bool:
    return bool(_COMMIT_RE.fullmatch(str(value or "")))


def valid_image_binding(image_ref: object, image_digest: object) -> bool:
    match = _DIGEST_REF_RE.fullmatch(str(image_ref or ""))
    return bool(match and image_digest == f"sha256:{match.group(1)}")


def build_release_binding(
    release: Mapping[str, Any], *, expected_source_commit: str
) -> tuple[dict[str, Any], list[str]]:
    """Derive an immutable worker binding from independent release evidence."""

    blockers: list[str] = []
    expected_source_commit = str(expected_source_commit or "")
    if not valid_source_commit(expected_source_commit):
        blockers.append("qualification_expected_source_commit_invalid")
    if release.get("schema_version") != "groot_oscar_thin_remote_build_result.v1":
        blockers.append("qualification_release_evidence_schema_invalid")
    if release.get("status") != "completed":
        blockers.append("qualification_release_evidence_not_completed")
    if release.get("blockers") != []:
        blockers.append("qualification_release_evidence_has_blockers")

    source_commit = str(release.get("source_commit") or "")
    if not valid_source_commit(source_commit):
        blockers.append("qualification_release_source_commit_invalid")
    elif source_commit != expected_source_commit:
        blockers.append("qualification_release_source_commit_mismatch")
    if release.get("source_patch_sha256") != EMPTY_PATCH_SHA256:
        blockers.append("qualification_release_source_patch_not_empty")

    image_ref = str(release.get("resolved_digest_ref") or "")
    digest_match = _DIGEST_REF_RE.fullmatch(image_ref)
    if digest_match is None:
        blockers.append("qualification_release_image_not_digest_pinned")
        image_digest = ""
    else:
        image_digest = f"sha256:{digest_match.group(1)}"
    if release.get("release_image_ref") != image_ref:
        blockers.append("qualification_release_image_ref_mismatch")
    if release.get("runnable_platform") != "linux/amd64":
        blockers.append("qualification_release_platform_not_linux_amd64")
    if release.get("models_embedded") is not False:
        blockers.append("qualification_release_models_not_externalized")

    thin = release.get("thin_release_contract")
    thin = dict(thin) if isinstance(thin, Mapping) else {}
    if thin.get("schema_version") != "groot_oscar_thin_release_image_contract.v1":
        blockers.append("qualification_thin_release_contract_schema_invalid")
    if thin.get("status") != "passed" or release.get("thin_release_contract_status") != "passed":
        blockers.append("qualification_thin_release_contract_not_passed")
    if thin.get("blockers") != []:
        blockers.append("qualification_thin_release_contract_has_blockers")
    if thin.get("release_image_ref") != image_ref:
        blockers.append("qualification_thin_release_image_mismatch")
    if thin.get("models_externalized") is not True:
        blockers.append("qualification_thin_release_models_not_externalized")

    required_cuda_version = str(release.get("required_cuda_version") or "")
    required_cuda_source = str(release.get("required_cuda_version_source") or "")
    if not required_cuda_version:
        blockers.append("qualification_release_cuda_version_missing")
    if not required_cuda_source.startswith("image_config_env:"):
        blockers.append("qualification_release_cuda_source_unverified")

    binding = {
        "schema_version": RELEASE_BINDING_SCHEMA_VERSION,
        "image_ref": image_ref or None,
        "image_digest": image_digest or None,
        "source_commit": source_commit or None,
        "source_patch_sha256": release.get("source_patch_sha256"),
        "runnable_platform": release.get("runnable_platform"),
        "required_cuda_version": required_cuda_version or None,
        "required_cuda_version_source": required_cuda_source or None,
        "models_externalized": release.get("models_embedded") is False
        and thin.get("models_externalized") is True,
        "release_evidence_status": release.get("status"),
        "thin_release_contract_status": thin.get("status"),
    }
    return binding, sorted(set(blockers))


def qualification_gate_matrix() -> list[dict[str, Any]]:
    """Forward proof ledger, separating historical attempts from future gates."""

    proven_016 = "attempt016_proven"
    proven_017 = "attempt017_proven"
    pending = "pending"
    rows = (
        (
            "image_bundle_assets",
            "Exact sealed image, episode bundle, kitchen assets, and runtime overlays",
            proven_016,
            "Exact image/bundle binding, startup asset gate, and image healthcheck passed.",
            proven_017,
            "Attempt 017 repeated the exact image, bundle, asset, and runtime-overlay proof.",
        ),
        (
            "groot_checkpoint_server",
            "GR00T N1.7 SONIC checkpoint preflight and live policy server readiness",
            proven_016,
            "Checkpoint preflight passed and the worker advanced beyond GR00T server readiness.",
            proven_017,
            "Attempt 017 recorded the live GR00T server ready and accepting the first real query.",
        ),
        (
            "isaac_scene_baseline",
            "Isaac RTX, kitchen stage, live G1, standing pose, and microwave baseline",
            proven_016,
            "Attempt 016 recorded RTX startup, live articulation, standing initialization, and the exact microwave door baseline.",
            proven_017,
            "Attempt 017 repeated RTX, kitchen, live G1 standing, and microwave-door baseline proof.",
        ),
        (
            "controller_init_done",
            "Official GEAR-SONIC controller emits Init Done",
            pending,
            "Attempt 016 stopped at official controller readiness.",
            proven_017,
            "Attempt 017 recorded the official controller Init Done marker.",
        ),
        (
            "native_dds_freshness",
            "Native DDS bridge identity, advancing publication count, and fresh Isaac samples",
            proven_016,
            "Attempt 016 recorded the compiled ELF-audited bridge and fresh live Isaac snapshots; this did not prove controller readiness.",
            proven_017,
            "Attempt 017 recorded the native DDS bridge ready with fresh live Isaac state.",
        ),
        (
            "first_groot_query",
            "First real GR00T policy query",
            pending,
            None,
            proven_017,
            "Attempt 017 produced a fresh 40-frame by 78-dimension receding-horizon response from the real initial observation.",
        ),
        (
            "protocol_v4_token_receipt",
            "Matching protocol-v4 action token received by the official GEAR-SONIC controller",
            pending,
            None,
            proven_017,
            "Attempt 017 recorded the matching 64D token and hands payload at frame/step 1.",
        ),
        (
            "first_official_action",
            "First matching official GEAR-SONIC FK/action output produced from the learned policy response",
            pending,
            None,
            "partial_protocol_v4_token_receipt_only",
            "The matching protocol-v4 token arrived in Attempt 017, but no matching g1_debug/FK output was produced.",
        ),
        (
            "isaac_apply_readback",
            "First action applied to live Isaac and read back from the same articulation",
            pending,
            None,
            pending,
            "Attempt 017 failed before an official FK output or Isaac apply/readback.",
        ),
        (
            "first_learned_oscar_transition",
            "First learned OSCAR/WAM transition",
            pending,
            None,
            pending,
            "Attempt 017 failed before the first learned OSCAR/WAM transition.",
        ),
        (
            "semantic_microwave_transition",
            "Microwave door semantic state transition satisfies the task contract",
            pending,
            None,
            pending,
            "Attempt 017 failed at step 1 before any semantic microwave-door transition.",
        ),
        (
            "ordered_review_render",
            "Ordered first-person and third-person review render",
            pending,
            None,
            pending,
            "The startup canary frame is not an ordered episode review render.",
        ),
        (
            "dynamic_termination",
            "Dynamic task-completion termination or the explicit 48-step safety cap",
            pending,
            None,
            pending,
            "Attempt 017 terminated on a step-1 component failure, not task completion or the safety cap.",
        ),
        (
            "validated_final_review_upload",
            "Validated final_review video, ordering evidence, digest, decode, and upload",
            pending,
            None,
            pending,
            "Attempt 017 final-review validation was blocked and produced no episode MP4.",
        ),
    )
    return [
        {
            "gate_id": gate_id,
            "requirement": requirement,
            "attempt_016_status": status_016,
            "attempt_016_evidence_note": note_016,
            "attempt_017_status": status_017,
            "attempt_017_evidence_note": note_017,
            "current_session_status": "pending",
            "evidence_note": note_017 or note_016,
            "required_for_full_episode_video": True,
        }
        for gate_id, requirement, status_016, note_016, status_017, note_017 in rows
    ]


def session_claim_boundary() -> dict[str, Any]:
    return {
        "allocation_is_not_runtime_readiness": True,
        "runtime_readiness_is_not_episode_success": True,
        "video_arrival_is_not_validated_review_video": True,
        "validated_review_video_is_not_semantic_task_success": True,
        "current_target": (
            "One dynamic G1 kitchen episode using real GR00T N1.7 SONIC, the official "
            "GEAR-SONIC controller, learned OSCAR/WAM, live Isaac action apply/readback, "
            "semantic microwave-door success, and a validated ordered final review video."
        ),
        "prior_persistent_result": {
            "policy_calls": 2,
            "learned_wam_transitions": 1,
            "isaac_kitchen_semantic_success_proven": False,
            "full_episode_video_proven": False,
            "must_not_be_promoted_to_current_goal_completion": True,
        },
        "attempt_016_result": {
            "exact_image_assets": True,
            "groot_server_ready": True,
            "isaac_rtx_kitchen_g1_baseline": True,
            "native_dds_freshness": True,
            "controller_init_done": False,
            "real_groot_query_proven": False,
            "learned_oscar_transition_proven": False,
            "semantic_success_proven": False,
            "full_episode_video_proven": False,
            "failure_boundary": "official_controller_readiness",
        },
        "attempt_017_result": {
            "exact_image_assets": True,
            "groot_server_ready": True,
            "fresh_action_horizon": {
                "frame_count": 40,
                "frame_dimension": 78,
                "selection_mode": "fresh_receding_horizon_first_frame",
                "real_initial_observation": True,
            },
            "isaac_rtx_kitchen_g1_baseline": True,
            "controller_init_done": True,
            "native_dds_ready": True,
            "protocol_v4_token_receipt_step": 1,
            "matching_g1_debug_fk_output_proven": False,
            "isaac_action_apply_readback_proven": False,
            "learned_oscar_transition_proven": False,
            "semantic_success_proven": False,
            "full_episode_video_proven": False,
            "failure_boundary": (
                "controller_fk_skeleton_command_nonzero_before_matching_g1_debug_output; "
                "plain_git_rev_parse_rejected_root_owned_opt_wbc_as_dubious_ownership"
            ),
            "must_not_be_promoted_to_current_goal_completion": True,
        },
    }
