"""Single-allocation paired execution contract for internal policy canaries.

This is deliberately separate from the qualified native policy campaign.  A
canary may bind a typed, nonblocking controls gap, but it can never raise its
claim ceiling, rank candidates, or promote a scene.  The provider-facing
executor opens one session, loads each frozen candidate once, and runs the same
ten immutable cells for both candidates without authorizing another allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_arena_policy_canary_session.v1"
AUTHORITY_SCHEMA_VERSION = "native_task_arena_policy_canary_session_authority.v1"
RESULT_SCHEMA_VERSION = "native_task_arena_policy_canary_session_result.v1"
RUN_KIND = "internal_policy_canary"
CLAIM_CEILING = "diagnostic_policy_execution"
CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")
EPISODES_PER_POLICY = 10
LEARNED_ROLLOUT_COUNT = 20
PROBE_KIND = "native-task-arena-policy-canary-session"
PROVIDER_BUNDLE_SCHEMA_VERSION = (
    "native_task_arena_policy_canary_provider_bundle.v1"
)
PROVIDER_RESULT_FILENAME = "native_task_arena_policy_canary_session_result.v1.json"
CANONICAL_ALLOCATOR = (
    "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
)
CONTROL_MODES = frozenset({"nonblocking_diagnostic_pending", "nonblocking_diagnostic_bound"})


class PolicyCanarySessionError(ValueError):
    """Raised before paid execution when the paired-session contract is invalid."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically retain provider output without importing host-only helpers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _record(value: Any, *, code: str) -> dict[str, Any]:
    record = _mapping(value)
    if (
        not str(record.get("path") or "").strip()
        or not _digest(record.get("sha256"))
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] <= 0
    ):
        raise PolicyCanarySessionError(code)
    return record


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _execution_release(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    release = _mapping(value)
    required_digests = (
        "overlay_digest",
        "test_receipt_digest",
        "exact_failure_input_digest",
        "release_digest",
    )
    if (
        release.get("mode") != "signed_hotfix_overlay"
        or not all(_digest(release.get(field)) for field in required_digests)
        or any(
            not isinstance(release.get(field), str)
            or len(release[field]) != 40
            for field in ("base_release_commit", "patch_commit")
        )
        or release.get("evidence_grade_ceiling") != "development_only"
        or release.get("qualification_authorized") is not False
        or release.get("official_ranking_authorized") is not False
        or release.get("scene_promotion_authorized") is not False
        or release.get("normal_deployment_required_for_promotion") is not True
        or release.get("release_digest")
        != canonical_digest(release, digest_field="release_digest")
    ):
        raise PolicyCanarySessionError("policy_canary_execution_release_invalid")
    return release


def validate_runtime_input_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canary-only base-scene and resolved-cell input set.

    Qualified control receipts are intentionally not accepted as a substitute
    for this discriminated record.  Each cell instead carries either an exact
    diagnostic receipt or an explicit pending typed gap.
    """

    payload = json.loads(json.dumps(value, allow_nan=False))
    if (
        payload.get("schema_version")
        != "task_evaluation_policy_canary_runtime_inputs.v1"
        or payload.get("run_kind") != RUN_KIND
        or payload.get("claim_ceiling") != CLAIM_CEILING
        or tuple(payload.get("candidate_ids") or ()) != CANDIDATE_IDS
    ):
        raise PolicyCanarySessionError("policy_canary_runtime_input_identity_invalid")
    for field in ("activation_digest", "configuration_digest", "plan_digest"):
        if not _digest(payload.get(field)):
            raise PolicyCanarySessionError("policy_canary_runtime_input_digest_invalid")
    if "matrix_digest" in payload and not _digest(payload.get("matrix_digest")):
        raise PolicyCanarySessionError("policy_canary_runtime_input_digest_invalid")
    for field in ("base_native_packet", "runtime_source", "construction_result"):
        _record(payload.get(field), code=f"policy_canary_runtime_input_{field}_invalid")
    execution_authority = _mapping(payload.get("execution_authority"))
    if (
        execution_authority.get("maximum_provider_allocations") != 1
        or execution_authority.get("retry_cap") != 0
        or execution_authority.get("single_warm_provider_session_required") is not True
        or execution_authority.get("caller_surviving_watchdog_required") is not True
        or execution_authority.get("billing_teardown_provider_zero_required") is not True
    ):
        raise PolicyCanarySessionError("policy_canary_runtime_input_authority_invalid")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != EPISODES_PER_POLICY:
        raise PolicyCanarySessionError("policy_canary_runtime_input_cells_invalid")
    seen_cells: set[str] = set()
    seen_seeds: set[int] = set()
    for cell in cells:
        row = _mapping(cell)
        cell_id = str(row.get("cell_id") or "")
        seed = row.get("seed")
        scenario = row.get("resolved_scenario")
        control = _mapping(row.get("control_diagnostic"))
        mode = str(control.get("mode") or "")
        if (
            not cell_id
            or cell_id in seen_cells
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed in seen_seeds
            or not _digest(row.get("cell_spec_digest"))
            or not str(row.get("family") or "")
            or not isinstance(scenario, Mapping)
            or not scenario
            or row.get("resolved_scenario_digest") != canonical_digest(scenario)
            or mode not in CONTROL_MODES
            or control.get("policy_execution_blocked") is not False
        ):
            raise PolicyCanarySessionError("policy_canary_runtime_input_cell_invalid")
        if mode == "nonblocking_diagnostic_pending":
            if control.get("typed_gap") != "controls_pending_at_submission" or "receipt" in control:
                raise PolicyCanarySessionError("policy_canary_control_gap_invalid")
        else:
            _record(control.get("receipt"), code="policy_canary_control_receipt_invalid")
        seen_cells.add(cell_id)
        seen_seeds.add(seed)
    if payload.get("runtime_inputs_digest") != canonical_digest(
        payload, digest_field="runtime_inputs_digest"
    ):
        raise PolicyCanarySessionError("policy_canary_runtime_input_self_digest_invalid")
    return payload


def validate_session_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(value, allow_nan=False))
    if (
        payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or payload.get("run_kind") != RUN_KIND
        or payload.get("claim_ceiling") != CLAIM_CEILING
        or tuple(payload.get("candidate_ids") or ()) != CANDIDATE_IDS
        or payload.get("episodes_per_policy") != EPISODES_PER_POLICY
        or payload.get("learned_policy_rollout_count") != LEARNED_ROLLOUT_COUNT
        or payload.get("maximum_provider_allocations") != 1
        or payload.get("retry_cap") != 0
        or payload.get("automatic_retry_authorized") is not False
        or payload.get("caller_surviving_watchdog_required") is not True
        or payload.get("provider_wide_launch_lock_required") is not True
        or payload.get("canonical_allocator") != CANONICAL_ALLOCATOR
        or payload.get("scene_promotion_authorized") is not False
        or payload.get("official_ranking_authorized") is not False
    ):
        raise PolicyCanarySessionError("policy_canary_session_authority_identity_invalid")
    for field in ("activation_manifest", "runtime_inputs"):
        _record(payload.get(field), code=f"policy_canary_session_{field}_invalid")
    _execution_release(payload.get("execution_release"))
    if not _digest(payload.get("runtime_inputs_digest")):
        raise PolicyCanarySessionError("policy_canary_session_runtime_inputs_digest_invalid")
    if (
        not _finite_positive(payload.get("hard_cap_usd"))
        or isinstance(payload.get("hard_ttl_seconds"), bool)
        or not isinstance(payload.get("hard_ttl_seconds"), int)
        or payload["hard_ttl_seconds"] <= 0
        or not str(payload.get("resource_name") or "").startswith(
            "blueprint-native-task-policy-canary-"
        )
    ):
        raise PolicyCanarySessionError("policy_canary_session_resource_bounds_invalid")
    if payload.get("authority_digest") != canonical_digest(
        payload, digest_field="authority_digest"
    ):
        raise PolicyCanarySessionError("policy_canary_session_authority_digest_invalid")
    return payload


def consume_session_authority_once(
    value: Mapping[str, Any], *, consumption_path: str | Path
) -> dict[str, Any]:
    authority = validate_session_authority(value)
    destination = Path(consumption_path).expanduser().resolve()
    receipt = {
        "schema_version": "native_task_arena_policy_canary_session_consumption.v1",
        "status": "consumed",
        "authority_digest": authority["authority_digest"],
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "consumption_digest": "",
    }
    receipt["consumption_digest"] = canonical_digest(
        receipt, digest_field="consumption_digest"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        prior = json.loads(destination.read_text(encoding="utf-8"))
        return {
            "schema_version": receipt["schema_version"],
            "status": "already_consumed",
            "authority_digest": authority["authority_digest"],
            "maximum_provider_allocations": 0,
            "retry_cap": 0,
            "prior_consumption_digest": prior.get("consumption_digest"),
            "blockers": ["policy_canary_session_authority_already_consumed"],
        }
    return receipt


def validate_provider_bundle(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    payload = json.loads(json.dumps(value, allow_nan=False))
    bound_authority = validate_session_authority(authority)
    execution_release = _execution_release(bound_authority.get("execution_release"))
    if (
        payload.get("schema_version") != PROVIDER_BUNDLE_SCHEMA_VERSION
        or payload.get("status") != "ready"
        or payload.get("execution_mode") != "internal_policy_canary_paired_session"
        or payload.get("run_kind") != RUN_KIND
        or payload.get("claim_ceiling") != CLAIM_CEILING
        or tuple(payload.get("candidate_ids") or ()) != CANDIDATE_IDS
        or payload.get("episodes_per_policy") != EPISODES_PER_POLICY
        or payload.get("learned_policy_rollout_count") != LEARNED_ROLLOUT_COUNT
        or payload.get("maximum_provider_allocations") != 1
        or payload.get("retry_cap") != 0
        or payload.get("candidate_policy_queried") is not False
        or payload.get("expected_output_filename") != PROVIDER_RESULT_FILENAME
        or payload.get("runtime_inputs_digest")
        != bound_authority["runtime_inputs_digest"]
        or payload.get("authority_digest") != bound_authority["authority_digest"]
        or payload.get("execution_release") != execution_release
    ):
        raise PolicyCanarySessionError("policy_canary_provider_bundle_invalid")
    if (
        not str(payload.get("bundle_path") or "").strip()
        or not _digest(payload.get("bundle_sha256"))
        or isinstance(payload.get("bundle_size_bytes"), bool)
        or not isinstance(payload.get("bundle_size_bytes"), int)
        or payload["bundle_size_bytes"] <= 0
    ):
        raise PolicyCanarySessionError("policy_canary_provider_bundle_bytes_invalid")
    return payload


def build_session_authority(
    *,
    activation_manifest: Mapping[str, Any],
    activation_record: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    runtime_input_record: Mapping[str, Any],
    resource_name: str,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    execution_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = validate_runtime_input_manifest(runtime_inputs)
    activation = json.loads(json.dumps(activation_manifest, allow_nan=False))
    if (
        activation.get("run_kind") != RUN_KIND
        or activation.get("claim_ceiling") != CLAIM_CEILING
        or activation.get("activation_digest") != inputs["activation_digest"]
        or tuple(activation.get("candidate_ids") or ()) != CANDIDATE_IDS
        or activation.get("campaign_unit_count") != EPISODES_PER_POLICY
        or activation.get("activation_digest")
        != canonical_digest(activation, digest_field="activation_digest")
    ):
        raise PolicyCanarySessionError("policy_canary_activation_manifest_invalid")
    payload: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "run_id": activation.get("run_id"),
        "candidate_ids": list(CANDIDATE_IDS),
        "episodes_per_policy": EPISODES_PER_POLICY,
        "learned_policy_rollout_count": LEARNED_ROLLOUT_COUNT,
        "activation_manifest": _record(
            activation_record, code="policy_canary_session_activation_manifest_invalid"
        ),
        "runtime_inputs": _record(
            runtime_input_record, code="policy_canary_session_runtime_inputs_invalid"
        ),
        "runtime_inputs_digest": inputs["runtime_inputs_digest"],
        "resource_name": resource_name,
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "caller_surviving_watchdog_required": True,
        "provider_wide_launch_lock_required": True,
        "canonical_allocator": CANONICAL_ALLOCATOR,
        "scene_promotion_authorized": False,
        "official_ranking_authorized": False,
        **(
            {"execution_release": _execution_release(execution_release)}
            if execution_release is not None
            else {}
        ),
        "authority_digest": "",
    }
    payload["authority_digest"] = canonical_digest(
        payload, digest_field="authority_digest"
    )
    return validate_session_authority(payload)


def _episode_evidence_valid(episode: Mapping[str, Any]) -> bool:
    common_digests = (
        "checkpoint_digest",
        "runtime_identity_digest",
    )
    if any(not _digest(episode.get(field)) for field in common_digests):
        return False
    visual = _mapping(episode.get("visual_evidence"))
    gap = _mapping(visual.get("media_gap"))
    if episode.get("status") != "completed":
        return (
            gap.get("type") == "before_first_observation"
            and bool(str(gap.get("reason") or ""))
            and episode.get("policy_outcome_interpretable") is False
        )
    required_digests = (
        "lossless_frame_manifest_digest",
        "review_video_digest",
        "returned_action_sequence_digest",
        "action_delivery_readback_digest",
        "state_trace_digest",
        "contact_force_digest",
        "task_object_trajectory_digest",
        "deterministic_score_digest",
    )
    return (
        all(_digest(episode.get(field)) for field in required_digests)
        and episode.get("candidate_policy_queried") is True
        and isinstance(episode.get("actions_reached_robot"), bool)
        and isinstance(episode.get("arm_moved"), bool)
        and episode.get("scoring_authority") == "deterministic_simulator_state"
        and not gap
    )


def validate_session_result(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(value, allow_nan=False))
    episodes = payload.get("episodes")
    closeout = _mapping(payload.get("session_closeout"))
    if (
        payload.get("schema_version") != RESULT_SCHEMA_VERSION
        or payload.get("run_kind") != RUN_KIND
        or payload.get("claim_ceiling") != CLAIM_CEILING
        or tuple(payload.get("candidate_ids") or ()) != CANDIDATE_IDS
        or payload.get("episodes_per_policy") != EPISODES_PER_POLICY
        or payload.get("learned_policy_rollout_count") != LEARNED_ROLLOUT_COUNT
        or payload.get("retry_cap") != 0
        or payload.get("warm_session_open_count") != 1
        or payload.get("scene_promotion_performed") is not False
        or payload.get("official_ranking_performed") is not False
        or not isinstance(episodes, list)
        or len(episodes) != LEARNED_ROLLOUT_COUNT
    ):
        raise PolicyCanarySessionError("policy_canary_session_result_identity_invalid")
    allocations = closeout.get("provider_allocations_observed")
    if (
        allocations != 1
        or closeout.get("teardown_completed") is not True
        or closeout.get("provider_zero_confirmed") is not True
        or payload.get("provider_allocations_observed") != allocations
    ):
        raise PolicyCanarySessionError("policy_canary_session_result_closeout_invalid")
    observed: set[tuple[str, str]] = set()
    by_candidate: dict[str, list[tuple[str, int]]] = {candidate: [] for candidate in CANDIDATE_IDS}
    for episode in episodes:
        row = _mapping(episode)
        candidate = str(row.get("candidate_id") or "")
        cell_id = str(row.get("cell_id") or "")
        seed = row.get("seed")
        key = (candidate, cell_id)
        if (
            candidate not in CANDIDATE_IDS
            or not cell_id
            or key in observed
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or row.get("run_kind") != RUN_KIND
            or row.get("claim_ceiling") != CLAIM_CEILING
            or row.get("ranking_eligible") is not False
            or not _episode_evidence_valid(row)
        ):
            raise PolicyCanarySessionError("policy_canary_session_episode_invalid")
        observed.add(key)
        by_candidate[candidate].append((cell_id, seed))
    if (
        len(observed) != LEARNED_ROLLOUT_COUNT
        or sorted(by_candidate[CANDIDATE_IDS[0]])
        != sorted(by_candidate[CANDIDATE_IDS[1]])
    ):
        raise PolicyCanarySessionError("policy_canary_session_pairing_invalid")
    completed = all(row.get("status") == "completed" for row in episodes)
    expected_status = "completed_unqualified" if completed else "blocked"
    if payload.get("status") != expected_status:
        raise PolicyCanarySessionError("policy_canary_session_terminal_status_invalid")
    if payload.get("result_digest") != canonical_digest(
        payload, digest_field="result_digest"
    ):
        raise PolicyCanarySessionError("policy_canary_session_result_digest_invalid")
    return payload


def execute_paired_session(
    *,
    authority: Mapping[str, Any],
    runtime_inputs: Mapping[str, Any],
    open_session: Callable[[Mapping[str, Any]], Any],
    load_policy: Callable[[Any, str], Any],
    run_episode: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]],
    close_policy: Callable[[Any], None],
    close_session: Callable[[Any], Mapping[str, Any]],
    output_path: str | Path | None = None,
    provider_closeout_pending: bool = False,
    selected_cell_index: int | None = None,
) -> dict[str, Any]:
    """Execute all twenty learned rollouts in one caller-owned warm session."""

    bound_authority = validate_session_authority(authority)
    inputs = validate_runtime_input_manifest(runtime_inputs)
    if bound_authority["runtime_inputs_digest"] != inputs["runtime_inputs_digest"]:
        raise PolicyCanarySessionError("policy_canary_session_runtime_binding_mismatch")
    if selected_cell_index is not None and (
        isinstance(selected_cell_index, bool)
        or not isinstance(selected_cell_index, int)
        or not 0 <= selected_cell_index < len(inputs["cells"])
    ):
        raise PolicyCanarySessionError("policy_canary_session_cell_selection_invalid")
    selected_cells = (
        inputs["cells"]
        if selected_cell_index is None
        else [inputs["cells"][selected_cell_index]]
    )
    expected_episode_count = len(CANDIDATE_IDS) * len(selected_cells)
    session = None
    episodes: list[dict[str, Any]] = []
    policy_loads: list[dict[str, Any]] = []
    closeout: dict[str, Any] = {
        "status": "not_opened",
        "provider_allocations_observed": 0,
        "teardown_completed": False,
        "provider_zero_confirmed": False,
    }
    open_failure: Exception | None = None
    try:
        session = open_session(inputs)
        if not isinstance(session, Mapping) or (
            not provider_closeout_pending
            and session.get("provider_allocations_observed") != 1
        ):
            raise PolicyCanarySessionError("policy_canary_session_open_receipt_invalid")
        for candidate_id in CANDIDATE_IDS:
            policy = load_policy(session, candidate_id)
            policy_loads.append({"candidate_id": candidate_id, "loaded_once": True})
            try:
                for cell in selected_cells:
                    context = {
                        **cell,
                        "candidate_id": candidate_id,
                        "run_kind": RUN_KIND,
                        "claim_ceiling": CLAIM_CEILING,
                        "policy_outcome_interpretable": False,
                        "ranking_eligible": False,
                    }
                    try:
                        observed = dict(run_episode(session, policy, context))
                    except Exception as exc:  # preserve the typed episode gap
                        observed = {
                            "status": "blocked",
                            "candidate_policy_queried": False,
                            "actions_reached_robot": False,
                            "typed_harness_failure": type(exc).__name__,
                            "visual_evidence": {
                                "status": "unavailable_before_first_observation",
                                "media_gap": {
                                    "type": "before_first_observation",
                                    "reason": "policy_canary_episode_runner_failed",
                                },
                            },
                            "checkpoint_digest": str(
                                _mapping(policy).get("checkpoint_digest") or ""
                            ),
                            "runtime_identity_digest": str(
                                _mapping(policy).get("runtime_identity_digest") or ""
                            ),
                        }
                    observed.update(context)
                    observed["policy_outcome_interpretable"] = bool(
                        observed.get("candidate_policy_queried") is True
                        and observed.get("actions_reached_robot") is True
                        and observed.get("arm_moved") is True
                        and observed.get("scoring_authority")
                        == "deterministic_simulator_state"
                    )
                    observed["ranking_eligible"] = False
                    episodes.append(observed)
            finally:
                close_policy(policy)
    except Exception as exc:
        open_failure = exc
    finally:
        if session is not None:
            try:
                closeout = dict(close_session(session))
            except Exception as exc:
                closeout = {
                    "status": "close_failed",
                    "provider_allocations_observed": _mapping(session).get(
                        "provider_allocations_observed"
                    ),
                    "teardown_completed": False,
                    "provider_zero_confirmed": False,
                    "failure_type": type(exc).__name__,
                }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "run_kind": RUN_KIND,
        "claim_ceiling": CLAIM_CEILING,
        "candidate_ids": list(CANDIDATE_IDS),
        "episodes_per_policy": EPISODES_PER_POLICY,
        "learned_policy_rollout_count": LEARNED_ROLLOUT_COUNT,
        "provider_allocations_observed": (
            None
            if provider_closeout_pending
            else closeout.get("provider_allocations_observed")
        ),
        "retry_cap": 0,
        "warm_session_open_count": 1,
        "policy_loads": policy_loads,
        "episodes": episodes,
        "session_closeout": closeout,
        "session_failure_type": type(open_failure).__name__ if open_failure else None,
        "scene_promotion_performed": False,
        "official_ranking_performed": False,
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in episodes
        ),
        "provider_zero_required_after_return": True,
        "result_digest": "",
    }
    if selected_cell_index is not None:
        result["selected_cell_index"] = selected_cell_index
    if (
        len(episodes) == LEARNED_ROLLOUT_COUNT
        and all(row.get("status") == "completed" for row in episodes)
        and closeout.get("provider_allocations_observed") == 1
        and closeout.get("teardown_completed") is True
        and closeout.get("provider_zero_confirmed") is True
    ):
        result["status"] = "completed_unqualified"
    if (
        provider_closeout_pending
        and len(episodes) == expected_episode_count
        and closeout.get("runtime_closed") is True
    ):
        result["status"] = (
            "runtime_completed_unqualified_pending_closeout"
            if selected_cell_index is None
            else "runtime_selected_cell_completed_pending_aggregation"
        )
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    if output_path is not None:
        _write_json(Path(output_path), result)
    return result


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CANDIDATE_IDS",
    "CANONICAL_ALLOCATOR",
    "CLAIM_CEILING",
    "PolicyCanarySessionError",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_SCHEMA_VERSION",
    "PROVIDER_RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "RUN_KIND",
    "SCHEMA_VERSION",
    "build_session_authority",
    "consume_session_authority_once",
    "execute_paired_session",
    "validate_runtime_input_manifest",
    "validate_provider_bundle",
    "validate_session_authority",
    "validate_session_result",
]
