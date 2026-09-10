"""First-cell diagnostic admission from an independent witness and a proved rejection.

This opt-in contract changes matrix advancement, never candidate action admission,
task scoring, or the required-controls/qualified evaluation path.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

PROTOCOL_KEY = "diagnostic_continuation_protocol"
PROTOCOL_SCHEMA = "policy_canary_diagnostic_continuation_protocol.v1"
PROTOCOL_MODE = "verified_candidate_joint_bound_rejection_with_paired_native_witness"
GATE_SCHEMA = "policy_canary_diagnostic_continuation_gate.v1"
GATE_FILENAME = "diagnostic_continuation_gate.v1.json"


class DiagnosticContinuationError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise DiagnosticContinuationError("diagnostic_continuation_" + code)


def _expected_protocol(inputs: Mapping[str, Any]) -> dict[str, Any]:
    base = deepcopy(dict(inputs))
    base.pop(PROTOCOL_KEY, None)
    base["runtime_inputs_digest"] = canonical_digest(base, digest_field="runtime_inputs_digest")
    cells = base.get("cells") or []
    _require(base.get("run_kind") == "internal_policy_canary"
        and base.get("claim_ceiling") == "diagnostic_policy_execution"
        and base.get("candidate_ids") == ["pi05_droid", "groot_n17_droid"], "scope_invalid")
    _require(len(cells) == 10 and cells[0].get("family") == "canonical_anchor"
        and all((cell.get("control_diagnostic") or {}).get("mode") == "nonblocking_omitted_by_user"
                for cell in cells)
        and (base.get("task_success_contract", {}).get("criteria") or {}).get("controls") is None,
        "controls_mode_not_admitted")
    value = {"schema_version": PROTOCOL_SCHEMA, "mode": PROTOCOL_MODE,
        "scope": "first_canonical_cell", "controls_mode": "nonblocking_omitted_by_user",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "run_id": base["run_id"], "candidate_ids": base["candidate_ids"],
        "matrix_digest": base["matrix_digest"], "anchor_cell_spec_digest": cells[0]["cell_spec_digest"],
        "task_success_contract_digest": base["task_success_contract_digest"],
        "base_runtime_inputs_digest": base["runtime_inputs_digest"],
        "accepted_failure_kind": "finite_absolute_joint_position_bounds_rejection",
        "rejection_candidate_id": "pi05_droid",
        "witness_requirement": "other_frozen_candidate_unchanged_embodiment_parity_pass",
        "evidence_contract": "sealed_readiness_reset_request_response_and_lossless_media",
        "qualified_comparison_permitted": False, "action_admission_changed": False,
        "protocol_digest": ""}
    value["protocol_digest"] = canonical_digest(value, digest_field="protocol_digest")
    return value


def validate_diagnostic_continuation_protocol(inputs: Mapping[str, Any]) -> dict[str, Any] | None:
    if PROTOCOL_KEY not in inputs:
        return None
    expected = _expected_protocol(inputs)
    _require(inputs[PROTOCOL_KEY] == expected, "protocol_binding_invalid")
    return expected


def bind_diagnostic_continuation_protocol(runtime_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return newly frozen draft inputs; callers build a new ordinary authority next."""
    from .native_task_arena_policy_canary_session import validate_runtime_input_manifest

    inputs = validate_runtime_input_manifest(runtime_inputs)
    inputs[PROTOCOL_KEY] = _expected_protocol(inputs)
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    return validate_runtime_input_manifest(inputs)


def _verify_media(root: Path, media: Any) -> None:
    from .policy_canary_interrupted_cell_recovery import _verify_artifact

    _require(isinstance(media, list) and bool(media), "media_inventory_missing")
    for record in media:
        _require(isinstance(record, Mapping), "media_record_invalid")
        _verify_artifact(root / "episodes", record)


def _readiness(root: Path, value: Any, *, candidate: str, episode_id: str, reset: Mapping[str, Any]) -> dict:
    from .episode_visual_evidence import validate_multicamera_frame_manifest
    from .policy_canary_interrupted_cell_recovery import _read, _safe_path
    from .policy_episode_lifecycle import seal_prestart_readiness
    from .policy_scientific_reset import compare_reset_readbacks

    _require(isinstance(value, Mapping), "prestart_readiness_missing")
    receipt = dict(value)
    _require(receipt.get("readiness_digest") == canonical_digest(receipt, digest_field="readiness_digest")
        and seal_prestart_readiness(receipt) == receipt
        and receipt.get("candidate_id") == candidate and receipt.get("episode_id") == episode_id
        and receipt.get("outcome_blind") is True, "prestart_readiness_invalid")
    _verify_media(root, receipt.get("media_artifacts"))
    manifests = [row for row in receipt["media_artifacts"] if row.get("role") == "multicamera_observation_frame_manifest"]
    _require(len(manifests) == 1, "readiness_frame_manifest_missing")
    manifest = _read(_safe_path(root / "episodes", manifests[0]["relative_path"]))
    validate_multicamera_frame_manifest(manifest, output_dir=root / "episodes", verify_files=True)
    _require(compare_reset_readbacks(receipt["restored_scientific_reset"], reset)["status"] == "matched",
        "readiness_reset_mismatch")
    return receipt


def _reset(value: Any, *, binding: Mapping[str, Any], candidate: str) -> dict:
    from .policy_scientific_reset import validate_reset_readback

    _require(isinstance(value, Mapping), "scientific_reset_missing")
    receipt = validate_reset_readback(value)
    expected = {key: binding[key] for key in ("cell_id", "seed", "task_spec_digest", "resolved_scenario_digest")}
    expected["candidate_id"] = candidate
    _require(receipt.get("complete") is True and receipt.get("gaps") == []
        and receipt.get("binding") == expected, "scientific_reset_unverified")
    return receipt


def _witness(root: Path, row: Mapping[str, Any], *, binding: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    import numpy as np
    from PIL import Image
    from .native_policy_canary_matrix_gate import _episode_embodiment_parity_diagnostic
    from .policy_canary_interrupted_cell_recovery import _read, _safe_path, _verify_artifact
    from .policy_canary_media_integrity import require_completed_episode_media
    from .policy_episode_lifecycle import validate_policy_episode_lifecycle

    candidate = row["candidate_id"]
    episode_id = f"{binding['run_id']}--{binding['cell_id']}--{candidate}"
    episode = dict(row.get("episode") or {})
    core = dict(episode)
    core.pop("embodiment_parity_diagnostic", None)
    _require(row.get("status") == "completed" and row.get("scoring_authority") == "deterministic_simulator_state"
        and episode.get("episode_id") == episode_id and episode.get("candidate_id") == candidate
        and episode.get("task_spec_digest") == binding["task_spec_digest"]
        and episode.get("max_policy_queries") == spec["max_policy_queries"]
        and episode.get("open_loop_horizon") == spec["open_loop_horizon"], "native_witness_identity_invalid")
    validate_policy_episode_lifecycle(core)
    require_completed_episode_media(root, episode)
    from .policy_canary_interrupted_cell_recovery import _request_records
    requests, images = _request_records(root, episode_id, candidate, dict(spec), dict(binding))
    frames = episode.get("candidate_exact_policy_input_frames") or []
    _require(episode.get("policy_request_evidence_complete") is True
        and len(requests) == len(frames) == episode.get("policy_queries"), "native_witness_wire_evidence_missing")
    for index, frame in enumerate(frames):
        with Image.open(_safe_path(root / "episodes", frame["relative_path"])) as image:
            pixels = np.asarray(image)
        _require(np.array_equal(pixels, np.concatenate([images[index]["external"], images[index]["wrist"]], axis=1)),
            "native_witness_wire_frame_mismatch")
    for artifact in (row.get("evidence_artifacts") or {}).values():
        _require(isinstance(artifact, Mapping), "native_witness_artifact_missing")
        _verify_artifact(root, artifact)
    receipt_ref = row["evidence_artifacts"]["episode_receipt"]
    _require(_read(_safe_path(root, receipt_ref["relative_path"])) == episode,
        "native_witness_episode_receipt_mismatch")
    diagnostic = _episode_embodiment_parity_diagnostic(episode,
        observation_support_qualified=row.get("observation_support_qualified") is True)
    _require(diagnostic == row.get("embodiment_parity_diagnostic") and diagnostic["status"] == "passed",
        "paired_native_witness_not_passed")
    reset = _reset(episode.get("scientific_reset"), binding=binding, candidate=candidate)
    readiness = _readiness(root, episode.get("prestart_readiness"), candidate=candidate, episode_id=episode_id, reset=reset)
    _require((episode.get("score") or {}).get("status") == "scored", "native_witness_score_unavailable")
    return {"candidate_id": candidate, "classification": "paired_native_witness",
        "parity_receipt_digest": diagnostic["receipt_digest"], "prestart_readiness_digest": readiness["readiness_digest"],
        "scientific_reset": reset, "episode_receipt": receipt_ref,
        "task_success_required": False, "existing_approach_threshold_changed": False}


def _candidate_rejection(root: Path, row: Mapping[str, Any], *, binding: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    import numpy as np
    from .adp009d_droid_action_execution import DroidActionExecutionError, validate_candidate_action_bounds
    from .episode_visual_evidence import FAILED_POLICY_FRAME_MANIFEST_SCHEMA_VERSION, _verified_retained_rgb_frame
    from .openpi_droid_policy_runtime import OpenPIDroidPolicySpec, normalize_openpi_inference_response, validate_server_metadata
    from .policy_canary_interrupted_cell_recovery import _read, _safe_path, _verify_artifact
    from .policy_request_evidence import validate_request_evidence

    candidate = row["candidate_id"]
    _require(candidate == "pi05_droid" and row.get("status") == "blocked"
        and row.get("failure_type") == "DroidActionExecutionError"
        and row.get("typed_harness_failure") == "DroidActionExecutionError"
        and spec["policy_spec"].get("action_space") == "joint_position", "candidate_failure_not_admitted")
    episode_id = f"{binding['run_id']}--{binding['cell_id']}--{candidate}"
    failure_path = root / "episodes" / f"{episode_id}.failure_evidence.json"
    failure = _read(failure_path)
    _require(failure.get("gap_digest") == canonical_digest(failure, digest_field="gap_digest")
        and all(row.get(key) == value for key, value in failure.items()), "candidate_failure_receipt_mismatch")
    reset = _reset(failure.get("scientific_reset"), binding=binding, candidate=candidate)
    readiness = _readiness(root, failure.get("prestart_readiness"), candidate=candidate, episode_id=episode_id, reset=reset)
    policy_spec = OpenPIDroidPolicySpec(**spec["policy_spec"])
    server = (readiness.get("policy_control_plane") or {}).get("server_metadata")
    _require(isinstance(server, Mapping), "candidate_server_identity_missing")
    validate_server_metadata(server, expected=policy_spec)
    visual = failure.get("visual_evidence") or {}
    _require(visual.get("status") == "complete" and visual.get("episode_terminal_status") == "failed_after_first_observation"
        and visual.get("terminal_observation_invented") is False
        and set(visual.get("videos") or {}) == {"external", "wrist", "overview"}, "candidate_failure_media_incomplete")
    _verify_media(root, (failure.get("episode") or {}).get("media_artifacts"))
    for artifact in (failure.get("evidence_artifacts") or {}).values():
        if artifact is not None:
            _verify_artifact(root, artifact)
    manifest_ref = failure["evidence_artifacts"]["frame_manifest"]
    manifest = _read(_safe_path(root, manifest_ref["relative_path"]))
    _require(manifest.get("schema_version") == FAILED_POLICY_FRAME_MANIFEST_SCHEMA_VERSION
        and manifest.get("frame_manifest_digest") == canonical_digest(manifest, digest_field="frame_manifest_digest")
        and manifest.get("episode_id") == episode_id, "candidate_frame_manifest_invalid")
    frames = manifest.get("candidate_exact_policy_input_frames") or []
    for frame in frames:
        _verified_retained_rgb_frame(frame, output_dir=root / "episodes")
    queries = failure.get("candidate_policy_action_queries") or []
    requests = (failure.get("episode") or {}).get("policy_request_artifacts") or []
    _require(bool(queries) and len(queries) == len(requests) == len(frames)
        and len(queries) <= spec["max_policy_queries"], "candidate_request_response_pairing_missing")
    from .policy_canary_interrupted_cell_recovery import _request_records, _recover_frames
    verified_requests, images = _request_records(root, episode_id, candidate, dict(spec), dict(binding))
    composites, _streams, media_gaps = _recover_frames(root, episode_id, candidate, images)
    _require(len(verified_requests) == len(queries) == len(composites)
        and all(frame.get("exact_wire_pixels_verified") is True for frame in composites)
        and not media_gaps, "candidate_wire_frame_join_unproven")
    for index, (query, request_ref) in enumerate(zip(queries, requests, strict=True)):
        _verify_artifact(root / "episodes", request_ref)
        request = validate_request_evidence(_read(_safe_path(root / "episodes", request_ref["relative_path"])))
        request_binding = {**reset["binding"], "episode_id": episode_id, "query_index": index}
        _require(request.get("episode_binding") == request_binding and request.get("serialization_verified") is True
            and query.get("query_index") == index and query.get("action_payload_returned") is True
            and query.get("raw_vendor_action_response_digest") == canonical_digest({"raw_vendor_action_response": query.get("raw_vendor_action_response")}),
            "candidate_wire_evidence_invalid")
    query = queries[-1]
    response = failure.get("policy_inference_evidence")
    _require(isinstance(response, Mapping) and response.get("server_response_received") is True
        and response.get("actions_extracted") is True
        and response.get("server_identity_sha256") == policy_spec.server_metadata()["identity_sha256"]
        and response.get("raw_vendor_action_response_digest") == query["raw_vendor_action_response_digest"]
        and response.get("raw_vendor_action_response") == query["raw_vendor_action_response"],
        "candidate_last_response_identity_unproven")
    values = np.asarray(normalize_openpi_inference_response(query["raw_vendor_action_response"]), dtype=float)
    horizon = int(spec["open_loop_horizon"])
    limits = np.asarray(reset["observed"]["robot"]["joint_limits"], dtype=float).reshape(-1, 2)[:7]
    _require(values.shape == (policy_spec.action_chunk_rows, 8) and np.isfinite(values).all()
        and query.get("executed_prefix_rows") == horizon and query.get("executed_prefix_bounds_validated") is False
        and query.get("shape_validated") is True and query.get("finite_values_validated") is True,
        "candidate_rejected_response_contract_invalid")
    try:
        validate_candidate_action_bounds(values[:horizon], action_space="joint_position", joint_limits=limits, candidate_id=candidate)
    except DroidActionExecutionError as exc:
        errors = list(exc.errors)
    else:
        raise DiagnosticContinuationError("diagnostic_continuation_candidate_rejection_not_reproduced")
    _require(bool(errors) and all(error.startswith("candidate_action_joint_position_bounds_invalid:") for error in errors)
        and errors == query.get("executed_prefix_bound_validation_errors")
        and failure.get("failure_message") == ";".join(errors), "candidate_rejection_reason_mismatch")
    rejected_index = len(queries) - 1
    commands = failure.get("commanded_actions") or []
    expected_commands = {(index, position) for index in range(rejected_index) for position in range(horizon)}
    _require(len(commands) == len(expected_commands)
        and {(command.get("query_index"), command.get("action_index_within_query")) for command in commands} == expected_commands,
        "prior_command_sequence_incomplete_or_ambiguous")
    for prior in queries[:rejected_index]:
        prior_values = np.asarray(prior["raw_vendor_action_response"]["actions"], dtype=float)
        validate_candidate_action_bounds(prior_values[:horizon], action_space="joint_position", joint_limits=limits, candidate_id=candidate)
    for command in commands:
        index = command.get("query_index")
        position = command.get("action_index_within_query")
        _require(type(index) is int and 0 <= index < rejected_index and type(position) is int and 0 <= position < horizon,
            "rejected_query_action_applied_or_ambiguous")
        expected = np.asarray(queries[index]["raw_vendor_action_response"]["actions"][position], dtype=float)[:7]
        before, after = np.asarray(command.get("observed_before_rad"), dtype=float), np.asarray(command.get("observed_after_rad"), dtype=float)
        _require(all(command.get(flag) is True for flag in ("environment_step_applied", "native_command_validated",
            "joint_state_before_validated", "joint_state_after_validated"))
            and command.get("joint_limit_clamped") is False
            and before.shape == after.shape == (7,) and np.isfinite(before).all() and np.isfinite(after).all()
            and np.array_equal(np.asarray(command.get("joint_position_target_rad")), expected)
            and np.array_equal(np.asarray(command.get("isaac_action"))[:7], expected), "prior_native_command_unverified")
    return {"candidate_id": candidate, "classification": "verified_candidate_joint_bound_rejection",
        "rejected_query_index": rejected_index, "replayed_rejection": errors,
        "raw_response_digest": query["raw_vendor_action_response_digest"], "source_failure_digest": failure["gap_digest"],
        "prestart_readiness_digest": readiness["readiness_digest"], "scientific_reset": reset,
        "retained_request_count": len(requests), "prior_applied_action_count": len(commands),
        "rejected_query_applied_action_count": 0, "action_bounds_changed": False, "score_invented": False}


def assess_diagnostic_first_cell(*, runtime_root: Path, child_root: Path) -> dict[str, Any]:
    """Read retained evidence and return a sealed gate decision without mutations."""
    from .native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME
    from .policy_canary_interrupted_cell_recovery import _load_binding, _read, _verify_artifact
    from .policy_scientific_reset import compare_reset_readbacks

    receipt: dict[str, Any] = {"schema_version": GATE_SCHEMA, "status": "blocked", "scope": "first_canonical_cell",
        "run_kind": "internal_policy_canary", "claim_ceiling": "diagnostic_policy_execution",
        "candidate_adjudications": [], "blockers": [], "qualified_comparison_permitted": False,
        "action_admission_changed": False, "policy_execution_performed": False, "result_digest": ""}
    try:
        root = Path(child_root).resolve()
        inputs, authority, cell, specs, binding = _load_binding(Path(runtime_root).resolve(), 0)
        protocol = validate_diagnostic_continuation_protocol(inputs)
        _require(protocol is not None, "protocol_not_enabled")
        child = _read(root / PROVIDER_RESULT_FILENAME)
        receipt.update(protocol_digest=protocol["protocol_digest"], runtime_inputs_digest=inputs["runtime_inputs_digest"],
            authority_digest=authority["authority_digest"], child_result_digest=child.get("result_digest"), binding=binding)
        _require(child.get("result_digest") == canonical_digest(child, digest_field="result_digest")
            and child.get("selected_cell_index") == 0
            and child.get("task_success_contract") == inputs["task_success_contract"]
            and child.get("matrix_digest") == inputs["matrix_digest"], "child_identity_invalid")
        rows = child.get("episodes") or []
        expected = {(candidate, cell["cell_id"], cell["seed"]) for candidate in inputs["candidate_ids"]}
        _require(len(rows) == 2 and {(row.get("candidate_id"), row.get("cell_id"), row.get("seed")) for row in rows} == expected,
            "paired_episode_identity_invalid")
        inventory = child.get("artifact_inventory")
        _require(isinstance(inventory, list) and bool(inventory)
            and child.get("artifact_inventory_digest") == canonical_digest({"value": inventory}), "child_inventory_invalid")
        for artifact in inventory:
            _verify_artifact(root, artifact)
        checked = []
        for row in rows:
            spec = specs[row["candidate_id"]]
            _require(all(row.get(key) == spec.get(key) for key in ("checkpoint_digest", "runtime_identity_digest")),
                "candidate_execution_identity_invalid")
            checked.append((_witness if row.get("status") == "completed" else _candidate_rejection)(
                root, row, binding=binding, spec=spec))
        _require(any(item["classification"] == "paired_native_witness" for item in checked), "independent_native_witness_missing")
        parity = compare_reset_readbacks(checked[0]["scientific_reset"], checked[1]["scientific_reset"])
        _require(parity["status"] == "matched", "paired_scientific_reset_unverified")
        receipt["candidate_adjudications"] = [{key: value for key, value in item.items() if key != "scientific_reset"} for item in checked]
        receipt["paired_reset_parity"] = parity
        receipt["status"] = "passed"
    except (KeyError, TypeError, ValueError, OSError) as exc:
        receipt["blockers"] = [str(exc) if isinstance(exc, DiagnosticContinuationError)
                               else f"diagnostic_continuation_evidence_invalid:{type(exc).__name__}:{str(exc)[:256]}"]
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    return receipt


__all__ = ["PROTOCOL_KEY", "GATE_FILENAME", "bind_diagnostic_continuation_protocol",
           "validate_diagnostic_continuation_protocol", "assess_diagnostic_first_cell"]
