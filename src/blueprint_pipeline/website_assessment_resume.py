"""ADP-010/day14: wake the existing browser producer after assessment admission.

No new queue, publisher, credentials or job. Only the explicit pre-provider
assessment refusal may arm this record in the existing handoff job ledger.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from pathlib import Path

from .common import write_json
from . import website_task_context as context_reader
from .website_preparation_contracts import validate_preparation_proposal as _proposal_admitted

PENDING_CODE = "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending"
SCHEMA = "website_assessment_resume.v1"
MAX_BYTES = 65_536


class AssessmentPreparationPending(ValueError):
    def __init__(self, resume_record):
        super().__init__(PENDING_CODE)
        self.resume_record = resume_record


def _require(condition):
    if not condition:
        raise ValueError("website_assessment_resume_binding_invalid")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _browser_wire_metadata(value):
    """Validate current browser-builder fields without retaining arbitrary records.

    These fields follow BlueprintCapture.extractFrames. The server-authored
    browser manifest supplies no lineage, upstream or robot candidate records.
    Raw bytes stay private in the existing ledger, never in status projections.
    """
    shapes = {
        "media_metadata": "source_device original_video_uri original_video_object frame_timestamps_uri stream_metadata_uri width height fps_source device_model device_model_marketing capture_start_epoch_ms",
        "task_site_context": "workflow_name task_steps target_kpi zone shift owner facility_template required_coverage_areas benchmark_stations adjacent_systems privacy_security_limits known_blockers non_routine_modes people_traffic_notes capture_restrictions lighting_windows shift_traffic_windows movable_obstacles floor_condition_notes reflective_surface_notes access_rules robot_eval_task_anchor_candidates robot_eval_scene_asset_hints robot_eval_robot_profile_candidates robot_eval_route_anchor_candidates",
        "scene_memory_capture": "continuity_score lighting_consistency dynamic_object_density sensor_availability operator_notes inaccessible_areas semantic_anchors_observed relocalization_count overlap_checkpoint_count world_model_candidate_reasoning motion_provenance motion_timestamps_capture_relative world_model_candidate geometry_source geometry_ready",
        "capture_rights": "derived_scene_generation_allowed data_licensing_allowed capture_contributor_payout_eligible consent_status permission_document_uri consent_scope consent_notes",
        "identity": "scene_id capture_id manifest_scene_id manifest_capture_id completion_scene_id completion_capture_id site_submission_id buyer_request_id capture_job_id upstream_handoff hosted_review_blockers completion_raw_prefix",
        "pipeline_status_event": "event_type scene_id capture_id raw_prefix raw_prefix_uri upload_completion_marker_uri trigger_object trigger_kind qa_status pipeline_handoff_uri source_finalize source_membership_status",
        "robot_eval_task_thresholds": "threshold_source target_kpi zone claim_boundary",
        "robot_eval_cpu_preflight_inputs": "task_anchor_candidates scene_asset_hints robot_profile_candidates route_anchor_candidates source_policy claim_boundary",
        "robot_eval_episode_spec_inputs": "task_anchor_candidate_count scene_asset_hint_count robot_profile_candidate_count route_anchor_candidate_count review_required claim_boundary",
        "sensor_availability": "arkit_poses arkit_intrinsics arkit_depth arkit_confidence arkit_meshes motion video_sync arkit_frames arkit_frame_quality arkit_feature_points arkit_planes arkit_light_estimates camera_pose camera_intrinsics depth depth_confidence point_cloud planes tracking_state light_estimate geospatial companion_phone_pose companion_phone_intrinsics companion_phone_calibration",
    }
    empty_arrays = {"robot_eval_task_anchor_candidates", "robot_eval_scene_asset_hints", "robot_eval_robot_profile_candidates",
                    "robot_eval_route_anchor_candidates", "task_anchor_candidates", "scene_asset_hints", "robot_profile_candidates", "route_anchor_candidates"}

    def check(item, field, depth=0):
        _require(depth < 8)
        if field in {"privacy_lineage", "provenance_lineage", "upstream_handoff"}:
            _require(item is None)
        elif field in empty_arrays:
            _require(item == [] and type(item) is list)
        elif type(item) is dict:
            if field in {"source_finalize", "source_membership_selector"}:
                _require(item == value.get(field))
                return  # Existing source parser validates these exact selectors.
            _require(field in shapes and set(item) <= set(shapes[field].split()))
            for key, child in item.items():
                check(child, key, depth + 1)
        elif type(item) is list:
            _require(len(item) <= 1000 and all(type(child) is str for child in item))
            for child in item:
                check(child, field, depth + 1)
        else:
            _require(item is None or type(item) in {str, int, float, bool})
            if type(item) is float:
                _require(math.isfinite(item))
            if type(item) is str:
                _require(len(item) <= 16_384 and re.search(r"[?&](?:x-goog-signature|x-amz-signature|signature|access_token)=", item, re.I) is None)

    for key, item in value.items():
        check(item, key)


def _payload_bytes(listener, payload, digest):
    raw = (payload if isinstance(payload, bytes) else payload.encode("utf-8") if isinstance(payload, str)
           else json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode())
    _require(0 < len(raw) <= MAX_BYTES and listener.payload_sha256(raw) == digest)
    value = json.loads(raw, object_pairs_hook=_unique)
    # Preserve exact bytes, rather than sanitizing/re-encoding a different event.
    # These are the existing handoff wire fields, not arbitrary private metadata.
    _require(type(value) is dict and set(value) <= {
        "schema_version", "bucket", "scene_id", "capture_id", "raw_prefix_uri", "pipeline_handoff_uri",
        "source_finalize", "source_membership_selector", "triggered_at", "source",
        "robot_eval_job_request_uri", "robot_eval_job_request_path", "robot_eval_request_inbox_uri",
        "robot_eval_request_inbox_path", "robot_eval_job_id", "robot_eval_provisioner", "robot_eval_simulator",
        "robot_eval_evaluation_substrate", "robot_eval_budget_usd",
        "handoff_source", "handoff_envelope_version", "handoff_topic", "handoff_trigger_object", "handoff_trigger_kind",
        "site_submission_id", "buyer_request_id", "capture_job_id", "region_id", "rights_profile", "capture_source",
        "source_device", "capture_modality", "raw_video_uri", "media_metadata", "qa_status", "requested_outputs", "requested_lanes",
        "raw_prefix", "frames_index_uri", "capture_descriptor_uri", "qa_report_uri", "keyframe_uri", "pipeline_status_event",
        "source_membership_status", "task_site_context", "scene_memory_capture", "capture_rights", "privacy_lineage",
        "provenance_lineage", "identity", "preview_simulation_requested", "worldlabs_request_manifest_uri",
        "worldlabs_input_manifest_uri", "worldlabs_input_video_uri", "robot_eval_dataset_requested",
        "robot_eval_publication_gate_required", "robot_eval_required_artifacts", "robot_eval_missing_proof_labels",
        "robot_eval_task_thresholds", "robot_eval_cpu_preflight_inputs", "robot_eval_episode_spec_inputs",
        "robot_eval_publication_blockers", "generated_at"})
    listener.parse_handoff_payload(raw)
    _browser_wire_metadata(value)
    return raw


def admit_browser_preparation(listener, *, payload, handoff, capture_root, observation,
                              producer_delivery_key, payload_digest, allow_resume):
    """Explicit current browser admission before run_e2e or any provider stage."""
    if observation["producer_delivery"]["kind"] != "website_browser_capture_delivery":
        return
    _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None
             and producer_delivery_key == observation["producer_delivery"]["delivery_key"])
    context = context_reader.load_current_website_task_context(request_id=observation["request_id"],
        scene_id=handoff.scene_id, capture_id=handoff.capture_id, purpose="scene_preparation")
    path = capture_root / "pipeline" / "website_task_context.json"
    from .website_preparation_status import _read
    if path.is_file():
        retained = context_reader.validate_website_task_context(_read(path), request_id=observation["request_id"],
            scene_id=handoff.scene_id, capture_id=handoff.capture_id, purpose="scene_preparation")
        _require(retained["context_digest"] == context["context_digest"])
    else:
        write_json(path, context)
    try:
        authority = context_reader.load_website_scene_sponsorship(task_context=context, now=time.time())
    except ValueError as exc:
        if str(exc) != PENDING_CODE:
            raise
        record = None
        if allow_resume:
            raw = _payload_bytes(listener, payload, payload_digest)
            record = {"schema_version": SCHEMA, "payload_base64": base64.b64encode(raw).decode(),
                "source_payload_sha256": payload_digest, "producer_delivery_key": producer_delivery_key,
                "task_context_digest": context["context_digest"]}
        raise AssessmentPreparationPending(record) from None
    _proposal_admitted(authority, context)


def safe_to_arm(attempt_count, previous_history, recovered_expired_lease):
    """Provider-stage failures/unknown outcomes never become fresh retry capacity."""
    return (not recovered_expired_lease and len(previous_history) == attempt_count - 1
        and all(row.get("stage") == "website_assessment_preparation" and row.get("error") == PENDING_CODE
                and type(row.get("assessment_resume")) is dict
                and row["assessment_resume"].get("schema_version") == SCHEMA for row in previous_history))


def reconcile_waiting_assessments(listener, *, storage_root: Path, process_args, limit=1):
    """Reuse the existing drain tick, original bytes, leases and source checks."""
    from .website_preparation_status import _read, _root, read_preparation_status
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    counts = {"resumed": 0, "pending": 0}
    roots = sorted(path.parent for path in Path(storage_root).glob("*/scenes/site-*/captures/walkthrough-*/pipeline_job_ledger.json"))
    # The preceding status reconciliation already rotates this durable cursor,
    # including empty/error states. Reuse it; no second cursor/queue framework.
    cursor_path = Path(storage_root) / ".website_preparation_delivery_cursor.json"
    try:
        cursor = _read(cursor_path).get("last_capture_digest") if cursor_path.is_file() else None
    except (OSError, ValueError):
        cursor = None
    keys = [hashlib.sha256(str(root).encode()).hexdigest() for root in roots]
    if cursor in keys:
        pivot = keys.index(cursor) + 1
        roots = roots[pivot:] + roots[:pivot]
    for root in roots[:max(0, min(limit, 50))]:
        try:
            _root(root)
            _require(not (root / listener.JOB_LEDGER_FILENAME).is_symlink())
            before = listener._read_job_ledger(root)
            record = before.get("assessment_resume")
            if before.get("status") != "failed_retryable" or before.get("last_error") != PENDING_CODE or not record:
                continue
            counts["pending"] += 1
            _require(type(record) is dict and set(record) == {"schema_version", "payload_base64",
                "source_payload_sha256", "producer_delivery_key", "task_context_digest"}
                and record["schema_version"] == SCHEMA and type(record["payload_base64"]) is str
                and len(record["payload_base64"]) <= 4 * ((MAX_BYTES + 2) // 3))
            raw = _payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True), record["source_payload_sha256"])
            handoff = listener.parse_handoff_payload(raw)
            _require(root == listener._handoff_capture_root(handoff, storage_root=storage_root)
                and before.get("source_payload_sha256") == record["source_payload_sha256"]
                and before.get("producer_delivery_key") == record["producer_delivery_key"])
            _require(handoff.source_finalize is not None and handoff.source_membership_selector is not None)
            request_id = handoff.scene_id.removeprefix("site-")
            # Reuse the actual status reader's complete active birth/membership,
            # fresh owner/video/rights, local consent, immutable current context
            # and unchanged ledger checks; a historical sidecar grants nothing.
            status = read_preparation_status(capture_root=root, selectors={"request_id": request_id,
                "scene_id": handoff.scene_id, "capture_id": handoff.capture_id,
                "completion_marker_generation": handoff.source_finalize["generation"],
                "producer_delivery_key": record["producer_delivery_key"],
                "source_payload_sha256": record["source_payload_sha256"], "task_context_digest": record["task_context_digest"]})
            _require(status["state"] == "failed_retryable" and status["revision"] == before["revision"])
            birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
            _require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector
                and _read(Path(birth["birth_delivery_raw_ref"]["path"]))["producer_delivery"]["kind"] == "website_browser_capture_delivery")
            context = context_reader.validate_website_task_context(_read(root / "pipeline" / "website_task_context.json"),
                request_id=request_id, scene_id=handoff.scene_id, capture_id=handoff.capture_id, purpose="scene_preparation")
            _proposal_admitted(context_reader.load_website_scene_sponsorship(task_context=context, now=time.time()), context)
            _require(listener._read_job_ledger(root) == before)
            # The SAME successful lease claim consumes the pending field; its
            # historical failure row retains the bytes. Never synthesize an ack.
            result = listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=record["source_payload_sha256"], expected_assessment_resume={"revision": before["revision"], "record": record},
                **({"expected_preparation_purpose": "scene_preparation"}
                   if birth.get("capture_observation_purpose") == "scene_preparation" else {}), **process_args)
            if result.get("status") in {"processed", "retryable_blocked"}:
                counts["resumed"] += 1
                counts["pending"] -= 1
        except Exception:
            listener.logger.debug("pubsub_handoff.assessment_resume_pending")
    return counts
