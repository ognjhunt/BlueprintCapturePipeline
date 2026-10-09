"""Submit only prepared website images; retain one Marble operation per admitted attempt."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from copy import deepcopy

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .paid_resource_admission import PaidResourceAdmissionGrant, require_paid_resource_admission_grant

# Multi-image Marble 1.1 Plus: 100 + 1500 + at most 1500 credits.
# https://docs.worldlabs.ai/api/pricing (verified 2026-09-19); no HQ mesh export.
MAX_GENERATION_COST_USD = 3100 / 1250


def prepare_atlas_pose_inputs(*, inputs: Mapping[str, Any], input_root: Path,
                              output_root: Path) -> dict[str, Any]:
    """CPU-only handoff for the documented unposed-image operation. Never submits."""
    from .website_scene_geometry import _bound_frames, INPUT_SCHEMA

    if inputs.get("schema_version") != INPUT_SCHEMA or inputs.get("heldout_pixels_included") is not False:
        raise ValueError("website_atlas_candidate_inputs_required")
    frames = _bound_frames(inputs, input_root, geometry=False)
    result = {"schema_version": "website_atlas_pose_inputs.v1", "status": "pending",
              "backend": "atlas", "endpoint": "/api/v2/tasks:images2PosedRGBD",
              "binding": {"source_video_digest": inputs["binding"]["source_video_digest"],
                          "geometry_input_digest": inputs["digest"]},
              "frames": [{"frame_id": row["frame_id"], "image_path": row["image_path"],
                          "image_digest": row["image_digest"]} for row in frames],
              "source_sample_count": inputs.get("source_sample_count"),
              "heldout_frame_count": inputs.get("heldout_frame_count"),
              "planned_provider_input_count": len(frames), "actual_provider_input_count": 0,
              "heldout_pixels_included": False, "claim_ceiling": "development_only",
              "metric_measurement_proven": False, "physical_evidence": False,
              "provider_mutation_performed": False,
              "blockers": ["website_atlas_admitted_pose_operation_required"]}
    result["digest"] = canonical_digest(result, digest_field="digest")
    path = output_root / "atlas_pose_inputs.json"
    if path.is_file():
        if json.loads(path.read_text()) != result:
            raise ValueError("website_atlas_pose_input_identity_conflict")
    else:
        write_json(path, result)
    pose_path = output_root / "atlas_posed_rgbd_result.json"
    if pose_path.is_file():
        # A producer must retain the real admitted operation with its exact
        # source/input binding. A bare operation from another capture is not
        # sufficient, and this reader grants no upload/spending authority.
        pose = json.loads(pose_path.read_text())
        if (pose.get("schema_version") != "website_atlas_pose_result.v1"
                or pose.get("digest") != canonical_digest(pose, digest_field="digest")
                or pose.get("binding") != result["binding"]
                or pose.get("endpoint") != result["endpoint"]):
            raise ValueError("website_atlas_pose_result_binding_mismatch")
        operation = pose.get("operation")
        if not isinstance(operation, Mapping):
            raise ValueError("website_atlas_pose_operation_not_succeeded")
        body = atlas_generate_from_pose_operation(operation=operation,
            expected_frame_count=len(frames), target_cameras=pose.get("target_cameras") or [],
            prompt=pose.get("prompt"))
        context = {**result, "pose_operation_id": operation["id"],
                   "pose_result_digest": pose["digest"], "generation_endpoint": "/api/v2/tasks:atlasGenerate",
                   "generation_request": body, "posed_rgbd_available": True,
                   # A returned bundle proves output count, not provider input
                   # usage. Preserve unknown input usage until the producer
                   # supplies its actual admitted operation receipt.
                   "actual_provider_input_count": None,
                   "returned_pose_frame_count": len(body["contextFrames"]),
                   "native_geometry_available": False,
                   "blockers": ["website_atlas_admitted_generation_operation_required",
                                "website_atlas_final_scene_assets_required",
                                "website_atlas_native_geometry_binding_required"]}
        context["digest"] = canonical_digest(context, digest_field="digest")
        context_path = output_root / "atlas_generation_inputs.json"
        if context_path.is_file():
            if json.loads(context_path.read_text()) != context:
                raise ValueError("website_atlas_generation_input_identity_conflict")
        else:
            write_json(context_path, context)
        return context
    return result


def atlas_generate_from_pose_operation(*, operation: Mapping[str, Any],
                                       expected_frame_count: int,
                                       target_cameras: Sequence[Mapping[str, Any]],
                                       prompt: str | None = None) -> dict[str, Any]:
    """Use images2PosedRGBD's returned bundle unchanged; build, never submit.

    The caller owns operation/source admission. This is a generation payload,
    not native source geometry, a splat, a collider, or measured metric evidence.
    https://atlas-beta.worldlabs.ai/docs/images-to-posed-images
    https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images
    """
    if (operation.get("done") is not True or operation.get("error") is not None
            or not isinstance(operation.get("id"), str) or not operation["id"].strip()):
        raise ValueError("website_atlas_pose_operation_not_succeeded")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("website_atlas_generation_prompt_invalid")
    response = operation.get("response")
    frames = response.get("frames") if isinstance(response, Mapping) else None
    if (isinstance(expected_frame_count, bool) or not isinstance(expected_frame_count, int)
            or expected_frame_count < 1 or not isinstance(frames, list)
            or len(frames) != expected_frame_count or not target_cameras):
        raise ValueError("website_atlas_pose_frames_invalid")

    def camera_grid(camera: Mapping[str, Any]) -> tuple[int, int]:
        if not isinstance(camera, Mapping):
            raise ValueError("website_atlas_camera_invalid")
        k, pose = camera.get("intrinsics") or {}, camera.get("extrinsics") or {}
        if not isinstance(k, Mapping) or not isinstance(pose, Mapping):
            raise ValueError("website_atlas_camera_invalid")
        width, height = k.get("width"), k.get("height")
        position, quaternion = pose.get("position"), pose.get("quaternion")
        def numeric(x: Any) -> bool:
            return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        if (any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (width, height))
                or not all(numeric(k.get(key)) for key in ("fx", "fy", "cx", "cy"))
                or k["fx"] <= 0 or k["fy"] <= 0
                or pose.get("coordinateSystem") not in ("rub", "rdf")
                or not isinstance(position, (list, tuple)) or len(position) != 3
                or not all(numeric(x) for x in position)
                or not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4
                or not all(numeric(x) for x in quaternion)
                or not math.isclose(sum(x * x for x in quaternion), 1, abs_tol=1e-3)):
            raise ValueError("website_atlas_camera_invalid")
        return width, height

    def asset(reference: Any) -> bool:
        # Retained project asset IDs avoid exporting transient access URLs.
        return isinstance(reference, Mapping) and isinstance(reference.get("assetId"), str) and bool(reference["assetId"].strip())

    grids = set()
    context = []
    for frame in frames:
        if (not isinstance(frame, Mapping) or not asset(frame.get("imageAsset"))
                or not isinstance(frame.get("depth"), Mapping)
                or not asset(frame["depth"].get("depthAsset"))):
            raise ValueError("website_atlas_posed_rgbd_bundle_required")
        grids.add(camera_grid(frame.get("camera")))
        # Never resize, crop, relabel axes, substitute source RGB or infer a
        # source-pixel transform for the provider's reframed output.
        depth = {"depthAsset": {"assetId": frame["depth"]["depthAsset"]["assetId"]}}
        if frame["depth"].get("confidenceAsset") is not None:
            if not asset(frame["depth"]["confidenceAsset"]):
                raise ValueError("website_atlas_confidence_asset_invalid")
            depth["confidenceAsset"] = {"assetId": frame["depth"]["confidenceAsset"]["assetId"]}
        context.append({"imageAsset": {"assetId": frame["imageAsset"]["assetId"]},
                        "camera": deepcopy(frame["camera"]), "depth": depth})
    if len(grids) != 1:
        raise ValueError("website_atlas_context_grid_mismatch")
    for camera in target_cameras:
        if camera_grid(camera) != (1280, 720):
            raise ValueError("website_atlas_target_grid_invalid")
    return {"contextFrames": context, "targetCameras": deepcopy(list(target_cameras)), "prompt": prompt}

#: A pre-generation 402 buys nothing and settles at $0, so each one admits one
#: more attempt bound to the rejected one. On 2026-09-27 a website capture's
#: only retry ran minutes before the owner added credits, and a single retry
#: stranded the scene. Bounded, so an account that stays empty cannot loop.
MAX_CREDIT_REJECTION_RETRIES = 3


#: An attempt whose generate outcome was never recorded (the process died, or a
#: later pass overwrote the run manifest before the 402 was reconciled) is
#: resolved only from the provider's own index: every generate request carries
#: its bp-<digest> tag and World Labs lists a world from PENDING onward. Any
#: in-flight POST holds submission.lock, and this grace outlasts its timeout.
UNRECORDED_ATTEMPT_GRACE_SECONDS = 600
_RESOLVED_ATTEMPT_STATUSES = frozenset({"rejected_insufficient_credits", "no_world_generated"})


def _attempt_name(stem: str, attempt: int) -> str:
    return f"{stem}.json" if attempt == 0 else f"{stem}_retry_{attempt}.json"


def _latest_attempt(root: Path) -> int:
    attempt = 0
    while attempt < MAX_CREDIT_REJECTION_RETRIES and (root / _attempt_name("submission", attempt + 1)).is_file():
        attempt += 1
    return attempt


def settle_website_reconstruction(*, provider_run: Mapping[str, Any], capture_root: Path,
                                  task_context: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return unused quote capacity only from the retained terminal provider bill."""
    root = capture_root / "pipeline" / "website_reconstruction"
    attempt = _latest_attempt(root)
    submission_path = root / _attempt_name("submission", attempt)
    admission_path = root / _attempt_name("controller_admission", attempt)
    operation_path = Path(provider_run.get("worldlabs_operation_manifest_uri") or root / "missing")
    if attempt == 1 and not admission_path.is_file() and submission_path.is_file():
        # The first deployed retry controller wrote its new admission at the
        # original path. Recognize only that exact binding, with the first
        # rejection already settled, so its completed operation can bill.
        legacy_path = root / "controller_admission.json"
        if legacy_path.is_file() and (root / "rejection_settlement.json").is_file():
            candidate = json.loads(legacy_path.read_text())
            submission = json.loads(submission_path.read_text())
            if candidate.get("allocation_binding_digest") == submission.get("request_digest"):
                admission_path = legacy_path
    if not submission_path.is_file() or not admission_path.is_file() or not operation_path.is_file():
        return None  # No settled bill means the full reservation remains charged.
    if not operation_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
        raise ValueError("website_reconstruction_billing_path_invalid")
    submission = json.loads(submission_path.read_text())
    admission = json.loads(admission_path.read_text())
    operation = json.loads(operation_path.read_text())
    credits = (operation.get("cost") or {}).get("total_credits")
    if operation.get("done") is not True or type(credits) is not int or credits < 0:
        return None
    if (operation.get("operation_id") != submission.get("operation_id")
            or operation.get("operation_id") != provider_run.get("provider_run_id")
            or submission.get("request_digest") != admission.get("allocation_binding_digest")
            or admission.get("task_context_digest") != task_context.get("context_digest")):
        raise ValueError("website_reconstruction_billing_binding_mismatch")
    from .website_task_context import website_webapp_request
    command = {"task_context_digest": task_context["context_digest"],
               "allocation_binding_digest": admission["allocation_binding_digest"], "provider": "world_labs",
               "operation_id": operation["operation_id"], "operation_done": True, "total_credits": credits,
               "provider_receipt_digest": canonical_digest(operation)}
    receipt = website_webapp_request(capture_id=task_context["capture_id"], operation="preparation-settlement",
        payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"], "settlement": command})
    if (any(receipt.get(key) != value for key, value in command.items())
            or receipt.get("status") != "settled" or receipt.get("actual_cost_usd") != credits / 1250):
        raise ValueError("website_reconstruction_settlement_receipt_invalid")
    write_json(root / _attempt_name("settlement", attempt), receipt)
    return receipt


def _require_settled_rejection(root: Path, state: Mapping[str, Any], attempt: int) -> None:
    if state.get("status") == "no_world_generated":
        proof_path = root / _attempt_name("unrecorded_attempt_proof", attempt)
        proof = json.loads(proof_path.read_text()) if proof_path.is_file() else {}
        if (state.get("operation_id") or proof.get("worlds") != []
                or proof.get("request_digest") != state.get("request_digest")
                or state.get("no_world_proof_digest") != canonical_digest(proof)):
            raise ValueError("website_reconstruction_unrecorded_attempt_proof_invalid")
        return  # Its full reservation stays charged: nothing proves why it was refused.
    rejection = state.get("provider_rejection") or {}
    settlement_path = root / _attempt_name("rejection_settlement", attempt)
    if (state.get("status") != "rejected_insufficient_credits" or state.get("operation_id")
            or rejection.get("code") != "worldlabs_api_402"
            or not str(rejection.get("detail") or "").startswith("Insufficient API credits to start world generation")
            or not settlement_path.is_file()):
        raise ValueError("website_reconstruction_rejection_evidence_invalid")
    settlement = json.loads(settlement_path.read_text())
    if (settlement.get("status") != "settled" or settlement.get("actual_cost_usd") != 0
            or settlement.get("allocation_binding_digest") != state["request_digest"]
            or settlement.get("rejection_code") != "insufficient_api_credits_before_generation"):
        raise ValueError("website_reconstruction_rejection_settlement_invalid")


def _attempt_binding(root: Path, base_binding: Mapping[str, Any], attempt: int) -> tuple[dict[str, Any], str | None]:
    """The binding attempt ``attempt`` carries: each retry names the rejected attempt before it."""
    binding, digest = dict(base_binding), None
    for index in range(attempt):
        state = json.loads((root / _attempt_name("submission", index)).read_text())
        if state.get("request_digest") != canonical_digest(binding):
            raise ValueError("website_reconstruction_already_bound_to_other_inputs")
        _require_settled_rejection(root, state, index)
        digest = canonical_digest(state)
        binding = {**base_binding, "rejected_attempt_digest": digest}
    return binding, digest


def rejected_generation_binding(*, capture_root: Path, base_binding: Mapping[str, Any]
                                ) -> tuple[dict[str, Any], str, int] | None:
    """Bind the current retry to the retained, explicit pre-generation credit rejections."""
    root = capture_root / "pipeline" / "website_reconstruction"
    first_path = root / "submission.json"
    if not first_path.is_file():
        return None
    first = json.loads(first_path.read_text())
    if first.get("request_digest") != canonical_digest(base_binding):
        raise ValueError("website_reconstruction_already_bound_to_other_inputs")
    latest = _latest_attempt(root)
    state = json.loads((root / _attempt_name("submission", latest)).read_text())
    target = latest + 1 if state.get("status") in _RESOLVED_ATTEMPT_STATUSES else latest
    if target == 0:
        return None
    if target > MAX_CREDIT_REJECTION_RETRIES:
        raise ValueError("website_reconstruction_credit_retry_limit_reached")
    binding, digest = _attempt_binding(root, base_binding, target)
    return binding, str(digest), target


def website_reconstruction_retry_state(*, descriptor: Mapping[str, Any], capture_root: Path,
                                       base_binding: Mapping[str, Any], provider: Any
                                       ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None, int]:
    """Retain old operations and admit only explicit pre-generation 402 retries."""
    root = capture_root / "pipeline" / "website_reconstruction"
    if not (root / "submission.json").is_file():
        return dict(base_binding), None, None, 0
    reconcile_website_credit_rejection(capture_root=capture_root, base_binding=base_binding,
        task_context=descriptor["metadata"]["site_task_context"])
    retry = rejected_generation_binding(capture_root=capture_root, base_binding=base_binding)
    if retry is None:
        return dict(base_binding), None, provider.submit(descriptor=descriptor, capture_root=capture_root), 0
    binding, retry_digest, attempt = retry
    if (root / _attempt_name("submission", attempt)).is_file():
        prepared = {**descriptor, "metadata": {**descriptor["metadata"],
            "website_reconstruction_retry_digest": retry_digest}}
        return binding, retry_digest, provider.submit(descriptor=prepared, capture_root=capture_root), attempt
    return binding, retry_digest, None, attempt


def reconcile_website_credit_rejection(*, capture_root: Path, base_binding: Mapping[str, Any],
                                       task_context: Mapping[str, Any]) -> bool:
    """Release a full reservation only for the controller's exact recorded HTTP 402."""
    root = capture_root / "pipeline" / "website_reconstruction"
    if not (root / "submission.json").is_file():
        return False
    first = json.loads((root / "submission.json").read_text())
    if first.get("request_digest") != canonical_digest(base_binding):
        raise ValueError("website_reconstruction_already_bound_to_other_inputs")
    attempt = _latest_attempt(root)
    path = root / _attempt_name("submission", attempt)
    state = json.loads(path.read_text())
    if state.get("status") in _RESOLVED_ATTEMPT_STATUSES:
        return True
    if state.get("status") != "submitting" or state.get("operation_id"):
        return False
    admission_path = root / _attempt_name("controller_admission", attempt)
    if not admission_path.is_file():
        return False
    admission = json.loads(admission_path.read_text())
    if admission.get("allocation_binding_digest") != state["request_digest"]:
        return False
    observed = state.get("observed_generation_refusal") or {}
    detail = _credit_rejection_detail(str(observed.get("failure_reason") or ""))
    if detail is not None:
        rejection = {"code": "worldlabs_api_402", "detail": detail,
                     "observed_generation_refusal_digest": canonical_digest(observed)}
        evidence = dict(observed)
    else:
        provider_path = capture_root / "pipeline" / "provider_run_manifest.json"
        provider = json.loads(provider_path.read_text()) if provider_path.is_file() else {}
        # The run manifest is rewritten per attempt; a 402 recorded before this
        # attempt's intent belongs to an earlier attempt and proves nothing here.
        detail = (_credit_rejection_detail(str(provider.get("failure_reason") or ""))
                  if (provider.get("status") == "failed" and not provider.get("provider_run_id")
                      and provider_path.stat().st_mtime_ns > path.stat().st_mtime_ns) else None)
        if detail is None:
            return _resolve_unrecorded_attempt(root=root, path=path, state=state, attempt=attempt)
        rejection = {"code": "worldlabs_api_402", "detail": detail,
                     "provider_run_manifest_digest": canonical_digest(provider)}
        evidence = provider
    state = {**state, "status": "rejected_insufficient_credits", "provider_rejection": rejection}
    write_json(root / _attempt_name("rejection_evidence", attempt), evidence)
    command = {"task_context_digest": task_context["context_digest"],
               "allocation_binding_digest": state["request_digest"], "provider": "world_labs",
               "rejection_code": "insufficient_api_credits_before_generation",
               "provider_receipt_digest": canonical_digest(rejection)}
    from .website_task_context import website_webapp_request
    receipt = website_webapp_request(capture_id=task_context["capture_id"], operation="preparation-settlement",
        payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"], "settlement": command})
    if (any(receipt.get(key) != value for key, value in command.items())
            or receipt.get("status") != "settled" or receipt.get("actual_cost_usd") != 0):
        raise ValueError("website_reconstruction_rejection_settlement_receipt_invalid")
    write_json(root / _attempt_name("rejection_settlement", attempt), receipt)
    write_json(path, state)
    return True


def _credit_rejection_detail(failure: str) -> str | None:
    """The provider's detail for an explicit pre-generation credit refusal, else None."""
    if not failure.startswith("worldlabs_api_402:"):
        return None
    try:
        error = json.loads(failure.partition(":")[2])
    except json.JSONDecodeError:
        return None
    detail = error.get("detail") if isinstance(error, dict) else None
    return str(detail) if str(detail or "").startswith("Insufficient API credits to start world generation") else None


def _resolve_unrecorded_attempt(*, root: Path, path: Path, state: Mapping[str, Any], attempt: int) -> bool:
    """Close an attempt with no recorded outcome only when World Labs holds no world for it."""
    if time.time() - path.stat().st_mtime < UNRECORDED_ATTEMPT_GRACE_SECONDS:
        return False
    with (root / "submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        from .provider_preview import _worldlabs_api_request
        query = {"tags": [f"bp-{state['request_digest'][7:31]}"], "page_size": 10}
        listing = _worldlabs_api_request("/marble/v1/worlds:list", method="POST", body=query)
        worlds = listing.get("worlds")
        if not isinstance(worlds, list) or listing.get("next_page_token"):
            return False
        if worlds:
            # A world was bought but never recorded; adopting it is a separate,
            # reviewed recovery. Never buy another over it.
            raise ValueError("website_reconstruction_unrecorded_attempt_generated_world")
        proof = {"schema_version": "website_reconstruction_unrecorded_attempt_proof.v1",
                 "request_digest": state["request_digest"], "query": query, "worlds": [],
                 "checked_at_iso": datetime.now(timezone.utc).isoformat()}
        write_json(root / _attempt_name("unrecorded_attempt_proof", attempt), proof)
        write_json(path, {**state, "status": "no_world_generated", "no_world_proof_digest": canonical_digest(proof)})
    return True


def validate_website_prepared_views(*, descriptor: Mapping[str, Any], capture_root: Path) -> tuple[list[Path], dict[str, Any]]:
    """Shared no-network preflight for the allocator and provider adapter."""
    metadata = descriptor.get("metadata") or {}
    clean_plate = metadata.get("clean_plate") or {}
    preparation = clean_plate.get("prepared_views") or {}
    profile = preparation.get("reconstruction_profile")
    if profile is not None and profile != {"provider": "world_labs", "model": "marble-1.1-plus", "max_input_images": 8}:
        raise ValueError("website_reconstruction_profile_requires_matching_provider_adapter")
    frames = preparation.get("frames") or []
    task_context = metadata.get("site_task_context") or {}
    task_digest = sha256(json.dumps(task_context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (clean_plate.get("status") not in {"noop", "objects_removed"}
            or clean_plate.get("privacy_verified") is not True
            or task_context.get("confirmed") is not True
            or preparation.get("status") != "ready" or not 2 <= len(frames) <= 8
            or preparation.get("task_context_sha256") != task_digest):
        raise ValueError("website_prepared_images_required")
    if preparation.get("digest") != canonical_digest(preparation, digest_field="digest"):
        raise ValueError("website_prepared_images_digest_mismatch")
    image_paths = []
    for frame in frames:
        path = Path(frame["image_path"]).resolve()
        if not path.is_relative_to(capture_root.resolve() / "pipeline") or not path.is_file() or _sha256_file(path) != frame.get("image_digest"):
            raise ValueError("website_prepared_image_changed_or_outside_pipeline")
        image_paths.append(path)
    binding = {"prepared_views_digest": preparation["digest"], "model": "marble-1.1-plus",
               "scene_id": descriptor["scene_id"], "capture_id": descriptor["capture_id"]}
    return image_paths, _equivalent_retained_binding(capture_root=capture_root, binding=binding, frames=frames)


def _view_sequence(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"frame_id": frame.get("frame_id"), "image_digest": frame.get("image_digest")} for frame in frames]


def _equivalent_retained_binding(*, capture_root: Path, binding: Mapping[str, Any],
                                 frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A purchased world stays bound to the exact images it was generated from.

    The prepared-views record also carries review receipts, so a re-review of
    byte-identical images changes its digest but not what Marble consumed.
    Only the same images in the same order (the first view anchors the world)
    with the same model, scene and capture reuse the submitted operation; any
    other difference keeps the refusal. The reuse is recorded, never silent.
    """
    root = capture_root / "pipeline" / "website_reconstruction"
    path = root / "submission.json"
    if not path.is_file():
        return dict(binding)
    state = json.loads(path.read_text())
    prior = state.get("binding") or {}
    if (state.get("request_digest") == canonical_digest(binding) or state.get("status") != "submitted"
            or not state.get("operation_id") or state.get("request_digest") != canonical_digest(prior)
            or {key: value for key, value in prior.items() if key != "prepared_views_digest"}
            != {key: value for key, value in binding.items() if key != "prepared_views_digest"}):
        return dict(binding)
    retained = state.get("prepared_views")
    if retained is None:
        # Submissions before the view sequence was retained: the run manifest
        # recorded the frames sent under this exact request digest.
        manifest_path = capture_root / "pipeline" / "worldlabs_request_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        retained = (_view_sequence(manifest.get("prepared_frames") or [])
                    if manifest.get("request_digest") == state["request_digest"] else None)
    if not retained or retained != _view_sequence(frames):
        return dict(binding)
    receipt = {"schema_version": "website_reconstruction_rebinding.v1", "basis": "identical_ordered_prepared_images",
               "prior_request_digest": state["request_digest"],
               "prior_prepared_views_digest": prior["prepared_views_digest"],
               "prepared_views_digest": binding["prepared_views_digest"], "view_sequence": retained}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    receipt_path = root / "rebindings" / f"{binding['prepared_views_digest'][7:]}.json"
    if not receipt_path.is_file():
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, receipt)
    return dict(prior)


def validate_website_reconstruction_admission(admission: Mapping[str, Any], request_digest: str) -> None:
    budget = admission.get("maximum_cost_usd")
    if (isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget)
            or budget < MAX_GENERATION_COST_USD):
        raise ValueError("website_reconstruction_budget_insufficient")
    if admission.get("external_disclosure_allowed") is not True:
        raise ValueError("website_reconstruction_disclosure_not_authorized")
    if admission.get("allocation_binding_digest") != request_digest:
        raise ValueError("website_reconstruction_spend_binding_missing")


def submit_website_prepared_views(*, descriptor: Mapping[str, Any], capture_root: Path,
                                 api_request: Callable[..., Any], upload: Callable[..., Any],
                                 admission_grant: PaidResourceAdmissionGrant | None = None) -> dict[str, Any]:
    image_paths, binding = validate_website_prepared_views(descriptor=descriptor, capture_root=capture_root)
    metadata = descriptor["metadata"]
    frames = metadata["clean_plate"]["prepared_views"]["frames"]
    retry_digest = metadata.get("website_reconstruction_retry_digest")
    attempt = 0
    if retry_digest:
        retry_binding = rejected_generation_binding(capture_root=capture_root, base_binding=binding)
        if retry_binding is None or retry_binding[1] != retry_digest:
            raise ValueError("website_reconstruction_retry_binding_invalid")
        binding, _, attempt = retry_binding
    request_digest = canonical_digest(binding)
    root = capture_root / "pipeline" / "website_reconstruction"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / _attempt_name("submission", attempt)
    with (root / "submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("website_reconstruction_submission_in_progress") from exc
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if state.get("request_digest") != request_digest:
                raise ValueError("website_reconstruction_already_bound_to_other_inputs")
            if not state.get("operation_id"):
                # A timeout after POST may have purchased a world. Do not buy
                # another while the first request's outcome is unknown.
                raise ValueError("website_reconstruction_submission_requires_reconciliation")
            operation = state["operation"]
            generation = state["generation_request"]
        else:
            admission = metadata.get("website_reconstruction_admission") or {}
            validate_website_reconstruction_admission(admission, request_digest)
            require_paid_resource_admission_grant(admission_grant, resource_class="provider_reconstruction_api",
                                                  allocation_binding_digest=request_digest, require_allocation_binding=True)
            content = []
            for path in image_paths:
                prepared = api_request("/marble/v1/media-assets:prepare_upload", method="POST",
                                       body={"file_name": path.name, "extension": path.suffix.lstrip("."), "kind": "image"})
                asset = prepared.get("media_asset") or {}
                asset_id = asset.get("media_asset_id") or asset.get("id")
                info = prepared.get("upload_info") or {}
                if not asset_id or not info.get("upload_url"):
                    raise ValueError("website_reconstruction_image_upload_invalid")
                upload(info["upload_url"], method=info.get("upload_method") or "PUT", content_type="image/png",
                       data=path.read_bytes(), required_headers=info.get("required_headers") or {})
                # Camera headings are estimates without a measured gravity
                # frame. Marble supports automatic layout when azimuth is absent.
                content.append({"content": {"source": "media_asset", "media_asset_id": asset_id}})
            generation = {"model": "marble-1.1-plus", "permission": {"public": False},
                          "display_name": str(descriptor["capture_id"])[:64],
                          "tags": [f"bp-{request_digest[7:31]}"],
                          "world_prompt": {"type": "multi-image", "multi_image_prompt": content,
                                           "reconstruct_images": True,
                                           "text_prompt": "Reconstruct the same real work area shown in these prepared views. Preserve its layout, supports and remaining objects. Do not add objects into cleared regions."}}
            state = {"request_digest": request_digest, "binding": binding, "status": "submitting",
                     "generation_request": generation, "prepared_views": _view_sequence(frames)}
            # Persist intent before the billable mutation.
            with state_path.open("x") as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            try:
                operation = api_request("/marble/v1/worlds:generate", method="POST", body=generation)
            except RuntimeError as exc:
                if _credit_rejection_detail(str(exc)) is not None:
                    # Keep the refusal with this attempt: the run manifest that
                    # also records it is rewritten by every later pass.
                    state.update(observed_generation_refusal={"failure_reason": str(exc)[:2000],
                        "observed_at_iso": datetime.now(timezone.utc).isoformat()})
                    temporary = root / "submission.tmp"
                    write_json(temporary, state)
                    os.replace(temporary, state_path)
                raise
            operation_id = operation.get("operation_id") or operation.get("id")
            if not operation_id:
                raise ValueError("website_reconstruction_operation_id_missing")
            state.update(status="submitted", operation_id=operation_id, operation=operation)
            temporary = root / "submission.tmp"
            write_json(temporary, state)
            os.replace(temporary, state_path)
        return {"provider_name": "world_labs", "provider_model": "marble-1.1-plus",
                "provider_run_id": state["operation_id"], "worldlabs_operation_id": state["operation_id"],
                "status": "processing", "artifact_uris": {}, "worldlabs_operation": operation,
                "generation_source_type": "prepared_multi_image", "privacy_safe_input": True,
                "worldlabs_request_manifest": {"request_digest": request_digest, "binding": binding,
                                                "generation_request": generation, "prepared_frames": frames},
                "cost_usd": 0.0, "failure_reason": None}
