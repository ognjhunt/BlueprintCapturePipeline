"""Durable admitted Atlas generation, editing and mesh tasks. ADP-040 / Day 28.

Reuses the existing pose adapter's audited transport, uploads, locking and
retention. Posing, native refined splats and mesh extraction share one ledger.
"""
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from . import website_atlas_pose as pose
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .paid_resource_admission import require_paid_resource_admission_grant

TASKS = frozenset({"atlasGenerate", "atlasMasked", "atlasChisel", "atlasTextToImage", "posedImages2Splats", "splats2Mesh"})
MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".exr": "image/x-exr", ".ply": "application/octet-stream", ".spz": "application/octet-stream"}
_CAPACITY_CODES = frozenset({"RATE_LIMIT_EXCEEDED", "CONCURRENCY_LIMIT_EXCEEDED", "TASK_UNAVAILABLE"})


def _capacity_refusal(payload: bytes, retry_after: str | None, *, now: float,
                      attempt: int) -> dict[str, Any] | None:
    """A documented 429 proves no purchase; unknown replies still reconcile.

    https://atlas-beta.worldlabs.ai/docs/rate-limits
    Retain the deadline so a worker restart cannot shorten the provider's wait.
    """
    try:
        body = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if (not isinstance(error, dict) or not isinstance(error.get("code"), str)
            or error["code"] not in _CAPACITY_CODES):
        return None
    wait = 0.0
    if retry_after:
        try:
            wait = float(retry_after)
        except ValueError:
            try:
                wait = parsedate_to_datetime(retry_after).timestamp() - now
            except (ValueError, TypeError, OverflowError):
                pass
    if not math.isfinite(wait):
        wait = 0.0
    backoff = min(300, 60 * 2 ** min(attempt - 1, 3))
    delay = max(wait, backoff) + random.uniform(0, 5)
    return {"status": 429, "code": error["code"], "accepted": False,
            "observed_at_epoch": now, "retry_not_before_epoch": now + delay}


def _capacity_pending(state: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "processing", "operation_id": None, "waiting_for": "provider_capacity",
            "retry_not_before_epoch": state["capacity_refusal"]["retry_not_before_epoch"]}


def validate_task_request(task: str, request: Mapping[str, Any]) -> None:
    """Pinned official OpenAPI request schema plus Blueprint disclosure limits."""
    if task not in TASKS:
        raise ValueError("atlas_task_unavailable")
    import jsonschema
    schema = json.loads((Path(__file__).parent / "_atlas_schemas" / (task + ".json")).read_text())
    root = schema["$ref"].split("/")[-1]
    allowed = schema["components"]["schemas"][root]["properties"]
    if set(request) - set(allowed):
        raise ValueError("atlas_task_unknown_fields")
    try:
        jsonschema.Draft202012Validator(schema).validate(dict(request))
    except jsonschema.ValidationError:
        raise ValueError("atlas_task_request_schema_invalid") from None
    if request.get("webhookUrl") or request.get("base64Outputs") is True:
        raise ValueError("atlas_task_durable_assets_required")

    def visit(value):
        if isinstance(value, Mapping):
            if any(key in value for key in ("url", "base64")):
                raise ValueError("atlas_task_private_project_assets_required")
            if "assetId" in value and not (isinstance(value["assetId"], str) and value["assetId"]):
                raise ValueError("atlas_task_asset_invalid")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("atlas_task_nonfinite_value")
    visit(request)
    if task == "posedImages2Splats":
        frames = request["posedFrames"]
        grids = set()
        for frame in frames:
            k, e = frame["camera"]["intrinsics"], frame["camera"]["extrinsics"]
            grids.add((k["width"], k["height"]))
            if not (frame.get("depth") or {}).get("depthAsset"):
                raise ValueError("atlas_splat_linear_depth_required")
            if not math.isclose(sum(v * v for v in e["quaternion"]), 1, abs_tol=1e-3):
                raise ValueError("atlas_task_camera_quaternion_invalid")
            if k["fx"] <= 0 or k["fy"] <= 0:
                raise ValueError("atlas_splat_focal_length_invalid")
        if len(grids) != 1:
            raise ValueError("atlas_splat_image_grids_must_match")
    for camera in request.get("targetCameras", []):
        if (camera["intrinsics"]["width"], camera["intrinsics"]["height"]) != (1280, 720):
            raise ValueError("atlas_task_target_grid_invalid")
        q = camera["extrinsics"]["quaternion"]
        if not math.isclose(sum(v * v for v in q), 1, abs_tol=1e-3):
            raise ValueError("atlas_task_camera_quaternion_invalid")
    if task == "atlasGenerate":
        pose_context = {"id": "validation", "done": True, "response": {"frames": request["contextFrames"]}}
        from .website_worldlabs import _atlas_posed_context
        _atlas_posed_context(operation=pose_context, expected_frame_count=len(request["contextFrames"]),
                             target_cameras=request["targetCameras"])
        if request.get("returnDepth") is True:
            centers = [r["camera"]["extrinsics"]["position"] for r in request["contextFrames"]]
            centers += [r["extrinsics"]["position"] for r in request["targetCameras"]]
            if max(math.dist(centers[0], p) for p in centers) < 1e-6:
                raise ValueError("atlas_generated_depth_requires_translation")
            # Routing is automatic. Some routes only reconstruct context on
            # the centered output grid. Portrait context cannot be assumed to
            # anchor a one-target reconstruction on every available route.
            eligible = []
            for frame in request["contextFrames"]:
                k = frame["camera"]["intrinsics"]
                if ((k["width"], k["height"]) == (1280, 720)
                        and math.isclose(k["cx"], 640, abs_tol=1e-3)
                        and math.isclose(k["cy"], 360, abs_tol=1e-3)):
                    eligible.append(frame["camera"]["extrinsics"]["position"])
            reliable_centers = eligible + [r["extrinsics"]["position"] for r in request["targetCameras"]]
            if max(math.dist(reliable_centers[0], p) for p in reliable_centers) < 1e-6:
                raise ValueError("atlas_generated_depth_anchor_baseline_missing")
    if task == "atlasMasked":
        context = request["contextFrames"]
        if len(context) != len(request["targetCameras"]):
            raise ValueError("atlas_masked_view_pairing_invalid")
        for frame, target in zip(context, request["targetCameras"], strict=True):
            if frame["camera"] != target:
                raise ValueError("atlas_masked_view_pairing_invalid")
    if task == "atlasChisel":
        context = request.get("contextFrames") or []
        targets = request["targetCameras"]
        if context and (len(context) != len(targets) or any(
                frame["camera"] != target for frame, target in zip(context, targets, strict=True))):
            raise ValueError("atlas_chisel_depth_pairing_invalid")
        if len(context) + len(targets) + len(request.get("sourceFrames") or []) > 64:
            raise ValueError("atlas_chisel_rollout_limit_exceeded")
    if task in {"atlasMasked", "atlasChisel"} and request.get("returnDepth") is True:
        targets = request["targetCameras"]
        centers = [r["extrinsics"]["position"] for r in targets]
        if len(targets) < 2 or max(math.dist(centers[0], p) for p in centers) < 1e-6:
            raise ValueError("atlas_task_depth_requires_translated_targets")


def _price(review: Mapping[str, Any], task: str, request: Mapping[str, Any]) -> float:
    if (review.get("schema_version") != "website_atlas_task_price_review.v1"
            or review.get("task") != task or review.get("request_digest") != canonical_digest(request)
            or not pose._positive_finite(review.get("maximum_cost_usd"))
            or review["maximum_cost_usd"] > pose.MAX_REVIEWED_COST_USD
            or not review.get("rate_card_reference") or not review.get("reviewed_by")
            or not pose._reviewed_recently(review.get("reviewed_on"), datetime.now(timezone.utc).date())):
        raise ValueError("atlas_task_reviewed_price_required")
    return float(review["maximum_cost_usd"])


def _replace_uploads(value: Any, uploaded: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {k: (uploaded[v[7:]] if k == "assetId" and isinstance(v, str) and v.startswith("upload:")
                    else _replace_uploads(v, uploaded)) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_uploads(v, uploaded) for v in value]
    return value


def _collect(client: pose.AtlasClient, operation: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    refs = []
    response = operation["response"]
    for count, ref in (response.get("refinedSplats") or {}).items():
        if not isinstance(count, str) or not count.isdecimal() or int(count) <= 0:
            raise ValueError("atlas_splat_count_invalid")
        refs.append((f"refinedSplats-{count}", pose._safe_id(ref.get("assetId"))))
    if response.get("mesh"):
        refs.append(("mesh", pose._safe_id(response["mesh"].get("assetId"))))
    for index, frame in enumerate(response.get("frames") or []):
        refs.append((f"frames-{index}-imageAsset", pose._safe_id(frame["imageAsset"].get("assetId"))))
        for role in ("depthAsset", "logDepthAsset", "confidenceAsset"):
            ref = (frame.get("depth") or {}).get(role)
            if ref:
                refs.append((f"frames-{index}-depth-{role}", pose._safe_id(ref.get("assetId"))))
    if not refs:
        raise ValueError("atlas_task_output_assets_missing")
    retained = []
    for role, asset_id in refs:
        url = client.call("POST", f"/assets/{asset_id}:createReadUrl").get("readUrl")
        is_splat = role.startswith("refinedSplats-")
        data = client.fetch_bytes(url, max_bytes=pose.MAX_MESH_OUTPUT_BYTES if role == "mesh" or is_splat else pose.MAX_OUTPUT_BYTES)
        asset = client.call("GET", f"/assets/{asset_id}")
        if asset.get("id") != asset_id:
            raise ValueError("atlas_task_asset_identity_mismatch")
        pose._require_reported_size(asset, len(data), required=True)
        if role.endswith("depthAsset") and not data.startswith(pose._EXR_MAGIC):
            raise ValueError("website_atlas_depth_not_exr")
        if role == "mesh":
            import struct
            if len(data) < 12 or data[:4] != b"glTF" or struct.unpack("<II", data[4:12]) != (2, len(data)):
                raise ValueError("atlas_task_mesh_not_glb")
        if is_splat:
            import gzip
            import io
            import struct
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                    header = compressed.read(16)
                magic, _version, count = struct.unpack_from("<III", header)
            except (OSError, EOFError, struct.error):
                raise ValueError("atlas_task_splat_not_spz") from None
            if magic != 0x5053474E or count != int(role.split("-")[-1]):
                raise ValueError("atlas_task_splat_count_mismatch")
        mime = asset.get("mimeType", "")
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/x-exr": ".exr",
                  "model/gltf-binary": ".glb"}.get(mime, ".bin")
        if is_splat:
            suffix = ".spz"
        elif data.startswith(b"glTF"):
            suffix = ".glb"
        elif data.startswith(pose._EXR_MAGIC):
            suffix = ".exr"
        elif data.startswith(b"\x89PNG"):
            suffix = ".png"
        path = Path("outputs") / (role + suffix)
        pose._write_durable(root / path, data)
        retained.append({"role": role, "asset_id": asset_id, "path": str(path),
                         "sha256": pose._sha256_bytes(data), "bytes": len(data)})
    return retained


def _camera_matches(returned: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
    """Allow float32 wire rounding, never a different camera or pixel grid."""
    if any(returned["intrinsics"][key] != requested["intrinsics"][key] for key in ("width", "height")):
        return False
    if returned["extrinsics"].get("coordinateSystem", "rub") != requested["extrinsics"].get("coordinateSystem", "rub"):
        return False
    pairs = [(returned["intrinsics"][key], requested["intrinsics"][key]) for key in ("fx", "fy", "cx", "cy")]
    for key in ("position", "quaternion"):
        pairs.extend(zip(returned["extrinsics"][key], requested["extrinsics"][key], strict=True))
    return all(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in pairs)


def _validate_response(task: str, request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    import jsonschema
    schema = json.loads((Path(__file__).parent / "_atlas_schemas" / (task + ".json")).read_text())
    try:
        jsonschema.Draft202012Validator({**schema, "$ref": schema["x-response-schema"]}).validate(response)
    except jsonschema.ValidationError:
        raise ValueError("atlas_task_response_schema_invalid") from None
    if task == "posedImages2Splats" and not response.get("refinedSplats"):
        raise ValueError("atlas_task_refined_splats_missing")
    if "targetCameras" in request:
        frames = response.get("frames")
        if not isinstance(frames, list) or len(frames) != len(request["targetCameras"]):
            raise ValueError("atlas_task_response_frame_count_mismatch")
        for frame, camera in zip(frames, request["targetCameras"], strict=True):
            if not _camera_matches(frame["camera"], camera):
                raise ValueError("atlas_task_response_camera_mismatch")
            if request.get("returnDepth") is True and not (frame.get("depth") or {}).get("depthAsset"):
                raise ValueError("atlas_task_response_depth_missing")


def _reconcile(client: pose.AtlasClient, state: dict[str, Any], path: Path, *, task: str,
               now: float) -> dict[str, Any]:
    """Settle an uncertain send from the operation list (the caller holds the lock).

    A bought operation is adopted; a send proven to have bought nothing is
    reopened for the identical resend under the same key, at most
    RECONCILE_REOPEN_LIMIT times, then closed.
    """
    # Older records kept no send times; the record was written beside the send.
    sent = [row["at_epoch"] for row in state.get("sends") or []] or [path.stat().st_mtime]
    found = pose.reconcile_atlas_submission(client, task_name=task, request=state["request"],
                                            owned_assets=state.get("created_assets", []), sent_at=sent, now=now)
    state.setdefault("reconciliations", []).append(found)
    reopens = state.get("reconciliation_reopens", 0)
    if found["outcome"] == "bought":
        state.update(status="submitted", operation_id=found["operation_id"])
    elif found["outcome"] == "lapsed":
        state.update(status="failed", terminal_error={"reason": "reconciliation_window_lapsed"})
    elif reopens >= pose.RECONCILE_REOPEN_LIMIT:
        state.update(status="rejected",
                     provider_rejection={"status": None, "retryable": False, "reconciled_unbought": True})
    else:
        state.update(status="resend", reconciliation_reopens=reopens + 1)
    pose._write_state(path, state)
    return state


def _run_admitted_atlas_task_attempt(*, task: str, request: Mapping[str, Any], output_root: Path,
                            source_video_digest: str, task_context: Mapping[str, Any],
                            price_review: Mapping[str, Any], release_blockers: Callable,
                            uploads: Mapping[str, Mapping[str, Any]] | None = None,
                            transport=None, now=time.time, sleep=time.sleep, max_polls=2,
                            poll_interval_seconds=15, retry_lineage=None) -> dict[str, Any]:
    """One exact, reserved task. Ambiguous submissions hold; reruns reuse intent."""
    from .website_task_context import reserve_website_preparation_spend
    from .atlas_preparation_runtime import preparation_source_checkpoint
    from .atlas_preparation_context import validate_preparation_task_context
    from .owned_capture_atlas_project import owned_atlas_project_checkpoint
    validate_task_request(task, request)
    validate_preparation_task_context(task_context)
    uploads = dict(uploads or {})
    referenced = set()
    masks = set()
    def refs(value, is_mask=False):
        if isinstance(value, Mapping):
            asset = value.get("assetId")
            if isinstance(asset, str) and asset.startswith("upload:"):
                referenced.add(asset[7:])
                if is_mask:
                    masks.add(asset[7:])
            for key, item in value.items():
                refs(item, key == "maskAsset")
        elif isinstance(value, list):
            for item in value:
                refs(item)
    refs(request)
    if referenced != set(uploads):
        raise ValueError("atlas_task_upload_references_mismatch")
    for label, row in uploads.items():
        pose._safe_id(label)
        path = Path(row["path"])
        if path.is_symlink() or path.suffix.lower() not in MIMES or _sha256_file(path) != row["sha256"]:
            raise ValueError("atlas_task_upload_binding_invalid")
        if label in masks:
            from PIL import Image
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (1280, 720):
                    raise ValueError("atlas_task_grayscale_keep_mask_required")
    cost = _price(price_review, task, request)
    retention = pose.atlas_provider_retention()
    binding = {"task": task, "request_template": dict(request), "source_video_digest": source_video_digest,
               "task_context_digest": task_context["context_digest"], "uploads": uploads,
               "maximum_cost_usd": cost, "price_review_digest": canonical_digest(price_review),
               "provider_retention": retention, "provider": pose.PROVIDER}
    if retry_lineage is not None:
        binding["retry_lineage"] = retry_lineage
    request_digest = canonical_digest(binding)
    root = output_root.resolve()
    state_path = root / "atlas_pose" / "submission.json"
    base, key = pose.atlas_api_config()
    client = pose.AtlasClient(base_url=base, api_key=key, transport=transport,
                             source_checkpoint=preparation_source_checkpoint(task_context),
                             project_checkpoint=owned_atlas_project_checkpoint(task_context))
    with pose._locked(root):
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        if state and state.get("request_digest") != request_digest:
            raise ValueError("atlas_task_binding_changed")
        if state and state.get("status") == "completed":
            result = json.loads((root / "result.json").read_text())
            if result.get("digest") != canonical_digest(result, digest_field="digest") or result["digest"] != state["result_digest"]:
                raise ValueError("atlas_task_result_changed")
            for row in result["outputs"]:
                if _sha256_file(root / row["path"]) != row["sha256"]:
                    raise ValueError("atlas_task_output_changed")
            return result
        if state and state.get("status") == "submitting":
            if state.get("api_base") != base:
                raise ValueError("atlas_task_api_base_changed")
            state = _reconcile(client, state, state_path, task=task, now=now())
        if state and state.get("status") == "capacity_wait":
            if state.get("api_base") != base:
                raise ValueError("atlas_task_api_base_changed")
            deadline = state["capacity_refusal"]["retry_not_before_epoch"]
            if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                raise ValueError("atlas_task_capacity_deadline_invalid")
            if now() < deadline:
                return _capacity_pending(state)
            state["status"] = "resend"
            pose._write_state(state_path, state)
        if state and state.get("status") == "rejected":
            raise ValueError("atlas_task_attempt_rejected")
        if state and state.get("status") == "failed" and not state.get("operation_id"):
            raise ValueError("atlas_task_reconciliation_window_lapsed")
        if not state or not state.get("operation_id"):
            commit, blockers = release_blockers()
            if blockers:
                raise ValueError("atlas_task_release_not_admitted:" + ",".join(blockers))
            admission_path = root / "controller_admission.json"
            retained = json.loads(admission_path.read_text()) if admission_path.is_file() else None
            admission, grant = reserve_website_preparation_spend(task_context=task_context, binding_digest=request_digest,
                maximum_cost_usd=cost, request_count=1, resource_class=pose.RESOURCE_CLASS,
                provider=pose.PROVIDER, source_video_digest=source_video_digest, retained_admission=retained,
                retain_admission=lambda value: pose._write_state(admission_path, {**value, "source_commit": commit}))
            pose._require_admission(admission, request_digest=request_digest, binding=binding, now=now())
            require_paid_resource_admission_grant(grant, resource_class=pose.RESOURCE_CLASS,
                allocation_binding_digest=request_digest, require_allocation_binding=True)
            if admission.get("provider_retention", retention) != retention:
                raise ValueError("atlas_task_retention_admission_mismatch")
            pose._write_state(admission_path, {**admission, "source_commit": commit})
            if not state:
                state = {"status": "uploading", "request_digest": request_digest, "binding": binding,
                         "created_assets": [], "uploads": [], "api_base": base,
                         "idempotency_key": "bp-atlas-task-" + request_digest[7:39]}
                pose._write_state(state_path, state)
            if state.get("status") == "uploading":
                for label, row in uploads.items():
                    if any(r["frame_id"] == label for r in state["uploads"]):
                        continue
                    frame = {"frame_id": label, "image_path": row["path"], "image_digest": row["sha256"]}
                    uploaded = pose._upload_frame(client, frame, name="bp-" + request_digest[7:19], state=state,
                        path=state_path, content_type=MIMES[Path(row["path"]).suffix.lower()])
                    state["uploads"].append(uploaded)
                    pose._write_state(state_path, state)
                pose._await_ready(client, state, state_path, sleep)
                body = _replace_uploads(request, {r["frame_id"]: r["asset_id"] for r in state["uploads"]})
                validate_task_request(task, body)
                state.update(request=body, request_body=pose._canonical_json(body))
            elif state.get("status") != "resend":
                # Never invent a second purchase after an unknown response.
                raise ValueError("atlas_task_submission_requires_reconciliation")
            pose._require_admission(admission, request_digest=request_digest, binding=binding, now=now())
            # Until an answer proves otherwise, this send may buy an operation.
            state["status"] = "submitting"
            state.setdefault("sends", []).append({"at_epoch": now(), "status": None})
            pose._write_state(state_path, state)
            # The same bytes under the same key: a lost answer replays the original operation.
            status, payload = client.send_raw("POST", "/tasks:" + task, state["request_body"].encode(),
                                              headers={"Idempotency-Key": state["idempotency_key"]})
            state["submission_http_status"] = state["sends"][-1]["status"] = status
            if status == 429:
                count = state.get("capacity_refusals", 0) + 1
                refusal = _capacity_refusal(payload, client.retry_after, now=now(), attempt=count)
                if refusal is not None:
                    state.update(status="capacity_wait", capacity_refusals=count, capacity_refusal=refusal)
                    state["sends"][-1]["capacity_refusal"] = refusal
                    pose._write_state(state_path, state)
                    return _capacity_pending(state)
            if 400 <= status < 500 and status not in pose._UNCERTAIN_4XX and status != 402:
                # Any earlier send was reconciled as bought nothing, so a definitive
                # refusal of this one closes the task without a purchase.
                state.update(status="rejected", provider_rejection={"status": status, "retryable": False})
            pose._write_state(state_path, state)
            operation = pose._without_urls(pose._json_object(status, payload))
            state.update(status="submitted", operation_id=pose._safe_id(operation.get("id")))
            pose._write_state(state_path, state)
        if state["api_base"] != base:
            raise ValueError("atlas_task_api_base_changed")
        operation = state.get("terminal_operation") if state.get("status") == "failed" else None
        if operation is None:
            for attempt in range(max_polls):
                operation = pose._without_urls(client.call("GET", "/operations/" + state["operation_id"]))
                if operation.get("id") != state["operation_id"]:
                    raise ValueError("atlas_task_operation_identity_mismatch")
                if operation.get("done") is True:
                    break
                if attempt + 1 < max_polls:
                    sleep(poll_interval_seconds)
            else:
                return {"status": "processing", "operation_id": state["operation_id"]}
        if operation.get("id") != state["operation_id"] or operation.get("done") is not True:
            raise ValueError("atlas_task_terminal_operation_invalid")
        if operation.get("error") is not None:
            if state.get("status") != "failed":
                state.update(status="failed", terminal_operation=operation, terminal_observed_at_epoch=now())
                pose._write_state(state_path, state)
            raise ValueError("atlas_task_operation_failed")
        _validate_response(task, state["request"], operation.get("response"))
        outputs = _collect(client, operation, root)
        result = {"schema_version": "website_atlas_task_result.v1", "status": "completed", "task": task,
                  "operation": operation, "binding": binding, "request_digest": request_digest,
                  "outputs": outputs, "billing": {"status": "unreported", "reserved_usd": cost},
                  "claim_ceiling": "development_only", "physical_evidence": False,
                  "metric_measurement_proven": False, "scale_status": "unscaled_provider_world_units",
                  "generated_content": task != "splats2Mesh"}
        result["digest"] = canonical_digest(result, digest_field="digest")
        pose._write_state(root / "result.json", result)
        state.update(status="completed", result_digest=result["digest"])
        pose._write_state(state_path, state)
        return result


def _transient_upstream_failure(state):
    return pose.resubmittable_terminal_failure(state)


def run_admitted_atlas_task(*, task: str, request: Mapping[str, Any], output_root: Path,
                            source_video_digest: str, task_context: Mapping[str, Any],
                            price_review: Mapping[str, Any], release_blockers: Callable,
                            uploads: Mapping[str, Mapping[str, Any]] | None = None,
                            transport=None, now=time.time, sleep=time.sleep, max_polls=2,
                            poll_interval_seconds=15) -> dict[str, Any]:
    """At most three admitted operations; only confirmed upstream failures retry.

    Every attempt keeps its own intent, reservation and idempotency key. A
    pending or ambiguous send never creates another purchase. Completed inputs
    and the caller's request remain unchanged across attempts and restarts.
    """
    options = dict(task=task, request=request, source_video_digest=source_video_digest,
                   task_context=task_context, price_review=price_review,
                   release_blockers=release_blockers, uploads=uploads, transport=transport,
                   now=now, sleep=sleep, max_polls=max_polls,
                   poll_interval_seconds=poll_interval_seconds)

    def _retry_task(root, lineage=None):
        try:
            return _run_admitted_atlas_task_attempt(output_root=root, retry_lineage=lineage, **options)
        except ValueError as exc:
            if str(exc) != "atlas_task_operation_failed":
                raise
            state_path = root / "atlas_pose/submission.json"
            with pose._locked(root):
                failed = json.loads(state_path.read_text())
                attempt = (lineage or {}).get("attempt", 0)
                if not _transient_upstream_failure(failed) or attempt >= 2:
                    raise
                failure_digest = canonical_digest(failed["terminal_operation"])
                failure_observed_at = failed.get("terminal_observed_at_epoch", state_path.stat().st_mtime)
                following = {"attempt": attempt + 1,
                             "parent_request_digest": failed["request_digest"],
                             "parent_operation_id": failed["operation_id"],
                             "terminal_operation_digest": failure_digest}
            retry_root = root / "retry"
            if retry_root.is_symlink():
                raise ValueError("atlas_task_retry_root_symlink")
            if not (retry_root / "atlas_pose/submission.json").exists():
                sleep(30)
            result = _retry_task(retry_root, following)
            if result.get("status") != "completed":
                return result
            # Keep the normal portable result at the caller's root. Failed
            # provider receipts remain in the parent state and retry history.
            result = {**result, "outputs": [
                {**row, "path": str(Path("retry") / row["path"])} for row in result["outputs"]],
                "retry_history": [{"operation_id": failed["operation_id"],
                                   "request_digest": failed["request_digest"],
                                   "terminal_operation": failed["terminal_operation"],
                                   "observed_at_epoch": failure_observed_at,
                                   "reserved_usd": failed["binding"]["maximum_cost_usd"]}]
                                  + result.get("retry_history", []),
                "billing": {**result["billing"], "total_reserved_usd": round(
                    failed["binding"]["maximum_cost_usd"]
                    + result["billing"].get("total_reserved_usd", result["billing"]["reserved_usd"]), 6)}}
            result["digest"] = canonical_digest(result, digest_field="digest")
            with pose._locked(root):
                current = json.loads(state_path.read_text())
                if (current["request_digest"] != failed["request_digest"]
                        or current["operation_id"] != failed["operation_id"]
                        or canonical_digest(current.get("terminal_operation")) != failure_digest):
                    raise ValueError("atlas_task_retry_parent_changed")
                pose._write_state(root / "result.json", result)
                current.update(status="completed", result_digest=result["digest"],
                               completed_operation_id=result["operation"]["id"],
                               retry_history=result["retry_history"])
                pose._write_state(state_path, current)
            return result

    return _retry_task(output_root.resolve())
