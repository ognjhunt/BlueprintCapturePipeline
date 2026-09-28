"""ADP-011/day 7: the observation boundary for customer-hosted inference.

The trusted simulator supplies bytes and numbers, never asset paths. This
projector constructs a new wire object instead of redacting a scene manifest.
Image metadata is stripped so a PNG text chunk cannot export a private path.
The exact returned PNG bytes must be retained as the policy-input evidence.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
from typing import Any, Mapping

from PIL import Image

from .company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from .controlled_policy_actions import validate_action_response
from .core.security_controls import fetch_bounded_https

MAX_WIRE_BYTES = 8 * 1024 * 1024


def _state(value: Any, shape: list[int]) -> Any:
    if not shape:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("controlled_policy_state_value_invalid")
        return float(value)
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("controlled_policy_state_shape_invalid")
    return [_state(child, shape[1:]) for child in value]


def project_controlled_observation(
    *, contract: Mapping[str, Any], request_id: str, prompt: str,
    camera_pngs: Mapping[str, bytes], robot_state: Mapping[str, Any],
    synthetic: bool = False,
) -> dict[str, Any]:
    """Construct only approved observations, with no simulator or asset metadata."""
    normalized = validate_company_policy_container_contract_v2(contract)
    if not isinstance(synthetic, bool):
        raise ValueError("controlled_policy_synthetic_flag_invalid")
    if not re.fullmatch(r"[a-f0-9]{32,64}", request_id):
        raise ValueError("controlled_policy_opaque_request_id_required")
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 4096:
        raise ValueError("controlled_policy_prompt_invalid")
    schema = normalized["observation_schema"]
    cameras: dict[str, Any] = {}
    for camera in schema["cameras"]:
        raw = camera_pngs.get(camera["name"])
        if not isinstance(raw, bytes) or len(raw) > MAX_WIRE_BYTES:
            raise ValueError("controlled_policy_camera_bytes_invalid")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.size != (camera["width"], camera["height"]):
                raise ValueError("controlled_policy_camera_shape_invalid")
            if image.mode != "RGB":
                raise ValueError("controlled_policy_camera_rgb_required")
            canonical = io.BytesIO()
            Image.frombytes("RGB", image.size, image.tobytes()).save(canonical, format="PNG")
        frame = canonical.getvalue()
        cameras[camera["name"]] = {
            "encoding": "lossless_png", "width": camera["width"], "height": camera["height"],
            "color_space": camera["color_space"],
            "sha256": "sha256:" + hashlib.sha256(frame).hexdigest(),
            "data_base64": base64.b64encode(frame).decode("ascii"),
        }
    state = {
        field["name"]: _state(robot_state.get(field["name"]), field["shape"])
        for field in schema["state_fields"]
    }
    wire = {
        "schema_version": "blueprint_company_policy_observation.v1",
        "request_id": request_id, "synthetic": synthetic,
        "prompt": prompt, "cameras": cameras, "state": state,
    }
    if len(json.dumps(wire, allow_nan=False).encode()) > MAX_WIRE_BYTES:
        raise ValueError("controlled_policy_observation_too_large")
    return wire


def call_customer_hosted_policy(
    *, endpoint: str, allowed_origins: tuple[str, ...], contract: Mapping[str, Any],
    request_id: str, prompt: str, camera_pngs: Mapping[str, bytes],
    robot_state: Mapping[str, Any], synthetic: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """HTTPS inference only. The simulator independently executes and scores actions.

    Origins are operator-approved, never sourced from a customer's request.
    Return the exact policy observation for the trusted lossless evidence writer.
    This function neither trusts a policy's success claim nor settles a run.
    """
    normalized = validate_company_policy_container_contract_v2(contract)
    wire = project_controlled_observation(
        contract=normalized, request_id=request_id, prompt=prompt,
        camera_pngs=camera_pngs, robot_state=robot_state, synthetic=synthetic,
    )
    encoded = json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    response = fetch_bounded_https(
        endpoint, method="POST", data=encoded,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout_seconds=min(30, normalized["container"]["resources"]["request_timeout_ms"] / 1000),
        max_bytes=65_536, allowed_origins=allowed_origins,
        allowed_content_types=("application/json",), max_redirects=0,
    )
    actions = validate_action_response(json.loads(response.body), action_schema=normalized["action_schema"])
    return actions, wire, {
        "schema_version": "blueprint.controlled_policy_call.v1",
        "request_id": request_id, "http_status": response.status,
        "observation_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "action_sha256": "sha256:" + hashlib.sha256(
            json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scene_files_exported": False, "scoring_harness_exported": False,
        "observation_access": "controlled_camera_frames_robot_state_and_instruction",
        "task_success_proven": False, "physical_success_proven": False,
    }
