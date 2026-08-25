"""Canonical live-candidate contract for company-supplied policy containers.

Version 1 proved the shape of the policy declaration, but its prototype
runtime used host networking, a writable root filesystem, caller-selected
files from Blueprint's shared secret directory, and raw Docker logs.  Those
properties are incompatible with showing untrusted code a private scene.

Version 2 is deliberately narrower and live-safe by construction:

* the OCI image is immutable and digest pinned;
* only the Blueprint HTTP/JSON wire protocol is admitted initially;
* robot, cameras, state, actions, units and timing are explicit;
* rights evidence is URI-and-digest bound;
* registry credentials are not part of this immutable artifact;
* the fixed security profile is injected, never caller-configurable.

The validator is pure.  It never pulls an image, reads a credential, opens a
socket, or grants launch authority.  A valid contract still requires separate
credential, sandbox qualification, synthetic conformance, launch profile and
paid-attempt authority receipts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "company_policy_container_contract.v2"
CLAIM_CEILING = "development_only"
LIVE_HANDSHAKE_KIND = "http_json_v1"
LIVE_PROTOCOL_VERSION = "1.0"
ACTION_ROUTE = "/v1/actions"

BLOCKER_INVALID = "company_policy_container_v2_invalid"
BLOCKER_IMAGE = "company_policy_container_v2_image_not_digest_pinned"
BLOCKER_SECURITY = "company_policy_container_v2_security_profile_invalid"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$")
_ROUTE = re.compile(r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]{1,255}$")

SECURITY_PROFILE: dict[str, Any] = {
    "profile_id": "blueprint_company_policy_sandbox_v2",
    "network_mode": "none_with_blueprint_proxy",
    "egress_policy": "deny_all_measured_before_first_observation",
    "dns_available": False,
    "ipv6_available": False,
    "host_network_available": False,
    "scene_mounts_allowed": False,
    "capture_mounts_allowed": False,
    "evidence_mounts_allowed": False,
    "output_mounts_allowed": False,
    "docker_socket_mounted": False,
    "registry_credential_mounted": False,
    "runtime_credential_mounts_allowed": False,
    "root_filesystem_read_only": True,
    "capabilities_dropped": "all",
    "no_new_privileges": True,
    "raw_logs_after_scene_access": "quarantined_not_customer_visible",
    "blueprint_owned_proxy_only": True,
    "container_and_image_removal_required": True,
}

_TOP = frozenset(
    {
        "schema_version",
        "policy_id",
        "company_id",
        "display_name",
        "checkpoint_identity",
        "claim_ceiling",
        "rights",
        "container",
        "robot",
        "observation_schema",
        "action_schema",
        "security_profile",
        "contract_digest",
    }
)
_CHECKPOINT = frozenset({"repository", "revision", "inventory_digest"})
_RIGHTS = frozenset(
    {
        "license",
        "rights_provenance",
        "rights_evidence_uri",
        "rights_evidence_digest",
        "provider_use_status",
        "redistribution_status",
        "rights_ready",
    }
)
_CONTAINER = frozenset(
    {
        "image",
        "visibility",
        "serve_command",
        "port",
        "handshake",
        "run_as_uid",
        "run_as_gid",
        "gpu_required",
        "resources",
    }
)
_HANDSHAKE = frozenset({"kind", "protocol_version", "action_route"})
_RESOURCES = frozenset(
    {
        "cpus",
        "memory_mib",
        "pids_limit",
        "tmpfs_mib",
        "startup_timeout_seconds",
        "request_timeout_ms",
    }
)
_ROBOT = frozenset(
    {
        "embodiment_id",
        "definition_uri",
        "definition_digest",
        "joint_names",
        "joint_limits",
        "gripper",
    }
)
_JOINT_LIMIT = frozenset({"name", "lower", "upper", "unit"})
_GRIPPER = frozenset({"name", "command_interval", "unit", "executed_semantics"})
_OBSERVATION = frozenset({"cameras", "state_fields", "prompt", "control_frequency_hz"})
_CAMERA = frozenset(
    {
        "name",
        "width",
        "height",
        "color_space",
        "dtype",
        "layout",
        "encoding",
        "calibration_uri",
        "calibration_digest",
    }
)
_STATE = frozenset({"name", "shape", "dtype", "unit"})
_PROMPT = frozenset({"mode", "required"})
_ACTION = frozenset({"adapter_id", "chunk_rows", "channels", "normalization"})
_CHANNEL = frozenset(
    {
        "name",
        "kind",
        "command_interval",
        "raw_accepted_bounds",
        "unit",
        "executed_semantics",
    }
)
_NORMALIZATION = frozenset({"observation", "action", "gripper"})


class CompanyPolicyContainerContractV2Error(ValueError):
    """Fail-closed error carrying stable, sorted blocker identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _unknown(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> list[str]:
    return [
        f"{BLOCKER_INVALID}:unknown_field:{path}{key}"
        for key in sorted(str(key) for key in value)
        if key not in allowed
    ]


def _mapping(
    payload: Mapping[str, Any], key: str, allowed: frozenset[str], errors: list[str]
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{BLOCKER_INVALID}:{key}")
        return {}
    errors.extend(_unknown(value, allowed, f"{key}."))
    return dict(value)


def _digest(value: Any, *, field: str, errors: list[str]) -> str:
    text = _text(value)
    if not _DIGEST.fullmatch(text):
        errors.append(f"{BLOCKER_INVALID}:{field}")
    return text


def _interval(value: Any, *, field: str, errors: list[str]) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        errors.append(f"{BLOCKER_INVALID}:{field}")
        return []
    lower, upper = _number(value[0]), _number(value[1])
    if lower is None or upper is None or lower >= upper:
        errors.append(f"{BLOCKER_INVALID}:{field}")
        return []
    return [lower, upper]


def validate_company_policy_container_contract_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate, normalize and digest one company policy declaration."""

    if not isinstance(value, Mapping):
        raise CompanyPolicyContainerContractV2Error([f"{BLOCKER_INVALID}:not_mapping"])
    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CompanyPolicyContainerContractV2Error([f"{BLOCKER_INVALID}:not_json"]) from exc
    errors: list[str] = []
    errors.extend(_unknown(payload, _TOP, ""))
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{BLOCKER_INVALID}:schema_version")

    policy_id = _text(payload.get("policy_id"))
    company_id = _text(payload.get("company_id"))
    if not _IDENTIFIER.fullmatch(policy_id):
        errors.append(f"{BLOCKER_INVALID}:policy_id")
    if not _IDENTIFIER.fullmatch(company_id):
        errors.append(f"{BLOCKER_INVALID}:company_id")
    display_name = _text(payload.get("display_name"))
    if not display_name:
        errors.append(f"{BLOCKER_INVALID}:display_name")
    if payload.get("claim_ceiling") != CLAIM_CEILING:
        errors.append(f"{BLOCKER_INVALID}:claim_ceiling")

    checkpoint = _mapping(payload, "checkpoint_identity", _CHECKPOINT, errors)
    checkpoint_normalized = {
        "repository": _text(checkpoint.get("repository")),
        "revision": _text(checkpoint.get("revision")),
        "inventory_digest": _digest(
            checkpoint.get("inventory_digest"),
            field="checkpoint_inventory_digest",
            errors=errors,
        ),
    }
    if not checkpoint_normalized["repository"]:
        errors.append(f"{BLOCKER_INVALID}:checkpoint_repository")
    if not checkpoint_normalized["revision"]:
        errors.append(f"{BLOCKER_INVALID}:checkpoint_revision")

    rights = _mapping(payload, "rights", _RIGHTS, errors)
    rights_normalized: dict[str, Any] = {}
    for field in (
        "license",
        "rights_provenance",
        "rights_evidence_uri",
        "provider_use_status",
        "redistribution_status",
    ):
        rights_normalized[field] = _text(rights.get(field))
        if not rights_normalized[field]:
            errors.append(f"{BLOCKER_INVALID}:rights_{field}")
    rights_normalized["rights_evidence_digest"] = _digest(
        rights.get("rights_evidence_digest"),
        field="rights_evidence_digest",
        errors=errors,
    )
    rights_normalized["rights_ready"] = rights.get("rights_ready") is True
    if not rights_normalized["rights_ready"]:
        errors.append(f"{BLOCKER_INVALID}:rights_ready")

    container = _mapping(payload, "container", _CONTAINER, errors)
    image = _text(container.get("image"))
    if not _IMAGE.fullmatch(image):
        errors.append(f"{BLOCKER_IMAGE}:{image or 'missing'}")
    visibility = _text(container.get("visibility"))
    if visibility not in {"public", "private"}:
        errors.append(f"{BLOCKER_INVALID}:container_visibility")
    command = container.get("serve_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or not arg for arg in command)
    ):
        errors.append(f"{BLOCKER_INVALID}:container_serve_command")
        command = []
    port = _positive_int(container.get("port"))
    if port is None or not 1024 <= port <= 65535:
        errors.append(f"{BLOCKER_INVALID}:container_port")
    uid = _positive_int(container.get("run_as_uid"))
    gid = _positive_int(container.get("run_as_gid"))
    if uid is None or uid > 2**31 - 1:
        errors.append(f"{BLOCKER_INVALID}:container_run_as_uid")
    if gid is None or gid > 2**31 - 1:
        errors.append(f"{BLOCKER_INVALID}:container_run_as_gid")
    gpu_required = container.get("gpu_required")
    if not isinstance(gpu_required, bool):
        errors.append(f"{BLOCKER_INVALID}:container_gpu_required")

    handshake = _mapping(container, "handshake", _HANDSHAKE, errors)
    handshake_normalized = {
        "kind": _text(handshake.get("kind")),
        "protocol_version": _text(handshake.get("protocol_version")),
        "action_route": _text(handshake.get("action_route")),
    }
    if handshake_normalized["kind"] != LIVE_HANDSHAKE_KIND:
        errors.append(f"{BLOCKER_INVALID}:handshake_kind")
    if handshake_normalized["protocol_version"] != LIVE_PROTOCOL_VERSION:
        errors.append(f"{BLOCKER_INVALID}:handshake_protocol_version")
    if (
        not _ROUTE.fullmatch(handshake_normalized["action_route"])
        or handshake_normalized["action_route"] != ACTION_ROUTE
    ):
        errors.append(f"{BLOCKER_INVALID}:handshake_action_route")

    resources = _mapping(container, "resources", _RESOURCES, errors)
    resource_bounds = {
        "cpus": (0.1, 64.0),
        "memory_mib": (256, 262_144),
        "pids_limit": (16, 4096),
        "tmpfs_mib": (16, 65_536),
        "startup_timeout_seconds": (1, 900),
        "request_timeout_ms": (1, 120_000),
    }
    resources_normalized: dict[str, Any] = {}
    for field, (minimum, maximum) in resource_bounds.items():
        raw = _number(resources.get(field))
        integer = field != "cpus"
        if raw is None or raw < minimum or raw > maximum or (integer and raw != int(raw)):
            errors.append(f"{BLOCKER_INVALID}:resource_{field}")
            continue
        resources_normalized[field] = int(raw) if integer else raw

    robot = _mapping(payload, "robot", _ROBOT, errors)
    embodiment_id = _text(robot.get("embodiment_id"))
    if not _IDENTIFIER.fullmatch(embodiment_id):
        errors.append(f"{BLOCKER_INVALID}:robot_embodiment_id")
    definition_uri = _text(robot.get("definition_uri"))
    if not definition_uri:
        errors.append(f"{BLOCKER_INVALID}:robot_definition_uri")
    definition_digest = _digest(
        robot.get("definition_digest"), field="robot_definition_digest", errors=errors
    )
    joint_names = robot.get("joint_names")
    if (
        not isinstance(joint_names, list)
        or not joint_names
        or any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in joint_names)
        or len(set(joint_names or [])) != len(joint_names or [])
    ):
        errors.append(f"{BLOCKER_INVALID}:robot_joint_names")
        joint_names = []
    limits = robot.get("joint_limits")
    normalized_limits: list[dict[str, Any]] = []
    if not isinstance(limits, list) or len(limits) != len(joint_names):
        errors.append(f"{BLOCKER_INVALID}:robot_joint_limits")
    else:
        for index, limit in enumerate(limits):
            if not isinstance(limit, Mapping):
                errors.append(f"{BLOCKER_INVALID}:robot_joint_limit:{index}")
                continue
            errors.extend(_unknown(limit, _JOINT_LIMIT, f"robot.joint_limits[{index}]."))
            name = _text(limit.get("name"))
            lower, upper = _number(limit.get("lower")), _number(limit.get("upper"))
            unit = _text(limit.get("unit"))
            if (
                index >= len(joint_names)
                or name != joint_names[index]
                or lower is None
                or upper is None
                or lower >= upper
                or not unit
            ):
                errors.append(f"{BLOCKER_INVALID}:robot_joint_limit:{index}")
                continue
            normalized_limits.append({"name": name, "lower": lower, "upper": upper, "unit": unit})
    gripper = _mapping(robot, "gripper", _GRIPPER, errors)
    gripper_normalized = {
        "name": _text(gripper.get("name")),
        "command_interval": _interval(
            gripper.get("command_interval"), field="robot_gripper_interval", errors=errors
        ),
        "unit": _text(gripper.get("unit")),
        "executed_semantics": _text(gripper.get("executed_semantics")),
    }
    if not _IDENTIFIER.fullmatch(gripper_normalized["name"]):
        errors.append(f"{BLOCKER_INVALID}:robot_gripper_name")
    if not gripper_normalized["unit"]:
        errors.append(f"{BLOCKER_INVALID}:robot_gripper_unit")
    if not gripper_normalized["executed_semantics"]:
        errors.append(f"{BLOCKER_INVALID}:robot_gripper_semantics")

    observation = _mapping(payload, "observation_schema", _OBSERVATION, errors)
    cameras = observation.get("cameras")
    normalized_cameras: list[dict[str, Any]] = []
    if not isinstance(cameras, list) or not cameras:
        errors.append(f"{BLOCKER_INVALID}:observation_cameras")
    else:
        seen: set[str] = set()
        for index, camera in enumerate(cameras):
            if not isinstance(camera, Mapping):
                errors.append(f"{BLOCKER_INVALID}:observation_camera:{index}")
                continue
            errors.extend(_unknown(camera, _CAMERA, f"observation_schema.cameras[{index}]."))
            name = _text(camera.get("name"))
            width = _positive_int(camera.get("width"))
            height = _positive_int(camera.get("height"))
            normalized_camera = {
                "name": name,
                "width": width,
                "height": height,
                "color_space": _text(camera.get("color_space")),
                "dtype": _text(camera.get("dtype")),
                "layout": _text(camera.get("layout")),
                "encoding": _text(camera.get("encoding")),
                "calibration_uri": _text(camera.get("calibration_uri")),
                "calibration_digest": _digest(
                    camera.get("calibration_digest"),
                    field=f"observation_camera_calibration_digest:{index}",
                    errors=errors,
                ),
            }
            if not _IDENTIFIER.fullmatch(name) or name in seen:
                errors.append(f"{BLOCKER_INVALID}:observation_camera_name:{index}")
            seen.add(name)
            if width is None or height is None or width > 8192 or height > 8192:
                errors.append(f"{BLOCKER_INVALID}:observation_camera_dimensions:{index}")
            if (
                normalized_camera["color_space"] != "rgb"
                or normalized_camera["dtype"] != "uint8"
                or normalized_camera["layout"] != "hwc"
                or normalized_camera["encoding"] != "lossless_png"
                or not normalized_camera["calibration_uri"]
            ):
                errors.append(f"{BLOCKER_INVALID}:observation_camera_format:{index}")
            normalized_cameras.append(normalized_camera)
    state_fields = observation.get("state_fields")
    normalized_state: list[dict[str, Any]] = []
    if not isinstance(state_fields, list) or not state_fields:
        errors.append(f"{BLOCKER_INVALID}:observation_state_fields")
    else:
        state_names: set[str] = set()
        for index, field in enumerate(state_fields):
            if not isinstance(field, Mapping):
                errors.append(f"{BLOCKER_INVALID}:observation_state_field:{index}")
                continue
            errors.extend(_unknown(field, _STATE, f"observation_schema.state_fields[{index}]."))
            name = _text(field.get("name"))
            shape = field.get("shape")
            dtype = _text(field.get("dtype"))
            unit = _text(field.get("unit"))
            if (
                not _IDENTIFIER.fullmatch(name)
                or name in state_names
                or not isinstance(shape, list)
                or not shape
                or any(_positive_int(item) is None for item in shape)
                or dtype not in {"float32", "float64"}
                or not unit
            ):
                errors.append(f"{BLOCKER_INVALID}:observation_state_field:{index}")
                continue
            state_names.add(name)
            normalized_state.append(
                {"name": name, "shape": list(shape), "dtype": dtype, "unit": unit}
            )
    prompt = _mapping(observation, "prompt", _PROMPT, errors)
    prompt_normalized = {
        "mode": _text(prompt.get("mode")),
        "required": prompt.get("required") is True,
    }
    if prompt_normalized != {"mode": "text", "required": True}:
        errors.append(f"{BLOCKER_INVALID}:observation_prompt")
    frequency = _number(observation.get("control_frequency_hz"))
    if frequency is None or not 0.1 <= frequency <= 240.0:
        errors.append(f"{BLOCKER_INVALID}:observation_control_frequency_hz")

    action = _mapping(payload, "action_schema", _ACTION, errors)
    adapter_id = _text(action.get("adapter_id"))
    if not _IDENTIFIER.fullmatch(adapter_id):
        errors.append(f"{BLOCKER_INVALID}:action_adapter_id")
    chunk_rows = _positive_int(action.get("chunk_rows"))
    if chunk_rows is None or chunk_rows > 1024:
        errors.append(f"{BLOCKER_INVALID}:action_chunk_rows")
    channels = action.get("channels")
    normalized_channels: list[dict[str, Any]] = []
    if not isinstance(channels, list) or not channels:
        errors.append(f"{BLOCKER_INVALID}:action_channels")
    else:
        channel_names: set[str] = set()
        for index, channel in enumerate(channels):
            if not isinstance(channel, Mapping):
                errors.append(f"{BLOCKER_INVALID}:action_channel:{index}")
                continue
            errors.extend(_unknown(channel, _CHANNEL, f"action_schema.channels[{index}]."))
            name = _text(channel.get("name"))
            kind = _text(channel.get("kind"))
            command_interval = _interval(
                channel.get("command_interval"),
                field=f"action_channel_command_interval:{index}",
                errors=errors,
            )
            raw_bounds = _interval(
                channel.get("raw_accepted_bounds"),
                field=f"action_channel_raw_bounds:{index}",
                errors=errors,
            )
            unit = _text(channel.get("unit"))
            semantics = _text(channel.get("executed_semantics"))
            if (
                not _IDENTIFIER.fullmatch(name)
                or name in channel_names
                or kind not in {"bounded_continuous", "threshold_scalar"}
                or not unit
                or not semantics
            ):
                errors.append(f"{BLOCKER_INVALID}:action_channel:{index}")
            channel_names.add(name)
            if (
                command_interval
                and raw_bounds
                and (raw_bounds[0] > command_interval[0] or raw_bounds[1] < command_interval[1])
            ):
                errors.append(f"{BLOCKER_INVALID}:action_channel_raw_narrower:{index}")
            normalized_channels.append(
                {
                    "name": name,
                    "kind": kind,
                    "command_interval": command_interval,
                    "raw_accepted_bounds": raw_bounds,
                    "unit": unit,
                    "executed_semantics": semantics,
                }
            )
    normalization = _mapping(action, "normalization", _NORMALIZATION, errors)
    normalization_normalized = {
        key: _text(normalization.get(key)) for key in ("observation", "action", "gripper")
    }
    if any(not value for value in normalization_normalized.values()):
        errors.append(f"{BLOCKER_INVALID}:action_normalization")

    supplied_security = payload.get("security_profile")
    if supplied_security is not None and supplied_security != SECURITY_PROFILE:
        errors.append(BLOCKER_SECURITY)

    if errors:
        raise CompanyPolicyContainerContractV2Error(errors)
    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": policy_id,
        "company_id": company_id,
        "display_name": display_name,
        "checkpoint_identity": checkpoint_normalized,
        "claim_ceiling": CLAIM_CEILING,
        "rights": rights_normalized,
        "container": {
            "image": image,
            "visibility": visibility,
            "serve_command": list(command),
            "port": port,
            "handshake": handshake_normalized,
            "run_as_uid": uid,
            "run_as_gid": gid,
            "gpu_required": gpu_required,
            "resources": resources_normalized,
        },
        "robot": {
            "embodiment_id": embodiment_id,
            "definition_uri": definition_uri,
            "definition_digest": definition_digest,
            "joint_names": list(joint_names),
            "joint_limits": normalized_limits,
            "gripper": gripper_normalized,
        },
        "observation_schema": {
            "cameras": normalized_cameras,
            "state_fields": normalized_state,
            "prompt": prompt_normalized,
            "control_frequency_hz": frequency,
        },
        "action_schema": {
            "adapter_id": adapter_id,
            "chunk_rows": chunk_rows,
            "channels": normalized_channels,
            "normalization": normalization_normalized,
        },
        "security_profile": dict(SECURITY_PROFILE),
    }
    digest = canonical_digest(normalized, digest_field="contract_digest")
    if "contract_digest" in payload and payload.get("contract_digest") != digest:
        raise CompanyPolicyContainerContractV2Error([f"{BLOCKER_INVALID}:contract_digest_mismatch"])
    normalized["contract_digest"] = digest
    return normalized


__all__ = [
    "ACTION_ROUTE",
    "BLOCKER_IMAGE",
    "BLOCKER_INVALID",
    "BLOCKER_SECURITY",
    "CLAIM_CEILING",
    "CompanyPolicyContainerContractV2Error",
    "LIVE_HANDSHAKE_KIND",
    "LIVE_PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "SECURITY_PROFILE",
    "validate_company_policy_container_contract_v2",
]
