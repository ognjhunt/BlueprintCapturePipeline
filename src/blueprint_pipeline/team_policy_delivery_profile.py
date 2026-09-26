"""Verify an owner-stored policy delivery against a trusted task setup.

Registration describes bytes and an interface. It does not qualify an
endpoint, grant model or scene rights, authorize GPU spend, or start a policy.
The same profile contract covers HTTPS, OCI, and noncontainer artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import cross_runtime_canonical_digest


SCHEMA = "team_policy_delivery_profile.v1"
PROTOCOL = "jsonl_observation_action_v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SECRET = re.compile(r"^secretref:[A-Za-z0-9][A-Za-z0-9._:/-]{0,190}$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,400}@sha256:[0-9a-f]{64}$")
_ENTRYPOINT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,190}$")
_FIELDS = frozenset(
    {
        "schema_version",
        "owner",
        "label",
        "source_setup_digest",
        "source_scene_id",
        "source_task_id",
        "robot_preset_id",
        "embodiment_id",
        "observation_schema_id",
        "action_schema_id",
        "delivery",
        "status",
        "claim_ceiling",
        "provider_mutation_performed",
        "public_redistribution_authorized",
        "profile_digest",
    }
)


def _https_url(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 2048:
        return False
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        return False
    host = (url.hostname or "").lower().rstrip(".")
    if (
        url.scheme != "https"
        or not host
        or "." not in host
        or url.username
        or url.password
        or url.query
        or url.fragment
        or port not in {None, 443}
        or host == "localhost"
        or host.endswith((".localhost", ".local", ".internal"))
    ):
        return False
    # IP literals are never a team-supplied public service identity.
    from ipaddress import ip_address

    try:
        ip_address(host)
    except ValueError:
        return True
    return False


def _delivery(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    mode = value.get("mode")
    if mode == "authenticated_endpoint":
        timeout = value.get("timeout_ms")
        return (
            set(value) == {"mode", "endpoint_url", "auth_secret_ref", "timeout_ms"}
            and _https_url(value.get("endpoint_url"))
            and isinstance(value.get("auth_secret_ref"), str)
            and _SECRET.fullmatch(value["auth_secret_ref"]) is not None
            and type(timeout) is int
            and 100 <= timeout <= 30_000
        )
    if mode == "container":
        return (
            set(value) == {"mode", "image_ref", "protocol"}
            and isinstance(value.get("image_ref"), str)
            and _IMAGE.fullmatch(value["image_ref"]) is not None
            and value.get("protocol") == PROTOCOL
        )
    if mode == "noncontainer_artifact":
        entrypoint = value.get("entrypoint")
        return (
            set(value) == {"mode", "artifact_uri", "artifact_sha256", "entrypoint", "protocol"}
            and _https_url(value.get("artifact_uri"))
            and isinstance(value.get("artifact_sha256"), str)
            and _DIGEST.fullmatch(value["artifact_sha256"]) is not None
            and isinstance(entrypoint, str)
            and _ENTRYPOINT.fullmatch(entrypoint) is not None
            and ".." not in entrypoint.split("/")
            and not entrypoint.startswith("/")
            and value.get("protocol") == PROTOCOL
        )
    return False


def validate_team_policy_delivery_profile(
    value: Mapping[str, Any] | Any,
    *,
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the WebApp digest and exact owner/task/robot interface binding."""

    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("team_policy_delivery_profile_shape_invalid")
    profile = dict(value)
    setup_digest = trusted_setup.get("setup_digest")
    if (
        not isinstance(setup_digest, str)
        or _DIGEST.fullmatch(setup_digest) is None
        or setup_digest
        != cross_runtime_canonical_digest(trusted_setup, digest_field="setup_digest")
        or not isinstance(trusted_setup.get("scene_id"), str)
        or not trusted_setup["scene_id"]
        or not isinstance(trusted_setup.get("task_id"), str)
        or not trusted_setup["task_id"]
    ):
        raise ValueError("team_policy_delivery_setup_invalid")
    robots = trusted_setup.get("robot_presets")
    if not isinstance(robots, list):
        raise ValueError("team_policy_delivery_setup_invalid")
    matches = [
        robot
        for robot in robots
        if isinstance(robot, Mapping)
        and robot.get("robot_preset_id") == profile.get("robot_preset_id")
    ]
    if len(matches) != 1:
        raise ValueError("team_policy_delivery_robot_binding_invalid")
    robot = matches[0]
    observation = robot.get("observation_schema")
    action = robot.get("action_schema")
    owner = profile.get("owner")
    label = profile.get("label")
    if (
        profile.get("schema_version") != SCHEMA
        or not isinstance(owner, dict)
        or set(owner) != {"user_id", "organization_id"}
        or owner != dict(authenticated_owner)
        or any(not isinstance(item, str) or not item for item in owner.values())
        or not isinstance(label, str)
        or not 1 <= len(label.strip()) <= 120
        or profile.get("source_setup_digest") != trusted_setup.get("setup_digest")
        or profile.get("source_scene_id") != trusted_setup.get("scene_id")
        or profile.get("source_task_id") != trusted_setup.get("task_id")
        or profile.get("embodiment_id") != robot.get("embodiment_id")
        or not isinstance(observation, Mapping)
        or profile.get("observation_schema_id") != observation.get("schema_id")
        or not isinstance(action, Mapping)
        or profile.get("action_schema_id") != action.get("schema_id")
        or not isinstance(profile.get("robot_preset_id"), str)
        or _IDENTIFIER.fullmatch(profile["robot_preset_id"]) is None
        or profile.get("status") != "registered_for_runtime_review"
        or profile.get("claim_ceiling") != "planning_only"
        or profile.get("provider_mutation_performed") is not False
        or profile.get("public_redistribution_authorized") is not False
        or not _delivery(profile.get("delivery"))
        or not isinstance(profile.get("profile_digest"), str)
        or _DIGEST.fullmatch(profile["profile_digest"]) is None
        or profile["profile_digest"]
        != cross_runtime_canonical_digest(profile, digest_field="profile_digest")
    ):
        raise ValueError("team_policy_delivery_profile_binding_invalid")
    return profile
