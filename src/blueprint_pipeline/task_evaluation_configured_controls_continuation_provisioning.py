"""Provision the configured-controls continuation for one scene at one commit.

For 839873 the continuation inputs were nineteen per-commit directories of
hand-written files: a runtime bundle rebound by editing a manifest, a camera
template lifted from an older packet, a trajectory plan lifted from an older
diagnostic, and per-phase authority files typed by hand.  Every one of them is
derivable from the owner's task request, the scene's preparation result, the
deployed commit, and the retained runtime payload.  This module derives them,
publishes what the activation worker must read back, seals the autostart
intent with the first-run-only inputs deferred, and installs it into the
registry the activation and progression workers consult.

It allocates nothing and never touches a provider.  Provider zero is observed
through the production seam for the construction lineage; everything else is
CPU-only file authoring with full-byte publication readback.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .task_evaluation_configured_controls_autostart import (
    DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD,
    TaskEvaluationConfiguredControlsAutostartError,
    configured_controls_autostart_registry_name,
    materialize_configured_controls_autostart_intent,
    validate_configured_controls_autostart_intent,
)
from .task_evaluation_configured_controls_deferred_inputs import (
    OVERVIEW_MODE,
    SCENE_BUNDLE_MODE,
    TRAJECTORY_MODE,
)
from .task_evaluation_native_arena_preparation_adapter import (
    TaskEvaluationNativeArenaAdapterError,
    build_task_evaluation_runtime_source_bundle,
)
from .vast_evidence_contracts import valid_vast_provider_zero_api_call


RUNTIME_IDENTITY = {"id": "native-arena", "version": "isaac-2026-1"}
RUNTIME_ENTRYPOINT = ["/opt/blueprint/run-task-evaluation"]
RUNTIME_REQUIREMENTS = {"cpu_cores": 8, "memory_gib": 64, "gpu_count": 1, "disk_gib": 100}
RUNTIME_OUTPUT_LIMIT_BYTES = 20_000_000_000
HEALTH_PROTOCOL_SCHEMA_VERSION = "task_evaluation_native_arena_health_protocol.v1"
REQUIRED_HEALTH_MARKERS = [
    "native_task_arena_runtime_preflight",
    "native_task_arena_construction_result",
    "provider_teardown_manifest",
    "post_teardown_provider_zero_receipt",
]
RELEASE_WINDOW_TEMPLATE_SCHEMA_VERSION = (
    "task_evaluation_configured_controls_release_window_template.v1"
)
CAMERA_TEMPLATE_SCHEMA_VERSION = "native_task_arena_packet_request.v1"
WRIST_PARENT_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
RIGHTS_SCOPE = "internal_noncommercial_research_and_development_Task_Evaluation_Run"
PHASES = ("construction", "controls")
DEFAULT_PHASE_HARD_CAP_USD = 2.0
DEFAULT_PHASE_TTL_SECONDS = 9_000
DEFAULT_HOURLY_RATE_USD = 0.8
DEFAULT_AUTHORITY_VALID_SECONDS = 7 * 24 * 3600
RELEASE_WINDOW_VALID_SECONDS = 3_600
DEFAULT_POLICY_CAMERA_RESOLUTION = (640, 360)
DEFAULT_OVERVIEW_CAMERA_RESOLUTION = (1280, 720)
DEFAULT_EXTERNAL_LAYER_MIN_BYTES = 64 * 1024 * 1024
DEFAULT_INTENT_ROOT = "/etc/blueprint/task-evaluation-configured-controls-intents"
DEFAULT_PROFILE_DIR = "/etc/blueprint/task-evaluation-launch-profiles"
DEFAULT_SERVICE_GROUP = "blueprint"
PREPARATION_CONTRACT_PATHS = {
    "task_definition": "task.definition",
    "robot_mount_interface": "scene.registration.robot_mount_interface",
    "camera_calibration": "scene.registration.camera_calibration",
    "rights_admission": "scene.rights.admission",
}
ARTIFACT_KINDS = {
    "runtime_source": "native-runtime-source",
    "health_protocol": "native-health-protocol",
    "provider_zero": "initial-provider-zero",
    "project_spend": "prior-official-spend",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

ArtifactPublisher = Callable[..., Mapping[str, Any]]
LayerPublisher = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ProviderZeroCollector = Callable[[], Mapping[str, Any]]


class ConfiguredControlsProvisioningError(RuntimeError):
    """A continuation input could not be derived from exact, admitted sources."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise ConfiguredControlsProvisioningError(blocker)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredControlsProvisioningError(blocker) from exc
    if not isinstance(value, Mapping):
        raise ConfiguredControlsProvisioningError(blocker)
    return dict(value)


def _payload(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``value`` immutably, or return the identical document already retained.

    Time-stamped authority documents are authored exactly once per controls
    root: a re-run reuses the retained bytes instead of minting a new stamp.
    """

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.exists() or path.is_symlink():
        return _load(path, blocker=f"configured_controls_provisioning_retained_invalid:{path.name}")
    with path.open("xb") as stream:
        stream.write(_payload(value))
    path.chmod(0o440)
    return dict(value)


def _reference(value: Any, *, blocker: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not str(value.get("uri") or "").strip()
        or _DIGEST.fullmatch(str(value.get("digest") or "")) is None
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] < 1
    ):
        raise ConfiguredControlsProvisioningError(blocker)
    return {
        "uri": str(value["uri"]),
        "digest": str(value["digest"]),
        "size_bytes": int(value["size_bytes"]),
    }


def _publish(
    *, path: Path, kind: str, publisher: ArtifactPublisher, reference_path: Path
) -> dict[str, Any]:
    """Publish one file with readback proof; reuse the retained reference when identical."""

    expected_digest = _sha256(path)
    expected_size = path.stat().st_size
    if reference_path.is_file() and not reference_path.is_symlink():
        retained = _load(
            reference_path, blocker=f"configured_controls_provisioning_retained_invalid:{kind}"
        )
        if retained.get("digest") == expected_digest and retained.get("size_bytes") == expected_size:
            return _reference(retained, blocker=f"configured_controls_provisioning_reference_invalid:{kind}")
        raise ConfiguredControlsProvisioningError(
            f"configured_controls_provisioning_retained_invalid:{kind}"
        )
    observed = dict(publisher(path=path, artifact_kind=kind))
    if (
        observed.get("status") != "remote_verified"
        or observed.get("remote_identity_verified") is not True
        or observed.get("full_byte_service_account_readback_passed") is not True
        or observed.get("digest") != expected_digest
        or observed.get("size_bytes") != expected_size
    ):
        raise ConfiguredControlsProvisioningError(
            f"configured_controls_provisioning_publication_readback_invalid:{kind}"
        )
    reference = _reference(
        observed, blocker=f"configured_controls_provisioning_reference_invalid:{kind}"
    )
    _write_once(reference_path, {**observed, "artifact_kind": kind})
    return reference


# ------------------------------------------------------------------ preparation


def _preparation_context(
    *, preparation_result_path: Path, preparation_queue_root: Path, expected_production_commit: str
) -> dict[str, Any]:
    result = _load(
        preparation_result_path,
        blocker="configured_controls_provisioning_preparation_result_invalid",
    )
    if (
        result.get("schema_version") != "task_evaluation_launch_preparation_result.v1"
        or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_preparation_result_invalid"
        )
    if result.get("run_mode") != "scene_configuration":
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_preparation_not_scene_configuration"
        )
    if result.get("source_commit") != expected_production_commit:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_preparation_commit_mismatch"
        )
    envelope = _load(
        preparation_queue_root / "materialized" / preparation_result_path.name,
        blocker="configured_controls_provisioning_preparation_envelope_invalid",
    )
    request = envelope.get("request")
    if (
        not isinstance(request, Mapping)
        or envelope.get("envelope_digest")
        != canonical_digest(envelope, digest_field="envelope_digest")
        or request.get("preparation_id") != result.get("preparation_id")
        or request.get("expected_production_commit") != expected_production_commit
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_preparation_envelope_invalid"
        )
    rows = {
        str(row.get("contract_path")): dict(row)
        for row in result.get("references") or []
        if isinstance(row, Mapping)
    }
    documents: dict[str, dict[str, Any]] = {}
    for name, contract_path in PREPARATION_CONTRACT_PATHS.items():
        row = rows.get(contract_path)
        path = Path(str((row or {}).get("materialized_path") or "")).expanduser()
        if (
            row is None
            or path.is_symlink()
            or not path.is_file()
            or _sha256(path) != row.get("digest")
            or path.stat().st_size != row.get("size_bytes")
            or row.get("full_byte_service_account_readback_passed") is not True
        ):
            raise ConfiguredControlsProvisioningError(
                f"configured_controls_provisioning_reference_missing:{contract_path}"
            )
        documents[name] = {
            "path": path,
            "reference": _reference(
                row, blocker=f"configured_controls_provisioning_reference_invalid:{contract_path}"
            ),
        }
    template = _load(
        documents["task_definition"]["path"],
        blocker="configured_controls_provisioning_task_template_invalid",
    )
    target = template.get("target_center_xyz_m")
    if (
        template.get("schema_version") != "task_evaluation_rigid_relocation_template.v1"
        or not isinstance(target, list)
        or len(target) != 3
        or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in target)
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_task_template_invalid"
        )
    return {
        "team_namespace": str(request["team_namespace"]),
        "scene_id": str(request["scene"]["identity"]["id"]),
        "task_id": str(request["task"]["identity"]["id"]),
        "target_position_world_m": [float(item) for item in target],
        "documents": documents,
    }


# ------------------------------------------------------------------ cameras


def _scaled_intrinsics(base: Mapping[str, Any], *, resolution: tuple[int, int]) -> dict[str, Any]:
    width, height = int(resolution[0]), int(resolution[1])
    try:
        base_width = int(base["width"])
        base_height = int(base["height"])
        fx = float(base["fx"])
        fy = float(base["fy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_embodiment_intrinsics_invalid"
        ) from exc
    if width < 16 or height < 16 or base_width < 1 or base_height < 1:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_camera_resolution_invalid"
        )
    if not math.isclose(width / height, base_width / base_height, rel_tol=1e-6):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_camera_aspect_mismatch"
        )
    scale = width / base_width
    return {
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "fx": fx * scale,
        "fy": fy * scale,
        "height": height,
        "width": width,
    }


def author_camera_template(
    *,
    embodiment_template: Mapping[str, Any],
    source_commit: str,
    policy_camera_resolution: tuple[int, int] = DEFAULT_POLICY_CAMERA_RESOLUTION,
    overview_camera_resolution: tuple[int, int] = DEFAULT_OVERVIEW_CAMERA_RESOLUTION,
) -> dict[str, Any]:
    """Re-author the embodiment's DROID cameras at the requested resolutions.

    The wrist mount pose is embodiment truth and is copied verbatim.  World
    camera poses are non-authoritative here: the autostart recomputes them from
    the selected base pose and the derived trajectory.
    """

    rows = embodiment_template.get("cameras")
    by_role = {
        str(row.get("role") or ""): dict(row) for row in rows or [] if isinstance(row, Mapping)
    }
    wrist = by_role.get("wrist") or {}
    if (
        not isinstance(rows, list)
        or set(by_role) != {"external", "wrist", "overview"}
        or wrist.get("pose_frame") != "robot_body"
        or wrist.get("parent_prim_path") != WRIST_PARENT_PRIM_PATH
        or wrist.get("optical_convention") != "opencv"
        or not isinstance(wrist.get("frame_from_camera_matrix"), list)
        or len(wrist["frame_from_camera_matrix"]) != 16
        or not isinstance(by_role["external"].get("intrinsics"), Mapping)
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_embodiment_camera_template_invalid"
        )
    policy_intrinsics = _scaled_intrinsics(
        by_role["external"]["intrinsics"], resolution=policy_camera_resolution
    )
    overview_intrinsics = _scaled_intrinsics(
        by_role["external"]["intrinsics"], resolution=overview_camera_resolution
    )

    def world_camera(role: str, *, policy_input: bool, intrinsics: Mapping[str, Any]) -> dict[str, Any]:
        matrix = by_role[role].get("frame_from_camera_matrix")
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_embodiment_camera_template_invalid"
            )
        return {
            "role": role,
            "policy_input": policy_input,
            "scoring_input": False,
            "pose_frame": "world",
            "parent_prim_path": "{ENV_REGEX_NS}",
            "optical_convention": "opencv",
            "frame_from_camera_matrix": [float(item) for item in matrix],
            "intrinsics": dict(intrinsics),
        }

    cameras = [
        world_camera("external", policy_input=True, intrinsics=policy_intrinsics),
        {
            "role": "wrist",
            "policy_input": True,
            "scoring_input": False,
            "pose_frame": "robot_body",
            "parent_prim_path": WRIST_PARENT_PRIM_PATH,
            "optical_convention": "opencv",
            "frame_from_camera_matrix": [float(item) for item in wrist["frame_from_camera_matrix"]],
            "intrinsics": dict(policy_intrinsics),
        },
        world_camera("overview", policy_input=False, intrinsics=overview_intrinsics),
    ]
    return {
        "schema_version": CAMERA_TEMPLATE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "status": "immutable_intrinsics_and_wrist_mount_template",
        "world_camera_poses_authoritative": False,
        "world_camera_poses_recomputed_after_cpu_placement": True,
        "native_observability_readback_required": True,
        "policy_camera_resolution": [int(policy_camera_resolution[0]), int(policy_camera_resolution[1])],
        "overview_camera_resolution": [
            int(overview_camera_resolution[0]),
            int(overview_camera_resolution[1]),
        ],
        "cameras": cameras,
    }


# ------------------------------------------------------------------ runtime source


def _runtime_source_reference(
    *,
    payload_dir: Path,
    controls_root: Path,
    source_commit: str,
    artifact_publisher: ArtifactPublisher,
    layer_publisher: LayerPublisher,
    external_layer_bucket: str | None,
    external_layer_min_bytes: int,
) -> dict[str, Any]:
    wrapper = controls_root / "native_task_runtime_source_adapter_bundle.zip"
    receipt_path = controls_root / "native_task_runtime_source_build_receipt.v1.json"
    if wrapper.exists() and receipt_path.is_file():
        receipt = _load(
            receipt_path, blocker="configured_controls_provisioning_retained_invalid:runtime_source"
        )
        if receipt.get("sha256") != _sha256(wrapper):
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_retained_invalid:runtime_source"
            )
    else:
        if payload_dir.is_symlink() or not payload_dir.is_dir():
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_runtime_payload_invalid"
            )
        try:
            receipt = dict(
                build_task_evaluation_runtime_source_bundle(
                    source_root=payload_dir,
                    output_path=wrapper,
                    expected_production_commit=source_commit,
                    runtime_identity=RUNTIME_IDENTITY,
                    external_layer_store_root=(
                        controls_root / "runtime-source-layers"
                        if external_layer_bucket is not None
                        else None
                    ),
                    external_layer_bucket=external_layer_bucket,
                    external_layer_min_bytes=external_layer_min_bytes,
                )
            )
        except TaskEvaluationNativeArenaAdapterError as exc:
            raise ConfiguredControlsProvisioningError(
                f"configured_controls_provisioning_runtime_source_build_failed:{exc}"
            ) from exc
        _write_once(receipt_path, receipt)
    layers = receipt.get("external_layers") or []
    if layers:
        publication = dict(layer_publisher(receipt))
        published = publication.get("layers")
        if (
            publication.get("status") != "remote_verified"
            or not isinstance(published, list)
            or len(published) != len(layers)
        ):
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_runtime_layer_publication_invalid"
            )
    return _publish(
        path=wrapper,
        kind=ARTIFACT_KINDS["runtime_source"],
        publisher=artifact_publisher,
        reference_path=controls_root / "native-runtime-source-artifact-reference.json",
    )


# ------------------------------------------------------------------ provider zero


def _validated_provider_zero(value: Mapping[str, Any]) -> dict[str, Any]:
    zero = dict(value)
    if (
        zero.get("schema_version") != "adp_paid_provider_zero.v1"
        or zero.get("provider") != "vast"
        or zero.get("api_confirmed") is not True
        or not valid_vast_provider_zero_api_call(zero.get("api_command"))
        or zero.get("global_live_resource_count") != 0
        or zero.get("provider_zero") is not True
        or zero.get("inventory") != []
        or zero.get("raw_secret_values_recorded") is not False
        or zero.get("provider_zero_digest")
        != canonical_digest(zero, digest_field="provider_zero_digest")
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_provider_not_zero"
        )
    return zero


def default_provider_zero_collector() -> dict[str, Any]:
    from .adp_task_evaluation_abstention import collect_vast_provider_zero_receipt

    return collect_vast_provider_zero_receipt()


def default_artifact_publisher(*, path: Path, artifact_kind: str) -> Mapping[str, Any]:
    from .task_evaluation_configured_scene_object_store import publish_configured_scene_artifact

    return publish_configured_scene_artifact(path=path, artifact_kind=artifact_kind)


def default_layer_publisher(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    from .task_evaluation_configured_scene_object_store import (
        publish_runtime_source_external_layers,
    )

    return publish_runtime_source_external_layers(receipt)


def default_external_layer_bucket() -> str:
    from .task_evaluation_configured_scene_object_store import _object_store_client

    return _object_store_client()[1]


# ------------------------------------------------------------------ provisioning


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def provision_configured_controls_continuation(
    *,
    expected_production_commit: str,
    preparation_result_path: str | Path,
    preparation_queue_root: str | Path,
    robot_asset_usd_path: str | Path,
    runtime_source_payload_dir: str | Path,
    embodiment_camera_template_path: str | Path,
    project_spend_reconciliation_path: str | Path,
    controls_root: str | Path,
    profile_dir: str | Path,
    authorization_reference: str,
    authorized_by: str,
    release_reference: str,
    openai_project_id: str,
    openai_api_key_id: str,
    policy_camera_resolution: tuple[int, int] = DEFAULT_POLICY_CAMERA_RESOLUTION,
    overview_camera_resolution: tuple[int, int] = DEFAULT_OVERVIEW_CAMERA_RESOLUTION,
    phase_hard_cap_usd: float = DEFAULT_PHASE_HARD_CAP_USD,
    phase_ttl_seconds: int = DEFAULT_PHASE_TTL_SECONDS,
    maximum_hourly_rate_usd: float = DEFAULT_HOURLY_RATE_USD,
    authority_valid_seconds: int = DEFAULT_AUTHORITY_VALID_SECONDS,
    max_inference_cost_usd: float = DEFAULT_MAX_PLACEMENT_INFERENCE_COST_USD,
    provider_zero_collector: ProviderZeroCollector = default_provider_zero_collector,
    artifact_publisher: ArtifactPublisher = default_artifact_publisher,
    layer_publisher: LayerPublisher = default_layer_publisher,
    external_layer_bucket: str | None = None,
    external_layer_min_bytes: int = DEFAULT_EXTERNAL_LAYER_MIN_BYTES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Author, publish, and seal every continuation input; return the intent path."""

    commit = str(expected_production_commit)
    if _COMMIT.fullmatch(commit) is None:
        raise ConfiguredControlsProvisioningError("configured_controls_provisioning_commit_invalid")
    if not str(authorization_reference or "").strip() or not str(authorized_by or "").strip():
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_authority_text_missing"
        )
    if (
        isinstance(phase_hard_cap_usd, bool)
        or not isinstance(phase_hard_cap_usd, (int, float))
        or not 0 < float(phase_hard_cap_usd) <= 50.0
        or isinstance(phase_ttl_seconds, bool)
        or not isinstance(phase_ttl_seconds, int)
        or phase_ttl_seconds <= 0
        or phase_ttl_seconds * float(maximum_hourly_rate_usd) / 3600.0 > float(phase_hard_cap_usd) + 1e-9
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_phase_spend_invalid"
        )
    root = Path(controls_root).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    context = _preparation_context(
        preparation_result_path=Path(preparation_result_path).expanduser(),
        preparation_queue_root=Path(preparation_queue_root).expanduser(),
        expected_production_commit=commit,
    )
    robot_asset = Path(robot_asset_usd_path).expanduser()
    if robot_asset.is_symlink() or not robot_asset.is_file():
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_robot_asset_missing"
        )
    issued = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    expires = issued + timedelta(seconds=int(authority_valid_seconds))

    # Published inputs the activation worker reads back.
    runtime_source = _runtime_source_reference(
        payload_dir=Path(runtime_source_payload_dir).expanduser(),
        controls_root=root,
        source_commit=commit,
        artifact_publisher=artifact_publisher,
        layer_publisher=layer_publisher,
        external_layer_bucket=external_layer_bucket,
        external_layer_min_bytes=int(external_layer_min_bytes),
    )
    health_protocol = {
        "schema_version": HEALTH_PROTOCOL_SCHEMA_VERSION,
        "source_commit": commit,
        "runtime_identity": dict(RUNTIME_IDENTITY),
        "required_markers": list(REQUIRED_HEALTH_MARKERS),
        "success_requires_native_result_and_terminal_closure": True,
        "health_protocol_digest": "",
    }
    health_protocol["health_protocol_digest"] = canonical_digest(
        health_protocol, digest_field="health_protocol_digest"
    )
    _write_once(root / "runtime_health_protocol.v1.json", health_protocol)
    health_reference = _publish(
        path=root / "runtime_health_protocol.v1.json",
        kind=ARTIFACT_KINDS["health_protocol"],
        publisher=artifact_publisher,
        reference_path=root / "native-health-protocol-artifact-reference.json",
    )
    zero_path = root / "initial_provider_zero_receipt.v1.json"
    if zero_path.is_file():
        zero = _validated_provider_zero(
            _load(zero_path, blocker="configured_controls_provisioning_retained_invalid:provider_zero")
        )
    else:
        zero = _validated_provider_zero(provider_zero_collector())
        _write_once(zero_path, zero)
    zero_reference = _publish(
        path=zero_path,
        kind=ARTIFACT_KINDS["provider_zero"],
        publisher=artifact_publisher,
        reference_path=root / "initial-provider-zero-artifact-reference.json",
    )
    spend_source = Path(project_spend_reconciliation_path).expanduser()
    spend_value = _load(
        spend_source, blocker="configured_controls_provisioning_project_spend_invalid"
    )
    if spend_value.get("schema_version") != "adp_project_spend_reconciliation.v1":
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_project_spend_invalid"
        )
    spend_path = root / "prior_official_spend_reconciliation.v1.json"
    if not spend_path.exists():
        spend_path.write_bytes(spend_source.read_bytes())
        spend_path.chmod(0o440)
    elif spend_path.read_bytes() != spend_source.read_bytes():
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_retained_invalid:project_spend"
        )
    spend_reference = _publish(
        path=spend_path,
        kind=ARTIFACT_KINDS["project_spend"],
        publisher=artifact_publisher,
        reference_path=root / "prior-official-spend-artifact-reference.json",
    )

    # Local intent inputs.
    camera_template = author_camera_template(
        embodiment_template=_load(
            Path(embodiment_camera_template_path).expanduser(),
            blocker="configured_controls_provisioning_embodiment_camera_template_invalid",
        ),
        source_commit=commit,
        policy_camera_resolution=tuple(policy_camera_resolution),
        overview_camera_resolution=tuple(overview_camera_resolution),
    )
    cameras_path = root / "camera_template.v1.json"
    _write_once(cameras_path, camera_template)
    runtime_binding = {
        "runtime": {
            "identity": dict(RUNTIME_IDENTITY),
            "oci_image": NATIVE_TASK_ARENA_IMAGE,
            "entrypoint": list(RUNTIME_ENTRYPOINT),
            "health_protocol": health_reference,
            "requirements": dict(RUNTIME_REQUIREMENTS),
            "network": {"default": "deny", "allowlist": []},
            "secret_refs": [],
            "mounts": [
                {
                    "source": {"deferred": SCENE_BUNDLE_MODE},
                    "container_path": "/inputs",
                    "mode": "read_only",
                },
                {"container_path": "/outputs", "mode": "output"},
            ],
            "output_limit_bytes": RUNTIME_OUTPUT_LIMIT_BYTES,
        },
        "execution_adapter": {
            "kind": "native_task_arena",
            "version": "v1",
            "runtime_source_bundle": runtime_source,
        },
        "spend": {
            "maximum_hourly_rate_usd": float(maximum_hourly_rate_usd),
            "hard_cap_usd": float(phase_hard_cap_usd),
            "hard_ttl_seconds": int(phase_ttl_seconds),
            "retry_cap": 0,
            "selected_provider": "vast",
            "provider_allowlist": ["vast"],
        },
    }
    runtime_binding_path = root / "runtime_binding.v1.json"
    _write_once(runtime_binding_path, runtime_binding)
    rights = context["documents"]["rights_admission"]["reference"]
    phases: dict[str, dict[str, str]] = {}
    for phase in PHASES:
        phase_root = root / phase
        template = {
            "schema_version": RELEASE_WINDOW_TEMPLATE_SCHEMA_VERSION,
            "status": "authorized_for_dynamic_release",
            "team_namespace": context["team_namespace"],
            "expected_production_commit": commit,
            "allowed_mutations": [
                "catalog_synchronization",
                "profile_publication",
                "standing_authorization",
            ],
            "provider_allowlist": ["vast"],
            "maximum_hard_cap_usd": float(phase_hard_cap_usd),
            "valid_for_seconds": RELEASE_WINDOW_VALID_SECONDS,
            "released_by": str(authorized_by),
            "release_reference": str(release_reference),
            "provider_resource_allocation_allowed": False,
            "paid_request_allowed": False,
            "template_digest": "",
        }
        template["template_digest"] = canonical_digest(template, digest_field="template_digest")
        _write_once(phase_root / "release_window_template.v1.json", template)
        authorization = _write_once(
            phase_root / "authorization.v1.json",
            {
                "reference": str(authorization_reference),
                "authorized_by": str(authorized_by),
                "authorized_on": _iso(issued),
                "standing_authorization_expires_at": _iso(expires),
                "profile_revision": f"scene-{context['scene_id']}-{commit[:12]}",
            },
        )
        _write_once(
            phase_root / "launch_authority.v1.json",
            {
                "rights_scope": RIGHTS_SCOPE,
                "rights_evidence": dict(rights),
                "max_spend_usd": float(phase_hard_cap_usd),
                "expires_at": authorization["standing_authorization_expires_at"],
            },
        )
        rows = {
            "release_window_template_path": str(phase_root / "release_window_template.v1.json"),
            "authorization_path": str(phase_root / "authorization.v1.json"),
            "launch_authority_path": str(phase_root / "launch_authority.v1.json"),
        }
        if phase == "construction":
            _write_once(
                phase_root / "lineage.v1.json",
                {
                    "kind": "initial_project",
                    "project_spend_reconciliation": spend_reference,
                    "initial_provider_zero": zero_reference,
                },
            )
            rows["lineage_path"] = str(phase_root / "lineage.v1.json")
        phases[phase] = rows
    paths = {
        "robot_asset_usd_path": str(robot_asset),
        "robot_mount_interface_path": str(context["documents"]["robot_mount_interface"]["path"]),
        "scene_camera_calibration_path": str(context["documents"]["camera_calibration"]["path"]),
        "native_trajectory_plan_path": {"deferred": TRAJECTORY_MODE},
        "cameras_path": str(cameras_path),
        "runtime_binding_path": str(runtime_binding_path),
        "overview_image_paths": {"deferred": OVERVIEW_MODE},
    }
    intent_path = root / configured_controls_autostart_registry_name(
        team_namespace=context["team_namespace"],
        scene_id=context["scene_id"],
        task_id=context["task_id"],
    )
    try:
        intent = materialize_configured_controls_autostart_intent(
            expected_production_commit=commit,
            submitted_by="configured-controls-continuation-provisioning",
            team_namespace=context["team_namespace"],
            scene_id=context["scene_id"],
            task_id=context["task_id"],
            target_position_world_m=context["target_position_world_m"],
            paths=paths,
            phases=phases,
            profile_dir=Path(profile_dir).expanduser(),
            output_path=intent_path,
            max_inference_cost_usd=float(max_inference_cost_usd),
            openai_project_id=str(openai_project_id),
            openai_api_key_id=str(openai_api_key_id),
        )
    except TaskEvaluationConfiguredControlsAutostartError as exc:
        raise ConfiguredControlsProvisioningError(
            f"configured_controls_provisioning_intent_invalid:{exc}"
        ) from exc
    return {
        "schema_version": "task_evaluation_configured_controls_continuation_provisioning.v1",
        "status": "configured_controls_continuation_provisioned",
        "expected_production_commit": commit,
        "team_namespace": context["team_namespace"],
        "scene_id": context["scene_id"],
        "task_id": context["task_id"],
        "controls_root": str(root),
        "intent_path": str(intent_path),
        "intent_digest": intent["intent_digest"],
        "runtime_source_bundle": runtime_source,
        "health_protocol": health_reference,
        "initial_provider_zero": zero_reference,
        "project_spend_reconciliation": spend_reference,
        "deferred_inputs": sorted(
            name for name, value in paths.items() if isinstance(value, Mapping)
        ),
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


# ------------------------------------------------------------------ registry


def _retire_superseded_registration(
    registry_path: Path, *, expected_production_commit: str
) -> Path | None:
    """Move a live registration bound to another release out of the workers' view.

    Every deploy changes the commit and the same continuation is re-provisioned
    at the new release.  Workers resolve only ``<identity>.json``; the previous
    release's entry keeps its bytes under a name no worker reads.  Two
    registrations of the same release still conflict unless byte-identical.
    """

    if not registry_path.exists() and not registry_path.is_symlink():
        return None
    if registry_path.is_symlink():
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_registry_conflict"
        )
    try:
        existing = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_registry_conflict"
        ) from exc
    previous = str(existing.get("expected_production_commit") or "") if isinstance(existing, Mapping) else ""
    if previous == expected_production_commit or re.fullmatch(r"[0-9a-f]{40}", previous) is None:
        return None
    retired = registry_path.with_name(
        f"{registry_path.name.removesuffix('.json')}.superseded-{previous}.json"
    )
    if retired.exists() or retired.is_symlink():
        if retired.is_symlink() or retired.read_bytes() != registry_path.read_bytes():
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_registry_conflict"
            )
        registry_path.unlink()
        return retired
    os.replace(registry_path, retired)
    return retired


def install_intent_into_registry(
    *,
    intent_path: str | Path,
    intent_root: str | Path,
    expected_production_commit: str,
    service_group: str | None = DEFAULT_SERVICE_GROUP,
) -> dict[str, Any]:
    """Install one sealed intent read-only where the activation and progression workers look."""

    source = Path(intent_path).expanduser()
    intent = validate_configured_controls_autostart_intent(
        _load(source, blocker="configured_controls_provisioning_intent_invalid")
    )
    if intent["expected_production_commit"] != expected_production_commit:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_intent_commit_mismatch"
        )
    if intent["configuration_adoption"] != {"mode": "same_commit_automatic"}:
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_intent_adoption_invalid"
        )
    root = Path(intent_root).expanduser()
    if not root.is_absolute() or root.is_symlink():
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_intent_root_invalid"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    destination = root / configured_controls_autostart_registry_name(
        team_namespace=intent["team_namespace"],
        scene_id=intent["scene_id"],
        task_id=intent["task_id"],
    )
    payload = source.read_bytes()
    retired = _retire_superseded_registration(
        destination, expected_production_commit=expected_production_commit
    )
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise ConfiguredControlsProvisioningError(
                "configured_controls_provisioning_registry_conflict"
            )
    else:
        with destination.open("xb") as stream:
            stream.write(payload)
    group_id = None
    if service_group is not None:
        try:
            group_id = grp.getgrnam(service_group).gr_gid
        except KeyError as exc:
            raise ConfiguredControlsProvisioningError(
                f"configured_controls_provisioning_service_group_missing:{service_group}"
            ) from exc
        os.chown(destination, os.geteuid() if os.geteuid() != 0 else 0, group_id)
        if retired is not None:
            os.chown(retired, os.geteuid() if os.geteuid() != 0 else 0, group_id)
    destination.chmod(0o440)
    if retired is not None:
        retired.chmod(0o440)
    metadata = destination.stat()
    if (
        stat.S_IMODE(metadata.st_mode) != 0o440
        or _sha256(destination) != _sha256(source)
        or (group_id is not None and metadata.st_gid != group_id)
    ):
        raise ConfiguredControlsProvisioningError(
            "configured_controls_provisioning_registry_readback_mismatch"
        )
    return {
        "status": "installed",
        "registry_path": str(destination),
        "intent_digest": intent["intent_digest"],
        "provider_mutation_performed": False,
    }


# ------------------------------------------------------------------ CLI


def _resolution(value: str) -> tuple[int, int]:
    width, _, height = str(value).lower().partition("x")
    try:
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must look like 640x360") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    provision = commands.add_parser("provision", help="Author, publish, and seal the continuation intent.")
    provision.add_argument("--expected-production-commit", required=True)
    provision.add_argument("--preparation-result", required=True)
    provision.add_argument(
        "--preparation-queue-root",
        default=os.getenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT")
        or "/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations",
    )
    provision.add_argument("--robot-asset-usd", required=True)
    provision.add_argument("--runtime-source-payload-dir", required=True)
    provision.add_argument("--embodiment-camera-template", required=True)
    provision.add_argument("--project-spend-reconciliation", required=True)
    provision.add_argument("--controls-root", required=True)
    provision.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    provision.add_argument("--authorization-reference", required=True)
    provision.add_argument("--authorized-by", required=True)
    provision.add_argument("--release-reference", required=True)
    provision.add_argument("--openai-project-id", default=os.getenv("OPENAI_PROJECT_ID"))
    provision.add_argument(
        "--openai-api-key-id", default=os.getenv("OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID")
    )
    provision.add_argument("--policy-camera-resolution", type=_resolution, default=DEFAULT_POLICY_CAMERA_RESOLUTION)
    provision.add_argument("--overview-camera-resolution", type=_resolution, default=DEFAULT_OVERVIEW_CAMERA_RESOLUTION)
    provision.add_argument("--phase-hard-cap-usd", type=float, default=DEFAULT_PHASE_HARD_CAP_USD)
    provision.add_argument("--phase-ttl-seconds", type=int, default=DEFAULT_PHASE_TTL_SECONDS)
    provision.add_argument("--authority-valid-seconds", type=int, default=DEFAULT_AUTHORITY_VALID_SECONDS)
    provision.add_argument("--external-layer-min-bytes", type=int, default=DEFAULT_EXTERNAL_LAYER_MIN_BYTES)
    install = commands.add_parser("install-intent", help="Install a sealed intent into the registry (run as root).")
    install.add_argument("--intent", required=True)
    install.add_argument("--intent-root", default=DEFAULT_INTENT_ROOT)
    install.add_argument("--expected-production-commit", required=True)
    install.add_argument("--service-group", default=DEFAULT_SERVICE_GROUP)
    args = parser.parse_args(argv)
    if args.command == "install-intent":
        print(
            json.dumps(
                install_intent_into_registry(
                    intent_path=args.intent,
                    intent_root=args.intent_root,
                    expected_production_commit=args.expected_production_commit,
                    service_group=args.service_group,
                ),
                sort_keys=True,
            )
        )
        return 0
    if not args.openai_project_id or not args.openai_api_key_id:
        parser.error("--openai-project-id and --openai-api-key-id are required")
    result = provision_configured_controls_continuation(
        expected_production_commit=args.expected_production_commit,
        preparation_result_path=args.preparation_result,
        preparation_queue_root=args.preparation_queue_root,
        robot_asset_usd_path=args.robot_asset_usd,
        runtime_source_payload_dir=args.runtime_source_payload_dir,
        embodiment_camera_template_path=args.embodiment_camera_template,
        project_spend_reconciliation_path=args.project_spend_reconciliation,
        controls_root=args.controls_root,
        profile_dir=args.profile_dir,
        authorization_reference=args.authorization_reference,
        authorized_by=args.authorized_by,
        release_reference=args.release_reference,
        openai_project_id=args.openai_project_id,
        openai_api_key_id=args.openai_api_key_id,
        policy_camera_resolution=args.policy_camera_resolution,
        overview_camera_resolution=args.overview_camera_resolution,
        phase_hard_cap_usd=args.phase_hard_cap_usd,
        phase_ttl_seconds=args.phase_ttl_seconds,
        authority_valid_seconds=args.authority_valid_seconds,
        external_layer_bucket=default_external_layer_bucket(),
        external_layer_min_bytes=args.external_layer_min_bytes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "ConfiguredControlsProvisioningError",
    "DEFAULT_OVERVIEW_CAMERA_RESOLUTION",
    "DEFAULT_POLICY_CAMERA_RESOLUTION",
    "author_camera_template",
    "install_intent_into_registry",
    "main",
    "provision_configured_controls_continuation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
