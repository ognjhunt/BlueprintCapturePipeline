"""One direct GPU execution of the prepared G1 kitchen GR00T+OSCAR episode."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import lzma
import math
import re
import shlex
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse
import urllib.request

from .common import ensure_dir, write_json
from .gpu_render_providers import get_render_provider
from .gear_sonic_isaac_dds_bridge import (
    BRIDGE_HEARTBEAT_PATH,
    BRIDGE_LOG_PATH,
    BRIDGE_PID_ENV,
    BRIDGE_REQUIRED_ENV,
    SNAPSHOT_DEFAULT_PATH,
    SNAPSHOT_ENV,
    bridge_prepare_script,
    bridge_start_script,
)
from .groot_oscar_digitalocean_closed_loop_job import (
    build_launch_spec,
    build_worker_bootstrap_script,
)
from .g1_microwave_groot_finetune_component import REMOTE_FINAL_CHECKPOINT
from .groot_oscar_runpod_watchdog import arm_watchdog, terminate_canary_resources
from .groot_oscar_worker_startup_script import qualification_checkpoint_preflight_script
from .isaac_particlefield_render_job import watch_and_collect
from .oscar_official_release import (
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_SOURCE_URL,
)
from .oscar_runtime_asset_contract import (
    DEFAULT_OSCAR_CHECKPOINT_ROOT,
    DEFAULT_RUNTIME_MODEL_CACHE_ROOT,
    offline_runtime_environment,
    render_offline_export_script,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    require_paid_resource_admission,
)
from .single_g1_kitchen_qualification_admission import qualification_pre_spend_preflight
from .wam_derived_observation_harness import (
    GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND,
)
from .wam_provider_object_store import (
    SCHEMA_VERSION as OBJECT_STORE_STAGING_SCHEMA_VERSION,
    SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION,
    signed_output_object_binding_sha256,
)


SCHEMA_VERSION = "single_g1_kitchen_episode_runpod.v1"
MANIPULATION_POLICY_TASK_COMPATIBILITY_SCHEMA_VERSION = "single_g1_kitchen_manipulation_policy_task_compatibility.v2"
MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER = "single_episode_manipulation_policy_task_compatibility_unproven"
DIRECT_OWNER_SCHEMA_VERSION = "single_g1_kitchen_episode_direct_owner.v1"
DIRECT_OWNER_NAME = "single_episode_direct_owner.json"
DIRECT_OWNER_LANE = "single_g1_kitchen_episode"
PROBE_KIND = "single-kitchen-episode"
IMAGE_REF = (
    "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:"
    "ab8fbccb714242b55811aa5142933001dfba76d56b5cc29dead4d0bdf1346e88"
)
IMAGE_DIGEST = IMAGE_REF.rsplit("@", 1)[-1]
# The checked-in legacy direct-lane image predates the build-time OSCAR source
# seal.  Keep this reviewed allowlist empty until that exact pin is replaced by
# a seal-capable image; release-bound qualification sessions use their explicit
# build result instead of this static pin.
OSCAR_RUNTIME_SOURCE_SEAL_CAPABLE_IMAGE_DIGESTS: frozenset[str] = frozenset()
BUNDLE_SHA256 = "e3274f1468fb998e48779d7865c67e2d58f2ed4eb14f41886c2d14438ed98692"
SOURCE_COMMIT = "fca4712e6bb78bf251512cead5ee7787ed2fb249"
SEALED_SONIC_CHECKPOINT_REPO = "LucaFrat/groot-bs16"
SEALED_SONIC_CHECKPOINT_REVISION = "86b17337379926a8d8f1ad5c4580c7c33deeb49f"
SEALED_SONIC_TRAINING_DATASET_REPO = "LucaFrat/dataset_100"
SEALED_SONIC_TRAINING_DATASET_REVISION = (
    "98976e7b62650a3848192ecf285902b296b276a2"
)
# Reviewed public provenance for the exact sealed checkpoint.  The public
# dataset declares one bag-manipulation task and no Blueprint canonical task
# ids.  Keep these values code-reviewed rather than trusting a launch-plan
# declaration: otherwise the current checkpoint could be relabeled as a
# microwave fine-tune without changing a single model byte.
SEALED_SONIC_REVIEWED_TRAINING_TASK_IDS: tuple[str, ...] = ()
SEALED_SONIC_REVIEWED_TRAINING_TASK_DESCRIPTIONS = (
    "grab the bag, turn 180 degrees and drop the bag",
)
REQUIRED_SONIC_EMBODIMENT = "UNITREE_G1_SONIC"
REQUIRED_MANIPULATION_TASK_ID = "microwave_door"
REQUIRED_MANIPULATION_TRANSITION_ID = "microwave_door_open_angle"
# The sealed scene already places G1 at the manipulation start pose.  A
# navigation preamble made the first learned horizon remain in a standing
# posture with both hands below a true head-mounted camera, so it could not
# provide the visible action conditioning that the OSCAR manipulation rollout
# requires.  Keep the instruction task-direct and bind the intended reach arm,
# matching the validated late-June egocentric manipulation lane.
TASK_PROMPT = "Open the microwave door using the right hand."
GPU_TYPES = (
    "NVIDIA A40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
)
QUALIFICATION_RUNPOD_GPU_TYPES = ("NVIDIA A40",)
SUPPORTED_PROVIDERS = ("runpod", "vast")
VAST_PREFERRED_GPU_KEYWORDS = ("L40S",)
VAST_MIN_RELIABILITY = 0.99
VAST_REQUIRE_KNOWN_SUPPORTED_ISAAC_DRIVER = True
MAX_HOURLY_RATE_USD = 1.10
WALL_SECONDS = 18_000
WATCH_SECONDS = 17_700
EPISODE_TIMEOUT_SECONDS = 14_400
PROGRESS_TIMEOUT_SECONDS = EPISODE_TIMEOUT_SECONDS + 900
# Restore the trained OSCAR/Wan per-query horizon. This is not the episode
# length: the outer live-Isaac loop is task-adaptive, stops on the registered
# task transition, and uses its 48-step value only as a safety cap. Shortening
# this model horizon to 8/9 frames made frame 1 coherent but every decoded
# frame after it collapse into artifact soup on the paid L40S run.
DIRECT_OSCAR_NUM_FRAMES = 81
DIRECT_OSCAR_NUM_STEPS = 35
# OSCAR's first live clip remained seed-coherent, but feeding its selected
# generated frame directly into the next generation caused the second clip to
# decorrelate immediately. Re-anchor every outer step to the newly captured,
# validated post-action Isaac robot-POV frame while retaining the generated
# observation as review and policy-requery evidence. This advances with the
# live task state and keeps the coherence floor fail-closed instead of
# weakening it to admit drifted video.
DIRECT_CLEAN_FRAME_REANCHOR_INTERVAL = 1
# The signed baseline rules out a trivially pre-opened microwave.  Therefore a
# real registered transition on the first outer policy query is sufficient and
# must terminate immediately instead of executing two unnecessary queries.
DIRECT_EPISODE_MIN_STEPS = 1
# Bound paid episodes that are visibly going nowhere.  Completion remains
# task-driven, but three consecutive live post-action measurements without
# registered task progress now terminate the run instead of consuming the
# entire 48-step safety cap.  Unsafe stance/fall detection is independently
# terminal so a fallen robot never spends minutes generating another WAM clip.
DIRECT_NO_PROGRESS_PATIENCE_STEPS = 3
# The frozen July bundle selected its nearest 0.35 m stance.  Live RTX proved
# that this leaves only ~0.22 m from the physical head lens to the microwave
# handle, filling the frame with the appliance and adjacent wall.  Select the
# scenario's already validated 0.63 m candidate for the direct egocentric lane;
# this changes the standing initialization, never the head-local camera mount.
DIRECT_EGOCENTRIC_STANCE_MIN_DISTANCE_M = 0.63
# GR00T N1.7 SONIC returns a 40-frame control horizon. This direct evaluation
# executes that complete horizon; the generic closed-loop CLI retains its
# one-frame default so other lanes do not change behavior implicitly.
DIRECT_GROOT_SONIC_EXECUTION_FRAME_COUNT = 40
QUALIFICATION_CHECKPOINT_PART_GET_URLS_ENV = (
    "BLUEPRINT_G1_MICROWAVE_QUALIFICATION_CHECKPOINT_PART_GET_URLS"
)
QUALIFICATION_CHECKPOINT_RESTORE_SCHEMA_VERSION = (
    "single_g1_kitchen_qualification_checkpoint_restore.v1"
)
QUALIFICATION_CHECKPOINT_RESTORE_REPORT_PATH = "/workspace/closed_loop_out/qualification_checkpoint_restore.json"
MAX_QUALIFICATION_CHECKPOINT_PARTS = 16
SINGLE_EPISODE_PROGRESS_PHASES = (
    "container_bash_started",
    "inputs_ready",
    "provider_allocation_bound",
    "oscar_runtime_dependency_install_started",
    "oscar_runtime_dependencies_bound",
    "oscar_runtime_asset_prepare_started",
    "oscar_runtime_assets_prepared",
    "oscar_runtime_asset_offline_preflight_passed",
    "oscar_runtime_import_preflight_passed",
    "groot_checkpoint_preflight_passed",
    "healthcheck_passed",
    "groot_server_ready",
    "gear_sonic_controller_ready",
    "isaac_task_executor_ready",
)
EXTERNAL_CONSISTENCY_OPTIONS = {
    "--wam-consistency-command": True,
    "--wam-consistency-timeout-seconds": True,
    "--allow-wam-consistency-scoring": False,
    "--require-forward-inverse-consistency": False,
}
EXTERNAL_CONSISTENCY_ENV = {
    "BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING",
    "BLUEPRINT_WAM_EPISODE_CONSISTENCY_COMMAND",
}
DIRECT_RGB_OBSERVATION_REMOVED_REQUIREMENTS = (
    "--require-real-perception-backend",
    "--require-sam3-completed",
    "--require-da3-completed",
)
OSCAR_PYTHON = "/opt/oscar-venv/bin/python"
ISAAC_PYTHON = "/isaac-sim/python.sh"
SYSTEM_PYTHON = "python3"
SEALED_GROOT_HF_HOME = "/opt/blueprint/hf_home"
SEALED_GROOT_HF_HUB_CACHE = f"{SEALED_GROOT_HF_HOME}/hub"
SEALED_COSMOS_BACKBONE_REPO = "nvidia/Cosmos-Reason2-2B"
SEALED_COSMOS_BACKBONE_REVISION = (
    "9ce19a195e423419c349abfc86fd07178b230561"
)
RUNTIME_PACKAGE_OVERLAY_DIR = "/workspace/runtime_overlay/package"
RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV = "BLUEPRINT_SINGLE_EPISODE_RUNTIME_OVERLAY_XZ_BASE64"
RUNTIME_PACKAGE_OVERLAY_SHA256_ENV = "BLUEPRINT_SINGLE_EPISODE_RUNTIME_OVERLAY_SHA256"
RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH = "/workspace/single_episode_runtime_overlay_xz_base64.txt"
OVERLAY_PAYLOAD_TRANSPORT_ENV = "BLUEPRINT_SINGLE_EPISODE_OVERLAY_PAYLOAD_TRANSPORT"
OVERLAY_PAYLOAD_FILE_TRANSPORT = "file_v1"
LINUX_MAX_ARG_STRLEN_BYTES = 128 * 1024
CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV = "BLUEPRINT_CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT"
INITIAL_POLICY_FRAME_PATH = "/workspace/initial_policy_frame.png"
CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH = (
    "/workspace/controller_fk_camera_projection_context.json"
)
RUNTIME_PACKAGE_OVERLAY_MODULES = (
    "groot_oscar_episode_review.py",
    "oscar_official_release.py",
    "oscar_runtime_source_provenance.py",
    "oscar_runtime_asset_contract.py",
    "unitree_groot_n17_sonic_policy_server_command.py",
    "oscar_isaac_closed_loop_eval.py",
    "oscar_wam_provider_bundle.py",
    "wam_derived_observation_harness.py",
    "groot_sonic_policy_endpoint.py",
    "gear_sonic_joint_order_contract.py",
    "gear_sonic_controller_fk_adapter.py",
    "gear_sonic_official_zmq_executor.py",
    "gear_sonic_process_supervisor.py",
    "isaac_persistent_task_executor_service.py",
    "isaac_persistent_task_completion_client.py",
    "task_episode_baseline.py",
    "isaac_task_review_renderer.py",
    "isaac_review_media.py",
    "g1_kitchen_worker_proof_emission.py",
)
REQUIRED_RUNTIME_PYTHONPATHS = (
    RUNTIME_PACKAGE_OVERLAY_DIR,
    "/opt/wbc",
    "/opt/OSCAR",
)
ISAAC_TASK_EXECUTOR_MODULE = "blueprint_pipeline.isaac_persistent_task_executor_service"
GEAR_SONIC_PROCESS_SUPERVISOR_MODULE = (
    "blueprint_pipeline.gear_sonic_process_supervisor"
)
ISAAC_RUNTIME_BACKEND_MODULE = "blueprint_pipeline.isaac_runtime_task_backend"
ISAAC_RUNTIME_OVERLAY_DIR = "/workspace/runtime_overlay"
ISAAC_RUNTIME_OVERLAY_SOURCE = f"{ISAAC_RUNTIME_OVERLAY_DIR}/isaac_runtime_task_backend.py"
ISAAC_RUNTIME_OVERLAY_WRAPPER = f"{ISAAC_RUNTIME_OVERLAY_DIR}/run_patched_isaac_executor.py"
ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV = "BLUEPRINT_ISAAC_RUNTIME_BACKEND_OVERLAY_GZIP_BASE64"
ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH = (
    "/workspace/single_episode_isaac_backend_overlay_gzip_base64.txt"
)
OSCAR_RUNTIME_PROVENANCE_ARTIFACT = "/workspace/closed_loop_out/oscar_runtime_provenance.json"
OSCAR_CUDNN_LIB_DIR_ENV = "BLUEPRINT_OSCAR_CUDNN_LIB_DIR"
OSCAR_CUDNN_LIB_DIR = "/opt/oscar-venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
OSCAR_FOUNDATION_REQUIREMENTS_LOCK_RELATIVE_PATH = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/requirements_oscar_foundation.lock"
)
OSCAR_FOUNDATION_REQUIREMENTS_LOCK_SHA256 = (
    "307778ea92f22091d0e10607d30323fc422b70052843cd74c619faf91ea6dd6d"
)
OSCAR_FOUNDATION_REQUIREMENTS_LOCK_PACKAGE_COUNT = 121
OSCAR_RUNTIME_REQUIRED_PYTEST_VERSION = "9.1.1"
OSCAR_RUNTIME_DEPENDENCY_LOCK_PATH = "/workspace/oscar_runtime_requirements_foundation.lock"
OSCAR_RUNTIME_DEPENDENCY_TARGET = "/workspace/oscar_runtime_deps"
GROOT_RUNTIME_PYTHONPATH_ENV = "BLUEPRINT_GROOT_RUNTIME_PYTHONPATH"
GROOT_VENV_ROOT_ENV = "BLUEPRINT_GROOT_VENV_ROOT"
GROOT_VENV_ROOT = "/opt/gr00t-venv"
OSCAR_RUNTIME_DEPENDENCY_TARGET_ENV = (
    "BLUEPRINT_OSCAR_RUNTIME_DEPENDENCY_TARGET"
)
OSCAR_RUNTIME_DEPENDENCY_ARTIFACT = (
    "/workspace/closed_loop_out/oscar_runtime_dependency_repair.json"
)
OSCAR_RUNTIME_PREFLIGHT_ARTIFACT = "/workspace/closed_loop_out/oscar_runtime_import_preflight.json"
OSCAR_RUNTIME_ASSET_CACHE_ROOT = str(DEFAULT_RUNTIME_MODEL_CACHE_ROOT)
OSCAR_RUNTIME_ASSET_CHECKPOINT_ROOT = str(DEFAULT_OSCAR_CHECKPOINT_ROOT)
OSCAR_RUNTIME_ASSET_PREPARE_ARTIFACT = "/workspace/closed_loop_out/oscar_runtime_asset_prepare.json"
OSCAR_RUNTIME_ASSET_PREFLIGHT_ARTIFACT = (
    "/workspace/closed_loop_out/oscar_runtime_asset_offline_preflight.json"
)
RUNPOD_WORKSPACE_VOLUME_GB = 60
# Vast maps /workspace to the instance disk requested by container_disk_gb;
# its adapter ignores this legacy generic volume field.
VAST_IGNORED_LEGACY_VOLUME_GB = 20
VAST_BOOTSTRAP_URL_ENV = "BLUEPRINT_VAST_REMOTE_BOOTSTRAP_SIGNED_GET_URL"
VAST_BOOTSTRAP_SHA256_ENV = "BLUEPRINT_VAST_REMOTE_BOOTSTRAP_SHA256"
VAST_BOOTSTRAP_ARTIFACT_NAME = "provider_bootstrap.sh"
VAST_BOOTSTRAP_MAX_BYTES = 8 * 1024 * 1024
SIGNED_OUTPUT_STAGING_MANIFEST_NAME = "wam_provider_object_store_staging_manifest.json"
SIGNED_OUTPUT_MIN_REMAINING_SECONDS = WALL_SECONDS + 15 * 60

VAST_BOOTSTRAP_DOWNLOADER_PYTHON = r"""import hashlib
import os
import urllib.parse
import urllib.request
from pathlib import Path

url = os.environ.pop("BLUEPRINT_VAST_REMOTE_BOOTSTRAP_SIGNED_GET_URL", "").strip()
expected = os.environ.pop("BLUEPRINT_VAST_REMOTE_BOOTSTRAP_SHA256", "").strip().lower()
parsed = urllib.parse.urlsplit(url)
if (
    parsed.scheme != "https"
    or not parsed.hostname
    or parsed.username is not None
    or parsed.password is not None
    or parsed.fragment
):
    raise SystemExit("vast_remote_bootstrap_url_invalid")
if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
    raise SystemExit("vast_remote_bootstrap_expected_sha256_invalid")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("vast_remote_bootstrap_redirect_forbidden")

try:
    opener = urllib.request.build_opener(NoRedirect())
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with opener.open(request, timeout=90) as response:
        if int(getattr(response, "status", 0)) != 200 or response.geturl() != url:
            raise RuntimeError("vast_remote_bootstrap_response_invalid")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > 8388608:
            raise RuntimeError("vast_remote_bootstrap_too_large")
        payload = response.read(8388609)
except Exception as exc:
    raise SystemExit(
        f"vast_remote_bootstrap_download_failed:{type(exc).__name__}:{exc}"
    ) from exc
if len(payload) > 8388608:
    raise SystemExit("vast_remote_bootstrap_too_large")
if hashlib.sha256(payload).hexdigest() != expected:
    raise SystemExit("vast_remote_bootstrap_sha256_mismatch")
destination = Path("/tmp/blueprint-provider-bootstrap.sh")
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
if temporary.exists() or temporary.is_symlink():
    raise SystemExit("vast_remote_bootstrap_temporary_path_unsafe")
with temporary.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, destination)
os.execv("/bin/bash", ["bash", str(destination)])
"""


def _open_direct_episode_owner(
    *,
    root: Path,
    provider_name: str,
    run_id: str,
    pod_name: str,
    pod_name_prefix: str,
    watchdog_deadline_epoch: float,
    watchdog_pid: int,
    registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Record direct-attempt ownership before the first billable mutation.

    This is a generic crash-recovery teardown obligation, not campaign state or
    a campaign lease.  It makes the direct controller plus its attempt-local
    hard-TTL watchdog the sole resource owner without pulling the one-episode
    test back into campaign orchestration.
    """

    pending = open_pending_teardown(
        provider=provider_name,
        lane=DIRECT_OWNER_LANE,
        run_id=run_id,
        resource_name=pod_name,
        job_dir=root,
        max_age_seconds=WALL_SECONDS + 1800,
        registry_dir=registry_dir,
    )
    owner = {
        "schema_version": DIRECT_OWNER_SCHEMA_VERSION,
        "status": "armed_before_provider_launch",
        "owner_kind": "direct_single_episode_controller_and_attempt_local_watchdog",
        "run_id": run_id,
        "provider": provider_name,
        "pod_name": pod_name,
        "pod_name_prefix": pod_name_prefix,
        "instance_id": None,
        "started_instance_id_file": str(root / "started_vast_instance_id.txt")
        if provider_name == "vast"
        else None,
        "watchdog_evidence_file": str(root / "groot_oscar_runpod_canary_watchdog.json"),
        "watchdog_deadline_epoch": float(watchdog_deadline_epoch),
        "watchdog_pid": int(watchdog_pid),
        "pending_teardown_record": pending["path"],
        "pending_teardown_lane": DIRECT_OWNER_LANE,
        "campaign_ownership_required": False,
        "campaign_lane_handoff_performed": False,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    write_json(root / DIRECT_OWNER_NAME, owner)
    return {"pending": pending, "owner": owner}


def _bind_direct_episode_owner(
    *,
    root: Path,
    ownership: Mapping[str, Any],
    instance_id: str,
    watchdog_pid: int,
) -> dict[str, Any]:
    owner = dict(ownership.get("owner") or {})
    pending = dict(ownership.get("pending") or {})
    pending_path = str(pending.get("path") or "").strip()
    if not pending_path:
        raise ValueError("direct_episode_pending_teardown_record_missing")
    bound_pending = bind_pending_teardown_instance(pending_path, instance_id)
    owner.update(
        {
            "status": "allocation_bound_to_direct_owner",
            "instance_id": str(instance_id),
            "watchdog_pid": int(watchdog_pid),
            "provider_mutations_performed": 1,
        }
    )
    write_json(root / DIRECT_OWNER_NAME, owner)
    return {"pending": bound_pending, "owner": owner}


def _settle_direct_episode_owner(
    *,
    root: Path,
    ownership: Mapping[str, Any],
    teardown: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    owner = dict(ownership.get("owner") or {})
    pending = dict(ownership.get("pending") or {})
    pending_path = str(pending.get("path") or "").strip()
    instance_id = str(launch.get("instance_id") or owner.get("instance_id") or "").strip()
    if not pending_path:
        owner.update(
            {
                "status": "control_plane_unverified",
                "blockers": ["direct_episode_pending_teardown_record_missing"],
            }
        )
    elif not instance_id:
        allocation_ambiguous = launch.get("allocation_outcome_ambiguous") is True
        if allocation_ambiguous:
            pending = mark_pending_teardown_ambiguous(
                pending_path,
                reason="direct_episode_provider_create_outcome_ambiguous",
                evidence={"status": launch.get("status")},
            )
        else:
            pending = cancel_pending_teardown(
                pending_path,
                reason="direct_episode_launch_returned_no_allocation",
                evidence={
                    "status": launch.get("status"),
                    "allocation_outcome_ambiguous": False,
                },
            )
        owner.update(
            {
                "status": (
                    "allocation_outcome_ambiguous_pending_record_retained_open"
                    if allocation_ambiguous
                    else "cancelled_no_allocation"
                ),
                "provider_mutations_performed": 0,
            }
        )
    elif teardown.get("provider_absence_confirmed") is True:
        pending = close_pending_teardown(
            pending_path,
            {
                "status": "PASS",
                "provider": owner.get("provider"),
                "allocation_id": instance_id,
                "provider_absence_confirmed": True,
                "status_source": "attempt_local_watchdog_exact_id_and_inventory",
            },
        )
        owner.update(
            {
                "status": "provider_terminal_and_control_plane_closed",
                "instance_id": instance_id,
                "provider_absence_confirmed": True,
                "pending_teardown_status": pending.get("status"),
            }
        )
    else:
        owner.update(
            {
                "status": "teardown_unverified_pending_record_retained_open",
                "instance_id": instance_id,
                "provider_absence_confirmed": False,
                "pending_teardown_status": pending.get("status"),
                "blockers": ["direct_episode_provider_teardown_not_proven"],
            }
        )
    owner["teardown_status"] = teardown.get("status")
    write_json(root / DIRECT_OWNER_NAME, owner)
    return {"pending": pending, "owner": owner}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_secret_url_file(path: str | Path) -> str:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"signed_url_file_missing_or_unsafe:{unresolved.name}")
    source = unresolved.resolve()
    if not source.is_file():
        raise ValueError(f"signed_url_file_missing_or_unsafe:{source.name}")
    if source.stat().st_mode & 0o077:
        raise ValueError(f"signed_url_file_permissions_not_0600:{source.name}")
    value = source.read_text(encoding="utf-8").strip()
    if not value.startswith("https://"):
        raise ValueError(f"signed_url_not_https:{source.name}")
    return value


class _HttpsRangeReader(io.RawIOBase):
    """Minimal seekable reader for a fixed presigned HTTPS object."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._position = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if int(getattr(response, "status", 0)) != 206 or response.geturl() != url:
                raise ValueError("single_episode_remote_bundle_range_probe_invalid")
            content_range = str(response.headers.get("Content-Range") or "")
            response.read(1)
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if not match or int(match.group(1)) <= 0:
            raise ValueError("single_episode_remote_bundle_size_invalid")
        self._size = int(match.group(1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("single_episode_remote_bundle_seek_whence_invalid")
        if position < 0:
            raise ValueError("single_episode_remote_bundle_seek_negative")
        self._position = min(position, self._size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        if size == 0:
            return b""
        end = min(self._size - 1, self._position + size - 1)
        request = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={self._position}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            if int(getattr(response, "status", 0)) != 206 or response.geturl() != self._url:
                raise ValueError("single_episode_remote_bundle_range_read_invalid")
            payload = response.read(end - self._position + 1)
        expected = end - self._position + 1
        if len(payload) != expected:
            raise ValueError("single_episode_remote_bundle_range_read_short")
        self._position += len(payload)
        return payload


def _sha256_remote_bundle(url: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=300) as response:
        if int(getattr(response, "status", 0)) != 200 or response.geturl() != url:
            raise ValueError("single_episode_remote_bundle_download_invalid")
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_signed_output_staging_proof(
    *,
    provider_output_put_url_file: str | Path,
    provider_output_get_url_file: str | Path,
    put_url: str,
    get_url: str,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Bind allocation to a cleaned, distinct-key PUT/GET staging probe.

    The actual output object is never populated by the probe.  Object-store
    staging exercises a separate run-unique sentinel key, verifies exact bytes
    and SHA-256 over signed HTTPS, deletes it, and proves it absent.  This
    verifier then binds the untouched run-unique output URL pair to that proof.
    """

    put_path = Path(provider_output_put_url_file).expanduser().resolve()
    get_path = Path(provider_output_get_url_file).expanduser().resolve()
    blockers: list[str] = []
    if put_path.parent != get_path.parent:
        blockers.append("single_episode_signed_output_url_files_not_co_staged")
    manifest_path = put_path.parent / SIGNED_OUTPUT_STAGING_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        blockers.append("single_episode_signed_output_staging_manifest_missing_or_unsafe")
        manifest: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        manifest = dict(loaded) if isinstance(loaded, Mapping) else {}
        if not manifest:
            blockers.append("single_episode_signed_output_staging_manifest_unreadable")

    if manifest:
        if manifest.get("schema_version") != OBJECT_STORE_STAGING_SCHEMA_VERSION:
            blockers.append("single_episode_signed_output_staging_schema_invalid")
        if manifest.get("status") != "completed" or manifest.get("blockers"):
            blockers.append("single_episode_signed_output_staging_not_completed")
        for label, path in (("put", put_path), ("get", get_path)):
            status = dict(manifest.get(f"provider_output_{label}_url_file") or {})
            try:
                recorded_path = Path(str(status.get("path") or "")).resolve()
            except (OSError, RuntimeError):
                recorded_path = Path("/")
            stat = path.stat()
            if (
                recorded_path != path
                or status.get("present") is not True
                or status.get("mode_is_0600") is not True
                or int(status.get("size_bytes") or -1) != stat.st_size
                or int(status.get("mtime_ns") or -1) != stat.st_mtime_ns
            ):
                blockers.append(f"single_episode_signed_output_{label}_url_file_not_staging_bound")

        round_trip = dict(manifest.get("signed_output_round_trip") or {})
        put_probe = dict(round_trip.get("put") or {})
        get_probe = dict(round_trip.get("get") or {})
        cleanup = dict(round_trip.get("cleanup") or {})
        if (
            round_trip.get("schema_version") != SIGNED_OUTPUT_ROUND_TRIP_SCHEMA_VERSION
            or round_trip.get("status") != "passed"
            or round_trip.get("blockers")
            or put_probe.get("status") != "passed"
            or get_probe.get("status") != "passed"
            or get_probe.get("exact_bytes_and_sha256") is not True
            or cleanup.get("status") != "passed"
            or cleanup.get("absence_confirmed") is not True
            or round_trip.get("actual_output_key_was_not_used") is not True
            or round_trip.get("raw_signed_urls_recorded") is not False
        ):
            blockers.append("single_episode_signed_output_round_trip_not_proven")
        sentinel_sha256 = str(round_trip.get("sentinel_sha256") or "").lower()
        received_sha256 = str(get_probe.get("received_sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sentinel_sha256) or received_sha256 != sentinel_sha256:
            blockers.append("single_episode_signed_output_round_trip_sha256_invalid")

        output_absence = dict(manifest.get("fresh_output_key_absence") or {})
        output_key = str(manifest.get("output_key") or "")
        if (
            output_absence.get("status") != "passed"
            or output_absence.get("absence_confirmed") is not True
            or manifest.get("output_key_run_unique") is not True
            or not re.search(
                r"/runpod_provider_runtime_output_[0-9a-f]{32}\.zip$",
                output_key,
            )
        ):
            blockers.append("single_episode_fresh_output_key_absence_not_proven")

        try:
            actual_binding = signed_output_object_binding_sha256(put_url, get_url)
        except ValueError:
            actual_binding = ""
        recorded_binding = str(manifest.get("output_url_object_binding_sha256") or "").lower()
        if (
            not re.fullmatch(r"[0-9a-f]{64}", recorded_binding)
            or actual_binding != recorded_binding
        ):
            blockers.append("single_episode_signed_output_url_object_binding_mismatch")

        expires_at = str(dict(manifest.get("presigned_url_expiry") or {}).get("expires_at") or "")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            remaining_seconds = expiry.timestamp() - (
                time.time() if now_epoch is None else float(now_epoch)
            )
        except (TypeError, ValueError, OverflowError):
            remaining_seconds = -1.0
        if remaining_seconds < SIGNED_OUTPUT_MIN_REMAINING_SECONDS:
            blockers.append("single_episode_signed_output_urls_expire_before_session_ttl")
        for label, value in (("put", put_url), ("get", get_url)):
            parsed_query = {
                key.lower(): values
                for key, values in parse_qs(
                    urlparse(value).query,
                    keep_blank_values=True,
                ).items()
            }
            date_values = parsed_query.get("x-amz-date") or []
            duration_values = parsed_query.get("x-amz-expires") or []
            try:
                signed_at = datetime.strptime(
                    str(date_values[0]),
                    "%Y%m%dT%H%M%SZ",
                ).replace(tzinfo=timezone.utc)
                exact_url_remaining = (
                    signed_at.timestamp()
                    + int(duration_values[0])
                    - (time.time() if now_epoch is None else float(now_epoch))
                )
            except (IndexError, TypeError, ValueError, OverflowError):
                exact_url_remaining = -1.0
            if exact_url_remaining < SIGNED_OUTPUT_MIN_REMAINING_SECONDS:
                blockers.append(
                    f"single_episode_signed_output_{label}_url_expiry_unverifiable_or_short"
                )
    else:
        round_trip = {}
        actual_binding = ""
        recorded_binding = ""
        expires_at = ""
        remaining_seconds = -1.0

    unique_blockers = sorted(set(blockers))
    if unique_blockers:
        raise ValueError(";".join(unique_blockers))
    return {
        "schema_version": "single_g1_kitchen_signed_output_staging_proof.v1",
        "status": "passed",
        "staging_manifest": str(manifest_path),
        "signed_output_round_trip_status": round_trip.get("status"),
        "sentinel_sha256": round_trip.get("sentinel_sha256"),
        "sentinel_cleanup_absence_confirmed": True,
        "actual_output_key_absence_confirmed": True,
        "actual_output_key_was_not_used_for_probe": True,
        "output_url_object_binding_sha256": actual_binding,
        "expires_at": expires_at,
        "remaining_seconds": int(remaining_seconds),
        "raw_signed_urls_recorded": False,
        "raw_secret_values_recorded": False,
    }


def _remote_bootstrap_script(
    inputs: Mapping[str, Any], *, provider_name: str
) -> str:
    """Embed non-secret runtime overlays in the remotely staged bootstrap."""

    bound_script = _bind_provider_allocation_identity(
        str(inputs["bootstrap_script"]), provider_name
    )
    runtime_payload = shlex.quote(str(inputs["runtime_package_overlay_xz_base64"]))
    isaac_payload = shlex.quote(str(inputs["isaac_runtime_backend_overlay_gzip_base64"]))
    return (
        "set -euo pipefail\n"
        "umask 077\n"
        # An inherited exported variable keeps its export attribute after a
        # plain assignment.  Unset both names first, then keep the very large
        # base64 values shell-local while bash's builtin printf writes them.
        # No external command is executed until the values have been unset,
        # avoiding Linux's 128 KiB MAX_ARG_STRLEN execve limit.
        f"unset {RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV} "
        f"{ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}\n"
        f"{RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV}={runtime_payload}\n"
        f"{ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}={isaac_payload}\n"
        f"builtin printf '%s' \"${{{RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV}}}\" > "
        f"{shlex.quote(RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH)}\n"
        f"builtin printf '%s' \"${{{ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}}}\" > "
        f"{shlex.quote(ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH)}\n"
        f"unset {RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV} "
        f"{ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV}\n"
        f"export {OVERLAY_PAYLOAD_TRANSPORT_ENV}={OVERLAY_PAYLOAD_FILE_TRANSPORT}\n" + bound_script
    )


def _vast_remote_bootstrap_script(inputs: Mapping[str, Any]) -> str:
    return _remote_bootstrap_script(inputs, provider_name="vast")


def _materialize_remote_bootstrap(
    root: Path, inputs: Mapping[str, Any], *, provider_name: str
) -> dict[str, Any]:
    path = root / VAST_BOOTSTRAP_ARTIFACT_NAME
    payload = _remote_bootstrap_script(inputs, provider_name=provider_name).encode("utf-8")
    path.write_bytes(payload)
    path.chmod(0o600)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "mode_is_0600": path.stat().st_mode & 0o777 == 0o600,
        "runtime_overlays_embedded": True,
        "provider_identity": provider_name,
    }


def _materialize_vast_remote_bootstrap(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    return _materialize_remote_bootstrap(root, inputs, provider_name="vast")


def _materialize_launch_session_nonce(root: Path, launch_session_id: str) -> dict[str, Any]:
    nonce = str(launch_session_id or "").strip()
    if (
        not nonce
        or not nonce.isascii()
        or any(not (character.isalnum() or character in "-_.") for character in nonce)
    ):
        raise ValueError("single_episode_launch_session_nonce_invalid")
    path = root / "launch_session_nonce.txt"
    path.write_text(nonce, encoding="utf-8")
    path.chmod(0o600)
    return {
        "path": str(path),
        "launch_session_id": nonce,
        "sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
        "mode_is_0600": path.stat().st_mode & 0o777 == 0o600,
    }


def _vast_signed_bootstrap_downloader_script() -> str:
    # Some Vast ssh-direct hosts expose port 22 before normalizing the image's
    # pre-existing /root/.ssh metadata.  sshd then rejects the correctly
    # injected account key with ``bad ownership or modes``.  Args-mode workers
    # may intentionally run as a non-root image user and do not use sshd at all,
    # so repair only when the container is already root.  In that root-only
    # branch, constrain changes to the exact key paths and reject symlinks or
    # unexpected file types instead of following them.
    return f"""set -euo pipefail
if [ "$(id -u)" -eq 0 ]; then
  if [ -L /root/.ssh ]; then
    echo vast_root_ssh_directory_symlink_forbidden >&2
    exit 72
  fi
  install -d -o root -g root -m 700 /root/.ssh
  if [ -e /root/.ssh/authorized_keys ] || [ -L /root/.ssh/authorized_keys ]; then
    if [ ! -f /root/.ssh/authorized_keys ] || [ -L /root/.ssh/authorized_keys ]; then
      echo vast_authorized_keys_not_regular_file >&2
      exit 72
    fi
    chown root:root /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
  fi
else
  echo vast_bootstrap_nonroot_ssh_repair_skipped >&2
fi
{SYSTEM_PYTHON} - <<'PY'
{VAST_BOOTSTRAP_DOWNLOADER_PYTHON.rstrip()}\nPY
"""


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            raise ValueError(f"closed_loop_option_value_missing:{option}")
        result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def _replace_repeated_option(command: list[str], option: str, values: list[str]) -> list[str]:
    result = _remove_option(command, option, takes_value=True)
    for value in values:
        result.extend([option, value])
    return result


def _remove_option(command: list[str], option: str, *, takes_value: bool) -> list[str]:
    result = list(command)
    while option in result:
        index = result.index(option)
        del result[index]
        if takes_value:
            if index >= len(result):
                raise ValueError(f"closed_loop_option_value_missing:{option}")
            del result[index]
    return result


def _pin_blueprint_command_interpreter(command: list[str]) -> list[str]:
    """Run Blueprint modules in the sealed venv that actually contains them."""
    result = list(command)
    if (
        len(result) >= 3
        and result[0] in {"python", "python3"}
        and result[1] == "-m"
        and result[2].startswith("blueprint_pipeline.")
    ):
        result[0] = OSCAR_PYTHON
    return result


def _workspace_volume_gb(provider_name: str) -> int:
    """Return storage requested by providers that honor the generic volume field."""

    if provider_name == "runpod":
        return RUNPOD_WORKSPACE_VOLUME_GB
    if provider_name == "vast":
        return VAST_IGNORED_LEGACY_VOLUME_GB
    raise ValueError(f"single_episode_provider_unsupported:{provider_name}")


def _runtime_pythonpath(value: Any) -> str:
    """Prepend the exact package overlay, then preserve sealed runtime roots."""
    ordered: list[str] = []
    for item in (
        *REQUIRED_RUNTIME_PYTHONPATHS,
        *str(value or "").split(":"),
    ):
        path = item.strip()
        if path and path not in ordered:
            ordered.append(path)
    return ":".join(ordered)


def _oscar_runtime_pythonpath(value: Any) -> str:
    """Add repaired OSCAR dependencies without contaminating sibling runtimes."""

    return ":".join(
        item
        for item in (
            OSCAR_RUNTIME_DEPENDENCY_TARGET,
            str(value or "").strip(),
        )
        if item
    )


def _scoped_oscar_python_command() -> str:
    """Return an OSCAR interpreter invocation with a command-local PYTHONPATH."""

    return (
        f"PYTHONPATH={shlex.quote(OSCAR_RUNTIME_DEPENDENCY_TARGET)}:"
        '"${PYTHONPATH:-}" '
        f"{shlex.quote(OSCAR_PYTHON)}"
    )


def _scope_command_pythonpath(command: list[str], pythonpath: str) -> list[str]:
    """Bind one child process to an explicit dependency namespace."""

    if not command:
        raise ValueError("single_episode_runtime_command_missing")
    if not pythonpath.strip():
        raise ValueError("single_episode_runtime_pythonpath_missing")
    return ["env", f"PYTHONPATH={pythonpath}", *command]


def _scenario_perception_target_prompts(
    scenario: Mapping[str, Any],
) -> list[str]:
    """Return deterministic microwave SAM3 prompts plus scenario-bound aliases."""
    candidates: list[Any] = ["microwave", "microwave handle", "microwave door"]
    for field in (
        "perception_target_prompts",
        "target_object_ids",
        "affordance_object_ids",
    ):
        value = scenario.get(field)
        if isinstance(value, list):
            candidates.extend(value)
    stance = scenario.get("accepted_stance_contract")
    if isinstance(stance, Mapping):
        for field in ("resolved_target", "resolved_affordance"):
            target = stance.get(field)
            if not isinstance(target, Mapping):
                continue
            for name in ("target_object_id", "object_id", "label", "name"):
                if target.get(name) is not None:
                    candidates.append(target[name])

    prompts: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        prompt = str(candidate or "").strip()
        key = prompt.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
    return prompts


def _runtime_package_overlay_script() -> tuple[str, str, str, dict[str, str]]:
    """Build one hash-pinned in-memory overlay for the direct episode modules."""
    source_root = Path(__file__).resolve().parent
    modules: dict[str, dict[str, str]] = {}
    source_sha256s: dict[str, str] = {}
    for filename in RUNTIME_PACKAGE_OVERLAY_MODULES:
        source_path = source_root / filename
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"single_episode_runtime_overlay_source_missing_or_unsafe:{filename}")
        source = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source).hexdigest()
        source_sha256s[filename] = source_sha256
        modules[filename] = {
            "source_base64": base64.b64encode(source).decode("ascii"),
            "source_sha256": source_sha256,
        }
    archive = json.dumps(
        {
            "schema_version": "single_g1_kitchen_runtime_package_overlay.v1",
            "modules": modules,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    encoded_archive = base64.b64encode(
        lzma.compress(archive, format=lzma.FORMAT_XZ, preset=9)
    ).decode("ascii")
    expected_filenames = list(RUNTIME_PACKAGE_OVERLAY_MODULES)
    sitecustomize_source = """from __future__ import annotations

from pathlib import Path

import blueprint_pipeline

OVERLAY_PACKAGE = str(
    (Path(__file__).resolve().parent / "blueprint_pipeline").resolve()
)
if OVERLAY_PACKAGE not in blueprint_pipeline.__path__:
    blueprint_pipeline.__path__.insert(0, OVERLAY_PACKAGE)
"""
    encoded_sitecustomize = base64.b64encode(sitecustomize_source.encode("utf-8")).decode("ascii")
    script = f"""PYTHONPATH=/opt/wbc:/opt/OSCAR /opt/oscar-venv/bin/python - <<'PY'
import base64
import hashlib
import importlib
import importlib.util
import json
import lzma
import os
import shutil
import sys
from pathlib import Path

root = Path({RUNTIME_PACKAGE_OVERLAY_DIR!r})
if root.exists():
    shutil.rmtree(root)
package = root / "blueprint_pipeline"
package.mkdir(parents=True, exist_ok=True)
payload_transport = os.environ.get({OVERLAY_PAYLOAD_TRANSPORT_ENV!r}, "").strip()
payload_path = Path({RUNTIME_PACKAGE_OVERLAY_PAYLOAD_PATH!r})
if payload_transport == {OVERLAY_PAYLOAD_FILE_TRANSPORT!r}:
    if payload_path.is_symlink() or not payload_path.is_file():
        raise RuntimeError("single_episode_runtime_overlay_payload_file_missing_or_unsafe")
    payload = payload_path.read_text(encoding="ascii").strip()
elif payload_transport:
    raise RuntimeError("single_episode_runtime_overlay_payload_transport_invalid")
else:
    # Legacy non-Vast providers still inject this non-secret payload directly.
    # The archive remains pinned to both the plan digest and embedded digest.
    payload = os.environ.get({RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV!r}, "").strip()
expected_archive_sha256 = os.environ.get(
    {RUNTIME_PACKAGE_OVERLAY_SHA256_ENV!r}, ""
).strip().lower()
if not payload or len(expected_archive_sha256) != 64:
    raise RuntimeError("single_episode_runtime_overlay_binding_missing")
archive = lzma.decompress(base64.b64decode(payload, validate=True))
observed_archive_sha256 = hashlib.sha256(archive).hexdigest()
if observed_archive_sha256 != expected_archive_sha256:
    raise RuntimeError("single_episode_runtime_overlay_sha256_mismatch")
if expected_archive_sha256 != {archive_sha256!r}:
    raise RuntimeError("single_episode_runtime_overlay_plan_digest_mismatch")
decoded = json.loads(archive)
if decoded.get("schema_version") != "single_g1_kitchen_runtime_package_overlay.v1":
    raise RuntimeError("single_episode_runtime_overlay_schema_mismatch")
module_rows = decoded.get("modules")
expected_filenames = {expected_filenames!r}
if not isinstance(module_rows, dict) or list(module_rows) != sorted(expected_filenames):
    raise RuntimeError("single_episode_runtime_overlay_module_set_mismatch")
source_sha256s = {{}}
for filename in expected_filenames:
    row = module_rows.get(filename)
    if not isinstance(row, dict):
        raise RuntimeError("single_episode_runtime_overlay_module_record_invalid")
    source = base64.b64decode(str(row.get("source_base64") or ""), validate=True)
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != str(row.get("source_sha256") or ""):
        raise RuntimeError("single_episode_runtime_overlay_source_sha256_mismatch")
    target = package / filename
    target.write_bytes(source)
    if hashlib.sha256(target.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("single_episode_runtime_overlay_materialized_sha256_mismatch")
    source_sha256s[filename] = source_sha256

sitecustomize = base64.b64decode({encoded_sitecustomize!r}, validate=True)
sitecustomize_path = root / "sitecustomize.py"
sitecustomize_path.write_bytes(sitecustomize)
sys.path.insert(0, str(root))
sitecustomize_spec = importlib.util.spec_from_file_location(
    "_blueprint_runtime_overlay_sitecustomize", sitecustomize_path
)
if sitecustomize_spec is None or sitecustomize_spec.loader is None:
    raise RuntimeError("single_episode_runtime_overlay_sitecustomize_spec_missing")
runtime_overlay_sitecustomize = importlib.util.module_from_spec(sitecustomize_spec)
sitecustomize_spec.loader.exec_module(runtime_overlay_sitecustomize)

imports = {{}}
for filename in expected_filenames:
    module_name = "blueprint_pipeline." + filename.removesuffix(".py")
    module = importlib.import_module(module_name)
    observed_path = Path(str(module.__file__)).resolve()
    expected_path = (package / filename).resolve()
    if observed_path != expected_path:
        raise RuntimeError("single_episode_runtime_overlay_import_path_mismatch")
    imports[module_name] = str(observed_path)

manifest = {{
    "schema_version": "single_g1_kitchen_runtime_package_overlay.v1",
    "status": "materialized_hash_verified_and_imported",
    "overlay_root": str(root),
    "archive_sha256": observed_archive_sha256,
    "source_sha256s": source_sha256s,
    "imports": imports,
    "installed_package_parent_preserved": True,
    "raw_secret_values_recorded": False,
}}
manifest_path = Path("/workspace/closed_loop_out/runtime_package_overlay.json")
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
PY
"""
    return script, archive_sha256, encoded_archive, source_sha256s


def _pin_isaac_executor_to_runtime_overlay(command: list[str]) -> list[str]:
    """Run the executor overlay and make it own both live camera artifacts."""
    expected = [ISAAC_PYTHON, "-m", ISAAC_TASK_EXECUTOR_MODULE]
    if command[: len(expected)] != expected:
        raise ValueError("single_episode_isaac_task_executor_command_unexpected")
    pinned = [
        ISAAC_PYTHON,
        ISAAC_RUNTIME_OVERLAY_WRAPPER,
        *command[len(expected) :],
    ]
    pinned = _replace_option(
        pinned,
        "--initial-frame-output",
        INITIAL_POLICY_FRAME_PATH,
    )
    return _replace_option(
        pinned,
        "--camera-projection-context-output",
        CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH,
    )


def _runtime_backend_overlay_script(*, backend_source: bytes) -> tuple[str, str, str]:
    """Materialize and verify the exact task backend before Isaac starts.

    The sealed image is intentionally immutable and predates the regular-joint
    microwave measurement fix.  This direct one-episode launcher therefore
    overlays only that module, under its canonical import name, while leaving
    the installed package and every model/runtime byte unchanged.
    """
    expected_sha256 = hashlib.sha256(backend_source).hexdigest()
    encoded_source = base64.b64encode(
        gzip.compress(backend_source, compresslevel=9, mtime=0)
    ).decode("ascii")
    wrapper = f"""from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path

MODULE_NAME = {ISAAC_RUNTIME_BACKEND_MODULE!r}
SOURCE_PATH = Path({ISAAC_RUNTIME_OVERLAY_SOURCE!r})
EXPECTED_SHA256 = {expected_sha256!r}
if os.environ.get("BLUEPRINT_ISAAC_RUNTIME_BACKEND_OVERLAY_SHA256") != EXPECTED_SHA256:
    raise SystemExit("isaac_runtime_backend_overlay_plan_digest_mismatch")
observed_sha256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
if observed_sha256 != EXPECTED_SHA256:
    raise SystemExit("isaac_runtime_backend_overlay_sha256_mismatch")

# Keep the installed Blueprint package as the parent so all relative imports
# resolve normally; replace only the backend module before the service lazily
# imports create_backend().
import blueprint_pipeline  # noqa: F401,E402

service_spec = importlib.util.find_spec({ISAAC_TASK_EXECUTOR_MODULE!r})
expected_service_path = (
    Path({RUNTIME_PACKAGE_OVERLAY_DIR!r})
    / "blueprint_pipeline"
    / "isaac_persistent_task_executor_service.py"
).resolve()
if (
    service_spec is None
    or service_spec.origin is None
    or Path(service_spec.origin).resolve() != expected_service_path
):
    raise SystemExit("isaac_task_executor_service_overlay_path_mismatch")

spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE_PATH)
if spec is None or spec.loader is None:
    raise SystemExit("isaac_runtime_backend_overlay_spec_unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)
runpy.run_module({ISAAC_TASK_EXECUTOR_MODULE!r}, run_name="__main__", alter_sys=True)
"""
    wrapper_bytes = wrapper.encode("utf-8")
    encoded_wrapper = base64.b64encode(wrapper_bytes).decode("ascii")
    wrapper_sha256 = hashlib.sha256(wrapper_bytes).hexdigest()
    script = (
        "python3 - <<'PY'\n"
        "import base64, gzip, hashlib, json, os\n"
        "from pathlib import Path\n\n"
        f"root = Path({ISAAC_RUNTIME_OVERLAY_DIR!r})\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        f"payload_transport = os.environ.get({OVERLAY_PAYLOAD_TRANSPORT_ENV!r}, '').strip()\n"
        f"payload_path = Path({ISAAC_RUNTIME_OVERLAY_PAYLOAD_PATH!r})\n"
        f"if payload_transport == {OVERLAY_PAYLOAD_FILE_TRANSPORT!r}:\n"
        "    if payload_path.is_symlink() or not payload_path.is_file():\n"
        "        raise RuntimeError('isaac_runtime_backend_overlay_payload_file_missing_or_unsafe')\n"
        "    payload = payload_path.read_text(encoding='ascii').strip()\n"
        "elif payload_transport:\n"
        "    raise RuntimeError('isaac_runtime_backend_overlay_payload_transport_invalid')\n"
        "else:\n"
        f"    payload = os.environ.get({ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV!r}, '').strip()\n"
        "if not payload:\n"
        "    raise RuntimeError('isaac_runtime_backend_overlay_payload_missing')\n"
        "source = gzip.decompress(base64.b64decode(payload, validate=True))\n"
        f"expected_source_sha = {expected_sha256!r}\n"
        "if hashlib.sha256(source).hexdigest() != expected_source_sha:\n"
        "    raise RuntimeError('isaac_runtime_backend_overlay_embedded_sha256_mismatch')\n"
        f"source_path = Path({ISAAC_RUNTIME_OVERLAY_SOURCE!r})\n"
        "source_path.write_bytes(source)\n"
        f"wrapper = base64.b64decode({encoded_wrapper!r}, validate=True)\n"
        f"expected_wrapper_sha = {wrapper_sha256!r}\n"
        "if hashlib.sha256(wrapper).hexdigest() != expected_wrapper_sha:\n"
        "    raise RuntimeError('isaac_runtime_backend_wrapper_embedded_sha256_mismatch')\n"
        f"wrapper_path = Path({ISAAC_RUNTIME_OVERLAY_WRAPPER!r})\n"
        "wrapper_path.write_bytes(wrapper)\n"
        "manifest = {\n"
        "    'schema_version': 'isaac_runtime_backend_overlay.v1',\n"
        "    'status': 'materialized_and_hash_verified',\n"
        f"    'canonical_module_name': {ISAAC_RUNTIME_BACKEND_MODULE!r},\n"
        "    'source_path': str(source_path),\n"
        "    'source_sha256': expected_source_sha,\n"
        "    'wrapper_path': str(wrapper_path),\n"
        "    'wrapper_sha256': expected_wrapper_sha,\n"
        "    'installed_package_parent_preserved': True,\n"
        "    'scene_physics_modified': False,\n"
        "    'raw_secret_values_recorded': False,\n"
        "}\n"
        "manifest_path = Path('/workspace/closed_loop_out/isaac_runtime_backend_overlay.json')\n"
        "manifest_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "PY\n"
    )
    return script, expected_sha256, encoded_source


def _oscar_runtime_provenance_script() -> str:
    """Verify the builder-sealed source tree before OSCAR execution.

    The thin foundation intentionally excludes Git and ``.git`` metadata.  Its
    build stage instead seals the reviewed origin/commit together with the
    post-patch runtime tree.  Recompute that tree here and compare it with the
    independently reviewed digest embedded in the verifier before exporting
    evaluator provenance. Mutable launch environment is not an authority.
    """
    return f"""OSCAR_RUNTIME_SOURCE_ROOT=/opt/OSCAR
OSCAR_RUNTIME_SOURCE_URL={OFFICIAL_OSCAR_SOURCE_URL!r}
OSCAR_RUNTIME_SOURCE_REF={OFFICIAL_OSCAR_SOURCE_COMMIT!r}
export OSCAR_RUNTIME_SOURCE_ROOT OSCAR_RUNTIME_SOURCE_URL OSCAR_RUNTIME_SOURCE_REF
set +e
/opt/oscar-venv/bin/python -m blueprint_pipeline.oscar_runtime_source_provenance verify \
  --source-root /opt/OSCAR \
  --seal /opt/blueprint/oscar_source_provenance.json \
  --artifact {OSCAR_RUNTIME_PROVENANCE_ARTIFACT!r}
OSCAR_RUNTIME_PROVENANCE_RC=$?
set -e
if [ "$OSCAR_RUNTIME_PROVENANCE_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$OSCAR_RUNTIME_PROVENANCE_RC" \
    BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_provenance_mismatch" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$OSCAR_RUNTIME_PROVENANCE_RC"
fi
export BLUEPRINT_OSCAR_WAM_SOURCE_URL="$OSCAR_RUNTIME_SOURCE_URL"
export BLUEPRINT_OSCAR_WAM_SOURCE_REF="$OSCAR_RUNTIME_SOURCE_REF"
"""


def _oscar_runtime_dependency_repair_script() -> str:
    """Bind the old sealed image to the reviewed OSCAR dependency closure.

    The image named by :data:`IMAGE_REF` filtered a nonexistent
    ``requirements.txt``.  The reviewed OSCAR commit names
    ``requirements_minimal.txt`` instead, and the historical filter silently
    emitted an empty file.  Carry the same hash-locked closure used by the
    replacement Foundation image so one bounded install repairs every missing
    OSCAR dependency before the real CLI import is attempted.
    """

    lock_path = (
        Path(__file__).resolve().parents[2] / OSCAR_FOUNDATION_REQUIREMENTS_LOCK_RELATIVE_PATH
    )
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("single_episode_oscar_foundation_requirements_lock_missing_or_unsafe")
    lock_bytes = lock_path.read_bytes()
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    if lock_sha256 != OSCAR_FOUNDATION_REQUIREMENTS_LOCK_SHA256:
        raise ValueError("single_episode_oscar_foundation_requirements_lock_digest_mismatch")
    compressed_payload = base64.b64encode(lzma.compress(lock_bytes, preset=9)).decode("ascii")

    return f"""upload_phase oscar_runtime_dependency_install_started
set +e
{_scoped_oscar_python_command()} - <<'PY'
import base64
import hashlib
import importlib.metadata
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

oscar_python = {OSCAR_PYTHON!r}
lock_path = Path({OSCAR_RUNTIME_DEPENDENCY_LOCK_PATH!r})
dependency_target = Path({OSCAR_RUNTIME_DEPENDENCY_TARGET!r})
base_site_packages = Path("/opt/oscar-venv/lib/python3.10/site-packages")
artifact_path = Path({OSCAR_RUNTIME_DEPENDENCY_ARTIFACT!r})
expected_lock_sha256 = {OSCAR_FOUNDATION_REQUIREMENTS_LOCK_SHA256!r}
required_pytest_version = {OSCAR_RUNTIME_REQUIRED_PYTEST_VERSION!r}
compressed_lock = {compressed_payload!r}
blockers = []
checks = {{}}

try:
    lock_bytes = lzma.decompress(base64.b64decode(compressed_lock, validate=True))
except Exception as exc:
    lock_bytes = b""
    blockers.append(f"oscar_dependency_lock_decode_failed:{{type(exc).__name__}}")
lock_sha256 = hashlib.sha256(lock_bytes).hexdigest() if lock_bytes else None
checks["foundation_lock_sha256_exact"] = lock_sha256 == expected_lock_sha256
if not checks["foundation_lock_sha256_exact"]:
    blockers.append("oscar_dependency_lock_digest_mismatch")

if lock_bytes:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(lock_bytes)

checks["base_site_packages_present"] = base_site_packages.is_dir()
checks["dependency_target_under_workspace"] = dependency_target.parent == Path("/workspace")
if not checks["base_site_packages_present"]:
    blockers.append("oscar_base_site_packages_missing")
if not checks["dependency_target_under_workspace"]:
    blockers.append("oscar_dependency_target_escaped_workspace")
if dependency_target.is_symlink():
    blockers.append("oscar_dependency_target_is_symlink")
elif dependency_target.exists() and not dependency_target.is_dir():
    blockers.append("oscar_dependency_target_not_directory")
elif checks["dependency_target_under_workspace"]:
    if dependency_target.exists():
        shutil.rmtree(dependency_target)
    dependency_target.mkdir(parents=True, exist_ok=False)

def canonical_name(value):
    return re.sub(r"[-_.]+", "-", str(value).strip()).lower()

def parse_locked_versions(payload):
    versions = {{}}
    for raw_line in payload.decode("utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if "==" not in line or not line.endswith("\\\\"):
            continue
        name, version = line[:-1].split("==", 1)
        normalized = canonical_name(name)
        version = version.strip()
        if not normalized or not version or normalized in versions:
            raise ValueError("invalid_or_duplicate_locked_requirement")
        versions[normalized] = version
    return versions

try:
    expected_versions = parse_locked_versions(lock_bytes) if lock_bytes else {{}}
except Exception as exc:
    expected_versions = {{}}
    blockers.append(f"oscar_dependency_lock_parse_failed:{{type(exc).__name__}}")
checks["foundation_lock_package_count"] = len(expected_versions)
checks["foundation_lock_package_count_complete"] = (
    len(expected_versions) == {OSCAR_FOUNDATION_REQUIREMENTS_LOCK_PACKAGE_COUNT}
)
checks["einops_locked_to_reviewed_version"] = expected_versions.get("einops") == "0.8.1"
checks["torch_locked_to_cuda_128_version"] = expected_versions.get("torch") == "2.10.0+cu128"
checks["pytest_locked_to_reviewed_version"] = (
    expected_versions.get("pytest") == required_pytest_version
)
for check_name in (
    "foundation_lock_package_count_complete",
    "einops_locked_to_reviewed_version",
    "torch_locked_to_cuda_128_version",
    "pytest_locked_to_reviewed_version",
):
    if not checks[check_name]:
        blockers.append(f"oscar_dependency_lock_contract_failed:{{check_name}}")

seeded_base_distributions = []
if not blockers:
    for metadata_dir in base_site_packages.glob("*.dist-info"):
        if not metadata_dir.is_dir() or metadata_dir.is_symlink():
            continue
        distribution = importlib.metadata.Distribution.at(metadata_dir)
        name = canonical_name(distribution.metadata.get("Name") or "")
        if expected_versions.get(name) != distribution.version:
            continue
        shutil.copytree(metadata_dir, dependency_target / metadata_dir.name)
        seeded_base_distributions.append(
            {{"name": name, "version": distribution.version}}
        )

def installed_distributions():
    return list(importlib.metadata.distributions(path=[str(dependency_target)]))

def installed_versions():
    versions = {{}}
    for distribution in installed_distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[canonical_name(name)] = distribution.version
    return versions

def mismatches(installed):
    return [
        {{"name": name, "expected": expected, "installed": installed.get(name)}}
        for name, expected in sorted(expected_versions.items())
        if installed.get(name) != expected
    ]

installed_before = installed_versions()
mismatches_before = mismatches(installed_before)
uv_candidates = [
    Path("/opt/oscar-venv/bin/uv"),
    Path("/usr/local/bin/uv"),
]
located_uv = shutil.which("uv")
if located_uv:
    uv_candidates.append(Path(located_uv))
uv_path = next((candidate for candidate in uv_candidates if candidate.is_file()), None)
checks["uv_installer_present"] = uv_path is not None
checks["uv_installer_required"] = bool(mismatches_before)
if checks["uv_installer_required"] and uv_path is None:
    blockers.append("oscar_dependency_uv_installer_missing")

install_attempted = bool(mismatches_before) and not blockers
install_returncode = None
install_stdout_tail = ""
install_stderr_tail = ""
if install_attempted:
    try:
        completed = subprocess.run(
            [
                str(uv_path),
                "pip",
                "install",
                "--python",
                oscar_python,
                "--target",
                str(dependency_target),
                "--require-hashes",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
                "--extra-index-url",
                "https://pypi.org/simple",
                "--index-strategy",
                "unsafe-best-match",
                "-r",
                str(lock_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1200,
        )
        install_returncode = completed.returncode
        install_stdout_tail = (completed.stdout or "")[-4000:]
        install_stderr_tail = (completed.stderr or "")[-6000:]
        if completed.returncode != 0:
            blockers.append(f"oscar_dependency_install_returncode:{{completed.returncode}}")
    except Exception as exc:
        blockers.append(f"oscar_dependency_install_failed:{{type(exc).__name__}}")

transformer_engine_metadata_dir = dependency_target / "transformer_engine-2.0.0.dist-info"
transformer_engine_metadata_files = {{
    "METADATA": (
        "Metadata-Version: 2.1\\n"
        "Name: transformer-engine\\n"
        "Version: 2.0.0\\n"
        "Summary: Blueprint OSCAR PyTorch SDPA compatibility shim metadata\\n"
    ),
    "WHEEL": (
        "Wheel-Version: 1.0\\n"
        "Generator: blueprint-oscar-wam-image\\n"
        "Root-Is-Purelib: true\\n"
        "Tag: py3-none-any\\n"
    ),
    "top_level.txt": "transformer_engine\\n",
}}
transformer_engine_metadata_expected_sha256s = {{
    "METADATA": "1094a00f29f0c1cdb6da87ae97d126098c129dc03ee293f22e94d6d2966772e3",
    "WHEEL": "b7ff9a708a45ce5df8c8013303ae24c116ce1a1e0bb8f395e3ae1a8e2d3b000e",
    "top_level.txt": "896975903ac1a386f9bb60f27e53e0a6e822923c966d9fbf077aeb6d24683274",
}}
transformer_engine_module_path = None
transformer_engine_metadata_sha256s = {{}}
transformer_engine_distribution_version = None
transformer_engine_distribution_metadata_path = None
checks["transformer_engine_dependency_target_on_pythonpath"] = str(dependency_target) in sys.path
if not checks["transformer_engine_dependency_target_on_pythonpath"]:
    blockers.append("oscar_transformer_engine_dependency_target_not_on_pythonpath")
if not blockers:
    try:
        import transformer_engine as transformer_engine_module
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        transformer_engine_module_path = str(Path(transformer_engine_module.__file__).resolve())
        trusted_transformer_engine_root = Path("/opt/OSCAR/transformer_engine").resolve()
        checks["transformer_engine_compat_shim_marker_verified"] = (
            getattr(transformer_engine_module, "BLUEPRINT_COMPAT_SHIM", False) is True
        )
        checks["transformer_engine_shim_source_root_verified"] = Path(
            transformer_engine_module_path
        ).is_relative_to(trusted_transformer_engine_root)
        checks["transformer_engine_attention_api_verified"] = callable(apply_rotary_pos_emb)
        checks["transformer_engine_package_code_not_overlaid"] = not (
            dependency_target / "transformer_engine"
        ).exists()
        for check_name in (
            "transformer_engine_compat_shim_marker_verified",
            "transformer_engine_shim_source_root_verified",
            "transformer_engine_attention_api_verified",
            "transformer_engine_package_code_not_overlaid",
        ):
            if not checks[check_name]:
                blockers.append(f"oscar_transformer_engine_shim_contract_failed:{{check_name}}")
    except Exception as exc:
        blockers.append(f"oscar_transformer_engine_shim_verification_failed:{{type(exc).__name__}}")

if not blockers:
    transformer_engine_metadata_dir.mkdir(parents=False, exist_ok=False)
    for filename, text in transformer_engine_metadata_files.items():
        metadata_path = transformer_engine_metadata_dir / filename
        metadata_path.write_text(text, encoding="utf-8")
        transformer_engine_metadata_sha256s[filename] = hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest()
    checks["transformer_engine_metadata_files_exact"] = (
        transformer_engine_metadata_sha256s
        == transformer_engine_metadata_expected_sha256s
    )
    if not checks["transformer_engine_metadata_files_exact"]:
        blockers.append("oscar_transformer_engine_metadata_digest_mismatch")
    try:
        transformer_engine_distribution = importlib.metadata.distribution(
            "transformer-engine"
        )
        transformer_engine_distribution_version = importlib.metadata.version(
            "transformer-engine"
        )
        transformer_engine_distribution_metadata_path = str(
            Path(transformer_engine_distribution._path).resolve()
        )
    except importlib.metadata.PackageNotFoundError:
        transformer_engine_distribution_version = None
        transformer_engine_distribution_metadata_path = None
    checks["transformer_engine_distribution_version_exact"] = (
        transformer_engine_distribution_version == "2.0.0"
        and Version(transformer_engine_distribution_version) >= Version("1.13.0")
    )
    checks["transformer_engine_distribution_metadata_path_exact"] = (
        transformer_engine_distribution_metadata_path
        == str(transformer_engine_metadata_dir.resolve())
    )
    if not checks["transformer_engine_distribution_version_exact"]:
        blockers.append("oscar_transformer_engine_distribution_metadata_invalid")
    if not checks["transformer_engine_distribution_metadata_path_exact"]:
        blockers.append("oscar_transformer_engine_distribution_metadata_path_invalid")

installed_after = installed_versions()
mismatches_after = mismatches(installed_after)
checks["locked_dependency_closure_exact"] = not mismatches_after and bool(expected_versions)
checks["einops_runtime_version_exact"] = installed_after.get("einops") == "0.8.1"
if not checks["locked_dependency_closure_exact"]:
    blockers.append("oscar_locked_dependency_closure_mismatch")
if not checks["einops_runtime_version_exact"]:
    blockers.append("oscar_einops_runtime_version_mismatch")

pytest_distribution_version = None
pytest_module_version = None
pytest_module_path = None
importlib.invalidate_caches()
sys.path_importer_cache.pop(str(dependency_target), None)
try:
    import pytest as pytest_module

    pytest_distribution_version = importlib.metadata.version("pytest")
    pytest_module_version = str(getattr(pytest_module, "__version__", ""))
    pytest_module_path = str(Path(pytest_module.__file__).resolve())
except Exception as exc:
    blockers.append(f"oscar_pytest_runtime_import_failed:{{type(exc).__name__}}")
checks["pytest_runtime_importable"] = bool(
    pytest_module_path and Path(pytest_module_path).is_file()
)
checks["pytest_runtime_distribution_version_exact"] = (
    pytest_distribution_version == required_pytest_version
)
checks["pytest_runtime_module_version_exact"] = pytest_module_version == required_pytest_version
checks["pytest_runtime_module_path_trusted"] = bool(
    pytest_module_path
    and (
        Path(pytest_module_path).is_relative_to(dependency_target.resolve())
        or Path(pytest_module_path).is_relative_to(base_site_packages.resolve())
    )
)
for check_name in (
    "pytest_runtime_importable",
    "pytest_runtime_distribution_version_exact",
    "pytest_runtime_module_version_exact",
    "pytest_runtime_module_path_trusted",
):
    if not checks[check_name]:
        blockers.append(f"oscar_pytest_runtime_contract_failed:{{check_name}}")

pytest_fresh_subprocess_returncode = None
pytest_fresh_subprocess_stdout_tail = ""
pytest_fresh_subprocess_stderr_tail = ""
pytest_fresh_distribution_version = None
pytest_fresh_module_version = None
pytest_fresh_module_path = None
pytest_fresh_probe_source = (
    "import importlib.metadata, json\\n"
    "from pathlib import Path\\n"
    "import pytest\\n"
    "print(json.dumps(dict("
    "distribution_version=importlib.metadata.version('pytest'), "
    "module_version=str(getattr(pytest, '__version__', '')), "
    "module_path=str(Path(pytest.__file__).resolve())"
    "), sort_keys=True))\\n"
)
pytest_fresh_env = os.environ.copy()
pytest_fresh_env["PYTHONPATH"] = os.pathsep.join(
    value
    for value in (
        str(dependency_target),
        pytest_fresh_env.get("PYTHONPATH", "").strip(),
    )
    if value
)
try:
    pytest_fresh_completed = subprocess.run(
        [oscar_python, "-c", pytest_fresh_probe_source],
        cwd="/opt/OSCAR",
        env=pytest_fresh_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    pytest_fresh_subprocess_returncode = pytest_fresh_completed.returncode
    pytest_fresh_subprocess_stdout_tail = (pytest_fresh_completed.stdout or "")[-2000:]
    pytest_fresh_subprocess_stderr_tail = (pytest_fresh_completed.stderr or "")[-4000:]
    if pytest_fresh_completed.returncode == 0:
        pytest_fresh_lines = [
            line for line in pytest_fresh_completed.stdout.splitlines() if line.strip()
        ]
        pytest_fresh_payload = json.loads(pytest_fresh_lines[-1])
        pytest_fresh_distribution_version = pytest_fresh_payload.get(
            "distribution_version"
        )
        pytest_fresh_module_version = pytest_fresh_payload.get("module_version")
        pytest_fresh_module_path = pytest_fresh_payload.get("module_path")
except Exception as exc:
    blockers.append(f"oscar_pytest_fresh_subprocess_failed:{{type(exc).__name__}}")
checks["pytest_fresh_subprocess_returncode_zero"] = (
    pytest_fresh_subprocess_returncode == 0
)
checks["pytest_fresh_distribution_version_exact"] = (
    pytest_fresh_distribution_version == required_pytest_version
)
checks["pytest_fresh_module_version_exact"] = (
    pytest_fresh_module_version == required_pytest_version
)
checks["pytest_fresh_module_path_in_dependency_target"] = bool(
    pytest_fresh_module_path
    and Path(pytest_fresh_module_path).is_file()
    and Path(pytest_fresh_module_path).is_relative_to(dependency_target.resolve())
)
checks["pytest_fresh_module_path_trusted"] = bool(
    pytest_fresh_module_path
    and Path(pytest_fresh_module_path).is_file()
    and (
        Path(pytest_fresh_module_path).is_relative_to(dependency_target.resolve())
        or Path(pytest_fresh_module_path).is_relative_to(base_site_packages.resolve())
    )
)
for check_name in (
    "pytest_fresh_subprocess_returncode_zero",
    "pytest_fresh_distribution_version_exact",
    "pytest_fresh_module_version_exact",
    "pytest_fresh_module_path_trusted",
):
    if not checks[check_name]:
        blockers.append(f"oscar_pytest_fresh_subprocess_contract_failed:{{check_name}}")

# The repaired OSCAR closure must never shadow GR00T's separately sealed
# interpreter. Attempt 025 proved that a global target PYTHONPATH can replace
# GR00T's compatible accelerate/transformers pair and trigger a circular import.
# Exercise the exact GR00T import sequence in a fresh, explicitly clean process
# before downloading the large OSCAR auxiliary assets.
groot_clean_import_returncode = None
groot_clean_import_stdout_tail = ""
groot_clean_import_stderr_tail = ""
groot_clean_import_module_paths = {{}}
groot_clean_env = os.environ.copy()
groot_clean_env["PYTHONPATH"] = os.pathsep.join(
    value
    for value in groot_clean_env.get("PYTHONPATH", "").split(os.pathsep)
    if value and value != str(dependency_target)
)
groot_clean_probe_source = (
    "import json\\n"
    "from pathlib import Path\\n"
    "import gr00t.model\\n"
    "from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls\\n"
    "import transformers\\n"
    "from transformers import AutoConfig, AutoProcessor, Qwen3VLProcessor\\n"
    "import accelerate\\n"
    "assert callable(get_backbone_cls)\\n"
    "assert all(value is not None for value in (AutoConfig, AutoProcessor, Qwen3VLProcessor))\\n"
    "print(json.dumps(dict("
    "accelerate=str(Path(accelerate.__file__).resolve()), "
    "gr00t_model=str(Path(gr00t.model.__file__).resolve()), "
    "transformers=str(Path(transformers.__file__).resolve())"
    "), sort_keys=True))\\n"
)
try:
    groot_clean_completed = subprocess.run(
        [os.environ.get("BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON", "/opt/gr00t-venv/bin/python"), "-c", groot_clean_probe_source],
        cwd="/opt/gr00t",
        env=groot_clean_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    groot_clean_import_returncode = groot_clean_completed.returncode
    groot_clean_import_stdout_tail = (groot_clean_completed.stdout or "")[-2000:]
    groot_clean_import_stderr_tail = (groot_clean_completed.stderr or "")[-4000:]
    if groot_clean_completed.returncode == 0:
        groot_clean_lines = [
            line for line in groot_clean_completed.stdout.splitlines() if line.strip()
        ]
        groot_clean_import_module_paths = json.loads(groot_clean_lines[-1])
except Exception as exc:
    blockers.append(f"groot_clean_import_probe_failed:{{type(exc).__name__}}")
checks["groot_clean_pythonpath_excludes_oscar_dependency_target"] = (
    str(dependency_target) not in groot_clean_env.get("PYTHONPATH", "").split(os.pathsep)
)
checks["groot_clean_import_probe_returncode_zero"] = groot_clean_import_returncode == 0
checks["groot_clean_import_modules_outside_oscar_dependency_target"] = bool(
    groot_clean_import_module_paths
    and all(
        Path(module_path).is_file()
        and not Path(module_path).is_relative_to(dependency_target.resolve())
        for module_path in groot_clean_import_module_paths.values()
    )
)
checks["groot_clean_accelerate_resolved_from_groot_venv"] = bool(
    groot_clean_import_module_paths.get("accelerate")
    and Path(groot_clean_import_module_paths["accelerate"]).is_relative_to(
        Path(os.environ["BLUEPRINT_GROOT_VENV_ROOT"]).resolve()
    )
)
for check_name in (
    "groot_clean_pythonpath_excludes_oscar_dependency_target",
    "groot_clean_import_probe_returncode_zero",
    "groot_clean_import_modules_outside_oscar_dependency_target",
    "groot_clean_accelerate_resolved_from_groot_venv",
):
    if not checks[check_name]:
        blockers.append(f"groot_runtime_dependency_isolation_failed:{{check_name}}")

dependency_check_returncode = None
dependency_check_stdout_tail = ""
dependency_check_stderr_tail = ""
dependency_violations = []
if not mismatches_after:
    environment = default_environment()
    for distribution in installed_distributions():
        requiring_name = distribution.metadata.get("Name") or "unknown"
        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                dependency_violations.append(
                    {{"requiring": requiring_name, "requirement": raw_requirement, "reason": "invalid"}}
                )
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            installed_version = installed_after.get(canonical_name(requirement.name))
            if installed_version is None:
                dependency_violations.append(
                    {{"requiring": requiring_name, "requirement": raw_requirement, "reason": "missing"}}
                )
            elif requirement.specifier and not requirement.specifier.contains(
                installed_version, prereleases=True
            ):
                dependency_violations.append(
                    {{
                        "requiring": requiring_name,
                        "requirement": raw_requirement,
                        "installed": installed_version,
                        "reason": "version_mismatch",
                    }}
                )
checks["dependency_graph_consistent"] = not dependency_violations and not mismatches_after
if not checks["dependency_graph_consistent"]:
    blockers.append("oscar_dependency_graph_inconsistent")
dependency_check_returncode = 0 if checks["dependency_graph_consistent"] else 1

passed = not blockers
payload = {{
    "schema_version": "single_g1_kitchen_oscar_runtime_dependency_repair.v1",
    "status": "passed" if passed else "blocked",
    "checks": checks,
    "foundation_lock_path": str(lock_path),
    "dependency_target": str(dependency_target),
    "foundation_lock_sha256": lock_sha256,
    "expected_foundation_lock_sha256": expected_lock_sha256,
    "locked_package_count": len(expected_versions),
    "mismatches_before": mismatches_before,
    "mismatches_after": mismatches_after,
    "seeded_base_distributions": seeded_base_distributions,
    "install_attempted": install_attempted,
    "install_returncode": install_returncode,
    "install_stdout_tail": install_stdout_tail,
    "install_stderr_tail": install_stderr_tail,
    "dependency_check_returncode": dependency_check_returncode,
    "dependency_check_stdout_tail": dependency_check_stdout_tail,
    "dependency_check_stderr_tail": dependency_check_stderr_tail,
    "dependency_violations": dependency_violations,
    "pytest_distribution_version": pytest_distribution_version,
    "pytest_module_version": pytest_module_version,
    "pytest_module_path": pytest_module_path,
    "pytest_fresh_subprocess_returncode": pytest_fresh_subprocess_returncode,
    "pytest_fresh_subprocess_stdout_tail": pytest_fresh_subprocess_stdout_tail,
    "pytest_fresh_subprocess_stderr_tail": pytest_fresh_subprocess_stderr_tail,
    "pytest_fresh_distribution_version": pytest_fresh_distribution_version,
    "pytest_fresh_module_version": pytest_fresh_module_version,
    "pytest_fresh_module_path": pytest_fresh_module_path,
    "groot_clean_import_returncode": groot_clean_import_returncode,
    "groot_clean_import_stdout_tail": groot_clean_import_stdout_tail,
    "groot_clean_import_stderr_tail": groot_clean_import_stderr_tail,
    "groot_clean_import_module_paths": groot_clean_import_module_paths,
    "transformer_engine_module_path": transformer_engine_module_path,
    "transformer_engine_metadata_dir": str(transformer_engine_metadata_dir),
    "transformer_engine_metadata_sha256s": transformer_engine_metadata_sha256s,
    "transformer_engine_distribution_version": transformer_engine_distribution_version,
    "transformer_engine_distribution_metadata_path": (
        transformer_engine_distribution_metadata_path
    ),
    "uv_path": str(uv_path) if uv_path is not None else None,
    "blockers": sorted(set(blockers)),
    "claim_boundary": {{
        "hash_locked_dependency_closure_bound": checks["foundation_lock_sha256_exact"],
        "verified_transformer_engine_shim_metadata_bound": checks.get(
            "transformer_engine_distribution_version_exact", False
        ),
        "verified_pytest_runtime_import_bound": checks.get(
            "pytest_runtime_importable", False
        ) and checks.get("pytest_runtime_distribution_version_exact", False),
        "verified_pytest_fresh_subprocess_bound": checks.get(
            "pytest_fresh_subprocess_returncode_zero", False
        ) and checks.get("pytest_fresh_distribution_version_exact", False),
        "groot_runtime_dependency_isolation_verified": all(
            checks.get(check_name, False)
            for check_name in (
                "groot_clean_pythonpath_excludes_oscar_dependency_target",
                "groot_clean_import_probe_returncode_zero",
                "groot_clean_import_modules_outside_oscar_dependency_target",
                "groot_clean_accelerate_resolved_from_groot_venv",
            )
        ),
        "oscar_cli_import_exercised_elsewhere": True,
        "checkpoint_loaded": False,
        "oscar_inference_ran": False,
        "generated_video_proven": False,
    }},
    "raw_secret_values_recorded": False,
}}
artifact_path.parent.mkdir(parents=True, exist_ok=True)
artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
OSCAR_RUNTIME_DEPENDENCY_RC=$?
set -e
if [ "$OSCAR_RUNTIME_DEPENDENCY_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$OSCAR_RUNTIME_DEPENDENCY_RC" \
    BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_dependency_repair_failed" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$OSCAR_RUNTIME_DEPENDENCY_RC"
fi
upload_phase oscar_runtime_dependencies_bound
"""


def _oscar_runtime_asset_prepare_and_preflight_script() -> str:
    """Prepare the immutable auxiliary cache, then prove it works offline.

    The network-enabled preparation process runs only after the reviewed
    dependency closure is bound.  A fresh process then verifies every pinned
    byte, constructs the Reason1 processor with local files only, and parses
    the exact OSCAR DCP metadata before the OSCAR dynamic config is imported.
    """

    offline_exports = render_offline_export_script(OSCAR_RUNTIME_ASSET_CACHE_ROOT)
    prepare_hf_home = offline_runtime_environment(OSCAR_RUNTIME_ASSET_CACHE_ROOT)["HF_HOME"]
    prepare_hub_cache = offline_runtime_environment(OSCAR_RUNTIME_ASSET_CACHE_ROOT)["HF_HUB_CACHE"]
    return f"""upload_phase oscar_runtime_asset_prepare_started
export HF_HOME={shlex.quote(prepare_hf_home)}
export HF_HUB_CACHE={shlex.quote(prepare_hub_cache)}
export HUGGINGFACE_HUB_CACHE={shlex.quote(prepare_hub_cache)}
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
set +e
{_scoped_oscar_python_command()} - <<'PY'
import json
import os
import time
from pathlib import Path

from blueprint_pipeline.oscar_runtime_asset_contract import (
    asset_contract_payload,
    prepare_runtime_asset_cache,
)

cache_root = Path({OSCAR_RUNTIME_ASSET_CACHE_ROOT!r})
artifact_path = Path({OSCAR_RUNTIME_ASSET_PREPARE_ARTIFACT!r})
prepare_attempts = []
try:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    runtime_evidence = {{}}
    for attempt_number, retry_delay_seconds in enumerate((0, 10, 30), start=1):
        if retry_delay_seconds:
            time.sleep(retry_delay_seconds)
        runtime_evidence = prepare_runtime_asset_cache(cache_root, token=token)
        prepare_attempts.append(
            {{
                "attempt_number": attempt_number,
                "status": runtime_evidence.get("status"),
                "blockers": list(runtime_evidence.get("blockers") or []),
                "retry_delay_seconds": retry_delay_seconds,
            }}
        )
        if runtime_evidence.get("status") == "passed":
            break
except Exception as exc:
    runtime_evidence = {{
        "schema_version": "oscar_runtime_asset_evidence.v1",
        "status": "blocked",
        "blockers": [f"single_episode_runtime_asset_prepare_exception:{{type(exc).__name__}}"],
        "raw_secret_values_recorded": False,
    }}
passed = runtime_evidence.get("status") == "passed"
payload = {{
    "schema_version": "single_g1_kitchen_oscar_runtime_asset_prepare.v1",
    "status": "passed" if passed else "blocked",
    "blockers": list(runtime_evidence.get("blockers") or []),
    "cache_root": str(cache_root),
    "asset_contract": asset_contract_payload(),
    "runtime_asset_evidence": runtime_evidence,
    "prepare_attempts": prepare_attempts,
    "prepare_attempt_count": len(prepare_attempts),
    "bounded_resume_retry_enabled": True,
    "claim_boundary": {{
        "runtime_auxiliary_assets_byte_verified": passed,
        "oscar_checkpoint_loaded": False,
        "oscar_inference_ran": False,
        "generated_video_proven": False,
    }},
    "raw_secret_values_recorded": False,
}}
artifact_path.parent.mkdir(parents=True, exist_ok=True)
artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"status": payload["status"], "blockers": payload["blockers"]}}, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY
OSCAR_RUNTIME_ASSET_PREPARE_RC=$?
set -e
if [ "$OSCAR_RUNTIME_ASSET_PREPARE_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$OSCAR_RUNTIME_ASSET_PREPARE_RC" \
    BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_asset_prepare_failed" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$OSCAR_RUNTIME_ASSET_PREPARE_RC"
fi
upload_phase oscar_runtime_assets_prepared
{offline_exports.rstrip()}
set +e
{_scoped_oscar_python_command()} - <<'PY'
import json
from pathlib import Path

from blueprint_pipeline.oscar_runtime_asset_contract import offline_preflight

cache_root = Path({OSCAR_RUNTIME_ASSET_CACHE_ROOT!r})
checkpoint_root = Path({OSCAR_RUNTIME_ASSET_CHECKPOINT_ROOT!r})
artifact_path = Path({OSCAR_RUNTIME_ASSET_PREFLIGHT_ARTIFACT!r})
try:
    offline_evidence = offline_preflight(
        cache_root,
        oscar_checkpoint_root=checkpoint_root,
        processor_probe=True,
        dcp_metadata_probe=True,
    )
except Exception as exc:
    offline_evidence = {{
        "schema_version": "oscar_runtime_asset_offline_preflight.v1",
        "status": "blocked",
        "blockers": [f"single_episode_runtime_asset_preflight_exception:{{type(exc).__name__}}"],
        "raw_secret_values_recorded": False,
    }}
passed = offline_evidence.get("status") == "passed"
payload = {{
    "schema_version": "single_g1_kitchen_oscar_runtime_asset_offline_preflight.v1",
    "status": "passed" if passed else "blocked",
    "blockers": list(offline_evidence.get("blockers") or []),
    "cache_root": str(cache_root),
    "checkpoint_root": str(checkpoint_root),
    "offline_preflight": offline_evidence,
    "claim_boundary": {{
        "offline_processor_and_dcp_metadata_preflight_passed": passed,
        "checkpoint_loaded_into_model": False,
        "oscar_inference_ran": False,
        "generated_video_proven": False,
    }},
    "raw_secret_values_recorded": False,
}}
artifact_path.parent.mkdir(parents=True, exist_ok=True)
artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"status": payload["status"], "blockers": payload["blockers"]}}, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY
OSCAR_RUNTIME_ASSET_PREFLIGHT_RC=$?
set -e
if [ "$OSCAR_RUNTIME_ASSET_PREFLIGHT_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$OSCAR_RUNTIME_ASSET_PREFLIGHT_RC" \
    BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_asset_offline_preflight_failed" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$OSCAR_RUNTIME_ASSET_PREFLIGHT_RC"
fi
upload_phase oscar_runtime_asset_offline_preflight_passed
"""


def _oscar_runtime_import_preflight_script() -> str:
    """Exercise OSCAR's real import/dynamic-link path before model residency.

    July 6 reached the OSCAR CLI only to fail on ``libcudnn_graph.so.9``. This
    probe uses the same interpreter, source checkout, PYTHONPATH, and cuDNN
    search path as per-step inference, while explicitly stopping short of a
    checkpoint load or generated-video claim.
    """

    return f"""set +e
{_scoped_oscar_python_command()} - <<'PY'
import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

cudnn_lib_dir = Path(os.environ.get({OSCAR_CUDNN_LIB_DIR_ENV!r}, ""))
entrypoint = Path("/opt/OSCAR/inference/inference_oscar.py")
required_libraries = ("libcudnn.so.9", "libcudnn_graph.so.9")
required_pytest_version = {OSCAR_RUNTIME_REQUIRED_PYTEST_VERSION!r}
checks = {{}}
blockers = []
checks["cudnn_runtime_directory_exact"] = str(cudnn_lib_dir) == {OSCAR_CUDNN_LIB_DIR!r}
checks["cudnn_runtime_directory_present"] = cudnn_lib_dir.is_dir()
if not checks["cudnn_runtime_directory_exact"] or not checks["cudnn_runtime_directory_present"]:
    blockers.append("oscar_cudnn_runtime_directory_invalid")
loaded_libraries = []
for library_name in required_libraries:
    library = cudnn_lib_dir / library_name
    checks[f"{{library_name}}_present"] = library.is_file()
    if not library.is_file():
        blockers.append(f"oscar_required_dynamic_library_missing:{{library_name}}")
        continue
    try:
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        loaded_libraries.append(library_name)
    except OSError as exc:
        blockers.append(f"oscar_dynamic_library_load_failed:{{library_name}}:{{type(exc).__name__}}")
checks["required_dynamic_libraries_loaded"] = len(loaded_libraries) == len(required_libraries)

pytest_distribution_version = None
pytest_module_version = None
pytest_module_path = None
try:
    import pytest as pytest_module

    pytest_distribution_version = importlib.metadata.version("pytest")
    pytest_module_version = str(getattr(pytest_module, "__version__", ""))
    pytest_module_path = str(Path(pytest_module.__file__).resolve())
    checks["oscar_pytest_importable"] = Path(pytest_module_path).is_file()
    checks["oscar_pytest_distribution_version_exact"] = (
        pytest_distribution_version == required_pytest_version
    )
    checks["oscar_pytest_module_version_exact"] = (
        pytest_module_version == required_pytest_version
    )
except Exception as exc:
    checks["oscar_pytest_importable"] = False
    checks["oscar_pytest_distribution_version_exact"] = False
    checks["oscar_pytest_module_version_exact"] = False
    blockers.append(f"oscar_pytest_import_failed:{{type(exc).__name__}}")
for check_name in (
    "oscar_pytest_importable",
    "oscar_pytest_distribution_version_exact",
    "oscar_pytest_module_version_exact",
):
    if not checks[check_name]:
        blockers.append(f"oscar_pytest_runtime_contract_failed:{{check_name}}")

transformer_engine_distribution_name = None
transformer_engine_distribution_version = None
try:
    transformer_engine_distribution = importlib.metadata.distribution("transformer-engine")
    transformer_engine_distribution_name = transformer_engine_distribution.metadata.get("Name")
    transformer_engine_distribution_version = importlib.metadata.version("transformer-engine")
    checks["oscar_transformer_engine_distribution_name_exact"] = (
        transformer_engine_distribution_name == "transformer-engine"
    )
    checks["oscar_transformer_engine_distribution_version_exact"] = (
        transformer_engine_distribution_version == "2.0.0"
    )
except importlib.metadata.PackageNotFoundError:
    checks["oscar_transformer_engine_distribution_name_exact"] = False
    checks["oscar_transformer_engine_distribution_version_exact"] = False
if (
    not checks["oscar_transformer_engine_distribution_name_exact"]
    or not checks["oscar_transformer_engine_distribution_version_exact"]
):
    blockers.append("oscar_transformer_engine_distribution_metadata_invalid")

try:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    checks["oscar_transformer_engine_attention_importable"] = callable(apply_rotary_pos_emb)
except Exception as exc:
    checks["oscar_transformer_engine_attention_importable"] = False
    blockers.append(f"oscar_transformer_engine_import_failed:{{type(exc).__name__}}")

checks["oscar_inference_entrypoint_present"] = entrypoint.is_file()
help_returncode = None
help_stdout_tail = ""
help_stderr_tail = ""
config_returncode = None
config_stdout_tail = ""
config_stderr_tail = ""
if entrypoint.is_file() and not blockers:
    child_env = os.environ.copy()
    existing = child_env.get("LD_LIBRARY_PATH", "").strip()
    child_env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(cudnn_lib_dir), existing) if value
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(entrypoint), "--help"],
            cwd="/opt/OSCAR",
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        help_returncode = completed.returncode
        help_stdout_tail = (completed.stdout or "")[-2000:]
        help_stderr_tail = (completed.stderr or "")[-4000:]
        if completed.returncode != 0:
            blockers.append(f"oscar_inference_cli_import_returncode:{{completed.returncode}}")
        else:
            config_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from worldsim._src.configs.agibot_control.config "
                        "import make_config; "
                        "config = make_config(); "
                        "assert config is not None; "
                        "print('BLUEPRINT_OSCAR_DYNAMIC_CONFIG_OK')"
                    ),
                ],
                cwd="/opt/OSCAR",
                env=child_env,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            config_returncode = config_probe.returncode
            config_stdout_tail = (config_probe.stdout or "")[-2000:]
            config_stderr_tail = (config_probe.stderr or "")[-4000:]
            if config_probe.returncode != 0:
                blockers.append(
                    f"oscar_dynamic_config_import_returncode:{{config_probe.returncode}}"
                )
    except Exception as exc:
        blockers.append(f"oscar_inference_cli_import_failed:{{type(exc).__name__}}")
else:
    if not entrypoint.is_file():
        blockers.append("oscar_inference_entrypoint_missing")
checks["oscar_dynamic_config_constructible"] = config_returncode == 0

passed = not blockers
payload = {{
    "schema_version": "single_g1_kitchen_oscar_runtime_import_preflight.v1",
    "status": "passed" if passed else "blocked",
    "checks": checks,
    "cudnn_runtime_directory": str(cudnn_lib_dir),
    "required_dynamic_libraries": list(required_libraries),
    "loaded_dynamic_libraries": loaded_libraries,
    "pytest_distribution_version": pytest_distribution_version,
    "pytest_module_version": pytest_module_version,
    "pytest_module_path": pytest_module_path,
    "transformer_engine_distribution_name": transformer_engine_distribution_name,
    "transformer_engine_distribution_version": transformer_engine_distribution_version,
    "oscar_inference_entrypoint": str(entrypoint),
    "oscar_inference_entrypoint_sha256": (
        hashlib.sha256(entrypoint.read_bytes()).hexdigest() if entrypoint.is_file() else None
    ),
    "oscar_help_returncode": help_returncode,
    "oscar_help_stdout_tail": help_stdout_tail,
    "oscar_help_stderr_tail": help_stderr_tail,
    "oscar_dynamic_config_returncode": config_returncode,
    "oscar_dynamic_config_stdout_tail": config_stdout_tail,
    "oscar_dynamic_config_stderr_tail": config_stderr_tail,
    "blockers": sorted(set(blockers)),
    "claim_boundary": {{
        "real_oscar_interpreter_import_path_exercised": True,
        "oscar_dynamic_config_constructed": checks["oscar_dynamic_config_constructible"],
        "dynamic_cudnn_linkage_exercised": True,
        "pytest_runtime_import_verified_before_dynamic_config": all(
            checks.get(check_name, False)
            for check_name in (
                "oscar_pytest_importable",
                "oscar_pytest_distribution_version_exact",
                "oscar_pytest_module_version_exact",
            )
        ),
        "checkpoint_loaded": False,
        "oscar_inference_ran": False,
        "generated_video_proven": False,
    }},
    "raw_secret_values_recorded": False,
}}
path = Path({OSCAR_RUNTIME_PREFLIGHT_ARTIFACT!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
raise SystemExit(0 if passed else 1)
PY
OSCAR_RUNTIME_PREFLIGHT_RC=$?
set -e
if [ "$OSCAR_RUNTIME_PREFLIGHT_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$OSCAR_RUNTIME_PREFLIGHT_RC" \
    BLUEPRINT_WORKER_FAILURE="official_oscar_runtime_import_preflight_failed" \
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit "$OSCAR_RUNTIME_PREFLIGHT_RC"
fi
upload_phase oscar_runtime_import_preflight_passed
"""


def _selected_scenario_payload(
    *,
    archive: zipfile.ZipFile,
    bundle_path: Path,
    attempt: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load the exact scenario digest named by the frozen attempt manifest."""
    artifacts = attempt.get("artifacts")
    scenario_ref = dict(artifacts.get("scenario") or {}) if isinstance(artifacts, Mapping) else {}
    expected_sha = str(scenario_ref.get("sha256") or "").strip().lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("single_episode_selected_scenario_digest_missing_or_invalid")

    candidates: list[tuple[str, bytes]] = []
    if "selected_isaac_scenario.json" in archive.namelist():
        candidates.append(
            ("bundle:selected_isaac_scenario.json", archive.read("selected_isaac_scenario.json"))
        )
    raw_path = str(scenario_ref.get("path") or "").strip()
    if raw_path:
        scenario_path = Path(raw_path).expanduser()
        if not scenario_path.is_absolute():
            scenario_path = bundle_path.parent / scenario_path
        if scenario_path.is_file() and not scenario_path.is_symlink():
            candidates.append((str(scenario_path), scenario_path.read_bytes()))
    if not candidates:
        raise ValueError("single_episode_selected_scenario_missing")

    matched = next(
        (
            (source, payload)
            for source, payload in candidates
            if hashlib.sha256(payload).hexdigest() == expected_sha
        ),
        None,
    )
    if matched is None:
        raise ValueError("single_episode_selected_scenario_digest_mismatch")
    _source, payload = matched
    try:
        scenario = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("single_episode_selected_scenario_unreadable") from exc
    if not isinstance(scenario, Mapping):
        raise ValueError("single_episode_selected_scenario_not_object")
    return dict(scenario), expected_sha


def _manipulation_policy_task_compatibility(
    *,
    plan: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    """Require sealed task qualification before admitting a paid episode.

    Checkpoint loadability and a valid SONIC action shape prove runtime
    compatibility only. The reviewed declaration belongs in the hash-bound
    launch plan so task fitness cannot be inferred from a model name, prompt,
    or visually plausible WAM prediction.
    """

    raw_declaration = plan.get("manipulation_policy_task_compatibility")
    declaration = (
        dict(raw_declaration) if isinstance(raw_declaration, Mapping) else {}
    )
    raw_evidence = declaration.get("qualification_evidence")
    evidence = dict(raw_evidence) if isinstance(raw_evidence, Mapping) else {}
    def _string_set(value: Any) -> set[str]:
        return (
            {str(item).strip() for item in value if str(item).strip()}
            if isinstance(value, list)
            else set()
        )

    training_task_ids = _string_set(declaration.get("training_task_ids"))
    training_task_descriptions = _string_set(
        declaration.get("training_task_descriptions")
    )
    reviewed_training_task_ids = set(SEALED_SONIC_REVIEWED_TRAINING_TASK_IDS)
    reviewed_training_task_descriptions = set(
        SEALED_SONIC_REVIEWED_TRAINING_TASK_DESCRIPTIONS
    )
    artifact_sha256 = str(evidence.get("artifact_sha256") or "").strip().lower()
    checks = {
        "declaration_present": bool(declaration),
        "schema_version_exact": declaration.get("schema_version")
        == MANIPULATION_POLICY_TASK_COMPATIBILITY_SCHEMA_VERSION,
        "status_qualified": declaration.get("status") == "qualified",
        "task_id_exact": declaration.get("task_id") == task_id,
        "embodiment_exact": declaration.get("embodiment")
        == REQUIRED_SONIC_EMBODIMENT,
        "checkpoint_repo_exact": declaration.get("checkpoint_repo")
        == SEALED_SONIC_CHECKPOINT_REPO,
        "checkpoint_revision_exact": declaration.get("checkpoint_revision")
        == SEALED_SONIC_CHECKPOINT_REVISION,
        "training_dataset_repo_exact": declaration.get("training_dataset_repo")
        == SEALED_SONIC_TRAINING_DATASET_REPO,
        "training_dataset_revision_exact": declaration.get(
            "training_dataset_revision"
        )
        == SEALED_SONIC_TRAINING_DATASET_REVISION,
        "training_task_ids_match_reviewed_provenance": training_task_ids
        == reviewed_training_task_ids,
        "training_task_descriptions_match_reviewed_provenance": (
            training_task_descriptions == reviewed_training_task_descriptions
        ),
        "requested_task_in_training_tasks": task_id in training_task_ids,
        "requested_task_in_reviewed_checkpoint_tasks": task_id
        in reviewed_training_task_ids,
        "qualification_evidence_passed": evidence.get("status") == "passed",
        "qualification_task_exact": evidence.get("task_id") == task_id,
        "qualification_environment_isaac": evidence.get("environment")
        == "isaac_sim",
        "registered_transition_exact": evidence.get("registered_transition_id")
        == REQUIRED_MANIPULATION_TRANSITION_ID,
        "qualification_artifact_sha256_valid": len(artifact_sha256) == 64
        and all(character in "0123456789abcdef" for character in artifact_sha256),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    qualified = not failed_checks
    return {
        "schema_version": MANIPULATION_POLICY_TASK_COMPATIBILITY_SCHEMA_VERSION,
        "status": "qualified" if qualified else "blocked",
        "task_id": task_id,
        "required_embodiment": REQUIRED_SONIC_EMBODIMENT,
        "sealed_checkpoint": {
            "repo": SEALED_SONIC_CHECKPOINT_REPO,
            "revision": SEALED_SONIC_CHECKPOINT_REVISION,
        },
        "reviewed_checkpoint_training_provenance": {
            "dataset_repo": SEALED_SONIC_TRAINING_DATASET_REPO,
            "dataset_revision": SEALED_SONIC_TRAINING_DATASET_REVISION,
            "canonical_task_ids": sorted(reviewed_training_task_ids),
            "task_descriptions": sorted(reviewed_training_task_descriptions),
            "requested_task_covered": task_id in reviewed_training_task_ids,
        },
        "declaration": declaration or None,
        "checks": checks,
        "failed_checks": failed_checks,
        "blockers": (
            [] if qualified else [MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER]
        ),
        "claim_boundary": {
            "checkpoint_loadability_is_not_task_qualification": True,
            "sonic_action_shape_is_not_task_qualification": True,
            "wam_visual_prediction_is_not_registered_transition_success": True,
            "qualification_is_not_episode_success": True,
        },
    }


def _qualification_checkpoint_restore_evidence(
    *,
    worker_report_path: str | Path,
    part_stage_dirs: tuple[str | Path, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Validate one completed fine-tune and bind its ordered remote parts."""

    report_path = Path(worker_report_path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("qualification_checkpoint_worker_report_invalid") from exc
    if not isinstance(report, Mapping) or report.get("status") != "completed":
        raise ValueError("qualification_checkpoint_worker_not_completed")
    open_loop = report.get("open_loop_qualification")
    open_loop = dict(open_loop) if isinstance(open_loop, Mapping) else {}
    if (
        open_loop.get("status") != "passed"
        or open_loop.get("exact_owned_training_trajectory_only") is not True
        or open_loop.get("isaac_registered_transition_not_proven") is not True
        or int(open_loop.get("steps") or 0) < 120
    ):
        raise ValueError("qualification_checkpoint_open_loop_not_qualified")
    checkpoint = report.get("checkpoint_archive")
    checkpoint = dict(checkpoint) if isinstance(checkpoint, Mapping) else {}
    upload = checkpoint.get("upload")
    upload = dict(upload) if isinstance(upload, Mapping) else {}
    archive_sha256 = str(checkpoint.get("sha256") or "").strip().lower()
    archive_size_bytes = int(checkpoint.get("size_bytes") or 0)
    rows = upload.get("parts")
    rows = [dict(row) for row in rows] if isinstance(rows, list) else []
    if (
        upload.get("status") != "passed"
        or upload.get("transport") != "ordered_parts"
        or upload.get("uploaded_sha256") != archive_sha256
        or int(upload.get("uploaded_size_bytes") or 0) != archive_size_bytes
        or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
        or not rows
        or len(rows) > MAX_QUALIFICATION_CHECKPOINT_PARTS
    ):
        raise ValueError("qualification_checkpoint_archive_transport_invalid")
    ordered_rows = sorted(rows, key=lambda row: int(row.get("part_number") or 0))
    if [int(row.get("part_number") or 0) for row in ordered_rows] != list(
        range(1, len(ordered_rows) + 1)
    ):
        raise ValueError("qualification_checkpoint_part_order_invalid")
    if len(part_stage_dirs) != len(ordered_rows):
        raise ValueError("qualification_checkpoint_part_stage_count_mismatch")
    urls: list[str] = []
    safe_rows: list[dict[str, Any]] = []
    total_size = 0
    for row, stage_dir in zip(ordered_rows, part_stage_dirs, strict=True):
        part_sha256 = str(row.get("sha256") or "").strip().lower()
        part_size_bytes = int(row.get("size_bytes") or 0)
        part_upload = row.get("upload")
        part_upload = dict(part_upload) if isinstance(part_upload, Mapping) else {}
        if (
            len(part_sha256) != 64
            or any(character not in "0123456789abcdef" for character in part_sha256)
            or part_size_bytes <= 0
            or part_upload.get("status") != "passed"
            or part_upload.get("uploaded_sha256") != part_sha256
            or int(part_upload.get("uploaded_size_bytes") or 0) != part_size_bytes
        ):
            raise ValueError("qualification_checkpoint_part_evidence_invalid")
        urls.append(
            _read_secret_url_file(
                Path(stage_dir).expanduser().resolve() / "provider_output_get_url.txt"
            )
        )
        safe_rows.append(
            {
                "part_number": int(row["part_number"]),
                "size_bytes": part_size_bytes,
                "sha256": part_sha256,
            }
        )
        total_size += part_size_bytes
    if total_size != archive_size_bytes:
        raise ValueError("qualification_checkpoint_part_total_size_mismatch")
    return (
        {
            "schema_version": QUALIFICATION_CHECKPOINT_RESTORE_SCHEMA_VERSION,
            "status": "qualified_for_isaac_evaluation",
            "worker_report_path": str(report_path),
            "checkpoint_path": REMOTE_FINAL_CHECKPOINT,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "parts": safe_rows,
            "open_loop_qualification": {
                "status": "passed",
                "steps": int(open_loop["steps"]),
                "mse_ratio": open_loop.get("mse_ratio"),
                "mae_ratio": open_loop.get("mae_ratio"),
                "maximum_error_ratio": open_loop.get("maximum_error_ratio"),
            },
            "task_compatibility_claimed": False,
            "isaac_registered_transition_required": True,
            "raw_signed_urls_recorded": False,
        },
        urls,
    )


def _apply_qualification_checkpoint_override(
    plan: dict[str, Any], checkpoint_path: str | None
) -> dict[str, Any]:
    if checkpoint_path in {None, ""}:
        return plan
    if checkpoint_path != REMOTE_FINAL_CHECKPOINT:
        raise ValueError("qualification_trained_checkpoint_path_not_fixed")
    command = [str(item) for item in plan.get("groot_server_command") or []]
    positions = [index for index, item in enumerate(command) if item == "--model-path"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("qualification_groot_model_path_option_invalid")
    command[positions[0] + 1] = checkpoint_path
    plan["groot_server_command"] = command
    plan["qualification_checkpoint_override"] = {
        "schema_version": "single_g1_kitchen_qualification_checkpoint_override.v1",
        "checkpoint_path": checkpoint_path,
        "open_loop_qualification_required": True,
        "isaac_registered_transition_required": True,
        "task_compatibility_claimed": False,
    }
    return plan


def _qualification_checkpoint_restore_script(evidence: Mapping[str, Any]) -> str:
    """Stream ordered parts once, verify both hash layers, and safely extract."""

    safe_payload = base64.b64encode(
        json.dumps(dict(evidence), sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return f"""
python3 - <<'PY'
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

evidence = json.loads(base64.b64decode({safe_payload!r}))
urls = json.loads(os.environ[{QUALIFICATION_CHECKPOINT_PART_GET_URLS_ENV!r}])
parts = evidence["parts"]
if not isinstance(urls, list) or len(urls) != len(parts):
    raise RuntimeError("qualification_checkpoint_part_url_count_mismatch")
archive = Path("/workspace/g1_microwave_qualification_checkpoint.zip")
aggregate = hashlib.sha256()
total = 0
with archive.open("wb") as output:
    for url, part in zip(urls, parts, strict=True):
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(url, timeout=300) as response:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                aggregate.update(chunk)
                digest.update(chunk)
                size += len(chunk)
                total += len(chunk)
        if size != int(part["size_bytes"]) or digest.hexdigest() != part["sha256"]:
            raise RuntimeError("qualification_checkpoint_part_digest_mismatch")
if total != int(evidence["archive_size_bytes"]):
    raise RuntimeError("qualification_checkpoint_archive_size_mismatch")
if aggregate.hexdigest() != evidence["archive_sha256"]:
    raise RuntimeError("qualification_checkpoint_archive_digest_mismatch")
destination = Path({REMOTE_FINAL_CHECKPOINT!r})
destination.parent.mkdir(parents=True, exist_ok=True)
snapshot = Path(tempfile.mkdtemp(prefix=".qualification-checkpoint-", dir=str(destination.parent)))
try:
    with zipfile.ZipFile(archive) as checkpoint_zip:
        members = checkpoint_zip.infolist()
        if not members:
            raise RuntimeError("qualification_checkpoint_archive_empty")
        for member in members:
            relative = Path(member.filename)
            target = (snapshot / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(snapshot.resolve()):
                raise RuntimeError("qualification_checkpoint_archive_member_unsafe")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError("qualification_checkpoint_archive_link_forbidden")
        checkpoint_zip.extractall(snapshot)
    if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
        raise RuntimeError("qualification_checkpoint_model_tree_invalid")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(snapshot), str(destination))
finally:
    shutil.rmtree(snapshot, ignore_errors=True)
archive.unlink()
restore_report_path = Path({QUALIFICATION_CHECKPOINT_RESTORE_REPORT_PATH!r})
restore_report_path.parent.mkdir(parents=True, exist_ok=True)
restore_report_path.write_text(
    json.dumps({{
        "schema_version": {QUALIFICATION_CHECKPOINT_RESTORE_SCHEMA_VERSION!r},
        "status": "passed",
        "checkpoint_path": str(destination),
        "archive_sha256": evidence["archive_sha256"],
        "archive_size_bytes": total,
        "part_count": len(parts),
        "raw_signed_urls_recorded": False,
    }}, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY
upload_phase qualification_checkpoint_restored
"""


def _resolve_microwave_task_contract(
    *,
    task_contract: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scenario_sha256: str,
    attempt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the selected microwave affordance prim into the direct-run contract."""
    if str(task_contract.get("task_id") or "") != "microwave_door":
        raise ValueError("single_episode_task_contract_not_microwave_door")
    if str(scenario.get("task_id") or "") != "microwave_door":
        raise ValueError("single_episode_selected_scenario_not_microwave_door")

    artifacts = attempt.get("artifacts")
    artifact_map = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    selection_ref = artifact_map.get("selection")
    selected_sha = (
        str(dict(selection_ref).get("sha256") or "").strip().lower()
        if isinstance(selection_ref, Mapping)
        else ""
    )
    if len(selected_sha) != 64:
        raise ValueError("single_episode_selection_digest_missing")
    for source, observed in (
        ("task_contract", task_contract.get("source_selection_sha256")),
        ("selected_scenario", scenario.get("source_selection_sha256")),
    ):
        if str(observed or "").strip().lower() != selected_sha:
            raise ValueError(f"single_episode_{source}_selection_digest_mismatch")

    stance = scenario.get("accepted_stance_contract")
    stance_contract = dict(stance) if isinstance(stance, Mapping) else {}
    if stance_contract.get("status") != "accepted":
        raise ValueError("single_episode_selected_scenario_stance_not_accepted")
    affordance = stance_contract.get("resolved_affordance")
    resolved_affordance = dict(affordance) if isinstance(affordance, Mapping) else {}
    target = stance_contract.get("resolved_target")
    resolved_target = dict(target) if isinstance(target, Mapping) else {}
    prim_path = str(resolved_affordance.get("prim_path") or "").strip()
    target_path = str(resolved_target.get("prim_path") or "").strip()
    lower_prim = prim_path.lower()
    if (
        not prim_path.startswith("/")
        or "microwave" not in lower_prim
        or "door" not in lower_prim
        or not target_path.startswith("/")
        or not prim_path.startswith(target_path.rstrip("/") + "/")
        or str(resolved_affordance.get("target_object_id") or "").lower() != "door"
    ):
        raise ValueError("single_episode_selected_scenario_microwave_door_prim_invalid")

    criteria_raw = task_contract.get("registered_criteria") or task_contract.get("criteria")
    criteria = [dict(item) for item in criteria_raw or [] if isinstance(item, Mapping)]
    if len(criteria) != 1:
        raise ValueError("single_episode_task_contract_requires_one_criterion")
    criterion = criteria[0]
    if str(criterion.get("criterion_id") or "") != "microwave_door_open_angle":
        raise ValueError("single_episode_task_contract_criterion_mismatch")
    existing_path = str(criterion.get("articulation_prim_path") or "").strip()
    if existing_path and existing_path != prim_path:
        raise ValueError("single_episode_task_contract_prim_conflicts_with_scenario")
    criterion["articulation_prim_path"] = prim_path
    criterion["articulation_prim_path_resolution"] = {
        "mode": "exact_selected_scenario_affordance",
        "selected_scenario_sha256": scenario_sha256,
        "selected_scenario_field": ("accepted_stance_contract.resolved_affordance.prim_path"),
    }
    resolved = copy.deepcopy(dict(task_contract))
    resolved["registered_criteria"] = [criterion]
    resolved.pop("criteria", None)
    resolution = {
        "schema_version": "single_g1_kitchen_task_contract_resolution.v1",
        "status": "resolved",
        "task_id": "microwave_door",
        "selected_scenario_sha256": scenario_sha256,
        "selected_scenario_field": ("accepted_stance_contract.resolved_affordance.prim_path"),
        "articulation_prim_path": prim_path,
        "stage_wide_joint_scan_required": False,
    }
    return resolved, resolution


def _task_contract_overlay_script(
    *,
    resolved_contract_bytes: bytes,
    source_contract_sha256: str,
    resolution: Mapping[str, Any],
) -> str:
    """Materialize the scenario-bound contract and bind its digest to this attempt."""
    encoded = base64.b64encode(resolved_contract_bytes).decode("ascii")
    resolution_json = json.dumps(dict(resolution), sort_keys=True, separators=(",", ":"))
    return (
        "python3 - <<'PY'\n"
        "import base64, hashlib, json, os\n"
        "from pathlib import Path\n\n"
        f"contract_bytes = base64.b64decode({encoded!r}, validate=True)\n"
        "contract_path = Path('/workspace/task_success_contract.json')\n"
        "attempt_path = Path('/workspace/attempt_input_manifest.json')\n"
        "resolution_path = Path('/workspace/closed_loop_out/direct_task_contract_resolution.json')\n"
        "contract_path.write_bytes(contract_bytes)\n"
        "resolved_sha = hashlib.sha256(contract_bytes).hexdigest()\n"
        "attempt = json.loads(attempt_path.read_text(encoding='utf-8'))\n"
        "if attempt.get('selected_task_id') != 'microwave_door':\n"
        "    raise RuntimeError('direct_task_contract_attempt_identity_mismatch')\n"
        "launch_session_id = os.environ.get('BLUEPRINT_LAUNCH_SESSION_ID', '').strip()\n"
        "if not launch_session_id:\n"
        "    raise RuntimeError('direct_task_contract_launch_session_identity_missing')\n"
        "qualification_nonce = os.environ.get('BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE', '').strip()\n"
        "qualification_nonce_sha256 = os.environ.get('BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE_SHA256', '').strip().lower()\n"
        "qualification_sequence_text = os.environ.get('BLUEPRINT_QUALIFICATION_ATTEMPT_SEQUENCE', '').strip()\n"
        "qualification_attempt_bound = bool(qualification_nonce)\n"
        "if qualification_attempt_bound:\n"
        "    if not qualification_sequence_text.isdigit() or int(qualification_sequence_text) < 1:\n"
        "        raise RuntimeError('qualification_attempt_sequence_invalid')\n"
        "    computed_qualification_sha256 = hashlib.sha256(qualification_nonce.encode('utf-8')).hexdigest()\n"
        "    if qualification_nonce_sha256 != computed_qualification_sha256:\n"
        "        raise RuntimeError('qualification_attempt_nonce_sha256_mismatch')\n"
        "else:\n"
        "    if qualification_sequence_text or qualification_nonce_sha256:\n"
        "        raise RuntimeError('qualification_attempt_identity_incomplete')\n"
        "    computed_qualification_sha256 = None\n"
        "active_launch_nonce = qualification_nonce or launch_session_id\n"
        "prepared_launch_nonce = str(attempt.get('launch_nonce') or '')\n"
        "attempt['prepared_launch_nonce'] = prepared_launch_nonce\n"
        "attempt['allocation_launch_session_id'] = launch_session_id\n"
        "attempt['launch_nonce'] = active_launch_nonce\n"
        "attempt['qualification_attempt_bound'] = qualification_attempt_bound\n"
        "attempt['qualification_attempt_sequence'] = int(qualification_sequence_text) if qualification_attempt_bound else None\n"
        "attempt['qualification_attempt_nonce'] = qualification_nonce or None\n"
        "attempt['qualification_attempt_nonce_sha256'] = computed_qualification_sha256\n"
        "artifacts = attempt.setdefault('artifacts', {})\n"
        "artifacts['task_success_contract'] = {\n"
        "    'path': str(contract_path),\n"
        "    'sha256': resolved_sha,\n"
        "    'size_bytes': len(contract_bytes),\n"
        f"    'derived_from_sha256': {source_contract_sha256!r},\n"
        "    'resolution_artifact_path': str(resolution_path),\n"
        "}\n"
        "attempt_bytes = (json.dumps(attempt, indent=2, sort_keys=True) + '\\n').encode('utf-8')\n"
        "attempt_path.write_bytes(attempt_bytes)\n"
        "collected_attempt_path = Path('/workspace/closed_loop_out/attempt_input_manifest.json')\n"
        "collected_attempt_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "collected_attempt_path.write_bytes(attempt_bytes)\n"
        f"resolution = json.loads({resolution_json!r})\n"
        f"resolution['source_task_contract_sha256'] = {source_contract_sha256!r}\n"
        "resolution['resolved_task_contract_sha256'] = resolved_sha\n"
        "resolution['prepared_launch_nonce'] = prepared_launch_nonce\n"
        "resolution['active_launch_nonce'] = active_launch_nonce\n"
        "resolution['allocation_launch_session_id'] = launch_session_id\n"
        "resolution['qualification_attempt_bound'] = qualification_attempt_bound\n"
        "resolution['qualification_attempt_sequence'] = int(qualification_sequence_text) if qualification_attempt_bound else None\n"
        "resolution['qualification_attempt_nonce_sha256'] = computed_qualification_sha256\n"
        "resolution['launch_nonce_rebound_to_current_allocation_session'] = not qualification_attempt_bound\n"
        "resolution['launch_nonce_rebound_to_qualification_attempt'] = qualification_attempt_bound\n"
        "resolution['attempt_input_manifest_path'] = str(attempt_path)\n"
        "resolution['collected_attempt_input_manifest_path'] = str(collected_attempt_path)\n"
        "resolution_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "resolution_path.write_text(json.dumps(resolution, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "PY\n"
    )


def _pin_bootstrap_interpreters(script: str) -> str:
    """Remove every dependency on the image having a bare ``python`` alias."""
    # OSCAR's runtime asset preparation intentionally exports its own HF_HOME
    # for the learned Reason1/Wan assets.  The sealed-image healthcheck instead
    # verifies GR00T's baked Reason2 backbone.  Scope that one subprocess back
    # to the immutable image cache so an ambient OSCAR cache cannot make the
    # healthcheck inspect the wrong model family or attempt a network fallback.
    healthcheck_entrypoint = (
        "python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py"
    )
    if script.count(healthcheck_entrypoint) != 1:
        raise ValueError("single_episode_bootstrap_healthcheck_marker_ambiguous")
    sealed_healthcheck = (
        f"HF_HOME={SEALED_GROOT_HF_HOME} "
        f"HF_HUB_CACHE={SEALED_GROOT_HF_HUB_CACHE} "
        f"HUGGINGFACE_HUB_CACHE={SEALED_GROOT_HF_HUB_CACHE} "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        f"COSMOS_BACKBONE_REPO={SEALED_COSMOS_BACKBONE_REPO} "
        f"COSMOS_BACKBONE_REVISION={SEALED_COSMOS_BACKBONE_REVISION} "
        f"{OSCAR_PYTHON} "
        "/opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py"
    )
    resolved = script.replace(healthcheck_entrypoint, sealed_healthcheck, 1)
    resolved = re.sub(
        r"(?<!\S)python(?= -m blueprint_pipeline\.)",
        OSCAR_PYTHON,
        resolved,
    )
    resolved = re.sub(r"(?<!\S)python(?= /workspace/)", SYSTEM_PYTHON, resolved)
    resolved = re.sub(r"(?<!\S)python(?= - <<'PY')", SYSTEM_PYTHON, resolved)
    forbidden = re.search(
        r"(?<!\S)python(?=(?: -m blueprint_pipeline\.| /workspace/| - <<'PY'))",
        resolved,
    )
    if forbidden:
        raise ValueError("single_episode_bootstrap_bare_python_remaining")
    if "python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py" in resolved:
        raise ValueError("single_episode_bootstrap_healthcheck_wrong_interpreter")
    return resolved


def _inject_gear_sonic_isaac_dds_bridge(script: str) -> str:
    """Start and own the native bridge beside the one persistent Isaac process."""

    isaac_pid_marker = "ISAAC_TASK_PID=$!\n"
    export_marker = "export GROOT_PID GEAR_SONIC_PID ISAAC_TASK_PID\n"
    cleanup_marker = '  kill "$ISAAC_TASK_PID" >/dev/null 2>&1 || true\n'
    for marker, error in (
        (isaac_pid_marker, "single_episode_isaac_pid_marker_ambiguous"),
        (export_marker, "single_episode_process_export_marker_ambiguous"),
        (cleanup_marker, "single_episode_process_cleanup_marker_ambiguous"),
    ):
        if script.count(marker) != 1:
            raise ValueError(error)
    script = script.replace(
        isaac_pid_marker,
        isaac_pid_marker + bridge_start_script(),
        1,
    )
    script = script.replace(
        export_marker,
        f"export GROOT_PID GEAR_SONIC_PID ISAAC_TASK_PID {BRIDGE_PID_ENV}\n",
        1,
    )
    return script.replace(
        cleanup_marker,
        cleanup_marker + f'  kill "${{{BRIDGE_PID_ENV}}}" >/dev/null 2>&1 || true\n',
        1,
    )


def _bind_provider_allocation_identity(script: str, provider_name: str) -> str:
    """Resolve the provider's injected allocation id before attempt trust is made."""
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"single_episode_provider_unsupported:{provider_name}")
    marker = "upload_phase inputs_ready\n"
    if script.count(marker) != 1:
        raise ValueError("single_episode_allocation_identity_marker_ambiguous")
    if provider_name == "vast":
        candidates = "${CONTAINER_ID:-} ${VAST_CONTAINERLABEL:-}"
        candidate_validation = """case "$BLUEPRINT_ALLOCATION_CANDIDATE" in
    ''|*[!0-9]*) continue ;;
  esac"""
    else:
        candidates = "${RUNPOD_POD_ID:-} ${RUNPOD_WORKER_ID:-}"
        candidate_validation = """if [ -z "$BLUEPRINT_ALLOCATION_CANDIDATE" ]; then
    continue
  fi"""
    resolver = f"""for BLUEPRINT_ALLOCATION_CANDIDATE in {candidates}; do
  BLUEPRINT_ALLOCATION_CANDIDATE="${{BLUEPRINT_ALLOCATION_CANDIDATE#C.}}"
  {candidate_validation}
  BLUEPRINT_PROVIDER_ALLOCATION_ID="$BLUEPRINT_ALLOCATION_CANDIDATE"
  export BLUEPRINT_PROVIDER_ALLOCATION_ID
  break
done
if [ -z "${{BLUEPRINT_PROVIDER_ALLOCATION_ID:-}}" ]; then
  BLUEPRINT_CLOSED_LOOP_RC=42 \\
    BLUEPRINT_WORKER_FAILURE="provider_allocation_identity_unavailable" \\
    python3 /workspace/write_result.py
  upload_phase runner_done
  exit 42
fi
upload_phase provider_allocation_bound
"""
    return script.replace(marker, marker + resolver, 1)


def _egocentric_camera_safe_route(
    *, route: Mapping[str, Any], scenario: Mapping[str, Any]
) -> dict[str, Any]:
    """Move the accepted stance to an existing camera-safe distance candidate."""

    stance_raw = scenario.get("accepted_stance_contract")
    stance = dict(stance_raw) if isinstance(stance_raw, Mapping) else {}
    pose = stance.get("pose_xyz")
    focus = stance.get("stance_focus_xyz")
    candidates = scenario.get("stance_distance_candidates_m")
    route_points = route.get("route_points")
    if (
        stance.get("status") != "accepted"
        or not isinstance(pose, list)
        or len(pose) != 3
        or not isinstance(focus, list)
        or len(focus) != 3
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(route_points, list)
        or len(route_points) != 2
    ):
        raise ValueError("single_episode_egocentric_stance_contract_incomplete")
    pose_xyz = [float(value) for value in pose]
    focus_xyz = [float(value) for value in focus]
    if not all(math.isfinite(value) for value in (*pose_xyz, *focus_xyz)):
        raise ValueError("single_episode_egocentric_stance_value_invalid")
    for point in route_points:
        if (
            not isinstance(point, list)
            or len(point) != 3
            or any(
                not math.isclose(
                    float(point[index]), pose_xyz[index], rel_tol=0.0, abs_tol=1e-6
                )
                for index in range(3)
            )
        ):
            raise ValueError("single_episode_route_not_bound_to_accepted_stance")
    finite_candidates = sorted(
        float(value)
        for value in candidates
        if math.isfinite(float(value))
        and float(value) >= DIRECT_EGOCENTRIC_STANCE_MIN_DISTANCE_M
    )
    if not finite_candidates:
        raise ValueError("single_episode_egocentric_stance_candidate_missing")
    selected_distance = finite_candidates[0]
    away_x = pose_xyz[0] - focus_xyz[0]
    away_y = pose_xyz[1] - focus_xyz[1]
    away_norm = math.hypot(away_x, away_y)
    if away_norm <= 1e-6:
        raise ValueError("single_episode_egocentric_stance_direction_invalid")
    adjusted_pose = [
        focus_xyz[0] + away_x / away_norm * selected_distance,
        focus_xyz[1] + away_y / away_norm * selected_distance,
        pose_xyz[2],
    ]
    return {
        **dict(route),
        "route_points": [adjusted_pose, adjusted_pose],
        "route_semantics": (
            "Attempt begins at the scenario-validated camera-safe manipulation "
            "stance; the robot POV remains rigidly head mounted."
        ),
        "egocentric_camera_safe_stance": {
            "status": "selected_from_scenario_candidates",
            "source_pose_xyz": pose_xyz,
            "stance_focus_xyz": focus_xyz,
            "selected_pose_xyz": adjusted_pose,
            "selected_stance_distance_m": selected_distance,
            "minimum_stance_distance_m": (
                DIRECT_EGOCENTRIC_STANCE_MIN_DISTANCE_M
            ),
            "camera_mount_changed": False,
            "surrogate": False,
        },
    }


def _load_single_episode_inputs(
    bundle_path: Path,
    *,
    qualification_checkpoint_restore: Mapping[str, Any] | None = None,
    remote_bundle_url: str | None = None,
) -> dict[str, Any]:
    remote_reader: _HttpsRangeReader | None = None
    if bundle_path.is_file() and zipfile.is_zipfile(bundle_path):
        digest = _sha256(bundle_path)
        archive_source: Any = bundle_path
    elif remote_bundle_url:
        digest, remote_size = _sha256_remote_bundle(remote_bundle_url)
        remote_reader = _HttpsRangeReader(remote_bundle_url)
        if remote_reader._size != remote_size:
            raise ValueError("single_episode_remote_bundle_size_changed")
        archive_source = remote_reader
    else:
        raise ValueError("single_episode_bundle_missing_or_invalid_zip")
    if digest != BUNDLE_SHA256:
        raise ValueError("single_episode_bundle_digest_mismatch")
    required = {
        "initial_policy_frame.png",
        "route.json",
        "seed_provenance.json",
        "sealed_launch_plan.json",
        "task_success_contract.json",
        "attempt_input_manifest_episode_001.json",
        "episode_review_builder.py",
        "kitchen/KitchenRoom.usd",
    }
    with zipfile.ZipFile(archive_source) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            raise ValueError("single_episode_bundle_members_missing:" + ",".join(missing))
        plan = json.loads(archive.read("sealed_launch_plan.json"))
        route = json.loads(archive.read("route.json"))
        seed = json.loads(archive.read("seed_provenance.json"))
        start_frame = archive.read("initial_policy_frame.png")
        attempt = json.loads(archive.read("attempt_input_manifest_episode_001.json"))
        task_contract_raw = archive.read("task_success_contract.json")
        task_contract = json.loads(task_contract_raw)
        scenario, scenario_sha256 = _selected_scenario_payload(
            archive=archive,
            bundle_path=bundle_path,
            attempt=attempt,
        )
    if not isinstance(plan, Mapping) or plan.get("sealed_active") is not True:
        raise ValueError("single_episode_sealed_plan_not_active")
    if plan.get("blockers"):
        raise ValueError("single_episode_sealed_plan_blocked")
    if plan.get("image_ref") != IMAGE_REF:
        raise ValueError("single_episode_plan_image_mismatch")
    if attempt.get("attempt_id") != "episode_001":
        raise ValueError("single_episode_attempt_identity_mismatch")
    if attempt.get("image_digest") != IMAGE_DIGEST:
        raise ValueError("single_episode_attempt_image_mismatch")
    if attempt.get("selected_task_id") != "microwave_door":
        raise ValueError("single_episode_task_identity_mismatch")
    manipulation_policy_task_compatibility = (
        _manipulation_policy_task_compatibility(
            plan=plan,
            task_id=REQUIRED_MANIPULATION_TASK_ID,
        )
    )
    if not isinstance(task_contract, Mapping):
        raise ValueError("single_episode_task_contract_not_object")
    resolved_task_contract, task_contract_resolution = _resolve_microwave_task_contract(
        task_contract=task_contract,
        scenario=scenario,
        scenario_sha256=scenario_sha256,
        attempt=attempt,
    )
    route = _egocentric_camera_safe_route(route=route, scenario=scenario)
    perception_target_prompts = _scenario_perception_target_prompts(scenario)
    resolved_task_contract_bytes = (
        json.dumps(resolved_task_contract, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    plan = copy.deepcopy(dict(plan))
    restore_evidence = (
        dict(qualification_checkpoint_restore)
        if isinstance(qualification_checkpoint_restore, Mapping)
        else {}
    )
    plan = _apply_qualification_checkpoint_override(
        plan,
        str(restore_evidence.get("checkpoint_path") or "") or None,
    )
    command = [str(item) for item in plan.get("closed_loop_command") or []]
    if not command:
        raise ValueError("single_episode_closed_loop_command_missing")
    command = _pin_blueprint_command_interpreter(command)
    expected_command = [
        OSCAR_PYTHON,
        "-m",
        "blueprint_pipeline.oscar_isaac_closed_loop_eval",
    ]
    if command[: len(expected_command)] != expected_command:
        raise ValueError("single_episode_closed_loop_command_unexpected")
    command = _replace_option(command, "--output-dir", "/workspace/closed_loop_out/episode_001")
    command = _replace_option(command, "--oscar-seed", "1001")
    command = _replace_option(command, "--num-frames", str(DIRECT_OSCAR_NUM_FRAMES))
    command = _replace_option(
        command,
        "--oscar-num-steps",
        str(DIRECT_OSCAR_NUM_STEPS),
    )
    command = _replace_option(
        command,
        "--min-steps",
        str(DIRECT_EPISODE_MIN_STEPS),
    )
    # Manipulation tasks enable unsafe-stance and online no-progress stopping
    # internally when ``--stop-on-task-completion`` is active. They are not
    # CLI switches; keep only the supported patience control below.
    command = _replace_option(
        command,
        "--no-progress-patience-steps",
        str(DIRECT_NO_PROGRESS_PATIENCE_STEPS),
    )
    command = _replace_option(
        command,
        "--clean-frame-reanchor-interval",
        str(DIRECT_CLEAN_FRAME_REANCHOR_INTERVAL),
    )
    command = _replace_option(
        command,
        "--groot-sonic-execution-frame-count",
        str(DIRECT_GROOT_SONIC_EXECUTION_FRAME_COUNT),
    )
    command = _replace_repeated_option(command, "--task-prompt", [TASK_PROMPT])
    command = _replace_repeated_option(
        command,
        "--perception-target-prompt",
        perception_target_prompts,
    )
    # This direct goal feeds the exact fresh OSCAR RGB frame back to the RGB-only
    # GR00T observation contract. It does not request SAM3, DA3, or another
    # perception model; semantic success remains a live Isaac transition only.
    command = _replace_option(
        command,
        "--harness-backend-kind",
        GENERATED_RGB_POLICY_OBSERVATION_BACKEND_KIND,
    )
    for option in DIRECT_RGB_OBSERVATION_REMOVED_REQUIREMENTS:
        command = _remove_option(command, option, takes_value=False)
    # The prepared campaign plan required an additional external strict
    # forward/inverse judge whose service was never bundled.  This direct run
    # exercises the real learned OSCAR/WAM checkpoint itself and keeps the
    # external-judge claim explicitly out of scope.
    for option, takes_value in EXTERNAL_CONSISTENCY_OPTIONS.items():
        command = _remove_option(command, option, takes_value=takes_value)
    plan_env = copy.deepcopy(dict(plan.get("env") or {}))
    for name in EXTERNAL_CONSISTENCY_ENV:
        plan_env.pop(name, None)
    plan_env["PYTHONPATH"] = _runtime_pythonpath(plan_env.get("PYTHONPATH"))
    plan_env[GROOT_RUNTIME_PYTHONPATH_ENV] = plan_env["PYTHONPATH"]
    plan_env[GROOT_VENV_ROOT_ENV] = GROOT_VENV_ROOT
    plan_env[OSCAR_RUNTIME_DEPENDENCY_TARGET_ENV] = (
        OSCAR_RUNTIME_DEPENDENCY_TARGET
    )
    # The immutable image was built from this exact reviewed OSCAR checkout.
    # Override inherited prepared-plan values, then independently verify the
    # live checkout in the bootstrap before the closed-loop readiness gate.
    plan_env["BLUEPRINT_OSCAR_WAM_SOURCE_URL"] = OFFICIAL_OSCAR_SOURCE_URL
    plan_env["BLUEPRINT_OSCAR_WAM_SOURCE_REF"] = OFFICIAL_OSCAR_SOURCE_COMMIT
    plan_env[OSCAR_CUDNN_LIB_DIR_ENV] = OSCAR_CUDNN_LIB_DIR
    plan_env.update(offline_runtime_environment(OSCAR_RUNTIME_ASSET_CACHE_ROOT))
    plan_env[SNAPSHOT_ENV] = SNAPSHOT_DEFAULT_PATH
    plan_env[BRIDGE_REQUIRED_ENV] = "true"
    plan_env[CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV] = (
        CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH
    )
    (
        runtime_overlay_script,
        runtime_overlay_sha256,
        runtime_overlay_payload,
        runtime_overlay_source_sha256s,
    ) = _runtime_package_overlay_script()
    plan_env[RUNTIME_PACKAGE_OVERLAY_SHA256_ENV] = runtime_overlay_sha256
    backend_source_path = Path(__file__).with_name("isaac_runtime_task_backend.py")
    if not backend_source_path.is_file() or backend_source_path.is_symlink():
        raise ValueError("single_episode_isaac_runtime_backend_source_missing_or_unsafe")
    backend_source = backend_source_path.read_bytes()
    (
        overlay_script,
        backend_overlay_sha256,
        backend_overlay_payload,
    ) = _runtime_backend_overlay_script(backend_source=backend_source)
    plan_env["BLUEPRINT_ISAAC_RUNTIME_BACKEND_OVERLAY_SHA256"] = backend_overlay_sha256
    plan["env"] = plan_env
    plan["isaac_task_executor_command"] = _scope_command_pythonpath(
        _pin_isaac_executor_to_runtime_overlay(
            [str(item) for item in plan.get("isaac_task_executor_command") or []]
        ),
        plan_env["PYTHONPATH"],
    )
    groot_server_command = [
        str(item) for item in plan.get("groot_server_command") or []
    ]
    plan["groot_server_command"] = _scope_command_pythonpath(
        groot_server_command,
        plan_env[GROOT_RUNTIME_PYTHONPATH_ENV],
    )
    gear_sonic_controller_command = [
        str(item) for item in plan.get("gear_sonic_controller_command") or []
    ]
    if not gear_sonic_controller_command:
        raise ValueError("single_episode_gear_sonic_controller_command_missing")
    # Start the official controller immediately beside GR00T and Isaac.  The
    # controller and native DDS bridge both initialize Unitree's participant,
    # so making either process wait for the other's ready signal deadlocks cold
    # startup.  GEAR_SONIC_READY_SCRIPT remains the fail-closed authority: the
    # episode cannot start until the controller reports Init Done and the exact
    # hash-bound bridge publishes fresh, advancing state from this Isaac task.
    # deploy.sh starts ``just`` and the compiled controller below its shell.
    # A PID-only stop can therefore strand the compiled child and its 5557
    # listener across qualification attempts.  The refreshable runtime
    # supervisor creates a private child session, reaps that complete owned
    # tree, and only clears an inherited listener when its launch-session and
    # older qualification nonce prove it belongs to this same evaluation.
    plan["gear_sonic_controller_command"] = [
        OSCAR_PYTHON,
        "-m",
        GEAR_SONIC_PROCESS_SUPERVISOR_MODULE,
        "supervise",
        "--",
        *gear_sonic_controller_command,
    ]
    oscar_runtime_pythonpath = _oscar_runtime_pythonpath(plan_env["PYTHONPATH"])
    plan["closed_loop_command"] = [
        "env",
        f"PYTHONPATH={oscar_runtime_pythonpath}",
        "timeout",
        str(EPISODE_TIMEOUT_SECONDS),
        *command,
    ]
    script = build_worker_bootstrap_script(plan)
    eager_frame_materialization = """Path("/workspace/initial_policy_frame.png").write_bytes(
    base64.b64decode(os.environ["BLUEPRINT_INITIAL_POLICY_FRAME_B64"])
)"""
    live_observation_output_reset = f"""for live_output_name in (
    {INITIAL_POLICY_FRAME_PATH!r},
    {CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_PATH!r},
):
    live_output = Path(live_output_name)
    if live_output.is_symlink() or live_output.is_file():
        live_output.unlink()
    elif live_output.exists():
        raise RuntimeError("live_isaac_observation_output_not_regular_file:" + live_output_name)"""
    if script.count(eager_frame_materialization) != 1:
        raise ValueError("single_episode_frame_materialization_marker_ambiguous")
    script = script.replace(
        eager_frame_materialization,
        live_observation_output_reset,
        1,
    )
    input_marker = "upload_phase inputs_ready\n"
    if script.count(input_marker) != 1:
        raise ValueError("single_episode_bootstrap_input_marker_ambiguous")
    script = script.replace(
        input_marker,
        (
            "cp /workspace/attempt_input_manifest_episode_001.json "
            "/workspace/attempt_input_manifest.json\n"
            + (
                _qualification_checkpoint_restore_script(restore_evidence)
                + qualification_checkpoint_preflight_script()
                if restore_evidence
                else ""
            )
            + runtime_overlay_script
            + _oscar_runtime_provenance_script()
            + _oscar_runtime_dependency_repair_script()
            + _oscar_runtime_asset_prepare_and_preflight_script()
            + _oscar_runtime_import_preflight_script()
            + overlay_script
            + bridge_prepare_script()
            + _task_contract_overlay_script(
                resolved_contract_bytes=resolved_task_contract_bytes,
                source_contract_sha256=hashlib.sha256(task_contract_raw).hexdigest(),
                resolution=task_contract_resolution,
            )
            + input_marker
        ),
        1,
    )
    old_manifest = (
        'manifest_path = Path("/workspace/closed_loop_out/oscar_isaac_closed_loop_manifest.json")'
    )
    new_manifest = (
        'manifest_path = Path("/workspace/closed_loop_out/episode_001/'
        'oscar_isaac_closed_loop_manifest.json")'
    )
    if script.count(old_manifest) != 1:
        raise ValueError("single_episode_result_manifest_marker_ambiguous")
    script = script.replace(old_manifest, new_manifest, 1)
    terminal_upload_marker = """upload_phase() {
  if [ \"$1\" = \"runner_done\" ] || [ \"$1\" = \"runner_timeout\" ]; then
    BLUEPRINT_BOOTSTRAP_PHASE=\"$1\" python /workspace/upload_progress.py
  else
    BLUEPRINT_BOOTSTRAP_PHASE=\"$1\" python /workspace/upload_progress.py || true
  fi
}
"""
    terminal_upload_replacement = """upload_phase() {
  if [ \"$1\" = \"runner_done\" ] || [ \"$1\" = \"runner_timeout\" ]; then
    BLUEPRINT_BOOTSTRAP_PHASE=\"$1\" python /workspace/upload_progress.py
    # RunPod keeps a requested RUNNING pod under an Always-style restart
    # policy. Exiting immediately after a terminal upload can overwrite the
    # only failure or success bundle with a fresh container_bash_started
    # heartbeat before the direct owner polls it. Freeze the terminal bundle
    # until that owner or its watchdog retrieves the evidence and deletes the
    # pod.
    if declare -F cleanup >/dev/null 2>&1; then cleanup; fi
    while :; do sleep 300; done
  else
    BLUEPRINT_BOOTSTRAP_PHASE=\"$1\" python /workspace/upload_progress.py || true
  fi
}
"""
    if script.count(terminal_upload_marker) != 1:
        raise ValueError("single_episode_terminal_upload_marker_ambiguous")
    script = script.replace(
        terminal_upload_marker,
        terminal_upload_replacement,
        1,
    )
    closed_loop_result_marker = (
        'RC=$?\nset -e\n\nBLUEPRINT_CLOSED_LOOP_RC="$RC" python /workspace/write_result.py'
    )
    if script.count(closed_loop_result_marker) != 1:
        raise ValueError("single_episode_review_marker_ambiguous")
    script = script.replace(
        closed_loop_result_marker,
        (
            "EPISODE_RC=$?\n"
            f"env PYTHONPATH={shlex.quote(oscar_runtime_pythonpath)} "
            "/opt/oscar-venv/bin/python -m blueprint_pipeline.groot_oscar_episode_review "
            "/workspace/closed_loop_out/episode_001\n"
            "REVIEW_RC=$?\n"
            "RC=$EPISODE_RC\n"
            'if [ "$RC" -eq 0 ]; then RC=$REVIEW_RC; fi\n'
            "set -e\n\n"
            'BLUEPRINT_CLOSED_LOOP_RC="$RC" python /workspace/write_result.py'
        ),
        1,
    )
    if "attempt_input_manifest_smoke" in script:
        raise ValueError("single_episode_bootstrap_contains_smoke_execution")
    script = _inject_gear_sonic_isaac_dds_bridge(script)
    if BRIDGE_HEARTBEAT_PATH not in script or BRIDGE_LOG_PATH not in script:
        raise ValueError("single_episode_isaac_dds_bridge_artifacts_missing")
    script = _pin_bootstrap_interpreters(script)
    return {
        "plan": plan,
        "route": dict(route),
        "seed": dict(seed) if isinstance(seed, Mapping) else {},
        "start_frame": start_frame,
        "attempt": dict(attempt),
        "task_contract": resolved_task_contract,
        "source_task_contract_sha256": hashlib.sha256(task_contract_raw).hexdigest(),
        "task_contract_resolution": task_contract_resolution,
        "task_prompt": TASK_PROMPT,
        "manipulation_policy_task_compatibility": (
            manipulation_policy_task_compatibility
        ),
        "qualification_checkpoint_restore": restore_evidence or None,
        "perception_target_prompts": perception_target_prompts,
        "runtime_package_overlay_sha256": runtime_overlay_sha256,
        "runtime_package_overlay_xz_base64": runtime_overlay_payload,
        "runtime_package_overlay_source_sha256s": runtime_overlay_source_sha256s,
        "isaac_runtime_backend_overlay_sha256": backend_overlay_sha256,
        "isaac_runtime_backend_overlay_gzip_base64": backend_overlay_payload,
        "bootstrap_script": script,
        "bundle_sha256": digest,
    }


def _write_materialized_inputs(root: Path, inputs: Mapping[str, Any]) -> tuple[Path, Path]:
    start_frame = root / "initial_policy_frame.png"
    route = root / "route.json"
    start_frame.write_bytes(bytes(inputs["start_frame"]))
    route.write_text(json.dumps(inputs["route"], indent=2) + "\n", encoding="utf-8")
    return start_frame, route


def _validate_collected_final_review(root: Path) -> dict[str, Any]:
    """Validate the retrieved episode review independently of worker status."""
    episode_dir = root / "closed_loop_output" / "closed_loop_out" / "episode_001"
    validation_path = episode_dir / "final_review_validation.json"
    video_path = episode_dir / "final_review.mp4"
    blockers: list[str] = []
    validation: dict[str, Any] = {}
    if not validation_path.is_file():
        blockers.append("single_episode_final_review_validation_missing")
    else:
        try:
            value = json.loads(validation_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                validation = dict(value)
            else:
                blockers.append("single_episode_final_review_validation_not_object")
        except (OSError, json.JSONDecodeError):
            blockers.append("single_episode_final_review_validation_unreadable")

    if validation:
        if validation.get("schema_version") != "groot_oscar_episode_review_validation.v1":
            blockers.append("single_episode_final_review_validation_schema_invalid")
        if validation.get("status") != "passed":
            blockers.append("single_episode_final_review_not_passed")
        if list(validation.get("blockers") or []):
            blockers.append("single_episode_final_review_reports_blockers")
        if validation.get("episode_order_verified") is not True:
            blockers.append("single_episode_final_review_order_not_verified")
        if validation.get("review_source") != ("persistent_same_session_isaac_execution_frames"):
            blockers.append("single_episode_final_review_source_invalid")
        if validation.get("execution_truth") is not True:
            blockers.append("single_episode_final_review_not_execution_truth")
        if validation.get("same_session_isaac_frames") is not True:
            blockers.append("single_episode_final_review_not_same_session_isaac")
        if (
            validation.get("primary_camera_role") != "robot_pov"
            or validation.get("overview_excluded_from_primary_review") is not True
            or validation.get("concat_mode")
            != "primary_same_session_isaac_robot_pov_only"
        ):
            blockers.append("single_episode_final_review_primary_not_robot_pov_only")
        if list(validation.get("required_camera_roles") or []) != [
            "overview",
            "robot_pov",
        ]:
            blockers.append("single_episode_final_review_camera_roles_invalid")
        try:
            trace_count = int(validation.get("trace_step_count") or 0)
            clip_count = int(validation.get("ordered_clip_count") or 0)
            ordered_indices = [int(value) for value in validation.get("ordered_step_indices") or []]
            width = int(validation.get("width") or 0)
            height = int(validation.get("height") or 0)
            frame_count = int(validation.get("frame_count") or 0)
            duration = float(validation.get("duration_seconds") or 0)
        except (TypeError, ValueError):
            blockers.append("single_episode_final_review_metadata_invalid")
            trace_count = clip_count = width = height = frame_count = 0
            ordered_indices = []
            duration = 0.0
        if trace_count < 1 or clip_count != trace_count:
            blockers.append("single_episode_final_review_clip_count_invalid")
        if ordered_indices != list(range(1, trace_count + 1)):
            blockers.append("single_episode_final_review_step_order_invalid")
        if width != 640 or height != 480:
            blockers.append("single_episode_final_review_resolution_invalid")
        if frame_count < 1 or duration <= 0:
            blockers.append("single_episode_final_review_video_empty")

        expected_review_frame_count = 0
        frame_evidence = validation.get("isaac_frame_evidence")
        if not isinstance(frame_evidence, Mapping):
            blockers.append("single_episode_final_review_frame_evidence_missing")
        else:
            if frame_evidence.get("status") != "passed":
                blockers.append("single_episode_final_review_frame_evidence_not_passed")
            if list(frame_evidence.get("blockers") or []):
                blockers.append("single_episode_final_review_frame_evidence_reports_blockers")
            if len(list(frame_evidence.get("bound_steps") or [])) != trace_count:
                blockers.append("single_episode_final_review_bound_step_count_invalid")
            try:
                expected_review_frame_count = int(
                    frame_evidence.get("ordered_review_frame_count") or 0
                )
                review_indices = [
                    int(value) for value in frame_evidence.get("ordered_review_frame_indices") or []
                ]
                review_control_indices = [
                    int(value)
                    for value in frame_evidence.get("ordered_review_control_frame_indices") or []
                ]
                top_level_review_frame_count = int(
                    validation.get("ordered_review_frame_count") or 0
                )
            except (TypeError, ValueError):
                expected_review_frame_count = top_level_review_frame_count = 0
                review_indices = []
                review_control_indices = []
            if (
                expected_review_frame_count < 2
                or expected_review_frame_count != frame_count
                or top_level_review_frame_count != expected_review_frame_count
                or review_indices != list(range(expected_review_frame_count))
                or len(review_control_indices) != expected_review_frame_count
                or not review_control_indices
                or review_control_indices[0] != 0
            ):
                blockers.append("single_episode_final_review_sampled_frame_horizon_invalid")

        role_videos = validation.get("isaac_role_videos")
        if not isinstance(role_videos, Mapping):
            blockers.append("single_episode_final_review_role_videos_missing")
        else:
            expected_role_files = {
                "overview": "isaac_overview_review.mp4",
                "robot_pov": "isaac_robot_pov_review.mp4",
            }
            for role, filename in expected_role_files.items():
                role_evidence = role_videos.get(role)
                if not isinstance(role_evidence, Mapping):
                    blockers.append(f"single_episode_final_review_{role}_video_evidence_missing")
                    continue
                if role_evidence.get("status") != "passed" or list(
                    role_evidence.get("blockers") or []
                ):
                    blockers.append(f"single_episode_final_review_{role}_video_not_passed")
                if Path(str(role_evidence.get("path") or "")).name != filename:
                    blockers.append(f"single_episode_final_review_{role}_video_path_invalid")
                try:
                    role_frame_count = int(role_evidence.get("frame_count") or 0)
                    role_width = int(role_evidence.get("width") or 0)
                    role_height = int(role_evidence.get("height") or 0)
                except (TypeError, ValueError):
                    role_frame_count = role_width = role_height = 0
                if role_frame_count != expected_review_frame_count:
                    blockers.append(f"single_episode_final_review_{role}_frame_count_invalid")
                if role_width != 640 or role_height != 480:
                    blockers.append(f"single_episode_final_review_{role}_resolution_invalid")
                local_role_video = episode_dir / filename
                expected_role_sha = str(role_evidence.get("sha256") or "").lower()
                if (
                    not local_role_video.is_file()
                    or local_role_video.stat().st_size <= 0
                    or expected_role_sha != _sha256(local_role_video)
                ):
                    blockers.append(f"single_episode_final_review_{role}_sha256_invalid")

        wam_review = validation.get("wam_prediction_review")
        if not isinstance(wam_review, Mapping):
            blockers.append("single_episode_wam_prediction_review_missing")
        else:
            if wam_review.get("status") != "passed" or list(wam_review.get("blockers") or []):
                blockers.append("single_episode_wam_prediction_review_not_passed")
            if wam_review.get("review_source") != "oscar_wam_predicted_rollout_clips":
                blockers.append("single_episode_wam_prediction_review_source_invalid")
            if Path(str(wam_review.get("path") or "")).name != ("wam_prediction_review.mp4"):
                blockers.append("single_episode_wam_prediction_review_path_invalid")
            wam_video_path = episode_dir / "wam_prediction_review.mp4"
            wam_validation_path = episode_dir / "wam_prediction_review_validation.json"
            standalone_wam: dict[str, Any] = {}
            if not wam_validation_path.is_file() or wam_validation_path.is_symlink():
                blockers.append("single_episode_wam_prediction_validation_missing")
            else:
                try:
                    raw_wam = json.loads(wam_validation_path.read_text(encoding="utf-8"))
                    standalone_wam = dict(raw_wam) if isinstance(raw_wam, Mapping) else {}
                except (OSError, json.JSONDecodeError):
                    standalone_wam = {}
                if not standalone_wam:
                    blockers.append("single_episode_wam_prediction_validation_unreadable")
            if standalone_wam:
                if standalone_wam != dict(wam_review):
                    blockers.append("single_episode_wam_prediction_validation_mismatch")
                raw_executed_durations = standalone_wam.get(
                    "executed_prefix_duration_seconds_by_step"
                )
                try:
                    executed_durations = [
                        float(value) for value in raw_executed_durations
                    ]
                except (TypeError, ValueError):
                    executed_durations = []
                expected_executed_duration = sum(executed_durations)
                try:
                    declared_expected_duration = float(
                        standalone_wam.get(
                            "expected_executed_timeline_duration_seconds"
                        )
                    )
                    observed_wam_duration = float(
                        standalone_wam.get("duration_seconds")
                    )
                except (TypeError, ValueError):
                    declared_expected_duration = observed_wam_duration = -1.0
                executed_timeline_valid = bool(
                    len(executed_durations) == trace_count
                    and all(
                        math.isfinite(value) and value > 0
                        for value in executed_durations
                    )
                    and math.isfinite(declared_expected_duration)
                    and abs(declared_expected_duration - expected_executed_duration)
                    <= 1e-6
                    and math.isfinite(observed_wam_duration)
                    and abs(observed_wam_duration - expected_executed_duration)
                    <= (2.0 / 15.0)
                )
                if (
                    standalone_wam.get("schema_version")
                    != "groot_oscar_wam_prediction_review_validation.v1"
                    or standalone_wam.get("status") != "passed"
                    or list(standalone_wam.get("blockers") or [])
                    or standalone_wam.get("review_source") != "oscar_wam_predicted_rollout_clips"
                    or int(standalone_wam.get("trace_step_count") or 0) != trace_count
                    or int(standalone_wam.get("ordered_clip_count") or 0) != trace_count
                    or [int(value) for value in standalone_wam.get("ordered_step_indices") or []]
                    != ordered_indices
                    or standalone_wam.get("episode_order_verified") is not True
                    or standalone_wam.get("video_frame_count_mode")
                    != "dynamic_from_executed_controller_duration"
                    or standalone_wam.get("prediction_review_timeline_mode")
                    != "executed_control_prefix_per_decision"
                    or standalone_wam.get(
                        "full_prediction_horizons_preserved_in_source_clips"
                    )
                    is not True
                    or standalone_wam.get(
                        "overlapping_unexecuted_prediction_tails_excluded"
                    )
                    is not True
                    or not executed_timeline_valid
                ):
                    blockers.append("single_episode_wam_prediction_validation_invalid")
                expected_wam_sha256 = str(standalone_wam.get("sha256") or "").lower()
                if (
                    wam_video_path.is_symlink()
                    or not wam_video_path.is_file()
                    or wam_video_path.stat().st_size <= 0
                    or expected_wam_sha256 != _sha256(wam_video_path)
                ):
                    blockers.append("single_episode_wam_prediction_video_sha256_invalid")

    video_sha256: str | None = None
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        blockers.append("single_episode_final_review_video_missing_or_empty")
    else:
        video_sha256 = _sha256(video_path)
        expected_sha256 = str(validation.get("sha256") or "").strip().lower()
        if expected_sha256 != video_sha256:
            blockers.append("single_episode_final_review_sha256_mismatch")

    unique_blockers = sorted(set(blockers))
    return {
        "schema_version": "single_g1_kitchen_final_review_evidence.v1",
        "status": "passed" if not unique_blockers else "blocked",
        "blockers": unique_blockers,
        "validation_path": str(validation_path),
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "validation": validation,
    }


def _validate_collected_qualification_checkpoint(
    root: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not expected:
        return {"status": "not_requested", "blockers": []}
    path = (
        root
        / "closed_loop_output"
        / "closed_loop_out"
        / "qualification_checkpoint_restore.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    checks = {
        "report_present": bool(value),
        "schema_version_exact": value.get("schema_version")
        == QUALIFICATION_CHECKPOINT_RESTORE_SCHEMA_VERSION,
        "status_passed": value.get("status") == "passed",
        "checkpoint_path_exact": value.get("checkpoint_path") == REMOTE_FINAL_CHECKPOINT,
        "archive_sha256_exact": value.get("archive_sha256")
        == expected.get("archive_sha256"),
        "archive_size_exact": int(value.get("archive_size_bytes") or 0)
        == int(expected.get("archive_size_bytes") or -1),
        "part_count_exact": int(value.get("part_count") or 0)
        == len(expected.get("parts") or []),
        "raw_signed_urls_not_recorded": value.get("raw_signed_urls_recorded") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "blocked",
        "path": str(path),
        "checks": checks,
        "blockers": [f"qualification_checkpoint_restore_{name}" for name in failed],
    }


def _validate_collected_semantic_success(root: Path) -> dict[str, Any]:
    path = (
        root
        / "closed_loop_output"
        / "closed_loop_out"
        / "episode_001"
        / "oscar_isaac_closed_loop_manifest.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    proof = value.get("success_proof")
    proof = dict(proof) if isinstance(proof, Mapping) else {}
    checks = {
        "manifest_present": bool(value),
        "manifest_completed": value.get("status") == "completed",
        "manipulation_success_proven": value.get("manipulation_success_proven") is True,
        "task_target_reached": value.get("task_target_reached") is True,
        "success_answer_yes": proof.get("answer") in {"yes", "proven"},
        "target_manipulation_succeeded": proof.get("did_target_manipulation_succeed")
        is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "blocked",
        "path": str(path),
        "checks": checks,
        "steps_executed": value.get("steps_executed"),
        "steps_requested": value.get("steps_requested"),
        "blockers": [f"single_episode_semantic_{name}" for name in failed],
    }


def run_single_episode(
    *,
    provider_name: str = "runpod",
    episode_bundle: str | Path,
    provider_bundle_url_file: str | Path,
    provider_output_put_url_file: str | Path,
    provider_output_get_url_file: str | Path,
    provider_bootstrap_url_file: str | Path | None = None,
    release_evidence: str | Path,
    provider_launch_request: str | Path,
    preflight_bundle: str | Path,
    admission_out: str | Path,
    bound_request_out: str | Path,
    adapter_output: str | Path,
    pod_name: str,
    execute: bool,
    qualification_checkpoint_report: str | Path | None = None,
    qualification_checkpoint_part_stage_dirs: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    result_path = Path(adapter_output).expanduser().resolve()
    root = result_path.parent
    ensure_dir(root)
    bundle_path = Path(episode_bundle).expanduser().resolve()
    blockers: list[str] = []
    if IMAGE_DIGEST not in OSCAR_RUNTIME_SOURCE_SEAL_CAPABLE_IMAGE_DIGESTS:
        blockers.append("single_episode_pinned_image_missing_oscar_runtime_source_seal")
    manipulation_policy_task_compatibility: dict[str, Any] = {}
    qualification_checkpoint_restore: dict[str, Any] = {}
    qualification_checkpoint_part_get_urls: list[str] = []
    try:
        bundle_url = _read_secret_url_file(provider_bundle_url_file)
        put_url = _read_secret_url_file(provider_output_put_url_file)
        get_url = _read_secret_url_file(provider_output_get_url_file)
        if qualification_checkpoint_report not in {None, ""}:
            (
                qualification_checkpoint_restore,
                qualification_checkpoint_part_get_urls,
            ) = _qualification_checkpoint_restore_evidence(
                worker_report_path=qualification_checkpoint_report,
                part_stage_dirs=tuple(qualification_checkpoint_part_stage_dirs),
            )
        elif qualification_checkpoint_part_stage_dirs:
            raise ValueError("qualification_checkpoint_worker_report_missing")
        inputs = (
            _load_single_episode_inputs(
                bundle_path,
                qualification_checkpoint_restore=qualification_checkpoint_restore,
                remote_bundle_url=(bundle_url if not bundle_path.is_file() else None),
            )
            if qualification_checkpoint_restore
            else _load_single_episode_inputs(bundle_path)
        )
        manipulation_policy_task_compatibility = dict(
            inputs.get("manipulation_policy_task_compatibility") or {}
        )
        if (
            manipulation_policy_task_compatibility.get("status") != "qualified"
            and not qualification_checkpoint_restore
        ):
            blockers.extend(
                manipulation_policy_task_compatibility.get("blockers")
                or [MANIPULATION_POLICY_TASK_COMPATIBILITY_BLOCKER]
            )
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        inputs = {}
        bundle_url = put_url = get_url = ""
        blockers.append(str(exc))
    signed_output_staging_proof: dict[str, Any] = {
        "status": "not_checked_dry_run",
        "required_before_provider_allocation": True,
        "raw_signed_urls_recorded": False,
    }
    if execute and inputs and put_url and get_url:
        try:
            signed_output_staging_proof = _require_signed_output_staging_proof(
                provider_output_put_url_file=provider_output_put_url_file,
                provider_output_get_url_file=provider_output_get_url_file,
                put_url=put_url,
                get_url=get_url,
            )
        except (OSError, ValueError) as exc:
            signed_output_staging_proof = {
                "status": "blocked",
                "blockers": str(exc).split(";"),
                "raw_signed_urls_recorded": False,
            }
            blockers.extend(signed_output_staging_proof["blockers"])
    release_path = Path(release_evidence).expanduser().resolve()
    if not release_path.is_file():
        blockers.append("single_episode_release_evidence_missing")
    else:
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            release = {}
        if release.get("resolved_digest_ref") != IMAGE_REF:
            blockers.append("single_episode_release_evidence_image_mismatch")

    resolved_provider_name = str(provider_name or "").strip().lower()
    if resolved_provider_name not in SUPPORTED_PROVIDERS:
        blockers.append(f"single_episode_provider_unsupported:{resolved_provider_name}")
        resolved_provider_name = provider_name
    provider = get_render_provider(resolved_provider_name)
    bootstrap_url = ""
    bootstrap_artifact: dict[str, Any] = {}
    bootstrap_staging_required = False
    signed_bootstrap_required = resolved_provider_name == "vast" or bool(
        resolved_provider_name == "runpod" and qualification_checkpoint_restore
    )
    if inputs and signed_bootstrap_required:
        bootstrap_artifact = _materialize_remote_bootstrap(
            root, inputs, provider_name=str(resolved_provider_name)
        )
        if provider_bootstrap_url_file:
            try:
                bootstrap_url = _read_secret_url_file(provider_bootstrap_url_file)
            except (OSError, ValueError) as exc:
                blockers.append(str(exc))
        else:
            bootstrap_staging_required = True
    requested_pod_name = pod_name.strip()
    name_suffix = (
        hashlib.sha256(requested_pod_name.encode("utf-8")).hexdigest()[:10]
        if requested_pod_name
        else uuid.uuid4().hex[:10]
    )
    prefix = f"blueprint-groot-oscar-canary-single-episode-{name_suffix}"
    resolved_name = f"{prefix}-pod"
    request: dict[str, Any] = {}
    capacity: dict[str, Any] = {}
    pre_inventory: dict[str, Any] = {}
    launch_session_id = ""
    launch_session_nonce_artifact: dict[str, Any] = {}
    if inputs and not blockers and not bootstrap_staging_required:
        (root / "provider_bundle_url.txt").write_text(bundle_url, encoding="utf-8")
        (root / "provider_output_put_url.txt").write_text(put_url, encoding="utf-8")
        (root / "provider_output_get_url.txt").write_text(get_url, encoding="utf-8")
        for path in (
            root / "provider_bundle_url.txt",
            root / "provider_output_put_url.txt",
            root / "provider_output_get_url.txt",
        ):
            path.chmod(0o600)
        start_frame, _route_path = _write_materialized_inputs(root, inputs)
        launch_session_id = f"single-g1-kitchen-{name_suffix}"
        launch_session_nonce_artifact = _materialize_launch_session_nonce(root, launch_session_id)
        spec = build_launch_spec(
            job_dir=root,
            image_ref=IMAGE_REF,
            start_frame=start_frame,
            route_payload=inputs["route"],
            task_prompt=TASK_PROMPT,
            plan=inputs["plan"],
            launch_nonce=launch_session_id,
            seed_provenance=inputs["seed"],
            container_disk_gb=220,
            volume_gb=_workspace_volume_gb(resolved_provider_name),
            max_hourly_rate_usd=MAX_HOURLY_RATE_USD,
        )
        spec.name = resolved_name
        spec.gpu_types = (
            QUALIFICATION_RUNPOD_GPU_TYPES
            if resolved_provider_name == "runpod" and qualification_checkpoint_restore
            else GPU_TYPES
        )
        spec.min_gpu_ram_mb = 40_000
        if spec.env.get("BLUEPRINT_LAUNCH_SESSION_ID") != launch_session_id:
            raise ValueError("single_episode_launch_session_nonce_binding_mismatch")
        if signed_bootstrap_required:
            spec.bootstrap_argv = ["-lc", _vast_signed_bootstrap_downloader_script()]
        else:
            spec.bootstrap_argv = [
                "-lc",
                _bind_provider_allocation_identity(
                    str(inputs["bootstrap_script"]), resolved_provider_name
                ),
            ]
        # build_launch_spec retains the legacy staged-frame input contract, but
        # this direct lane never sends or consumes that frame at runtime. Isaac
        # writes the live frame and matching projection context before its
        # readiness marker; the bootstrap removed any bundle-extracted copies.
        spec.env.pop("BLUEPRINT_INITIAL_POLICY_FRAME_B64", None)
        spec.env.update(
            {
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "BLUEPRINT_SOURCE_COMMIT": SOURCE_COMMIT,
                "BLUEPRINT_SINGLE_EPISODE_ATTEMPT_ID": "episode_001",
            }
        )
        if qualification_checkpoint_restore:
            spec.env[QUALIFICATION_CHECKPOINT_PART_GET_URLS_ENV] = json.dumps(
                qualification_checkpoint_part_get_urls,
                separators=(",", ":"),
            )
        if signed_bootstrap_required:
            spec.env[VAST_BOOTSTRAP_URL_ENV] = bootstrap_url
            spec.env[VAST_BOOTSTRAP_SHA256_ENV] = str(bootstrap_artifact["sha256"])
            spec.env.pop(RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV, None)
            spec.env.pop(ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV, None)
        if not signed_bootstrap_required:
            spec.env.update(
                {
                    RUNTIME_PACKAGE_OVERLAY_PAYLOAD_ENV: str(
                        inputs["runtime_package_overlay_xz_base64"]
                    ),
                    ISAAC_RUNTIME_OVERLAY_PAYLOAD_ENV: str(
                        inputs["isaac_runtime_backend_overlay_gzip_base64"]
                    ),
                }
            )
        request = provider.build_request(spec, root)
        request["min_gpu_ram_mb"] = 40_000
        request["requires_rtx"] = True
        if resolved_provider_name == "vast":
            request.update(
                {
                    "bootstrap_transport": "signed_https_sha256",
                    "remote_bootstrap_sha256": bootstrap_artifact["sha256"],
                    "remote_bootstrap_size_bytes": bootstrap_artifact["size_bytes"],
                    "require_avx": True,
                    "min_reliability": VAST_MIN_RELIABILITY,
                    "require_known_supported_isaac_driver": (
                        VAST_REQUIRE_KNOWN_SUPPORTED_ISAAC_DRIVER
                    ),
                    "preferred_gpu_keywords": list(VAST_PREFERRED_GPU_KEYWORDS),
                }
            )
        pre_inventory = provider.billable_inventory(name_prefix="")
        capacity = provider.capacity_preflight(request)
        viable = [
            row
            for row in capacity.get("viable_gpu_types", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("on_demand_price_usd_per_hour"), (int, float))
            and float(row["on_demand_price_usd_per_hour"]) <= MAX_HOURLY_RATE_USD
        ]
        if pre_inventory.get("api_confirmed") is not True:
            blockers.append("single_episode_prelaunch_inventory_unverified")
        elif pre_inventory.get("live_resource_count") != 0:
            blockers.append(f"single_episode_prelaunch_{resolved_provider_name}_inventory_not_zero")
        if capacity.get("status") != "available" or not viable:
            blockers.append("single_episode_48gb_rtx_capacity_unavailable")
        pre_spend_preflight, pre_spend_blockers = qualification_pre_spend_preflight(
            root=root, capacity=capacity, pre_inventory=pre_inventory,
            image_ref=IMAGE_REF, execute=execute, provider=resolved_provider_name,
        )
        blockers.extend(pre_spend_blockers)
        request["pre_spend_preflight"] = pre_spend_preflight
        request["prelaunch_spend_guard"] = {
            "required_before_provider_launch": True,
            "can_launch": not blockers,
            "blockers": sorted(set(blockers)),
            "max_hourly_rate_usd": MAX_HOURLY_RATE_USD,
            "maximum_live_seconds": WALL_SECONDS,
            "maximum_estimated_spend_usd": round(MAX_HOURLY_RATE_USD * WALL_SECONDS / 3600.0, 2),
        }

    reported_blockers = list(blockers)
    reported_blockers += ["single_episode_bootstrap_staging_required"] if bootstrap_staging_required else []
    preflight = {
        "schema_version": "single_g1_kitchen_episode_preflight.v1",
        "status": (
            "bootstrap_staging_required"
            if bootstrap_staging_required and not blockers
            else ("ready" if not blockers else "blocked")
        ),
        "provider": resolved_provider_name,
        "image_ref": IMAGE_REF,
        "bundle_sha256": inputs.get("bundle_sha256"),
        "runtime_package_overlay_sha256": inputs.get("runtime_package_overlay_sha256"),
        "runtime_package_overlay_source_sha256s": inputs.get(
            "runtime_package_overlay_source_sha256s"
        ),
        "attempt_id": "episode_001",
        "task": "microwave_door",
        "manipulation_policy_task_compatibility": (
            manipulation_policy_task_compatibility or None
        ),
        "qualification_checkpoint_restore": qualification_checkpoint_restore or None,
        "episode_step_cap": 48,
        "oscar_seed": 1001,
        "network_volume_used": False,
        "capacity": capacity,
        "pre_inventory": pre_inventory,
        "bootstrap_artifact": bootstrap_artifact or None,
        "launch_session_nonce_artifact": launch_session_nonce_artifact or None,
        "signed_output_staging_proof": signed_output_staging_proof,
        "blockers": sorted(set(reported_blockers)),
    }
    admission = {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not reported_blockers else "blocked",
        "resource_class": "gpu_render",
        "scope": "one_g1_kitchen_groot_sonic_oscar_episode",
        "manipulation_policy_task_compatibility": (
            manipulation_policy_task_compatibility or None
        ),
        "qualification_checkpoint_restore": qualification_checkpoint_restore or None,
        "provider_mutations_performed": 0,
        "blockers": sorted(set(reported_blockers)),
        "raw_secret_values_recorded": False,
    }
    bound = {
        "schema_version": "single_g1_kitchen_episode_bound_request.v1",
        "status": "bound" if not reported_blockers else "blocked",
        "provider": resolved_provider_name,
        "pod_name": resolved_name,
        "pod_name_prefix": prefix,
        "image_ref": IMAGE_REF,
        "bundle_sha256": inputs.get("bundle_sha256"),
        "runtime_package_overlay_sha256": inputs.get("runtime_package_overlay_sha256"),
        "gpu_type_ids": (
            list(QUALIFICATION_RUNPOD_GPU_TYPES)
            if resolved_provider_name == "runpod" and qualification_checkpoint_restore
            else list(GPU_TYPES) if resolved_provider_name == "runpod" else []
        ),
        "vast_offer_requirements": (
            capacity.get("selection_policy") if resolved_provider_name == "vast" else None
        ),
        "vast_bound_offer": (
            capacity.get("selected_offer") if resolved_provider_name == "vast" else None
        ),
        "bootstrap_transport": (
            "signed_https_sha256"
            if signed_bootstrap_required
            else request.get("bootstrap_transport")
        ),
        "bootstrap_artifact": bootstrap_artifact or None,
        "launch_session_id": launch_session_id or None,
        "launch_session_nonce_sha256": launch_session_nonce_artifact.get("sha256"),
        "bootstrap_source_length": request.get("bootstrap_source_length"),
        "provider_args_length": request.get("provider_args_length"),
        "gpu_count": 1,
        "container_disk_gb": 220,
        "workspace_volume_gb": _workspace_volume_gb(resolved_provider_name),
        "workspace_storage_semantics": (
            "runpod_workspace_volume"
            if resolved_provider_name == "runpod"
            else "vast_instance_disk_container_disk_gb"
        ),
        "episode_attempt_id": "episode_001",
        "manipulation_policy_task_compatibility": (
            manipulation_policy_task_compatibility or None
        ),
        "qualification_checkpoint_restore": qualification_checkpoint_restore or None,
        "episode_seed": 1001,
        "signed_bundle_url_present": bool(bundle_url),
        "signed_output_urls_present": bool(put_url and get_url),
        "signed_output_staging_proof": signed_output_staging_proof,
        "blockers": sorted(set(reported_blockers)),
        "raw_secret_values_recorded": False,
    }
    write_json(Path(provider_launch_request), bound)
    write_json(Path(preflight_bundle), preflight)
    write_json(Path(admission_out), admission)
    write_json(Path(bound_request_out), bound)
    if not execute or reported_blockers:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "bootstrap_staging_required"
                if bootstrap_staging_required and not blockers
                else ("dry_run_ready" if not blockers else "blocked")
            ),
            "execute": bool(execute),
            "preflight": preflight,
            "admission": admission,
            "bound_request": bound,
            "provider_mutations_performed": 0,
            "bootstrap_artifact": bootstrap_artifact or None,
            "manipulation_policy_task_compatibility": (
                manipulation_policy_task_compatibility or None
            ),
            "qualification_checkpoint_restore": qualification_checkpoint_restore or None,
            "blockers": sorted(set(reported_blockers)),
        }
        write_json(result_path, result)
        return result

    grant = require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    started_at = time.time()
    deadline = started_at + WALL_SECONDS
    armed = arm_watchdog(
        out_dir=root,
        pod_name_prefix=prefix,
        deadline_epoch=deadline,
        provider_name=resolved_provider_name,
    )
    watchdog = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--out-dir",
            str(root),
            "--pod-name-prefix",
            prefix,
            "--deadline-epoch",
            str(deadline),
            "--provider",
            resolved_provider_name,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    direct_ownership = _open_direct_episode_owner(
        root=root,
        provider_name=resolved_provider_name,
        run_id=f"single-g1-kitchen-{name_suffix}",
        pod_name=resolved_name,
        pod_name_prefix=prefix,
        watchdog_deadline_epoch=deadline,
        watchdog_pid=watchdog.pid,
    )
    launch: dict[str, Any] = {}
    watch: dict[str, Any] = {}
    teardown: dict[str, Any] = {}
    ownership_settlement: dict[str, Any] = {}
    try:
        launch = provider.launch(
            root,
            request,
            cold=True,
            paid_resource_admission_grant=grant,
        )
        instance_id = str(launch.get("instance_id") or "")
        if instance_id:
            direct_ownership = _bind_direct_episode_owner(
                root=root,
                ownership=direct_ownership,
                instance_id=instance_id,
                watchdog_pid=watchdog.pid,
            )
            watch = watch_and_collect(
                root,
                root / "closed_loop_output",
                instance_id,
                provider=provider,
                max_seconds=WATCH_SECONDS,
                poll=20,
                stop_on_success=False,
                preserve_instance=False,
                preserve_blocked_instance=False,
                progress_timeout_seconds=PROGRESS_TIMEOUT_SECONDS,
                progress_stall_phases=SINGLE_EPISODE_PROGRESS_PHASES,
            )
    except BaseException as exc:
        if not launch.get("instance_id"):
            launch = {
                **launch,
                "status": launch.get("status") or "launch_or_watch_interrupted",
                "allocation_outcome_ambiguous": True, "error_type": type(exc).__name__,
            }
        watch = {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "blockers": ["single_episode_launch_or_watch_interrupted"],
        }
    finally:
        teardown = terminate_canary_resources(
            provider=provider,
            pod_name_prefix=prefix,
            armed=armed,
            provider_name=resolved_provider_name,
        )
        write_json(root / "single_episode_teardown.json", teardown)
        ownership_settlement = _settle_direct_episode_owner(
            root=root,
            ownership=direct_ownership,
            teardown=teardown,
            launch=launch,
        )
    final_global_inventory = provider.billable_inventory(name_prefix="")
    runner = watch.get("runner_result") if isinstance(watch, Mapping) else {}
    runner = dict(runner) if isinstance(runner, Mapping) else {}
    final_review = _validate_collected_final_review(root)
    qualification_checkpoint_validation = _validate_collected_qualification_checkpoint(
        root, qualification_checkpoint_restore
    )
    semantic_success = _validate_collected_semantic_success(root)
    run_blockers: list[str] = []
    if launch.get("status") != "launched":
        run_blockers.append("single_episode_pod_not_launched")
    if watch.get("status") != "completed":
        run_blockers.append("single_episode_worker_not_completed")
    if runner.get("status") != "completed":
        run_blockers.append("single_episode_runtime_result_not_completed")
    if final_review.get("status") != "passed":
        run_blockers.extend(final_review.get("blockers") or ["single_episode_final_review_invalid"])
    if qualification_checkpoint_restore:
        if qualification_checkpoint_validation.get("status") != "passed":
            run_blockers.extend(
                qualification_checkpoint_validation.get("blockers")
                or ["qualification_checkpoint_restore_not_proven"]
            )
        if semantic_success.get("status") != "passed":
            run_blockers.extend(
                semantic_success.get("blockers")
                or ["single_episode_semantic_success_not_proven"]
            )
    if teardown.get("provider_absence_confirmed") is not True:
        run_blockers.append("single_episode_teardown_not_proven")
    if (
        final_global_inventory.get("api_confirmed") is not True
        or final_global_inventory.get("live_resource_count") != 0
    ):
        run_blockers.append(f"single_episode_final_{resolved_provider_name}_inventory_not_zero")
    elapsed = max(0.0, time.time() - started_at)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not run_blockers else "blocked",
        "provider": resolved_provider_name,
        "image_ref": IMAGE_REF,
        "bundle_sha256": inputs.get("bundle_sha256"),
        "attempt_id": "episode_001",
        "task": "microwave_door",
        "manipulation_policy_task_compatibility": (
            manipulation_policy_task_compatibility or None
        ),
        "qualification_checkpoint_restore": qualification_checkpoint_restore or None,
        "episode_seed": 1001,
        "episode_step_cap": 48,
        "learned_oscar_wam_required": True,
        "external_forward_inverse_scorer_requested": False,
        "launch": launch,
        "watch": watch,
        "runner_result": runner,
        "final_review": final_review,
        "qualification_checkpoint_validation": qualification_checkpoint_validation,
        "semantic_success": semantic_success,
        "teardown": teardown,
        "direct_ownership": ownership_settlement.get("owner"),
        "pending_teardown_record": ownership_settlement.get("pending", {}).get("path"),
        "final_global_inventory": final_global_inventory,
        "watchdog_pid": watchdog.pid,
        "elapsed_seconds": round(elapsed, 3),
        "maximum_estimated_spend_usd": round(
            MAX_HOURLY_RATE_USD * min(elapsed, WALL_SECONDS) / 3600.0, 6
        ),
        "continuing_spend": not (
            teardown.get("provider_absence_confirmed") is True
            and final_global_inventory.get("api_confirmed") is True
            and final_global_inventory.get("live_resource_count") == 0
        ),
        "provider_mutations_performed": 1 if launch.get("instance_id") else 0,
        "blockers": sorted(set(run_blockers)),
        "proof_boundary": (
            "Completion requires the real single episode worker result, a hash-verified ordered "
            "final review video, and provider-zero teardown; model execution and semantic success "
            "remain fields of the retrieved "
            "episode manifest and are not inferred from allocation. The direct episode runs "
            "the learned OSCAR/WAM checkpoint but does not claim external forward/inverse "
            "consistency."
        ),
    }
    write_json(result_path, result)
    return result
