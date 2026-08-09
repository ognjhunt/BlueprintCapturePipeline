"""Build an immutable, scene-neutral Joint Agent + local OVRTX Vast bundle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .common import ensure_dir, utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .nvidia_nim_model_preflight import (
    DEFAULT_ENDPOINT as NIM_DEFAULT_ENDPOINT,
    DEFAULT_MODEL as NIM_DEFAULT_MODEL,
    SCHEMA_VERSION as NIM_PREFLIGHT_SCHEMA_VERSION,
)
from .paid_resource_admission import PaidResourceAdmissionGrant
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers
from .provider_bundle_rehearsal import (
    provider_bundle_rehearsal_blockers,
    rehearse_provider_bundle_entrypoint,
)
from .public_scene_execution_authority import validate_public_scene_execution_authority
from .usd_content_joint_agent_packet import inspect_joint_agent_checkout
from .vast_independent_watchdog_control import (
    arm_independent_vast_watchdog,
    close_independent_vast_watchdog,
)
from .vast_provider_adapter import run_vast_provider_adapter
from .vast_session_budget_contract import attempt_estimated_cost, attempt_runtime_seconds
from .wam_provider_object_store import (
    cleanup_staged_wam_provider_objects,
    stage_wam_provider_bundle_object_store,
)


PROVIDER_BUNDLE_KIND = "adp_joint_agent"
PROBE_KIND = "adp-usd-joint-agent"
SCHEMA_VERSION = "adp_joint_agent_provider_bundle.v2"
SOURCE_TREE = "d36ddaed4c3ea44ab81c9f8178ab40d2eb0f8fe3"
DEFAULT_IMAGE = (
    "docker.io/nvidia/cuda@"
    "sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10"
)
CONSTRUCTION_RENDER_MODE = "rt2"
CONSTRUCTION_RENDER_SENSOR_UPDATES = 32
RUNTIME_DISK_GB = 96
# The released OVRTX wheel is about 2.4 GiB compressed and is materialized in
# both uv's cache and the isolated runtime environment.  Agent dependencies,
# source, and retained outputs share that filesystem.  v12 began with 9.5 GiB
# free and failed partway through installation, so capacity is an admission
# input rather than something discovered one missing file at a time.
MINIMUM_FREE_BYTES_BEFORE_DEPENDENCY_INSTALL = 32 * 1024**3
# Exact public Scene Optimizer Core package the released optimize_usd step
# requires for its local backend. v8 proved the released CLI fails closed
# without it ("local backend unavailable"), so the bundle ships the pinned
# artifact instead of fetching at provider runtime.
SCENE_OPTIMIZER_CORE_PACKAGE = "scene_optimizer_core_usd_25.11_py_3.12"
SCENE_OPTIMIZER_CORE_RELEASE = "NVIDIA-Omniverse/usd-optimize v1.0.3"
SCENE_OPTIMIZER_CORE_LICENSE = "Apache-2.0"
SCENE_OPTIMIZER_CORE_SHA256 = (
    "sha256:9d98d22eed1eb31da3183bfd4155f3b8eca48576e6eb5947d126e781f0edc671"
)
RESULT_SCHEMA_VERSION = "adp_joint_agent_vast_run.v1"
REQUIRED_RETAINED_ARTIFACT_ROLES = frozenset(
    {
        "articulation_candidates",
        "candidate_bounds",
        "deterministic_articulation_review",
        "optimized_source",
        "owned_core_rigged_asset",
        "owned_core_diagnostics",
        "owned_core_validation",
    }
)
DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/joint-agent"
_VAST_MUTATION_ENV = (
    "BLUEPRINT_ALLOW_VAST_API_CALLS",
    "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH",
)
_VAST_SINGLE_ATTEMPT_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _canonical_receipt(path: Path, *, digest_field: str, error: str) -> dict[str, Any]:
    value = _read_json(path, error=error)
    if value.get(digest_field) != canonical_digest(value, digest_field=digest_field):
        raise ValueError(error)
    return value


def _write_executable(destination: Path, source: Path) -> None:
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _blueprint_identity(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _deterministic_zip(root: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(root.parent).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = (0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644) << 16
            archive.writestr(info, path.read_bytes())


def _provider_config(packet: Mapping[str, Any]) -> dict[str, Any]:
    config_path = Path(str((packet.get("config") or {}).get("path") or ""))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adp_joint_agent_config_invalid")
    config["project"]["working_dir"] = "runtime_output/joint_agent_work"
    config["input"]["usd_path"] = "input/articulated_source.usda"
    steps = config["steps"]
    steps["identify_asset"]["renderer"] = {"backend": "remote"}
    steps["build_dataset_usd"]["renderer"] = {"backend": "remote"}
    apply = steps["apply_joint_rigger"]
    apply.update(
        {
            "enabled": False,
            "adapter": "owned_core",
            "articulation_candidates_path": (
                "runtime_output/joint_agent_work/articulation_candidates/"
                "articulation_candidates.json"
            ),
            "output_usd_path": "runtime_output/joint_agent_work/joint_rigger/rigged.usdz",
            "diagnostics_path": (
                "runtime_output/joint_agent_work/joint_rigger/"
                "joint_rigger_diagnostics.json"
            ),
            "validation_path": (
                "runtime_output/joint_agent_work/joint_rigger/"
                "joint_rigger_validation.json"
            ),
            "apply_masses": False,
            "apply_collision": False,
        }
    )
    return config


def _review_contract(
    freeze: Mapping[str, Any], scope_amendment: Mapping[str, Any]
) -> dict[str, Any]:
    observation = freeze.get("member_geometry_observation") or {}
    scope = scope_amendment.get("joint_scope") or {}
    return {
        "schema_version": "joint_agent_task_topology_review_contract.v1",
        "minimum_assembly_joint_count": scope.get("minimum_assembly_joint_count"),
        "maximum_assembly_joint_count": scope.get("maximum_assembly_joint_count"),
        "commanded_task_joint_count": scope.get("commanded_task_joint_count"),
        "required_articulation_root_count": scope.get("required_articulation_root_count"),
        "non_task_joint_mode": scope.get("non_task_joint_mode"),
        "non_task_joint_motion_tolerance": scope.get("non_task_joint_motion_tolerance"),
        "allowed_joint_types": ["revolute", "prismatic"],
        "target_joint_type": "revolute",
        "target_axis_world": observation.get("joint_axis_world"),
        "target_axis_absolute_dot_minimum": 0.99,
        "target_moving_z_interval_m": observation.get("upper_member_vertical_interval_m"),
        "minimum_target_z_overlap_fraction": 0.85,
        "task_joint_id": (freeze.get("task_spec") or {}).get("target_joint_id"),
        "freeze_digest": freeze.get("freeze_digest"),
        "scope_amendment_digest": scope_amendment.get("amendment_digest"),
        "contract_digest": "",
    }


def build_joint_agent_vast_bundle(
    *,
    repo_root: str | Path,
    joint_agent_root: str | Path,
    packet_path: str | Path,
    execution_authority_path: str | Path,
    freeze_path: str | Path,
    scope_amendment_path: str | Path,
    nim_preflight_path: str | Path,
    scene_optimizer_core_zip_path: str | Path,
    job_dir: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Bind exact released source, derived USD, authority, config, and runtime."""

    repo = Path(repo_root).expanduser().resolve()
    source = Path(joint_agent_root).expanduser().resolve()
    optimizer_zip = Path(scene_optimizer_core_zip_path).expanduser().resolve()
    if (
        not optimizer_zip.is_file()
        or optimizer_zip.is_symlink()
        or _sha256(optimizer_zip) != SCENE_OPTIMIZER_CORE_SHA256
    ):
        raise ValueError("adp_joint_agent_scene_optimizer_core_binding_invalid")
    packet = _canonical_receipt(
        Path(packet_path).expanduser().resolve(),
        digest_field="packet_digest",
        error="adp_joint_agent_packet_invalid",
    )
    authority = validate_public_scene_execution_authority(
        _canonical_receipt(
            Path(execution_authority_path).expanduser().resolve(),
            digest_field="authorization_digest",
            error="adp_joint_agent_execution_authority_invalid",
        )
    )
    freeze = _canonical_receipt(
        Path(freeze_path).expanduser().resolve(),
        digest_field="freeze_digest",
        error="adp_joint_agent_freeze_invalid",
    )
    scope_amendment = _canonical_receipt(
        Path(scope_amendment_path).expanduser().resolve(),
        digest_field="amendment_digest",
        error="adp_joint_agent_scope_amendment_invalid",
    )
    nim_preflight = _canonical_receipt(
        Path(nim_preflight_path).expanduser().resolve(),
        digest_field="receipt_digest",
        error="adp_joint_agent_nim_preflight_invalid",
    )
    if (
        nim_preflight.get("schema_version") != NIM_PREFLIGHT_SCHEMA_VERSION
        or nim_preflight.get("status") != "qualified"
        or nim_preflight.get("endpoint") != NIM_DEFAULT_ENDPOINT
        or nim_preflight.get("model") != NIM_DEFAULT_MODEL
        or nim_preflight.get("http_status") != 200
        or nim_preflight.get("credential_validated") is not True
        or nim_preflight.get("required_model_present") is not True
        or nim_preflight.get("paid_inference_performed") is not False
        or nim_preflight.get("provider_mutations_performed") != 0
        or nim_preflight.get("raw_secret_values_recorded") is not False
        or nim_preflight.get("blockers") not in ([], None)
    ):
        raise ValueError("adp_joint_agent_nim_preflight_not_qualified")
    destination = Path(job_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("adp_joint_agent_bundle_job_dir_not_empty")
    blueprint = _blueprint_identity(repo)
    if blueprint["dirty"]:
        raise ValueError("adp_joint_agent_blueprint_source_not_clean")

    release = inspect_joint_agent_checkout(source)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if tree != SOURCE_TREE or dirty:
        raise ValueError("adp_joint_agent_source_tree_or_cleanliness_mismatch")

    packet_source = packet.get("source_asset") or {}
    source_asset = Path(str(packet_source.get("path") or "")).expanduser().resolve()
    if (
        not source_asset.is_file()
        or _sha256(source_asset) != packet_source.get("sha256")
        or authority["joint_agent_source_asset_digest"] != packet_source.get("sha256")
        or authority["joint_agent_source_receipt_digest"]
        != packet_source.get("source_receipt_digest")
        or authority["publisher_scene_id"] != str((freeze.get("scene") or {}).get("publisher_scene_id"))
        or authority["freeze_digest"] != freeze.get("freeze_digest")
    ):
        raise ValueError("adp_joint_agent_packet_authority_freeze_join_invalid")
    # All caller-controlled identities are validated before the first output
    # byte is created. A failed preflight therefore never leaves a partial
    # bundle that needs a manual cleanup workaround.
    runtime = destination / "provider_runtime"
    ensure_dir(runtime / "input")
    ensure_dir(runtime / "blueprint_src" / "blueprint_pipeline")
    shutil.copy2(source_asset, runtime / "input" / "articulated_source.usda")
    config = _provider_config(packet)
    (runtime / "joint_agent.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    review = _review_contract(freeze, scope_amendment)
    if (
        scope_amendment.get("task_family")
        != "one_commanded_joint_in_bounded_multi_joint_articulated_assembly"
        or review["minimum_assembly_joint_count"] != 1
        or not isinstance(review["maximum_assembly_joint_count"], int)
        or review["maximum_assembly_joint_count"] < 1
        or review["commanded_task_joint_count"] != 1
        or review["required_articulation_root_count"] != 1
        or review["non_task_joint_mode"]
        != "locked_at_frozen_reset_with_native_readback"
        or review["non_task_joint_motion_tolerance"]
        != (freeze.get("task_spec") or {}).get("non_task_joint_motion_tolerance_rad")
    ):
        raise ValueError("adp_joint_agent_preregistered_joint_scope_invalid")
    review["contract_digest"] = canonical_digest(review, digest_field="contract_digest")
    write_json(runtime / "joint_review_contract.json", review)
    write_json(runtime / "execution_authority.json", authority)
    write_json(runtime / "joint_agent_packet.json", packet)
    write_json(runtime / "nvidia_nim_model_preflight.json", nim_preflight)

    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={runtime / 'content_agents_source.zip'}", "HEAD"],
        cwd=source,
        check=True,
    )
    shutil.copy2(optimizer_zip, runtime / "scene_optimizer_core.zip")
    scripts = repo / "scripts"
    _write_executable(
        runtime / "run_adp_joint_agent_provider_runtime.sh",
        scripts / "run_adp_joint_agent_provider_runtime.sh",
    )
    shutil.copy2(
        scripts / "adp_joint_agent_provider_runner.py",
        runtime / "adp_joint_agent_provider_runner.py",
    )
    shutil.copy2(
        repo / "src/blueprint_pipeline/provider_archive.py",
        runtime / "provider_archive.py",
    )
    shutil.copy2(
        repo / "src/blueprint_pipeline/provider_runtime_capacity.py",
        runtime / "provider_runtime_capacity.py",
    )
    for name in (
        "__init__.py",
        "decision_evidence_contracts.py",
        "joint_agent_articulation_review.py",
    ):
        shutil.copy2(
            repo / "src" / "blueprint_pipeline" / name,
            runtime / "blueprint_src" / "blueprint_pipeline" / name,
        )

    entrypoint = runtime / "run_adp_joint_agent_provider_runtime.sh"
    runner = runtime / "adp_joint_agent_provider_runner.py"
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind=PROVIDER_BUNDLE_KIND,
        entrypoint_text=entrypoint.read_text(encoding="utf-8"),
        runner_text=runner.read_text(encoding="utf-8"),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "container_image": DEFAULT_IMAGE,
        "released_code": release,
        "blueprint_source": blueprint,
        "source_tree": tree,
        "source_archive_sha256": _sha256(runtime / "content_agents_source.zip"),
        "scene_optimizer_core": {
            "role": "released_code_public_build_resource",
            "package": SCENE_OPTIMIZER_CORE_PACKAGE,
            "release": SCENE_OPTIMIZER_CORE_RELEASE,
            "license": SCENE_OPTIMIZER_CORE_LICENSE,
            "sha256": _sha256(runtime / "scene_optimizer_core.zip"),
            "size_bytes": (runtime / "scene_optimizer_core.zip").stat().st_size,
        },
        "input_usd_sha256": _sha256(runtime / "input" / "articulated_source.usda"),
        "packet_digest": packet["packet_digest"],
        "execution_authority_digest": authority["authorization_digest"],
        "one_instance_at_a_time": authority.get("one_instance_at_a_time", True),
        "maximum_concurrent_paid_instances": authority.get(
            "maximum_concurrent_paid_instances", 1
        ),
        "freeze_digest": freeze["freeze_digest"],
        "scope_amendment_digest": scope_amendment["amendment_digest"],
        "nim_preflight_receipt_digest": nim_preflight["receipt_digest"],
        "config_sha256": _sha256(runtime / "joint_agent.yaml"),
        "review_contract_digest": review["contract_digest"],
        "renderer": {
            "implementation": "released_code_local_ovrtx_rendering_api",
            "endpoint": "http://127.0.0.1:8001",
            "purpose": "joint_agent_construction_preview_only",
            "render_mode": CONSTRUCTION_RENDER_MODE,
            "num_sensor_updates": CONSTRUCTION_RENDER_SENSOR_UPDATES,
            "profile_basis": "released_ovrtx_in_process_backend_defaults",
            "evaluation_authorized": False,
            "policy_input": False,
            "scene_bytes_leave_vast_instance": False,
        },
        "runtime_resource_requirements": {
            "requested_disk_gb": RUNTIME_DISK_GB,
            "minimum_free_bytes_before_dependency_install": (
                MINIMUM_FREE_BYTES_BEFORE_DEPENDENCY_INSTALL
            ),
            "measurement_path": "provider_runtime",
            "failure_blocker": "joint_agent_runtime_disk_headroom_insufficient",
        },
        "model": {"backend": "nvidia_nim", "id": "google/gemma-4-31b-it"},
        "completion_retries": 0,
        "execution_role": "optional_construction_enrichment",
        "failure_blocks_deterministic_asset_construction": False,
        "failure_blocks_native_simulator_qualification": False,
        "agent_output_is_simready_authority": False,
        "deterministic_usd_construction_remains_primary": True,
        "automatic_paid_retry_allowed": False,
        "raw_interiorgs_downloaded_bytes_included": False,
        "derived_sage_asset_included": True,
        "owned_core_publication_requires_deterministic_review": True,
        "provider_zero_required_after_return": True,
        "expected_output_filename": "adp_joint_agent_result.json",
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    write_json(runtime / "adp_joint_agent_provider_manifest.json", manifest)
    bundle = destination / "adp_joint_agent_provider_runtime_bundle.zip"
    _deterministic_zip(runtime, bundle)
    rehearsal = rehearse_provider_bundle_entrypoint(
        bundle_path=bundle,
        entrypoint_relative_path="provider_runtime/run_adp_joint_agent_provider_runtime.sh",
        evidence_path=destination / "adp_joint_agent_exact_bundle_rehearsal.json",
    )
    receipt = {
        **manifest,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "exact_bundle_entrypoint_rehearsal": rehearsal,
    }
    write_json(destination / "adp_joint_agent_bundle_receipt.json", receipt)
    return receipt


def _remaining_minutes(
    *, job: Path, hard_cap_usd: float, hard_ttl_seconds: int, max_hourly_rate_usd: float
) -> int:
    ledger_path = job / "adp_joint_agent_vast_session_budget.json"
    ledger = _read_json(ledger_path, error="adp_joint_agent_budget_ledger_invalid") if ledger_path.is_file() else {}
    attempts = [row for row in ledger.get("attempts") or [] if isinstance(row, Mapping)]
    prior_seconds = sum(attempt_runtime_seconds(row) for row in attempts)
    prior_cost = sum(attempt_estimated_cost(row) for row in attempts)
    return max(
        0,
        min(
            math.floor(max(0.0, hard_ttl_seconds - prior_seconds) / 60.0),
            math.floor(max(0.0, hard_cap_usd - prior_cost) * 60.0 / max_hourly_rate_usd),
        ),
    )


def _extract_provider_output(path: Path, destination: Path) -> dict[str, Any]:
    blockers: list[str] = []
    result_path = destination / "adp_joint_agent_result.json"
    if not path.is_file():
        return {
            "status": "blocked",
            "execution": {},
            "result_path": str(result_path),
            "blockers": ["joint_agent_provider_output_zip_missing"],
        }
    ensure_dir(destination)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    blockers.append("joint_agent_provider_output_zip_path_traversal")
            if not blockers:
                archive.extractall(destination)
    except (OSError, zipfile.BadZipFile):
        blockers.append("joint_agent_provider_output_zip_invalid")
    execution = _read_json(result_path, error="joint_agent_provider_result_invalid") if result_path.is_file() else {}
    if not execution:
        blockers.append("joint_agent_provider_result_missing")
    return {
        "status": "completed" if not blockers else "blocked",
        "execution": execution,
        "result_path": str(result_path),
        "blockers": sorted(set(blockers)),
    }


def _retained_execution_artifact_blockers(
    execution: Mapping[str, Any], *, extracted_root: Path
) -> list[str]:
    """Validate that topology outputs survived provider teardown packaging."""

    rows = execution.get("retained_artifacts")
    if not isinstance(rows, list):
        return ["joint_agent_required_retained_artifacts_missing"]
    by_role: dict[str, Mapping[str, Any]] = {}
    blockers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            blockers.append("joint_agent_retained_artifact_record_invalid")
            continue
        role = str(row.get("role") or "")
        if not role or role in by_role:
            blockers.append("joint_agent_retained_artifact_role_invalid")
            continue
        by_role[role] = row
    for role in sorted(REQUIRED_RETAINED_ARTIFACT_ROLES):
        row = by_role.get(role)
        if row is None:
            blockers.append(f"joint_agent_retained_artifact_missing:{role}")
            continue
        relative = Path(str(row.get("relative_path") or ""))
        path = (extracted_root / relative).resolve()
        root = extracted_root.resolve()
        if (
            relative.is_absolute()
            or (path != root and root not in path.parents)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            blockers.append(f"joint_agent_retained_artifact_invalid:{role}")
    return sorted(set(blockers))


def _nvidia_api_key() -> str:
    value = str(os.environ.get("NVIDIA_API_KEY") or "").strip()
    if value:
        return value
    path = Path("~/.blueprint-secrets/ngc_api_key").expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


@contextmanager
def _authority_environment():
    names = (
        *_VAST_MUTATION_ENV,
        _VAST_SINGLE_ATTEMPT_ENV,
        "NVIDIA_API_KEY",
        "BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS",
    )
    previous = {name: os.environ.get(name) for name in names}
    secret = _nvidia_api_key()
    if not secret:
        raise ValueError("adp_joint_agent_nvidia_api_key_missing")
    try:
        for name in _VAST_MUTATION_ENV:
            os.environ[name] = "1"
        os.environ[_VAST_SINGLE_ATTEMPT_ENV] = "0"
        os.environ["NVIDIA_API_KEY"] = secret
        os.environ["BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"] = "NVIDIA_API_KEY"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_joint_agent_vast(
    *,
    job_dir: str | Path,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    execute: bool,
    prepared_bundle: Mapping[str, Any],
    max_hourly_rate_usd: float = 1.0,
    hard_cap_usd: float = 3.0,
    hard_ttl_seconds: int = 10_800,
    public_image: str = DEFAULT_IMAGE,
    allowed_active_instance_ids: Sequence[int] = (),
    machine_avoidlist_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute exactly one zero-retry Joint Agent attempt with provider zero."""

    job = Path(job_dir).expanduser().resolve()
    ensure_dir(job)
    bundle = dict(prepared_bundle)
    bundle_path = Path(str(bundle.get("bundle_path") or "")).resolve()
    if public_image != DEFAULT_IMAGE:
        raise ValueError("adp_joint_agent_container_image_not_frozen")
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or _sha256(bundle_path) != bundle.get("bundle_sha256")
        or provider_bundle_rehearsal_blockers(
            bundle.get("exact_bundle_entrypoint_rehearsal"),
            bundle_sha256=str(bundle.get("bundle_sha256") or ""),
            entrypoint_relative_path=(
                "provider_runtime/run_adp_joint_agent_provider_runtime.sh"
            ),
        )
    ):
        raise ValueError("adp_joint_agent_prepared_bundle_binding_invalid")
    if not execute:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "dry_run_ready",
            "bundle": bundle,
            "provider_mutations_performed": 0,
            "retry_cap": 0,
            "blockers": [],
        }
        write_json(job / "adp_joint_agent_vast_result.json", result)
        return result
    if paid_resource_admission_grant is None:
        raise ValueError("adp_joint_agent_paid_resource_admission_grant_missing")
    remaining_minutes = _remaining_minutes(
        job=job,
        hard_cap_usd=hard_cap_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    if remaining_minutes < 90:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": ["adp_joint_agent_budget_below_minimum_live_window"],
        }
    staging_dir = job / "object_store_staging"
    staging = stage_wam_provider_bundle_object_store(
        job_dir=staging_dir,
        bundle_path=str(bundle_path),
        key_prefix=DEFAULT_KEY_PREFIX,
        expiration_seconds=max(hard_ttl_seconds + 1800, 18_000),
    )
    if staging.get("status") != "completed":
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "blockers": staging.get("blockers") or ["joint_agent_object_store_staging_blocked"],
        }
    provider_run = job / "vast_provider_run"
    output_zip = provider_run / "vast_provider_runtime_output.zip"
    watchdog_handoff, watchdog_handle = arm_independent_vast_watchdog(
        job_dir=job,
        max_live_minutes=remaining_minutes,
        generated_at=utc_now_iso(),
        allowed_active_instance_ids=allowed_active_instance_ids,
    )
    if watchdog_handle is None:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "provider_mutations_performed": 0,
            "all_staged_objects_absent": cleanup.get("all_objects_absent"),
            "independent_watchdog": watchdog_handoff,
            "blockers": ["joint_agent_independent_watchdog_not_armed"],
        }
    adapter: dict[str, Any] = {}
    try:
        with _authority_environment():
            adapter = run_vast_provider_adapter(
                job_dir=provider_run,
                mode="live-startup-probe",
                allow_vast_api_call=True,
                allow_instance_launch=True,
                max_hourly_rate=max_hourly_rate_usd,
                target_spend_usd=hard_cap_usd,
                hard_cap_usd=hard_cap_usd,
                max_live_minutes=remaining_minutes,
                session_max_live_minutes=hard_ttl_seconds // 60,
                public_image=public_image,
                isaac_image=public_image,
                ngc_image_login_mode="never",
                provider_bundle=bundle_path,
                provider_bundle_url=(staging_dir / "provider_bundle_url.txt").read_text().strip(),
                provider_output_put_url=(staging_dir / "provider_output_put_url.txt").read_text().strip(),
                provider_output_get_url=(staging_dir / "provider_output_get_url.txt").read_text().strip(),
                provider_runtime_output_zip=output_zip,
                enable_isaac_smoke=False,
                enable_blueprint_bundle=True,
                provider_bundle_kind=PROVIDER_BUNDLE_KIND,
                vast_launch_mode="ssh_direct",
                allow_cold_isaac_image_pull=False,
                disk_gb=RUNTIME_DISK_GB,
                min_gpu_ram_mb=24_000,
                poll_interval_seconds=15,
                startup_timeout_seconds=min(10_800, remaining_minutes * 60),
                heartbeat_no_progress_seconds=1800,
                session_budget_ledger_path=job / "adp_joint_agent_vast_session_budget.json",
                verify_staging_urls=True,
                require_known_supported_isaac_driver=False,
                preferred_gpu_keywords=("L40S", "RTX 4090", "RTX A6000", "A100"),
                prefer_isaac_rt=False,
                allowed_active_instance_ids=allowed_active_instance_ids,
                vast_launch_lock_file=job.parent / "joint_agent_paid_launch.lock",
                instance_label_prefix="blueprint-adp-joint-agent-",
                started_instance_id_path=watchdog_handle.started_instance_id_path,
                forward_hf_token=False,
                paid_resource_admission_grant=paid_resource_admission_grant,
                machine_avoidlist_path=machine_avoidlist_path,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        adapter = {
            "status": "blocked",
            "blockers": [f"adp_joint_agent_vast_adapter_failed:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
        }
    finally:
        cleanup = cleanup_staged_wam_provider_objects(staging_dir)
    extracted = _extract_provider_output(output_zip, job / "immutable_execution")
    execution = dict(extracted.get("execution") or {})
    teardown_path = provider_run / "vast_teardown_manifest.json"
    teardown = _read_json(teardown_path, error="joint_agent_teardown_manifest_invalid") if teardown_path.is_file() else {}
    instance_ids = [
        int(value)
        for value in (teardown.get("vast_instance_ids") or adapter.get("vast_instance_ids") or [])
        if isinstance(value, int) and value > 0
    ]
    watchdog_close = close_independent_vast_watchdog(
        job_dir=job,
        handle=watchdog_handle,
        instance_ids=instance_ids,
        provider_teardown_completed=teardown.get("continuing_spend_from_this_run") is False,
        provider_allocation_impossible=(
            not instance_ids and adapter.get("provider_create_attempted") is not True
        ),
    )
    blockers = list(adapter.get("blockers") or []) + list(extracted.get("blockers") or [])
    if execution.get("status") != "completed":
        blockers.extend(execution.get("blockers") or ["joint_agent_execution_not_completed"])
    else:
        blockers.extend(
            _retained_execution_artifact_blockers(
                execution,
                extracted_root=job / "immutable_execution",
            )
        )
    if teardown.get("continuing_spend_from_this_run") is not False:
        blockers.append("joint_agent_vast_provider_zero_not_proven")
    if cleanup.get("all_objects_absent") is not True:
        blockers.append("joint_agent_object_store_provider_zero_not_proven")
    if watchdog_close.get("status") not in {"provider_terminal", "cancelled_no_allocation"}:
        blockers.append("joint_agent_independent_watchdog_not_closed")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_result_path": extracted.get("result_path"),
        "adapter_result_path": str(provider_run / "vast_provider_adapter_result.json"),
        "teardown_manifest_path": str(teardown_path),
        "estimated_cost_usd": adapter.get("estimated_cost_usd"),
        "hard_cap_usd": hard_cap_usd,
        "hard_ttl_seconds": hard_ttl_seconds,
        "retry_cap": 0,
        "continuing_spend_from_this_run": teardown.get("continuing_spend_from_this_run"),
        "all_staged_objects_absent": cleanup.get("all_objects_absent"),
        "independent_watchdog": watchdog_close,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "raw_secret_values_recorded": False,
    }
    write_json(job / "adp_joint_agent_vast_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--joint-agent-root", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--scope-amendment", required=True)
    parser.add_argument("--nim-preflight", required=True)
    parser.add_argument("--scene-optimizer-core", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    receipt = build_joint_agent_vast_bundle(
        repo_root=args.repo_root,
        joint_agent_root=args.joint_agent_root,
        packet_path=args.packet,
        execution_authority_path=args.execution_authority,
        freeze_path=args.freeze,
        scope_amendment_path=args.scope_amendment,
        nim_preflight_path=args.nim_preflight,
        scene_optimizer_core_zip_path=args.scene_optimizer_core,
        job_dir=args.job_dir,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_IMAGE",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "build_joint_agent_vast_bundle",
    "run_joint_agent_vast",
]
