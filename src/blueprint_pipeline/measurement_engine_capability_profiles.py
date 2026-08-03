"""R1 source-verified engine capability profiles.

Filled ``method_capability_profile.v1`` drafts for the priority engines named
by the measurement-routing research: MuJoCo, Drake, Isaac Sim (PhysX physics
and the RTX/OpenUSD sensor path as separate methods), Newton, SAPIEN, and
Project Chrono.

Provenance discipline (R1 stage of the admission protocol):

- ``live_fetch`` facts were verified against the named primary source on
  ``PROFILE_VERIFICATION_DATE`` (release pages, license files, official docs).
- ``report_vf`` facts were VF-labeled rows of the accepted 2026-08-01 routing
  research, not independently re-fetched on the verification date.
- Every other capability boolean FAILS CLOSED to ``False``.  ``False`` here
  means "not established", never "measured as absent".
- ``deterministic_mode`` is ``"unverified"`` for every engine, which the
  routing kernel's replay allowlist rejects, so a deterministic-replay
  requirement can never be satisfied by an unverified claim.
- Commit hashes and container digests are explicitly ``unpinned:`` sentinels:
  pinning exact commits, containers, drivers, and numerics is R3/R7 work.

These profiles carry ZERO qualifications.  Feeding them to the router yields
``no_exact_verified_qualification`` abstentions by construction: a verified
feature list is a capability declaration, never task-scoped validity.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    validate_method_capability_profile,
)


PROFILE_VERIFICATION_DATE = "2026-08-02"
ENGINE_PROFILE_SET_VERSION = "measurement_engine_capability_profiles.2026-08-02"

_LIST_FIELDS = (
    "plugin_versions",
    "robot_model_formats",
    "supported_embodiments",
    "supported_end_effectors",
    "action_representation_types",
    "qualification_record_ids",
    "qualified_task_classes",
    "qualified_material_regimes",
    "qualified_robot_ids",
    "qualified_end_effector_ids",
    "qualified_controller_ids",
    "qualified_sensor_ids",
    "qualified_site_classes",
    "qualified_metric_ids",
    "known_failure_modes",
    "prohibited_extrapolations",
    "asset_license_ids",
    "model_license_ids",
    "subprocessor_regions",
    "output_formats",
)


class EngineProfileError(ValueError):
    pass


def _blank_capabilities(
    *,
    method_id: str,
    family: str,
    version: str,
    release_date: str,
    solver_backend: str,
    contact_formulation: str,
    operating_system: str,
    gpu_model: str,
    license_ids: tuple[str, ...],
    ceiling: str,
    commercial_use_allowed: bool,
    redistribution_allowed: bool,
) -> dict[str, Any]:
    values: dict[str, Any] = {field: False for field in ALL_CAPABILITY_FIELDS}
    for field in _LIST_FIELDS:
        values[field] = []
    values.update(
        {
            "method_id": method_id,
            "method_family": family,
            "version": version,
            "release_date": release_date,
            "commit_hash": f"unpinned:release-tag-v{version}",
            "container_digest": "unpinned:r1_no_container_built",
            "solver_backend": solver_backend,
            "numeric_precision": "unverified",
            "deterministic_mode": "unverified",
            "operating_system": operating_system,
            "gpu_model": gpu_model,
            "driver_version": "unpinned:r1_no_driver_manifest",
            "random_seed_policy": "unverified",
            "contact_formulation": contact_formulation,
            "maximum_control_rate_hz": 0,
            "qualified_parameter_ranges": {},
            "qualified_claim_ceiling": ceiling,
            "qualification_expiration": "not_applicable_no_qualifications",
            "harmful_false_negative_bound": 1.0,
            "maximum_latency_class": "unverified",
            "maximum_compute_class": "unverified",
            "estimated_cost_class": "local_compute",
            "data_retention_days": 0,
            "source_available": True,
            "local_offline_supported": True,
            "api_only": False,
            "commercial_use_allowed": commercial_use_allowed,
            "redistribution_allowed": redistribution_allowed,
            "asset_license_ids": list(license_ids),
            "model_license_ids": [],
            "provider_training_use_allowed": False,
            "deletion_right_supported": True,
            "output_export_supported": True,
        }
    )
    return values


def _profile(
    *,
    capabilities: dict[str, Any],
    live_fields: Mapping[str, tuple[str, ...]],
    reported_fields: Mapping[str, tuple[str, ...]],
    identity_sources: tuple[str, ...],
    notes: str,
    world_model_roles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Assemble one validated profile with a tiered source manifest.

    ``live_fields`` and ``reported_fields`` map source references to the
    capability fields they substantiate; every listed field is set ``True``.
    """

    manifest: list[dict[str, Any]] = [
        {
            "verification": "live_fetch",
            "reference": reference,
            "facts": ["method_identity", "version", "release_date", "license"],
            "verified_on": PROFILE_VERIFICATION_DATE,
        }
        for reference in identity_sources
    ]
    for reference, fields in sorted(live_fields.items()):
        for field in fields:
            if field not in ALL_CAPABILITY_FIELDS:
                raise EngineProfileError(f"unknown_capability_field:{field}")
            capabilities[field] = True
        manifest.append(
            {
                "verification": "live_fetch",
                "reference": reference,
                "facts": sorted(fields),
                "verified_on": PROFILE_VERIFICATION_DATE,
            }
        )
    for reference, fields in sorted(reported_fields.items()):
        for field in fields:
            if field not in ALL_CAPABILITY_FIELDS:
                raise EngineProfileError(f"unknown_capability_field:{field}")
            capabilities[field] = True
        manifest.append(
            {
                "verification": "report_vf",
                "reference": reference,
                "facts": sorted(fields),
                "verified_on": "2026-08-01",
            }
        )
    value: dict[str, Any] = {
        "schema_version": "method_capability_profile.v1",
        "method_id": capabilities["method_id"],
        "capabilities": capabilities,
        "evidence_quality": {
            "source": "r1_source_verification",
            "verified_on": PROFILE_VERIFICATION_DATE,
            "source_manifest": manifest,
            "unverified_fields_fail_closed": True,
            "public_research_is_qualification": False,
            "notes": notes,
        },
        "expected_cost_usd": 1.0,
        "expected_latency_seconds": 60.0,
    }
    if world_model_roles:
        value["world_model_roles"] = list(world_model_roles)
    return validate_method_capability_profile(value)


def _mujoco_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="mujoco-3",
        family="traditional_simulation",
        version="3.11.0",
        release_date="2026-07-27",
        solver_backend="mujoco_cpu",
        contact_formulation="soft_convex_analytically_invertible",
        operating_system="linux_macos_windows",
        gpu_model="optional_mjx_or_warp_backends",
        license_ids=("Apache-2.0",),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=True,
    )
    capabilities["robot_model_formats"] = ["mjcf", "urdf", "mjb", "mjz"]
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://github.com/google-deepmind/mujoco/releases",
            "https://raw.githubusercontent.com/google-deepmind/mujoco/main/LICENSE",
        ),
        live_fields={
            "https://mujoco.readthedocs.io/en/latest/overview.html": (
                "static_friction_supported",
                "dynamic_friction_supported",
                "torsional_friction_supported",
                "rolling_friction_supported",
                "joint_friction_supported",
                "joint_limits_supported",
                "revolute_joint_supported",
                "prismatic_joint_supported",
                "contact_compliance_supported",
                "cloth_shell_supported",
                "rod_cable_supported",
                "self_collision_supported",
                "force_torque_supported",
                "imu_supported",
                "contact_sensor_supported",
                "mjcf_supported",
                "urdf_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#mujoco-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "contact_force_output_supported",
                "position_control_supported",
                "velocity_control_supported",
                "torque_control_supported",
                "actuator_dynamics_supported",
                "rgb_supported",
            ),
        },
        notes=(
            "1D/2D/3D flex bodies verified live but flex remains solver- and "
            "task-specific, never generic cloth authority. MJX and MuJoCo Warp "
            "are separate backends requiring separate qualification; MJWarp "
            "lacks autodiff and full feature parity."
        ),
    )


def _drake_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="drake-1-55",
        family="traditional_simulation",
        version="1.55.0",
        release_date="2026-07-15",
        solver_backend="drake_multibody",
        contact_formulation="hydroelastic_with_point_fallback",
        operating_system="linux_macos",
        gpu_model="none_required",
        license_ids=("BSD-3-Clause",),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=True,
    )
    capabilities["robot_model_formats"] = ["urdf", "sdformat"]
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://github.com/RobotLocomotion/drake/releases",
            "https://raw.githubusercontent.com/RobotLocomotion/drake/master/LICENSE.TXT",
        ),
        live_fields={
            "https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html": (
                "hydroelastic_contact_supported",
                "contact_compliance_supported",
                "contact_force_output_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#drake-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "self_collision_supported",
                "revolute_joint_supported",
                "prismatic_joint_supported",
                "joint_limits_supported",
                "static_friction_supported",
                "dynamic_friction_supported",
                "position_control_supported",
                "torque_control_supported",
                "force_torque_supported",
                "urdf_supported",
            ),
        },
        notes=(
            "Compliant and rigid hydroelastic representations with explicit "
            "point-contact fallback verified live. Not a general visual-sensor "
            "or broad deformable simulator."
        ),
    )


def _isaac_physx_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="isaac-sim-6-physx",
        family="traditional_simulation",
        version="6.0.1",
        release_date="2026-06-30",
        solver_backend="physx_110.1.13",
        contact_formulation="physx_iterative_tgs",
        operating_system="linux_windows",
        gpu_model="nvidia_rtx_required",
        license_ids=(
            "Apache-2.0-core",
            "NVIDIA-Isaac-Sim-Additional-Software-and-Materials-License",
        ),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=False,
    )
    capabilities["robot_model_formats"] = ["urdf", "mjcf", "openusd"]
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html",
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/license-faq.html",
        ),
        live_fields={
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html": (
                "urdf_supported",
                "mjcf_supported",
                "openusd_supported",
                "revolute_joint_supported",
                "prismatic_joint_supported",
                "joint_limits_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#isaac-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "self_collision_supported",
                "static_friction_supported",
                "dynamic_friction_supported",
                "contact_force_output_supported",
                "position_control_supported",
                "velocity_control_supported",
                "torque_control_supported",
            ),
        },
        notes=(
            "PhysX 110.1.13 with experimental Newton backend verified live; the "
            "backends require separate qualification records. Deformable "
            "booleans stay False: legacy particle cloth is marked do-not-use "
            "and volume deformables are experimental. Visual meshes are not "
            "automatically colliders. Redistribution or delivery as a service "
            "requires an NVIDIA AI Enterprise license (internal R&D and "
            "selling simulation outputs verified permitted)."
        ),
    )


def _isaac_rtx_sensor_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="isaac-rtx-openusd-sensor-path",
        family="calibrated_renderer_sensor_simulation",
        version="6.0.1",
        release_date="2026-06-30",
        solver_backend="rtx_sensor_pipeline",
        contact_formulation="not_a_contact_method",
        operating_system="linux_windows",
        gpu_model="nvidia_rtx_required",
        license_ids=(
            "Apache-2.0-core",
            "NVIDIA-Isaac-Sim-Additional-Software-and-Materials-License",
        ),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=False,
    )
    capabilities["robot_model_formats"] = ["openusd"]
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html",
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/license-faq.html",
        ),
        live_fields={
            "https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html": (
                "rgb_supported",
                "lidar_supported",
                "radar_supported",
                "openusd_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#rtx-sensor-row": (
                "depth_supported",
                "intrinsics_import_supported",
                "extrinsics_import_supported",
                "material_brdf_supported",
                "sensor_timing_supported",
            ),
        },
        notes=(
            "RTX camera, lidar, radar, and the new acoustic sensor verified "
            "live. Sensor capability is not evidence that a configured output "
            "matches a particular physical sensor; physics fields remain False "
            "and the family can never hold physics authority."
        ),
    )


def _newton_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="newton-1-4",
        family="traditional_simulation",
        version="1.4.0",
        release_date="2026-07-16",
        solver_backend="multi_solver_mjwarp_xpbd_vbd_featherstone_semiimplicit_kamino_mpm",
        contact_formulation="per_solver_varies",
        operating_system="linux",
        gpu_model="nvidia_gpu_required",
        license_ids=("Apache-2.0",),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=True,
    )
    capabilities["robot_model_formats"] = ["openusd", "urdf", "mjcf"]
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://github.com/newton-physics/newton/releases",
            "https://raw.githubusercontent.com/newton-physics/newton/main/LICENSE.md",
        ),
        live_fields={
            "https://github.com/newton-physics/newton/releases": (
                "openusd_supported",
                "mpm_supported",
                "hyperelastic_fem_supported",
                "rod_cable_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#newton-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "self_collision_supported",
                "static_friction_supported",
                "dynamic_friction_supported",
                "cloth_shell_supported",
                "urdf_supported",
                "mjcf_supported",
                "position_control_supported",
                "torque_control_supported",
            ),
        },
        notes=(
            "Deformable USD workflows, VBD rigid-soft contact, cable damping, "
            "and MPM verified live but explicitly experimental. Each solver "
            "family requires its own qualification record: 'Newton qualified' "
            "without a solver scope is an invalid catalog entry, and version "
            "or commit changes trigger requalification."
        ),
    )


def _sapien_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="sapien-maniskill-3",
        family="traditional_simulation",
        version="3.0.3",
        release_date="2026-03-10",
        solver_backend="physx",
        contact_formulation="physx_iterative",
        operating_system="linux",
        gpu_model="optional_gpu",
        license_ids=("Apache-2.0",),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=True,
    )
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://github.com/haosulab/SAPIEN/releases",
            "https://raw.githubusercontent.com/haosulab/SAPIEN/master/LICENSE",
        ),
        live_fields={},
        reported_fields={
            "routing-research-2026-08-01#sapien-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "revolute_joint_supported",
                "prismatic_joint_supported",
                "joint_limits_supported",
                "static_friction_supported",
                "dynamic_friction_supported",
                "rgb_supported",
                "urdf_supported",
                "position_control_supported",
            ),
        },
        notes=(
            "Release identity and Apache-2.0 license verified live; feature "
            "rows carry only the research report's VF grading pending a "
            "documentation-level R1 pass. PartNet-Mobility or synthetic joints "
            "are never captured-site articulation evidence."
        ),
    )


def _chrono_profile() -> dict[str, Any]:
    capabilities = _blank_capabilities(
        method_id="project-chrono-10",
        family="multiphysics_engineering_solver",
        version="10.0.0",
        release_date="2026-03-27",
        solver_backend="chrono_multibody_fea_dem_fsi",
        contact_formulation="smc_and_nsc",
        operating_system="linux_macos_windows",
        gpu_model="optional_gpu_modules",
        license_ids=("BSD-3-Clause",),
        ceiling="C4",
        commercial_use_allowed=True,
        redistribution_allowed=True,
    )
    return _profile(
        capabilities=capabilities,
        identity_sources=(
            "https://projectchrono.org/news/",
            "https://github.com/projectchrono/chrono/releases",
        ),
        live_fields={
            "https://projectchrono.org/news/": (
                "sph_supported",
                "rgb_supported",
            ),
        },
        reported_fields={
            "routing-research-2026-08-01#chrono-row": (
                "metric_scale_supported",
                "dynamic_collision_supported",
                "dem_supported",
                "granular_cohesion_supported",
                "hyperelastic_fem_supported",
                "static_friction_supported",
                "dynamic_friction_supported",
                "rolling_friction_supported",
            ),
        },
        notes=(
            "SOURCE DISCREPANCY RECORDED: the GitHub releases page surfaced a "
            "10.0.0 entry dated 2024-04-07 while projectchrono.org dates the "
            "10.0.0 stable release 2026-03-27 with the refactored SPH/TDPF FSI "
            "module and new Peridynamics module; resolved in favor of the "
            "project site and flagged for monitoring. TDPF fluid is not "
            "recorded as general CFD."
        ),
    )


def engine_capability_profiles() -> tuple[dict[str, Any], ...]:
    """All R1 source-verified engine profiles, validated and digest-bound."""

    return (
        _mujoco_profile(),
        _drake_profile(),
        _isaac_physx_profile(),
        _isaac_rtx_sensor_profile(),
        _newton_profile(),
        _sapien_profile(),
        _chrono_profile(),
    )


def engine_profile_by_method_id(method_id: str) -> dict[str, Any]:
    for profile in engine_capability_profiles():
        if profile["method_id"] == method_id:
            return profile
    raise EngineProfileError(f"engine_profile_unknown:{method_id}")


def r1_source_verification_stage_data(method_id: str) -> dict[str, Any]:
    """R1 admission stage data derived from a profile's source manifest."""

    profile = engine_profile_by_method_id(method_id)
    quality = dict(profile["evidence_quality"])
    manifest = list(quality.get("source_manifest") or [])
    live = [row for row in manifest if row.get("verification") == "live_fetch"]
    reported = [row for row in manifest if row.get("verification") == "report_vf"]
    capabilities = dict(profile["capabilities"])
    return {
        "source_verification": {
            "verified_on": PROFILE_VERIFICATION_DATE,
            "live_fetch_references": sorted({row["reference"] for row in live}),
            "report_vf_references": sorted({row["reference"] for row in reported}),
            "unverified_fields_fail_closed": True,
        },
        "code_access": {
            "source_available": capabilities["source_available"],
            "local_offline_supported": capabilities["local_offline_supported"],
        },
        "license_records": {
            "license_ids": list(capabilities["asset_license_ids"]),
            "redistribution_allowed": capabilities["redistribution_allowed"],
        },
        "vendor_claim_separation": {
            "vendor_results_status": "external_claim_until_independent_execution",
            "report_vf_rows_pending_doc_level_reverification": bool(reported),
        },
    }


def engine_profile_set_snapshot() -> dict[str, Any]:
    profiles = engine_capability_profiles()
    import hashlib

    snapshot = {
        "schema_version": "measurement_engine_capability_profile_set.v1",
        "set_version": ENGINE_PROFILE_SET_VERSION,
        "verification_date": PROFILE_VERIFICATION_DATE,
        "profile_digests": sorted(profile["capability_profile_digest"] for profile in profiles),
        "qualification_record_count": 0,
        "profiles_are_routable_without_qualification": False,
    }
    encoded = json.dumps(
        {key: value for key, value in snapshot.items() if key != "set_digest"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    snapshot["set_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return snapshot


__all__ = [
    "ENGINE_PROFILE_SET_VERSION",
    "EngineProfileError",
    "PROFILE_VERIFICATION_DATE",
    "engine_capability_profiles",
    "engine_profile_by_method_id",
    "engine_profile_set_snapshot",
    "r1_source_verification_stage_data",
]
