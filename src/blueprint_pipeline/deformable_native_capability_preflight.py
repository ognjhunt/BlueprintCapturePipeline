"""One-shot, fail-closed static preflight for native deformable execution.

This module evaluates one retained inventory assembled before a scene run.  It
does not import Isaac, start SimulationApp, allocate a GPU, or launch a policy.
Instead it cross-checks exact source/runtime identities, dependency and symbol
inventories, selected-embodiment scope, device declarations, media tools, and
the two frozen policy identities.  All static rows are evaluated even after a
failure so one paid run is never spent discovering missing modules serially.

Static availability is deliberately separated from the native canary.  Source
inspection can prove that the pinned stack declares experimental volumetric FEM
soft-body APIs; it cannot prove USD cooking, CUDA execution, soft-body contact,
reset repeatability, settling, or rendering.  Thin-shell cloth, independent
bend/shear authoring, and physical towel equivalence are never claimed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .native_task_arena_import_scope import ROBOT_EMBODIMENT_MODULES
from .native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


REQUEST_SCHEMA_VERSION = "deformable_native_capability_preflight_request.v1"
MATRIX_SCHEMA_VERSION = "deformable_native_capability_preflight_matrix.v1"

FROZEN_CANDIDATES = ("pi05_droid", "groot_n17_droid")
POLICY_ADAPTER_MODULES = {
    "pi05_droid": "blueprint_pipeline.openpi_droid_policy_runtime",
    "groot_n17_droid": "blueprint_pipeline.groot_n17_droid_policy_runtime",
}

ISAACLAB_REQUIRED_RELATIVE_PATHS = (
    "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py",
    "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_cfg.py",
    "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_data.py",
    "source/isaaclab/isaaclab/sim/schemas/schemas.py",
    "source/isaaclab/isaaclab/sim/spawners/materials/physics_materials.py",
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py",
    "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py",
    "source/isaaclab/isaaclab/sensors/camera/camera.py",
    "source/isaaclab/isaaclab/sensors/camera/camera_cfg.py",
    "source/isaaclab/isaaclab/sim/simulation_context.py",
)
ARENA_REQUIRED_RELATIVE_PATHS = (
    "isaaclab_arena/embodiments/droid/droid.py",
)

REQUIRED_SYMBOLS_BY_CHECK = {
    "openusd_physx_volumetric_deformable_schemas": (
        "pxr.PhysxSchema:PhysxCollisionAPI",
        "pxr.PhysxSchema:PhysxDeformableAPI",
        "pxr.PhysxSchema:PhysxDeformableBodyAPI",
        "pxr.PhysxSchema:PhysxDeformableBodyMaterialAPI",
    ),
    "volumetric_deformable_authoring_apis": (
        "omni.physx.scripts.deformableUtils:add_physx_deformable_body",
        "isaaclab.sim.schemas.schemas:define_deformable_body_properties",
        "isaaclab.sim.spawners.materials.physics_materials:spawn_deformable_body_material",
        "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject",
        "isaaclab_physx.assets.deformable_object.deformable_object_cfg:DeformableObjectCfg",
    ),
    "contact_sensor_apis": (
        "isaaclab.sensors.contact_sensor.contact_sensor:ContactSensor",
        "isaaclab.sensors.contact_sensor.contact_sensor_cfg:ContactSensorCfg",
    ),
    "deformable_reset_state_readback_apis": (
        "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject.write_nodal_state_to_sim_index",
        "isaaclab_physx.assets.deformable_object.deformable_object:DeformableObject.write_nodal_kinematic_target_to_sim_index",
        "isaaclab_physx.assets.deformable_object.deformable_object_data:DeformableObjectData.nodal_state_w",
        "isaaclab_physx.assets.deformable_object.deformable_object_data:DeformableObjectData.sim_element_deform_gradient_w",
    ),
    "renderer_camera_apis": (
        "isaaclab.sensors.camera.camera:Camera",
        "isaaclab.sensors.camera.camera_cfg:CameraCfg",
        "isaaclab.sim.simulation_context:SimulationContext.render",
    ),
    "cuda_warp_apis": (
        "torch:cuda",
        "warp:get_device",
        "warp:launch",
    ),
}
ALL_REQUIRED_SYMBOLS = tuple(
    symbol
    for check_id in REQUIRED_SYMBOLS_BY_CHECK
    for symbol in REQUIRED_SYMBOLS_BY_CHECK[check_id]
)

MINIMUM_INTERNAL_RUNTIME_MODULES = (
    "blueprint_pipeline.native_task_entity_contract",
    "blueprint_pipeline.native_task_arena_import_scope",
    "blueprint_pipeline.native_task_arena_runtime",
    "blueprint_pipeline.native_task_arena_construction_worker",
    "blueprint_pipeline.native_task_episode_environment",
    "blueprint_pipeline.deformable_transfer_scoring",
    "blueprint_pipeline.openpi_droid_policy_runtime",
    "blueprint_pipeline.groot_n17_droid_policy_runtime",
)

DYNAMIC_NATIVE_CANARY_GATES = (
    (
        "dynamic_usd_composition_and_deformable_cooking",
        "compose one closed-mesh volumetric body, bind material, and read back cooked schemas",
    ),
    (
        "dynamic_cuda_warp_execution",
        "execute PhysX and Warp work on the admitted CUDA device",
    ),
    (
        "dynamic_genuine_gripper_deformable_contact",
        "prove collision/friction grasp, lift, release, and retained contact evidence",
    ),
    (
        "dynamic_nodal_reset_repeatability",
        "restore free-node state twice and compare native nodal readback",
    ),
    (
        "dynamic_deformable_settling_and_strain_readback",
        "settle without divergence and read back deformation gradients",
    ),
    (
        "dynamic_renderer_camera_capture",
        "render synchronized external, wrist, and overview frames",
    ),
    (
        "dynamic_applied_parameter_readback",
        "read back every applied solver, material, contact, and camera parameter",
    ),
)

_SHA256 = "sha256:"


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return bool(
        text.startswith(_SHA256)
        and len(text) == len(_SHA256) + 64
        and all(character in "0123456789abcdef" for character in text[len(_SHA256) :])
    )


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _add_row(
    rows: list[dict[str, Any]],
    *,
    check_id: str,
    category: str,
    blockers: Sequence[str],
    evidence: Mapping[str, Any],
) -> None:
    typed = sorted(set(str(blocker) for blocker in blockers if str(blocker)))
    rows.append(
        {
            "check_id": check_id,
            "category": category,
            "phase": "static_preflight",
            "required": True,
            "status": "blocked" if typed else "passed",
            "blockers": typed,
            "evidence": dict(evidence),
        }
    )


def _request_contract_check(request: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("deformable_preflight_request_schema_unsupported")
    preflight_id = str(request.get("preflight_id") or "").strip()
    if not preflight_id:
        blockers.append("deformable_preflight_id_missing")
    robot_id = str(request.get("selected_robot_id") or "").strip()
    if robot_id not in ROBOT_EMBODIMENT_MODULES:
        blockers.append(f"deformable_preflight_robot_unadmitted:{robot_id or 'missing'}")

    policies = _rows(request.get("policy_identities"))
    candidate_ids = [str(row.get("candidate_id") or "") for row in policies]
    if tuple(sorted(candidate_ids)) != tuple(sorted(FROZEN_CANDIDATES)):
        blockers.append("deformable_preflight_frozen_candidate_set_mismatch")

    internals = _rows(request.get("internal_runtime_modules"))
    internal_names = {str(row.get("module") or "") for row in internals}
    for module in MINIMUM_INTERNAL_RUNTIME_MODULES:
        if module not in internal_names:
            blockers.append(f"deformable_preflight_internal_requirement_missing:{module}")
    return blockers, {
        "preflight_id": preflight_id,
        "selected_robot_id": robot_id,
        "candidate_ids": sorted(candidate_ids),
        "internal_module_count": len(internal_names),
    }


def _source_identity_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    roots = _mapping(observations.get("source_roots"))
    expected_repositories = {
        "isaaclab": {
            "repository": ISAACLAB_REPOSITORY,
            "revision": ISAACLAB_COMMIT,
            "tree": ISAACLAB_TREE,
            "required_paths": ISAACLAB_REQUIRED_RELATIVE_PATHS,
        },
        "arena": {
            "repository": ARENA_REPOSITORY,
            "revision": ARENA_COMMIT,
            "tree": ARENA_TREE,
            "required_paths": ARENA_REQUIRED_RELATIVE_PATHS,
        },
    }
    evidence: dict[str, Any] = {"repositories": {}}
    for repository_id, expected in expected_repositories.items():
        observed = _mapping(roots.get(repository_id))
        root_text = str(observed.get("root_path") or "").strip()
        root = Path(root_text).expanduser() if root_text else Path("/__missing__")
        repository_blockers: list[str] = []
        if not root_text or not root.is_dir():
            repository_blockers.append(
                f"deformable_preflight_source_root_missing:{repository_id}"
            )
        if observed.get("repository") != expected["repository"]:
            repository_blockers.append(
                f"deformable_preflight_source_repository_mismatch:{repository_id}"
            )
        if observed.get("revision") != expected["revision"]:
            repository_blockers.append(
                f"deformable_preflight_source_revision_mismatch:{repository_id}"
            )
        if observed.get("tree") != expected["tree"]:
            repository_blockers.append(
                f"deformable_preflight_source_tree_mismatch:{repository_id}"
            )
        if not _valid_sha256(observed.get("source_receipt_digest")):
            repository_blockers.append(
                f"deformable_preflight_source_receipt_digest_invalid:{repository_id}"
            )
        missing_relative_paths = [
            relative
            for relative in expected["required_paths"]
            if not (root / relative).is_file()
        ]
        for relative in missing_relative_paths:
            repository_blockers.append(
                f"deformable_preflight_required_source_file_missing:{repository_id}:{relative}"
            )
        blockers.extend(repository_blockers)
        evidence["repositories"][repository_id] = {
            "root_path": root_text,
            "repository": observed.get("repository"),
            "revision": observed.get("revision"),
            "tree": observed.get("tree"),
            "source_receipt_digest": observed.get("source_receipt_digest"),
            "required_file_count": len(expected["required_paths"]),
            "missing_relative_paths": missing_relative_paths,
        }

    expected_simulator = _mapping(request.get("simulator_runtime_identity"))
    observed_simulator = _mapping(observations.get("simulator_runtime_identity"))
    simulator_root_text = str(observed_simulator.get("root_path") or "").strip()
    simulator_root = (
        Path(simulator_root_text).expanduser() if simulator_root_text else Path("/__missing__")
    )
    for field in ("runtime_id", "container_image"):
        if not expected_simulator.get(field) or (
            observed_simulator.get(field) != expected_simulator.get(field)
        ):
            blockers.append(f"deformable_preflight_simulator_identity_mismatch:{field}")
    if not simulator_root_text or not simulator_root.is_dir():
        blockers.append("deformable_preflight_simulator_root_missing")
    evidence["simulator_runtime"] = {
        "runtime_id": observed_simulator.get("runtime_id"),
        "container_image": observed_simulator.get("container_image"),
        "root_path": simulator_root_text,
    }
    return blockers, evidence


def _python_runtime_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected = _mapping(request.get("runtime_python"))
    observed = _mapping(observations.get("python_runtime"))
    for field in ("implementation", "version", "python_tag", "abi_tag"):
        if not expected.get(field) or observed.get(field) != expected.get(field):
            blockers.append(f"deformable_preflight_python_runtime_mismatch:{field}")
    expected_platforms = set(_strings(expected.get("platform_tags")))
    observed_platforms = set(_strings(observed.get("platform_tags")))
    if not expected_platforms or not expected_platforms.issubset(observed_platforms):
        blockers.append("deformable_preflight_python_platform_tags_incompatible")
    return blockers, {
        "expected": dict(expected),
        "observed": dict(observed),
        "missing_platform_tags": sorted(expected_platforms - observed_platforms),
    }


def _dependency_rows_by_package(
    value: Any, *, label: str, blockers: list[str]
) -> dict[str, Mapping[str, Any]]:
    rows = _rows(value)
    result: dict[str, Mapping[str, Any]] = {}
    if not rows:
        blockers.append(f"deformable_preflight_dependency_{label}_empty")
    for index, row in enumerate(rows):
        package = str(row.get("package") or "").strip()
        if not package:
            blockers.append(f"deformable_preflight_dependency_{label}_invalid:{index}")
            continue
        if package in result:
            blockers.append(f"deformable_preflight_dependency_{label}_duplicate:{package}")
        result[package] = row
    return result


def _wheel_tag_compatible(
    wheel_tag: str,
    *,
    python_tag: str,
    abi_tag: str,
    platform_tags: set[str],
) -> bool:
    parts = wheel_tag.split("-", 2)
    if len(parts) != 3:
        return False
    interpreter, abi, platform_tag = parts
    if platform_tag == "any":
        return interpreter in {"py3", python_tag} and abi == "none"
    return bool(
        interpreter == python_tag
        and abi in {abi_tag, "abi3"}
        and platform_tag in platform_tags
    )


def _dependency_closure_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected = _dependency_rows_by_package(
        request.get("dependency_closure"), label="requirement", blockers=blockers
    )
    observed = _dependency_rows_by_package(
        observations.get("installed_dependencies"),
        label="observation",
        blockers=blockers,
    )
    runtime = _mapping(observations.get("python_runtime"))
    python_tag = str(runtime.get("python_tag") or "")
    abi_tag = str(runtime.get("abi_tag") or "")
    platform_tags = set(_strings(runtime.get("platform_tags")))
    dependency_root_text = str(observations.get("dependency_root") or "").strip()
    dependency_root = (
        Path(dependency_root_text).expanduser()
        if dependency_root_text
        else Path("/__missing__")
    )
    if not dependency_root_text or not dependency_root.is_dir():
        blockers.append("deformable_preflight_dependency_root_missing")

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    for package in missing:
        blockers.append(f"deformable_preflight_dependency_missing:{package}")
    for package in unexpected:
        blockers.append(f"deformable_preflight_dependency_unexpected:{package}")
    for package in sorted(set(expected) & set(observed)):
        expected_row = expected[package]
        observed_row = observed[package]
        for field in ("version", "wheel_tag", "wheel_sha256", "import_module"):
            if not expected_row.get(field) or (
                observed_row.get(field) != expected_row.get(field)
            ):
                blockers.append(
                    f"deformable_preflight_dependency_identity_mismatch:{package}:{field}"
                )
        if not _valid_sha256(observed_row.get("wheel_sha256")):
            blockers.append(
                f"deformable_preflight_dependency_wheel_digest_invalid:{package}"
            )
        wheel_tag = str(observed_row.get("wheel_tag") or "")
        if not _wheel_tag_compatible(
            wheel_tag,
            python_tag=python_tag,
            abi_tag=abi_tag,
            platform_tags=platform_tags,
        ):
            blockers.append(
                f"deformable_preflight_dependency_wheel_abi_incompatible:{package}"
            )
        module_file_text = str(observed_row.get("module_file") or "").strip()
        module_file = (
            Path(module_file_text).expanduser()
            if module_file_text
            else Path("/__missing__")
        )
        if (
            not module_file_text
            or not module_file.is_file()
            or not _path_is_within(module_file, dependency_root)
        ):
            blockers.append(
                f"deformable_preflight_dependency_import_outside_closure:{package}"
            )
    return blockers, {
        "dependency_root": dependency_root_text,
        "required_packages": sorted(expected),
        "observed_packages": sorted(observed),
        "missing_packages": missing,
        "unexpected_packages": unexpected,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tags": sorted(platform_tags),
    }


def _embodiment_scope_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    robot_id = str(request.get("selected_robot_id") or "")
    expected_module = ROBOT_EMBODIMENT_MODULES.get(robot_id)
    selected = _mapping(observations.get("selected_embodiment"))
    if selected.get("robot_id") != robot_id:
        blockers.append("deformable_preflight_selected_robot_identity_mismatch")
    if not expected_module or selected.get("selected_module") != expected_module:
        blockers.append("deformable_preflight_selected_embodiment_module_mismatch")

    imported = set(_strings(observations.get("imported_modules")))
    embodiment_prefix = "isaaclab_arena.embodiments"
    imported_embodiments = sorted(
        module
        for module in imported
        if module == embodiment_prefix or module.startswith(embodiment_prefix + ".")
    )
    allowed_family_prefix = (
        expected_module.rsplit(".", 1)[0] if expected_module else "__unadmitted__"
    )
    unrelated = [
        module
        for module in imported_embodiments
        if module != embodiment_prefix
        and not (
            module == allowed_family_prefix
            or module.startswith(allowed_family_prefix + ".")
        )
    ]
    if unrelated:
        blockers.append("deformable_preflight_unrelated_embodiment_imported")

    arena_root_text = str(
        _mapping(_mapping(observations.get("source_roots")).get("arena")).get(
            "root_path"
        )
        or ""
    )
    module_file_text = str(selected.get("selected_module_file") or "").strip()
    module_file = (
        Path(module_file_text).expanduser() if module_file_text else Path("/__missing__")
    )
    arena_root = Path(arena_root_text).expanduser() if arena_root_text else Path("/__missing__")
    if (
        not module_file_text
        or not module_file.is_file()
        or not _path_is_within(module_file, arena_root)
    ):
        blockers.append("deformable_preflight_selected_embodiment_source_mismatch")
    return blockers, {
        "robot_id": robot_id,
        "expected_module": expected_module,
        "selected_module": selected.get("selected_module"),
        "selected_module_file": module_file_text,
        "imported_embodiment_modules": imported_embodiments,
        "unrelated_embodiment_modules": unrelated,
    }


def _symbol_check(
    *,
    check_id: str,
    required_symbols: Sequence[str],
    observations: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    available = set(_strings(observations.get("available_symbols")))
    module_files = _mapping(observations.get("module_files"))
    source_roots = _mapping(observations.get("source_roots"))
    isaaclab_root_text = str(_mapping(source_roots.get("isaaclab")).get("root_path") or "")
    arena_root_text = str(_mapping(source_roots.get("arena")).get("root_path") or "")
    simulator_root_text = str(
        _mapping(observations.get("simulator_runtime_identity")).get("root_path") or ""
    )
    dependency_root_text = str(observations.get("dependency_root") or "").strip()
    missing = sorted(set(required_symbols) - available)
    for symbol in missing:
        blockers.append(f"deformable_preflight_required_symbol_missing:{symbol}")
    required_modules = sorted({symbol.split(":", 1)[0] for symbol in required_symbols})
    invalid_module_files: list[str] = []
    for module in required_modules:
        module_file_text = str(module_files.get(module) or "").strip()
        module_file = (
            Path(module_file_text).expanduser()
            if module_file_text
            else Path("/__missing__")
        )
        if not module_file_text or not module_file.is_file():
            invalid_module_files.append(module)
            blockers.append(f"deformable_preflight_module_file_missing:{module}")
            continue
        if module.startswith("isaaclab_arena"):
            expected_root_text = arena_root_text
        elif module.startswith("isaaclab"):
            expected_root_text = isaaclab_root_text
        elif module in {"torch", "warp"} or module.startswith(("torch.", "warp.")):
            expected_root_text = dependency_root_text
        else:
            expected_root_text = simulator_root_text
        expected_root = (
            Path(expected_root_text).expanduser()
            if expected_root_text
            else Path("/__missing__")
        )
        if not expected_root_text or not _path_is_within(module_file, expected_root):
            blockers.append(f"deformable_preflight_module_outside_pinned_root:{module}")
    return blockers, {
        "check_id": check_id,
        "required_symbols": list(required_symbols),
        "missing_symbols": missing,
        "module_files": {
            module: module_files.get(module) for module in required_modules
        },
        "invalid_module_files": invalid_module_files,
    }


def _cuda_warp_declarations_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected = _mapping(request.get("cuda_warp_requirements"))
    observed = _mapping(observations.get("cuda_warp_declarations"))
    for field in ("torch_cuda_version", "warp_version", "selected_device"):
        if not expected.get(field) or observed.get(field) != expected.get(field):
            blockers.append(f"deformable_preflight_cuda_warp_mismatch:{field}")
    for field in ("torch_version", "cuda_driver_version", "device_name"):
        if not str(observed.get(field) or "").strip():
            blockers.append(f"deformable_preflight_cuda_warp_declaration_missing:{field}")
    selected_device = str(expected.get("selected_device") or "")
    warp_devices = set(_strings(observed.get("warp_devices")))
    if not selected_device or selected_device not in warp_devices:
        blockers.append("deformable_preflight_warp_selected_device_unavailable")

    minimum_capability = expected.get("minimum_compute_capability")
    observed_capability = observed.get("compute_capability")
    capability_valid = bool(
        isinstance(minimum_capability, Sequence)
        and not isinstance(minimum_capability, (str, bytes, bytearray))
        and isinstance(observed_capability, Sequence)
        and not isinstance(observed_capability, (str, bytes, bytearray))
        and len(minimum_capability) == 2
        and len(observed_capability) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in minimum_capability)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in observed_capability)
    )
    if not capability_valid:
        blockers.append("deformable_preflight_compute_capability_invalid")
    elif tuple(observed_capability) < tuple(minimum_capability):
        blockers.append("deformable_preflight_compute_capability_insufficient")
    return blockers, {
        "expected": dict(expected),
        "observed": dict(observed),
        "warp_selected_device_declared": selected_device in warp_devices,
        "execution_proven": False,
    }


def _media_tools_check(observations: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    rows = _rows(observations.get("media_tools"))
    by_name = {str(row.get("name") or ""): row for row in rows}
    evidence: dict[str, Any] = {}
    for name in ("ffmpeg", "ffprobe"):
        row = _mapping(by_name.get(name))
        executable = str(row.get("executable") or "").strip()
        returncode = row.get("returncode")
        version_line = str(row.get("version_line") or "").strip()
        if not executable or returncode != 0 or not version_line:
            blockers.append(f"deformable_preflight_media_tool_unavailable:{name}")
        evidence[name] = {
            "executable": executable,
            "returncode": returncode,
            "version_line": version_line,
        }
    return blockers, evidence


def _policy_identity_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected_rows = _rows(request.get("policy_identities"))
    observed_rows = _rows(observations.get("policy_identities"))
    expected = {str(row.get("candidate_id") or ""): row for row in expected_rows}
    observed = {str(row.get("candidate_id") or ""): row for row in observed_rows}
    if set(expected) != set(FROZEN_CANDIDATES):
        blockers.append("deformable_preflight_frozen_candidate_set_mismatch")
    if set(observed) != set(FROZEN_CANDIDATES):
        blockers.append("deformable_preflight_observed_candidate_set_mismatch")
    evidence: dict[str, Any] = {"candidates": {}}
    for candidate_id in FROZEN_CANDIDATES:
        expected_row = _mapping(expected.get(candidate_id))
        observed_row = _mapping(observed.get(candidate_id))
        expected_adapter = POLICY_ADAPTER_MODULES[candidate_id]
        candidate_blockers: list[str] = []
        if expected_row.get("adapter_module") != expected_adapter:
            candidate_blockers.append(
                f"deformable_preflight_policy_adapter_module_unfrozen:{candidate_id}"
            )
        if observed_row.get("adapter_module") != expected_adapter:
            candidate_blockers.append(
                f"deformable_preflight_policy_adapter_module_mismatch:{candidate_id}"
            )
        expected_adapter_sha = expected_row.get("adapter_sha256")
        observed_adapter_sha = observed_row.get("adapter_sha256")
        if (
            not _valid_sha256(expected_adapter_sha)
            or observed_adapter_sha != expected_adapter_sha
        ):
            candidate_blockers.append(
                f"deformable_preflight_policy_adapter_identity_mismatch:{candidate_id}"
            )
        adapter_file_text = str(observed_row.get("adapter_file") or "").strip()
        adapter_file = (
            Path(adapter_file_text).expanduser()
            if adapter_file_text
            else Path("/__missing__")
        )
        actual_adapter_sha = _sha256_file(adapter_file)
        if actual_adapter_sha != expected_adapter_sha:
            candidate_blockers.append(
                f"deformable_preflight_policy_adapter_file_mismatch:{candidate_id}"
            )

        expected_checkpoint = _mapping(expected_row.get("checkpoint_identity"))
        observed_checkpoint = _mapping(observed_row.get("checkpoint_identity"))
        if not expected_checkpoint:
            candidate_blockers.append(
                f"deformable_preflight_checkpoint_identity_missing:{candidate_id}"
            )
            expected_checkpoint_digest = None
        else:
            expected_checkpoint_digest = _canonical_digest(expected_checkpoint)
        observed_checkpoint_digest = (
            _canonical_digest(observed_checkpoint) if observed_checkpoint else None
        )
        if observed_checkpoint_digest != expected_checkpoint_digest:
            candidate_blockers.append(
                f"deformable_preflight_checkpoint_identity_mismatch:{candidate_id}"
            )
        blockers.extend(candidate_blockers)
        evidence["candidates"][candidate_id] = {
            "adapter_module": observed_row.get("adapter_module"),
            "adapter_file": adapter_file_text,
            "adapter_sha256": observed_adapter_sha,
            "actual_adapter_sha256": actual_adapter_sha,
            "expected_checkpoint_identity_digest": expected_checkpoint_digest,
            "observed_checkpoint_identity_digest": observed_checkpoint_digest,
            "blockers": sorted(set(candidate_blockers)),
        }
    return blockers, evidence


def _internal_runtime_modules_check(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    expected_rows = _rows(request.get("internal_runtime_modules"))
    observed_rows = _rows(observations.get("internal_runtime_modules"))
    expected = {str(row.get("module") or ""): row for row in expected_rows}
    observed = {str(row.get("module") or ""): row for row in observed_rows}
    for module in MINIMUM_INTERNAL_RUNTIME_MODULES:
        if module not in expected:
            blockers.append(f"deformable_preflight_internal_requirement_missing:{module}")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    for module in missing:
        blockers.append(f"deformable_preflight_internal_module_missing:{module}")
    for module in unexpected:
        blockers.append(f"deformable_preflight_internal_module_unexpected:{module}")
    module_evidence: dict[str, Any] = {}
    for module in sorted(set(expected) & set(observed)):
        expected_sha = expected[module].get("sha256")
        observed_sha = observed[module].get("sha256")
        module_file_text = str(observed[module].get("file") or "").strip()
        module_file = (
            Path(module_file_text).expanduser()
            if module_file_text
            else Path("/__missing__")
        )
        actual_sha = _sha256_file(module_file)
        if (
            not _valid_sha256(expected_sha)
            or observed_sha != expected_sha
            or actual_sha != expected_sha
        ):
            blockers.append(f"deformable_preflight_internal_module_identity_mismatch:{module}")
        module_evidence[module] = {
            "file": module_file_text,
            "expected_sha256": expected_sha,
            "observed_sha256": observed_sha,
            "actual_sha256": actual_sha,
        }
    return blockers, {
        "required_modules": sorted(expected),
        "observed_modules": sorted(observed),
        "missing_modules": missing,
        "unexpected_modules": unexpected,
        "module_identities": module_evidence,
    }


def _claim_boundary_check(
    observations: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    declared_model = str(observations.get("deformable_model") or "").strip()
    if declared_model != "volumetric_fem":
        blockers.append("deformable_preflight_volumetric_fem_model_not_declared")
    claims = set(_strings(observations.get("claimed_capabilities")))
    forbidden = sorted(
        claims
        & {
            "thin_shell_cloth",
            "independent_bend_shear",
            "physically_equivalent_towel",
        }
    )
    if forbidden:
        blockers.append("deformable_preflight_unsupported_cloth_claim")
    return blockers, {
        "declared_model": declared_model,
        "reported_claims": sorted(claims),
        "forbidden_claims": forbidden,
        "thin_shell_cloth_supported": False,
        "independent_bend_shear_supported": False,
        "physical_towel_equivalence_supported": False,
    }


def build_deformable_native_capability_preflight(
    *,
    request: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every static row and return one digest-bound capability matrix."""

    request_map = _mapping(request)
    observation_map = _mapping(observations)
    static_rows: list[dict[str, Any]] = []

    checks = (
        (
            "request_contract",
            "contract",
            lambda: _request_contract_check(request_map),
        ),
        (
            "runtime_source_roots_and_revisions",
            "runtime_sources",
            lambda: _source_identity_check(request_map, observation_map),
        ),
        (
            "python_runtime_abi_platform",
            "python_runtime",
            lambda: _python_runtime_check(request_map, observation_map),
        ),
        (
            "python_dependency_closure",
            "python_dependencies",
            lambda: _dependency_closure_check(request_map, observation_map),
        ),
        (
            "selected_robot_embodiment_scope",
            "arena_embodiment",
            lambda: _embodiment_scope_check(request_map, observation_map),
        ),
    )
    for check_id, category, check in checks:
        blockers, evidence = check()
        _add_row(
            static_rows,
            check_id=check_id,
            category=category,
            blockers=blockers,
            evidence=evidence,
        )

    for check_id, required_symbols in REQUIRED_SYMBOLS_BY_CHECK.items():
        blockers, evidence = _symbol_check(
            check_id=check_id,
            required_symbols=required_symbols,
            observations=observation_map,
        )
        _add_row(
            static_rows,
            check_id=check_id,
            category="runtime_api_inventory",
            blockers=blockers,
            evidence=evidence,
        )

    tail_checks = (
        (
            "cuda_warp_declarations",
            "accelerator_declarations",
            lambda: _cuda_warp_declarations_check(request_map, observation_map),
        ),
        (
            "ffmpeg_ffprobe",
            "media_tools",
            lambda: _media_tools_check(observation_map),
        ),
        (
            "frozen_policy_adapter_checkpoint_identities",
            "policy_runtime",
            lambda: _policy_identity_check(request_map, observation_map),
        ),
        (
            "internal_runtime_module_closure",
            "internal_runtime",
            lambda: _internal_runtime_modules_check(request_map, observation_map),
        ),
        (
            "deformable_claim_boundary",
            "claim_boundary",
            lambda: _claim_boundary_check(observation_map),
        ),
    )
    for check_id, category, check in tail_checks:
        blockers, evidence = check()
        _add_row(
            static_rows,
            check_id=check_id,
            category=category,
            blockers=blockers,
            evidence=evidence,
        )

    dynamic_rows = [
        {
            "check_id": check_id,
            "category": "native_canary",
            "phase": "dynamic_native_canary",
            "required": True,
            "status": "pending_native_canary",
            "blockers": [],
            "evidence": {
                "required_proof": required_proof,
                "static_preflight_cannot_prove": True,
            },
        }
        for check_id, required_proof in DYNAMIC_NATIVE_CANARY_GATES
    ]
    blockers = sorted(
        {
            blocker
            for row in static_rows
            for blocker in row["blockers"]
        }
    )
    static_passed = not blockers
    volumetric_static_available = all(
        next(row for row in static_rows if row["check_id"] == check_id)["status"]
        == "passed"
        for check_id in (
            "openusd_physx_volumetric_deformable_schemas",
            "volumetric_deformable_authoring_apis",
            "deformable_reset_state_readback_apis",
        )
    )
    result: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "preflight_id": str(request_map.get("preflight_id") or ""),
        "status": (
            "static_preflight_passed_native_canary_required"
            if static_passed
            else "blocked_static_preflight"
        ),
        "request_digest": _canonical_digest(request_map),
        "observation_digest": _canonical_digest(observation_map),
        "static_checks": static_rows,
        "dynamic_native_canary_gates": dynamic_rows,
        "static_checks_passed": static_passed,
        "native_canary_required": True,
        "native_canary_completed": False,
        "scene_run_admitted": False,
        "blockers": blockers,
        "claim_ceiling": {
            "volumetric_fem_static_api_available": volumetric_static_available,
            "volumetric_fem_native_qualified": False,
            "thin_shell_cloth_supported": False,
            "independent_bend_shear_supported": False,
            "physical_towel_equivalence_supported": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = _canonical_digest(result, digest_field="receipt_digest")
    return result


def write_deformable_native_capability_preflight(
    *,
    request: Mapping[str, Any],
    observations: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Evaluate and atomically retain one JSON capability matrix."""

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return result


__all__ = [
    "ALL_REQUIRED_SYMBOLS",
    "DYNAMIC_NATIVE_CANARY_GATES",
    "FROZEN_CANDIDATES",
    "MATRIX_SCHEMA_VERSION",
    "MINIMUM_INTERNAL_RUNTIME_MODULES",
    "POLICY_ADAPTER_MODULES",
    "REQUEST_SCHEMA_VERSION",
    "REQUIRED_SYMBOLS_BY_CHECK",
    "build_deformable_native_capability_preflight",
    "write_deformable_native_capability_preflight",
]
