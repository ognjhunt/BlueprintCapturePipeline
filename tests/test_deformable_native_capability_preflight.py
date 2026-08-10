from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from blueprint_pipeline import deformable_native_capability_preflight as preflight
from blueprint_pipeline.deformable_native_capability_preflight import (
    ALL_REQUIRED_SYMBOLS,
    DYNAMIC_NATIVE_CANARY_GATES,
    FROZEN_CANDIDATES,
    MINIMUM_INTERNAL_RUNTIME_MODULES,
    POLICY_ADAPTER_MODULES,
    REQUEST_SCHEMA_VERSION,
    REQUIRED_SYMBOLS_BY_CHECK,
    build_deformable_native_capability_preflight,
    write_deformable_native_capability_preflight,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ARENA_REPOSITORY,
    ARENA_TREE,
    ISAACLAB_COMMIT,
    ISAACLAB_REPOSITORY,
    ISAACLAB_TREE,
)


def _write(path: Path, value: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _symbol_module_files(
    *, isaaclab_root: Path, simulator_root: Path, dependency_root: Path
) -> dict[str, str]:
    known_isaaclab_files = {
        "isaaclab.sim.schemas.schemas": (
            "source/isaaclab/isaaclab/sim/schemas/schemas.py"
        ),
        "isaaclab.sim.spawners.materials.physics_materials": (
            "source/isaaclab/isaaclab/sim/spawners/materials/physics_materials.py"
        ),
        "isaaclab.sensors.contact_sensor.contact_sensor": (
            "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py"
        ),
        "isaaclab.sensors.contact_sensor.contact_sensor_cfg": (
            "source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py"
        ),
        "isaaclab.sensors.camera.camera": (
            "source/isaaclab/isaaclab/sensors/camera/camera.py"
        ),
        "isaaclab.sensors.camera.camera_cfg": (
            "source/isaaclab/isaaclab/sensors/camera/camera_cfg.py"
        ),
        "isaaclab.sim.simulation_context": (
            "source/isaaclab/isaaclab/sim/simulation_context.py"
        ),
        "isaaclab_physx.assets.deformable_object.deformable_object": (
            "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object.py"
        ),
        "isaaclab_physx.assets.deformable_object.deformable_object_cfg": (
            "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_cfg.py"
        ),
        "isaaclab_physx.assets.deformable_object.deformable_object_data": (
            "source/isaaclab_physx/isaaclab_physx/assets/deformable_object/deformable_object_data.py"
        ),
    }
    modules = {symbol.split(":", 1)[0] for symbol in ALL_REQUIRED_SYMBOLS}
    result: dict[str, str] = {}
    for module in modules:
        if module in known_isaaclab_files:
            path = isaaclab_root / known_isaaclab_files[module]
        elif module in {"torch", "warp"}:
            path = dependency_root / module / "__init__.py"
            _write(path, f"# {module}\n")
        else:
            path = simulator_root / "python" / (module.replace(".", "/") + ".py")
            _write(path, f"# {module}\n")
        result[module] = str(path)
    return result


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    isaaclab_root = tmp_path / "runtime/isaaclab"
    arena_root = tmp_path / "runtime/arena"
    simulator_root = tmp_path / "runtime/isaac-sim"
    dependency_root = tmp_path / "runtime/dependencies"
    internal_root = tmp_path / "bundle"
    for root in (
        isaaclab_root,
        arena_root,
        simulator_root,
        dependency_root,
        internal_root,
    ):
        root.mkdir(parents=True)
    for relative in preflight.ISAACLAB_REQUIRED_RELATIVE_PATHS:
        _write(isaaclab_root / relative, f"# {relative}\n")
    for relative in preflight.ARENA_REQUIRED_RELATIVE_PATHS:
        _write(arena_root / relative, f"# {relative}\n")

    dependency_requirements = [
        {
            "package": "fixture-pure",
            "version": "1.0.0",
            "wheel_tag": "py3-none-any",
            "wheel_sha256": _sha("a"),
            "import_module": "fixture_pure",
        },
        {
            "package": "fixture-binary",
            "version": "2.0.0",
            "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
            "wheel_sha256": _sha("b"),
            "import_module": "fixture_binary",
        },
    ]
    installed_dependencies = []
    for row in dependency_requirements:
        module_file = _write(
            dependency_root / str(row["import_module"]) / "__init__.py"
        )
        installed_dependencies.append({**row, "module_file": str(module_file)})

    internal_requirements = []
    internal_observations = []
    internal_paths: dict[str, Path] = {}
    for module in MINIMUM_INTERNAL_RUNTIME_MODULES:
        path = _write(internal_root / (module.replace(".", "/") + ".py"), module)
        digest = _sha256(path)
        internal_paths[module] = path
        internal_requirements.append({"module": module, "sha256": digest})
        internal_observations.append(
            {"module": module, "sha256": digest, "file": str(path)}
        )

    checkpoint_identities = {
        "pi05_droid": {
            "checkpoint_uri": "gs://fixture/pi05",
            "checkpoint_inventory_sha256": _sha("c"),
            "object_count": 3,
            "size_bytes": 1234,
        },
        "groot_n17_droid": {
            "model_id": "nvidia/GR00T-N1.7-DROID",
            "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
            "checkpoint_files_sha256": _sha("d"),
        },
    }
    policy_requirements = []
    policy_observations = []
    for candidate_id in FROZEN_CANDIDATES:
        adapter_module = POLICY_ADAPTER_MODULES[candidate_id]
        adapter_file = internal_paths[adapter_module]
        adapter_sha = _sha256(adapter_file)
        policy_requirements.append(
            {
                "candidate_id": candidate_id,
                "adapter_module": adapter_module,
                "adapter_sha256": adapter_sha,
                "checkpoint_identity": checkpoint_identities[candidate_id],
            }
        )
        policy_observations.append(
            {
                "candidate_id": candidate_id,
                "adapter_module": adapter_module,
                "adapter_file": str(adapter_file),
                "adapter_sha256": adapter_sha,
                "checkpoint_identity": checkpoint_identities[candidate_id],
            }
        )

    simulator_identity = {
        "runtime_id": "isaac-sim-fixture",
        "container_image": "fixture.invalid/isaac-sim@sha256:" + "e" * 64,
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "preflight_id": "fixture-deformable-preflight",
        "selected_robot_id": "franka_panda",
        "simulator_runtime_identity": simulator_identity,
        "runtime_python": {
            "implementation": "CPython",
            "version": "3.12.11",
            "python_tag": "cp312",
            "abi_tag": "cp312",
            "platform_tags": [
                "manylinux_2_28_x86_64",
                "manylinux_2_17_x86_64",
            ],
        },
        "dependency_closure": dependency_requirements,
        "cuda_warp_requirements": {
            "torch_cuda_version": "12.8",
            "warp_version": "1.8.1",
            "selected_device": "cuda:0",
            "minimum_compute_capability": [8, 0],
        },
        "policy_identities": policy_requirements,
        "internal_runtime_modules": internal_requirements,
    }
    observations = {
        "source_roots": {
            "isaaclab": {
                "root_path": str(isaaclab_root),
                "repository": ISAACLAB_REPOSITORY,
                "revision": ISAACLAB_COMMIT,
                "tree": ISAACLAB_TREE,
                "source_receipt_digest": _sha("1"),
            },
            "arena": {
                "root_path": str(arena_root),
                "repository": ARENA_REPOSITORY,
                "revision": ARENA_COMMIT,
                "tree": ARENA_TREE,
                "source_receipt_digest": _sha("2"),
            },
        },
        "simulator_runtime_identity": {
            **simulator_identity,
            "root_path": str(simulator_root),
        },
        "python_runtime": copy.deepcopy(request["runtime_python"]),
        "dependency_root": str(dependency_root),
        "installed_dependencies": installed_dependencies,
        "selected_embodiment": {
            "robot_id": "franka_panda",
            "selected_module": "isaaclab_arena.embodiments.droid.droid",
            "selected_module_file": str(
                arena_root / "isaaclab_arena/embodiments/droid/droid.py"
            ),
        },
        "imported_modules": [
            "isaaclab_arena.embodiments",
            "isaaclab_arena.embodiments.droid",
            "isaaclab_arena.embodiments.droid.droid",
        ],
        "available_symbols": list(ALL_REQUIRED_SYMBOLS),
        "module_files": _symbol_module_files(
            isaaclab_root=isaaclab_root,
            simulator_root=simulator_root,
            dependency_root=dependency_root,
        ),
        "cuda_warp_declarations": {
            "torch_version": "2.7.0",
            "torch_cuda_version": "12.8",
            "warp_version": "1.8.1",
            "cuda_driver_version": "570.86.15",
            "selected_device": "cuda:0",
            "warp_devices": ["cpu", "cuda:0"],
            "device_name": "Fixture GPU",
            "compute_capability": [8, 9],
        },
        "media_tools": [
            {
                "name": "ffmpeg",
                "executable": "/fixture/bin/ffmpeg",
                "returncode": 0,
                "version_line": "ffmpeg version fixture",
            },
            {
                "name": "ffprobe",
                "executable": "/fixture/bin/ffprobe",
                "returncode": 0,
                "version_line": "ffprobe version fixture",
            },
        ],
        "policy_identities": policy_observations,
        "internal_runtime_modules": internal_observations,
        "deformable_model": "volumetric_fem",
        "claimed_capabilities": ["volumetric_fem"],
    }
    return request, observations


def _row(result: dict, check_id: str) -> dict:
    return next(row for row in result["static_checks"] if row["check_id"] == check_id)


def test_complete_static_matrix_passes_but_requires_every_dynamic_canary(
    tmp_path: Path,
) -> None:
    request, observations = _fixture(tmp_path)

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert result["status"] == "static_preflight_passed_native_canary_required"
    assert result["static_checks_passed"] is True
    assert result["blockers"] == []
    assert result["native_canary_required"] is True
    assert result["native_canary_completed"] is False
    assert result["scene_run_admitted"] is False
    assert {row["check_id"] for row in result["dynamic_native_canary_gates"]} == {
        check_id for check_id, _ in DYNAMIC_NATIVE_CANARY_GATES
    }
    assert {
        row["status"] for row in result["dynamic_native_canary_gates"]
    } == {"pending_native_canary"}
    assert all(row["status"] == "passed" for row in result["static_checks"])


def test_retained_matrix_is_digest_bound_and_portable(tmp_path: Path) -> None:
    request, observations = _fixture(tmp_path)
    output = tmp_path / "evidence/deformable_native_preflight.json"

    result = write_deformable_native_capability_preflight(
        request=request,
        observations=observations,
        output_path=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["request_digest"].startswith("sha256:")
    assert result["observation_digest"].startswith("sha256:")
    assert result["receipt_digest"].startswith("sha256:")
    assert not output.with_name(output.name + ".tmp").exists()


def test_every_static_failure_is_aggregated_in_one_matrix(tmp_path: Path) -> None:
    request, observations = _fixture(tmp_path)
    observations["source_roots"]["isaaclab"]["revision"] = "0" * 40
    observations["installed_dependencies"].pop(0)
    observations["imported_modules"].append("isaaclab_arena.embodiments.g1.g1")
    missing_symbol = REQUIRED_SYMBOLS_BY_CHECK["contact_sensor_apis"][0]
    observations["available_symbols"].remove(missing_symbol)
    observations["media_tools"][1]["returncode"] = 1
    observations["policy_identities"][0]["checkpoint_identity"] = {"wrong": "bytes"}
    observations["internal_runtime_modules"].pop()
    observations["deformable_model"] = "thin_shell_cloth"
    observations["claimed_capabilities"].append("thin_shell_cloth")

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert result["status"] == "blocked_static_preflight"
    assert result["static_checks_passed"] is False
    assert "deformable_preflight_source_revision_mismatch:isaaclab" in result["blockers"]
    assert "deformable_preflight_dependency_missing:fixture-pure" in result["blockers"]
    assert "deformable_preflight_unrelated_embodiment_imported" in result["blockers"]
    assert (
        f"deformable_preflight_required_symbol_missing:{missing_symbol}"
        in result["blockers"]
    )
    assert "deformable_preflight_media_tool_unavailable:ffprobe" in result["blockers"]
    assert (
        "deformable_preflight_checkpoint_identity_mismatch:pi05_droid"
        in result["blockers"]
    )
    assert any(
        blocker.startswith("deformable_preflight_internal_module_missing:")
        for blocker in result["blockers"]
    )
    assert "deformable_preflight_unsupported_cloth_claim" in result["blockers"]
    assert "deformable_preflight_volumetric_fem_model_not_declared" in result["blockers"]
    assert len(result["static_checks"]) == 5 + len(REQUIRED_SYMBOLS_BY_CHECK) + 5
    assert all(
        row["status"] == "pending_native_canary"
        for row in result["dynamic_native_canary_gates"]
    )


def test_binary_wheel_abi_and_platform_are_checked_without_rejecting_pure_python(
    tmp_path: Path,
) -> None:
    request, observations = _fixture(tmp_path)
    binary = next(
        row
        for row in observations["installed_dependencies"]
        if row["package"] == "fixture-binary"
    )
    binary["wheel_tag"] = "cp311-cp311-manylinux_2_17_x86_64"

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert (
        "deformable_preflight_dependency_wheel_abi_incompatible:fixture-binary"
        in result["blockers"]
    )
    assert not any(
        blocker.endswith(":fixture-pure") and "abi_incompatible" in blocker
        for blocker in result["blockers"]
    )
    assert _row(result, "python_dependency_closure")["status"] == "blocked"


def test_loaded_api_module_cannot_shadow_the_pinned_source_root(tmp_path: Path) -> None:
    request, observations = _fixture(tmp_path)
    module = "isaaclab_physx.assets.deformable_object.deformable_object"
    shadow = _write(tmp_path / "shadow/deformable_object.py", "# shadow\n")
    observations["module_files"][module] = str(shadow)

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert (
        f"deformable_preflight_module_outside_pinned_root:{module}"
        in result["blockers"]
    )
    assert _row(result, "volumetric_deformable_authoring_apis")["status"] == "blocked"


def test_exact_policy_checkpoint_identity_is_required_for_both_candidates(
    tmp_path: Path,
) -> None:
    request, observations = _fixture(tmp_path)
    observations["policy_identities"] = [
        row
        for row in observations["policy_identities"]
        if row["candidate_id"] != "groot_n17_droid"
    ]

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert "deformable_preflight_observed_candidate_set_mismatch" in result["blockers"]
    assert (
        "deformable_preflight_policy_adapter_module_mismatch:groot_n17_droid"
        in result["blockers"]
    )
    assert (
        "deformable_preflight_checkpoint_identity_mismatch:groot_n17_droid"
        in result["blockers"]
    )


def test_static_volume_api_never_becomes_a_thin_cloth_or_native_claim(
    tmp_path: Path,
) -> None:
    request, observations = _fixture(tmp_path)

    result = build_deformable_native_capability_preflight(
        request=request,
        observations=observations,
    )

    assert result["claim_ceiling"] == {
        "volumetric_fem_static_api_available": True,
        "volumetric_fem_native_qualified": False,
        "thin_shell_cloth_supported": False,
        "independent_bend_shear_supported": False,
        "physical_towel_equivalence_supported": False,
    }
    assert _row(result, "contact_sensor_apis")["status"] == "passed"
    assert next(
        row
        for row in result["dynamic_native_canary_gates"]
        if row["check_id"] == "dynamic_genuine_gripper_deformable_contact"
    )["status"] == "pending_native_canary"
