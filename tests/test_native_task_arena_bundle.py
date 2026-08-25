from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import (
    native_task_arena_construction_bundle as construction_bundle_module,
)
from blueprint_pipeline import native_task_arena_controls_bundle as controls_bundle_module
from blueprint_pipeline import native_task_arena_policy_bundle as policy_bundle_module
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.adp009d_policy_rights import build_candidate_policy_rights
from blueprint_pipeline.adp009d_scene_policy_readiness import (
    load_scene_policy_readiness,
)
from blueprint_pipeline.native_task_arena_bundle import (
    NativeTaskArenaBundleError,
    _entrypoint,
    build_native_task_arena_bundle,
)
from blueprint_pipeline.native_task_arena_construction_bundle import (
    CONSTRUCTION_RUNTIME_MODULE_NAMES,
    PROBE_KIND,
    build_native_task_arena_construction_bundle,
    load_verified_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_controls_bundle import (
    CONTROLS_RUNTIME_MODULE_NAMES,
    PROBE_KIND as CONTROLS_PROBE_KIND,
    build_native_task_arena_controls_bundle,
    load_verified_native_task_arena_controls_bundle,
)
from blueprint_pipeline.native_task_arena_execution_contract import (
    EXECUTION_MODE_CONTRACTS,
    NATIVE_TASK_ARENA_POLICY_CANDIDATES,
    native_task_arena_execution_transport_completed,
)
from blueprint_pipeline.native_task_arena_policy_bundle import (
    ADP009D_POLICY_READINESS_PATH,
    ADP009D_SCENARIO_SUITE_PATH,
    build_native_task_arena_policy_bundle,
    build_native_task_policy_execution_spec,
    load_verified_native_task_arena_policy_bundle,
    materialize_native_task_policy_execution_spec,
    validate_native_task_policy_execution_spec,
)
from blueprint_pipeline.native_task_arena_runtime_preflight_bundle import (
    PROBE_KIND as RUNTIME_PREFLIGHT_PROBE_KIND,
    RESULT_FILENAME as RUNTIME_PREFLIGHT_RESULT_FILENAME,
    build_native_task_arena_runtime_preflight_bundle,
    load_verified_native_task_arena_runtime_preflight_bundle,
)
from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
    _plain_nurec_volume_contract,
)
from blueprint_pipeline.native_task_arena_vast import (
    run_native_task_arena_controls_vast,
    run_native_task_arena_policy_diagnostic_vast,
    run_native_task_arena_policy_vast,
    run_native_task_arena_runtime_preflight_vast,
    run_native_task_arena_vast,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ISAACLAB_PACKAGE_NAMES,
    RUNTIME_DEPENDENCY_WHEELS,
    materialize_native_task_runtime_source_packet,
)
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from blueprint_pipeline.droid_policy_bridge import OPENPI_SOURCE_REVISION
from blueprint_pipeline.groot_n17_droid_policy_runtime import (
    GrootN17DroidPolicySpec,
)
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
    _resolve_probe_image,
    _select_offer,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_execution_contract_freezes_all_modes_and_candidates() -> None:
    assert set(EXECUTION_MODE_CONTRACTS) == {
        "runtime_preflight",
        "construction_canary",
        "controls",
        "policy",
        "policy_diagnostic",
    }


def test_transport_accepts_only_exact_nonqualifying_downstream_diagnostic() -> None:
    request = {
        "schema_version": (
            "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1"
        ),
        "enabled": True,
        "development_only": True,
        "qualification_effect": "none",
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request["status"] = "requested"
    diagnostic = {
        "schema_version": (
            "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
        ),
        "status": "measured",
        "phase5_qualified": False,
        "qualification_effect": "none",
        "control_passed": False,
        "receipt_digest": "",
    }
    diagnostic["receipt_digest"] = canonical_digest(
        diagnostic, digest_field="receipt_digest"
    )
    result = {
        "schema_version": "native_task_arena_control_result.v1",
        "status": "diagnostic_completed",
        "controls_qualified": False,
        "qualification_effect": "none",
        "development_only": True,
        "diagnostic_only": True,
        "phase5_qualified": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "blockers": [],
        "phase_reached": (
            "synthetic_post_phase5_downstream_diagnostic_complete"
        ),
        "synthetic_post_phase5_downstream_diagnostic_request": request,
        "synthetic_post_phase5_downstream_diagnostic": diagnostic,
        "downstream_phase_posture_matrix": {
            "status": "not_run",
            "executed_cell_count": 0,
            "represented_configuration_count": 0,
        },
    }
    assert native_task_arena_execution_transport_completed(
        result,
        expected_output_filename="native_task_arena_control_result.v1.json",
    )
    assert not native_task_arena_execution_transport_completed(
        result,
        expected_output_filename="native_task_arena_policy_result.v1.json",
    )
    for field, value in (
        ("controls_qualified", True),
        ("qualification_effect", "qualify"),
        ("control_pair", {}),
    ):
        invalid = json.loads(json.dumps(result))
        invalid[field] = value
        assert not native_task_arena_execution_transport_completed(
            invalid,
            expected_output_filename=(
                "native_task_arena_control_result.v1.json"
            ),
        )
    invalid_digest = json.loads(json.dumps(result))
    invalid_digest[
        "synthetic_post_phase5_downstream_diagnostic"
    ]["receipt_digest"] = "sha256:" + "0" * 64
    assert not native_task_arena_execution_transport_completed(
        invalid_digest,
        expected_output_filename="native_task_arena_control_result.v1.json",
    )
    assert NATIVE_TASK_ARENA_POLICY_CANDIDATES == {
        "groot_n17_droid",
        "pi05_droid",
    }


def _packet(root: Path, *, scene_id: str) -> Path:
    packet = root / f"packet-{scene_id}"
    assets = packet / "assets"
    assets.mkdir(parents=True)
    source_bindings = []
    for role in ("scene_collision", "scene_appearance", "task_object"):
        path = assets / f"{role}.usd"
        path.write_text(f"exact:{scene_id}:{role}\n", encoding="utf-8")
        source_bindings.append(
            {
                "semantic_role": role,
                "source": {"root": "evidence", "relative_path": path.name},
                "staged_relative_path": f"assets/{path.name}",
                "staged_size_bytes": path.stat().st_size,
                "staged_sha256": _sha(path),
            }
        )
    documents = {
        "native_task_arena_packet_request.v1.json": {"scene_id": scene_id},
        "native_task_runtime_contract.v1.json": {"contract_digest": "sha256:" + "c" * 64},
        "native_task_arena_scene_plan.v1.json": {"plan_digest": "sha256:" + "p" * 64},
    }
    artifacts = []
    for role, (name, value) in zip(
        ("packet_request", "runtime_contract", "arena_scene_plan"),
        documents.items(),
        strict=True,
    ):
        path = packet / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append(
            {
                "role": role,
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    receipt = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "status": "construction_packet_completed",
        "scene_id": scene_id,
        "task_id": f"task-{scene_id}",
        "request_digest": "sha256:" + "a" * 64,
        "runtime_contract_digest": "sha256:" + "c" * 64,
        "arena_scene_plan_digest": "sha256:" + "b" * 64,
        "scenario_instance_digest": "sha256:" + "d" * 64,
        "source_bindings": source_bindings,
        "artifacts": artifacts,
        "source_bytes_mutated": False,
        "native_application_claimed": False,
        "policy_episode_claimed": False,
        "simulator_execution_is_not_physical_truth": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    (packet / "native_task_arena_packet_receipt.v1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return packet


def _runtime_source_packet(root: Path) -> Path:
    destination = root / "runtime-source-packet"
    receipt_path = destination / "native_task_runtime_source_packet.v1.json"
    if receipt_path.is_file():
        return receipt_path

    def repository(path: Path, *, arena: bool) -> tuple[Path, str, str]:
        path.mkdir(parents=True)
        subprocess.run(["git", "-C", str(path), "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "fixture@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Fixture"],
            check=True,
        )
        if arena:
            files = {
                "LICENSE.md": "Apache fixture\n",
                "setup.py": "from setuptools import setup; setup(name='isaaclab_arena')\n",
                "pyproject.toml": "[build-system]\nrequires=['setuptools']\n",
                "extension.toml": "[package]\nversion='fixture'\n",
                "isaaclab_arena/__init__.py": "VERSION='fixture'\n",
            }
        else:
            files = {
                "LICENSE": "BSD fixture\n",
                "apps/isaaclab.python.kit": (
                    "[dependencies]\n\"omni.physics.physx\" = {}\n"
                    "[settings.app.extensions]\n"
                    "excluded = [\"omni.warp.core\"]\n"
                ),
                "apps/isaaclab.python.headless.kit": (
                    "[dependencies]\n\"omni.physics.physx\" = {}\n"
                    "[settings]\n"
                    "app.extensions.excluded = [\"omni.warp.core\"]\n"
                ),
                "apps/isaaclab.python.headless.rendering.kit": (
                    "[dependencies]\n\"isaaclab.python.headless\" = {}\n"
                ),
            }
            for name in ISAACLAB_PACKAGE_NAMES:
                files[f"source/{name}/setup.py"] = (
                    "from setuptools import setup; "
                    f"setup(name='{name}', install_requires="
                    f"{['warp-lang==1.13.0', 'torch>=2.10'] if name == 'isaaclab' else []!r})\n"
                )
                files[f"source/{name}/pyproject.toml"] = (
                    "[build-system]\nrequires=['setuptools']\n"
                )
                files[f"source/{name}/config/extension.toml"] = (
                    "[package]\nversion='fixture'\n"
                )
                files[f"source/{name}/{name}/__init__.py"] = "VERSION='fixture'\n"
        for relative, value in files.items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value, encoding="utf-8")
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "-m", "fixture"],
            check=True,
            capture_output=True,
        )

        def git_value(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(path), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        return path, git_value("rev-parse", "HEAD"), git_value("rev-parse", "HEAD^{tree}")

    isaaclab, isaaclab_commit, isaaclab_tree = repository(
        root / "runtime-source-repos/isaaclab", arena=False
    )
    arena, arena_commit, arena_tree = repository(
        root / "runtime-source-repos/arena", arena=True
    )
    wheelhouse = root / "runtime-source-wheelhouse"
    wheelhouse.mkdir()
    for contract in RUNTIME_DEPENDENCY_WHEELS:
        distribution = contract["filename"].split("-", 1)[0]
        dist_info = f"{distribution}-{contract['version']}.dist-info"
        pure_python = bool(contract.get("pure_python", True))
        wheel_tag = str(contract.get("wheel_tag", "py3-none-any"))
        with zipfile.ZipFile(wheelhouse / contract["filename"], "w") as archive:
            archive.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\n"
                f"Root-Is-Purelib: {str(pure_python).lower()}\n"
                f"Tag: {wheel_tag}\n",
            )
            archive.writestr(f"{distribution}/__init__.py", "FIXTURE = True\n")
            if contract["package"] == "setuptools":
                archive.writestr(
                    "setuptools/_vendor/example-1.0.dist-info/WHEEL",
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                )
    materialize_native_task_runtime_source_packet(
        output_dir=destination,
        isaaclab_repo=isaaclab,
        arena_repo=arena,
        dependency_wheel_dir=wheelhouse,
        generated_at="fixed",
        isaaclab_commit=isaaclab_commit,
        isaaclab_tree=isaaclab_tree,
        isaaclab_runtime_compatibility_commit=isaaclab_commit,
        isaaclab_runtime_compatibility_tree=isaaclab_tree,
        arena_commit=arena_commit,
        arena_tree=arena_tree,
    )
    return receipt_path


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_rigid_and_articulated_packets_use_the_same_bundle_contract(
    tmp_path: Path, scene_id: str
) -> None:
    packet = _packet(tmp_path, scene_id=scene_id)
    worker = tmp_path / f"worker-{scene_id}.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    module = tmp_path / "runtime_helper.py"
    module.write_text("HELPER = 1\n", encoding="utf-8")

    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / f"job-{scene_id}",
        packet_dir=packet,
        worker_source=worker,
        runtime_module_sources=[module],
        implementation_commit="a" * 40,
        generated_at="fixed",
    )

    assert receipt["status"] == "ready"
    assert receipt["scene_reconstructed_by_bundle"] is False
    assert receipt["packet_receipt_digest"].startswith("sha256:")
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert (
            "provider_runtime/native_task_packet/assets/task_object.usd" in names
        )
        assert "provider_runtime/blueprint_pipeline/runtime_helper.py" in names
        assert archive.read(
            "provider_runtime/adp_arena_provider_runner.py"
        ) == worker.read_bytes()


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_bound_runtime_inputs_are_immutable_and_scene_neutral(
    tmp_path: Path, scene_id: str
) -> None:
    packet = _packet(tmp_path, scene_id=scene_id)
    worker = tmp_path / f"worker-{scene_id}.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    bound = tmp_path / f"input-{scene_id}.json"
    bound.write_text(json.dumps({"scene_id": scene_id}) + "\n", encoding="utf-8")

    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / f"bound-{scene_id}",
        packet_dir=packet,
        worker_source=worker,
        runtime_module_sources=[],
        implementation_commit="b" * 40,
        execution_mode="controls",
        expected_output_filename="controls.json",
        bound_runtime_inputs={"qualification/input.json": bound},
        generated_at="fixed",
    )

    assert receipt["bound_runtime_inputs"] == [
        {
            "relative_path": "runtime_inputs/qualification/input.json",
            "size_bytes": bound.stat().st_size,
            "sha256": _sha(bound),
        }
    ]
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        assert (
            archive.read("provider_runtime/runtime_inputs/qualification/input.json")
            == bound.read_bytes()
        )
    assert not (
        Path(receipt["bundle_path"]).parent
        / "provider_runtime/runtime_inputs/qualification/input.json"
    ).exists()


def _articulated_packet(root: Path) -> tuple[Path, dict]:
    packet = _packet(root, scene_id="840796")
    motion = {
        "schema_version": "native_articulated_motion_geometry.v1",
        "target_joint_id": "fixture_hinge",
        "hinge_point_world_m": [0.0, 0.0, 1.0],
        "hinge_axis_world_unit": [0.0, 0.0, 1.0],
        "handle_grasp_point_closed_world_m": [0.5, 0.0, 1.0],
        "authored_limits_degrees": [0.0, 90.0],
        "scripted_sweep_angle_degrees": 50.0,
        "motion_geometry_digest": "",
    }
    motion["motion_geometry_digest"] = canonical_digest(
        motion, digest_field="motion_geometry_digest"
    )
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_id": "fixture_articulated_task",
        "task_kind": "articulated_open_close",
        "robot": {"robot_id": "franka_panda"},
        "scenario": {"cell_id": "articulated-canonical", "seed": 17},
        "articulation": {"motion_geometry": motion},
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
            "prompt": "Open the articulated fixture.",
            "settle_window_samples": 40,
            "maximum_action_steps": 450,
        },
        "plan_digest": "",
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    plan_path.write_text(json.dumps(scene, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arena_scene_plan_digest"] = scene["plan_digest"]
    artifact = next(
        row for row in receipt["artifacts"] if row["role"] == "arena_scene_plan"
    )
    artifact["size_bytes"] = plan_path.stat().st_size
    artifact["sha256"] = _sha(plan_path)
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return packet, scene


def _qualified_construction(root: Path, scene: dict) -> Path:
    clearance = {
        "scene_plan_digest": scene["plan_digest"],
        "phases": [{"phase_id": "approach"}],
        "plan_digest": "",
    }
    clearance["plan_digest"] = canonical_digest(
        clearance, digest_field="plan_digest"
    )
    result = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed",
        "construction_gate_qualified": True,
        "blockers": [],
        "scene_plan_digest": scene["plan_digest"],
        "phase_results": [{"phase_id": "approach", "target_reached": True}],
        "camera_gates": {
            role: {"passed": True} for role in ("external", "wrist", "overview")
        },
        "reset_replay": {"passed": True},
        "construction_phase_plan": clearance,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path = root / "native_task_arena_construction_result.v1.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def test_executable_construction_bundle_freezes_local_phase_plan(
    tmp_path: Path,
) -> None:
    packet, _scene = _articulated_packet(tmp_path)
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "construction-with-phase-plan",
        packet_dir=packet,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="9" * 40,
        generated_at="fixed",
    )

    assert [row["relative_path"] for row in receipt["bound_runtime_inputs"]] == [
        "runtime_inputs/native_task_construction_phase_plan.v1.json"
    ]
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        frozen = json.loads(
            archive.read(
                "provider_runtime/runtime_inputs/"
                "native_task_construction_phase_plan.v1.json"
            )
        )
    # The bundle is what reaches the GPU, so the two bounds the servo executes
    # have to be frozen into it alongside the tolerances.
    assert frozen["execution_parameters"] == {
        "arrival_tolerance_m": 0.02,
        "stable_samples": 2,
        "maximum_steps_per_phase": 64,
        "articulated_waypoint_count": 8,
        "max_joint_delta_rad": 0.10,
        "max_joint_setpoint_lead_rad": 1.00,
        "velocity_feedforward_scale": 1.0,
    }
    assert frozen["plan_digest"] == canonical_digest(
        frozen, digest_field="plan_digest"
    )


def _qualified_controls(root: Path, scene: dict, construction: Path) -> Path:
    construction_result = json.loads(construction.read_text())
    pair = {
        "schema_version": "adp_task_control_pair.v1",
        "cell_id": scene["scenario"]["cell_id"],
        "task_spec_digest": canonical_digest(scene["task_spec"]),
        "cell_admitted_for_policy_execution": True,
        "policy_execution_blockers": [],
        "candidate_policy_queried": False,
        "pair_digest": "",
    }
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    result = {
        "schema_version": "native_task_arena_control_result.v1",
        "status": "completed",
        "controls_qualified": True,
        "scene_plan_digest": scene["plan_digest"],
        "construction_result_digest": construction_result["result_digest"],
        "control_pair": pair,
        "candidate_policy_queried": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    path = root / "native_task_arena_control_result.v1.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def _policy_spec(scene: dict, construction: Path, controls: Path) -> dict:
    construction_result = json.loads(construction.read_text())
    control_result = json.loads(controls.read_text())
    inventory = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/experiments/policy_ranking_thesis_20260726/"
            "openpi_polaris_checkpoint_inventory.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        row
        for row in inventory["entries"]
        if row["policy_id"] == "pi05_droid_jointpos_polaris"
    )
    spec = {
        "schema_version": "native_task_arena_policy_execution_spec.v1",
        "candidate_id": "pi05_droid",
        "task_id": scene["task_id"],
        "cell_id": scene["scenario"]["cell_id"],
        "prompt": "Open the articulated fixture.",
        "scene_plan_digest": scene["plan_digest"],
        "construction_result_digest": construction_result["result_digest"],
        "control_result_digest": control_result["result_digest"],
        "control_pair_digest": control_result["control_pair"]["pair_digest"],
        "policy_endpoint": {
            "host": "127.0.0.1",
            "port": 8000,
            "credential_env": "BLUEPRINT_PI05_API_KEY",
        },
        "policy_spec": {
            "policy_id": "pi05_droid_jointpos_polaris",
            "config_name": "pi05_droid_jointpos_polaris",
            "checkpoint_uri": entry["checkpoint_uri"],
            "checkpoint_object_manifest_sha256": entry[
                "legacy_object_manifest_sha256"
            ],
            "checkpoint_generation_manifest_sha256": entry[
                "generation_manifest_sha256"
            ],
            "checkpoint_inventory_sha256": inventory["inventory_sha256"],
            "checkpoint_object_count": entry["object_count"],
            "checkpoint_size_bytes": entry["size_bytes"],
            "action_space": "joint_position",
            "action_chunk_rows": 15,
            "open_loop_horizon": 8,
            "openpi_revision": OPENPI_SOURCE_REVISION,
        },
        "policy_identity_receipt": {"identity_verified": True},
        "max_policy_queries": 56,
        "open_loop_horizon": 8,
        "overview_camera_policy_input": False,
        "policy_may_grade_itself": False,
        "execution_authority": "qualified_controls_evaluation",
        "execution_spec_digest": "",
    }
    readiness = load_scene_policy_readiness(
        ADP009D_POLICY_READINESS_PATH,
        scenario_suite_path=ADP009D_SCENARIO_SUITE_PATH,
    )
    spec["candidate_rights_binding"] = build_candidate_policy_rights(
        readiness,
        candidate_id="pi05_droid",
        policy_spec=spec["policy_spec"],
        runtime_robot_id=scene["robot"]["robot_id"],
        scene_plan_digest=scene["plan_digest"],
    )
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )
    return spec


def _groot_policy_spec(scene: dict, construction: Path, controls: Path) -> dict:
    spec = _policy_spec(scene, construction, controls)
    groot = GrootN17DroidPolicySpec()
    spec.update(
        {
            "candidate_id": "groot_n17_droid",
            "policy_endpoint": {
                "host": "127.0.0.1",
                "port": 5555,
                "credential_env": "BLUEPRINT_GROOT_API_TOKEN",
            },
            "policy_spec": {
                "model_id": groot.model_id,
                "embodiment_tag": groot.embodiment_tag,
                "groot_source_revision": groot.groot_source_revision,
                "checkpoint_revision": groot.checkpoint_revision,
                "open_loop_horizon": groot.open_loop_horizon,
            },
            "policy_identity_receipt": {
                "status": "runtime_measurement_required",
                "relative_path": (
                    "adp009d_groot_worker_identity.groot_n17_droid.json"
                ),
            },
        }
    )
    readiness = load_scene_policy_readiness(
        ADP009D_POLICY_READINESS_PATH,
        scenario_suite_path=ADP009D_SCENARIO_SUITE_PATH,
    )
    spec["candidate_rights_binding"] = build_candidate_policy_rights(
        readiness,
        candidate_id="groot_n17_droid",
        policy_spec=spec["policy_spec"],
        runtime_robot_id=scene["robot"]["robot_id"],
        scene_plan_digest=scene["plan_digest"],
    )
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )
    return spec


@pytest.mark.parametrize("host", ["0.0.0.0", "policy.example", "10.0.0.7"])
def test_policy_execution_spec_refuses_non_loopback_endpoint(
    tmp_path: Path, host: str
) -> None:
    _, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    spec = _groot_policy_spec(scene, construction, controls)
    spec["policy_endpoint"]["host"] = host
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )

    with pytest.raises(ValueError, match="native_task_policy_endpoint_invalid"):
        validate_native_task_policy_execution_spec(spec)


def test_policy_bundle_requires_exact_qualified_construction_and_controls(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    policy_spec = _policy_spec(scene, construction, controls)
    receipt = build_native_task_arena_policy_bundle(
        job_dir=tmp_path / "policy-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=policy_spec,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="d" * 40,
        generated_at="fixed",
    )

    assert receipt["execution_mode"] == "policy"
    assert receipt["policy_candidate_id"] == "pi05_droid"
    assert receipt["candidate_policy_queried"] is False
    assert receipt["policy_execution_spec_digest"] == policy_spec[
        "execution_spec_digest"
    ]
    assert receipt["policy_execution_authority"] == (
        "qualified_controls_evaluation"
    )
    assert receipt["policy_rights_binding"] == policy_spec[
        "candidate_rights_binding"
    ]
    assert receipt["expected_output_filename"] == (
        "native_task_arena_policy_result.v1.json"
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {
            "provider_runtime/runtime_inputs/adp009d_scene_840920_policy_readiness.v1.json",
            "provider_runtime/runtime_inputs/third_scene_840920_task_a_scenario_suite.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_control_result.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_policy_execution_spec.v1.json",
        }.issubset(names)
        assert {
            "provider_runtime/adp009d_policy_provisioning.pi05_droid.sh",
            "provider_runtime/adp009d_policy_execution_spec.json",
            "provider_runtime/adp009d_openpi_checkpoint_inventory.json",
            "provider_runtime/adp009d_policy_server_worker.py",
            "provider_runtime/adp009d_checkpoint_fetch_worker.py",
            "provider_runtime/adp009d_provisioning_preflight.py",
            "provider_runtime/openpi_droid_policy_runtime.py",
            "provider_runtime/droid_policy_bridge.py",
        }.issubset(names)
        entrypoint = archive.read(
            "provider_runtime/run_adp_arena_provider_runtime.sh"
        ).decode()
        assert 'export RUNTIME_DIR OUT_DIR' in entrypoint
        assert entrypoint.index("policy_provisioning:started") < entrypoint.index(
            '"$RUNTIME_DIR/adp_arena_provider_runner.py"'
        )
        provisioning = archive.read(
            "provider_runtime/adp009d_policy_provisioning.pi05_droid.sh"
        ).decode()
        assert "BLUEPRINT_ADP009D_POLICY_PROVISIONED:pi05_droid" in provisioning
        assert "XLA_PYTHON_CLIENT_PREALLOCATE=\"false\"" in provisioning
        worker = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
        assert "840313" not in worker
        assert "840796" not in worker
        assert "refrigerator" not in worker
    loaded = load_verified_native_task_arena_policy_bundle(
        tmp_path
        / "policy-bundle/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="d" * 40,
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]

    receipt_path = (
        tmp_path
        / "policy-bundle/native_task_arena_provider_bundle_receipt.v1.json"
    )
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["container_image"] = "registry.invalid/image@sha256:" + "0" * 64
    tampered["input_digest"] = canonical_digest(
        {
            key: value
            for key, value in tampered.items()
            if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
        },
        digest_field="input_digest",
    )
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="native_task_policy_bundle_image_mismatch"):
        load_verified_native_task_arena_policy_bundle(
            receipt_path,
            expected_implementation_commit="d" * 40,
        )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    control_result = json.loads(controls.read_text())
    control_result["controls_qualified"] = False
    control_result["result_digest"] = canonical_digest(
        control_result, digest_field="result_digest"
    )
    controls.write_text(json.dumps(control_result))
    spec = _policy_spec(scene, construction, controls)
    with pytest.raises(ValueError, match="native_task_policy_controls_not_qualified"):
        build_native_task_arena_policy_bundle(
            job_dir=tmp_path / "blocked-policy-bundle",
            packet_dir=packet,
            construction_result_path=construction,
            control_result_path=controls,
            policy_execution_spec=spec,
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="d" * 40,
        )


def test_policy_bundle_rejects_self_digested_forged_rights_projection(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    spec = _policy_spec(scene, construction, controls)
    spec["candidate_rights_binding"]["rights"]["provider_use_status"] = (
        "caller_asserted_but_not_authoritative"
    )
    spec["candidate_rights_binding"]["rights_receipt_digest"] = canonical_digest(
        spec["candidate_rights_binding"], digest_field="rights_receipt_digest"
    )
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )

    # The narrow syntax validator deliberately has no filesystem authority;
    # the bundle gate must compare it to the separately bound readiness bytes.
    validate_native_task_policy_execution_spec(spec)
    with pytest.raises(
        ValueError,
        match="candidate_policy_rights_authoritative_projection_mismatch",
    ):
        build_native_task_arena_policy_bundle(
            job_dir=tmp_path / "forged-policy-bundle",
            packet_dir=packet,
            construction_result_path=construction,
            control_result_path=controls,
            policy_execution_spec=spec,
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="d" * 40,
        )


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_policy_provisioning_shell_failures_retain_typed_media_gap(
    candidate_id: str,
) -> None:
    """Shell-side failures bypass the policy worker's normal result finalizer."""

    entrypoint = _entrypoint(
        expected_output_filename="native_task_arena_policy_result.v1.json",
        expected_result_schema="native_task_arena_policy_result.v1",
        runtime_source_packet_required=True,
        policy_provisioning_script_name=(
            f"adp009d_policy_provisioning.{candidate_id}.sh"
        ),
    )

    for reason in (
        "native_task_runtime_source_provisioning_failed",
        "native_task_arena_policy_provisioning_failed",
    ):
        blocker = f'"blockers": ["{reason}"]'
        start = entrypoint.index(blocker)
        result_writer = entrypoint[
            start : entrypoint.index("provider_zero_required_after_return", start)
        ]
        assert '"status": "unavailable_before_first_observation"' in result_writer
        assert '"type": "before_first_observation"' in result_writer
        assert f'"reason": "{reason}"' in result_writer


def test_policy_execution_spec_can_be_sealed_without_calling_an_endpoint(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    request = _policy_spec(scene, construction, controls)
    request.pop("execution_spec_digest")
    output = tmp_path / "policy-execution-spec.json"

    result = materialize_native_task_policy_execution_spec(
        request=request, output_path=output
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["candidate_id"] == "pi05_droid"
    assert result["execution_spec_digest"] == canonical_digest(
        result, digest_field="execution_spec_digest"
    )
    with pytest.raises(ValueError, match="output_exists"):
        materialize_native_task_policy_execution_spec(
            request=request, output_path=output
        )


def test_policy_bundle_cli_forwards_explicit_authority_paths(
    tmp_path: Path, monkeypatch
) -> None:
    spec = tmp_path / "policy-spec.json"
    spec.write_text("{}", encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        policy_bundle_module,
        "build_native_task_arena_policy_bundle",
        lambda **kwargs: observed.update(kwargs) or {"status": "ready"},
    )

    exit_code = policy_bundle_module.main(
        [
            "--job-dir",
            str(tmp_path / "job"),
            "--packet-dir",
            str(tmp_path / "packet"),
            "--construction-result",
            str(tmp_path / "construction.json"),
            "--control-result",
            str(tmp_path / "controls.json"),
            "--runtime-source-packet-receipt",
            str(tmp_path / "runtime.json"),
            "--implementation-commit",
            "a" * 40,
            "--policy-execution-spec",
            str(spec),
            "--scene-policy-readiness-path",
            str(tmp_path / "readiness.json"),
            "--scenario-suite-path",
            str(tmp_path / "suite.json"),
        ]
    )

    assert exit_code == 0
    assert observed["scene_policy_readiness_path"] == str(
        tmp_path / "readiness.json"
    )
    assert observed["scenario_suite_path"] == str(tmp_path / "suite.json")


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_provider_free_spec_builder_derives_each_frozen_candidate(
    tmp_path: Path, candidate_id: str
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    output = tmp_path / f"{candidate_id}.execution-spec.json"

    result = build_native_task_policy_execution_spec(
        candidate_id=candidate_id,
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=output,
    )

    assert result["candidate_id"] == candidate_id
    assert result["task_id"] == scene["task_id"]
    assert result["prompt"] == scene["task_spec"]["prompt"]
    assert result["max_policy_queries"] == 56
    assert result["policy_may_grade_itself"] is False
    assert result["execution_spec_digest"] == canonical_digest(
        result, digest_field="execution_spec_digest"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == result
    if candidate_id == "pi05_droid":
        assert result["policy_spec"]["checkpoint_uri"].endswith(
            "/polaris/pi05_droid_jointpos_polaris"
        )
    else:
        assert result["policy_identity_receipt"] == (
            policy_bundle_module.GROOT_RUNTIME_IDENTITY_DECLARATION
        )


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_provider_free_spec_cli_seals_each_frozen_candidate(
    tmp_path: Path, candidate_id: str
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    output = tmp_path / f"{candidate_id}.cli-execution-spec.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_native_task_arena_policy_execution_spec.py",
            "--candidate-id",
            candidate_id,
            "--scene-plan",
            str(packet / "native_task_arena_scene_plan.v1.json"),
            "--construction-result",
            str(construction),
            "--control-result",
            str(controls),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["candidate_id"] == candidate_id
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_id"] == (
        candidate_id
    )


def test_provider_free_spec_cli_fails_closed_before_controls_qualify(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    control_result = json.loads(controls.read_text(encoding="utf-8"))
    control_result["controls_qualified"] = False
    control_result["result_digest"] = canonical_digest(
        control_result, digest_field="result_digest"
    )
    controls.write_text(json.dumps(control_result), encoding="utf-8")
    output = tmp_path / "must-not-exist.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_native_task_arena_policy_execution_spec.py",
            "--candidate-id",
            "pi05_droid",
            "--scene-plan",
            str(packet / "native_task_arena_scene_plan.v1.json"),
            "--construction-result",
            str(construction),
            "--control-result",
            str(controls),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "native_task_policy_controls_not_qualified" in completed.stdout
    assert not output.exists()


def test_groot_execution_spec_refuses_predeclared_verified_runtime_identity(
    tmp_path: Path,
) -> None:
    """Checkpoint bytes do not exist on the worker when the spec is sealed."""

    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    request = _groot_policy_spec(scene, construction, controls)
    request["policy_identity_receipt"] = {
        "status": "verified",
        "checkpoint_files_sha256": "4" * 64,
        "environment_lock_sha256": "5" * 64,
    }
    request.pop("execution_spec_digest")

    with pytest.raises(
        ValueError, match="native_task_policy_spec_or_identity_invalid"
    ):
        materialize_native_task_policy_execution_spec(
            request=request,
            output_path=tmp_path / "dishonest-groot-execution-spec.json",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        (
            "prompt",
            "Open a different appliance.",
            "native_task_policy_prompt_task_spec_mismatch",
        ),
        (
            "max_policy_queries",
            55,
            "native_task_policy_shared_query_budget_mismatch",
        ),
        (
            "execution_authority",
            None,
            "native_task_policy_execution_authority_invalid",
        ),
    ),
)
def test_policy_bundle_binds_prompt_and_shared_query_budget_to_task(
    tmp_path: Path,
    field: str,
    value,
    error: str,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    spec = _policy_spec(scene, construction, controls)
    spec[field] = value
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )

    with pytest.raises(ValueError, match=error):
        build_native_task_arena_policy_bundle(
            job_dir=tmp_path / f"blocked-{field}",
            packet_dir=packet,
            construction_result_path=construction,
            control_result_path=controls,
            policy_execution_spec=spec,
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="d" * 40,
        )


def test_canonical_allocator_routes_native_task_policy_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    execution_spec = _policy_spec(scene, construction, controls)
    execution_path = tmp_path / "policy-execution.json"
    execution_path.write_text(json.dumps(execution_spec))
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "d" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_native_task_arena_policy_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        "native-task-arena-policy",
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "policy-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "policy-adapter.json"),
        "--pod-name",
        "native-task-policy",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(_runtime_source_packet(tmp_path)),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-control-result",
        str(controls),
        "--native-task-arena-policy-execution-spec",
        str(execution_path),
        "--adp-job-dir",
        str(tmp_path / "policy-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
        "--adp-allowed-active-vast-instance-id",
        "47373597",
    ]

    assert allocator.main(args) == 0
    assert observed["prepared_bundle"]["execution_mode"] == "policy"
    assert observed["prepared_bundle"]["policy_candidate_id"] == "pi05_droid"
    assert observed["allowed_active_instance_ids"] == [47373597]
    admission = json.loads((tmp_path / "policy-admission.json").read_text())
    assert admission["candidate_policy_queried"] is True


def test_allocator_refuses_prebuilt_bundle_with_a_different_external_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    runtime_sources = _runtime_source_packet(tmp_path)
    bundled_spec = _policy_spec(scene, construction, controls)
    bundle_job = tmp_path / "bound-policy-bundle"
    build_native_task_arena_policy_bundle(
        job_dir=bundle_job,
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=bundled_spec,
        runtime_source_packet_receipt=runtime_sources,
        implementation_commit="d" * 40,
        generated_at="fixed",
    )
    external_spec = json.loads(json.dumps(bundled_spec))
    external_spec["max_policy_queries"] -= 1
    external_spec["execution_spec_digest"] = canonical_digest(
        external_spec, digest_field="execution_spec_digest"
    )
    external_path = tmp_path / "different-policy-execution.json"
    external_path.write_text(json.dumps(external_spec), encoding="utf-8")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "d" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_native_task_arena_policy_vast",
        lambda **_kwargs: {"status": "dry_run_ready"},
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        "native-task-arena-policy",
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "mismatch-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "mismatch-adapter.json"),
        "--pod-name",
        "native-task-policy-mismatch",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(runtime_sources),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-control-result",
        str(controls),
        "--native-task-arena-policy-execution-spec",
        str(external_path),
        "--native-task-arena-bundle-receipt",
        str(bundle_job / "native_task_arena_provider_bundle_receipt.v1.json"),
        "--adp-job-dir",
        str(tmp_path / "policy-mismatch-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
    ]

    assert allocator.main(args) == 0
    admission = json.loads(
        (tmp_path / "mismatch-admission.json").read_text(encoding="utf-8")
    )
    assert "native_task_arena_policy_execution_binding_mismatch" in admission[
        "blockers"
    ]


def test_canonical_allocator_binds_groot_gated_backbone_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    execution_spec = _groot_policy_spec(scene, construction, controls)
    execution_path = tmp_path / "groot-policy-execution.json"
    execution_path.write_text(json.dumps(execution_spec))
    observed: dict = {}
    access = {"receipt_digest": "sha256:" + "9" * 64, "blockers": []}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "d" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(allocator, "normalize_model_access_env", lambda: None)
    monkeypatch.setattr(
        allocator,
        "model_access_secret_status",
        lambda: {"huggingface": {"auth_ready": True}},
    )
    monkeypatch.setattr(
        allocator, "probe_gated_backbone_access", lambda: dict(access)
    )
    monkeypatch.setattr(
        allocator,
        "run_native_task_arena_policy_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        "native-task-arena-policy",
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "groot-policy-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "groot-policy-adapter.json"),
        "--pod-name",
        "native-task-groot-policy",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(_runtime_source_packet(tmp_path)),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-control-result",
        str(controls),
        "--native-task-arena-policy-execution-spec",
        str(execution_path),
        "--adp-job-dir",
        str(tmp_path / "groot-policy-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
        "--adp009d-authorize-gated-backbone",
    ]

    assert allocator.main(args) == 0
    assert observed["prepared_bundle"]["policy_candidate_id"] == "groot_n17_droid"
    assert observed["authorize_gated_backbone"] is True
    admission = json.loads(
        (tmp_path / "groot-policy-admission.json").read_text()
    )
    assert admission["gated_backbone_access"] == access
    assert admission["allocation_binding"]["gated_backbone_authorized"] is True
    assert admission["allocation_binding"][
        "gated_backbone_access_receipt_digest"
    ] == access["receipt_digest"]
    assert admission["allocation_binding"]["execution_mode"] == "policy"


def test_qualified_construction_builds_one_complete_controls_bundle(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    receipt = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "controls-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="c" * 40,
        generated_at="fixed",
    )

    assert receipt["execution_mode"] == "controls"
    assert receipt["expected_output_filename"] == (
        "native_task_arena_control_result.v1.json"
    )
    assert receipt["policy_candidate_id"] is None
    assert receipt["candidate_policy_queried"] is False
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {
            "provider_runtime/runtime_inputs/adp_task_control_plan.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json",
        }.issubset(names)
        assert {
            f"provider_runtime/blueprint_pipeline/{name}"
            for name in CONTROLS_RUNTIME_MODULE_NAMES
        }.issubset(names)
        worker = archive.read(
            "provider_runtime/adp_arena_provider_runner.py"
        ).decode("utf-8")
        assert "840313" not in worker
        assert "840796" not in worker
        assert "refrigerator" not in worker
    loaded = load_verified_native_task_arena_controls_bundle(
        tmp_path
        / "controls-bundle/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="c" * 40,
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]


def test_controls_bundle_seals_bounded_orientation_reference_into_plan(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    reference = [
        1.8153258562,
        0.8945093155,
        -1.6013997793,
        -2.5417878628,
        -2.8766772747,
        2.3462493420,
        -0.8545385003,
    ]

    receipt = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "controls-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="c" * 40,
        generated_at="fixed",
        bounded_orientation_reference_joint_positions_rad=reference,
    )

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        plan = json.loads(
            archive.read(
                "provider_runtime/runtime_inputs/adp_task_control_plan.v1.json"
            )
        )
    assert plan[
        "bounded_orientation_reference_joint_positions_rad"
    ] == pytest.approx(reference)
    assert plan["plan_digest"] == canonical_digest(
        plan, digest_field="plan_digest"
    )


def test_controls_bundle_refuses_invalid_bounded_orientation_reference(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)

    with pytest.raises(
        ValueError,
        match="native_task_controls_bounded_orientation_reference_invalid",
    ):
        build_native_task_arena_controls_bundle(
            job_dir=tmp_path / "controls-bundle",
            packet_dir=packet,
            construction_result_path=construction,
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            implementation_commit="c" * 40,
            generated_at="fixed",
            bounded_orientation_reference_joint_positions_rad=[0.0] * 6,
        )


def test_downstream_diagnostic_is_default_off_and_requires_immutable_bundle_opt_in(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    common = {
        "packet_dir": packet,
        "construction_result_path": construction,
        "runtime_source_packet_receipt": _runtime_source_packet(tmp_path),
        "implementation_commit": "c" * 40,
        "generated_at": "fixed",
    }
    ordinary = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "ordinary-controls", **common
    )
    diagnostic = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "diagnostic-controls",
        enable_synthetic_post_phase5_downstream_diagnostic=True,
        **common,
    )

    request_member = (
        "provider_runtime/runtime_inputs/"
        + controls_bundle_module.DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME
    )
    with zipfile.ZipFile(ordinary["bundle_path"]) as archive:
        assert request_member not in archive.namelist()
    with zipfile.ZipFile(diagnostic["bundle_path"]) as archive:
        assert request_member in archive.namelist()
        request = json.loads(archive.read(request_member))
    assert request["enabled"] is True
    assert request["development_only"] is True
    assert request["qualification_effect"] == "none"
    assert request["request_digest"] == canonical_digest(
        request, digest_field="request_digest"
    )
    loaded = load_verified_native_task_arena_controls_bundle(
        tmp_path
        / "diagnostic-controls/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="c" * 40,
    )
    assert loaded["bundle_sha256"] == diagnostic["bundle_sha256"]


def test_bundle_is_deterministic_for_one_sealed_packet(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    kwargs = {
        "packet_dir": packet,
        "worker_source": worker,
        "runtime_module_sources": [],
        "implementation_commit": "b" * 40,
        "generated_at": "fixed",
    }
    first = build_native_task_arena_bundle(job_dir=tmp_path / "first", **kwargs)
    second = build_native_task_arena_bundle(job_dir=tmp_path / "second", **kwargs)

    assert first["bundle_sha256"] == second["bundle_sha256"]


def test_packet_asset_tamper_fails_before_bundle_creation(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    (packet / "assets/task_object.usd").write_text("tampered\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="c" * 40,
        )

    assert any(
        error.startswith("native_task_arena_bundle_packet_asset_identity_mismatch")
        for error in excinfo.value.errors
    )


def test_policy_mode_requires_an_exact_candidate_binding(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="d" * 40,
            execution_mode="policy",
        )

    assert excinfo.value.errors == (
        "native_task_arena_bundle_policy_binding_invalid",
    )


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_construction_bundle_has_one_scene_neutral_import_closure(
    tmp_path: Path, scene_id: str
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / f"construction-{scene_id}",
        packet_dir=_packet(tmp_path, scene_id=scene_id),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="e" * 40,
        generated_at="fixed",
    )
    assert receipt["container_image"] == NATIVE_TASK_ARENA_IMAGE
    extracted = tmp_path / f"extracted-{scene_id}"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        archive.extractall(extracted)
    package = extracted / "provider_runtime/blueprint_pipeline"
    expected = {
        f"provider_runtime/blueprint_pipeline/{name}"
        for name in CONSTRUCTION_RUNTIME_MODULE_NAMES
    }
    assert expected.issubset(names)
    assert (
        "provider_runtime/native_task_runtime_sources/native_task_runtime_sources.zip"
        not in names
    )
    bundle_root = Path(receipt["bundle_path"]).parent / "provider_runtime"
    assert not (bundle_root / "native_task_packet").exists()
    assert not (bundle_root / "native_task_runtime_sources").exists()
    assert receipt["runtime_source_packet"]["redistribution_permitted"] is True
    assert receipt["runtime_source_packet"]["transport"] == (
        "content_addressed_external_layer.v1"
    )
    assert receipt["runtime_source_packet"]["embedded_in_provider_bundle"] is False
    assert "provider_runtime/blueprint_pipeline/native_task_arena_scene_plan.py" not in names
    assert "provider_runtime/blueprint_pipeline/adp009d_approach_capture.py" not in names
    assert not any(
        name.startswith("provider_runtime/blueprint_pipeline/adp009d")
        for name in names
    )
    modules = [Path(name).stem for name in CONSTRUCTION_RUNTIME_MODULE_NAMES]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib,sys;"
                f"sys.path.insert(0,{str(package.parent)!r});"
                f"[importlib.import_module('blueprint_pipeline.'+name) for name in {modules!r}]"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    worker = (extracted / "provider_runtime/adp_arena_provider_runner.py").read_text()
    assert "840313" not in worker
    assert "840796" not in worker
    assert "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena" in worker


def test_construction_bundle_binds_customer_supplied_digest_pinned_image(
    tmp_path: Path,
) -> None:
    image = "registry.example/robot-team/runtime@sha256:" + "4" * 64
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "custom-runtime",
        packet_dir=_packet(tmp_path, scene_id="customer-scene"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="e" * 40,
        container_image=image,
        generated_at="fixed",
    )

    assert receipt["container_image"] == image
    loaded = load_verified_native_task_arena_construction_bundle(
        tmp_path
        / "custom-runtime/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="e" * 40,
        expected_packet_receipt_digest=receipt["packet_receipt_digest"],
        expected_runtime_source_packet_digest=receipt["runtime_source_packet"][
            "receipt_digest"
        ],
    )
    assert loaded["container_image"] == image


def test_runtime_preflight_bundle_reuses_exact_packet_and_stops_before_motion(
    tmp_path: Path,
) -> None:
    receipt = build_native_task_arena_runtime_preflight_bundle(
        job_dir=tmp_path / "runtime-preflight",
        packet_dir=_packet(tmp_path, scene_id="840920"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="e" * 40,
        generated_at="fixed",
    )
    assert receipt["execution_mode"] == "runtime_preflight"
    assert receipt["expected_output_filename"] == RUNTIME_PREFLIGHT_RESULT_FILENAME
    assert receipt["container_image"] == NATIVE_TASK_ARENA_IMAGE
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        worker = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
        extracted = tmp_path / "runtime-preflight-extracted"
        archive.extractall(extracted)
    assert "native_task_arena_runtime_preflight_worker.py" not in names
    assert "task_motion_executed" in worker
    assert "task_motion_executed\"] = True" not in worker
    assert worker.index("(output_root / RESULT_FILENAME).write_text") < worker.index(
        "simulation_app.close()"
    )
    assert (
        "provider_runtime/blueprint_pipeline/native_task_torch_runtime_lock.py"
        in names
    )
    assert "provider_runtime/blueprint_pipeline/rigid_frame_transforms.py" in names
    loaded = load_verified_native_task_arena_runtime_preflight_bundle(
        tmp_path
        / "runtime-preflight/native_task_arena_provider_bundle_receipt.v1.json",
        expected_implementation_commit="e" * 40,
        expected_packet_receipt_digest=receipt["packet_receipt_digest"],
        expected_runtime_source_packet_digest=receipt["runtime_source_packet"][
            "receipt_digest"
        ],
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "runtime-preflight-static",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["blockers"] == []
    assert preflight["status"] == "passed"
    entrypoint = (
        extracted / "provider_runtime/run_adp_arena_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    assert 'if [ ! -f "$OUT_DIR/native_task_arena_runtime_preflight.v1.json" ]' in entrypoint
    assert '"worker_exit_code": runner_rc' in entrypoint
    from blueprint_pipeline.native_task_arena_construction_worker import (
        _load_and_verify_manifest,
    )

    verified_manifest = _load_and_verify_manifest(
        extracted / "provider_runtime",
        expected_execution_mode="runtime_preflight",
    )
    assert verified_manifest["execution_mode"] == "runtime_preflight"


def test_runtime_preflight_classifies_plain_volume_without_spg(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    asset = packet / "assets" / "scene_appearance.usdz"
    asset.parent.mkdir(parents=True)
    with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "gauss.usda",
            'def Volume "gauss" {\n'
            "  custom bool omni:nurec:isNuRecVolume = 1\n"
            '  def OmniNuRecFieldAsset "density_field" {}\n'
            "}\n",
        )
    plan = {
        "objects": [
            {
                "semantic_role": "scene_appearance",
                "usd_path": "assets/scene_appearance.usdz",
            }
        ]
    }
    result = _plain_nurec_volume_contract(packet, plan)
    assert result["passed"] is True
    assert result["render_path"] == "plain_nurec_volume"
    assert result["spg_graph_execution_required"] is False
    assert result["renderer_extension_activation_expected"] is True


def test_runtime_preflight_refuses_spg_asset_on_plain_volume_path(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    asset = packet / "assets" / "scene_appearance.usdz"
    asset.parent.mkdir(parents=True)
    with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "gauss.usda",
            'def Volume "gauss" {\n'
            "  custom bool omni:nurec:isNuRecVolume = 1\n"
            '  custom asset info:spg:sourceAsset = @graph.spg@\n'
            '  def OmniNuRecFieldAsset "density_field" {}\n'
            "}\n",
        )
    plan = {
        "objects": [
            {
                "semantic_role": "scene_appearance",
                "usd_path": "assets/scene_appearance.usdz",
            }
        ]
    }
    result = _plain_nurec_volume_contract(packet, plan)
    assert result["passed"] is False
    assert result["spg_graph_execution_required"] is True
    assert result["blockers"] == [
        "native_task_arena_spg_asset_requires_separate_launch_path"
    ]


def test_runtime_preflight_accepts_sealed_particlefield_alignment(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    asset = packet / "assets" / "scene_appearance.usdc"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"sealed binary ParticleField bytes")
    plan = {
        "objects": [
            {
                "semantic_role": "scene_appearance",
                "usd_path": "assets/scene_appearance.usdc",
            }
        ],
        "appearance_frame_alignment": {
            "status": "aligned",
            "representation": "particlefield_3d_gaussian_splat",
            "measurement_authority": "particlefield_position_quantiles",
        },
    }

    result = _plain_nurec_volume_contract(packet, plan)

    assert result["passed"] is True
    assert result["render_path"] == "particlefield_3d_gaussian_splat"
    assert result["particlefield_alignment_receipt_present"] is True
    assert result["spg_graph_execution_required"] is False


def test_runtime_preflight_transport_is_ada_only_and_requires_no_task_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    observed: dict = {}
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    result = run_native_task_arena_runtime_preflight_vast(
        job_dir=tmp_path / "preflight-run",
        prepared_bundle={
            "schema_version": "native_task_arena_provider_bundle.v1",
            "status": "ready",
            "execution_mode": "runtime_preflight",
            "policy_candidate_id": None,
            "candidate_policy_queried": False,
            "expected_output_filename": RUNTIME_PREFLIGHT_RESULT_FILENAME,
            "container_image": NATIVE_TASK_ARENA_IMAGE,
        },
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert result["status"] == "dry_run_ready"
    assert observed["preferred_gpu_keywords"] == (
        "L40S",
        "RTX 6000 Ada",
        "RTX 4090",
    )
    assert observed["min_gpu_ram_mb"] == 24_000
    assert "RTX A6000" not in observed["preferred_gpu_keywords"]
    assert observed["gpu_selection_policy"] == {
        "policy_id": "native_task_arena_runtime_preflight_ada_only",
        "allowed_gpu_keywords": ("L40S", "6000ADA", "RTX 4090"),
        "reason": "NuRec runtime preflight is qualified only on Ada GPUs",
        "minimum_cuda_max_good": 12.8,
    }
    assert observed["candidate_policy_query_expected"] is False


def test_runtime_preflight_ada_policy_accepts_provider_slug_and_rejects_a6000() -> None:
    policy = {
        "policy_id": "native_task_arena_runtime_preflight_ada_only",
        "allowed_gpu_keywords": ("L40S", "6000ADA", "RTX 4090"),
        "minimum_cuda_max_good": 12.8,
    }
    selected = _select_offer(
        [
            {
                "ask_contract_id": 1,
                "gpu_name": "RTX A6000",
                "gpu_ram_mb": 49_140,
                "dph_total": 0.45,
                "driver_version": "580.159.03",
                "cuda_max_good": 13.0,
                "machine_id": 1,
            },
            {
                "ask_contract_id": 2,
                "gpu_name": "RTX 4090",
                "gpu_ram_mb": 24_576,
                "dph_total": 0.62,
                "driver_version": "580.119.02",
                "cuda_max_good": 13.0,
                "machine_id": 2,
            },
        ],
        max_hourly_rate=0.80,
        min_gpu_ram_mb=24_000,
        minimum_driver_version="580.65.06",
        gpu_selection_policy=policy,
    )
    assert selected is not None
    assert selected["ask_contract_id"] == 2
    assert selected["gpu_name"] == "RTX 4090"

def test_construction_bundle_passes_native_vast_static_preflight(tmp_path: Path) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="f" * 40,
        generated_at="fixed",
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    assert (
        _resolve_probe_image(
            public_image="public",
            isaac_image="isaac",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="native_task_arena",
        )
        == "isaac"
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=True,
            provider_bundle_kind="native_task_arena",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        forward_hf_token=False,
    )
    assert env["ACCEPT_EULA"] == "Y"
    assert "run_adp_arena_provider_runtime.sh" in _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="native_task_arena",
    )

    dry_run = run_native_task_arena_vast(
        job_dir=tmp_path / "dry-run",
        prepared_bundle=receipt,
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0


def _native_bundle_preflight(tmp_path: Path, receipt: dict, *, name: str) -> dict:
    return _blueprint_bundle_preflight(
        job_dir=tmp_path / f"{name}-preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )


def _rewrite_zip_json_member(
    bundle_path: str | Path, member_name: str, payload: dict
) -> None:
    path = Path(bundle_path)
    with zipfile.ZipFile(path) as archive:
        rows = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for info, value in rows:
            if info.filename == member_name:
                value = json.dumps(payload, indent=2).encode("utf-8")
            archive.writestr(info, value)


def test_native_preflight_reads_readiness_from_immutable_bundle_member(
    tmp_path: Path,
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="f" * 40,
        generated_at="fixed",
    )
    sidecar = (
        Path(receipt["bundle_path"]).parent
        / "provider_runtime/adp_arena_provider_manifest.json"
    )
    sidecar.chmod(0)
    try:
        preflight = _native_bundle_preflight(
            tmp_path, receipt, name="immutable-readiness"
        )
    finally:
        sidecar.chmod(0o600)

    member = "provider_runtime/adp_arena_provider_manifest.json"
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    assert preflight["provider_bundle_readiness_source"] == (
        "immutable_bundle_member"
    )
    assert preflight["provider_bundle_readiness_member"] == member
    assert preflight["provider_bundle_readiness_path"] == (
        f"{receipt['bundle_path']}!/{member}"
    )
    assert preflight["provider_bundle_readiness_present"] is True
    assert preflight["provider_bundle_local_ready_for_remote_staging"] is True


@pytest.mark.parametrize("invalid_field", ["status", "blockers", "input_digest"])
def test_native_preflight_rejects_invalid_immutable_readiness(
    tmp_path: Path, invalid_field: str
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / f"bundle-{invalid_field}",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="f" * 40,
        generated_at="fixed",
    )
    member = "provider_runtime/adp_arena_provider_manifest.json"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = json.loads(archive.read(member))
    if invalid_field == "status":
        manifest["status"] = "blocked"
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
    elif invalid_field == "blockers":
        manifest["blockers"] = ["fixture_blocker"]
        manifest["input_digest"] = canonical_digest(
            manifest, digest_field="input_digest"
        )
    else:
        manifest["input_digest"] = "sha256:" + "0" * 64
    _rewrite_zip_json_member(receipt["bundle_path"], member, manifest)

    preflight = _native_bundle_preflight(
        tmp_path, receipt, name=f"invalid-readiness-{invalid_field}"
    )

    assert preflight["status"] == "blocked"
    assert "native_task_arena_provider_manifest_invalid" in preflight["blockers"]


def _assert_bundle_has_isolated_import_closure(
    tmp_path: Path, receipt: dict, *, name: str
) -> set[str]:
    extracted = tmp_path / f"{name}-extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        archive.extractall(extracted)
    package = extracted / "provider_runtime/blueprint_pipeline"
    modules = sorted(
        path.stem for path in package.glob("*.py") if path.name != "__init__.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib,sys;"
                f"sys.path.insert(0,{str(package.parent)!r});"
                f"[importlib.import_module('blueprint_pipeline.'+name) for name in {modules!r}]"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return names


def test_real_controls_bundle_passes_preflight_and_imports_cleanly(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    receipt = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "controls-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="7" * 40,
        generated_at="fixed",
    )

    preflight = _native_bundle_preflight(tmp_path, receipt, name="controls")
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    _assert_bundle_has_isolated_import_closure(tmp_path, receipt, name="controls")


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_real_policy_bundles_pass_preflight_and_import_cleanly(
    tmp_path: Path, candidate_id: str
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    spec = build_native_task_policy_execution_spec(
        candidate_id=candidate_id,
        scene_plan_path=packet / "native_task_arena_scene_plan.v1.json",
        construction_result_path=construction,
        control_result_path=controls,
        output_path=tmp_path / f"{candidate_id}.execution-spec.json",
    )
    receipt = build_native_task_arena_policy_bundle(
        job_dir=tmp_path / f"policy-{candidate_id}",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=spec,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="6" * 40,
        generated_at="fixed",
    )

    preflight = _native_bundle_preflight(
        tmp_path, receipt, name=f"policy-{candidate_id}"
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    names = _assert_bundle_has_isolated_import_closure(
        tmp_path, receipt, name=f"policy-{candidate_id}"
    )
    assert "provider_runtime/blueprint_pipeline/policy_ranking_thesis.py" not in names


@pytest.mark.parametrize(
    "execution_mode,missing_module",
    [
        ("construction_canary", "articulated_control_planner.py"),
        ("controls", "adp009d_contact_envelope.py"),
        ("policy", "adp009d_droid_action_execution.py"),
    ],
)
def test_preflight_rejects_one_missing_runtime_module_per_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
    missing_module: str,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    runtime_source_packet = _runtime_source_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    if execution_mode == "construction_canary":
        monkeypatch.setattr(
            construction_bundle_module,
            "CONSTRUCTION_RUNTIME_MODULE_NAMES",
            tuple(
                name
                for name in construction_bundle_module.CONSTRUCTION_RUNTIME_MODULE_NAMES
                if name != missing_module
            ),
        )
        receipt = build_native_task_arena_construction_bundle(
            job_dir=tmp_path / execution_mode,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet,
            implementation_commit="5" * 40,
            generated_at="fixed",
        )
    elif execution_mode == "controls":
        monkeypatch.setattr(
            controls_bundle_module,
            "CONTROLS_RUNTIME_MODULE_NAMES",
            tuple(
                name
                for name in controls_bundle_module.CONTROLS_RUNTIME_MODULE_NAMES
                if name != missing_module
            ),
        )
        receipt = build_native_task_arena_controls_bundle(
            job_dir=tmp_path / execution_mode,
            packet_dir=packet,
            construction_result_path=construction,
            runtime_source_packet_receipt=runtime_source_packet,
            implementation_commit="5" * 40,
            generated_at="fixed",
        )
    else:
        monkeypatch.setattr(
            policy_bundle_module,
            "POLICY_EXTRA_RUNTIME_MODULE_NAMES",
            tuple(
                name
                for name in policy_bundle_module.POLICY_EXTRA_RUNTIME_MODULE_NAMES
                if name != missing_module
            ),
        )
        receipt = build_native_task_arena_policy_bundle(
            job_dir=tmp_path / execution_mode,
            packet_dir=packet,
            construction_result_path=construction,
            control_result_path=controls,
            policy_execution_spec=_policy_spec(scene, construction, controls),
            runtime_source_packet_receipt=runtime_source_packet,
            implementation_commit="5" * 40,
            generated_at="fixed",
        )

    preflight = _native_bundle_preflight(
        tmp_path, receipt, name=f"missing-{execution_mode}"
    )
    expected_member = f"provider_runtime/blueprint_pipeline/{missing_module}"
    assert preflight["status"] == "blocked"
    assert "provider_runtime_bundle_required_entries_missing" in preflight["blockers"]
    assert expected_member in preflight["missing_zip_entries"]


def test_preflight_rejects_unknown_native_result_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    monkeypatch.setattr(
        controls_bundle_module,
        "RESULT_FILENAME",
        "native_task_arena_garbage_result.v1.json",
    )
    receipt = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "wrong-result",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="4" * 40,
        generated_at="fixed",
    )

    preflight = _native_bundle_preflight(tmp_path, receipt, name="wrong-result")
    assert preflight["status"] == "blocked"
    assert "native_task_arena_provider_manifest_invalid" in preflight["blockers"]
    assert (
        "provider_entrypoint_missing_runtime_result_crash_fallback"
        in preflight["blockers"]
    )


def test_preflight_rejects_cross_mode_runtime_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    monkeypatch.setattr(
        controls_bundle_module,
        "CONTROLS_RUNTIME_MODULE_NAMES",
        (
            *controls_bundle_module.CONTROLS_RUNTIME_MODULE_NAMES,
            "native_task_arena_scene_plan.py",
        ),
    )
    receipt = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "extra-module",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="3" * 40,
        generated_at="fixed",
    )

    preflight = _native_bundle_preflight(tmp_path, receipt, name="extra-module")
    assert preflight["status"] == "blocked"
    assert "native_task_arena_provider_manifest_invalid" in preflight["blockers"]


@pytest.mark.parametrize(
    "filename,schema_version",
    (
        (
            "native_task_arena_construction_result.v1.json",
            "native_task_arena_construction_result.v1",
        ),
        (
            "native_task_arena_control_result.v1.json",
            "native_task_arena_control_result.v1",
        ),
        (
            "native_task_arena_policy_result.v1.json",
            "native_task_arena_policy_result.v1",
        ),
    ),
)
def test_provider_output_recognizes_task_neutral_native_result(
    tmp_path: Path, filename: str, schema_version: str
) -> None:
    from blueprint_pipeline.wam_provider_output import (
        inspect_provider_runtime_output_zip,
    )

    output_zip = tmp_path / "native-task-output.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            f"runtime/{filename}",
            json.dumps(
                {
                    "schema_version": schema_version,
                    "status": "blocked",
                    "blockers": ["native_task_construction_dependency_preflight_failed"],
                    "candidate_policy_queried": False,
                }
            ),
        )

    inspection = inspect_provider_runtime_output_zip(output_zip)

    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "blocked"
    assert inspection["runtime_result_blockers"] == [
        "native_task_construction_dependency_preflight_failed"
    ]


def test_explicit_concurrent_authority_uses_a_scoped_launch_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    observed = []

    def fake_run(**kwargs):
        observed.append(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(module, "run_arena_native_control_vast", fake_run)
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "construction_canary",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_construction_result.v1.json",
        "container_image": "image@sha256:" + "a" * 64,
    }

    module.run_native_task_arena_vast(
        job_dir=tmp_path / "serialized",
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
    )
    module.run_native_task_arena_vast(
        job_dir=tmp_path / "concurrent",
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
        allowed_active_instance_ids=(47358598,),
    )

    assert observed[0]["vast_launch_lock_file"] is None
    assert observed[1]["allowed_active_instance_ids"] == (47358598,)
    assert observed[1]["vast_launch_lock_file"] == (
        (tmp_path / "concurrent").resolve()
        / "native_task_arena_paid_launch.lock"
    )


@pytest.mark.parametrize(
    ("runner", "execution_mode", "expected_output", "expected_prefix", "candidate"),
    (
        (
            run_native_task_arena_vast,
            "construction_canary",
            "native_task_arena_construction_result.v1.json",
            "blueprint-native-task-arena-",
            None,
        ),
        (
            run_native_task_arena_controls_vast,
            "controls",
            "native_task_arena_control_result.v1.json",
            "blueprint-native-task-controls-",
            None,
        ),
        (
            run_native_task_arena_policy_vast,
            "policy",
            "native_task_arena_policy_result.v1.json",
            "blueprint-native-task-policy-",
            "pi05_droid",
        ),
        (
            run_native_task_arena_policy_diagnostic_vast,
            "policy_diagnostic",
            "native_task_arena_policy_diagnostic_result.v1.json",
            "blueprint-native-task-policy-diagnostic-",
            "pi05_droid",
        ),
    ),
)
def test_each_native_task_arena_stage_requires_its_exact_watchdog_scope(
    runner,
    execution_mode,
    expected_output,
    expected_prefix,
    candidate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    observed: dict = {}
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": execution_mode,
        "policy_candidate_id": candidate,
        "candidate_policy_queried": False,
        "expected_output_filename": expected_output,
        "container_image": "image@sha256:" + "a" * 64,
    }

    runner(
        job_dir=tmp_path / execution_mode,
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert observed["require_independent_watchdog"] is True
    assert observed["instance_label_prefix"] == expected_prefix
    if execution_mode in {"construction_canary", "controls"}:
        assert observed["min_gpu_ram_mb"] == 24_000
        assert "RTX 4090" in observed["preferred_gpu_keywords"]
        assert "RTX A6000" not in observed["preferred_gpu_keywords"]
    else:
        assert observed["min_gpu_ram_mb"] == 46_000
        assert "RTX A6000" in observed["preferred_gpu_keywords"]


def test_policy_vast_adapter_marks_candidate_query_and_external_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    observed: dict = {}
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "policy",
        "policy_candidate_id": "pi05_droid",
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_policy_result.v1.json",
        "container_image": "image@sha256:" + "a" * 64,
    }

    run_native_task_arena_policy_vast(
        job_dir=tmp_path / "policy",
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
        allowed_active_instance_ids=(47373597,),
    )

    assert observed["candidate_policy_query_expected"] is True
    assert observed["allowed_active_instance_ids"] == (47373597,)
    assert observed["object_store_key_prefix"].endswith("/policy/pi05_droid")
    assert observed["vast_launch_lock_file"] is None


def test_controls_and_policy_share_the_canonical_provider_semaphore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module
    from blueprint_pipeline import vast_provider_adapter

    canonical_lock = tmp_path / "provider-locks" / "vast_paid_launch.lock"
    monkeypatch.setenv(
        vast_provider_adapter.VAST_LAUNCH_LOCK_FILE_ENV,
        str(canonical_lock),
    )
    observed: list[dict] = []
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready"},
    )
    controls = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "controls",
        "policy_candidate_id": None,
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_control_result.v1.json",
        "container_image": "image@sha256:" + "a" * 64,
    }
    policy = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "policy",
        "policy_candidate_id": "pi05_droid",
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_policy_result.v1.json",
        "container_image": "image@sha256:" + "a" * 64,
    }

    run_native_task_arena_controls_vast(
        job_dir=tmp_path / "controls",
        prepared_bundle=controls,
        paid_resource_admission_grant=None,
        execute=False,
        allowed_active_instance_ids=(48606888,),
    )
    run_native_task_arena_policy_vast(
        job_dir=tmp_path / "policy",
        prepared_bundle=policy,
        paid_resource_admission_grant=None,
        execute=False,
        allowed_active_instance_ids=(48606888, 48607367),
    )

    assert [row["vast_launch_lock_file"] for row in observed] == [None, None]
    assert [row["allowed_active_instance_ids"] for row in observed] == [
        (48606888,),
        (48606888, 48607367),
    ]
    expected_slots = [
        canonical_lock,
        canonical_lock.with_name("vast_paid_launch.slot1.lock"),
        canonical_lock.with_name("vast_paid_launch.slot2.lock"),
    ]
    assert vast_provider_adapter.vast_launch_lock_paths() == expected_slots


@pytest.mark.parametrize(
    ("runner", "execution_mode", "expected_output", "candidate"),
    (
        (
            run_native_task_arena_controls_vast,
            "controls",
            "native_task_arena_control_result.v1.json",
            None,
        ),
        (
            run_native_task_arena_policy_vast,
            "policy",
            "native_task_arena_policy_result.v1.json",
            "pi05_droid",
        ),
    ),
)
def test_shared_semaphore_links_remain_retry_zero_when_authority_is_consumed(
    runner,
    execution_mode: str,
    expected_output: str,
    candidate: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    monkeypatch.setattr(
        module,
        "validate_native_task_arena_paid_attempt_authority",
        lambda *_args, **_kwargs: {"authorization_digest": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        module,
        "consume_native_task_arena_authority_once",
        lambda _authority: {
            "status": "already_consumed",
            "blockers": ["native_task_arena_paid_attempt_authority_already_consumed"],
        },
    )
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **_kwargs: pytest.fail("consumed authority reached the provider seam"),
    )
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": execution_mode,
        "policy_candidate_id": candidate,
        "candidate_policy_queried": False,
        "expected_output_filename": expected_output,
        "container_image": "image@sha256:" + "a" * 64,
    }

    result = runner(
        job_dir=tmp_path / execution_mode,
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=True,
        allowed_active_instance_ids=(48606888,),
        paid_attempt_authority={"authorization_digest": "sha256:" + "a" * 64},
    )

    assert result["status"] == "blocked"
    assert result["retry_cap"] == 0
    assert result["provider_mutations_performed"] == 0


def test_groot_policy_vast_requires_and_forwards_gated_backbone_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_vast as module

    observed: dict = {}
    monkeypatch.setattr(
        module,
        "run_arena_native_control_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    prepared = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "policy",
        "policy_candidate_id": "groot_n17_droid",
        "candidate_policy_queried": False,
        "expected_output_filename": "native_task_arena_policy_result.v1.json",
        "container_image": "image@sha256:" + "a" * 64,
    }

    with pytest.raises(ValueError, match="gated_backbone_authority_missing"):
        run_native_task_arena_policy_vast(
            job_dir=tmp_path / "missing",
            prepared_bundle=prepared,
            paid_resource_admission_grant=None,
            execute=False,
        )

    run_native_task_arena_policy_vast(
        job_dir=tmp_path / "authorized",
        prepared_bundle=prepared,
        paid_resource_admission_grant=None,
        execute=False,
        authorize_gated_backbone=True,
    )

    assert observed["forward_hf_token"] is True


def test_bundle_rejects_an_unpinned_runtime_image(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="f" * 40,
            container_image="nvcr.io/nvidia/isaac-sim:latest",
        )
    assert excinfo.value.errors == (
        "native_task_arena_bundle_container_image_not_digest_pinned",
    )

    with pytest.raises(NativeTaskArenaBundleError, match="not_digest_pinned"):
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job-nonhex",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="f" * 40,
            container_image="registry.example/runtime@sha256:" + "z" * 64,
        )


def test_dry_run_bundle_receipt_reloads_exact_bytes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    receipt_path = tmp_path / "bundle/native_task_arena_provider_bundle_receipt.v1.json"
    loaded = load_verified_native_task_arena_construction_bundle(
        receipt_path,
        expected_implementation_commit="a" * 40,
        expected_packet_receipt_digest=receipt["packet_receipt_digest"],
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]

    Path(receipt["bundle_path"]).write_bytes(
        Path(receipt["bundle_path"]).read_bytes() + b"tamper"
    )
    with pytest.raises(ValueError, match="native_task_arena_bundle_bytes_identity_mismatch"):
        load_verified_native_task_arena_construction_bundle(
            receipt_path,
            expected_implementation_commit="a" * 40,
        )


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_routes_sealed_native_task_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    frozen_bundle = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "frozen-bundle",
        packet_dir=packet,
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_native_task_arena_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "native-task-arena",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(_runtime_source_packet(tmp_path)),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
    ]
    if execute:
        authority_path = tmp_path / "native-task-arena-authority.json"
        write_json(
            authority_path,
            {"authorization_digest": "sha256:" + "9" * 64},
        )
        monkeypatch.setattr(
            allocator,
            "validate_native_task_arena_paid_attempt_authority",
            lambda authority, **_kwargs: authority,
        )
        args.extend(
            [
                "--native-task-arena-bundle-receipt",
                str(
                    tmp_path
                    / "frozen-bundle/native_task_arena_provider_bundle_receipt.v1.json"
                ),
                "--native-task-arena-attempt-authority",
                str(authority_path),
                "--execute",
            ]
        )

    assert allocator.main(args) == 0
    assert observed["execute"] is execute
    assert isinstance(
        observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant
    ) is execute
    if execute:
        assert (
            observed["prepared_bundle"]["bundle_sha256"]
            == frozen_bundle["bundle_sha256"]
        )
        assert observed["paid_attempt_authority"]["authorization_digest"].startswith(
            "sha256:"
        )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["private_data_uploaded"] is True
    assert admission["raw_dataset_bytes_uploaded"] is False
    assert admission["retry_cap"] == 0
    assert admission["allocation_binding"]["packet_receipt_digest"].startswith(
        "sha256:"
    )
    assert admission["allocation_binding"][
        "runtime_source_packet_receipt_digest"
    ].startswith("sha256:")


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_routes_no_motion_runtime_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    packet = _packet(tmp_path, scene_id="840920")
    source_packet = _runtime_source_packet(tmp_path)
    frozen = build_native_task_arena_runtime_preflight_bundle(
        job_dir=tmp_path / "frozen-preflight-bundle",
        packet_dir=packet,
        runtime_source_packet_receipt=source_packet,
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_native_task_arena_runtime_preflight_vast",
        lambda **kwargs: observed.update(kwargs)
        or {"status": "completed" if kwargs["execute"] else "dry_run_ready"},
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        RUNTIME_PREFLIGHT_PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "preflight-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "preflight-adapter.json"),
        "--pod-name",
        "native-task-arena-runtime-preflight",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-runtime-source-packet",
        str(source_packet),
        "--adp-job-dir",
        str(tmp_path / "preflight-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
    ]
    if execute:
        args.extend(
            [
                "--native-task-arena-bundle-receipt",
                str(
                    tmp_path
                    / "frozen-preflight-bundle/native_task_arena_provider_bundle_receipt.v1.json"
                ),
                "--execute",
            ]
        )
    assert allocator.main(args) == 0
    assert observed["execute"] is execute
    assert isinstance(
        observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant
    ) is execute
    if execute:
        assert observed["prepared_bundle"]["bundle_sha256"] == frozen["bundle_sha256"]
    assert "paid_attempt_authority" not in observed
    admission = json.loads((tmp_path / "preflight-admission.json").read_text())
    assert admission["probe_kind"] == RUNTIME_PREFLIGHT_PROBE_KIND
    assert admission["retry_cap"] == 0
    assert admission["allocation_binding"]["execution_mode"] == "runtime_preflight"


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_routes_qualified_native_controls_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    source_packet = _runtime_source_packet(tmp_path)
    frozen_bundle = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "frozen-controls-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=source_packet,
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_native_task_arena_controls_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        CONTROLS_PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "controls-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "controls-adapter.json"),
        "--pod-name",
        "native-task-arena-controls",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-runtime-source-packet",
        str(source_packet),
        "--adp-job-dir",
        str(tmp_path / "controls-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
    ]
    if execute:
        authority_path = tmp_path / "native-task-arena-controls-authority.json"
        write_json(
            authority_path,
            {"authorization_digest": "sha256:" + "8" * 64},
        )
        monkeypatch.setattr(
            allocator,
            "validate_native_task_arena_paid_attempt_authority",
            lambda authority, **_kwargs: authority,
        )
        args.extend(
            [
                "--native-task-arena-bundle-receipt",
                str(
                    tmp_path
                    / "frozen-controls-bundle/native_task_arena_provider_bundle_receipt.v1.json"
                ),
                "--native-task-arena-attempt-authority",
                str(authority_path),
                "--execute",
            ]
        )

    assert allocator.main(args) == 0
    assert observed["execute"] is execute
    assert observed["prepared_bundle"]["execution_mode"] == "controls"
    if execute:
        assert (
            observed["prepared_bundle"]["bundle_sha256"]
            == frozen_bundle["bundle_sha256"]
        )
    admission = json.loads((tmp_path / "controls-admission.json").read_text())
    assert admission["probe_kind"] == CONTROLS_PROBE_KIND
    assert admission["allocation_binding"]["execution_mode"] == "controls"
    assert admission["candidate_policy_queried"] is False


def test_canonical_allocator_attaches_controls_without_new_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    source_packet = _runtime_source_packet(tmp_path)
    frozen_bundle = build_native_task_arena_controls_bundle(
        job_dir=tmp_path / "frozen-controls-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        runtime_source_packet_receipt=source_packet,
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    warm_session = {
        "schema_version": "native_task_arena_warm_session.v1",
        "session_digest": "sha256:" + "7" * 64,
        "instance_id": 123,
    }
    warm_session_path = tmp_path / "warm-session.json"
    write_json(warm_session_path, warm_session)
    warm_authority = {
        "schema_version": "native_task_arena_warm_attempt_authority.v1",
        "authorization_digest": "sha256:" + "8" * 64,
    }
    warm_authority_path = tmp_path / "warm-authority.json"
    write_json(warm_authority_path, warm_authority)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "validate_native_task_arena_warm_session",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        allocator,
        "validate_native_task_arena_warm_attempt_authority",
        lambda value, **_kwargs: value,
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed", "provider_allocations_performed": 0}

    monkeypatch.setattr(
        allocator, "run_native_task_arena_warm_controls_vast", fake_run
    )
    args = [
        "gpu-canary",
        "--probe-kind",
        CONTROLS_PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "controls-admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "controls-adapter.json"),
        "--pod-name",
        "native-task-arena-warm-controls",
        "--native-task-arena-packet",
        str(packet),
        "--native-task-arena-construction-result",
        str(construction),
        "--native-task-arena-runtime-source-packet",
        str(source_packet),
        "--native-task-arena-bundle-receipt",
        str(
            tmp_path
            / "frozen-controls-bundle/native_task_arena_provider_bundle_receipt.v1.json"
        ),
        "--native-task-arena-attempt-authority",
        str(warm_authority_path),
        "--native-task-arena-warm-session",
        str(warm_session_path),
        "--adp-allowed-active-vast-instance-id",
        "123",
        "--adp-job-dir",
        str(tmp_path / "controls-job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.2",
        "--adp-hard-ttl-seconds",
        "5400",
        "--execute",
    ]

    assert allocator.main(args) == 0
    assert observed["execute"] is True
    assert observed["prepared_bundle"]["bundle_sha256"] == frozen_bundle[
        "bundle_sha256"
    ]
    assert observed["warm_session"] == warm_session
    assert observed["warm_attempt_authority"] == warm_authority
    assert "machine_avoidlist_path" not in observed
    admission = json.loads((tmp_path / "controls-admission.json").read_text())
    assert admission["allocation_binding"]["execution_transport"] == (
        "retained_warm_instance"
    )
    assert admission["allocation_binding"]["warm_session_digest"] == (
        warm_session["session_digest"]
    )


def _repack_scene(packet: Path, scene: dict) -> None:
    """Rewrite a packet's scene plan and rebind its receipt to the new bytes."""

    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    scene = dict(scene)
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    plan_path.write_text(json.dumps(scene, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["arena_scene_plan_digest"] = scene["plan_digest"]
    artifact = next(
        row for row in receipt["artifacts"] if row["role"] == "arena_scene_plan"
    )
    artifact["size_bytes"] = plan_path.stat().st_size
    artifact["sha256"] = _sha(plan_path)
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_bundle_refuses_a_packet_restating_stale_servo_limits(
    tmp_path: Path,
) -> None:
    """A hardlinked-forward packet must stop the launch, not run.

    r19 executed the predecessor's servo limits because the chain rebuilds the
    bundle from the deployed commit but carries the packet forward. PR #788
    caught that in the construction launch chain only; controls and policy
    consume the same packet, and the controls bundle recompiles its control
    plan from it. The refusal has to name the offending values.
    """

    from blueprint_pipeline.native_articulated_control_plan import (
        MAX_JOINT_DELTA_RAD,
        MAX_JOINT_SETPOINT_LEAD_RAD,
    )

    packet, scene = _articulated_packet(tmp_path)
    scene["task_spec"] = {
        **scene["task_spec"],
        "interaction_affordance": {
            "max_joint_delta_rad": 0.03,
            "max_joint_setpoint_lead_rad": 0.20,
        },
    }
    _repack_scene(packet, scene)
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "stale-job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="e" * 40,
            execution_mode="controls",
            expected_output_filename="native_task_arena_control_result.v1.json",
            runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
            generated_at="fixed",
        )

    blocker = next(
        item
        for item in excinfo.value.errors
        if item.startswith("native_task_arena_bundle_packet_servo_limits_stale")
    )
    assert "packet=0.03,0.2" in blocker
    assert f"deployed={MAX_JOINT_DELTA_RAD},{MAX_JOINT_SETPOINT_LEAD_RAD}" in blocker


def test_bundle_accepts_a_packet_restating_the_deployed_servo_limits(
    tmp_path: Path,
) -> None:
    """The gate must fire on staleness only, never on a fresh restatement."""

    from blueprint_pipeline.native_articulated_control_plan import (
        MAX_JOINT_DELTA_RAD,
        MAX_JOINT_SETPOINT_LEAD_RAD,
    )

    packet, scene = _articulated_packet(tmp_path)
    scene["task_spec"] = {
        **scene["task_spec"],
        "interaction_affordance": {
            "max_joint_delta_rad": MAX_JOINT_DELTA_RAD,
            "max_joint_setpoint_lead_rad": MAX_JOINT_SETPOINT_LEAD_RAD,
        },
    }
    _repack_scene(packet, scene)
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / "fresh-job",
        packet_dir=packet,
        worker_source=worker,
        runtime_module_sources=[],
        implementation_commit="e" * 40,
        execution_mode="controls",
        expected_output_filename="native_task_arena_control_result.v1.json",
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        generated_at="fixed",
    )

    assert receipt["status"] == "ready"
