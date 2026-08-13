from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.common import write_json
from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    DEFAULT_IMAGE as QUALIFIED_ADP_IMAGE,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import (
    NativeTaskArenaBundleError,
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
from blueprint_pipeline.native_task_arena_policy_bundle import (
    build_native_task_arena_policy_bundle,
    load_verified_native_task_arena_policy_bundle,
)
from blueprint_pipeline.native_task_arena_vast import (
    run_native_task_arena_policy_vast,
    run_native_task_arena_vast,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ISAACLAB_PACKAGE_NAMES,
    RUNTIME_DEPENDENCY_WHEELS,
    materialize_native_task_runtime_source_packet,
)
from blueprint_pipeline.droid_policy_bridge import OPENPI_SOURCE_REVISION
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
    _resolve_probe_image,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
                    f"{['warp-lang==1.12.0'] if name == 'isaaclab' else []!r})\n"
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
        "scenario": {"cell_id": "articulated-canonical", "seed": 17},
        "articulation": {"motion_geometry": motion},
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "articulated_open_close",
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
    assert frozen["execution_parameters"] == {
        "arrival_tolerance_m": 0.02,
        "stable_samples": 2,
        "maximum_steps_per_phase": 64,
        "articulated_waypoint_count": 8,
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
            "policy_id": "pi05_droid",
            "config_name": "pi05_droid",
            "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi05_droid",
            "checkpoint_object_manifest_sha256": "1" * 64,
            "checkpoint_generation_manifest_sha256": "2" * 64,
            "checkpoint_inventory_sha256": "3" * 64,
            "checkpoint_object_count": 1,
            "checkpoint_size_bytes": 1024,
            "action_space": "joint_position",
            "action_chunk_rows": 10,
            "open_loop_horizon": 8,
            "openpi_revision": OPENPI_SOURCE_REVISION,
        },
        "policy_identity_receipt": {"identity_verified": True},
        "max_policy_queries": 4,
        "open_loop_horizon": 8,
        "overview_camera_policy_input": False,
        "policy_may_grade_itself": False,
        "execution_spec_digest": "",
    }
    spec["execution_spec_digest"] = canonical_digest(
        spec, digest_field="execution_spec_digest"
    )
    return spec


def test_policy_bundle_requires_exact_qualified_construction_and_controls(
    tmp_path: Path,
) -> None:
    packet, scene = _articulated_packet(tmp_path)
    construction = _qualified_construction(tmp_path, scene)
    controls = _qualified_controls(tmp_path, scene, construction)
    receipt = build_native_task_arena_policy_bundle(
        job_dir=tmp_path / "policy-bundle",
        packet_dir=packet,
        construction_result_path=construction,
        control_result_path=controls,
        policy_execution_spec=_policy_spec(scene, construction, controls),
        runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
        implementation_commit="d" * 40,
        generated_at="fixed",
    )

    assert receipt["execution_mode"] == "policy"
    assert receipt["policy_candidate_id"] == "pi05_droid"
    assert receipt["candidate_policy_queried"] is False
    assert receipt["expected_output_filename"] == (
        "native_task_arena_policy_result.v1.json"
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert {
            "provider_runtime/runtime_inputs/native_task_arena_construction_result.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_control_result.v1.json",
            "provider_runtime/runtime_inputs/native_task_arena_policy_execution_spec.v1.json",
        }.issubset(names)
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
        "1.0",
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
    assert receipt["container_image"] == QUALIFIED_ADP_IMAGE
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
        in names
    )
    bundle_root = Path(receipt["bundle_path"]).parent / "provider_runtime"
    assert not (bundle_root / "native_task_packet").exists()
    assert not (bundle_root / "native_task_runtime_sources").exists()
    assert receipt["runtime_source_packet"]["redistribution_permitted"] is True
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
    assert observed["vast_launch_lock_file"] == (
        (tmp_path / "policy").resolve()
        / "native_task_arena_policy_paid_launch.lock"
    )


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
        "1.0",
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
        "1.0",
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
