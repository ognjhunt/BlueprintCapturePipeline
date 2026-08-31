from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_collision_aware_candidate_generation import (
    CandidateGeneratorContext,
    CollisionAwareCandidateGenerationError,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_curobo_candidate_generator import (
    CUROBO_BACKEND_IDENTITY,
    CuroboCandidateGenerator,
    RemoteCuroboCandidateGenerator,
    curobo_gpu_runtime_capability_contract,
)
import blueprint_pipeline.task_evaluation_curobo_candidate_generator as curobo_adapter
from blueprint_pipeline.task_evaluation_moveit_task_constructor_candidate_generator import (
    MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY,
    MoveItTaskConstructorCandidateGenerator,
    moveit_task_constructor_runtime_capability_contract,
)
from blueprint_pipeline.task_evaluation_curobo_context import (
    materialize_remote_curobo_context,
)
import blueprint_pipeline.task_evaluation_curobo_context as curobo_context


def _sealed(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _reference(root: Path, role: str) -> dict:
    path = root / f"{role}.json"
    path.write_text(json.dumps({"role": role}, sort_keys=True), encoding="utf-8")
    import hashlib

    data = path.read_bytes()
    return {
        "role": role,
        "path": str(path),
        "size_bytes": len(data),
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }


def _context(root: Path) -> CandidateGeneratorContext:
    return CandidateGeneratorContext(
        run_id="scene-839873-native-construction",
        expected_production_commit="a" * 40,
        robot_configuration=_reference(root, "robot_configuration"),
        world_configuration=_reference(root, "world_configuration"),
        task_trajectory=_reference(root, "task_trajectory"),
        analytic_candidate_inventory=_reference(root, "analytic_candidate_inventory"),
        maximum_incremental_cost_usd=0.2,
        maximum_runtime_seconds=120.0,
    )


def _solution(request: dict) -> dict:
    stages = []
    for index, kind in enumerate(("entry", "approach", "contact", "release", "retreat")):
        stages.append(
            {
                "stage_id": f"{index:02d}-{kind}",
                "stage_kind": kind,
                "waypoints": [
                    {
                        "waypoint_id": f"{kind}-00",
                        "robot_joint_positions_rad": {
                            f"panda_joint{joint}": 0.01 * joint
                            for joint in range(1, 8)
                        },
                    }
                ],
            }
        )
    return _sealed(
        {
            "solution_id": "analytic-03-entry-vertical",
            "deterministic_rank": 0,
            "source_analytic_candidate_id": "analytic-03",
            "robot_base_pose_world": {
                "position_world_m": [2.925996, -6.132664, 0.752958],
                "orientation_xyzw": [0.0, 0.0, 0.608761429, -0.79335334],
            },
            "support_surface_id": "/Site/counter@z=0.752958",
            "robot_joint_reset_positions_rad": {
                f"panda_joint{joint}": 0.01 * joint for joint in range(1, 8)
            },
            "joins_authored_phase_id": "precontact",
            "stages": stages,
            "cameras": [
                {
                    "role": "external",
                    "pose_frame": "world",
                    "position_world_m": [2.9, -6.1, 1.8],
                    "target_world_m": [3.0, -6.7, 0.82],
                },
                {
                    "role": "overview",
                    "pose_frame": "world",
                    "position_world_m": [3.0, -5.8, 2.2],
                    "target_world_m": [3.0, -6.7, 0.82],
                },
                {
                    "role": "wrist",
                    "pose_frame": "robot_body",
                    "configuration_id": "profile-wrist-default",
                },
            ],
            "addressed_feedback_codes": request["addressable_feedback_codes"],
            "minimum_world_clearance_m": 0.012,
            "minimum_self_clearance_m": 0.018,
            "joint_limit_compliance_observed": True,
            "collision_aware_motion_generated": True,
            "solution_digest": "",
        },
        "solution_digest",
    )


class _Process:
    def __init__(self, backend_identity: dict) -> None:
        self.backend_identity = backend_identity
        self.requests: list[dict] = []

    def __call__(self, argv, **_kwargs):
        environment = _kwargs["env"]
        if self.backend_identity == CUROBO_BACKEND_IDENTITY:
            assert environment["BLUEPRINT_CUROBO_SOURCE_REVISION"] == (
                CUROBO_BACKEND_IDENTITY["source_revision"]
            )
        else:
            assert environment["ROS_DISTRO"] == "jazzy"
        output = Path(argv[argv.index("--result-json") + 1])
        if "--probe" in argv:
            probe = _sealed(
                {
                    "schema_version": "task_evaluation_candidate_generator_runtime_probe.v1",
                    "runtime_ready": True,
                    "backend_identity": self.backend_identity,
                    "cuda_available": self.backend_identity == CUROBO_BACKEND_IDENTITY,
                    "cuda_device_count": (
                        1 if self.backend_identity == CUROBO_BACKEND_IDENTITY else 0
                    ),
                    "probe_digest": "",
                },
                "probe_digest",
            )
            output.write_text(json.dumps(probe), encoding="utf-8")
        else:
            request_path = Path(argv[argv.index("--request-json") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.requests.append(request)
            result = _sealed(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "backend_identity": self.backend_identity,
                    "request_digest": request["request_digest"],
                    "solutions": [_solution(request)],
                    "result_digest": "",
                },
                "result_digest",
            )
            output.write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_curobo_emits_controller_inventory_without_native_claims(tmp_path: Path) -> None:
    process = _Process(CUROBO_BACKEND_IDENTITY)
    generator = CuroboCandidateGenerator(
        context=_context(tmp_path),
        command=("/usr/bin/blueprint-curobo-candidate-service",),
        runner=process,
    )
    feedback = {
        "feedback_digest": "sha256:" + "b" * 64,
        "feedback_codes": ["forbidden_robot_task_contact:precontact"],
    }

    inventory = generator.generate(
        source_native_feedback=feedback,
        prior_history=[],
        round_index=1,
        maximum_candidates=4,
    )

    assert inventory["schema_version"] == (
        "task_evaluation_native_construction_candidate_inventory.v1"
    )
    assert inventory["model_authored_candidates"] is False
    assert inventory["source_native_feedback_digest"] == feedback["feedback_digest"]
    assert inventory["inventory_digest"] == canonical_digest(
        inventory, digest_field="inventory_digest"
    )
    candidate = inventory["candidates"][0]
    assert candidate["candidate_id"].startswith("curobo_v2_motion_generation-r1-")
    assert [row["stage_kind"] for row in candidate["entry_trajectory_variant"]["waypoints"]] == [
        "entry",
        "approach",
        "contact",
        "release",
        "retreat",
    ]
    assert candidate["generation_evidence"]["native_requirements_unresolved"] == [
        "orientation_execution",
        "collision_and_contact_readback",
        "camera_observability",
        "task_execution",
    ]
    serialized = json.dumps(candidate, sort_keys=True)
    assert '"success"' not in serialized
    assert '"status"' not in serialized
    assert '"gate"' not in serialized


def test_curobo_reopens_every_sealed_input_before_process(tmp_path: Path) -> None:
    context = _context(tmp_path)
    Path(str(context.world_configuration["path"])).write_text(
        '{"mutated":true}', encoding="utf-8"
    )
    generator = CuroboCandidateGenerator(
        context=context,
        command=("/usr/bin/blueprint-curobo-candidate-service",),
        runner=_Process(CUROBO_BACKEND_IDENTITY),
    )
    with pytest.raises(
        CollisionAwareCandidateGenerationError,
        match="candidate_generation_world_configuration_invalid",
    ):
        generator.generate(
            source_native_feedback=None,
            prior_history=[],
            round_index=0,
            maximum_candidates=2,
        )


def test_backend_rejects_missing_required_stage(tmp_path: Path) -> None:
    class MissingStage(_Process):
        def __call__(self, argv, **kwargs):
            completed = super().__call__(argv, **kwargs)
            if "--probe" not in argv:
                output = Path(argv[argv.index("--result-json") + 1])
                result = json.loads(output.read_text(encoding="utf-8"))
                solution = result["solutions"][0]
                solution["stages"] = solution["stages"][:-1]
                solution["solution_digest"] = canonical_digest(
                    solution, digest_field="solution_digest"
                )
                result["result_digest"] = canonical_digest(
                    result, digest_field="result_digest"
                )
                output.write_text(json.dumps(result), encoding="utf-8")
            return completed

    generator = CuroboCandidateGenerator(
        context=_context(tmp_path),
        command=("/usr/bin/blueprint-curobo-candidate-service",),
        runner=MissingStage(CUROBO_BACKEND_IDENTITY),
    )
    with pytest.raises(
        CollisionAwareCandidateGenerationError,
        match="candidate_generation_solution_invalid",
    ):
        generator.generate(
            source_native_feedback=None,
            prior_history=[],
            round_index=0,
            maximum_candidates=2,
        )


def test_moveit_requires_separate_exact_runtime_and_emits_same_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        CollisionAwareCandidateGenerationError,
        match="moveit_task_constructor_runtime_unavailable",
    ):
        MoveItTaskConstructorCandidateGenerator(
            context=_context(tmp_path), command=None
        )

    process = _Process(MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY)
    generator = MoveItTaskConstructorCandidateGenerator(
        context=_context(tmp_path),
        command=("/opt/ros/jazzy/bin/blueprint-mtc-candidate-service",),
        runner=process,
    )
    inventory = generator.generate(
        source_native_feedback=None,
        prior_history=[],
        round_index=0,
        maximum_candidates=2,
    )
    assert inventory["generator_backend"] == MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY
    assert inventory["candidates"][0]["candidate_id"].startswith(
        "moveit_task_constructor_ros2_jazzy-r0-"
    )


def test_runtime_contracts_pin_source_version_license_and_claim_boundary() -> None:
    curobo = curobo_gpu_runtime_capability_contract()
    assert curobo["backend_identity"] == CUROBO_BACKEND_IDENTITY
    assert curobo["backend_identity"]["source_revision"] == (
        "4ea77366ca48ee453e7df139e39fa6532af49f3b"
    )
    assert curobo["backend_identity"]["license_expression"] == "Apache-2.0"
    assert curobo["required_capabilities"]["nvidia_gpu"] is True
    assert curobo["claim_boundary"]["native_task_execution_unresolved"] is True

    moveit = moveit_task_constructor_runtime_capability_contract()
    assert moveit["backend_identity"] == MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY
    assert moveit["backend_identity"]["package_version"] == "0.1.4-2"
    assert moveit["backend_identity"]["license_expression"] == "BSD-3-Clause"
    assert moveit["provisioning"]["coinstallation_in_current_isaac_image_claimed"] is False
    assert moveit["provisioning"]["fail_closed_when_process_or_identity_unavailable"] is True


def test_remote_curobo_uses_retained_worker_without_allocating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote: dict[str, bytes] = {}

    monkeypatch.setattr(
        curobo_adapter,
        "_enroll_warm_host_key",
        lambda *_args, **_kwargs: {
            "status": "enrolled",
            "known_hosts_file": str(tmp_path / "known_hosts"),
        },
    )

    def ssh(*, remote_argv, stdin=None, **_kwargs):
        if remote_argv[:2] == ["/bin/bash", "-c"]:
            script = remote_argv[2]
            assert "git clone --filter=blob:none --no-checkout" in script
            assert CUROBO_BACKEND_IDENTITY["source_revision"] in script
            assert CUROBO_BACKEND_IDENTITY["source_tree"] in script
            assert "--no-deps --no-build-isolation" in script
            return {
                "status": "completed",
                "stdout": "BLUEPRINT_CUROBO_RUNTIME_READY\n",
            }
        if remote_argv[:2] == ["/isaac-sim/python.sh", "-c"]:
            remote_path = next(
                value
                for value in reversed(remote_argv)
                if isinstance(value, str) and value.startswith("/workspace/")
            )
            remote[remote_path] = bytes(stdin)
        elif remote_argv[0] == "env":
            output = remote_argv[remote_argv.index("--result-json") + 1]
            if "--probe" in remote_argv:
                value = _sealed(
                    {
                        "schema_version": "task_evaluation_candidate_generator_runtime_probe.v1",
                        "runtime_ready": True,
                        "backend_identity": CUROBO_BACKEND_IDENTITY,
                        "cuda_available": True,
                        "cuda_device_count": 1,
                        "probe_digest": "",
                    },
                    "probe_digest",
                )
            else:
                request_path = remote_argv[remote_argv.index("--request-json") + 1]
                request = json.loads(remote[request_path])
                for role in (
                    "robot_configuration",
                    "world_configuration",
                    "task_trajectory",
                    "analytic_candidate_inventory",
                ):
                    assert request[role]["path"].startswith(
                        "/workspace/blueprint-curobo-inputs/"
                    )
                    assert "/private/" not in request[role]["path"]
                value = _sealed(
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "backend_identity": CUROBO_BACKEND_IDENTITY,
                        "request_digest": request["request_digest"],
                        "solutions": [_solution(request)],
                        "result_digest": "",
                    },
                    "result_digest",
                )
            remote[output] = json.dumps(value).encode()
        elif remote_argv[0] == "cat":
            return {
                "status": "completed",
                "stdout": remote[remote_argv[-1]].decode(),
                "stdout_truncation": {"truncated": False},
            }
        return {"status": "completed", "stdout": ""}

    monkeypatch.setattr(curobo_adapter, "_run_warm_ssh", ssh)
    generator = RemoteCuroboCandidateGenerator(
        context=_context(tmp_path),
        warm_session={
            "ssh_host": "worker.example",
            "ssh_port": 22022,
            "remote_work_dir": "/workspace",
        },
        local_transport_root=tmp_path / "transport",
    )
    inventory = generator.generate(
        source_native_feedback=None,
        prior_history=[],
        round_index=0,
        maximum_candidates=2,
    )
    assert inventory["generator_backend"] == CUROBO_BACKEND_IDENTITY
    assert inventory["candidates"][0]["candidate_id"].startswith(
        "curobo_v2_motion_generation-r0-"
    )
    assert all("provider" not in path for path in remote)


def test_context_materializer_binds_packet_mesh_and_five_native_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    assets = packet / "assets"
    assets.mkdir(parents=True)
    collision = assets / "scene.usd"
    collision.write_bytes(b"sealed-scene-collision")
    import hashlib

    collision_digest = "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest()
    scene = _sealed(
        {
            "schema_version": "native_task_arena_scene_plan.v1",
            "scene_id": "839873",
            "task_id": "simple-relocation",
            "task_kind": "rigid_pick_place",
            "asset_directory": str(assets),
            "objects": [
                {
                    "semantic_role": "scene_collision",
                    "filename": collision.name,
                    "sha256": collision_digest,
                }
            ],
            "robot": {"robot_id": "franka_panda"},
            "plan_digest": "",
        },
        "plan_digest",
    )
    (packet / "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps(scene), encoding="utf-8"
    )

    phases = []
    for phase_id, x in (
        ("precontact", 2.8),
        ("push_contact", 2.9),
        ("push_01", 3.0),
        ("push_release", 3.1),
        ("retreat", 3.2),
        ("recovery", 2.8),
    ):
        phases.append(
            {
                "phase_id": phase_id,
                "position_world_m": [x, -6.7, 0.82],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            }
        )
    phase_plan = _sealed(
        {"phases": phases, "plan_digest": ""}, "plan_digest"
    )
    monkeypatch.setattr(
        curobo_context,
        "materialize_native_task_construction_phase_plan",
        lambda _scene: phase_plan,
    )
    monkeypatch.setattr(
        curobo_context,
        "_write_world_obj",
        lambda _source, output: output.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        ),
    )
    candidate = {
        "candidate_id": "analytic-00",
        "deterministic_rank": 0,
        "robot_base_pose_world": {
            "position_world_m": [2.925996, -6.132664, 0.752958],
            "orientation_xyzw": [0.0, 0.0, 0.608761429, -0.79335334],
        },
        "support_surface_id": "/Site/counter@z=0.752958",
        "reset_variant": {
            "robot_joint_reset_positions_rad": {
                f"panda_joint{index}": 0.01 * index for index in range(1, 8)
            }
        },
        "camera_variant": {
            "cameras": [
                {"role": "external"},
                {"role": "overview"},
                {"role": "wrist"},
            ]
        },
        "addressed_feedback_codes": [],
    }
    context, remote_root = materialize_remote_curobo_context(
        packet_dir=packet,
        universe={"inventory_digest": "sha256:" + "c" * 64, "candidates": [candidate]},
        output_root=tmp_path / "context",
        commit="a" * 40,
        warm_session={"remote_work_dir": "/workspace"},
    )
    world = json.loads(Path(context.world_configuration["path"]).read_text())
    task = json.loads(Path(context.task_trajectory["path"]).read_text())
    assert context.world_configuration["attachments"][0]["role"] == "world_collision_mesh"
    assert world["source_scene_collision_digest"] == collision_digest
    assert [
        row["stage_kind"] for row in task["candidate_phases"]["analytic-00"]
    ] == ["entry", "approach", "contact", "release", "retreat"]
    assert remote_root == "/workspace/adp_arena_provider_bundle/provider_runtime"
