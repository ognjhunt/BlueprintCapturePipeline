from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    SCHEMA_VERSION,
    TaskEvaluationLaunchPreparationContractError,
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)


DIGESTS = [f"sha256:{index:064x}" for index in range(1, 32)]


def ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/object-{index}.json",
        "digest": DIGESTS[index],
        "size_bytes": 1000 + index,
    }


def request() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_production_commit": "a" * 40,
        "preparation_id": "team-a-scene-841007-zero-v1",
        "team_namespace": "team-a",
        "run_id": "run-scene-841007-zero-v1",
        "scene": {
            "identity": {"id": "interiorgs-841007", "version": "v1"},
            "source_manifest": ref(1),
            "appearance": {
                "kind": "interiorgs",
                "representation": ref(2),
                "renderer_qualification": ref(3),
            },
            "geometry": {
                "kind": "sage_derived",
                "collision": ref(4),
                "validation": ref(5),
            },
            "registration": {
                "metric_registration": ref(6),
                "support_plane": ref(7),
                "robot_base": ref(8),
                "camera_calibration": ref(9),
            },
            "rights": {
                "admission": ref(10),
                "source_bytes_redistributable": False,
                "provider_disclosure_scope": "derived_only",
            },
        },
        "robot": {
            "identity": {"id": "franka-panda", "version": "isaac-2026-1"},
            "configuration": ref(11),
            "kinematics": ref(12),
            "joint_bounds": ref(13),
            "controller_configuration": ref(14),
        },
        "controller": {
            "identity": {"id": "zero-action", "version": "v1"},
            "kind": "zero_action",
            "configuration": ref(15),
        },
        "task": {
            "identity": {"id": "planar-can-push", "version": "v1"},
            "definition": ref(16),
            "success_criteria": ref(17),
            "execution": ref(18),
        },
        "sensors": {"configuration": ref(19)},
        "runtime": {
            "identity": {"id": "native-arena", "version": "v1"},
            "oci_image": "registry.example/arena@sha256:" + "a" * 64,
            "entrypoint": ["/opt/blueprint/run-task-evaluation"],
            "health_protocol": ref(20),
            "requirements": {
                "cpu_cores": 8,
                "memory_gib": 32,
                "gpu_count": 1,
                "disk_gib": 64,
            },
            "network": {"default": "deny", "allowlist": []},
            "secret_refs": [],
            "mounts": [
                {"source": ref(21), "container_path": "/inputs", "mode": "read_only"},
                {"container_path": "/outputs", "mode": "output"},
            ],
            "output_limit_bytes": 20_000_000_000,
        },
        "execution_adapter": {
            "kind": "native_task_arena",
            "version": "v1",
            "construction_packet_bundle": ref(22),
            "runtime_source_bundle": ref(23),
        },
        "publication": {
            "input_namespace": "team-a-scene-841007-v1",
            "service_account_readback_required": True,
        },
        "spend": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 0.75,
            "hard_ttl_seconds": 3300,
            "retry_cap": 0,
            "provider_allowlist": [],
        },
    }


def test_accepts_scene_neutral_customer_contract_and_has_stable_digest() -> None:
    value = request()
    assert validate_launch_preparation_request(value) == value
    assert launch_preparation_request_digest(value) == launch_preparation_request_digest(
        copy.deepcopy(value)
    )


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda value: value["scene"]["identity"].update(id="different-scene"),
            None,
        ),
        (
            lambda value: value["runtime"].update(oci_image="registry.example/arena:latest"),
            "launch_preparation_request_invalid:runtime.oci_image",
        ),
        (
            lambda value: value["runtime"]["network"].update(default="allow"),
            "launch_preparation_request_invalid:runtime.network.default",
        ),
        (
            lambda value: value["spend"].update(retry_cap=1),
            "launch_preparation_request_invalid:spend.retry_cap",
        ),
        (
            lambda value: value["execution_adapter"].update(kind="unknown_adapter"),
            "launch_preparation_execution_adapter_unavailable",
        ),
    ],
)
def test_request_contract_is_scene_neutral_and_fail_closed(mutate, blocker) -> None:
    value = request()
    original_digest = launch_preparation_request_digest(value)
    mutate(value)
    if blocker is None:
        assert launch_preparation_request_digest(value) != original_digest
        return
    with pytest.raises(TaskEvaluationLaunchPreparationContractError, match=blocker):
        validate_launch_preparation_request(value)


def test_rejects_rights_disclosure_conflict_and_host_paths() -> None:
    value = request()
    value["scene"]["rights"]["provider_disclosure_scope"] = "source_and_derived"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_scene_disclosure_conflicts_with_rights",
    ):
        validate_launch_preparation_request(value)

    value = request()
    value["scene"]["source_manifest"]["uri"] = "/var/lib/blueprint/scene.json"
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_request_invalid:scene.source_manifest.uri",
    ):
        validate_launch_preparation_request(value)


def test_requires_one_bounded_output_mount_and_gpu() -> None:
    value = request()
    value["runtime"]["mounts"][1]["mode"] = "read_only"
    value["runtime"]["mounts"][1]["source"] = ref(22)
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_output_mount_count_invalid",
    ):
        validate_launch_preparation_request(value)

    value = request()
    value["runtime"]["requirements"]["gpu_count"] = 0
    with pytest.raises(
        TaskEvaluationLaunchPreparationContractError,
        match="launch_preparation_gpu_requirement_missing",
    ):
        validate_launch_preparation_request(value)
