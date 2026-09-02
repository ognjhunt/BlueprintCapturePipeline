"""Hermetic full-lifecycle rehearsal of the paired policy canary worker.

Every paid Quick-10 attempt before this test existed discovered a defect that
needed no GPU to find: a module missing from the bundle, a client refusing a
second readiness preflight after inference, a result lost when Isaac's
``SimulationApp.close`` ended the interpreter, a 20 Hz scene fed to 15 Hz
policies, and a Replicator graph destroyed by rebuilding an environment after
closing it.  This rehearsal drives the worker's real orchestration
(:func:`_run_selected_cell` and :func:`_run_isolated_cell_processes`), the real
episode runner, and the real OpenPI and GR00T client classes over forced
transports.  Only Isaac itself is replaced, by a fake that keeps the semantics
that bit production: ``close`` terminates the interpreter, and a closed
environment can never be rebuilt inside the same process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_action_execution import GripperConvention
from blueprint_pipeline.adp009d_groot_worker_identity import (
    expected_checkpoint_content_binding,
)
from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
from blueprint_pipeline.adp009d_task_scoring import SUPPORT_PLANE_Z_M
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.groot_n17_droid_policy_runtime import (
    CHECKPOINT_REVISION,
    EMBODIMENT_TAG,
    GROOT_SOURCE_REVISION,
    LANGUAGE_KEY,
    MODEL_ID,
    GrootN17DroidPolicyClient,
    GrootN17DroidPolicySpec,
)
# Dotted import on purpose: the impacted-test selector maps a changed source
# module to the tests whose source names it, so a worker-only change runs this
# rehearsal (pinned in tests/test_impacted_test_selection.py).
import blueprint_pipeline.native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    PROVIDER_RESULT_FILENAME,
    build_session_authority,
)
from blueprint_pipeline.native_task_arena_policy_worker import (
    GROOT_RUNTIME_IDENTITY_FILENAME,
    _runtime_groot_worker_identity,
    _to_tensor,
)
from blueprint_pipeline.openpi_droid_policy_runtime import (
    OpenPIDroidPolicySpec,
    OpenPIWebsocketDroidPolicyClient,
)
from tests.test_adp009d_policy_episode import _DESTINATION, _LifecycleEnvironment


RUN_ID = "scene-839873-canary-rehearsal"
PI05_POLICY_SPEC = {
    "policy_id": "pi05_droid_jointpos_polaris",
    "config_name": "pi05_droid_jointpos_polaris",
    "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris",
    "checkpoint_object_manifest_sha256": "1" * 64,
    "checkpoint_generation_manifest_sha256": "2" * 64,
    "checkpoint_inventory_sha256": "3" * 64,
    "checkpoint_object_count": 1,
    "checkpoint_size_bytes": 1,
    "action_space": "joint_position",
    "action_chunk_rows": 10,
}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _activation() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_campaign_activation.v1",
        "run_id": RUN_ID,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "campaign_unit_count": 10,
        "campaign_units": [
            {
                "campaign_unit_id": f"unit-{index}",
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "candidate_ids": list(CANDIDATE_IDS),
            }
            for index in range(10)
        ],
        "activation_digest": "",
    }
    value["activation_digest"] = canonical_digest(value, digest_field="activation_digest")
    return value


def _scene_plan() -> dict[str, Any]:
    pose = {
        "position_world_m": [3.4681748, -3.3100837, SUPPORT_PLANE_Z_M],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "task_id": "scene-839873-mug-planar-push",
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "franka_panda"},
        "objects": [
            {
                "name": "task_object",
                "task_subject": True,
                "pose_world": json.loads(json.dumps(pose)),
                "reset_state": {"root_pose_world": json.loads(json.dumps(pose))},
            }
        ],
        "cameras": [
            {"role": "external", "frame_from_camera_matrix": [1.0] * 16},
            {"role": "wrist", "frame_from_camera_matrix": [1.0] * 16},
        ],
        # The compiled scene still carries the pre-canary 20 Hz cadence; the
        # worker must resolve it to the frozen DROID adapters' 15 Hz.
        "cadence": {
            "control_frequency_hz": 20.0,
            "physics_frequency_hz": 120.0,
            "physics_dt_seconds": 1.0 / 120.0,
            "control_decimation": 6,
            "maximum_action_steps": 240,
            "settle_window_samples": 1,
            "episode_length_seconds": 13.05,
        },
        "task_spec": {
            "schema_version": "adp_task_spec.v1",
            "task_kind": "rigid_pick_place",
            "destination_position_world_m": list(_DESTINATION),
            "support_plane_z_m": SUPPORT_PLANE_Z_M,
            "settle_window_samples": 1,
            "require_sealed_start_pose": True,
            "control_frequency_hz": 20.0,
            "maximum_action_steps": 240,
        },
        "plan_digest": "",
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def _execution_spec(candidate: str, *, port: int) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_execution_spec.v1",
        "candidate_id": candidate,
        "execution_authority": "internal_policy_canary_unqualified",
        "claim_ceiling": "diagnostic_policy_execution",
        "ranking_permitted": False,
        "qualification_permitted": False,
        "scene_promotion_permitted": False,
        "policy_endpoint": {
            "host": "127.0.0.1",
            "port": port,
            "credential_env": (
                "BLUEPRINT_PI05_API_KEY"
                if candidate == "pi05_droid"
                else "BLUEPRINT_GROOT_API_TOKEN"
            ),
        },
        "policy_spec": dict(PI05_POLICY_SPEC) if candidate == "pi05_droid" else {},
        "candidate_rights_binding": {"status": "admitted"},
        "checkpoint_digest": "sha256:" + ("c" if candidate == "pi05_droid" else "d") * 64,
        "runtime_identity_digest": "sha256:" + ("e" if candidate == "pi05_droid" else "f") * 64,
        "prompt": "push the mug across the table",
        "max_policy_queries": 1,
        "open_loop_horizon": 8,
        "execution_spec_digest": "",
    }
    value["execution_spec_digest"] = canonical_digest(
        value, digest_field="execution_spec_digest"
    )
    return value


def _groot_identity_receipt() -> dict[str, Any]:
    return {
        "status": "verified",
        "model_id": MODEL_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "groot_source_revision": GROOT_SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files_sha256": "1" * 64,
        "checkpoint_content_manifest_digest": expected_checkpoint_content_binding()[
            "file_manifest_digest"
        ],
        "environment_lock_sha256": "2" * 64,
    }


def _stage_runtime_root(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out the provider bundle exactly as the worker reads it on a GPU host."""

    runtime = tmp_path / "provider_runtime"
    provider_output = tmp_path / "runtime_output"
    provider_output.mkdir(parents=True)
    packet_receipt = _write(
        runtime / "native_task_packet" / "native_task_arena_packet_receipt.v1.json",
        {"schema_version": "native_task_arena_packet_receipt.v1"},
    )
    _write(runtime / "native_task_packet" / "native_task_arena_scene_plan.v1.json", _scene_plan())
    runtime_source = _write(
        runtime / "native_task_runtime_sources" / "native_task_runtime_source_packet.v1.json",
        {"schema_version": "native_task_runtime_source_packet.v1"},
    )
    activation = _activation()
    activation_path = _write(
        runtime / "runtime_inputs" / "task_evaluation_policy_campaign_activation.v1.json",
        activation,
    )
    scene_revision_digest = "sha256:" + "9" * 64
    construction = {
        "schema_version": "task_evaluation_episode_compilation_result.v1",
        "status": "compiled_for_production_launch",
        "blockers": [],
        "configured_scene_revision_digest": scene_revision_digest,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    construction["result_digest"] = canonical_digest(construction, digest_field="result_digest")
    construction_path = _write(
        runtime / "runtime_inputs" / "native_task_arena_construction_result.v1.json",
        construction,
    )
    cells = []
    for index in range(10):
        scenario = {
            "family": "canonical_anchor" if index < 2 else "placement_approach",
            "parameters": {} if index < 2 else {"object_start_y_delta_m": 0.005 * index},
        }
        cells.append(
            {
                "cell_id": f"cell-{index}",
                "seed": 3100 + index,
                "cell_spec_digest": "sha256:" + f"{index:064x}",
                "family": scenario["family"],
                "resolved_scenario": scenario,
                "resolved_scenario_digest": canonical_digest(scenario),
                "control_diagnostic": {
                    "mode": "nonblocking_diagnostic_pending",
                    "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False,
                },
            }
        )
    inputs: dict[str, Any] = {
        "schema_version": "task_evaluation_policy_canary_runtime_inputs.v1",
        "run_id": RUN_ID,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "activation_digest": activation["activation_digest"],
        "scene_revision_digest": scene_revision_digest,
        "matrix_digest": "sha256:" + "8" * 64,
        "configuration_digest": "sha256:" + "1" * 64,
        "plan_digest": "sha256:" + "2" * 64,
        "base_native_packet": _record(packet_receipt),
        "runtime_source": _record(runtime_source),
        "construction_result": _record(construction_path),
        "cells": cells,
        "execution_authority": {
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
            "single_warm_provider_session_required": True,
            "caller_surviving_watchdog_required": True,
            "billing_teardown_provider_zero_required": True,
        },
        "runtime_inputs_digest": "",
    }
    inputs["runtime_inputs_digest"] = canonical_digest(
        inputs, digest_field="runtime_inputs_digest"
    )
    inputs_path = _write(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json", inputs)
    authority = build_session_authority(
        activation_manifest=activation,
        activation_record=_record(activation_path),
        runtime_inputs=inputs,
        runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-0123456789abcdef",
        hard_cap_usd=4.0,
        hard_ttl_seconds=9_000,
    )
    _write(runtime / "runtime_inputs" / "policy_canary_session_authority.json", authority)
    manifest: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_provider_bundle.v1",
        "execution_mode": "internal_policy_canary_paired_session",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "runtime_inputs_digest": inputs["runtime_inputs_digest"],
        "authority_digest": authority["authority_digest"],
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    _write(runtime / "adp_arena_provider_manifest.json", manifest)
    _write(
        runtime / "runtime_inputs" / "policy_execution_spec.pi05_droid.json",
        _execution_spec("pi05_droid", port=8000),
    )
    _write(
        runtime / "runtime_inputs" / "policy_execution_spec.groot_n17_droid.json",
        _execution_spec("groot_n17_droid", port=5555),
    )
    _write(provider_output / GROOT_RUNTIME_IDENTITY_FILENAME, _groot_identity_receipt())
    return runtime, provider_output


class FakeIsaac:
    """Isaac Sim as the paid runs observed it, without Isaac.

    ``SimulationApp.close`` ends the interpreter (the run that lost its result
    exited 0 exactly there), and once an environment has been closed the
    Replicator camera graph is gone for the rest of the process (the run whose
    19 later cells failed with ``Unable to retrieve replicator graph``).
    """

    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path
        self.launches = 0
        self.builds = 0
        self.environment_closes = 0
        self.closed = False
        self.graph_valid = True
        self.result_sealed_at_close: bool | None = None
        self.built_control_frequencies: list[float] = []
        self.built_cell_ids: list[str] = []

    def launch(self, receipt_path: Path, *, device: str) -> tuple[Any, dict[str, Any]]:
        self.launches += 1
        return _FakeSimulationApp(self), {"device": device, "receipt": receipt_path.name}

    def build(
        self,
        scene_plan: dict[str, Any],
        *,
        device: str,
        bundle_root: Path,
        preconstruction_receipt: dict[str, Any],
    ) -> Any:
        del device, bundle_root, preconstruction_receipt
        if self.closed or not self.graph_valid:
            raise RuntimeError("Unable to retrieve replicator graph")
        self.builds += 1
        self.built_control_frequencies.append(
            float(scene_plan["cadence"]["control_frequency_hz"])
        )
        self.built_cell_ids.append(str(scene_plan["scenario"]["cell_id"]))
        return SimpleNamespace(env=_FakeEnvironment(self))


class _FakeSimulationApp:
    def __init__(self, isaac: FakeIsaac) -> None:
        self._isaac = isaac

    def close(self) -> None:
        self._isaac.closed = True
        self._isaac.result_sealed_at_close = self._isaac.result_path.is_file()
        raise SystemExit(0)


class _FakeEnvironment:
    def __init__(self, isaac: FakeIsaac) -> None:
        self._isaac = isaac
        self.reset_seeds: list[int] = []
        self.unwrapped = SimpleNamespace(scene={"robot": object()})

    def reset(self, *, seed: int) -> None:
        self.reset_seeds.append(int(seed))

    def close(self) -> None:
        self._isaac.environment_closes += 1
        self._isaac.graph_valid = False


class _OpenPIVendor:
    """The pinned OpenPI websocket server as seen through its client."""

    def __init__(self, spec: OpenPIDroidPolicySpec) -> None:
        self._spec = spec
        self.inferences = 0

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            **self._spec.server_metadata(),
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "4" * 64,
            "local_checkpoint_object_count": 1,
            "local_checkpoint_size_bytes": 1,
        }

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        del observation
        self.inferences += 1
        chunk = np.zeros((10, 8), dtype=float)
        chunk[:, 0] = 0.25
        chunk[:, 7] = 0.9
        return {"actions": chunk, "policy_timing": {"infer_ms": 12.0}}


class _GrootVendor:
    """The pinned GR00T ZMQ server as seen through its client."""

    def __init__(self) -> None:
        self.inferences = 0

    def ping(self) -> bool:
        return True

    def reset(self) -> dict[str, Any]:
        return {"status": "reset"}

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "video": {
                "modality_keys": ["exterior_image_1_left", "wrist_image_left"],
                "delta_indices": [0],
            },
            "state": {
                "modality_keys": ["eef_9d", "gripper_position", "joint_position"],
                "delta_indices": [0],
            },
            "action": {
                "modality_keys": ["eef_9d", "gripper_position", "joint_position"],
                "delta_indices": list(range(40)),
            },
            "language": {"modality_keys": [LANGUAGE_KEY], "delta_indices": [0]},
        }

    def get_action(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        del request
        self.inferences += 1
        joints = np.zeros((1, 40, 7), dtype=float)
        joints[..., 0] = 0.25
        return (
            {
                "joint_position": joints,
                "gripper_position": np.zeros((1, 40, 1), dtype=float),
                "eef_9d": np.zeros((1, 40, 9), dtype=float),
            },
            {"served_by": "rehearsal"},
        )


def _real_policy_client(spec: dict[str, Any], *, groot_worker_identity_receipt=None) -> Any:
    """Construct the real client classes over forced loopback transports."""

    endpoint = spec["policy_endpoint"]
    if spec["candidate_id"] == "pi05_droid":
        policy_spec = OpenPIDroidPolicySpec(**spec["policy_spec"])
        return OpenPIWebsocketDroidPolicyClient(
            spec=policy_spec,
            host=str(endpoint["host"]),
            port=int(endpoint["port"]),
            client_factory=lambda **_kwargs: _OpenPIVendor(policy_spec),
        )
    if groot_worker_identity_receipt is None:
        raise RuntimeError("groot_runtime_worker_identity_receipt_missing")
    return GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(**spec["policy_spec"]),
        worker_identity_receipt=groot_worker_identity_receipt,
        host=str(endpoint["host"]),
        port=int(endpoint["port"]),
        client_factory=lambda **_kwargs: _GrootVendor(),
    )


def _rehearsal_runtime(isaac: FakeIsaac) -> worker.CellRuntime:
    return worker.CellRuntime(
        device="cuda:0",
        launch_isaac=isaac.launch,
        preflight_dependency_matrix=lambda *, robot_id: {
            "all_required_available": True,
            "robot_id": robot_id,
        },
        prepare_preconstruction=lambda *, expected_device: {
            "passed": True,
            "device": expected_device,
        },
        build_environment=isaac.build,
        read_device_binding=lambda built, *, expected_device: {
            "passed": built.env is not None and expected_device == "cuda:0"
        },
        gripper_probe=lambda *, env, robot, seed: {
            "status": "measured",
            "closed_command": 1.0,
            "open_command": 0.0,
            "seed": seed,
        },
        make_servo=lambda *, env, robot, gripper_convention: SimpleNamespace(
            current_grasp_frame_pose_world=lambda: None
        ),
        make_task_readback=lambda built, *, grasp_frame_pose_callback: None,
        build_episode_environment=lambda *, built, gripper_convention, servo, task_readback, to_tensor: (
            _LifecycleEnvironment(),
            {"schema_version": "rehearsal_episode_environment.v1", "seed": built.env.reset_seeds[-1]},
        ),
        to_tensor=_to_tensor,
        policy_client=_real_policy_client,
        groot_worker_identity=_runtime_groot_worker_identity,
        run_policy_episode=run_policy_episode,
    )


def _run_cell_in_process(
    *,
    index: int,
    runtime_root: Path,
    output_root: Path,
    child_root: Path,
    isaacs: list[FakeIsaac],
    runtime_factory=_rehearsal_runtime,
) -> int:
    """Stand in for one isolated cell interpreter; Isaac's close is SystemExit."""

    isaac = FakeIsaac(child_root / PROVIDER_RESULT_FILENAME)
    isaacs.append(isaac)
    try:
        return worker._run_selected_cell(
            index,
            runtime_root=runtime_root,
            output_root=child_root,
            provider_output_root=output_root,
            cell_runtime=runtime_factory(isaac),
        )
    except SystemExit as exc:
        return int(exc.code or 0)


def _sealed_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["result_digest"] == canonical_digest(value, digest_field="result_digest")
    return value


def test_selected_cell_queries_both_real_clients_and_seals_before_isaac_close(
    tmp_path: Path,
) -> None:
    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    child_root = provider_output / "cell_runs" / "03"
    child_root.mkdir(parents=True)
    isaac = FakeIsaac(child_root / PROVIDER_RESULT_FILENAME)

    with pytest.raises(SystemExit) as exited:
        worker._run_selected_cell(
            3,
            runtime_root=runtime_root,
            output_root=child_root,
            provider_output_root=provider_output,
            cell_runtime=_rehearsal_runtime(isaac),
        )

    assert exited.value.code == 0
    result = _sealed_result(child_root / PROVIDER_RESULT_FILENAME)
    assert result["status"] == "runtime_selected_cell_completed_pending_aggregation"
    assert result["selected_cell_index"] == 3
    assert [row["candidate_id"] for row in result["episodes"]] == list(CANDIDATE_IDS)
    for episode in result["episodes"]:
        assert episode["status"] == "completed"
        assert episode["candidate_policy_queried"] is True
        assert episode["actions_reached_robot"] is True
        assert episode["policy_outcome_interpretable"] is True
        assert episode["cell_id"] == "cell-3"
        assert episode["evidence_artifacts"]["frame_manifest"] is not None
        assert episode["evidence_artifacts"]["review_video"] is not None
        assert episode["episode"]["prestart_readiness"]["candidate_policy_queried"] is False
    # Exactly one Isaac launch, one environment build for the selected cell,
    # and the environment closed once after the second candidate.
    assert isaac.launches == 1
    assert isaac.builds == 1
    assert isaac.built_cell_ids == ["cell-3"]
    assert isaac.environment_closes == 1
    # The compiled 20 Hz scene reached the simulator at the policies' 15 Hz.
    assert isaac.built_control_frequencies == [15.0]
    # The provider result was durable before Isaac ended the interpreter.
    assert isaac.result_sealed_at_close is True
    assert not list((child_root / "episodes").glob("*.failure_gap.json"))
    assert (child_root / "policy_canary_telemetry_index.json").is_file()


def test_quick10_rehearsal_runs_twenty_real_client_rollouts_in_ten_isolated_processes(
    tmp_path: Path,
) -> None:
    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    isaacs: list[FakeIsaac] = []

    exit_code = worker._run_isolated_cell_processes(
        runtime_root=runtime_root,
        output_root=provider_output,
        run_cell_process=lambda **kwargs: _run_cell_in_process(**kwargs, isaacs=isaacs),
    )

    assert exit_code == 0
    result = _sealed_result(provider_output / PROVIDER_RESULT_FILENAME)
    assert result["status"] == "runtime_completed_unqualified_pending_closeout"
    assert result["isolated_simulation_process_count"] == 10
    assert result["construction_lineage_mode"] == "compiled_configured_scene_diagnostic"
    assert len(result["episodes"]) == 20
    assert all(row["status"] == "completed" for row in result["episodes"])
    assert all(row["candidate_policy_queried"] is True for row in result["episodes"])
    assert result["candidate_policy_queried"] is True
    assert {
        (row["candidate_id"], row["cell_id"], row["seed"]) for row in result["episodes"]
    } == {
        (candidate, f"cell-{index}", 3100 + index)
        for candidate in CANDIDATE_IDS
        for index in range(10)
    }
    assert [
        row["evidence_artifacts"]["reset_state"]["relative_path"].split("/")[:2]
        for row in result["episodes"]
    ] == [["cell_runs", f"{index:02d}"] for index in range(10) for _ in CANDIDATE_IDS]
    assert len(isaacs) == 10
    assert all(isaac.launches == 1 and isaac.builds == 1 for isaac in isaacs)
    assert all(isaac.result_sealed_at_close is True for isaac in isaacs)
    assert all(isaac.built_control_frequencies == [15.0] for isaac in isaacs)
    roles = {row["role"] for row in result["artifact_inventory"]}
    assert {"indexed_episode_telemetry", "review_video", "policy_query_receipt"} <= roles


def test_one_failed_cell_is_a_typed_gap_and_the_other_nineteen_rollouts_continue(
    tmp_path: Path,
) -> None:
    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    isaacs: list[FakeIsaac] = []

    def failing_runtime(isaac: FakeIsaac) -> worker.CellRuntime:
        runtime = _rehearsal_runtime(isaac)
        if len(isaacs) != 6:
            return runtime

        def build(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("camera graph failed at /workspace/private/cell.py")

        return worker.CellRuntime(**{**runtime.__dict__, "build_environment": build})

    exit_code = worker._run_isolated_cell_processes(
        runtime_root=runtime_root,
        output_root=provider_output,
        run_cell_process=lambda **kwargs: _run_cell_in_process(
            **kwargs, isaacs=isaacs, runtime_factory=failing_runtime
        ),
    )

    assert exit_code == 0
    result = _sealed_result(provider_output / PROVIDER_RESULT_FILENAME)
    assert len(result["episodes"]) == 20
    failed = [row for row in result["episodes"] if row["cell_id"] == "cell-5"]
    healthy = [row for row in result["episodes"] if row["cell_id"] != "cell-5"]
    assert len(failed) == 2 and len(healthy) == 18
    assert all(row["status"] == "blocked" for row in failed)
    assert all(row["typed_harness_failure"] == "RuntimeError" for row in failed)
    assert all(row["candidate_policy_queried"] is False for row in failed)
    assert all(row["status"] == "completed" for row in healthy)
    gaps = sorted((provider_output / "cell_runs" / "05" / "episodes").glob("*.failure_gap.json"))
    assert sorted(path.name.split("--")[-1] for path in gaps) == sorted(
        f"{candidate}.failure_gap.json" for candidate in CANDIDATE_IDS
    )
    for path in gaps:
        gap = json.loads(path.read_text(encoding="utf-8"))
        assert gap["failure_type"] == "RuntimeError"
        assert gap["failure_message"] == "camera graph failed at <path>"
        assert "/workspace" not in path.read_text(encoding="utf-8")
    # The failing cell still sealed its result before Isaac closed, and no
    # other process was disturbed.
    assert isaacs[5].result_sealed_at_close is True
    assert all(isaac.result_sealed_at_close is True for isaac in isaacs)


def test_cadence_mismatch_is_refused_by_the_real_episode_contract_and_still_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    child_root = provider_output / "cell_runs" / "00"
    child_root.mkdir(parents=True)
    isaac = FakeIsaac(child_root / PROVIDER_RESULT_FILENAME)
    resolved = worker._resolved_scene_plan

    def stale_cadence(base: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
        plan = resolved(base, cell)
        plan["task_spec"]["control_frequency_hz"] = 20.0
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
        return plan

    monkeypatch.setattr(worker, "_resolved_scene_plan", stale_cadence)

    with pytest.raises(SystemExit) as exited:
        worker._run_selected_cell(
            0,
            runtime_root=runtime_root,
            output_root=child_root,
            provider_output_root=provider_output,
            cell_runtime=_rehearsal_runtime(isaac),
        )

    assert exited.value.code == 0
    result = _sealed_result(child_root / PROVIDER_RESULT_FILENAME)
    assert [row["typed_harness_failure"] for row in result["episodes"]] == [
        "PolicyEpisodeError",
        "PolicyEpisodeError",
    ]
    assert all(row["candidate_policy_queried"] is False for row in result["episodes"])
    gaps = sorted((child_root / "episodes").glob("*.failure_gap.json"))
    assert len(gaps) == 2
    assert all(
        "policy_episode_control_frequency_task_spec_mismatch"
        in json.loads(path.read_text(encoding="utf-8"))["failure_message"]
        for path in gaps
    )
    assert isaac.result_sealed_at_close is True


def test_environment_rebuild_after_close_is_impossible_in_one_process(
    tmp_path: Path,
) -> None:
    """A second cell in the same process would rebuild after close; refuse it."""

    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    child_root = provider_output / "cell_runs" / "01"
    child_root.mkdir(parents=True)
    isaac = FakeIsaac(child_root / PROVIDER_RESULT_FILENAME)
    runtime = _rehearsal_runtime(isaac)

    with pytest.raises(SystemExit):
        worker._run_selected_cell(
            1,
            runtime_root=runtime_root,
            output_root=child_root,
            provider_output_root=provider_output,
            cell_runtime=runtime,
        )

    assert isaac.environment_closes == 1
    with pytest.raises(RuntimeError, match="Unable to retrieve replicator graph"):
        runtime.build_environment(
            _scene_plan(), device="cuda:0", bundle_root=runtime_root, preconstruction_receipt={}
        )


def test_real_clients_refuse_a_second_readiness_preflight_after_inference(
    tmp_path: Path,
) -> None:
    """Pins the invariant behind per-cell isolation: one client, one episode."""

    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    specs = {
        candidate: json.loads(
            (runtime_root / "runtime_inputs" / f"policy_execution_spec.{candidate}.json").read_text(
                encoding="utf-8"
            )
        )
        for candidate in CANDIDATE_IDS
    }
    groot_receipt, _evidence = _runtime_groot_worker_identity(
        output_root=provider_output, spec=specs["groot_n17_droid"]
    )
    gripper = GripperConvention(closed_command=1.0, open_command=0.0, measured_by_probe=True)

    def episode(client: Any, candidate: str, label: str) -> dict[str, Any]:
        return run_policy_episode(
            environment=_LifecycleEnvironment(),
            policy=client,
            candidate_id=candidate,
            prompt="push the mug across the table",
            gripper=gripper,
            task_spec=worker._resolved_scene_plan(
                _scene_plan(),
                {"cell_id": f"cell-{label}", "seed": 1, "resolved_scenario": {}},
            )["task_spec"],
            max_policy_queries=1,
            settle_window_samples=1,
            open_loop_horizon=8,
            media_output_dir=tmp_path / candidate / label,
            episode_id=f"{candidate}-{label}",
            require_complete_multicamera_media=True,
            require_prestart_readiness=True,
        )

    for candidate, receipt in (("pi05_droid", None), ("groot_n17_droid", groot_receipt)):
        client = _real_policy_client(specs[candidate], groot_worker_identity_receipt=receipt)
        first = episode(client, candidate, "first")
        assert first["candidate_policy_queried"] is True
        # The client refuses the second readiness preflight outright, and the
        # episode lifecycle refuses a readiness receipt from a queried client;
        # whichever boundary fires first, a second episode on one client is
        # a typed refusal, never a silent rollout.
        with pytest.raises(
            ValueError,
            match="preflight_after_inference_forbidden|policy_episode_readiness_queried_candidate",
        ):
            episode(client, candidate, "second")
