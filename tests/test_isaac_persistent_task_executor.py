from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_persistent_task_completion_client as client
from blueprint_pipeline import isaac_persistent_task_executor_service as service
from blueprint_pipeline import isaac_runtime_task_backend as backend_module
from blueprint_pipeline.g1_proprioception_map import (
    G1_CANONICAL_DOF_GROUPS,
    validate_g1_sonic_state_dims,
)
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_FULL_JOINT_ORDER,
    PROTOCOL_V4_MAPPING_DIGEST,
)
from blueprint_pipeline.task_episode_baseline import canonical_task_contract_sha256

IsaacPersistentTaskBackend = backend_module.IsaacPersistentTaskBackend


CONTRACT = {
    "registered_criteria": [
        {
            "criterion_id": "microwave_door_open_angle",
            "observable_transition": "articulation_angle_rad",
            "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
            "comparison": "increase_at_least",
            "tolerance": 0.35,
            "unit": "rad",
        }
    ]
}


def _signing_key_file(tmp_path: Path) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pem = Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_file = tmp_path / "signing_key.pem"
    key_file.write_bytes(pem)
    return key_file


class _DC:
    def __init__(self):
        self.task_value = 0.0
        self.robot_target = 0.0

    def get_articulation(self, path):
        return "robot" if path == "/World/G1" else "task"

    def find_articulation_dof(self, articulation, name):
        return f"{articulation}:{name}"

    def get_dof_position(self, dof):
        return self.task_value if dof.startswith("task:") else self.robot_target

    def set_dof_position_target(self, dof, value):
        assert dof.startswith("robot:")
        self.robot_target = value


class _App:
    def __init__(self, dc):
        self.dc = dc

    def update(self):
        self.dc.task_value += abs(self.dc.robot_target) * 0.05


class _ReviewRenderer:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def render(self, *, step_index, target_prim_path):
        return [
            {
                "camera_role": "overview",
                "frame_index": step_index,
                "path": str(self.tmp_path / f"overview_{step_index:04d}.png"),
                "sha256": "a" * 64,
                "target_prim_path": target_prim_path,
            }
        ]


def _hermetic_backend(tmp_path: Path) -> IsaacPersistentTaskBackend:
    backend = IsaacPersistentTaskBackend.__new__(IsaacPersistentTaskBackend)
    backend.dc = _DC()
    backend.app = _App(backend.dc)
    backend.robot_handle = "robot"
    backend.evidence_dir = tmp_path
    backend.session_id = "persistent-session-1"
    backend.stage_id = "stage-1"
    backend.review_renderer = _ReviewRenderer(tmp_path)
    return backend


def _request(step: int, value: float, contract=CONTRACT) -> dict:
    action = {"action_chunk": [value]}
    action_sha = hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "step_index": step,
        "action": action,
        "wam_output": {
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "joint_names": list(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_positions": [value] * len(PROTOCOL_V4_FULL_JOINT_ORDER),
                "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
                "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
            }
        },
        "task_success_contract": contract,
        "physics_steps_per_action": 4,
    }


def test_completion_client_posts_attempt_bound_request(monkeypatch, tmp_path: Path):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_payload = {
        "schema_version": "oscar_task_completion_evaluator_request.v1",
        "step_index": 3,
        "source_action_sha256": "a" * 64,
        "action": {"generated_robot_state": {"joint_positions": [0.1]}},
        "task_success_contract": {"criterion_id": "door-open"},
    }
    request_path.write_text(json.dumps(request_payload))

    result_payload = {
        "status": "completed",
        "passed": True,
        "simulator_session_id": "session-1",
        "stage_id": "stage-1",
        "runtime_result_id": "result-1",
        "source_action_sha256": "a" * 64,
        "articulation_prim_path": "/World/G1",
        "before_timestamp": "2026-07-10T00:00:00Z",
        "after_timestamp": "2026-07-10T00:00:01Z",
        "before_value": 0.0,
        "after_value": 0.4,
        "unit": "radian",
        "criterion_id": "door-open",
        "observable_transition": "door_joint_increases",
        "evaluator_attestation": {"verification_status": "verified"},
        "persistent_simulator_state_applied": True,
        "official_controller_action_applied": True,
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(result_payload).encode()

    def fake_open(request, timeout, policy):
        del policy
        assert request.full_url == "http://127.0.0.1:8765/apply-and-measure"
        assert timeout == 120
        assert json.loads(request.data) == request_payload
        return Response()

    monkeypatch.setattr(client.safe_outbound_http, "_open_with_policy", fake_open)
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_INPUT", str(request_path))
    monkeypatch.setenv("BLUEPRINT_TASK_COMPLETION_OUTPUT", str(output_path))

    assert client.main() == 0
    assert json.loads(output_path.read_text()) == result_payload


def test_backend_applies_two_actions_to_one_persistent_stage_and_measures_transition(
    tmp_path: Path,
) -> None:
    backend = _hermetic_backend(tmp_path)
    baseline = backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.install_episode_baseline_attestation({"signature_verified": True})
    assert baseline["episode_initial_value"] == pytest.approx(0.0)
    assert (tmp_path / "task_episode_baseline.json").is_file()

    first = backend.apply_and_measure(_request(1, 0.5))
    second = backend.apply_and_measure(_request(2, 1.0))
    assert first["simulator_session_id"] == second["simulator_session_id"]
    assert first["stage_id"] == second["stage_id"]
    assert first["runtime_result_id"] != second["runtime_result_id"]
    assert second["before_value"] == pytest.approx(first["after_value"])
    assert second["after_value"] > second["before_value"]
    assert first["source_action_sha256"] != second["source_action_sha256"]
    assert Path(first["evidence_artifacts"][0]["path"]).is_file()
    assert first["review_frames"][0]["camera_role"] == "overview"

    for result in (first, second):
        assert result["episode_initial_value"] == pytest.approx(
            baseline["episode_initial_value"]
        )
        assert result["step_before"] == pytest.approx(result["before_value"])
        assert result["step_after"] == pytest.approx(result["after_value"])
        assert result["step_delta"] == pytest.approx(
            result["after_value"] - result["before_value"]
        )
        assert result["episode_delta"] == pytest.approx(
            result["after_value"] - baseline["episode_initial_value"]
        )
        assert result["episode_baseline_digest"] == baseline["baseline_digest"]
    persisted = json.loads(
        Path(second["evidence_artifacts"][0]["path"]).read_text(encoding="utf-8")
    )
    assert persisted["episode_initial_value"] == pytest.approx(
        baseline["episode_initial_value"]
    )
    assert persisted["episode_delta"] == pytest.approx(second["episode_delta"])


def test_backend_blocks_apply_without_episode_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    with pytest.raises(RuntimeError, match="task_episode_baseline_missing"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_second_baseline_capture(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    with pytest.raises(RuntimeError, match="persistent_isaac_episode_baseline_already_captured"):
        backend.capture_episode_baseline(
            task_success_contract=CONTRACT,
            attempt_id="run-1-attempt-000001",
            launch_nonce="nonce-1",
        )


def test_backend_restart_cannot_recapture_same_attempt_baseline(tmp_path: Path):
    first = _hermetic_backend(tmp_path)
    first.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    restarted = _hermetic_backend(tmp_path)
    with pytest.raises(RuntimeError, match="baseline_artifact_already_exists"):
        restarted.capture_episode_baseline(
            task_success_contract=CONTRACT,
            attempt_id="run-1-attempt-000001",
            launch_nonce="nonce-1",
        )


def test_backend_blocks_tampered_episode_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.episode_baseline["episode_initial_value"] = -1.0
    with pytest.raises(RuntimeError, match="task_episode_baseline_digest_mismatch"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_session_restart_after_baseline(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    backend.session_id = "persistent-session-2-restarted"
    with pytest.raises(RuntimeError, match="task_episode_baseline_session_mismatch"):
        backend.apply_and_measure(_request(0, 0.5))


def test_backend_blocks_changed_target_prim(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.capture_episode_baseline(
        task_success_contract=CONTRACT,
        attempt_id="run-1-attempt-000001",
        launch_nonce="nonce-1",
    )
    changed = {
        "registered_criteria": [
            {
                **CONTRACT["registered_criteria"][0],
                "articulation_prim_path": "/root/Refrigerator001/Door",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="task_episode_baseline_prim_mismatch"):
        backend.apply_and_measure(_request(0, 0.5, contract=changed))


class _ProprioDC:
    def __init__(self, names):
        self.names = list(names)

    def get_articulation_dof_count(self, handle):
        return len(self.names)

    def get_articulation_dof(self, handle, index):
        return index

    def get_dof_name(self, dof):
        return self.names[dof]

    def get_dof_position(self, dof):
        return 0.01 * dof

    def get_articulation_root_body(self, handle):
        return f"{handle}:root"

    def get_rigid_body_pose(self, body):
        del body

        class Rotation:
            x = 0.0
            y = 0.0
            z = 0.0
            w = 1.0

        class Pose:
            r = Rotation()

        return Pose()


def _full_g1_dof_names() -> list[str]:
    return [name for names in G1_CANONICAL_DOF_GROUPS.values() for name in names]


def test_initial_policy_state_maps_full_g1_inventory_and_passes_dims_contract(
    tmp_path: Path,
):
    backend = _hermetic_backend(tmp_path)
    backend.dc = _ProprioDC(_full_g1_dof_names())
    state = backend.initial_policy_state()
    assert validate_g1_sonic_state_dims(state) == []
    assert state["left_leg"] == [pytest.approx(0.01 * index) for index in range(6)]
    mapping = state["proprioception_mapping"]
    assert len(mapping["mapping_digest"]) == 64
    assert len(mapping["observed_dof_inventory"]) == 43
    assert mapping["dimensions"]["left_arm"] == 7
    assert mapping["unmapped_observed_dofs"] == []
    assert state["measurement"]["source"] == (
        "live_isaac_articulation_dof_positions_and_base_orientation"
    )
    assert state["measurement"]["surrogate"] is False
    assert state["measurement"]["mapping_digest"] == mapping["mapping_digest"]


def test_initial_policy_state_blocks_on_missing_required_dof(tmp_path: Path):
    names = [name for name in _full_g1_dof_names() if name != "right_wrist_yaw_joint"]
    backend = _hermetic_backend(tmp_path)
    backend.dc = _ProprioDC(names)
    with pytest.raises(
        RuntimeError,
        match=r"persistent_isaac_initial_proprio_mapping_blocked:"
        r".*g1_proprioception_required_dof_missing:right_wrist_yaw_joint",
    ):
        backend.initial_policy_state()


def test_initial_policy_state_blocks_on_duplicate_dof(tmp_path: Path):
    backend = _hermetic_backend(tmp_path)
    backend.dc = _ProprioDC(_full_g1_dof_names() + ["left_hip_pitch_joint"])
    with pytest.raises(
        RuntimeError,
        match="g1_proprioception_observed_dof_duplicate:left_hip_pitch_joint",
    ):
        backend.initial_policy_state()


class _ServiceBackend:
    def __init__(self, tmp_path: Path):
        self.evidence_dir = tmp_path
        self.results: list[dict] = []

    def queue(self, result: dict) -> None:
        self.results.append(result)

    def apply_and_measure(self, request):
        return self.results.pop(0)


def _measurement(step: int, *, before: float, after: float, initial: float) -> dict:
    return {
        "schema_version": "task_transition_measurement.v1",
        "criterion_id": "microwave_door_open_angle",
        "observable_transition": "articulation_angle_rad",
        "before_value": before,
        "after_value": after,
        "episode_initial_value": initial,
        "step_before": before,
        "step_after": after,
        "step_delta": after - before,
        "episode_delta": after - initial,
        "episode_baseline_digest": "b" * 64,
        "unit": "rad",
        "source_step_index": step,
        "source_action_sha256": "a" * 64,
        "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
        "simulator_session_id": "persistent-session-1",
        "stage_id": "stage-1",
        "before_timestamp": "1",
        "after_timestamp": "2",
        "runtime_result_id": f"persistent-session-1-step-{step:04d}",
        "persistent_simulator_state_applied": True,
        "official_controller_action_applied": True,
        "evidence_artifacts": [],
        "review_frames": [],
    }


def test_service_two_small_steps_pass_episode_criterion_only_after_step_two(
    tmp_path: Path,
):
    backend = _ServiceBackend(tmp_path)
    backend.queue(_measurement(0, before=0.0, after=0.20, initial=0.0))
    backend.queue(_measurement(1, before=0.20, after=0.40, initial=0.0))
    key_file = _signing_key_file(tmp_path)

    results = [
        service._evaluate_completion_request(
            backend=backend,
            request=_request(step, 0.5),
            signing_key_file=str(key_file),
            attempt_input_manifest_sha256="c" * 64,
        )
        for step in (0, 1)
    ]
    assert results[0]["passed"] is False
    assert results[1]["passed"] is True
    for result in results:
        assert result["status"] == "completed"
        assert result["evaluation_basis"] == "episode_relative"
        assert result["comparison"] == "increase_at_least"
        assert result["attempt_input_manifest_sha256"] == "c" * 64
        assert result["evaluator_attestation"]["signature_verified"] is True
    assert results[0]["episode_delta"] == pytest.approx(0.20)
    assert results[1]["episode_delta"] == pytest.approx(0.40)


def test_service_blocks_result_without_episode_fields(tmp_path: Path):
    backend = _ServiceBackend(tmp_path)
    legacy = _measurement(0, before=0.0, after=0.40, initial=0.0)
    del legacy["episode_initial_value"]
    backend.queue(legacy)
    with pytest.raises(
        RuntimeError, match="persistent_isaac_task_result_episode_fields_missing"
    ):
        service._evaluate_completion_request(
            backend=backend,
            request=_request(0, 0.5),
            signing_key_file=str(_signing_key_file(tmp_path)),
            attempt_input_manifest_sha256="c" * 64,
        )


class _MainBackend:
    def __init__(self, evidence_dir: Path):
        self.evidence_dir = evidence_dir
        self.session_id = "persistent-session-main"
        self.stage_id = "d" * 64
        self.capture_calls: list[dict] = []
        self.baseline_attestation = None

    def initial_policy_state(self):
        return {"left_leg": [0.0] * 6}

    def capture_episode_baseline(self, *, task_success_contract, attempt_id, launch_nonce):
        self.capture_calls.append(
            {
                "task_success_contract": task_success_contract,
                "attempt_id": attempt_id,
                "launch_nonce": launch_nonce,
            }
        )
        from blueprint_pipeline.task_episode_baseline import build_task_episode_baseline

        return build_task_episode_baseline(
            episode_initial_value=0.0,
            attempt_id=attempt_id,
            launch_nonce=launch_nonce,
            simulator_session_id=self.session_id,
            stage_id=self.stage_id,
            articulation_prim_path="/root/Microwave017/Microwave017_Door",
            task_contract_sha256=canonical_task_contract_sha256(task_success_contract),
            criterion_id="microwave_door_open_angle",
            unit="rad",
            captured_timestamp="1",
        )

    def install_episode_baseline_attestation(self, attestation):
        self.baseline_attestation = dict(attestation)


def _write_main_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    contract_path = tmp_path / "task_success_contract.json"
    contract_path.write_text(json.dumps(CONTRACT, sort_keys=True), encoding="utf-8")
    attempt_path = tmp_path / "attempt_input_manifest.json"
    attempt_path.write_text(
        json.dumps(
            {
                "schema_version": "g1_kitchen_attempt_input_manifest.v1",
                "attempt_id": "run-1-attempt-000001",
                "launch_nonce": "nonce-1",
                "artifacts": {
                    "task_success_contract": {
                        "path": str(contract_path),
                        "sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return attempt_path, contract_path


def _run_main(monkeypatch, tmp_path: Path, *, corrupt_contract_sha: bool = False):
    attempt_path, contract_path = _write_main_fixtures(tmp_path)
    if corrupt_contract_sha:
        manifest = json.loads(attempt_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["task_success_contract"]["sha256"] = "0" * 64
        attempt_path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    backend = _MainBackend(evidence_dir)
    serve_calls: list[dict] = []

    monkeypatch.setattr(backend_module, "create_backend", lambda **kwargs: backend)
    monkeypatch.setattr(service, "serve", lambda **kwargs: serve_calls.append(kwargs))
    monkeypatch.setenv(
        "BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE",
        str(_signing_key_file(tmp_path)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "isaac_persistent_task_executor_service",
            "--stage",
            str(tmp_path / "stage.usd"),
            "--evidence-dir",
            str(evidence_dir),
            "--initial-state-output",
            str(tmp_path / "initial_state.json"),
            "--attempt-input-manifest",
            str(attempt_path),
        ],
    )
    exit_code = service.main()
    return exit_code, backend, serve_calls, attempt_path


def test_main_captures_signed_episode_baseline_before_serving(monkeypatch, tmp_path: Path):
    exit_code, backend, serve_calls, attempt_path = _run_main(monkeypatch, tmp_path)
    assert exit_code == 0
    assert backend.capture_calls == [
        {
            "task_success_contract": CONTRACT,
            "attempt_id": "run-1-attempt-000001",
            "launch_nonce": "nonce-1",
        }
    ]
    attestation = json.loads(
        (backend.evidence_dir / "task_episode_baseline_attestation.json").read_text(
            encoding="utf-8"
        )
    )
    assert attestation["signature_verified"] is True
    assert backend.baseline_attestation == attestation
    assert (backend.evidence_dir / "task_episode_baseline_signature.json").is_file()
    assert (tmp_path / "initial_state.json").is_file()
    assert len(serve_calls) == 1
    assert serve_calls[0]["attempt_input_manifest_sha256"] == hashlib.sha256(
        attempt_path.read_bytes()
    ).hexdigest()


def test_main_blocks_on_task_contract_sha_mismatch(monkeypatch, tmp_path: Path):
    with pytest.raises(SystemExit, match="persistent_isaac_task_contract_sha256_mismatch"):
        _run_main(monkeypatch, tmp_path, corrupt_contract_sha=True)
