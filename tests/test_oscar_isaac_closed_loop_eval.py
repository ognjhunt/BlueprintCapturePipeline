"""Hermetic test for the per-step policy <-> WAM <-> perception closed loop (no GPU).

Validates the loop STRUCTURE the real run depends on: each step the policy acts, a (stubbed)
WAM generates the next observation, and the perception harness runs on it immediately. The real
run swaps the stub WAM for per-step OSCAR-2B and the fixture harness backend for real SAM3/DA3,
along the same code path.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import oscar_isaac_closed_loop_eval as L
from blueprint_pipeline.oscar_wam_command_adapter import DEFAULT_NUM_FRAMES

pytestmark = [pytest.mark.slow, pytest.mark.integration]

Image = pytest.importorskip("PIL.Image")
TASK_COMPLETION_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x04" * 32)
CALIBRATION_REVIEWER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x05" * 32)
LABEL_RUNTIME_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x06" * 32)
LEARNED_POLICY_RUNTIME_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
FK_EXECUTOR_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x08" * 32)
COSMOS3_RUNTIME_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)


@pytest.fixture(autouse=True)
def _trusted_task_completion_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    public_key = TASK_COMPLETION_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        L.SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public_key).hexdigest(),
    )
    for name, key, trusted_env in (
        (
            "LEARNED_POLICY",
            LEARNED_POLICY_RUNTIME_PRIVATE_KEY,
            L.SC3_LEARNED_POLICY_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        ),
        (
            "FK_EXECUTOR",
            FK_EXECUTOR_PRIVATE_KEY,
            L.SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        ),
        (
            "COSMOS3_RUNTIME",
            COSMOS3_RUNTIME_PRIVATE_KEY,
            L.SC3_COSMOS3_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        ),
    ):
        private_path = tmp_path / f"{name.lower()}-private.pem"
        private_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        runtime_public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        monkeypatch.setenv(f"BLUEPRINT_TEST_{name}_PRIVATE_KEY_FILE", str(private_path))
        monkeypatch.setenv(
            trusted_env,
            hashlib.sha256(runtime_public_key).hexdigest(),
        )


def _attest_task_completion(payload: dict, tmp_path: Path, stem: str) -> dict:
    public_key = TASK_COMPLETION_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    signed_payload_sha256 = hashlib.sha256(message).hexdigest()
    report = tmp_path / f"{stem}-task-completion-signature.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": signed_payload_sha256,
                "signer_key_id": "task-evaluator-test",
                "verifier_id": "blueprint-test-verifier",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": "task-evaluator-test",
        "verifier_id": "blueprint-test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(TASK_COMPLETION_PRIVATE_KEY.sign(message)).decode(
            "ascii"
        ),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }


def _write_frame(path: Path, seed: int) -> Path:
    # a non-flat, non-dark frame so the harness frame-quality gate does not reject it
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48))
    pix = img.load()
    for y in range(48):
        for x in range(64):
            pix[x, y] = ((x * 4 + seed) % 256, (y * 5 + seed) % 256, (x + y + seed) % 256)
    img.save(path)
    return path


def _stub_wam(work: Path):
    """A local stand-in WAM: writes the next-observation frame (real PNG) per step."""

    def _generate(current_frame, action, step_index, history):
        out = work / "wam_generated" / f"step_{step_index:04d}.png"
        _write_frame(out, seed=step_index * 17)
        return {"generated_frame_path": str(out)}

    return _generate


def _stub_wam_with_video(work: Path):
    """A WAM stand-in that exposes a generated video artifact for consistency scoring."""

    def _generate(current_frame, action, step_index, history):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        out = work / "wam_generated" / f"step_{step_index:04d}.png"
        _write_frame(out, seed=step_index * 17)
        video = out.with_suffix(".mp4")
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            8.0,
            (320, 256),
        )
        yy, xx = np.indices((256, 320))
        for frame_index in range(4):
            frame = np.zeros((256, 320, 3), dtype=np.uint8)
            frame[..., 0] = (xx + step_index * 17 + frame_index * 7) % 255
            frame[..., 1] = (yy + step_index * 11 + frame_index * 9) % 255
            frame[..., 2] = (xx // 2 + yy // 2 + step_index * 13 + frame_index * 5) % 255
            writer.write(frame)
        writer.release()
        assert video.is_file() and video.stat().st_size > 0
        return {
            "generated_frame_path": str(out),
            "generated_video_path": str(video),
            "wam_backend": "oscar_wam",
        }

    return _generate


def _configure_success_label_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    criterion_ids = [
        "end_effector_reaches_target",
        "target_state_change_visible",
        "robot_caused_target_motion",
    ]
    criterion_ids_sha256 = hashlib.sha256(
        json.dumps(sorted(criterion_ids), separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for index in range(20):
        sample_id = f"calibration-sample-{index:02d}"
        blinded_sample_id = f"blind-{index:02d}"
        evidence = tmp_path / f"{sample_id}.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": "wam_success_label_calibration_sample.v1",
                    "sample_id": sample_id,
                    "blinded_sample_id": blinded_sample_id,
                    "criterion_ids_sha256": criterion_ids_sha256,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        verdict = bool(index % 2)
        rows.append(
            {
                "sample_id": sample_id,
                "blinded_sample_id": blinded_sample_id,
                "source_evidence_artifact": {
                    "path": str(evidence),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                },
                "reported_confidence": 0.95,
                "model_verdict": verdict,
                "blinded_rater_votes": [
                    {"rater_id": "rater-a", "verdict": verdict},
                    {"rater_id": "rater-b", "verdict": verdict},
                ],
                "rater_votes_blinded_to_model": True,
                "model_verdict_recorded_before_adjudication": True,
                "adjudicated_ground_truth": verdict,
                "adjudicator_id": "adjudicator-1",
            }
        )
    dataset_payload = {
        "schema_version": "wam_success_label_calibration_dataset.v1",
        "provider": "fake-generated-video-success",
        "model": "fake-vlm",
        "prompt_template_sha256": "c" * 64,
        "criterion_ids": criterion_ids,
        "registered_blinded_rater_ids": ["rater-a", "rater-b"],
        "blinded_human_labels": True,
        "adjudication_completed": True,
        "inter_rater_agreement": 1.0,
        "rows": rows,
    }
    public_key = CALIBRATION_REVIEWER_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed_bytes = json.dumps(dataset_payload, sort_keys=True, separators=(",", ":")).encode()
    signed_payload_sha256 = hashlib.sha256(signed_bytes).hexdigest()
    report = tmp_path / "success-label-calibration-signature-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
                "signed_payload_sha256": signed_payload_sha256,
                "signer_key_id": "calibration-review-board-test",
                "verifier_id": "blueprint-test-verifier",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dataset_payload["reviewer_attestation"] = {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": "calibration-review-board-test",
        "verifier_id": "blueprint-test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode(),
        "public_key_sha256": hashlib.sha256(public_key).hexdigest(),
        "signature_base64": base64.b64encode(
            CALIBRATION_REVIEWER_PRIVATE_KEY.sign(signed_bytes)
        ).decode(),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }
    dataset = tmp_path / "success-label-calibration-dataset.json"
    dataset.write_text(
        json.dumps(dataset_payload, sort_keys=True),
        encoding="utf-8",
    )
    calibration = tmp_path / "success-label-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": "wam_success_label_calibration.v1",
                "status": "accepted",
                "provider": "fake-generated-video-success",
                "model": "fake-vlm",
                "prompt_template_sha256": "c" * 64,
                "criterion_ids": criterion_ids,
                "confidence_floor": 0.8,
                "calibration_method": ("lowest_registered_threshold_meeting_accuracy_v1"),
                "registered_minimum_confidence_floor": 0.8,
                "minimum_high_confidence_samples": 10,
                "minimum_high_confidence_accuracy": 0.8,
                "calibration_dataset_artifact": {
                    "path": str(dataset),
                    "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_SUCCESS_LABEL_CALIBRATION_ARTIFACT",
        str(calibration),
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_SUCCESS_LABEL_CALIBRATION_SHA256",
        hashlib.sha256(calibration.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_SUCCESS_LABEL_CALIBRATION_TRUSTED_PUBLIC_KEY_SHA256",
        hashlib.sha256(public_key).hexdigest(),
    )
    runtime_private_key_path = tmp_path / "success-label-runtime-private.pem"
    runtime_private_key_path.write_bytes(
        LABEL_RUNTIME_PRIVATE_KEY.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    runtime_public_key = LABEL_RUNTIME_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_SUCCESS_LABEL_RUNTIME_SIGNING_PRIVATE_KEY_FILE",
        str(runtime_private_key_path),
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_SUCCESS_LABEL_RUNTIME_TRUSTED_PUBLIC_KEY_SHA256",
        hashlib.sha256(runtime_public_key).hexdigest(),
    )


def test_learned_policy_command_endpoint_exposes_strict_positive_cli_path(
    tmp_path: Path,
) -> None:
    command = tmp_path / "learned_policy.py"
    command.write_text(
        """
import hashlib, json, os
from pathlib import Path
from blueprint_pipeline.oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation

request = json.loads(Path(os.environ['BLUEPRINT_LEARNED_POLICY_INPUT']).read_text())
root = Path.cwd()
checkpoint = root / 'policy-checkpoint.bin'
model_code = root / 'policy-model-code.py'
checkpoint.write_bytes(b'trusted learned policy checkpoint')
model_code.write_text('trusted learned policy runtime')
checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
model_code_sha256 = hashlib.sha256(model_code.read_bytes()).hexdigest()
action = {
    'action_chunk': [float(request['step_index'])] * 7,
    'not_a_learned_robot_policy_action': False,
    'out_of_distribution_action_projection': False,
}
payload = {
    'status': 'completed',
    'learned_policy_action_proven': True,
    'policy_endpoint_id': 'real-policy-test',
    'runtime_result_id': f"policy-result-{request['step_index']}",
    'checkpoint_sha256': checkpoint_sha256,
    'model_code_sha256': model_code_sha256,
    'checkpoint_artifact': {'path': str(checkpoint), 'sha256': checkpoint_sha256},
    'model_code_artifact': {'path': str(model_code), 'sha256': model_code_sha256},
    'action': action,
}
signed_result = {
    'schema_version': 'sc3_learned_policy_runtime_result.v1',
    'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    'step_index': int(request['step_index']),
    'policy_endpoint_id': payload['policy_endpoint_id'],
    'runtime_result_id': payload['runtime_result_id'],
    'checkpoint_sha256': checkpoint_sha256,
    'model_code_sha256': model_code_sha256,
    'checkpoint_artifact': payload['checkpoint_artifact'],
    'model_code_artifact': payload['model_code_artifact'],
    'action': action,
}
payload['runtime_attestation'] = build_sc3_runtime_attestation(
    signed_result,
    private_key_file=os.environ['BLUEPRINT_TEST_LEARNED_POLICY_PRIVATE_KEY_FILE'],
    report_path=root / 'learned-policy-signature-report.json',
    signer_key_id='learned-policy-runtime-test',
    verifier_id='blueprint-test-verifier',
)
Path(os.environ['BLUEPRINT_LEARNED_POLICY_OUTPUT']).write_text(json.dumps(payload))
""".strip(),
        encoding="utf-8",
    )
    endpoint = L.make_learned_policy_command_endpoint(
        command=f"{sys.executable} {command}",
        work_dir=tmp_path / "endpoint",
    )

    action = endpoint({"frame": "generated.png"}, [], 2)

    assert action["action_chunk"] == [2.0] * 7
    assert action["not_a_learned_robot_policy_action"] is False
    assert len(action["learned_policy_checkpoint_sha256"]) == 64
    assert action["learned_policy_runtime_result_id"] == "policy-result-2"
    with pytest.raises(RuntimeError, match="runtime_result_id_missing_or_replayed"):
        endpoint({"frame": "generated.png"}, [], 2)


def test_learned_policy_command_endpoint_rejects_proxy_action(tmp_path: Path) -> None:
    command = tmp_path / "proxy_policy.py"
    command.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "payload = {",
                "  'status': 'completed',",
                "  'learned_policy_action_proven': True,",
                "  'checkpoint_sha256': 'a' * 64,",
                "  'model_code_sha256': 'b' * 64,",
                "  'action': {",
                "    'action_chunk': [0.0] * 7,",
                "    'not_a_learned_robot_policy_action': True,",
                "    'out_of_distribution_action_projection': False,",
                "  },",
                "}",
                "Path(os.environ['BLUEPRINT_LEARNED_POLICY_OUTPUT']).write_text(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    endpoint = L.make_learned_policy_command_endpoint(
        command=f"{sys.executable} {command}",
        work_dir=tmp_path / "endpoint",
    )

    with pytest.raises(RuntimeError, match="proxy_action_rejected"):
        endpoint({}, [], 1)


def test_cosmos3_per_step_command_consumes_exact_observation_action_and_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        L,
        "validate_checkpoint_attestation",
        lambda payload: {"status": "validated", "blockers": []},
    )
    command = tmp_path / "cosmos3_step.py"
    command.write_text(
        """
import hashlib, json, os
from pathlib import Path
from blueprint_pipeline.oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation

request = json.loads(Path(os.environ['BLUEPRINT_COSMOS3_CLOSED_LOOP_INPUT']).read_text())
root = Path.cwd()
request_sha256 = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
runtime_result_id = f"cosmos-result-{request['step_index']}"
checkpoint_sha256 = 'e' * 64
frame = root / 'generated.png'
frame.write_bytes(Path(request['source_observation_artifact']['path']).read_bytes())
frame_ref = {'path': str(frame), 'sha256': hashlib.sha256(frame.read_bytes()).hexdigest()}
modes = {}
for mode in ('forward_dynamics', 'inverse_dynamics', 'cross_view'):
    artifact = root / f'{mode}.json'
    artifact.write_text(json.dumps({
        'schema_version': 'cosmos3_sc3_mode_output.v1',
        'status': 'completed',
        'mode': mode,
        'runtime_session_id': request['runtime_session_id'],
        'runtime_result_id': runtime_result_id,
        'request_sha256': request_sha256,
        'action_sha256': request['action_sha256'],
        'source_observation_sha256': request['source_observation_artifact']['sha256'],
        'checkpoint_sha256': checkpoint_sha256,
        'mode_result': [float(request['step_index'])],
    }, sort_keys=True))
    modes[mode] = {
        'status': 'completed',
        'fresh_mode_execution_proven': True,
        'artifact': {'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
payload = {
    'schema_version': 'cosmos3_closed_loop_step_output.v1',
    'status': 'completed',
    'fresh_model_command_executed_this_invocation': True,
    'learned_wam_model_ran': True,
    'runtime_session_id': request['runtime_session_id'],
    'runtime_result_id': runtime_result_id,
    'request_sha256': request_sha256,
    'consumed_action_sha256': request['action_sha256'],
    'source_observation_sha256': request['source_observation_artifact']['sha256'],
    'generated_frame_path': str(frame),
    'generated_frame_artifact': frame_ref,
    'mode_outputs': modes,
    'sc3_checkpoint_attestation': {'status': 'attested', 'checkpoint_sha256': checkpoint_sha256},
}
signed_result = {
    'schema_version': 'sc3_cosmos3_runtime_result.v1',
    'runtime_session_id': request['runtime_session_id'],
    'runtime_result_id': runtime_result_id,
    'request_sha256': request_sha256,
    'checkpoint_sha256': checkpoint_sha256,
    'source_observation_sha256': request['source_observation_artifact']['sha256'],
    'action_sha256': request['action_sha256'],
    'generated_frame_artifact': frame_ref,
    'mode_outputs': modes,
}
payload['runtime_attestation'] = build_sc3_runtime_attestation(
    signed_result,
    private_key_file=os.environ['BLUEPRINT_TEST_COSMOS3_RUNTIME_PRIVATE_KEY_FILE'],
    report_path=root / 'cosmos3-signature-report.json',
    signer_key_id='cosmos3-runtime-test',
    verifier_id='blueprint-test-verifier',
)
Path(os.environ['BLUEPRINT_COSMOS3_CLOSED_LOOP_OUTPUT']).write_text(json.dumps(payload))
""".strip(),
        encoding="utf-8",
    )
    source = _write_frame(tmp_path / "cosmos-source.png", 11)
    action = {"policy_action": "move", "action_chunk": [0.1] * 7}
    action_sha256 = L._canonical_sha256(action)

    def project(source_action, step_index):
        assert L._canonical_sha256(source_action) == action_sha256
        return {
            "source_action_sha256": action_sha256,
            "derived_via_controller_fk": True,
            "controller_id": "controller",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "landmarks": [{"name": "wrist", "x": 0.1, "y": 0.0}],
            "generated_robot_state": {
                "source_action_sha256": action_sha256,
                "proxy_or_surrogate": False,
                "joint_positions": [0.1] * 7,
            },
        }

    backend = L.make_cosmos3_per_step_command_wam_backend(
        command=f"{sys.executable} {command}",
        work_dir=tmp_path / "cosmos-backend",
        task_prompt="move the object",
        skeleton_for_action=project,
    )
    result = backend(str(source), action, 1, [action])

    assert result["status"] == "completed"
    assert result["wam_backend"] == "cosmos3_nano_per_step"
    assert set(result["sc3_mode_outputs"]) == {
        "forward_dynamics",
        "inverse_dynamics",
        "cross_view",
    }
    replay = backend(str(source), action, 1, [action])
    assert replay["status"] == "blocked"
    assert "cosmos3_closed_loop_output_contract_invalid" in replay["wam_generation_blockers"]


def test_cosmos3_per_step_command_rejects_invalid_fk_projection_before_model(
    tmp_path: Path,
) -> None:
    command = tmp_path / "must_not_run.py"
    command.write_text(
        "from pathlib import Path\nPath('command-ran').write_text('unsafe')\n",
        encoding="utf-8",
    )
    source = _write_frame(tmp_path / "source.png", 12)
    action = {"policy_action": "move", "action_chunk": [0.1] * 7}
    action_sha256 = L._canonical_sha256(action)

    def invalid_projection(source_action, step_index):
        assert source_action == action
        assert step_index == 1
        return {
            "source_action_sha256": action_sha256,
            "derived_via_controller_fk": False,
            "controller_id": "controller",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "landmarks": [{"name": "wrist", "x": 0.1, "y": 0.0}],
            "generated_robot_state": {
                "source_action_sha256": action_sha256,
                "proxy_or_surrogate": False,
                "joint_positions": [0.1] * 7,
            },
        }

    work_dir = tmp_path / "cosmos-backend"
    backend = L.make_cosmos3_per_step_command_wam_backend(
        command=f"{sys.executable} {command}",
        work_dir=work_dir,
        task_prompt="move the object",
        skeleton_for_action=invalid_projection,
    )

    result = backend(str(source), action, 1, [action])

    assert result["status"] == "blocked"
    assert any(
        blocker.endswith("fresh_action_skeleton_not_derived_via_controller_fk")
        for blocker in result["blockers"]
    )
    assert not (work_dir / "step_0001" / "command-ran").exists()


def test_task_completion_command_evaluator_binds_typed_measurement(
    tmp_path: Path,
) -> None:
    command = tmp_path / "task_completion.py"
    command.write_text(
        "\n".join(
            [
                "import hashlib, json, os",
                "from pathlib import Path",
                "request = json.loads(Path(os.environ['BLUEPRINT_TASK_COMPLETION_INPUT']).read_text())",
                "step = request['step_index']",
                "measurement = {",
                "  'schema_version': 'task_transition_measurement.v1',",
                "  'criterion_id': 'door_angle',",
                "  'observable_transition': 'door_angle_increased',",
                "  'before_value': 0.0, 'after_value': 0.3,",
                "  'unit': 'radian', 'source_step_index': step,",
                "}",
                "artifact = Path.cwd() / 'measurement.json'",
                "artifact.write_text(json.dumps(measurement))",
                "payload = {**measurement, 'status': 'completed', 'tolerance': 0.2, 'passed': True,",
                "  'episode_initial_value': 0.0, 'step_delta': 0.3, 'episode_delta': 0.3,",
                "  'source_action_sha256': hashlib.sha256(json.dumps(request.get('action') or {}, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),",
                "  'simulator_session_id': 'isaac-session-1', 'runtime_result_id': f'task-result-{step}',",
                "  'persistent_simulator_state_applied': True, 'official_controller_action_applied': True,",
                "  'simulator_backend': 'isaac', 'stage_id': 'stage-1',",
                "  'articulation_prim_path': '/World/Microwave017/Microwave017_Door',",
                "  'before_timestamp': '2026-07-10T12:00:00.000Z', 'after_timestamp': '2026-07-10T12:00:00.020Z',",
                "  'evidence_artifacts': [{'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}]}",
                "Path(os.environ['BLUEPRINT_TASK_COMPLETION_OUTPUT']).write_text(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    evaluator = L.make_task_completion_command_evaluator(
        command=f"{sys.executable} {command}",
        work_dir=tmp_path / "task_evaluator",
    )
    result = evaluator(
        {
            "step_index": 2,
            "action": {"action_chunk": [0.1, -0.1]},
            "task_success_contract": {
                "task_kind": "manipulation",
                "criteria": [
                    {
                        "criterion_id": "door_angle",
                        "observable_transition": "door_angle_increased",
                        "comparison": "increase_at_least",
                        "tolerance": 0.2,
                        "unit": "radian",
                    }
                ],
            },
        }
    )
    result = dict(result)
    result["evaluator_attestation"] = _attest_task_completion(
        result,
        tmp_path,
        "command-evaluator",
    )
    validation = L._validate_task_completion_transition(
        completion_result=result,
        task_success_contract={
            "criteria": [
                {
                    "criterion_id": "door_angle",
                    "observable_transition": "door_angle_increased",
                    "comparison": "increase_at_least",
                    "tolerance": 0.2,
                    "unit": "radian",
                }
            ]
        },
        expected_source_step_index=2,
    )
    assert validation["registered_transition_passed"] is True


def test_cli_manipulation_completion_requires_contract_and_evaluator(
    tmp_path: Path,
) -> None:
    seed = _write_frame(tmp_path / "completion-seed.png", 7)
    route = tmp_path / "completion-route.json"
    route.write_text(
        json.dumps({"route_points": [[0, 0, 0.79], [1, 0, 0.79]]}),
        encoding="utf-8",
    )
    output = tmp_path / "completion-cli"

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(output),
            "--dry-run",
            "--stop-on-task-completion",
            "--perception-target-prompt",
            "open the door",
        ]
    )

    assert exit_code == 2
    readiness = json.loads(
        (output / "closed_loop_wam_backend_readiness.json").read_text(encoding="utf-8")
    )
    assert "blocked_manipulation_completion_requires_task_success_contract" in readiness["blockers"]
    assert (
        "blocked_manipulation_completion_requires_task_completion_command" in readiness["blockers"]
    )


def _write_episode_consistency_command(tmp_path: Path, *, inverse_consistent: bool) -> Path:
    command = tmp_path / (
        "wam_consistency_pass.py" if inverse_consistent else "wam_consistency_fail.py"
    )
    command.write_text(
        f"""
import hashlib
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_episode_consistency_request.v2"
assert request["claim_boundary"]["scorer_is_separate_from_wam_execution_and_evaluator"] is True
rollout = request["rollouts"][0]
strict = request["strict_action_aware_consistency"]
recovered = list(strict["commanded_action_vector"])
recovered_sha256 = hashlib.sha256(
    json.dumps(recovered, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
payload = {{
    "schema_version": "wam_episode_consistency.command.v1",
    "status": "completed",
    "provider": "fake-vlm-episode-consistency",
    "model": "fake-vlm",
    "rollout_checks": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "model_candidate": rollout.get("model_candidate"),
            "forward_consistent": True,
            "inverse_consistent": {inverse_consistent!r},
            "confidence": 0.91,
            "rationale": "The rollout was checked against action trace context.",
            "visual_evidence_used": True,
            "action_trace_evidence_used": True,
            "commanded_action_sha256": strict["commanded_action_sha256"],
            "recovered_action": recovered,
            "recovered_action_sha256": recovered_sha256,
            "per_dimension_error": [0.0] * len(recovered),
            "per_dimension_uncertainty": [0.01] * len(recovered),
            "calibration_identity": {{"calibration_id": "test-calibration", "sha256": "d" * 64}},
            "threshold": {{"max_abs_error": 0.05, "unit": strict["action_unit"]}},
            "action_timing": strict["action_timing"],
            "action_units": strict["action_units"],
            "controller_fk_state_sha256": strict["controller_fk_state_sha256"],
            "generated_state_sha256": strict["generated_state_sha256"],
            "generated_motion_sha256": strict["generated_motion_sha256"],
            "scorer_runtime_id": "test-scorer-runtime-1",
            "provider_output_replay_used": False,
            "forward_result": {{"passed": True, "method": "test-forward-model"}},
            "inverse_result": {{"passed": {inverse_consistent!r}, "method": "test-inverse-model"}},
            "evidence_refs": [rollout["generated_video_path"]],
            "termination_chunk": {{
                "step_index": strict["action_timing"]["step_index"],
                "commanded_action_sha256": strict["commanded_action_sha256"],
                "generated_motion_sha256": strict["generated_motion_sha256"],
            }},
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return command


def _write_success_label_command(tmp_path: Path, *, success: bool) -> Path:
    command = tmp_path / ("wam_success_pass.py" if success else "wam_success_fail.py")
    command.write_text(
        f"""
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_success_label_request.v1"
assert request["sim_only_constraint"]["real_world_data_allowed"] == "site_capture_only"
assert request["claim_boundary"]["judge_input_is_generated_video_not_raw_robot_evidence"] is True
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_success_labels.command.v1",
    "status": "completed",
    "provider": "fake-generated-video-success",
    "model": "fake-vlm",
    "prompt_template_sha256": "c" * 64,
    "labels": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "success": {success!r},
            "confidence": 0.93,
            "rationale": "The generated video was reviewed against the task prompt.",
            "task_completion_evidence": ["visible robot-caused target state change"]
            if {success!r}
            else [],
            "criterion_results": [
                {{
                    "criterion_id": "end_effector_reaches_target",
                    "passed": {success!r},
                    "evidence_refs": [
                        row["generated_video_path"]
                        for row in rollout["ordered_step_videos"]
                    ],
                }},
                {{
                    "criterion_id": "target_state_change_visible",
                    "passed": {success!r},
                    "evidence_refs": [
                        row["generated_video_path"]
                        for row in rollout["ordered_step_videos"]
                    ],
                }},
                {{
                    "criterion_id": "robot_caused_target_motion",
                    "passed": {success!r},
                    "evidence_refs": [
                        row["generated_video_path"]
                        for row in rollout["ordered_step_videos"]
                    ],
                }}
            ],
            "failure_modes": []
            if {success!r}
            else ["target_state_change_not_visible"],
            "visual_evidence_used": True,
        }}
    ],
}}
from blueprint_pipeline.wam_generated_video_success_label_gemini import attach_success_label_runtime_attestation
output_path = Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT"])
payload = attach_success_label_runtime_attestation(
    payload,
    inference_input_manifest_sha256=request["inference_input_manifest_sha256"],
    output_dir=output_path.parent,
)
output_path.write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return command


def test_strict_action_consistency_rejects_boolean_only_scorer_response() -> None:
    rollout = {
        "rollout_id": "r1",
        "scenario_eval_run_id": "s1",
        "policy_id": "p1",
        "generated_video_path": "/tmp/video.mp4",
    }
    result = L._normalize_wam_episode_consistency(
        command_payload={
            "status": "completed",
            "rollout_checks": [
                {
                    "rollout_id": "r1",
                    "forward_consistent": True,
                    "inverse_consistent": True,
                    "visual_evidence_used": True,
                    "action_trace_evidence_used": True,
                }
            ],
        },
        rollouts=[rollout],
        generated_at="now",
        action_conditioned_video_rollout_generated=True,
        action_conditioned_video_rollout_available=True,
        provider_output_replay_used=False,
        success_label_generated=False,
        visual_smoke_status="passed_visual_quality_smoke",
        visual_rollout_useful=True,
        strict_action_contract={
            "commanded_action_sha256": "a" * 64,
            "commanded_action_vector": [0.1, -0.2],
            "action_dimension": 2,
            "action_unit": "per_dimension",
            "action_units": ["latent", "latent"],
            "action_timing": {
                "step_index": 1,
                "sim_time_s": 0.0,
                "control_hz": 50.0,
                "sample_period_seconds": 0.02,
                "unit": "s",
            },
            "controller_fk_state_sha256": "b" * 64,
            "generated_state_sha256": "c" * 64,
            "generated_motion_sha256": "d" * 64,
        },
    )
    assert result["forward_inverse_consistency_proven"] is False
    assert "wam_consistency_recovered_action_missing_wrong_dim_or_nonfinite" in result[
        "what_is_needed_to_make_forward_inverse_consistency_true"
    ]
    assert result["rollout_checks"][0]["strict_action_aware_contract_passed"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        ("wrong_dimension", "wam_consistency_recovered_action_missing_wrong_dim_or_nonfinite"),
        ("nonfinite_error", "wam_consistency_per_dimension_error_missing_wrong_dim_or_nonfinite"),
        ("wrong_timing", "wam_consistency_action_timing_missing_or_invalid"),
        ("wrong_unit", "wam_consistency_action_units_missing_or_mismatch"),
    ],
)
def test_strict_action_consistency_rejects_malformed_numeric_contract(
    mutation: str,
    expected_blocker: str,
) -> None:
    contract = {
        "commanded_action_sha256": "a" * 64,
        "commanded_action_vector": [0.1, -0.2],
        "action_dimension": 2,
        "action_unit": "per_dimension",
        "action_units": ["latent", "latent"],
        "action_timing": {
            "step_index": 1,
            "sim_time_s": 0.02,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "unit": "s",
        },
        "controller_fk_state_sha256": "c" * 64,
        "generated_state_sha256": "d" * 64,
        "generated_motion_sha256": "e" * 64,
    }
    recovered = [0.1, -0.2]
    check = {
        "rollout_id": "r1",
        "forward_consistent": True,
        "inverse_consistent": True,
        "visual_evidence_used": True,
        "action_trace_evidence_used": True,
        "commanded_action_sha256": "a" * 64,
        "recovered_action": recovered,
        "recovered_action_sha256": hashlib.sha256(
            json.dumps(recovered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "per_dimension_error": [0.0, 0.0],
        "per_dimension_uncertainty": [0.01, 0.01],
        "calibration_identity": {"calibration_id": "cal", "sha256": "b" * 64},
        "threshold": {"max_abs_error": 0.05, "unit": "per_dimension"},
        "action_timing": contract["action_timing"],
        "action_units": contract["action_units"],
        "controller_fk_state_sha256": contract["controller_fk_state_sha256"],
        "generated_state_sha256": contract["generated_state_sha256"],
        "generated_motion_sha256": contract["generated_motion_sha256"],
        "scorer_runtime_id": "scorer-runtime-1",
        "provider_output_replay_used": False,
        "forward_result": {"passed": True, "method": "forward-model"},
        "inverse_result": {"passed": True, "method": "inverse-model"},
        "evidence_refs": ["video.mp4", "action.json"],
        "termination_chunk": {
            "step_index": 1,
            "commanded_action_sha256": "a" * 64,
            "generated_motion_sha256": contract["generated_motion_sha256"],
        },
    }
    if mutation == "wrong_dimension":
        check["recovered_action"] = [0.1]
    elif mutation == "nonfinite_error":
        check["per_dimension_error"] = [0.0, float("nan")]
    elif mutation == "wrong_timing":
        check["action_timing"] = {
            **contract["action_timing"],
            "sim_time_s": 0.03,
        }
    elif mutation == "wrong_unit":
        check["action_units"] = ["radians", "radians"]

    result = L._normalize_wam_episode_consistency(
        command_payload={"status": "completed", "rollout_checks": [check]},
        rollouts=[{"rollout_id": "r1", "generated_video_path": "video.mp4"}],
        generated_at="now",
        action_conditioned_video_rollout_generated=True,
        action_conditioned_video_rollout_available=True,
        provider_output_replay_used=False,
        success_label_generated=False,
        visual_smoke_status="passed_visual_quality_smoke",
        visual_rollout_useful=True,
        strict_action_contract=contract,
    )
    assert result["forward_inverse_consistency_proven"] is False
    assert expected_blocker in result["what_is_needed_to_make_forward_inverse_consistency_true"]


def _passed_visual_smoke(**kwargs):
    return {
        "schema_version": "wam_generated_rollout_visual_smoke.v1",
        "generated_at": "now",
        "status": "passed_visual_quality_smoke",
        "blockers": [],
        "rollout_count": len(kwargs.get("rollouts") or []),
        "claim_boundary": {
            "visual_rollout_useful_for_task_success_review": True,
            "generated_video_is_not_task_success_proof": True,
        },
    }


def _write_seed_geometry_route(tmp_path: Path) -> tuple[Path, Path]:
    render_dir = tmp_path / "render"
    source_render = render_dir / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=18)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=19)
    (render_dir / "manipulation_pov_geometry.json").write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [64, 48]},
                        "target_projection": {"available": True, "u_px": 50, "v_px": 24},
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 18,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "left_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 22,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "left_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 26,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "left_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 30,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                            {
                                "landmark_id": "right_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 46,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "right_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 42,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "right_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 38,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "right_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 34,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                        ],
                        "segments": [
                            {"from": "left_shoulder", "to": "left_elbow"},
                            {"from": "left_elbow", "to": "left_wrist"},
                            {"from": "left_wrist", "to": "left_hand"},
                            {"from": "right_shoulder", "to": "right_elbow"},
                            {"from": "right_elbow", "to": "right_wrist"},
                            {"from": "right_wrist", "to": "right_hand"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_trace = render_dir / "trace.jsonl"
    source_trace.write_text("{}\n", encoding="utf-8")
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
                "source_trace": str(source_trace),
            }
        ),
        encoding="utf-8",
    )
    return seed, route


def _write_passed_short_visual_sanity_manifest(root: Path, policy_observation_path: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_qa = root / "source_policy_observation_visual_qa.json"
    report = root / "wam_rollout_visual_quality_report.json"
    contact_sheet = _write_frame(root / "wam_rollout_contact_sheet.jpg", seed=91)
    video_status = root / "video_review_status.json"
    review_video = root / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
    review_video.parent.mkdir(parents=True, exist_ok=True)
    source_qa.write_text(
        json.dumps({"status": "passed_visual_quality_gate"}),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "review_quality",
                "visual_success": True,
                "profile_contract": {
                    "review_quality_profile": True,
                    "review_quality_minimum_satisfied": True,
                    "smoke_only": False,
                },
            }
        ),
        encoding="utf-8",
    )
    ffprobe_metadata = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "15/1",
                "nb_frames": "24",
            }
        ],
        "format": {"duration": "1.6", "size": "1000"},
    }
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": ffprobe_metadata,
            }
        ),
        encoding="utf-8",
    )
    review_video.write_bytes(b"mp4")
    task_success_judge = root / "persistent_wam_task_success_judge.json"
    task_success_judge.write_text(
        json.dumps(
            {
                "schema_version": "persistent_wam_task_success_judge.v1",
                "generated_at": "now",
                "status": "not_proven",
                "answer": "not_proven",
                "task_success_proven": False,
                "true_manipulation_success_proven": False,
                "blockers": ["true_manipulation_success_not_proven"],
                "generated_video_semantic_success_label": {
                    "status": "requires_review",
                    "success_label_from_generated_video": False,
                    "generated_video_semantic_success": False,
                    "generated_video_semantic_failure": False,
                },
                "claim_boundary": {
                    "visual_quality_is_not_task_success": True,
                    "generated_video_semantic_label_is_support_only": True,
                    "task_success_proof_requires_evaluator_or_physics_state": True,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "persistent_wam_short_visual_sanity_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "persistent_wam_short_visual_sanity.v1",
                "generated_at": "now",
                "status": "passed_short_visual_sanity",
                "short_visual_sanity_passed": True,
                "policy_observation_path": str(policy_observation_path.resolve()),
                "provider": "vast",
                "requested_transition_count": 2,
                "requested_loop_step_count": 3,
                "generated_transition_count": 2,
                "visual_profile": "review_quality",
                "claim_boundary": {
                    "short_visual_sanity_is_not_task_success_proof": True,
                    "visual_sanity_passed_is_not_task_success": True,
                    "visual_quality_is_not_task_success": True,
                    "task_success_judge_required_for_task_success_claim": True,
                    "generated_video_success_label_is_support_only": True,
                },
                "task_success_judge_path": str(task_success_judge),
                "task_success_judge_status": "not_proven",
                "task_success_proven": False,
                "true_manipulation_success_proven": False,
                "generated_video_task_success_label_from_generated_video": False,
                "generated_video_task_success_label_status": "requires_review",
                "generated_video_semantic_success": False,
                "generated_video_semantic_failure": False,
                "task_success_blockers": ["true_manipulation_success_not_proven"],
                "source_policy_observation_visual_qa_status": ("passed_visual_quality_gate"),
                "source_policy_observation_visual_qa_path": str(source_qa),
                "wam_rollout_visual_success": True,
                "wam_rollout_visual_quality_report_path": str(report),
                "wam_rollout_contact_sheet_path": str(contact_sheet),
                "video_review_status_path": str(video_status),
                "review_video_path": str(review_video),
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": ffprobe_metadata,
                "live_wam_generation_success_count": 2,
                "learned_wam_model_success_count": 2,
                "structural_fallback_used": False,
                "paid_provider": {
                    "provider": "vast",
                    "used": False,
                    "teardown_status": "not_required_no_paid_provider",
                    "teardown_performed": False,
                    "continuing_spend_from_this_run": False,
                },
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _allow_vast_paid_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-vast-test-key\n", encoding="utf-8")
    budget = tmp_path / "fresh_vast_budget.json"
    budget.write_text(
        json.dumps({"schema_version": "vast_session_cost_summary.v4", "attempts": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.25")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "10")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "30")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.50")
    return key_file


def test_closed_loop_runs_policy_wam_harness_per_step(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=3)
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["loop_kind"] == "per_step_policy_wam_perception_closed_loop"
    assert manifest["steps_executed"] == 4
    assert manifest["real_perception_backend_used"] is False  # fixture in this hermetic test

    # the trace proves per-step: policy action -> WAM frame -> harness ran, each step
    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace) == 4
    assert [r["step_index"] for r in trace] == [1, 2, 3, 4]
    for row in trace:
        assert row["policy_action"]  # policy acted
        assert Path(row["wam_generated_frame"]).is_file()  # WAM produced the next obs
        assert row["harness_step_status"] == "completed"  # harness ran on it
    # feed-forward: step 2's source observation is step 1's generated frame
    assert trace[1]["source_observation_frame"] == trace[0]["wam_generated_frame"]


def test_closed_loop_requeries_learned_policy_on_wam_observation(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=5)
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    endpoint_calls: list[dict[str, object]] = []
    endpoint_actions: list[dict[str, object]] = []

    def _policy_endpoint(obs, history, step_index):
        frame_path = Path(str(obs.get("camera_frame_path")))
        nested_visual = obs.get("visual_observation") or {}
        assert str(frame_path) == str(nested_visual.get("camera_frame_path"))
        red_channel = Image.open(frame_path).getpixel((0, 0))[0]
        action = {
            "root_position": [round(red_channel / 100.0, 6), round(step_index / 10.0, 6), 0.79],
            "root_yaw_radians": round(red_channel / 1000.0, 6),
            "policy_action": "learned_policy_action",
        }
        endpoint_calls.append(
            {
                "step_index": step_index,
                "camera_frame_path": str(frame_path),
                "history": list(history),
            }
        )
        endpoint_actions.append(action)
        return action

    base_wam = _stub_wam(tmp_path)

    def _action_conditioned_wam(current_frame, action, step_index, history):
        output = base_wam(current_frame, action, step_index, history)
        action_sha256 = L._canonical_sha256(action)
        root_position = action.get("root_position") or [0.0, 0.0, 0.0]
        projection = L._with_action_conditioning_digests(
            {
                "landmarks": [
                    {
                        "name": "wrist",
                        "x": float(root_position[0]),
                        "y": float(root_position[1]),
                    }
                ],
                "derived_via_controller_fk": True,
                "source_action_sha256": action_sha256,
                "controller_id": "unit-test-controller",
                "controller_sha256": "a" * 64,
                "robot_model_sha256": "b" * 64,
                "generated_robot_state": {
                    "source_action_sha256": action_sha256,
                    "proxy_or_surrogate": False,
                    "joint_positions": [float(value) for value in root_position],
                },
            }
        )
        return {
            **output,
            "skeleton_conditioning": projection,
            "generated_robot_state": projection["generated_robot_state"],
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_action_conditioned_wam,
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        policy_endpoint=_policy_endpoint,
    )

    assert manifest["status"] == "completed"
    assert manifest["proof"]["simulator_backend"] == "isaac"
    assert manifest["proof"]["learned_policy_requery_count"] == 3
    assert manifest["proof"]["policy_action_changed_count"] >= 2
    assert manifest["proof"]["policy_observes_wam_generated_next_observation"] is True
    assert not any(str(key).startswith("unitree_") for key in manifest["proof"])

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(endpoint_calls) == manifest["steps_executed"] == 3
    assert endpoint_calls[0]["camera_frame_path"] == str(start.resolve())
    for call, prior_row in zip(endpoint_calls[1:], trace[:-1]):
        assert call["camera_frame_path"] == prior_row["wam_generated_frame"]
    assert endpoint_actions[0]["root_position"] != endpoint_actions[1]["root_position"]
    assert trace[0]["policy_action_from_wam_requery"] is True
    assert trace[0]["policy_requeried_fresh"] is True
    assert trace[0]["policy_action_source"].endswith("initial_real_observation")
    assert trace[1]["policy_action_from_wam_requery"] is True
    assert trace[0]["root_position"] == endpoint_actions[0]["root_position"]
    assert trace[1]["root_position"] == endpoint_actions[1]["root_position"]
    assert trace[2]["root_position"] == endpoint_actions[2]["root_position"]
    assert all(row["policy_requeried_on_wam_observation"] is True for row in trace[:-1])
    assert all(row["requery_status"] == "completed" for row in trace[:-1])


def test_completion_does_not_overclaim_without_learned_requery(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=8)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 4
    assert manifest["policy_observes_wam_generated_next_observation"] is False
    assert manifest["wam_evaluator_in_control_loop"] is True
    assert manifest["clean_frame_reanchoring"] == {
        "enabled": False,
        "interval_steps": None,
        "source_frame_kind": "initial_policy_observation_clean_frame",
    }
    assert manifest["clean_frame_reanchor_event_count"] == 0
    assert manifest["periodic_clean_frame_reanchoring_used"] is False
    assert manifest["proof"]["fresh_learned_policy_requery_steps"] == 0
    assert manifest["proof"]["policy_observes_wam_generated_next_observation"] is False

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert trace[1]["source_observation_frame"] == trace[0]["wam_generated_frame"]
    assert all(row["clean_frame_reanchor_applied"] is False for row in trace)


def test_require_fresh_learned_policy_requery_blocks_deterministic_and_passes_with_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=9)
    route = [(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)]
    blocked = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "blocked",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        require_fresh_learned_policy_requery=True,
    )
    assert blocked["status"] == "blocked"
    assert "fresh_learned_policy_requery_not_proven" in blocked["blockers"]

    original_action_record = L.action_record

    def _fresh_action_record(**kwargs):
        record = original_action_record(**kwargs)
        record["policy_requeried_on_generated_observation"] = True
        record["policy_action_source"] = "test_fresh_learned_policy_requery"
        return record

    monkeypatch.setattr(L, "action_record", _fresh_action_record)
    completed = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "completed",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        require_fresh_learned_policy_requery=True,
    )
    assert completed["status"] == "completed"
    assert completed["proof"]["fresh_learned_policy_requery_steps"] == 3
    assert completed["policy_observes_wam_generated_next_observation"] is True
    assert completed["wam_evaluator_in_control_loop"] is True


def test_clean_frame_reanchoring_feeds_seed_frame_back(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=10)
    route = [(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)]
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "reanchor",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
        clean_frame_reanchor_interval=2,
    )
    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_start = str(start.resolve())
    assert manifest["status"] == "completed"
    assert manifest["clean_frame_reanchor_event_count"] == 2
    assert manifest["periodic_clean_frame_reanchoring_used"] is True
    assert manifest["clean_frame_reanchoring"]["enabled"] is True
    assert manifest["clean_frame_reanchoring"]["interval_steps"] == 2
    assert [event["step_index"] for event in manifest["clean_frame_reanchor_events"]] == [2, 4]
    assert (
        manifest["clean_frame_reanchor_events"][0]["next_policy_observation_frame_path"]
        == resolved_start
    )
    assert trace[1]["clean_frame_reanchor_applied"] is True
    assert trace[2]["source_observation_frame"] == resolved_start
    assert manifest["proof"]["feed_forward_verified"] is True

    default = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "default",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        clean_frame_reanchor_interval=0,
    )
    default_trace = [
        json.loads(line)
        for line in Path(default["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert default["clean_frame_reanchoring"]["enabled"] is False
    assert default["clean_frame_reanchor_event_count"] == 0
    assert default_trace[1]["source_observation_frame"] == default_trace[0]["wam_generated_frame"]


def test_closed_loop_emits_in_process_success_evaluator_not_proven_by_default(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=11)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    judge_path = Path(manifest["manipulation_success_evaluator_results_path"])
    assert judge_path.is_file()
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    assert judge["schema_version"] == "isaac_manipulation_success_evaluator_results.v1"
    assert judge["simulator_backend"] == "isaac"
    assert judge["success_proof_separate_from_structural_loop_proof"] is True
    assert "mujoco" not in json.dumps(judge).lower()
    assert "vast" not in json.dumps(judge).lower()
    assert manifest["status"] == "completed"
    assert manifest["task_target_reached"] is True
    assert judge["answer"] == "not_proven"
    assert judge["manipulation_success_proven"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["success_proof_separate_from_structural_loop_proof"] is True
    assert manifest["success_proof"]["structural_loop_completed"] is True
    assert "feed_forward_verified" in manifest["proof"]
    assert "external judge" not in manifest["claim_boundary"]
    assert "manipulation_success_evaluator" in manifest["claim_boundary"]


def test_closed_loop_external_episode_consistency_stays_separate_from_task_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=12)
    command = _write_episode_consistency_command(tmp_path, inverse_consistent=True)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        wam_consistency_command=f"{sys.executable} {command}",
        allow_wam_consistency_scoring=True,
        wam_consistency_timeout_seconds=5.0,
    )

    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 2
    assert manifest["external_episode_consistency_scorer_ran"] is True
    assert manifest["forward_inverse_consistency_proven"] is True
    assert manifest["wam_episode_consistency_early_termination_recommended"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["answer"] == "not_proven"
    assert "task success" in manifest["claim_boundary"]

    for request_path in manifest["wam_episode_consistency_request_paths"]:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert request["schema_version"] == "wam_episode_consistency_request.v2"
        assert request["rollouts"][0]["generated_video_path"].endswith(".mp4")
    checks = json.loads(
        Path(manifest["wam_consistency_checks_paths"][0]).read_text(encoding="utf-8")
    )
    assert checks["forward_inverse_consistency_proven"] is True
    assert (
        checks["claim_boundary"]["forward_inverse_consistency_does_not_prove_task_success"] is True
    )

    judge = json.loads(
        Path(manifest["manipulation_success_evaluator_results_path"]).read_text(encoding="utf-8")
    )
    assert judge["manipulation_success_proven"] is False


def test_closed_loop_generated_video_success_label_is_sim_only_and_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=31)
    consistency_command = _write_episode_consistency_command(tmp_path, inverse_consistent=True)
    success_command = _write_success_label_command(tmp_path, success=True)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING", "true")
    _configure_success_label_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        perception_target_prompts=["open the fridge"],
        wam_consistency_command=f"{sys.executable} {consistency_command}",
        allow_wam_consistency_scoring=True,
        require_forward_inverse_consistency=True,
        wam_success_label_command=f"{sys.executable} {success_command}",
        allow_wam_success_labeling=True,
        require_generated_video_success_label=True,
        wam_consistency_timeout_seconds=5.0,
        wam_success_label_timeout_seconds=5.0,
    )

    assert manifest["status"] == "completed"
    assert manifest["forward_inverse_consistency_proven"] is True
    assert manifest["generated_video_success_label_passed"] is True
    assert manifest["simulated_manipulation_success_shown"] is True
    assert manifest["real_world_task_success_proven"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["generated_video_success_label_is_sim_only"] is True
    assert manifest["success_proof"]["real_world_task_success_proven"] is False
    assert "generated_video_success_label_not_proven" not in manifest["blockers"]

    request = json.loads(
        Path(manifest["generated_video_success_label_request_path"]).read_text(encoding="utf-8")
    )
    assert request["schema_version"] == "wam_success_label_request.v1"
    assert request["sim_only_constraint"]["real_world_data_allowed"] == "site_capture_only"
    assert request["sim_only_constraint"]["physical_robot_rollout_used"] is False
    assert request["rollouts"][0]["generated_video_path"].endswith(".mp4")
    assert request["task_prompts"][0]["task_prompt"] == "open the fridge"

    labels = json.loads(
        Path(manifest["generated_video_success_labels_path"]).read_text(encoding="utf-8")
    )
    assert labels["status"] == "completed"
    assert labels["labels"][0]["review_task_success"] is True
    assert (
        labels["claim_boundary"]["success_label_does_not_prove_forward_inverse_consistency"] is True
    )


def test_closed_loop_required_generated_video_success_label_blocks_without_labeler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=32)
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        require_generated_video_success_label=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["generated_video_success_label_passed"] is False
    assert manifest["simulated_manipulation_success_shown"] is False
    assert "generated_video_success_label_not_proven" in manifest["blockers"]
    assert "generated_video_success_label:requires_wam_success_review" in manifest["blockers"]
    assert manifest["real_world_task_success_proven"] is False


def test_closed_loop_required_forward_inverse_consistency_blocks_without_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=33)
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        require_forward_inverse_consistency=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["forward_inverse_consistency_proven"] is False
    assert "forward_inverse_consistency_not_proven" in manifest["blockers"]


def test_closed_loop_episode_consistency_failure_early_terminates_feed_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=13)
    command = _write_episode_consistency_command(tmp_path, inverse_consistent=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        wam_consistency_command=f"{sys.executable} {command}",
        allow_wam_consistency_scoring=True,
        wam_consistency_timeout_seconds=5.0,
    )

    assert manifest["status"] == "blocked"
    assert manifest["steps_executed"] == 1
    assert manifest["external_episode_consistency_scorer_ran"] is True
    assert manifest["forward_inverse_consistency_proven"] is False
    assert manifest["wam_episode_consistency_early_termination_recommended"] is True
    assert any(
        blocker == "wam_episode_consistency_step_1:wam_consistency_inverse_not_proven"
        for blocker in manifest["blockers"]
    )
    assert manifest["manipulation_success_proven"] is False

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace) == 1
    assert trace[0]["wam_episode_consistency_early_termination_recommended"] is True
    assert "wam_consistency_inverse_not_proven" in trace[0]["wam_episode_consistency_blockers"]


def test_isaac_manipulation_success_evaluator_requires_registered_transition(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "transition.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "task_transition_measurement.v1",
                "criterion_id": "door_angle",
                "observable_transition": "door_angle_rad_increased",
                "before_value": 0.0,
                "after_value": 0.3,
                "unit": "radian",
                "source_step_index": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proof = {
        "learned_policy_requery_steps": 2,
        "fresh_oscar_provider_model_run_steps": 2,
        "real_perception_backend_steps": 1,
        "registered_task_completion_transition": {
            "criterion_id": "door_angle",
            "registered_transition_passed": True,
            "computed_transition_passed": True,
            "registered_criterion": {
                "criterion_id": "door_angle",
                "observable_transition": "door_angle_rad_increased",
                "comparison": "increase_at_least",
                "tolerance": 0.2,
                "unit": "radian",
            },
            "observable_transition": "door_angle_rad_increased",
            "before_value": 0.0,
            "after_value": 0.3,
            "tolerance": 0.2,
            "unit": "radian",
            "source_step_index": 2,
            "validation_blockers": [],
            "validated_evidence_artifacts": [
                {
                    "path": str(evidence),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    trace_rows = [
        {"policy_action_conditioned_on_wam_generated_observation": True},
        {"policy_action_conditioned_on_wam_generated_observation": True},
    ]
    judge = L.evaluate_isaac_manipulation_success(
        generated_at="now",
        status="completed",
        proof=proof,
        trace_rows=trace_rows,
        task_target_reached=True,
        perception_target_prompts=["open the fridge"],
    )
    assert judge["manipulation_success_proven"] is True
    assert judge["did_target_manipulation_succeed"] is True
    assert judge["answer"] == "yes"
    assert judge["question"] == "open the fridge"
    assert judge["success_proof_separate_from_structural_loop_proof"] is True

    unjudged = L.evaluate_isaac_manipulation_success(
        generated_at="now",
        status="completed",
        proof={
            key: value
            for key, value in proof.items()
            if key != "registered_task_completion_transition"
        },
        trace_rows=trace_rows,
        task_target_reached=True,
        perception_target_prompts=[],
    )
    assert unjudged["manipulation_success_proven"] is False
    assert unjudged["answer"] == "not_proven"
    assert "no task-success signal" in unjudged["reason"]
    assert unjudged["task_target_reached"] is True
    assert unjudged["kinematic_route_reached_is_not_manipulation_success"] is True


def test_closed_loop_blocks_on_missing_wam_frame(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)

    def _bad_wam(current_frame, action, step_index, history):
        return {"generated_frame_path": str(tmp_path / "does_not_exist.png")}

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (2.0, 0.0, 0.79)],
        wam_generate_next=_bad_wam,
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert any("wam_generation_missing_frame" in b for b in manifest["blockers"])


def test_build_oscar_per_step_request_shapes_conditioning(tmp_path: Path) -> None:
    action = {
        "policy_action": "accepted_direct_collision_checked_motion",
        "root_position": [1.0, 2.0, 0.79],
        "root_yaw_radians": 0.5,
    }
    landmarks = [
        {"landmark_id": "pelvis", "image_projection": {"available": True, "u_px": 1, "v_px": 2}}
    ]
    req = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action=action,
        step_index=3,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=landmarks,
        seed=42,
    )
    assert req["reference_frame_path"] == "/frames/cur.png"
    assert req["task_prompt"] == "walk to the sink"
    assert req["num_frames"] == 8
    assert req["seed"] == 45  # base seed + step_index
    assert req["projected_landmark_count"] == 1
    assert req["skeleton_landmarks"] == landmarks
    assert req["output_dir"].endswith("oscar_step_0003")


def test_oscar_per_step_backend_drives_the_loop(tmp_path: Path) -> None:
    """The real GPU path with OSCAR mocked: each step calls per-step OSCAR generation, the
    harness runs on the generated frame. Swapping the mock for a real OSCAR pod + real SAM3
    backend is the only change for the GPU run.
    """
    calls: list[dict] = []

    def _fake_oscar_generate(request):
        calls.append(dict(request))
        frame = tmp_path / "oscar_out" / f"step_{request['step_index']:04d}.png"
        _write_frame(frame, seed=request["step_index"] * 23 + 5)
        return {
            "status": "completed",
            "generated_frame_path": str(frame),
            "generated_video_path": str(frame.with_suffix(".mp4")),
        }

    def _skeleton_for_action(action, step_index):
        return [
            {
                "landmark_id": "pelvis",
                "image_projection": {"available": True, "u_px": step_index, "v_px": 1},
            }
        ]

    backend = L.make_oscar_per_step_wam_backend(
        oscar_generate=_fake_oscar_generate,
        work_dir=tmp_path / "oscar_work",
        task_prompt="walk to the sink",
        num_frames=8,
        skeleton_for_action=_skeleton_for_action,
    )
    start = _write_frame(tmp_path / "start.png", seed=2)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(-4.25, -3.35, 0.79), (1.75, 1.25, 0.79)],
        wam_generate_next=backend,
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
    )
    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 3
    assert len(calls) == 3  # OSCAR called once per step
    # each per-step request carried the step's action + projected skeleton conditioning
    assert calls[0]["task_prompt"] == "walk to the sink"
    assert all(c["projected_landmark_count"] == 1 for c in calls)
    assert [c["step_index"] for c in calls] == [1, 2, 3]


def test_provider_command_backend_writes_step_input_and_extracts_next_frame(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=4)
    captured: dict[str, object] = {}
    captured_input_paths: list[str] = []
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    projected_trace.write_text(
        json.dumps(
            {
                "schema_version": "blueprint.g1.projected_upper_body_skeleton.v1",
                "status": "completed",
                "projected_landmark_count": 1,
                "landmarks": [
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {"available": True, "u_px": 10, "v_px": 12},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_adapter(argv):
        captured["argv"] = list(argv or [])
        input_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"])
        output_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
        captured["input_path"] = str(input_path)
        captured_input_paths.append(str(input_path))
        captured["runtime_env"] = {
            "num_frames": os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES"),
            "num_steps": os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS"),
            "guidance": os.environ.get("BLUEPRINT_OSCAR_WAM_GUIDANCE"),
            "seed": os.environ.get("BLUEPRINT_OSCAR_WAM_SEED"),
            "height": os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT"),
            "width": os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH"),
            "fps": os.environ.get("BLUEPRINT_OSCAR_WAM_FPS"),
        }
        video = output_path.parent / "oscar_generated_rollout.mp4"
        video.write_bytes(b"fake mp4")
        payload = {
            "status": "completed",
            "fresh_provider_model_run_claimed": True,
            "provider_learned_wam_model_ran": True,
            "provider_generated_video_is_model_output": True,
            "rollouts": [{"generated_video_path": str(video)}],
            "blockers": [],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract(video_path, out_dir):
        assert Path(video_path).is_file()
        return _write_frame(Path(out_dir) / "next.png", seed=19)

    backend = L.make_oscar_provider_command_wam_backend(
        work_dir=tmp_path / "provider_loop",
        task_prompt="walk to the sink",
        num_frames=12,
        num_steps=27,
        guidance=4.25,
        seed=100,
        height=240,
        width=320,
        fps=10.0,
        provider="vast",
        allow_paid_provider_launch=True,
        adapter_run=_fake_adapter,
        extract_next_frame=_extract,
        projected_skeleton_trace_path=projected_trace,
    )
    result = backend(
        str(start),
        {
            "policy_action": "accepted_direct_collision_checked_motion",
            "root_position": [0, 0, 0.79],
        },
        1,
        [],
    )

    assert result["status"] == "completed"
    assert result["wam_backend"] == "oscar_2b_per_step_provider"
    assert result["fresh_provider_model_run_claimed"] is True
    assert Path(result["generated_frame_path"]).is_file()
    assert "--allow-paid-provider-launch" in captured["argv"]
    assert captured["runtime_env"] == {
        "num_frames": "12",
        "num_steps": "27",
        "guidance": "4.25",
        "seed": "101",
        "height": "240",
        "width": "320",
        "fps": "10.0",
    }
    step_input = json.loads(Path(str(captured["input_path"])).read_text(encoding="utf-8"))
    assert step_input["schema_version"] == "wam_generation_step_input.v1"
    assert step_input["source_policy_action"]["task_prompt"] == "walk to the sink"
    visual = step_input["current_policy_observation"]["visual_observation"]
    assert visual["g1_projected_skeleton_trace_jsonl"] == str(projected_trace.resolve())

    step2 = backend(
        str(result["generated_frame_path"]),
        {
            "policy_action": "accepted_direct_collision_checked_motion",
            "root_position": [0.25, 0, 0.79],
        },
        2,
        [{"step_index": 1, "wam_generated_frame": result["generated_frame_path"]}],
    )
    assert step2["status"] == "completed"
    step2_input = json.loads(Path(captured_input_paths[-1]).read_text(encoding="utf-8"))
    step2_visual = step2_input["current_policy_observation"]["visual_observation"]
    assert step2_visual["g1_projected_skeleton_trace_jsonl"] == str(projected_trace.resolve())
    assert step2_visual["projected_skeleton_trace_path"] == str(projected_trace.resolve())


def test_provider_command_backend_blocks_failed_visual_smoke_before_extracting_frame(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=41)
    extractor_called = False

    def _fake_adapter(_argv):
        output_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
        video = output_path.parent / "oscar_generated_rollout.mp4"
        video.write_bytes(b"fake mp4")
        payload = {
            "status": "completed",
            "fresh_provider_model_run_claimed": True,
            "provider_learned_wam_model_ran": True,
            "provider_generated_video_is_model_output": True,
            "generated_rollout_visual_smoke_status": "failed_visual_quality_smoke",
            "generated_rollout_visual_quality_blockers": [
                "generated_rollout_later_frames_edge_structure_drift",
                "generated_rollout_later_frames_entropy_drift",
            ],
            "rollouts": [{"generated_video_path": str(video)}],
            "blockers": [],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract(_video_path, _out_dir):
        nonlocal extractor_called
        extractor_called = True
        return None

    backend = L.make_oscar_provider_command_wam_backend(
        work_dir=tmp_path / "provider_loop",
        task_prompt="open the refrigerator",
        adapter_run=_fake_adapter,
        extract_next_frame=_extract,
    )
    result = backend(
        str(start),
        {"policy_action": "accepted_direct_collision_checked_motion"},
        1,
        [],
    )

    assert result["status"] == "blocked"
    assert result["generated_frame_path"] == ""
    assert Path(result["generated_video_path"]).is_file()
    assert extractor_called is False
    assert "provider_generated_rollout_visual_smoke_not_passed" in result["blockers"]
    assert "generated_rollout_later_frames_edge_structure_drift" in result["blockers"]
    assert "generated_rollout_later_frames_entropy_drift" in result["blockers"]


def test_materialize_projected_skeleton_trace_from_seed_geometry_scales_to_seed(
    tmp_path: Path,
) -> None:
    source_render = tmp_path / "render" / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=9)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=10)
    geometry = tmp_path / "render" / "manipulation_pov_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [128, 96]},
                        "target_projection": {
                            "available": True,
                            "u_px": 100,
                            "v_px": 60,
                        },
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_wrist_link",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 64,
                                    "v_px": 48,
                                    "depth_m": 0.2,
                                },
                            },
                            {
                                "landmark_id": "left_hand_link",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 96,
                                    "v_px": 72,
                                    "depth_m": 0.3,
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "render" / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")

    out = L.materialize_projected_skeleton_trace_from_seed_geometry(
        route_payload={"source_trace": str(trace)},
        start_frame_path=seed,
        output_dir=tmp_path / "conditioning",
    )

    assert out is not None and out.is_file()
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == DEFAULT_NUM_FRAMES
    first = rows[0]
    last = rows[-1]
    assert first["image_size_px"] == [64, 48]
    assert first["source_image_size_px"] == [128, 96]
    assert first["target_projection"]["u_px"] == 50.0
    assert first["target_projection"]["v_px"] == 30.0
    assert first["projected_landmark_count"] == 2
    assert first["landmarks"][0]["image_projection"]["u_px"] == 32.0
    assert first["landmarks"][0]["image_projection"]["v_px"] == 24.0
    assert first["segments"] == [{"from": "left_wrist_link", "to": "left_hand_link"}]
    assert last["temporal_progress"] == 1.0
    assert (
        last["landmarks"][1]["image_projection"]["u_px"]
        > first["landmarks"][1]["image_projection"]["u_px"]
    )
    assert (
        last["landmarks"][1]["image_projection"]["v_px"]
        < first["landmarks"][1]["image_projection"]["v_px"]
    )
    assert (
        last["claim_boundary"][
            "temporal_rows_are_target_conditioning_from_resolved_affordance_projection"
        ]
        is True
    )


def _real_backend_command(*, depth_kind: str = "depth_anything_3"):
    code = f"""
import json, os
out = os.environ["BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT"]
payload = {{
  "schema_version": "wam_perception_backend_result.v1",
  "status": "completed",
  "backend": {{
    "kind": "real_provider_probe",
    "status": "completed",
    "real_sam_or_depth_model_ran": True,
    "blockers": [],
    "provider_statuses": [
      {{"provider": "sam3", "ran": True, "blockers": [], "object_count": 1}},
      {{"provider": "depth", "kind": {depth_kind!r}, "ran": True, "blockers": []}}
    ]
  }},
  "objects": [{{"object_id": "sam3_target_0000", "label": "sink", "bbox": [1, 2, 10, 20], "confidence": 0.8}}],
  "depth_estimates": [{{"object_id": "generated_frame", "relative_depth": 0.5, "confidence": 0.7}}],
  "pose_estimates": [],
  "claim_boundary": {{"harness_outputs_are_derived_from_generated_pixels": True}}
}}
open(out, "w", encoding="utf-8").write(json.dumps(payload))
"""
    return [sys.executable, "-c", code]


def test_closed_loop_does_not_consume_blocked_wam_output_with_frame(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=61)

    def _blocked_wam(_current_frame, _action, step_index, _history):
        frame = _write_frame(
            tmp_path / "degraded_wam" / f"step_{step_index:04d}.png",
            seed=step_index * 61,
        )
        return {
            "status": "blocked",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "fresh_provider_model_run_claimed": False,
            "blockers": [
                "provider_generated_rollout_visual_smoke_not_passed",
                "generated_rollout_later_frames_edge_structure_drift",
                "generated_rollout_later_frames_entropy_drift",
            ],
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_blocked_wam,
        steps=2,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["steps_executed"] == 0
    assert (
        "blocked_wam_generation_at_step_1:"
        "provider_generated_rollout_visual_smoke_not_passed" in manifest["blockers"]
    )
    assert (
        "blocked_wam_generation_at_step_1:"
        "generated_rollout_later_frames_edge_structure_drift" in manifest["blockers"]
    )
    assert (
        "blocked_wam_generation_at_step_1:"
        "generated_rollout_later_frames_entropy_drift" in manifest["blockers"]
    )


def test_closed_loop_proof_requirements_pass_with_fresh_oscar_sam3_da3(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=6)

    def _fresh_oscar(current_frame, action, step_index, history):
        frame = _write_frame(
            tmp_path / "oscar" / f"step_{step_index:04d}.png", seed=step_index * 31
        )
        video = tmp_path / "oscar" / f"step_{step_index:04d}.mp4"
        video.write_bytes(b"fake mp4")
        return {
            "status": "completed",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "generated_video_path": str(video),
            "fresh_provider_model_run_claimed": True,
            "provider_payload": {
                "status": "completed",
                "fresh_provider_model_run_claimed": True,
                "provider_learned_wam_model_ran": True,
                "provider_generated_video_is_model_output": True,
            },
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_fresh_oscar,
        steps=2,
        harness_backend_kind="real_provider_probe",
        harness_backend_command=_real_backend_command(depth_kind="depth_anything_3"),
        allow_external_backend=True,
        require_fresh_oscar_provider=True,
        require_real_perception_backend=True,
        require_sam3_completed=True,
        require_da3_completed=True,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["proof"]["fresh_oscar_provider_model_run_steps"] == 2
    assert manifest["proof"]["sam3_completed_steps"] == 2
    assert manifest["proof"]["da3_completed_steps"] == 2
    assert manifest["proof"]["feed_forward_verified"] is True


def test_closed_loop_proof_does_not_count_depth_v2_as_da3(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=7)

    def _fresh_oscar(current_frame, action, step_index, history):
        frame = _write_frame(
            tmp_path / "oscar" / f"step_{step_index:04d}.png", seed=step_index * 33
        )
        return {
            "status": "completed",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "fresh_provider_model_run_claimed": True,
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_fresh_oscar,
        steps=1,
        harness_backend_kind="real_provider_probe",
        harness_backend_command=_real_backend_command(depth_kind="transformers_depth_anything_v2"),
        allow_external_backend=True,
        require_da3_completed=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "da3_provider_not_completed_at_step_1" in manifest["blockers"]
    assert manifest["proof"]["depth_completed_steps"] == 1
    assert manifest["proof"]["da3_completed_steps"] == 0


def test_build_oscar_inference_argv_mirrors_entrypoint(tmp_path: Path) -> None:
    argv = L.build_oscar_inference_argv(
        python="python",
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        first_frame_path="/frames/cur.png",
        prompt="walk to the sink",
        num_frames=8,
        num_steps=35,
        guidance=6.0,
        seed=45,
        height=480,
        width=640,
        fps=15.0,
        output_video=tmp_path / "out.mp4",
        skeleton_video=tmp_path / "skel.mp4",
    )
    assert any("inference_oscar.py" in a for a in argv)
    assert argv[argv.index("--first-frame") + 1] == "/frames/cur.png"
    assert argv[argv.index("--prompt") + 1] == "walk to the sink"
    assert argv[argv.index("--num-frames") + 1] == "8"
    assert argv[argv.index("--skeleton-video") + 1].endswith("skel.mp4")


def test_local_oscar_subprocess_generate_runs_and_extracts(tmp_path: Path) -> None:
    seen_argv: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = "oscar stdout"
        stderr = "oscar stderr"

    def _fake_run(argv, **kwargs):
        seen_argv.append(list(argv))
        # simulate inference_oscar.py writing the output clip
        out = argv[argv.index("--output") + 1]
        Path(out).write_bytes(b"\x00fakeclip")
        return _Done()

    def _fake_extract(video_path: Path, out_dir: Path):
        frame = out_dir / "next_obs.png"
        _write_frame(frame, seed=11)
        return frame

    def _fake_skeleton_video(landmarks, out_dir: Path):
        v = out_dir / "skel.mp4"
        v.write_bytes(b"\x00skel")
        return v

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        run=_fake_run,
        build_skeleton_video=_fake_skeleton_video,
        extract_next_frame=_fake_extract,
    )
    request = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action={"root_position": [1, 2, 0.79]},
        step_index=1,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=[{"landmark_id": "pelvis"}],
    )
    out = gen(request)
    assert out["status"] == "completed"
    assert Path(out["generated_frame_path"]).is_file()
    assert Path(out["stdout_log_path"]).read_text(encoding="utf-8") == "oscar stdout"
    assert Path(out["stderr_log_path"]).read_text(encoding="utf-8") == "oscar stderr"
    assert "--skeleton-video" in seen_argv[0]  # skeleton conditioning passed to OSCAR


def test_local_oscar_subprocess_generate_blocks_on_nonzero(tmp_path: Path) -> None:
    class _Fail:
        returncode = 1
        stdout = "usage"
        stderr = "error: --skeleton-video required"

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/c",
        run=lambda argv, **k: _Fail(),
        extract_next_frame=lambda v, d: None,
    )
    out = gen(
        {
            "output_dir": str(tmp_path),
            "reference_frame_path": "/f.png",
            "task_prompt": "t",
            "num_frames": 8,
            "seed": 1,
        }
    )
    assert out["status"] == "blocked"
    assert any("returncode" in b for b in out["blockers"])
    assert Path(out["stdout_log_path"]).read_text(encoding="utf-8") == "usage"
    assert (
        Path(out["stderr_log_path"]).read_text(encoding="utf-8")
        == "error: --skeleton-video required"
    )


def test_extract_next_observation_selects_earliest_usable_future_frame(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    usable_future = np.zeros((24, 32, 3), dtype=np.uint8)
    usable_future[::2, :] = (235, 235, 235)
    usable_future[:, ::4] = (24, 180, 240)
    dark_late = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, usable_future, dark_late, dark_late):
        writer.write(frame)
    writer.release()

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")
    assert out is not None and out.is_file()
    got = cv2.imread(str(out))
    assert got is not None and got.shape == (24, 32, 3)
    assert int(got.mean()) > 80  # selected frame 1, not the late collapsed dark frame
    selection = json.loads(
        (tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8")
    )
    assert selection["status"] == "completed"
    assert selection["selected_frame_index"] == 1
    assert selection["claim_boundary"]["scene_or_task_specific_pixels_used"] is False


def test_extract_next_observation_blocks_when_future_frames_are_not_useful(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    dark_future = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, dark_future, dark_future):
        writer.write(frame)
    writer.release()

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")

    assert out is None
    selection = json.loads(
        (tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8")
    )
    assert selection["status"] == "blocked"
    assert selection["selected_frame_index"] is None
    assert "no_usable_future_next_observation_frame" in selection["blockers"]
    assert any(
        "next_observation_candidate_too_dark" in candidate["blockers"]
        for candidate in selection["candidates"][1:]
    )


def test_extract_next_observation_blocks_static_noise_future_frame(tmp_path: Path) -> None:
    stats = {
        "mean_luma": 97.0,
        "std_luma": 16.0,
        "luma_range": 143,
        "dark_pixel_ratio": 0.001,
        "bright_pixel_ratio": 0.0,
        "edge_density": 0.203,
    }

    blockers = L._next_observation_signal_blockers(stats)

    assert "next_observation_candidate_static_noise_artifact" in blockers


def test_extract_last_frame_uses_ffmpeg_when_cv2_missing(tmp_path: Path, monkeypatch) -> None:
    import builtins
    import subprocess

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "cv2":
            raise ImportError("cv2 intentionally unavailable")
        return real_import(name, *args, **kwargs)

    def fake_run(argv: list[str], **_kwargs: object):
        Path(argv[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")

    assert out == tmp_path / "extracted" / "next_observation.png"
    assert out.is_file()
    selection = json.loads(
        (tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_frame_index"] == 1


def test_closed_loop_blocks_on_empty_route(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[],
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert "blocked_empty_route" in manifest["blockers"]


def test_closed_loop_wam_backend_readiness_accepts_wired_cosmos3_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "python cosmos3_adapter.py")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="cosmos3_wam",
        use_provider_command=True,
        oscar_repo=None,
        checkpoint=None,
        oscar_provider="vast",
        allow_paid_provider_launch=False,
    )

    assert readiness["status"] == "ready"
    assert readiness["selected_wam_backend"] == "cosmos3_wam"
    assert readiness["explicit_provider_command_configured"] is True
    assert readiness["supported_by_this_runner"] is True
    assert readiness["blockers"] == []
    assert readiness["claim_boundary"]["cosmos3_per_step_command_contract_wired"] is True
    assert (
        readiness["claim_boundary"]["cosmos3_strategy_preference_does_not_imply_runtime_execution"]
        is True
    )


def test_closed_loop_wam_backend_readiness_blocks_unpinned_local_oscar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_SOURCE_URL", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_SOURCE_REF", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_HF_REPO", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_HF_REVISION", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_EXPERIMENTAL_OSCAR_WAM_VERSION", raising=False)
    source = tmp_path / "oscar-source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=False,
        oscar_repo=str(source),
        checkpoint=str(checkpoint),
        oscar_provider="vast",
        allow_paid_provider_launch=False,
    )

    assert readiness["status"] == "blocked"
    assert "official_oscar_source_url_mismatch" in readiness["blockers"]
    assert "official_oscar_source_commit_not_pinned" in readiness["blockers"]
    assert "official_oscar_hf_revision_not_pinned" in readiness["blockers"]
    assert readiness["official_oscar_release"]["official_release_match"] is False
    assert readiness["claim_boundary"]["official_oscar_source_and_checkpoint_pinned"] is False


def test_closed_loop_wam_backend_readiness_surfaces_vast_paid_gate_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", raising=False)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "missing_vast_api_key"))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(tmp_path / "budget.json"))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="vast",
        allow_paid_provider_launch=True,
    )

    assert readiness["status"] == "blocked"
    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_VAST_API_CALLS" in readiness["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH" in readiness["blockers"]
    assert "missing_file_based_secret_VAST_API_KEY_FILE" in readiness["blockers"]
    assert preflight["claim_boundary"]["preflight_does_not_call_vast_api"] is True


def test_closed_loop_wam_backend_readiness_surfaces_vast_session_budget_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "schema_version": "vast_session_cost_summary.v4",
                "attempts": [
                    {
                        "estimated_cost_usd": 0.60,
                        "actual_live_runtime_seconds_observed_by_adapter": 55 * 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.45")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "45")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "50")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.75")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="vast",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert readiness["status"] == "blocked"
    assert preflight["prior_estimated_cost_usd"] == 0.6
    assert preflight["prior_live_runtime_minutes"] == 55.0
    assert "session_live_runtime_limit_exhausted" in readiness["blockers"]
    assert "session_estimated_spend_hard_cap_exhausted" in readiness["blockers"]
    assert preflight["raw_secret_values_recorded"] is False
    assert "redacted-test-key" not in json.dumps(preflight)


def test_closed_loop_cli_writes_no_spend_backend_readiness_for_cosmos3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=13)
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({"route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]]}), encoding="utf-8"
    )
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "python cosmos3_adapter.py")

    exit_code = L.main(
        [
            "--start-frame",
            str(start),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "cosmos3_wam",
            "--use-provider-command",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    readiness_path = tmp_path / "closed_loop" / "closed_loop_wam_backend_readiness.json"
    plan_path = tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "blocked"
    assert readiness["selected_wam_backend"] == "cosmos3_wam"
    assert plan["selected_wam_backend"] == "cosmos3_wam"
    assert plan["wam_backend_readiness_path"] == str(readiness_path)
    assert "blocked_cosmos3_wam_not_wired_into_isaac_closed_loop_runner" not in plan["blockers"]
    assert (
        "blocked_strict_evaluation_requires_action_skeleton_controller_fk_command"
        in plan["blockers"]
    )
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_cli_blocks_paid_multi_step_provider_without_projected_skeleton(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=17)
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({"route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]]}),
        encoding="utf-8",
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(start),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--allow-paid-provider-launch",
            "--steps",
            "2",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    readiness_path = tmp_path / "closed_loop" / "closed_loop_wam_backend_readiness.json"
    plan_path = tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    blocker = "closed_loop_projected_skeleton_trace_missing_for_paid_multi_step_provider_wam"
    assert readiness["status"] == "blocked"
    assert readiness["oscar_provider"] == "vast"
    assert readiness["paid_provider_preflight"]["provider"] == "vast"
    assert plan["oscar_provider"] == "vast"
    assert plan["num_frames_per_step"] == DEFAULT_NUM_FRAMES
    assert plan["oscar_runtime_settings"]["num_frames"] == DEFAULT_NUM_FRAMES
    assert readiness["seed_conditioning_preflight"]["required"] is True
    assert blocker in readiness["blockers"]
    assert blocker in plan["blockers"]
    assert plan["seed_conditioning_preflight"]["projected_skeleton_trace_present"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_cli_dry_run_writes_provider_input_contract_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    render_dir = tmp_path / "render"
    source_render = render_dir / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=18)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=19)
    (render_dir / "manipulation_pov_geometry.json").write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [64, 48]},
                        "target_projection": {"available": True, "u_px": 50, "v_px": 24},
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 18,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "left_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 22,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "left_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 26,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "left_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 30,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                            {
                                "landmark_id": "right_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 46,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "right_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 42,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "right_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 38,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "right_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 34,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                        ],
                        "segments": [
                            {"from": "left_shoulder", "to": "left_elbow"},
                            {"from": "left_elbow", "to": "left_wrist"},
                            {"from": "left_wrist", "to": "left_hand"},
                            {"from": "right_shoulder", "to": "right_elbow"},
                            {"from": "right_elbow", "to": "right_wrist"},
                            {"from": "right_wrist", "to": "right_hand"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_trace = render_dir / "trace.jsonl"
    source_trace.write_text("{}\n", encoding="utf-8")
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
                "source_trace": str(source_trace),
            }
        ),
        encoding="utf-8",
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "2",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    preflight = plan["provider_input_contract_preflight"]
    assert preflight["status"] == "ready"
    assert preflight["contract_status"] == "warning_high_risk"
    assert preflight["autoregressive_risk_level"] == "high"
    assert (
        "projected_skeleton_not_scene_faithful_policy_action_high_risk"
        in preflight["high_risk_flags"]
    )
    assert (
        "projected_skeleton_missing_scene_faithful_policy_action_bridge"
        in preflight["ranking_risk_flags"]
    )
    assert preflight["policy_ranking_claim_safe"] is False
    assert plan["short_visual_sanity_launch_plan"]["status"] == "not_required"
    assert Path(preflight["bundle_manifest_path"]).is_file()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_closed_loop_paid_long_run_requires_short_visual_sanity_after_input_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    gate = plan["short_rollout_sanity_gate"]
    assert gate["status"] == "blocked"
    assert gate["required"] is True
    assert gate["risk_recommends_short_sanity"] is True
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in gate["blockers"]
    assert "short_visual_sanity_manifest_env_missing" in gate["blockers"]
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in plan["blockers"]
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "ready"
    assert launch_plan["required"] is True
    assert launch_plan["provider"] == "vast"
    assert launch_plan["provider_resolution"] == "explicit_provider"
    assert launch_plan["blockers"] == []
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is True
    assert launch_plan["provider_launch_blockers"] == []
    policy_observation_path = Path(launch_plan["policy_observation_path"])
    assert policy_observation_path.is_file()
    policy_observation = json.loads(policy_observation_path.read_text(encoding="utf-8"))
    assert policy_observation["schema_version"] == "blueprint_policy_observation.v1"
    assert policy_observation["task_prompt"] == plan["task_prompt"]
    assert policy_observation["unitree_g1_sonic_state_source"] == (
        "neutral_unitree_g1_sonic_contract_state"
    )
    assert policy_observation["unitree_g1_sonic_state_metadata"]["complete"] is True
    assert (
        policy_observation["unitree_g1_sonic_state_metadata"][
            "scene_or_task_specific_coordinates_hardcoded"
        ]
        is False
    )
    state = policy_observation["unitree_g1_sonic_state"]
    assert {key: len(value) for key, value in state.items()} == {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 7,
        "right_hand": 7,
        "projected_gravity": 3,
    }
    assert state["projected_gravity"] == [0.0, 0.0, -1.0]
    assert policy_observation["visual_observation"]["camera_frame_path"] == str(seed.resolve())
    assert "blueprint_pipeline.persistent_wam_short_visual_sanity" in launch_plan["command_argv"]
    assert (
        launch_plan["command_argv"][launch_plan["command_argv"].index("--transition-count") + 1]
        == "2"
    )
    assert launch_plan["expected_manifest_path"].endswith(
        "persistent_wam_short_visual_sanity_manifest.json"
    )
    assert (
        launch_plan["unlock_env"][L.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV]
        == launch_plan["expected_manifest_path"]
    )
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_short_sanity_manifest_must_match_policy_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)
    stale_observation = tmp_path / "stale_policy_observation.json"
    stale_observation.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_policy_observation.v1",
                "visual_observation": {"camera_frame_path": str(seed.resolve())},
            }
        ),
        encoding="utf-8",
    )
    stale_manifest = _write_passed_short_visual_sanity_manifest(
        tmp_path / "stale_short_sanity", stale_observation
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--short-visual-sanity-manifest",
            str(stale_manifest),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    gate = plan["short_rollout_sanity_gate"]
    assert gate["status"] == "blocked"
    assert "short_visual_sanity_policy_observation_mismatch" in gate["blockers"]
    assert (
        gate["expected_policy_observation_path"]
        == plan["short_visual_sanity_launch_plan"]["policy_observation_path"]
    )
    assert (
        "short_visual_sanity_policy_observation_mismatch"
        in gate["short_visual_sanity_validation"]["blockers"]
    )
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_matching_short_sanity_manifest_unlocks_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)
    output_dir = tmp_path / "closed_loop"

    first_exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(output_dir),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )
    assert first_exit_code == 2
    first_plan = json.loads(
        (output_dir / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    policy_observation_path = Path(
        first_plan["short_visual_sanity_launch_plan"]["policy_observation_path"]
    )
    matching_manifest = _write_passed_short_visual_sanity_manifest(
        tmp_path / "matching_short_sanity", policy_observation_path
    )

    second_exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(output_dir),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--short-visual-sanity-manifest",
            str(matching_manifest),
            "--dry-run",
        ]
    )

    assert second_exit_code == 0
    plan = json.loads(
        (output_dir / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    gate = plan["short_rollout_sanity_gate"]
    assert plan["status"] == "prepared"
    assert gate["status"] == "ready"
    assert gate["short_visual_sanity_manifest_path"] == str(matching_manifest)
    assert gate["expected_policy_observation_path"] == str(policy_observation_path)
    assert gate["short_visual_sanity_validation"]["status"] == "passed_short_visual_sanity"
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" not in plan["blockers"]
    captured = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert captured[-1]["status"] == "prepared"


def test_closed_loop_short_sanity_launch_plan_blocks_vast_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    seed, route = _write_seed_geometry_route(tmp_path)
    monkeypatch.delenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", raising=False)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "missing_vast_api_key"))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(tmp_path / "budget.json"))

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "blocked_provider_authorization"
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is False
    assert launch_plan["provider"] == "vast"
    assert Path(launch_plan["policy_observation_path"]).is_file()
    assert "missing_env_BLUEPRINT_ALLOW_VAST_API_CALLS" in launch_plan["provider_launch_blockers"]
    assert (
        "missing_env_BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"
        in launch_plan["provider_launch_blockers"]
    )
    assert "missing_file_based_secret_VAST_API_KEY_FILE" in launch_plan["blockers"]
    assert launch_plan["paid_provider_preflight"]["status"] == "blocked"
    assert launch_plan["claim_boundary"]["plan_is_no_spend"] is True
    assert len(plan["blockers"]) == len(set(plan["blockers"]))
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_short_sanity_launch_plan_allows_fresh_vast_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    seed, route = _write_seed_geometry_route(tmp_path)
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    budget = tmp_path / "fresh_budget.json"
    budget.write_text(
        json.dumps({"schema_version": "vast_session_cost_summary.v4", "attempts": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.25")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "10")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "30")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.50")

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "ready"
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is True
    assert launch_plan["provider_launch_blockers"] == []
    assert launch_plan["provider"] == "vast"
    assert launch_plan["paid_provider_preflight"]["status"] == "ready"
    assert launch_plan["paid_provider_preflight"]["budget_ledger_present"] is True
    assert launch_plan["paid_provider_preflight"]["attempt_count"] == 0
    assert Path(launch_plan["policy_observation_path"]).is_file()
    assert launch_plan["blockers"] == []
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in plan["blockers"]
    assert "short_visual_sanity_manifest_env_missing" in plan["blockers"]
    assert "redacted-test-key" not in json.dumps(plan)
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_runpod_paid_preflight_ready_with_gates_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RunPod is a first-class paid closed-loop provider when its API/pod-launch
    gates are set, the key file exists, and the projected cost fits the cap."""
    key_file = tmp_path / "runpod_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(key_file))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "ready"
    assert preflight["provider"] == "runpod"
    assert preflight["projected_max_incremental_cost_usd"] <= preflight["hard_cap_usd"]


def test_closed_loop_runpod_paid_preflight_blocks_without_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(tmp_path / "missing_key"))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_RUNPOD_API_CALLS" in preflight["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH" in preflight["blockers"]
    assert "missing_file_based_secret_RUNPOD_API_KEY_FILE" in preflight["blockers"]


def test_closed_loop_runpod_paid_preflight_blocks_over_hard_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "runpod_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_MAX_HOURLY_RATE", "10.0")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_MAX_LIVE_MINUTES", "120")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_HARD_CAP_USD", "3.0")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert any("hard_cap" in b for b in preflight["blockers"])


def test_cli_default_num_frames_is_standard_oscar_clip_length() -> None:
    """A 'default' generation must be the standard 81-frame (5.4s) OSCAR clip.
    The old default of 8 produced ~5-frame unusable rollouts that looked like
    model failure (2026-07-02 sink_faucet incident)."""
    import argparse

    parser_actions = {}
    real_parse = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda self, *a, **k: (
            parser_actions.update({a.dest: a.default for a in self._actions})
            or (_ for _ in ()).throw(SystemExit(0))
        )
        try:
            L.main(["--start-frame", "x", "--route-file", "y", "--output-dir", "z"])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    assert parser_actions.get("num_frames") == 81


def _write_clip(path, frames):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (64, 48))
    for frame in frames:
        writer.write(np.ascontiguousarray(frame))
    writer.release()


def test_generated_clip_coherence_measures_drift(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(7)
    seed = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    # 3 seed-anchored frames (tiny jitter), then pure noise = drift.
    coherent = [
        np.clip(seed.astype(np.int16) + rng.integers(-6, 6, seed.shape), 0, 255).astype(np.uint8)
        for _ in range(3)
    ]
    noise = [rng.integers(0, 255, size=seed.shape, dtype=np.uint8) for _ in range(5)]
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, [seed, *coherent, *noise])

    result = L.generated_clip_coherence(clip)
    assert result["status"] == "measured"
    assert result["frame_count"] == 9
    # The 3 jittered frames stay anchored; the noise tail does not.
    assert 2 <= result["coherent_horizon_frames"] <= 5
    assert result["min_correlation"] < 0.5
    assert result["claim_boundary"]

    missing = L.generated_clip_coherence(tmp_path / "nope.mp4")
    assert missing["status"] == "not_measured"


def test_closed_loop_blocks_on_incoherent_generated_clip(tmp_path):
    pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(11)
    seed_frame = _write_frame(tmp_path / "seed.png", 3)
    noise_clip = tmp_path / "noise.mp4"
    seed_img = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    _write_clip(
        noise_clip,
        [seed_img] + [rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8) for _ in range(6)],
    )

    def wam_with_incoherent_video(frame, action, step, history):
        generated = tmp_path / f"gen_{step}.png"
        _write_frame(generated, 40 + step)
        return {
            "status": "completed",
            "generated_frame_path": str(generated),
            "generated_video_path": str(noise_clip),
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "out",
        start_frame_path=seed_frame,
        route_points=[[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
        wam_generate_next=wam_with_incoherent_video,
        steps=2,
        min_coherent_horizon_frames=2,
    )
    assert manifest["status"] == "blocked"
    assert any(
        blocker.startswith("blocked_generated_clip_coherence_below_floor_at_step_")
        for blocker in manifest["blockers"]
    )
    coherence = manifest["generated_clip_coherence"]
    assert coherence["min_coherent_horizon_frames_required"] == 2
    assert coherence["per_step"][0]["status"] == "measured"

    # Gate disabled (0): the same clips only get recorded, never block.
    relaxed = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "out_relaxed",
        start_frame_path=seed_frame,
        route_points=[[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
        wam_generate_next=wam_with_incoherent_video,
        steps=1,
        min_coherent_horizon_frames=0,
    )
    assert not any(
        blocker.startswith("blocked_generated_clip_coherence")
        for blocker in relaxed.get("blockers", [])
    )
    assert relaxed["generated_clip_coherence"]["per_step"]


def test_cli_blocks_sub_native_oscar_resolution_without_override(tmp_path):
    seed = _write_frame(tmp_path / "seed.png", 5)
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"route_points": [[0, 0, 0.79], [1, 0, 0.79]]}))

    def run_cli(extra):
        out = tmp_path / f"out_{len(extra)}"
        L.main(
            [
                "--start-frame",
                str(seed),
                "--route-file",
                str(route),
                "--steps",
                "1",
                "--output-dir",
                str(out),
                "--dry-run",
                "--oscar-height",
                "240",
                "--oscar-width",
                "320",
                *extra,
            ]
        )
        return json.loads((out / "closed_loop_wam_backend_readiness.json").read_text())

    blocked = run_cli([])
    contract = blocked["oscar_generation_resolution_contract"]
    assert contract["native_match"] is False
    assert contract["requested_height"] == 240
    assert "blocked_non_native_oscar_resolution_requires_explicit_override" in blocked["blockers"]
    assert blocked["status"] == "blocked"

    overridden = run_cli(["--allow-non-native-oscar-resolution"])
    assert (
        "blocked_non_native_oscar_resolution_requires_explicit_override"
        not in overridden["blockers"]
    )
    assert overridden["oscar_generation_resolution_contract"]["override_used"] is True


def _route_walking_wam(tmp_path):
    def wam(frame, action, step, history):
        generated = tmp_path / f"dyn_gen_{step}.png"
        _write_frame(generated, 60 + step)
        return {"status": "completed", "generated_frame_path": str(generated)}

    return wam


def test_episode_ends_when_task_target_reached(tmp_path):
    # Short route: the deterministic walk reaches the target well before the
    # steps cap, so the episode should terminate early with the task reason.
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "dyn_out",
        start_frame_path=_write_frame(tmp_path / "dyn_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=12,
        stop_on_task_completion=True,
    )
    termination = manifest["episode_termination"]
    assert termination["stop_on_task_completion"] is True
    assert termination["task_completed_early"] is True
    assert termination["reason"].startswith("task_criterion_navigation_goal_passed_at_step_")
    assert termination["steps_executed"] < 12
    assert termination["steps_cap"] == 12
    assert manifest["status"] == "completed"
    assert "Navigation smoke" in termination["claim_boundary"]


def test_manipulation_does_not_terminate_on_robot_root_proximity(tmp_path):
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "manipulation_proximity_out",
        start_frame_path=_write_frame(tmp_path / "manipulation_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=3,
        stop_on_task_completion=True,
        perception_target_prompts=["open the refrigerator"],
        task_success_contract={"task_kind": "manipulation", "criterion_id": "door_angle"},
    )

    termination = manifest["episode_termination"]
    assert termination["task_completed_early"] is False
    assert termination["steps_executed"] == 3
    assert termination["task_completion_evidence_status"] == (
        "blocked_missing_registered_task_completion_evaluator"
    )


def test_manipulation_terminates_only_on_registered_observable_transition(tmp_path):
    evidence = tmp_path / "door-state-transition.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "task_transition_measurement.v1",
                "criterion_id": "door_articulation_angle",
                "observable_transition": "door_angle_rad_increased",
                "before_value": 0.0,
                "after_value": 0.3,
                "unit": "radian",
                "source_step_index": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_ref = {
        "path": str(evidence),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }

    def evaluator(context):
        passed = context["step_index"] == 2
        result = {
            "status": "completed",
            "criterion_id": "door_articulation_angle",
            "observable_transition": "door_angle_rad_increased",
            "before_value": 0.0,
            "after_value": 0.3 if passed else 0.05,
            "episode_initial_value": 0.0,
            "tolerance": 0.2,
            "unit": "radian",
            "source_step_index": context["step_index"],
            "passed": passed,
            "evidence_artifacts": [evidence_ref],
        }
        result["evaluator_attestation"] = _attest_task_completion(
            result,
            tmp_path,
            f"door-step-{context['step_index']}",
        )
        return result

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "manipulation_transition_out",
        start_frame_path=_write_frame(tmp_path / "manipulation_transition_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=5,
        stop_on_task_completion=True,
        perception_target_prompts=["open the refrigerator"],
        task_success_contract={
            "task_kind": "manipulation",
            "criteria": [
                {
                    "criterion_id": "door_articulation_angle",
                    "observable_transition": "door_angle_rad_increased",
                    "comparison": "increase_at_least",
                    "tolerance": 0.2,
                    "unit": "radian",
                }
            ],
        },
        task_completion_evaluator=evaluator,
    )

    termination = manifest["episode_termination"]
    assert termination["task_completed_early"] is True
    assert termination["steps_executed"] == 2
    assert termination["reason"] == ("task_criterion_door_articulation_angle_passed_at_step_2")
    assert termination["task_completion_evidence_status"] == "passed"
    assert manifest["manipulation_success_proven"] is True
    assert manifest["success_proof"]["did_target_manipulation_succeed"] is True
    transition = manifest["proof"]["registered_task_completion_transition"]
    assert transition["registered_transition_passed"] is True
    assert transition["validated_evidence_artifacts"][0]["sha256"] == evidence_ref["sha256"]


def test_task_transition_measurement_artifact_must_bind_exact_content(tmp_path):
    contract = {
        "task_kind": "manipulation",
        "criteria": [
            {
                "criterion_id": "door_articulation_angle",
                "observable_transition": "door_angle_rad_increased",
                "comparison": "increase_at_least",
                "tolerance": 0.2,
                "unit": "radian",
            }
        ],
    }
    base_measurement = {
        "schema_version": "task_transition_measurement.v1",
        "criterion_id": "door_articulation_angle",
        "observable_transition": "door_angle_rad_increased",
        "before_value": 0.0,
        "after_value": 0.3,
        "unit": "radian",
        "source_step_index": 2,
    }

    def validate(measurement, *, name):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(measurement) + "\n", encoding="utf-8")
        result = {
            "status": "completed",
            "criterion_id": "door_articulation_angle",
            "observable_transition": "door_angle_rad_increased",
            "before_value": 0.0,
            "after_value": 0.3,
            "episode_initial_value": 0.0,
            "tolerance": 0.2,
            "unit": "radian",
            "source_step_index": 2,
            "passed": True,
            "evidence_artifacts": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        }
        result["evaluator_attestation"] = _attest_task_completion(
            result,
            tmp_path,
            f"measurement-{name}",
        )
        return L._validate_task_completion_transition(
            completion_result=result,
            task_success_contract=contract,
            expected_source_step_index=2,
        )

    assert validate(base_measurement, name="valid")["registered_transition_passed"] is True
    for field, bad_value in (
        ("criterion_id", "different_criterion"),
        ("observable_transition", "different_transition"),
        ("before_value", 99.0),
        ("after_value", 99.0),
        ("unit", "degree"),
        ("source_step_index", 1),
    ):
        invalid = validate(
            {**base_measurement, field: bad_value},
            name=f"bad-{field}",
        )
        assert invalid["registered_transition_passed"] is False
        assert (
            f"task_transition_measurement_binding_mismatch:{field}:0"
            in invalid["validation_blockers"]
        )

    untyped = validate({"before": 0.0, "after": 0.3}, name="untyped")
    assert untyped["registered_transition_passed"] is False
    assert "task_transition_measurement_schema_invalid:0" in untyped["validation_blockers"]


def test_manipulation_rejects_unregistered_or_unhashed_transition_evidence(tmp_path):
    def evaluator(context):
        return {
            "status": "completed",
            "criterion_id": "unregistered_root_distance",
            "observable_transition": "root_distance_changed",
            "before_value": 1.0,
            "after_value": 0.0,
            "tolerance": float("nan"),
            "passed": True,
            "evidence_refs": [f"state://step/{context['step_index']}"],
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "unregistered_transition_out",
        start_frame_path=_write_frame(tmp_path / "unregistered_transition_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=3,
        stop_on_task_completion=True,
        perception_target_prompts=["open the refrigerator"],
        task_success_contract={
            "task_kind": "manipulation",
            "criteria": [
                {
                    "criterion_id": "door_articulation_angle",
                    "observable_transition": "door_angle_rad_increased",
                    "comparison": "increase_at_least",
                    "tolerance": 0.2,
                }
            ],
        },
        task_completion_evaluator=evaluator,
    )

    assert manifest["episode_termination"]["task_completed_early"] is False
    assert manifest["manipulation_success_proven"] is False
    validation_blockers = manifest["episode_termination"]["task_completion_results"][0][
        "validation_blockers"
    ]
    assert "task_transition_criterion_not_registered" in validation_blockers
    assert "task_transition_tolerance_missing_nonfinite_or_unregistered" in validation_blockers
    assert "task_transition_hashed_evidence_artifacts_missing" in validation_blockers


def test_action_conditioning_rejects_proxy_and_binds_fresh_action_identity():
    proxy_action = {
        "policy_action": "learned_policy_action",
        "sonic_action_chunk": [0.1] * 7,
        "not_a_learned_robot_policy_action": True,
        "out_of_distribution_action_projection": True,
    }
    proxy_blockers = L._action_conditioning_blockers(
        action=proxy_action,
        wam_output={},
    )
    assert "not_a_learned_robot_policy_action" in proxy_blockers
    assert "surrogate_policy_action_projection_not_allowed" in proxy_blockers

    action = {"policy_action": "learned_policy_action", "action_chunk": [0.1] * 7}
    action_sha = L._canonical_sha256(action)
    projection = L._with_action_conditioning_digests(
        {
            "landmarks": [{"name": "wrist", "x": 0.1, "y": 0.0}],
            "derived_via_controller_fk": True,
            "source_action_sha256": action_sha,
            "controller_id": "unitree-g1-controller-v1",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "proxy_or_surrogate": False,
                "joint_positions": [0.1] * 7,
            },
        }
    )
    valid_output = {
        "skeleton_conditioning": projection,
        "generated_robot_state": projection["generated_robot_state"],
    }
    assert L._action_conditioning_blockers(action=action, wam_output=valid_output) == []

    perturbed = {**action, "action_chunk": [-0.1] * 7}
    assert "fresh_action_skeleton_identity_mismatch" in L._action_conditioning_blockers(
        action=perturbed,
        wam_output=valid_output,
    )

    empty_evidence = L._with_action_conditioning_digests(
        {
            "landmarks": [{}],
            "derived_via_controller_fk": True,
            "source_action_sha256": action_sha,
            "controller_id": "unitree-g1-controller-v1",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "proxy_or_surrogate": False,
            },
        }
    )
    empty_blockers = L._action_conditioning_blockers(
        action=action,
        wam_output={
            "skeleton_conditioning": empty_evidence,
            "generated_robot_state": empty_evidence["generated_robot_state"],
        },
    )
    assert "fresh_action_skeleton_landmark_id_missing:0" in empty_blockers
    assert "fresh_action_skeleton_landmark_numeric_evidence_missing:0" in empty_blockers
    assert "generated_robot_state_numeric_evidence_missing" in empty_blockers


def test_controller_fk_skeleton_command_binds_exact_action_and_state(tmp_path):
    command = tmp_path / "controller_fk.py"
    command.write_text(
        """
import hashlib, json, os
from pathlib import Path
from blueprint_pipeline.oscar_isaac_closed_loop_eval import build_sc3_runtime_attestation

request = json.load(open(os.environ['BLUEPRINT_CONTROLLER_FK_INPUT']))
root = Path.cwd()
action = request['action']['action_chunk']
controller_code = root / 'controller-code.bin'
robot_model = root / 'robot-model.xml'
controller_code.write_bytes(b'trusted controller FK implementation')
robot_model.write_bytes(b'trusted robot model')
controller_sha256 = hashlib.sha256(controller_code.read_bytes()).hexdigest()
robot_model_sha256 = hashlib.sha256(robot_model.read_bytes()).hexdigest()
payload = {
    'status': 'completed',
    'runtime_result_id': f"fk-result-{request['step_index']}",
    'source_action_sha256': request['source_action_sha256'],
    'derived_via_controller_fk': True,
    'controller_id': 'controller-v1',
    'controller_sha256': controller_sha256,
    'robot_model_sha256': robot_model_sha256,
    'controller_code_artifact': {'path': str(controller_code), 'sha256': controller_sha256},
    'robot_model_artifact': {'path': str(robot_model), 'sha256': robot_model_sha256},
    'landmarks': [{'name':'wrist','x':action[0],'y':action[1]}],
    'generated_robot_state': {
        'source_action_sha256': request['source_action_sha256'],
        'proxy_or_surrogate': False,
        'joint_positions': action,
    },
}
signed_result = {
    'schema_version': 'sc3_controller_fk_runtime_result.v1',
    'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    'step_index': int(request['step_index']),
    'source_action_sha256': request['source_action_sha256'],
    'runtime_result_id': payload['runtime_result_id'],
    'controller_id': payload['controller_id'],
    'controller_sha256': controller_sha256,
    'robot_model_sha256': robot_model_sha256,
    'controller_code_artifact': payload['controller_code_artifact'],
    'robot_model_artifact': payload['robot_model_artifact'],
    'derived_via_controller_fk': True,
    'landmarks': payload['landmarks'],
    'generated_robot_state': payload['generated_robot_state'],
}
payload['executor_attestation'] = build_sc3_runtime_attestation(
    signed_result,
    private_key_file=os.environ['BLUEPRINT_TEST_FK_EXECUTOR_PRIVATE_KEY_FILE'],
    report_path=root / 'fk-signature-report.json',
    signer_key_id='fk-runtime-test',
    verifier_id='blueprint-test-verifier',
)
json.dump(payload, open(os.environ['BLUEPRINT_CONTROLLER_FK_OUTPUT'], 'w'))
""".strip(),
        encoding="utf-8",
    )
    projector = L.make_controller_fk_skeleton_projector(
        command=f"{sys.executable} {command}",
        work_dir=tmp_path / "controller_fk",
    )
    action = {"policy_action": "learned", "action_chunk": [0.2, -0.1]}

    projection = projector(action, 1)

    assert projection["source_action_sha256"] == L._canonical_sha256(action)
    assert projection["derived_via_controller_fk"] is True
    assert projection["landmarks"][0]["x"] == 0.2
    assert projection["generated_robot_state"]["proxy_or_surrogate"] is False
    with pytest.raises(RuntimeError, match="runtime_result_id_missing_or_replayed"):
        projector(action, 1)


def test_per_step_backend_uses_distinct_fresh_action_skeletons(tmp_path):
    captured = []

    def projector(action, step_index):
        chunk = action["action_chunk"]
        action_sha = L._canonical_sha256(action)
        return {
            "landmarks": [
                {"name": "wrist", "x": float(value), "y": float(step_index)} for value in chunk
            ],
            "derived_via_controller_fk": True,
            "source_action_sha256": action_sha,
            "controller_id": "controller",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "proxy_or_surrogate": False,
                "joint_positions": [float(value) for value in chunk],
            },
        }

    def generate(request):
        captured.append(request)
        frame = _write_frame(tmp_path / f"projected_{len(captured)}.png", 20 + len(captured))
        return {"status": "completed", "generated_frame_path": str(frame)}

    backend = L.make_oscar_per_step_wam_backend(
        oscar_generate=generate,
        work_dir=tmp_path / "wam",
        task_prompt="move",
        skeleton_for_action=projector,
    )
    chunks = [
        [0.0, 0.0],
        [0.2, 0.2],
        [-0.2, -0.2],
        [0.2, -0.2],
    ]
    conditioning = [
        backend(
            str(_write_frame(tmp_path / f"source_{index}.png", index + 1)),
            {"action_chunk": chunk, "policy_action": "learned_policy_action"},
            index + 1,
            [],
        )["skeleton_conditioning"]
        for index, chunk in enumerate(chunks)
    ]

    signatures = {json.dumps(row["landmarks"], sort_keys=True) for row in conditioning}
    assert len(signatures) == len(chunks)
    assert all(row["derived_via_controller_fk"] is True for row in conditioning)


@pytest.mark.parametrize(
    ("distinct_evidence", "expected_status"),
    ((True, "completed"), (False, "blocked")),
)
def test_strict_closed_loop_requires_action_differentiated_fk_evidence(
    tmp_path, distinct_evidence, expected_status
):
    generation_count = 0

    def projector(action, step_index):
        chunk = action.get("action_chunk") or [0.0, 0.0]
        evidence_chunk = chunk if distinct_evidence else [0.0, 0.0]
        action_sha = L._canonical_sha256(action)
        return {
            "landmarks": [
                {
                    "name": "wrist",
                    "x": float(evidence_chunk[0]),
                    "y": float(evidence_chunk[1]),
                }
            ],
            "derived_via_controller_fk": True,
            "source_action_sha256": action_sha,
            "controller_id": "controller-v1",
            "controller_sha256": "a" * 64,
            "robot_model_sha256": "b" * 64,
            "generated_robot_state": {
                "source_action_sha256": action_sha,
                "proxy_or_surrogate": False,
                "joint_positions": [float(value) for value in evidence_chunk],
            },
        }

    def generate(request):
        nonlocal generation_count
        generation_count += 1
        frame = _write_frame(
            tmp_path / f"strict-generated-{generation_count}.png",
            30 + generation_count,
        )
        return {"status": "completed", "generated_frame_path": str(frame)}

    def policy_endpoint(observation, history, step_index):
        return {
            "policy_action": "learned_policy_action",
            "root_position": [step_index * 0.1, 0.0, 0.79],
            "action_chunk": [float(step_index), float(-step_index)],
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / f"strict-{distinct_evidence}",
        start_frame_path=_write_frame(tmp_path / f"strict-seed-{distinct_evidence}.png", 4),
        route_points=[[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
        wam_generate_next=L.make_oscar_per_step_wam_backend(
            oscar_generate=generate,
            work_dir=tmp_path / f"strict-wam-{distinct_evidence}",
            task_prompt="move the wrist",
            skeleton_for_action=projector,
        ),
        steps=4,
        policy_endpoint=policy_endpoint,
        require_fresh_learned_policy_requery=True,
        require_action_derived_skeleton_conditioning=True,
    )

    assert manifest["status"] == expected_status
    assert manifest["proof"]["fresh_action_conditioning_differentiation_proven"] is (
        distinct_evidence
    )
    if not distinct_evidence:
        assert any(
            "fresh_action_conditioning_not_action_differentiated" in blocker
            for blocker in manifest["blockers"]
        )


def test_sc3_closed_loop_blocks_egocentric_only_wam_output(tmp_path):
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "egocentric_only",
        start_frame_path=_write_frame(tmp_path / "egocentric_seed.png", 2),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=1,
        require_synchronized_calibrated_multiview=True,
    )

    assert manifest["status"] == "blocked"
    assert any(blocker.startswith("multiview_step_1:") for blocker in manifest["blockers"])


def test_success_review_manifest_preserves_all_ordered_step_clips(tmp_path):
    clips = []
    for step in (1, 2, 3):
        clip = tmp_path / f"step-{step}.mp4"
        clip.write_bytes(f"clip-{step}".encode())
        clips.append(clip)
    artifacts = L._closed_loop_generated_episode_artifacts(
        output_dir=tmp_path / "episode",
        generated_at="now",
        trace_rows=[
            {
                "step_index": step,
                "wam_generated_video": str(clip),
                "wam_generated_frame": f"frame-{step}.png",
                "source_observation_frame": f"source-{step}.png",
                "policy_action": f"action-{step}",
            }
            for step, clip in zip((1, 2, 3), clips)
        ],
        initial_frame_path="initial.png",
        policy_id="p",
        task_prompts=["open the door"],
        target=[0.0, 0.0, 0.0],
    )

    rollout = artifacts["rollouts"][0]
    assert rollout["episode_order_verified"] is True
    assert rollout["review_media_scope"] == "full_ordered_episode"
    assert [row["step_index"] for row in rollout["ordered_step_videos"]] == [1, 2, 3]
    assert [row["generated_video_path"] for row in rollout["ordered_step_videos"]] == [
        str(clip) for clip in clips
    ]
    assert [row["generated_video_sha256"] for row in rollout["ordered_step_videos"]] == [
        hashlib.sha256(clip.read_bytes()).hexdigest() for clip in clips
    ]


def test_success_review_manifest_blocks_missing_or_noncontiguous_step_clips(tmp_path):
    clip = tmp_path / "step-1.mp4"
    clip.write_bytes(b"clip-1")
    artifacts = L._closed_loop_generated_episode_artifacts(
        output_dir=tmp_path / "incomplete-episode",
        generated_at="now",
        trace_rows=[
            {"step_index": 1, "wam_generated_video": str(clip)},
            {"step_index": 3, "wam_generated_video": str(tmp_path / "missing.mp4")},
        ],
        initial_frame_path="initial.png",
        policy_id="p",
        task_prompts=["open the door"],
        target=[0.0, 0.0, 0.0],
    )

    assert artifacts["status"] == "blocked"
    assert artifacts["episode_order_verified"] is False
    assert artifacts["rollouts"] == []
    assert "closed_loop_step_video_file_missing:2" in artifacts["blockers"]
    assert "closed_loop_episode_order_not_verified" in artifacts["blockers"]


def test_episode_runs_full_cap_without_stop_flag(tmp_path):
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "cap_out",
        start_frame_path=_write_frame(tmp_path / "cap_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=3,
    )
    termination = manifest["episode_termination"]
    assert termination["task_completed_early"] is False
    assert termination["reason"] == "steps_cap_reached"
    assert termination["steps_executed"] == 3


def test_sealed_plan_and_cli_carry_stop_on_task_completion(tmp_path):
    from blueprint_pipeline import groot_oscar_closed_loop_image as gocl

    plan = gocl.build_sealed_launch_plan(
        start_frame="/workspace/seed.png",
        route_file="/workspace/route.json",
        steps=12,
        task_prompt="open the fridge",
        output_dir="/workspace/t4_out",
        env={"BLUEPRINT_GROOT_OSCAR_SEALED_IMAGE": "true"},
    )
    if plan["closed_loop_command"]:
        assert "--stop-on-task-completion" in plan["closed_loop_command"]
        assert (
            plan["closed_loop_command"][plan["closed_loop_command"].index("--min-steps") + 1] == "3"
        )
        assert plan["episode_length_contract"]["min_steps_before_task_completion"] == 3


def test_transition_verdict_uses_episode_baseline_not_step_pair(tmp_path):
    contract = {
        "task_kind": "manipulation",
        "criteria": [
            {
                "criterion_id": "door_articulation_angle",
                "observable_transition": "door_angle_rad_increased",
                "comparison": "increase_at_least",
                "tolerance": 0.35,
                "unit": "radian",
            }
        ],
    }

    def validate(*, step, before, after, initial, passed, name, episode_fields=True):
        measurement = {
            "schema_version": "task_transition_measurement.v1",
            "criterion_id": "door_articulation_angle",
            "observable_transition": "door_angle_rad_increased",
            "before_value": before,
            "after_value": after,
            "unit": "radian",
            "source_step_index": step,
        }
        if episode_fields:
            measurement["episode_initial_value"] = initial
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(measurement) + "\n", encoding="utf-8")
        result = {
            "status": "completed",
            "criterion_id": "door_articulation_angle",
            "observable_transition": "door_angle_rad_increased",
            "before_value": before,
            "after_value": after,
            "tolerance": 0.35,
            "unit": "radian",
            "source_step_index": step,
            "passed": passed,
            "evidence_artifacts": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
        }
        if episode_fields:
            result["episode_initial_value"] = initial
            result["step_delta"] = after - before
            result["episode_delta"] = after - initial
        result["evaluator_attestation"] = _attest_task_completion(
            result, tmp_path, f"episode-{name}"
        )
        return L._validate_task_completion_transition(
            completion_result=result,
            task_success_contract=contract,
            expected_source_step_index=step,
        )

    first = validate(
        step=0, before=0.0, after=0.2, initial=0.0, passed=False, name="step0"
    )
    assert first["registered_transition_passed"] is False
    assert "task_transition_reported_verdict_mismatch" not in first[
        "validation_blockers"
    ]

    second = validate(
        step=1, before=0.2, after=0.4, initial=0.0, passed=True, name="step1"
    )
    assert second["registered_transition_passed"] is True

    step_pair_only = validate(
        step=1,
        before=0.2,
        after=0.4,
        initial=0.0,
        passed=True,
        name="no-episode",
        episode_fields=False,
    )
    assert step_pair_only["registered_transition_passed"] is False
    assert any(
        "episode_initial_value" in item
        for item in step_pair_only["validation_blockers"]
    )
