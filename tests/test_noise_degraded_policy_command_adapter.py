from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline import noise_degraded_policy_command_adapter as adapter

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _fake_inner_adapter(tmp_path: Path, *, chunk_length: int = 8) -> str:
    runner = tmp_path / "fake_inner_adapter.py"
    chunk = [round(0.1 * index, 3) for index in range(chunk_length)]
    runner.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "payload = json.loads(sys.stdin.read() or '{}')",
                "response = {",
                "    'schema_version': 'unitree_groot_n17_sonic_policy_command_adapter.v1',",
                "    'status': 'completed',",
                "    'policy_id': 'unitree_groot_n17_sonic',",
                "    'model_ran': True,",
                "    'action': {",
                "        'action_type': 'unitree_g1_sonic_action_chunk',",
                f"        'action_chunk': {chunk!r},",
                "        'unitree_groot_n17_sonic_action_chunk_present': True,",
                "        'source_action_key': 'action_chunk',",
                "    },",
                "    'claim_boundary': {'generated_world_rank_fidelity_result_proven': False},",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )
    return f"{sys.executable} {runner}"


def _observation() -> dict:
    return {
        "task_id": "contact_or_push_light_object",
        "scenario_eval_run_id": "run_0001",
        "step_index": 3,
    }


def _registered_bounds(chunk_length: int, *, limit: float = 10.0) -> dict:
    return {
        "schema_version": adapter.REGISTERED_ACTION_BOUNDS_SCHEMA_VERSION,
        "contract_id": f"test-action-chunk-{chunk_length}-bounds.v1",
        "action_representation": "test_action_chunk",
        "fields": {
            "action_chunk": {
                "lower": [-limit] * chunk_length,
                "upper": [limit] * chunk_length,
            }
        },
    }


def _bounds_kwargs(chunk_length: int, *, limit: float = 10.0) -> dict:
    contract = _registered_bounds(chunk_length, limit=limit)
    return {
        "registered_action_bounds": contract,
        "registered_action_bounds_sha256_value": (
            adapter.registered_action_bounds_sha256(contract)
        ),
    }


def test_noise_adapter_blocks_without_inner_command_or_amplitude() -> None:
    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=None,
        amplitude=None,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert (
        f"set_{adapter.INNER_COMMAND_ENV}_to_runnable_inner_policy_adapter_command"
        in response["blockers"]
    )
    assert f"set_{adapter.AMPLITUDE_ENV}_to_nonnegative_noise_amplitude" in response["blockers"]
    assert response["claim_boundary"]["degraded_variant_for_ranker_validation_only"] is True
    assert response["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False


@pytest.mark.parametrize("amplitude", [-0.5, float("nan"), float("inf")])
def test_noise_adapter_blocks_invalid_amplitude(tmp_path: Path, amplitude: float) -> None:
    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=_fake_inner_adapter(tmp_path),
        amplitude=amplitude,
    )

    assert exit_code == 2
    assert "noise_amplitude_must_be_finite_and_nonnegative" in response["blockers"]


def test_noise_adapter_perturbs_action_chunk_deterministically(tmp_path: Path) -> None:
    inner_command = _fake_inner_adapter(tmp_path)
    original_chunk = [round(0.1 * index, 3) for index in range(8)]

    first, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=inner_command,
        amplitude=0.3,
        seed=7,
        **_bounds_kwargs(8),
    )
    second, _ = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=inner_command,
        amplitude=0.3,
        seed=7,
        **_bounds_kwargs(8),
    )
    other_seed, _ = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=inner_command,
        amplitude=0.3,
        seed=8,
        **_bounds_kwargs(8),
    )

    assert exit_code == 0
    assert first["status"] == "completed"
    assert first["policy_id"] == "unitree_groot_n17_sonic_noise_0p3"
    assert first["inner_policy_id"] == "unitree_groot_n17_sonic"
    assert first["model_ran"] is True
    assert first["action"]["action_chunk"] != original_chunk
    assert first["action"]["noise_injected"] is True
    assert first["action"]["noise_amplitude"] == 0.3
    assert first["noise_injection"]["perturbed_value_count"] == 8
    assert first["noise_injection"]["action_bounds_validated"] is True
    assert first["action"]["action_chunk"] == second["action"]["action_chunk"]
    assert first["action"]["action_chunk"] != other_seed["action"]["action_chunk"]
    assert first["claim_boundary"]["synthetic_noise_degradation_injected"] is True
    assert first["claim_boundary"]["degraded_action_is_not_policy_checkpoint_behavior"] is True
    assert first["claim_boundary"]["generated_world_rank_fidelity_result_proven"] is False


def test_noise_adapter_zero_amplitude_is_passthrough(tmp_path: Path) -> None:
    original_chunk = [round(0.1 * index, 3) for index in range(8)]
    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=_fake_inner_adapter(tmp_path),
        amplitude=0.0,
        **_bounds_kwargs(8),
    )

    assert exit_code == 0
    assert response["action"]["action_chunk"] == [float(v) for v in original_chunk]
    assert response["action"]["noise_injected"] is False
    assert response["noise_injection"]["action_values_perturbed"] is False
    assert response["claim_boundary"]["synthetic_noise_degradation_injected"] is False
    assert response["policy_id"] == "unitree_groot_n17_sonic_noise_0"


def test_noise_adapter_higher_amplitude_deviates_more(tmp_path: Path) -> None:
    inner_command = _fake_inner_adapter(tmp_path, chunk_length=64)
    original = [round(0.1 * index, 3) for index in range(64)]

    def mean_absolute_deviation(amplitude: float) -> float:
        response, exit_code = adapter.run_noise_degraded_policy(
            payload={"observation": _observation()},
            inner_command=inner_command,
            amplitude=amplitude,
            seed=7,
            **_bounds_kwargs(64),
        )
        assert exit_code == 0
        chunk = response["action"]["action_chunk"]
        return sum(abs(value - base) for value, base in zip(chunk, original)) / len(original)

    assert (
        mean_absolute_deviation(0.1) < mean_absolute_deviation(0.6) < mean_absolute_deviation(2.0)
    )


def test_noise_adapter_leaves_metadata_fields_untouched(tmp_path: Path) -> None:
    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=_fake_inner_adapter(tmp_path),
        amplitude=0.5,
        **_bounds_kwargs(8),
    )

    assert exit_code == 0
    action = response["action"]
    assert action["action_type"] == "unitree_g1_sonic_action_chunk"
    assert action["unitree_groot_n17_sonic_action_chunk_present"] is True
    assert action["source_action_key"] == "action_chunk"


def test_noise_adapter_propagates_inner_blockers(tmp_path: Path) -> None:
    runner = tmp_path / "blocked_inner_adapter.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "response = {",
                "    'status': 'blocked',",
                "    'policy_id': 'unitree_groot_n17_sonic',",
                "    'blockers': ['blocked_missing_policy_visual_observation_frame'],",
                "}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
                "raise SystemExit(2)",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=f"{sys.executable} {runner}",
        amplitude=0.3,
        **_bounds_kwargs(8),
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert "inner:blocked_missing_policy_visual_observation_frame" in response["blockers"]
    assert response["inner_policy_id"] == "unitree_groot_n17_sonic"
    assert response["noise_injection"]["action_values_perturbed"] is False


def test_noise_adapter_blocks_unbounded_action_instead_of_emitting_unsafe_noise(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "unbounded_inner_adapter.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "response = {'status':'completed','policy_id':'p','model_ran':True,",
                "            'action': {'action_chunk':[0.0, 0.1]}}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=f"{sys.executable} {runner}",
        amplitude=0.3,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert any("registered_action_bounds" in blocker for blocker in response["blockers"])


def test_noise_adapter_rejects_inner_reported_bounds_drift(tmp_path: Path) -> None:
    runner = tmp_path / "drifted_bounds_inner_adapter.py"
    runner.write_text(
        "\n".join(
            [
                "import json, os",
                "response = {'status':'completed','policy_id':'p','model_ran':True,",
                "            'action_bounds': {'action_chunk': {'lower':[-999.0,-999.0], 'upper':[999.0,999.0]}},",
                "            'action': {'action_chunk':[0.0, 0.1]}}",
                "open(os.environ['BLUEPRINT_POLICY_ACTION_OUTPUT'], 'w').write(json.dumps(response))",
            ]
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=f"{sys.executable} {runner}",
        amplitude=0.3,
        **_bounds_kwargs(2, limit=1.0),
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert (
        "blocked_noise_degraded_inner_action_bounds_drift_from_registered_contract"
        in response["blockers"]
    )


def test_noise_adapter_rejects_oversized_registered_bounds(tmp_path: Path) -> None:
    contract = _registered_bounds(8, limit=adapter.MAX_REGISTERED_ACTION_ABS_BOUND + 1.0)
    response, exit_code = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=_fake_inner_adapter(tmp_path),
        amplitude=0.3,
        registered_action_bounds=contract,
        registered_action_bounds_sha256_value=(adapter.registered_action_bounds_sha256(contract)),
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert any("registered_action_bounds_oversized" in blocker for blocker in response["blockers"])


def test_noise_adapter_explicit_policy_id_used_on_blocked_and_completed(tmp_path: Path) -> None:
    completed, _ = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=_fake_inner_adapter(tmp_path),
        amplitude=0.3,
        policy_id="ladder_rung_2",
        **_bounds_kwargs(8),
    )
    blocked, _ = adapter.run_noise_degraded_policy(
        payload={"observation": _observation()},
        inner_command=None,
        amplitude=0.3,
        policy_id="ladder_rung_2",
    )

    assert completed["policy_id"] == "ladder_rung_2"
    assert blocked["policy_id"] == "ladder_rung_2"


def test_noise_adapter_cli_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    inner_command = _fake_inner_adapter(tmp_path)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"observation": _observation()}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.noise_degraded_policy_command_adapter",
            "--inner-command",
            inner_command,
            "--noise-amplitude",
            "0.3",
            "--seed",
            "7",
            "--registered-action-bounds-json",
            json.dumps(_registered_bounds(8), sort_keys=True, separators=(",", ":")),
            "--registered-action-bounds-sha256",
            adapter.registered_action_bounds_sha256(_registered_bounds(8)),
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
            "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(output_path.read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["policy_id"] == "unitree_groot_n17_sonic_noise_0p3"
    assert response["action"]["noise_injected"] is True


def test_noise_adapter_manifest_declares_env_contract() -> None:
    manifest = adapter.adapter_manifest()

    assert manifest["schema_version"] == "policy_command_adapter_manifest.v1"
    assert manifest["wraps_inner_policy_adapter"] is True
    assert adapter.INNER_COMMAND_ENV in manifest["required_env"]
    assert adapter.AMPLITUDE_ENV in manifest["required_env"]
    assert manifest["claim_boundary"]["degraded_variant_for_ranker_validation_only"] is True


def test_environment_noise_amplitude_parser_is_finite_and_numeric() -> None:
    assert adapter._float_or_none("0.3") == 0.3
    assert adapter._float_or_none(0) == 0.0
    assert adapter._float_or_none("not-a-number") is None
    assert adapter._float_or_none("nan") is None


def test_noise_degraded_policy_id_labels() -> None:
    assert adapter.noise_degraded_policy_id("policy_a", 0.1) == "policy_a_noise_0p1"
    assert adapter.noise_degraded_policy_id("policy_a", 0.25) == "policy_a_noise_0p25"
    assert adapter.noise_degraded_policy_id("policy_a", 1.0) == "policy_a_noise_1"
