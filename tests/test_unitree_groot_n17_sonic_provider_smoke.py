from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

from blueprint_pipeline import unitree_groot_n17_sonic_provider_smoke as smoke


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\x81\xb3\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _provider_runner_namespace(tmp_path: Path) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": "blueprint_unitree_groot_sonic_provider_runner_test",
        "__file__": str(tmp_path / "unitree_groot_n17_sonic_provider_runner.py"),
    }
    exec(smoke.PROVIDER_RUNNER, namespace)
    return namespace


def _frame(path: Path) -> Path:
    path.write_bytes(PNG_1X1)
    return path


def test_groot_n17_sonic_provider_bundle_contains_runtime_contract(tmp_path: Path) -> None:
    manifest = smoke.build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        policy_command="python run_groot_sonic_policy.py",
        n17_checkpoint="nvidia/GR00T-N1.7-3B",
        sonic_checkpoint="/weights/g1_sonic/checkpoint-20000",
        groot_root="/workspace/Isaac-GR00T",
        wbc_root="/workspace/GR00T-WholeBodyControl",
        policy_server_url="tcp://127.0.0.1:5550",
        sim2sim_command="python gear_sonic/scripts/run_sim_loop.py",
    )

    bundle = Path(manifest["bundle_path"])
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        runner_text = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_provider_runner.py"
        ).decode("utf-8")
    assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
    assert "provider_runtime/unitree_groot_n17_sonic_policy_provider_manifest.json" in names
    assert "provider_runtime/policy_input.json" in names
    assert (
        "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py"
        in names
    )
    ast.parse(runner_text)
    assert "run_unitree_groot_n17_sonic_policy" in runner_text
    assert "uv run --project" not in runner_text
    assert "venv_python" in runner_text
    assert "snapshot_download" in runner_text
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_ATTEMPTS" in runner_text
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_MAX_WORKERS" in runner_text
    assert "HF_HUB_DISABLE_XET" in runner_text
    assert "HF_HUB_ENABLE_HF_TRANSFER" in runner_text
    assert "BLUEPRINT_GROOT_MODEL_SNAPSHOT_ATTEMPT_FAILED" in runner_text
    assert '"processor_config.json"' in runner_text
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE" in runner_text
    assert "system_python_minimal" in runner_text
    assert "sealed_image" in runner_text
    assert "sealed_image_uses_prebaked_system_python_deps" in runner_text
    assert "blocked_sealed_image_missing_local_gr00t_model_snapshot" in runner_text
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT" in runner_text
    assert "--filter=blob:none --sparse" in runner_text
    assert "httpx>=0.27.0" in runner_text
    assert "tokenizers==0.22.1" in runner_text
    assert "transformers==4.57.3" in runner_text
    assert "albumentations==1.4.18" in runner_text
    assert "albucore==0.0.17" in runner_text
    assert "diffusers==0.35.1" in runner_text
    assert "dm-tree" in runner_text
    assert "peft==0.17.1" in runner_text
    assert "pandas==2.2.3" in runner_text
    assert "pydantic==2.13.4" in runner_text
    assert "pydantic-core==2.46.4" in runner_text
    assert "msgpack-numpy==0.4.8" in runner_text
    assert "numkong>=0.1.0" in runner_text
    assert "scipy==1.15.3" in runner_text
    assert "scikit-image==0.25.2" in runner_text
    assert "imageio>=2.33.0" in runner_text
    assert "networkx>=3.0" in runner_text
    assert "tifffile>=2022.8.12" in runner_text
    assert "lazy-loader>=0.4" in runner_text
    assert "tyro==0.9.17" in runner_text
    assert 'modules=["huggingface_hub", "httpx", "zmq", "transformers"]' in runner_text
    assert "blocked_system_python_missing_torch" in runner_text
    assert manifest["env_contract"]["BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"] == (
        "<configured>"
    )
    assert manifest["ready_for_fresh_model_execution"] is True
    assert manifest["runtime_execution_blockers"] == []
    assert (
        manifest["truth_boundary"]["unitree_groot_n17_sonic_policy_action_command_ran"]
        is False
    )
    assert manifest["truth_boundary"]["generated_world_rank_fidelity_result_proven"] is False


def test_groot_provider_bundle_preserves_scene_policy_observation(tmp_path: Path) -> None:
    observation_path = tmp_path / "initial_policy_observation.json"
    observation = {
        "schema_version": "blueprint_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "visual_observation": {"camera_frame_path": "/old/frame.jpg"},
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "scene_packet_contract_probe_zero_state",
    }
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    manifest = smoke.build_unitree_groot_n17_sonic_policy_provider_bundle(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        policy_observation_path=observation_path,
    )

    assert manifest["policy_observation_preserved"] is True
    policy_input = json.loads(
        Path(manifest["policy_input_path"]).read_text(encoding="utf-8")
    )
    bundled_observation = policy_input["observation"]
    assert bundled_observation["task_id"] == "turn_on_sink_handle"
    assert sorted(bundled_observation["unitree_g1_sonic_state"]) == [
        "left_arm",
        "left_hand",
        "left_leg",
        "projected_gravity",
        "right_arm",
        "right_hand",
        "right_leg",
        "waist",
    ]
    assert bundled_observation["visual_observation"]["camera_frame_path"] == "input_frame.png"


def test_import_groot_n17_sonic_provider_output_completed(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider_output.zip"
    provider_payload = {
        "schema_version": "unitree_groot_n17_sonic_policy_provider_output.v1",
        "status": "completed",
        "unitree_groot_n17_sonic_model_executed": True,
        "unitree_groot_n17_sonic_policy_action_command_ran": True,
        "policy_action_model_command_ran": True,
        "action": {"action_type": "unitree_g1_sonic_action_chunk"},
    }
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(provider_payload),
        )

    imported = smoke.import_unitree_groot_n17_sonic_provider_output(
        provider_output_zip=output_zip,
        extraction_dir=tmp_path / "extracted",
        output_path=tmp_path / "import.json",
    )

    assert imported["status"] == "completed"
    assert imported["unitree_groot_n17_sonic_model_executed"] is True
    assert imported["unitree_groot_n17_sonic_policy_action_command_ran"] is True
    assert imported["action"]["action_type"] == "unitree_g1_sonic_action_chunk"
    assert imported["truth_boundary"]["generated_world_rank_fidelity_result_proven"] is False
    assert (
        imported["truth_boundary"]["provider_output_import_is_not_fresh_local_policy_execution"]
        is True
    )


def test_groot_n17_sonic_provider_smoke_dry_run(tmp_path: Path) -> None:
    summary = smoke.run_unitree_groot_n17_sonic_policy_provider_smoke(
        job_dir=tmp_path / "job",
        frame_path=_frame(tmp_path / "frame.png"),
        dry_run=True,
    )

    assert summary["status"] == "dry_run_ready"
    assert summary["unitree_groot_n17_sonic_model_executed"] is False
    assert summary["unitree_groot_n17_sonic_policy_action_command_ran"] is False
    assert summary["ready_for_fresh_model_execution"] is False
    assert "blocked_missing_unitree_groot_n17_sonic_policy_command" in summary[
        "runtime_execution_blockers"
    ]
    assert "blocked_missing_unitree_groot_n17_checkpoint" in summary[
        "runtime_execution_blockers"
    ]
    assert "blocked_missing_unitree_g1_sonic_checkpoint" in summary[
        "runtime_execution_blockers"
    ]
    assert Path(summary["bundle_manifest_path"]).is_file()


def test_groot_sealed_image_bootstrap_uses_prebaked_system_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    namespace = _provider_runner_namespace(tmp_path)
    repo_root = tmp_path / "Isaac-GR00T"
    (repo_root / "gr00t" / "eval").mkdir(parents=True)
    (repo_root / "gr00t" / "eval" / "run_gr00t_server.py").write_text(
        "print('server')\n",
        encoding="utf-8",
    )
    model_root = tmp_path / "model"
    (model_root / "processor").mkdir(parents=True)
    (model_root / "processor" / "processor_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_AUTO_START_POLICY_SERVER", "true")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE", "sealed-image")
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_REMOTE_ROOT", str(repo_root))

    monkeypatch.setitem(
        namespace,
        "_system_python_executable",
        lambda: (
            "python3",
            {"status": "completed", "configured": False, "executable": "python3"},
        ),
    )

    import_preflights: list[list[str]] = []

    def fake_import_preflight(*, python_executable, output_dir, modules, log_name):
        import_preflights.append(list(modules))
        return {
            "status": "completed",
            "python_executable": python_executable,
            "log_path": str(output_dir / log_name),
            "modules": list(modules),
        }

    monkeypatch.setitem(namespace, "_python_import_preflight", fake_import_preflight)
    monkeypatch.setitem(
        namespace,
        "_install_system_python_minimal_deps",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("sealed image mode must not install runtime deps")
        ),
    )

    tcp_checks = iter([False, True])

    def fake_tcp_ready(host: str, port: int) -> bool:
        return next(tcp_checks, True)

    class FakeProcess:
        pid = 1234
        returncode = None

        def poll(self) -> None:
            return None

    popen_calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        popen_calls.append(list(command))
        return FakeProcess()

    monkeypatch.setitem(namespace, "_tcp_ready", fake_tcp_ready)
    monkeypatch.setattr(namespace["subprocess"], "Popen", fake_popen)

    result, process = namespace["_bootstrap_gr00t_policy_server"](
        output_dir=tmp_path / "output",
        policy_server_url="tcp://127.0.0.1:5550",
        model_path=str(model_root),
    )

    assert process is not None
    assert result["status"] == "completed"
    assert result["bootstrap_mode"] == "sealed_image"
    assert result["system_python_deps"] == {
        "status": "skipped",
        "reason": "sealed_image_uses_prebaked_system_python_deps",
        "requirements_count": 0,
    }
    assert result["model_resolution"]["source"] == "local_path"
    assert result["model_resolution"]["snapshot_download_ran"] is False
    assert import_preflights == [
        ["torch"],
        ["huggingface_hub", "httpx", "zmq", "transformers"],
    ]
    assert popen_calls and popen_calls[0][0] == "python3"


def test_groot_sealed_image_blocks_missing_local_checkpoint_without_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    namespace = _provider_runner_namespace(tmp_path)
    monkeypatch.setenv(
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SEALED_MODEL_ROOT",
        str(tmp_path / "missing_model"),
    )

    result = namespace["_materialize_groot_model_path"](
        output_dir=tmp_path / "output",
        model_path="LucaFrat/groot-bs16",
        venv_python=Path("python3"),
        env={},
        allow_snapshot_download=False,
    )

    assert result["status"] == "blocked"
    assert result["snapshot_download_ran"] is False
    assert result["blockers"] == [
        "blocked_sealed_image_missing_local_gr00t_model_snapshot"
    ]
