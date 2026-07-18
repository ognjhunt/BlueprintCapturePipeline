from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from blueprint_pipeline.groot_oscar_worker_startup_script import (
    GEAR_SONIC_READY_SCRIPT,
    GROOT_CHECKPOINT_PREFLIGHT_SCRIPT,
    STARTUP_GATES_SCRIPT,
)


def _preflight_python() -> str:
    marker = "/opt/gr00t/.venv/bin/python - <<'PY'\n"
    assert marker in GROOT_CHECKPOINT_PREFLIGHT_SCRIPT
    source = GROOT_CHECKPOINT_PREFLIGHT_SCRIPT.split(marker, 1)[1]
    return source.split("\nPY\n", 1)[0]


def _gear_sonic_ready_python() -> str:
    marker = "/opt/oscar-venv/bin/python - <<'PY'\n"
    assert GEAR_SONIC_READY_SCRIPT.count(marker) == 1
    source = GEAR_SONIC_READY_SCRIPT.split(marker, 1)[1]
    return source.split("\nPY\n", 1)[0]


def test_same_allocation_isaac_startup_gates_are_bounded_and_fail_closed() -> None:
    assert STARTUP_GATES_SCRIPT.count(
        "timeout --signal=TERM --kill-after=15s 180s"
    ) == 1
    assert (
        STARTUP_GATES_SCRIPT.index("timeout --signal=TERM --kill-after=15s 180s")
        < STARTUP_GATES_SCRIPT.index(
            "blueprint_pipeline.isaac_worker_runtime_preflight"
        )
    )
    assert (
        'BLUEPRINT_REVIEW_CANARY_TIMEOUT_SECONDS="${BLUEPRINT_REVIEW_CANARY_TIMEOUT_SECONDS:-600}"'
        in STARTUP_GATES_SCRIPT
    )
    second_timeout = STARTUP_GATES_SCRIPT.index(
        "timeout --signal=TERM --kill-after=15s 630s"
    )
    assert second_timeout < STARTUP_GATES_SCRIPT.index(
        "blueprint_pipeline.isaac_review_renderer_canary"
    )
    assert "FAST_GATE_RC=$?" in STARTUP_GATES_SCRIPT
    assert "REVIEW_GATE_RC=$?" in STARTUP_GATES_SCRIPT
    fast_command = STARTUP_GATES_SCRIPT.split(
        "blueprint_pipeline.isaac_worker_runtime_preflight", 1
    )[1].split("FAST_GATE_RC=$?", 1)[0]
    assert "--require-nvidia-smi" in fast_command
    assert "--require-rtx-render" not in fast_command
    assert "review canary is the rendered-frame authority" in STARTUP_GATES_SCRIPT
    assert "attempt 042" in STARTUP_GATES_SCRIPT
    assert "239.222s" in STARTUP_GATES_SCRIPT


def test_gear_sonic_readiness_exercises_exact_protocol_v4_action_state_boundary() -> None:
    source = _gear_sonic_ready_python()

    compile(source, "<gear-sonic-ready>", "exec")
    assert "from blueprint_pipeline.gear_sonic_official_zmq_executor import (" in source
    assert "    _zmq_roundtrip," in source
    assert "_zmq_roundtrip(" in source
    assert "frame_index=-1" in source
    assert "BLUEPRINT_LAUNCH_SESSION_ID" in source
    assert "BLUEPRINT_QUALIFICATION_ATTEMPT_NONCE" in source
    assert "'qualification_attempt_nonce' if qualification_attempt_nonce" in source
    assert "'probe_identity_kind': probe_identity_kind" in source
    assert "'probe_identity_sha256':" in source
    assert "launch_session_id_sha256" not in source
    assert "canary_vector('motion', probe_attempt" in source
    assert "canary_vector('left', probe_attempt" in source
    assert "canary_vector('right', probe_attempt" in source
    assert "integer / 32768.0" in source
    assert "token_state == motion" in source
    assert "echoed_left == left" in source
    assert "echoed_right == right" in source
    assert "finite_vector(accepted.get('body_q_target'), 29)" in source
    assert "finite_vector(accepted.get('body_q_measured'), 29)" in source
    assert "finite_vector(accepted.get('base_quat_measured'), 4)" in source
    assert "GEAR_SONIC_ISAAC_BRIDGE_PID" in source
    assert "0 <= live_heartbeat_age_ns <= 500_000_000" in source
    assert "0 <= live_snapshot_age_ns <= 500_000_000" in source
    assert "bridge_revalidation_deadline = min(probe_deadline, time.time() + 20.0)" in source
    assert "bridge_revalidation_floor_publish_count" in source
    assert "> bridge_revalidation_floor_publish_count" in source
    assert "for process_name in required_process_names:" in source
    assert "readiness['bridge_revalidation_attempts']" in source
    assert "readiness['checks']['dds_bridge_still_ready'] = bridge_revalidated" in source
    assert "official_gear_sonic_dds_bridge_not_fresh_after_matching_g1_debug" in source
    assert "controller_readiness.json" in source
    assert "raw_probe_values_recorded': False" in source
    assert "readiness_probe_is_not_episode_policy_action': True" in source


def test_gear_sonic_readiness_reuses_canary_for_full_fk_boundary_without_wire_action() -> None:
    source = _gear_sonic_ready_python()

    compile(source, "<gear-sonic-ready-controller-fk>", "exec")
    assert "execute as execute_controller_fk" in source
    assert "def retained_canary_transport(**kwargs):" in source
    assert "return dict(accepted)" in source
    assert "transport=retained_canary_transport" in source
    assert "'camera_projection_context': projection_context" in source
    assert "kwargs.get('motion_token') != motion" in source
    assert "kwargs.get('action_frames') is not None" in source
    assert "'no_additional_wire_action_sent': True" in source
    assert "controller_fk_projection_context_attempt_identity_mismatch" in source
    assert "controller_fk_projection_context_session_identity_mismatch" in source
    assert "context.get('launch_nonce') == probe_identity" in source
    assert (
        "context.get('allocation_launch_session_id') == launch_session_id"
        in source
    )
    assert "controller_fk_readiness.json" in source
    assert "'raw_controller_state_recorded': False" in source
    assert "'raw_joint_positions_recorded': False" in source
    assert "'raw_landmarks_recorded': False" in source
    assert "'raw_projection_context_recorded': False" in source
    assert "'retained_controller_state_sha256': canonical_sha256(accepted)" in source
    assert "'joint_positions_sha256': canonical_sha256(joint_positions)" in source
    assert "'landmarks_sha256': canonical_sha256(landmarks)" in source
    assert "'pinned_controller_revision_verified': bool(" in source
    assert "'official_mujoco_fk_completed': bool(landmarks)" in source
    assert "'protocol_v4_joint_mapping_verified': mapping_bound" in source
    assert "'standing_cross_simulator_registration_passed': (" in source
    assert "'live_camera_projection_completed': (" in source
    assert "projections_bound and projection_status_explicit" in source
    assert "'initial_fk_landmark_visibility_is_not_controller_readiness': True" in source
    assert "'visibility_required_for_controller_readiness': False" in source
    assert "'isaac_action_not_applied_by_subprobe': True" in source
    assert "controller_fk_readiness['failure_code'] = failure_code" in source
    assert "controller_fk_readiness['failure_exception_type']" in source
    assert "controller_fk_readiness['failure_detail_sha256']" in source
    assert "'controller_fk_readiness_subprobe_failed:' + failure_code" in source


def _install_preflight_module_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_GROOT_VENV_ROOT", sys.prefix)
    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_RUNTIME_DEPENDENCY_TARGET",
        "/workspace/oscar_runtime_deps",
    )
    gr00t = ModuleType("gr00t")
    gr00t.__path__ = []  # type: ignore[attr-defined]
    model = ModuleType("gr00t.model")
    model.__path__ = []  # type: ignore[attr-defined]
    n1d7_package = ModuleType("gr00t.model.gr00t_n1d7")
    n1d7_package.__path__ = []  # type: ignore[attr-defined]
    n1d7 = ModuleType("gr00t.model.gr00t_n1d7.gr00t_n1d7")

    class Qwen3Backbone:
        pass

    def get_backbone_cls(_config: object) -> type[Qwen3Backbone]:
        return Qwen3Backbone

    n1d7.get_backbone_cls = get_backbone_cls  # type: ignore[attr-defined]
    gr00t.model = model  # type: ignore[attr-defined]
    model.gr00t_n1d7 = n1d7_package  # type: ignore[attr-defined]
    n1d7_package.gr00t_n1d7 = n1d7  # type: ignore[attr-defined]

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace()

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            return cls()

    class Qwen3VLProcessor(AutoProcessor):
        pass

    transformers = ModuleType("transformers")
    transformers.AutoConfig = AutoConfig  # type: ignore[attr-defined]
    transformers.AutoProcessor = AutoProcessor  # type: ignore[attr-defined]
    transformers.Qwen3VLProcessor = Qwen3VLProcessor  # type: ignore[attr-defined]

    accelerate = ModuleType("accelerate")
    accelerate.__file__ = str(
        Path(sys.prefix) / "lib/python/site-packages/accelerate/__init__.py"
    )
    accelerate.__version__ = "test-version"  # type: ignore[attr-defined]

    for name, module in {
        "accelerate": accelerate,
        "gr00t": gr00t,
        "gr00t.model": model,
        "gr00t.model.gr00t_n1d7": n1d7_package,
        "gr00t.model.gr00t_n1d7.gr00t_n1d7": n1d7,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "cosmos-model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"local-model-bytes")

    checkpoint = tmp_path / "sonic"
    (checkpoint / "processor").mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "blueprint_original_model_name": "nvidia/Cosmos-Reason2-2B",
                "model_name": str(model),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (checkpoint / "processor/processor_config.json").write_text(
        json.dumps(
            {"processor_kwargs": {"model_name": "nvidia/Cosmos-Reason2-2B"}}
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, model


def _run_preflight() -> int:
    try:
        exec(compile(_preflight_python(), "<groot-checkpoint-preflight>", "exec"), {})
    except SystemExit as exc:
        return int(exc.code)
    raise AssertionError("checkpoint preflight did not exit")


def test_checkpoint_preflight_is_idempotent_across_two_consecutive_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_preflight_module_stubs(monkeypatch)
    checkpoint, model = _checkpoint(tmp_path)
    alias_root = tmp_path / "aliases" / "cosmos"
    artifact = tmp_path / "checkpoint-preflight.json"
    monkeypatch.setenv("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_GROOT_MODEL_ALIAS_ROOT", str(alias_root))
    monkeypatch.setenv("BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT", str(artifact))

    assert _run_preflight() == 0
    first_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    alias_name = str(alias_root.resolve() / "nvidia/Cosmos-Reason2-2B/../..")
    assert first_config["blueprint_original_model_name"] == "nvidia/Cosmos-Reason2-2B"
    assert first_config["blueprint_runtime_model_path"] == str(model.resolve())
    assert first_config["model_name"] == alias_name

    assert _run_preflight() == 0
    second_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert second_config == first_config
    assert result["status"] == "passed"
    assert result["alias_activation_state"] == "verified_existing"
    assert result["checks"]["existing_alias_binding_verified"] is True
    assert result["checks"]["groot_interpreter_prefix_exact"] is True
    assert result["checks"]["oscar_dependency_target_absent_from_pythonpath"] is True
    assert result["checks"]["oscar_dependency_target_absent_from_sys_path"] is True
    assert result["checks"]["accelerate_resolved_from_groot_venv"] is True
    assert result["accelerate_version"] == "test-version"
    assert (alias_root / "config.json").resolve() == (model / "config.json").resolve()


def test_checkpoint_preflight_rejects_oscar_dependency_target_on_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_preflight_module_stubs(monkeypatch)
    checkpoint, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "checkpoint-preflight.json"
    dependency_target = tmp_path / "oscar-runtime-deps"
    dependency_target.mkdir()
    monkeypatch.setenv("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT", str(artifact))
    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_RUNTIME_DEPENDENCY_TARGET", str(dependency_target)
    )
    monkeypatch.setenv("PYTHONPATH", str(dependency_target))

    assert _run_preflight() == 1
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["checks"]["oscar_dependency_target_absent_from_pythonpath"] is False
    assert (
        result["error"]
        == "groot_runtime_isolation_failed:"
        "oscar_dependency_target_absent_from_pythonpath"
    )


def test_checkpoint_preflight_rejects_accelerate_outside_groot_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_preflight_module_stubs(monkeypatch)
    checkpoint, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "checkpoint-preflight.json"
    accelerate = sys.modules["accelerate"]
    accelerate.__file__ = str(tmp_path / "outside/accelerate/__init__.py")
    monkeypatch.setenv("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT", str(artifact))

    assert _run_preflight() == 1
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["checks"]["accelerate_resolved_from_groot_venv"] is False
    assert (
        result["error"]
        == "groot_runtime_isolation_failed:accelerate_resolved_from_groot_venv"
    )


def test_checkpoint_preflight_fails_closed_on_changed_active_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_preflight_module_stubs(monkeypatch)
    checkpoint, _model = _checkpoint(tmp_path)
    alias_root = tmp_path / "aliases" / "cosmos"
    artifact = tmp_path / "checkpoint-preflight.json"
    monkeypatch.setenv("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_GROOT_MODEL_ALIAS_ROOT", str(alias_root))
    monkeypatch.setenv("BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT", str(artifact))

    assert _run_preflight() == 0
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_name"] = str(alias_root / "nvidia/Cosmos-Reason2-2B/../../changed")
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    assert _run_preflight() == 1
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["blockers"] == ["groot_sonic_checkpoint_preflight_failed"]
    assert "groot_sonic_checkpoint_runtime_alias_mismatch" in result["error"]


def test_checkpoint_preflight_recovers_baked_selector_then_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_preflight_module_stubs(monkeypatch)
    checkpoint, model = _checkpoint(tmp_path)
    baked_selector = tmp_path / "baked-cosmos-selector"
    baked_selector.mkdir()
    for source in model.iterdir():
        (baked_selector / source.name).symlink_to(
            source,
            target_is_directory=source.is_dir(),
        )
    baked_anchor = baked_selector / "nvidia/Cosmos-Reason2-2B"
    baked_anchor.mkdir(parents=True)
    baked_alias = str(baked_anchor / "../..")

    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_name"] = baked_alias
    config.pop("blueprint_runtime_model_path", None)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    processor_path = checkpoint / "processor/processor_config.json"
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    processor["processor_kwargs"]["model_name"] = baked_alias
    processor_path.write_text(json.dumps(processor) + "\n", encoding="utf-8")

    runtime_alias_root = tmp_path / "runtime-aliases" / "cosmos"
    artifact = tmp_path / "checkpoint-preflight.json"
    monkeypatch.setenv("BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BLUEPRINT_GROOT_MODEL_ALIAS_ROOT", str(runtime_alias_root))
    monkeypatch.setenv("BLUEPRINT_GROOT_CHECKPOINT_PREFLIGHT_ARTIFACT", str(artifact))

    assert _run_preflight() == 0
    first_config = json.loads(config_path.read_text(encoding="utf-8"))
    first_result = json.loads(artifact.read_text(encoding="utf-8"))
    runtime_alias = str(
        runtime_alias_root.resolve() / "nvidia/Cosmos-Reason2-2B/../.."
    )
    assert first_config["model_name"] == runtime_alias
    assert first_config["blueprint_runtime_model_path"] == str(
        baked_selector.resolve()
    )
    assert (
        first_config["blueprint_runtime_model_kind"]
        == "baked_selector_root_without_namespace"
    )
    assert first_result["status"] == "passed"
    assert first_result["alias_activation_state"] == "recovered_baked_selector"
    assert (runtime_alias_root / "config.json").resolve() == (
        model / "config.json"
    ).resolve()

    assert _run_preflight() == 0
    second_config = json.loads(config_path.read_text(encoding="utf-8"))
    second_result = json.loads(artifact.read_text(encoding="utf-8"))
    assert second_config == first_config
    assert second_result["status"] == "passed"
    assert second_result["alias_activation_state"] == "verified_existing"
