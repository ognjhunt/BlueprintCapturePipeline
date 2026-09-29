"""Provider legacy calls must not require host-private registered authority code."""

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("caller", ["runtime", "supervisor"])
def test_actual_provider_default_none_never_imports_host_registration(
    caller, tmp_path, monkeypatch
):
    from blueprint_pipeline import native_g1_runtime_assembly as runtime
    from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in ("control_plane_lane_experiment_consumer", "native_g1_registered_containment"):
            pytest.fail("provider legacy call imported host-only registration adapter")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    if caller == "runtime":
        with pytest.raises(ValueError, match="g1_supervised_episode_scene_or_candidate_invalid"):
            runtime.run_g1_supervised_built_scene_episode(
                built=SimpleNamespace(plan={}),
                candidate_id="candidate",
                preflight_inputs={},
                python_executable=Path("/python"),
                port=8000,
                device="cuda:0",
                max_steps=1,
                output_dir=tmp_path / "ordinary",
                to_tensor=lambda v: v,
                make_action_tensor=lambda **kw: kw,
            )
    else:
        with pytest.raises(ValueError):
            supervisor.start_g1_policy_server(
                preflight_inputs={},
                python_executable=Path("/python"),
                port=0,
                device="cuda:0",
                log_path=tmp_path / "ordinary.log",
            )


@pytest.mark.parametrize("caller", ["runtime", "supervisor"])
def test_provider_native_call_without_use_still_refuses_reserved_target(caller, tmp_path):
    from blueprint_pipeline import native_g1_runtime_assembly as runtime
    from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor

    target = tmp_path / "g1" / ("registered-" + "a" * 32)
    if caller == "runtime":
        with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
            runtime.run_g1_supervised_built_scene_episode(
                built=SimpleNamespace(plan={}),
                candidate_id="candidate",
                preflight_inputs={},
                python_executable=Path("/python"),
                port=8000,
                device="cuda:0",
                max_steps=1,
                output_dir=target / "output",
                to_tensor=lambda v: v,
                make_action_tensor=lambda **kw: kw,
            )
    else:
        with pytest.raises(ValueError, match="experiment_consumer_authority_required"):
            supervisor.start_g1_policy_server(
                preflight_inputs={},
                python_executable=Path("/python"),
                port=0,
                device="cuda:0",
                log_path=target / "log",
            )
