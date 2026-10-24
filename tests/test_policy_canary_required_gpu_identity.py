"""Reject a different physical GPU before cells or learned clients are started."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline.native_task_arena_policy_canary_session import PolicyCanarySessionError
from tests.test_native_task_arena_policy_canary_session import _authority

GPU = "GPU-00000000-0000-0000-0000-000000000001"
OTHER = "GPU-00000000-0000-0000-0000-000000000002"


def test_existing_authority_does_not_add_a_probe(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unrequested GPU probe")

    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    assert worker._required_gpu_identity_preflight({}, tmp_path) is None


@pytest.mark.parametrize("value", ["", "GPU-test", True, "MIG-" + GPU, None])
def test_required_uuid_is_validated_before_publication(tmp_path, value):
    if value is None:
        from blueprint_pipeline.native_task_arena_policy_canary_session import (
            validate_session_authority,
        )

        authority, _ = _authority(tmp_path)
        authority["required_gpu_uuid"] = None
        with pytest.raises(PolicyCanarySessionError, match="required_gpu_uuid_invalid"):
            validate_session_authority(authority)
    else:
        with pytest.raises(PolicyCanarySessionError, match="required_gpu_uuid_invalid"):
            _authority(tmp_path, required_gpu_uuid=value)


def test_matching_physical_card_is_recorded(tmp_path, monkeypatch):
    authority, _ = _authority(tmp_path, required_gpu_uuid=GPU)
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=GPU + "\n")
    )
    result = worker._required_gpu_identity_preflight(authority, tmp_path)
    assert result["passed"] is True
    assert result["observed_gpu_uuids"] == [GPU]
    assert result["candidate_policy_queried"] is False
    assert result["media_gap"] is None


@pytest.mark.parametrize("stdout", [OTHER + "\n", GPU + "\n" + OTHER + "\n", ""])
def test_wrong_or_ambiguous_card_refuses_parent_before_spawning_cells(
    tmp_path, monkeypatch, stdout
):
    authority, inputs = _authority(tmp_path, required_gpu_uuid=GPU)
    runtime = tmp_path / "runtime"
    directory = runtime / "runtime_inputs"
    directory.mkdir(parents=True)
    (directory / "policy_canary_runtime_inputs.json").write_text(json.dumps(inputs))
    (directory / "policy_canary_session_authority.json").write_text(json.dumps(authority))
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=stdout))

    def forbidden(**kwargs):
        raise AssertionError("mismatched GPU reached a cell or learned client")

    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="required_physical_gpu_identity_mismatch"):
        worker._run_isolated_cell_processes(
            runtime_root=runtime, output_root=output, run_cell_process=forbidden
        )
    result = json.loads((output / "policy_canary_required_gpu_identity.v1.json").read_text())
    assert result["passed"] is False
    assert result["media_gap"]["type"] == "before_first_observation"
    assert result["candidate_policy_queried"] is False


def test_failed_probe_is_a_typed_refusal(tmp_path, monkeypatch):
    authority, _ = _authority(tmp_path, required_gpu_uuid=GPU)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(worker.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="required_physical_gpu_identity_mismatch"):
        worker._required_gpu_identity_preflight(authority, tmp_path)
    result = json.loads((tmp_path / "policy_canary_required_gpu_identity.v1.json").read_text())
    assert result["probe_error"] == "CalledProcessError"


def test_static_off_host_preflight_stays_cpu_only_and_provider_preflight_checks_gpu(
    tmp_path, monkeypatch
):
    authority, inputs = _authority(tmp_path, required_gpu_uuid=GPU)
    runtime = tmp_path / "runtime"
    files = {
        "runtime_inputs/policy_canary_runtime_inputs.json": inputs,
        "runtime_inputs/policy_canary_session_authority.json": authority,
        "adp_arena_provider_manifest.json": {},
        "native_task_packet/native_task_arena_scene_plan.v1.json": {},
        "native_task_packet/native_task_arena_packet_request.v1.json": {},
        "runtime_inputs/native_task_arena_construction_result.v1.json": {},
        **{
            f"runtime_inputs/policy_execution_spec.{candidate}.json": {}
            for candidate in inputs["candidate_ids"]
        },
    }
    for name, value in files.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(
        worker,
        "preflight_policy_canary_static_inputs",
        lambda **kw: {"status": "passed", "blockers": []},
    )
    monkeypatch.setattr(worker, "_validate_provider_manifest", lambda *a, **kw: None)

    def forbidden(*a, **kw):
        raise AssertionError("host static preflight queried GPU")

    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    assert worker.preflight_policy_canary_runtime_directory(runtime)["status"] == "passed"
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=OTHER + "\n")
    )
    with pytest.raises(RuntimeError, match="required_physical_gpu_identity_mismatch"):
        worker.preflight_policy_canary_runtime_directory(
            runtime, physical_gpu_output_root=tmp_path / "out"
        )


def test_rendered_provider_entrypoint_activates_guard_before_checkpoint_provisioning(
    tmp_path, monkeypatch
):
    import sys
    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint
    from blueprint_pipeline.policy_canary_output_members import POLICY_CANARY_OUTPUT_CONTRACT

    script = _entrypoint()
    marker = "<<'PYINPUT'\n"
    start = script.index(marker) + len(marker)
    block = script[start : script.index("\nPYINPUT", start)]
    assert start < script.index('bash "$RUNTIME_DIR/adp009d_policy_provisioning.pi05_droid.sh"')
    authority, _ = _authority(tmp_path, required_gpu_uuid=GPU)
    output = tmp_path / "out"
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("OUT_DIR", str(output))
    monkeypatch.setattr(
        worker.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=OTHER + "\n")
    )

    def actual_gpu_check(runtime, *, physical_gpu_output_root):
        assert physical_gpu_output_root == output
        worker._required_gpu_identity_preflight(authority, physical_gpu_output_root)
        raise AssertionError("mismatched GPU passed checkpoint boundary")

    monkeypatch.setitem(
        sys.modules,
        "adp_arena_provider_runner",
        SimpleNamespace(preflight_policy_canary_runtime_directory=actual_gpu_check),
    )
    with pytest.raises(RuntimeError, match="required_physical_gpu_identity_mismatch"):
        exec(compile(block, "<sealed-provider-input-preflight>", "exec"), {})
    assert POLICY_CANARY_OUTPUT_CONTRACT.needed("policy_canary_required_gpu_identity.v1.json")
    assert POLICY_CANARY_OUTPUT_CONTRACT.needed(
        "cell_runs/00/policy_canary_required_gpu_identity.v1.json"
    )
