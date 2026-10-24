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
