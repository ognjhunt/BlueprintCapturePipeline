"""Capacity may shrink between observing a refusal and reserving its successor."""
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_capacity_controller as controller
from blueprint_pipeline import control_plane_disk_budget as disk
from blueprint_pipeline import task_evaluation_scene_capacity_recovery as recovery


def measured_capacity(tmp_path, monkeypatch):
    cpu = (tmp_path / "cpu").resolve()
    output = (tmp_path / "output").resolve()
    cpu.mkdir()
    output.mkdir()
    available = {"cpu": 1400, "output": 1000}

    def whole_chain(*_args, **_kwargs):
        return {
            "required_workspace_bytes": 1000,
            "measurement": {
                "status": "measured", "floor_bytes": 100,
                "reserved_bytes": 200, "free_bytes": available["cpu"],
            },
        }

    monkeypatch.setattr(controller, "whole_chain_admission", whole_chain)
    monkeypatch.setitem(disk.ROLE_FOOTPRINT_BYTES, "semantic_pretraining", 600)
    monkeypatch.setattr(recovery.shutil, "disk_usage", lambda _path: SimpleNamespace(free=available["output"]))
    producer = {"provider_output_disk_requirements": {"required_free_bytes_before_download": 900}}
    observation = {"values": {"bundle": {"bundle_size_bytes": 100}, "result": producer}}
    config = {"factory_output_root": str(cpu), "launch_execution_root": str(output)}
    return observation, config, available


@pytest.mark.parametrize("which", ["cpu", "output"])
def test_either_capacity_floor_refuses_even_when_other_volume_has_room(tmp_path, monkeypatch, which):
    observation, config, available = measured_capacity(tmp_path, monkeypatch)
    admitted = recovery.capacity_admission(observation, config, 100)
    assert admitted["status"] == "admitted"
    assert admitted["cpu_required_free_bytes"] == 1400  # Includes existing reservations and the new bundle.
    assert admitted["output_required_free_bytes"] == 1000
    available[which] -= 1
    assert recovery.capacity_admission(observation, config, 100)["status"] == "waiting_for_capacity"


def test_saved_admission_cannot_authorize_after_space_disappears(tmp_path, monkeypatch):
    observation, config, available = measured_capacity(tmp_path, monkeypatch)
    admission = recovery.capacity_admission(observation, config, 100)
    reference = {"path": "/retained/result.json", "sha256": "sha256:" + "a" * 64, "size_bytes": 1}
    failure = {"construction_records": {"result": reference,
               "factory": {"path": str(tmp_path / "cpu/intent/attempt/factory.json")},
               "profile": {"path": str(tmp_path / "output/launch/launch_profile.json")}}, "producer_result": reference,
               "capacity_admission": admission}
    monkeypatch.setattr(recovery, "validate_source", lambda *_args, **_kwargs: observation["values"])
    available["output"] = 999
    with pytest.raises(ValueError, match="capacity_not_recovered"):
        recovery.validate_capacity_failure(failure, observation["values"]["result"], {}, 101)


def test_old_capacity_measurement_requires_fresh_observation(tmp_path, monkeypatch):
    observation, config, _available = measured_capacity(tmp_path, monkeypatch)
    reference = {"path": "/retained/result.json", "sha256": "sha256:" + "a" * 64, "size_bytes": 1}
    failure = {"construction_records": {"result": reference,
               "factory": {"path": str(tmp_path / "cpu/intent/attempt/factory.json")},
               "profile": {"path": str(tmp_path / "output/launch/launch_profile.json")}}, "producer_result": reference,
               "capacity_admission": recovery.capacity_admission(observation, config, 100)}
    monkeypatch.setattr(recovery, "validate_source", lambda *_args, **_kwargs: observation["values"])
    with pytest.raises(ValueError, match="capacity_admission_stale"):
        recovery.validate_capacity_failure(failure, observation["values"]["result"], {}, 401)
