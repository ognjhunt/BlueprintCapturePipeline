"""ADP-009D: the load scene starts with real intake and publication joins."""

from __future__ import annotations

import json
import subprocess
from collections import namedtuple

import pytest

from scripts.control_plane_concurrency_scene import (
    advance_fixture_intake, advance_fixture_preparation, advance_fixture_configuration,
)


@pytest.mark.slow
def test_intake_factory_publication_and_owned_queue_share_actual_identity(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    objects = tmp_path / "objects"
    objects.mkdir()
    result = advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects,
                                    source_commit=source)
    assert result["claim_ceiling"] == "development_only"
    assert result["source_commit"] == source
    assert result["progression"]["results"][0]["status"] == "running", result
    assert result["publication"]["status"] == "published_and_read_back"
    assert result["publication"]["raw_source_uploaded"] is False
    assert result["publication"]["provider_allocated"] is False
    request = json.loads(result["request_path"].read_text())
    assert request["expected_production_commit"] == source
    assert request["scene_intent_digest"] == result["intent"]["intent_digest"]
    pending = list((result["preparation_queue"] / "pending").glob("*.json"))
    assert len(pending) == 1
    envelope = json.loads(pending[0].read_text())
    assert envelope["request"] == request
    assert result["publication"]["manifest_sha256"] == result["factory"]["submission_manifest"]["sha256"]
    assert result["object_bytes_uploaded"] > 0


def test_intake_fixture_cannot_replace_the_actual_checkout_validator(tmp_path):
    objects = tmp_path / "objects"
    objects.mkdir()
    with pytest.raises(ValueError, match="harness_checkout_source_mismatch"):
        advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects,
                               source_commit="0" * 40)
    assert not (tmp_path / "scene").exists()


@pytest.mark.slow
def test_preparation_reads_real_objects_and_seals_same_scene_construction(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    usage = namedtuple("Usage", "total used free")(200 * 1024**3, 50 * 1024**3, 150 * 1024**3)

    def measured_test_reservation(role, **kwargs):
        # This local contract test supplies disk capacity, retaining the real
        # ledger/reservation implementation. Live harness code has no seam.
        return reserve_control_plane_disk(role, **kwargs, disk_usage=lambda _: usage)

    monkeypatch.setattr(worker, "reserve_control_plane_disk", measured_test_reservation)
    from blueprint_pipeline import control_plane_disk_budget as budget
    real_headroom = budget.disk_headroom
    monkeypatch.setattr(budget, "disk_headroom",
        lambda **kwargs: real_headroom(**kwargs, disk_usage=lambda _: usage))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    objects = tmp_path / "objects"
    objects.mkdir()
    first = advance_fixture_intake(host_root=tmp_path / "scene", object_root=objects, source_commit=source)
    result = advance_fixture_preparation(intake=first, object_root=objects,
                                        reservation_root=tmp_path / "reservations", pins_root=tmp_path / "pins")
    assert result["run"]["results"][0]["status"] == "queued_for_production_scene_configuration", result
    envelope = result["construction_envelope"]
    assert envelope["request"] == json.loads(first["request_path"].read_text())
    assert envelope["envelope_digest"] == result["run"]["results"][0]["construction_queue_envelope_digest"]
    assert all(row["full_byte_service_account_readback_passed"] for row in envelope["materialized_references"])
    assert envelope["render_inputs_result"]["fixture_provider"] is True
    assert not list((tmp_path / "reservations").glob("*.json"))
    configured = advance_fixture_configuration(preparation=result, object_root=objects,
                                               output_root=tmp_path / "scene/configured")
    assert configured["publication"]["status"] == "configured_scene_published"
    revision = configured["revision"]
    assert revision["configuration_run_id"] == envelope["run_id"]
    assert revision["scene_identity"] == envelope["request"]["scene"]["identity"]
    assert revision["task_template"]["identity"] == envelope["request"]["task"]["identity"]
    assert configured["terminal"]["configured_scene_revision_digest"] == revision["revision_digest"]
    assert configured["terminal"]["scene_construction_queue_finalization"]["queue_state"] == "completed"
    assert configured["terminal"]["fixture_provider"] is True
    from scripts.control_plane_concurrency_readiness import materialize_fixture_placement
    placement = materialize_fixture_placement(configured=configured, object_root=objects,
                                             output_root=tmp_path / "placement")
    assert placement["trajectory"]["source_plan_digest"] == placement["plan"]["plan_digest"]
    assert placement["candidate"]["configured_scene_revision_digest"] == revision["revision_digest"]
    assert placement["candidate"]["position_ik_qualified_on_cpu"] is True
    assert placement["candidate"]["robot_base_qualified"] is False
    assert placement["receipt"]["physical_execution_authorized"] is False
    from scripts.control_plane_concurrency_readiness import stage_fixture_episode_preparation
    from tests.test_native_task_arena_bundle import _runtime_source_packet
    from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import build_task_evaluation_runtime_source_bundle
    from tests.test_task_evaluation_configured_controls_progression import _runtime
    from scripts.control_plane_concurrency_provider import fixture_publisher
    runtime = _runtime()
    fixture_source = _runtime_source_packet(tmp_path)
    bundle = tmp_path / "runtime-source.zip"
    build_task_evaluation_runtime_source_bundle(source_root=fixture_source.parent, output_path=bundle,
        expected_production_commit=source, runtime_identity=runtime["runtime"]["identity"])
    publish = fixture_publisher(objects)
    runtime["execution_adapter"]["runtime_source_bundle"] = {key: value for key, value in
        publish(path=bundle, object_name="runtime-source.zip").items() if key in {"uri", "digest", "size_bytes"}}
    health = tmp_path / "fixture-health.json"
    health.write_text('{"fixture_runtime":true}')
    runtime["runtime"]["health_protocol"] = {key: value for key, value in
        publish(path=health, object_name="health.json").items() if key in {"uri", "digest", "size_bytes"}}
    runtime["runtime"]["mounts"][0]["source"] = revision["configured_scene_bundle"]
    with __import__("scripts.control_plane_concurrency_scene", fromlist=["fixture_environment"]).fixture_environment(first["environment"]):
        episode = stage_fixture_episode_preparation(configured=configured, placement=placement,
            object_root=objects, runtime_binding=runtime, output_root=tmp_path / "readiness",
            queue_root=first["preparation_queue"], source_commit=source)
    assert episode["status"] == "episode_preparation_queued"
    assert episode["configured_scene_revision_digest"] == revision["revision_digest"]
    ready = advance_fixture_preparation(intake=first, object_root=objects,
        reservation_root=tmp_path / "reservations", pins_root=tmp_path / "pins")
    assert ready["run"]["results"][0]["status"] == "queued_for_production_episode_compilation", ready
    assert ready["compilation_envelope"]["configured_scene_revision_digest"] == revision["revision_digest"]
    from blueprint_pipeline import task_evaluation_episode_compilation_worker as compilation
    monkeypatch.setattr(compilation, "reserve_control_plane_disk", measured_test_reservation)
    with __import__("scripts.control_plane_concurrency_scene", fromlist=["fixture_environment"]).fixture_environment(first["environment"]):
        compiled = compilation.process_episode_compilation_queue(queue_root=ready["compilation_queue"],
            input_root=tmp_path / "prepared-references", output_root=tmp_path / "compiled",
            source_commit=source, disk_reservation_root=tmp_path / "reservations", storage_pins_root=tmp_path / "pins")
    if compiled["results"][0]["status"] != "compiled_for_production_launch":
        from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import compile_native_arena_episode
        envelope = ready["compilation_envelope"]
        (tmp_path / "failed-compiler-diagnostic").mkdir()
        try:
            compile_native_arena_episode(envelope=envelope,
                materialized_references={row["contract_path"]: row for row in envelope["materialized_references"]},
                output_root=tmp_path / "failed-compiler-diagnostic")
        except ValueError as exc:
            raise AssertionError(getattr(exc, "errors", str(exc))) from exc
    assert compiled["results"][0]["status"] == "compiled_for_production_launch", json.dumps(compiled, indent=2)
    from scripts.control_plane_concurrency_provider import fixture_scene_artifacts
    from blueprint_pipeline.task_evaluation_scene_configuration_publication import TaskEvaluationSceneConfigurationPublicationError

    def corrupt_provider(**kwargs):
        rows = fixture_scene_artifacts(**kwargs)
        path = kwargs["output_root"] / "collision.usda"
        path.write_text("changed provider output")
        return rows

    with pytest.raises(TaskEvaluationSceneConfigurationPublicationError,
                       match="scene_configuration_publication_artifact_invalid:configured_collision_without_source_object"):
        advance_fixture_configuration(preparation=result, object_root=objects,
                                      output_root=tmp_path / "failed-configuration", provider=corrupt_provider)
    assert (tmp_path / "failed-configuration/fixture-provider/collision.usda").read_text() == "changed provider output"
