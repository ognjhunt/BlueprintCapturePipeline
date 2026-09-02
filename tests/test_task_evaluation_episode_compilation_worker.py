from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_episode_compilation_worker as worker
from blueprint_pipeline.control_plane_disk_budget import ControlPlaneDiskBudgetError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_episode_compilation_worker import (
    COMPILER_OUTPUT_SCHEMA_VERSION,
    process_episode_compilation_queue,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
    TaskEvaluationNativeArenaAdapterError,
)
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    ensure_scene_construction_queue_root,
)
from tests.test_task_evaluation_launch_preparation_contract import request


def _record(path: Path, contract_path: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "contract_path": contract_path,
        "uri": f"s3://blueprint-production-inputs/{path.name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "materialized_path": str(path),
        "full_byte_service_account_readback_passed": True,
    }


def _stage(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = inputs / "configured-scene-bundle.json"
    bundle.write_bytes(b'{"configured":"scene"}\n')
    robot = inputs / "robot.json"
    robot.write_bytes(b'{"robot":"team"}\n')
    bundle_record = _record(
        bundle, "scene.configured_revision.configured_scene_bundle"
    )
    robot_record = _record(robot, "robot.configuration")
    value = request()
    value["task"]["configured_scene_revision_digest"] = "sha256:" + "9" * 64
    envelope: dict[str, object] = {
        "schema_version": "task_evaluation_episode_compilation_envelope.v1",
        "compilation_id": value["preparation_id"],
        "preparation_id": value["preparation_id"],
        "run_id": value["run_id"],
        "team_namespace": value["team_namespace"],
        "expected_production_commit": value["expected_production_commit"],
        "configured_scene_revision_digest": value["task"][
            "configured_scene_revision_digest"
        ],
        "configured_scene_bundle": {
            key: bundle_record[key]
            for key in ("uri", "digest", "size_bytes")
        },
        "materialized_references": [bundle_record, robot_record],
        "request": value,
        "preparation_result_digest": "sha256:" + "8" * 64,
        "automatic_progression_required": True,
        "robot_specific_episode_packet_compiled_in_production": True,
        "customer_supplied_prebuilt_episode_packet": False,
        "production_compiler_owns_episode_packet": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    queue = ensure_scene_construction_queue_root(tmp_path / "queue")
    write_launch_preparation_record_exclusive(
        queue / "pending" / "episode.json", envelope
    )
    return queue, inputs, envelope


def test_production_compiler_joins_configured_scene_and_team_inputs(
    tmp_path: Path,
) -> None:
    queue, inputs, envelope = _stage(tmp_path)
    envelope["materialized_references"].append(
        dict(envelope["materialized_references"][1])
    )
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = queue / "pending" / "episode.json"
    pending.unlink()
    write_launch_preparation_record_exclusive(pending, envelope)
    calls = 0

    def compile_episode(*, envelope, materialized_references, output_root):
        nonlocal calls
        calls += 1
        assert "robot.configuration" in materialized_references
        packet = output_root / "native-task-arena-bundle.zip"
        packet.write_bytes(b"production-compiled-episode-packet")
        adapter_path = output_root / "adapter-result.json"
        adapter = {
            "schema_version": "task_evaluation_native_arena_adapter_result.v1",
            "status": "native_arena_adapter_materialized",
            "packet_receipt_digest": "sha256:" + "a" * 64,
            "runtime_source_receipt_digest": "sha256:" + "b" * 64,
            "result_digest": "",
        }
        adapter["result_digest"] = canonical_digest(
            adapter, digest_field="result_digest"
        )
        adapter_path.write_text(
            __import__("json").dumps(adapter) + "\n", encoding="utf-8"
        )
        result = {
            "schema_version": COMPILER_OUTPUT_SCHEMA_VERSION,
            "status": "completed",
            "run_id": envelope["run_id"],
            "configured_scene_revision_digest": envelope[
                "configured_scene_revision_digest"
            ],
            "compiled_episode_packet": {
                "format": "native_task_arena_bundle_zip",
                "path": str(packet),
                "digest": "sha256:" + hashlib.sha256(packet.read_bytes()).hexdigest(),
                "size_bytes": packet.stat().st_size,
            },
            "adapter_result": {
                "path": str(adapter_path),
                "digest": adapter["result_digest"],
                "packet_receipt_digest": adapter["packet_receipt_digest"],
                "runtime_source_receipt_digest": adapter[
                    "runtime_source_receipt_digest"
                ],
            },
            "compiled_by_production": True,
            "customer_supplied_prebuilt_episode_packet": False,
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "raw_secret_values_recorded": False,
            "compiler_output_digest": "",
        }
        result["compiler_output_digest"] = canonical_digest(
            result, digest_field="compiler_output_digest"
        )
        return result

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=envelope["expected_production_commit"],
        episode_compiler=compile_episode,
    )

    assert calls == 1
    assert run["results"][0]["status"] == "compiled_for_production_launch"
    assert run["results"][0]["compiled_by_production"] is True
    assert run["results"][0]["customer_supplied_prebuilt_episode_packet"] is False
    assert run["provider_mutation_performed"] is False


def test_compilation_rejects_conflicting_duplicate_contract_reference(
    tmp_path: Path,
) -> None:
    queue, inputs, envelope = _stage(tmp_path)
    replacement = inputs / "other-robot.json"
    replacement.write_bytes(b'{"robot":"other"}\n')
    envelope["materialized_references"].append(
        _record(replacement, "robot.configuration")
    )
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    pending = queue / "pending" / "episode.json"
    pending.unlink()
    write_launch_preparation_record_exclusive(pending, envelope)

    def forbidden(**_kwargs):
        raise AssertionError("compiler must not see conflicting inputs")

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=envelope["expected_production_commit"],
        episode_compiler=forbidden,
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "episode_compilation_materialized_reference_invalid"
    ]


def test_compilation_blocks_before_compiler_on_changed_materialized_bytes(
    tmp_path: Path,
) -> None:
    queue, inputs, envelope = _stage(tmp_path)
    (inputs / "robot.json").write_bytes(b"changed-after-readback")

    def forbidden(**_kwargs):
        raise AssertionError("compiler must not see changed inputs")

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=envelope["expected_production_commit"],
        episode_compiler=forbidden,
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "episode_compilation_materialized_reference_invalid"
    ]


def test_compilation_preserves_typed_native_adapter_blocker(
    tmp_path: Path,
) -> None:
    queue, inputs, envelope = _stage(tmp_path)

    def blocked_adapter(**_kwargs):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_identity_mismatch"
        )

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=tmp_path / "outputs",
        source_commit=envelope["expected_production_commit"],
        episode_compiler=blocked_adapter,
    )

    assert run["results"][0]["status"] == "blocked"
    assert run["results"][0]["blockers"] == [
        "task_evaluation_adapter_bundle_identity_mismatch"
    ]


def test_compilation_reports_errno_and_removes_partial_output(
    tmp_path: Path,
) -> None:
    queue, inputs, envelope = _stage(tmp_path)
    outputs = tmp_path / "outputs"

    def disk_full(*, output_root, **_kwargs):
        (Path(output_root) / "partial.bin").write_bytes(b"partial")
        raise OSError(28, "No space left on device", "/secret/host/path")

    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=outputs,
        source_commit=envelope["expected_production_commit"],
        episode_compiler=disk_full,
    )

    result = run["results"][0]
    assert result["status"] == "blocked"
    assert result["blockers"] == ["episode_compilation_failed:OSError:errno_28"]
    assert "/secret/host/path" not in json.dumps(result)
    assert not (outputs / envelope["compilation_id"]).exists()


def test_compilation_refuses_low_disk_before_creating_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, inputs, envelope = _stage(tmp_path)
    outputs = tmp_path / "outputs"

    def refuse(*_args, **_kwargs):
        raise ControlPlaneDiskBudgetError(
            "control_plane_disk_budget_exceeded:episode_compilation:"
            "need_bytes=6:available_bytes=0:free_bytes=8:"
            "floor_bytes=8:reserved_bytes=0"
        )

    monkeypatch.setattr(worker, "reserve_control_plane_disk", refuse)
    run = process_episode_compilation_queue(
        queue_root=queue,
        input_root=inputs,
        output_root=outputs,
        source_commit=envelope["expected_production_commit"],
        episode_compiler=lambda **_kwargs: pytest.fail("compiler must not run"),
        disk_reservation_root=tmp_path / "reservations",
    )

    assert run["results"][0]["blockers"] == [
        "control_plane_disk_budget_exceeded:episode_compilation:"
        "need_bytes=6:available_bytes=0:free_bytes=8:"
        "floor_bytes=8:reserved_bytes=0"
    ]
    assert not (outputs / envelope["compilation_id"]).exists()
