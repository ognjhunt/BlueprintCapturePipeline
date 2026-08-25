from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
    TaskEvaluationNativeArenaAdapterError,
    build_task_evaluation_adapter_bundle,
    materialize_native_arena_adapter,
)
from tests.test_native_task_arena_bundle import _packet, _runtime_source_packet
from tests.test_task_evaluation_launch_preparation_contract import request
from tests.test_task_evaluation_configured_scene_revision import revision


def _identity(path: Path) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/{path.name}",
        "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _bundles(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path]:
    value = request()
    value["scene"]["identity"] = {"id": "public-scene-17", "version": "v1"}
    value["task"]["identity"] = {
        "id": "task-public-scene-17",
        "version": "v1",
    }
    packet = _packet(tmp_path, scene_id="public-scene-17")
    task_object = packet / "assets" / "task_object.usd"
    value["task"]["subject"]["identity"] = {
        "id": "admitted-can",
        "version": "v1",
    }
    configured = revision()
    configured["team_namespace"] = value["team_namespace"]
    configured["scene_identity"] = value["scene"]["identity"]
    configured["source_commit"] = value["expected_production_commit"]
    configured["replacement"]["identity"] = value["task"]["subject"][
        "identity"
    ]
    configured["replacement"]["asset"] = _identity(task_object)
    configured["task_template"]["identity"] = value["task"]["identity"]
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    value["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]
    runtime_receipt = _runtime_source_packet(tmp_path)
    construction_bundle = tmp_path / "construction-packet.zip"
    runtime_bundle = tmp_path / "runtime-source.zip"
    build_task_evaluation_adapter_bundle(
        source_root=packet,
        output_path=construction_bundle,
        request=value,
        role="construction_packet",
    )
    build_task_evaluation_adapter_bundle(
        source_root=runtime_receipt.parent,
        output_path=runtime_bundle,
        request=value,
        role="runtime_source",
    )
    value["execution_adapter"]["runtime_source_bundle"] = _identity(
        runtime_bundle
    )
    return (
        value,
        configured,
        construction_bundle,
        runtime_bundle,
    )


def test_builds_and_materializes_scene_neutral_native_arena_bundles(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(tmp_path)

    result = materialize_native_arena_adapter(
        request=value,
        compiled_episode_packet_path=construction_bundle,
        compiled_episode_packet_reference=_identity(construction_bundle),
        configured_revision=configured,
        runtime_source_bundle_path=runtime_bundle,
        output_root=tmp_path / "adapter-output",
    )

    assert result["status"] == "native_arena_adapter_materialized"
    assert result["source_commit"] == value["expected_production_commit"]
    assert result["provider_mutation_performed"] is False
    assert result["catalog_mutation_performed"] is False
    assert result["paid_execution_requested"] is False
    assert Path(result["packet_root"]).is_dir()
    assert Path(result["runtime_source_receipt"]).is_file()


def test_adapter_refuses_bundle_bytes_that_do_not_match_website_request(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(tmp_path)
    changed_reference = _identity(construction_bundle)
    changed_reference["digest"] = "sha256:" + "0" * 64

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_bundle_source_identity_mismatch",
    ):
        materialize_native_arena_adapter(
            request=value,
            compiled_episode_packet_path=construction_bundle,
            compiled_episode_packet_reference=changed_reference,
            configured_revision=configured,
            runtime_source_bundle_path=runtime_bundle,
            output_root=tmp_path / "adapter-output",
        )


def test_adapter_manifest_binds_independent_scene_identity(tmp_path: Path) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(tmp_path)
    changed = copy.deepcopy(value)
    changed["scene"]["identity"] = {"id": "different-scene", "version": "v1"}

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_configured_revision_binding_mismatch",
    ):
        materialize_native_arena_adapter(
            request=changed,
            compiled_episode_packet_path=construction_bundle,
            compiled_episode_packet_reference=_identity(construction_bundle),
            configured_revision=configured,
            runtime_source_bundle_path=runtime_bundle,
            output_root=tmp_path / "adapter-output",
        )


def test_adapter_refuses_task_subject_bytes_or_strategy_not_in_packet(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(tmp_path)

    changed = copy.deepcopy(value)
    changed["task"]["strategy"] = "pick_and_place"
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_task_subject_binding_mismatch",
    ):
        materialize_native_arena_adapter(
            request=changed,
            compiled_episode_packet_path=construction_bundle,
            compiled_episode_packet_reference=_identity(construction_bundle),
            configured_revision=configured,
            runtime_source_bundle_path=runtime_bundle,
            output_root=tmp_path / "adapter-output",
        )
