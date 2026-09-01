from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
    TaskEvaluationNativeArenaAdapterError,
    build_task_evaluation_adapter_bundle,
    build_task_evaluation_runtime_source_bundle,
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
    build_task_evaluation_runtime_source_bundle(
        source_root=runtime_receipt.parent,
        output_path=runtime_bundle,
        expected_production_commit=value["expected_production_commit"],
        runtime_identity=value["runtime"]["identity"],
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


def test_runtime_source_bundle_is_prelaunch_reusable_across_revision_digest(
    tmp_path: Path,
) -> None:
    value, configured, _construction_bundle, runtime_bundle = _bundles(tmp_path)
    with zipfile.ZipFile(runtime_bundle) as archive:
        manifest = json.loads(
            archive.read("task_evaluation_adapter_bundle_manifest.v1.json")
        )
    assert manifest["identity_bindings"] == {
        "expected_production_commit": value["expected_production_commit"],
        "runtime": value["runtime"]["identity"],
    }

    future_revision = copy.deepcopy(configured)
    future_revision["configuration_run_id"] = "future-configured-scene-run"
    future_revision["revision_digest"] = canonical_digest(
        future_revision, digest_field="revision_digest"
    )
    future_request = copy.deepcopy(value)
    future_request["task"]["configured_scene_revision_digest"] = future_revision[
        "revision_digest"
    ]
    future_construction = tmp_path / "future-construction-packet.zip"
    source_packet = _packet(tmp_path / "future-packet", scene_id="public-scene-17")
    task_object = source_packet / "assets" / "task_object.usd"
    future_revision["replacement"]["asset"] = _identity(task_object)
    future_revision["revision_digest"] = canonical_digest(
        future_revision, digest_field="revision_digest"
    )
    future_request["task"]["configured_scene_revision_digest"] = future_revision[
        "revision_digest"
    ]
    build_task_evaluation_adapter_bundle(
        source_root=source_packet,
        output_path=future_construction,
        request=future_request,
        role="construction_packet",
    )

    result = materialize_native_arena_adapter(
        request=future_request,
        compiled_episode_packet_path=future_construction,
        compiled_episode_packet_reference=_identity(future_construction),
        configured_revision=future_revision,
        runtime_source_bundle_path=runtime_bundle,
        output_root=tmp_path / "future-adapter-output",
    )

    assert result["status"] == "native_arena_adapter_materialized"


def test_prelaunch_runtime_source_builder_needs_no_future_request(
    tmp_path: Path,
) -> None:
    runtime_receipt = _runtime_source_packet(tmp_path)
    destination = tmp_path / "prelaunch-runtime-source.zip"

    receipt = build_task_evaluation_runtime_source_bundle(
        source_root=runtime_receipt.parent,
        output_path=destination,
        expected_production_commit="a" * 40,
        runtime_identity={"id": "native-arena", "version": "isaac-2026-1"},
    )

    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(
            archive.read("task_evaluation_adapter_bundle_manifest.v1.json")
        )
    assert receipt["status"] == "built"
    assert manifest["identity_bindings"] == {
        "expected_production_commit": "a" * 40,
        "runtime": {"id": "native-arena", "version": "isaac-2026-1"},
    }


def test_runtime_source_bundle_forces_zip64_for_every_payload_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runtime-source"
    source.mkdir()
    (source / "runtime.bin").write_bytes(b"runtime")
    observed: list[bool] = []
    original_open = zipfile.ZipFile.open

    def tracked_open(
        archive: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        mode: str = "r",
        pwd: bytes | None = None,
        *,
        force_zip64: bool = False,
    ):
        if (
            mode == "w"
            and isinstance(name, zipfile.ZipInfo)
            and name.filename.startswith("payload/")
        ):
            observed.append(force_zip64)
        return original_open(
            archive,
            name,
            mode=mode,
            pwd=pwd,
            force_zip64=force_zip64,
        )

    monkeypatch.setattr(zipfile.ZipFile, "open", tracked_open)

    receipt = build_task_evaluation_runtime_source_bundle(
        source_root=source,
        output_path=tmp_path / "runtime-source.zip",
        expected_production_commit="a" * 40,
        runtime_identity={"id": "native-arena", "version": "isaac-2026-1"},
    )

    assert receipt["status"] == "built"
    assert observed == [True]


@pytest.mark.parametrize(
    ("commit", "runtime_identity"),
    [
        ("not-a-commit", {"id": "native-arena", "version": "v1"}),
        ("a" * 40, {"id": "native-arena"}),
        ("a" * 40, {"id": "native arena", "version": "v1"}),
    ],
)
def test_prelaunch_runtime_source_builder_rejects_unbound_identity(
    tmp_path: Path, commit: str, runtime_identity: dict[str, str]
) -> None:
    runtime_receipt = _runtime_source_packet(tmp_path)

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_runtime_source_bundle_identity_invalid",
    ):
        build_task_evaluation_runtime_source_bundle(
            source_root=runtime_receipt.parent,
            output_path=tmp_path / "runtime-source.zip",
            expected_production_commit=commit,
            runtime_identity=runtime_identity,
        )


def test_runtime_source_bundle_rejects_different_production_commit(
    tmp_path: Path,
) -> None:
    value, configured, _construction_bundle, runtime_bundle = _bundles(tmp_path)
    changed = copy.deepcopy(value)
    changed["expected_production_commit"] = "b" * 40
    changed_revision = copy.deepcopy(configured)
    changed_revision["source_commit"] = changed["expected_production_commit"]
    changed_revision["revision_digest"] = canonical_digest(
        changed_revision, digest_field="revision_digest"
    )
    changed["task"]["configured_scene_revision_digest"] = changed_revision[
        "revision_digest"
    ]
    changed_construction = tmp_path / "changed-construction-packet.zip"
    source_packet = _packet(tmp_path / "changed-packet", scene_id="public-scene-17")
    task_object = source_packet / "assets" / "task_object.usd"
    changed_revision["replacement"]["asset"] = _identity(task_object)
    changed_revision["revision_digest"] = canonical_digest(
        changed_revision, digest_field="revision_digest"
    )
    changed["task"]["configured_scene_revision_digest"] = changed_revision[
        "revision_digest"
    ]
    build_task_evaluation_adapter_bundle(
        source_root=source_packet,
        output_path=changed_construction,
        request=changed,
        role="construction_packet",
    )

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_bundle_manifest_invalid",
    ):
        materialize_native_arena_adapter(
            request=changed,
            compiled_episode_packet_path=changed_construction,
            compiled_episode_packet_reference=_identity(changed_construction),
            configured_revision=changed_revision,
            runtime_source_bundle_path=runtime_bundle,
            output_root=tmp_path / "changed-adapter-output",
        )

    changed["execution_adapter"]["runtime_source_implementation_commit"] = value[
        "expected_production_commit"
    ]
    compatible = materialize_native_arena_adapter(
        request=changed,
        compiled_episode_packet_path=changed_construction,
        compiled_episode_packet_reference=_identity(changed_construction),
        configured_revision=changed_revision,
        runtime_source_bundle_path=runtime_bundle,
        output_root=tmp_path / "compatible-adapter-output",
    )
    assert compatible["status"] == "native_arena_adapter_materialized"


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
