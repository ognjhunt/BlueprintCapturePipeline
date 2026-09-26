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
    _verify_task_subject_binding,
    build_task_evaluation_adapter_bundle,
    build_task_evaluation_runtime_source_bundle,
    materialize_native_arena_adapter,
    _manifest_from_archive,
)
from tests.test_native_task_arena_bundle import _packet, _runtime_source_packet
from tests.test_task_evaluation_launch_preparation_contract import request
from tests.test_task_evaluation_configured_scene_revision import revision


def _passive_joint_overlay_binding(tmp_path: Path) -> tuple[dict, dict, Path, dict]:
    from pxr import Usd, UsdGeom, UsdPhysics

    from blueprint_pipeline.native_task_arena_runtime import author_passive_joint_friction_overlay

    source = tmp_path / "sealed-cabinet.usda"
    stage = Usd.Stage.CreateNew(str(source))
    cabinet = UsdGeom.Xform.Define(stage, "/Cabinet")
    stage.SetDefaultPrim(cabinet.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/Cabinet/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    anchor = UsdPhysics.FixedJoint.Define(stage, "/Cabinet/fixed_base_anchor")
    anchor.GetBody1Rel().SetTargets([body.GetPath()])
    joint_path = "/Cabinet/joints/middle_drawer_joint"
    UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    stage.GetRootLayer().Save()

    packet = _packet(tmp_path, scene_id="public-scene-17")
    staged = packet / "assets/task_object.usd"
    friction = author_passive_joint_friction_overlay(
        source, staged, joint_prim_path=joint_path)
    assert friction is not None
    adaptation = {
        "adaptation": "estimated_passive_joint_friction_overlay",
        "fixed_base_body_prim_path": "/Cabinet/body",
        "candidate_bytes_modified": False,
        "derived_from_sha256": _identity(source)["digest"],
        "passive_joint_friction": friction,
    }
    value = request()
    value["task"]["kind"] = "articulated_manipulation"
    value["task"]["strategy"] = "articulated_open_close"
    value["task"]["subject"]["identity"] = {"id": "admitted-can", "version": "v1"}
    value["task"]["configured_scene_revision_digest"] = "sha256:" + "a" * 64
    configured = revision()
    configured["revision_digest"] = value["task"]["configured_scene_revision_digest"]
    configured["replacement"]["identity"] = value["task"]["subject"]["identity"]
    configured["replacement"]["asset"] = _identity(source)

    contract_path = packet / "native_task_runtime_contract.v1.json"
    contract = json.loads(contract_path.read_text())
    contract["task_kind"] = "articulated_open_close"
    contract["task_spec"] = {
        "manipulation_strategy": "articulated_open_close",
        "subject_asset_id": "admitted_can",
        "source_subject_identity": "admitted-can",
    }
    contract["task_subject_asset_id"] = "admitted_can"
    contract["objects"] = [{
        "asset_id": "admitted_can", "task_subject": True,
        "object_type": "ARTICULATION", "sha256": _identity(staged)["digest"],
        "articulation_adaptation": adaptation,
    }]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    receipt = json.loads((packet / "native_task_arena_packet_receipt.v1.json").read_text())
    binding = next(row for row in receipt["source_bindings"]
                   if row["semantic_role"] == "task_object")
    binding.update({
        "runtime_asset_id": "admitted_can", "source": {
            "sha256": _identity(staged)["digest"],
            "size_bytes": staged.stat().st_size,
        },
        "staged_relative_path": "assets/task_object.usd",
        "staged_sha256": _identity(staged)["digest"],
        "staged_size_bytes": staged.stat().st_size,
    })
    return value, configured, packet, receipt


def test_adapter_accepts_only_readback_verified_passive_joint_derivation(tmp_path: Path) -> None:
    value, configured, packet, receipt = _passive_joint_overlay_binding(tmp_path)
    _verify_task_subject_binding(
        request=value, configured_revision=configured,
        packet_root=packet, packet_receipt=receipt)

    wrong_source = copy.deepcopy(receipt)
    binding = next(row for row in wrong_source["source_bindings"]
                   if row["semantic_role"] == "task_object")
    binding["source"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(TaskEvaluationNativeArenaAdapterError,
                       match="task_subject_binding_mismatch"):
        _verify_task_subject_binding(
            request=value, configured_revision=configured,
            packet_root=packet, packet_receipt=wrong_source)

    contract_path = packet / "native_task_runtime_contract.v1.json"
    contract = json.loads(contract_path.read_text())
    contract["objects"][0]["articulation_adaptation"]["passive_joint_friction"]["static_effort_n"] = 9.0
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(TaskEvaluationNativeArenaAdapterError,
                       match="task_subject_overlay_invalid"):
        _verify_task_subject_binding(
            request=value, configured_revision=configured,
            packet_root=packet, packet_receipt=receipt)


def _identity(path: Path) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/{path.name}",
        "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _bundles(
    tmp_path: Path, *, empty_runtime_marker: bool = False,
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
    if empty_runtime_marker:
        marker = runtime_receipt.parent / "IsaacLab-Arena/.git/objects/pack/fixture.promisor"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")
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


@pytest.mark.parametrize("empty_runtime_marker", [False, True])
def test_builds_and_materializes_scene_neutral_native_arena_bundles(
    tmp_path: Path, empty_runtime_marker: bool,
) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(
        tmp_path, empty_runtime_marker=empty_runtime_marker)

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
    if empty_runtime_marker:
        marker = Path(result["runtime_source_receipt"]).parent / "IsaacLab-Arena/.git/objects/pack/fixture.promisor"
        assert marker.read_bytes() == b""


@pytest.mark.parametrize("tamper", ["empty_digest", "negative_size"])
def test_runtime_empty_metadata_still_requires_exact_size_and_digest(tmp_path: Path, tamper: str) -> None:
    value, _, _, bundle = _bundles(tmp_path, empty_runtime_marker=True)
    name = "task_evaluation_adapter_bundle_manifest.v1.json"
    with zipfile.ZipFile(bundle) as archive:
        payloads = {n: archive.read(n) for n in archive.namelist()}
    manifest = json.loads(payloads[name])
    marker = next(row for row in manifest["entries"] if row["relative_path"].endswith("fixture.promisor"))
    if tamper == "empty_digest":
        marker["sha256"] = "sha256:" + "f" * 64
    else:
        marker["size_bytes"] = -1
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    payloads[name] = json.dumps(manifest).encode()
    changed = tmp_path / "changed-runtime.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        for member, payload in payloads.items():
            archive.writestr(member, payload)
    with zipfile.ZipFile(changed) as archive, pytest.raises(TaskEvaluationNativeArenaAdapterError, match="member_identity_invalid"):
        _manifest_from_archive(archive, request=value, expected_role="runtime_source")


def test_adapter_hardlinks_verified_members_from_shared_content_store(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, runtime_bundle = _bundles(tmp_path)
    content_store = tmp_path / "compiled-content" / "sha256"

    first = materialize_native_arena_adapter(
        request=value,
        compiled_episode_packet_path=construction_bundle,
        compiled_episode_packet_reference=_identity(construction_bundle),
        configured_revision=configured,
        runtime_source_bundle_path=runtime_bundle,
        output_root=tmp_path / "adapter-output-a",
        content_store_root=content_store,
    )
    second = materialize_native_arena_adapter(
        request=value,
        compiled_episode_packet_path=construction_bundle,
        compiled_episode_packet_reference=_identity(construction_bundle),
        configured_revision=configured,
        runtime_source_bundle_path=runtime_bundle,
        output_root=tmp_path / "adapter-output-b",
        content_store_root=content_store,
    )

    first_packet = (
        Path(first["runtime_source_receipt"]).parent
        / "native_task_runtime_sources.zip"
    )
    second_packet = (
        Path(second["runtime_source_receipt"]).parent
        / "native_task_runtime_sources.zip"
    )
    cached = content_store / hashlib.sha256(first_packet.read_bytes()).hexdigest()
    assert cached.is_file()
    assert first_packet.stat().st_ino == cached.stat().st_ino
    assert second_packet.stat().st_ino == cached.stat().st_ino
    assert cached.stat().st_nlink == 3


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
    reference = {
        "uri": "s3://blueprint-production-inputs/destination/qualified.json",
        "digest": "sha256:" + "d" * 64,
        "size_bytes": 123,
    }
    changed["task"]["destination"] = {
        "schema_version": "task_evaluation_rigid_destination_asset.v1",
        "identity": {"id": "document-tray", "version": "v1"},
        "relation": "inside",
        "visible_label": "blue document tray",
        "asset": reference,
        "rights_admission": reference,
        "static_qualification": reference,
        "native_import_qualification": reference,
        "geometry": reference,
        "placement_qualification": reference,
        "pose_world": {
            "position_world_m": [3.2, -6.76, 0.82],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "provider_disclosure_allowed": True,
    }
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


def test_adapter_bundle_stores_incompressible_payload_members(tmp_path: Path) -> None:
    """The compile-side archive must not deflate splat or checkpoint bytes.

    Every payload member was deflated regardless of content, which made the
    no-spend episode compilation the slowest control-plane hop (about four CPU
    minutes per run).  Entropy-coded payloads are stored, text still deflates,
    the manifest still deflates, and every member reads back byte-exact.
    """

    source = tmp_path / "runtime-source"
    source.mkdir()
    splat_bytes = b"".join(
        hashlib.sha256(index.to_bytes(8, "big")).digest() for index in range(65_536)
    )
    (source / "scene.ply").write_bytes(splat_bytes)
    (source / "runtime.json").write_text(
        json.dumps({"rows": ["row"] * 4096}), encoding="utf-8"
    )
    output = tmp_path / "runtime-source.zip"

    receipt = build_task_evaluation_runtime_source_bundle(
        source_root=source,
        output_path=output,
        expected_production_commit="a" * 40,
        runtime_identity={"id": "native-arena", "version": "isaac-2026-1"},
    )

    assert receipt["status"] == "built"
    with zipfile.ZipFile(output) as archive:
        kinds = {info.filename: info.compress_type for info in archive.infolist()}
        assert kinds["payload/scene.ply"] == zipfile.ZIP_STORED
        assert kinds["payload/runtime.json"] == zipfile.ZIP_DEFLATED
        manifest_members = [name for name in kinds if not name.startswith("payload/")]
        assert manifest_members == ["task_evaluation_adapter_bundle_manifest.v1.json"]
        assert kinds[manifest_members[0]] == zipfile.ZIP_DEFLATED
        assert archive.read("payload/scene.ply") == splat_bytes
        assert json.loads(archive.read("payload/runtime.json")) == {"rows": ["row"] * 4096}
    assert output.stat().st_size < len(splat_bytes) + 64 * 1024
