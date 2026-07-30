from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
import zipfile

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.nurec_openusd_packaging import (
    NuRecOpenUSDPackagingError,
    package_nurec_openusd,
)
from blueprint_pipeline.reconstruction_geometry_contracts import (
    ReconstructionGeometryContractError,
    build_nurec_openusd_packaging_request,
)
from blueprint_pipeline.reconstruction_appearance_asset import (
    build_appearance_asset_manifest,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import ToolRegistry, non_spend_tool_bindings


REPO = Path(__file__).resolve().parents[1]
DIGESTS = ["sha256:" + character * 64 for character in "abcdef"]


def test_recorded_local_packaging_fixture_does_not_promote_scientific_claims() -> None:
    receipt = json.loads(
        (
            REPO / "docs/evidence/openusd_local_packaging_fixture_5d9675f6.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["status"] == "completed_packaging_fixture_only"
    assert receipt["package"]["exact_package_reopened"] is True
    assert receipt["package"]["deterministic_replay_exact"] is True
    assert receipt["package"]["particlefield_prim_count"] == 1
    assert receipt["package"]["collision_api_prim_count"] == 1
    assert receipt["package"]["missing_asset_count"] == 0
    assert receipt["package"]["unresolved_dependency_count"] == 0
    assert receipt["claim_ceiling"]["openusd_package"] is True
    assert all(
        receipt["claim_ceiling"][claim] is False
        for claim in (
            "metric_reference_geometry",
            "collision_geometry",
            "physics_readiness",
            "isaac_load_render_compatibility",
            "simulator_task_evidence",
            "physical_task_success",
            "deployment_readiness",
        )
    )
    assert receipt["contract_artifacts"][
        "qualification_measurements_are_fixture_declarations"
    ] is True
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sources(root: Path) -> tuple[Path, Path]:
    from pxr import Usd, UsdGeom, UsdPhysics

    appearance = root / "appearance.usda"
    stage = Usd.Stage.CreateNew(str(appearance))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Appearance", "ParticleField3DGaussianSplat")
    stage.GetRootLayer().Save()

    collider = root / "collider.usda"
    stage = Usd.Stage.CreateNew(str(collider))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    collision_root = UsdGeom.Xform.Define(stage, "/World/Collision")
    floor = UsdGeom.Cube.Define(stage, "/World/Collision/Floor")
    floor.CreateSizeAttr(4.0)
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    stage.SetDefaultPrim(collision_root.GetPrim())
    stage.GetRootLayer().Save()
    return appearance, collider


def _appearance_manifest(appearance: Path) -> dict:
    appearance_digest = _sha256(appearance)
    return build_appearance_asset_manifest(
        {
            "stable_run_identity": "packaging-run-1",
            "source_capture_identity": "capture-1",
            "source_capture_digest": DIGESTS[0],
            "original_file_references": [
                {"artifact_id": "capture.mov", "digest": DIGESTS[1]}
            ],
            "producing_method": "fixture_particlefield_compiler",
            "implementation_version": "1",
            "container_image_digest": None,
            "source_commit_sha": "1" * 40,
            "deterministic_configuration_digest": DIGESTS[1],
            "input_digests": [
                {"artifact_id": "appearance_candidate.ply", "digest": DIGESTS[0]},
                {"artifact_id": "training_result", "digest": DIGESTS[1]},
            ],
            "output_digests": [
                {"artifact_id": appearance.name, "digest": appearance_digest}
            ],
            "train_heldout_split_digest": DIGESTS[2],
            "camera_calibration_binding": {"digest": DIGESTS[3]},
            "coordinate_frame_declaration": {"frame": "world", "up_axis": "Z"},
            "units": "meters",
            "metric_scale_status": "validated",
            "provider_runtime_identity": {"provider": "local"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"mode": "execute_non_spend"},
            "warnings": [],
            "blockers": [],
            "proof_effect": "appearance_asset_candidate_only",
            "claim_ceiling": "appearance_reconstruction",
            "parent_artifact_or_event": {"digest": DIGESTS[1]},
            "timestamp": "2026-07-30T13:00:00Z",
            "status": "completed",
            "reconstruction_training_request_digest": DIGESTS[0],
            "reconstruction_training_result_digest": DIGESTS[1],
            "source_appearance_asset_reference": "appearance_candidate.ply",
            "source_appearance_asset_digest": DIGESTS[0],
            "source_asset_format": "standard_3dgs_ply",
            "appearance_asset_reference": appearance.name,
            "appearance_asset_digest": appearance_digest,
            "appearance_asset_format": "particlefield_usd",
            "source_prim_path": "/World/Appearance",
            "splat_count": 1,
            "sh_degree": 0,
            "captured_observation": False,
            "raw_evidence": False,
            "metric_geometry_proven": False,
            "collision_geometry_proven": False,
            "heldout_evaluated": False,
        }
    )


def _request(root: Path, **updates):
    appearance = root / "appearance.usda"
    collider = root / "collider.usda"
    appearance_manifest = _appearance_manifest(appearance)
    value = {
        "stable_run_identity": "packaging-run-1",
        "source_capture_identity": "capture-1",
        "source_capture_digest": DIGESTS[0],
        "original_file_references": [
            {"artifact_id": "appearance", "digest": _sha256(appearance)},
            {"artifact_id": "collider", "digest": _sha256(collider)},
        ],
        "producing_method": "packaging_request_compiler",
        "implementation_version": "1",
        "source_commit_sha": "1" * 40,
        "deterministic_configuration_digest": DIGESTS[1],
        "input_digests": [
            {"artifact_id": "appearance", "digest": _sha256(appearance)},
            {"artifact_id": "collider", "digest": _sha256(collider)},
            {
                "artifact_id": "appearance_manifest",
                "digest": appearance_manifest["appearance_asset_manifest_digest"],
            },
        ],
        "output_digests": [],
        "train_heldout_split_digest": DIGESTS[2],
        "camera_calibration_binding": {"digest": DIGESTS[3]},
        "coordinate_frame_declaration": {"frame": "world", "up_axis": "Z"},
        "units": "meters",
        "provider_runtime_identity": {"provider": "local"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {"mode": "execute_non_spend"},
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {"digest": DIGESTS[0]},
        "timestamp": "2026-07-30T13:00:00Z",
        "appearance_asset": {
            "relative_path": appearance.name,
            "digest": _sha256(appearance),
            "source_prim_path": "/World/Appearance",
            "manifest_digest": appearance_manifest["appearance_asset_manifest_digest"],
        },
        "appearance_asset_manifest_digest": appearance_manifest[
            "appearance_asset_manifest_digest"
        ],
        "appearance_asset_manifest": appearance_manifest,
        "metric_geometry_manifest_digest": DIGESTS[3],
        "collider_asset": {
            "relative_path": collider.name,
            "digest": _sha256(collider),
            "source_prim_path": "/World/Collision",
        },
        "collider_candidate_manifest_digest": DIGESTS[4],
        "collider_qualification_digest": DIGESTS[5],
        "collider_qualification_decision": "accepted_bounded_navigation",
        "stage_meters_per_unit": 1.0,
        "up_axis": "Z",
        "shared_visual_physics_frame": True,
        "target_prim_paths": {
            "appearance": "/World/BlueprintReconstruction/Appearance",
            "collision": "/World/BlueprintReconstruction/Collision",
        },
        "output_format": "usdz",
        "output_name": "blueprint_reconstruction.usdz",
        "proof_effect": "packaging_request_only",
        "claim_ceiling": "none",
    }
    value.update(updates)
    return build_nurec_openusd_packaging_request(value)


def _rebind_appearance(request: dict, *, relative_path: str, digest: str) -> None:
    manifest_value = json.loads(json.dumps(request["appearance_asset_manifest"]))
    manifest_value.pop("appearance_asset_manifest_digest", None)
    manifest_value["appearance_asset_reference"] = relative_path
    manifest_value["appearance_asset_digest"] = digest
    manifest_value["output_digests"] = [
        {"artifact_id": relative_path, "digest": digest}
    ]
    manifest = build_appearance_asset_manifest(manifest_value)
    request["appearance_asset_manifest"] = manifest
    request["appearance_asset_manifest_digest"] = manifest[
        "appearance_asset_manifest_digest"
    ]
    request["appearance_asset"] = {
        "relative_path": relative_path,
        "digest": digest,
        "source_prim_path": "/World/Appearance",
        "manifest_digest": manifest["appearance_asset_manifest_digest"],
    }
    request["input_digests"] = [
        row for row in request["input_digests"] if row.get("artifact_id") != "appearance_manifest"
    ]
    request["input_digests"].append(
        {
            "artifact_id": "appearance_manifest",
            "digest": manifest["appearance_asset_manifest_digest"],
        }
    )


def test_real_openusd_packaging_is_self_contained_digest_bound_and_replayable(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics, UsdUtils

    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sources(source_root)
    request = _request(source_root)
    request_schema = json.loads(
        (REPO / "docs/schemas/nurec_openusd_packaging_request.v1.schema.json").read_text()
    )
    jsonschema.validate(request, request_schema)

    first = package_nurec_openusd(
        source_artifact=request,
        artifact_root=source_root,
        output_root=tmp_path / "out-a",
    )
    second = package_nurec_openusd(
        source_artifact=request,
        artifact_root=source_root,
        output_root=tmp_path / "out-b",
    )
    replay = package_nurec_openusd(
        source_artifact=request,
        artifact_root=source_root,
        output_root=tmp_path / "out-a",
    )
    assert replay == first
    assert second["package_digest"] == first["package_digest"]
    assert first["claim_ceiling"] == "openusd_package"
    assert first["collision_api_configured"] is True
    assert first["collider_qualification_decision"] == "accepted_bounded_navigation"

    package = tmp_path / "out-a" / first["package_artifact_reference"]
    assert _sha256(package) == first["package_digest"]
    stage = Usd.Stage.Open(str(package))
    assert stage is not None
    appearance = stage.GetPrimAtPath("/World/BlueprintReconstruction/Appearance")
    collision = stage.GetPrimAtPath("/World/BlueprintReconstruction/Collision")
    assert str(appearance.GetTypeName()) == "ParticleField3DGaussianSplat"
    assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(collision))
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(package))
    assert list(unresolved) == []

    with zipfile.ZipFile(package) as archive:
        assert archive.infolist()[0].filename.endswith(".usda")
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_STORED
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
            assert offset % 64 == 0

    result_schema = json.loads(
        (REPO / "docs/schemas/nurec_openusd_packaging_result.v1.schema.json").read_text()
    )
    jsonschema.validate(first, result_schema)


def test_packaging_request_rejects_unqualified_collider_and_path_traversal(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    _write_sources(tmp_path)
    with pytest.raises(
        ReconstructionGeometryContractError, match="qualified_collider_required_for_packaging"
    ):
        _request(tmp_path, collider_qualification_decision="rejected")
    raw = dict(_request(tmp_path))
    raw.pop("packaging_request_digest")
    raw["appearance_asset"] = dict(raw["appearance_asset"], relative_path="../appearance.usda")
    raw["packaging_request_digest"] = canonical_digest(
        raw, digest_field="packaging_request_digest"
    )
    with pytest.raises(ReconstructionGeometryContractError, match="appearance_asset_relative_path_unsafe"):
        build_nurec_openusd_packaging_request(raw)

    unbound = dict(_request(tmp_path))
    unbound.pop("packaging_request_digest")
    unbound["appearance_asset"] = dict(unbound["appearance_asset"], digest=DIGESTS[0])
    with pytest.raises(
        ReconstructionGeometryContractError, match="appearance_asset_provenance_binding_missing"
    ):
        build_nurec_openusd_packaging_request(unbound)

    spoofed = dict(_request(tmp_path))
    spoofed.pop("packaging_request_digest")
    spoofed["appearance_asset"] = dict(
        spoofed["appearance_asset"], digest=spoofed["collider_asset"]["digest"]
    )
    with pytest.raises(
        ReconstructionGeometryContractError,
        match="appearance_asset_manifest_asset_binding_mismatch",
    ):
        build_nurec_openusd_packaging_request(spoofed)


def test_packager_rejects_digest_mismatch_and_symlink(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    appearance, _collider = _write_sources(source_root)
    request = dict(_request(source_root))
    request.pop("packaging_request_digest")
    _rebind_appearance(request, relative_path="appearance.usda", digest=DIGESTS[0])
    request["original_file_references"] = [
        {"artifact_id": "appearance", "digest": DIGESTS[0]},
        request["original_file_references"][1],
    ]
    request["input_digests"] = [
        {"artifact_id": "appearance", "digest": DIGESTS[0]},
        request["input_digests"][1],
        request["input_digests"][2],
    ]
    request = build_nurec_openusd_packaging_request(request)
    with pytest.raises(NuRecOpenUSDPackagingError, match="appearance_asset_digest_mismatch"):
        package_nurec_openusd(
            source_artifact=request,
            artifact_root=source_root,
            output_root=tmp_path / "out",
        )

    link = source_root / "appearance-link.usda"
    link.symlink_to(appearance)
    request = dict(_request(source_root))
    request.pop("packaging_request_digest")
    _rebind_appearance(
        request,
        relative_path=link.name,
        digest=request["appearance_asset"]["digest"],
    )
    request = build_nurec_openusd_packaging_request(request)
    with pytest.raises(NuRecOpenUSDPackagingError, match="appearance_asset_symlink_forbidden"):
        package_nurec_openusd(
            source_artifact=request,
            artifact_root=source_root,
            output_root=tmp_path / "out-link",
        )


def test_registered_supervisor_tool_executes_only_frozen_packaging_request(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_sources(source_root)
    request = _request(source_root)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="openusd-package-tool",
        customer_question="Package the qualified reconstruction",
        supervisor_output_dir=str(tmp_path / "supervisor"),
        nurec_packaging_request=request,
        nurec_openusd_packager=partial(package_nurec_openusd, artifact_root=source_root),
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[request["packaging_request_digest"]],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    observation = bindings["package_nurec_openusd"].invoke(
        {"packaging_request_digest": request["packaging_request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["claim_ceiling"] == "openusd_package"
    assert observation["typed_result"]["proof_state_changed"] is False
    assert observation["proof_effect"] == "none"

    called = False

    def must_not_run(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid request reached the packaging runtime")

    invalid = dict(request)
    invalid["collider_qualification_decision"] = "rejected"
    invalid["packaging_request_digest"] = canonical_digest(
        invalid, digest_field="packaging_request_digest"
    )
    refused_context = SupervisorContext(
        run_id="openusd-package-tool-refusal",
        customer_question="Package an unqualified collider",
        supervisor_output_dir=str(tmp_path / "refused"),
        nurec_packaging_request=invalid,
        nurec_openusd_packager=must_not_run,
    )
    refused_authority = default_authority_envelope(
        run_id=refused_context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[invalid["packaging_request_digest"]],
    ).to_mapping()
    refused_bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=refused_context,
            registry=registry,
            authority=refused_authority,
        )
    }
    refused = refused_bindings["package_nurec_openusd"].invoke(
        {"packaging_request_digest": invalid["packaging_request_digest"]}
    )
    assert refused["status"] == "refused"
    assert called is False


def test_packager_rejects_unbound_usd_dependencies_and_compressed_usdz(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    source_root = tmp_path / "source"
    source_root.mkdir()
    appearance, _collider = _write_sources(source_root)
    extra = source_root / "unbound.usda"
    stage = Usd.Stage.CreateNew(str(extra))
    stage.DefinePrim("/Extra", "Xform")
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(appearance))
    stage.GetPrimAtPath("/World/Appearance").GetReferences().AddReference(
        str(extra), "/Extra"
    )
    stage.GetRootLayer().Save()
    request = dict(_request(source_root))
    request.pop("packaging_request_digest")
    digest = _sha256(appearance)
    _rebind_appearance(request, relative_path="appearance.usda", digest=digest)
    request["original_file_references"][0] = {"artifact_id": "appearance", "digest": digest}
    request["input_digests"][0] = {"artifact_id": "appearance", "digest": digest}
    request = build_nurec_openusd_packaging_request(request)
    with pytest.raises(
        NuRecOpenUSDPackagingError, match="openusd_source_external_dependency_unbound"
    ):
        package_nurec_openusd(
            source_artifact=request,
            artifact_root=source_root,
            output_root=tmp_path / "out-unbound",
        )

    compressed = source_root / "compressed.usdz"
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
    request = dict(_request(source_root))
    request.pop("packaging_request_digest")
    digest = _sha256(compressed)
    request["collider_asset"] = {
        "relative_path": compressed.name,
        "digest": digest,
        "source_prim_path": "/World/Collision",
    }
    request["original_file_references"][1] = {"artifact_id": "collider", "digest": digest}
    request["input_digests"][1] = {"artifact_id": "collider", "digest": digest}
    request = build_nurec_openusd_packaging_request(request)
    with pytest.raises(
        NuRecOpenUSDPackagingError, match="collider_asset_usdz_compressed_member"
    ):
        package_nurec_openusd(
            source_artifact=request,
            artifact_root=source_root,
            output_root=tmp_path / "out-compressed",
        )
