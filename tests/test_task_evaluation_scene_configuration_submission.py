"""Scene bytes plus task objects in; one validated production submission out."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import (
    resolve_scene_configuration_disclosure,
)
from blueprint_pipeline.task_evaluation_scene_configuration_source_preflight import (
    TaskEvaluationSceneConfigurationSourcePreflightError,
    validate_scene_configuration_source_bindings,
    validate_scene_configuration_source_preflight,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import (
    validate_immutable_stage_configurations,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission import (
    SceneConfigurationSubmissionError,
    materialize_scene_configuration_submission,
)
from blueprint_pipeline.task_evaluation_scene_construction_recipe import (
    validate_scene_construction_recipe,
)


SHA = "0662bb89ff29df030dd893a0b170edbdd27a30d5"
BOOK_CORNERS = [
    {"x": -1.8563, "y": -3.6400, "z": 0.2755},
    {"x": -2.1516, "y": -3.6400, "z": 0.2755},
    {"x": -2.1516, "y": -3.2423, "z": 0.2755},
    {"x": -1.8563, "y": -3.2423, "z": 0.2755},
    {"x": -1.8563, "y": -3.6400, "z": 0.29664},
    {"x": -2.1516, "y": -3.6400, "z": 0.29664},
    {"x": -2.1516, "y": -3.2423, "z": 0.29664},
    {"x": -1.8563, "y": -3.2423, "z": 0.29664},
]
CABINET_CORNERS = [
    {"x": -1.8418, "y": -4.2363, "z": 0.0},
    {"x": -2.2168, "y": -4.2363, "z": 0.0},
    {"x": -2.2168, "y": 0.9322, "z": 0.0},
    {"x": -1.8418, "y": 0.9322, "z": 0.0},
    {"x": -1.8418, "y": -4.2363, "z": 0.275},
    {"x": -2.2168, "y": -4.2363, "z": 0.275},
    {"x": -2.2168, "y": 0.9322, "z": 0.275},
    {"x": -1.8418, "y": 0.9322, "z": 0.275},
]
BOOK_PRIM = "/Root/_PYZVUZVAUUF2PTUKY888888"
CABINET_PRIM = "/Root/_J6IMDBVAV27YPTUKI888888"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _digested(value: dict, field: str) -> dict:
    value[field] = ""
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _identity_receipt(*, labels: Path, collision: Path, instance: str, label: str, corners, prim: str, points: int, faces: int) -> dict:
    lower = [min(c[a] for c in corners) for a in "xyz"]
    upper = [max(c[a] for c in corners) for a in "xyz"]
    match = {
        "prim_path": prim,
        "world_aabb_min_m": [round(v + 0.0003, 9) for v in lower],
        "world_aabb_max_m": [round(v - 0.0002, 9) for v in upper],
        "point_count": points,
        "face_count": faces,
        "collision_api_applied": True,
        "aabb_iou": 0.86,
        "target_coverage_fraction": 0.92,
        "mesh_coverage_fraction": 0.93,
    }
    receipt = {
        "schema_version": "interiorgs_sage_collision_identity.v1",
        "source_files": {
            "interiorgs_labels": {"path": labels.name, "size_bytes": labels.stat().st_size, "sha256": _sha(labels)},
            "sage_collision_usd": {"path": collision.name, "size_bytes": collision.stat().st_size, "sha256": _sha(collision)},
        },
        "coordinate_frame": {"up_axis": "Z", "meters_per_unit": 1.0, "transform_applied": "identity"},
        "target": {
            "interiorgs_instance_id": instance,
            "semantic_label": label,
            "world_aabb_min_m": [round(v, 9) for v in lower],
            "world_aabb_max_m": [round(v, 9) for v in upper],
        },
        "thresholds": {"minimum_whole_object_iou": 0.85, "minimum_part_mesh_coverage": 0.5, "minimum_part_target_coverage": 0.1, "maximum_part_target_coverage": 0.9},
        "overlapping_meshes": [match],
        "whole_object_matches": [match],
        "candidate_subpart_meshes": [],
        "whole_object_collision_identity_passed": True,
        "candidate_subpart_count": 0,
        "claim_boundary": {"physical_equivalence_proven": False},
    }
    return _digested(receipt, "receipt_digest")


def production_fixture(tmp_path: Path, *, room_topology: bool = False) -> dict:
    root = tmp_path / "production"
    inputs = root / "task-evaluation-inputs" / "scene-841757-raw-v2"
    inputs.mkdir(parents=True)
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nend_header\n"
    ply = inputs / "3dgs_compressed.ply"
    ply.write_bytes(header + struct.pack("<3f", 0.1, 0.2, 0.3))
    _write_json(inputs / "labels.json", [
        {"ins_id": "115", "label": "book", "bounding_box": BOOK_CORNERS},
        {"ins_id": "85", "label": "TV cabinet", "bounding_box": CABINET_CORNERS},
        {"ins_id": "7", "label": "sofa", "bounding_box": [{"x": 1.0 + (i % 2), "y": 1.0 + (i // 2 % 2), "z": 0.4 * (i // 4)} for i in range(8)]},
    ])
    _write_json(inputs / "structure.json", {"rooms": [
        {"profile": [[-3., -5.], [1., -5.], [1., 2.], [-3., 2.]]}
    ] if room_topology else []})
    collision = inputs / "841757_collision.usd"
    collision.write_bytes(b"PXR-USDC-fixture-collision-bytes")
    usdz = inputs / "841757.usdz"
    usdz.write_bytes(b"PK-fixture-usdz")
    files = [
        {"role": "appearance_3dgs", "relative_path": "inputs/3dgs_compressed.ply"},
        {"role": "semantic_metadata", "relative_path": "inputs/labels.json"},
        {"role": "scene_structure", "relative_path": "inputs/structure.json"},
        {"role": "collision_usd", "relative_path": "inputs/841757_collision.usd"},
        {"role": "publisher_scene_usdz", "relative_path": "inputs/841757.usdz"},
    ]
    (inputs / "inputs").mkdir()
    for row in files:
        source = inputs / Path(row["relative_path"]).name
        target = inputs / row["relative_path"]
        target.write_bytes(source.read_bytes())
        row["sha256"] = _sha(target)
        row["size_bytes"] = target.stat().st_size
    installation = _digested({
        "schema_version": "public_scene_host_input_installation_receipt.v1",
        "status": "installed",
        "service_readable": True,
        "scene_id": "841757",
        "packet_id": "scene-841757-raw-v2",
        "source_commit_sha": SHA,
        "destination_root": str(inputs),
        "files": files,
    }, "receipt_digest")
    installation_path = _write_json(inputs / "public_scene_host_input_installation_receipt.v1.json", installation)
    publisher = {
        "schema_version": "scene_841757_publisher_source_intake.v1",
        "scene_id": "841757",
        "status": "publisher_pinned_sources_verified_on_production",
        "publisher_direct_download": True,
        "source_uploaded_by_blueprint": False,
        "public_redistribution_allowed": False,
        "artifacts": [
            {"relative_path": "interiorgs/3dgs_compressed.ply", "publisher_revision": "334dfeea4e0241033b4e5de97c01bc7c9c080530", "publisher_url": "https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/334dfeea4e0241033b4e5de97c01bc7c9c080530/0839_841757/3dgs_compressed.ply", "sha256": files[0]["sha256"], "size_bytes": files[0]["size_bytes"]},
            {"relative_path": "interiorgs/labels.json", "publisher_revision": "334dfeea4e0241033b4e5de97c01bc7c9c080530", "publisher_url": "https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/334dfeea4e0241033b4e5de97c01bc7c9c080530/0839_841757/labels.json", "sha256": files[1]["sha256"], "size_bytes": files[1]["size_bytes"]},
            {"relative_path": "interiorgs/structure.json", "publisher_revision": "334dfeea4e0241033b4e5de97c01bc7c9c080530", "publisher_url": "https://huggingface.co/datasets/spatialverse/InteriorGS/resolve/334dfeea4e0241033b4e5de97c01bc7c9c080530/0839_841757/structure.json", "sha256": files[2]["sha256"], "size_bytes": files[2]["size_bytes"]},
            {"relative_path": "sage/841757_collision.usd", "publisher_revision": "3ba75cc7887b62bf84211d5db08adfa64d691597", "publisher_url": "https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh/resolve/3ba75cc7887b62bf84211d5db08adfa64d691597/Collision_Mesh/841757/841757_collision.usd", "sha256": files[3]["sha256"], "size_bytes": files[3]["size_bytes"]},
            {"relative_path": "sage/841757.usdz", "publisher_revision": "d0b6cdc0aa052d38d743339bb629799ae81e7966", "publisher_url": "https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz/resolve/d0b6cdc0aa052d38d743339bb629799ae81e7966/InteriorGS_usdz/841757.usdz", "sha256": files[4]["sha256"], "size_bytes": files[4]["size_bytes"]},
        ],
    }
    publisher_path = _write_json(root / "scene_841757_publisher_source_intake.v1.json", publisher)

    prepared = root / "task-evaluation-inputs" / "scene-841757-raw-v2.source-preparation"
    prepared.mkdir()
    subject_identity = _identity_receipt(labels=inputs / "inputs/labels.json", collision=inputs / "inputs/841757_collision.usd", instance="115", label="book", corners=BOOK_CORNERS, prim=BOOK_PRIM, points=1200, faces=2400)
    support_identity = _identity_receipt(labels=inputs / "inputs/labels.json", collision=inputs / "inputs/841757_collision.usd", instance="85", label="TV cabinet", corners=CABINET_CORNERS, prim=CABINET_PRIM, points=800, faces=1600)
    _write_json(prepared / "source_identity_00.json", subject_identity)
    _write_json(prepared / "source_identity_01.json", support_identity)
    _write_json(prepared / "shared_frame_candidate.json", _digested({"schema_version": "interiorgs_sage_shared_frame_candidate.v1", "correspondence_count": 2}, "receipt_digest"))
    _write_json(prepared / "room_topology_survey.json", {"schema_version": "adp009a_room_viewpoint_survey.v1"})
    def _record(name: str) -> dict:
        return {"relative_path": name, "sha256": _sha(prepared / name), "size_bytes": (prepared / name).stat().st_size}
    preparation = _digested({
        "schema_version": "public_scene_source_preparation.v1",
        "status": "source_context_prepared_pending_calibrated_views",
        "source_commit": SHA,
        "scene_id": "841757",
        "source_installation_digest": installation["receipt_digest"],
        "task_objects": [
            {"role": "movable_subject", "source_instance_id": "115"},
            {"role": "source_support", "source_instance_id": "85"},
            {"role": "supplemental_destination", "description": "blue document tray"},
        ],
        "artifacts": [_record("room_topology_survey.json"), _record("source_identity_00.json"), _record("source_identity_01.json"), _record("shared_frame_candidate.json")],
        "source_identities": [
            {"role": "movable_subject", "source_instance_id": "115", "identity_receipt_digest": subject_identity["receipt_digest"], "target": subject_identity["target"]},
            {"role": "source_support", "source_instance_id": "85", "identity_receipt_digest": support_identity["receipt_digest"], "target": support_identity["target"]},
        ],
        "blockers": [],
    }, "receipt_digest")
    preparation_path = _write_json(prepared / "public_scene_source_preparation.v1.json", preparation)

    tray = root / "tray"
    tray.mkdir()
    asset = tray / "passive_destination_simready.usdz"
    asset.write_bytes(b"PK-tray-simready")
    tray_identity = {"id": "document-tray", "version": "v2"}
    static = _digested({
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": tray_identity,
        "replacement_usd": {"path": str(asset), "sha256": _sha(asset), "size_bytes": asset.stat().st_size},
        "authored_structure_statically_qualified": True,
        "structural_findings": [],
        "claim_boundary": {"native_simulator_import_qualified": False},
        "observed_structure": {
            "collision_bounds_body_frame_m": {"minimum": [-0.165, -0.24, 0.0], "maximum": [0.165, 0.24, 0.035]},
            "rigid_body_paths": ["/Asset"],
            "collision_prim_paths": ["/Asset/Colliders/Bottom", "/Asset/Colliders/Left", "/Asset/Colliders/Right", "/Asset/Colliders/Front", "/Asset/Colliders/Back"],
        },
    }, "result_digest")
    static_path = _write_json(tray / "passive_destination_static_qualification.v1.json", static)
    rights = _digested({"schema_version": "task_evaluation_rigid_destination_rights_admission.v1", "status": "admitted", "destination_identity": tray_identity, "private_provider_processing_allowed": True, "provider_training_allowed": False, "public_redistribution_allowed": False, "license_identifier": "Blueprint-generated-development-asset"}, "rights_admission_digest")
    rights_path = _write_json(tray / "passive_destination_rights_admission.v1.json", rights)
    authoring = _digested({"schema_version": "task_evaluation_rigid_replacement_authoring_result.v1", "status": "authored_candidate_pending_qualification", "replacement_identity": tray_identity, "physics_authority_granted": False, "output_usd": {"sha256": _sha(asset), "size_bytes": asset.stat().st_size}}, "result_digest")
    authoring_path = _write_json(tray / "passive_destination_authoring_receipt.v1.json", authoring)
    def _rec(path: Path) -> dict:
        return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}
    simready = _digested({
        "schema_version": "task_evaluation_passive_destination_simready.v1",
        "status": "static_qualified_pending_native_import_and_placement",
        "destination_identity": tray_identity,
        "asset": _rec(asset), "authoring_receipt": _rec(authoring_path), "static_qualification": _rec(static_path), "rights_admission": _rec(rights_path),
        "intended_support_prim_paths": ["/Asset"],
        "intended_support_collision_prim_paths": ["/Asset/Colliders/Bottom"],
        "interior_bounds_body_frame_m": {"minimum": [-0.16, -0.235, 0.005], "maximum": [0.16, 0.235, 0.035]},
        "static_result_digest": static["result_digest"],
        "native_import_qualified": False, "placement_qualified": False,
    }, "result_digest")
    simready_path = _write_json(tray / "passive_destination_simready_result.v1.json", simready)

    state = root / "pipeline-control-plane"
    (state / SHA).mkdir(parents=True)
    provenance = {"schema_version": "blueprint.deploy_release_provenance.v1", "status": "verified", "git_sha": SHA, "run_id": 33927082294, "run_url": "https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/33927082294", "collection": {"test_count": 16665, "skipped_count": 0}, "workflow_name": "Full Test Lane", "workflow_path": ".github/workflows/full-test-lane.yml", "job_name": "Full pytest lane on CPU runner", "claim_boundary": {"canonical_full_lane_verified": True}}
    provenance_path = _write_json(state / SHA / "deploy-release-provenance.json", provenance)
    receipt_path = _write_json(state / "deploy-receipts" / "production_0662bb89.json", {
        "schema_version": "control_plane_commit_deploy_receipt.v1", "source_commit": SHA, "provider_mutation_performed": False,
        "intake_runtime": {"commit_proven": True, "source_commit": SHA},
        "release_provenance": {"provenance_status": "verified", "promotion_eligible": True, "canonical_full_lane_verified": True, "run_id": 33927082294, "sha256": _sha(provenance_path), "git_sha": SHA},
    })
    runtimes = root / "task-evaluation-inputs" / "system-runtimes"
    _write_json(runtimes / "scene-configuration" / f"{SHA}.publication.v1.json", _digested({"schema_version": "task_evaluation_scene_configuration_toolchain_publication.v1", "source_commit": SHA, "toolchain_digest": "sha256:" + "a" * 64, "receipt_digest": "sha256:" + "b" * 64, "status": "published_and_read_back", "full_byte_service_account_readback_passed": True, "readback_actor": "service-account:blueprint", "file_count": 3}, "receipt_digest"))
    _write_json(runtimes / "splat-render" / f"{SHA}.publication.v1.json", _digested({"schema_version": "task_evaluation_splat_render_runtime_publication.v1", "source_commit": SHA, "runtime_digest": "sha256:" + "c" * 64, "receipt_digest": "sha256:" + "d" * 64, "status": "published_and_read_back", "full_byte_service_account_readback_passed": True, "readback_actor": "service-account:blueprint", "file_count": 3}, "receipt_digest"))
    env_path = root / "task-evaluation-scene-configuration-release.env"
    env_path.write_text("BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT=/x\n")
    evidence = root / "rights-evidence"
    evidence.mkdir()
    terms = evidence / "InteriorGS_Terms_of_Use.pdf"
    terms.write_bytes(b"%PDF-1.4 fixture terms")
    igs_readme = evidence / "InteriorGS_README.md"
    igs_readme.write_text("# InteriorGS\n")
    sage_readme = evidence / "SAGE_README.md"
    sage_readme.write_text("# SAGE\n")

    task_request = {
        "schema_version": "task_evaluation_minimal_task_request.v1",
        "team_namespace": "blueprint-adp",
        "publisher_scene_id": "841757",
        "appearance_removal_method": "registered_source_bounds",
        "run_prefix": "adp-new-scene-book-to-tray-841757",
        "scene_identity": {"id": "interiorgs-841757", "version": "book-tray-v1"},
        "task_identity": {"id": "scene-841757-book-to-tray", "version": "v1"},
        "output_identity": {"id": "interiorgs-841757-configured-runtime-packet", "version": "book-tray-v1"},
        "strategy": "pick_and_place",
        "instruction": "Pick up the open book, place it fully inside the blue document tray, release it, and move the gripper clear.",
        "subject": {
            "source_instance_id": "115",
            "review_label": "open_book",
            "replacement_identity": {"id": "scene-841757-book-replacement", "version": "v1"},
            "authoring_target": "rigid open book lying flat that matches the source-visible object envelope",
            "physics_bounds": {"mass_kg_bounds": [0.3, 1.2], "static_friction_bounds": [0.3, 0.9], "dynamic_friction_bounds": [0.2, 0.8], "restitution_bounds": [0.0, 0.15]},
        },
        "support": {"source_instance_id": "85"},
        "destination": {"relation": "inside", "visible_label": "blue document tray", "clearance_gap_m": 0.05, "support_edge_margin_m": 0.02},
        "success": {
            "control_frequency_hz": 15, "maximum_episode_seconds": 24.0,
            "minimum_lift_m": 0.05, "pregrasp_clearance_m": 0.10,
            "minimum_planar_displacement_m": 0.1, "maximum_final_planar_target_error_m": 0.05,
            "maximum_retries": 0, "maximum_regrasps": 0,
        },
        "human_authority": {"accepted_by": "Synthetic test owner", "accepted_on": "2026-09-04", "authority_reference": "synthetic-fixture-only", "private_derived_frame_disclosure_authorized": True, "provider_retention_terms_accepted": True, "provider_training_terms_accepted": True, "provider_training_authorized": False},
    }
    task_request_path = _write_json(root / "task_request.json", task_request)
    return {
        "root": root,
        "installation_receipt": installation_path,
        "publisher_intake": publisher_path,
        "source_preparation": preparation_path,
        "destination_simready": simready_path,
        "deploy_receipt": receipt_path,
        "release_provenance": provenance_path,
        "release_environment": env_path,
        "runtime_publication_root": runtimes,
        "rights_evidence": {"interiorgs_terms": terms, "interiorgs_readme": igs_readme, "sage_readme": sage_readme},
        "task_request": task_request_path,
        "staging_root": root / "staging",
        "files": files,
    }


def _materialize(fixture: dict, **overrides):
    arguments = dict(
        task_request_path=fixture["task_request"],
        installation_receipt_path=fixture["installation_receipt"],
        publisher_intake_path=fixture["publisher_intake"],
        source_preparation_receipt_path=fixture["source_preparation"],
        destination_simready_result_path=fixture["destination_simready"],
        deploy_receipt_path=fixture["deploy_receipt"],
        release_provenance_path=fixture["release_provenance"],
        release_environment_path=fixture["release_environment"],
        runtime_publication_root=fixture["runtime_publication_root"],
        rights_evidence=fixture["rights_evidence"],
        staging_root=fixture["staging_root"],
        expected_production_commit=SHA,
        namespace_timestamp="20260904T230000Z",
    )
    arguments.update(overrides)
    return materialize_scene_configuration_submission(**arguments)


def _staged_references(staging: Path, request: dict, namespace: str) -> dict[str, dict]:
    prefix = f"s3://blueprint/task-evaluation/production-inputs/{namespace}/"
    rows: dict[str, dict] = {}
    manifest = json.loads((staging / "bundle_manifest.v1.json").read_text())
    inventory = {row["uri"]: row for row in manifest["files"]}

    def walk(node, path):
        if isinstance(node, dict):
            if set(node) == {"uri", "digest", "size_bytes"}:
                entry = inventory[node["uri"]]
                if node["uri"].startswith(prefix):
                    assert entry["publication_allowed"] is True
                else:
                    assert node["uri"].startswith("https://huggingface.co/datasets/")
                    assert "/resolve/" in node["uri"]
                    assert entry["publication_allowed"] is False
                local = staging / entry["relative_path"]
                assert local.is_file(), node["uri"]
                assert _sha(local) == node["digest"] and local.stat().st_size == node["size_bytes"], path
                rows[path] = {"contract_path": path, **node, "materialized_path": str(local), "full_byte_service_account_readback_passed": True}
                return
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}.{index}")

    walk(request, "")
    return rows


def test_scene_and_task_objects_become_one_validated_production_submission(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    result = _materialize(fixture)
    staging = Path(result["staging_root"])
    request = json.loads((staging / "scene_configuration_preparation_request.v1.json").read_text())
    assert validate_launch_preparation_request(request) == request
    namespace = result["input_namespace"]
    assert namespace == f"adp-new-scene-book-to-tray-841757-{SHA}-20260904T230000Z"
    assert request["run_mode"] == "scene_configuration"
    assert request["expected_production_commit"] == SHA
    assert request["task"]["strategy"] == "pick_and_place"
    references = _staged_references(staging, request, namespace)
    # Every reference is staged with exact bytes, and the recipe binds the same stage files.
    recipe = json.loads((staging / "configuration/scene_construction_recipe.v1.json").read_text())
    assert validate_scene_construction_recipe(recipe) == recipe
    assert recipe["supplemental_destination"]["identity"] == {"id": "document-tray", "version": "v2"}
    assert recipe["supplemental_destination"]["asset"] == request["task"]["destination"]["asset"]
    destination = request["task"]["destination"]
    assert "native_import_qualification" not in destination and "geometry" not in destination
    assert destination["native_probe"]["placement_support_scene_prim_paths"] == [CABINET_PRIM]
    assert destination["pose_world"]["position_world_m"][2] == pytest.approx(0.275)
    assert destination["pose_world"]["position_world_m"][1] == pytest.approx(-3.2423 + 0.05 + 0.24)
    # Stage configurations pass the exact preflight validators the bundle runs.
    configurations = {}
    for index, stage in enumerate(recipe["stage_sequence"]):
        path = staging / stage["configuration"]["uri"].split(f"{namespace}/", 1)[1]
        configurations[stage["stage_id"]] = json.loads(path.read_text())
    stage_one = configurations[recipe["stage_sequence"][0]["stage_id"]]
    rights_admission = json.loads((staging / "rights/rights_admission.v1.json").read_text())
    decision = resolve_scene_configuration_disclosure(stage_one_configuration=stage_one, rights_admission=rights_admission)
    assert decision["render_execution_site"] == "control_plane", decision
    envelope = {
        "request": request,
        "recipe": recipe,
        "materialized_references": list(references.values()),
        "render_inputs_result": {
            "disclosure_decision": decision,
            "source_splat_digest": request["scene"]["appearance"]["representation"]["digest"],
            "source_object_masks": {"source_object_identity": {"publisher_instance_id": "115"}},
        },
    }
    validate_immutable_stage_configurations(envelope=envelope, configurations=configurations)
    validate_scene_configuration_source_preflight(envelope=envelope, configurations=configurations)
    unrendered = {key: value for key, value in envelope.items() if key != "render_inputs_result"}
    validate_scene_configuration_source_bindings(envelope=unrendered, configurations=configurations)
    with pytest.raises(TaskEvaluationSceneConfigurationSourcePreflightError, match="manifest_binding_invalid"):
        validate_scene_configuration_source_preflight(envelope=unrendered, configurations=configurations)
    # The book's grasp is authored from its own bounds, not hardcoded.
    template = json.loads((staging / "configuration/task_template.v1.json").read_text())
    assert template["strategy"] == "pick_and_place"
    assert template["interaction_affordance"]["jaw_unit_scoring_frame"] == [0.0, 0.0, 1.0]
    assert template["interaction_affordance"]["contact_point_scoring_frame_m"][1] == pytest.approx(-(3.6400 - 3.2423) / 2.0)
    assert template["interaction_affordance"]["approach_unit_scoring_frame"] == [0.0, -1.0, 0.0]
    assert template["control_frequency_hz"] == 15 and template["maximum_step_count"] == 360
    success = json.loads((staging / "configuration/task_success_criteria.v1.json").read_text())
    execution = json.loads((staging / "configuration/task_execution_spec.v1.json").read_text())
    assert success["target_center_xyz_m"] == template["target_center_xyz_m"] == execution["target_center_xyz_m"]
    assert template["target_center_xyz_m"][2] == pytest.approx(0.275 + 0.005 + (0.29664 - 0.2755) / 2.0)
    manifest = json.loads((staging / "bundle_manifest.v1.json").read_text())
    assert manifest["manifest_digest"] == canonical_digest(manifest, digest_field="manifest_digest")
    staged_files = {p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()}
    assert {row["relative_path"] for row in manifest["files"]} | {"bundle_manifest.v1.json"} == staged_files
    assert result["request_digest"].startswith("sha256:")
    assert manifest["raw_source_upload_allowed"] is False
    assert manifest["native_qualification_claimed"] is False
    assert manifest["provider_allocated"] is False
    assert template["claim_boundary"]["native_grasp_qualified"] is False
    assert template["claim_boundary"]["robot_reachability_established"] is False
    assert template["claim_boundary"]["policy_execution_authorized"] is False
    assert not any("render_inputs_result" in name for name in staged_files)
    raw_rows = [row for row in manifest["files"] if row["relative_path"].startswith("source/")]
    assert len(raw_rows) == 5
    assert all(row["publication_allowed"] is False for row in raw_rows)
    assert stage_one["provider_disclosure"]["raw_interiorgs_bytes"] is False
    assert stage_one["provider_disclosure"]["source_appearance_bytes"] is False
    disclosure = rights_admission["provider_disclosure"]
    for field in ("raw_interiorgs_downloaded_bytes_may_be_uploaded",
                  "source_appearance_downloaded_bytes_may_be_uploaded",
                  "interiorgs_labels_or_structure_may_be_uploaded"):
        assert disclosure[field] is False
    assert rights_admission["amendments"] == []
    original_authority = json.loads(fixture["task_request"].read_text())["human_authority"]
    assert json.loads((staging / "rights/human_authority.v1.json").read_text()) == original_authority
    assert rights_admission["authority_records"][0]["authority_reference"] == original_authority["authority_reference"]


def test_submission_refuses_a_source_preparation_that_is_blocked_or_stale(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    preparation = json.loads(fixture["source_preparation"].read_text())
    preparation["status"] = "blocked"
    preparation["blockers"] = ["source_preparation_whole_object_match_not_unique:115"]
    _write_json(fixture["source_preparation"], _digested(preparation, "receipt_digest"))
    with pytest.raises(SceneConfigurationSubmissionError, match="scene_configuration_submission_source_preparation_blocked"):
        _materialize(fixture)


def test_submission_refuses_a_deploy_receipt_for_another_commit(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    with pytest.raises(SceneConfigurationSubmissionError, match="scene_configuration_submission_release_commit_mismatch"):
        _materialize(fixture, expected_production_commit="1" * 40)


def test_submission_refuses_a_destination_that_cannot_hold_the_subject(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    simready = json.loads(fixture["destination_simready"].read_text())
    simready["interior_bounds_body_frame_m"]["maximum"][2] = 0.02
    _write_json(fixture["destination_simready"], _digested(simready, "result_digest"))
    with pytest.raises(SceneConfigurationSubmissionError, match="scene_configuration_submission_destination_cannot_contain_subject"):
        _materialize(fixture)


@pytest.mark.parametrize("field", ["source_preparation", "installation_receipt"])
def test_submission_refuses_tampered_receipt_digest(tmp_path: Path, field: str) -> None:
    fixture = production_fixture(tmp_path)
    record = json.loads(fixture[field].read_text())
    record["scene_id"] = "another-scene"
    _write_json(fixture[field], record)
    with pytest.raises(SceneConfigurationSubmissionError, match="input_digest_mismatch"):
        _materialize(fixture)


def test_submission_refuses_tampered_source_bytes(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    installation = json.loads(fixture["installation_receipt"].read_text())
    path = Path(installation["destination_root"]) / installation["files"][0]["relative_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(SceneConfigurationSubmissionError, match="input_bytes_mismatch"):
        _materialize(fixture)


def test_submission_binds_persistent_scene_intent(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    _materialize(fixture, scene_intent_digest="sha256:" + "c" * 64)
    request = json.loads((fixture["staging_root"] / "scene_configuration_preparation_request.v1.json").read_text())
    assert request["scene_intent_digest"] == "sha256:" + "c" * 64
    assert request["expected_production_commit"] == SHA


def test_submission_refuses_invalid_source_preparation_provenance(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    record = json.loads(fixture["source_preparation"].read_text())
    record["source_commit"] = "invalid-commit"
    _write_json(fixture["source_preparation"], _digested(record, "receipt_digest"))
    with pytest.raises(SceneConfigurationSubmissionError, match="source_preparation_commit_mismatch"):
        _materialize(fixture)


@pytest.mark.parametrize("field", ["publisher_intake", "task_request"])
def test_submission_refuses_wrong_publisher_scene(tmp_path: Path, field: str) -> None:
    fixture = production_fixture(tmp_path)
    record = json.loads(fixture[field].read_text())
    record["publisher_scene_id" if field == "task_request" else "scene_id"] = "840873"
    _write_json(fixture[field], record)
    with pytest.raises(SceneConfigurationSubmissionError, match="scene_identity_mismatch"):
        _materialize(fixture)


def test_submission_refuses_shared_subject_and_support_collider(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    preparation = json.loads(fixture["source_preparation"].read_text())
    path = fixture["source_preparation"].parent / "source_identity_00.json"
    identity = json.loads(path.read_text())
    for field in ("whole_object_matches", "overlapping_meshes"):
        identity[field][0]["prim_path"] = CABINET_PRIM
    _write_json(path, _digested(identity, "receipt_digest"))
    for row in preparation["artifacts"]:
        if row["relative_path"] == path.name:
            row.update(sha256=_sha(path), size_bytes=path.stat().st_size)
    preparation["source_identities"][0]["identity_receipt_digest"] = identity["receipt_digest"]
    _write_json(fixture["source_preparation"], _digested(preparation, "receipt_digest"))
    with pytest.raises(SceneConfigurationSubmissionError, match="subject_support_collider_shared"):
        _materialize(fixture)


def test_submission_refuses_to_overwrite_existing_staging(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    fixture["staging_root"].mkdir()
    sentinel = fixture["staging_root"] / "retained-user-evidence.txt"
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(SceneConfigurationSubmissionError, match="staging_root_exists"):
        _materialize(fixture)
    assert sentinel.read_bytes() == b"preserve me"
    assert list(fixture["staging_root"].iterdir()) == [sentinel]


def test_submission_derives_instruction_from_configured_labels(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task.pop("instruction")
    task["subject"]["review_label"] = "rigid_folder"
    task["destination"]["visible_label"] = "green inbox"
    _write_json(fixture["task_request"], task)
    result = _materialize(fixture)
    template = json.loads((Path(result["staging_root"]) / "configuration/task_template.v1.json").read_text())
    assert template["instruction"] == (
        "Pick up the rigid folder, place it fully inside the green inbox, "
        "release it, and move the gripper clear."
    )


@pytest.mark.parametrize("kind", ["scene-configuration", "splat-render"])
def test_submission_refuses_unreadable_runtime_publication(tmp_path: Path, kind: str) -> None:
    fixture = production_fixture(tmp_path)
    path = fixture["runtime_publication_root"] / kind / f"{SHA}.publication.v1.json"
    publication = json.loads(path.read_text())
    publication["full_byte_service_account_readback_passed"] = False
    _write_json(path, _digested(publication, "receipt_digest"))
    with pytest.raises(SceneConfigurationSubmissionError, match="runtime_publication_unproven"):
        _materialize(fixture)


@pytest.mark.parametrize("test_count,skipped_count", [(0, 0), (16665, 1)])
def test_submission_refuses_ineligible_full_lane_provenance(
    tmp_path: Path, test_count: int, skipped_count: int
) -> None:
    fixture = production_fixture(tmp_path)
    provenance = json.loads(fixture["release_provenance"].read_text())
    provenance["collection"] = {"test_count": test_count, "skipped_count": skipped_count}
    _write_json(fixture["release_provenance"], provenance)
    deploy = json.loads(fixture["deploy_receipt"].read_text())
    deploy["release_provenance"]["sha256"] = _sha(fixture["release_provenance"])
    _write_json(fixture["deploy_receipt"], deploy)
    with pytest.raises(SceneConfigurationSubmissionError, match="release_provenance_unproven"):
        _materialize(fixture)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing_authority", "provider_authority_missing"),
        ("training_allowed", "provider_training_forbidden"),
        ("missing_method", "requested_appearance_method_not_implemented"),
        ("sam31_method", "requested_appearance_method_not_implemented"),
        ("zero_seed", "resolved_seed_invalid"),
    ],
)
def test_submission_refuses_ungranted_authority_or_unsupported_method_before_staging(
    tmp_path: Path, mutation: str, error: str,
) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    if mutation == "missing_authority":
        task["human_authority"].pop("provider_retention_terms_accepted")
    elif mutation == "training_allowed":
        task["human_authority"]["provider_training_authorized"] = True
    elif mutation == "missing_method":
        task.pop("appearance_removal_method")
    elif mutation == "sam31_method":
        task["appearance_removal_method"] = "sam31_accepted_exact_masks"
    elif mutation == "zero_seed":
        task["resolved_seed"] = 0
    _write_json(fixture["task_request"], task)
    with pytest.raises(SceneConfigurationSubmissionError, match=error):
        _materialize(fixture)
    assert not fixture["staging_root"].exists()


@pytest.mark.parametrize(("field", "bounds"), [
    ("static_friction_bounds", [0.2, 1.2]),
    ("dynamic_friction_bounds", [0.2, 1.2]),
    ("restitution_bounds", [0.0, 1.01]),
    ("mass_kg_bounds", [0.0, 1.2]),
    ("mass_kg_bounds", [1.2, 0.3]),
    ("static_friction_bounds", [-0.2, 0.8]),
    ("static_friction_bounds", [0.2, float("nan")]),
    ("mass_kg_bounds", [0.2, float("inf")]),
    ("mass_kg_bounds", [0.3]),
    ("mass_kg_bounds", None),
    (None, None),
])
def test_submission_rejects_invalid_subject_physics_before_staging(
        tmp_path: Path, field: str | None, bounds: list | None) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    if field is None:
        task["subject"].pop("physics_bounds")
    else:
        task["subject"]["physics_bounds"][field] = bounds
    _write_json(fixture["task_request"], task)
    with pytest.raises(SceneConfigurationSubmissionError, match="task_subject_physics_bounds_invalid"):
        _materialize(fixture)
    assert not fixture["staging_root"].exists()


def test_submission_rejects_infeasible_friction_before_staging(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["subject"]["physics_bounds"].update(
        static_friction_bounds=[0.2, 0.4], dynamic_friction_bounds=[0.5, 0.6])
    _write_json(fixture["task_request"], task)
    with pytest.raises(SceneConfigurationSubmissionError, match="task_subject_friction_bounds_infeasible"):
        _materialize(fixture)
    assert not fixture["staging_root"].exists()


def test_submission_accepts_canonical_friction_ceiling_without_changing_it(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    task["subject"]["physics_bounds"]["static_friction_bounds"] = [0.2, 1.0]
    _write_json(fixture["task_request"], task)
    result = _materialize(fixture)
    stages = [json.loads(path.read_text()) for path in Path(result["staging_root"]).rglob("*.json")]
    authoring = next(stage for stage in stages
                     if stage.get("schema_version") == "rigid_replacement_authoring_configuration.v1")
    assert authoring["required_output"]["static_friction_bounds"] == [0.2, 1.0]
    assert authoring["required_output"]["mass_kg_bounds"] == [0.3, 1.2]


def test_submission_preserves_scoped_full_source_authority_references(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path)
    task = json.loads(fixture["task_request"].read_text())
    authority_path = tmp_path / "fixture-authority-reference.json"
    authority_path.write_text(json.dumps({"fixture_only": True}))
    reference = {"path": str(authority_path), "sha256": "sha256:" + hashlib.sha256(authority_path.read_bytes()).hexdigest(),
                 "size_bytes": authority_path.stat().st_size}
    task["human_authority"]["full_source_provider_disclosure_authorities"] = {
        "configured_scene_partitioned_source_processing": reference}
    _write_json(fixture["task_request"], task)
    result = _materialize(fixture)
    stage = json.loads((Path(result["staging_root"]) / "configuration/stage_1.v1.json").read_text())
    assert stage["human_authority"]["full_source_provider_disclosure_authorities"] == {
        "configured_scene_partitioned_source_processing": reference}
    # Carrying a reference is not source-upload authority; the later byte-bound
    # purpose validator must open and admit its actual contents before staging.
    assert stage["provider_disclosure"]["source_appearance_bytes"] is False


def _iteration_fixture(tmp_path: Path) -> dict:
    fixture = production_fixture(tmp_path)
    provenance = {"schema_version": "blueprint.deploy_release_provenance.v1", "status": "iteration",
        "git_sha": SHA, "promotion_eligible": False, "claim_boundary": {
            "canonical_full_lane_verified": False, "promotion_eligible": False, "evidence_grade": "development_only"}}
    _write_json(fixture["release_provenance"], provenance)
    deploy = json.loads(fixture["deploy_receipt"].read_text())
    deploy["release_provenance"].update(provenance_status="iteration", promotion_eligible=False,
        canonical_full_lane_verified=False, run_id=None, run_url=None, sha256=_sha(fixture["release_provenance"]))
    _write_json(fixture["deploy_receipt"], deploy)
    return fixture


def test_iteration_release_requires_explicit_development_admission(tmp_path: Path) -> None:
    fixture = _iteration_fixture(tmp_path)
    with pytest.raises(SceneConfigurationSubmissionError, match="release_provenance_unproven"):
        _materialize(fixture)
    assert not fixture["staging_root"].exists()
    result = _materialize(fixture, release_admission_mode="development_iteration")
    binding = json.loads((Path(result["staging_root"]) / "release/exact_production_release_binding.v1.json").read_text())
    assert binding["release_admission_mode"] == "development_iteration"
    assert binding["claim_ceiling"] == binding["promotion"]["evidence_grade"] == "development_only"
    assert binding["promotion"] == {"workflow": None, "provenance_status": "iteration",
        "promotion_eligible": False, "canonical_full_lane_verified": False, "evidence_grade": "development_only",
        "run_id": None, "test_count": None, "skip_count": None, "provenance_sha256": _sha(fixture["release_provenance"])}
    manifest = json.loads((Path(result["staging_root"]) / "bundle_manifest.v1.json").read_text())
    assert manifest["release_admission_mode"] == "development_iteration"
    assert manifest["claim_ceiling"] == "development_only"
    assert manifest["native_qualification_claimed"] is manifest["provider_allocated"] is False


@pytest.mark.parametrize("defect", ["canary", "claim_upgrade", "numeric_false", "fake_run", "fake_workflow",
                                     "digest", "live_commit", "runtime_readback", "unknown_mode"])
def test_iteration_admission_preserves_exact_release_and_truth_boundaries(tmp_path: Path, defect: str) -> None:
    fixture = _iteration_fixture(tmp_path)
    provenance = json.loads(fixture["release_provenance"].read_text())
    deploy = json.loads(fixture["deploy_receipt"].read_text())
    mode = "development_iteration"
    if defect == "canary":
        provenance["status"] = deploy["release_provenance"]["provenance_status"] = "canary"
    elif defect == "claim_upgrade":
        provenance["claim_boundary"]["promotion_eligible"] = True
    elif defect == "numeric_false":
        provenance["claim_boundary"]["canonical_full_lane_verified"] = 0
    elif defect == "fake_run":
        provenance["run_id"] = 123
    elif defect == "fake_workflow":
        provenance["workflow_name"] = "Full Test Lane"
    elif defect == "live_commit":
        deploy["intake_runtime"]["source_commit"] = "b" * 40
    elif defect == "runtime_readback":
        path = fixture["runtime_publication_root"] / "splat-render" / f"{SHA}.publication.v1.json"
        value = json.loads(path.read_text())
        value["full_byte_service_account_readback_passed"] = False
        _write_json(path, _digested(value, "receipt_digest"))
    elif defect == "unknown_mode":
        mode = "skip_checks"
    _write_json(fixture["release_provenance"], provenance)
    deploy["release_provenance"]["sha256"] = "sha256:" + "b" * 64 if defect == "digest" else _sha(fixture["release_provenance"])
    _write_json(fixture["deploy_receipt"], deploy)
    with pytest.raises(SceneConfigurationSubmissionError):
        _materialize(fixture, release_admission_mode=mode)
    assert not fixture["staging_root"].exists()


def test_sam31_submission_binds_real_source_geometry_before_any_render(tmp_path: Path) -> None:
    fixture = production_fixture(tmp_path, room_topology=True)
    task = json.loads(fixture["task_request"].read_text())
    task["appearance_removal_method"] = "sam31"
    _write_json(fixture["task_request"], task)
    profile = _write_json(tmp_path / "sam-profile.json", _digested({
        "schema_version": "task_evaluation_sam31_preparation_profile.v1",
        "source_commit": SHA, "review_model": "gpt-5.6-terra", "review_maximum_cost_usd": 1.0,
        "candidate_policy_queried": False,
    }, "profile_digest"))
    result = _materialize(fixture, sam31_server_profile_path=profile)
    plan = json.loads((Path(result["staging_root"]) / "configuration/sam31_preparation_plan.v1.json").read_text())
    assert len(plan["camera_policy"]["views"]) == 16
    assert len(plan["camera_policy"]["replacement_views"]) == 16
    screen = plan["camera_policy"]["geometry_screen"]
    assert screen["target_instance_id"] == "115"
    assert screen["source_files"]["labels"]["path"].endswith("labels.json")
    assert screen["source_files"]["collision_identity"]["path"].endswith("source_identity_00.json")
    assert screen["source_files"]["structure"]["sha256"] == _sha(Path(screen["source_files"]["structure"]["path"]))
