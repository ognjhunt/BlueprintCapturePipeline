from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_suite_materializer as materializer
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_suite_materializer import (
    PublicSceneSuiteMaterializationError,
    materialize_public_scene_suite,
)


TARGET_PRIM = "/Root/target"
SUPPORT_PRIM = "/Root/support"
ROOT = Path(__file__).resolve().parents[1]


def test_content_agents_match_v2_binding_verifies_exact_approval_chain() -> None:
    control = json.loads(
        (ROOT / materializer.CONTENT_AGENTS_MATCH_V2_CONTROL_RECEIPT).read_text()
    )
    replacement = json.loads(
        (ROOT / materializer.CONTENT_AGENTS_MATCH_V2_REPLACEMENT_RECEIPT).read_text()
    )
    human = json.loads(
        (ROOT / materializer.CONTENT_AGENTS_MATCH_V2_HUMAN_REVIEW_RECEIPT).read_text()
    )
    bundle = {
        "input_variant": "match_v2",
        "reference_image_authority": (
            "blueprint_cad_snapshot_bound_to_human_approved_match_v2_not_"
            "interiorgs_dataset_bytes"
        ),
        "input_variant_bindings": {
            "control_receipt_digest": control["receipt_digest"],
            "replacement_receipt_digest": replacement["receipt_digest"],
            "human_review_receipt_digest": human["receipt_digest"],
        },
        "input_usd_normalization": {
            "source_input_usd_sha256": control["usd"]["sha256"]
        },
    }

    binding = materializer._content_agents_input_variant_binding(
        bundle=bundle, repo_root=ROOT
    )

    assert binding["input_variant"] == "match_v2"
    assert binding["approved_v2_receipt_chain_verified"] is True


def test_content_agents_match_v2_binding_rejects_changed_human_receipt_digest() -> None:
    control = json.loads(
        (ROOT / materializer.CONTENT_AGENTS_MATCH_V2_CONTROL_RECEIPT).read_text()
    )
    replacement = json.loads(
        (ROOT / materializer.CONTENT_AGENTS_MATCH_V2_REPLACEMENT_RECEIPT).read_text()
    )
    bundle = {
        "input_variant": "match_v2",
        "reference_image_authority": (
            "blueprint_cad_snapshot_bound_to_human_approved_match_v2_not_"
            "interiorgs_dataset_bytes"
        ),
        "input_variant_bindings": {
            "control_receipt_digest": control["receipt_digest"],
            "replacement_receipt_digest": replacement["receipt_digest"],
            "human_review_receipt_digest": "sha256:" + "0" * 64,
        },
        "input_usd_normalization": {
            "source_input_usd_sha256": control["usd"]["sha256"]
        },
    }

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="match_v2_receipt_chain_invalid",
    ):
        materializer._content_agents_input_variant_binding(bundle=bundle, repo_root=ROOT)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _box(
    ins_id: str, label: str, lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> dict:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return {
        "ins_id": ins_id,
        "label": label,
        "bounding_box": [
            {"x": x, "y": y, "z": z}
            for x, y, z in (
                (x0, y0, z0),
                (x1, y0, z0),
                (x0, y1, z0),
                (x1, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x0, y1, z1),
                (x1, y1, z1),
            )
        ],
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    methods = tmp_path / "methods"
    output = repo / "outputs"
    scene = data / "shortlist" / "InteriorGS" / "0001_123456"
    labels = [
        _box("10", "canned beverage", (0.0, 0.0, 1.0), (0.1, 0.1, 1.2)),
        _box("20", "TV cabinet", (-0.5, -0.5, 0.0), (0.5, 0.5, 1.0)),
    ]
    _write_json(scene / "labels.json", labels)
    _write_json(scene / "structure.json", {"rooms": []})
    (scene / "3dgs_compressed.ply").write_bytes(b"real-splat-bytes")
    usdz = data / "shortlist" / "SAGE-3D_InteriorGS_usdz" / "InteriorGS_usdz" / "123456.usdz"
    collision = (
        data
        / "shortlist"
        / "SAGE-3D_Collision_Mesh"
        / "Collision_Mesh"
        / "123456"
        / "123456_collision.usd"
    )
    usdz.parent.mkdir(parents=True)
    collision.parent.mkdir(parents=True)
    usdz.write_bytes(b"real-usdz-bytes")
    collision.write_bytes(b"real-collision-bytes")
    _write_json(
        data / "survey.json",
        {
            "scene_id": "123456",
            "camera_count": 8,
            "rooms": [{"room_id": "room_00"}],
            "survey_digest": "sha256:" + "1" * 64,
            "target_closeup": {"target_ins_id": "10", "camera_count": 4},
        },
    )
    terms = data / "rights" / "terms.pdf"
    terms.parent.mkdir(parents=True)
    terms.write_bytes(b"publisher terms")
    terms_digest = "sha256:" + hashlib.sha256(terms.read_bytes()).hexdigest()
    rights = repo / "rights.json"
    _write_json(
        rights,
        {
            "revision": "a" * 40,
            "agent_accepted_terms": False,
            "terms_text_sha256": terms_digest,
        },
    )
    for name in ("Inpaint360GS", "Infusion", "AuraFusion360_official"):
        (methods / name).mkdir(parents=True)
        (methods / name / "LICENSE").write_text("test license", encoding="utf-8")

    commits = {"Inpaint360GS": "b" * 40, "Infusion": "c" * 40, "AuraFusion360_official": "d" * 40}

    def fake_git(path: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return commits[path.name]
        if args == ("rev-parse", "HEAD^{tree}"):
            return "e" * 40
        if args == ("status", "--porcelain"):
            return ""
        if args == ("submodule", "status", "--recursive"):
            return ""
        raise AssertionError(args)

    def fake_usd(_path: Path, prim_paths: tuple[str, ...]) -> dict:
        prims = {
            TARGET_PRIM: {
                "type_name": "Mesh",
                "collision_api": True,
                "world_aabb_min_m": [0.0, 0.0, 1.0],
                "world_aabb_max_m": [0.1, 0.1, 1.2],
            },
            SUPPORT_PRIM: {
                "type_name": "Mesh",
                "collision_api": True,
                "world_aabb_min_m": [-0.5, -0.5, 0.0],
                "world_aabb_max_m": [0.5, 0.5, 1.0],
            },
        }
        return {
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "unresolved_dependency_count": 0,
            "prims": {key: prims[key] for key in prim_paths},
        }

    monkeypatch.setattr("blueprint_pipeline.public_scene_suite_materializer._git", fake_git)
    monkeypatch.setattr("blueprint_pipeline.public_scene_suite_materializer._inspect_usd", fake_usd)
    request = {
        "schema_version": "public_scene_suite_materialization_request.v1",
        "index_id": "test-index",
        "evaluated_on": "2026-08-04",
        "scene": {
            "publisher_scene_id": "123456",
            "interiorgs_folder": "0001_123456",
            "interiorgs_revision": "a" * 40,
            "sage_usdz_revision": "f" * 40,
            "sage_collision_revision": "9" * 40,
            "splat_path": "shortlist/InteriorGS/0001_123456/3dgs_compressed.ply",
            "labels_path": "shortlist/InteriorGS/0001_123456/labels.json",
            "structure_path": "shortlist/InteriorGS/0001_123456/structure.json",
            "sage_usdz_path": "shortlist/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/123456.usdz",
            "sage_collision_path": "shortlist/SAGE-3D_Collision_Mesh/Collision_Mesh/123456/123456_collision.usd",
            "survey_path": "survey.json",
            "interiorgs_rights_authority_path": "rights.json",
            "interiorgs_terms_path": "rights/terms.pdf",
            "target_instance_id": "10",
            "target_label": "canned_beverage",
            "target_collision_prim": TARGET_PRIM,
            "minimum_target_collider_iou": 0.9,
            "support_instance_id": "20",
            "support_label": "tv_cabinet",
            "support_collision_prim": SUPPORT_PRIM,
            "minimum_support_collider_iou": 0.9,
        },
        "methods": {
            "inpaint360_author_smoke": {
                "repository": "https://example/Inpaint360GS",
                "local_path": "Inpaint360GS",
                "commit": commits["Inpaint360GS"],
                "source_license": "Apache-2.0",
                "source_license_path": "LICENSE",
                "smallest_blocker": "author_smoke_missing",
            },
            "infusion_primary_adapter": {
                "repository": "https://example/Infusion",
                "local_path": "Infusion",
                "commit": commits["Infusion"],
                "source_license": "MIT",
                "source_license_path": "LICENSE",
                "smallest_blocker": "author_smoke_missing",
            },
            "aurafusion360_quality_challenger": {
                "repository": "https://example/Aura",
                "local_path": "AuraFusion360_official",
                "commit": commits["AuraFusion360_official"],
                "source_license": "Apache-2.0",
                "source_license_path": "LICENSE",
                "smallest_blocker": "author_smoke_missing",
            },
        },
        "missing_roles": {
            "controlled_background_truth": "truth_missing",
            "exact_simready_object": "simready_missing",
            "usd_content_agents_candidate": "content_agent_missing",
            "physics_positive_control": "physics_control_missing",
            "scannetpp_real_transfer": "scannet_missing",
        },
    }
    request_path = repo / "request.json"
    _write_json(request_path, request)
    return {
        "repo": repo,
        "data": data,
        "methods": methods,
        "output": output,
        "request": request_path,
    }


def _run(paths: dict[str, Path]) -> dict:
    return materialize_public_scene_suite(
        request_path=paths["request"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        method_root=paths["methods"],
        output_root=paths["output"],
    )


def _add_method_prerequisite_receipt(paths: dict[str, Path]) -> Path:
    checkpoint = paths["data"] / "methods" / "cropformer.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"real-cropformer-checkpoint")
    rights_file = paths["methods"] / "Entity" / "LICENSE"
    rights_file.parent.mkdir(parents=True, exist_ok=True)
    rights_file.write_text("CC BY-NC 4.0", encoding="utf-8")
    checkpoint_record = {
        "artifact_id": "cropformer",
        "role": "method_checkpoint",
        "relative_path": "methods/cropformer.pth",
        "size_bytes": checkpoint.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "publisher": {"path": "weights/cropformer.pth"},
        "rights_authority_id": "cropformer-rights",
        "rights_established": True,
    }
    rights_record = {
        "relative_path": "Entity/LICENSE",
        "size_bytes": rights_file.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(rights_file.read_bytes()).hexdigest(),
    }
    receipt: dict[str, object] = {
        "schema_version": "public_scene_method_prerequisite_receipt.v1",
        "methods": {
            "inpaint360_author_smoke": {
                "artifacts": [checkpoint_record],
                "remote_snapshots": [
                    {
                        "artifact_id": "author-data",
                        "category": "author_data",
                        "materialized": False,
                        "publisher": {
                            "repository": "example/author-data",
                            "revision": "3" * 40,
                            "snapshot_digest": "sha256:" + "4" * 64,
                        },
                        "rights": {"license_id": None, "allowed_use_ceiling": None},
                        "rights_established": False,
                    }
                ],
                "rights_authorities": [
                    {
                        "authority_id": "cropformer-rights",
                        "established": True,
                        "evidence_files": [rights_record],
                    }
                ],
                "checkpoint_rights_established": True,
                "author_data_rights_established": False,
            }
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = paths["data"] / "methods" / "prerequisite.json"
    _write_json(receipt_path, receipt)
    request = json.loads(paths["request"].read_text())
    request["methods"]["inpaint360_author_smoke"]["prerequisite_receipt"] = (
        "methods/prerequisite.json"
    )
    request["methods"]["inpaint360_author_smoke"]["smallest_blocker"] = (
        "author_dataset_rights_missing"
    )
    _write_json(paths["request"], request)
    return checkpoint


def _add_completed_aura_author_smoke(paths: dict[str, Path]) -> Path:
    request = json.loads(paths["request"].read_text())
    method = request["methods"]["aurafusion360_quality_challenger"]
    prerequisite: dict[str, object] = {
        "schema_version": "public_scene_method_prerequisite_receipt.v1",
        "methods": {
            "aurafusion360_quality_challenger": {
                "artifacts": [],
                "rights_authorities": [],
                "remote_snapshots": [
                    {
                        "artifact_id": "author-data",
                        "category": "author_data",
                        "rights_established": True,
                    },
                    {
                        "artifact_id": "checkpoint",
                        "category": "checkpoint",
                        "rights_established": True,
                    },
                ],
                "checkpoint_rights_established": True,
                "author_data_rights_established": True,
            }
        },
        "receipt_digest": "",
    }
    prerequisite["receipt_digest"] = canonical_digest(
        prerequisite, digest_field="receipt_digest"
    )
    prerequisite_path = paths["data"] / "aura/prerequisite.json"
    _write_json(prerequisite_path, prerequisite)

    bundle = paths["data"] / "aura/bundle/runtime.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"real-immutable-aura-bundle")
    bundle_sha = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    bundle_receipt = {
        "schema_version": "adp_aura_author_smoke_provider_bundle.v1",
        "status": "ready",
        "blockers": [],
        "retry_cap": 0,
        "source_commit": method["commit"],
        "source_tree": method.get("tree", ""),
        "bundle_path": str(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": bundle_sha,
        "wonderworld_source_commit": materializer.AURA_WONDERWORLD_SOURCE_COMMIT,
        "wonderworld_source_tree": materializer.AURA_WONDERWORLD_SOURCE_TREE,
        "wonderworld_marigold_runtime_archive_sha256": "sha256:" + "a" * 64,
        "wonderworld_marigold_runtime_manifest_digest": canonical_digest(
            {
                "files": [
                    {
                        "path": "utils/utils/LICENSE.txt",
                        "size_bytes": 1,
                        "sha256": "sha256:" + "0" * 64,
                    },
                    {
                        "path": "utils/utils/batchsize.py",
                        "size_bytes": 1,
                        "sha256": "sha256:" + "1" * 64,
                    },
                    {
                        "path": "utils/utils/ensemble.py",
                        "size_bytes": 1,
                        "sha256": "sha256:" + "2" * 64,
                    },
                    {
                        "path": "utils/utils/image_util.py",
                        "size_bytes": 1,
                        "sha256": "sha256:" + "3" * 64,
                    },
                ]
            }
        ),
        "wonderworld_marigold_runtime_license": "Apache-2.0",
    }
    bundle_receipt_path = paths["data"] / "aura/bundle/receipt.json"
    _write_json(bundle_receipt_path, bundle_receipt)

    immutable = paths["data"] / "aura/live/immutable_execution"
    artifacts = immutable / "artifacts"
    artifacts.mkdir(parents=True)
    produced = artifacts / "point_cloud.ply"
    produced.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n")
    produced_sha = "sha256:" + hashlib.sha256(produced.read_bytes()).hexdigest()
    command_log = immutable / "aura-inpaint-init.log"
    command_log.write_text("author command completed", encoding="utf-8")
    hardware_log = immutable / "nvidia-smi.log"
    hardware_log.write_text("RTX 4090", encoding="utf-8")
    freeze = immutable / "pip-freeze.txt"
    freeze.write_text("torch==2.5.1+cu124\n", encoding="utf-8")
    execution = {
        "schema_version": "adp_aura_author_smoke_result.v1",
        "status": "completed",
        "blockers": [],
        "source_commit": method["commit"],
        "source_tree": method.get("tree", ""),
        "source_identity_before": {"matches": True, "changed_files": []},
        "source_identity_after": {"matches": True, "changed_files": []},
        "author_source_modified": False,
        "author_command": {
            "command": [
                "/worker/AuraFusion360_official/.venv/bin/python",
                "inpaint.py",
                "--config",
                "configs/360-USID/sunflower/inpaint.config",
            ],
            "returncode": 0,
            "timed_out": False,
            "runtime_seconds": 12.5,
            "stdout_stderr_sha256": "sha256:"
            + hashlib.sha256(command_log.read_bytes()).hexdigest(),
            "log": command_log.name,
        },
        "hardware_probe": {
            "returncode": 0,
            "timed_out": False,
            "runtime_seconds": 0.1,
            "stdout_stderr_sha256": "sha256:"
            + hashlib.sha256(hardware_log.read_bytes()).hexdigest(),
            "log": hardware_log.name,
        },
        "inpaint_init_executed": True,
        "published_expected_output_bound": True,
        "produced_point_cloud": {
            "size_bytes": produced.stat().st_size,
            "sha256": produced_sha,
            "vertex_count": 1,
            "retained_relative_path": "artifacts/point_cloud.ply",
        },
        "python_environment": {
            "path": freeze.name,
            "sha256": "sha256:" + hashlib.sha256(freeze.read_bytes()).hexdigest(),
        },
        "depth_anything3_used": False,
        "retry_cap": 0,
        "wonderworld_marigold_runtime": {
            "repository": "https://github.com/KovenYu/WonderWorld",
            "commit": materializer.AURA_WONDERWORLD_SOURCE_COMMIT,
            "tree": materializer.AURA_WONDERWORLD_SOURCE_TREE,
            "license": "Apache-2.0",
            "license_sha256": materializer.AURA_WONDERWORLD_MARIGOLD_LICENSE_SHA256,
            "archive_sha256": "sha256:" + "a" * 64,
            "source_files": [
                {
                    "path": "utils/utils/LICENSE.txt",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "0" * 64,
                },
                {
                    "path": "utils/utils/batchsize.py",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "1" * 64,
                },
                {
                    "path": "utils/utils/ensemble.py",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "2" * 64,
                },
                {
                    "path": "utils/utils/image_util.py",
                    "size_bytes": 1,
                    "sha256": "sha256:" + "3" * 64,
                },
            ],
        },
    }
    execution_path = immutable / "adp_aura_author_smoke_result.json"
    _write_json(execution_path, execution)

    provider = paths["data"] / "aura/live/vast_provider_run"
    provider.mkdir(parents=True)
    instance_ids = [12345]
    adapter_path = provider / "vast_provider_adapter_result.json"
    _write_json(
        adapter_path,
        {
            "schema_version": "vast_provider_adapter_result.v1",
            "status": "completed",
            "reason": "vast_startup_probe_completed",
            "blockers": [],
            "vast_instance_ids": instance_ids,
            "continuing_spend_from_this_run": False,
        },
    )
    teardown_path = provider / "vast_teardown_manifest.json"
    _write_json(
        teardown_path,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "vast_instance_ids": instance_ids,
            "continuing_spend_from_this_run": False,
        },
    )
    _write_json(
        provider / "vast_final_validation.json",
        {
            "schema_version": "vast_final_validation.v1",
            "status": "passed",
            "vast_instance_ids": instance_ids,
            "all_vast_instances_destroyed_by_adapter": True,
            "continuing_spend_from_this_run": False,
        },
    )
    cleanup_path = paths["data"] / "aura/live/object_store_staging/cleanup.json"
    _write_json(
        cleanup_path,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "blockers": [],
        },
    )
    run_path = paths["data"] / "aura/live/adp_aura_author_smoke_vast_result.json"
    _write_json(
        run_path,
        {
            "schema_version": "adp_aura_author_smoke_vast_run.v1",
            "status": "completed",
            "blockers": [],
            "retry_cap": 0,
            "bundle_sha256": bundle_sha,
            "execution_result_path": str(execution_path),
            "adapter_result_path": str(adapter_path),
            "teardown_manifest_path": str(teardown_path),
            "estimated_cost_usd": 0.1,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
        },
    )
    method.update(
        {
            "prerequisite_receipt": prerequisite_path.relative_to(
                paths["data"]
            ).as_posix(),
            "author_smoke_receipt": run_path.relative_to(paths["data"]).as_posix(),
            "author_smoke_bundle_receipt": bundle_receipt_path.relative_to(
                paths["data"]
            ).as_posix(),
            "author_smoke_object_cleanup_receipt": cleanup_path.relative_to(
                paths["data"]
            ).as_posix(),
        }
    )
    _write_json(paths["request"], request)
    return produced


def _add_statically_validated_simready_control(paths: dict[str, Path]) -> Path:
    asset = paths["repo"] / "assets" / "control.usda"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"real-cad-derived-usd")
    blocker = "isaac_dynamic_contact_drop_slide_tip_gripper_probes_missing"
    receipt: dict[str, object] = {
        "schema_version": "adp009a_parametric_simready_receipt.v1",
        "control_id": "test-simready-control",
        "source_scene_id": "123456",
        "source_instance_id": "10",
        "status": "statically_validated",
        "blockers": [blocker],
        "cad_evidence": {
            "source_skill": {
                "repository": "https://example/cad-skill",
                "commit": "1" * 40,
                "license": "MIT",
            }
        },
        "usd": {
            "relative_path": "assets/control.usda",
            "size_bytes": asset.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
        "checks": {"usd_readback_passed": True},
        "simready_foundation_validation": {
            "passed": True,
            "repository": "https://example/simready-foundation",
            "commit": "2" * 40,
            "tree": "3" * 40,
            "license": "Apache-2.0",
            "profile": "Prop-Robotics-Physx",
            "profile_version": "2.0.0",
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = paths["repo"] / "simready-receipt.json"
    _write_json(receipt_path, receipt)
    request = json.loads(paths["request"].read_text())
    del request["missing_roles"]["exact_simready_object"]
    request["simready_control"] = {
        "receipt_path": "simready-receipt.json",
        "smallest_blocker": blocker,
    }
    _write_json(paths["request"], request)
    return asset


def _add_content_agents_preflight(paths: dict[str, Path]) -> Path:
    receipt: dict[str, object] = {
        "schema_version": "adp009a_usd_content_agents_preflight_receipt.v1",
        "status": "prepared_static_validation_passed",
        "source": {
            "repository": "https://example/content-agents",
            "commit": "4" * 40,
            "tree": "5" * 40,
            "version": "0.5.2",
            "license": "Apache-2.0",
        },
        "runtime": {
            "image_digest": "sha256:" + "6" * 64,
            "platform": "linux/arm64",
            "paid_resource_allocated": False,
            "model_or_remote_renderer_called": False,
        },
        "agents": {
            name: {
                "dry_run_executed": True,
                "full_agent_executed": False,
            }
            for name in ("material", "texture", "physics")
        },
        "blockers": ["gpu_execution_missing"],
        "receipt_digest": "",
    }
    receipt["agents"]["validation"] = {
        "executed": True,
        "dry_run": False,
        "verdict": "pass",
    }
    receipt["agents"]["joint"] = {
        "applicable": False,
        "executed": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = paths["repo"] / "content-agents-receipt.json"
    _write_json(receipt_path, receipt)
    request = json.loads(paths["request"].read_text())
    del request["missing_roles"]["usd_content_agents_candidate"]
    request["content_agents"] = {
        "receipt_path": "content-agents-receipt.json",
        "smallest_blocker": "gpu_execution_missing",
    }
    _write_json(paths["request"], request)
    return receipt_path


def _add_content_agents_execution(paths: dict[str, Path]) -> Path:
    _add_content_agents_preflight(paths)
    candidate_root = paths["data"] / "content_agents" / "candidate"
    run_root = paths["data"] / "content_agents" / "run"
    execution_root = run_root / "immutable_execution"
    bundle_path = candidate_root / "bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(b"immutable-provider-bundle")
    input_sha256 = "sha256:" + "7" * 64
    bundle_receipt = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "status": "ready",
        "source_commit": "4" * 40,
        "source_tree": "5" * 40,
        "source_version": "0.5.2",
        "container_image": "example/content-agents@sha256:" + "6" * 64,
        "container_platform": "linux/amd64",
        "input_usd_sha256": input_sha256,
        "reference_image_authority": "blueprint_cad_render_not_interiorgs_dataset_bytes",
        "input_usd_normalization": {
            "source_input_usd_sha256": "sha256:" + "8" * 64,
            "normalized_input_usd_sha256": input_sha256,
            "default_purpose_bbox_nonempty": True,
        },
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "retry_cap": 0,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    bundle_receipt_path = candidate_root / "bundle_receipt.json"
    _write_json(bundle_receipt_path, bundle_receipt)

    agents: dict[str, object] = {}
    for name in ("material", "texture", "physics"):
        artifact = execution_root / f"{name}_workdir" / "output" / f"{name}.usd"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"{name}-output".encode())
        extra_records = []
        if name in {"material", "physics"}:
            state_path = execution_root / f"{name}_workdir" / ".pipeline_state.json"
            steps = (
                [
                    "validate_input",
                    "build_dataset_usd",
                    "build_dataset_prepare_dataset",
                    "predict",
                    "validate_predictions",
                    "apply",
                    "validate_output",
                    "render",
                ]
                if name == "material"
                else [
                    "build_dataset_usd",
                    "build_dataset_prepare_dataset",
                    "predict",
                    "apply_physics",
                ]
            )
            _write_json(state_path, {"completed_steps": steps, "failed_steps": []})
            extra_records.append(
                {
                    "relative_path": ".pipeline_state.json",
                    "size_bytes": state_path.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(state_path.read_bytes()).hexdigest(),
                }
            )
        else:
            manifest_path = execution_root / "texture_workdir" / "artifacts_manifest.json"
            _write_json(
                manifest_path,
                {
                    "schema_version": "texture-agent-artifacts.v1",
                    "status": {"state": "completed", "failed_step": None},
                },
            )
            extra_records.append(
                {
                    "relative_path": "artifacts_manifest.json",
                    "size_bytes": manifest_path.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                }
            )
        agents[name] = {
            f"{name}_agent_attempted": True,
            f"{name}_agent_executed": True,
            "retry_count": 0,
            "execution": {"returncode": 0, "timed_out": False},
            "produced_artifacts": [
                {
                    "relative_path": f"output/{name}.usd",
                    "size_bytes": artifact.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ]
            + extra_records,
        }
    validation_result = execution_root / "validation_agent" / "validation_result.json"
    _write_json(validation_result, {"verdict": "pass"})
    agents["validation"] = {
        "validation_agent_attempted": True,
        "validation_agent_executed": True,
        "verdict": "pass",
        "result_sha256": "sha256:" + hashlib.sha256(validation_result.read_bytes()).hexdigest(),
        "execution": {"returncode": 0, "timed_out": False},
    }
    agents["joint"] = {
        "joint_agent_executed": False,
        "joint_agent_inapplicable_single_rigid_body": True,
    }
    execution_result = {
        "schema_version": "adp_content_agents_vast_result.v1",
        "status": "completed",
        "source_commit": "4" * 40,
        "source_tree": "5" * 40,
        "source_version": "0.5.2",
        "input_usd_sha256": input_sha256,
        "material_agent_executed": True,
        "texture_agent_executed": True,
        "physics_agent_executed": True,
        "validation_agent_executed": True,
        "paid_gpu_execution": True,
        "model_backend_call_authorized": True,
        "retry_cap": 0,
        "blockers": [],
        "raw_secret_values_recorded": False,
        "agents": agents,
    }
    execution_result_path = execution_root / "adp_content_agents_vast_result.json"
    _write_json(execution_result_path, execution_result)
    allocator_result_path = run_root / "allocator_result.live.json"
    _write_json(
        allocator_result_path,
        {
            "schema_version": "adp_content_agents_vast_run.v1",
            "status": "completed",
            "bundle_sha256": bundle_receipt["bundle_sha256"],
            "estimated_cost_usd": 0.1,
            "hard_cap_usd": 2.0,
            "hard_ttl_seconds": 7200,
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
        },
    )
    final_validation_path = run_root / "vast_provider_run" / "vast_final_validation.json"
    _write_json(
        final_validation_path,
        {
            "schema_version": "vast_final_validation.v1",
            "status": "passed",
            "estimated_cost_usd": 0.1,
            "continuing_spend_from_this_run": False,
            "all_vast_instances_destroyed_by_adapter": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
        },
    )
    teardown_path = run_root / "vast_provider_run" / "vast_teardown_manifest.json"
    _write_json(
        teardown_path,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "runner_gpu_teardown_completed": True,
            "raw_secret_values_recorded": False,
        },
    )
    cleanup_path = run_root / "object_store_staging" / "cleanup.json"
    _write_json(
        cleanup_path,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "raw_secret_values_recorded": False,
            "blockers": [],
        },
    )
    request = json.loads(paths["request"].read_text())
    request["content_agents"].update(
        {
            "execution_result_path": execution_result_path.relative_to(paths["data"]).as_posix(),
            "allocator_result_path": allocator_result_path.relative_to(paths["data"]).as_posix(),
            "bundle_receipt_path": bundle_receipt_path.relative_to(paths["data"]).as_posix(),
            "bundle_path": bundle_path.relative_to(paths["data"]).as_posix(),
            "final_validation_path": final_validation_path.relative_to(paths["data"]).as_posix(),
            "teardown_manifest_path": teardown_path.relative_to(paths["data"]).as_posix(),
            "object_cleanup_path": cleanup_path.relative_to(paths["data"]).as_posix(),
        }
    )
    request["content_agents"].pop("smallest_blocker")
    _write_json(paths["request"], request)
    return execution_result_path


def test_materializer_derives_two_admissions_and_blocks_source_only_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _run(paths)
    index = json.loads((paths["output"] / "public_scene_suite_index.v1.json").read_text())
    assert result["status"] == "blocked"
    assert result["admitted_role_count"] == 2
    assert len(index["components"]) == 10
    statuses = {row["role"]: row["status"] for row in index["components"]}
    assert statuses["interiorgs_appearance_scene"] == "admitted"
    assert statuses["sage3d_collision_companion"] == "admitted"
    assert statuses["inpaint360_author_smoke"] == "blocked"


def test_actual_file_mutation_changes_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _run(paths)
    before = json.loads(
        (paths["output"] / "interiorgs_appearance_scene.component_manifest.json").read_text()
    )
    splat = paths["data"] / "shortlist/InteriorGS/0001_123456/3dgs_compressed.ply"
    splat.write_bytes(splat.read_bytes() + b"changed")
    _run(paths)
    after = json.loads(
        (paths["output"] / "interiorgs_appearance_scene.component_manifest.json").read_text()
    )
    assert before["manifest_digest"] != after["manifest_digest"]
    assert (
        before["materialized_artifacts"][0]["sha256"]
        != after["materialized_artifacts"][0]["sha256"]
    )


def test_method_prerequisite_binds_checkpoint_without_claiming_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_method_prerequisite_receipt(paths)
    _run(paths)
    manifest = json.loads(
        (paths["output"] / "inpaint360_author_smoke.component_manifest.json").read_text()
    )
    receipt = json.loads(
        (paths["output"] / "inpaint360_author_smoke.component_receipt.json").read_text()
    )
    assert manifest["rights"]["checkpoint_rights_established"] is True
    assert manifest["rights"]["author_data_rights_established"] is False
    assert manifest["observed_evidence"]["checkpoint_bytes_verified"] is True
    assert manifest["observed_evidence"]["author_method_executed"] is False
    assert manifest["prerequisite_evidence"]["local_artifact_count"] == 1
    assert manifest["prerequisite_evidence"]["remote_snapshots"][0]["rights_established"] is False
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["author_dataset_rights_missing"]


def test_method_prerequisite_rejects_changed_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    checkpoint = _add_method_prerequisite_receipt(paths)
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="method_prerequisite_artifact_bytes_changed:inpaint360_author_smoke",
    ):
        _run(paths)


def test_aura_author_smoke_is_admitted_only_from_executed_digest_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_completed_aura_author_smoke(paths)

    result = _run(paths)

    manifest = json.loads(
        (paths["output"] / "aurafusion360_quality_challenger.component_manifest.json").read_text()
    )
    receipt = json.loads(
        (paths["output"] / "aurafusion360_quality_challenger.component_receipt.json").read_text()
    )
    assert result["admitted_role_count"] == 3
    assert manifest["observed_evidence"]["author_method_executed"] is True
    assert manifest["observed_evidence"]["inpaint_init_executed"] is True
    assert manifest["observed_evidence"]["provider_zero_proven"] is True
    assert manifest["rights"]["author_data_rights_established"] is True
    assert receipt["status"] == "admitted"
    assert receipt["blockers"] == []


def test_aura_author_smoke_rejects_unbound_wonderworld_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_completed_aura_author_smoke(paths)
    bundle_receipt_path = paths["data"] / "aura/bundle/receipt.json"
    bundle_receipt = json.loads(bundle_receipt_path.read_text())
    bundle_receipt["wonderworld_source_commit"] = "f" * 40
    _write_json(bundle_receipt_path, bundle_receipt)

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="aura_author_smoke_bundle_receipt_invalid",
    ):
        _run(paths)


def test_aura_author_smoke_rejects_mutated_produced_point_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    produced = _add_completed_aura_author_smoke(paths)
    produced.write_bytes(produced.read_bytes() + b"mutated")

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="aura_author_smoke_output_bytes_changed",
    ):
        _run(paths)


def test_aura_author_smoke_rejects_success_flag_without_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_completed_aura_author_smoke(paths)
    teardown_path = paths["data"] / "aura/live/vast_provider_run/vast_teardown_manifest.json"
    teardown = json.loads(teardown_path.read_text())
    teardown["continuing_spend_from_this_run"] = True
    _write_json(teardown_path, teardown)

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="aura_author_smoke_provider_evidence_invalid",
    ):
        _run(paths)


def test_statically_validated_simready_control_is_bound_but_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_statically_validated_simready_control(paths)

    _run(paths)

    manifest = json.loads(
        (paths["output"] / "exact_simready_object.component_manifest.json").read_text()
    )
    receipt = json.loads(
        (paths["output"] / "exact_simready_object.component_receipt.json").read_text()
    )
    index = json.loads(
        (paths["output"] / "public_scene_suite_index.v1.json").read_text()
    )
    index_row = next(
        row for row in index["components"] if row["role"] == "exact_simready_object"
    )
    assert manifest["source_project_id"] == "Blueprint-controlled"
    assert index_row["source_project_id"] == manifest["source_project_id"]
    assert index_row["component_manifest_digest"] == manifest["manifest_digest"]
    assert index_row["component_admission_receipt_digest"] == receipt["receipt_digest"]
    assert manifest["observed_evidence"]["simready_foundation_profile_passed"] is True
    assert manifest["observed_evidence"]["isaac_dynamic_probes_executed"] is False
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["isaac_dynamic_contact_drop_slide_tip_gripper_probes_missing"]


def test_content_agents_preflight_is_bound_without_promoting_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_content_agents_preflight(paths)

    _run(paths)

    manifest = json.loads(
        (paths["output"] / "usd_content_agents_candidate.component_manifest.json").read_text()
    )
    receipt = json.loads(
        (paths["output"] / "usd_content_agents_candidate.component_receipt.json").read_text()
    )
    assert manifest["observed_evidence"]["validation_agent_static_check_passed"] is True
    assert manifest["observed_evidence"]["material_agent_full_execution"] is False
    assert manifest["observed_evidence"]["joint_agent_inapplicable_single_rigid_body"] is True
    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == ["gpu_execution_missing"]


def test_content_agents_preflight_rejects_mutated_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt_path = _add_content_agents_preflight(paths)
    receipt = json.loads(receipt_path.read_text())
    receipt["runtime"]["paid_resource_allocated"] = True
    _write_json(receipt_path, receipt)

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="content_agents_preflight_receipt_digest_mismatch",
    ):
        _run(paths)


def test_content_agents_full_execution_is_admitted_from_verified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_content_agents_execution(paths)

    result = _run(paths)

    manifest = json.loads(
        (paths["output"] / "usd_content_agents_candidate.component_manifest.json").read_text()
    )
    receipt = json.loads(
        (paths["output"] / "usd_content_agents_candidate.component_receipt.json").read_text()
    )
    assert result["admitted_role_count"] == 3
    assert manifest["observed_evidence"]["material_agent_full_execution"] is True
    assert manifest["observed_evidence"]["continuing_spend_zero_verified"] is True
    assert receipt["status"] == "admitted"
    assert receipt["blockers"] == []


def test_content_agents_full_execution_rejects_changed_output_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _add_content_agents_execution(paths)
    artifact = (
        paths["data"]
        / "content_agents/run/immutable_execution/material_workdir/output/material.usd"
    )
    artifact.write_bytes(b"mutated")

    with pytest.raises(
        PublicSceneSuiteMaterializationError,
        match="content_agents_artifact_(size|digest)_mismatch",
    ):
        _run(paths)


def test_simready_control_rejects_changed_usd_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    asset = _add_statically_validated_simready_control(paths)
    asset.write_bytes(asset.read_bytes() + b"mutation")

    with pytest.raises(
        PublicSceneSuiteMaterializationError, match="simready_control_usd_bytes_changed"
    ):
        _run(paths)


def test_caller_asserted_admission_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    request["status"] = "admitted"
    _write_json(paths["request"], request)
    with pytest.raises(
        PublicSceneSuiteMaterializationError, match="caller_asserted_admission_forbidden"
    ):
        _run(paths)


def test_scene_id_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    request["scene"]["interiorgs_folder"] = "0001_654321"
    _write_json(paths["request"], request)
    with pytest.raises(PublicSceneSuiteMaterializationError, match="interiorgs_scene_id_mismatch"):
        _run(paths)


def test_path_outside_approved_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    request = json.loads(paths["request"].read_text())
    request["scene"]["splat_path"] = "../outside.ply"
    _write_json(paths["request"], request)
    with pytest.raises(PublicSceneSuiteMaterializationError, match="path_outside_approved_roots"):
        _run(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("rights", "interiorgs_terms_digest_mismatch"),
        ("units", "usd_units_not_meters"),
        ("collider", "target_collider_identity_below_threshold"),
    ],
)
def test_rights_units_and_collider_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, message: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    if mutation == "rights":
        rights = json.loads((paths["repo"] / "rights.json").read_text())
        rights["terms_text_sha256"] = "sha256:" + "0" * 64
        _write_json(paths["repo"] / "rights.json", rights)
    else:
        from blueprint_pipeline import public_scene_suite_materializer as module

        original = module._inspect_usd

        def changed(path: Path, prim_paths: tuple[str, ...]) -> dict:
            result = original(path, prim_paths)
            if mutation == "units":
                result["meters_per_unit"] = 0.01
            elif TARGET_PRIM in result["prims"]:
                result["prims"][TARGET_PRIM]["world_aabb_min_m"] = [9.0, 9.0, 9.0]
                result["prims"][TARGET_PRIM]["world_aabb_max_m"] = [10.0, 10.0, 10.0]
            return result

        monkeypatch.setattr(module, "_inspect_usd", changed)
    with pytest.raises(PublicSceneSuiteMaterializationError, match=message):
        _run(paths)
