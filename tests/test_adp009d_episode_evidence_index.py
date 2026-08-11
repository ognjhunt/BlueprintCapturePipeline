from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from blueprint_pipeline.adp_episode_evidence_index import (
    EpisodeEvidenceIndexError,
    HTML_FILENAME,
    INDEX_FILENAME,
    agent_cad_content_agents_supporting_artifacts,
    materialize_episode_evidence_index,
    materialize_supporting_evidence_inventory,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simready_cad_agent_contract import (
    INSPECTION_SCHEMA_VERSION,
    file_record,
    seal_cad_agent_execution_receipt,
    seal_cad_agent_matrix,
    seal_cad_agent_output,
    seal_cad_agent_reference_manifest,
    seal_cad_agent_request,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_A_FREEZE = (
    ROOT
    / "docs/arm_decision_proof_v1/manifests/third_scene_840920_task_a_freeze.v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative_path: str, content: bytes) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_text_or_bytes(path: Path, payload: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _cad_backend(root: Path, backend_id: str) -> dict[str, object]:
    archive = root / f"source-{backend_id}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as source:
        source.comment = b"1" * 40
        source.writestr("LICENSE", "MIT\n")
    return {
        "backend_id": backend_id,
        "execution_mode": (
            "codex_skill_step_first"
            if backend_id == "earthtojake_text_to_cad"
            else "codex_agent_direct_repo_route"
        ),
        "agent_authored_geometry": True,
        "deterministic_geometry_generator_used": False,
        "graph_geometry_used_for_cad_authoring": False,
        "deterministic_format_conversion_only": True,
        "repository_url": (
            "https://github.com/earthtojake/text-to-cad"
            if backend_id == "earthtojake_text_to_cad"
            else "https://github.com/Pan-Chera/Multi-Agent-CAD"
        ),
        "commit": "1" * 40,
        "tree": "2" * 40,
        "source_archive": file_record(archive),
        "license": "MIT",
        "model_id": "codex_fixture",
    }


def _cad_output_fixture(
    source_root: Path, *, backend_id: str, slot: int = 1
) -> dict[str, object]:
    candidate_root = source_root / "task_a" / backend_id
    brief = _write_text_or_bytes(candidate_root / "brief.md", "CAD brief\n")
    reference = _write_text_or_bytes(candidate_root / "reference.png", b"PNG")
    reference_manifest = seal_cad_agent_reference_manifest(
        scene_id="fixture_scene",
        objects=[
            {
                "replacement_slot": slot,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "task_freeze_path": TASK_A_FREEZE,
                "reference_image_paths": [reference],
            }
        ],
    )
    reference_manifest_path = candidate_root / "reference_manifest.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    request = seal_cad_agent_request(
        request_id=f"request-{slot}-{backend_id}",
        scene_id="fixture_scene",
        task_id="task_a_washer_door_open",
        asset_id="840920_simready_washer_candidate",
        replacement_slot=slot,
        backend=_cad_backend(source_root / "sources", backend_id),
        task_freeze_path=TASK_A_FREEZE,
        cad_brief_path=brief,
        metric_envelope_mm=[600.0, 604.0, 848.0],
        reference_manifest_path=reference_manifest_path,
    )
    generator = _write_text_or_bytes(candidate_root / "candidate.py", "pass\n")
    step = _write_text_or_bytes(candidate_root / "candidate.step", b"STEP")
    snapshot = _write_text_or_bytes(candidate_root / "snapshot.png", b"PNG")
    inspection = candidate_root / "inspection.json"
    inspection_payload = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "passed",
        "step": file_record(step),
        "measured_envelope_mm": [600.0, 604.0, 848.0],
        "measured_center_mm": [0.0, 0.0, 424.0],
        "topology": {"face_count": 6, "edge_count": 12},
        "inspector": {
            "backend": "build123d_import_step",
            "build123d_version": "fixture",
            "ocp_version": "fixture",
            "python_version": "fixture",
            "module_source": file_record(
                ROOT / "src/blueprint_pipeline/simready_cad_agent_contract.py"
            ),
        },
        "claim_boundary": {
            "step_bytes_reopened": True,
            "metric_envelope_measured": True,
            "visual_quality_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    inspection_payload["receipt_digest"] = canonical_digest(
        inspection_payload, digest_field="receipt_digest"
    )
    inspection.write_text(json.dumps(inspection_payload), encoding="utf-8")
    execution = candidate_root / "cad_agent_execution_receipt.v1.json"
    execution_payload = seal_cad_agent_execution_receipt(
        request=request,
        generator_source_path=generator,
        cad_brief_path=brief,
        output_step_path=step,
        event_rows=[{"event": "agent_authored", "status": "passed"}],
    )
    execution.write_text(json.dumps(execution_payload), encoding="utf-8")
    output = seal_cad_agent_output(
        request=request,
        generator_source_path=generator,
        step_path=step,
        inspection_receipt_path=inspection,
        snapshot_paths=[snapshot],
        execution_receipt_path=execution,
        measured_envelope_mm=[600.0, 604.0, 848.0],
        actual_cost_usd=0.0,
    )
    projection_dir = candidate_root / "content_agents_input_v1"
    usd = _write_text_or_bytes(projection_dir / "agent_input.usda", "#usda 1.0\n")
    packet = projection_dir / "mesh_packet.v1.json"
    packet_payload = {
        "schema_version": "cad_agent_mesh_packet.v1",
        "backend_id": backend_id,
    }
    packet_payload["packet_digest"] = canonical_digest(packet_payload)
    packet.write_text(json.dumps(packet_payload), encoding="utf-8")
    projection = {
        "schema_version": "cad_agent_mesh_usd_projection.v1",
        "status": "mesh_working_copy_authored",
        "content_agents_input_eligible": True,
        "canonical_simulator_asset": False,
        "claim_boundary": {
            "deterministic_format_conversion_only": True,
            "cad_authored_by_projection": False,
            "collision_authority": False,
            "physics_authority": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        "step": file_record(step),
        "packet": file_record(packet),
        "packet_digest": packet_payload["packet_digest"],
        "output_usd": file_record(usd),
        "mesh_count": 1,
        "mesh_prim_paths": ["/Asset/links/body/geometry/panel"],
        "default_material_path": "/Asset/materials/agent_input_neutral",
        "point_count": 8,
        "triangle_count": 12,
        "receipt_digest": "",
    }
    projection["receipt_digest"] = canonical_digest(
        projection, digest_field="receipt_digest"
    )
    (projection_dir / "projection_receipt.v1.json").write_text(
        json.dumps(projection), encoding="utf-8"
    )
    bundle_dir = candidate_root / "content_agents_bundle"
    bundle_zip = _write_text_or_bytes(bundle_dir / "bundle.zip", b"ZIP")
    bundle_receipt = {
        "schema_version": "adp_content_agents_provider_bundle.v1",
        "status": "ready",
        "bundle_sha256": file_record(bundle_zip)["sha256"],
        "receipt_digest": "",
    }
    bundle_receipt["receipt_digest"] = canonical_digest(
        bundle_receipt, digest_field="receipt_digest"
    )
    bundle_receipt_path = bundle_dir / "adp_content_agents_bundle_receipt.json"
    bundle_receipt_path.write_text(json.dumps(bundle_receipt), encoding="utf-8")
    return {
        "output": output,
        "projection": projection,
        "bundle": file_record(bundle_zip),
        "bundle_receipt": file_record(bundle_receipt_path),
    }


def _content_bundle_matrix(*, fixture_rows: list[dict[str, object]]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for fixture in fixture_rows:
        output = fixture["output"]
        request = output["request"]
        backend_id = request["backend"]["backend_id"]
        projection = fixture["projection"]
        items.append(
            {
                "replacement_slot": request["replacement_slot"],
                "task_id": request["task_id"],
                "asset_id": request["asset_id"],
                "cad_agent_backend_id": backend_id,
                "cad_agent_output_receipt_digest": output["receipt_digest"],
                "mesh_projection_receipt_digest": projection["receipt_digest"],
                "mesh_packet_digest": projection["packet_digest"],
                "candidate_step_sha256": output["artifacts"]["step"]["sha256"],
                "bundle": fixture["bundle"],
                "bundle_receipt": fixture["bundle_receipt"],
                "blockers": [],
                "exact_bundle_entrypoint_rehearsal_status": "passed",
                "agent_output_is_simready_authority": False,
                "canonical_simready_construction_unresolved": True,
            }
        )
    matrix = {
        "schema_version": "third_scene_agent_cad_content_agents_bundle_matrix.v1",
        "status": "ready",
        "input_variant": "agent_cad_v1",
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": 5,
            "sealed_slots": 1,
        },
        "candidate_count": len(items),
        "items": items,
        "claim_boundary": {
            "content_agents_bundles_built": True,
            "exact_entrypoint_rehearsed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    return matrix


def _preflight_receipt(
    source_root: Path,
    *,
    fixture: dict[str, object],
    schema_version: str,
    status: str,
    relative_name: str,
) -> dict[str, object]:
    output = fixture["output"]
    request = output["request"]
    backend_id = request["backend"]["backend_id"]
    path = (
        source_root
        / "task_a"
        / backend_id
        / "content_agents_bundle"
        / relative_name
    )
    receipt = {
        "schema_version": schema_version,
        "status": status,
        "bundle_sha256": fixture["bundle"]["sha256"],
        "bundle_receipt_sha256": fixture["bundle_receipt"]["sha256"],
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return file_record(path) | {"receipt_digest": receipt["receipt_digest"]}


def _content_execution_readiness(
    *,
    bundle_matrix: dict[str, object],
    fixture_rows: list[dict[str, object]],
    source_root: Path,
) -> dict[str, object]:
    rows = []
    for fixture in fixture_rows:
        output = fixture["output"]
        request = output["request"]
        static = _preflight_receipt(
            source_root,
            fixture=fixture,
            schema_version="adp_content_agents_static_bundle_config_preflight.v1",
            status="static_passed_docker_and_paid_model_access_not_checked",
            relative_name="static_bundle_config_preflight_v1/"
            "adp_content_agents_static_bundle_config_preflight.json",
        )
        local = _preflight_receipt(
            source_root,
            fixture=fixture,
            schema_version="adp_content_agents_local_bundle_config_preflight.v1",
            status="local_passed_paid_model_access_not_checked",
            relative_name="local_bundle_config_preflight_v1/"
            "adp_content_agents_bundle_config_preflight.json",
        )
        rows.append(
            {
                "replacement_slot": request["replacement_slot"],
                "task_id": request["task_id"],
                "asset_id": request["asset_id"],
                "cad_agent_backend_id": request["backend"]["backend_id"],
                "bundle": fixture["bundle"],
                "bundle_receipt": fixture["bundle_receipt"],
                "config_preflight": None,
                "local_config_preflight": local,
                "static_config_preflight": static,
                "execute_admitted": False,
                "provider_mutations_performed": 0,
            }
        )
    readiness = {
        "schema_version": "adp_content_agents_execution_readiness.v1",
        "status": "blocked_before_paid_execution",
        "input_variant": "agent_cad_v1",
        "content_agents_bundle_matrix_digest": bundle_matrix["receipt_digest"],
        "items": rows,
        "receipt_digest": "",
    }
    readiness["receipt_digest"] = canonical_digest(
        readiness, digest_field="receipt_digest"
    )
    return readiness


def _receipt(
    root: Path,
    *,
    episode_id: str,
    subject_id: str,
    learned: bool,
) -> Path:
    artifacts = []
    manifest = _artifact(
        root,
        f"media/{episode_id}/multicamera_frame_manifest.json",
        b'{"schema_version":"fixture"}\n',
    )
    artifacts.append(
        {"role": "multicamera_observation_frame_manifest", **manifest}
    )
    videos = {}
    for camera_id in ("external", "wrist", "overview"):
        artifact = _artifact(
            root, f"media/{episode_id}/{camera_id}.mp4", camera_id.encode("utf-8")
        )
        artifacts.append(
            {"role": "camera_review_video", "camera_id": camera_id, **artifact}
        )
        videos[camera_id] = {
            **artifact,
            "camera_id": camera_id,
            "derived_from_frame_manifest_digest": "sha256:fixture",
        }
    receipt = {
        "schema_version": (
            "adp009d_policy_episode.v3" if learned else "adp009d_control_episode.v2"
        ),
        "episode_id": episode_id,
        ("candidate_id" if learned else "control_id"): subject_id,
        "score": {
            "status": "scored",
            "outcome": "placed" if learned else "never_moved",
            "task_succeeded": learned,
            "outcome_rank": 4 if learned else 0,
        },
        "grader_authority": "deterministic_simulator_state",
        "visual_evidence": {
            "status": "complete",
            "required_camera_ids": ["external", "wrist", "overview"],
            "review_only_camera_ids": ["overview"],
            "videos": videos,
        },
        "media_artifacts": artifacts,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path = root / "receipts" / f"{episode_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("scene_id", "task_id"),
    [
        ("840313", "canned_beverage_pick_place"),
        ("840796", "upper_refrigerator_door_open"),
    ],
)
def test_portable_episode_index_covers_original_and_second_scene_fixtures(
    tmp_path: Path, scene_id: str, task_id: str
) -> None:
    zero = _receipt(
        tmp_path,
        episode_id=f"{scene_id}-canonical-zero",
        subject_id="zero_action_negative",
        learned=False,
    )
    learned = _receipt(
        tmp_path,
        episode_id=f"{scene_id}-canonical-pi05",
        subject_id="pi05_droid",
        learned=True,
    )
    result = materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[learned, zero],
        run_identity={
            "scene_id": scene_id,
            "task_id": task_id,
            "scenario_suite_digest": "sha256:frozen-suite",
        },
    )

    assert result["index"]["episode_count"] == 2
    assert result["index"]["run_identity"]["scene_id"] == scene_id
    assert result["index"]["overview_is_review_only"] is True
    assert (tmp_path / INDEX_FILENAME).is_file()
    html = (tmp_path / HTML_FILENAME).read_text(encoding="utf-8")
    assert f"media/{scene_id}-canonical-pi05/external.mp4" in html
    assert f"media/{scene_id}-canonical-pi05/wrist.mp4" in html
    assert f"media/{scene_id}-canonical-pi05/overview.mp4" in html
    assert "deterministic simulator state" in html


def test_portable_episode_index_rejects_tampered_video(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path,
        episode_id="canonical-pi05",
        subject_id="pi05_droid",
        learned=True,
    )
    (tmp_path / "media/canonical-pi05/wrist.mp4").write_bytes(b"tampered")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="episode_artifact_(digest|size)_mismatch:canonical-pi05:wrist",
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[receipt],
            run_identity={
                "scene_id": "840796",
                "task_id": "upper_refrigerator_door_open",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
        )


def test_portable_episode_index_rejects_overview_as_policy_input(tmp_path: Path) -> None:
    receipt_path = _receipt(
        tmp_path,
        episode_id="canonical-groot",
        subject_id="groot_n17_droid",
        learned=True,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["visual_evidence"]["review_only_camera_ids"] = []
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="episode_overview_not_review_only:canonical-groot",
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[receipt_path],
            run_identity={
                "scene_id": "840796",
                "task_id": "upper_refrigerator_door_open",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
        )


def test_portable_index_represents_terminal_abstention_without_fake_episodes(
    tmp_path: Path,
) -> None:
    abstention = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": "joint_agent_local_ovrtx_renderer_not_ready",
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "receipt_digest": "",
    }
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )

    result = materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[],
        run_identity={
            "scene_id": "840796",
            "task_id": "upper_refrigerator_door_open",
            "scenario_suite_digest": "not_materialized_before_abstention",
        },
        abstention_receipt=abstention,
    )

    assert result["index"]["episode_count"] == 0
    assert result["index"]["typed_abstention"] == abstention
    html = (tmp_path / HTML_FILENAME).read_text(encoding="utf-8")
    assert "No control or learned-policy episode exists" in html
    assert "joint_agent_local_ovrtx_renderer_not_ready" in html


def test_abstention_index_verifies_and_links_supporting_receipts(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    mask = _artifact(external_root, "removal/masks/front.png", b"mask")
    teardown = _artifact(
        external_root,
        "removal/run/teardown.json",
        b'{"schema_version":"vast_teardown_manifest.v1"}\n',
    )
    inventory = materialize_supporting_evidence_inventory(
        source_root=external_root,
        output_root=package_root,
        output_relative_path="supporting_evidence_inventory.v1.json",
        source_root_id="rights_bounded_construction_root",
        artifacts=[
            {"role": "source_mask", **mask},
            {"role": "paid_teardown", **teardown},
        ],
        disclosure_class="digest_receipt_only",
    )
    recovery = {
        "schema_version": "adp_gaussian_excision_recovery_readiness.v1",
        "status": "ready_for_new_authority_not_executed",
        "receipt_digest": "",
    }
    recovery["receipt_digest"] = canonical_digest(
        recovery, digest_field="receipt_digest"
    )
    recovery_path = package_root / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    abstention = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": (
            "fresh_paid_authority_for_qualified_gaussian_contribution_missing"
        ),
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "receipt_digest": "",
    }
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )

    result = materialize_episode_evidence_index(
        run_root=package_root,
        episode_receipt_paths=[],
        run_identity={
            "scene_id": "fixture_scene",
            "task_id": "fixture_task",
            "scenario_suite_digest": "sha256:frozen-suite",
        },
        abstention_receipt=abstention,
        supporting_receipt_paths=[
            "supporting_evidence_inventory.v1.json",
            "recovery.json",
        ],
    )

    assert inventory["artifact_count"] == 2
    assert len(result["index"]["supporting_evidence"]) == 2
    html = (package_root / HTML_FILENAME).read_text(encoding="utf-8")
    assert "Supporting construction evidence" in html
    assert "supporting_evidence_inventory.v1.json" in html


def test_supporting_inventory_rejects_tampered_external_artifact(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    artifact = _artifact(external_root, "mask.png", b"mask")
    (external_root / "mask.png").write_bytes(b"tampered")

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="supporting_evidence_artifact_(digest|size)_mismatch:source_mask",
    ):
        materialize_supporting_evidence_inventory(
            source_root=external_root,
            output_root=package_root,
            output_relative_path="supporting_evidence_inventory.v1.json",
            source_root_id="rights_bounded_construction_root",
            artifacts=[{"role": "source_mask", **artifact}],
            disclosure_class="digest_receipt_only",
        )


def test_supporting_inventory_requires_explicit_replace_existing(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    first = _artifact(external_root, "mask-a.png", b"mask-a")
    second = _artifact(external_root, "mask-b.png", b"mask-b")
    materialize_supporting_evidence_inventory(
        source_root=external_root,
        output_root=package_root,
        output_relative_path="supporting_evidence_inventory.v1.json",
        source_root_id="rights_bounded_construction_root",
        artifacts=[{"role": "source_mask", **first}],
        disclosure_class="digest_receipt_only",
    )
    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="supporting_evidence_inventory_overwrite_forbidden",
    ):
        materialize_supporting_evidence_inventory(
            source_root=external_root,
            output_root=package_root,
            output_relative_path="supporting_evidence_inventory.v1.json",
            source_root_id="rights_bounded_construction_root",
            artifacts=[{"role": "source_mask", **second}],
            disclosure_class="digest_receipt_only",
        )

    updated = materialize_supporting_evidence_inventory(
        source_root=external_root,
        output_root=package_root,
        output_relative_path="supporting_evidence_inventory.v1.json",
        source_root_id="rights_bounded_construction_root",
        artifacts=[{"role": "source_mask", **second}],
        disclosure_class="digest_receipt_only",
        replace_existing=True,
    )

    assert updated["artifacts"][0]["relative_path"] == "mask-b.png"


def test_agent_cad_content_agents_rows_materialize_task_inventory(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    output_root = tmp_path / "package"
    output_root.mkdir()
    shared = _artifact(
        source_root,
        "shared_scene/cad_agent_comparison_v1/all_four.png",
        b"comparison",
    )
    earth = _cad_output_fixture(
        source_root, backend_id="earthtojake_text_to_cad"
    )
    mac = _cad_output_fixture(
        source_root, backend_id="pan_chera_multi_agent_cad"
    )
    cad_matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "candidates": [earth["output"], mac["output"]],
            }
        ]
    )
    bundle_matrix = _content_bundle_matrix(fixture_rows=[earth, mac])
    readiness = _content_execution_readiness(
        bundle_matrix=bundle_matrix,
        fixture_rows=[earth, mac],
        source_root=source_root,
    )

    rows = agent_cad_content_agents_supporting_artifacts(
        source_root=source_root,
        cad_agent_matrix=cad_matrix,
        content_agents_bundle_matrix=bundle_matrix,
        content_agents_execution_readiness=readiness,
        task_id="task_a_washer_door_open",
        shared_artifacts=[
            {
                "role": "agent_cad_comparison:all_four_front_best",
                "relative_path": shared["relative_path"],
            }
        ],
    )

    roles = {row["role"] for row in rows}
    assert "agent_cad_comparison:all_four_front_best" in roles
    assert (
        "agent_cad:earthtojake_text_to_cad:content_agents_bundle_receipt"
        in roles
    )
    assert (
        "agent_cad:earthtojake_text_to_cad:content_agents_static_preflight"
        in roles
    )
    assert (
        "agent_cad:pan_chera_multi_agent_cad:content_agents_local_docker_preflight"
        in roles
    )
    assert "agent_cad:pan_chera_multi_agent_cad:mesh_projection_receipt" in roles
    inventory = materialize_supporting_evidence_inventory(
        source_root=source_root,
        output_root=output_root,
        output_relative_path="supporting_evidence_inventory.v1.json",
        source_root_id="fixture_rights_bounded_root",
        artifacts=rows,
        disclosure_class="digest_receipt_only",
    )
    assert inventory["artifact_count"] == len(rows)
    assert inventory["artifact_bytes_embedded"] is False


def test_agent_cad_content_agents_rows_reject_projection_join_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external"
    earth = _cad_output_fixture(
        source_root, backend_id="earthtojake_text_to_cad"
    )
    mac = _cad_output_fixture(
        source_root, backend_id="pan_chera_multi_agent_cad"
    )
    cad_matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": 1,
                "task_id": "task_a_washer_door_open",
                "asset_id": "840920_simready_washer_candidate",
                "candidates": [earth["output"], mac["output"]],
            }
        ]
    )
    bundle_matrix = _content_bundle_matrix(fixture_rows=[earth, mac])
    bundle_matrix["items"][0]["candidate_step_sha256"] = "sha256:" + "0" * 64
    bundle_matrix["receipt_digest"] = canonical_digest(
        bundle_matrix, digest_field="receipt_digest"
    )

    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="agent_cad_supporting_projection_join_mismatch",
    ):
        agent_cad_content_agents_supporting_artifacts(
            source_root=source_root,
            cad_agent_matrix=cad_matrix,
            content_agents_bundle_matrix=bundle_matrix,
            task_id="task_a_washer_door_open",
        )


def test_supporting_inventory_and_index_reject_symlinks(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    package_root = tmp_path / "package"
    external_root.mkdir()
    package_root.mkdir()
    target = external_root / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = external_root / "linked.json"
    link.symlink_to(target)
    record = {
        "role": "linked_receipt",
        "relative_path": "linked.json",
        "sha256": "sha256:" + _sha256(target),
        "size_bytes": target.stat().st_size,
    }

    with pytest.raises(
        EpisodeEvidenceIndexError, match="episode_artifact_symlink_forbidden"
    ):
        materialize_supporting_evidence_inventory(
            source_root=external_root,
            output_root=package_root,
            output_relative_path="inventory.json",
            source_root_id="fixture_root",
            artifacts=[record],
            disclosure_class="digest_receipt_only",
        )

    with pytest.raises(
        EpisodeEvidenceIndexError, match="supporting_receipt_symlink_forbidden"
    ):
        abstention = {
            "schema_version": "adp_task_evaluation_run_abstention.v1",
            "status": "typed_evidence_backed_abstention",
            "smallest_missing_capability": "fixture_blocker",
            "controls_executed": False,
            "learned_candidate_episodes_executed": False,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "receipt_digest": "",
        }
        abstention["receipt_digest"] = canonical_digest(
            abstention, digest_field="receipt_digest"
        )
        materialize_episode_evidence_index(
            run_root=external_root,
            episode_receipt_paths=[],
            run_identity={
                "scene_id": "fixture_scene",
                "task_id": "fixture_task",
                "scenario_suite_digest": "sha256:frozen-suite",
            },
            abstention_receipt=abstention,
            supporting_receipt_paths=["linked.json"],
        )


def test_episode_index_refresh_is_opt_in_and_validates_owned_source(
    tmp_path: Path,
) -> None:
    abstention = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": "first_blocker",
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "receipt_digest": "",
    }
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )
    identity = {
        "scene_id": "fixture_scene",
        "task_id": "fixture_task",
        "scenario_suite_digest": "sha256:frozen-suite",
    }
    materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[],
        run_identity=identity,
        abstention_receipt=abstention,
    )

    abstention["smallest_missing_capability"] = "corrected_blocker"
    abstention["receipt_digest"] = canonical_digest(
        abstention, digest_field="receipt_digest"
    )
    with pytest.raises(
        EpisodeEvidenceIndexError, match="episode_evidence_index_overwrite_forbidden"
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[],
            run_identity=identity,
            abstention_receipt=abstention,
        )

    refreshed = materialize_episode_evidence_index(
        run_root=tmp_path,
        episode_receipt_paths=[],
        run_identity=identity,
        abstention_receipt=abstention,
        replace_existing=True,
    )
    assert (
        refreshed["index"]["typed_abstention"]["smallest_missing_capability"]
        == "corrected_blocker"
    )
    assert "corrected_blocker" in (tmp_path / HTML_FILENAME).read_text(
        encoding="utf-8"
    )

    existing = json.loads((tmp_path / INDEX_FILENAME).read_text(encoding="utf-8"))
    existing["index_digest"] = "sha256:tampered"
    (tmp_path / INDEX_FILENAME).write_text(json.dumps(existing), encoding="utf-8")
    with pytest.raises(
        EpisodeEvidenceIndexError,
        match="episode_evidence_index_refresh_source_invalid",
    ):
        materialize_episode_evidence_index(
            run_root=tmp_path,
            episode_receipt_paths=[],
            run_identity=identity,
            abstention_receipt=abstention,
            replace_existing=True,
        )
