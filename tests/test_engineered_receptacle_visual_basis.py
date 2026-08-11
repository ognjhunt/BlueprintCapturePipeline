from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import blueprint_pipeline.engineered_receptacle_visual_basis as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.engineered_receptacle_visual_basis import (
    EngineeredReceptacleVisualBasisError,
    VisualBasisLimits,
    verify_engineered_receptacle_visual_basis,
)
from blueprint_pipeline.semantic_review_attestation import (
    TRUSTED_PUBLIC_KEY_SHA256_ENV,
    canonical_semantic_authority_selection_bytes,
    canonical_semantic_review_attestation_bytes,
    materialize_semantic_authority_selection,
    materialize_semantic_review_attestation,
    materialize_semantic_review_payload,
    semantic_frame_evidence_digest,
    semantic_review_signature_message,
)


CURRENT_TOPOLOGY_DIGEST = "sha256:" + "7" * 64
SEMANTIC_AUTHORITY_KEY = Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)


def _public_key_bytes() -> bytes:
    return SEMANTIC_AUTHORITY_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


@pytest.fixture(autouse=True)
def _configured_semantic_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        TRUSTED_PUBLIC_KEY_SHA256_ENV,
        "sha256:" + hashlib.sha256(_public_key_bytes()).hexdigest(),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _pattern_image(path: Path, *, rgba: bool = False) -> None:
    mode = "RGBA" if rgba else "RGB"
    image = Image.new(mode, (4, 3))
    if rgba:
        image.putdata(
            [(30 + index * 10, 50 + index * 7, 70 + index * 5, 180) for index in range(12)]
        )
    else:
        image.putdata([(20 + index * 12, 40 + index * 8, 60 + index * 4) for index in range(12)])
    image.save(path, format="PNG")


def _refresh_semantic_authority(fixture: dict[str, Any]) -> None:
    review = fixture["review"]
    manifest = fixture["manifest"]
    cameras = {row["id"]: row for row in manifest["cameras"]}
    frame_evidence: list[dict[str, Any]] = []
    for target in review["targets"]:
        for citation in target["cited_frames"]:
            camera = cameras[citation["camera_id"]]
            with Image.open(camera["path"]) as image:
                rgb_bytes = image.convert("RGB").tobytes()
            frame_evidence.append(
                {
                    "target_id": target["target_id"],
                    "camera_id": citation["camera_id"],
                    "sha256": camera["digest"],
                    "size_bytes": camera["bytes"],
                    "decoded_rgb_sha256": ("sha256:" + hashlib.sha256(rgb_bytes).hexdigest()),
                }
            )
    basket = next(
        target for target in review["targets"] if target["target_kind"] == "destination_receptacle"
    )
    payload = materialize_semantic_review_payload(
        attestation_id="fixture-basket-semantic-review-001",
        selection_id="fixture-basket-semantic-authority-freeze-001",
        authority_id="fixture-semantic-review-authority",
        authority_key_id="fixture-semantic-key-2026-08",
        scene_id=review["scene_id"],
        target_id=basket["target_id"],
        source_instance_id=basket["publisher_instance_id"],
        semantic_role=basket["target_kind"],
        visual_review_digest=review["review_digest"],
        render_manifest_digest=manifest["render_manifest_digest"],
        collision_topology_receipt_digest=CURRENT_TOPOLOGY_DIGEST,
        cited_frames_digest=semantic_frame_evidence_digest(frame_evidence),
        learned_policy_outcomes_inspected=False,
        semantic_assertions={
            "rigid_exterior_observed": basket["rigid_exterior_observed"],
            "open_rim_observed": basket["open_rim_observed"],
            "interior_occupied": basket["interior_occupied"],
            "complete_interior_appearance_observed": basket[
                "complete_interior_appearance_observed"
            ],
            "source_destination_admitted": basket["source_destination_admitted"],
            "engineered_twin_design_basis_admitted": basket[
                "engineered_twin_design_basis_admitted"
            ],
            "selection_role": basket["selection_role"],
        },
    )
    public_key = _public_key_bytes()
    attestation = materialize_semantic_review_attestation(
        payload=payload,
        public_key_base64=base64.b64encode(public_key).decode("ascii"),
        signature_base64=base64.b64encode(
            SEMANTIC_AUTHORITY_KEY.sign(semantic_review_signature_message(payload))
        ).decode("ascii"),
    )
    selection = materialize_semantic_authority_selection(attestation=attestation)
    fixture["semantic_review_attestation_path"].write_bytes(
        canonical_semantic_review_attestation_bytes(attestation)
    )
    fixture["semantic_authority_selection_path"].write_bytes(
        canonical_semantic_authority_selection_bytes(selection)
    )


def _calibration(camera_id: str, *, width: int = 4, height: int = 3) -> dict[str, Any]:
    return {
        "id": camera_id,
        "pose_convention": "world_position_target_up_look_at_z_up",
        "position_world_m": [1.0, 2.0, 3.0],
        "target_world_m": [0.0, 0.0, 0.0],
        "up_world": [0.0, 0.0, 1.0],
        "intrinsics": {
            "model": "pinhole_centered_square_pixels",
            "fx": 4.2,
            "fy": 4.2,
            "cx": width / 2,
            "cy": height / 2,
            "width": width,
            "height": height,
            "vertical_fov_deg": 45.0,
        },
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    frame_root = tmp_path / "render" / "frames"
    frame_root.mkdir(parents=True)
    frame_paths = {
        "basket_e00": frame_root / "basket_e00.png",
        "towel_e00": frame_root / "towel_e00.png",
    }
    _pattern_image(frame_paths["basket_e00"])
    _pattern_image(frame_paths["towel_e00"], rgba=True)

    cameras = []
    for camera_id, path in frame_paths.items():
        cameras.append(
            {
                "id": camera_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "nonblank": True,
                "digest": _sha256(path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "splat_scene_render.v1",
        "source_digest": "sha256:" + "1" * 64,
        "renderer_identity": {
            "name": "hermetic_reference_renderer",
            "harness_sha256": "sha256:" + "2" * 64,
            "entry_sha256": "sha256:" + "3" * 64,
            "width": 4,
            "height": 3,
            "pixel_ratio": 1,
            "supersampling": 1,
            "alpha": False,
            "background_rgb_hex": "0x000000",
            "output_format": "lossless_png",
            "color_space": "srgb",
            "node_version": "v1.0.0",
            "dependency_versions": {"renderer": "1.2.3"},
        },
        "render": {"status": "completed", "returncode": 0},
        "appearance_fidelity": {
            "source_splat_count": 100,
            "retained_splat_count": 100,
            "appearance_fidelity_qualified": False,
            "evaluation_input_authorized": False,
        },
        "cameras": cameras,
        "camera_calibration": [_calibration(camera_id) for camera_id in frame_paths],
        "render_manifest_digest": "",
    }
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )

    camera_by_id = {row["id"]: row for row in cameras}
    review: dict[str, Any] = {
        "schema_version": "adp_deformable_scene_visual_review.v2",
        "scene_id": "fixture_scene",
        "reviewer_id": "fixture_reviewer",
        "reviewed_at": "2026-08-10T12:00:00Z",
        "learned_policy_outcomes_inspected": False,
        "reconnaissance_only": True,
        "render_manifest_digest": manifest["render_manifest_digest"],
        "collision_topology_receipt_digest": CURRENT_TOPOLOGY_DIGEST,
        "targets": [
            {
                "target_id": "towel_79",
                "publisher_instance_id": "79",
                "target_kind": "movable_deformable",
                "selection_role": "selected_movable_design_basis",
                "cited_frames": [
                    {
                        "camera_id": "towel_e00",
                        "size_bytes": camera_by_id["towel_e00"]["bytes"],
                        "sha256": camera_by_id["towel_e00"]["digest"],
                    }
                ],
            },
            {
                "target_id": "basket_87",
                "publisher_instance_id": "87",
                "target_kind": "destination_receptacle",
                "rigid_exterior_observed": True,
                "open_rim_observed": True,
                "interior_occupied": True,
                "complete_interior_appearance_observed": False,
                "source_destination_admitted": False,
                "engineered_twin_design_basis_admitted": True,
                "selection_role": "engineered_twin_design_basis",
                "cited_frames": [
                    {
                        "camera_id": "basket_e00",
                        "size_bytes": camera_by_id["basket_e00"]["bytes"],
                        "sha256": camera_by_id["basket_e00"]["digest"],
                    }
                ],
            },
        ],
        "selected_movable_instance_id": "79",
        "selected_destination_design_basis_instance_id": "87",
        "source_destination_is_occupied": True,
        "source_destination_complete_interior_appearance_observed": False,
        "composition_required": True,
        "claim_boundary": {
            "virtual_closeup_recovers_missing_source_observations": False,
            "collision_cavity_establishes_hidden_appearance": False,
            "engineered_twin_hidden_geometry_is_source_truth": False,
            "review_is_evaluation_policy_media": False,
            "physical_material_equivalence_proven": False,
        },
        "review_digest": "",
    }
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")

    review_path = tmp_path / "review.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(review_path, review)
    _write_json(manifest_path, manifest)
    fixture = {
        "review": review,
        "manifest": manifest,
        "review_path": review_path,
        "manifest_path": manifest_path,
        "frame_root": frame_root,
        "frame_paths": frame_paths,
        "semantic_review_attestation_path": tmp_path / "semantic-attestation.json",
        "semantic_authority_selection_path": tmp_path / "semantic-selection.json",
    }
    _refresh_semantic_authority(fixture)
    return fixture


def _verify(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = {
        "visual_review_receipt_path": fixture["review_path"],
        "render_manifest_path": fixture["manifest_path"],
        "frame_root": fixture["frame_root"],
        "semantic_review_attestation_path": fixture["semantic_review_attestation_path"],
        "semantic_authority_selection_path": fixture["semantic_authority_selection_path"],
        "current_topology_receipt_digest": CURRENT_TOPOLOGY_DIGEST,
        "source_instance_id": "87",
    }
    kwargs.update(overrides)
    return verify_engineered_receptacle_visual_basis(**kwargs)


def test_verifies_exact_frames_and_preserves_source_vs_authored_twin_boundary(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = _verify(fixture)

    assert result["status"] == "verified_visual_design_basis"
    assert result["cited_frame_count"] == 2
    assert {row["camera_id"] for row in result["cited_frames"]} == {
        "basket_e00",
        "towel_e00",
    }
    assert all(row["decoded_mode"] == "RGB" for row in result["cited_frames"])
    assert all(row["decoded_rgb_nontrivial"] for row in result["cited_frames"])
    assert all(
        row["visual_content_metrics"]["differing_luminance_fraction"] >= 0.10
        and row["visual_content_metrics"]["luminance_entropy_bits"] >= 0.50
        and row["visual_content_metrics"]["luminance_p05_p95_spread"] >= 8
        for row in result["cited_frames"]
    )
    assert all(
        row["visual_content_thresholds"]
        == {
            "minimum_differing_luminance_fraction": 0.10,
            "minimum_luminance_entropy_bits": 0.50,
            "minimum_luminance_p05_p95_spread": 8,
        }
        for row in result["cited_frames"]
    )
    assert all(row["calibration"]["intrinsics"]["fx"] == 4.2 for row in result["cited_frames"])
    assert result["render_manifest"]["renderer_identity"]["supersampling"] == 1
    assert result["source_receptacle_observation"] == {
        "observation_state": "occupied_and_incomplete",
        "rigid_exterior_observed": True,
        "open_rim_observed": True,
        "interior_occupied": True,
        "complete_interior_appearance_observed": False,
        "empty_source_interior_observed": False,
        "source_destination_admitted": False,
    }
    assert (
        result["engineered_twin_design_basis"]["authored_empty_interior_is_source_observation"]
        is False
    )
    assert result["engineered_twin_design_basis"]["native_simulator_qualification_required"] is True
    assert result["semantic_authority"]["semantic_authority_verified"] is True
    assert result["semantic_authority"]["learned_policy_outcomes_inspected"] is False
    assert result["claim_boundary"]["signed_semantic_authority_is_native_qualification"] is False
    assert result["basis_digest"] == canonical_digest(result, digest_field="basis_digest")


def test_rejects_stale_visual_review_topology_join(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(
            fixture,
            current_topology_receipt_digest="sha256:" + "8" * 64,
        )

    assert error.value.errors == ("visual_basis_topology_join_invalid",)


def test_mapping_only_review_cannot_cross_path_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture, visual_review_receipt_path=fixture["review"])

    assert error.value.errors == ("visual_basis_review_path_invalid",)


def test_self_rehashed_nonexistent_frame_still_fails(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing = fixture["frame_root"] / "never_existed.png"
    manifest = fixture["manifest"]
    manifest["cameras"][0]["path"] = str(missing)
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_open_failed",)


def test_rejects_frame_symlink_even_when_manifest_hash_matches_target(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = fixture["frame_paths"]["basket_e00"]
    link = fixture["frame_root"] / "linked.png"
    link.symlink_to(target)
    manifest = fixture["manifest"]
    basket = next(row for row in manifest["cameras"] if row["id"] == "basket_e00")
    basket["path"] = str(link)
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_open_failed",)


def test_hash_valid_non_png_bytes_do_not_become_visual_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = fixture["frame_paths"]["basket_e00"]
    frame.write_bytes(b"not a png despite a fully joined digest")
    manifest = fixture["manifest"]
    camera = next(row for row in manifest["cameras"] if row["id"] == "basket_e00")
    camera["bytes"] = frame.stat().st_size
    camera["digest"] = _sha256(frame)
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    citation = review["targets"][1]["cited_frames"][0]
    citation["size_bytes"] = camera["bytes"]
    citation["sha256"] = camera["digest"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_png_invalid",)


def test_every_cited_target_frame_must_exist_not_only_selected_basket(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["frame_paths"]["towel_e00"].unlink()

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_open_failed",)


def test_incomplete_source_interior_cannot_be_rewritten_as_observed_empty(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    review = fixture["review"]
    basket = review["targets"][1]
    basket["complete_interior_appearance_observed"] = True
    basket["interior_occupied"] = False
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_source_observation_invalid",)


def test_png_dimensions_must_match_bound_renderer_and_intrinsics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    manifest["camera_calibration"][0]["intrinsics"]["width"] = 5
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_camera_calibration_invalid",)


def test_final_evidence_files_are_each_opened_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    real_open = module.os.open
    counts: dict[str, int] = {}

    def counted_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        text = os.fspath(path)
        tracked = {
            str(fixture["review_path"]),
            str(fixture["manifest_path"]),
            "basket_e00.png",
            "towel_e00.png",
        }
        if text in tracked:
            counts[text] = counts.get(text, 0) + 1
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", counted_open)

    _verify(fixture)

    assert counts == {
        str(fixture["review_path"]): 1,
        str(fixture["manifest_path"]): 1,
        "basket_e00.png": 1,
        "towel_e00.png": 1,
    }


def test_same_exact_frame_cited_by_two_cameras_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture["manifest"]
    basket = next(row for row in manifest["cameras"] if row["id"] == "basket_e00")
    towel = next(row for row in manifest["cameras"] if row["id"] == "towel_e00")
    towel.update(path=basket["path"], bytes=basket["bytes"], digest=basket["digest"])
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    towel_citation = review["targets"][0]["cited_frames"][0]
    towel_citation.update(size_bytes=basket["bytes"], sha256=basket["digest"])
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)
    _refresh_semantic_authority(fixture)
    real_open = module.os.open
    frame_open_count = 0

    def counted_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal frame_open_count
        if os.fspath(path) == "basket_e00.png":
            frame_open_count += 1
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", counted_open)

    result = _verify(fixture)

    assert result["cited_frame_count"] == 2
    assert frame_open_count == 1


def test_path_replacement_during_read_fails_closed_without_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    review_path = fixture["review_path"]
    displaced = tmp_path / "displaced-review.json"
    real_open = module.os.open
    real_read = module.os.read
    review_fd: int | None = None
    replaced = False

    def tracked_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal review_fd
        fd = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == str(review_path):
            review_fd = fd
        return fd

    def replacing_read(fd: int, count: int) -> bytes:
        nonlocal replaced
        if fd == review_fd and not replaced:
            review_path.rename(displaced)
            review_path.write_text("{}\n", encoding="utf-8")
            replaced = True
        return real_read(fd, count)

    monkeypatch.setattr(module.os, "open", tracked_open)
    monkeypatch.setattr(module.os, "read", replacing_read)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert replaced is True
    assert error.value.errors == ("visual_basis_review_changed_during_read",)
    assert displaced.exists()


def test_limits_reject_oversized_json_before_parsing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture, limits=VisualBasisLimits(max_json_bytes=8))

    assert error.value.errors == ("visual_basis_review_size_invalid",)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["review_path"].write_text(
        '{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8"
    )

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_review_json_invalid",)


def test_review_digest_tampering_is_rejected_before_frame_use(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review = copy.deepcopy(fixture["review"])
    review["reviewer_id"] = "tampered"
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_review_digest_invalid",)


def test_unsigned_review_cannot_authorize_visual_design_basis(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["semantic_review_attestation_path"].unlink()

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == (
        "visual_basis_semantic_authority_unqualified:semantic_review_attestation_file_open_failed",
    )


def test_self_rehashed_review_cannot_replace_signed_semantic_decision(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    review = fixture["review"]
    review["reviewer_id"] = "self_rehashed_attacker"
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["review_path"], review)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_semantic_authority_join_invalid",)


def test_even_signed_solid_color_frame_cannot_authorize_semantics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = fixture["frame_paths"]["basket_e00"]
    Image.new("RGB", (4, 3), (90, 70, 40)).save(frame, format="PNG")
    manifest = fixture["manifest"]
    camera = next(row for row in manifest["cameras"] if row["id"] == "basket_e00")
    camera["bytes"] = frame.stat().st_size
    camera["digest"] = _sha256(frame)
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    citation = review["targets"][1]["cited_frames"][0]
    citation["size_bytes"] = camera["bytes"]
    citation["sha256"] = camera["digest"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)
    _refresh_semantic_authority(fixture)

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_visually_trivial",)


def test_signed_one_pixel_outlier_cannot_authorize_semantics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    frame = fixture["frame_paths"]["basket_e00"]
    image = Image.new("RGB", (4, 3))
    image.putdata([(90, 70, 40)] * 11 + [(99, 79, 49)])
    image.save(frame, format="PNG")
    manifest = fixture["manifest"]
    camera = next(row for row in manifest["cameras"] if row["id"] == "basket_e00")
    camera["bytes"] = frame.stat().st_size
    camera["digest"] = _sha256(frame)
    manifest["render_manifest_digest"] = canonical_digest(
        manifest, digest_field="render_manifest_digest"
    )
    review = fixture["review"]
    citation = review["targets"][1]["cited_frames"][0]
    citation["size_bytes"] = camera["bytes"]
    citation["sha256"] = camera["digest"]
    review["render_manifest_digest"] = manifest["render_manifest_digest"]
    review["review_digest"] = canonical_digest(review, digest_field="review_digest")
    _write_json(fixture["manifest_path"], manifest)
    _write_json(fixture["review_path"], review)
    _refresh_semantic_authority(fixture)

    decoded = module._decode_png_rgb(  # noqa: SLF001 - adversarial metric regression
        frame.read_bytes(),
        expected_width=4,
        expected_height=3,
        max_pixels=12,
    )
    assert decoded["visual_content_metrics"] == {
        "pixel_count": 12,
        "dominant_luminance_count": 11,
        "dominant_luminance_fraction": pytest.approx(11 / 12),
        "differing_luminance_fraction": pytest.approx(1 / 12),
        "luminance_entropy_bits": pytest.approx(0.413816850304),
        "luminance_mean": pytest.approx(73.75),
        "luminance_standard_deviation": pytest.approx(2.487468592766),
        "luminance_p05": 73,
        "luminance_p95": 73,
        "luminance_p05_p95_spread": 0,
    }
    assert decoded["decoded_rgb_nontrivial"] is False

    with pytest.raises(EngineeredReceptacleVisualBasisError) as error:
        _verify(fixture)

    assert error.value.errors == ("visual_basis_frame_visually_trivial",)
