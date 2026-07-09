"""R010 — location-generalized privacy redaction.

Person-only redaction leaves worker badges/ID cards, monitors/screens with
proprietary data, whiteboards/signage, and vehicle license plates exposed on
industrial sites. These tests pin the site-type-aware behavior:

- ``privacy_processing`` expands the SAM3 detection class-set for industrial site
  types (person + badges/screens/plates/signage) and records that the
  industrial-sensitive classes were handled; non-industrial captures stay
  person-only and unchanged.
- ``proof_contracts.build_rights_provenance_review`` refuses to clear privacy for
  an industrial site until the industrial-sensitive classes were actually handled,
  surfacing an explicit ``industrial_sensitive_classes_not_handled`` blocker, while
  non-industrial captures keep clearing on person-only removal.
"""

from __future__ import annotations

from pathlib import Path

import blueprint_pipeline.privacy_processing as pp
from blueprint_pipeline.privacy_processing import (
    INDUSTRIAL_SENSITIVE_REDACTION_CLASSES,
    _redaction_classes_for_site,
    run_privacy_postprocess,
)
from blueprint_pipeline.proof_contracts import build_rights_provenance_review


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _write_video(path: Path, payload: bytes = b"fake-video") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _capture_paths(tmp_path: Path, name: str = "cap-1") -> tuple[Path, Path, Path]:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / name
    pipeline_dir = capture_root / "pipeline"
    raw_video = capture_root / "raw" / "walkthrough.mov"
    _write_video(raw_video)
    return capture_root, pipeline_dir, raw_video


def _cleared_rights_summary() -> dict:
    return {
        "consent_status": "policy_only",
        "derived_scene_generation_allowed": True,
    }


def _grounded_provenance() -> dict:
    return {"status": "grounded", "record": {"canonical_truth": True}}


# --------------------------------------------------------------------------------------
# Redaction class-set resolution
# --------------------------------------------------------------------------------------


def test_industrial_site_expands_redaction_class_set() -> None:
    classes, is_industrial = _redaction_classes_for_site("warehouse distribution center")
    assert is_industrial is True
    assert classes[0] == "person"
    # Every industrial-sensitive class is targeted in addition to person.
    for cls in INDUSTRIAL_SENSITIVE_REDACTION_CLASSES:
        assert cls in classes
    for expected in ("badge", "screen", "monitor", "vehicle license plate", "signage", "whiteboard"):
        assert expected in classes


def test_non_industrial_site_stays_person_only() -> None:
    classes, is_industrial = _redaction_classes_for_site("residential home living room")
    assert is_industrial is False
    assert classes == ["person"]


def test_unknown_site_type_stays_person_only() -> None:
    classes, is_industrial = _redaction_classes_for_site(None)
    assert is_industrial is False
    assert classes == ["person"]


# --------------------------------------------------------------------------------------
# SAM3 detection prompt carries the expanded classes (fixes both hardcoded sites)
# --------------------------------------------------------------------------------------


def test_run_sam3_http_body_prompt_includes_industrial_classes(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_http(**kwargs):
        captured["body"] = kwargs["body"]
        return {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []}

    monkeypatch.setenv("PRIVACY_SAM3_URL", "https://sam3.example")
    monkeypatch.setattr(pp, "_run_http_json", _fake_http)

    input_video = tmp_path / "raw" / "walkthrough.mov"
    _write_video(input_video)
    classes, _ = _redaction_classes_for_site("factory assembly line")
    pp._run_sam3(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_json=tmp_path / "out.json",
        output_json_uri="gs://bucket/out.json",
        masks_dir=tmp_path / "masks",
        masks_prefix_uri="gs://bucket/masks",
        stage_name="initial",
        detection_classes=classes,
    )

    body = captured["body"]
    assert body["prompt_classes"] == classes
    assert "person" in body["prompt"]
    assert "badge" in body["prompt"]
    assert "vehicle license plate" in body["prompt"]


def test_run_sam3_defaults_to_person_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_http(**kwargs):
        captured["body"] = kwargs["body"]
        return {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []}

    monkeypatch.setenv("PRIVACY_SAM3_URL", "https://sam3.example")
    monkeypatch.setattr(pp, "_run_http_json", _fake_http)

    input_video = tmp_path / "raw" / "walkthrough.mov"
    _write_video(input_video)
    pp._run_sam3(
        input_video=input_video,
        input_video_uri="gs://bucket/in.mov",
        output_json=tmp_path / "out.json",
        output_json_uri="gs://bucket/out.json",
        masks_dir=tmp_path / "masks",
        masks_prefix_uri="gs://bucket/masks",
        stage_name="initial",
    )
    assert captured["body"]["prompt"] == "person"
    assert captured["body"]["prompt_classes"] == ["person"]


# --------------------------------------------------------------------------------------
# run_privacy_postprocess threads the class-set and records handling
# --------------------------------------------------------------------------------------


def test_postprocess_industrial_site_targets_expanded_classes(monkeypatch, tmp_path: Path) -> None:
    capture_root, pipeline_dir, raw_video = _capture_paths(tmp_path, "industrial")
    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")

    seen: dict[str, object] = {}

    def _fake_sam3(**kwargs):
        seen["detection_classes"] = list(kwargs["detection_classes"])
        return {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []}

    monkeypatch.setattr(pp, "_run_sam3", _fake_sam3)
    monkeypatch.setattr(
        pp,
        "_run_depth_anything",
        lambda **_kwargs: {"status": "succeeded", "source": "depth_anything"},
    )

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="industrial",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
        site_type="warehouse loading dock",
    )

    assert result["status"] == "no_people_detected"
    assert result["is_industrial_site"] is True
    assert result["industrial_sensitive_classes_handled"] is True
    for cls in ("person", "badge", "screen", "vehicle license plate", "signage"):
        assert cls in seen["detection_classes"]
    assert set(result["industrial_sensitive_classes"]) == set(INDUSTRIAL_SENSITIVE_REDACTION_CLASSES)


def test_postprocess_non_industrial_site_unchanged(monkeypatch, tmp_path: Path) -> None:
    capture_root, pipeline_dir, raw_video = _capture_paths(tmp_path, "home")
    monkeypatch.setenv("PRIVACY_PIPELINE_ENABLED", "true")

    seen: dict[str, object] = {}

    def _fake_sam3(**kwargs):
        seen["detection_classes"] = list(kwargs["detection_classes"])
        return {"status": "succeeded", "people_detected": False, "people_count": 0, "mask_paths": []}

    monkeypatch.setattr(pp, "_run_sam3", _fake_sam3)
    monkeypatch.setattr(
        pp,
        "_run_depth_anything",
        lambda **_kwargs: {"status": "succeeded", "source": "depth_anything"},
    )

    result = run_privacy_postprocess(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="home",
        capture_root=capture_root,
        pipeline_dir=pipeline_dir,
        raw_video_path=raw_video,
        site_type="residential home kitchen",
    )

    assert result["status"] == "no_people_detected"
    assert result["is_industrial_site"] is False
    assert result["industrial_sensitive_classes_handled"] is False
    assert seen["detection_classes"] == ["person"]
    assert result["industrial_sensitive_classes"] == []


# --------------------------------------------------------------------------------------
# proof_contracts 'cleared' gate is site-type-aware
# --------------------------------------------------------------------------------------


def test_industrial_review_blocks_person_only_removal() -> None:
    """An industrial site processed person-only (classes not handled) must not clear."""

    review = build_rights_provenance_review(
        rights_summary=_cleared_rights_summary(),
        privacy_processing={"status": "person_removed"},
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        site_type="warehouse fulfillment center",
    )

    assert review["privacy"]["status"] != "cleared"
    assert review["privacy"]["is_industrial_site"] is True
    assert review["privacy"]["external_delivery_allowed"] is False
    assert "industrial_sensitive_classes_not_handled" in review["blockers"]
    assert review["status"] == "blocked"


def test_industrial_review_clears_when_classes_handled() -> None:
    review = build_rights_provenance_review(
        # R011: an industrial site that clears must carry operator authorization
        # (permission document or lawful-basis attestation). This test isolates the
        # R010 privacy-class concern, so supply a permission document to satisfy the
        # site-aware consent-evidence gate; the assertions below still verify that
        # privacy clears only when the industrial-sensitive classes were handled.
        rights_summary={
            **_cleared_rights_summary(),
            "permission_document_uri": "gs://bucket/rights/operator-permission.pdf",
        },
        privacy_processing={
            "status": "person_removed",
            "is_industrial_site": True,
            "industrial_sensitive_classes_handled": True,
        },
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        site_type="warehouse fulfillment center",
    )

    assert review["privacy"]["status"] == "cleared"
    assert review["privacy"]["external_delivery_allowed"] is True
    assert "industrial_sensitive_classes_not_handled" not in review["blockers"]
    assert review["status"] == "cleared"


def test_industrial_flag_from_privacy_manifest_gates_without_site_type() -> None:
    """Industrial-ness carried on the privacy manifest gates even when site_type is unset."""

    review = build_rights_provenance_review(
        rights_summary=_cleared_rights_summary(),
        privacy_processing={
            "status": "person_removed",
            "is_industrial_site": True,
            "industrial_sensitive_classes_handled": False,
        },
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
    )

    assert "industrial_sensitive_classes_not_handled" in review["blockers"]
    assert review["status"] == "blocked"


def test_non_industrial_review_unchanged() -> None:
    review = build_rights_provenance_review(
        rights_summary=_cleared_rights_summary(),
        privacy_processing={"status": "person_removed"},
        provenance_summary=_grounded_provenance(),
        site_identity={"site_id": "site-1"},
        adjacent_systems=[],
        site_type="residential home",
    )

    assert review["privacy"]["status"] == "cleared"
    assert review["privacy"]["is_industrial_site"] is False
    assert "industrial_sensitive_classes_not_handled" not in review["blockers"]
    assert review["status"] == "cleared"
