"""Hermetic contract tests for the conditional clean-plate frame-editing stage.

No network, no paid model call: the Gemini analysis is fail-closed behind its
gate, and the "analysis completed" paths are exercised by monkeypatching the
analysis function with a canned plan. These pin the scaffold contract:

- default-off flag is a pure no-op;
- the gate-off path blocks without a provider call;
- removal-plan parse/validate/build behave;
- an empty plan is a no-op, a movable-object plan defers the (unbuilt) fill;
- privacy ``failed_closed`` hard-stops;
- originals are never touched;
- the stage manifest is ``development_only`` and rejects any claim elevation;
- the stage never emits a clean-plate video in the scaffold.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.clean_plate_removal_analysis_gemini import (
    GATE_ENV,
    PROMPT_TEMPLATE_SHA256,
    analyze_removal_targets,
    build_removal_plan,
    empty_removal_plan,
    parse_removal_plan_response,
    validate_removal_plan,
)
from blueprint_pipeline.clean_plate_stage import (
    ADP_ITEM_ENV,
    ALLOWED_ADP_ITEMS,
    CLAIM_CEILING,
    FLAG_ENV,
    PROGRAM_ID,
    STAGE_MANIFEST_SCHEMA_VERSION,
    CleanPlatePolicy,
    apply_clean_plate_to_reconstruction_input,
    run_clean_plate_stage,
    validate_clean_plate_stage_manifest,
)
from blueprint_pipeline.common import read_json

_ANALYSIS_ATTR = "blueprint_pipeline.clean_plate_stage.analyze_removal_targets"
_SAFE_PRIVACY = {"status": "no_people_detected", "world_model_video_uri": "gs://b/x.mov"}


@pytest.mark.parametrize("status", ["blocked", "failed_closed", "disabled", "unknown"])
def test_website_preparation_hold_cannot_fall_back_to_original_video(status):
    original = {"status": "ready", "output_video_uri": "gs://b/original.mov"}
    result = apply_clean_plate_to_reconstruction_input(original, {"status": status}, required=True)
    assert result["status"] == "blocked"
    assert result["output_video_uri"] is None
    assert original["output_video_uri"] == "gs://b/original.mov"


def test_only_validated_preparation_selects_reconstruction_media():
    original = {"status": "ready", "output_video_uri": "gs://b/original.mov"}
    assert apply_clean_plate_to_reconstruction_input(original, {"status": "disabled"}, required=False) == original
    noop = {"status": "noop", "privacy_verified": True, "blockers": []}
    assert apply_clean_plate_to_reconstruction_input(original, noop, required=True) == original
    edited = {**noop, "status": "objects_removed", "clean_plate_video_uri": "gs://b/edited.mov"}
    assert apply_clean_plate_to_reconstruction_input(original, edited, required=True)["output_video_uri"] == "gs://b/edited.mov"
    for invalid in ({**noop, "blockers": ["task_object_ambiguous"]},
                    {**noop, "privacy_verified": False},
                    {**edited, "clean_plate_video_uri": None}):
        assert apply_clean_plate_to_reconstruction_input(original, invalid, required=True)["status"] == "blocked"

_GATE_AND_KEY_ENVS = (
    FLAG_ENV,
    GATE_ENV,
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GOOGLE_AI_API_KEY",
    "GEMINI_API_KEY_FILE",
    "GOOGLE_GENAI_API_KEY_FILE",
    "GOOGLE_AI_API_KEY_FILE",
    ADP_ITEM_ENV,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in _GATE_AND_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def _make_capture(tmp_path: Path, *, with_input_video: bool = True) -> Path:
    capture_root = tmp_path / "root" / "bucket" / "scenes" / "s1" / "captures" / "c1"
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "pipeline").mkdir(parents=True)
    (capture_root / "raw" / "walkthrough.mp4").write_bytes(b"RAWVIDEO")
    (capture_root / "raw" / "manifest.json").write_text('{"schema_version": "v3"}')
    if with_input_video:
        wl = capture_root / "pipeline" / "worldlabs_input"
        wl.mkdir(parents=True)
        (wl / "worldlabs_input.mp4").write_bytes(b"CLEANINPUTVIDEO")
    return capture_root


def _hash_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------- #
# Policy + flag
# --------------------------------------------------------------------------- #


def test_policy_default_off_and_serializes():
    policy = CleanPlatePolicy.from_env()
    assert policy.enabled is False
    assert policy.adp_item == "ADP-009B"
    assert policy.adp_item in ALLOWED_ADP_ITEMS
    round_trip = policy.to_dict()
    assert round_trip["enabled"] is False
    assert round_trip["require_privacy_verified"] is True


def test_policy_reads_flag(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    assert CleanPlatePolicy.from_env().enabled is True
    monkeypatch.setenv(FLAG_ENV, "0")
    assert CleanPlatePolicy.from_env().enabled is False


def test_disabled_is_pure_noop(tmp_path):
    capture_root = _make_capture(tmp_path)
    before = _hash_tree(capture_root)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    assert result["status"] == "disabled"
    assert result["clean_plate_video_uri"] is None
    # Nothing written anywhere, including no clean_plate dir.
    assert _hash_tree(capture_root) == before
    assert not (capture_root / "pipeline" / "clean_plate").exists()


@pytest.mark.parametrize("geometry_missing", [False, True])
def test_website_preparation_preserves_original_geometry_without_requiring_measured_scale(tmp_path, monkeypatch, geometry_missing):
    capture_root = _make_capture(tmp_path)
    source = capture_root / "raw" / "walkthrough.mp4"
    plan = empty_removal_plan(status="completed", model="test", processing="agentic")
    monkeypatch.setattr(_ANALYSIS_ATTR, lambda **_kwargs: plan)
    estimated = {"status": "estimated", "scale_status": "model_estimated", "metric_measurement_proven": False}

    def geometry(**kwargs):
        assert kwargs["source_video"] == source
        assert source.read_bytes() == b"RAWVIDEO"
        if geometry_missing:
            raise ValueError("mapanything_local_checkpoint_missing")
        return estimated

    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.run_website_scene_geometry", geometry)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY,
                                   policy=CleanPlatePolicy(enabled=True), website_source_video=source)
    if geometry_missing:
        assert result["status"] == "blocked"
        assert "mapanything_local_checkpoint_missing" in result["blockers"]
    else:
        assert result["status"] == "noop"
        assert result["source_geometry"] == estimated
        assert result["blockers"] == []


def test_website_stage_runs_geometry_masks_and_recovery_before_image_handoff(tmp_path, monkeypatch):
    capture_root = _make_capture(tmp_path)
    source = capture_root / "raw/walkthrough.mp4"
    order = []
    plan = build_removal_plan(targets=[{
        "target_id": "box", "semantic_label": "blue box", "target_class": "movable_object",
        "target_role": "task_object", "task_effect": "manipulated", "disposition": "remove",
        "rebuild_intent": "rebuild_and_compose", "spatial_evidence": [], "confidence": 0.9,
    }], status="completed", model="test", processing="agentic")
    plan["task_context_sha256"] = "task-digest"
    geometry = {"status": "estimated", "digest": "geometry-digest", "frames": [{"frame_id": "f0"}, {"frame_id": "f1"}]}
    masks = {"targets": [{"track": {"observations": [{"source_frame_id": "f0"}, {"source_frame_id": "f1"}]}}]}

    def analyze(**kwargs):
        order.append("analysis")
        assert kwargs["video_path"] == source
        return plan

    def estimate(**kwargs):
        order.append("geometry")
        return geometry

    def track(**kwargs):
        order.append("masks")
        assert kwargs["source_geometry"] == geometry
        return masks

    def recover(**kwargs):
        order.append("recovery")
        assert kwargs["task_masks"] == masks
        return [{"frame_id": frame["frame_id"], "remaining_pixel_count": 0, "recovered_pixel_count": 2}
                for frame in geometry["frames"]]

    monkeypatch.setattr(_ANALYSIS_ATTR, analyze)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.run_website_scene_geometry", estimate)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.run_website_task_masks", track)
    monkeypatch.setattr("blueprint_pipeline.clean_plate_stage.recover_observed_background", recover)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY,
                                   policy=CleanPlatePolicy(enabled=True), website_source_video=source)
    assert order == ["analysis", "geometry", "masks", "recovery"]
    assert result["status"] == "objects_removed"
    assert len(result["prepared_views"]["frames"]) == 2
    forwarded = apply_clean_plate_to_reconstruction_input({"output_video_uri": "gs://raw.mov"}, result, required=True)
    assert forwarded["output_video_uri"] is None
    assert forwarded["prepared_views"] == result["prepared_views"]


# --------------------------------------------------------------------------- #
# Fail-closed analysis (no paid call)
# --------------------------------------------------------------------------- #


def test_analysis_fail_closed_without_gate():
    plan = analyze_removal_targets(video_path=None)
    assert plan["status"] == "blocked"
    assert f"missing_env_{GATE_ENV}" in plan["blockers"]
    assert "clean_plate_input_video_not_found" in plan["blockers"]
    assert plan["targets"] == []
    assert plan["prompt_template_sha256"] == PROMPT_TEMPLATE_SHA256
    # Person removal is never authorized by the analysis.
    assert plan["authority_boundary"]["authorizes_person_pixel_removal"] is False


def test_stage_gate_off_blocks_without_provider_call(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")  # stage enabled, analysis gate still off
    capture_root = _make_capture(tmp_path)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    assert result["status"] == "blocked"
    assert result["mode"] == "analysis_blocked"
    assert f"missing_env_{GATE_ENV}" in result["blockers"]
    assert result["clean_plate_video_uri"] is None
    # Artifacts still emitted for review.
    assert Path(result["stage_manifest_path"]).is_file()
    assert Path(result["removal_plan_path"]).is_file()


# --------------------------------------------------------------------------- #
# Removal-plan schema
# --------------------------------------------------------------------------- #


def test_parse_and_validate_removal_plan():
    text = (
        "```json\n"
        '{"targets": ['
        '{"target_id": "tote_1", "semantic_label": "blue tote", '
        '"target_class": "movable_object", "disposition": "remove", '
        '"rebuild_intent": "rebuild_and_compose", "confidence": 1.4, '
        '"spatial_evidence": [{"timestamp_seconds": 3.5, '
        '"box_xywh_normalized": [0.1, 0.2, 0.3, 0.4]}]},'
        '{"target_id": "worker", "semantic_label": "person", '
        '"target_class": "person", "disposition": "remove", "rebuild_intent": "none"},'
        '{"target_id": "shelving", "semantic_label": "wall shelf", '
        '"target_class": "fixed_clutter", "disposition": "keep", "rebuild_intent": "none"},'
        '{"target_class": "not_a_class"}'
        "]}\n```"
    )
    targets = parse_removal_plan_response(text)
    assert {t["target_id"] for t in targets} == {"tote_1", "worker", "shelving"}
    tote = next(t for t in targets if t["target_id"] == "tote_1")
    assert tote["confidence"] == 1.0  # clamped
    assert tote["spatial_evidence"][0]["timestamp_seconds"] == 3.5

    plan = build_removal_plan(
        targets=targets, status="completed", model="gemini-3.7-flash", processing="agentic"
    )
    assert validate_removal_plan(plan) == []
    assert plan["movable_removal_count"] == 1
    assert plan["person_target_count"] == 1


def test_validate_removal_plan_rejects_person_authority():
    plan = empty_removal_plan(status="completed", model="m", processing="agentic")
    plan["authority_boundary"]["authorizes_person_pixel_removal"] = True
    assert "removal_plan_person_authority_boundary_invalid" in validate_removal_plan(plan)


# --------------------------------------------------------------------------- #
# Stage decisioning (monkeypatched completed analysis; no provider call)
# --------------------------------------------------------------------------- #


def test_empty_plan_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    capture_root = _make_capture(tmp_path)

    def _fake(**_kwargs):
        return empty_removal_plan(status="completed", model="gemini-3.7-flash", processing="agentic")

    monkeypatch.setattr(_ANALYSIS_ATTR, _fake)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    assert result["status"] == "noop"
    assert result["mode"] == "noop"
    assert result["movable_removal_count"] == 0
    assert result["clean_plate_video_uri"] is None
    manifest = read_json(Path(result["stage_manifest_path"]))
    assert validate_clean_plate_stage_manifest(manifest) == []


def test_movable_objects_defer_fill_machinery(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    capture_root = _make_capture(tmp_path)

    def _fake(**_kwargs):
        return build_removal_plan(
            targets=[
                {
                    "target_id": "tote_1",
                    "semantic_label": "blue tote",
                    "target_class": "movable_object",
                    "disposition": "remove",
                    "rebuild_intent": "rebuild_and_compose",
                    "spatial_evidence": [],
                    "confidence": 0.9,
                }
            ],
            status="completed",
            model="gemini-3.7-flash",
            processing="agentic",
        )

    monkeypatch.setattr(_ANALYSIS_ATTR, _fake)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    assert result["status"] == "blocked"
    assert result["mode"] == "fill_machinery_pending"
    assert result["reason"] == "clean_plate_fill_machinery_not_implemented"
    assert result["movable_removal_count"] == 1
    assert result["clean_plate_video_uri"] is None
    removal_manifest = read_json(Path(result["removal_manifest_path"]))
    assert removal_manifest["removed_target_count"] == 1
    entry = removal_manifest["entries"][0]
    assert entry["compose_back"]["replacement_asset_id"] is None
    assert entry["mask_track_ref"] is None


def test_privacy_failed_closed_hard_stops(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    capture_root = _make_capture(tmp_path)

    def _boom(**_kwargs):  # must never be called on the hard-stop path
        raise AssertionError("analysis must not run when privacy failed closed")

    monkeypatch.setattr(_ANALYSIS_ATTR, _boom)
    result = run_clean_plate_stage(
        capture_root=capture_root,
        privacy_processing={"status": "failed_closed", "reason": "raw_video_missing"},
    )
    assert result["status"] == "failed_closed"
    assert result["mode"] == "privacy_failed_closed"
    assert "privacy_pipeline_failed_closed" in result["blockers"]
    assert result["clean_plate_video_uri"] is None


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def test_originals_never_touched(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    capture_root = _make_capture(tmp_path)
    raw_before = _hash_tree(capture_root / "raw")

    def _fake(**_kwargs):
        return build_removal_plan(
            targets=[
                {
                    "target_id": "tote_1",
                    "target_class": "movable_object",
                    "disposition": "remove",
                    "rebuild_intent": "rebuild_and_compose",
                }
            ],
            status="completed",
            model="m",
            processing="agentic",
        )

    monkeypatch.setattr(_ANALYSIS_ATTR, _fake)
    run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    assert _hash_tree(capture_root / "raw") == raw_before
    # Only pipeline/clean_plate was created.
    assert (capture_root / "pipeline" / "clean_plate" / "clean_plate_stage_manifest.json").is_file()


def test_stage_manifest_is_development_only_and_valid(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    capture_root = _make_capture(tmp_path)

    def _fake(**_kwargs):
        return empty_removal_plan(status="completed", model="m", processing="agentic")

    monkeypatch.setattr(_ANALYSIS_ATTR, _fake)
    result = run_clean_plate_stage(capture_root=capture_root, privacy_processing=_SAFE_PRIVACY)
    manifest = read_json(Path(result["stage_manifest_path"]))
    assert manifest["schema_version"] == STAGE_MANIFEST_SCHEMA_VERSION
    assert manifest["program_id"] == PROGRAM_ID
    assert manifest["claim_ceiling"] == CLAIM_CEILING
    assert manifest["originals_retained"] is True
    assert all(value is False for value in manifest["claim_boundary"].values())
    assert validate_clean_plate_stage_manifest(manifest) == []


def test_validator_rejects_claim_elevation():
    base = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": "ADP-009B",
        "day_gate": "public_scene_day_14",
        "claim_ceiling": CLAIM_CEILING,
        "clean_plate_video_uri": None,
        "generated_regions_present": False,
        "claim_boundary": {
            "reconstruction_qualified": False,
            "metric_authority": False,
            "collision_authority": False,
            "hidden_surface_authority": False,
            "privacy_authority": False,
            "physical_evidence": False,
        },
    }
    assert validate_clean_plate_stage_manifest(base) == []

    elevated = {**base, "claim_ceiling": "qualified_evidence"}
    assert "clean_plate_stage_claim_ceiling_invalid" in validate_clean_plate_stage_manifest(elevated)

    wrong_program = {**base, "program_id": "some-other-program"}
    assert "clean_plate_stage_program_id_invalid" in validate_clean_plate_stage_manifest(wrong_program)

    bad_backlog = {**base, "adp_item": "ADP-999"}
    assert "clean_plate_stage_adp_item_invalid" in validate_clean_plate_stage_manifest(bad_backlog)

    with_video = {**base, "clean_plate_video_uri": "gs://b/clean.mp4"}
    assert "clean_plate_stage_video_uri_unexpected_in_scaffold" in validate_clean_plate_stage_manifest(
        with_video
    )

    elevated_boundary = {
        **base,
        "claim_boundary": {**base["claim_boundary"], "metric_authority": True},
    }
    assert "clean_plate_stage_boundary_metric_authority_elevated" in validate_clean_plate_stage_manifest(
        elevated_boundary
    )
