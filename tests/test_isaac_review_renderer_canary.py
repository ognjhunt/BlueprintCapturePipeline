from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import isaac_review_renderer_canary as canary
from blueprint_pipeline.isaac_review_renderer_canary import (
    ISAAC_REVIEW_RENDERER_CANARY_SCHEMA_VERSION,
    REVIEW_FRAME_HEIGHT,
    REVIEW_FRAME_ORIENTATIONS,
    REVIEW_FRAME_WIDTH,
    main as canary_main,
    run_isaac_review_renderer_canary,
    validate_review_render_frame,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
GOOD_FRACTIONS = {"g1_pixel_fraction": 0.05, "target_marker_pixel_fraction": 0.02}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _good_frame(rows: int = 480, cols: int = 640) -> np.ndarray:
    y = np.linspace(20.0, 230.0, rows)[:, None]
    x = np.linspace(0.0, 20.0, cols)[None, :]
    base = y + x
    rng = np.random.default_rng(7)
    noise = rng.uniform(0.0, 4.0, size=(rows, cols))
    frame = np.stack([base + noise, base * 0.9, base * 0.8], axis=-1)
    return np.clip(frame, 1.0, 254.0).astype(np.uint8)


def _fake_backend(**overrides):
    result = {
        "frames": [_good_frame()],
        "g1_pixel_fraction": 0.05,
        "target_marker_pixel_fraction": 0.02,
        "rtx_driver_verifier_errors": [],
        "renderer": "FakeRayTracedLighting",
    }
    result.update(overrides)
    return lambda: result


def test_review_frame_orientation_constants() -> None:
    assert REVIEW_FRAME_HEIGHT == 480
    assert REVIEW_FRAME_WIDTH == 640
    assert REVIEW_FRAME_ORIENTATIONS["landscape"] == (480, 640)
    assert REVIEW_FRAME_ORIENTATIONS["portrait"] == (640, 480)


def test_validate_good_landscape_frame_passes() -> None:
    result = validate_review_render_frame(_good_frame(), **GOOD_FRACTIONS)

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert len(result["frame_sha256"]) == 64
    assert result["orientation"]["expected"] == "landscape"
    assert result["orientation"]["matches_expected"] is True
    assert result["clipping"]["fraction_saturated_total"] == 0.0


def test_validate_good_portrait_frame_passes() -> None:
    result = validate_review_render_frame(
        _good_frame(640, 480),
        expected_orientation="portrait",
        **GOOD_FRACTIONS,
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["orientation"]["expected_rows"] == 640
    assert result["orientation"]["expected_cols"] == 480


def test_validate_orientation_mismatch_blocked() -> None:
    result = validate_review_render_frame(
        _good_frame(640, 480),
        expected_orientation="landscape",
        **GOOD_FRACTIONS,
    )

    assert result["status"] == "blocked"
    assert "review_frame_wrong_dimensions" in result["blockers"]
    assert result["orientation"]["matches_expected"] is False


def test_validate_wrong_dimensions_blocked() -> None:
    result = validate_review_render_frame(
        np.full((240, 320, 3), 128, dtype=np.uint8),
        **GOOD_FRACTIONS,
    )

    assert "review_frame_wrong_dimensions" in result["blockers"]


def test_validate_float_zero_to_one_frame_accepted() -> None:
    frame = _good_frame().astype(np.float64) / 255.0
    result = validate_review_render_frame(frame, **GOOD_FRACTIONS)

    assert result["status"] == "passed"
    assert result["blockers"] == []


def test_validate_non_finite_pixels_blocked() -> None:
    frame = _good_frame().astype(np.float64) / 255.0
    frame[10:20, 10:20, 0] = np.nan
    frame[0, 0, 1] = np.inf
    result = validate_review_render_frame(frame, **GOOD_FRACTIONS)

    assert result["status"] == "blocked"
    assert "review_frame_non_finite_pixels" in result["blockers"]
    assert result["non_finite_pixel_count"] > 0


def test_validate_blank_black_frame_blocked() -> None:
    result = validate_review_render_frame(
        np.zeros((480, 640, 3), dtype=np.uint8),
        **GOOD_FRACTIONS,
    )

    assert result["status"] == "blocked"
    assert "review_frame_blank_black" in result["blockers"]


def test_validate_blank_white_frame_blocked() -> None:
    result = validate_review_render_frame(
        np.full((480, 640, 3), 255, dtype=np.uint8),
        **GOOD_FRACTIONS,
    )

    assert "review_frame_blank_white" in result["blockers"]


def test_validate_flat_frame_blocked() -> None:
    result = validate_review_render_frame(
        np.full((480, 640, 3), 128, dtype=np.uint8),
        **GOOD_FRACTIONS,
    )

    assert "review_frame_flat" in result["blockers"]
    assert "review_frame_blank_black" not in result["blockers"]
    assert "review_frame_blank_white" not in result["blockers"]


def test_validate_severe_clipping_blocked() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 320:, :] = 255
    result = validate_review_render_frame(frame, **GOOD_FRACTIONS)

    assert "review_frame_severe_clipping" in result["blockers"]
    assert result["clipping"]["fraction_saturated_total"] == pytest.approx(1.0)


def test_validate_missing_or_insufficient_g1_blocked() -> None:
    missing = validate_review_render_frame(
        _good_frame(),
        g1_pixel_fraction=None,
        target_marker_pixel_fraction=0.02,
    )
    insufficient = validate_review_render_frame(
        _good_frame(),
        g1_pixel_fraction=0.0001,
        target_marker_pixel_fraction=0.02,
    )

    assert "review_frame_unitree_g1_not_visible" in missing["blockers"]
    assert "review_frame_unitree_g1_not_visible" in insufficient["blockers"]


def test_validate_missing_or_insufficient_target_marker_blocked() -> None:
    missing = validate_review_render_frame(
        _good_frame(),
        g1_pixel_fraction=0.05,
        target_marker_pixel_fraction=None,
    )
    insufficient = validate_review_render_frame(
        _good_frame(),
        g1_pixel_fraction=0.05,
        target_marker_pixel_fraction=0.0001,
    )

    assert "review_frame_target_marker_missing" in missing["blockers"]
    assert "review_frame_target_marker_missing" in insufficient["blockers"]


def test_validate_checksum_reuse_blocked_and_fresh_frame_allowed() -> None:
    frame = _good_frame()
    first = validate_review_render_frame(frame, **GOOD_FRACTIONS)
    reused = validate_review_render_frame(
        frame,
        prior_frame_sha256=first["frame_sha256"],
        **GOOD_FRACTIONS,
    )
    fresh = validate_review_render_frame(
        frame,
        prior_frame_sha256="0" * 64,
        **GOOD_FRACTIONS,
    )

    assert first["status"] == "passed"
    assert "review_frame_checksum_reused_from_prior_attempt" in reused["blockers"]
    assert fresh["status"] == "passed"


def test_run_canary_blocked_without_renderer_backend(tmp_path: Path) -> None:
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=None,
    )

    persisted = _read_json(tmp_path / "isaac_review_renderer_canary.json")
    assert result["status"] == "blocked"
    assert "review_renderer_backend_unavailable" in result["blockers"]
    assert result["isaac_review_renderer_operational"] is False
    assert persisted["schema_version"] == ISAAC_REVIEW_RENDERER_CANARY_SCHEMA_VERSION
    assert persisted["fast_startup_canary_is_not_review_proof"] is True
    claim_boundary = persisted["claim_boundary"]  # type: ignore[index]
    assert claim_boundary["kitchen_scene_placement_proven"] is False
    assert claim_boundary["policy_execution_proven"] is False
    assert claim_boundary["proves_task_success"] is False
    assert claim_boundary["fast_startup_canary_is_not_review_proof"] is True
    assert persisted["secret_values_in_artifact"] is False


def test_run_canary_blocked_on_missing_nonce_and_digest(tmp_path: Path) -> None:
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="   ",
        image_digest="",
        renderer_backend=_fake_backend(),
    )

    assert result["status"] == "blocked"
    assert "review_canary_launch_nonce_missing" in result["blockers"]
    assert "review_canary_image_digest_missing" in result["blockers"]
    assert result["isaac_review_renderer_operational"] is False
    assert (tmp_path / "isaac_review_renderer_canary.json").is_file()


def test_run_canary_blocked_on_rtx_driver_verifier_errors(tmp_path: Path) -> None:
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(
            rtx_driver_verifier_errors=["rtx verifier rejected driver 570.124.06"]
        ),
    )

    assert result["status"] == "blocked"
    assert "rtx_driver_verifier_error" in result["blockers"]
    assert (tmp_path / "isaac_review_renderer_canary.json").is_file()
    assert (tmp_path / "review_renderer_canary_frame.png").is_file()


def test_run_canary_passes_and_writes_artifacts(tmp_path: Path) -> None:
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(),
    )

    frame_png = tmp_path / "review_renderer_canary_frame.png"
    persisted = _read_json(tmp_path / "isaac_review_renderer_canary.json")
    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["isaac_review_renderer_operational"] is True
    assert result["launch_session_id"] == "nonce-1"
    assert result["image_digest"] == "sha256:abc"
    assert frame_png.read_bytes().startswith(PNG_SIGNATURE)
    assert not (tmp_path / "review_renderer_canary_contact_sheet.png").exists()
    assert persisted["isaac_review_renderer_operational"] is True
    assert persisted["artifacts"]["frame_png"] == str(frame_png)  # type: ignore[index]
    assert persisted["artifacts"]["contact_sheet_png"] is None  # type: ignore[index]


def test_run_canary_writes_contact_sheet_for_multiple_frames(tmp_path: Path) -> None:
    frames = [_good_frame(), np.flipud(_good_frame()).copy()]
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(frames=frames),
    )

    contact_sheet = tmp_path / "review_renderer_canary_contact_sheet.png"
    assert result["status"] == "passed"
    assert contact_sheet.read_bytes().startswith(PNG_SIGNATURE)
    assert result["artifacts"]["contact_sheet_png"] == str(contact_sheet)
    assert result["artifacts"]["frame_count"] == 2


def test_run_canary_failed_frame_still_writes_png_and_json(tmp_path: Path) -> None:
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(frames=[np.zeros((480, 640, 3), dtype=np.uint8)]),
    )

    assert result["status"] == "blocked"
    assert "review_frame_blank_black" in result["blockers"]
    assert result["isaac_review_renderer_operational"] is False
    assert (tmp_path / "review_renderer_canary_frame.png").read_bytes().startswith(PNG_SIGNATURE)
    assert _read_json(tmp_path / "isaac_review_renderer_canary.json")["status"] == "blocked"


def test_run_canary_rejects_reused_prior_frame_checksum(tmp_path: Path) -> None:
    first = run_isaac_review_renderer_canary(
        output_dir=tmp_path / "attempt1",
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(),
    )
    reused = run_isaac_review_renderer_canary(
        output_dir=tmp_path / "attempt2",
        launch_session_id="nonce-2",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(),
        prior_frame_sha256=first["frame_validation"]["frame_sha256"],
    )

    assert first["status"] == "passed"
    assert reused["status"] == "blocked"
    assert "review_frame_checksum_reused_from_prior_attempt" in reused["blockers"]


def test_run_canary_calls_backend_close_after_json_written(tmp_path: Path) -> None:
    closed: list[bool] = []
    json_path = tmp_path / "isaac_review_renderer_canary.json"

    def _close() -> None:
        assert json_path.is_file()
        assert _read_json(json_path)["status"] == "passed"
        closed.append(True)

    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(close=_close),
    )

    assert result["status"] == "passed"
    assert closed == [True]


def test_run_canary_backend_exception_blocked(tmp_path: Path) -> None:
    def _boom() -> dict[str, object]:
        raise RuntimeError("renderer exploded")

    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_boom,
    )

    assert result["status"] == "blocked"
    assert "review_renderer_backend_failed" in result["blockers"]
    assert (tmp_path / "isaac_review_renderer_canary.json").is_file()


def test_builtin_png_writer_used_when_pillow_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    result = run_isaac_review_renderer_canary(
        output_dir=tmp_path,
        launch_session_id="nonce-1",
        image_digest="sha256:abc",
        renderer_backend=_fake_backend(),
    )

    assert result["status"] == "passed"
    assert result["artifacts"]["png_encoder"] == "builtin_zlib"
    frame_bytes = (tmp_path / "review_renderer_canary_frame.png").read_bytes()
    assert frame_bytes.startswith(PNG_SIGNATURE)
    assert frame_bytes.endswith(b"IEND\xaeB`\x82")


def test_cli_exits_two_when_backend_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "isaacsim", None)

    exit_code = canary_main(
        [
            "--output-dir",
            str(tmp_path),
            "--launch-session-id",
            "nonce-1",
            "--image-digest",
            "sha256:abc",
        ]
    )

    persisted = _read_json(tmp_path / "isaac_review_renderer_canary.json")
    assert exit_code == 2
    assert persisted["status"] == "blocked"
    assert "review_renderer_backend_unavailable" in persisted["blockers"]  # type: ignore[operator]


def test_cli_exits_zero_on_pass(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        canary,
        "_resolve_renderer_backend",
        lambda **_kwargs: _fake_backend(),
    )

    exit_code = canary_main(
        [
            "--output-dir",
            str(tmp_path),
            "--launch-session-id",
            "nonce-1",
            "--image-digest",
            "sha256:abc",
            "--orientation",
            "landscape",
        ]
    )

    summary = json.loads(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert summary == {"status": "passed", "blockers": []}
    assert _read_json(tmp_path / "isaac_review_renderer_canary.json")["status"] == "passed"


def test_cli_portrait_orientation_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        canary,
        "_resolve_renderer_backend",
        lambda **_kwargs: _fake_backend(frames=[_good_frame(640, 480)]),
    )

    exit_code = canary_main(
        [
            "--output-dir",
            str(tmp_path),
            "--launch-session-id",
            "nonce-1",
            "--image-digest",
            "sha256:abc",
            "--orientation",
            "portrait",
        ]
    )

    persisted = _read_json(tmp_path / "isaac_review_renderer_canary.json")
    assert exit_code == 0
    assert persisted["expected_orientation"] == "portrait"
