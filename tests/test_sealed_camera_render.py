from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.sealed_camera_render import (
    SealedCameraRenderError,
    _render_harness_failure_codes,
    _wait_for_renderer_with_progress_watchdog,
    render_splat_at_exact_cameras,
    transform_camera_into_provider_frame,
)


DIGEST = "sha256:" + "a" * 64


class _StalledRendererProcess:
    """Minimal Popen double whose renderer never completes on its own."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self.terminated or self.killed:
            self.returncode = -15
            return "", "stopped"
        raise subprocess.TimeoutExpired(cmd="fixture-renderer", timeout=timeout)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_renderer_progress_watchdog_stops_before_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _StalledRendererProcess()
    clock = iter((0.0, 0.0, 1.1))
    monkeypatch.setattr(
        "blueprint_pipeline.sealed_camera_render.time.monotonic", lambda: next(clock)
    )

    with pytest.raises(SealedCameraRenderError) as exc:
        _wait_for_renderer_with_progress_watchdog(
            process=process,  # type: ignore[arg-type]
            expected_frame_paths=[tmp_path / "frames/first.png"],
            render_timeout_seconds=10.0,
            initial_progress_timeout_seconds=1.0,
            progress_timeout_seconds=1.0,
        )

    assert exc.value.codes == ("render_harness_initial_progress_timeout",)
    assert process.terminated is True


def test_renderer_progress_watchdog_stops_after_frame_progress_stalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = tmp_path / "frames/first.png"
    frame.parent.mkdir()
    frame.write_bytes(b"complete-frame")
    process = _StalledRendererProcess()
    clock = iter((0.0, 0.0, 1.0, 2.0, 4.1))
    monkeypatch.setattr(
        "blueprint_pipeline.sealed_camera_render.time.monotonic", lambda: next(clock)
    )

    with pytest.raises(SealedCameraRenderError) as exc:
        _wait_for_renderer_with_progress_watchdog(
            process=process,  # type: ignore[arg-type]
            expected_frame_paths=[frame, tmp_path / "frames/second.png"],
            render_timeout_seconds=10.0,
            initial_progress_timeout_seconds=5.0,
            progress_timeout_seconds=3.0,
        )

    assert exc.value.codes == ("render_harness_frame_progress_timeout",)
    assert process.terminated is True


def test_render_harness_failure_classifies_missing_playwright_browser() -> None:
    codes = _render_harness_failure_codes(
        stderr=(
            "browserType.launch: Executable doesn't exist at /cache/chromium "
            "Looks like Playwright was just installed. Run playwright install."
        ),
        stdout="",
        harness_output={},
    )
    assert codes == ["render_harness_failed", "render_playwright_browser_missing"]


def _write_standard_3dgs_ply(path: Path, rows: list[tuple[float, float, float, float, float, float]]) -> None:
    properties = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    body = b""
    for x, y, z, r, g, b in rows:
        body += struct.pack(
            "<14f", x, y, z, r, g, b, 8.0, -3.4, -3.4, -3.4, 1.0, 0.0, 0.0, 0.0
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + body)


def test_transform_camera_into_provider_frame_inverts_alignment() -> None:
    angle = np.deg2rad(25.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale, translation = 2.0, np.array([1.0, -2.0, 0.5])
    alignment = {
        "estimated_scale_factor": scale,
        "rotation_matrix": rotation.tolist(),
        "translation": translation.tolist(),
    }
    candidate_pose = np.eye(4)
    candidate_pose[:3, 3] = [0.4, 0.7, 1.9]
    provider_pose = np.asarray(
        transform_camera_into_provider_frame(
            camera_to_world_candidate=candidate_pose.tolist(), alignment=alignment
        )
    )
    # Mapping the provider-frame camera center forward must recover the
    # candidate-frame center: x_c = s R x_p + t.
    recovered = scale * rotation @ provider_pose[:3, 3] + translation
    assert recovered == pytest.approx(candidate_pose[:3, 3], abs=1e-12)
    assert provider_pose[:3, :3] == pytest.approx(rotation.T @ candidate_pose[:3, :3])

    with pytest.raises(SealedCameraRenderError, match="pose_invalid"):
        transform_camera_into_provider_frame(
            camera_to_world_candidate=[[float("nan")] * 4] * 4, alignment=alignment
        )


@pytest.mark.parametrize("background_rgb", [-1, 0x1000000, True, "black"])
def test_exact_camera_render_rejects_unbound_background_rgb(
    tmp_path: Path, background_rgb: object
) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            background_rgb=background_rgb,  # type: ignore[arg-type]
        )

    assert exc.value.codes == ("render_background_rgb_invalid",)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"render_timeout": 0}, "render_timeout_invalid"),
        (
            {"initial_progress_timeout_seconds": float("nan")},
            "render_initial_progress_timeout_invalid",
        ),
        (
            {"progress_timeout_seconds": 0},
            "render_progress_timeout_invalid",
        ),
        (
            {"render_timeout": 5, "progress_timeout_seconds": 6},
            "render_progress_timeout_exceeds_render_timeout",
        ),
    ],
)
def test_exact_camera_render_rejects_invalid_progress_watchdog_configuration(
    tmp_path: Path, overrides: dict[str, object], expected: str
) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])
    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            **overrides,
        )

    assert expected in exc.value.codes


def test_evaluation_authorized_render_requires_durable_camera_calibration(
    tmp_path: Path,
) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            authorization_class="evaluation_authorized",
        )

    assert exc.value.codes == (
        "render_evaluation_calibrated_camera_file_missing",
        "render_evaluation_purpose_missing",
    )


@pytest.mark.parametrize("authorization_class", ["method_input", "review_only"])
def test_other_qualified_render_classes_require_durable_camera_calibration(
    tmp_path: Path, authorization_class: str
) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            authorization_class=authorization_class,
        )

    assert exc.value.codes == (
        "render_evaluation_calibrated_camera_file_missing",
        "render_evaluation_purpose_missing",
    )


def test_render_rejects_unknown_authorization_class(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            authorization_class="evaluation-ish",
        )

    assert exc.value.codes == ("render_authorization_class_invalid",)


def test_evaluation_authorized_render_rejects_camera_file_mismatch(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    _write_standard_3dgs_ply(splat, [(0.0, 0.0, 2.0, 1.0, 1.0, 1.0)])
    camera_file = tmp_path / "calibrated_cameras.json"
    camera_file.write_text(
        json.dumps(
            [
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 31.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SealedCameraRenderError) as exc:
        render_splat_at_exact_cameras(
            splat_path=splat,
            cameras=[
                {
                    "camera_id": "fixture",
                    "T_world_camera_provider_frame": np.eye(4).tolist(),
                    "intrinsics": {
                        "fx": 32.0,
                        "fy": 32.0,
                        "cx": 16.0,
                        "cy": 16.0,
                        "width": 32,
                        "height": 32,
                    },
                }
            ],
            output_dir=tmp_path / "render",
            provider_splat_import_receipt_digest=DIGEST,
            alignment_digest=DIGEST,
            camera_set_label="fixture",
            calibrated_camera_file=camera_file,
            purpose="source_object_ownership",
            authorization_class="evaluation_authorized",
        )

    assert "render_calibrated_camera_file_mismatch" in exc.value.codes


@pytest.mark.slow
def test_exact_camera_render_places_known_gaussians_at_predicted_pixels(tmp_path: Path) -> None:
    splat = tmp_path / "scene.ply"
    bright = 1.77
    _write_standard_3dgs_ply(
        splat,
        [
            (0.0, 0.0, 2.0, bright, -1.0, -1.0),   # red at image center
            (0.4, 0.0, 2.0, -1.0, bright, -1.0),   # green right of center
            (0.0, 0.3, 2.0, -1.0, -1.0, bright),   # blue below center (OpenCV +y down)
        ],
    )
    fx = fy = 100.0
    cx, cy, width, height = 32.0, 24.0, 64, 48
    cameras = [
        {
            "camera_id": "sealed_check",
            "T_world_camera_provider_frame": np.eye(4).tolist(),
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "near": 0.05,
                "far": 100.0,
            },
        }
    ]
    calibrated_camera_file = tmp_path / "calibrated_cameras.json"
    calibrated_camera_file.write_text(json.dumps(cameras), encoding="utf-8")
    manifest = render_splat_at_exact_cameras(
        splat_path=splat,
        cameras=cameras,
        output_dir=tmp_path / "render",
        provider_splat_import_receipt_digest=DIGEST,
        alignment_digest=DIGEST,
        camera_set_label="fixture_exact_check",
        calibrated_camera_file=calibrated_camera_file,
        purpose="renderer_projection_conformance",
        authorization_class="evaluation_authorized",
        background_rgb=0x102030,
    )
    assert manifest["status"] == "rendered_exact_cameras"
    assert manifest["render_count"] == 1
    assert manifest["renderer_identity"]["background_rgb"] == "#102030"
    assert manifest["renderer_identity"]["repository_revision"]
    assert manifest["renderer_identity"]["repository_renderer_files_clean"] is True
    assert manifest["renderer_identity"]["package_version"] == "0.0.0"
    assert manifest["renderer_identity"]["dependency_versions"]["@sparkjsdev/spark"] == "2.1.0"
    assert manifest["source_splat"]["retained_gaussian_count"] == 3
    assert manifest["source_splat"]["retained_count_source"] == "verified_standard_ply_header"
    assert manifest["calibrated_camera_file"]["binding"] == "caller_file_exact_match"
    assert manifest["calibrated_camera_file"]["digest"].startswith("sha256:")
    assert manifest["calibrated_cameras"][0]["id"] == "sealed_check"
    assert manifest["render_settings"] == {
        "dimensions": {"width": width, "height": height},
        "supersampling": 1,
        "color_space": "srgb",
        "alpha_mode": "opaque_rgb",
        "background_rgb": "#102030",
        "exposure": {"mode": "renderer_default_unmodified", "ev": None},
    }
    assert manifest["purpose"] == "renderer_projection_conformance"
    assert manifest["authorization_class"] == "evaluation_authorized"
    frame = np.asarray(
        Image.open(tmp_path / "render" / manifest["renders"][0]["relative_path"]).convert("RGB")
    ).astype(np.float64)

    def peak_channel(u: float, v: float) -> int:
        patch = frame[int(v) - 4 : int(v) + 5, int(u) - 4 : int(u) + 5]
        return int(np.argmax(patch.reshape(-1, 3).max(axis=0)))

    # OpenCV projection: u = fx*x/z + cx, v = fy*y/z + cy.
    assert peak_channel(cx, cy) == 0  # red at center
    assert peak_channel(fx * 0.2 + cx, cy) == 1  # green at (52, 24)
    assert peak_channel(cx, fy * 0.15 + cy) == 2  # blue at (24, 39)
    background = frame[2:8, 2:8]
    assert float(background.max()) < 60.0
