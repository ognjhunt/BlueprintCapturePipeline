"""Hermetic test for the per-step policy <-> WAM <-> perception closed loop (no GPU).

Validates the loop STRUCTURE the real run depends on: each step the policy acts, a (stubbed)
WAM generates the next observation, and the perception harness runs on it immediately. The real
run swaps the stub WAM for per-step OSCAR-2B and the fixture harness backend for real SAM3/DA3,
along the same code path.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline import oscar_isaac_closed_loop_eval as L


def _write_frame(path: Path, seed: int) -> Path:
    # a non-flat, non-dark frame so the harness frame-quality gate does not reject it
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48))
    pix = img.load()
    for y in range(48):
        for x in range(64):
            pix[x, y] = ((x * 4 + seed) % 256, (y * 5 + seed) % 256, (x + y + seed) % 256)
    img.save(path)
    return path


def _stub_wam(work: Path):
    """A local stand-in WAM: writes the next-observation frame (real PNG) per step."""

    def _generate(current_frame, action, step_index, history):
        out = work / "wam_generated" / f"step_{step_index:04d}.png"
        _write_frame(out, seed=step_index * 17)
        return {"generated_frame_path": str(out)}

    return _generate


def test_closed_loop_runs_policy_wam_harness_per_step(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=3)
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["loop_kind"] == "per_step_policy_wam_perception_closed_loop"
    assert manifest["steps_executed"] == 4
    assert manifest["real_perception_backend_used"] is False  # fixture in this hermetic test

    # the trace proves per-step: policy action -> WAM frame -> harness ran, each step
    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace) == 4
    assert [r["step_index"] for r in trace] == [1, 2, 3, 4]
    for row in trace:
        assert row["policy_action"]  # policy acted
        assert Path(row["wam_generated_frame"]).is_file()  # WAM produced the next obs
        assert row["harness_step_status"] == "completed"  # harness ran on it
    # feed-forward: step 2's source observation is step 1's generated frame
    assert trace[1]["source_observation_frame"] == trace[0]["wam_generated_frame"]


def test_closed_loop_blocks_on_missing_wam_frame(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)

    def _bad_wam(current_frame, action, step_index, history):
        return {"generated_frame_path": str(tmp_path / "does_not_exist.png")}

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (2.0, 0.0, 0.79)],
        wam_generate_next=_bad_wam,
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert any("wam_generation_missing_frame" in b for b in manifest["blockers"])


def test_build_oscar_per_step_request_shapes_conditioning(tmp_path: Path) -> None:
    action = {
        "policy_action": "accepted_direct_collision_checked_motion",
        "root_position": [1.0, 2.0, 0.79],
        "root_yaw_radians": 0.5,
    }
    landmarks = [{"landmark_id": "pelvis", "image_projection": {"available": True, "u_px": 1, "v_px": 2}}]
    req = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action=action,
        step_index=3,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=landmarks,
        seed=42,
    )
    assert req["reference_frame_path"] == "/frames/cur.png"
    assert req["task_prompt"] == "walk to the sink"
    assert req["num_frames"] == 8
    assert req["seed"] == 45  # base seed + step_index
    assert req["projected_landmark_count"] == 1
    assert req["skeleton_landmarks"] == landmarks
    assert req["output_dir"].endswith("oscar_step_0003")


def test_oscar_per_step_backend_drives_the_loop(tmp_path: Path) -> None:
    """The real GPU path with OSCAR mocked: each step calls per-step OSCAR generation, the
    harness runs on the generated frame. Swapping the mock for a real OSCAR pod + real SAM3
    backend is the only change for the GPU run.
    """
    calls: list[dict] = []

    def _fake_oscar_generate(request):
        calls.append(dict(request))
        frame = tmp_path / "oscar_out" / f"step_{request['step_index']:04d}.png"
        _write_frame(frame, seed=request["step_index"] * 23 + 5)
        return {
            "status": "completed",
            "generated_frame_path": str(frame),
            "generated_video_path": str(frame.with_suffix(".mp4")),
        }

    def _skeleton_for_action(action, step_index):
        return [{"landmark_id": "pelvis", "image_projection": {"available": True, "u_px": step_index, "v_px": 1}}]

    backend = L.make_oscar_per_step_wam_backend(
        oscar_generate=_fake_oscar_generate,
        work_dir=tmp_path / "oscar_work",
        task_prompt="walk to the sink",
        num_frames=8,
        skeleton_for_action=_skeleton_for_action,
    )
    start = _write_frame(tmp_path / "start.png", seed=2)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(-4.25, -3.35, 0.79), (1.75, 1.25, 0.79)],
        wam_generate_next=backend,
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
    )
    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 3
    assert len(calls) == 3  # OSCAR called once per step
    # each per-step request carried the step's action + projected skeleton conditioning
    assert calls[0]["task_prompt"] == "walk to the sink"
    assert all(c["projected_landmark_count"] == 1 for c in calls)
    assert [c["step_index"] for c in calls] == [1, 2, 3]


def test_build_oscar_inference_argv_mirrors_entrypoint(tmp_path: Path) -> None:
    argv = L.build_oscar_inference_argv(
        python="python",
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        first_frame_path="/frames/cur.png",
        prompt="walk to the sink",
        num_frames=8,
        num_steps=35,
        guidance=6.0,
        seed=45,
        height=480,
        width=640,
        fps=15.0,
        output_video=tmp_path / "out.mp4",
        skeleton_video=tmp_path / "skel.mp4",
    )
    assert any("inference_oscar.py" in a for a in argv)
    assert argv[argv.index("--first-frame") + 1] == "/frames/cur.png"
    assert argv[argv.index("--prompt") + 1] == "walk to the sink"
    assert argv[argv.index("--num-frames") + 1] == "8"
    assert argv[argv.index("--skeleton-video") + 1].endswith("skel.mp4")


def test_local_oscar_subprocess_generate_runs_and_extracts(tmp_path: Path) -> None:
    seen_argv: list[list[str]] = []

    class _Done:
        returncode = 0

    def _fake_run(argv, **kwargs):
        seen_argv.append(list(argv))
        # simulate inference_oscar.py writing the output clip
        out = argv[argv.index("--output") + 1]
        Path(out).write_bytes(b"\x00fakeclip")
        return _Done()

    def _fake_extract(video_path: Path, out_dir: Path):
        frame = out_dir / "next_obs.png"
        _write_frame(frame, seed=11)
        return frame

    def _fake_skeleton_video(landmarks, out_dir: Path):
        v = out_dir / "skel.mp4"
        v.write_bytes(b"\x00skel")
        return v

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        run=_fake_run,
        build_skeleton_video=_fake_skeleton_video,
        extract_next_frame=_fake_extract,
    )
    request = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action={"root_position": [1, 2, 0.79]},
        step_index=1,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=[{"landmark_id": "pelvis"}],
    )
    out = gen(request)
    assert out["status"] == "completed"
    assert Path(out["generated_frame_path"]).is_file()
    assert "--skeleton-video" in seen_argv[0]  # skeleton conditioning passed to OSCAR


def test_local_oscar_subprocess_generate_blocks_on_nonzero(tmp_path: Path) -> None:
    class _Fail:
        returncode = 1

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/c",
        run=lambda argv, **k: _Fail(),
        extract_next_frame=lambda v, d: None,
    )
    out = gen({"output_dir": str(tmp_path), "reference_frame_path": "/f.png", "task_prompt": "t", "num_frames": 8, "seed": 1})
    assert out["status"] == "blocked"
    assert any("returncode" in b for b in out["blockers"])


def test_extract_last_frame_via_opencv_roundtrip(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    # 4 distinct frames; the last is solid-ish so we can verify it's the one extracted
    for i in range(4):
        frame = np.full((24, 32, 3), i * 60, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out = L.extract_last_frame_via_opencv(video, tmp_path / "extracted")
    assert out is not None and out.is_file()
    got = cv2.imread(str(out))
    assert got is not None and got.shape == (24, 32, 3)
    assert int(got.mean()) > 120  # the brightest (last) frame, not an earlier dark one


def test_closed_loop_blocks_on_empty_route(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[],
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert "blocked_empty_route" in manifest["blockers"]
