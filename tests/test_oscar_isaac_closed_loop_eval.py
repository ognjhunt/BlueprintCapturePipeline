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
