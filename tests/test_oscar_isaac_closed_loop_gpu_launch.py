"""Hermetic test for the closed-loop GPU pod startup builder (no GPU, no network)."""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import oscar_isaac_closed_loop_gpu_launch as G
from blueprint_pipeline.oscar_official_release import (
    OFFICIAL_OSCAR_HF_REVISION,
    OFFICIAL_OSCAR_SOURCE_COMMIT,
)


def test_startup_packages_setup_inputs_and_run(tmp_path: Path) -> None:
    start = tmp_path / "start.png"
    Image.new("RGB", (16, 12), (120, 90, 60)).save(start)

    script = G.build_closed_loop_pod_startup(
        start_frame_path=start,
        route_points=[[-4.25, -3.35, 0.79], [1.75, 1.25, 0.79]],
        steps=3,
        task_prompt="walk to the sink",
        num_frames=8,
        harness_backend_kind="fixture",
        output_put_url="https://spaces.example/out.tgz?sig=A",
    )

    # the proven OSCAR setup is reused: clone OSCAR + fetch the same checkpoint
    assert "wuzy2115/oscar-public.git" in script
    assert OFFICIAL_OSCAR_SOURCE_COMMIT in script
    assert "checkout --detach FETCH_HEAD" in script
    assert "oscar_source_commit_mismatch" in script
    assert "zywu2115/OSCAR-2B" in script
    assert f"--revision {OFFICIAL_OSCAR_HF_REVISION}" in script
    assert f'BLUEPRINT_OSCAR_WAM_HF_REVISION="{OFFICIAL_OSCAR_HF_REVISION}"' in script
    assert "HF_HUB_ENABLE_HF_TRANSFER=1" in script  # this startup installs hf_transfer first
    # Blueprint installed on the pod so the closed-loop CLI is importable
    assert "ognjhunt/BlueprintCapturePipeline.git" in script
    assert "pip install -q -e /opt/blueprint" in script
    # the loop is actually invoked with the wired args
    assert "-m blueprint_pipeline.oscar_isaac_closed_loop_eval" in script
    assert "--steps 3" in script
    assert "--oscar-repo /opt/oscar" in script
    assert "--harness-backend-kind fixture" in script
    # inputs baked in (no extra staging): start frame round-trips, route is inline
    b64 = base64.b64encode(start.read_bytes()).decode("ascii")
    assert b64 in script
    assert '"route_points"' in script
    # results upload to the signed URL
    assert "out.tgz?sig=A" in script
    assert "BLUEPRINT_CLOSED_LOOP_DONE" in script


def test_startup_omits_upload_when_no_url(tmp_path: Path) -> None:
    start = tmp_path / "s.png"
    Image.new("RGB", (8, 8), (100, 100, 100)).save(start)
    script = G.build_closed_loop_pod_startup(
        start_frame_path=start, route_points=[[0, 0, 0.79], [1, 0, 0.79]], steps=2
    )
    # no put URL -> the curl PUT guard is empty, no upload attempted
    assert 'if [ -n "" ]' in script
    assert "real_provider_probe" not in script  # default harness kind is fixture for the safe v1
