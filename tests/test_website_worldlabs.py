"""Atlas request building for website captures, and the legacy Marble poll.

Website captures no longer buy Marble worlds (owner decision 2026-10-22); the
poll tests pin the legacy device-capture provider that still uses it.
"""
import pytest


@pytest.mark.parametrize("mutation", [None, "depth", "context_grid", "target_grid", "pending"])
def test_atlas_generation_preserves_returned_bundles_and_requires_valid_grids(mutation):
    from copy import deepcopy
    from blueprint_pipeline.website_worldlabs import atlas_generate_from_pose_operation

    # Synthetic API contract only; this is not a provider receipt or usage proof.
    camera = {"intrinsics": {"width": 640, "height": 480, "fx": 500, "fy": 500,
                             "cx": 320, "cy": 240},
              "extrinsics": {"coordinateSystem": "rdf", "position": [0, 0, 0],
                             "quaternion": [0, 0, 0, 1]}}
    frames = [{"imageAsset": {"assetId": f"synthetic-rgb-{i}"}, "camera": deepcopy(camera),
               "depth": {"depthAsset": {"assetId": f"synthetic-depth-{i}"},
                         "confidenceAsset": {"assetId": f"synthetic-confidence-{i}"}}}
              for i in range(2)]
    targets = [deepcopy(camera)]
    targets[0]["intrinsics"].update(width=1280, height=720)
    operation = {"id": "synthetic-pose", "done": True, "response": {"frames": frames}}
    if mutation == "depth":
        frames[0].pop("depth")
    elif mutation == "context_grid":
        frames[1]["camera"]["intrinsics"]["width"] = 1280
    elif mutation == "target_grid":
        targets[0]["intrinsics"]["width"] = 640
    elif mutation == "pending":
        operation["done"] = False
    if mutation:
        with pytest.raises(ValueError, match="website_atlas_"):
            atlas_generate_from_pose_operation(operation=operation, expected_frame_count=2,
                                               target_cameras=targets)
        return
    before = deepcopy(operation)
    body = atlas_generate_from_pose_operation(operation=operation, expected_frame_count=2,
                                             target_cameras=targets, prompt="Synthetic scene")
    assert body["contextFrames"] == frames
    assert body["targetCameras"] == targets
    assert operation == before
    body["contextFrames"][0]["camera"]["extrinsics"]["position"][0] = 1
    assert operation == before


