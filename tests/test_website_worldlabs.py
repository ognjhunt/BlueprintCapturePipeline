"""Atlas request building for website captures, and the legacy Marble poll.

Website captures no longer buy Marble worlds (owner decision 2026-10-22); the
poll tests pin the legacy device-capture provider that still uses it.
"""
import pytest

from blueprint_pipeline import provider_preview
from blueprint_pipeline.provider_preview import WorldLabsPreviewProvider


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


@pytest.mark.parametrize("credits", [1600, 3100, None, True, -1])
def test_marble_poll_reports_settled_provider_cost_without_inventing_zero(monkeypatch, credits):
    monkeypatch.setattr(provider_preview, "_worldlabs_api_request", lambda *_a, **_k: {
        "done": True, "response": {"world_id": "world", "world_marble_url": "https://marble.worldlabs.ai/world/world"},
        "cost": {"total_credits": credits}})
    result = WorldLabsPreviewProvider().poll(run_id="operation")
    assert result["status"] == "ready"
    if credits in (1600, 3100):
        assert result["billing_status"] == "settled"
        assert result["cost_credits"] == credits
        assert result["cost_usd"] == credits / 1250
    else:
        assert result["billing_status"] == "unreported"
        assert "cost_usd" not in result


def test_marble_terminal_error_refuses_placeholder_world_url_without_a_bill(monkeypatch):
    def operation_only(path, **_kwargs):
        assert path == "/marble/v1/operations/operation"
        return {
            "done": True,
            "metadata": {"world_id": "world"},
            "response": {"world_id": "world", "world_marble_url": "https://marble.worldlabs.ai/world/world"},
            "error": {"code": 500, "message": "An error has happened, please retry it."},
            "cost": None,
        }

    monkeypatch.setattr(provider_preview, "_worldlabs_api_request", operation_only)
    result = WorldLabsPreviewProvider().poll(run_id="operation")
    assert result["status"] == result["operation_terminal_status"] == "failed"
    assert result["operation_done"] is True
    assert result["failure_reason"] == "An error has happened, please retry it."
    assert result["launch_url"] is None and result["worldlabs_world"] is None
    assert result["billing_status"] == "unreported" and "cost_usd" not in result
