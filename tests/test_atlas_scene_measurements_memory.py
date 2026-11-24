"""No provider calls: a measurement pass reads each retained view's pixels once and keeps none of them.

2026-11-24, itest-cce2beee (204 posed 720x1280 views): ``_source_pose`` measured its
views, then the scale review loaded and measured them again while the first load was
still held. Each measured load kept every view's image, depth and confidence, about
2 GB; the second pass peaked at 5.46 GiB (replayed offline on the job's own views),
above the batch slice's 5 GiB memory bound, and the kernel slowed the worker there
for two hours until the unit's limit killed it. Replayed with this fix: 1.49 GiB,
the same measurements byte for byte.
"""
import gc
import tracemalloc

import numpy as np
from PIL import Image

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline import atlas_scene_measurements as sizes
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.atlas_posed_rgbd_synthetic import room_frame, write_room_pose_root

PIXELS = {"rgb", "depth", "confidence"}
WIDTH, HEIGHT, VIEWS = 320, 180, 6
VIEW_BYTES = WIDTH * HEIGHT * (3 + 4 + 4)


def posed_views(tmp_path):
    """A retained pose of ``VIEWS`` room views, and pose inputs naming each view's own source image."""
    pose_root = write_room_pose_root(tmp_path / "pose", VIEWS, width=WIDTH, height=HEIGHT)
    frames = []
    for index in range(VIEWS):
        _, rgb, _, _ = room_frame(index, VIEWS, width=WIDTH, height=HEIGHT)
        source = tmp_path / "sources" / f"{index:06d}.png"
        source.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(source)
        frames.append({"frame_id": f"view-{index:03d}", "image_path": str(source), "image_digest": _sha256_file(source),
                       "timestamp_seconds": float(index)})
    return pose_root, {"frames": frames}


def whole_view(frame_id):
    return {"source_frame_id": frame_id, "width": WIDTH, "height": HEIGHT,
            "runs": [{"start": 0, "length": WIDTH * HEIGHT}]}


def middle(frame_id):
    return {"source_frame_id": frame_id, "width": WIDTH, "height": HEIGHT,
            "runs": [{"start": y * WIDTH + WIDTH // 4, "length": WIDTH // 2} for y in range(HEIGHT // 4, HEIGHT // 2)]}


# The room is about a metre to the provider unit (its cameras walk 1.4 units above its floor).
PLAN = {"scale_references": [{"frame_id": "view-000", "label": "back wall", "endpoints_normalized": [[0.3, 0.5], [0.7, 0.5]],
                              "assumed_distance_m": 2.4, "plausible_distance_range_m": [2.2, 2.6],
                              "assumption": "the span of wall between the marks", "segmentation_prompt": "wall"}],
        "objects": []}
MASKS = [{"reference_index": 0, "frame_id": "view-000", "instances": [whole_view("view-000")]}]
TRACKS = [{"target_id": "box", "label": "box", "track": {"observations": [middle("view-001"), middle("view-002")]}}]
SURFACES = [{"target_id": "counter", "target_role": "support", "semantic_label": "kitchen counter",
             "segmentation_prompt": "kitchen counter", "track": {"observations": [middle("view-003")]}}]


def measure(frames, output):
    return sizes.estimate_sizes(frames=frames, automation=PLAN, pose_digest="sha256:pose", output=output,
                                tracks=TRACKS, reference_masks=MASKS, camera_height=True, surfaces=SURFACES)


def keeps_no_pixels(frames):
    return all(isinstance(frame, depth_splat.PosedFrame) and frame._values.keys().isdisjoint(PIXELS)
               for frame in frames)


def test_the_scale_review_measures_a_second_load_without_either_load_keeping_its_views(tmp_path):
    """``_source_pose``'s measurement, then ``repair_source_scale``'s own load and measurement, the first still held."""
    pose_root, inputs = posed_views(tmp_path)
    measure(depth_splat.load_posed_rgbd(pose_root)[1], tmp_path / "warm.json")  # first-call allocations
    gc.collect()
    tracemalloc.start()
    try:
        _, first = sizes.load_views(pose_root, inputs)
        measured = measure(first, tmp_path / "first.json")
        _, second = sizes.load_views(pose_root, inputs)
        remeasured = measure(second, tmp_path / "second.json")
        gc.collect()
        held = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    # Both loads together hold less than one view's pixels; they held two of every view's.
    assert held < VIEW_BYTES, (held, VIEW_BYTES)
    assert keeps_no_pixels(first) and keeps_no_pixels(second)
    assert all(frame["pixel_alignment"]["method"] == "pixel_checked_resize" for frame in first)
    assert measured["scale_references"] and measured["objects"][0]["status"] == "estimated"
    assert remeasured == measured


def test_a_view_read_once_per_pass_measures_exactly_what_the_decoded_view_measures(tmp_path):
    pose_root, inputs = posed_views(tmp_path)
    _, lazy = sizes.load_views(pose_root, inputs)
    _, kept = sizes.load_views(pose_root, inputs)
    eager = [{key: frame[key] for key in frame} for frame in kept]  # every pixel decoded and held, as before
    assert all(isinstance(frame["depth"], np.ndarray) for frame in eager)
    streamed, held = measure(lazy, tmp_path / "lazy.json"), measure(eager, tmp_path / "eager.json")
    assert streamed["digest"] == held["digest"]
    assert (tmp_path / "lazy.json").read_bytes() == (tmp_path / "eager.json").read_bytes()
    assert keeps_no_pixels(lazy)


def test_retained_surfaces_are_placed_on_posed_views_without_decoding_them(tmp_path):
    pose_root, inputs = posed_views(tmp_path)
    _, frames = sizes.load_views(pose_root, inputs)
    selection = {"status": "selected",
                 "source_frame_registry": [{"source_frame_id": "view-003", "decoded_pts_seconds": 3.0,
                                            "width": WIDTH, "height": HEIGHT}],
                 "targets": [{"target_id": "counter", "target_role": "support", "semantic_label": "kitchen counter",
                              "segmentation_prompt": "kitchen counter",
                              "source_track": {"track_id": "t1", "observations": [middle("view-003")]}}]}
    surfaces = sizes.surfaces_on_posed_views(frames, inputs, selection)
    assert [len(row["track"]["observations"]) for row in surfaces] == [1]
    assert surfaces[0]["track"]["observations"][0]["runs"] == middle("view-003")["runs"]
    assert keeps_no_pixels(frames)
