"""Automatic estimates/editing preserve raw geometry, source caps and quality rejection."""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import atlas_scene_measurements as sizes
from blueprint_pipeline import atlas_background_automation as edits
from blueprint_pipeline.atlas_automation_plan import AUTOMATION_INSTRUCTIONS, AutomationPlan
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def test_replay_refuses_missing_prepared_images_before_editing(tmp_path):
    with pytest.raises(ValueError, match='replay_prepared_background_missing'):
        edits.run_background(frames=[], objects=[{'disposition': 'remove'}], automation={},
            inputs={'digest': 'inputs'}, pose={'digest': 'pose'}, root=tmp_path,
            context={}, policy={}, source_digest='video', task_runner=None,
            release_blockers=lambda: ('commit', []), require_cached=True)


@pytest.mark.parametrize('missing_cache', [False, True])
def test_export_replay_skips_paid_mesh_and_preserves_canonical_result(tmp_path, monkeypatch, missing_cache):
    import json
    from blueprint_pipeline import atlas_scene_automation as worker

    frame = view(tmp_path)
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'already-paid'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    monkeypatch.setattr(sizes, 'load_views', lambda *args: (pose, [frame]))
    monkeypatch.setattr(sizes, 'estimate_sizes', lambda **kw: {})
    def retained(**kw):
        assert kw['require_cached'] is True
        if missing_cache:
            raise ValueError('atlas_export_replay_prepared_background_missing')
        return {'status': 'passed', 'stage_roots': [], 'result_digests': ['edited'],
                'visual_review_digest': 'review'}
    monkeypatch.setattr(edits, 'run_background', retained)
    monkeypatch.setattr(worker, '_edited_frames', lambda background: [frame])
    canonical = tmp_path / 'result.json'
    canonical.write_text('{"retained_production_result":true}')
    before = canonical.read_bytes()
    result = worker.run_automation(pose_root=tmp_path, inputs={}, review={'automation': plan()},
        output_root=tmp_path, task_context={}, configuration={'policy': {'mesh_export_cost_usd': .5}},
        task_runner=lambda **_: pytest.fail('paid mesh called'), pose_runner=None, release_blockers=None,
        replay_exports_only=True)
    assert result['schema_version'] == 'atlas_scene_export_replay.v1' and result['new_paid_calls'] == 0
    if missing_cache:
        assert result['status'] == 'needs_input'
        assert result['background']['reason'] == 'atlas_export_replay_prepared_background_missing'
    else:
        assert result['status'] == 'completed' and result['retained_frame_count'] == 1
        assert result['whole_scene_complete'] is False
    assert result['simulation_qualified'] is False
    assert canonical.read_bytes() == before
    saved = next((tmp_path / 'export_replays').glob('*.json'))
    assert json.loads(saved.read_text()) == result


@pytest.mark.parametrize('refused', [False, True])
def test_retained_finalization_exports_bundle_without_any_provider(tmp_path, monkeypatch, refused):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_result_bundle as bundles
    from blueprint_pipeline import atlas_scene_coverage as coverage

    frame = view(tmp_path)
    pose = {'digest': 'pose', 'binding': {'source_video_digest': 'video'},
            'operation': {'id': 'already-paid'}, 'depth_unit': 'estimated', 'scale_status': 'estimated'}
    monkeypatch.setattr(sizes, 'load_views', lambda *args: (pose, [frame]))
    monkeypatch.setattr(sizes, 'estimate_sizes', lambda **kw: {})
    def retained(**kw):
        assert kw['require_cached'] is True
        return {'status': 'passed', 'stage_roots': [], 'result_digests': ['edited'],
                'visual_review_digest': 'review'}
    monkeypatch.setattr(edits, 'run_background', retained)
    monkeypatch.setattr(worker, '_edited_frames', lambda background: [frame])
    monkeypatch.setattr(coverage, '_refresh_fusion', lambda **kw: None)
    mesh_root = tmp_path / 'completed-mesh-original-path'
    adopted, exported = [], []
    def adopt(**kw):
        assert kw['mesh_root'] == mesh_root and kw['task_context_digest'] == 'context'
        adopted.append(kw)
        if refused:
            raise ValueError('atlas_retained_mesh_output_changed')
        return {'status': 'completed'}
    monkeypatch.setattr(bundles, 'retained_mesh_for_splat', adopt)
    monkeypatch.setattr(bundles, 'export_pose_bundle', lambda **kw:
                        exported.append(kw) or {'zip_path': 'retained-scene.zip'})
    def forbidden(**kw):
        pytest.fail('provider called while finalizing retained exports')
    canonical = tmp_path / 'result.json'
    canonical.write_text('{"earlier_completed_result":true}')
    before = canonical.read_bytes()
    result = worker.run_automation(pose_root=tmp_path, inputs={}, review={'automation': plan()},
        output_root=tmp_path, task_context={'context_digest': 'context'},
        configuration={'policy': {'mesh_export_cost_usd': .5}},
        task_runner=forbidden, pose_runner=forbidden, planner_runner=forbidden, release_blockers=None,
        finalize_retained_exports=True, retained_mesh_root=mesh_root)
    assert result['new_paid_calls'] == 0
    assert result['finalized_retained_exports'] is True and result['simulation_qualified'] is False
    if refused:
        assert result['status'] == 'needs_input'
        assert result['background']['reason'] == 'atlas_retained_mesh_output_changed'
        assert canonical.read_bytes() == before and not exported
        return
    assert result['status'] == 'completed'
    assert result['background']['download']['zip_path'] == 'retained-scene.zip'
    assert len(adopted) == len(exported) == 1
    assert exported[0]['mesh_root'] == mesh_root


def view(tmp_path, key="a", position=0, width=80, height=60, centre=None, quaternion=(0.0, 0.0, 0.0, 1.0),
         depth=2.0):
    camera = {
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 100.0,
            "fy": 100.0,
            "cx": width / 2,
            "cy": height / 2,
        },
        "extrinsics": {
            "position": [position, 0.0, 0.0] if centre is None else list(centre),
            "quaternion": list(quaternion),
            "coordinateSystem": "rub",
        },
    }
    f = posed_frame(
        frame_id=key,
        rgb=np.full((height, width, 3), 90, np.uint8),
        depth=np.full((height, width), depth, np.float32),
        confidence=np.full((height, width), 255, np.uint8),
        camera=camera,
    )
    f["pixel_alignment"] = {"matrix": np.eye(3).tolist(), "source_grid": [width, height]}
    path = tmp_path / f"{key}.png"
    Image.fromarray(f["rgb"]).save(path)
    f.update(image_path=str(path), image_digest=_sha256_file(path))
    return f


def plan():
    return {
        "scale_references": [
            {
                "frame_id": "a",
                "label": "visible known-sized item",
                "endpoints_normalized": [[0.25, 0.5], [0.75, 0.5]],
                "assumed_distance_m": 0.8,
                "plausible_distance_range_m": [0.7, 0.9],
                "assumption": "class size assumption",
            }
        ],
        "objects": [
            {
                "target_id": "obj",
                "label": "task object",
                "segmentation_prompt": "task object",
                "disposition": "remove",
                "regions": [{"frame_id": "a", "box_xywh_normalized": [0.2, 0.2, 0.6, 0.6]}],
            }
        ],
        "removal_instruction": "Remove the complete task object and its parts; reveal the realistic empty space.",
        "mask_margin_fraction": 0.01,
    }


def track(frames):
    observations = []
    for f in frames:
        h, w = f["depth"].shape
        observations.append(
            {
                "source_frame_id": f["frame_id"],
                "width": w,
                "height": h,
                "runs": [
                    {"start": int(y * w + w // 4), "length": w // 2}
                    for y in range(h // 4, 3 * h // 4)
                ],
            }
        )
    return {
        "target_id": "obj",
        "label": "task object",
        "disposition": "remove",
        "track": {"observations": observations},
    }


def test_estimates_include_units_ranges_objects_and_preserve_raw_geometry(tmp_path):
    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    before = deepcopy(frames[0]["depth"])
    result = sizes.estimate_sizes(
        frames=frames,
        automation=plan(),
        pose_digest="pose",
        output=tmp_path / "estimate.json",
        tracks=[track(frames)],
    )
    assert result["meters_per_unit_estimate"] == pytest.approx(1.0)
    assert result["meters_per_unit_range"] == pytest.approx([0.875, 1.125])
    assert result["scene"]["extent_xyz_feet_estimate"] is not None
    assert result["objects"][0]["extent_xyz_m_estimate"] is not None
    assert result["metric_measurement_proven"] is False and result["raw_geometry_rescaled"] is False
    assert np.array_equal(frames[0]["depth"], before)
    assert (
        sizes.estimate_sizes(
            frames=frames,
            automation=plan(),
            pose_digest="pose",
            output=tmp_path / "estimate.json",
            tracks=[track(frames)],
        )
        == result
    )
    with pytest.raises(ValueError, match="binding_changed"):
        sizes.estimate_sizes(
            frames=frames,
            automation=plan(),
            pose_digest="changed",
            output=tmp_path / "estimate.json",
            tracks=[track(frames)],
        )


def test_missing_reference_depth_yields_relative_sizes_without_inventing_meters(tmp_path):
    f = view(tmp_path)
    f["depth"][:] = np.nan
    result = sizes.estimate_sizes(
        frames=[f], automation=plan(), pose_digest="pose", output=tmp_path / "none.json"
    )
    assert result["status"] == "relative_sizes_only" and result["meters_per_unit_estimate"] is None
    assert result["reference_gaps"][0]["reason"] == "atlas_reference_depth_missing"
    assert result["scene"]["status"] == "unavailable"


def reference(label, metres, spread=0.05):
    """A prior on the test view, whose endpoints span 0.8 provider units: metres / 0.8 m per unit."""
    return {"frame_id": "a", "label": label, "endpoints_normalized": [[0.25, 0.5], [0.75, 0.5]],
            "assumed_distance_m": metres,
            "plausible_distance_range_m": [metres * (1 - spread), metres * (1 + spread)],
            "assumption": "class size assumption"}


def scaled(tmp_path, *references):
    return sizes.estimate_sizes(frames=[view(tmp_path)], automation={**plan(), "scale_references": list(references)},
                                pose_digest="pose", output=tmp_path / "references.json")


def test_one_reference_still_scales_but_says_it_stood_alone(tmp_path):
    result = scaled(tmp_path, reference("range width", 0.8))
    assert result["meters_per_unit_estimate"] == pytest.approx(1.0)
    assert result["reference_agreement"] == {"basis": "single_object", "object_count": 1, "agreeing_object_count": 1}


def test_independent_references_that_agree_set_the_scale_together(tmp_path):
    result = scaled(tmp_path, reference("range width", 0.8), reference("countertop height", 0.82))
    assert result["status"] == "estimated"
    assert result["meters_per_unit_estimate"] == pytest.approx((1.0 + 1.025) / 2)
    assert result["reference_agreement"]["basis"] == "agreeing_objects" and not result["reference_gaps"]


def test_a_minority_reference_that_disagrees_is_set_aside(tmp_path):
    result = scaled(tmp_path, reference("range width", 0.8), reference("countertop height", 0.82),
                    reference("dishwasher opening", 1.2))
    assert result["meters_per_unit_estimate"] == pytest.approx((1.0 + 1.025) / 2)
    assert result["reference_agreement"] == {"basis": "majority_after_outliers", "object_count": 3,
                                             "agreeing_object_count": 2}
    assert [(gap["reference"]["label"], gap["reason"]) for gap in result["reference_gaps"]] == [
        ("dishwasher opening", "atlas_reference_disagrees_with_consensus")]


def test_two_priors_that_disagree_give_no_scale(tmp_path):
    result = scaled(tmp_path, reference("range width", 0.8), reference("dishwasher opening", 1.2))
    assert result["status"] == "relative_sizes_only" and result["meters_per_unit_estimate"] is None
    assert result["reference_agreement"]["basis"] == "no_majority"
    assert len(result["reference_gaps"]) == 2 and not result["scale_references"]


def test_several_views_of_one_object_never_outvote_another_object(tmp_path):
    result = scaled(tmp_path, reference("Dishwasher opening", 1.2), reference("dishwasher  opening", 1.2),
                    reference("range width", 0.8))
    assert result["reference_agreement"] == {"basis": "no_majority", "object_count": 2, "agreeing_object_count": 1}
    assert result["meters_per_unit_estimate"] is None


#: xyzw: a camera turned to look straight down.
DOWN = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
CAMERA = sizes.CAMERA_HEIGHT_LABEL
DESK = "task surface height: desk"


def walkthrough(tmp_path, heights, *, desk_top=None):
    """View ``a`` (its references span 0.8 units on a wall 2 units away) and views down at a floor at y = 0.

    Each camera stands ``heights[i]`` provider units above the floor; a camera
    looking straight down sees the floor at exactly that depth. With ``desk_top``
    the first floor view looks down on a desk top that high instead.
    """
    frames = [view(tmp_path, centre=(0.0, heights[0], 0.0))]
    spots = [(x, z) for z in (-0.8, 0.8) for x in (-1.5, -0.5, 0.5, 1.5)]
    for index, (rise, (x, z)) in enumerate(zip(heights[1:], spots, strict=False)):
        frames.append(view(tmp_path, f"floor{index}", width=40, height=30, centre=(x, rise, z), quaternion=DOWN,
                           depth=rise))
    if desk_top is not None:
        # The desk fills the middle of that view; the floor shows around it.
        desk = frames[1]
        desk["depth"][8:22, 10:30] = heights[1] - desk_top
    return frames


def surface(label, *, role="support", frame_id="floor0", prompt=None):
    """A ``website_task_masks`` target tracked over the desk region of ``walkthrough``'s first floor view."""
    observation = {"source_frame_id": frame_id, "width": 40, "height": 30,
                   "runs": [{"start": y * 40 + 10, "length": 20} for y in range(8, 22)]}
    return {"target_id": label.replace(" ", "_"), "target_role": role, "semantic_label": label,
            "segmentation_prompt": prompt or label, "disposition": "keep", "task_effect": "static_contact",
            "track": {"observations": [observation]}}


def walked(tmp_path, frames, *references, camera_height=True, surfaces=(), name="walk"):
    return sizes.estimate_sizes(frames=frames, automation={**plan(), "scale_references": list(references)},
                                pose_digest="pose", output=tmp_path / f"{name}.json", camera_height=camera_height,
                                surfaces=list(surfaces))


def reasons(result):
    return [(gap["reference"]["label"], gap["reason"]) for gap in result["reference_gaps"]]


def test_the_camera_height_breaks_a_tie_between_references_that_disagree(tmp_path):
    # Cameras 1.4 units above the floor: the 1.4 m prior says 1.0 m per unit, as the range does.
    frames = walkthrough(tmp_path, [1.4] * 9)
    tie = (reference("range width", 0.8), reference("dishwasher opening", 1.2))
    # Two disagreeing references and no third cue: unknown, exactly as before.
    assert walked(tmp_path, frames, *tie, camera_height=False, name="none")["reference_agreement"]["basis"] == \
        "no_majority"
    result = walked(tmp_path, frames, *tie)
    assert result["status"] == "estimated" and result["scale_status"] == "estimated_from_named_assumptions"
    # The range sets the scale and its range; the wide prior only voted.
    assert result["meters_per_unit_estimate"] == pytest.approx(1.0)
    assert result["meters_per_unit_range"] == pytest.approx([0.95, 1.05])
    assert [row["label"] for row in result["scale_references"]] == ["range width"]
    assert reasons(result) == [("dishwasher opening", "atlas_reference_disagrees_with_consensus")]
    agreement = result["reference_agreement"]
    assert agreement["basis"] == "majority_with_height_priors"
    assert agreement["object_count"] == 3 and agreement["agreeing_object_count"] == 2
    assert agreement["agreeing_cues"] == ["range width", CAMERA]
    assert agreement["references_alone"]["basis"] == "no_majority"
    camera = result["height_priors"]["camera"]
    assert camera["status"] == "measured" and camera["agrees_with_consensus"] is True
    assert camera["basis"] == "camera_height_prior" and camera["span_provider_units"] == pytest.approx(1.4)
    assert camera["plausible_distance_range_m"] == [1.2, 1.6] and camera["source"]
    assert result["height_priors"]["floor"]["status"] == "observed"
    # A camera between them agrees with both and so picks neither.
    between = walkthrough(tmp_path, [1.4 / 1.06] * 9)
    torn = walked(tmp_path, between, reference("range width", 0.8), reference("dishwasher opening", 0.896),
                  name="torn")
    assert torn["status"] == "relative_sizes_only" and torn["meters_per_unit_estimate"] is None
    assert torn["reference_agreement"]["basis"] == "no_majority"
    assert torn["reference_agreement"]["competing_majorities"] == 2
    assert torn["height_priors"]["camera"]["agrees_with_consensus"] is False


def test_the_camera_height_confirms_a_single_reference_and_a_contradiction_unscales_it(tmp_path):
    frames = walkthrough(tmp_path, [1.4] * 9)
    alone = walked(tmp_path, frames, reference("range width", 0.8), camera_height=False, name="alone")
    confirmed = walked(tmp_path, frames, reference("range width", 0.8), name="confirmed")
    assert confirmed["reference_agreement"]["basis"] == "single_object_with_height_priors"
    assert confirmed["reference_agreement"]["agreeing_cues"] == ["range width", CAMERA]
    # Confirmation leaves the reference's own scale and range as they were.
    for key in ("status", "meters_per_unit_estimate", "meters_per_unit_range", "scale_references", "scene"):
        assert confirmed[key] == alone[key]
    # Cameras 0.93 units up say 1.5 m per unit: two independent priors disagree, so the scale is unknown.
    low = walkthrough(tmp_path, [1.4 / 1.5] * 9)
    contradicted = walked(tmp_path, low, reference("range width", 0.8), name="contradicted")
    assert contradicted["status"] == "relative_sizes_only" and contradicted["meters_per_unit_estimate"] is None
    assert contradicted["reference_agreement"]["basis"] == "no_majority"
    assert reasons(contradicted) == [("range width", "atlas_reference_disagrees_with_consensus")]
    assert contradicted["height_priors"]["camera"]["meters_per_unit"] == pytest.approx(1.5)


def test_with_no_usable_reference_the_camera_height_alone_gives_a_wide_scale(tmp_path):
    frames = walkthrough(tmp_path, [1.75] * 9)
    before = [deepcopy(frame["depth"]) for frame in frames]
    assert walked(tmp_path, frames, camera_height=False, name="none")["status"] == "relative_sizes_only"
    result = walked(tmp_path, frames)
    assert result["status"] == "estimated" and result["scale_references"] == [] and not result["reference_gaps"]
    assert result["reference_agreement"]["basis"] == "camera_height_prior_only"
    assert result["reference_agreement"]["agreeing_cues"] == [CAMERA]
    assert result["meters_per_unit_estimate"] == pytest.approx(1.4 / 1.75)
    assert result["meters_per_unit_range"] == pytest.approx([1.2 / 1.75, 1.6 / 1.75])
    assert result["scene"]["extent_xyz_m_estimate"] is not None
    assert result["metric_measurement_proven"] is False and result["raw_geometry_rescaled"] is False
    assert all(np.array_equal(frame["depth"], depth) for frame, depth in zip(frames, before, strict=True))


@pytest.mark.parametrize("references", [
    (), (("range width", 0.8),), (("range width", 0.8), ("dishwasher opening", 1.2))])
def test_without_a_floor_the_height_priors_change_nothing(tmp_path, references):
    # Three views of one wall: no level surface, so no floor to stand cameras or a desk on.
    frames = [view(tmp_path, "a"), view(tmp_path, "b", 0.1), view(tmp_path, "c", 0.2)]
    rows = [reference(label, metres) for label, metres in references]
    before = walked(tmp_path, frames, *rows, camera_height=False, name="before")
    after = walked(tmp_path, frames, *rows, surfaces=[surface("desk", frame_id="a")], name="after")
    priors = after.pop("height_priors")
    assert priors["floor"] == {"status": "unavailable", "reason": "floor_unobserved"}
    assert priors["camera"] == {"status": "unavailable", "reason": "floor_unobserved"}
    assert [row["status"] for row in priors["surfaces"]] == ["unavailable"]
    assert {**after, "digest": None, "binding": None} == {**before, "digest": None, "binding": None}


def test_camera_heights_no_single_hold_explains_are_no_cue(tmp_path):
    tie = (reference("range width", 0.8), reference("dishwasher opening", 1.2))
    # Stairs, a drone or a phone held up: most views sit far from their median height.
    spread = walked(tmp_path, walkthrough(tmp_path, [1.4, 0.6, 3.0, 4.2, 1.0, 2.6, 5.0, 0.8, 3.6]), *tie,
                    name="spread")
    assert spread["height_priors"]["camera"]["reason"] == "camera_height_spread_implausible"
    assert spread["reference_agreement"]["basis"] == "no_majority" and spread["meters_per_unit_estimate"] is None
    # A camera below the floor means the plane is not the floor it stood on.
    frames = walkthrough(tmp_path, [1.4] * 8) + [view(tmp_path, "below", centre=(0.0, -0.5, 2.0),
                                                      depth=float("nan"))]
    below = walked(tmp_path, frames, *tie, name="below")
    assert below["height_priors"]["floor"]["reason"] == "camera_not_above_floor"
    assert below["meters_per_unit_estimate"] is None
    floor = {"status": "observed", "camera_heights_provider_units": [1.4, 1.4]}
    assert sizes.camera_height_reference([], floor) == {
        "status": "unavailable", "reason": "camera_height_too_few_views", "camera_count": 2}


@pytest.mark.parametrize("references, basis", [
    ((("range width", 0.8), ("countertop height", 0.82)), "agreeing_objects"),
    ((("range width", 0.8), ("countertop height", 0.82), ("dishwasher opening", 1.2)), "majority_after_outliers"),
])
def test_references_that_already_agree_are_byte_identical_with_the_height_priors(tmp_path, references, basis):
    # These cameras say 1.5 m per unit, siding with the outlier: consulted, the prior would split the vote.
    frames = walkthrough(tmp_path, [1.4 / 1.5] * 9)
    points = np.concatenate([sizes.frame_points(frame) for frame in frames])
    floor = sizes.observed_floor(frames, points)
    assert sizes.camera_height_reference(frames, floor)["meters_per_unit"] == pytest.approx(1.5)
    rows = [reference(label, metres) for label, metres in references]
    before = walked(tmp_path, frames, *rows, camera_height=False, name="before")
    walked(tmp_path, frames, *rows, name="after")
    assert before["reference_agreement"]["basis"] == basis and "height_priors" not in before
    assert (tmp_path / "after.json").read_bytes() == (tmp_path / "before.json").read_bytes()
    # A retained result keeps its binding and bytes when the priors are switched on.
    retained = (tmp_path / "before.json").read_bytes()
    assert walked(tmp_path, frames, *rows, name="before") == before
    assert (tmp_path / "before.json").read_bytes() == retained


def test_the_floor_under_the_cameras_is_found_in_a_tilted_world_at_any_unit(tmp_path):
    from tests.atlas_level_room_fixture import provider_units, room_parts, tilt

    rotation, offset, mpu = tilt(18.0), (0.3, -1.3, 0.2), 0.9
    cameras = [{"position": centre} for centre in provider_units(
        [(0.0, 1.2, 0.0), (0.5, 1.35, 0.3), (-0.5, 1.5, -0.2)], rotation=rotation, offset_m=offset,
        meters_per_unit=mpu)]

    def observed(**surfaces):
        world = np.concatenate([points for points, _ in room_parts(**surfaces).values()])
        world = world + np.random.default_rng(0).normal(0.0, 0.003, world.shape)
        return provider_units(world, rotation=rotation, offset_m=offset, meters_per_unit=mpu)

    floor = sizes.observed_floor(cameras, observed())
    # 1.2, 1.35 and 1.5 m over a floor at 0.9 m per unit, measured in the provider's own units.
    assert floor["camera_heights_provider_units"] == pytest.approx([1.2 / mpu, 1.35 / mpu, 1.5 / mpu], abs=0.01)
    assert np.dot(floor["normal"], rotation @ [0.0, 1.0, 0.0]) > math.cos(math.radians(0.5))
    found = sizes.camera_height_reference(cameras, floor)
    assert found["meters_per_unit"] == pytest.approx(1.4 / (1.35 / mpu), rel=0.01)
    # A counter seen without its floor has walls running on below it.
    assert sizes.observed_floor(cameras, observed(floor=False))["reason"] == "floor_unobserved"


def test_a_desk_alone_scales_the_scene_from_its_standard_height(tmp_path):
    # The cameras say nothing here (the prior is off); the desk top stands 0.8 units up: 0.74 m / 0.8.
    frames = walkthrough(tmp_path, [1.4] * 9, desk_top=0.8)
    result = walked(tmp_path, frames, camera_height=False, surfaces=[surface("desk")])
    assert result["status"] == "estimated" and result["scale_references"] == []
    assert result["reference_agreement"]["basis"] == "surface_height_prior_only"
    assert result["reference_agreement"]["agreeing_cues"] == [DESK]
    assert result["meters_per_unit_estimate"] == pytest.approx(0.74 / 0.8, rel=1e-3)
    assert result["meters_per_unit_range"] == pytest.approx([0.70 / 0.8, 0.77 / 0.8], rel=1e-3)
    desk = result["height_priors"]["surfaces"][0]
    assert desk["status"] == "measured" and desk["basis"] == "standard_surface_height_prior"
    assert desk["label"] == DESK and desk["target_role"] == "support" and desk["source"]
    assert desk["span_provider_units"] == pytest.approx(0.8, abs=1e-3)
    assert desk["observed_frame_ids"] == ["floor0"] and desk["agrees_with_consensus"] is True
    assert "camera" not in result["height_priors"]
    # Surfaces are no part of the binding: the result is read back without them.
    assert "surfaces_digest" not in result["binding"]
    assert walked(tmp_path, frames, camera_height=False) == result


def test_the_camera_height_and_a_desk_agree_or_disagree(tmp_path):
    # Cameras 1.4 units up say 1.0 m per unit; a desk top 0.74 units up says the same.
    frames = walkthrough(tmp_path, [1.4] * 9, desk_top=0.74)
    both = walked(tmp_path, frames, surfaces=[surface("standing height desk", prompt="desk"), surface(
        "office desk", role="destination")], name="both")
    # The standing desk names no standard height; the office desk is one vote.
    assert [row["status"] for row in both["height_priors"]["surfaces"]] == ["unmatched", "measured"]
    assert both["reference_agreement"]["basis"] == "height_priors_only"
    assert both["reference_agreement"]["agreeing_cues"] == [CAMERA, DESK]
    assert both["meters_per_unit_estimate"] == pytest.approx(1.0, rel=1e-3)
    # Both guesstimates set the range: the union of the two plausible ranges.
    assert both["meters_per_unit_range"] == pytest.approx([1.2 / 1.4, 1.6 / 1.4], rel=1e-3)
    # A desk top 0.5 units up says 1.48 m per unit: the two guesstimates disagree, so no scale.
    low = walkthrough(tmp_path, [1.4] * 9, desk_top=0.5)
    torn = walked(tmp_path, low, surfaces=[surface("desk")], name="torn")
    assert torn["status"] == "relative_sizes_only" and torn["meters_per_unit_estimate"] is None
    assert torn["reference_agreement"]["basis"] == "no_majority"
    assert torn["height_priors"]["camera"]["agrees_with_consensus"] is False
    assert torn["height_priors"]["surfaces"][0]["agrees_with_consensus"] is False


@pytest.mark.parametrize("label, prompt", [
    ("table", None), ("shelf above the desk", None), ("floor", None), ("bench", None),
    ("desk", "kitchen counter"), ("coffee table", "coffee table drawer")])
def test_a_surface_whose_label_names_no_one_standard_height_casts_no_vote(tmp_path, label, prompt):
    frames = walkthrough(tmp_path, [1.4] * 9, desk_top=0.8)
    result = walked(tmp_path, frames, camera_height=False, surfaces=[surface(label, prompt=prompt)])
    assert result["status"] == "relative_sizes_only" and result["meters_per_unit_estimate"] is None
    assert result["height_priors"]["surfaces"][0]["status"] == "unmatched"
    assert result["reference_agreement"]["basis"] == "no_reference"


def test_a_surface_votes_only_from_its_own_observations(tmp_path):
    frames = walkthrough(tmp_path, [1.4] * 9, desk_top=0.8)
    # A task object is no surface prior, and a surface tracked only on views not measured here is unobserved.
    obstacle, elsewhere = surface("desk", role="task_object"), surface("desk", frame_id="not-posed")
    result = walked(tmp_path, frames, camera_height=False, surfaces=[obstacle, elsewhere])
    assert result["height_priors"]["surfaces"] == [{
        "target_id": "desk", "target_role": "support", "semantic_label": "desk", "segmentation_prompt": "desk",
        "status": "unavailable", "reason": "surface_unobserved", "point_count": 0}]
    assert result["status"] == "relative_sizes_only"
    off_grid = surface("desk")
    off_grid["track"]["observations"] = [{"source_frame_id": "floor0", "width": 80, "height": 60,
                                          "runs": [{"start": 0, "length": 10}]}]
    assert walked(tmp_path, frames, camera_height=False, surfaces=[off_grid], name="grid")["height_priors"][
        "surfaces"][0]["reason"] == "atlas_measurement_mask_grid_mismatch"


def test_standard_surface_heights_are_named_ranged_and_sourced():
    for kind, phrases, assumed, (low, high), source in sizes.SURFACE_HEIGHTS:
        assert phrases and low <= assumed <= high and source
        assert sizes.standard_surface(kind)[0] == kind
    assert sizes.standard_surface("Kitchen countertop")[0] == "kitchen counter"
    assert sizes.standard_surface("bar counter")[0] == "bar counter"
    assert sizes.standard_surface("kitchen table")[0] == "dining table"
    for label in ("table", "counter", "bench", "kitchen island", "reception desk", "desk drawer", "bookshelf"):
        assert sizes.standard_surface(label) is None


@pytest.mark.parametrize(
    "change",
    [
        {"assumed_distance_m": float("nan")},
        {"endpoints_normalized": [[0, 0], [2, 1]]},
        {"plausible_distance_range_m": [1.0, 2.0]},
    ],
)
def test_invalid_scale_priors_cannot_reach_depth_sampling(change):
    value = plan()
    value["scale_references"][0].update(change)
    with pytest.raises(ValueError):
        AutomationPlan.model_validate(value)


def test_source_crop_is_checked_instead_of_silently_pairing_wrong_pixels(tmp_path):
    f = view(tmp_path)
    checked = sizes.source_to_posed(Path(f["image_path"]), f["rgb"])
    assert checked["method"] == "pixel_checked_resize"
    with pytest.raises(ValueError, match="alignment_unavailable"):
        sizes.source_to_posed(Path(f["image_path"]), np.full((60, 80, 3), 240, np.uint8))


def test_portrait_tiles_cover_full_height_and_transform_camera_rays(tmp_path):
    f = view(tmp_path, width=720, height=1280)
    prepared = edits.prepare_masked_frames(
        frames=[f], targets=[track([f])], margin=0.01, root=tmp_path / "masks"
    )
    tiles = edits.tile_views(
        frames=[f], prepared=prepared, anchor=prepared[0], root=tmp_path / "tiles"
    )
    assert len(tiles) == 4
    first, last = tiles[0], tiles[-1]
    assert first["pixel_transform"]["crop_xy"][1] == 0
    assert last["pixel_transform"]["crop_xy"][1] + 720 == round(1280 * 1280 / 720)
    for t in tiles:
        assert Image.open(t["keep_mask_path"]).mode == "L"
        assert Image.open(t["image_path"]).size == (1280, 720)
        assert np.asarray(Image.open(t["keep_mask_path"])).min() == 255
        k = t["camera"]["intrinsics"]
        sx, sy = t["pixel_transform"]["resize_xy"]
        x, y = t["pixel_transform"]["crop_xy"]
        assert k["fx"] == pytest.approx(f["intrinsics"]["fx"] * sx)
        assert k["cy"] == pytest.approx(f["intrinsics"]["cy"] * sy - y)


def test_batching_has_no_scene_frame_ceiling_and_every_batch_keeps_clean_anchor(tmp_path):
    a = view(tmp_path)
    b = view(tmp_path, "b", 0.2)
    prepared = edits.prepare_masked_frames(
        frames=[a, b], targets=[track([a, b])], margin=0.01, root=tmp_path / "masks"
    )
    tiles = edits.tile_views(
        frames=[a, b], prepared=prepared, anchor=prepared[0], root=tmp_path / "tiles"
    )
    rows = [
        {**tiles[i % len(tiles)], "frame_id": str(i), "clean_anchor": i == 0} for i in range(100)
    ]
    groups = edits.batches(rows)
    assert {r["frame_id"] for group in groups for r in group} == {str(i) for i in range(100)}
    assert all(g[0]["clean_anchor"] and len(g) <= 16 for g in groups)
    for group in groups:
        request, uploads = edits.masked_request(
            group, "Remove complete appliance; keep the surrounding room."
        )
        assert request["contextFrames"][0]["camera"] == request["targetCameras"][0]
        assert len(uploads) == 2 * len(group)


def test_empty_or_misaligned_mask_track_refuses_before_editing(tmp_path):
    f = view(tmp_path)
    t = track([f])
    t["track"]["observations"][0]["width"] = 79
    with pytest.raises(ValueError, match="grid_mismatch"):
        edits.prepare_masked_frames(frames=[f], targets=[t], margin=0.01, root=tmp_path / "bad")


def test_rejected_anchor_never_calls_atlas_or_splat_builder(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit

    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(
        edits, "segment_objects", lambda **_: [{**plan()["objects"][0], **track(frames)}]
    )
    monkeypatch.setattr(
        image_edit,
        "_validated_backend",
        lambda *a, **k: ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"),
    )
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **k: k["frames"])
    monkeypatch.setattr(
        image_edit,
        "verify_completed_background",
        lambda **_: {"status": "blocked", "review": {"task_objects_removed": False}},
    )

    def forbidden(**_):
        raise AssertionError("Rejected pixels reached Atlas")

    stages = []
    result = edits.run_background(
        frames=frames,
        objects=plan()["objects"],
        automation=plan(),
        inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
        pose={"digest": "pose"},
        root=tmp_path / "run",
        context={},
        policy={"rgbd_view_cost_usd": 0.05},
        source_digest="source",
        task_runner=forbidden,
        release_blockers=lambda: ("commit", []),
        configuration={"remaining_cost_usd": 10},
        checkpoint=stages.append,
    )
    assert result["status"] == "quality_rejected" and result["stage"] == "anchor"
    assert stages == ["automation_segmentation", "automation_anchor", "automation_anchor_review"]


def test_reframed_anchor_is_reedited_once_with_its_camera_named(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit
    from blueprint_pipeline.website_image_repair_agent import CAMERA_CHANGED_INSTRUCTION

    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(
        edits, "segment_objects", lambda **_: [{**plan()["objects"][0], **track(frames)}]
    )
    monkeypatch.setattr(
        image_edit,
        "_validated_backend",
        lambda *a, **k: ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"),
    )
    edited = []
    monkeypatch.setattr(image_edit, "complete_background_images",
                        lambda **k: edited.append(k["repair_instruction"]) or k["frames"])
    reviews = iter([{"status": "blocked", "basis": "deterministic_view_geometry",
                     "review": {"camera_changed_frame_ids": ["a"]}},
                    {"status": "blocked", "review": {"task_objects_removed": False}}])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **_: next(reviews))

    def forbidden(**_):
        raise AssertionError("Rejected pixels reached Atlas")

    stages = []
    result = edits.run_background(
        frames=frames, objects=plan()["objects"], automation=plan(),
        inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]}, pose={"digest": "pose"},
        root=tmp_path / "run", context={}, policy={"rgbd_view_cost_usd": 0.05}, source_digest="source",
        task_runner=forbidden, release_blockers=lambda: ("commit", []),
        configuration={"remaining_cost_usd": 10}, checkpoint=stages.append,
    )
    # One camera re-edit, then the independent review's verdict stands.
    assert edited == [plan()["removal_instruction"], plan()["removal_instruction"] + " " + CAMERA_CHANGED_INSTRUCTION]
    assert result["status"] == "quality_rejected" and result["stage"] == "anchor"
    assert stages == ["automation_segmentation", "automation_anchor", "automation_anchor_review",
                      "automation_anchor_camera_reedit"]


def test_insufficient_entire_path_budget_refuses_before_sam(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(
        image_edit,
        "_validated_backend",
        lambda *a, **k: ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"),
    )
    monkeypatch.setattr(
        edits, "segment_objects", lambda **_: (_ for _ in ()).throw(AssertionError("SAM called"))
    )
    with pytest.raises(ValueError, match="budget_insufficient"):
        edits.run_background(
            frames=frames,
            objects=plan()["objects"],
            automation=plan(),
            inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
            pose={},
            root=tmp_path / "run",
            context={},
            policy={"rgbd_view_cost_usd": 0.05},
            source_digest="source",
            task_runner=None,
            release_blockers=lambda: ("commit", []),
            configuration={"remaining_cost_usd": 0},
        )


def test_failed_automation_preserves_measurement_result_without_reconstructing(
    tmp_path, monkeypatch
):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline import atlas_depth_splat

    f = view(tmp_path)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [f]))
    monkeypatch.setattr(
        edits, "run_background", lambda **_: {"status": "quality_rejected", "stage": "anchor"}
    )
    monkeypatch.setattr(
        atlas_depth_splat,
        "write_depth_splat",
        lambda **_: pytest.fail("rejected edit was reconstructed"),
    )
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan()},
        output_root=tmp_path / "out",
        task_context={},
        configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("provider called"),
        pose_runner=None,
        release_blockers=lambda: ("commit", []),
    )
    assert (
        result["status"] == "quality_rejected" and result["measurements"]["status"] == "estimated"
    )
    assert result["raw_capture_preserved"] and not result["simulation_qualified"]


def test_boxes_are_clipped_in_returned_grid_and_outside_boxes_refuse_before_sam(tmp_path):
    f = view(tmp_path)
    f["pixel_alignment"]["matrix"][0][2] = 50
    args = dict(frames=[f], objects=plan()["objects"], inputs={"frames": [{"frame_id": "a"}]})
    box = edits.object_evidence(**args)["obj"][0]["box_xywh_normalized"]
    assert box[0] + box[2] == pytest.approx(1)
    f["pixel_alignment"]["matrix"][0][2] = 500
    with pytest.raises(ValueError, match="outside_returned_view"):
        edits.object_evidence(**args)


def test_offline_measurements_retain_public_citation_lineage(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker

    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [view(tmp_path)]))
    source = {
        "frame_id": "a",
        "label": plan()["scale_references"][0]["label"],
        "url": "https://manufacturer.example/spec",
        "status": "unconfirmed_reference",
    }
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan(), "scale_reference_sources": [source], "anchor_frame_id": "a"},
        output_root=tmp_path / "offline",
        task_context={},
        configuration={},
        execute=False,
        task_runner=lambda **_: pytest.fail("offline provider call"),
        pose_runner=None,
        release_blockers=None,
    )
    assert result["measurements"]["scale_references"][0]["public_reference_readback"] == source


def test_passed_edit_constructs_and_exports_generated_splat_after_review(tmp_path, monkeypatch):
    from blueprint_pipeline import (
        atlas_scene_automation as worker,
        atlas_depth_splat,
        atlas_result_bundle,
    )

    f = view(tmp_path)
    pose = {"digest": "pose", "binding": {"source_video_digest": "source"}}
    monkeypatch.setattr(sizes, "load_views", lambda *a: (pose, [f]))
    monkeypatch.setattr(
        edits,
        "run_background",
        lambda **_: {
            "status": "passed",
            "provider": "atlas",
            "stage_roots": [],
            "result_digests": ["edited-task"],
            "visual_review_digest": "review",
            "batch_frame_ids": [],
        },
    )
    monkeypatch.setattr(worker, "_edited_frames", lambda _: [f])
    calls = []

    def build(**kw):
        calls.append(kw)
        assert (
            kw["generated_result_digests"] == ["edited-task"]
            and kw["visual_review_digest"] == "review"
        )
        return {"digest": "generated-splat"}

    monkeypatch.setattr(atlas_depth_splat, "write_depth_splat", build)
    monkeypatch.setattr(
        atlas_result_bundle,
        "export_pose_bundle",
        lambda **kw: {"zip_path": "generated-background.zip"},
    )
    result = worker.run_automation(
        pose_root=tmp_path,
        inputs={},
        review={"automation": plan()},
        output_root=tmp_path / "out",
        task_context={},
        configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("unexpected provider call"),
        pose_runner=None,
        release_blockers=lambda: ("commit", []),
    )
    assert result["status"] == "completed" and len(calls) == 1
    assert result["background"]["download"]["zip_path"] == "generated-background.zip"
    assert result["background"]["mesh_available"] is False


def test_public_reference_quote_requires_retained_page_text_and_matching_units(tmp_path):
    from blueprint_pipeline.atlas_reference_research import retain_references

    class Fetcher:
        log = []

        def fetch(self, url):
            path = tmp_path / "public-page.txt"
            path.write_text("Range width: 30 inches. Category dimensions vary by model.")
            self.log.append({"url": url, "status": "ok", "text_path": str(path)})
            return {"status": "ok", "url": url}

    proposal = {"automation": plan()}
    reference = proposal["automation"]["scale_references"][0]
    reference.update(
        assumed_distance_m=0.762,
        citation_url="https://manufacturer.example/spec",
        citation_quote="Range width: 30 inches.",
    )
    valid = retain_references(proposal, tmp_path, fetcher=Fetcher())
    assert (
        valid[0]["status"] == "retained_public_reference" and not valid[0]["physical_measurement"]
    )
    reference["citation_quote"] = "Range width: 99 inches."
    invalid = retain_references(proposal, tmp_path, fetcher=Fetcher())
    assert invalid[0]["status"] == "unconfirmed_reference" and invalid[0]["quote_found"] is False


def test_reviewed_background_resume_reuses_images_before_any_provider_preflight(tmp_path, monkeypatch):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    image = view(tmp_path)
    row = {"frame_id": "a", "image_path": image["image_path"],
           "image_digest": image["image_digest"], "original_image_path": image["image_path"],
           "original_image_digest": image["image_digest"]}
    review_binding = {"frames": [{"frame_id": "a", "image_digest": row["image_digest"]}]}
    background = {"status": "fallback_prepared", "ready_for_reconstruction": True,
                  "frames": [row], "review": {"status": "passed", "binding": review_binding,
                  "request_digest": canonical_digest(review_binding), "review": {
                      "consistent_background": True, "task_objects_removed": True,
                      "unrelated_objects_preserved": True}}}
    args = dict(frames=[image], objects=plan()["objects"], automation=plan(),
                inputs={"digest": "inputs"}, pose={"digest": "pose"}, root=tmp_path / "run",
                context={}, policy={}, source_digest="source", task_runner=None,
                release_blockers=lambda: ("commit", []))
    args["root"].mkdir()
    binding = {"source_video_digest": "source", "source_pose_digest": "pose", "inputs_digest": "inputs",
               "automation_digest": canonical_digest(plan()), "task_context_digest": canonical_digest({}),
               "selected_frame_ids": None, "editor_first": True}
    path = args["root"] / "prepared_background.json"
    edits._prepared_background(path, binding, background)
    monkeypatch.setattr(edits, "_provider_preflight", lambda: pytest.fail("bought completed edits again"))
    monkeypatch.setattr(edits, "object_evidence", lambda **_: pytest.fail("repeated completed masks"))
    assert edits.run_background(**args) == background
    with pytest.raises(ValueError, match="binding_changed"):
        edits.run_background(**{**args, "source_digest": "different-video"})
    Path(row["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="pixels_changed"):
        edits.run_background(**args)


def test_rejected_or_unbound_background_cannot_be_adopted(tmp_path):
    for review in ({"status": "blocked"}, {"status": "passed", "binding": {"frames": []}}):
        with pytest.raises(ValueError, match="review_invalid"):
            edits._prepared_background(tmp_path / "prepared.json", {}, {
                "status": "fallback_prepared", "ready_for_reconstruction": True,
                "frames": [{"frame_id": "a", "image_digest": "sha256:a"}], "review": review})
    assert not (tmp_path / "prepared.json").exists()


def test_source_cleanup_needs_no_pose_and_keeps_only_current_reviewed_frames(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    inputs = {"frames": [{k: f[k] for k in ("frame_id", "image_path", "image_digest")} for f in (a, b)],
              "binding": {"source_video_digest": "source"}, "digest": "inputs"}
    observed = []

    def reject(**kw):
        observed.append(kw)
        assert [f["frame_id"] for f in kw["frames"]] == ["a"]
        assert "depth" not in kw["frames"][0] and "extrinsics" not in kw["frames"][0]
        assert not kw["frames"][0]["camera_calibrated"]
        return {"status": "quality_rejected"}

    monkeypatch.setattr(edits, "run_background", reject)
    result = worker.run_automation(pose_root=None, inputs=inputs,
        review={"automation": plan(), "coverage_requirements": [{"required_frame_ids": ["a"]}]},
        output_root=tmp_path / "out", task_context={}, configuration={"policy": {}},
        task_runner=lambda **_: pytest.fail("rejected image reached Atlas"),
        pose_runner=lambda **_: pytest.fail("rejected image reached Atlas"), release_blockers=None)
    assert result["status"] == "quality_rejected" and result["source_pose_digest"] is None
    assert observed[0]["selected_frame_ids"] == ["a"]
    assert result["digest"] == canonical_digest(result, digest_field="digest")


def test_accepted_rgb_passes_real_pose_capture_boundary_before_any_paid_runner(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_automation as worker, website_atlas_pose as atlas

    monkeypatch.setenv(atlas.ATLAS_PRICE_REVIEW_ENV, "")
    frame, second = view(tmp_path, width=80, height=45), view(tmp_path, "b", .1, width=80, height=45)
    root = tmp_path / "job"
    legacy = root / "fallback_pose/atlas_pose_inputs.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"refused_legacy_handoff": true}')
    called = []

    def validate(**kw):
        inputs = atlas.load_pending_atlas_pose_inputs(kw["output_root"])
        assert inputs["generated_input_content"] is True
        assert Path(inputs["frames"][0]["image_path"]).is_relative_to(kw["output_root"])
        assert inputs["full_view_sources"][0]["image_digest"] == frame["image_digest"]
        from PIL import Image
        from blueprint_pipeline.atlas_context_fov import full_view_sources
        full_view_sources(inputs, kw["output_root"])
        assert Image.open(inputs["frames"][0]["image_path"]).width == Image.open(inputs["frames"][0]["image_path"]).height
        assert atlas.atlas_pose_target_resolution(inputs) == [1280, 720]
        called.append(kw)
        return {"status": "processing", "operation_id": "not-purchased"}

    result, state = worker._fallback_pose({"frames": [frame, second]}, root, "source", {"capture_id": "c"},
        {"pose_input_cost_usd": .01, "rate_card_reference": "reviewed", "reviewed_by": "test",
         "reviewed_on": "2026-10-22"}, validate, 2)
    assert result is None and state["status"] == "processing" and len(called) == 1
    assert legacy.read_text() == '{"refused_legacy_handoff": true}'
    (legacy.parent / "atlas_pose").mkdir()
    (legacy.parent / "atlas_pose/submission.json").write_text('{"status":"submitting"}')
    with pytest.raises(ValueError, match="legacy_pose_requires_reconciliation"):
        worker._fallback_pose({"frames": [frame]}, root, "source", {}, {}, validate, 2)
    assert len(called) == 1


def test_rejected_atlas_fallback_enters_existing_repair_and_retains_passed_set(tmp_path, monkeypatch):
    from blueprint_pipeline import website_image_completion as image_edit, website_image_repair_agent as repair
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    monkeypatch.setattr(edits, "segment_objects", lambda **_: [{**plan()["objects"][0], **track(frames)}])
    monkeypatch.setattr(image_edit, "_validated_backend", lambda *a, **k:
                        ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"))
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **k: [{
        **row, "original_image_path": row["image_path"], "original_image_digest": row["image_digest"],
    } for row in k["frames"]])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **k:
                        {"status": "passed" if k["output_root"].name == "anchor_review" else "blocked"})
    monkeypatch.setattr(repair, "image_repair_enabled", lambda: True)
    calls = []

    def repaired(**kw):
        calls.append(kw)
        binding = {"frames": [{"frame_id": r["frame_id"], "image_digest": r["image_digest"]}
                              for r in kw["selected"]]}
        return kw["selected"], {"status": "passed", "binding": binding,
                                "request_digest": canonical_digest(binding), "review": {
                                    "consistent_background": True, "task_objects_removed": True,
                                    "unrelated_objects_preserved": True}}

    monkeypatch.setattr(repair, "repair_rejected_views", repaired)

    def task(**kw):
        return {"status": "completed", "outputs": [
            {"role": f"frames-{i}-imageAsset", "path": "unused.png", "sha256": "unused"}
            for i in range(len(kw["request"]["contextFrames"]))]}

    args = dict(frames=frames, objects=plan()["objects"], automation=plan(), editor_first=False,
                inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]},
                pose={"digest": "pose"}, root=tmp_path / "run", context={},
                policy={"rgbd_view_cost_usd": 0.05, "rate_card_reference": "reviewed",
                        "reviewed_by": "operator", "reviewed_on": "2026-10-22"},
                source_digest="video", task_runner=task,
                release_blockers=lambda: ("commit", []), configuration={"remaining_cost_usd": 10})
    result = edits.run_background(**args)
    assert result["status"] == "fallback_prepared" and len(calls) == 1
    assert calls[0]["max_repairs"] == 2 and calls[0]["source_video_digest"] == "video"
    assert calls[0]["repair_reference_path"].is_file()
    assert edits.run_background(**args) == result and len(calls) == 1


@pytest.mark.parametrize("makes_progress", [True, False])
def test_editor_repairs_follow_latest_review_and_stop_when_pixels_do_not_change(tmp_path, monkeypatch, makes_progress):
    from blueprint_pipeline import website_image_completion as image_edit, website_image_repair_agent as repair
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    prepared = edits.prepare_masked_frames(frames=frames, targets=[track(frames)], margin=.01,
                                           root=tmp_path / "originals")
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(image_edit, "complete_background_images", lambda **kw: kw["frames"])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **_: {"status": "blocked"})
    monkeypatch.setattr(repair, "image_repair_enabled", lambda: True)
    calls = []

    def fix(**kw):
        calls.append(kw)
        selected = list(map(dict, kw["selected"]))
        if makes_progress:
            index = len(calls) - 1
            path = tmp_path / f"repair-{index}.png"
            Image.new("RGB", (80, 60), "red" if index == 0 else "blue").save(path)
            selected[index].update(image_path=str(path), image_digest=_sha256_file(path))
        binding = {"frames": [{"frame_id": row["frame_id"], "image_digest": row["image_digest"]}
                              for row in selected]}
        passed = makes_progress and len(calls) == 2
        review = {"status": "passed" if passed else "blocked", "binding": binding,
                  "request_digest": canonical_digest(binding), "review": {
                      "consistent_background": passed, "task_objects_removed": True,
                      "unrelated_objects_preserved": True}}
        return selected, review

    monkeypatch.setattr(repair, "repair_rejected_views", fix)
    result = edits._reviewed_editor_background(prepared=prepared, anchor=prepared[0],
        plan={"task_context_sha256": "task", "targets": []}, automation=plan(), root=root, context={},
        source_digest="video", measurements={}, selected_frame_ids=None, check=lambda _: None, prepared_binding={})
    assert len(calls) == (2 if makes_progress else 1)
    assert result["status"] == ("fallback_prepared" if makes_progress else "quality_rejected")
    if makes_progress:
        assert calls[1]["failed_review"]["status"] == "blocked"
        assert calls[1]["selected"][0]["image_path"].endswith("repair-0.png")
        assert calls[1]["output_root"].name == "fallback_repairs-0002"
    assert (root / "prepared_background.json").exists() is makes_progress


def _sam_person_tracks(frames):
    """Two people in the first view (one instance each), nobody in the second."""
    first = frames[0]
    h, w = first["rgb"].shape[:2]
    people = []
    for index, columns in enumerate([(2, 10), (60, 70)]):
        runs = [{"start": y * w + columns[0], "length": columns[1] - columns[0]} for y in range(10, 50)]
        people.append({"track_id": f"p{index}", "label": "transient-0", "observations": [
            {"source_frame_id": first["frame_id"], "width": w, "height": h, "runs": runs}]})
    return people


def _transient_run(tmp_path, monkeypatch, tracks):
    from blueprint_pipeline import meta_sam31, website_image_completion as image_edit

    frames = [view(tmp_path), view(tmp_path, "b", 0.1)]
    for frame in frames:
        del frame["depth"]  # Editor-first runs before any depth is bought.
    prompts, edits_seen = [], []
    monkeypatch.setattr(edits, "_provider_preflight", lambda: None)
    monkeypatch.setattr(meta_sam31, "run_meta_sam31",
                        lambda **kw: prompts.extend(kw["prompts"]) or {"tracks": tracks(frames)})
    monkeypatch.setattr(image_edit, "_validated_backend", lambda *a, **k:
                        ({}, {"pricing_binding": {"max_cost_per_request_usd": 0.2}}, "b"))
    monkeypatch.setattr(image_edit, "complete_background_images",
                        lambda **k: edits_seen.append(k) or k["frames"])
    monkeypatch.setattr(image_edit, "verify_completed_background", lambda **_: {"status": "blocked"})
    result = edits.run_background(
        frames=frames, objects=[], automation={**plan(), "objects": []},
        inputs={"frames": [{"frame_id": f["frame_id"]} for f in frames]}, pose={"digest": "pose"},
        root=tmp_path / "run", context={}, policy={"rgbd_view_cost_usd": 0.05}, source_digest="video",
        task_runner=None, release_blockers=lambda: ("commit", []), configuration={"remaining_cost_usd": 10},
        transients=[{"target_id": "transient-0", "label": "person", "segmentation_prompt": "person"}])
    return result, prompts, edits_seen


def test_every_person_in_the_one_video_is_erased_even_with_no_task_object_to_remove(tmp_path, monkeypatch):
    result, prompts, edits_seen = _transient_run(tmp_path, monkeypatch, _sam_person_tracks)
    assert prompts == [{"prompt_id": "transient-0", "output_label": "transient-0", "text": "person"}]
    anchor = edits_seen[0]["frames"][0]
    # Both people are erased, not one chosen instance.
    assert anchor["frame_id"] == "a" and anchor["removed_transient_ids"] == ["transient-0"]
    assert anchor["removed_task_object_ids"] == [] and anchor["remaining_pixel_count"] > 2 * 8 * 40
    mask = np.asarray(Image.open(anchor["remaining_mask_path"])) == 255
    assert mask[30, 5] and mask[30, 65] and not mask[30, 35]
    empty = np.asarray(Image.open(tmp_path / "run/originals/0001-edit.png"))
    assert not empty.any()  # Nobody in the second view: nothing to edit there.
    transient = next(t for t in edits_seen[0]["targets"] if t["target_id"] == "transient-0")
    assert transient["task_effect"] == "transient" and transient["disposition"] == "remove"
    assert result["status"] == "quality_rejected" and result["stage"] == "anchor"


def test_a_scene_with_no_people_found_and_nothing_to_remove_buys_no_edit(tmp_path, monkeypatch):
    result, prompts, edits_seen = _transient_run(tmp_path, monkeypatch, lambda frames: [])
    assert result["status"] == "not_requested" and result["reason"] == "all_objects_kept"
    assert prompts and edits_seen == []


def test_an_object_track_missing_a_view_refuses_but_an_absent_transient_is_empty(tmp_path):
    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    partial = track([a])
    with pytest.raises(ValueError, match="mask_view_coverage_incomplete"):
        edits.prepare_masked_frames(frames=[a, b], targets=[partial], margin=0.01, root=tmp_path / "bad")
    prepared = edits.prepare_masked_frames(
        frames=[a, b], targets=[{**partial, "task_effect": "transient"}], margin=0.01, root=tmp_path / "ok")
    assert prepared[0]["removed_transient_ids"] == ["obj"] and prepared[1]["remaining_pixel_count"] == 0
    assert "removed_transient_ids" not in prepared[1]


def _segment(tmp_path, monkeypatch, frames, regions, found):
    from blueprint_pipeline import meta_sam31
    sam = {"track_id": "meta-sam31-obj-0", "label": "obj", **track(found)["track"]}
    monkeypatch.setattr(meta_sam31, "run_meta_sam31", lambda **_: {"tracks": [sam]})
    target = {**plan()["objects"][0], "regions": regions}
    inputs = {"frames": [{"frame_id": f["frame_id"], "timestamp_seconds": i} for i, f in enumerate(frames)]}
    return edits.segment_objects(frames=frames, objects=[target], inputs=inputs, task_context={},
                                 source_digest="sha256:" + "a" * 64, root=tmp_path / "sam")


def test_a_sam_track_that_found_some_views_covers_the_rest_without_leaving_the_object(tmp_path, monkeypatch):
    """2026-10-24: the chosen dish track covered 3 of 10 views and missed one the planner marked."""
    a, b, c = view(tmp_path), view(tmp_path, "b", 0.1), view(tmp_path, "c", 0.2)
    box = {"box_xywh_normalized": [0.2, 0.2, 0.6, 0.6]}
    # SAM found the object in a; the planner also marked b; c shows nothing to remove.
    [selected] = _segment(tmp_path, monkeypatch, [a, b, c],
                          [{"frame_id": "a", **box}, {"frame_id": "b", **box}], found=[a])
    observations = {row["source_frame_id"]: row for row in selected["track"]["observations"]}
    assert set(observations) == {"a", "b", "c"} and selected["track"]["guide_only"] is True
    assert observations["a"] == track([a])["track"]["observations"][0]  # SAM's own mask is kept
    assert observations["b"]["runs"] and observations["c"]["runs"] == []
    prepared = edits.prepare_masked_frames(frames=[a, b, c], targets=[selected], margin=0.01, root=tmp_path / "masks")
    assert [row["remaining_pixel_count"] > 0 for row in prepared] == [True, True, False]


def test_a_sam_track_missing_only_unmarked_views_stays_a_segmentation(tmp_path, monkeypatch):
    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    [selected] = _segment(tmp_path, monkeypatch, [a, b],
                          [{"frame_id": "a", "box_xywh_normalized": [0.2, 0.2, 0.6, 0.6]}], found=[a])
    assert [row["runs"] == [] for row in selected["track"]["observations"]] == [False, True]
    assert "guide_only" not in selected["track"]


@pytest.mark.parametrize("scope, edited", [("required_and_additional", ["a"]), ("all_selected_views", ["a", "b"])])
def test_the_trial_scope_cleans_every_selected_view_and_the_default_only_the_reviewed_ones(
        tmp_path, monkeypatch, scope, edited):
    from blueprint_pipeline import atlas_scene_automation as worker

    a, b = view(tmp_path), view(tmp_path, "b", 0.1)
    inputs = {"frames": [{k: f[k] for k in ("frame_id", "image_path", "image_digest")} for f in (a, b)],
              "binding": {"source_video_digest": "source"}, "digest": "inputs"}
    observed = []
    monkeypatch.setattr(edits, "run_background", lambda **kw: observed.append(kw) or {"status": "quality_rejected"})
    args = dict(pose_root=None, inputs=inputs,
                review={"automation": plan(), "coverage_requirements": [{"required_frame_ids": ["a"]}]},
                output_root=tmp_path / "out", task_context={}, configuration={"policy": {}},
                task_runner=None, pose_runner=None, release_blockers=None)
    worker.run_automation(**args, edit_scope=scope)
    assert observed[0]["selected_frame_ids"] == edited
    assert [f["frame_id"] for f in observed[0]["frames"]] == edited
    with pytest.raises(ValueError, match="edit_scope_invalid"):
        worker.run_automation(**args, edit_scope="everything")


def test_scale_guidance_serves_any_site_and_keeps_endpoints_on_the_object():
    text = " ".join(AUTOMATION_INSTRUCTIONS.split())
    # Examples come from building elements any site has, not from one kind of room.
    for example in ("doorway", "stair riser", "ceiling tile", "work surface height", "standard pallet",
                    "rack or shelving upright", "parking bay", "built-in appliance"):
        assert example in text
    assert "range or oven" not in text and "appliance model" not in text
    # 2026-10-23: one endpoint sat on paper towels in front of the measured opening.
    assert "Put both endpoints on the named object itself" in text and "face-on" in text


def box_mask(frame, x0, x1, y0, y1):
    h, w = frame["depth"].shape
    return {"source_frame_id": frame["frame_id"], "width": w, "height": h,
            "runs": [{"start": int(y * w + x0), "length": int(x1 - x0)} for y in range(y0, y1)]}


def test_a_reference_counts_only_when_both_ends_lie_on_the_object_it_names(tmp_path):
    frame = view(tmp_path)  # the reference spans x = 20..60 at y = 30 on this 80 x 60 view

    def measured(name, instances, masks=None):
        rows = [{"reference_index": 0, "frame_id": "a", "instances": instances}] if masks is None else masks
        return sizes.estimate_sizes(frames=[frame], automation={**plan(), "scale_references": [reference("range width", 0.8)]},
                                    pose_digest="pose", output=tmp_path / f"{name}.json", reference_masks=rows)

    def reasons(result):
        return [gap["reason"] for gap in result["reference_gaps"]]

    assert measured("on", [box_mask(frame, 20, 61, 20, 41)])["meters_per_unit_estimate"] == pytest.approx(1.0)
    # 2026-10-23: one end sat on paper towels in front of the measured object.
    off = measured("off", [box_mask(frame, 30, 61, 20, 41)])
    assert off["status"] == "relative_sizes_only" and reasons(off) == ["atlas_reference_endpoint_off_object"]
    # One end on each of two instances spans the gap between objects, not one object.
    split = measured("split", [box_mask(frame, 18, 30, 20, 41), box_mask(frame, 50, 62, 20, 41)])
    assert reasons(split) == ["atlas_reference_endpoint_off_object"]
    assert reasons(measured("unseen", [box_mask(frame, 0, 0, 0, 0)])) == ["atlas_reference_object_unsegmented"]
    assert reasons(measured("unnamed", [], masks=[])) == ["atlas_reference_object_unnamed"]


def test_reference_objects_are_segmented_once_on_their_own_views(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_reference_masks, meta_sam31

    frames = [view(tmp_path, "a"), view(tmp_path, "b", position=1)]
    calls, events = [], []

    def run(**kwargs):
        calls.append(kwargs)
        events.append("dispatch")
        return {"tracks": [
            {"label": "scale_reference_0", "observations": [box_mask(frames[0], 20, 61, 20, 41),
                                                             box_mask(frames[1], 0, 5, 0, 5)]},
            {"label": "scale_reference_0", "observations": [box_mask(frames[0], 0, 5, 0, 5)]},
            {"label": "scale_reference_2", "observations": [box_mask(frames[1], 10, 20, 10, 20)]}]}
    monkeypatch.setattr(meta_sam31, "run_meta_sam31", run)
    references = [{**reference("range width", 0.8), "segmentation_prompt": " freestanding range "},
                  reference("countertop height", 0.82),
                  {**reference("doorway", 2.0), "frame_id": "b", "segmentation_prompt": "doorway"}]
    options = {"frames": frames, "root": tmp_path / "masks", "task_context": {"context_digest": "c"},
               "source_digest": "sha256:" + "a" * 64, "checkpoint": events.append}
    masks = atlas_reference_masks.segment_scale_references(references=references, **options)
    (call,) = calls
    # The withdrawal check runs before any view reaches the provider.
    assert events == ["reference_segmentation", "dispatch"]
    assert [row["source_frame_id"] for row in call["frame_registry"]] == ["a", "b"]
    assert [row["path"] for row in call["frame_artifacts"]] == [f["image_path"] for f in frames]
    assert [prompt["text"] for prompt in call["prompts"]] == ["freestanding range", "doorway"]
    assert call["task_context"] == {"context_digest": "c"} and call["source_video_digest"] == options["source_digest"]
    assert [row["reference_index"] for row in masks] == [0, 2]
    assert [o["source_frame_id"] for o in masks[0]["instances"]] == ["a", "a"]
    assert [o["source_frame_id"] for o in masks[1]["instances"]] == ["b"]
    # A retained plan whose references name no object is measured as before, with no call.
    assert atlas_reference_masks.segment_scale_references(references=[reference("x", 1.0)], **options) is None
    assert len(calls) == 1

    def withdrawn(stage):
        raise ValueError("website_capture_withdrawn")
    # A capture withdrawn while it was posed sends nothing more to the provider.
    with pytest.raises(ValueError, match="website_capture_withdrawn"):
        atlas_reference_masks.segment_scale_references(references=references, **{**options, "checkpoint": withdrawn})
    assert len(calls) == 1


def test_a_retained_plan_without_reference_names_keeps_its_measurement_binding():
    from blueprint_pipeline.atlas_scene_automation import measurement_plan

    assert "segmentation_prompt" not in measurement_plan({"automation": plan()})["scale_references"][0]
    named = {**plan(), "scale_references": [{**plan()["scale_references"][0], "segmentation_prompt": "range"}]}
    assert measurement_plan({"automation": named})["scale_references"][0]["segmentation_prompt"] == "range"
