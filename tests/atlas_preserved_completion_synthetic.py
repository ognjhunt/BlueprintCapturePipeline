"""An offline preserved-scene completion over a synthetic retained pose: no provider call.

The completion code runs for real, from the gap survey through generation,
joint registration, recovery and the completed splat, and then the refined
splat's gap composite. Only the paid and judging calls are stand-ins: the
planner picks the first gaps it is shown, ``atlasGenerate`` and
images2PosedRGBD render the synthetic room at the cameras they are asked for,
and the visual review passes. Used to measure the completion path's memory and
to check that its outputs do not depend on how the views are held.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np

from blueprint_pipeline import atlas_depth_splat as depth_splat
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from tests.atlas_posed_rgbd_synthetic import render_room, write_room_pose_root
from tests.test_atlas_depth_splat import write_exr

# A patch of back wall and a patch of side wall that no view ranged: the export's holes.
HOLES = ((np.array([-1.0, -0.6, -4.01]), np.array([1.0, 0.6, -3.99])),
         (np.array([2.99, -0.8, -1.5]), np.array([3.01, 0.4, 0.5])))
POLICY = {'rgbd_view_cost_usd': 0.05, 'pose_input_cost_usd': 0.01, 'rate_card_reference': 'synthetic',
          'reviewed_by': 'synthetic', 'reviewed_on': '2026-11-09'}


def _pixels_key(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def _write_png(path: Path, rgb) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path, format='PNG')


def _sealed(value):
    return {**value, 'digest': canonical_digest(value, digest_field='digest')}


class Rehearsal:
    """The inputs of one preserved completion, and the stand-ins for its paid calls."""

    def __init__(self, root: Path, count: int, *, width: int, height: int, views: int | None = None):
        from blueprint_pipeline.website_atlas_pose import ATLAS_POSE_ENDPOINT, INPUTS_SCHEMA

        self.root = Path(root)
        self.pose_root = write_room_pose_root(self.root / 'pose', count, width=width, height=height, views=views,
                                              holes=HOLES)
        result = json.loads((self.pose_root / depth_splat.POSE_RESULT).read_text())
        # The pending pose inputs the context field-of-view audit reads: the images that were posed.
        rows = []
        for row in result['operation_receipt']['input_usage']['frames']:
            image = next(output for output in row['outputs'] if output['role'] == 'image')
            rgb = self._decode(self.pose_root / image['path'])
            path = self.pose_root / 'inputs' / f"{row['frame_id']}.png"
            _write_png(path, rgb)
            rows.append({'frame_id': row['frame_id'], 'image_path': str(path), 'image_digest': _sha256_file(path)})
        inputs = _sealed({'schema_version': INPUTS_SCHEMA, 'status': 'pending', 'endpoint': ATLAS_POSE_ENDPOINT,
                          'heldout_pixels_included': False, 'claim_ceiling': 'development_only',
                          'metric_measurement_proven': False, 'frames': rows})
        (self.pose_root / 'atlas_pose_inputs.json').write_text(json.dumps(inputs))
        survey = _sealed({'schema_version': 'atlas_video_survey.v1',
                          'source_video_digest': result['binding']['source_video_digest'],
                          'heldout_pixels_included': False, 'display_rotation_degrees': 0,
                          'frames': [{'frame_id': row['frame_id'], 'timestamp_seconds': index,
                                      'source_image_path': row['image_path'],
                                      'source_image_digest': row['image_digest']}
                                     for index, row in enumerate(rows)]})
        self.survey_path = self.root / 'source_survey.json'
        self.survey_path.write_text(json.dumps(survey))
        # Every image a stand-in may be asked to pose, by its pixels: the camera it was taken from.
        self.cameras = {}

    @staticmethod
    def _decode(path: Path):
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))

    def planner(self, *, completion, output_root, **_):
        """The two context-supported cameras with the largest unflagged gaps, that gap each."""
        views = {row['camera_id']: row for row in completion['gap_survey']['views']}
        largest = []
        for order, candidate in enumerate(completion['candidate_camera_ids']):
            gaps = [gap for gap in views[candidate['camera_id']]['gaps'] if not gap.get('photographed_without_depth')]
            if gaps:
                gap = max(gaps, key=lambda row: row['pixel_count'])
                largest.append((-gap['pixel_count'], order, candidate, gap))
        chosen = [(candidate, gap) for _, _, candidate, gap in sorted(largest, key=lambda row: row[:2])[:2]]
        reasons = [{'camera_id': candidate['camera_id'], 'reason': 'Unranged wall patch', 'gap_ids': [gap['gap_id']],
                    'source_evidence': [{'gap_id': gap['gap_id'], 'visibility': 'observed_in_conditioning',
                                         'source_frame_ids': [candidate['source_frame_id']],
                                         'reason': 'The wall is photographed; its depth is missing.'}]}
                   for candidate, gap in chosen]
        return {'proposal': {'status': 'proposed', 'selected_camera_ids': [row['camera_id'] for row in reasons],
                             'gap_assessment': ['Wall patches'], 'remaining_unknowns': ['Generated wall'],
                             'scene_caption': 'The photographed room.', 'view_gap_reasons': reasons}}

    def task_runner(self, *, task, request, output_root, **_):
        """``atlasGenerate``: the room rendered at each target camera, with its depth."""
        assert task == 'atlasGenerate', task
        output_root.mkdir(parents=True, exist_ok=True)
        frames, outputs = [], []
        for index, cam in enumerate(request['targetCameras']):
            rgb, depth, _ = render_room(cam, 0.5)
            image, depth_path = output_root / f'{index}-image.png', output_root / f'{index}-depth.exr'
            _write_png(image, rgb)
            write_exr(depth_path, depth.astype(np.float32))
            self.cameras[_pixels_key(rgb)] = cam
            frames.append({'camera': cam})
            outputs += [{'role': f'frames-{index}-imageAsset', 'path': image.name, 'sha256': _sha256_file(image)},
                        {'role': f'frames-{index}-depth-depthAsset', 'path': depth_path.name,
                         'sha256': _sha256_file(depth_path)}]
        result = _sealed({'status': 'completed', 'operation': {'id': 'synthetic-generate',
                                                               'response': {'frames': frames}}, 'outputs': outputs})
        (output_root / 'result.json').write_text(json.dumps(result))
        return {'status': 'completed', 'operation_id': 'synthetic-generate'}

    def pose_runner(self, *, output_root, **_):
        """images2PosedRGBD: each posed image returned whole, with the room's depth at its camera."""
        import cv2

        inputs = json.loads((output_root / 'atlas_pose_inputs.json').read_text())
        cameras, usage = [], []
        for index, (row, source) in enumerate(zip(inputs['frames'], inputs['full_view_sources'], strict=True)):
            rgb = self._decode(Path(source['image_path']))
            cam = self.cameras[_pixels_key(rgb)]
            _, depth, _ = render_room(cam)
            folder = output_root / 'atlas_pose' / 'outputs' / f'{index:06d}'
            folder.mkdir(parents=True)
            write_exr(folder / 'depth.bin', depth.astype(np.float32))
            confidence = np.where(np.isfinite(depth) & (depth > 0), 255, 0).astype(np.uint8)
            for role, pixels in (('image', np.ascontiguousarray(rgb[..., ::-1])), ('confidence', confidence)):
                assert cv2.imwrite(str(folder / f'{role}.png'), pixels)
                (folder / f'{role}.png').rename(folder / f'{role}.bin')
            cameras.append({'camera': cam})
            usage.append({'frame_id': row['frame_id'], 'outputs': [
                {'role': role, 'path': str((folder / f'{role}.bin').relative_to(output_root)),
                 'sha256': _sha256_file(folder / f'{role}.bin')} for role in ('image', 'depth', 'confidence')]})
        result = _sealed({'binding': inputs['binding'], 'operation': {'id': 'synthetic-pose', 'response': {
            'frames': cameras}}, 'operation_receipt': {'input_usage': {'frames': usage}}})
        (output_root / depth_splat.POSE_RESULT).write_text(json.dumps(result))
        return {'status': 'completed', 'operation_id': 'synthetic-pose'}

    @contextlib.contextmanager
    def stand_ins(self, fused):
        """The visual review passes; reference photographs are remembered with their cameras; fusions are kept."""
        from blueprint_pipeline import atlas_scene_chisel, website_image_completion

        prepare, write = atlas_scene_chisel.prepare_source_photographs, depth_splat.write_depth_splat

        def remembered(**options):
            rows = prepare(**options)
            for row in rows:
                self.cameras[_pixels_key(self._decode(Path(row['image_path'])))] = row['camera']
            return rows

        def kept(**options):
            fused.append(list(options['frames']))
            return write(**options)

        saved = (website_image_completion.verify_completed_background, prepare, write)
        website_image_completion.verify_completed_background = lambda **_: {'status': 'passed'}
        atlas_scene_chisel.prepare_source_photographs = remembered
        depth_splat.write_depth_splat = kept
        try:
            yield
        finally:
            (website_image_completion.verify_completed_background, atlas_scene_chisel.prepare_source_photographs,
             depth_splat.write_depth_splat) = saved

    def run(self, frames, pose):
        """The preserved completion, then the refined splat's gap composite of the views it admitted.

        Returns the completion's result and each composited view's receipt with
        the digests of its pixels and depth.
        """
        from blueprint_pipeline.atlas_native_splat import composited_views
        from blueprint_pipeline.atlas_preserved_completion import run_preserved_completion

        splat_root = self.root / 'splat'
        if not (splat_root / 'atlas_depth_splat_manifest.json').is_file():
            depth_splat.write_depth_splat(frames=frames, result=pose, output_root=splat_root)
        fused = []
        with self.stand_ins(fused):
            result = run_preserved_completion(
                pose_root=self.pose_root, pose=pose, frames=frames,
                review={'subject_origin': 'no_manipulated_subject', 'uncaptured_regions': [],
                        'automation': {'objects': [], 'removal_instruction': ''}},
                root=self.root / 'preserved_completion', job_root=self.root,
                context={'context_digest': 'synthetic-task', 'capture_id': 'synthetic-capture'},
                configuration={'remaining_cost_usd': 50.0, 'policy': dict(POLICY)}, splat_root=splat_root,
                source_survey_path=self.survey_path, task_runner=self.task_runner,
                planner_runner=self.planner, pose_runner=self.pose_runner,
                release_blockers=lambda: ('synthetic', []), max_polls=1)
        composites = []
        if fused:
            assert result.get('generated_geometry_admitted') and len(fused) == 1, result.get('status')
            additions = fused[0][len(frames):]
            for _, view, _ in composited_views([*frames, *additions]):
                if 'gap_composite' in view:
                    composites.append((view['gap_composite'], _pixels_key(view['rgb']),
                                       hashlib.sha256(np.ascontiguousarray(view['depth']).tobytes()).hexdigest()))
            del fused[:], additions
        return result, composites


def output_tree(root: Path) -> dict:
    """Every file a rehearsal wrote, by path relative to its root: the bytes, as a digest."""
    return {str(path.relative_to(root)): _sha256_file(path) for path in sorted(Path(root).rglob('*'))
            if path.is_file() and not path.is_symlink()}
