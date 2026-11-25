"""What a completion's gap survey and camera aiming will cost its step, from counts, before any of it runs.

A completion surveys the room's gaps from candidate cameras, aims new cameras at them and surveys those, all
on the host's CPU inside one Atlas preview step, before its planner (a paid call) is asked which gaps to fill.
Austin (itest-cce2beee, 2026-11-25), a room kept as filmed: 204 posed views, 121 of them the selection's
reference views, 201 survey cameras and a draft splat of 16.2 M Gaussians. Measured offline with one BLAS
thread, as the unit runs, its first survey took 30.7 minutes and the aiming 47.4; the host runs such work 2.4
to 3 times slower, and ends a step 65 minutes after its unit started. The survey was written only when it
ended, so every retry began it again, and the job would have ended failed after its retries.

So the work is estimated from counts: cameras times Gaussians, views times points, voxels times views and
placements times context samples, each times a cost per unit measured here (``UNIT_SECONDS``), never the
clock, so the same job always gets the same answer. When it cannot fit, the completion takes the floor:
``WORK_EXCEEDS_STEP`` is a completion floor code (``atlas_automation_plan.COMPLETION_FLOOR_ERRORS``), nothing
is bought, and the room ships as filmed, flagged ``scene_completion_not_admitted``. It is checked before the
survey (``admit_survey``: everything the counts already name), then, as the aiming names them, before each
gap's placements are scored (``admit_placements``) and before each aimed survey (``admit_aimed_survey``):
all before the planner.

What fits (``fits``): the step must finish the whole pass before it is ended (``STEP_END_SECONDS`` after the
unit's start), so its surveys are written; and what a later step does again before it may reach the planner
(reading the views, the field-of-view audit, the camera pool and the aiming, which is not kept) must fit
before no paid stage may start (``STEP_START_BUDGET_SECONDS``), since a pass that ends after that leaves its
planner to the next step. Both are host seconds, divided by ``HOST_SLOWDOWN``.

Derivation (``UNIT_SECONDS``, reference seconds: the container round 8b measured Austin in, one BLAS thread):
each phase's time divided by the count that drives it. Austin there gave the step's work before the survey (6.7
minutes for 204 views), the support check (121 views: 3.87 minutes on this code), the export projection (1.70 s a
camera for 16.2 M Gaussians), the observed space (1.7 minutes, 204 views of 4.2 M voxels) and the aiming's reads
(0.8 minutes). A captured object's room is calibrated on the host itself, divided by ``HOST_SLOWDOWN``: the lamp
job (itest-314bd4bc, 24 Nov; 30 views, 281 survey cameras, 131 of them away from every view's centre, no export)
made its context views in 688 host seconds and surveyed from the retained views in 1,570, split between
projection, reading and each camera's own pixels as a run of the same work here splits them. One placement score
over Austin's 2.9 M context samples took 0.44 s in this container after a restart that made it 1.9 times faster
than the reference (the export projection's 4.51 s a camera became 2.41). ``HOST_SLOWDOWN`` is the slow end of
what the host measured: its fusion of Austin took 38 minutes against 15.7 in the reference (2.4 times). The
estimate for the lamp's room comes to about 9% above what the host took from its field-of-view audit to its
planner's inputs (49 minutes), and about 87% of a step with the reading before it; Austin's context phase to 8%
above what it took. The record (``RECORD``) keeps the counts, costs and
estimate.
"""
from __future__ import annotations

import math
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest

#: The completion floor code a pass whose work cannot fit takes.
WORK_EXCEEDS_STEP = 'atlas_completion_work_exceeds_step'
RECORD = 'completion-work-bound.json'
SCHEMA_VERSION = 'atlas_completion_work_bound.v1'
#: How much slower the host runs this work than the container the costs were measured in.
HOST_SLOWDOWN = 3.0
#: Reference seconds (this container, one BLAS thread) per unit of each phase's count.
UNIT_SECONDS = {
    # Reading the posed views, the field-of-view audit and the camera pool: per view pixel (Austin).
    'prelude_per_view_pixel': 2.14e-6,
    # The cross-view support check: per source pixel checked against one other view (Austin).
    'support_per_point_pair': 1.73e-8,
    # A context view away from every photograph's centre: per photograph pixel projected into it (lamp).
    'context_per_point_camera': 4.0e-8,
    # Reading and lifting the views once per batch of cameras that needs them: per view pixel (lamp).
    'lift_per_point': 8.1e-8,
    # A survey camera's depth from the export: per Gaussian projected (Austin).
    'splat_per_gaussian_camera': 1.05e-7,
    # A survey camera's depth from the retained views (no export): per view pixel projected (lamp).
    'depth_per_point_camera': 5.2e-8,
    # The observed-space grid: per voxel per view (Austin).
    'space_per_voxel_view': 1.2e-7,
    # A survey camera's own pixels: its context images written, gap labels, rays walked, survey images (lamp).
    'camera_per_pixel': 4.6e-7,
    # The aiming's reads of every view (planning cloud, sight lines, context samples): per view pixel (Austin).
    'aiming_per_view_pixel': 2.6e-7,
    # Scoring one candidate placement of an aimed camera: per context sample.
    'placement_per_sample': 2.85e-7,
}
#: The phases a later step does again before it may reach the planner: the rest is kept on disk.
REPEATED = ('prelude', 'aiming_reads', 'aiming_support', 'placement_scoring')


def step_seconds() -> dict[str, float]:
    """Reference seconds a step has: to finish the pass, and to reach its planner, at the host's pace."""
    from .website_atlas_preview import STEP_END_SECONDS, STEP_START_BUDGET_SECONDS

    return {'whole_pass': STEP_END_SECONDS / HOST_SLOWDOWN,
            'repeated_before_planner': STEP_START_BUDGET_SECONDS / HOST_SLOWDOWN}


def _pixels(frame) -> int:
    k = frame['intrinsics']
    return int(k['width']) * int(k['height'])


def _gaussians(splat_root) -> int:
    import json

    manifest = json.loads((Path(splat_root) / 'atlas_depth_splat_manifest.json').read_text())
    count = manifest.get('gaussian_count')
    if type(count) is int and count >= 0:
        return count
    from .atlas_scene_gaps import load_exported_splat
    return int(load_exported_splat(splat_root)[1].count)


class CompletionWork:
    """One completion pass's work, estimated from counts and admitted as its parts are named."""

    def __init__(self, *, root, frames, photographs, geometry_frames, candidates, splat_root=None,
                 expected_frames=()):
        self.root = Path(root)
        self.frames, self.photographs = list(frames), list(photographs)
        self.geometry_frames, self.expected_frames = list(geometry_frames), list(expected_frames)
        self.gaussians = None if splat_root is None else _gaussians(splat_root)
        self.counts = {'views': len(self.frames), 'photographs': len(self.photographs),
                       'geometry_views': len(self.geometry_frames), 'survey_cameras': len(candidates),
                       'gaussians': self.gaussians, 'scored_placements': 0, 'aimed_survey_cameras': 0}
        self.phases = {'prelude': UNIT_SECONDS['prelude_per_view_pixel'] * sum(map(_pixels, self.frames))}
        self._add(self._survey(candidates, first=True))
        pixels = sum(map(_pixels, self.frames))
        self.phases['aiming_reads'] = UNIT_SECONDS['aiming_per_view_pixel'] * pixels
        # Their support masks, unless the survey's photographs are these views (then they are read again).
        if not (len(self.frames) == len(self.photographs)
                and all(a is b for a, b in zip(self.frames, self.photographs))):
            self.phases['aiming_support'] = UNIT_SECONDS['support_per_point_pair'] * pixels * (len(self.frames) - 1)

    def _add(self, phases):
        for name, value in phases.items():
            self.phases[name] = self.phases.get(name, 0.) + value

    def _survey(self, cameras, *, first):
        """One ``survey_gaps`` over ``cameras``; the support masks and the observed space on the first only, since
        a pass's later surveys read them again from their caches."""
        from .atlas_scene_chisel import DEPTH_PROJECTION_BATCH_CAMERAS
        from .atlas_scene_preservation import CONTEXT_BATCH_CAMERAS, _same_center

        cost, phases, prefix = UNIT_SECONDS, {}, 'survey_' if first else 'aimed_survey_'
        photograph_pixels = sum(map(_pixels, self.photographs))
        moving = [not any(_same_center(frame, row['camera']) for frame in self.photographs) for row in cameras]
        if first and any(moving):
            phases['support'] = cost['support_per_point_pair'] * photograph_pixels * (len(self.photographs) - 1)
        batches = sum(any(moving[start:start + CONTEXT_BATCH_CAMERAS])
                      for start in range(0, len(moving), CONTEXT_BATCH_CAMERAS))
        phases['context'] = (cost['context_per_point_camera'] * sum(moving) * photograph_pixels
                             + cost['lift_per_point'] * batches * photograph_pixels)
        if self.gaussians is not None:
            phases['depth'] = cost['splat_per_gaussian_camera'] * len(cameras) * self.gaussians
        else:
            # Every geometry view not at the camera's centre is lifted into it; one at its centre is resampled.
            projected = sum(_pixels(frame) for row in cameras for frame in self.geometry_frames
                            if not _same_center(frame, row['camera']))
            phases['depth'] = (cost['depth_per_point_camera'] * projected + cost['lift_per_point']
                               * math.ceil(len(cameras) / DEPTH_PROJECTION_BATCH_CAMERAS)
                               * sum(map(_pixels, self.geometry_frames)))
        if self.expected_frames:
            phases['depth'] += cost['depth_per_point_camera'] * len(cameras) * max(map(_pixels, self.expected_frames))
        if first:
            # One grid over the observed bounds, every view projected over all of it (``observed_space``).
            from .atlas_observed_space import OBSERVED_SPACE_VOXELS

            views = {id(frame): frame for frame in [*self.geometry_frames, *self.photographs]}
            voxels = min(OBSERVED_SPACE_VOXELS, sum(map(_pixels, views.values())))
            phases['observed_space'] = cost['space_per_voxel_view'] * voxels * len(views)
        phases['cameras'] = cost['camera_per_pixel'] * sum(_pixels(row['camera']) for row in cameras)
        return {prefix + name: value for name, value in phases.items()}

    def total(self) -> float:
        return sum(self.phases.values())

    def repeated(self) -> float:
        return sum(self.phases.get(name, 0.) for name in REPEATED)

    def fits(self) -> bool:
        available = step_seconds()
        return self.total() <= available['whole_pass'] and self.repeated() <= available['repeated_before_planner']

    def _admit(self, stage):
        if self.fits():
            return
        record = {'schema_version': SCHEMA_VERSION, 'status': 'exceeds_step', 'stage': stage, 'counts': self.counts,
                  'reference_seconds_by_phase': {name: round(value, 1) for name, value in self.phases.items()},
                  'reference_seconds': round(self.total(), 1), 'repeated_reference_seconds': round(self.repeated(), 1),
                  'available_reference_seconds': {name: round(value, 1) for name, value in step_seconds().items()},
                  'host_slowdown': HOST_SLOWDOWN, 'unit_seconds': UNIT_SECONDS,
                  'basis': 'counts_times_measured_unit_costs_not_the_clock', 'claim_ceiling': 'development_only'}
        record['digest'] = canonical_digest(record, digest_field='digest')
        write_json(self.root / RECORD, record)
        raise ValueError(WORK_EXCEEDS_STEP)

    def admit_survey(self):
        """Before the survey: the step's reading of the views, the survey and the aiming's reads must fit."""
        self._admit('before_gap_survey')

    def admit_placements(self, *, placements, samples):
        """Before one gap's ``placements`` locations are scored over ``samples`` context samples."""
        self.counts['scored_placements'] += placements
        self._add({'placement_scoring': UNIT_SECONDS['placement_per_sample'] * placements * samples})
        self._admit('before_placement_scoring')

    def admit_aimed_survey(self, cameras):
        """Before the aimed cameras (or their source-centred fallbacks) are surveyed."""
        self.counts['aimed_survey_cameras'] += len(cameras)
        self._add(self._survey(cameras, first=False))
        self._admit('before_aimed_survey')
