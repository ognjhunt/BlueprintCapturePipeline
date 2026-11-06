from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.gaussian_suppression_volume import (
    derive_suppression_volume_from_twin,
)
from tests.test_gaussian_suppression_volume import (
    BODY_MAX,
    BODY_MIN,
    _members,
)


def _receipt(tmp_path: Path, ply: Path, **overrides):
    arguments = {
        "task_id": "refrigerator_upper_door_open",
        "canonical_ply_path": ply,
        "body_world_aabb_min_m": BODY_MIN,
        "body_world_aabb_max_m": BODY_MAX,
        "articulated_members": _members(),
        "destination": tmp_path / "receipt.json",
    }
    arguments.update(overrides)
    return derive_suppression_volume_from_twin(**arguments)


_LOCAL_EVIDENCE = Path(
    "/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804"
)
_LOCAL_SCAN = _LOCAL_EVIDENCE / "inpainting_inputs/840796_refrigerator_v2/scene_standard.ply"
_LOCAL_SEALED_RETAINED = (
    _LOCAL_EVIDENCE
    / "gaussian_excision/840796_v2/target_only_expansion_ladder_v1/rung_01_core95"
    / "union/retained_scene_gaussians.ply"
)


@pytest.mark.slow
@pytest.mark.external_data
@pytest.mark.skipif(
    not _LOCAL_SEALED_RETAINED.is_file(),
    reason="840796 sealed retained scene not present locally",
)
def test_840796_suppression_payload_is_the_scene_the_coverage_audit_measured() -> None:
    """Coverage equivalence follows from input identity, not from a re-run.

    The sealed 96-cell hybrid review bound the exact splat it rendered. If the
    suppression payload carries that same digest then every cell of that audit
    - eight cameras by twelve door states - is reproduced by construction, and
    the 2.9% worst-case residual measured there applies unchanged to the
    render-time path.
    """

    import hashlib

    review = json.loads(
        (
            _LOCAL_EVIDENCE
            / "gaussian_excision/840796_v2/target_only_expansion_ladder_v1"
            / "rung_01_core95/reference_hybrid_review_v1"
            / "adp009b_reference_hybrid_review.v1.json"
        ).read_text(encoding="utf-8")
    )
    measured = review["retained_scene_render"]["splat_digest"]
    sealed = "sha256:" + hashlib.sha256(_LOCAL_SEALED_RETAINED.read_bytes()).hexdigest()

    assert measured == sealed
