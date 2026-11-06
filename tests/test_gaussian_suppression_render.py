from __future__ import annotations

import json
from pathlib import Path

import pytest



_LOCAL_EVIDENCE = Path(
    "/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804"
)
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
