"""The contracts the Pipeline owns and the WebApp mirrors byte for byte stay the same in both checkouts.

Pipeline #3653 added ``base_footprint_l_w_m`` to the robot spec catalog and the WebApp's mirror was not updated: the
WebApp's own check (``scripts/pipeline/verify-robot-spec-catalog.ts``) runs only in the WebApp, so nothing on this
side failed. This test fails when a sibling WebApp checkout holds a mirror that differs from the Pipeline's file, and
skips when no sibling checkout is found (CI, a lone checkout). Change both sides in the same change set.

The sibling is ``BLUEPRINT_WEBAPP_CHECKOUT`` when set, else the first ``Blueprint-WebApp`` directory beside this
checkout or beside any directory above it (a worktree sits inside its main checkout).

Mirrored, Pipeline file -> WebApp file:

- ``src/blueprint_pipeline/_catalog_data/robot_spec_catalog.v1.json`` -> ``contracts/pipeline/robot-spec-catalog.v1.json``
- ``config/website_job_spend_cap.v1.json`` -> ``contracts/pipeline/website-job-spend-cap.v1.json``
- ``docs/schemas/site_task_measurement_catalog.v1.json`` -> ``contracts/pipeline/site-task-measurement-catalog.v1.json``
- ``docs/schemas/website_task_target_choice.v1.schema.json`` ->
  ``contracts/pipeline/website-task-target-choice.v1.schema.json``
- every file of ``docs/webapp_handoff/decision-evidence-router.v1/`` -> ``contracts/pipeline/decision-evidence-router.v1/``
- ``tests/fixtures/website-withdrawal-vector.json`` -> ``server/tests/fixtures/website-withdrawal-vector.json``

Not mirrored from here: the WebApp's ``contracts/pipeline/decision-evidence-router.v1.json`` (its legacy public-reader
schema, with no Pipeline source) and ``contracts/shared-doctrine.lock.json`` (generated from ``doctrine/``, checked
by its own lock).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

#: A sibling WebApp checkout is not installed on the hermetic CI runners, whose wide lane refuses a skipped test: the
#: lane deselects these, and the WebApp's own CI compares the same mirrors with Pipeline main on every run
#: (`npm run pipeline:mirrors:verify`). Locally, beside a WebApp checkout, they run.
pytestmark = pytest.mark.external_data

REPO = Path(__file__).resolve().parents[1]
ROUTER = "docs/webapp_handoff/decision-evidence-router.v1"
MIRRORS = (
    ("src/blueprint_pipeline/_catalog_data/robot_spec_catalog.v1.json", "contracts/pipeline/robot-spec-catalog.v1.json"),
    ("config/website_job_spend_cap.v1.json", "contracts/pipeline/website-job-spend-cap.v1.json"),
    ("docs/schemas/site_task_measurement_catalog.v1.json", "contracts/pipeline/site-task-measurement-catalog.v1.json"),
    ("docs/schemas/website_task_target_choice.v1.schema.json",
     "contracts/pipeline/website-task-target-choice.v1.schema.json"),
    *((f"{ROUTER}/{path.name}", f"contracts/pipeline/decision-evidence-router.v1/{path.name}")
      for path in sorted((REPO / ROUTER).iterdir()) if path.is_file()),
    ("tests/fixtures/website-withdrawal-vector.json", "server/tests/fixtures/website-withdrawal-vector.json"),
)


def sibling_webapp() -> Path | None:
    """The WebApp checkout beside this one, or None."""
    configured = os.environ.get("BLUEPRINT_WEBAPP_CHECKOUT", "").strip()
    candidates = [Path(configured)] if configured else [parent / "Blueprint-WebApp" for parent in REPO.parents]
    return next((path for path in candidates if (path / "contracts" / "pipeline").is_dir()), None)


def test_every_mirrored_contract_exists_here():
    assert len(MIRRORS) >= 6
    for pipeline, _webapp in MIRRORS:
        assert (REPO / pipeline).is_file(), pipeline


@pytest.mark.parametrize("pipeline, webapp", MIRRORS, ids=[pipeline for pipeline, _webapp in MIRRORS])
def test_the_sibling_webapp_mirror_is_the_pipelines_file_byte_for_byte(pipeline, webapp):
    sibling = sibling_webapp()
    if sibling is None:
        pytest.skip("no sibling Blueprint-WebApp checkout")
    mirror = sibling / webapp
    assert mirror.is_file(), f"the WebApp checkout at {sibling} has no mirror {webapp} of {pipeline}"
    same = hashlib.sha256(mirror.read_bytes()).hexdigest() == hashlib.sha256((REPO / pipeline).read_bytes()).hexdigest()
    assert same, (f"{webapp} in the WebApp checkout at {sibling} differs from the Pipeline's {pipeline}: "
                  "update the mirror in the same change set")
