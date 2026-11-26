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
- ``docs/webapp_handoff/website-pipeline-codes.v1.json`` -> ``contracts/pipeline/website-pipeline-codes.v1.json``
  (every code the website is sent about a job's simulation: ``test_website_pipeline_codes``)

Mirrored the other way, as code: the WebApp's rules for a several-object job's ``task.additional_objects`` (its
``server/utils/taskEvaluationSceneIntake.ts`` and ``server/utils/websiteSceneSponsorship.ts``) and the Pipeline's copy
in ``task_evaluation_scene_intent_contracts`` (``ADDITIONAL_OBJECTS_*``), compared by their literals.

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
    ("docs/webapp_handoff/website-pipeline-codes.v1.json", "contracts/pipeline/website-pipeline-codes.v1.json"),
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


def test_the_sibling_webapps_further_object_rules_are_the_pipelines_copy():
    """The WebApp's ``additional_objects`` rules, read from its source, against the Pipeline's copy: at most
    ``ADDITIONAL_OBJECTS_MAX`` rows (taskEvaluationSceneIntake.ts ``ADDITIONAL_OBJECTS_MAX``), each exactly a rank of
    2 to that plus one and a subject, ranks rising (``additionalObjects``), on a pick-and-place task alone
    (``sponsoredSceneTask``); a subject removed before reconstruction or designed and never a test environment's
    (websiteSceneSponsorship.ts ``WEBSITE_SUBJECT_ORIGINS``, ``validateWebsiteSponsoredIntake``)."""
    import re

    from blueprint_pipeline import task_evaluation_scene_intent_contracts as contracts
    sibling = sibling_webapp()
    if sibling is None:
        pytest.skip("no sibling Blueprint-WebApp checkout")
    intake = (sibling / "server/utils/taskEvaluationSceneIntake.ts").read_text(encoding="utf-8")
    sponsorship = (sibling / "server/utils/websiteSceneSponsorship.ts").read_text(encoding="utf-8")
    where = (f"the WebApp checkout at {sibling} reads further objects otherwise than task_evaluation_scene_intent_contracts:"
             " change both in the same change set (or set BLUEPRINT_WEBAPP_CHECKOUT to a current checkout)")
    maximum = re.search(r"export const ADDITIONAL_OBJECTS_MAX = (\d+);", intake)
    assert maximum and int(maximum[1]) == contracts.ADDITIONAL_OBJECTS_MAX, where
    rows = re.search(r"export const additionalObjects = z\s*\.array\(z\.object\(\{ rank: z\.number\(\)\.int\(\)"
                     r"\.min\((\d+)\)\.max\(ADDITIONAL_OBJECTS_MAX \+ (\d+)\), subject: mapping \}\)\.strict\(\)\)"
                     r"\s*\.min\(1\)\s*\.max\(ADDITIONAL_OBJECTS_MAX\)\s*\.refine\(\(rows\) => rows\.every\(\(row, index\)"
                     r" => index === 0 \|\| row\.rank > rows\[index - 1\]\.rank\)", intake)
    assert rows and (int(rows[1]), contracts.ADDITIONAL_OBJECTS_MAX + int(rows[2])) == contracts.ADDITIONAL_OBJECT_RANKS, where
    # On the sponsored pick-and-place task alone: the public command and every other strategy refuse the key.
    assert re.findall(r"\badditional_objects\b: \w+", intake) == ["additional_objects: additionalObjects"], where
    assert "z.object({ ...pickAndPlaceFields, additional_objects: additionalObjects.optional() }).strict()" in intake, where
    origins = re.search(r"const WEBSITE_SUBJECT_ORIGINS = new Set\(\[([^\]]*)\]\);", sponsorship)
    assert origins and tuple(re.findall(r'"([a-z_]+)"', origins[1])) == contracts.ADDITIONAL_OBJECT_ORIGINS, where
    refusal = re.search(r"request\.task\?\.additional_objects \?\? \[\]\)\.some\([^)]*\) =>\s*"
                        r"!WEBSITE_SUBJECT_ORIGINS\.has\(row\?\.subject\?\.geometry_origin\) \|\| "
                        r'Object\.hasOwn\(row\?\.subject \?\? \{\}, "(\w+)"\)\)', sponsorship)
    assert refusal and (refusal[1],) == contracts.ADDITIONAL_OBJECT_FORBIDDEN_SUBJECT_FIELDS, where
