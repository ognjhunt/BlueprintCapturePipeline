"""The longest window a website job's preparation grant may carry, mirrored from the WebApp.

A website job's preparation grant and the scene intent bound to it share one
expiry (``validate_preparation_request`` requires the intent's
``execution.expires_at_epoch`` to equal the grant's). Its length is WebApp
policy: ``ttl_seconds`` in ``BLUEPRINT_WEBSITE_SCENE_SPONSORSHIP_JSON``, bounded
by ``WEBSITE_JOB_WINDOW_MAX_SECONDS`` in the WebApp's
``server/utils/websiteJobWindow.ts``. The Pipeline only checks the bound.

Rollout order: raise this value and deploy the Pipeline before the WebApp may
issue a longer window, so neither side refuses the other's grants. Lowering it
refuses grants already issued under the old bound.

Only time is bounded here. Caps, attempt counts, rights and consent are checked
where they always were. An owner-approved extension (door
``extend-scene-execution-window``) is separate: up to 7 days past the issued
expiry (``task_evaluation_scene_execution_window_evidence.MAXIMUM_EXTENSION_SECONDS``).
"""

# Proposed 2026-11-03 (up from one day), held for the owner's approval of the length.
WEBSITE_JOB_WINDOW_MAX_SECONDS = 7 * 86400
