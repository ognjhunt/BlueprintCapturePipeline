# Synthetic browser handoff fixture

`browser_handoff.json` is OFFLINE / NO MODEL-QUALITY OR PARTNER EVIDENCE. It was
generated from the real `BlueprintCapture` extractor's pure
`buildPipelineHandoffPayload` and companion routing, lineage, task, preview,
robot-handoff and status builders, not a second producer implementation.

Source: Capture commit `97ed402ef486bb9ad4a4a0ad2ec108c2329e2ec5`,
`cloud/extract-frames/src/index.ts`, SHA-256
`51a015ca081abf229ccf79c804de7853c647be3ecd5f61b8c98a5e70567515ea`.
Source was compiled outside the checkout; a synthetic Firebase bucket setting
allowed module registration. No function handler, network operation or provider
was invoked. All IDs, timestamps, source generations, hashes and task values are
synthetic. Browser manifests have no arbitrary lineage/upstream/candidate records;
this narrow resume retention rejects those records rather than granting access.

The exact-byte test reads the compact JSON produced by the real builder. The
existing birth/owner fixture substitutes its validated selectors before freezing
bytes for actual pending-preflight and same-worker resume tests. Known nested
fields, duplicate/unknown fields, signed URLs, nonfinite values, source changes,
terminal state and an intervening unknown-provider failure are independently
covered. The public preparation status remains bounded and excludes raw bytes.

ADP-010 / partner day-14 blocker: finite upload notification delivery can expire
before the positive assessment. The existing worker tick and job ledger suffice;
no new endpoint, queue, account or permission is introduced. Run the focused
provider-free contracts from the repo root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_website_assessment_resume.py tests/test_website_preparation_status.py tests/test_pubsub_handoff_source_versions.py tests/test_pubsub_handoff_listener.py -q
ruff check src/blueprint_pipeline/website_assessment_resume.py src/blueprint_pipeline/pubsub_handoff_listener.py src/blueprint_pipeline/pubsub_handoff_scene_operations.py tests/test_website_assessment_resume.py tests/test_website_preparation_status.py
```

165 tests and changed-file lint passed locally. Deployment and an actual
source-bound dispatch receipt remain separate coordinated release/acceptance
work; this fixture does not prove them.
