# Post-capture evidence-production spine

## Purpose

`blueprint-produce-post-capture-evidence` turns an admitted ARKit source into
the typed prerequisites consumed by `new_site_task_evaluation_run`. It verifies
native producer artifacts and exact file bytes; callers provide paths to typed
receipts, not site-specific reshaped JSON.

The command is provider-neutral. Canonical/Postshot/Splatfacto and Teleport
outputs enter through `native_3dgs_candidate.v1`; neither a provider receipt nor
an analyzer result can qualify itself. Appearance quality, metric registration,
collision geometry, robot placement, scene composition, routing, and execution
authorization remain independent gates.

## Real retained ARKitScenes command

From the repository root, with the retained ARKitScenes 40958756 bytes present:

```bash
blueprint-produce-post-capture-evidence \
  --run-id arkitscenes-40958756-real-post-capture \
  --source-artifact docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json \
  --source-root output/public_dataset_smokes/arkitscenes/40958756 \
  --output-root output/post_capture_evidence_runs
```

The current retained input terminates scientifically at:

```text
stage: reconstruction_registration
smallest missing measurement: native_3dgs_appearance_missing
```

That is an expected exit status of `2`, not a command failure. The retained
content-addressed result is under
`docs/evidence/arkitscenes_40958756_post_capture_a42a9edf/`. It binds the six
real source files (including `40958756.mov`) and explicitly remains
provider-derived public-dataset support, never Blueprint Raw Contract V3.2 or
fixture R7 evidence.

## Continuing a run

Add only native typed artifacts that have actually been produced:

```bash
blueprint-produce-post-capture-evidence \
  --run-id <site-task-run-id> \
  --source-artifact <arkit-proxy-or-raw-v3.2-validation.json> \
  --source-root <retained-source-root> \
  --appearance-candidate <native-3dgs-candidate.json> \
  --depth-surface-result <arkit-depth-surface-result.json> \
  --depth-surface-root <root-containing-the-bound-surface> \
  --geometry-qualification <independent-site-geometry-qualification.json> \
  --registration-qualification <independent-registration-qualification.json> \
  --target-orchestration <automatic-target-orchestration.json> \
  --placement-candidate <robot-placement-candidate.json> \
  --placement-qualification <independent-placement-qualification.json> \
  --routing-bundle <exact-task-site-routing-catalog-bundle.json> \
  --task-metric <frozen-task-metric.json> \
  --policy-candidate <candidate-1.json> \
  --policy-candidate <candidate-2.json> \
  --policy-candidate <candidate-3.json> \
  --policy-candidate <candidate-4.json> \
  --policy-candidate <candidate-5.json> \
  --policy-attempt <attempt-1.json> \
  --policy-attempt <attempt-2.json> \
  --policy-attempt <attempt-3.json> \
  --policy-attempt <attempt-4.json> \
  --policy-attempt <attempt-5.json> \
  --output-root output/post_capture_evidence_runs
```

For interaction tasks, also provide
`--simready-task-zone-qualification <qualification.json>`. Inspection tasks do
not require task-zone replacement. A support mount is unnecessary only when
the exact placement qualification says the source collider is qualified.

To execute the current registered-view analyzer instead of supplying a recorded
target orchestration, use `--target-pipeline-request`. Supplying both target
inputs is rejected.

## Artifact order

The content-addressed run directory contains the available prefix:

```text
01_source_profile.json
02_native_3dgs_candidate.json
03_derived_site_geometry.json
04_registered_site_reconstruction.json
05_target_orchestration.json
06_task_robot_selection.json
07_robot_placement.json
08_scene_composition.json
09_routing_inputs.json
09_routing_decision.json
10_policy_execution_authorization.json
terminal_new_site_task_evaluation_run.json
post_capture_evidence_run.json
```

Absent stages are not fabricated. The terminal run records the first missing
gate and its smallest measurement.

## Replay and invalidation

Each run directory is derived from an invocation digest over all upstream
artifact digests. Repeating the same invocation reuses byte-identical immutable
files. Changing any upstream digest selects a different run directory, so stale
downstream files cannot be silently reused.

## Claim boundaries

- ARKitScenes is public-dataset provider-derived support, not Raw V3.2.
- Raw V3.2 admission proves capture contract and calibrated trajectory only.
- ARKit depth/mesh is a hole-preserving geometry candidate until independently
  qualified for metric scale and collision.
- A native 3DGS is appearance, never dynamics geometry.
- Appearance evaluation cannot qualify metric registration.
- Target analyzers propose; the deterministic 3D binder selects or abstains.
- Robot selection is not placement qualification.
- Placement candidates cannot qualify their own reach or collision evidence.
- SimReady preflight cannot self-qualify a task-zone asset.
- Routing and policy authorization are deterministic exact joins; agents,
  providers, analyzers, and selected methods cannot authorize themselves.
- Policy execution authorization is simulation-only and never authorizes a
  physical robot, physical success, safety, or deployment readiness.

## Verification

```bash
.venv/bin/pytest -q tests/test_post_capture_evidence_spine.py \
  tests/test_new_site_task_evaluation_run.py
.venv/bin/pytest -q -m slow tests/test_post_capture_evidence_spine.py
```

The slow test reads the retained 40958756 source bytes and reproduces the
committed terminal artifacts. It skips only when those large retained bytes are
not installed in the test environment.
