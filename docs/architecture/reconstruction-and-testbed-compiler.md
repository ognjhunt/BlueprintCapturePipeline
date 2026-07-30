# Reconstruction Capability and Testbed Compiler

Status: implemented local contract boundary, version 1 (2026-07-29)

## Decision

Reconstruction is a replaceable capability graph beneath a Task Evaluation Run.
It is not a product and it does not always produce a 3D Gaussian splat. The
planner derives the representations required by the requested claim types and
selects the lowest-total-cost set of authorized, applicable method profiles.
Provider availability is never treated as qualification.

`reconstruction_method_profile.v1` names the exact method, version,
implementation digest, provider identity, execution mode, outputs, capture and
claim-ceiling prerequisites, authorization, qualification status, expected
cost, rights/provider constraints, and failure modes. Supported method kinds
include pose/SfM, metric scaffolds, depth fusion, photogrammetry, 3DGS, semantic
graphs, segmentation, structural priors, collision proxies, articulated assets,
USD composition, generated visual completion, and owner-attested correction.

`reconstruction_plan.v1` binds the exact intake and capture digest. It records
required representations, selected methods, rejected candidates, missing
representations, total expected cost, and the next cheapest experiment. A
generated visual completion cannot satisfy metric, collision, physics, or
articulation outputs.

## Normalized results and layers

`reconstruction_result.v1` binds exact capture, method-profile,
implementation, runtime, and result digests. It records camera/coordinate-frame
solutions, assets, coverage, observed and generated regions, uncertainty,
invalid regions, held-out metrics, cost, provider receipt, rights/retention,
deletion evidence, and an explicit claim ceiling. Every generated region needs
a mask. A result that contains generated regions and physics-like outputs must
explicitly exclude those regions from physics use.

The compiler keeps four independent testbed layers:

- appearance: splats, images, textures, and visual meshes;
- metric/reference: calibrated frames, scale, depth, and structural planes;
- semantic: object identities, regions, relations, and uncertainty;
- physics: independently checked collision geometry, bodies, joints, and
  physics properties.

Appearance never becomes collision truth. Generated content never becomes
observed truth.

## SimReady and robot placement

`simready_asset_decision.v1` decides necessity per object and requested claim.
Physics-dependent claims require an asset; perception/visibility-only claims do
not. An asset is selectable only when it binds the source capture, is not
generated-only, and an identity independent from its provider verifies scale,
transform, support, orientation, penetration, reprojection, and physics
properties. A realistic-looking provider output cannot self-qualify.

`robot_placement_result.v1` binds the exact robot, embodiment, footprint,
sensors, controller, end effector, task object, target, approved task, capture,
method qualification, and evidence digests. It filters support, footprint,
access, collision, reset, human-clearance, and coverage failures before scoring
reach, manipulability, visibility, approach, cable/controller constraints,
stability, and calibration uncertainty. If no covered candidate remains it
abstains and requests targeted capture or measurement.

## Immutable testbed versions

`blueprint-compile-site-task-testbed` consumes an accepted, digest-verified QA
report; the approved task; the reconstruction plan/results; SimReady and
placement decisions; exact Card/evaluator/reset artifact references; and
supported condition ranges. It emits the existing
`maintained_site_task_testbed.v1`, including raw source identity, layered
reconstruction, evidence inventory, robot binding, task objects/targets, reset,
governance, validation envelope, unsupported conditions, invalidation triggers,
and provenance.

Artifacts are stored by testbed ID, version, and digest using create-once
semantics. An inter-process lock and immutable version binding prevent two
digests from occupying one logical testbed version. A correction or
reconstruction must name a new version and bind the predecessor digest. The
compiler rejects unaccepted QA, stale capture/result bindings, results from
methods absent from the exact plan, credential-bearing artifact URIs, and
same-version successors.

The signed `/api/live-pipeline/testbeds/compile` service loads the authoritative
approved task from Pipeline state rather than trusting a caller-supplied task.
After compilation it can publish the full digest-bound testbed to WebApp through
`PIPELINE_TESTBED_WEBAPP_URL`, authenticated with `PIPELINE_SYNC_TOKEN`. A 2xx
response is insufficient: Pipeline accepts only a receipt matching the exact
session, intake, task digest, testbed ID/version/digest, artifact reference, and
proof boundary. Set `PIPELINE_TESTBED_WEBAPP_SYNC_REQUIRED=true` in a deployment
that requires customer-visible state before reporting service success.

## Proof boundary

Hermetic compilation proves deterministic composition and contract integrity.
It does not prove reconstruction fidelity, collision correctness, task success,
deployment readiness, safety certification, or physical performance. The
comparative policy-ranking verdict remains `thesis_not_supported`.
