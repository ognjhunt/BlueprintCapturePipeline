# Deployment embodiment shortlist support

Owner scope: October 21, 2026. This supports beta job-to-evidence preparation,
ADP-011 / partner day 7 (freeze the actual partner stack), and ADP-050 / partner
day 28 (integrate the selected policy/runtime). The completion artifact for this
change is an internal operator shortlist that Pipeline can validate and resolve into exact
integration requirements. A completed simulator episode is a separate artifact.

The existing embodiment registry, job capability routing, and internal job
console are extended. Customer setup stays focused on the customer's actual
robot and policy. The existing DROID executable pair
and all paid execution gates are unchanged. No new dashboard or execution
service is introduced.

## Options

The canonical source is
[`deployment_embodiment_support.v1.json`](../../src/blueprint_pipeline/_catalog_data/deployment_embodiment_support.v1.json).
It is packaged in the Python distribution and mirrored byte-for-byte in
Blueprint-WebApp's `client/src/data/deploymentEmbodimentSupport.v1.json`.

The 16 exact options are:

- Universal Robots UR5e, one arm and two arms as separate configurations;
- two TRLC-DK1 arms;
- Almond Axol;
- FANUC CRX-10iA/L and M-710iC/45M;
- Yaskawa GP12 and GP20;
- two I2RT YAM arms;
- two AgileX PiPER arms;
- Standard Bots RO1 and Core;
- two Franka Research 3 arms;
- two Kinova Gen3 arms;
- two UFACTORY xArm 7 arms;
- KUKA LBR iiwa 14.

Existing single-arm CRX-10iA, Panda, Gen3 and xArm profiles remain distinct.
CRX-10iA/L cannot inherit a CRX-10iA asset. RO1 cannot inherit Core geometry.
FR3 cannot inherit the Panda/DROID runtime. GP20HL is not GP20.

Nine policy options cover six interface types: Dream-1 and Physical
Intelligence hosted APIs; embodiment-specific customer pi0.5, GR00T and ACT
checkpoints; customer controller; planning endpoint; recorded skill trace; and
Praxis-1 partner access. Praxis-1 is listed only for the publisher-demonstrated
RO1 pairing. GR00T and ACT UR5e entries require customer-supplied checkpoints,
without claiming those exact pairings ran at a customer site. These are different
interfaces, not interchangeable model checkpoints. A named API association
does not pin a version or fine-tune. Candidate and controller identities are
supplied during integration.

Each association retains whether it was publisher-demonstrated, an interface
candidate, or a customer-supplied route. Asset sources include the original
URL, known revision and path, license notes, and required import/composition
work. Documentation-only sources and unknown revisions remain explicit.

## Website to Pipeline

Staff use **Robot integration planning** in the existing `/admin/jobs` console.
The visual catalog, compatible policy routes, asset sources and setup handoff
are internal tools. Customer setup pages do not expose this catalog. The
`GET /api/workspace/deployment-embodiments` endpoint requires admin/ops authority.

Only staff can create or change a saved `support_option` planning description.
Existing pending descriptions are preserved during ordinary customer edits,
including after a catalog refresh. They cannot inherit an executable binding
or start an evaluation.

The shortlist download uses `deployment_embodiment_shortlist.v1`. It binds the
full catalog digest, optional source launch ID, exact embodiment IDs, ordered
policy route IDs and its own digest. It grants no execution or provider access.

Replay it with the installed Pipeline package:

```bash
python -m blueprint_pipeline.deployment_embodiment_support --catalog
python -m blueprint_pipeline.deployment_embodiment_support \
  --shortlist robot-policy-embodiment-shortlist.json \
  --output integration-plan.json
```

The plan resolves every choice into sources, requirements and typed blockers.
Changed catalogs, duplicate choices, substituted variants and incompatible
routes fail closed. Job capability routing includes the same sealed catalog
as `support_options`; the new rows do not enter its executable configurations.

Existing signed Pipeline HTTP admission exposes the same read-only operations:

- `GET /api/live-pipeline/deployment-embodiment-support`
- `POST /api/live-pipeline/deployment-embodiment-shortlist-preflight`

Both require the existing authenticated Website issuer. The POST accepts the
downloaded shortlist directly and performs no provider or queue mutation.

## Exact robot structural preflight

A selected single fixed arm can proceed through the existing team-embodiment
USD checks using its real manifest, staged model and scene plan:

```bash
python -m blueprint_pipeline.deployment_embodiment_support \
  --shortlist robot-policy-embodiment-shortlist.json \
  --embodiment-id universal_robots_ur5e \
  --native-manifest robot-embodiment.json \
  --native-model robot.usd \
  --scene-plan resolved-scene-plan.json \
  --output robot-preflight.json
```

This calls the existing manifest, model digest, USD joint/link, world-anchor,
reach and binding validators. It retains the exact scene/task digests and emits
a `robot_embodiment` handoff for `blueprint_policy_https.v1` /
`team_evaluation_scored_v1`. It does not generate endpoint credentials or an
admission/readiness receipt. OEM identity remains self-declared until source
provenance and licensing are checked: matching a SHA-256 and joint count does
not establish that a USD is the manufacturer's robot.

The native team path currently admits rigid pick/place tasks. Two-arm options
retain `independent_bimanual_tool_channels_required`: the current team manifest
has one normalized gripper channel, so two grippers must not silently share it.
The pinned Axol URDF also models fixed grippers, requiring an actuated gripper
asset. Dream additionally requires a controlled-observation API bridge,
confirmed tool calibration, reviewed account/data terms and model identity.
The Dream SDK's license does not establish hosted-service rights or recording
behavior. These gaps stay visible when an option is shortlisted.

Actual execution still enters the existing team candidate admission path with
the exact policy interface, source/rights records, controlled observations,
operator scope, spend cap/TTL, watchdog and teardown. Only retained simulator
execution can establish runtime support for a specific combination.

## Verification

Focused coverage protects catalog identities and source metadata, Website save
and readback behavior, canonical browser/Python handoff equivalence, stale and
incompatible selection rejection, unchanged runnable routing, signed no-write
preflight, packaged data, and native structural checks. Hermetic USD fixtures
test admission behavior; they are not manufacturer assets or execution proof.
