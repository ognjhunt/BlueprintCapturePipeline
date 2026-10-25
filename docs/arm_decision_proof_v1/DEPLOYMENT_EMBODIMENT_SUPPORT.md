# Deployment embodiment shortlist support

Owner scope: October 21, 2026. This supports beta job-to-evidence preparation,
ADP-011 / partner day 7 (freeze the actual partner stack), and ADP-050 / partner
day 28 (integrate the selected policy/runtime). The completion artifact for this
change is an internal operator shortlist that Pipeline can validate and resolve into exact
integration requirements. The subsequent owner request, "Set this up for all
embodiments", extends that seam to exact configured robots and the existing
independent episode runtime. A completed simulator episode is a separate artifact.

The subsequent runnable implementation, exact Dream protocol, native arm servo,
portable dependency staging and remaining provider/asset inputs are described in
[configured runtime execution](DEPLOYMENT_RUNTIME_EXECUTION.md).

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

Only staff can create or change a saved `support_option` description.
Existing pending descriptions are preserved during ordinary customer edits,
including after a catalog refresh. A catalog selection alone cannot inherit an
executable binding. A complete matching manifest can use the existing setup,
checkpoint and evaluation request, retaining the same scene access, rights and
runtime admission. This general robot route does not extend the frozen ADP pair.

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
- `GET /api/live-pipeline/deployment-embodiment-asset-recipes`
- `POST /api/live-pipeline/deployment-embodiment-native-preflight`

These require the existing authenticated Website issuer. Shortlist preparation
accepts the downloaded shortlist. Native configuration inspection accepts only
`embodiment_id` and `robot_embodiment`, reads the actual bounded HTTPS model,
verifies its digest and USD joints/links/anchor, and returns
`configured_not_executed` or `requires_setup`. It accepts no caller-authored
scene and claims no task qualification. No provider or queue mutation occurs.

In the existing admin job console, save the complete robot configuration,
inspect it, register its exact policy endpoint/checkpoint and store its token
through the separate protected credential action. Choose that saved setup and
checkpoint when creating the normal evaluation request for an authorized job.
The normal free-evaluation admission binds the real scene, checkpoint, license,
qualification receipt and existing authority. Replacing a selected checkpoint
at admission is rejected. No configuration action starts a paid worker.

## Source preparation

Pinned sources and actual limitations are available through the same setup flow
and [source notes](EMBODIMENT_ASSET_RECIPE_SOURCES.md). Retain a source closure
without cloning whole repositories:

```bash
python -m blueprint_pipeline.deployment_embodiment_assets --list
python -m blueprint_pipeline.deployment_embodiment_assets \
  --embodiment i2rt_yam_bimanual --destination /path/to/new-yam-source-bundle
```

This retains the exact licensed source bytes, notices, Git blob identities and
SHA-256 inventory. Retries verify and reuse the same bundle. Missing models,
LFS payloads, tools, inertias or exact mounts remain explicit. The installed
Isaac importer can prepare source assets on its existing approved runtime;
unavailable importers do not trigger an installation or paid allocation.
Single-arm sources can be composed using explicit station parameters; there are
no default physical mount transforms or generated substitute tools.
A full mounted model, declared reset/controller/tool/cameras and exact manifest
are still needed for native execution. Imported source is not qualified evidence.

On the existing installed and admitted Isaac worker, source conversion is an
explicit command; it allocates no worker and never accepts a runtime license:

```bash
python -m blueprint_pipeline.deployment_embodiment_asset_import \
  --source-bundle /path/to/new-yam-source-bundle \
  --output /path/to/new-yam-import --fixed-base --initialize-isaac
```

The receipt retains importer identity/settings, output hashes and a relocatable
dependency closure. Additional content stays inside the prepared directory;
the original source bundle remains provenance and is not a runtime path dependency.
An unavailable importer or Xacro dependency returns a named reason. Omit
`--initialize-isaac` only when calling within an already initialized runtime.
The public one-file inspection route still requires a single self-contained
USD; closed multi-file preparations use operator-owned closure staging.
For an explicit two-arm station, the source composer and importer's optional
`--station-parameters` preserve source geometry while requiring both recorded
simulation mounts; see the [composition contract](EMBODIMENT_ASSET_RECIPE_SOURCES.md#explicit-bimanual-source-composition).

## Exact robot structural preflight

A selected fixed single-arm or bimanual configuration can proceed through the
existing team-embodiment USD checks using its real manifest, staged model and
scene plan:

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

The native team path admits rigid pick/place tasks. Single-arm v1 manifests,
digests and action tensors remain readable. Bimanual configurations use
`blueprint.robot_embodiment.v2`: ordered `grippers` replace the single `gripper`.
Each channel declares its unique name and tool link, finger joints with open
and closed positions, maximum opening, and contact links. The primary
`tool_link` names one declared tool. All arm/finger joints have distinct names
and position-driven joints have explicit reset positions. Each policy row
contains the declared arm joints followed by one closure per tool in channel
order. The observation includes closures and matching `gripper_names`; hidden
object state and scoring truth remain local.

The native runtime drives each arm and gripper independently, resets every
joint, renders each declared mounted/world camera, and retains separate tool
widths, joint positions, body origins and filtered task contact forces. Terminal
release or closed-state criteria apply to every tool; aggregate contact clear
requires both tools to let go. It never averages wrist positions or copies one
gripper width across tools. A required retreat remains undetermined when the
request lacks calibrated per-tool grasp geometry: raw body origins cannot
inherit the frozen Franka's physical-pad calibration. A structural preflight
is not an executed episode, admitted provider, or physical performance claim.

The pinned Axol URDF models fixed grippers, requiring an actuated gripper
asset. Dream additionally requires a controlled-observation API bridge,
confirmed tool calibration, reviewed account/data terms and model identity.
The Dream SDK's license does not establish hosted-service rights or recording
behavior. Its 20 Hz interface also cannot inherit the frozen task's 15 Hz
sampling cadence without a validated adapter. These gaps stay visible when an
option is shortlisted; Dream account access has not been established here.

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
