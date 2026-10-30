# Configured robot and Dream execution

Owner scope: 2026-10-21, "Don't stop until implemented fully - they need to be
runnable." This extends the existing beta L5 execution seam and ADP-050 / partner
day-28 policy integration. It does not change the frozen two-candidate ADP
campaign. The existing native worker, scene qualification, task scorer, paid
allocator, watchdog, media retention and teardown remain authoritative.

The former shortlist resolved sources without executing end-effector actions.
This change supplies the missing Dream protocol client, measured arm servo,
explicit clock conversion, portable USD dependencies and mounted URDF composition.
Each is additive to the existing team robot contract. A complete execution
artifact is an actual episode receipt with observed motion, exact policy input
frames, independent scoring, review video and teardown. Source preparation,
mock transport tests and a CPU controller smoke are separate evidence.

## Execution path

1. Retain the exact licensed OEM source with
   `python -m blueprint_pipeline.deployment_embodiment_assets`.
2. Expand publisher Xacro using its pinned package roots. For a two-arm station,
   `deployment_embodiment_station` takes explicit mount transforms and actual
   source root names. It duplicates a single-arm source or remounts a publisher's
   existing pair. It preserves inertia, collision geometry, tools and source
   provenance; it does not invent a gripper or physical site mounting.
3. On the installed Isaac runtime, run `deployment_embodiment_asset_import`.
   `--station-parameters parameters.json` composes before import. The retained
   import includes the root USD and a hash-bound relative dependency closure.
4. Save the exact robot manifest. Version 2 provides independent `grippers` and
   optional `arms`: ordered joint names, arm base, tool, associated gripper and
   the measured/configured `tool_from_tcp` transform. Native validation checks
   actual model links, anchoring, joint limits and complete independent groups.
5. Register the actual policy endpoint/checkpoint and store its credential in
   the existing protected credential flow. For Dream, choose the exact account
   model ID and map `overhead`, `wrist_l`, `wrist_r` to the declared cameras.
6. The existing request/admission process binds the site/task, rights, checkpoint,
   model, scene qualification and founder funding. The ordinary canonical worker
   bundle retains the robot, dependencies and adapter. No setup action allocates
   resources or releases an existing dispatch hold.
7. The worker observes actual TCP poses, sends policy inputs, converts returned
   arm-base targets into measured Jacobian joint commands, and scores actual
   simulator state. Policy input PNGs decode the exact sent JPEG bytes; source
   renders and derived review normalization retain separate provenance.

## Public-source runtime configurations

`deployment_embodiment_runtime_configuration` registers 21 exact source profiles:
eleven include public source grippers (YAM, PiPER, dual FR3 plus Franka Hand,
single FR3 plus Robotiq2F85 simulation assembly, Kinova single/dual,
xArm7 single/dual, Axol, TRLC-DK1 and dual UR5e with Robotiq2F85); ten are bare arms
(UR5e, UR10e, KUKA iiwa14, three FANUC variants, Yaskawa GP12/GP20 and exact
ABB GoFa CRB15000-5/0.95 and CRB15000-12/1.27 variants). Bare arms do not acquire grasping
capability from their registration. The remaining catalog rows retain an
individual unavailable reason or the existing DROID runtime.

Prepare every available source and its joint/tool configuration without Isaac:

```bash
python -m blueprint_pipeline.deployment_embodiment_assets \
  --all-available --destination work/robot-sources
python -m blueprint_pipeline.deployment_embodiment_runtime_configuration \
  --all-available --source-bundle work/robot-sources --output work/robot-configurations
```

The generated `runtime-configuration.json` binds retained source bytes, serial
joint order, source limits, mimic relations and authored TCP frames. PD gains,
interior reset poses, camera rigs and duplicate-arm mount offsets are labeled
simulation assumptions. `source_dynamics` reports moving links without positive
authored mass/inertia; no estimates are silently added. GP12/GP20 currently lack
those dynamics and must retain that reason until a usable explicit estimate or
vendor model is integrated. Other source-authored inertias are not physical
validation. The two GoFa variants retain their original missing inertias and zero
effort placeholders as source evidence. Their generated simulation models add
explicit geometry-based inertias normalized to ABB's documented total robot mass,
and positive estimated effort limits/PD gains. Saved USD mass/inertia/effort
readback is required and retained; these estimates establish no physical dynamics
or policy equivalence. See the [GoFa source/modeling notes](EMBODIMENT_ASSET_RECIPE_SOURCES.md#exact-gofa-variants-and-explicit-simulation-dynamics).
Source configurations contain no invented USD hash or execution result.

The second phase runs on an installed Isaac worker obtained through the existing
paid allocator (or already authorized local Isaac runtime), before dispatching a
policy episode. For example:

```bash
python -m blueprint_pipeline.deployment_embodiment_runtime_configuration \
  --source-bundle work/robot-sources/i2rt_yam_bimanual \
  --output work/robot-configurations/i2rt_yam_bimanual \
  --import-prepared work/robot-usd/i2rt_yam_bimanual --initialize-isaac \
  --model-url https://YOUR_ASSET_HOST/i2rt_yam_bimanual/robot.usda \
  > work/yam-binding-inputs.json
```

This command allocates no cloud resource and uploads no asset. Publish the actual
prepared USD plus its dependency closure at the supplied operator-owned location,
then register the resulting manifest/model/provenance through the existing Team
registration and `prepared_robot_models` flow. The adapter
`beta_policy_screening_team.registration_from_prepared_inputs` creates the
reusable registry row from those verified inputs, the actual model path and
the operator's policy/rights settings. `bind_prepared_configuration`
checks the actual imported USD names, bodies, world anchor, digest and composition
before producing those inputs. The registration is task-independent; the beta
runner subsequently seals the actual task station and execution budget. Import
must precede that sealing because the worker requires the real USD digest.

Existing source/configuration/import outputs are verified and reused on retry;
changed bytes are never silently replaced. The FR3/Robotiq addition also accepts
the exact corrected `9bd34559c` generator for its original 18 profiles, after
reconstructing every derived configuration field with the current matching
profile and verifying the retained source, model, inventory and receipt digests.
Unknown generators, pre-correction TRLC/Axol generators and that old identity on
the new FR3/Robotiq profile remain invalid. A different policy cadence can be
bound with `--runtime-settings settings.json` containing supported `control_hz`
and `action_chunk_rows`. The final manifest seals those values. Actual policy
compatibility, successful Isaac import and scored task execution remain separate
observed outcomes.

## Dream contract

The direct client implements the published `dreamlabs-client==1.0.3` HTTP
contract without installing or instantiating its optional SDK recorder:

- inference: `https://api.dreamlabs.ai/v1/get_action`;
- model discovery: `https://platform.dreamlabs.ai/v1/models`;
- exact account model ID, without beta/stable aliases;
- overview and per-arm wrist JPEG histories at 5 Hz, up to nine actual frames;
- arm-base TCP xyz/rotation and independent opening commands at 20 Hz;
- fixed native arms at 60 Hz, holding each policy row for three native ticks;
- explicit XYZW/rot6 conversion and opening-to-closure inversion;
- bounded HTTPS, no redirects, credential-free evidence and no automatic POST
  retry after an ambiguous session outcome.

Provider metadata identity is **not a verified checkpoint-weights digest**.
The receipt records that limit. Disabling the separate SDK recorder does not
establish provider inference retention or training behavior. The operator rights
record must contain an exact, self-digested
`blueprint.dreamlabs_provider_disclosure.v1` receipt accepting those terms for
the selected model, candidate and scene before credentials or imagery are used.
No live approval is generated by this implementation.

The disclosure also records explicit team-account inference spending authority.
The sponsored GPU cap does not authorize a separate hosted-provider bill.
Blueprint-billed Dream access is refused until it has its own admitted paid
resource path; no advertised price is used as a cost or billing receipt.

## Operator-prepared portable assets

Only the operator registry can set `prepared_robot_models`; scene-derived
profiles and customer JSON cannot select a local directory. Each row binds
`team_id`, `checkpoint_id`, `model_url`, root `sha256`, an operator-owned
`directory` and `closure_digest`. Staging re-verifies the import receipt and
every dependency, then copies with exclusive creation, link refusal, byte bounds
and streaming hashes. Bundle assembly and the native runtime verify again.
The complete self-digested import provenance survives in the sealed model
binding, alongside the content closure. Descriptor-relative directory traversal
keeps source and destination copies on their selected filesystem trees.
The public single-file model preflight still rejects external dependencies.

## Availability and acceptance

The exact source catalog remains enumerated in
[source notes](EMBODIMENT_ASSET_RECIPE_SOURCES.md), with 21 source-to-runtime
configuration profiles described above. The same native manifest
runtime executes configured OEM USDs; a separate hardcoded adapter per vendor
is unnecessary. Exact tools, mounts, resets, cameras and licenses are execution
inputs, and cannot be inferred from a robot-family name.

Dream has a documented direct adapter. Customer controllers, planners and
embodiment-specific pi0.5/GR00T/ACT policies can supply the existing authenticated
Blueprint motor-action endpoint contract; their trained weights and
normalization must actually exist. The older skill-intent JSON is not a motor
trajectory. Physical Intelligence's API and Praxis require their actual private
robotics contracts and access; no public model metadata, generic Runway media API
or generic serializer substitutes for them. Standard Bots asset verification and
Axol's controller binding and estimated-physics limitations remain explicit in
the source matrix; the selected Axol standalone source includes actuated fingers.

Acceptance recorded during implementation includes focused native, client,
admission, asset, media and import-closure tests. A retained MuJoCo 3.11.0 CPU
controller smoke exercised the generated source models: 14 of 16 passed joint
motion, separate tool commands and repeatable reset; GP12/GP20 reported missing
moving-body mass/inertia. That smoke retains source inertias but disables gravity
and contacts, omits visuals, and omits Kinova COLLADA collision surfaces because
MuJoCo has no COLLADA decoder. It establishes controller isolation only, with no
collision, camera-rendering, Isaac, manipulation-task or policy-execution claim.
The later dual-UR5e/Robotiq2F85 CPU isolation also passed: both arms, both
independent tools and repeatable reset, with all 24 moving source joints and
source inertias retained; it omitted visuals and disabled gravity/contacts.
KUKA source metadata was separately compared against MuJoCo.
Actual hosted
Dream execution and Isaac acceptance require admitted inputs and their own
observed receipts. Deploying these adapters does not supply private access,
release a founder hold or mark any candidate qualified.
