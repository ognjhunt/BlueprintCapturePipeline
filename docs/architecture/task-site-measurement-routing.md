# Task-and-site-specific measurement routing

Status: implemented contract, taxonomy, admission, catalog, and agent slice
(2026-08-01, completed 2026-08-02)

## Outcome

The Decision/Evidence Router now has an additive measurement-routing kernel for
questions whose validity depends on a specific task, captured site, robot,
material, sensor, controller, or claim ceiling. It answers:

> Which measurement substrate is independently qualified for this exact scope,
> and what evidence bounds the answer?

It does not answer “which simulator is best,” select a universal backend, or
turn public research into production qualification.

The implementation is in
`blueprint_pipeline.task_site_measurement_routing`. Its six checked contracts
are:

1. `task_measurement_requirements.v1`
2. `site_evidence_profile.v1`
3. `method_capability_profile.v1`
4. `measurement_qualification_record.v1`
5. `task_site_measurement_routing_decision.v1`
6. `abstention_or_next_action.v1`

The JSON Schema is
`docs/schemas/task_site_measurement_routing.v1.schema.json`, which also checks
the `site_evidence_audit.v1` capture-evidence audit. These are sidecars inside
the existing maintained testbed, request, method, qualification, and Evidence
Plan contracts. Existing v1 readers remain valid.

Sibling modules complete the repo-local governance and execution-preparation
loop:

- `blueprint_pipeline.measurement_research_admission` — the human-controlled
  R0-R8 admission state machine
  (`docs/schemas/measurement_research_admission.v1.schema.json`);
- `blueprint_pipeline.measurement_method_research_catalog` — the research
  landscape as validated, non-production candidate dossiers plus the
  Q-KIN..Q-HRI protocol taxonomy, the three qualification-benchmark
  blueprints, the top-10 priority investigations, and the twenty standing
  abstentions;
- `blueprint_pipeline.measurement_engine_capability_profiles` — R1
  source-verified capability profiles for MuJoCo, Drake, Isaac Sim (PhysX and
  the RTX sensor path separately), Newton, SAPIEN, and Chrono. Every `True`
  field is traceable to a live-fetch or report-VF source-manifest entry,
  unverified fields fail closed, and the profiles carry zero qualifications,
  so the router abstains on all of them: a verified feature list is a
  capability declaration, never task-scoped validity;
- `blueprint_pipeline.measurement_adapter_runtime` — validated, version-pinned
  adapter descriptors, side-effect-free environment probes, tri-state
  capability drafts, and R0/R1 admission packets for the priority engine,
  sensor, deformable, tactile, observation, and provider candidates. A probe
  observes installation state only; it never imports or launches the engine,
  reads credentials, authorizes execution, or creates a qualification;
  the Chrono descriptor deliberately probes no PyPI distribution because the
  official PyChrono route is the projectchrono conda channel or a source build.
  `bootstrap_measurement_chrono_development.py` instead creates or inspects an
  isolated conda runtime, binds the exact `conda-meta` version/build/channel,
  preloads the environment-owned OpenMP runtime when present, and verifies a
  real `pychrono.core` system construction. The environment receipt cannot
  establish a benchmark; the separately bound NSC worker and suite do that;
- `blueprint_pipeline.measurement_adapter_execution` — the uniform argv-only
  local development executor and receipt contract. It binds requests, public
  cases, implementation bytes, exact runtime settings, worker results, and log
  digests; rejects qualification-split execution; and never grants production,
  provider-spend, physical, R6, or R7 authority;
- `blueprint_pipeline.measurement_site_evidence_bridge` — the deterministic
  join from an observed metric surface, no-fill collider candidate, independent
  bounded-navigation collider report, and signed measured robot-base-to-site-
  frame SE(3) record into `site_evidence_profile.v1`. It verifies source,
  asset, task-region, evaluator, signature, threshold, and digest lineage while
  denying qualification, R5-R7, physical-success, deployment, and safety
  authority;
- `blueprint_pipeline.measurement_pinocchio_coal_kinematic_adapter` and
  `blueprint_pipeline.measurement_geometry_kinematic_development_suite` — an
  exact-pinned Pinocchio 4.1.0/Coal 3.0.3 Q-KIN development port and checked
  three-case corpus for planar two-link reachability plus finite-sample GJK
  signed-distance queries. They explicitly exclude captured meshes,
  registration, continuous collision, physical evidence, and R5-R7 authority;
- `blueprint_pipeline.measurement_mujoco_adapter` — the first real narrow
  worker, pinned to MuJoCo 3.11.0. It runs a rigid drop/contact protocol twice
  and requires exact replay agreement before emitting development predictions;
- `blueprint_pipeline.measurement_geometry_contact_development_suite` — the
  checked two-case sphere/box rigid-drop runner. It binds public case bytes,
  implementation, solver, exact runtime, receipts, first-contact timing, and
  penetration aggregates while schemas force synthetic, non-instrumented,
  development-only, non-R5/R6/R7 authority;
- `blueprint_pipeline.measurement_newton_rigid_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_cross_engine_development_suite`
  — an exact-pinned Newton 1.4.0/Warp 1.15.0 CPU XPBD port and paired
  MuJoCo/Newton runner over the same method-neutral sphere/box corpus. It
  preserves per-engine receipt identities plus contact-time and transient-
  penetration deltas instead of treating engine agreement as qualification;
- `blueprint_pipeline.measurement_drake_rigid_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_drake_development_suite`
  — an exact Drake 1.55.0 MultibodyPlant CPU/SAP/point-contact development
  worker and checked runner over the same method-neutral sphere/box corpus.
  They bind an explicit isolated Python 3.13/3.14 interpreter, worker and
  wrapper source, solver, cases, logs, receipts, and exact double replay while
  denying renderer, physical, qualification, and R5-R7 authority;
- `blueprint_pipeline.measurement_isaac_physx_rigid_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_isaac_physx_development_suite`
  — an exact Isaac Sim 6.0.1 CPU PhysX/TGS/SAP development port over the same
  method-neutral sphere/box corpus. The worker starts `SimulationApp` before
  Isaac imports, creates a fresh metric Z-up USD stage, observes the live body
  pose and PhysX contact report, repeats the complete trace, and binds the
  runtime and solver without invoking RTX sensors or a renderer. Plan-time
  validation is complete; actual Isaac execution remains false until an exact
  external runtime returns both checked receipts;
- `blueprint_pipeline.measurement_sapien_rigid_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_sapien_development_suite`
  — an exact-pinned SAPIEN 3.0.3/PhysX CPU development worker and checked
  corpus runner over the same method-neutral sphere/box cases. They use no
  renderer or ManiSkill task layer, enable TGS and enhanced determinism with
  zero CPU workers, require exact double replay, and emit a schema-checked
  non-authorizing aggregate;
- `blueprint_pipeline.measurement_mujoco_articulation_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_articulation_development_suite`
  — a separately identified Q-ART port and two-case door-hinge/drawer-slide
  runner. They bind joint type, mass, damping, effort, limits, target travel,
  solver, exact runtime, and replay while explicitly excluding captured joints,
  instrumented force, policy interaction, and R5/R6/R7 authority;
- `blueprint_pipeline.measurement_mujoco_insertion_adapter` and
  `blueprint_pipeline.measurement_geometry_contact_insertion_development_suite`
  — a separately identified centered-clearance/interference square-peg port.
  It records signed clearance, side-contact sequence, insertion outcome, and
  penetration under exact replay while excluding captured geometry,
  instrumented force, robot control, and qualification authority;
- `blueprint_pipeline.measurement_opencv_observation_adapter` — a second real
  development worker for `direct-captured-observations`. It runs OpenCV PnP and
  calibrated reprojection twice over public, digest-bound cases; measures RGB
  reprojection, depth availability, depth residual, and timestamp alignment;
  and refuses version drift, noncanonical calibration, degenerate targets, or
  replay disagreement. The checked synthetic corpus exercises the port but is
  explicitly neither held-out nor physical evidence;
- `blueprint_pipeline.measurement_observation_development_suite` — the corpus
  runner that plans or executes every public observation case through the
  uniform boundary and emits a digest-bound aggregate suite. The suite schema
  hard-codes development/synthetic truth and makes held-out, physical, R5, R6,
  R7, production, and agent-promotion claims impossible;
- `blueprint_pipeline.measurement_pyelastica_cable_adapter` — a real CPU
  PyElastica 0.3.3.post2 Q-DLO development worker. It executes a bounded
  fixed-free Cosserat rod under a known load, samples the full tip trajectory,
  measures displacement, segment strain, and applied force, and requires exact
  double-run replay before emitting a cable-lane prediction;
- `blueprint_pipeline.measurement_deformation_cable_development_suite` — the
  two-regime PyElastica corpus runner. It binds the solver, material, geometry,
  timestep, load, corpus, cases, and receipts while forcing synthetic,
  development-only, nonphysical, non-R5/R6/R7 status;
- `blueprint_pipeline.measurement_mujoco_flex_cloth_adapter` — a separately
  identified MuJoCo 3.11 Q-CLOTH worker for the `flexcomp` 2D stretch
  formulation. It binds grid topology, pinning, material, collision, solver,
  timestep, warnings, sag, strain, penetration, and contact traces and refuses
  to inherit identity or validity from the rigid-drop worker;
- `blueprint_pipeline.measurement_deformation_cloth_development_suite` — the
  two-regime flex-cloth runner covering free sag and ground contact. Its solver
  scope is fixed to `mujoco-flex-elastic2d-stretch`; self-contact, bending,
  garments, topology change, and physical material validity remain excluded;
- `blueprint_pipeline.measurement_mujoco_granular_adapter` — a separately
  identified MuJoCo 3.11 Q-GRAN development reference for identical rigid
  spheres. It runs a staggered column collapse twice, measuring spread,
  settling, contact topology, normal force, and penetration while forcing
  cohesion, nonspherical grains, and physical characterization out of scope;
- `blueprint_pipeline.measurement_deformation_granular_development_suite` —
  the two-regime spherical-particle corpus runner. It keeps the solver scope at
  `mujoco-rigid-monodisperse-sphere-contact` and cannot claim Chrono DEM,
  pouring/tool interaction, material calibration, scale transfer, or granular
  qualification;
- `blueprint_pipeline.measurement_chrono_granular_adapter` and
  `blueprint_pipeline.measurement_deformation_granular_chrono_development_suite`
  — an exact PyChrono 10.0.0 core NSC CPU/Bullet/PSOR worker and distinct
  two-regime 27-sphere corpus. They bind the official conda build/channel,
  environment-owned OpenMP preload, material parameters, solver, collision
  system, timestep, force, penetration, settling, contact scope, and double
  replay. The specialized Chrono::Granular GPU module, material calibration,
  nonspherical/cohesive particles, pouring/tool interaction, and R5-R7 remain
  explicitly false;
- `blueprint_pipeline.measurement_direct_tactile_adapter` — the direct-sensor
  Q-TACT development port. It deterministically reduces synchronized optical
  marker displacement, contact intensity, and normal/shear-force sequences,
  but the checked corpus is synthetic and cannot represent real calibration or
  force truth;
- `blueprint_pipeline.measurement_tactile_development_suite` — the stable-
  contact/incipient-slip two-case runner. It binds the tactile lane and forces
  physical measurements, real-sensor calibration, independent execution, R5,
  R6, R7, production, and physical-success authority false;
- `blueprint_pipeline.measurement_world_model_action_fidelity_adapter` and
  `blueprint_pipeline.measurement_world_model_action_fidelity_suite` — a
  model-neutral Q-WM development port over the existing strict numeric action-
  recovery and cross-step replay contracts. It executes synthetic within/out-
  of-envelope cases but never generates model output, scores policy ranking,
  claims physics, or changes `thesis_not_supported`;
- `blueprint_pipeline.measurement_qualification_benchmarks` — executable
  benchmark-contract machinery for preregistering the three qualification
  programs, sealing development versus qualification splits, binding adapter
  predictions to execution receipts, ingesting independently measured hidden
  labels, computing deterministic metrics, and producing R4/R5 evidence
  candidates. A passing report cannot make the R6 human decision or R7 catalog
  admission;
- `blueprint_pipeline.measurement_research_monitoring` (with
  `scripts/measurement_research_monitor.py`) — release monitoring that emits
  version-change alerts, stale-profile flags, R0 intake drafts, regression
  checks, and requalification-trigger proposals. It proposes only; humans
  approve, advance, and apply through the admission machinery.
- `blueprint_pipeline.measurement_research_monitor` — the broader primary-source
  snapshot lane: it sanitizes injected source metadata, persists no source
  prose or credentials, diffs immutable snapshots, recommends benchmarks,
  emits bounded regression plans, and defines a monthly schedule contract. It
  neither fetches arbitrary sources by itself nor installs an external host
  scheduler.

The two monitoring modules are intentionally layered rather than competing
authorities. `measurement_research_monitoring` is the lightweight
release/version watcher and GitHub-release CLI. `measurement_research_monitor`
is the catalog-wide, source-type-neutral snapshot/diff contract used after an
operator or approved fetch adapter has reduced a primary source to the checked
metadata shape. Both are proposal-only inputs to human-controlled admission.

Research candidates progress through the separately checked
`measurement_research_admission.v1` R0-R8 state machine and
`docs/schemas/measurement_research_admission.v1.schema.json`. Production
measurement qualifications bind the digest of an R7/R8 admission record.

## Request-to-route flow

```text
customer question (untrusted)
        |
        v
interpretation agent: proposed task/claim/material/sensor scope
        |
        v
validated Decision/Evidence Request + exact maintained testbed
        |
        v
deterministic requirement compiler
        |
        +--> required solver/sensor/geometry capabilities
        +--> required site measurements
        +--> requested claim level C0-C8
        +--> rights/privacy/replay/budget constraints
        |
        v
immutable catalog snapshot
        |
        +--> complete method capability profiles
        +--> independently approved, signature-verified qualifications
        |
        v
hard filtering -> exact scope containment -> lexicographic ranking
        |
        +--> single method
        +--> composite route
        +--> authoritative physical-evidence route
        `--> abstention + smallest next action
```

The selected route remains a plan. It has `execution_authorized=false`, never
sets proof, and cannot initiate provider spend or physical robot work.

## Dynamic per-site and per-task behavior

The kernel derives requirements from both the claim and the immutable testbed:

- `task_distribution.measurement_task_class` selects the controlled task class;
- the claim selects the requested measurement and C0-C8 ceiling;
- task material, robot, end effector, controller, sensor, metric, and parameter
  ranges narrow the scope;
- the testbed-bound `site_evidence_profile` supplies validated local facts;
- request constraints enforce local-only, commercial-use, provider-training,
  retention, portability, latency, and cost rules.

An explicit `task_measurement_requirements` object may be supplied after
boundary validation. Otherwise the deterministic compiler derives the minimum
requirements from the controlled task/claim taxonomy. An agent interpretation
is always marked non-authoritative.

The controlled task classes cover static reachability, collision-free motion,
rigid pick/place, insertion/assembly, doors/drawers/handles,
valves/switches/buttons, contact-rich dexterous manipulation, visual
perception, visual navigation/active perception, transparent/reflective
objects, small/thin/occluded objects, locomotion, mobile manipulation in
clutter, human-robot interaction, long-horizon execution, garment
manipulation, cable/hose routing, granular, fluid, food, and tactile
manipulation. Unknown task classes are rejected, never approximated by a
nearby class.

"Deformable" is not a router value. Material regimes are controlled
(garment cloth, towel/sheet, rope/cable/hose, paper/cardboard/thin sheet,
elastomer, foam, elastoplastic dough/clay, granular media, viscous/free-surface
fluid, plastic/fabric bag, carton/box packaging, cuttable/multiphase food,
surgical tissue, plus rigid/none), each adding its own capability and
site-evidence requirements; generic words such as `deformable`, `soft`, or
`cloth` fail closed. Structured interaction sections (contact regime,
continuous collision, self-contact, friction axes, force output, deformation
family) and sensor modalities add further requirements, and solver
alternatives (for example SPH/CFD/MPM for free-surface fluid) are alternative
groups of which the composite route must cover at least one member.

The union of task class, claim, material regimes, sensor modalities, and
interaction is a deterministic minimum: agent-prepared requirements may narrow
or extend the scope but are rejected if they fall below that floor.

## Hard eligibility

A candidate is rejected before ranking when any required boundary fails:

- required capability absent;
- capability not covered by the exact qualification record;
- required site measurement unavailable or unvalidated;
- task, material, robot, end effector, controller, sensor, metric, action-rate,
  or parameter range outside qualification scope;
- method version or capability-profile digest mismatch;
- qualification not independently approved, dual-signed, and
  admission-record-bound (R7/R8);
- qualification expired;
- requested claim above the qualified ceiling;
- qualification ceiling above the method-family cap (exact kinematics C2; every
  simulation, renderer, reconstruction, world-model, provider, or framework
  family C4; physical evidence alone reaches C6-C8);
- physics authority (contact, articulation, collision, material dynamics)
  asserted for an appearance, renderer, world-model, capture, or framework
  family;
- declared robot embodiment/end-effector interface or control rate
  unsupported;
- deterministic replay required but unsupported;
- subprocessor regions unknown or outside the allowed set;
- site privacy forbids external processing and the method is not local;
- site commercial rights not cleared, or the claim is in the site's
  `forbidden_claims` limitations;
- local-only, retention (including a hard zero-retention requirement),
  training-use, portability, latency, or cost constraint violated.

Unknown scope fields are gaps, never wildcards. Price and speed cannot rescue an
ineligible method.

## Composite routing

A method is eligible for only the qualified capabilities it covers. The kernel
then composes the smallest route whose union covers every required capability.
Within the eligible tier, it ranks lexicographically by:

1. independent physical accuracy error;
2. uncertainty;
3. scope distance;
4. harmful false-negative rate;
5. reproducibility;
6. privacy preference;
7. latency;
8. cost;
9. stable method identity.

This prevents a visually strong renderer from becoming a collision method and
allows, for example, raw captured observations, a qualified renderer, validated
colliders, an articulated-contact solver, and physical evidence to occupy
different stages of one route.

## Site-evidence semantics

Every evidence record has separate `available` and `validated` booleans plus a
record identity. Availability alone is insufficient. The current controlled
evidence vocabulary includes metric scale, robot/site registration, coverage
uncertainty, validated meshes and colliders, articulation and actuation
measurements, mass/inertia, friction/contact, material parameters, initial
material state, calibrated RGB/depth/LiDAR/IMU, sensor calibration and timing,
force/tactile data, controller calibration, physical specimens, and physical
outcomes.

The raw capture bundle and provenance remain authoritative. The profile audits
what exists; it does not rewrite or repair capture truth.

## Agentic integration

The existing Task Evaluation Supervisor remains the agent harness. Its roles
are deliberately asymmetric:

- The claim/task interpreter proposes per-claim `task_measurement_requirements`
  from the controlled taxonomy when the testbed carries a site evidence
  profile, distinguishes reach/open/rank/safety claims, and turns unknown task
  classes or generic material words ("deformable", an ambiguous "bag") into
  the smallest clarification. It cannot authorize that interpretation or lower
  the deterministic minimum.
- The capture/testbed supervisor runs the deterministic capture-evidence audit
  and proposes the smallest per-gap measurement (metric-scale check,
  registration, collider validation, articulation measurement, material
  identification, sensor calibration, force/tactile collection, targeted
  recapture). It cannot infer mass, friction, collision validity,
  articulation, material behavior, or rights from appearance.
- The scenario/adversarial proposer doubles as the qualification designer: it
  drafts frozen R4 benchmark preregistrations (splits, physical measurements,
  metrics, thresholds, failure criteria) under the matching Q-protocols and
  the three benchmark blueprints. It can never approve its own experiment,
  reveal held-out labels or hidden material parameters, or grade
  vendor-submitted results.
- The evaluation-method specialist may explain the embedded deterministic
  measurement decision, rejections, composite stages, claim boundary, and next
  action. It cannot select or qualify the route.
- The deterministic router validates all sidecars and embeds the complete
  `measurement_routing_decision` into the claim's Evidence Plan.
- Human approval remains required for initial qualification, rights exceptions,
  safety/human-sensitive protocols, claim-ceiling changes, conflict overrides,
  and any override of deterministic abstention.

The OpenAI Agents SDK prompt explicitly states that a splat is not a collider,
a mesh is not a validated collider, OpenUSD is not physics readiness, and a
feature is not scoped qualification. Agent results remain non-authoritative and
have no proof effect.

## Claim hierarchy

| Level | Permitted meaning | Planning boundary |
| --- | --- | --- |
| C0 | capture provenance or observation | raw capture authority |
| C1 | reachability/kinematics | exact metric geometry and registration |
| C2 | visibility/geometric collision | calibrated sensor geometry or validated collider |
| C3 | named-simulator dynamic behavior | exact qualified solver/contact/material/controller scope |
| C4 | comparative policy ranking | held-out paired real/sim evidence for the exact population |
| C5 | sim-to-real transfer hypothesis | C4 plus transfer studies; not physical success |
| C6 | physical task success | accepted real robot execution |
| C7 | deployment readiness | real operations/reliability/recovery/human factors |
| C8 | safety certification | applicable formal safety process and physical evidence |

The route's claim boundary is a permission for evidence collection. Planning
never establishes the claim. In particular, `physical_success_established`,
`deployment_readiness_established`, and `safety_certification_established`
remain false in a routing decision.

## Qualification governance

Each measurement qualification binds:

- exact method/version and capability-profile digest;
- qualified capabilities and C0-C8 ceiling;
- task/material/site/robot/end-effector/controller/sensor/metric/range scope;
- physical accuracy, uncertainty, scope distance, harmful false-negative,
  reproducibility, and privacy metrics;
- expiration;
- an independently verified approval signature;
- `agent_approved=false` and `self_grading=false`.

The repository does not manufacture trust in a detached signature. The catalog
ingestion boundary records only an approval already verified by the owning
identity/governance system. Invalid, unverified, expired, self-graded, or
version-mismatched records are excluded by the kernel.

Engine, solver, plugin, adapter, driver/numeric backend, capture-pipeline,
license/privacy, or qualification-scope changes require a new record or an
explicit compatibility decision. Public papers and vendor benchmarks remain
research candidates until Blueprint executes the frozen admission protocol.

The admission state machine enforces sequential R0-R8 transitions, signed human
roles at every stage, hidden-label and vendor-self-grading prohibitions, frozen
benchmark preregistration, independent held-out execution, independent R6
qualification, R7 catalog admission, and R8 monitoring/requalification. Agents
may prepare candidate material but cannot sign or advance the state.

## Abstention

Abstention is a successful scientific outcome. It contains stable blocker
codes, prohibited fallbacks, the smallest next action, and — whenever site
evidence is the blocker — the embedded capture-evidence audit. Actions follow
the research taxonomy: targeted recapture, metric-scale check, robot/site
registration, sensor calibration, collider validation, articulation
measurement, material identification, force/tactile collection, adapter work,
qualification benchmark, rights approval, physical execution, and contract
clarification. C6-C8 claims always resolve to physical execution: more
simulation can never unblock a physical-success, deployment, or safety claim.

The kernel does not silently select an available simulator, relax a claim,
extrapolate a qualification, or use visual similarity as physics evidence.

## Implemented versus not yet proven

Implemented and hermetically tested:

- six digest-bound contracts, the capture-evidence audit contract, and the
  checked JSON Schemas;
- complete machine-readable capability-field enforcement;
- deterministic task/claim/material/sensor/interaction requirement derivation
  with a floor agents cannot lower;
- controlled site-evidence vocabulary, per-gap audit, and smallest-action map;
- version/signature/expiration/scope/rights/privacy/replay/region/interface/
  operations hard filters;
- method-family claim-ceiling caps and physics-authority denial for
  appearance, renderer, world-model, capture, and framework families,
  including the world-model role enum;
- lexicographic ranking, composite coverage, and solver-alternative groups;
- C0-C8 claim boundary with route type (single, composite, direct
  observation, physical test) and evidence-package binding;
- abstention, smallest-next-action, and mandatory physical execution for
  C6-C8;
- the R0-R8 admission state machine with retained stage evidence, R6-gated
  catalog admission, split-leakage and held-out binding guards, vendor
  self-grading prohibitions, and requalification-trigger suspension;
- the research intake catalog (landscape dossiers, Q-protocols, benchmark
  blueprints, priorities, standing abstentions) with zero production routes;
- adapter descriptors and non-invasive local probes for the priority methods,
  with complete tri-state capability drafts and fail-closed admission packets;
- a uniform development execution request/worker/receipt/bundle contract and a
  real MuJoCo 3.11.0 rigid-contact worker with exact implementation/runtime/
  solver/case/log binding and deterministic double-run replay;
- a real Pinocchio 4.1.0/Coal 3.0.3 planar Q-KIN worker and three-case corpus
  covering reachable-clear, reachable-discrete-collision, and unreachable
  boundaries, with exact forward-kinematic verification, GJK signed distance,
  and deterministic replay. It is finite-sample synthetic development evidence,
  not continuous collision, captured-site registration, or physical proof;
- a checked qualified-geometry-to-site-evidence bridge that can admit an
  independently qualified metric mesh/collider and a signed measured
  robot/site registration into a maintained testbed without agent promotion or
  proof inflation;
- a broad artifact compiler that consumes raw capture manifests and the
  capture/reconstruction/qualification artifact families, attaches the
  resulting `site_evidence_profile.v1` to a maintained testbed, and stays
  fail-closed: raw streams and metric-geometry manifests are candidates, while
  only the existing collider, articulation, and material gates can validate
  their physical evidence records;
- a real SAPIEN 3.0.3 physics-only rigid-contact worker over the shared
  sphere/box corpus, with exact SAPIEN/PhysX/implementation/solver binding,
  headless operation, enhanced-determinism settings, and deterministic double
  replay, plus a schema-checked plan/execute corpus aggregate. This does not
  implement or validate ManiSkill tasks, policies, or rendering;
- a real Drake 1.55.0 MultibodyPlant CPU rigid-contact worker over the shared
  sphere/box corpus, with SAP discrete contact, point contact, an isolated
  supported Python runtime, exact implementation/solver/case/receipt binding,
  and deterministic double replay. It does not establish hydroelastic contact,
  general robot/task support, captured-site accuracy, or qualification;
- a real OpenCV 4.11 calibrated Capture-to-Observation development worker plus
  a schema-checked two-trial synthetic corpus and aggregate corpus runner, with
  exact nanosecond timestamp handling, non-coplanar target enforcement,
  deterministic replay, and nonqualification flags at the corpus, receipt,
  prediction, suite, and supervisor boundaries;
- a real PyElastica 0.3.3.post2 Capture-to-Deformation cable development
  worker, schema-checked two-regime Cosserat-rod corpus, and aggregate runner,
  including exact material/geometry/load/timestep binding, sampled trajectory
  replay, and displacement/strain/force summaries;
- a separate real MuJoCo 3.11 Capture-to-Deformation cloth worker,
  schema-checked sag/contact corpus, and aggregate runner for the
  `elastic2d=stretch` flex formulation, with solver-warning rejection and
  explicit exclusions for bending, self-contact, garment topology, and
  captured-material claims;
- a separate real MuJoCo 3.11 spherical-particle granular development worker,
  schema-checked two-regime column-collapse corpus, and aggregate runner with
  exact particle/contact/solver binding, deterministic replay, spread,
  settling, normal-force, and penetration summaries. Its contracts force
  synthetic-only parameters and exclude DEM, cohesion, nonspherical grains,
  pouring/tool interaction, and physical material characterization;
- a direct tactile-sequence development worker and schema-checked stable/slip
  corpus with synchronized marker/contact/normal/shear channels, deterministic
  reduction, and explicit exclusion of real-sensor calibration, physical force
  truth, and TacSL/DiffTactile qualification;
- a model-neutral world-model action-fidelity development worker and two-case
  corpus that reuse the strict WAM numeric consistency contract, reject motion
  reuse across different commands, keep ranking metrics unmeasured, and
  preserve the frozen `thesis_not_supported` verdict;
- runnable benchmark contracts for geometry/contact, observation, and
  deformation (cloth, cable, granular, and tactile lanes), including hidden-label,
  split-isolation, independent-execution, threshold, and R4/R5 evidence
  guards;
- receipt-backed predictions, independent-executor proof fields, a minimum
  repeated-trial gate, evaluator-identity binding, and computed 95% confidence
  intervals for R5 evidence candidates;
- a route-to-execution development bridge that binds a selected measurement
  stage to the uniform worker boundary and then creates a separate immutable
  Evidence Plan attachment containing the exact plan, claim, route, receipt,
  case, and prediction digests. A same-logical-case MuJoCo/Drake report retains
  both engine identities and numeric metric ranges without treating agreement
  as qualification;
- an on-demand SimReady object lane that retains 3DGS as appearance, inserts
  per-object USD/MJCF physics drafts, and feeds all generated collider/mass/
  friction records back to the router as `validated=false` candidates. Its
  schema-checked preflight records the local trimesh/USD/MuJoCo checks and
  optional Blender/NVIDIA validator availability before target-simulator
  admission; load/stability never upgrades physical validity. Dynamic display
  additionally requires an exact source-digest-bound Gaussian object partition:
  selected rows are absent from the static background, preserved in one
  object-local splat, and driven with the collider by one body-pose channel;
- release monitoring plus immutable primary-source snapshot/diff monitoring,
  candidate/version/access alerts, benchmark recommendations, bounded
  regression plans, a monthly cadence contract, and a read-only monthly GitHub
  Actions lane for public release feeds and bounded regressions;
- proposal-only interpreter, capture-audit, and qualification-designer agent
  roles, with research descriptors/specifications/alerts visible to the agents
  but protected admission, execution, catalog, and qualification fields forced
  false, plus the worked-example suite from the routing research;
- embedding inside the existing Evidence Plan and agent supervisor result.

Not established by this repository change:

- an engine-specific production execution wrapper and R7 qualification for
  MuJoCo, Drake, Isaac/PhysX/RTX, Newton, SAPIEN, Chrono, FLASH, RGBench,
  SimWeaver, SOFA, tactile systems, or any provider. The repo-local descriptors,
  probes, draft profiles, and benchmark ports exist. The narrow MuJoCo 3.11.0
  rigid-contact, articulation, insertion, flex-cloth, and spherical-granular
  workers, the Newton 1.4.0 CPU XPBD rigid-contact worker, the Drake 1.55.0
  CPU SAP/point-contact worker, the SAPIEN 3.0.3 PhysX CPU rigid-contact worker,
  and the execution-bound Isaac Sim 6.0.1 PhysX worker and guarded Vast canary,
  plus the OpenCV calibrated-observation worker now run correctly, but none of
  their development fixtures is captured-site
  physical accuracy evidence or a general method qualification. The SAPIEN
  worker is deliberately physics-only and does not establish ManiSkill task,
  policy, sensor, renderer, or benchmark support;
- Chrono::Granular GPU, characterized-material DEM, pouring, or tool-interaction
  evidence. The executable PyChrono 10.0.0 development port uses core
  `ChSystemNSC` CPU/Bullet/PSOR for a bounded synthetic 27-sphere collapse and
  explicitly records that the specialized GPU module was not used;
- execution of a real customer capture through the existing observed-surface
  compiler, independent collider qualification, and the new signed robot/site
  registration bridge. The contracts and deterministic join exist, but this
  change contains no customer-site metrology artifact. The Q-KIN corpus remains
  a synthetic planar two-link boundary test with primitive Coal geometry and
  finite joint interpolation; it does not establish a general URDF, mesh,
  self-collision, continuous-collision, MoveIt, Drake, cuRobo, or
  site-metrology route;
- a completed Capture-to-Geometry-and-Contact, Capture-to-Observation, or
  Capture-to-Deformation physical benchmark using independently collected
  held-out labels. The observation lane now has a complete two-case synthetic
  development execution corpus; the geometry/contact lane has complete
  two-case MuJoCo sphere/box drop, door/drawer articulation, and clear-versus-
  interference square-peg insertion development corpora; and the deformation
  cable lane has a
  complete two-regime PyElastica development corpus. The cloth lane also has a
  complete two-regime MuJoCo flex stretch-only development corpus. The
  granular lane has a complete two-regime rigid-sphere column-collapse
  development corpus, but not the required characterized materials,
  pouring/tool-interaction cases, Chrono::Granular GPU/commercial/research-MPM
  comparisons, or physical labels. A separate two-regime PyChrono core-NSC
  development corpus is complete, but it is not a method-neutral physical
  comparison. None is a physical or held-out qualification dataset;
  the external physical datasets, qualification executions, and evaluator
  signatures are not present;
- a completed Q-TACT physical benchmark against the exact real sensor. The
  direct tactile development corpus validates sequence reduction and slip-rule
  plumbing only; it contains no real calibration, physical labels, TacSL or
  DiffTactile predictions, or independent evaluator receipt;
- a completed held-out world-model policy-ranking benchmark. The action-
  fidelity development suite has no provider output, real-policy outcomes,
  ranking labels, action-motion correlation, or ranking-regret measurement and
  therefore does not qualify OSCAR, Cosmos, RoboWorld, IWS, or GigaWorld;
- an external host scheduler or a credentialed universal source fetcher. The
  checked monthly GitHub Actions workflow becomes operational only after merge
  with Actions enabled and covers public GitHub release feeds; non-GitHub and
  restricted sources remain deployment configuration;
- Blender or NVIDIA Content-Agent Validation execution in the current local
  preflight. The exact environment probe found neither executable; their
  optional validator slots are recorded as typed unavailable without install,
  network access, or provider calls. The current green preflight proves only
  local geometry/USD generation plus headless MJCF load and numerical
  stability;
- independent R6 human decisions or R7 production catalog admissions for the
  research candidates;
- policy-ranking validity (the current verdict remains `thesis_not_supported`);
- a completed Isaac/PhysX Vast development execution. The exact official
  Isaac 6.0.1 NGC digest, clean-commit input bundle, canonical paid allocator,
  zero-retry spend/TTL gate, independent watchdog, output validator, teardown,
  and provider-zero code paths are implemented, but no provider result may be
  claimed until the paid canary returns and those receipts validate;
- physical task success, deployment readiness, or safety certification.

Those are evidence-generation gates, not reasons to weaken the router.
