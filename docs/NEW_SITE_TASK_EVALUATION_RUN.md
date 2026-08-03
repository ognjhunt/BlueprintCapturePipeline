# Provider-Neutral New-Site Task Evaluation Run

`blueprint_pipeline.new_site_task_evaluation_run` is the deterministic top-level
compiler for a new capture after provider/source admission. It does not choose
Isaac by default and it does not let an analyzer, provider, or agent authorize
its own evidence.

This compiler is downstream of, and distinct from,
`blueprint_pipeline.new_site_task_evaluation`. That upstream command validates
capture materialization and an explicit task through a zero-spend development
lane; it intentionally proves no comparative policy ranking. A caller may use
its digest-bound capture and site evidence to prepare this compiler's admitted
source, reconstruction, target, placement, and routing artifacts, but it must
still supply five independently identified learned-policy execution receipts.
Neither command may reinterpret the other's development output as stronger
evidence.

The compiler binds, in order:

1. a digest-bound Blueprint or provider-derived source profile;
2. the same capture's full-resolution native 3DGS appearance artifact and its
   separately registered dynamics geometry;
3. automatic visible-object/affordance/task proposals that passed deterministic
   3D target binding;
4. a qualified robot placement, using Franka Panda for manipulation and Unitree
   G1 only for an explicitly humanoid task class;
5. qualified floor/support composition only when the source collision lane needs
   it, and a qualified SimReady task-zone replacement only when the selected
   interaction semantics require one;
6. the existing deterministic task/site measurement router's best exactly
   qualified engine or composite stack; and
7. exactly five immutable learned-policy identities and five execution receipts
   under one frozen reset and observation/action contract.

The target stage accepts the canonical output of
`rendered_scene_task_target_orchestrator` directly. It verifies both the outer
orchestration digest and nested target-analysis digest, joins the source scene
and native splat to the registered reconstruction, and derives the selected
binding only from the matching digest-bound binding result. The compiler maps
the versioned analyzer task-family vocabulary into the measurement router's
controlled task classes and preserves the analyzer's Franka/G1 robot binding.
An unknown family, robot/family mismatch, foreign scene/splat binding, or
ambiguous task-zone interaction mode fails closed before placement. The
placement stage likewise recognizes
`external_scene_robot_placement_candidate.v1`, but that
artifact remains an analytic/runtime-visualization candidate: it produces a
`qualified_robot_placement_missing` abstention until an independent placement
qualification exists. No caller-side, site-specific JSON reshaping upgrades
either artifact.

The task outcome metric is digest-bound and timestamped before execution. Each
attempt must bind the selected route and placement, show at least one fresh
learned-policy query and action, and preserve reset, observation, action,
contact, collision, and metric evidence. Only attempts whose task metric is
supported are ranked. A completed attempt with incomplete outcome evidence is
retained but excluded from ranking. Equal best metric values produce an explicit
set of shared winners and no sole `winner_candidate_id`.

For the Franka inspection lane, use
`blueprint_pipeline.franka_inspection_learned_policy_lane`. It freezes the
embodiment, two-camera 224x224 DROID observation, per-step observation sequence,
8-D DROID action conversion, 15 Hz control, matched reset, inspection-coverage
metric, and provider-neutral runtime interface. The runtime queries the learned
policy anew for every control step, retains the complete native chunk, and
executes only native row zero before observing again.

Real executions enter this compiler through
`learned_policy_execution_bundle.v1`. The compiler recomputes every embedded
observation, native output, normalized action, simulator action, contact,
collision, terminal-observation, metric, and attempt digest. When that bundle is
present, caller-supplied parallel candidates, attempts, or metrics are rejected;
the top-level authorization must byte-match the authorization inside the bundle.
Hermetic fake receipts are refused by the real bundle builder.

Run the compiler with:

```bash
python -m blueprint_pipeline.new_site_task_evaluation_run \
  --request <new-site-task-evaluation-request.json> \
  --output <new-site-task-evaluation-run.json>
```

The portable schemas are:

- `docs/schemas/new_site_task_evaluation_request.v1.schema.json`
- `docs/schemas/new_site_task_evaluation_run.v1.schema.json`

## Frozen policy-by-scenario matrix (v2)

V2 preserves the complete v1 reader and replaces the single shared reset with
an immutable five-policy by scenario matrix. An inspection pack contains three
to five admitted scenarios and binds the task, source/site, reconstruction,
target, robot, placement, metric, and aggregation rule. Each scenario freezes
its own reset-state digest and simulator seed, target and distractor state,
public observation contract, evaluator-only state digest, observation settings,
and any task-valid perturbation or qualified geometry/material variant.

The smallest recommended inspection pack is:

1. nominal;
2. one evidence-bounded robot-base or camera/observation perturbation; and
3. one visibility/occlusion stress case only when target visibility and the
   observation change are qualified within the exact evidence ceiling.

Unsupported variants belong in `excluded_scenarios` with a rationale. Scenario
generation always records `scenario_generation_may_authorize_new_claims=false`;
it cannot create scale, physics, material, collision, sensor, or metric
authority.

`execute_policy_scenario_matrix` constructs the exact 5 x N grid and invokes
every cell in deterministic scenario/candidate order. A runner exception,
failed receipt, or missing receipt becomes a terminal cell record; later cells
still run. The policy query payload omits evaluator-only state. Every returned
receipt must bind both immutable policy identity and scenario identity, plus the
scenario-specific reset, seed, route, placement, metric, and execution traces.
The v2 `request_digest` covers the frozen pre-execution request projection and
therefore excludes only `matrix_execution_packet`; the packet binds that digest
and has its own digest, avoiding a circular self-reference.

Aggregation uses only scenarios supported for all five policies. It reports
the excluded scenarios and exact cells, attempt/supported/paired coverage per
candidate, a deterministic paired-bootstrap interval, preregistered ties, and
supported versus unsupported metrics. A catastrophic cell is always listed and
makes that candidate ineligible for a winner claim even when its mean is high.
Missing cells, insufficient paired coverage, or an all-catastrophic cohort
produce a terminal abstention while retaining the diagnostic matrix.

The v2 contracts and replayable hermetic examples are:

- `docs/schemas/new_site_task_scenario_pack.v1.schema.json`
- `docs/schemas/new_site_task_evaluation_request.v2.schema.json`
- `docs/schemas/new_site_policy_scenario_execution_packet.v1.schema.json`
- `docs/schemas/new_site_task_evaluation_run.v2.schema.json`
- `docs/examples/new_site_task_evaluation_request.v2.example.json`
- `docs/examples/new_site_task_evaluation_run.v2.example.json`

`migrate_v1_request_to_v2` emits an explicitly labeled one-scenario legacy
projection. It preserves readability but states that no multi-scenario evidence
or ranking upgrade was created. `project_v2_result_to_v1` provides a v1-shaped
compatibility projection whose matrix provenance remains visible. Learned-policy
and scripted-controller matrices are separate evidence types; a controller
candidate cannot enter this contract.

The retained ARKitScenes 40958756 packet at
`docs/evidence/arkitscenes_40958756_scenario_matrix_preexecution_packet.v1.json`
is a real public-dataset evidence binding and an exact pre-execution abstention.
It has zero admitted cells because appearance reconstruction, independent metric
scale, collision, placement, engine, and site metric gates remain missing. It is
not a completed real-site matrix and does not reinterpret ARKitScenes as
Blueprint Raw Contract truth.

## Fail-closed behavior

The result is either `completed` or `abstained`. An abstention identifies the
terminal stage and smallest missing measurement. It never silently substitutes:

- a provider export for Blueprint Raw Contract truth;
- provider-declared units for independently qualified scale;
- native 3DGS appearance for collision or dynamics authority;
- DiFix, Artifixer, or another presentation output for evaluation evidence;
- Isaac for an unqualified or better-qualified task/site engine;
- a scripted controller for a learned policy;
- mismatched starting states for a comparative ranking; or
- controller ranking for learned-policy ranking.

The result remains simulation evidence. Even a completed five-policy ranking
does not prove physical task success, safety, deployment readiness, or transfer
to a live robot.

## Current evidence boundary

Repository tests exercise the contracts with synthetic fixtures. They prove
deterministic validation, routing, abstention, and ranking behavior only. A real
Polycam Developer Mode ZIP, registered native splat, qualified site geometry,
and five real learned-policy execution receipts are still required before a new
physical site can produce a real completed run.
