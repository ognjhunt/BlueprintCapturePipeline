# Provider-Neutral New-Site Task Evaluation Run

Production of the compiler prerequisites is owned by
`blueprint_pipeline.post_capture_evidence_spine`; see
[`runbooks/post-capture-evidence-spine.md`](runbooks/post-capture-evidence-spine.md).
The final compiler remains a strict verifier. Callers no longer need to reshape
site-specific source, reconstruction, geometry, target, placement, composition,
routing, or authorization JSON by hand.

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
