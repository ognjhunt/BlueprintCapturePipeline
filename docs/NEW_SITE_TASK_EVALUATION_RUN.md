# Provider-Neutral New-Site Task Evaluation Run

`blueprint_pipeline.new_site_task_evaluation_run` is the deterministic top-level
compiler for a new capture after provider/source admission. It does not choose
Isaac by default and it does not let an analyzer, provider, or agent authorize
its own evidence.

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

The task outcome metric is digest-bound and timestamped before execution. Each
attempt must bind the selected route and placement, show at least one fresh
learned-policy query and action, and preserve reset, observation, action,
contact, collision, and metric evidence. Only attempts whose task metric is
supported are ranked. A completed attempt with incomplete outcome evidence is
retained but excluded from ranking.

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
