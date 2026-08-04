# ADP-010 Blocker And Human Action Packet

Status: **ADP-010 BLOCKED**
Audit date: 2026-08-04
Audited checkout: `codex/arm-decision-proof-v1` at
`d6b99c038841a34a3b8a700aa02f731974f73ee8` before the local decision-design
correction

## Observed Result

The checkout contains no admissible design-partner packet. The authoritative
partner-selection document is an unfilled intake specification. The ADP-008
SIMPLER artifacts are a retrospective public reference with a
`development_only` ceiling, not a partner. The older controlled-beta ledger and
fixtures explicitly do not prove a rights-cleared customer capture, a partner
task, two partner candidates, physical holdout authority, or partner value.
Repository history contains no ADP-010 admission, ADP-011 integration freeze,
or ADP-020 partner protocol approval artifact.

Consequently:

- admitted partner count: `0`;
- ADP-010 score: not computable from evidence;
- required-row zero check: not computable from evidence;
- ADP-011: dependency-blocked by ADP-010;
- ADP-020: dependency-blocked by ADP-010 and three missing same-digest human
  approvals;
- no capture, reconstruction, simulator-scene construction, production policy
  evaluation, holdout access, or physical trial was started by this goal.

## Typed Blockers

| Blocker | Exact missing human or partner evidence |
| --- | --- |
| `partner_identity_and_task_owner_missing` | Legal/organizational partner identity plus a named task owner/operator with authority to state task and reset truth. |
| `real_allocation_decision_missing` | A written statement of the unresolved physical-test allocation decision and how each possible result changes it. |
| `two_runnable_candidates_missing` | Exactly two current runnable candidate identities, immutable weight/configuration digests, shared-interface evidence, and recent execution receipts. |
| `bounded_task_and_evaluator_missing` | One fixed-arm rigid-object pick/place task, machine-verifiable success evaluator, termination rules, and repeatable reset evidence. |
| `scorecard_evidence_missing` | Evidence for all 12 scorecard rows sufficient to reach at least 20/24 with no zero in a required row. |
| `physical_holdout_authority_missing` | Named independent custodian, affordable trial budget, disjoint holdout reservation, randomization/interleaving authority, and release procedure. |
| `rights_and_case_study_authority_missing` | Durable permission for existing-asset use or bounded capture, policy receipts, outcomes, privacy/storage, and agreed bounded case-study evidence. |
| `integration_inventory_missing` | Exact robot, gripper, cameras, workcell, objects, CAD/calibration/capture/scene/log assets, policy runtime, schemas/transforms, rates, reset/logging behavior, simulator, dependency, storage, compute, and adapter requirements. |
| `protocol_design_inputs_missing` | Explicit baseline and alternative, minimum decision-relevant difference, conditions and frequencies, calibration/holdout partitions, exclusions/invalid region, cost/time limits, optional owner-preregistered secondary metrics, and amendment policy. |
| `same_digest_approvals_missing` | Task owner, holdout custodian, and Blueprint approval receipts that each quote the final immutable protocol digest. |

## Ready-To-Send Partner Request

Subject: Bounded two-candidate fixed-arm evaluation study intake

> Blueprint is selecting exactly one design partner for a prospective,
> preregistered evaluation of one existing fixed-arm rigid-object pick/place
> task. The study will compare exactly two candidates that already run through
> the same interface, seal a simulation decision before physical outcome
> access, and adjudicate it with a randomized or interleaved holdout controlled
> by an independent custodian. It is not deployment certification, safety
> approval, or a success guarantee.
>
> Please reply with the evidence listed below. Do not send private assets or
> credentials by ordinary email. Provide immutable digests, current receipts,
> and storage references first; we will agree on an authorized transfer channel
> separately if the scorecard passes. Missing facts will remain blockers rather
> than being inferred.

### A. Partner And Decision

1. Partner legal/organizational name.
2. Named task owner/operator, role, and durable approval identity.
3. Exact unresolved allocation decision and the next physical-test budget it
   controls.
4. Minimum success-rate difference that would change that allocation.
5. Why physical testing is scarce and how many future candidate pairs could
   reuse the result.

### B. Task, Reset, And Evaluator

1. Robot, gripper, fixed cameras, workcell, rigid objects, and target identities
   with model/serial or immutable asset identifiers where appropriate.
2. One bounded pick/place trial definition, termination and timeout rules, and
   machine-verifiable primary success metric.
3. Reset checklist/script, reset-equivalence check, invalid-trial region, and
   intervention rule.
4. Normal task conditions and approximate frequencies; proposed calibration
   conditions and disjoint physical-holdout conditions.
5. Any proposed secondary metric. For each, explicitly state that the partner
   task owner preregisters it before outcomes; otherwise Blueprint will omit it.

### C. Exactly Two Candidates And Existing Stack

For the explicit **baseline** and **alternative** separately, provide:

1. candidate ID, weight/configuration digest, code revision, prompts if any,
   and dependency/container lock;
2. a current runnable receipt showing the real candidate loads and queries;
3. shared observation schema, preprocessing/transforms, action schema,
   controller/inference rates, and termination behavior;
4. runtime entrypoint and logging/media behavior.

Also inventory existing CAD, URDF/USD/MJCF, calibrations, captures, simulator
scenes, task assets, and logs with versions, digests, provenance, storage
location, and access constraints. State the simulator and dependency versions
already in use. Do not propose a new engine solely for this intake.

### D. Authority And Rights

Provide durable receipts identifying who may approve each item:

1. the exact protocol;
2. use of existing assets and, only if later required, bounded new capture;
3. randomized/interleaved physical holdout execution;
4. outcome release and use;
5. policy/log/media processing, retention, deletion, and privacy terms;
6. bounded case-study evidence and redaction limits.

Name a holdout custodian independent of candidate execution/analysis. Confirm
that the custodian can keep holdout outcomes inaccessible until Blueprint
presents a valid sealed-decision digest.

### E. Scorecard Evidence

Return a `0`, `1`, or `2` score for each row in
[`PARTNER_SELECTION_PACKET.md`](PARTNER_SELECTION_PACKET.md), with one or more
evidence references per score. Admission requires at least `20/24` and no zero
in any required row. Blueprint will recompute the score; self-scoring alone is
not admission.

## Human Approval Sequence After A Passing Intake

1. Blueprint validates the scorecard and rights receipts and emits one
   digest-bound ADP-010 admission or a typed rejection.
2. Blueprint and the partner freeze only the existing-stack integration fields
   required by ADP-011.
3. Local deterministic compilation produces the exact candidate × condition ×
   reset × seed × repetition schedule. The supported conservative v1 method is
   an independent two-proportion fixed-sample approximation at planning
   variance `p=0.5`; at `alpha=0.05`, power `0.80`, and a 20-point minimum
   decision-relevant difference it requires 99 scheduled trials per candidate.
   A partner-specific protocol may choose different frozen inputs, but the
   declared method and calculation must agree.
4. Blueprint provides one canonical protocol file and digest. No outcome access
   or execution begins.
5. Each approver replies separately with this exact statement:

> I approve protocol `<protocol_id>` with digest
> `sha256:<64 lowercase hex characters>` in my role as `<partner task owner |
> physical holdout custodian | Blueprint approver>`. I confirm that no physical
> holdout outcome was disclosed to Blueprint or candidate operators before this
> approval.

6. Deterministic validation admits ADP-020 only when all three receipts quote
   the identical protocol digest. A changed field creates a new digest and
   requires all three approvals again.

## Execution Boundary

This packet authorizes only human review and response. It does not authorize an
upload, private-data processing, terms acceptance, paid compute, capture,
simulation production run, physical robot motion, outcome release, publication,
push, merge, deploy, or external outreach by an agent.
