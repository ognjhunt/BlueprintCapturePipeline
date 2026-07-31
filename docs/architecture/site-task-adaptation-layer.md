# Site/Task Adaptation Layer

Status: proposed, version 0 (2026-07-31)

Amends [`decision-evidence-router.md`](decision-evidence-router.md) (accepted,
version 1, 2026-07-29). It does not weaken that decision; it names a layer the
router already routes across and closes two gaps that layer opens.

## Decision

Blueprint routes across third-party evidence backends and does not build
simulators, world models, reconstruction models, or physics engines. Blueprint
*may* adapt a third-party backend to a specific site and task — conditioning,
calibration, fine-tuning, augmentation — when that measurably improves accuracy
for a claim.

An adapted backend is a **distinct method identity**. It never inherits the base
backend's qualification, and Blueprint's authorship earns it no benefit of the
doubt.

## Why this needs its own decision

Adaptation is already shipped, unnamed. `g1_microwave_*_finetune*`,
`droid_oscar_skeleton_conditioning`, `franka_droid_skeleton_conditioning`,
`policy_ranking_causal_conditioning`, `oscar_visual_augmentation_*`, and
`wam_conditioning_fidelity` all adapt an external backend to a site or task.
Doctrine describes Blueprint as routing across replaceable backends and says
nothing about authoring variants of them. That silence hides two problems.

## Problem 1: method identity has no home in the leaf contract

`evaluation_run_contract.EvaluationRunSpec` has six replaceable components:
`scene_bundle`, `robot_adapter`, `task_scenario_pack`, `policy_adapter`,
`runtime_provider_profile`, `proof_contract`. A site-tuned backend variant is
none of them.

Qualification binds to method identity. If the adaptation is not *in* that
identity, two different tunings of the same base backend collapse to one
qualification record — and the ledger silently reports evidence from tuning A as
qualified when tuning B produced it. That is a correctness failure, not a
documentation gap.

**Resolution:** an adaptation is part of the method identity and must be
digest-bound. Either (a) `runtime_provider_profile` gains a required
`adaptation` field whose digest participates in method identity, or (b) a
seventh component `adaptation_profile` is added. (a) is preferred: it avoids a
schema-version break in the leaf contract and keeps the adaptation bound to the
runtime that executes it.

An unadapted backend carries an explicit null adaptation, not an absent field,
so "no adaptation" is a recorded fact rather than a missing one.

## Problem 2: Blueprint becomes a method author it also grades

`decision_evidence_contracts.py:421-426` enforces `self_grading:must_be_false`
and `provider_self_grading_forbidden`. `site_task_testbed_compiler.py:196` sets
`provider_or_model_self_grading_allowed: False`. The router ADR states the
subject provider or model may not grade itself.

Those rules are written as *provider*-scoped, where "provider" means the
external vendor. When Blueprint authors an adaptation, Blueprint is the
provider. Read literally the existing rule already fires on us; read as
intended, it does not fire at all. Both readings are wrong to leave standing.

This is the channel conflict `VISION.md` defers to rungs 4–5, arriving at rung 1
through a decision that sounds purely technical. Neutrality is the asset the
routing position depends on. It is spent quietly, by defaults, not loudly.

**Resolution — three holds, all required:**

1. **The qualification record is the firewall.** A Blueprint-adapted method
   earns its own record against held-out accepted real anchors under the same
   process as any third-party method. It may not inherit, borrow, or extrapolate
   from the base backend's record.
2. **Blueprint-authored methods face a stricter bar, not a looser one.** We
   control both the method and the grader, so the evidence requirement is
   higher: an adapted method requires held-out anchors disjoint from every
   partition used to fit the adaptation, and the disjointness is checked
   mechanically, as `physical_outcome_learning` already checks calibration and
   held-out sample IDs.
3. **Publish the delta.** When an adapted variant is selected over its base
   backend, the run exposes base-versus-adapted performance on the same held-out
   set. A routing decision that favors our own method must show its work. This
   converts the conflict into a credential.

## The layer stays deliberately thin

Adaptation is a cost of making a rented backend fit a site, not a capability
Blueprint accumulates for its own sake. It is expected to commoditize as
backends improve. Bias every decision here toward the smallest adaptation that
moves a claim across a qualification threshold, and toward deleting adaptations
that a newer backend version makes unnecessary.

An adaptation that cannot be justified by a specific claim on a specific
qualification record should not exist.

## Consequence for compounding

`physical_outcome_learning.py` disables transfer across site, task, and
embodiment unless held-out evidence earns a wider record. That gate is correct
and stays.

But per-site adaptation plus disabled transfer means every site gets bespoke
tuning whose calibration informs no other site. That is consulting margin, not
platform margin — and it is the same failure that turns a field network into a
headcount business rather than a moat. Whatever the durable moat turns out to
be, it depends on site N+1 costing less than site N. Adaptation without a
qualified generalization path guarantees the opposite.

Designing that path is therefore the central engineering problem behind both the
routing thesis and the network thesis. This ADR does not solve it and must not
be read as authorizing transfer.

## Non-goals

This decision does not authorize building a simulator, world model,
reconstruction model, or physics engine; does not permit an adapted method to
claim its base backend's qualification; does not enable cross-site, cross-task,
or cross-embodiment transfer; and does not upgrade the frozen
`thesis_not_supported` policy-ranking verdict.

## Open questions

- (a) versus (b) for adaptation identity in the leaf contract.
- Whether an adapted method may ever be the *only* qualified method for a claim,
  or whether a Blueprint-authored method always requires an independent
  qualified alternative in the candidate set before it can be selected.
- Whether the base-versus-adapted delta is exposed to the buyer in every run or
  only on request.
