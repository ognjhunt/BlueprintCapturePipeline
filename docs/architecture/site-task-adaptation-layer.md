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

**Adaptation** means a change to a backend's parameters or inference behavior:
fine-tuned weights, conditioning channels supplied at inference, or generation
inputs Blueprint constructs. It does **not** mean evaluator machinery, scoring,
diagnostics, or admission gates. Those grade a method; they are not part of the
method being graded, and attaching method-identity or self-grading obligations
to them would invert the independence this ADR is trying to protect.

By that definition adaptation is already shipped, unnamed:
`g1_microwave_*_finetune*` (fine-tuned weights),
`droid_oscar_skeleton_conditioning` and `franka_droid_skeleton_conditioning`
(conditioning channels at inference), and `oscar_visual_augmentation_*`
(constructed generation inputs).

Explicitly **outside** this ADR: `policy_ranking_causal_conditioning` is a
label-blind causal/placebo diagnostic over generated outputs, and
`wam_conditioning_fidelity` is a fail-closed admission contract. Neither
modifies or conditions an executable backend. They are grader-side machinery
and are governed by the independence rule below, not by method identity.

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
digest-bound.

**Both shapes require a new contract version.** `EVALUATION_RUN_SCHEMA_VERSION`
is `evaluation_run.v1` and the compiler canonicalizes the whole spec into the
digest that *is* method identity. Adding a required `adaptation` field —
nullable or not — invalidates stored leaves that omit it, and defaulting it
silently changes their canonical digests, breaking exactly the digest-bound
joins that `physical_outcome_learning` and the plan/authorization path rely on.
An earlier draft of this ADR claimed the field could be added
schema-compatibly; that was wrong.

So: introduce `evaluation_run.v2` with explicit translation from v1, following
the precedent already set by `LegacyEvaluationPackSpec` and
`legacy_evaluation_pack_to_leaf_spec`. v1 leaves keep their existing digests and
translate forward to an explicit null adaptation; they are never silently
reinterpreted.

Within v2, prefer (a) `runtime_provider_profile` gains the `adaptation` field
over (b) a seventh `adaptation_profile` component, because it keeps the
adaptation bound to the runtime that executes it and leaves the six-part
interface intact. An unadapted backend carries an explicit null adaptation, not
an absent field, so "no adaptation" is a recorded fact rather than a missing
one.

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

**Resolution — four holds, all required:**

1. **The qualification record is the firewall.** A Blueprint-adapted method
   earns its own record against held-out accepted real anchors under the same
   process as any third-party method. It may not inherit, borrow, or extrapolate
   from the base backend's record.
2. **The grader must be independent of the adaptation author.** Data
   independence is not author independence: disjoint partitions prevent
   leakage, but they do not satisfy the router's rule that the subject may not
   grade itself. An adapted method is graded through the replaceable
   external-scorer boundary, by an evaluator that is not the party that authored
   the adaptation. Where no independent evaluator is available for a claim, the
   adapted method is not qualified for it and the run abstains — the router does
   not fall back to self-grading.
3. **Blueprint-authored methods face a stricter bar, not a looser one.** We
   control both the method and the grader's inputs, so the evidence requirement
   is higher: held-out anchors must be disjoint from every partition used to fit
   the adaptation, checked mechanically, as `physical_outcome_learning` already
   checks calibration and held-out sample IDs.
4. **Publish the delta.** When an adapted variant is selected over its base
   backend, the run exposes base-versus-adapted performance on the same held-out
   set. A routing decision that favors our own method must show its work. This
   converts the conflict into a credential.

Holds 2 and 3 are separate requirements and neither substitutes for the other.
This ADR does not amend the router's independence rule; it applies that rule to
Blueprint as a method author.

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

- Whether the v1 → v2 translation is performed eagerly for stored leaves or
  lazily on read. Eager rewriting touches historical artifacts and needs an
  explicit immutability argument.
- Whether an adapted method may ever be the *only* qualified method for a claim,
  or whether a Blueprint-authored method always requires an independent
  qualified alternative in the candidate set before it can be selected.
- Whether the base-versus-adapted delta is exposed to the buyer in every run or
  only on request.
