# Decision/Evidence Router

Status: accepted, version 1 (2026-07-29)

## Decision

Blueprint has one customer-facing product: a **Task Evaluation Run**. A
maintained Site-Task Testbed is the immutable-version substrate reused by
successive runs. Each run starts with a provider-neutral Decision/Evidence
Request, decomposes the decision into claims, plans the cheapest currently
qualified evidence for each claim, and ends with a Decision Envelope or explicit
abstention.

The control plane is:

```text
Maintained Site-Task Testbed + Decision/Evidence Request
                       |
             deterministic claim router
                       |
      Evidence Plan (zero, one, or many leaf run specs)
          |             |                 |
   analytic/capture   EvaluationRunSpec   physical request/read-only outcome
          \             |                 /
              Normalized Evidence Results
                       |
                 Decision Envelope
                       |
           append-only Physical Outcome Join
                       |
         new testbed version + narrow calibration
```

The router minimizes evidence acquisition cost, delay cost, and expected
decision loss subject to claim-specific false-safe limits, minimum coverage,
rights/privacy constraints, exact applicability, budget/latency,
reproducibility, and available physical evidence. Provider identity, visual
realism, parameter count, and runnable defaults are not qualification.

## Stable boundaries

`blueprint_pipeline.evaluation_run_contract.EvaluationRunSpec` and
`evaluation_run.v1` are the canonical leaf execution boundary. The six
replaceable components remain `scene_bundle`, `robot_adapter`,
`task_scenario_pack`, `policy_adapter`, `runtime_provider_profile`, and
`proof_contract`.

The older class historically named `evaluation_run.EvaluationRunSpec` is a
static pack definition, not a runtime leaf. It is now named
`LegacyEvaluationPackSpec`; its old import remains an alias and
`legacy_evaluation_pack_to_leaf_spec` performs explicit, tested translation.
Legacy defaults produce candidates only and never qualify a method.

Geometry, captured observations, and accepted prior physical outcomes use
non-simulator adapters. A physical step with no accepted outcome emits a bounded
evidence request and cannot initiate a robot run.

## Qualification

A qualification record binds the exact method/profile/implementation/evaluator
digests to one claim type, task family, site conditions, embodiment, sensors,
controller/action representation, calibration partition, predictions, accepted
real outcomes, confidence intervals, coverage, abstention, and error behavior.
Transfer across any of those axes is disabled unless held-out evidence earns a
new, wider record. The subject provider or model may not grade itself.

Methods that are uncalibrated, unavailable, out of domain, over budget, missing
inputs, rights-incompatible, non-reproducible, or above the false-safe ceiling
are rejected. Correlated methods do not count as independent evidence. Invalid,
uncertain, contradictory, unavailable, or under-covered results conditionally
execute the next qualified method. If none exists, the run abstains and names
the next cheapest experiment.

## Scientific boundary

The current policy-ranking result remains `thesis_not_supported`. Published
world-model work, including arXiv:2606.10366v1, is relevant to method
qualification because it emphasizes exact adaptation identity, calibration,
perturbation sensitivity, and scope-specific validation. It does not replace
the router, prove cross-site or cross-embodiment transfer, or upgrade the frozen
Blueprint verdict. An adapted policy or model version is a distinct method
identity and needs a new qualification record.

## Product and compatibility migration

Policy Improvement Run and Post-Training Data Package contracts remain readable
legacy/internal machinery. Translators turn their requests into provider-neutral
Task Evaluation Run requests; translation grants no qualification. Their CLIs
are explicitly deprecated and default robot-eval orchestration no longer emits
the legacy evidence export unless the request opts in.

Post-training is an allowed use of qualifying evidence inside a run. It requires
rights, consent/revocation, provenance, robot-action alignment, quality, and
held-out leakage gates. An export never implies that training occurred or a
policy improved. Optional internal candidate generation is not a SKU.

## Truth and immutability

Raw capture remains authoritative. Plans, results, decisions, and physical
outcomes bind exact digests. Later outcomes append a join record and may create
a new narrow qualification plus a new testbed version; they never rewrite a
historical plan, result, decision, or testbed version. Generated artifacts never
upgrade raw, physical, deployment, or safety claims.

## Non-goals

This decision does not build a foundation world model, reconstruction model,
physics engine, robot, teleoperation stack, marketplace, tournament, generalized
physical lab, procurement system, or deployment/safety-certification authority.

## Operator entrypoint

`blueprint-route-task-evaluation` exposes `plan`, `execute`, `aggregate`, and
`ingest-outcome`. Version 1 execution accepts only explicitly authorized
hermetic fixture adapters. It does not discover providers from defaults, spend
money, call a live provider, or initiate physical execution.
