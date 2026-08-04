# AI Onboarding Map

## One Program

BlueprintCapturePipeline has one active program: **Arm Decision Proof v1**.

Read completely, in order:

1. [`../../AGENTS.md`](../../AGENTS.md)
2. [`../arm_decision_proof_v1/north_star_contract.json`](../arm_decision_proof_v1/north_star_contract.json)
3. [`../arm_decision_proof_v1/README.md`](../arm_decision_proof_v1/README.md)
4. [`../arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md`](../arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md)
5. [`../arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md`](../arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md)
6. [`../../PLATFORM_CONTEXT.md`](../../PLATFORM_CONTEXT.md)
7. [`../../WORLD_MODEL_STRATEGY_CONTEXT.md`](../../WORLD_MODEL_STRATEGY_CONTEXT.md)
8. [`source-of-truth-map.md`](source-of-truth-map.md)
9. [`../DOCTRINE_PRECEDENCE.md`](../DOCTRINE_PRECEDENCE.md)

The north star is one prospective two-candidate fixed-arm decision on a new
partner workcell, adjudicated by matched held-out physical trials. The Task
Evaluation Run is the product; the Site-Task Testbed is the substrate; the
candidate MSER is the construction method; the Physical Outcome Join is the
proof.

## Active Architecture Route

| Program stage | Existing implementation to inspect first | Required v1 output |
| --- | --- | --- |
| Raw capture truth | `materialization.py`, capture Raw Contract consumers, reconstruction control plane | immutable capture descriptor, timestamps/poses/intrinsics/depth/provenance |
| Partner task truth | `site_task_testbed_compiler.py`, decision/evidence contracts | owner-approved task distribution, reset, outcome, invalid region |
| Testbed compilation | `site_task_testbed_compiler.py`, `canonical_site_package.py` | versioned testbed bound to exact evidence |
| Claim routing | `decision_evidence_contracts.py`, `decision_evidence_router.py`, `decision_evidence_execution.py` | qualified evidence plan or smallest blocker |
| Leaf run boundary | `evaluation_run_contract.py`, `evaluation_run_execution.py` | scene/robot/task/policy/runtime/proof bindings |
| Condition matrix | `new_site_task_evaluation_matrix.py`, scenario variation modules | stable candidate/condition IDs and frozen partitions |
| Runtime | existing Franka/DROID, MuJoCo/Isaac, and provider-neutral adapters | closed-loop steps and complete episode receipts |
| Statistics | `rank_fidelity_statistics.py`, `robot_eval_calibration.py` | power plan, uncertainty, decision rule, claim ceiling |
| Prospective seal | decision envelopes and immutable artifact graph | result digest before holdout release |
| Physical adjudication | `physical_outcome_learning.py` | exact condition-ID Physical Outcome Join and new testbed version |
| Showable result | buyer readout and report renderers | evidence matrix and bounded case study |

Inspect before adding. Prefer a thin adapter over a parallel SiteBench stack.

## Development Substrates

Audit and pin SIMPLER first as the single external real-to-sim harness candidate.
Use existing local assets to isolate downstream paths:

- `tests/fixtures/decision_evidence_rigid_object_v1/vertical_slice.json` —
  routing, partial decisions, abstention, and outcome versioning;
- `tests/fixtures/new_site_loading_bay_v1` — capture-to-testbed compiler shape;
- `tests/fixtures/kitchen_task_min` — existing USD/runtime plumbing only.

Select at most one other existing SimReady/OpenUSD artifact when a recorded
runtime blocker cannot be exercised by these fixtures.

All are `development_only`. They cannot establish the partner capture, task
truth, site/robot registration, task dynamics, observation-domain match,
sim-to-real decision fidelity, or customer value. Do not hand-author substitute
evidence. A missing partner measurement remains a typed blocker.

Complete ADP-008 before new capture/reconstruction features. Public outcomes may
be programmatically withheld until after sealing to test software separation,
but the run remains retrospective because the labels are published.

## Truth Route

```text
raw capture and provenance
  + owner-approved task/reset/outcome truth
  + exact policy/robot/runtime identities
  -> candidate replica and qualified evidence plan
  -> condition-level simulation receipts
  -> sealed prospective decision
  -> authoritative held-out physical outcomes
  -> exact Physical Outcome Join
  -> bounded verdict or explicit abstention
```

Raw capture and authoritative physical outcomes outrank derived geometry,
generated appearance, SimReady assets, simulation, world-model/provider output,
and reports. Each derived method keeps its own claim ceiling.

## Task Admission

Before editing, answer:

1. Which ADP backlog item is this?
2. Which day-7/day-14/day-28/day-35/day-42 gate does it unblock?
3. What observed artifact proves completion?
4. Why is the existing implementation insufficient?
5. What is the smallest reversible change?

If any answer is missing, stop. Historical docs and code do not authorize their
former lanes.

## Frozen Areas

Humanoid/G1, locomotion, deformables, cables, cloth, granular tasks, tight
insertion, force-sensitive assembly, five-policy/general-ranking campaigns,
world-model/evaluator expansion, reconstruction/provider bakeoffs, universal
runtime support, dynamic-scene research, post-training products, multi-site
generalization, and unrelated buyer/growth polish are frozen.

Touch a frozen component only after recording an observed Arm Decision Proof
blocker and demonstrating that the component is the smallest path. Preserve
compatibility readers and safety gates, but do not extend the old lane.

## Authority

Agents may inspect, plan, implement, run hermetic development fixtures, validate,
and draft packets. Deterministic code authorizes scientific transitions. Humans
own partner admission, task/reset truth, rights/privacy, physical safety and
robot motion, holdout release, spend/provider authorization, publication, and
claim promotion.

No live provider, paid compute, external upload, physical action, or publication
is authorized by program status alone.

## Focused Verification

```bash
python -m pytest -q tests/test_arm_decision_proof_focus.py
python3 scripts/verify_shared_doctrine.py
python -m blueprint_pipeline.impacted_test_selection
```

Name the claim or risk protected by every command. Do not run a broad suite by
default.
