# BlueprintCapturePipeline

## Sole Active Program: Arm Decision Proof v1

Blueprint's only active objective is to produce one prospectively physically
validated, site-specific fixed-arm decision.

> Qualify the reusable service on exact public datasets first—including one
> metric 3DGS/collision object removal, released-code inpainting, and exact
> SimReady USD replacement—then take one fresh capture of one previously unseen
> fixed-arm workcell, make a prospective decision between exactly two frozen
> candidates or abstain, and adjudicate it with held-out physical trials.

North-star metric:
`prospectively_physically_validated_new_site_task_decisions`, current `0`, target
`1`.

Start here:

1. [`docs/arm_decision_proof_v1/north_star_contract.json`](docs/arm_decision_proof_v1/north_star_contract.json)
2. [`docs/arm_decision_proof_v1/README.md`](docs/arm_decision_proof_v1/README.md)
3. [`docs/arm_decision_proof_v1/PARTNER_SELECTION_PACKET.md`](docs/arm_decision_proof_v1/PARTNER_SELECTION_PACKET.md)
4. [`docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md`](docs/arm_decision_proof_v1/IMPLEMENTATION_BACKLOG.md)
5. [`docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md`](docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md)
6. [`docs/arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md`](docs/arm_decision_proof_v1/PUBLIC_REFERENCE_SUBSTRATE.md) — historical ADP-008 decision
7. [`docs/arm_decision_proof_v1/MASTER_GOAL_PROMPT.md`](docs/arm_decision_proof_v1/MASTER_GOAL_PROMPT.md)

[`docs/README.md`](docs/README.md) defines documentation authority. Other docs
remain stable contracts, compatibility material, or historical evidence; they
are not independent roadmap authority.

## Product Boundary

Blueprint has one customer-facing product: a **Task Evaluation Run**.

| Term | Meaning |
| --- | --- |
| Task Evaluation Run | Bounded decision, partial decision, or explicit abstention |
| Site-Task Testbed | Versioned reusable substrate behind a run |
| Candidate MSER | Smallest replica believed sufficient for one decision; unproven until physical adjudication |
| Physical Outcome Join | Exact link from the sealed result to authoritative held-out physical outcomes |
| SiteBench | Optional name for the bounded case study, not a second product |

The first envelope is one partner, one site, one fixed arm, one bounded
rigid-object pick-and-place task, and two genuine frozen candidates. Humanoids,
deformables, five-policy campaigns, universal runtime support, general ranking,
provider bakeoffs, world-model expansion, post-training products, and multi-site
generalization are frozen.

## Public Datasets, Inpainting, And SimReady Replacement

ADP-008's SIMPLER decision-harness replay is observed complete. The immediate
engineering objective is **ADP-009 public-scene qualification**, not new capture
technology: smoke exact Inpaint360GS on unchanged author data, then run an
InFusion primary adapter and AuraFusion360 quality challenger on one exact
rights-admitted InteriorGS/SAGE-3D pair; generate frozen object-present
render-derived method inputs where original cameras are unavailable, while
withholding external validation geometry and clean-background truth and
preserving the published metric frame;
remove one source object from appearance and collision; insert one exact
SimReady USD; compare manual authoring with NVIDIA USD Content Agents; then
transfer the frozen contracts to one targeted ScanNet++ scene after access.

Use existing assets now to exercise every downstream seam that does not require
new physical truth. Waiting for the partner capture would unnecessarily
serialize the valuable work.

The bounded development corpus is:

- `tests/fixtures/decision_evidence_rigid_object_v1/vertical_slice.json`;
- `tests/fixtures/new_site_loading_bay_v1`;
- `tests/fixtures/kitchen_task_min` only for existing USD/runtime plumbing.
- one exact rights-admitted InteriorGS scene and its matching SAGE-3D collision
  companion; an authored scene is a separate physics control, not a substitute;
- one targeted access-admitted ScanNet++ scene;
- exact Inpaint360GS, InFusion, and AuraFusion360 revisions, weights,
  dependencies, licenses, and author-data smokes;
- one exact replacement SimReady USD;
- one manual authoring control and one pinned NVIDIA USD Content Agents
  candidate after deterministic geometry exists.

GOR-IS may be added only as a separately admitted glossy/shadow ablation. A
paper-only method cannot enter the critical path. This is a claim-complete
ladder, not a provider or reconstruction bakeoff.

All reused assets remain `development_only`. They cannot qualify partner
capture, task-owner truth, registration, task-specific dynamics, policy-domain
match, sim-to-real decision fidelity, or customer value. Fixture success is a
software result, not the proof.

Until the one-command ADP-009 rehearsal passes, new capture feature development
is frozen unless a gate identifies a specific missing measurement. Partner
discovery and protocol/rights work continue as a small parallel human lane. The
next construction phase uses the existing Raw V3.2 path for one fresh workcell
with clean-background and object-present observations; physical holdout remains
required for the prospective north-star claim.

## Architecture To Reuse

Build on the existing control plane rather than creating a parallel SiteBench
stack:

- `decision_evidence_contracts.py`, `decision_evidence_router.py`, and
  `decision_evidence_execution.py` for claim routing and abstention;
- `site_task_testbed_compiler.py` for maintained testbed artifacts;
- `evaluation_run_contract.py` and `evaluation_run_execution.py` for stable
  scene/robot/task/policy/runtime/proof seams;
- `new_site_task_evaluation_matrix.py` and existing runtime adapters for
  condition execution;
- `rank_fidelity_statistics.py` for power and uncertainty discipline;
- `physical_outcome_learning.py` for immutable physical outcome joins;
- raw-capture, rights, privacy, provenance, paid-resource, watchdog, teardown,
  and provider-zero contracts as existing guardrails.

Preserve five-policy and legacy formats through explicit readers or translators.
Do not create fake candidates to satisfy them.

## Repository Map

- `src/blueprint_pipeline/` — capture-to-testbed orchestration, evaluation
  contracts, adapters, runtime, decisions, and evidence joins
- `tests/` — deterministic contract and fixture coverage
- `tests/fixtures/` — development-only inputs; never physical qualification
- `docs/arm_decision_proof_v1/` — sole active program
- `docs/schemas/` — versioned schemas, including the north-star focus lock
- `docs/` — stable dependency docs, compatibility material, and historical evidence
- `doctrine/` — canonical shared doctrine synced to Capture and WebApp
- `scripts/` — setup, verification, retention, and authorized runtime helpers

## Setup And Focused Verification

```bash
python -m pip install -e .[dev]
python -m pytest -q tests/test_arm_decision_proof_focus.py
python3 scripts/verify_shared_doctrine.py
python -m blueprint_pipeline.impacted_test_selection
```

Run only the smallest deterministic tests covering a change. The repository fast
lane and full suite are integration/promotion tools, not default build-loop
commands. See [`AGENTS.md`](AGENTS.md).

Paid CPU, model-volume, GPU, and provider reconstruction paths remain behind the
canonical allocator and do not become authorized merely because this program is
active:

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator model-volume <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction <arguments>
```

## Task Admission

Before accepting work, answer:

1. Which Arm Decision Proof backlog item and day-7/day-14/day-21/day-28/day-35/day-42
   gate does it unblock?
2. What observed artifact proves the blocker is removed?
3. Why is existing infrastructure insufficient?
4. What is the smallest reversible change?

If any answer is missing, the work is out of focus.

Raw capture, owner task truth, and authoritative physical outcomes remain above
all generated, simulated, provider, and presentation artifacts. Agents may
propose and assemble; deterministic contracts authorize or abstain. Humans own
rights, physical safety/motion, task/reset truth, holdout release, and external
publication.
