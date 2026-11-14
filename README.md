# BlueprintCapturePipeline

> Current owner direction (2026-11-12, recorded in the shared doctrine): a
> site films a job once, and Blueprint turns the video into a **job packet**
> that any robot deployment company can evaluate without visiting. This repo
> is the packet factory: it builds the **simulation of the site** with the
> job's object in it and carries the success rule. The simulation is a means,
> not the product. Blueprint's own test of openly available robot software is
> an **early trial** inside the packet; it is not the site's answer, and a
> scripted run only checks the simulation. Hardening the automatic test for
> any kind of job, several objects or unattended scale is paused. See
> [`PLATFORM_CONTEXT.md`](PLATFORM_CONTEXT.md) and [`AGENTS.md`](AGENTS.md).

Beta task coverage is open across task families and robot forms (owner
direction 2026-10-27); execution still needs the actual assets, reset,
independent state measurements and a compatible runtime. Spend stays inside
approved allowances.

For capture-derived evaluations, the
[supervised beta runbook](docs/runbooks/first-site-team-beta.md) covers exact
runtime delivery, simulation qualification, results and recovery. It is a
conditional evidence route rather than a required stage for every job.

[`docs/README.md`](docs/README.md) defines documentation authority. Other docs
remain stable contracts, compatibility material, or historical evidence; they
are not independent roadmap authority.

## Product Boundary

Beta task admission is open across task families and robot forms (owner
direction, 2026-10-27). Video and task analysis assess applicable current robot
capability evidence. A keyword, an absent runtime adapter, or one incompatible
robot configuration cannot establish that no robot team can do the job.
Missing capability evidence remains unresolved; missing simulator support keeps
the accepted job in evaluation preparation and demand discovery. Exact runtime
compatibility, independent scoring, rights and spending controls still apply
before execution. This serves the beta job-to-evidence journey.

This uses the existing WebApp Site assessment agent, which combines video
observations with Firestore `robotTeams` capability/provenance records and
authorized company-history research. Its `read_robot_registry` tool and
evidence bindings are documented in the WebApp's
[task-assessment audit](https://github.com/ognjhunt/Blueprint-WebApp/blob/main/docs/research/task-assessment-audit-2026-10-07.md).
Pipeline's runnable-policy registry is a separate execution inventory.

Website task contexts retain an optional, digest-bound `task_family`. It is an
open string, not an intake allowlist. Native preparation has explicit profiles
and independent state scorers for rigid placement, articulated open/close,
pouring, wiping, insertion, stacking, button presses, cloth folding and ordered
multi-step work. Additional families retain their exact missing adapter and
measurement requirements. The family adapter consumes a sealed task contract,
scene assets, reset, native metric bindings and runtime identity; a registered
profile does not supply missing engine physics or establish a successful run.
Older contexts keep their compatibility path.

An ambiguous `general_task` hint can be resolved by operation semantics retained
by the existing Atlas review. That derivation binds the original task context,
source video and completed planner response; it never rewrites the owner's
brief. Resolution uses the current profile catalog and preserves every requested
operation. Missing semantics, unknown operations or ambiguous stage identities
keep the job in preparation. Registered families still need their actual native
contracts, assets, reset and runtime inputs before execution.

A retained reading may leave overall job coverage false because CAD, metric
scale, numeric criteria or execution qualification is pending. A separate,
source-bound operation clarification preserves that reading and every issue,
while distinguishing unresolved robot operations from preparation requirements.
It can accompany an unstarted scale review. Completed scale/scene stages remain
adopted; the fallback is a text-only review capped at $0.25 plus at most one
schema repair. Read-only inspection and default handoff preparation only adopt
retained clarifications. The existing Atlas controller and authorized capture
execution can request one under the retained site grant. This resolves routing
only; owner confirmation and execution qualification remain independent.

See [task-family evaluation](docs/runbooks/task-family-evaluation.md) for the
native contracts, state measurements and execution boundary.

Placement into an open container uses the capture-derived floor, surrounding
walls and unobstructed interior. Native scoring checks the whole oriented
object volume, support, release and stability. A destination mask or the
object's center alone cannot establish containment.

A job assessment can use existing evidence without creating a capture,
testbed or run. WebApp owns the site side: the job page, the first look on
paper, the site's permissions and the responses of deployment companies.
Pipeline owns the simulation of the site, the success rule, the early trial,
normalized results, uncertainty and abstention.

When new technical evidence is useful, reuse the Decision/Evidence Router and
existing bounded **Task Evaluation Run**. Blueprint can operate supported
licensed policies itself; approved team-supplied endpoints and integrations
remain available. A model or robot profile does not establish executable
compatibility, delivery support or physical performance. Capture delivery does
not itself start evaluation or authorize spending.

| Term | Meaning |
| --- | --- |
| Task Evaluation Run | Bounded decision, partial decision, or explicit abstention |
| Site-Task Testbed | Versioned reusable substrate behind a run |
| Candidate MSER | Smallest replica believed sufficient for one decision; unproven until physical adjudication |
| Physical Outcome Join | Exact link from the sealed result to authoritative held-out physical outcomes |
| SiteBench | Optional name for the bounded case study, not a second product |

Beta task coverage across task families and robot forms (arms, mobile
manipulators, wheeled humanoids and humanoids) is authorized by the 2026-10-27
owner direction above. General-ranking campaigns, universal runtime support,
provider bakeoffs, world-model expansion, post-training products, and
multi-site generalization stay frozen.

## Architecture To Reuse

Build on the existing control plane rather than creating a parallel SiteBench
stack:

- `decision_evidence_contracts.py`, `decision_evidence_router.py`, and
  `decision_evidence_execution.py` for claim routing and abstention;
- `site_task_testbed_compiler.py` for maintained testbed artifacts;
- `evaluation_run_contract.py` and `evaluation_run_execution.py` for stable
  scene/robot/task/policy/runtime/proof seams;
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
- `docs/schemas/` — versioned schemas
- `docs/` — stable dependency docs, compatibility material, and historical evidence
- `doctrine/` — canonical shared doctrine synced to Capture and WebApp
- `scripts/` — setup, verification, retention, and authorized runtime helpers

## Setup And Focused Verification

```bash
python -m pip install -e .[dev]
python3 scripts/verify_shared_doctrine.py
python -m blueprint_pipeline.impacted_test_selection
```

Run only the smallest deterministic tests covering a change. The repository fast
lane and full suite are integration/promotion tools, not default build-loop
commands. See [`AGENTS.md`](AGENTS.md).

Paid CPU, GPU, and provider reconstruction paths remain behind the
canonical allocator:

```bash
python -m blueprint_pipeline.paid_resource_allocator cpu-build <arguments>
python -m blueprint_pipeline.paid_resource_allocator gpu-canary <arguments>
python -m blueprint_pipeline.paid_resource_allocator provider-reconstruction <arguments>
```

## Task Admission

Before accepting work, answer:

1. Which job-packet step and observed blocker does it unblock?
2. What observed artifact proves the blocker is removed?
3. Why is existing infrastructure insufficient?
4. What is the smallest reversible change?

If any answer is missing, the work is out of focus.

Raw capture, owner task truth, and authoritative physical outcomes remain above
all generated, simulated, provider, and presentation artifacts. Agents may
propose and assemble; deterministic contracts authorize or abstain. Humans own
rights, physical safety/motion, task/reset truth, holdout release, and external
publication.
