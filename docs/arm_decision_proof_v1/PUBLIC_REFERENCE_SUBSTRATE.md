# Public Reference Substrate Decision

Status: candidate selection for the `public_reference_harness` phase
Decision: audit and pin **SIMPLER** first; do not run a multi-backend bakeoff

## Why The Program Is Harness-First

Near-term engineering value is in the evaluation harness, not in inventing
another capture or reconstruction system. Capture and reconstruction are inputs
behind a versioned evidence contract. New capture feature development is frozen
until a downstream harness run or partner proof identifies a specific missing
measurement that existing sources cannot provide.

The harness must accept an already-built public environment and still produce:

```text
source and rights manifest
-> normalized scene/robot/task/policy/proof bindings
-> two-candidate condition matrix
-> closed-loop execution and complete episode receipts
-> sealed development-only decision
-> externally sourced physical-reference outcome join
-> uncertainty, invalid region, and explicit claim ceiling
-> inspectable evidence matrix and replay command
```

If this cannot work with an existing environment, building a new capture stack
would hide a harness problem rather than solve it.

## Primary Candidate: SIMPLER

[SIMPLER](https://github.com/simpler-env/SimplerEnv) is the best first harness
substrate because its public repository provides:

- prepackaged real-to-sim manipulation environments;
- standard Gym-style reset/step execution;
- Google Robot and WidowX setups;
- rigid-object tasks including pick, move-near, stack, and place-in-container;
- policy integration examples for RT-1, RT-1-X, and Octo;
- visual-matching and variant-aggregation condition strategies;
- public real and simulated performance tables and metric code;
- environment, robot, object, controller, policy, and logging code;
- an MIT-licensed repository.

The [official project](https://simpler-env.github.io/) reports paired real and
simulated evaluation across about 1,500 episodes from each domain and explicitly
positions the environments for checkpoint selection and distribution-shift
analysis. That makes it useful for replaying a complete evidence chain without
pretending Blueprint produced the physical outcomes.

Before admission, pin:

- exact repository and submodule commits;
- task, robot, two policy/checkpoint identities, and inference dependencies;
- environment and asset digests;
- license and any model/checkpoint terms;
- public real-performance source and its condition granularity;
- control frequencies, action transformation, and evaluator semantics;
- compute requirements and a zero-spend local feasibility result.

Recommended first task shape: the simplest rigid pick/place or
place-in-container task for which two public policies and real reference outcomes
are available under the same interface. Choose based on the admission audit, not
the prettiest visualization.

## Secondary Candidates, Not Parallel Work

### PolaRiS

[PolaRiS](https://polaris-evals.github.io/) is the closest reference for the
later scene-compilation seam: it reports a 2–5 minute calibrated scan, 2DGS
appearance, collision meshes, inserted robots/objects, downloadable evaluation
environments, and 20 real plus 50 simulated rollouts per policy-task pair.

Use it only if SIMPLER proves insufficient for a recorded capture-to-replica or
site-variation seam. Do not integrate it in parallel merely because it is newer.

### REALM

[REALM](https://martin-sedlacek.com/realm/) is a possible alternate validated
manipulation benchmark, especially for a later DROID/OpenPI-shaped partner. It
requires its own exact code, asset, outcome, and license audit before admission.

### DROID

[DROID](https://droid-dataset.github.io/) provides 76,000 trajectories across
564 scenes and 86 tasks, with Franka robot state/actions, multiple camera views,
language, and calibration data. It is valuable for policy-interface and observed
trajectory fixtures.

DROID alone is not a complete SiteBench reference: it does not provide a
qualified interactive simulator replica and matched per-condition simulated and
physical evaluation result for each captured scene.

### Existing Site-Capture Datasets

ARKitScenes and similar reconstruction datasets remain useful for import,
provenance, scale, registration, and abstention tests. They do not contain the
paired robot policies, task/reset truth, and physical decision outcomes needed
to test the whole evaluation claim.

## What Public Data Can And Cannot Prove

Public references can prove that Blueprint's software:

- consumes an external scene without owning capture;
- normalizes two real candidates;
- executes conditions and records replayable receipts;
- seals a result before programmatic outcome release;
- joins external reference outcomes without rewriting them;
- reports uncertainty, contradiction, and abstention honestly.

They cannot prove:

- a prospective scientific result, because the physical outcomes are already
  public and may have influenced design;
- an unseen partner-site capture or task distribution;
- new site/robot registration and task-specific physics;
- customer value or avoided physical testing;
- general rank fidelity, deployment, or safety.

The public run is therefore a **retrospective harness qualification**, not the
north-star result. The partner run remains necessary only after the harness is
complete.

## Engineering Allocation

Until the public-reference harness passes:

- nearly all implementation effort goes to source ingestion, normalized
  contracts, two-candidate execution, receipts, sealing, outcome joins,
  statistics, abstention, replay, and evidence presentation;
- new capture/reconstruction feature work receives zero effort unless a measured
  blocker proves it is required;
- partner discovery, rights, task/protocol design, and physical access continue
  as a small parallel human lane so the harness is not optimized for a fictional
  customer interface.
