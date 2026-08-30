# Evaluation-ready policy run contract

This contract is the no-spend bridge from an authenticated configured-scene
offering to the existing Task Evaluation preparation, activation, and terminal
delivery paths. It does not allocate a provider or execute an episode.

The Arm Decision Proof scope is the day-21 learned-policy execution/evidence
gate: one Franka Panda + Robotiq 2F-85 embodiment and exactly the ordered frozen
pair `pi05_droid`, `groot_n17_droid`. A different embodiment, candidate, order,
cell, seed, or policy-specific scenario change fails closed.

## Website boundary

The browser chooses only a preset. Team identity and the notification recipient
come from the authenticated WebApp session; neither is accepted as policy-run
configuration. The WebApp binds the route's source launch, immutable offering
digest, and server-owned inline `policy_run_setup` before forwarding the existing
preparation request.

The first release exposes these ordered presets:

| Preset | Cells per policy | Availability | Learned | Controls | Total |
| --- | ---: | --- | ---: | ---: | ---: |
| `quick_10` | 10 | enabled, default | 20 | 20 | 40 |
| `standard_100` | 100 | coming later | 200 | 200 | 400 |
| `deep_500` | 500 | coming later | 1,000 | 1,000 | 2,000 |

Every cell runs once for each learned candidate, once for the zero-action
negative, and once for the deterministic scripted positive. Thus a preset of
size `N` always has `2N` learned episodes, `2N` controls, and `4N` total
simulator episodes.

Quick has exactly one canonical anchor, two placement/approach cells, one
illumination cell, one camera/sensor cell, one bounded-physics cell, two
pairwise cells, and two held-out-composition cells. Standard and Deep are
disabled until their complete ordered manifests are published. Their preset
descriptors bind the parent prefix and a nesting proof so that enabling them
requires Quick to be the first 10 cells of Standard and Standard to be the
first 100 cells of Deep.

Duration and cost are never guessed. Each preset carries either an unavailable
estimate or a range with an as-of timestamp and digest-bound basis.

## Deterministic compilation

The server selection is
`task_evaluation_policy_run_selection.v1`. It contains the run ID, source launch
ID, offering digest, setup digest, and preset ID. It has no cells, seeds,
provider, email, or team field.

Pipeline compiles `task_evaluation_policy_run_configuration.v1` from the
published ordered prefix. One seed per cell is derived from the setup digest,
run ID, preset ID, and cell ID. The compiled configuration carries exact cells,
cell-spec digests, seeds, counts, no-retry guards, and evidence requirements.
An agent may propose or classify future variations, but cannot select compiled
cells or inspect outcomes during compilation.

## Controls-gated activation

Preparation seals a `task_evaluation_policy_run_plan.v1` and performs no
execution or provider mutation. Activation requires one digest-bound
`task_evaluation_policy_controls_qualification.v1` manifest with a passed
zero-action and scripted-positive receipt for every exact compiled cell.

The activation adapter then materializes `N` ordered campaign units. Each unit
uses the existing `native_task_arena_policy_campaign.v1` exact two-member
runtime contract for one cell and seed. Activation only writes this paired
campaign queue; it publishes no live profile or standing authority, requests no
paid execution, and allocates no provider. Later execution still requires the
existing paid-resource admission and authority gates.

## Terminal projection

`task_evaluation_policy_run_result_projection.v1` is delivered through the
existing authenticated Task Evaluation result-delivery record. A decision is
valid only when all `4N` episodes are complete, both candidates have the same
cells and seeds, controls are complete, all seven families are reported, and
every learned episode has its lossless policy-input frame manifest and derived
review video. A pre-observation failure must be represented by a typed media
gap. Incomplete evidence produces an abstention or partial result with an
explicit blocker; it cannot be promoted to a decision.

Simulation and review video remain non-physical evidence, and a policy never
grades itself.
