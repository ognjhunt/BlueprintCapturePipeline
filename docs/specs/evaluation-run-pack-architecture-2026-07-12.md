# Evaluation Run pack architecture — 2026-07-12

## Why

The product goal is general and dynamic: any captured site, any robot, any
task family, any policy, evaluated through one engine. Historically the G1
kitchen lane grew into both the first proven vertical slice *and* too much of
the engine: `isaac_g1_kitchen_parity_job.py` mixed scene-agnostic
orchestration (launch retry, spend guards, teardown proofs) with hardcoded
kitchen/G1 assumptions (`KitchenRoom.usd`, `/workspace/bundle/kitchen`, G1 USD
paths, kitchen-named schemas).

## Target

```
Evaluation Run
  + scene bundle              (which assets, how the worker addresses them)
  + robot adapter             (which embodiment; resolves a RobotProfile)
  + task/scenario pack        (which scenarios / task file)
  + policy adapter            (which policy; remote-runtime env contract)
  + runtime/provider profile  (providers, image resolution, spend caps, lane)
  + proof contract            (evidence schemas + closure contract the run must emit)
```

Sites ("g1_kitchen", "g1_warehouse", a new customer capture) are **packs**:
data passed into the engine, never the engine's shape.

## What is implemented (this change)

`src/blueprint_pipeline/evaluation_run.py`:

- Frozen dataclasses `SceneBundle`, `RobotAdapter`, `TaskScenarioPack`,
  `PolicyAdapter`, `RuntimeProviderProfile`, `ProofContractBinding`, composed
  by `EvaluationRunSpec` (`evaluation_run_spec.v1`). All components expose
  `validation_blockers()` (repo convention) and the spec fails closed:
  strict `evaluation_run_spec_from_dict` rejects unknown keys, missing
  components, and schema drift; `assert_valid()` raises with the full blocker
  list.
- Pack registry: `register_evaluation_pack` / `get_evaluation_pack` /
  `known_evaluation_pack_ids`.
- Generic engine seams (scene-parameterized, formerly kitchen-hardcoded):
  `inspect_scene_asset_namelist`, `inspect_scene_asset_dir_layout`,
  `inspect_scene_asset_zip` (layout + sha256 content inventory), and
  `build_runner_request` (request keys come from the spec, e.g. legacy
  `kitchen_usd`/`g1_usd` vs neutral `scene_usd`/`robot_usd`).
- CLI: `python -m blueprint_pipeline.evaluation_run --list-packs`,
  `--pack <id> [--out <path>]`, `--spec-json <file>` (validate an external
  spec; nonzero exit on blockers).

Built-in packs:

- **`g1_kitchen`** — the historical lane as pure configuration. Every legacy
  identifier is pinned verbatim (`isaac_g1_kitchen_parity_job.v1`,
  `kitchen_asset_layout_validation.v1`, `g1_kitchen_attempt_closure.v1`, …) so
  previously emitted evidence stays valid.
- **`g1_warehouse`** — proves a second site is configuration only. Claim
  boundary is explicit: `pack_definition_only`; no GPU run has executed it and
  nothing beyond spec composition may be claimed until a live run emits its
  proof contract.

`isaac_g1_kitchen_parity_job.py` now consumes the pack instead of
re-declaring values: scene/robot/policy/runtime/proof constants are
single-sourced from `get_evaluation_pack("g1_kitchen")`; the kitchen asset
inspectors and `build_request` delegate to the generic functions
(byte-identical output, verified by equivalence tests in
`tests/test_evaluation_run.py` and the existing 97-test kitchen job suite).

## Deliberately NOT changed

- The kitchen job's public entrypoints, CLI, evidence schemas, filenames, and
  the `g1_kitchen_attempt_closure.v1` contract — the live-run closure work on
  this branch depends on them.
- `build_parity_bundle`'s shipped-module list and the runner
  (`scripts/run_isaac_g1_kitchen_parity_eval.py`) — the runner already
  consumes `RobotProfile` via `--robot-id`/`--robot-profile-json`.
- No renames of `isaac_g1_kitchen_*` modules; renaming would disguise
  coupling and break historical evidence.

## Migration path (next steps, in order)

1. **Plumb the robot adapter through the orchestrator CLI**: pass
   `--robot-id`/`--robot-profile-json` from the job CLI to the runner request
   so the embodiment is data end to end (the runner side already exists).
2. **Generalize `build_parity_bundle`**: mount name and shipped scene assets
   from `SceneBundle`; required-file list per pack.
3. **Spec-driven launch**: derive `RenderLaunchSpec` / spend-guard /
   `StartupSupervisorRequest` inputs from `RuntimeProviderProfile` in
   `run_isaac_g1_kitchen_parity_job`, then rename the engine (keeping the
   kitchen module as a thin alias) once the FABLE live-run episode closes.
4. **New packs**: express `warehouse_isaac_scenarios` /
   `lightwheel_kitchen_isaac_scenarios` scenario families and customer-site
   canonical packages as packs; join with `evaluation_prep_stage` catalogs and
   `success_claim_contracts` layer requirements per task.

## Tests

`tests/test_evaluation_run.py` (38 tests): pack registry, fail-closed
validation, manifest/JSON round-trip, legacy-equivalence of inspectors and
runner request against the kitchen job, scene-neutral behavior for the
warehouse pack, CLI. Fast lane green: 3497 passed.
