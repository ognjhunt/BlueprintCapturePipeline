# Provider-Agnostic WAM Compute - Design Spec

Date: 2026-06-27
Status: Draft, validated against current repo/artifacts

## Goal

Unify the provider-backed WAM execution path so OSCAR/Cosmos WAM bundles can run through a
single compute-provider interface with RunPod, Vast, and later providers behind it.

The intended shape is one orchestration core plus thin provider adapters. Adding provider
number 3 should mean adding one adapter and capability profile, not copying the WAM lane.

## Validation Summary

The pasted conversation is directionally right, but it needs two corrections before becoming
implementation truth.

Verified true:

- The render lane already has the desired provider abstraction:
  `src/blueprint_pipeline/gpu_render_providers.py` defines `RenderLaunchSpec`,
  `GpuRenderProvider`, `RunPodRenderProvider`, `VastRenderProvider`, and
  `get_render_provider("runpod"|"vast")`.
- The Isaac G1 `artic3` robot-POV artifact exists at:
  `output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611/pipeline/isaac_g1_parity_artic3/render_output/lightwheel_kitchen_g1_01_entry_to_sink/robot_pov.mp4`.
  It is H.264, 640x480, 8 frames, 3 fps. Its paired `trace.jsonl` has 8 rows,
  all with `policy_action: accepted_direct_collision_checked_motion`; final root position
  equals the target `[2.2, 0.9, 0.79]`.
- The paired `g1_projected_skeleton_trace.jsonl` has 8 rows and each checked row records
  `projected_landmark_count: 48`.
- `g1_binding.json` for the same run reports `controllable_articulation_detected: true`
  and `collision_enabled_verified: true`.
- The exact Isaac-to-OSCAR WAM input manifest at
  `output/.../pipeline/isaac_oscar_wam_job/wam_rollout_input_manifest.json`
  points to that `robot_pov.mp4`, the OSCAR skeleton trace, and the locomotion trace.
- `src/blueprint_pipeline/oscar_wam_provider_command_adapter.py` is still Vast-only for
  fresh paid launches: parser choices are `auto`, `replay-existing-provider-output`, and
  `vast-provider`, and fresh execution calls `create_async_vast_wam_run` /
  `poll_async_vast_wam_run`.

Corrections:

- It is not globally true that there is no RunPod WAM path. The repo has
  `src/blueprint_pipeline/runpod_wam_async_runner.py`, and
  `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py` can launch
  provider `runpod` with `provider_bundle_kind="wam"`. The accurate statement is:
  the standalone `oscar_wam_provider_command_adapter` does not yet expose RunPod as a
  first-class fresh-launch provider.
- The exact Isaac `artic3` OSCAR output was not complete at validation time.
  `wam_provider_output.json` was `blocked` on `vast_wam_provider_poll_blocked`; a newer
  Vast state file showed `instance_created` for instance `42880112`, with the phase log
  ending at `vast_heartbeat_started`. Do not claim a fresh OSCAR-generated MP4 for this
  Isaac job until a valid extracted `oscar_generated_rollout.mp4`, final poll manifest,
  and teardown/spend state exist.
- In the exact `artic3` render output directory I found `overview_000*.png` frames, but
  no `overview.mp4`. Older sibling render outputs do have `overview.mp4`; do not call the
  `artic3` third-person video complete unless the MP4 is assembled or located.

## Problem

The WAM compute surface is fragmented:

- `vast_wam_async_runner.py` owns Vast-specific create/poll/teardown, offer selection,
  phase logs, budget ledgers, signed URL download, output inspection, and dud detection.
- `runpod_wam_async_runner.py` owns RunPod-specific create/poll/delete, pod status handling,
  nonterminal output handling, output zip inspection, and teardown/spend reporting.
- `oscar_wam_provider_command_adapter.py` only launches Vast for fresh OSCAR WAM requests,
  even though it can import either Vast or RunPod-style completed output zips.
- `unitree_groot_n17_sonic_vast_persistent_session.py` has a RunPod path, but it does not
  share a stable provider-neutral WAM compute contract with the OSCAR adapter.

This makes future providers expensive and makes reliability fixes live in provider-specific
files instead of a shared WAM compute orchestration layer.

## Non-Goals

- Do not replace `gpu_render_providers.py`; render and WAM compute are related patterns but
  different contracts.
- Do not claim generated WAM video quality, physical robot readiness, safety validation,
  collision truth, or deployment approval.
- Do not make paid provider launches possible without explicit CLI and env gates.
- Do not print or persist raw provider credentials, signed URLs, or secret values.
- Do not remove existing Vast/RunPod entrypoints before compatibility tests prove the new
  layer preserves behavior.

## Proposed Architecture

Add `src/blueprint_pipeline/wam_compute_providers.py`.

### Neutral Spec

`WamComputeLaunchSpec` should carry:

- `name`
- `bundle_path`
- `provider_bundle_kind` (`wam`, `unitree_unifolm`, `unitree_groot_n17_sonic`)
- `image`
- `env`
- signed or file-backed transport URLs
- GPU sizing and provider capability hints
- budget controls and max-live controls
- expected output contract, including expected video count and output zip path
- entrypoint/watchdog timeouts
- claim-boundary metadata

### Provider Protocol

`WamComputeProvider` should expose:

- `available() -> dict`
- `build_request(spec, job_dir) -> dict`
- `create(spec, job_dir, *, allow_paid_launch: bool) -> WamComputeRunResult`
- `poll(job_dir, *, max_wait_seconds: int, teardown: bool) -> WamComputeRunResult`
- `teardown(job_dir, instance_id) -> WamComputeRunResult`
- `inspect_output(job_dir, output_zip_path) -> dict`

The shared result shape should include:

- `schema_version`
- `provider`
- `status`: `planned`, `blocked`, `running`, `completed`, `teardown_completed`
- `provider_command_status`
- `instance_id`
- `output_zip_path`
- `output_zip_present`
- `mp4_count`
- `extracted_video_paths`
- `runtime_result_status`
- `runtime_result_blockers`
- `phase_log_path`
- `budget_ledger_path`
- `teardown_manifest_path`
- `continuing_spend_from_this_run`
- `blockers`
- `raw_secret_values_recorded: false`

### Provider Adapters

Initial adapters should wrap current code, not rewrite all provider mechanics at once:

- `VastWamComputeProvider`
  - wraps `create_async_vast_wam_run`, `poll_async_vast_wam_run`
  - preserves offer selection, budget ledger, phase logging, and dud detection
  - maps Vast manifests into the neutral result shape
- `RunPodWamComputeProvider`
  - wraps `create_runpod_wam_async_run`, `poll_runpod_wam_async_run`
  - preserves pod status handling, nonterminal zip handling, delete/teardown, and
    `continuing_spend_from_this_run`
  - maps RunPod manifests into the neutral result shape

Add a registry:

```python
def get_wam_compute_provider(name: str | None) -> WamComputeProvider: ...
def list_wam_compute_providers() -> list[dict]: ...
```

### Shared Orchestrator

Add a thin orchestration helper:

```python
run_wam_compute_job(
    spec: WamComputeLaunchSpec,
    job_dir: Path,
    provider_order: Sequence[str],
    allow_paid_launch: bool,
    failover_on_blockers: Sequence[str],
) -> WamComputeRunResult
```

It should:

1. validate local bundle and transport inputs
2. choose a provider from explicit CLI/env order
3. create only when paid gates are explicitly open
4. poll until completed, blocked, or watchdog boundary
5. inspect output zip and extracted videos
6. teardown when completed or terminal-blocked
7. preserve exact phase and spend state
8. optionally fail over to the next provider only for configured, non-destructive blockers

Failover must be conservative. It is allowed for dud/startup/no-container/provider-output
transport blockers; it must not hide model/runtime failures such as bad checkpoint layout,
input materialization failure, or model dependency failure.

## Integration Plan

1. Add the neutral module and unit tests for provider registry, result normalization,
   output inspection, and no-spend blocking.
2. Wrap the existing RunPod and Vast runners behind the neutral provider adapters.
3. Update `oscar_wam_provider_command_adapter.py`:
   - add `--provider {vast,runpod,auto}`
   - preserve existing `--mode vast-provider` compatibility
   - add a new fresh-launch path through `run_wam_compute_job`
   - keep replay/import behavior unchanged
4. Update the persistent-session runner to use the neutral adapter where possible while
   preserving current RunPod behavior and artifacts.
5. Add docs and manifest fields that distinguish:
   - provider launch success
   - model/runtime success
   - valid downloadable output zip
   - generated MP4 extraction
   - visual usefulness
   - teardown/spend closure
6. Add an optional provider order env such as:
   `BLUEPRINT_WAM_COMPUTE_PROVIDER_ORDER=runpod,vast`
7. Keep existing command entrypoints (`blueprint-run-runpod-wam-async-runner`,
   `blueprint-run-vast-wam-async-runner`, `blueprint-run-oscar-wam-provider-command-adapter`)
   working while the new abstraction lands.

## Acceptance Criteria

- No paid provider launch occurs unless both explicit CLI flags and required env gates are set.
- `oscar_wam_provider_command_adapter` can run a no-spend blocked plan for both `--provider vast`
  and `--provider runpod` without throwing.
- Existing completed provider output import still accepts both `vast_provider_runtime_output.zip`
  and `runpod_provider_runtime_output.zip`.
- RunPod and Vast WAM manifests normalize into the same `WamComputeRunResult` contract.
- Invalid or zero-byte output zips remain `not_available` / `blocked`; they do not become
  completed outputs.
- `continuing_spend_from_this_run`, teardown status, provider phase, and output availability
  are all present in final manifests.
- Existing focused tests for RunPod, Vast, OSCAR provider adapter, and persistent session pass.
- If implementation touches shared orchestrator behavior, full `python -m pytest` passes before
  calling the work complete.

## Suggested Test Set

Run focused tests first:

```bash
python -m pytest \
  tests/test_wam_compute_providers.py \
  tests/test_oscar_wam_provider_command_adapter.py \
  tests/test_runpod_wam_async_runner.py \
  tests/test_vast_wam_async_runner.py \
  tests/test_unitree_groot_n17_sonic_vast_persistent_session.py \
  -q
```

Then run the full suite if shared behavior changed:

```bash
python -m pytest
```

## Proof Boundary

This work proves provider-package portability and WAM compute orchestration reliability. It
does not prove that a generated rollout is visually useful, that WAM output is capture truth,
that collision truth is validated, or that any physical robot is deployment-ready.
