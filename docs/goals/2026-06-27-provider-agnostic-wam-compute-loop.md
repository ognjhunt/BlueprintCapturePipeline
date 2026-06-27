# Provider-Agnostic WAM Compute Loop Goal

Date: 2026-06-27

Status: handoff prompt for next session

## Copy/Paste Goal Prompt

```text
/goal In /Users/nijelhunt_1/workspace/BlueprintCapturePipeline, implement the provider-agnostic WAM compute abstraction described in /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/superpowers/specs/2026-06-27-provider-agnostic-wam-compute-design.md and keep working in a loop until it is implemented, verified, or blocked by a concrete external provider condition.

Read first, in order:
1. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/AGENTS.md
2. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/PLATFORM_CONTEXT.md
3. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/WORLD_MODEL_STRATEGY_CONTEXT.md
4. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/README.md
5. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/pyproject.toml
6. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/superpowers/specs/2026-06-27-provider-agnostic-wam-compute-design.md

Before editing, inspect current state:
- git status --short --branch
- src/blueprint_pipeline/gpu_render_providers.py
- src/blueprint_pipeline/oscar_wam_provider_command_adapter.py
- src/blueprint_pipeline/vast_wam_async_runner.py
- src/blueprint_pipeline/runpod_wam_async_runner.py
- src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py
- tests/test_oscar_wam_provider_command_adapter.py
- tests/test_runpod_wam_async_runner.py
- tests/test_vast_wam_async_runner.py
- tests/test_unitree_groot_n17_sonic_vast_persistent_session.py

Also inspect the current Isaac OSCAR job state before launching or polling anything:
ISAAC_OSCAR_JOB=/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611/pipeline/isaac_oscar_wam_job
Check $ISAAC_OSCAR_JOB/wam_rollout_input_manifest.json, $ISAAC_OSCAR_JOB/wam_provider_output.json, and $ISAAC_OSCAR_JOB/provider_job/vast_provider_run/vast_wam_async_state.json if present. At the time of this handoff the exact Isaac artic3 WAM output was not complete: the prior output was blocked on vast_wam_provider_poll_blocked, and a newer Vast state showed instance_created for 42880112 with the phase log ending at vast_heartbeat_started. Do not claim a real OSCAR-generated MP4 until a valid extracted oscar_generated_rollout.mp4, final poll manifest, and teardown/spend state exist. If an instance is active, first do a bounded poll/resume using existing repo commands and report phase, output, teardown, and continuing_spend_from_this_run exactly; do not start another paid run unless the existing run is closed or blocked and explicit paid gates are present.

Implementation objective:
- Add src/blueprint_pipeline/wam_compute_providers.py with a provider-neutral WamComputeLaunchSpec, WamComputeProvider protocol/base, WamComputeRunResult normalization, get_wam_compute_provider(), list_wam_compute_providers(), VastWamComputeProvider, and RunPodWamComputeProvider.
- Wrap existing provider code rather than rewriting provider APIs from scratch: Vast should reuse create_async_vast_wam_run and poll_async_vast_wam_run; RunPod should reuse create_runpod_wam_async_run and poll_runpod_wam_async_run.
- Update src/blueprint_pipeline/oscar_wam_provider_command_adapter.py so fresh provider launches can select --provider vast|runpod|auto while preserving existing --mode auto, --mode replay-existing-provider-output, and --mode vast-provider compatibility.
- Preserve import/replay of completed provider jobs and support both vast_provider_runtime_output.zip and runpod_provider_runtime_output.zip.
- If practical in the same pass, update src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py to use the neutral adapter where it reduces duplication without breaking existing artifact paths.
- Add focused hermetic tests in tests/test_wam_compute_providers.py and update existing tests for the adapter/provider-selection contract.

Hard rules:
- No paid launch by default. Paid provider calls must require explicit CLI flags plus existing env gates.
- Never print raw API keys, signed URLs, secret env values, or token contents.
- Keep render-provider abstraction separate from WAM compute abstraction; reuse its pattern, not its exact contract.
- Keep WAM/generated-video success, forward/inverse episode consistency, review quality, capture truth, collision truth, deployment approval, safety validation, and physical robot readiness separate.
- Invalid, zero-byte, or nonterminal output zips must not count as completed generated-video artifacts.
- Every completed or blocked provider result must expose provider phase, output availability, teardown status, continuing_spend_from_this_run, and blockers.

Verification loop:
1. Run python -m pytest tests/test_wam_compute_providers.py -q as soon as the new module has tests.
2. Run:
   python -m pytest tests/test_oscar_wam_provider_command_adapter.py tests/test_runpod_wam_async_runner.py tests/test_vast_wam_async_runner.py tests/test_unitree_groot_n17_sonic_vast_persistent_session.py -q
3. Run python -m pytest if shared WAM/provider behavior changed broadly.
4. If any test fails, inspect the failure, patch the code, and rerun the relevant tests. Keep looping until green or until the remaining blocker is an external provider state that cannot be advanced locally.
5. Run git diff --check before final status.

Definition of done:
- The new provider-neutral WAM compute contract exists and is tested.
- OSCAR provider command adapter can select RunPod or Vast through the same provider-neutral path while preserving old Vast mode compatibility.
- Existing RunPod/Vast WAM behavior remains covered by focused tests.
- No-spend blocked paths are explicit and clean.
- Paid-provider artifacts, if inspected or resumed, report exact phase, output state, teardown/spend state, and claim boundary.
- Final answer names modified files, tests run, any active/closed provider state, and any remaining blocker. Do not stop at a plan or compile-clean-only state.
```

## Proof Boundary

This goal is about provider-package and WAM compute portability. It must not upgrade any
generated rollout into capture truth, visual usefulness, rank-fidelity truth, safety
validation, physical readiness, or deployment approval.
