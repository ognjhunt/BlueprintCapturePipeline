# Isaac Lab 3.0 GA runtime qualification

ADP-009D Day-28 rehearsal / Day-35 simulation matrix. This opt-in experiment
compares the retained native workcell against the pinned beta runtime. Production
runtime defaults, PhysX, RTX, task inputs, controls and scoring remain fixed.
Neither Newton nor standalone rendering is included.

The GA profile pins Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`,
Arena `b5708b72b21f37f74c8c66b1ee9cedff362a424b`, and the full NVIDIA Lab 3.0.0
image by digest in `native_task_isaaclab_ga.py`. The source archive verifies each
file, license and tree, plus twenty hashed Arena supplements from the released
lock. The image supplies Torch 2.11/CUDA 13 and Warp 1.17. Provisioning checks
actual versions and a CUDA operation before scene execution.

`native_task_runtime_comparison` instruments the existing worker without changing
its actions, observations or scoring. It records startup/build/step/reset timing,
process GPU memory, actual reset joints, camera gates and episode/contact/media
receipts. Timing synchronizes CUDA for steps and resets. Memory is a sampled peak
lower bound. An exact GPU UUID is required for a causal timing comparison.

Prepare a construction precursor with:

```bash
python -m blueprint_pipeline.native_task_runtime_comparison_cli prepare-construction \
  --packet-dir RETAINED_PACKET --variant baseline --output baseline-request.json
python -m blueprint_pipeline.native_task_runtime_comparison_cli prepare-construction \
  --packet-dir RETAINED_PACKET --variant isaaclab_3_ga --output ga-request.json
```

Pass each request to the existing construction bundle builder using
`--runtime-comparison-request-path`, with its matching image and source packet.
Allocation still uses only `paid_resource_allocator gpu-canary`, a sealed
zero-retry authority, independent watchdog, hard budget/TTL and provider-zero.
A construction replay is a precursor; it does not qualify learned-policy
performance. Policy comparisons bind both frozen specs and identical resolved
scenario cells. Existing paused controls remain paused.

Compare retained measurement files with:

```bash
python -m blueprint_pipeline.native_task_runtime_comparison_cli compare \
  --baseline BASELINE_MEASUREMENT --candidate GA_MEASUREMENT --output comparison.json
```

The comparator refuses changed inputs, different GPUs, missing measurements or
incomplete receipts. Numerical improvement alone does not authorize promotion:
camera, reset, contact, scoring and complete policy-media evidence need review.
Actual GPU qualification results will be recorded here after execution.

Focused validation covers release identity, dependency tamper refusal, lifecycle
close semantics, unchanged runner values, exact frozen inputs and the sealed
provider import closure. Existing worker/bundle/lifecycle tests protect the
unchanged production path. No test result by itself establishes runtime uplift.
