# Isaac Lab 3.0 GA runtime qualification

ADP-009D Day-28 rehearsal / Day-35 simulation matrix. This opt-in experiment
compares the retained native workcell against the pinned beta runtime. Production
runtime defaults, PhysX, RTX, task inputs, controls and scoring remain fixed.
Neither Newton nor standalone rendering is included.

The GA profile pins Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`,
Arena `b5708b72b21f37f74c8c66b1ee9cedff362a424b`, and the full NVIDIA Lab 3.0.0
image by digest in `native_task_isaaclab_ga.py`. The source archive verifies each
file, license and tree, plus twenty-one hashed Arena supplements from the released
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

The first allocated GA attempt on 2026-10-23 (Vast instance 57403653)
terminated as `stopped_before_start`, before any simulator observation. The
provider posted $0.006 of disk cost and $0 of GPU cost. Teardown and later
exact-object cleanup confirmed absence. The original failed result is retained;
a digest-bound post-cleanup result preserves its blockers and nonzero cost.
This attempt provides no startup, memory, stepping, camera, reset, contact, or
episode measurements.

Readback of the pinned NVIDIA image configuration confirmed `User=isaaclab`.
The Vast SSH bootstrap requires root, so this exact image now requests
`user=root`; other images and launch modes retain their previous behavior.
Provider polling also retains redacted startup status text. This repairs an
observed launch-contract mismatch; the original provider log did not establish
that it caused the failure. Focused authority/CLI/launch tests passed (55), and
provider liveness/log-poll tests passed (29). A fresh admitted native attempt is
still required to qualify the runtime.

`native_task_runtime_comparison_cli prepare-policy` prepares the existing paired
Quick-10 harness from retained Pi0.5/GR00T execution specs, readiness, construction,
and control receipts. It requires the caller to name the human authorization to
reuse the frozen task. It verifies the retained model configuration against the
current pinned adapters, preserves the exact instruction and query limits, and
binds every cell to the original task digest. The prompt compatibility path is
opt-in and rejects any instruction that differs from the sealed base task.

Both variants bind the same task criteria, cells, models, and historical control
receipt. Failed historical controls remain failed diagnostic evidence; preparation
does not qualify a policy comparison, resume the paused controls stage, or create
a Website activation. Static preflight reports unresolved scenario/observation
gates before allocation. The observation gate verifies the same retained prompt
digest instead of regenerating a modern instruction. All 45 focused worker,
lifecycle, import-closure and retained-input tests passed; native policy
execution remains pending.

The next allocated attempt (57404798) completed the pinned GA provisioning and
native simulator startup, measured at 287.031 seconds including cold shader
compilation. It stopped before environment construction: the complete native
dependency matrix found `msgpack` absent. The image's source lock listed the
package, but that did not establish installation in its environment. This was a
preflight process defect. The source packet now carries the exact Arena-locked
`msgpack` 1.1.0 wheel and verifies its import/version before Kit starts.

The same run exposed two identical driver rows for one process/GPU memory
sample. The reader now deduplicates identical rows and still refuses conflicting
observations. The original measurement remains incomplete; no memory, stepping,
camera, reset, contact, or episode qualification is claimed. Official charges
were $0.169; teardown, provider absence and object cleanup were confirmed.
All 80 focused source, launch, measurement, lifecycle and import tests passed
after the repairs. The original task, source assets and controls remain frozen.
