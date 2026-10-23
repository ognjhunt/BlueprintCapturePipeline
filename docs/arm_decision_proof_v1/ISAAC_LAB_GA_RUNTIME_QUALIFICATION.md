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
On a host configuring `BLUEPRINT_GPU_SPEND_GUARD_OUTPUT_ROOT`, place
`--adp-job-dir` beneath that physical directory so the shared guard can discover
the live watchdog. Preflight and watchdog arming reject an outside path or a
symlinked guard root. Keep bundles and source in their existing retained paths.
A construction replay is a precursor; it does not qualify learned-policy
performance. Policy comparisons bind both frozen specs and identical resolved
scenario cells. Existing paused controls remain paused.

Compare retained measurement files with:

```bash
python -m blueprint_pipeline.native_task_runtime_comparison_cli compare \
  --baseline BASELINE_MEASUREMENT --candidate GA_MEASUREMENT --output comparison.json
python -m blueprint_pipeline.native_task_runtime_comparison_cli compare-set \
  --baseline-root BASELINE_POLICY_OUTPUT --candidate-root GA_POLICY_OUTPUT \
  --output policy-comparison.json
```

The comparator refuses changed inputs, different GPUs, missing measurements or
incomplete receipts. Numerical improvement alone does not authorize promotion:
camera, reset, contact, scoring and complete policy-media evidence need review.
Policy measurements retain the selected scenario and seed. Set comparison
requires every expected scenario exactly once and one GPU across the entire set.
After observing the GA policy machine, prepare the baseline with
`prepare-policy --allowed-machine-id MACHINE_ID`. The sealed session authority
constrains provider selection; invalid or altered constraints are refused. The
actual GPU UUID still has to match, because a machine can contain several GPUs.
Preparation creates and validates a unique watchdog-compatible resource name.
The 39 focused comparison, retained-input, lifecycle and import-closure tests
cover mismatched, missing and duplicated scenarios plus cross-scenario GPU drift.
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

When later object cleanup completes, retain a derived closeout without rewriting
the failed result with `python scripts/seal_native_task_arena_provider_zero.py
--authority AUTHORITY --result ORIGINAL_RESULT --post-cleanup-output-dir NEW_DIR`.
The command verifies the original failure, costs, cleanup, watchdog and teardown.

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

Attempt 57412520 passed all 78 native dependency imports and completed simulator
startup in 500.973 seconds, including 450 seconds of cold shader compilation.
Environment construction then exposed Arena GA's removed
`init_franka_arm_pose` event. No stepping, camera, reset, contact or episode result
was produced. Provider absence, watchdog completion and object cleanup were
verified; the official charge was $0.113.

The compatibility repair preserves the retained reset event and ordering. Source
comparison also found changed robot asset roots (6.0 to 6.2), an embedded stand,
end-effector frames, camera parents and gripper velocity. Comparison bundles now
carry the same original 6.0 robot USD and two texture files (28,091,506 bytes),
verify their hashes before allocation and use the retained configuration on both
runtimes. Normal runtime defaults remain unchanged. The measurement records the
concrete robot configuration and refuses a pair with missing or unequal readback.

The original USD references a missing `Rubber_Textured.mdl` (HTTP 404); this gap
is retained on both sides. `OmniPBR.mdl` remains supplied by the pinned renderer.
Camera review must establish whether the missing material affects task views;
there is no complete-appearance or runtime-improvement qualification yet. Asset
files belong beside each comparison request under `retained_droid_assets/`, with
the original relative layout. Bundling does not fetch or replace them.

Attempt 57418653 ended during provider container provisioning, before the
Blueprint worker started. Its official charge was $0.016 for disk and zero for
GPU execution. The instance and exact staged objects were confirmed absent.
The later-cleanup receipt exposed a canonical-digest bug: a derived result had
retained the original result digest. Recovery now validates the original digest
and seals the derived result separately, preserving the original bytes, failure
and costs. All 107 focused authority, CLI and spend-reconciliation tests passed.

The shared guard journal later established the cause of that provisioning exit:
it reaped instance 57418653 as unowned after its 480-second boot limit. The live
allocator and watchdog were outside its configured evidence root. This was an
ownership-discovery defect, not evidence of an Isaac or provider-host failure.
For the already running replacement, an additional real watchdog was started
under the configured root with the same exact instance, prefix and deadline;
the deployed guard's ownership checker recognized it. No guard limit or service
configuration was relaxed. The encoded path preflight passed 81 focused
watchdog and native allocator tests, including dry-run refusal before artifacts
or provider work and symlink-escape refusal. Future jobs use the registered root.

Attempt 57429691 passed all 78 dependency imports and measured simulator startup
at 173.414 seconds. Camera attachment then stopped at GA's changed render-product
interface. The compatibility reader now accepts GA's HydraTexture path while
retaining exact camera prim binding and rejecting conflicting identities.
Memory sampling also exposed host/container PID mismatch; both comparison sides
now use one visible GPU's device memory and require the worker to hold an NVIDIA
device. This includes renderer and other process allocations and is explicitly
labeled device-wide. The original incomplete measurement is preserved. No
stepping, reset, camera or episode result was completed; the official charge was
$0.054 and provider absence was verified through the existing recovery boundary.

Attempt 57432938 could enumerate CUDA devices but could not create a Vulkan
instance (`ERROR_INCOMPATIBLE_DRIVER`). Direct loading confirmed that its NVIDIA
graphics libraries, including the library named by its Vulkan ICD, were absent.
The CUDA-only bootstrap incorrectly continued to download the bundle. The exact
host was destroyed after retaining the diagnostics, before the workcell started.
This is a bootstrap preflight defect, not an Isaac comparison result. The smoke
now loads the selected NVIDIA Vulkan ICD library before starting Kit or fetching
the bundle. Missing libraries and software-only ICDs fail closed. Actual rendering
still requires the native camera checks; loading a library is not rendering proof.
Eight focused bootstrap/launch tests and changed-file lint passed for this repair.
