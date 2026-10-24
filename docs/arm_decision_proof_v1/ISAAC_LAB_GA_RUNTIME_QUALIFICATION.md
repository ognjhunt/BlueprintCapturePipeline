# Isaac Lab 3.0 GA runtime qualification

ADP-009D Day-28 rehearsal / Day-35 simulation matrix. This opt-in experiment
compares the retained native workcell against the pinned beta runtime. Production
runtime defaults, PhysX, RTX, task inputs, controls and scoring remain fixed.
Neither Newton nor standalone rendering is included in that comparison.

## Separate native Newton/RTX smoke

The user selected one additional native feature test on 2026-10-24. This opt-in
ADP-009D / Day-35 compatibility experiment reuses the retained workcell and RTX
camera path with Newton's MuJoCo-Warp solver. The existing factory and launcher
hardwired PhysX; a sealed feature request now selects Newton explicitly at both
boundaries without changing the production default or the paired measurements.

Prepare the request and bundle with:

```bash
python -m blueprint_pipeline.native_task_runtime_comparison_cli prepare-newton-smoke \
  --packet-dir RETAINED_PACKET --reference-construction-request GA_REQUEST \
  --output newton-request.json
python -m blueprint_pipeline.native_task_arena_construction_bundle \
  --job-dir NEW_BUNDLE --packet-dir RETAINED_PACKET \
  --runtime-source-packet-receipt GA_SOURCE_RECEIPT \
  --implementation-commit IMMUTABLE_COMMIT --container-image PINNED_GA_IMAGE \
  --runtime-comparison-request-path GA_REQUEST \
  --native-feature-request-path newton-request.json
```

The canonical construction allocator, authority, watchdog, budget, TTL and
teardown still apply. The worker reads back the actual Newton manager, solver,
CUDA device and package versions, then checks 64 hold-position steps, finite
joint state, three cameras before/after, and a repeated seeded reset. It retains
joint drift and timing without assigning a task-success score. A native refusal
is retained with its phase and an explicit media gap where applicable.

The completion artifact is `native_task_newton_feature_result.v1.json`. The
construction transport envelope always retains `construction_gate_qualified:
false` and `qualification_effect: none`. Newton may interpret the retained USD's
PhysX properties differently; this separate smoke cannot establish physical
fidelity, a policy outcome or a comparable performance improvement. Native
physics stepping remains unqualified; source tests are preparation evidence only.

The first separate native attempt, Vast `57469156` on execution commit
`cad3e2f9162b4479db8236679dacb1a50d872315`, failed before scene construction.
Bootstrap passed in 29.006 seconds; the native Kit launch took 589.150 seconds.
Arena then refused `env_cfg_callback changed the physics backend away from
PhysX.` The adapter had changed the physics configuration in its callback while
leaving Arena's environment backend selection at its PhysX default. No Newton
physics step or camera observation occurred. The typed before-first-observation
gap and failure traceback are retained. Instance, allocator, watchdog and staged
objects are absent; canonical provider-zero and official billing passed at
$0.125. Fifteen evidence files were hash-verified locally and on the host.

This was a paid preflight process defect. The repair declares Newton before
Arena adapts the embodiment and preserves Arena's released Newton solver preset,
including its manipulation-specific manager and contact/constraint capacities.
Bundle preparation now replays the actual pinned vendor constructor, backend
selector, physics preset and callback guard without Kit, a GPU, geometry or a
provider call. The old behavior reproduces the exact native assertion; the
candidate selects Newton and enables the required physics replication. That
replay is a preparation artifact, not a native success claim.

The released Newton DROID adapter changes gravity compensation, collision and
gripper settings. The experiment records its resulting configuration separately
from the retained PhysX reference and leaves the source asset bytes unchanged.
It cannot borrow the primary pair's controls or configuration-equivalence claim.
The correctly selected rerun is recorded below.

The second separate attempt executed `be697a32edb81e3c5f361f3e17c21e5bd0663262`
on Vast `57473847`, RTX 3090 UUID
`GPU-dbe176f1-fb1e-bcb8-b17c-c923a025b448`, driver 580.173.02. Bootstrap passed
in 26.743 seconds and the native app launched in 146.550 seconds. The run
reported Newton 1.6.1, MuJoCo/MuJoCo-Warp 3.12.0, Warp 1.17.0 and Torch
2.11.0+cu130. During workcell construction, CoACD's manifold edge check was
followed by `std::logic_error: unexpected code path was hit`; the process
aborted with provider entrypoint exit code 134. It retained no observation
PNGs and never reached the requested hold-step/reset probe or backend readback.

The canonical outer process result is blocked. The feature JSON remains a
`running`/`build` checkpoint after the native abort and must not be interpreted
as a final result. All 15 evidence files, bootstrap logs and two native-log
snapshots were retained; the final evidence and the locally copied snapshots
were hash-verified. `newton-feature-observed-outcome-20261024.json` binds the
raw records and explicit before-first-observation media gap without modifying
the checkpoint. Canonical teardown/provider-zero passed, all instance,
allocator, watchdog and staging resources are absent, and official spend is
$0.048. This is a failed compatibility test for this scene/configuration; it
does not qualify Newton or alter the primary PhysX pair.

## Observed construction pair, 2026-10-24

The GA and baseline construction replays completed on the same physical RTX
4090, UUID `GPU-3b954a77-b580-0fc9-0149-d9990de42321`, driver 580.159.03.
Both executed immutable source `5265bdd80b58212a3f320a5c4cb8ebfb5716459c`
with the same retained workcell, controls, camera configuration and scoring.

| Measurement | Baseline | GA | Observed GA change |
| --- | ---: | ---: | ---: |
| Native app startup | 16.972 s | 212.904 s | +1154.44% |
| Environment build | 34.003 s | 41.543 s | +22.17% |
| Median step, 218 samples each | 167.576 ms | 167.143 ms | -0.26% |
| Median reset, 5 samples each | 73.857 ms | 67.719 ms | -8.31% |
| Sampled device-wide peak | 6145 MiB | 6411 MiB | +4.33% |

Both runs reached all ten motion targets over 150 action steps, passed all
three camera gates and reset replay, and retained 150 task-state samples. All
four measured contact-force channels remained at 0 N, with no collision,
containment or joint-limit failure. Both global-IK preflights solved six of ten
targets; actual motion reached all ten. No candidate policy was queried.

All 33 camera pairs have identical intrinsics, matching raw RGB shapes and
sources, and identical camera positions. All five recorded robot joint-position
arrays and reset seeds match exactly. The 44 original PNGs per runtime were
verified against their native snapshot receipts. RGB mean absolute differences
range from 6.43 to 11.06 on a 0–255 scale. Representative reset/approach views and
the largest-difference prealign/wrist pair have matching view geometry but
visible texture/noise differences. Their cause and policy impact remain
unresolved; the original robot rubber-material gap is retained.

This single construction pair does not establish repeatability or an overall
runtime improvement. Startup and build were slower and the observed memory peak
was higher under GA. The canonical comparator reports
`measured_pending_correctness_review`, with `improvement_qualified: false` and
`runtime_promotion_authorized: false`. Memory remains a sampled device-wide
lower-bound peak, including other processes.

The full comparison still requires both frozen learned policies on all ten
identical cells in each runtime: 40 episodes, complete policy-input frames,
digest-bound manifests, derived review videos, deterministic scores and contact
evidence. The separate Newton attempt failed before physics stepping;
standalone rendering was not executed.
Fresh capacity at 2026-10-24 04:29 UTC offered no qualifying US 48 GB policy GPU
within the unchanged resource and driver limits.

A subsequent read-only capacity audit found that the RT-candidate list omitted
the RTX 5880 Ada. NVIDIA documents 48 GB and third-generation RT cores at
https://www.nvidia.com/en-us/products/workstations/rtx-5880/. The hardware
registry now recognizes both provider spellings and caps advertised memory at
48 GB. This corrects hardware metadata; native qualification, driver,
architecture, geography and total-transfer-cost gates still apply. The observed
UK offer is not admitted by the existing US-only execution authorities.

For ADP-009D's Day-35 matrix, the observed US capacity blocker justifies an
opt-in private UK compute authority for the same admitted inputs. This is a
capacity preparation change, not a new evaluation or provider. Existing
authorities retain US placement. A UK internal canary must bind its exact
runtime-input digest, named authorizer and authorization reference, unchanged
private Vast disclosure scope, and an explicit machine constraint. Production
team evaluations cannot use it. The authority remains digest-bound; driver,
architecture, RT, VRAM, requested disk, total transfer cost, watchdog, output
admission and zero-retry gates remain mandatory. Source checks alone do not
establish transfer performance; the first observed UK attempt is recorded below.

At 2026-10-24 08:30 UTC, the repaired US preparation passed the complete
canonical billing, output-disk and allocator preflight. Its immediate dispatch
readback then found no offer on machine `117959`; dispatch stopped before
allocation or authority consumption. The 08:33 retained US capacity response
had no eligible US offer, and the prepared UK host was unavailable at 08:34.
Unsupported 595-series driver offers remain excluded.

That same response listed a 48 GB RTX 6000 Ada on machine `40508` in Estonia,
driver 580.178.04, at a quoted $0.8119/hour including 200 GB of disk. This is
provider metadata, not an observed GPU or a charge. The owner's approval for
the same retained comparison supports one pinned private Estonia preparation
under the unchanged $8 cap, $1/hour limit, 7200-second TTL and zero automatic
retries. The opt-in authority now accepts one country, either `GB` or `EE`,
with the same input digest, named authority, private Vast disclosure scope and
machine constraint. US remains the default; mixed and other countries remain
refused. Inputs, controls, scoring and publisher/provider restrictions do not
change. Fresh capacity, complete preflight, actual native execution and the
same observed GPU UUID for baseline are still required.

A baseline session authority can additionally require the GA session's observed
physical GPU UUID. Provider input preflight checks it before checkpoint
provisioning; the parent worker and each isolated cell recheck the one visible
UUID before starting cells or loading learned clients. Host static preflight
does not probe a GPU. A mismatch,
ambiguous device inventory or failed probe records a digest-bound refusal and
before-first-observation gap. It does not query a policy. This supplements the
existing comparison's requirement that every scenario and both runtimes share
the actual GPU; a machine ID alone can refer to several physical cards.

The first UK policy attempt, instance `57479820`, executed source
`bf9709aa010d8b71f2e9f270df1a87b00617dba4` on machine 49863. SSH readback
proved RTX 5880 Ada Generation, 49,140 MiB, driver 580.159.03, UUID
`GPU-1925818e-d48f-b7e1-8211-cef77c827d10`. Bootstrap passed in 28.2 seconds;
the exact source packet provisioned and immutable input preflight passed.
Before checkpoint provisioning, the entrypoint exited 86: GA's Python launcher
added an informational stdout line to a shell-captured control-mode value.
No policy was queried and no policy observation was produced. The original
fallback and secondary missing-server teardown diagnostics remain unchanged.
The derived evidence manifest records the before-first-observation gap.

The repair passes control mode through a file rather than launcher stdout,
retains the input-preflight receipt, and preserves typed prepolicy failures
without inventing a server teardown when no receipt exists. A CPU fixture
replay, bound to the retained provider log, reproduces exit 86 with the original
entrypoint and reaches the worker with the candidate. Five shell-entrypoint
checks cover both control paths and missing/invalid modes. These are software
boundary checks with source provisioning and servers stubbed, not simulator,
checkpoint-loading, policy-execution or runtime-improvement evidence. A native
rerun is required. All 24 terminal evidence files were copied and hash-verified;
provider-zero, exact staging cleanup and official $0.039 billing are closed.

Vast GA instance `57459128` and baseline instance `57462894` are confirmed
absent. Their allocators and watchdogs exited; staging cleanup and canonical
provider-zero passed. Official charges were $0.179 and $0.263. Official Vast
charges for all task attempts, including SimReady intake and the separate failed
Newton attempts and the first UK policy attempt, total $1.572.
Baseline attempt 1 allocated nothing and has a separate official $0 closeout.

Retained evidence is under `output/isaaclab_ga_20261023/` locally and beneath
`/mnt/blueprint-work/isaaclab-ga-adf55cb3f/output/` on the control-plane host.
The GA attempt 19 and baseline attempt 2 evidence directories each contain 14
hash-verified files. `construction-pair-20261024/` retains the comparison,
per-frame review, original extracted PNGs and a side-by-side figure. The
canonical comparison receipt is
`sha256:8197110ed86e342af9f96c7d0553ee812d7cda076f4908a1047dd94bc4ed84c1`.

Before these runs, 87 focused material/comparison tests and the 25 required
policy lifecycle/import-closure tests passed on the executed source commit.
The native run source and bundles did not change between the paired attempts.

The GA profile pins Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`,
Arena `b5708b72b21f37f74c8c66b1ee9cedff362a424b`, and the full NVIDIA Lab 3.0.0
image by digest in `native_task_isaaclab_ga.py`. The source archive verifies each
file, license and tree, plus twenty-one hashed Arena supplements from the released
lock. The image supplies Torch 2.11/CUDA 13 and Warp 1.17. Provisioning checks
actual versions and a CUDA operation before scene execution.

`native_task_runtime_comparison` instruments the existing worker without changing
its actions, observations or scoring. It records startup/build/step/reset timing,
device-wide GPU memory, actual reset joints, camera gates and episode/contact/media
receipts. Timing synchronizes CUDA for steps and resets. Memory is a sampled peak
lower bound. An exact GPU UUID is required for the paired timing comparison.

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

Attempt 57434567 loaded the graphics ICD, then Isaac's Python process segfaulted
before Kit emitted diagnostics. No workcell was downloaded or executed. The
allocator destroyed the instance; exact-object cleanup and provider absence were
verified, and the official charge was $0.051. The same-process library preflight
had changed Kit's native initialization conditions, so the probe now runs in a
separate interpreter before the unchanged Kit/Warp smoke. This removes that
interference; the retained log alone does not establish the segfault's cause.
Ten focused tests passed, including a subprocess regression that rejects driver
state leaking into Kit and verifies that an unavailable library prevents Kit
startup. Changed-file lint and shell syntax checks passed. A native rerun is
still required; no paired runtime or camera qualification is claimed.

Both bootstrap interpreters enable Python's fatal-error handler so a native
startup crash leaves its Python stack in the retained provider log. Eleven
focused tests pass, including a controlled child-process segmentation fault
with core dumps disabled that verifies the diagnostic reaches stderr. This
adds diagnostic evidence without changing the workcell, actions or scoring.
