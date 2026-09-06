# Arena Lane Handover — 2026-08-19

Hand this whole file to the next agent. It is written to be self-contained: you
should not need to read the prior transcript.

---

## 1. Mission and non-negotiable constraints

**Goal:** prove the Scene **840920 Task A** (`task_a_washer_door_open`) *native
task arena construction* lane end-to-end **on production GPUs**. Not locally,
not for this scene only — the fixes must be general.

**User's standing rules, verbatim:**

- "remember our solution needs to work in PROD - not just locally for this
  specific scene/tasks"
- "once we confirm a fix make sure that it is merged/goes to prod - nothing stays
  local"
- "we should A/B test a pod to find all possible issues"

**Repo doctrine (`AGENTS.md`, binding):** never resolve a failure by hand or by
one-off workaround. Every fix lands as **code on `main`** with a **hermetic
fast-lane test** pinning the contract and, where a paid path exists, a
**fail-closed gate**. Precedents: PR #180, #181.

**Program context:** Arm Decision Proof v1 is the sole active program. Humanoid,
deformable, world-model, multi-site and general-ranking work is frozen. Read
`docs/arm_decision_proof_v1/north_star_contract.json` and `AGENTS.md` before
making scope calls.

---

## 2. Where things stand

`origin/main` = **`0ea8fe333`** ("Measure which frame the controlled gripper body
is actually in (#804)").

All §7.1 line numbers are pinned at `714ba20d2` (three commits back); #803 and
#804 touched `native_task_camera_observability.py`,
`native_task_arena_construction_worker.py`, `native_franka_pose_servo.py`,
`native_franka_action_math.py` and their tests, so re-pin those four before
quoting a line from them.

Merged on 2026-08-19:

| PR | What |
|----|------|
| #783 | Bounded the sealed-camera Chromium install so one wedged shard can't hold the lane |
| #798 | Refuse an unauthored grasp orientation instead of executing 120° of it |
| #799 | Derive the grasp orientation instead of hard-coding a placeholder |
| #800 | Refuse an appearance volume composed outside the scene it depicts |
| #801 | Ship the appearance volume at identity instead of the exporter's axis matrix |
| #802 | Author the gripper approach axis the panel normal could not supply |
| #803 | Decide the camera gate from pixels, not from scene-graph membership |
| #804 | Measure which frame the controlled gripper body is actually in |

**Paid runs r13–r23 could never have passed.** Do not treat any of their
"passing" gates as evidence of anything. Total spend across the campaign was
about **$1.58**; nothing is currently billing (`vastai show instances` → zero
rows).

### ⚠️ First thing to do: the primary working tree is stale

`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline` is checked out at
`a9b28b176`, **503 commits behind `origin/main`**, and carries **pre-existing
uncommitted changes that are not part of this work** (a deleted
`scripts/materialize_semantic_teacher_image_edit_closeout.py`, modifications to
`materializer_cli.py`, `policy_ranking_evidence.py`,
`public_scene_suite_materializer.py`, `adp_content_agents_candidate_comparison.py`
and their tests, plus untracked
`scripts/seal_content_agents_candidate_comparison.py` and
`tests/test_content_agents_candidate_comparison_cli.py`).

**Do not discard those** — they predate this session and belong to a different
lane. Stash them or work in a fresh worktree off `origin/main`. A plain
`git merge --ff-only` aborts because of them.

---

## 3. The defects — what was actually wrong

Four independent causes. Two are fixed, three are open. All of them were
invisible to the lane's own gates, which is the meta-problem (§6).

### 3.1 ✅ FIXED (#802) — the grasp frame was mathematically degenerate

`paired_target_interaction_affordance_candidate.py` set **both**
`pinch_axis = normal_world` **and** `approach = normal_world` — the same vector.
So `ee_x = ee_y × ee_z = (0,0,0)`; no hand orientation existed. The lane fell
back on an identity placeholder, which — read in Isaac Lab's `wxyz` order — is a
**180° yaw**, and sat **120° from the Franka reset pose** against a **0.08 rad**
arrival tolerance. No phase could ever have passed.

The fix was not a choice. The producer *already computed* `radial` (hinge →
panel centre) to place the contact at the free edge, then **discarded it**. The
gripper approach is its negative, now emitted as
`gripper_approach_axis_registered_stage`.

Authored value: `[-0.7071067811865476, 0.0, 0.7071067811865476, 0.0]`,
`deg_from_reset = 180.000`.

Reachability was verified on CPU against the committed r22/r23 packet: all **31**
commanded phases solve on **one continuous IK branch** (worst residual 9.3e-10 m),
no solution-family alternation of the kind r22/r23 measured; joint 4 sits
**+0.0088 rad** inside its limit at the grasp (pinned in a test). Repeated at
tool offsets 0.10 / 0.16 / 0.22 m — all reach.

### 3.2 ✅ FIXED (#801) — the appearance volume was spawned outside the room

`scene_appearance.usdz` carried 3DGRUT's exporter matrix

```
((-1,0,0,0), (0,0,-1,0), (0,-1,0,0), (0,0,0,1))    # p -> (-x, -z, -y)
```

applied to gaussian centres that were **already in metric registered-stage
coordinates**. The volume landed ~**13 m** away, mirrored, below the floor. Every
camera pointed at correctly-built geometry inside an empty void.

Two traps inside this one:

- `apply_normalizing_transform = False` in
  `scripts/public_scene_artifixer3d_runner.py` was **the wrong lever** — it
  suppresses a recenter baked into the **point data**; the axis matrix is authored
  separately into the **USD layer**, which no exporter setting touches. Since the
  matrix is an involution, turning the flag off leaves exactly the mirror.
- `public_scene_artifixer3d_vast.py` **hard-required the defective matrix**, so
  fixing only the producer would have made a *correct* export fail closed. Both
  are fixed.

#800's spurious-matrix rule only fired *after* containment already failed, so it
never independently refused anything — a volume symmetric about the mapping's
fixed point stays "inside" while mirrored. It is now judged on the linear part
alone.

### 3.3 ⛔ OPEN, AND THE REAL BLOCKER — the arena IMAGE lacks NuRec

> **Correction, read this first.** An earlier version of this section said the
> pinned *Kit* declares no NuRec extension. **That was wrong**, and so was the
> GPU probe it rested on. The conclusion — the image lacks NuRec — survives and is
> now proven far better, for **$0**. Details below so nobody re-runs the bad probe.

#### What the first probe got wrong

1. **`omni.nurec`, `omni.rtx.nre` and `omni.usd.schema.omni_nurec` are not real
   extension IDs.** No NVIDIA source publishes them. The real ones are
   **`omni.rtx.spg`** and **`isaacsim.replicator.nurec_utils`**; `omni.rtx.nre`
   exists only as a *carb settings namespace*. "All three fail to enable" measured
   the absence of things that never existed, and the grep
   `nre|nurec|gsplat|ujitso` **cannot match `omni.rtx.spg` at all.**
2. **It ran the wrong Kit** — `/isaac-sim/apps/isaacsim.exp.base.python.kit`,
   Isaac Sim's own base experience, **not the lane's pinned compat Kit**. That is
   why `rendererHints` was null. The pinned compat tree (`a4abce12`)
   `apps/isaaclab.python.headless.rendering.kit` declares `# NuRec` /
   `omni.rtx.nre.compositing.rendererHints = 3` at **lines 99-100**, plus
   `app.useFabricSceneDelegate = true` and `renderer.multiGpu.enabled = false`.
   **The pinned Kit is the one component in the stack that already configures
   NuRec correctly.**

The positive control (lit cube **237.5881** vs **237.5906** with the volume added)
is still valid evidence that nothing rendered *on that configuration* — it just
was not the lane's configuration. Instance 48161680, $0.062, destroyed.

#### What is actually true — proven for $0

Authenticated to nvcr.io with the existing NGC key and stream-listed each image's
large layer with `crane`. No GPU, no pod, nothing written to disk:

| Image | `omni.rtx.spg` | `isaacsim.replicator.nurec_utils` |
|---|---|---|
| `6.0.0-dev2@c3e7bef5` (pinned arena) | **absent** | **absent** |
| `6.0.0` GA | **absent** | **absent** |
| `6.0.1@b1c542b2` | `omni.rtx.spg-0.2.1+f9bf0dda.lx64.r` | present (+3 render tests) |

**"6.0 GA documents it" is refuted — GA does not ship it.** Full published tag
list: `4.0.0 4.1.0 4.2.0 4.5.0 5.0.0 5.1.0 6.0.0 6.0.0-dev2 6.0.0-dev3 6.0.1`.
**6.0.1 is the only candidate** (`6.0.0-dev3` unscanned but predates GA, which
lacks it). NVIDIA's `rendering_setup.py` requires `omni.rtx.spg >= 0.2.0` and Kit
>= 110.1.2; dev2 ships Kit 110.0.0, GA 110.1.1.

**Consequence stands: fixing alignment is necessary and not sufficient.** Every
arena run on the current image renders the robot and SimReady asset against a
black captured scene regardless of alignment.

#### Runtime provisioning — confirmed, so this is a swap

`native_task_runtime_source_packet.py:1-8` states it outright: *"The qualified
Isaac Sim image is intentionally treated as a simulator base, not as evidence that
companion Python packages are installed."* Sources install from exact git objects
via `native_task_runtime_source_provision.py`. Pins: `ISAACLAB_COMMIT e57379c6`,
`ARENA_COMMIT 8b4a3a47`, `ISAACLAB_RUNTIME_COMPATIBILITY_COMMIT a4abce12`.

#### Recommendation

Swap the arena lane to **`6.0.1@b1c542b2`** **and** pass **`--enable
omni.rtx.spg`**. Both are required — neither pinned Kit declares `spg` as a
dependency, and NVIDIA's gate checks exactly two things: `omni.rtx.spg` enabled
and `/renderer/multiGpu/enabled=false` (the compat Kit already does the latter).

6.0.1 is **already in this repo**: `public_scene_simready_isaac_bundle.DEFAULT_IMAGE`
(`@783444c7`, multi-arch index) and `adp_isaac_lab_arena_vast.DEFAULT_IMAGE`
(`@b1c542b2`, its amd64 manifest) are the same image, already booting Isaac and
running PhysX probes.

#### ⚠️ Do NOT edit the shared `DEFAULT_IMAGE`

`adp009d_native_microcheck_bundle.DEFAULT_IMAGE` is consumed by **seven modules**,
including `adp009d_franka_vast.py` (the working ADP-009D microcheck lane) and
`paired_target_native_import_bundle` / `_vast.py` (**the IMPORT run the arena
authority chains off**). Editing it in place moves working lanes onto an
unvalidated image. There are also **four independent hardcoded digest copies** —
`adp009d_agent_skill_audit.py:243` is a hard-equality check that *rejects*
substitution, plus `adp009d_physics_backend_comparison.py:32` and
`articulated_native_diagnostic_bundle.py:30` — and four retained doc/manifest
pins, one reading *"do not substitute a rolling main image."*
**The correct shape is a separate arena image constant.**

#### The unvalidated assumption that blocks landing it

**Nothing establishes that IsaacLab `e57379c6` + Arena `8b4a3a47` run on 6.0.1** —
they were pinned against a dev2-era Isaac Sim. This is exactly the
silent-reintroduction class: the Warp 1.12.0 contract in
`native_task_isaaclab_launch.verify_*` would catch bundled-Warp drift, but
**device coherence would not self-report**. The prior cuda/cpu failure came from a
bare `SimulationApp` instead of `AppLauncher`, and nothing warned. **"It boots" is
not evidence the swap worked.**

#### The one pod that discriminates (~15 min, Ada-class; a 4090 also sidesteps an unresolved "Ampere may not be supported" thread on the `nrend` error)

On `6.0.1@b1c542b2`, launched via the lane's own `launch_native_task_isaaclab`
(AppLauncher, compat experience, `device="cuda:0"`), with
`--enable omni.rtx.spg --/renderer/multiGpu/enabled=false`:

1. Install the runtime-source packet; assert `import isaaclab_arena` succeeds on
   6.0.1. **If this fails, the answer is a revision bump, not an image problem —
   and you learn it before spending on step 5.**
2. Assert `omni.rtx.spg` is enabled and
   `/omni/rtx/nre/compositing/rendererHints == 3`.
3. Readback: every articulation on `cuda:0` (the #769 regression check).
4. Schema check with the **correct** API — the last probe died on
   `AttributeError: 'SchemaRegistry' object has no attribute 'GetPrimDefinition'`.
   Use `Usd.SchemaRegistry().FindConcretePrimDefinition("OmniNuRecFieldAsset")`.
5. Keep the positive control: lit cube alone vs cube + NuRec volume, 320×180,
   report `rgb_mean` for both. Last time the delta was 237.5881 → 237.5906; a real
   200k-gaussian room must move it far more.

#### Do NOT fall back to ParticleField

It never rendered across ~12 attempts; settled. **But note for planning:** NVIDIA
staff state on the forums that *NuRec will be deprecated and replaced by
ParticleField*, and a 5.1-container report matches our exact symptom
(ParticleField black, nurec renders). Long-term asset-format risk, not a fix for
r24.

#### Side finding, unrelated to the image

`native_task_arena_policy_bundle.py` has **no load-time `container_image`
verification**, unlike construction (`:131`) and controls (`:138`). The policy
lane relies solely on the shape-only digest check, which accepts any registry.

### 3.4 ✅ FIXED (#803) — the camera gate could not see black frames

`src/blueprint_pipeline/native_task_camera_observability.py` decides pass/fail
purely from a **semantic segmentation mask**:

```python
passed = (count >= minimum_pixels and fraction >= minimum_pixel_fraction and centroid_framed)
```

and self-reports, at **line 102**:

```python
"rgb_or_model_label_used": False,
```

Segmentation masks are populated from **scene-graph membership**, not radiance —
they come back fully populated whether or not a single photon was rendered. This
is why eleven paid runs reported **PASS** against frames that were **88–92%
black**. Nothing acted on that `False`.

**How #803 resolved it.** RGB is now a required keyword with no default; an
absent frame is a structured refusal; `passed` is a conjunction; schema is `.v2`
with every v1 field preserved. Radiance is split into **three separately named
verdicts**, because *nothing rendered* and *this content did not render* are
different claims:

- `frame_rendered` — not void, not one repeated value. **Gated always.**
- `target_rendered` — the pixels the mask calls the target actually carry an
  image. **Gated always.** This is the direct repair: the mask can no longer
  vouch for pixels that show nothing.
- `site_rendered` — the frame outside the target is not mostly void. **Measured
  always, gated only against an explicit declaration the caller cannot omit.**

That last one is conditional precisely because of §3.3: the captured site is
*legitimately* black today. Gating it unconditionally would fail every run for a
cause the module cannot name; leaving it unmeasured is how r13–r23 happened. So
the worker declares `SITE_APPEARANCE_RENDER_EXPECTED = False` with the GPU
evidence in the comment, and each gate records `site_appearance_claimed: false`,
`site_void_pixel_fraction`, and
`claim: camera_observes_task_object_without_site_appearance`.

**Reciprocal guard — this is the part that matters after the image is fixed:** a
run whose site renders anyway emits
`native_task_camera_site_rendered_while_unclaimed`. So swapping in a NuRec-capable
image **cannot** leave the declaration stale and unnoticed. When you fix §3.3,
expect this refusal and flip `SITE_APPEARANCE_RENDER_EXPECTED` to `True`.

A passing gate today means "the object is framed and its pixels rendered" — **not**
"the frames show the site". Do not quote it as the latter.

**Threshold provenance:** derived from 17 real renders from this Isaac RTX stack
under `output/` (gitignored, so tests pin the derived numbers and stay hermetic):
void `0.00000–0.01281`, luminance std `19.59–69.24`, `196–256` distinct levels.
Synthetically rebuilding the r13–r23 signature gives void `0.880/0.900/0.921` at
std `47–58` over `222–229` levels — **proving variance and tonal range cannot see
that failure and void fraction is the only statistic that can.** Ceiling `0.50` is
39× above the darkest real render and 38 points below the shallowest failure, and
matches `isaac_review_renderer_canary.SEVERE_CLIPPING_MIN_FRACTION`. 16 of the 17
real frames pass the finished gate; the one that fails is genuinely a single RGB
value.

Two ordering fixes came with it: the frame PNG is written **before** measurement
so a refusal retains its evidence, and `best_observability` ranks on
`(passed, pixel_count)` so the gate cannot cite a black frame as its best
evidence.

Call site verified by test with a fake Isaac camera, not by reading:
`_camera_snapshot` at `native_task_arena_construction_worker.py:739` (measuring at
`:780`), the only producer of `camera_gates`, called from `:1251` and `:1407`.

### 3.5 ✅ INSTRUMENTED (#804) — the coupler contradiction, now measurable

Three sealed artifacts **pairwise contradict** on what frame the controlled body
is in. Any two can be true; all three cannot:

1. The DROID asset has **no `panda_hand`** — `native_task_robot_contact_topology.py`
   lists `panda_link0..7` plus Robotiq bodies, so the controlled body is
   `base_link` (= `robotiq_85_base_link`; URDF not vendored). But the
   `+Z_ee`=approach / `+Y_ee`=jaw convention that #799 derived and #802 authored
   against is **`panda_hand`'s**.
2. At the sealed reset joints the repo's own FK puts the flange `+Z` **straight
   down** (matching `APPROACH_TOOL_QUAT_XYZW`). The measured reset body quaternion
   `(0.5, 0.5, 0.5, 0.5)` has no axis down except `-Y` → approach would be
   `-Y_body`. Implied coupler is exactly **`Rx(-90°)`** — clean enough to look
   like a real measurement, not noise.
3. The sealed wrist-camera extrinsic in `base_link` has its optical axis at
   `(0.9498, -0.3130, 0.0022)` ≈ **`+X_body`**.

**If the convention is wrong for this body, every authored quaternion — including
#802's — is off by the coupler rotation.** The sealed artifacts cannot settle it,
because they are the things that disagree.

**Resolution is a logging change, not an experiment.** Record at reset:
`base_link`'s world quaternion **plus the world positions of the two
`*_inner_finger` bodies**. The jaw axis is the direction between the fingers;
approach is fixed relative to it — no convention needed. The servo
(`native_franka_pose_servo.py`, or its caller in
`native_task_arena_construction_worker.py`) **already reads both buffers**; only
the finger *norm* is recorded and the direction is discarded.

**#804 landed this.** `gripper_frame_axis_readback()`, schema
`native_franka_gripper_frame_readback.v1`, retained at the sealed reset *before
any phase moves the arm*, by the construction, controls **and** policy workers.
Field path:
`native_task_arena_construction_result.v1.json → gripper_frame_axis_readback →
{measured, derived, assessment}`. `measured` holds the raw body pose and both
finger world positions; a test recomputes every derived axis from `measured`
alone, so the assessment sits *beside* the numbers rather than replacing them.

**Reading the next run is arithmetic, not judgement.** At the measured reset
quaternion `(0.5,0.5,0.5,0.5)` the body axes are a cyclic permutation of world
axes, so the three hypotheses predict three **mutually orthogonal** world
directions (orthogonality itself pinned by a test):

| `approach_unit_body` | fingers sit, in world | verdict |
|---|---|---|
| `[0,0,1]` (`+z`) | along world `+X` | convention holds; **#802 correct as landed** |
| `[0,-1,0]` (`-y`) | **straight down** | coupler real; every authored quaternion off by it |
| `[1,0,0]` (`+x`) | along world `+Y` | the camera-extrinsic axis is the approach axis |
| anything else | — | `none_within_tolerance`; raw numbers still derive the truth |

A hypothesis is named only when exactly one is within the declared 0.25 rad
tolerance.

**Judgement call worth preserving:** the jaw *sign* is recorded as a label, not a
measurement (`jaw_axis_sign_is_a_label_not_a_measurement: true` beside
`jaw_axis_ordering`). The two fingers are interchangeable, so the jaw is a line,
not an arrow — claiming its direction was measured would be the same class of
mistake this PR exists to stop.

Refusals, all fail-closed with no default axis:
`native_franka_pose_servo_finger_body_missing:<name>` (pre-existing, already
pinned), `..._gripper_frame_position_invalid:<label>`,
`..._gripper_frame_jaw_degenerate`, `..._gripper_frame_approach_degenerate`,
`..._quaternion_invalid`.

Where the answer was being discarded, at `714ba20d2`:
`native_franka_pose_servo.py:257` averages both finger poses to a midpoint and
drops the difference; `native_task_arena_construction_worker.py:369-373` computes
the difference then immediately reduces it to a norm;
`native_franka_action_math.py:281-286` computes the approach lever arm in body
coordinates **every tick** and never records it.

---

## 4. Do this next, in this order

1. **Base image with NuRec (§3.3) — investigation is DONE, execution is not.**
   The answer is `6.0.1@b1c542b2` plus `--enable omni.rtx.spg`, landed as a
   **separate arena image constant** (never an edit to the shared
   `DEFAULT_IMAGE`). Run the five-step discriminating pod in §3.3 first — step 1
   alone tells you whether IsaacLab `e57379c6` + Arena `8b4a3a47` even import on
   6.0.1, which is the assumption the whole swap rests on.
2. **Finger-direction readback (§3.5).** Small, CPU-only. Makes the next run
   settle the coupler question instead of leaving it open for another eleven runs.
   A branch may already exist — check `claude/arena-gripper-frame-readback-20260819`
   before starting from scratch.
3. **When you fix the image (step 1), flip `SITE_APPEARANCE_RENDER_EXPECTED` to
   `True`** in the construction worker. #803's reciprocal guard
   (`native_task_camera_site_rendered_while_unclaimed`) will refuse the run until
   you do, which is intended — it is what stops a swapped image from leaving the
   declaration stale.
4. **Fire r24** — only after 1–3, and only after the pre-flight in §5 passes
   against the **built artifact**.

Still parked, lower priority: `task_b` carries the identical placeholder in
`gripper_orientation_scoring_frame_xyzw` (load-bearing, and its axes *are*
independent, so it is derivable); pi05
`license.allowed_use = not_admitted_until_checkpoint_specific_terms_are_bound`;
`data_origin_invalid` tamper gate not raising on clean main.

---

## 5. Mechanics — exact commands

Everything below is on `origin/main` under `scripts/`.

### Pre-flight (run BEFORE any GPU spend, against the BUILT artifact)

**r19 was a wasted paid run** because the fix was merged and deployed but the
packet **hardlinked an older copy forward**. Verify the value in the artifact,
never in the source tree.

```bash
python - 'arena-launch-<RUN>/arena_packet/*/native_task_runtime_contract.v1.json' <<'PY'
import json, math, sys, glob
hits = sorted(glob.glob(sys.argv[1]))
assert hits, "no native_task_runtime_contract.v1.json under that glob"
reset = [0.5, 0.5, 0.5, 0.5]
bad = 0
for path in hits:
    q = json.load(open(path))["task_spec"]["interaction_affordance"]["gripper_orientation_contact_xyzw"]
    identity = all(abs(v) <= 1e-6 for v in q[:3]) and abs(abs(q[3]) - 1.0) <= 1e-6
    deg = math.degrees(2 * math.acos(min(1.0, abs(sum(a*b for a, b in zip(q, reset))))))
    print(f"{path}\n  gripper_orientation_contact_xyzw={q}\n  unauthored_identity={identity}  deg_from_reset={deg:.3f}")
    bad += identity
sys.exit(1 if bad else 0)
PY
```

Verified both ways: on the committed r22/r23 packet it prints
`unauthored_identity=True  deg_from_reset=120.000` and exits **1**; with the
authored value, `unauthored_identity=False  deg_from_reset=180.000`, exit **0**.

The appearance equivalent — assert the layer transform in the **packaged**
`scene_appearance.usdz` is identity — is in #801's PR body and commit message.

### Iteration deploy (~3 min instead of ~21)

```bash
SHA=<40-char commit already on origin/main> bash scripts/deploy_control_plane_iteration.sh
```

Skips the ~15-minute Full Test Lane. Stamps `promotion_eligible=false`; every run
from it records `evidence_grade_ceiling=development_only`. **It fails closed if
the SHA is not an ancestor of `origin/main`** — the guard lives in
`scripts/deploy_control_plane_commit.py`, not just the wrapper, because within an
hour of the wrapper being written its `git fetch` hit a permission error and the
obvious workaround was to call the tool directly. Promote with the normal
lane-verified deploy (`--release-provenance`) before sealing evidence.

Note the wrapper fetches `origin main` **specifically** — a bare `git fetch
origin` fails with "Permission denied" creating ref locks for unrelated branches.

### Launch chain

```bash
bash scripts/arena_construction_launch_chain.sh   # 9 steps: build -> stage -> authorize
bash scripts/arena_construction_fire.sh           # submit
```

Step 2b (#788) refuses a **stale packet** that disagrees with deployed constants.
Steps 1 and 4 walk back past non-allocating runs when picking the spend
predecessor. The execute gate is armed **before** submission.

Supporting scripts on main: `build_native_task_arena_live_profile.py`,
`build_arena_native_control_live_profile.py`,
`issue_native_task_arena_paid_attempt_authority.py`,
`seal_native_task_arena_provider_zero.py`,
`seal_native_task_arena_recovered_provider_zero.py`.

### Test lanes

```bash
python -m blueprint_pipeline.impacted_test_selection   # changed tests + sentinels, hard-capped 120s
ruff check <changed files>                             # default build loop
scripts/pytest_fast.sh                                 # bounded integration diagnostic
scripts/pytest_full.sh                                 # promotion / scheduled / cross-cutting only
```

Bare `pytest` deselects `slow`/`gpu` but has **no guaranteed wall-time** — it is
not the default build-loop or ordinary-PR gate.

---

## 6. Traps that have each cost real money or a wasted run

**Conventions**
- Contracts are **xyzw**; Isaac Lab is **wxyz**. Identity xyzw `[0,0,0,1]` read as
  wxyz is `w=0, z=1` — a **180° yaw**. Every one of these seams fails *silently*.
- **DO NOT** convert the camera offset to wxyz "by symmetry" with the robot spawn.
  PR #775 did; it blinded both world cameras (external 21871 → 0, overview
  9053 → 0). Reverted in #785 with a test pinning the measured pixel evidence.
  The comment in `native_task_arena_runtime.py` says so — believe it.

**Artifacts and deploys**
- The packet is **hardlinked forward**; a merged-and-deployed fix can still not be
  in the artifact (r19). Always pre-flight the built artifact.
- **Every deploy invalidates prepared bundles / preflights / profiles** — rebuild
  them *after* deploying, not before.
- A deploy is **two surfaces**. `git archive` strips `.git` and empties the
  identity probe. Use `scripts/deploy_control_plane_commit.py`.
- Render's `update_in_progress` deploys silently do not apply env vars.

**Provider / pods**
- Vast `gpu_ram` is in **MEGABYTES**. A GB value is a silent no-op that returns
  junk offers.
- "No offers" is usually the **query**, not the market — constraints filtered
  after a bounded page starve allocation.
- `success: False` from create can still return a contract id with
  `intended_status: stopped` — that machine had no capacity. Verify the instance
  reaches `running` before waiting on logs.
- **Terminate explicitly.** An EXITED instance still bills for disk.
- SSH key propagation is **machine-dependent**, not account-wide — it failed on
  two machines and worked fine on 48161680. `vastai create --onstart <script>` +
  `vastai logs` is the reliable path; SSH is a bonus when it works.

**Repo**
- Squash merges erase the parentage git needs to see that long branches agree —
  N branches on one hunk re-conflict. Push the byte-identical union to all N.
- A worktree's pytest reads the **worktree's** `src`, but a bare `python -c` reads
  the **main tree's** editable install.
- 380+ branches "ahead of main" are mostly squash-landed no-ops. Trust only
  merge-then-diff-vs-main.
- There are **115 git worktrees** on this machine. `scripts/agent_workspace_gc.py`
  reaps stale agent scratch clones (dry-run by default; delete needs
  `--apply --ack reap-agent-scratch`). Agent clones have filled the disk before
  (~40 GB in 6 days).

---

## 7. The meta-problem — read this before adding any gate

Every one of today's findings had the same shape: **a sealed artifact asserting
something nobody measured.**

- The camera gate claimed visibility it never checked (`rgb_or_model_label_used:
  False`).
- The receipt *declared* the exporter transform as a hardcoded constant no code
  had ever read from the asset.
- The grasp convention was inherited from a hand (`panda_hand`) the robot does not
  have.
- #800's own spurious-matrix rule could never fire independently.

This is systemic, not three unrelated bugs. When you add or touch a gate, the
test that matters is: *what would this report if the thing it describes were
entirely absent?* If the answer is "pass", the gate is decorative.

**Three more of the same shape, found by #803, named but deliberately not fixed
there.** Each still needs work:

- `isaac_review_renderer_canary.py:191` and `:194` —
  `review_frame_unitree_g1_not_visible` / `review_frame_target_marker_missing`
  decided from `_semantic_label_fraction` at `:662-664`. Partly mitigated (the
  module does read RGB at `:182-189`), but its
  `BLANK_BLACK_MAX_MEAN_LUMINANCE = 2.0` would **not** have fired on an r13–r23
  frame either.
- `adp009d_isaac_runtime.py:2040-2049` — `quality_diagnostics` writes
  `rgb_min/max/mean` plus `foreground_semantic_pixel_fraction` and **nothing gates
  any of them**. The only references outside the writer are two tests asserting
  the field names exist, one by substring-matching the source.
- `lightwheel_sink_isaac_worker.py:890` — `nonzero_value_count > 0` does read
  pixels, but a **single** non-black pixel satisfies it.

### 7.1 A dedicated audit found ELEVEN more. This is bigger than the arena lane.

Ranked by severity. Line numbers pinned against `714ba20d2`. High confidence on
1–6 and 8 (producer *and* consumer traced); medium on 7, 9, 11; 10 is
high-confidence as a shape but low severity because the validator is unreached.

**These gate PAID paths — treat as highest priority:**

1. `adaptive_task_stance_configurator.py:344` — `render_evidence_fresh` claims
   (module docstring) "fresh robot-POV and third-person PNG render evidence from a
   real Isaac RGB render". `render_ok` at `:336` only tests that two path strings
   are non-empty existing files and that a **caller-supplied** `render_source`
   string equals `"isaac_rtx_rgb"`. **It never opens either PNG and never checks
   freshness.** Feeds `all_gates_passed` (`:357`), which accepts the stance
   candidate at `:599` inside the paid Isaac G1 kitchen parity eval.
2. `adaptive_task_stance_configurator.py:304` — `affordance_visibility` thresholds
   `affordance_visible_fraction` at 0.6, but the only producer
   (`scripts/run_isaac_g1_kitchen_parity_eval.py:7399`) sets it to
   `1.0 if target_visible else 0.0` from `bool(geometry["target_in_frame"])`. An
   occluded or wholly unrendered affordance scores a full **1.0**.
6. `kitchen_task_scaling_preflight.py:548` — launch preflight gate labelled
   "target visible in manipulation POV" whose value is
   `bool(geometry.get("target_in_frame"))`; sibling at `:553` likewise. Both are
   projection membership, and the same projection report gates **paid** placement
   validation at `run_isaac_g1_kitchen_parity_eval.py:11939`.

**Hardcoded verdicts:**

5. `isaac_task_review_renderer.py:1386` — `framing_validation` written with a
   hardcoded `"status": "PASS"` from a pure pinhole projection, no depth buffer or
   occlusion test. Stays "PASS" even when `all_required_points_in_frame` is False,
   because the `raise` above only fires when `required_in_frame` is True — which
   for `robot_pov` is False until the mount is calibrated.
   **Corroborated:** `all_required_points_in_frame` (`:1389`) and
   `task_target_required_in_frame_this_update` (`:1390`) have **zero readers
   anywhere, tests included**. So the block can be stamped `"PASS"` with the
   in-frame check False and *nothing in the repo, production or test, would
   notice.*
8. `isaac_task_review_renderer.py:809` — prewarm evidence hardcodes
   `"status": "passed"` and `"render_products_realized": True` (`:818`), and its
   claim_boundary asserts the render products "returned live RGB". The only
   preceding test is `data.ndim == 3` plus a resolution match. **An all-black
   annotator buffer satisfies it identically.**

**Projection or segmentation standing in for pixels:**

3. `isaac_runtime_task_backend.py:2930` — `active_forearm_visibility_passed` is
   just `not missing_active_arm_links` (FK poses projecting inside the image
   rectangle), while the same receipt declares
   `active_forearm_visibility_required_for_policy_observation: True` at `:2943`.
   No production code reads either field (only
   `tests/test_isaac_persistent_task_executor.py:695`).
4. `g1_render_noise_audit.py:584` / `:586` — `both_arms_visible` and
   `target_visible` lifted from `left_arm_visible` / `target_in_frame`, whose own
   producer's `claim_boundary` says it is "not pixel-level visual proof unless the
   per-variant robot pixel mask agrees". **This module contains no pixel-mask read
   at all**, so the disclosure is inert.
7. `adp009d_live_hybrid_frames.py:462` — `approved_task_object_present` and the
   `hybrid_observation_approved_task_object_absent` blocker are decided from
   `dynamic_segmentation` label pixels, while the composed RGB written beside them
   is hashed into the receipt (`:497`) and **never measured for radiance**. A black
   composite with a fully-populated mask passes. (This is the r13–r23 bug, in a
   second module.)
9. `wam_isaac_evaluation_hierarchy.py:525` — `visible` (surfaced as
   `projected_motion_capability_passed`) is the pixel displacement of an
   FK-projected effector point, so it cannot distinguish visible motion from motion
   behind an occluder or in a frame that never rendered.
11. `g1_render_noise_audit.py:590` — `no_camera_self_occlusion_suspected` is
    assigned the value of `no_large_black_edge_wedge`: it asserts absence of camera
    self-occlusion while measuring only the dark-pixel ratio in an edge wedge.

**A claim with no producer and no consumer:**

10. `adp009d_live_hybrid_observation.py:429` — requires
    `camera_motion_occlusion_probe_passed`, `static_occlusion_probe_passed` and
    `moving_occlusion_probe_passed` to be True, but **nothing in the repo computes
    those keys** (only test fixtures set them), and
    `validate_live_hybrid_runtime_receipt` has no production caller. The occlusion
    claim is a declaration the producer writes about itself into a validator that
    never runs. Decide whether to wire it up or delete it.

**⚠️ Coverage is NOT exhaustive — one category is explicitly unfinished.**
Entries under the "recorded-but-unused statistics" shape (3, 4, 9, 11, and the
`adp009d_isaac_runtime.py` finding below) were produced by **targeted grep with
hand-traced consumers**, not by a mechanical sweep. Each reported entry is
high-confidence, but **the category is not closed** — assume more exist. The
exhaustive sweep was attempted and died when its worktree was deleted underneath
it mid-run. To finish it, run it against a **live checkout of the current main**;
do not run it against the primary clone at `/Users/nijelhunt_1/workspace/…`,
which sits at `a9b28b176` where the line numbers do not match.

A cheap mechanical form of the sweep: for every key written into a receipt or
`*_gates` dict, count readers outside the writer (`src/`, `scripts/`, `tools/`,
`tests/`). Zero readers, or readers that only assert the key's *name* exists,
means the field is decorative.

**And `adp009d_isaac_runtime.py:2045` is worse than "never gated":**
`rgb_min`/`rgb_max`/`rgb_mean`/`foreground_semantic_pixel_fraction` have **zero
readers** anywhere in `src/`, `scripts/`, `tools/` or `tests/`. The only test
touching them (`tests/test_adp009d_native_microcheck_bundle.py:946`) asserts the
literal **string** `"foreground_semantic_pixel_fraction"` appears in the module
source — which locks the dead field in place. Note the contrast: the *depth*
sibling in the same dict **is** genuinely gated (`require_metric_depth` →
`metric_depth_valid`, `:1978`), so the RGB fields are the only ungated members of
a block that otherwise looks enforced.

Corollary that cost a pod: `rgb_mean 0.0` proves nothing without a **positive
control**. Always render a known-good object in the same stage.

---

## 8. Environment

- Repo: `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline` (see §2 — stale).
- Control plane host: `/opt/blueprint/BlueprintCapturePipeline`, venv at
  `.venv/bin/python`, releases under
  `/opt/blueprint/task-evaluation-control-plane-releases`, state under
  `/var/lib/blueprint/pipeline-control-plane`, active symlink
  `/opt/blueprint/task-evaluation-control-plane`. Runs as user `blueprint`.
- Upstream Isaac Lab reference sources (no GPU, read-only):
  `ssh root@174.138.76.111`, at
  `/var/lib/blueprint/task-evaluation-inputs/native-task-runtime-source-c3e8b79a-sources`.
  Useful files: `IsaacLab/source/isaaclab_tasks/.../manipulation/cabinet/cabinet_env_cfg.py`
  (`rot=(0.5,-0.5,-0.5,0.5)  # align with end-effector frame`),
  `IsaacLab/scripts/environments/state_machine/open_cabinet_sm.py`,
  `.../direct/franka_cabinet/franka_cabinet_env.py`, and Arena's `DroidSceneCfg`
  at `isaaclab_arena/embodiments/droid/droid.py` (stiffness 400 / damping 80,
  `disable_gravity=True`).
- Image: `isaac-sim` **6.0.0-dev2**, digest-pinned `sha256:c3e7bef5…`.
- Scene geometry (sealed, registered stage, metres):
  hinge axis `[~0, 0, -1.0]`, hinge point
  `[3.2704862952316285, 9.456716013828277, 0.42999999415255735]`, contact point
  `[3.7634863044329236, 9.456664008775391, 0.40499998738356097]`,
  `parallel_jaw_stroke_m` 0.085, `pinch_span_m` 0.05509902644189424.
  Robot base pos `[3.5154863, 9.208716, 0.090782]`, quat xyzw
  `[0, 0, 0.7071067811865841, 0.7071067811865109]`. Franka hand reset orientation
  `[0.5, 0.5, 0.5, 0.5]`.

---

## 9. What "done" means

The lane is proven when a **paid production run** of the arena construction lane
for scene 840920 task_a completes with:

- the captured scene **visibly rendered** (not black), confirmed by a gate that
  actually reads pixels and by a human looking at a frame;
- the Franka reaching its commanded phases within tolerance, with the grasp
  orientation confirmed against the **measured** finger-direction readback rather
  than an assumed convention;
- provider zero sealed, teardown proven with `status_source="provider_api"`, and
  no open billing risk;
- every fix that made it work already on `main` with hermetic tests — nothing
  local, nothing hand-applied.

Downstream and still pending after that: chain the arena **controls** link off the
construction result, then the **policy** link (spec + run).
