# MuJoCo vs Isaac Sim lane — bidirectional capability gap analysis

**Blueprint Capture Pipeline · world-model + robot-policy evaluation**
Scope: iPhone capture → 3D scene → sim → GR00T/SONIC policy + WAM world-model eval
Method: 12-agent code-grounded deep-dive (4 lane inventories + 6 dimension comparisons + synthesis + adversarial verification). 13/15 highest-stakes claims verified directly against code; the 2 corrections are folded in below.

---

## 1. Executive summary

Today the two lanes are **not redundant — they are complementary halves of one eval system**. **Isaac is the photoreal seed/render + placement lane**: it is the only lane that boots a real RTX renderer, RTX-renders captured 3DGS/NuRec site splats, and resolves collision-checked stand poses from a USD scene — but it runs the robot kinematically (gravity off by default), executes **no learned policy in-process**, and has no native depth/segmentation passes. **MuJoCo is the contact-rich physics + closed-loop policy lane**: it steps real rigid-body dynamics under gravity every frame, drives PD torques to the G1's actuators, reads true 6D contact forces, and runs two genuine closed loops that requery a *learned* VLA policy against WAM-generated observations — but it renders only a flat-shaded OBJ proxy and provably cannot decode splats.

The single biggest asymmetry is **directional and clean: Isaac has the pixels, MuJoCo has the physics and the policy-in-the-loop.** Your suspicion is confirmed outright — **MuJoCo cannot render a photoreal scene; it is a diagnostic-grade rasterizer, not a sensor renderer** (verified: single grey material `rgba="0.45 0.50 0.55 1"`, GLB→OBJ drops PBR image textures, no splat decoder, no depth pass).

**Headline recommendation: do NOT pursue full parity. Specialize the lanes behind the existing simulator-agnostic execution contract** — MuJoCo = fast contact-rich physics rollout + learned-policy closed loop; Isaac = photoreal seed/render + placement — and close only the *narrow, high-value* gaps that make the division of labor work: a learned-policy requery in Isaac, native depth/seg passes in both, and a cross-lane observation handoff so the MuJoCo policy can consume Isaac/WAM photoreal frames.

---

## 2. Roles today (evidence-based)

**Isaac owns:**
- **Photoreal rendering.** Real RTX (`SimulationApp({"renderer": "RayTracedLighting"})`, `run_isaac_g1_kitchen_parity_eval.py:1858`), true path tracing + OptiX denoiser + firefly filter (`:5270–5274`), process-level render-step watchdog for C++ RTX hangs (`:6765–6841`).
- **Captured-site splat rendering.** The only lane that can display a reconstructed kitchen: 3DGS PLY → NuRec USDZ / ParticleField USD (`isaac_nurec_export.py`) → RTX render via `omni.hydra.rtx` / `omni.kit.converter.gsplat` (`run_isaac_splat_nurec_render.py:40,154,169`).
- **USD scene loading + placement.** USD references, official G1 USD articulation, a pure unit-testable `scene_placement` package (USD bbox index → VLM target resolution → collision-checked stand pose → footprint/boundary validation).
- **Photoreal seed + skeleton conditioning** for the WAM loop (projected-G1-skeleton trace that conditions OSCAR-2B next-observation generation).

**MuJoCo owns:**
- **Contact-rich dynamics.** Real `mj_step` forward dynamics under gravity every control step (`mujoco_g1_wam_vla_policy_endpoint_eval.py:8967`), PD-torque control to named leg actuators (`mujoco_g1_wam_vla_policy_endpoint_eval.py:7217–7219`), true 6D `mj_contactForce` (`mujoco_g1_simulator_command.py:1822–1824`), authored object mass/friction (`mujoco_g1_wam_vla_policy_endpoint_eval.py:5809`).
- **The learned-policy closed loop.** Two real loops requery a learned VLA/GR00T policy with the WAM-generated next observation (`mujoco_g1_wam_vla_policy_endpoint_eval.py:4609–4636`; `unitree_groot_n17_sonic_vast_persistent_session.py:1944–2062`), with honest completion gates (≥2 fresh non-replay calls + ≥1 action-conditioned generation).
- **Physics ground-truth replay.** `unitree_groot_n17_sonic_sim2sim_command.py` replays one GR00T action chunk open-loop and scores hand-object contact / nearest-hand distance.
- **Correct, emitted camera intrinsics.** `focal_px = 0.5·H/tan(fovy/2)` + camera world pose, emitted per frame (`mujoco_g1_wam_vla_policy_endpoint_eval.py:6358–6427`) — *more complete than Isaac*, which never reads K back.
- **The true warm persistent session** — one GPU instance hosts policy server + WAM worker, reused across every infer call, with the strictest zero-continuing-spend teardown gate in the stack.

**Shared (the contract that makes specialization legal):** both lanes run the same simulator-agnostic G1 policy. The policy server speaks a normalized obs (ego RGB + 46-dim structured state + language) / action (motion_token + hand joints) schema and **never imports a physics engine** (`unitree_groot_n17_sonic_policy_server_command.py` — verified: zero `import mujoco|physx|isaacsim|render_product|mj_step|SimulationApp` hits). The architecture docs codify this as a hard rule: **MuJoCo evidence ≠ Isaac evidence** (non-interchangeable proof boundary).

> **Naming caveat:** in `unitree_groot_n17_sonic_vast_persistent_session.py`, "Isaac-GR00T" is **only the policy-server host repo / venv root** — there is no `SimulationApp`, render product, or PhysX there. Isaac-as-a-stepped-simulator lives only in the kitchen-parity / splat-render scripts.

---

## 3. Bidirectional gap matrix

Legend: ✅ strong · 🟡 partial · 🟧 stub · ❌ absent. **Gap direction** = which lane is behind.

| Capability | Isaac | MuJoCo | Gap direction | Why it matters | Effort |
|---|---|---|---|---|---|
| Photoreal RGB rendering | ✅ RTX + path trace + OptiX | ❌ flat-shaded OBJ proxy | **→ MuJoCo (unfixable natively)** | VLA policies need the real captured site; MuJoCo gives a visually meaningless scene | n/a (route to Isaac) |
| 3DGS / NuRec splat render | ✅ native NuRec/ParticleField | ❌ no PLY/SPZ decoder (proven) | **→ MuJoCo (architecturally infeasible)** | Captured-site fidelity is the moat | n/a (orchestration) |
| Textured/PBR scene materials | ✅ USD materials | 🟧 grey material; GLB→OBJ keeps vertex color, drops PBR textures | → MuJoCo | Visual realism for policy + customer video | Small |
| HDRI / multi-light lighting | ✅ DomeLight/HDRI + fill | 🟡 1 headlight + 1 directional | → MuJoCo | Plausibility for VLA + render realism | Small |
| Native depth render pass | 🟡 analytic pinhole only | ❌ `supports_depth=False` | **→ both** | RGBD policies + WAM perception want co-registered depth | S (Isaac) / M (MuJoCo) |
| Instance/semantic segmentation pass | ❌ rgb-only; external SAM3 | ❌ body-name matching only | **→ both** | Free deterministic masks for success scoring | Medium |
| Camera intrinsics (K read-back + per-frame pose) | 🟡 author-side analytic, not emitted | ✅ derived + emitted per frame | **→ Isaac** | Sim-vs-real calibration, target-in-view scoring | Small |
| Contact-rich dynamics (gravity-on step) | 🟡 opt-in bounded settle, gravity OFF default | ✅ `mj_step` under gravity every frame | **→ Isaac** | An Isaac "pass" never integrated dynamics = weak physics evidence | Medium |
| Torque/effort articulation control | 🟧 position-target only | ✅ PD torques to actuators | **→ Isaac** | Force-level contact realism separates real manipulation from kinematic pose | Large |
| Accurate collision geometry | 🟧 forced boundingCube/convexHull | 🟡 real scene mesh + AABB box proxies | **→ both (different ways)** | Coarse colliders mis-score reach/clearance | Medium each |
| Authored contact material (friction/mass/restitution) | ❌ none in parity runner | ✅ mass=0.25, friction authored | → Isaac | Undefined friction = meaningless contact signal | Small |
| Closed-loop requery of a **learned** policy | 🟡 loop wired, in-loop policy deterministic; VLA stubbed | ✅ two real loops requery learned VLA | **→ Isaac (the big one)** | The entire point of the eval is policy-on-generated-world | Large |
| In-process success/manipulation judge | ❌ defers to external judge | ✅ `manipulation_success_evaluator` | → Isaac | No success label = output not comparable to MuJoCo | Medium |
| In-loop clean-frame reanchoring / drift gate | 🟡 pre-launch sanity gate only | ✅ in-loop reanchor + drift blocker | → Isaac | Long autoregressive rollouts drift to flat/dark | Small |
| Open-loop physics ground-truth replay (sim2sim) | ❌ none | ✅ scores hand-object contact | → Isaac | Cheap physics sanity-check of an action chunk | Large |
| Projected-skeleton WAM conditioning | ✅ from articulated pose | 🟡 numeric action vector, weaker | → MuJoCo | Constrains WAM to plausible robot poses | Medium |
| USD scene support | ✅ native | ❌ no USD load path | → MuJoCo | Can't share authored USD scenes | Large |
| Articulated fixtures (door/drawer joints) | 🟡 USD can carry joints | ❌ static box/mesh only | → MuJoCo | "Open the fridge" tasks unsimulatable | Large |
| Provider race (RunPod ↔ Vast + breaker) | ✅ `provider_race` race+reaper | 🟡 serial fallback, one path | → MuJoCo | Launch latency gates how often the cheap lane runs | Large |
| True warm persistent in-process session | 🟡 `--serve` render-only | ✅ one instance hosts policy+WAM | **→ Isaac** | Amortizes boot over a whole episode | Large |
| Git-worktree provenance gate on paid launch | ✅ dirty-tree fail-closed | ❌ none | → MuJoCo | Paid rollout from uncommitted code mistaken for evidence | Small |
| Boot-marker flaky-cold-pod reaper | ✅ kills bill-but-no-boot pods | 🟡 heartbeat poll only | → MuJoCo | Burns full wait window on dead nodes | Medium |
| Runtime preflight that boots the sim before paying | 🟡 image/window heuristics only | ✅ boots MuJoCo + EGL render | **→ Isaac** | Catches broken GL/RTX cheaply | Medium |
| Zero-continuing-spend teardown gate | ✅ strong | ✅ strictest (manifest must exist/complete) | ~ even (MuJoCo slightly ahead) | Spend safety | — |

---

## 4. What to ADD to MuJoCo (that Isaac has)

### Priority 0 — Settle the rendering question definitively (READ THIS FIRST)

**Verdict: MuJoCo *can* render, but it CANNOT render photoreal, and it never will natively.** Verified directly in code:
- `mujoco.Renderer` produces real offscreen RGB via EGL (`MUJOCO_GL=egl`) — frames → PNG → MP4. So "does MuJoCo render?" = **yes, RGB frames.**
- But the scene is painted with **one flat grey material**, `rgba="0.45 0.50 0.55 1"` (`mujoco_g1_simulator_command.py:806`), lit by a single headlight + one directional light. The GLB→OBJ conversion (`:486–503`) **drops PBR image textures/materials** (vertex colors *are* preserved — `obj_vertex_color_summary:504`). The code self-documents this: *"still simulator visual evidence, not photoreal Marble/SPZ or PBR texture proof"* (`:530`).
- MuJoCo **provably has no 3DGS/PLY/SPZ decoder** — the `g1_site_3dgs_mujoco_preview.py` probe writes test assets and concludes `mujoco_3_9_probe_no_ply_mesh_decoder` / `..._no_spz_mesh_decoder` (`:163,172`; also `:2729`). Splats must be pre-converted to a box/OBJ proxy.

**Fidelity ceiling: MuJoCo rendering is diagnostic-grade rasterization (flat-shaded, no IBL, no path tracing, no splats). It is fit for "did the robot move / is the target in view," NOT for giving a VLA policy a representative egocentric view of the deployment site. This gap cannot be closed inside MuJoCo.** The work splits into two distinct moves:

**P0a — Texture/material preservation (narrows, does not close, the gap).** Small.
- In `mujoco_g1_simulator_command.py` `_convert_glb_to_obj` (`:486`) preserve `trimesh` `visual.material`/UV and export **OBJ + MTL with `map_Kd`**; in `_write_mjcf_wrapper` (`:759`) emit MuJoCo `<texture>`/`<material>` assets bound to the visual geom instead of the single grey override.
- Dependency: upstream GLBs must actually carry PBR (many World Labs/Marble assets are collider-only — verify per-asset). Verify: render the same fridge scene before/after; assert >1 distinct material and non-uniform texture; pixel variance increases.

**P0b — Photoreal observation handoff (the real fix: orchestration, not MuJoCo code).** Medium.
- When a scenario needs splat-grounded RGB, **route photoreal frames from the Isaac splat renderer (`run_isaac_splat_nurec_render.py`) or the WAM generator into the MuJoCo policy's visual channel**, while MuJoCo keeps physics/contacts. The policy server already accepts an ego frame path, so this is a frame-source swap, not a physics change.
- Verify: a closed-loop MuJoCo run where `observation_source` is an Isaac/WAM photoreal frame, with the policy still stepping MuJoCo physics.

> **Definitive answer:** MuJoCo renders RGB at flat-shaded diagnostic fidelity only — no PBR textures, no photoreal lighting, no splats, no depth. Photoreal site rendering is **Isaac-only**, and the correct design is to *hand Isaac/WAM frames to the MuJoCo policy*, not to make MuJoCo photoreal.

### Priority 1 — Native render-pass depth buffer · Medium
- `supports_depth=False` (`mujoco_g1_wam_vla_policy_endpoint_eval.py:2776`); the only "depth" is geometric projection of a known point (`:6405–6427`). Enable `Renderer.enable_depth_rendering()` in the render paths, save depth per camera, flip the flag. Verified: zero depth-render API usage today, so this is a genuine absent→present add (MuJoCo ≥2.3). Verify: non-constant depth co-registered with RGB; reproject a known target and check agreement.

### Priority 2 — Segmentation render pass · Medium
- No seg pass anywhere; object identity is geom/body-name string matching. Enable MuJoCo segmentation in `update_scene` (`scene.flags`/seg mode) and map geom/body ids → labels (the body-name map already exists in the contact code). Verify: per-pixel instance ids match contact body names.

### Priority 3 — Convex / oriented collision proxies · Medium
- Per-component **axis-aligned** box proxies over-approximate footprints (an L-counter becomes one big box), inflating false collisions vs Isaac's convexHull. Add an optional convex-decomposition path (VHACD/coacd → convex mesh geoms) or `trimesh.bounds.oriented_bounds` OBBs in `_collision_proxy_geoms_from_mesh`, gated behind a flag mirroring Isaac's `--collision-approximation`; keep AABB as fast default. Verify: stand-pose acceptance on a non-boxy fixture matches Isaac's overlap decision more closely.

### Priority 4 — Articulated fixtures (door/drawer joints) + richer lighting · Large / Small
- Scene fixtures are static box/mesh only; lighting is 1+1. Extend `_write_mjcf_wrapper` to emit `<body><joint type=hinge|slide>` for labeled fixtures (bbox metadata already tags doors/drawers); add scene-derived directional/area lights + shadows. Verify: an "open drawer" task produces a moving slide joint and contact on the handle.

*(Tuning solver `condim`/`solref`/`solimp` on floor + proxies is a small independent hardening win — currently MuJoCo defaults are inherited.)*

---

## 5. What to ADD to Isaac (that MuJoCo has)

This list is **long and high-value** — Isaac is the lane further from the eval's actual purpose (policy-on-generated-world). Isaac's render is excellent, but its *physics and policy-in-the-loop are the weaker half.*

### Priority 1 — Closed-loop requery of a LEARNED policy (the single most important Isaac gap) · Large
- Isaac's loop is plumbing-complete but the in-loop policy is `DeterministicWalkToTargetPolicy`, which **ignores the WAM-generated frame and walks a fixed route by step index** (`isaac_g1_policy.py:205–249`); `Groot17SonicPolicy.reset/step` raise `NotImplementedError` (`:280,283`). So Isaac proves render + conditioning + perception, **never policy-on-generated-world**. MuJoCo already does this in two places.
- In `oscar_isaac_closed_loop_eval.py:run_oscar_isaac_closed_loop`, make the in-loop policy pluggable (accept a policy object/endpoint instead of hardcoding the deterministic policy at `:1679`); after the WAM observation is produced (`:1706,1723`), **requery the policy to get the next action** (mirror `mujoco_g1_wam_vla_policy_endpoint_eval.py:4609–4636`), reusing the ZMQ policy client / policy-server path.
- Verify: trace shows the next action changing as a function of the generated observation (perturb the frame → action changes); `policy_observes_wam_generated_next_observation=True` only when fresh non-replay calls occurred.

### Priority 2 — In-process manipulation success evaluator · Medium
- Isaac defers success to an external judge (`oscar_isaac_closed_loop_eval.py:1874–1877`); no success label even when the loop runs. Add a `manipulation_success_evaluator` step after the trace is written, mirroring `vast_persistent_session.py:4157–4173`; keep success_proof separate from structural loop proof. Do after P1 (else it judges the deterministic route).

### Priority 3 — Honest completion gating + in-loop drift reanchoring · Small
- Isaac sets `completed` when trace rows exist and no blockers (`:1824`), with no requirement that a fresh learned policy was requeried — overclaims vs MuJoCo. Strengthen completion to require fresh learned-policy requery counts; add `policy_observes_wam_generated_next_observation` / `wam_evaluator_in_control_loop` fields. Add optional `clean_frame_reanchoring` feeding the seed frame back at intervals (mirror `vast_persistent_session.py:611–645,1998–2013`). Needs P1 first.

### Priority 4 — Gravity-on stepping + torque control + real contact materials · Medium / Large / Small
- The main parity render runs **gravity OFF** (`run_isaac_g1_kitchen_parity_eval.py:2919–2920`) holding a kinematic pose; drive is position-target only (`:2522`); colliders forced to boundingCube/convexHull (`:5066–5083`); no friction/mass on interaction objects. Extend `_settle_dynamic_standing_contacts` (`:3112`) into a per-step gravity-on loop as the standard *physics-proof* path; add an effort/torque drive (`ArticulationAction(joint_efforts=...)`, port MuJoCo's kp/kd PD law) behind the existing `physics_articulation_drive` opt-in; author `UsdPhysics.MassAPI` + PhysicsMaterial on the resolved target only, with per-prim convexDecomposition cooked *just for that target*.
- **Blocking dependency:** the official G1 USD invalidates Isaac's tensor view on joint drive/read (`:2936–2961`) — needs a G1 USD variant that survives drive/read, plus authored per-joint kp/kd, plus GPU time and a balance/settle acceptance. Verify: a physics-proof run where the robot integrates under gravity with non-trivial contact forces and falls correctly when destabilized.

### Priority 5 — Provider-runtime parity (warm session + runtime preflight) · Large / Medium
- Isaac races providers and reaps cold pods but has **no true in-process persistent session** (`--serve` is render-only, so each closed-loop WAM step round-trips a separate worker) and **no runtime preflight that actually boots SimulationApp** (only image/window heuristics). Extend `--serve` to host a long-lived policy/WAM server analogous to `vast_persistent_session.py:1389–1573`; add `isaac_worker_runtime_preflight.py` that boots `SimulationApp` headless + renders a tiny RTX smoke frame before paying (mirror `mujoco_worker_runtime_preflight.py`).

**Honest note:** items 1–4 matter for product correctness. Item 5 makes the lane cheaper, not more correct — do it last.

---

## 6. Recommended division of labor / target architecture

**Recommendation: specialize, do not bring to parity.** The simulator-agnostic execution contract exists precisely so the lanes can differ. Forcing parity means (a) a splat rasterizer inside MuJoCo (infeasible) and (b) a full learned-policy + dynamic-physics loop inside Isaac (large, and duplicates a working MuJoCo capability). Neither earns its cost.

**Target roles:**
- **MuJoCo = fast contact-rich physics rollout + learned-policy closed loop.** Cheap, CPU-capable primary proof path and canonical policy↔WAM host. Invest in depth/seg passes, convex colliders, and photoreal *frame intake* (P0b) — not in making MuJoCo render.
- **Isaac = photoreal seed/render + placement + skeleton conditioning.** The only lane that produces a representative egocentric site view and resolves stand poses. Invest in the learned-policy requery + success judge so its rollouts become *comparable*, and gravity-on/torque physics so an "Isaac pass" means something.
- **WAM (OSCAR/Cosmos) = generated-world policy ranking** behind the replaceable adapter, fed by Isaac's photoreal seed + skeleton, consumed by whichever lane hosts the policy loop.

**Minimal shared contract the two lanes must guarantee:**
1. **Same policy obs/action schema** (already true; no physics-engine import in the policy server).
2. **Per-frame camera contract** — every RGB frame from *either* lane emits `{K (fx,fy,cx,cy), camera world pose, resolution, projection_method}`. MuJoCo already does; **Isaac must add K read-back + per-frame pose.** Cheapest cross-lane win; unblocks sim-vs-real calibration.
3. **Optional co-registered depth + segmentation artifacts** with a shared schema (both lanes currently lack the render passes).
4. **Non-interchangeable proof markers preserved** (`simulator_backend` / `*_proven` flags) so MuJoCo physics ≠ Isaac render.
5. **A frame-source handoff** so the MuJoCo policy loop can consume Isaac/WAM photoreal frames as its visual channel while MuJoCo owns physics — this is what makes "MuJoCo physics + Isaac pixels" one eval.
6. **One provider-runtime contract** (race + marker-reap + git-provenance + warm-candidates + strict teardown) promoted into `provider_race`/`gpu_render_providers`, parameterized by `bundle_kind` + readiness marker, so a guard added to one lane reaches both.

---

## 7. Sequenced roadmap (value-per-effort)

1. **Emit the per-frame camera contract in Isaac** (K read-back + camera pose) — Small; unblocks calibration; brings Isaac to MuJoCo's intrinsics bar.
2. **Native depth render pass in both lanes** — Small (Isaac: add `distance_to_image_plane` annotator near `:5128`) / Medium (MuJoCo: `enable_depth_rendering`). Highest-value shared modality gap; verified absent in both.
3. **Learned-policy requery in Isaac** + **strengthen completion gating** — Large; converts Isaac from plumbing-proof to policy-on-generated-world, the eval's actual purpose.
4. **Photoreal observation handoff** (Isaac/WAM frame → MuJoCo policy visual channel) — Medium; makes "MuJoCo physics + Isaac pixels" one system and resolves the rendering gap *correctly* (orchestration, not MuJoCo rendering).
5. **In-process manipulation success judge in Isaac** — Medium; makes Isaac rollouts comparable. After #3.
6. **Segmentation render pass in both** + **convex/OBB colliders in MuJoCo** — Medium each.
7. **MuJoCo texture/material preservation** + **scene-derived lighting** — Small; narrows (not closes) the visual gap for assets that carry PBR.
8. **Isaac gravity-on dynamic stepping + torque control** (resolve tensor-view first) — Medium/Large; makes an "Isaac physics pass" real. Behind `physics_articulation_drive`.
9. **Provider-runtime convergence** — promote race + marker-reap + git-gate + warm-session + runtime-preflight into a shared module; port git-provenance to MuJoCo (Small), boot+render preflight to Isaac (Medium). Last (cost/ops, not correctness).

---

## Key file index (for action)

- **Isaac render/physics/policy:** `scripts/run_isaac_g1_kitchen_parity_eval.py` (rgb-only annotator `:5128`, path-trace settings `:5270–5274`, gravity-off `:2919–2920`, position drive `:2522`, cheap-collision `:5066–5083`, tensor-view invalidation `:2936–2961`, render watchdog `:6765–6841`); `scripts/run_isaac_splat_nurec_render.py`; `src/blueprint_pipeline/isaac_nurec_export.py`; `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py` (loop `:1639–1882`, deterministic policy `:1679`, external-judge boundary `:1874–1877`); `src/blueprint_pipeline/isaac_g1_policy.py` (deterministic `:205–249`, NotImplemented stubs `:280,283`); `src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`; `src/blueprint_pipeline/scene_placement/`.
- **MuJoCo physics/render/closed-loop:** `src/blueprint_pipeline/mujoco_g1_simulator_command.py` (flat grey material `:806`, GLB→OBJ `:486–503`, vertex-color `:504`, fidelity boundary `:527–531`, `mj_contactForce` `:1822–1824`); `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py` (closed loop `:4609–4636`, intrinsics `:6358–6427`, `mj_step` `:8967`, PD torque `:7217–7219`, object mass/friction `:5809`, `supports_depth=False` `:2776`); `src/blueprint_pipeline/g1_site_3dgs_mujoco_preview.py` (no-decoder probe `:163,172,2729`); `src/blueprint_pipeline/unitree_groot_n17_sonic_sim2sim_command.py`; `src/blueprint_pipeline/unitree_groot_n17_sonic_vast_persistent_session.py` (closed loop `:1944–2062`, success eval `:4157–4173`, reanchor `:611–645,1998–2013`); `src/blueprint_pipeline/mujoco_worker_runtime_preflight.py`.
- **Shared contract:** `src/blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py` (no physics-engine import); `vast_provider_adapter.py`; `provider_race.py`; `gpu_render_providers.py`.

---

### Verification notes (corrections folded into the report)
1. **"GLB→OBJ drops ALL materials"** was overstated — PBR **image textures** are dropped, but **vertex colors survive** (`mujoco_g1_simulator_command.py:504`). Matrix/text updated to "drops PBR textures (vertex color survives)."
2. **PD torque law and object mass/friction** live in `mujoco_g1_wam_vla_policy_endpoint_eval.py` (lines 7217–7219 and 5809), not `mujoco_g1_simulator_command.py` (which is only 5582 lines). Citations corrected.
3. **Policy server file** is `unitree_groot_n17_sonic_policy_server_command.py`, not `policy_server_command.py`. The "no physics engine imported" capability claim is verified correct.

*Coverage note: 11/12 analysis agents succeeded; the dedicated sensors/observation agent hit a structured-output retry cap, but that dimension is covered by the rendering, camera-intrinsics, and policy/WAM agents (depth, segmentation, intrinsics, proprioception all appear in the matrix).*
