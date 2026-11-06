# Policy & WAM Benchmark Research — 2026-07-26

Question: which policies should we benchmark next (task is flexible — kitchen, warehouse,
factory all acceptable; we can re-stage the scene), and should policies come before
world-model (WAM) work?

Method: four parallel web-research agents (policy landscape, Unitree G1 ecosystem,
world-model alternatives + world-model-as-evaluator prior art, proven checkpoint+task
pairs), all verifying against live sources dated 2026-07-26, plus grounding in this
repo's contracts (`UNITREE_ACTION_COMMAND_CANDIDATES`,
`configs/kitchen_unitree_g1_task_registry.json`, sealed-checkpoint pinning,
`swap_policy.yaml`, `WORLD_MODEL_STRATEGY_CONTEXT.md`). Claims below marked LOW
CONFIDENCE where the agents could not verify a primary source. This document is
research, not a run artifact; nothing here proves live-provider, deployment, or
physical-robot readiness.

---

## A. Sequencing verdict: policies first — but via proven pairs, not a microwave fine-tune

1. **The current baseline cannot produce task signal.** `LucaFrat/groot-bs16` is a
   GR00T N1.7-3B `UNITREE_G1_SONIC` fine-tune by a private individual (one of 17
   sibling models in a personal hyperparameter sweep), trained on
   `LucaFrat/dataset_100`: 106 episodes, exactly one task — *"grab the bag, turn 180
   degrees and drop the bag"* — and **the dataset declares no license**. Evaluating it
   on "open the microwave" measures zero-shot off-task behavior. Door angle 0.0 across
   rev9's 11 clean steps is the expected result. (Repo comment at
   `single_g1_kitchen_episode_runpod.py:106` already recorded the task mismatch;
   the provenance and license findings are new.)
2. **The loop is proven; the policy is the missing ingredient.** Rev9 executed 11
   clean WAM→coherence→Isaac transitions. That is enough horizon to measure
   differential task progress from a policy that actually does its task — even with
   today's OSCAR chaining fragility.
3. **Policy comparison is the product.** The candidate registry, pack architecture, and
   ranker validation ladder all exist for exactly this. Published prior art (SNU,
   arXiv:2512.01358) gives the expected known-ordering ladder on a G1: **0% zero-shot →
   48% standard fine-tune → 94% contact-augmented** — a policy-side ladder our ranker
   can be validated against.
4. **The WAM fix is an architecture change, chartered separately.** OSCAR's public
   paper (arXiv 2606.04463, June 2026) validates it **open-loop only**; no chained
   durability was ever published. The 2026 literature names our collapse (exposure
   bias) and converges on short action-conditioned chunks + explicit memory — no
   published system chains 81-frame single-seed generations. That work maps to the
   strategy doc's existing Cosmos-3-behind-the-adapter charter, sequenced after
   policy signal exists.

**The task pivot matters as much as the policy pivot.** The fastest route to a
provably-working policy is adopting a (checkpoint + task) pair with existing success
evidence and re-staging our scene around it — not forcing any particular appliance.
Rigid pick-and-place pairs are also *easier* for our evaluator than articulated doors
(clean displacement/contact transitions; no articulated USD requirement).

**The microwave niche stays open.** No published closed-loop G1 microwave-door result
exists anywhere as of 2026-07-26. Keep the microwave lane as the stretch/novelty lane;
do not make it the qualification-critical path. The only public G1 microwave dataset
(`niravpanchalmerai/dtwin_g1_microwave_bowl`, 200 eps, SONIC-consistent schema) is
**unlicensed** — rights-blocked for our use per repo rules.

---

## B. Tier 1 — positive-control pairs (adopt first)

| Rank | Pair | Why | License |
|---|---|---|---|
| 1 | **`nvidia/GR00T-N1.6-G1-PnPAppleToPlate`** (official G1 checkpoint) on the apple→plate task; or re-run NVIDIA's e2e N1.7 recipe | Official checkpoint; NVIDIA reports ~90–92% sim success (secondary source, LOW CONFIDENCE on exact figure); **independent replication measured 50% (5/10)** closed-loop (`cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate`, Apache-2.0); ego-POV camera; contact-force success termination is public; scene = table + apple + plate | Weights NVIDIA Open Model License (commercial OK); data `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` subset `unitree_g1.LMPnPAppleToPlateDC` (102 trajs) CC-BY-4.0. Verify N1.6 card license before commercial deliverable (N1.5 precedent below). |
| 2 | **GR00T N1.7-3B fine-tuned on `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1`** (real G1, **Dex3-1 tri-finger**, fruit→basket, 250 teleop trajs/task, single ego RealSense 640×480) | Exact embodiment incl. our hands; language-conditioned fruit choice doubles as an instruction-discrimination probe; NVIDIA's course quotes ~2–3 h fine-tune on one RTX 6000 Ada (batch 12, 20k steps, horizon 40 @ 50 Hz); SNU published the success ladder on this embodiment+family | Base commercial-OK; dataset CC-BY-4.0 |

Runner-up positive controls: `JeffrinSam/GR00T-N1.7-G1-BrainCo-Pick` (Apache-2.0, real
G1 picks, single ego cam — but BrainCo hand mismatch, no reported SR);
`florianmoedl/G1sortscubesinboxes_GR00T17` (warehouse-flavored cube-sorting,
Apache-2.0, paired dataset, LOW CONFIDENCE quality); π0.5 G1 box-move precedent
(`xiaopeng-wu/pi05_unitree_g1` on `nepyope/unitree_box_move_blue_full`, 550 eps,
Apache-2.0 — global cam, not ego).

## C. Tier 2 — cross-family comparator bench (the product demo)

Fine-tune each family on the SAME dataset (Tier-1 pair #2's fruits set, or
`unitreerobotics/G1_Dex3_ToastedBread_Dataset` — 418 eps, Apache-2.0, kitchen-native)
and evaluate on the same task through the same lane. Known-ordering + cross-family
ranking is exactly what the ranker validation ladder needs.

| Family | Checkpoint | Size / est. inference VRAM | License | Adaptation | Notes |
|---|---|---|---|---|---|
| GR00T N1.7 (control arm) | `nvidia/GR00T-N1.7-3B` | 3B / ~16 GB | NVIDIA OML (commercial OK) | native `UNITREE_G1_SONIC`; official recipe | Already runs on our pod (groot-bs16 is this family) |
| Ψ₀ (Psi-0) | `USC-PSI-Lab/psi-model` | ~2.5B / ~6–8 GB (LC) | Apache-2.0 code; **confirm weights card** | G1-native, ego+proprio+language; 48-DoF joint-space — runs via its own deploy path (different-controller arm; record controller lane in scorecards) or SONIC-head retrain | Only non-NVIDIA G1-native open humanoid VLA (RSS 2026) |
| π0.5 | `lerobot/pi05_base` (mirror openpi `pi05_base`) | ~4B / ~8–10 GB (LC) | Apache-2.0 / Gemma terms on LeRobot port — **legal check** | new-embodiment fine-tune via LeRobot; G1 precedent exists | π0.6 is closed; 0.5 is the newest open |
| WALL-OSS 0.5 | `x-square-robot/wall-oss-0.5` | 4.2B / ~10 GB (LC) | Apache-2.0 | LeRobot-native `wall_x`; no G1 precedent (higher tuning risk) | Architecture diversity (MoE + CoT) |
| SmolVLA | `lerobot/smolvla_base` | 0.45B / ~3 GB | Apache-2.0 (card tag absent — pin) | full fine-tune; weak humanoid transfer expected | Calibration floor + cheap harness smoke-test |
| ACT (non-VLA floor) | `myx160/unitree_lerobot_act_g1d_*` or in-house (~hours to train) | ~80M / negligible | Apache-2.0 | `unitree_lerobot` toolchain (supports ACT/DP/pi0/pi05/GR00T on G1) | Non-VLA baseline for the ladder |

Excluded, with reason: RDT2-VQ/FM (hard binocular wrist-cam requirement), RDT-1B
(wrist-view prior + T5-XXL overhead), OpenVLA-OFT (15.9–18 GB at budget edge, no
humanoid), X-VLA (20-dim action ceiling in released head), EO-1 (MIT but no G1 path),
MolmoAct 2 (no humanoid out of the box), GigaBrain-0.1 (Apache-2.0 but no G1
evidence). Watch: **GR00T N2** (DreamZero world-action architecture, GTC 2026 preview,
weights end of 2026).

## D. Kitchen/articulated licensed data (when we return to appliances)

- Unitree Apache-2.0 whole-body-teleop sets: `G1_WBT_Inspire_Put_Drinks_Into_Fridge`
  (fridge door, 300 eps), dishwasher plate sets (Brainco/Inspire), washing-machine
  sets. **Hand-hardware mismatch is the dominant transfer killer**: these are
  Inspire/Brainco-hand sets; our platform is Dex3-1. Dex3-native lanes:
  NVIDIA Teleop-G1 (fruits), `G1_Dex3_*` (13 sets incl. ToastedBread, Pouring,
  CameraPackaging), Humanoid Everyday (MIT, 260 tasks × 40 eps, G1+Dex3, no
  checkpoints).
- For a licensed microwave lane, we must generate our own data (VR teleop → LeRobot
  v2.1 via the public GR00T-WholeBodyControl recipe, ≥50–100 demos) or adapt Unitree's
  fridge-door set with a hand-swap in sim (`unitree_sim_isaaclab` supports Dex3, Dex1,
  Inspire).

## E. License traps (verified 2026-07-26)

- **`nvidia/GR00T-N1.5-3B` is now carded non-commercial** ("Nvidia License") — N1.5-era
  checkpoints and community fine-tunes are unusable in commercial deliverables. N1.7
  is explicitly commercial (NVIDIA Open Model License). Check N1.6 cards individually
  (`GR00T-N1.5-3B-WaveHand` is "One-Way Noncommercial").
- **UnifoLM weights are CC BY-NC-SA** (WMA-0 and VLA per card checks) despite our
  registry having wired slots — datasets from Unitree are Apache-2.0, the *weights*
  are not. NC also blocks: AgiBot GO-1, Galaxea G0, Fourier ActionNet (data).
- **Unlicensed** (= blocked by our provenance rules): `LucaFrat/dataset_100` (current
  baseline's training set), `niravpanchalmerai/dtwin_g1_microwave_bowl`, TrajBooster
  retarget dataset (no license file).
- Action: pin the card license (repo + revision) at candidate admission time, same
  pattern as the sealed-checkpoint provenance comment.

## F. WAM track (sequenced second; two cheap actions now)

- **OSCAR public identity**: arXiv 2606.04463 (June 2026) — Cosmos-Predict2.5-2B
  backbone, 81-frame native, skeleton-latent action conditioning, policy-eval Pearson
  r=0.750 on RoboArena, **open-loop only; no chained-durability evidence published**.
  Our 5–11-generation collapse is the literature's exposure-bias failure mode.
- **Architecture implication**: every published system with stable long
  policy-in-the-loop rollouts chains short action-conditioned chunks with explicit
  memory (Ctrl-World 1-s chunks/20+ s; WEAVER 15-step chunks; PiL-World ~225 frames;
  GigaWorld-1 40 s with memory buffers + RoPE re-init; RoboWorld "Step Forcing"
  r=0.989). Cosmos-Predict2.5's Feb 2026 action-conditioned lane is built on 13-frame
  chunks. Nobody chains 81-frame single-seed generations. The WAM upgrade is chunked
  rollout + memory, not just a bigger 81-frame model.
- **Benchmark shortlist (fits 48 GB, commercial-clean)**: Cosmos-Predict2.5
  action-cond/distilled (NVIDIA OML), Ctrl-World (MIT), WEAVER (license LC), LingBot-VA
  (Apache-2.0), GigaWorld-1 (CC BY 4.0). Blocked: UnifoLM-WMA (NC), GE-Sim-V2 (NC
  portions), DreamZero-14B (2-GPU min; 5B variant possible), Genie 3 (closed).
- **Cheap now**: (1) reimplement dWorldEval's round-trip LPIPS and PiL-World's
  Hallucination-Free Ratio against our existing rev9/rev11 clip artifacts — free,
  local, quantifies the drift we currently infer from coherence-gate trips; (2) record
  per-generation drift curves in the manifest so WAM swaps have a baseline metric.
- Evaluator prior art for positioning: WorldEval r=0.942, RoboWorld r=0.989,
  PiL-World 0.94, dWorldEval 0.91–0.93, WorldGym r=0.78/3.3% gap, MiraBench (finds
  optimism bias widespread; visual fidelity ⊄ action fidelity). NVIDIA's
  IsaacLab-Arena is the nearest competitive analog (pick-place/loco-manip only, no
  articulated kitchen closed-loop).

## G. Repo integration map (what adoption actually costs)

1. **Task registry**: add pair tasks to
   `configs/kitchen_unitree_g1_task_registry.json`-style registries. Apple/fruit
   pick-place needs one new registered `observable_transition` type
   (object-displacement or contact-force; current registered criteria are
   articulation-angle) — small evaluator addition; contracts stay data-driven.
   `swap_policy.yaml` already enumerates warehouse manipulables.
2. **Candidate slots**: Tier-2 families ride `UNITREE_ACTION_COMMAND_CANDIDATES`
   (checkpoint envs per candidate). Record controller lane (SONIC-latent 78-dim vs
   joint-space) as a scorecard axis so cross-family comparisons stay honest.
3. **Prefix parity**: every candidate's native chunk (24–50 steps) exceeds our
   16-frame executed prefix; pin per-policy replan-rate/prefix truncation in the run
   spec (same class of contract as the rev11 duration-scaled gate).
4. **Provenance**: pin HF repo + revision for every checkpoint AND dataset (mirror
   openpi GCS weights into an HF repo we control); reuse the sealed-checkpoint
   reviewed-provenance pattern; NVIDIA OML attribution where applicable.
5. **Scene staging**: pick-place staging is commodity props + existing scene tooling
   (`scene_placement` resolves task string → stand pose). No articulated USD needed
   for Tier 1.
6. **Unverified claim to close out**: the "microwave-qualified checkpoint previously
   passed open-loop qualification" claim from the live session has no artifact in
   local `output/` and no public counterpart found. Require path + digest or drop it
   from planning.

## H. Open questions / measurements needed

- Measure real Ada-class inference latency for each Tier-2 family during smoke tests
  (published numbers are H100/Orin; Ada figures are interpolations).
- Confirm Ψ₀ weights-repo license text and SmolVLA card license tag before
  contract-grade use.
- N1.6 checkpoint card licenses (per-artifact, given the N1.5 re-card).
- Whether Tier-1 pair #1 runs against our Isaac scene directly or via
  IsaacLab-Arena/cloudwalk MuJoCo+gear_wbc loop first (independent replication used
  the latter).
