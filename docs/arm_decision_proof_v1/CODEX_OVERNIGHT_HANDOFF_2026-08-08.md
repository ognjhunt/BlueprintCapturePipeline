# ADP-009D overnight handoff — autonomous continuation

You are continuing ADP-009D: a Franka arm manipulating a sealed SimReady can in a
public scene (InteriorGS 840313), evaluated against learned DROID policies. The
pipeline now runs end to end. Your job is to close the open questions, harden
what is fragile, and reduce cost per run — unsupervised, for several hours.

## Where to work

```
worktree : /private/tmp/blueprint-adp009d-harness-20260806
branch   : codex/adp009d-franka-harness-20260806   (HEAD 54f70c8e0)
main repo: /Users/nijelhunt_1/workspace/BlueprintCapturePipeline
```

Read `AGENTS.md` and `CLAUDE.md` first. They are binding. Note especially the
**no hand-fixes rule**: every fix lands as code on the branch with a hermetic
fast-lane test pinning the contract. A manual workaround that is not encoded is
not a fix.

Verification loop:
```bash
PYTHONPATH="$PWD/src" pytest tests/ -q -k "adp009d or droid or episode or nurec or aura"
ruff check src/ tests/
```
Both must pass before every commit. Commit and push after each landed change.

## State of play — what is proven

**The episode pipeline runs.** Run v62 (`native_microcheck_v62_pi05_episodes`):

```
status: completed   blockers: []
candidate_policy_queried: True   candidate_outcomes_accessed: True
3 episodes · 3 scored · 0 failed · 60 policy queries each · 520 env steps each
outcome: never_moved ×3   ranking: ["pi05_droid"]   mean_outcome_rank 0.0
```

**Appearance is solved.** Aura's ghost-removed 2DGS is authored as a hand-built
NuRec volume and renders in Isaac. Current best asset:
```
.../adp009d_franka_public_scene_sim_20260806/aura_nurec_v3_recentred_exposed/aura_ghost_removed_nurec.usdz
```
- v51: room renders (mean 27 → 213 vs ParticleField)
- v53: exposure fixed via `omni:nurec:ccm*` diagonal 0.689 → saturation 35% → 0%
- Recentring verified: room composes in the correct world position
- Measured: our render carries **more** high-frequency detail than Aura's own
  512×384 reference render. The perceived softness is magnification, not loss.

**Efficiency is solved.** Rendering at 320×180 — the exact resolution both
candidates consume — gives byte-identical observations for 1/16 the pixels:

| | 1280×720 | 320×180 | speedup |
|---|---|---|---|
| first render | 25.40 s | 0.146 s | 174× |
| per frame | 22.5 s | 0.195 s | 115× |
| one episode | ~3.2 h | 89 s | 131× |

**pi05_droid is fully proven**: provisions rc=0, server ready in ~53 s, returns a
well-formed 15×8 action chunk, four separate runs.

## Open questions — in priority order

### P0. `never_moved` ×3 is uninterpretable. Fix the evidence, then interpret.

All three episodes scored `never_moved` (the can never moved). **Two readings
cannot currently be distinguished**: π0.5 genuinely failing an out-of-distribution
scene, or actions never reaching the robot.

`adp009d_episode_batch.run_episode_batch` keeps only a per-episode summary and
discards the per-query detail that would settle it. `run_policy_episode` already
computes it — `queries[]` carries `chunk_shape`, `executed_rows`,
`discarded_rows`, `any_joint_limit_clamped`.

Do this:
1. Retain in the batch row, per episode: joint position at reset vs at end,
   max per-joint delta over the episode, `any_joint_limit_clamped` count, and a
   summary of the commanded action magnitudes.
2. Add a derived field that names which reading applies, e.g.
   `arm_moved: bool` and `actions_reached_robot: bool`. **Do not infer a policy
   verdict from an absent one** — if the arm never moved, that is a harness
   finding, not a policy result.
3. Re-run and interpret only then.

This is the single most important item. A `never_moved` reported as a policy
result when the actions were being dropped would be a false claim.

### P1. groot_n17_droid never serves.

It provisions completely — venv, pinned install, checkpoint fetch all succeed —
then hangs at server start. Observed **47 minutes past the worker's own
15-minute readiness timeout**, which means it is blocked *inside*
`attempt_round_trip` rather than looping around it.

Suspects, in order: the `gr00t.policy.server_client` import blocking; the
`PolicyClient` constructor blocking on connect (its `timeout_ms=15000` governs
the request, not the connect); the server subprocess never binding.

Do this:
1. Make `wait_for_round_trip` enforce its deadline even when a single attempt
   blocks — the current loop can be starved by one hanging call. A per-attempt
   timeout, or run the attempt in a thread with a join deadline.
2. Capture and retain the server log for a failed candidate so the failure is
   diagnosable without another run (`adp009d_policy_server.<candidate>.log`
   already exists; make sure it survives into the output zip on failure).
3. Then diagnose with the receipt rather than by guessing.

Provisioning is now bounded per candidate
(`BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS`, default 1500 s), so a hang no
longer consumes the whole run.

### P2. Untested asset variants.

- `aura_nurec_v4_float32/` was built but never rendered (run killed). float32
  removes a 0.43 mm p95 position quantisation against 0.81 mm surfels.
  **Expectation: low value (~15–20%)** — measurement showed our render already
  out-resolves the reference. Test it once, cheaply, then decide whether the
  84 MB payload earns its size. If null, keep v3 and delete v4.
- 30.3% of DC coefficients decode outside displayable range (peak 4.64×). The
  `ccm` scale of 0.689 was derived from p99. Sweep 0.47 (p99.9) only if the
  frames show remaining clipping.

### P3. Cost and cycle-time optimisation.

Each run currently costs ~15 min boot + ~9 min provisioning before any work.
Ideas worth investigating, cheapest first:

1. **Reuse a warm instance across runs** rather than boot-per-run. There is
   prior art in the repo (`unitree_groot_n17_sonic_vast_persistent_session.py`).
   This is the largest single win.
2. **Cache the policy venv and checkpoint** on a persistent volume so
   provisioning is not repeated. pi05's checkpoint alone is 12.4 GB.
3. Profile where the 89 s per episode goes — policy inference vs render vs
   physics. If inference dominates, batching or a shorter horizon may help; if
   render dominates, there may be more headroom below 320×180 for the wrist
   camera specifically.
4. Consider whether `cfg.sim.render_interval` can exceed `decimation` so frames
   render only on policy-query steps (every 8th). Verify no stale-frame hazard
   before adopting.

### P4. Then, and only then: the two-policy ranked run.

Once P0 and P1 are closed, run both candidates, 3 episodes each, one scene, and
produce the ranking. The runtime already supports a comma-separated candidate
list and ranks by mean outcome rung with the sample-size caveat attached.

## Launch command (copy exactly, adjust job name)

Always verify provider zero **before** launching and **after** teardown.

```bash
cd /private/tmp/blueprint-adp009d-harness-20260806
V=/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804
JOB=$V/adp009d_franka_public_scene_sim_20260806/native_microcheck_vNN_description
mkdir -p "$JOB"
cp $V/adp009d_franka_public_scene_sim_20260806/native_microcheck_v62_pi05_episodes/adp_arena_vast_machine_avoidlist.json "$JOB/"
export BLUEPRINT_ADP009D_CAMERA_RESOLUTION=policy
export BLUEPRINT_ADP009D_EPISODES=3
export BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS=1500
PYTHONPATH="$PWD/src" python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --probe-kind adp009d-franka-native-microcheck --provider vast \
  --provider-launch-request /dev/null --release-evidence /dev/null \
  --model-cache-evidence /dev/null --preflight-bundle /dev/null \
  --bound-request-out "$JOB/bound_request.json" --pod-name blueprint-adp009d-vNN \
  --admission-out "$JOB/paid_resource_admission.json" \
  --adapter-output "$JOB/adapter_result.json" --adp-job-dir "$JOB" \
  --adp009d-approved-can "$V/simready/validation_inputs/840313_canned_beverage_match_v2/adp009a_840313_canned_beverage_match_v2.usda" \
  --adp009d-sage-collision "$V/simready/replacement_840313_match_v2/isaac_probe_v1/scene/assets/840313_collision.usd" \
  --adp009d-harness-manifest "$V/adp009d_franka_public_scene_sim_20260806/native_microcheck_v25_wrist_approach/bundle/provider_runtime/adp009d_franka_eval_harness_manifest.v1.json" \
  --adp009d-policy-candidate "pi05_droid" \
  --adp009d-aura-particlefield "$V/adp009d_franka_public_scene_sim_20260806/aura_nurec_v3_recentred_exposed/aura_ghost_removed_nurec.usdz" \
  --adp-machine-avoidlist "$JOB/adp_arena_vast_machine_avoidlist.json" \
  --adp-max-hourly-rate-usd 1.4 --adp-max-spend-usd 6.0 --adp-hard-ttl-seconds 5400 \
  --execute > "$JOB/launch.log" 2>&1
```

Provider-zero check (run before every launch, after every teardown):
```bash
python -c "
import json,os,urllib.request
key=open(os.path.expanduser('~/.config/vastai/vast_api_key')).read().strip()
r=urllib.request.Request('https://console.vast.ai/api/v0/instances/',headers={'Authorization':f'Bearer {key}'})
i=json.load(urllib.request.urlopen(r,timeout=40)).get('instances',[])
print('active:',len(i),[(x['id'],x.get('actual_status')) for x in i])"
```

## Spend discipline — hard constraints

- **Total overnight budget: $25.** Session spend to date is ~$6. Stop launching
  paid runs if cumulative spend reaches the cap and write up what remains.
- One instance at a time. Verify provider zero before each launch.
- Destroy the instance and re-verify zero after every run, including failures.
- Never launch a paid run to test something a local test can settle. Three runs
  tonight were spent on defects a local test caught afterwards for free.
- Prefer one well-instrumented run over two speculative ones. Hold everything
  constant except the variable under test, or the result is unattributable.

## Hard-won constraints — violating these has already cost runs

1. **The bundle ships modules flat, with no package.** A top-level relative
   import cannot resolve. Use the guarded dual-layout form. A test enforces this
   (`test_no_shipped_module_uses_an_unguarded_top_level_relative_import`).
2. **Every module the runtime imports must be in the bundle's file list.** A test
   derives the imports and enforces it.
3. **Policy server in the policy venv; policy client in Isaac's interpreter.**
   Only the thin client goes into Isaac. Never the full policy tree — JAX or a
   mismatched torch will take the card out from under Isaac.
4. **One venv per candidate.** openpi and GR00T pin incompatible torch versions;
   a shared venv silently breaks whichever installed first.
5. **The no-progress watchdog reads container stdout**, not redirected logs.
   Progress markers must be echoed by the entrypoint itself.
6. **The readiness probe and the episode must agree on the response shape.**
   Both vendors wrap their chunk as `{"actions": ...}`. A disagreement passes
   readiness and fails the episode.
7. **Machine class matters enormously.** An A6000 rendered this scene at 65 s/frame
   with a 60 s `omni.usd` idle timeout on every frame; L40 and RTX 6000Ada do the
   same work at ~22 s (full size) with zero timeouts. Keep the avoidlist current
   and add any host that stalls.
8. **A USD reference into a layer with no `defaultPrim` resolves to nothing.** This
   cost five runs. Assets must declare one.
9. **Scales in NuRec are stored pre-activation (log space).** "Flat" is a large
   negative number. A small linear value decodes as `exp(small) ≈ 1` — a one-metre
   thickness. This bug buried the camera in opaque needles.

## Reporting

Write findings to `docs/arm_decision_proof_v1/` as you go. For each paid run
record: what changed, what was held constant, the measured result, and what it
rules in or out. A null result is a result — record it rather than discarding it.

Leave a summary at `docs/arm_decision_proof_v1/OVERNIGHT_RESULTS_2026-08-08.md`
with: what landed, what each run cost and returned, what remains open, and the
single next action you would take. Be explicit about anything you could not
distinguish — do not resolve an ambiguity by choosing the flattering reading.
