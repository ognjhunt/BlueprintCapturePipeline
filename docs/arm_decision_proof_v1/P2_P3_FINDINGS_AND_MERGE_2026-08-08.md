# P2 closed, P3 built — findings and how to prove them

Written from branch `claude/adp009d-p3-render-interval` (`4bdc3e935`), which is
based on `dca08f185` and therefore already contains P0 (`c9103d311`) and P1
(`dca08f185`).

## P2 — float32 is closed. Do not spend a run on it.

The float16 quantisation argument was made when the cameras rendered 1280×720.
They now render **320×180**, the resolution both candidates actually consume,
which makes one pixel 4× larger and collapses the argument.

Recentred float16 p95 position error is **0.426 mm**. At 90° FOV:

| render | 1 px @ 2 m | error in pixels |
|---|---|---|
| 1280×720 | 3.12 mm | 0.136 px |
| **320×180 (current)** | 12.50 mm | **0.034 px** |

Worst case at 1 m: **0.068 px**. Two orders of magnitude below a pixel cannot
change the rasterised image.

**Action: drop `aura_nurec_v4_float32/` (84 MB), keep
`aura_nurec_v3_recentred_exposed/` (42 MB).** No GPU run required.

Note it was already sub-pixel at 1280×720, so float32 was never going to matter
for a policy observation. The earlier sharpness reasoning was invalidated by the
resolution change and I did not recheck it until now.

## P3 — built and guarded, needs one run to prove

Measured from v62: `policy_episode` 267.99 s / 3 = **89.3 s per episode**, 520
env steps, 60 policy queries, warmup gives **195.1 ms per frame**.

- 520 × 195 ms = **101 s of render** against 89 s measured wall clock. Render is
  ~100% of the episode.
- **Policy inference is effectively free** — ~0 s across 60 queries. There is
  nothing to optimise there.
- The policy consumes an observation **once per query**, not once per step.
  520 renders serve 60 observations: **460 wasted, 88%**.

`cfg.sim.render_interval` was 8, equal to `decimation`, so exactly one render per
`env.step`. It is now `decimation × RENDER_QUERY_HORIZON` (64), one render per
policy query. Opt out with `BLUEPRINT_ADP009D_RENDER_PER_QUERY=0` for a
diagnostic capture that wants every frame.

**Projected: 89.3 s → ~11.7 s per episode (7.6×); a 3-episode two-policy ranked
run from ~8.9 min to ~1.2 min of episode time.**

### The guard is the load-bearing part

Rendering less often is sound only if the frame the policy sees was rendered
*after* the actions it is responding to. The adapter stamps every observation
with `observation_sim_time`; the episode raises
`policy_episode_observation_did_not_advance` if it does not advance, and the
receipt records `observation_interval_seconds` so a claimed saving shows the
cadence that actually ran.

Without that guard a misaligned cadence hands the policy a frame from the
previous chunk, which presents as *a policy that ignores the scene* rather than
as a harness bug — a plausible-looking policy verdict, which is the most
expensive kind of wrong and exactly what P0 exists to prevent.

**A fast run is not the proof. A fast run that also scores is** — if the cadence
were misaligned the guard fires and the episode fails.

## How to prove both

One run does it, because this branch contains P0 and P1:

```bash
cd /private/tmp/blueprint-adp009d-p3      # or merge this branch first
export BLUEPRINT_ADP009D_CAMERA_RESOLUTION=policy
export BLUEPRINT_ADP009D_EPISODES=3
export BLUEPRINT_ADP009D_PROVISION_TIMEOUT_SECONDS=1500
# ... standard gpu-canary launch, --adp009d-policy-candidate "pi05_droid,groot_n17_droid"
```

Read from the receipt:

| check | where |
|---|---|
| P3 saving | `timings_seconds.policy_episode` ≈ 35 s for 3 episodes, not 268 s |
| P3 guard held | episodes `scored`, no `observation_did_not_advance` blocker |
| P3 cadence real | `observation_interval_seconds` ≈ 0.533 s, not 0.067 s |
| P0 resolved | `interpretation` populated; `never_moved` is finally readable |
| P4 | both candidates in `comparison.ranking` |

## Two hazards observed while trying to run this

**Concurrent paid runs are unsafe, not merely blocked.** The lane refuses a
second launch with `vast_paid_launch_lock_busy`, which is correct. But a run
that *did* start while another agent was active died as
`vast_probe_interrupted_before_completion` moments before the other agent's
instance appeared — consistent with one agent's orphan reaper destroying the
other's instance. **One agent owns the lane at a time.**

**RTX A6000 is a slow host for this scene.** Measured 65 s/frame with a 60 s
`omni.usd` idle timeout on *every* frame, against ~22 s on L40 and RTX 6000Ada
with zero timeouts. A run that lands on one will look hung and may exhaust its
TTL. Consider adding the class to the avoidlist rather than individual machine
ids.

## Branches

- `claude/adp009d-p3-render-interval` (`4bdc3e935`) — P3, merge this
- `claude/adp009d-p1-groot-serve` (`35eb43cd2`) — superseded by Codex's
  `dca08f185`, which is better (named `RoundTripAttemptTimeout`, 30 s bound).
  Keep for reference or delete.
