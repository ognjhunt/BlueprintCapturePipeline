# Takeover runbook — Scene 840920 production loop

Written 2026-08-17 while the codex operator was still live, against verified
host and repo state. Everything here was read from the control plane or the
repository, not from a summary.

**Control plane:** `root@174.138.76.111`
**Deployed commit:** `6bcc65db` — live tree is `/opt/blueprint/BlueprintCapturePipeline`
**Service venv:** `/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python3`

The `prep-*` trees under `/opt/blueprint` are **not** what serves. `prep-53b84015`
is stale and its system python has no numpy. Always use the live tree and its
venv.

## Ledger state

Proven and not to be re-run: SAM source tracks, Gaussian excision,
retained-scene render/reproof, ArtiFixer3D → paired-native import.

**USD Content Agents is executionally complete** as of 2026-08-17. Both subruns
are terminal with official billing and a succeeded WebApp sync:

| Run | Status | Cost | Provider zero | Official billing | WebApp sync |
| --- | --- | --- | --- | --- | --- |
| Task A r3 | completed | $0.288955 | confirmed | landed | succeeded |
| Task B r2 | completed | $0.432352 | confirmed | landed | succeeded |

One caveat before counting it: the sync records carry
`webapp_trigger_proven: None`, not `true`. `LIVE_LANE_REACHABILITY.md` records
that completed runs historically returned `website_trigger_proven: false` with
`webapp_launch_record_missing`. "Sync succeeded" and "the run is bound to a
website record" may not be the same claim. Confirm which the ledger requires.

## Spend to date

**$6.7014 across 61 launch runs** — 12 completed, 49 blocked or failed.

Today: Task A $0.288955, Task B r1 $0.267443 (failed), Task B r2 $0.432352,
paired-native $0.125617. Total **$1.114367**.

Individual runs are cheap. The cost is in failed attempts, and the failures
repeat.

## Failure patterns worth knowing before you spend

Read from the 49 non-completed runs. These are the shapes that actually recur:

| Blocker | Seen | Meaning |
| --- | --- | --- |
| `material_agent_full_execution_failed` + `physics_agent_full_execution_failed` | 3× on Content Agents Task B (v8r10, 3b671992, r1 today) | **Not one-off.** Task B r2 succeeded on retry, but this has failed three times. Budget for a retry; if it fails twice consecutively, stop and diagnose rather than paying a third time. |
| `content_agents_full_execution_not_completed` + `content_agents_provider_output_zip_missing` | 4× | The run executed but produced no output zip. Costs real money ($0.70–0.81 on v8r8/v8r9). |
| `paid_resource_admission_has_blockers` / `not_admitted` | 3× | Authority not valid at launch. Costs $0 — fails before allocation. Cheap, fix and relaunch. |
| `simready_isaac_execution_not_completed` + `native_execution_not_proven` | 2× on scene 840313 | SimReady has never completed a proving run; the two 840313 attempts cost $0.13 and $0.10. |
| `sam31_runtime_claim_ceiling_mismatch` | 3× | Fails at $0. |
| `execute_launch_id_required` | Task A r2 today | Dispatcher could not see the standing authority. Fixed by placing the authority in the dispatcher's configured directory; cost $0. |

## The three next lanes

### SimReady Isaac — blocked on a scene mismatch and a deploy

**All 14 SimReady input dirs are scene 840313.** Zero 840920 bundles or
authorities exist. The published profile rehearses `would_pass`, but it is
bound to 840313 — firing it produces a green, correctly-sealed validation of
the wrong scene. The terminal contract does not check scene identity.

The 840920 inputs to build from now exist. The paired-native run completed at
$0.125617 and its probes confirm both twins are already import-qualified in
Isaac:

- `840920_simready_washer_candidate` — 6 rigid bodies, **5 joint prims**,
  70 composed prims, `native_simulator_import_qualified: true`
- `840920_simready_notebook_candidate` — **1 joint prim**,
  `native_simulator_import_qualified: true`

Both still carry `joint_physics_behavior_qualified: false` and
`contact_or_support_qualified: false` — which is exactly what SimReady and
Joint Agent are for.

Candidate USDs, digest-matched to the probes:

```
.../scene840920-native-assets-53b84015-r1/paired_native_bundle-846bce86-r3/provider_runtime/assets/
  replacement_00.usda   sha256:9b0c47a374742ced…   washer
  replacement_01.usda   sha256:9807d51715b0f166…   notebook
```

**Command chain — every link now has a CLI:**

```
1. public_scene_simready_replacement.py --request <req>          → replacement receipt
2. public_scene_simready_native.py                               → probe root
     --evidence-root … --replacement-receipt … --destination …
3. public_scene_simready_isaac_bundle.py                         → bundle
     --probe-root … --native-probe-manifest … --scene-id 840920
     --candidate-usd … --job-dir … --worker-source …
     --source-commit-sha <deployed>
4. scripts/issue_simready_isaac_paid_attempt_authority.py        → authority
5. scripts/build_simready_isaac_live_profile.py                  → profile
6. scripts/stage_paid_lane_bundle.py                             → host-resident staging
7. scripts/rehearse_lane_terminal_contract.py                    → $0 check, do this first
```

**Step 2 did not exist until 2026-08-17.** `public_scene_simready_native.py`
had no entry point, so the probe root could only be produced by calling a
Python function — meaning the bundle was unbuildable at the deployed commit.
The fix is on branch `claude/openai-spend-closure-20260817` and **is not
merged**. Until it lands and deploys, step 2 cannot run at the deployed commit,
and a bundle built anywhere else is refused by the allocator.

### Joint Agent — needs 840920 artifacts from scratch

Both existing bundle receipts are scene **840796**
(`/var/lib/blueprint/task-evaluation-inputs/joint-agent-840796-live{,-r2}`).
Zero 840920 artifacts. Two blocked launches on the host are 840796-era, from
2026-08-13.

Correction to the working plan: **Task B is not "rigid/inapplicable."** The
notebook USD contains 1 joint prim. It needs validating as correctly locked,
not skipped.

### Native Task Arena — three sequential runs, not one launch

Nothing exists: 0 packets, 0 construction results, 0 control results, 0 policy
specs, 0 bundle. The builder requires `--construction-result`,
`--control-result`, and `--policy-execution-spec`, which are **outputs of the
earlier stages**. The policy profile cannot be built until construction and
controls have actually run. Any plan treating Arena as a single launch is
wrong.

## Profiles: catalog entry ≠ launchable

The catalog at
`/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-profile-catalog.json`
lists 109 profile IDs, but only 58 files exist under
`task-evaluation-profile-releases`. A catalog entry does not mean the profile
file is present.

Profile **files** exist for: gaussian-excision, content-agents, simready-isaac
(840313), sam31-source-tracks, artifixer3d, semantic-teacher, and a
franka dry-run.

Profile files **absent** for: joint-agent, paired-target, arena,
native-camera, diagnostic-canary, reconstruction-smoke — six families.

## OpenAI cost attribution

Official OpenAI per-run cost requires `OpenAIProjectCandidateCostAuthority`,
which reserves against a **pre-run zero baseline**. Once money has moved the
window is non-zero and no official cost can ever be produced. No production
lane is wired to it — it appears only in the CLI, an export, its own module,
and its test.

Two modules landed on the unmerged branch address this:

- `openai_unattributable_spend.py` — the honest closure for runs already
  executed. Reserves the full authority cap, names which of two structural
  causes applies, refuses a smaller estimate, refuses to close while a
  reservation is in flight, never sets `cost_is_final`.
- `openai_official_cost_gate.py` — the forward fix. `preflight_…` answers
  "would a reservation succeed?" for $0; `require_…` returns a reservation or
  raises, with no path that both fails and returns.

The scope attestation is deliberately an operator artifact — the validator
requires `issued_by_agent` to be exactly `False`. **You cannot self-authorize
this.** It needs a dedicated OpenAI project, an API key, an admin key file at
mode `0600`, and an operator-signed attestation.

## Traps

- **Worktree imports.** `pytest` reads the worktree's `src`; bare `python -m`
  reads the main tree's editable install. Set `PYTHONPATH` explicitly or you
  will test the wrong code and believe the wrong answer.
- **Rehearse before firing.** `rehearse_lane_terminal_contract.py` costs $0,
  rents nothing, reads no credential. But it does **not** check scene identity
  — a `would_pass` on a wrong-scene profile is false confidence.
- **Every deploy invalidates bundles.** The allocator refuses a bundle whose
  commit is not the running one.
- **Local disk.** The workstation sat at 99% (under 1 GB free) on 2026-08-17.
  Clearing `~/Library/Caches` reclaimed 7.4 GB. `agent_workspace_gc.py`
  reclaimed nothing — everything was in-window or stash-bearing.

## Unmerged work gating this

Branch `claude/openai-spend-closure-20260817`, three commits, **local and
unpushed** in a scratch worktree. Not visible to any other operator.

1. `openai_unattributable_spend.py` + 12 tests
2. `openai_official_cost_gate.py` + 10 tests
3. `public_scene_simready_native.py` CLI + rediscovery-contract extension
   (74 tests in that file pass; the new test fails on the parent commit)

Item 3 is on the critical path for SimReady 840920.
