# Policy Diagnostic Launch Runbook — 2026-08-24

Companion to PR #1021 (non-scoring canonical policy diagnostic lane) and PR #1020
(continuous phases 6–11 synthetic-checkpoint diagnostic). Written from review of both PR
heads (all focused tests pass under worktree PYTHONPATH: #1020 136/136, #1021 100/100;
lint clean). Purpose: first REAL policy-motion data, tonight, without touching the
controls lane's cadence or the decision-grade claim boundary.

User authorization on record: multiple GPUs + parallel policy runs approved
(2026-08-24 evening).

---

## 0. What each PR contributes (so nobody launches the wrong thing)

- **#1020 is controls-side.** The continuous jp01→retreat diagnostic from a synthetic
  post-phase-5 checkpoint is embedded in the controls worker — it rides the NEXT controls
  run automatically. No separate launch exists or is needed.
- **#1021 is its own paid lane**: allocator/profile/bundle/result mode
  `policy-diagnostic` (profile id prefix `arena-policy-diagnostic-live`, job dir
  `arena-policy-diagnostic-job`). One frozen candidate acts from canonical reset;
  scoring disabled and sealed off (`scientific_scoring_permitted: false`); requires
  qualified construction + **un**qualified controls receipt + separately-passed
  zero-action negative; ordinary policy admission explicitly rejects diagnostic
  authority (`native_task_policy_diagnostic_authority_invalid`).

## 1. Preconditions (in order)

1. Merge #1020 and #1021 (reviews complete; see session log).
2. **Full suite before production promotion** — #1021 is a cross-cutting paid-allocator
   change and its own impacted planner says `requires_full_suite` for promotion. For the
   first development_only diagnostic, an iteration deploy is acceptable (ceiling is
   development_only anyway); run `scripts/pytest_full.sh` in parallel and promote via the
   #1009 same-SHA path when green.
3. Deploy the merged SHA to the control plane
   (`deploy_control_plane_commit.py … --iteration` first; promote later).
4. **Rebuild everything after the deploy** — bundles/preflights/profiles built before a
   deploy are invalid (r19 class).

## 2. Inputs — the four digests (all exist today)

| Input | Source artifact |
|---|---|
| scene plan | the sealed 840920 task_a scene plan (unchanged all campaign) |
| construction result | **c76 construction run — COMPLETED** (`adp-arena-construction-…-c7dc8461-…`) |
| control result | **C79's sealed result** (`adp-arena-controls-…-c79-live-dls-phase5-first-cold-…`) — the highest-quality unqualified pair (reached contact_open at 4.43 mm; clean zero-action negative). Do NOT bind C82's (it regressed at approach — worse diagnostic substrate). `status=blocked` is REQUIRED by the diagnostic contract: it refuses if controls are qualified, forcing graduation to the real lane. |
| zero-action negative | bound separately from the same controls pair; validator checks `control_id == "zero_action_negative"` passed |

## 3. Build the spec (one per candidate)

```bash
python scripts/materialize_native_task_arena_policy_diagnostic_spec.py \
  --candidate-id pi05_droid \
  --scene-plan <scene_plan.json> \
  --construction-result <c76_construction_result.json> \
  --control-result <latest_controls_result.json> \
  --output <pd1_pi05_spec.json>
```

Run order recommendation: **pi05_droid first** (lighter serving risk), groot_n17_droid
second (GR00T has a documented "provisions but never serves" history — that is exactly
what run 2 exists to retire).

## 4. Bundle → manifest → profile → rehearsal (c73-era chain, new link name)

- Build the policy-diagnostic bundle at the DEPLOYED SHA; manifest publish needs
  `--destination-prefix r2://blueprint/task-evaluation/immutable-manifests` and the
  `BLUEPRINT_WAM_OBJECT_STORE_*` env vars passed through sudo (gs:// and bare-bucket
  forms both throw credential errors — known trap).
- Profile: `build_native_task_arena_live_profile.py` with the **`policy-diagnostic`**
  link; add `--preferred-geolocation-regex "(^|, )(US|United States)(,|$)"` (C27
  precedent: US host halved the cycle).
- Terminal-contract rehearsal before any authority is issued (`would_pass` required).

## 5. Authority + spend (defaults verified in the transport)

- Transport defaults: **max $0.80/hr, hard cap $1.00, TTL 5400 s, retry 0**, own
  watchdog + teardown. Authority math: rate×TTL/3600 ≤ cap (0.80×1.5 = 1.20 > 1.00 —
  so either issue at rate ≤ 0.66 for the full TTL or accept the cap as the binding
  limit; do NOT raise the cap for run 1).
- GPU floor: inherited from the policy path — `min_gpu_ram_mb=46_000`
  (native_task_arena_vast.py:352; units are MB — never "46").
- Mandatory pre-spend preflight (`require_pre_spend_preflight`) as everywhere.
- **Fleet coexistence:** `allowed_active_instance_ids` for this lane must reference only
  the policy pod. Do not include the controls 4090 (24 GB — below floor, and attaching
  the wrong lane to it is the C75-warm-refusal class). Both pods coexist under the #998
  fleet leases with independent watchdogs.

## 6. Fire

Two-gate pattern as always: arm EXECUTE_ID in `/etc/blueprint/pipeline-control-plane.env`
(backup first) for launch id `…policy-diagnostic…pd1-pi05…`, then
`submit_task_evaluation_launch_via_webapp.py` with the Render submit secret. Instant
`blocked` with `execute_launch_id_required` = the gate, not a failure; arm and resubmit.

## 7. What run 1 must produce to count as "real policy data"

- `native_task_arena_policy_diagnostic_result.v1.json` sealed, with: lossless policy
  observations, exact actions, three-camera media, per-step timings including policy
  query latency (the 15 Hz cadence check), and the claim-boundary strings intact
  (`development_only_policy_motion_diagnostic_not_scoring_not_ranking…`).
- Spend reconciliation + provider-zero with `status_source=provider_api` teardown proof.
- Triage map for the three likely first-run failures, each a known class: checkpoint
  serving (GR00T precedent), VRAM at load (floor/offer selection), query-cadence stalls.

## 8. What these runs may never be called

No scoring, no ranking, no candidate comparison, no policy admission, no phase-5 or
downstream qualification — the contracts enforce all of this (verified in review: the
diagnostic REFUSES qualified controls; ordinary policy bundles REFUSE diagnostic
authority). When phase 5 admits, the diagnostic lane retires and the real policy link
fires with integration risk already at zero.
