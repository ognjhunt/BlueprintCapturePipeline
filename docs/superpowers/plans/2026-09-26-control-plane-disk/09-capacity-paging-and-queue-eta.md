# PR 9 — Capacity pages a human, and intake queues with an ETA (design doc phases 4 and 3)

> Read `00-index.md` first. Builds on the integrated stack (PRs 1–8).

**Goal:**
- Capacity is an SLO with an action. A human is paged at three days of headroom, whenever admission
  is refused, and whenever growth is blocked.
- An unrouted alert is itself loud.
- Scene intake accepts a request while capacity is short (`queued_for_capacity`) instead of
  answering 503. The capacity wait records its shortfall and an ETA when one can be known.

**Branch / base / worktree:** `claude/disk-9-capacity-paging` from the integration branch,
`$WORKSPACE/BlueprintCapturePipeline-disk-9-20260926`.

## Facts (verified)

- **Forecast and alerts.**
  - `forecast` (capacity controller 176-203) was accurate on 2026-09-26 ("floor within 0.39 days").
    What failed was delivery: `BLUEPRINT_OPERATOR_ALERT_WEBHOOK_URL` did not reach a person, and
    nothing said so.
  - `alert_due` (281-290) alerts on escalation and repeats hourly while critical.
  - `post_alert` (293-323) sends `level`, `alerts`, `mounts` and `text`.
  - Volume growth blocked for lack of `BLUEPRINT_CAPACITY_AUTORESIZE_ACK`, or at the provider's
    100 GB maximum, appears only in `report.volume_resize`, never as an alert.
- **The 503.** Scene intake returns 503 `"scene intake disk admission refused"` when
  `launch_preparation` is refused (`task_evaluation_scene_intake_http.py:136-137`). Yet scene
  progression already waits for capacity before starting a chain (`scene_whole_chain_capacity_insufficient`,
  `awaiting_execution` / phase `capacity`).
- **Report visibility.**
  - The storage-GC report (`storage-gc/latest.json`) is root-only.
  - PR 2's `capacity/summary.json` is door- and service-readable (0644).
  - PR 5 writes `release-retention/latest-deploy-retirement.json`.
  - PR 8 writes break-glass notes in `cleanup-receipts/`.
- **G1 is out of scope.** G1 team intake has the same 503 but is humanoid work outside the ADP focus
  (`AGENTS.md`); leave it unchanged.

## Design

**Alert severity.** Every alert row gains `severity`.
- **`page`:**
  - `floor_within_three_days`, `admission_refused`, `critical_admission_refused`, `mount_unreadable`
  - `volume_growth_blocked` (new; `reason` is the resize block reason)
  - `operator_alert_route_unconfigured` (new; emitted when the webhook URL is empty, and it makes
    the level at least `warning`)
- **`warn`:**
  - `utilization_warning`, `utilization_critical`
  - `usage_unclassified_root`, `usage_attribution_low` (PR 2)
  - `release_retirement_attention` (new; from the retirement summary when its status is not
    `applied` or it carries alerts)
  - `break_glass_notes_unreported` (new; the count of unreported notes)
  - `project_spend_refresh_blocked`, provider funding

**Paging semantics** (`alert_due`):
- The fingerprint is `sha256` of the sorted `(code, mount, severity)` tuples of page-severity alerts.
- Post when:
  - the level escalates, as today; or
  - the page fingerprint differs from the last posted one; or
  - page alerts persist and an hour has passed since the last post.
- Record `last_alert_fingerprint` beside `last_alert_epoch`.

**Payload** (`post_alert`), additional fields:
- `severity`: `page` if any page alert, else `warn`
- `page` (bool), `fingerprint`, `runbook: "docs/runbooks/control-plane-capacity.md"`
- `summary`: one line, at most 200 characters, most urgent first. For example:
  `"/var/lib/blueprint reaches its floor in 2.1 days; scene chains refused"`.

`text` is kept for existing receivers.

**Reclaim outlook.** Added to `latest.json` and `summary.json`. The controller runs as root, so it
can read the GC report:

```json
"reclaim_outlook": {"observed_at_epoch": 0, "next_reclaim_epoch": 0,
                    "reclaimable_bytes": 0,
                    "sources": {"scene_workspaces": 0, "evidence_offload": 0, "derived_directories": 0, "content_store": 0},
                    "volume_growth": "planned" | "applied" | "blocked" | "not_needed" | "not_configured"}
```

- Sum the dry-run candidate bytes of the latest GC report, counting only phases that would actually
  apply: offload and scene retirement only when enabled.
- `next_reclaim_epoch = gc_report.observed_at_epoch + 3600`.

**ETA.**

```python
def capacity_eta(shortfall_bytes: int, *, summary: Mapping | None, now: float) -> dict:
    """{"eta_epoch": float | None, "eta_basis": "volume_growth" | "reclaim_scheduled" | "operator_action_required" | "unknown"}"""
```

- Volume growth planned or applied → `now + 600`.
- Reclaimable bytes at least equal to the shortfall → `next_reclaim_epoch`.
- A summary that exists but covers neither → `operator_action_required`, with no ETA.
- No readable summary → `unknown`.

**Scene progression.** The whole-chain gate (`task_evaluation_scene_progression.py:572-599`) keeps
its status, phase and blocker, so WebApp compatibility is unchanged. It adds
`state["capacity_wait"]`:

```json
{"since_epoch": 0, "required_bytes": 0, "available_bytes": 0,
 "shortfall_bytes": 0, "basis": "measured_p95", "devices": [], "eta_epoch": null,
 "eta_basis": "operator_action_required", "next_check_epoch": 0}
```

- The summary path is `BLUEPRINT_CAPACITY_SUMMARY_PATH`, defaulting to
  `/var/lib/blueprint/pipeline-control-plane/capacity/summary.json`.
- `since_epoch` is preserved across passes.
- The object is removed once admitted.

**Scene intake.**
- Replace the 503 with acceptance. The success JSON gains
  `"capacity": {"state": "queued_for_capacity", "refused_roles": [...], "eta_epoch": …, "eta_basis": …}`,
  or `{"state": "available"}`.
- Execution is still gated by progression.
- Idempotency, signatures and every other refusal are unchanged.

## Tasks (each: failing test → implement → pass → commit)

- [ ] **9.1 Severity, unrouted and growth-blocked alerts.** Tests in `tests/test_control_plane_capacity_controller.py`:
  - **`test_three_days_of_headroom_pages_and_unrouted_alerts_are_loud`**:
    - a history that forecasts about 2 days, with a webhook → `floor_within_three_days`, severity
      `page`; the posted payload has `page: true`, `severity: "page"`, a fingerprint and a runbook;
    - the same state with no webhook → `operator_alert_route_unconfigured` present;
    - the summary carries both.
  - `test_blocked_volume_growth_pages`
  - `test_new_page_alert_posts_even_at_the_same_level` (fingerprint change within the hour)

  Update the existing escalation test's expected code set only by adding severities. Commit "Page a human for capacity, and say so when no one can be paged".
- [ ] **9.2 Retirement and break-glass attention.** Tests:
  - the retirement summary with `status: "blocked"` → `release_retirement_attention`;
  - one unreported note → `break_glass_notes_unreported`.

  Commit "Surface release-retirement and break-glass attention in the capacity report".
- [ ] **9.3 Reclaim outlook and ETA.** Tests:
  - the outlook sums only enabled phases;
  - `capacity_eta` covers each basis.

  Commit "Forecast when a capacity wait ends".
- [ ] **9.4 Progression capacity wait.** Tests in `tests/test_task_evaluation_scene_progression.py`:
  - `test_capacity_wait_records_shortfall_and_eta` (stub `whole_chain_admission` and a summary file);
  - `since_epoch` is preserved across passes;
  - the object is cleared once admitted.

  Commit "Record a scene's capacity shortfall and ETA while it waits".
- [ ] **9.5 Scene intake queues.** Tests in the scene intake HTTP test file:
  - `test_scene_intake_accepts_and_queues_for_capacity` (refused roles → 2xx with a `capacity`
    state; the intent is staged);
  - `test_scene_intake_reports_available_capacity`.

  Commit "Accept scene intents while capacity is short and say when they can run".
- [ ] **9.6 Runbook.** New `docs/runbooks/control-plane-capacity.md` covering:
  - thresholds and severities; what a page means; the first command
    (`python3 scripts/operator_door.py status`, then `usage`);
  - growing (volume maximum, DigitalOcean limit request, the `BLUEPRINT_CAPACITY_AUTORESIZE_ACK` value);
  - reclaiming (start the GC unit through the door, `retire-scene-workspace`, the offload opt-in);
  - holds instead of stops; break-glass notes; "nobody hand-deletes evidence";
  - the owner's Phase-0 checklist from the design doc.

  Link it from `docs/CONTROL_PLANE_STORAGE.md` and `docs/CONTROL_PLANE_CAPACITY_PLAN.md`. Commit "Add the control-plane capacity runbook".

## PR verification

- `tests/test_control_plane_capacity_controller.py`, `tests/test_task_evaluation_scene_progression.py`;
- the scene intake HTTP tests (`grep -l task-evaluation-scene-intents tests`);
- `tests/test_live_pipeline_intake_service.py`.
